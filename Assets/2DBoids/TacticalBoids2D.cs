using UnityEngine;
using UnityEngine.UI;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine.Rendering;
using System.Collections.Generic;
using System;
using System.Linq;
using System.Runtime.InteropServices;

struct Boid
{
    public float2 pos;
    public float2 vel;
    public float2 target; // Added target position for battalion formation
    public int battalionId; // Battalion this boid belongs to
}

public class TacticalBoids2D : MonoBehaviour
{
    const float blockSize = 1024f;

    [Header("Performance")] [SerializeField]
    public int numBoids = 5000;

    bool useGpu = true;

    [Header("Boid Settings")] [SerializeField]
    float maxSpeed = 2;

    [SerializeField] float edgeMargin = .5f;
    [SerializeField] float visualRange = .5f;
    float visualRangeSq => visualRange * visualRange;
    [SerializeField] float minDistance = 0.15f;
    float minDistanceSq => minDistance * minDistance;
    [SerializeField] float cohesionFactor = 2;
    [SerializeField] float separationFactor = 1;
    [SerializeField] float alignmentFactor = 5;
    [SerializeField] float targetFactor = 3f; // How strongly boids move toward their target
    [SerializeField] float battalionCohesionFactor = 1.5f; // Stronger cohesion within battalion

    [Header("Prefabs")] [SerializeField] Text fpsText;
    [SerializeField] Text boidText;
    [SerializeField] Text selectionText;
    [SerializeField] Slider numSlider;
    [SerializeField] Button modeButton;
    [SerializeField] ComputeShader boidShader;
    [SerializeField] ComputeShader gridShader;
    [SerializeField] Material boidMat;
    Vector2[] triangleVerts;
    GraphicsBuffer trianglePositions;

    [Header("UI")] [SerializeField] Button lineFormationButton;
    [SerializeField] Button squareFormationButton;
    [SerializeField] Button columnFormationButton;
    [SerializeField] Button wedgeFormationButton;

    float minSpeed;
    float turnSpeed;

    NativeArray<Boid> boids;
    NativeArray<Boid> boidsTemp;

    int updateBoidsKernel, generateBoidsKernel;
    int updateGridKernel, clearGridKernel, prefixSumKernel, sumBlocksKernel, addSumsKernel, rearrangeBoidsKernel;
    int blocks;

    ComputeBuffer boidBuffer;
    ComputeBuffer boidBufferOut;
    ComputeBuffer gridBuffer;
    ComputeBuffer gridOffsetBuffer;
    ComputeBuffer gridOffsetBufferIn;
    ComputeBuffer gridSumsBuffer;
    ComputeBuffer gridSumsBuffer2;
    ComputeBuffer battalionBuffer; // Buffer for battalion data

    // Index is particle ID, x value is position flattened to 1D array, y value is grid cell offset
    NativeArray<int2> grid;
    NativeArray<int> gridOffsets;
    int gridDimY, gridDimX, gridTotalCells;
    float gridCellSize;

    float xBound, yBound;
    RenderParams rp;

    readonly int cpuLimit = 1 << 16;
    readonly int gpuLimit = (int)blockSize * 65535;

    // Battalion management
    BattalionManager battalionManager;
    bool isDragging = false;
    Vector2 dragStart;
    Vector2 currentMousePos;
    bool showSelectionBox = false;

    void Awake()
    {
        numSlider.maxValue = Mathf.Log(useGpu ? gpuLimit : cpuLimit, 2);
        triangleVerts = GetTriangleVerts();

        // Initialize battalion manager if not already assigned
        if (battalionManager == null)
        {
            battalionManager = GetComponent<BattalionManager>();
            if (battalionManager == null)
                battalionManager = gameObject.AddComponent<BattalionManager>();
        }
        Debug.Log($"TacticalBoids2D.Awake() called on GameObject: {gameObject.name}, Active: {gameObject.activeInHierarchy}");

    }

    // Start is called before the first frame update
    void Start()
    {
        Debug.Log("TacticalBoids2D.Start() - Beginning initialization");
        // Zoom camera based on number of boids
        Camera.main.orthographicSize = Mathf.Max(2, Mathf.Sqrt(numBoids) / 10 + edgeMargin);
        Camera.main.transform.position = new Vector3(0, 0, -10);
        
        Debug.Log("Starting camera");
        
        GetComponent<MoveCamera2D>().Start();

        boidText.text = "Boids: " + numBoids;
        xBound = Camera.main.orthographicSize * Camera.main.aspect - edgeMargin;
        yBound = Camera.main.orthographicSize - edgeMargin;
        turnSpeed = maxSpeed * 3;
        minSpeed = maxSpeed * 0.75f;
        
        Debug.Log("Getting kernel IDs");
        
        // Get kernel IDs
        updateBoidsKernel = boidShader.FindKernel("UpdateBoids");
        generateBoidsKernel = boidShader.FindKernel("GenerateBoids");
        updateGridKernel = gridShader.FindKernel("UpdateGrid");
        clearGridKernel = gridShader.FindKernel("ClearGrid");
        prefixSumKernel = gridShader.FindKernel("PrefixSum");
        sumBlocksKernel = gridShader.FindKernel("SumBlocks");
        addSumsKernel = gridShader.FindKernel("AddSums");
        rearrangeBoidsKernel = gridShader.FindKernel("RearrangeBoids");
        
        Debug.Log("Setting up compute buffers");

        // Setup compute buffer - modified to include target and battalionId
        boidBuffer = new ComputeBuffer(numBoids, Marshal.SizeOf<Boid>()); // 5 float2s (pos, vel, target) + 1 int
        boidBufferOut = new ComputeBuffer(numBoids, Marshal.SizeOf<Boid>());
        boidShader.SetBuffer(updateBoidsKernel, "boidsIn", boidBufferOut);
        boidShader.SetBuffer(updateBoidsKernel, "boidsOut", boidBuffer);
        boidShader.SetInt("numBoids", numBoids);
        boidShader.SetFloat("maxSpeed", maxSpeed);
        boidShader.SetFloat("minSpeed", minSpeed);
        boidShader.SetFloat("edgeMargin", edgeMargin);
        boidShader.SetFloat("visualRangeSq", visualRangeSq);
        boidShader.SetFloat("minDistanceSq", minDistanceSq);
        boidShader.SetFloat("turnSpeed", turnSpeed);
        boidShader.SetFloat("xBound", xBound);
        boidShader.SetFloat("yBound", yBound);
        boidShader.SetFloat("cohesionFactor", cohesionFactor);
        boidShader.SetFloat("separationFactor", separationFactor);
        boidShader.SetFloat("alignmentFactor", alignmentFactor);
        boidShader.SetFloat("targetFactor", targetFactor);
        boidShader.SetFloat("battalionCohesionFactor", battalionCohesionFactor);
        
        Debug.Log("Initialising battalions");
        
        // Initialize battalions
        battalionManager.InitializeBattalions(numBoids);
        
        if (battalionBuffer != null)
            battalionBuffer.Release();
        
        // Create battalion buffer
        battalionBuffer =
            new ComputeBuffer(Mathf.Max(1, battalionManager.battalions.Count), 32); 
        UpdateBattalionBuffer();
        
        if (battalionBuffer != null && battalionBuffer.IsValid())
            boidShader.SetBuffer(updateBoidsKernel, "battalions", battalionBuffer);
        
        boidShader.SetInt("numBattalions", battalionManager.battalions.Count);

        // Generate boids on GPU if over CPU limit
        if (numBoids <= cpuLimit)
        {
            // Populate initial boids
            boids = new NativeArray<Boid>(numBoids, Allocator.Persistent);
            boidsTemp = new NativeArray<Boid>(numBoids, Allocator.Persistent);
            for (int i = 0; i < numBoids; i++)
            {
                var pos = new float2(UnityEngine.Random.Range(-xBound, xBound),
                    UnityEngine.Random.Range(-yBound, yBound));
                var vel = new float2(UnityEngine.Random.Range(-maxSpeed, maxSpeed),
                    UnityEngine.Random.Range(-maxSpeed, maxSpeed));
                int battalionId = -1;

                // Assign battalion id
                foreach (Battalion battalion in battalionManager.battalions)
                {
                    if (i >= battalion.startIndex && i < battalion.startIndex + battalion.count)
                    {
                        battalionId = battalion.id;
                        break;
                    }
                }

                var boid = new Boid();
                boid.pos = pos;
                boid.vel = vel;
                boid.target = pos; // Initialize target to current position
                boid.battalionId = battalionId;
                boids[i] = boid;
            }

            boidBuffer.SetData(boids);
        }
        else
        {
            boidShader.SetBuffer(generateBoidsKernel, "boidsOut", boidBuffer);
            boidShader.SetInt("randSeed", UnityEngine.Random.Range(0, int.MaxValue));
            boidShader.Dispatch(generateBoidsKernel, Mathf.CeilToInt(numBoids / blockSize), 1, 1);
        }

        // Set render params
        rp = new RenderParams(boidMat);
        rp.matProps = new MaterialPropertyBlock();
        rp.matProps.SetBuffer("boids", boidBuffer);
        rp.worldBounds = new Bounds(Vector3.zero, Vector3.one * 3000);
        trianglePositions = new GraphicsBuffer(GraphicsBuffer.Target.Structured, 3, 8);
        trianglePositions.SetData(triangleVerts);
        rp.matProps.SetBuffer("_Positions", trianglePositions);

        // Spatial grid setup
        gridCellSize = visualRange;
        gridDimX = Mathf.Max(1, Mathf.FloorToInt(xBound * 2 / gridCellSize) + 30);
        gridDimY = Mathf.Max(1, Mathf.FloorToInt(yBound * 2 / gridCellSize) + 30);
        gridTotalCells = gridDimX * gridDimY;
        blocks = Mathf.Max(1, Mathf.CeilToInt(gridTotalCells / blockSize));

        Debug.Log($"Grid dimensions: {gridDimX}x{gridDimY}, Total cells: {gridTotalCells}, Blocks: {blocks}");
        // Don't generate grid on CPU if over CPU limit
        if (numBoids <= cpuLimit)
        {
            grid = new NativeArray<int2>(numBoids, Allocator.Persistent);
            gridOffsets = new NativeArray<int>(gridTotalCells, Allocator.Persistent);
        }

        Debug.Log("Setting buffers");

        gridBuffer = new ComputeBuffer(numBoids, 8);
        gridOffsetBuffer = new ComputeBuffer(gridTotalCells, 4);
        gridOffsetBufferIn = new ComputeBuffer(gridTotalCells, 4);
        blocks = Mathf.CeilToInt(gridTotalCells / blockSize);
        gridSumsBuffer = new ComputeBuffer(blocks, 4);
        gridSumsBuffer2 = new ComputeBuffer(blocks, 4);
        gridShader.SetInt("numBoids", numBoids);
        gridShader.SetBuffer(updateGridKernel, "boids", boidBuffer);
        gridShader.SetBuffer(updateGridKernel, "gridBuffer", gridBuffer);
        gridShader.SetBuffer(updateGridKernel, "gridOffsetBuffer", gridOffsetBufferIn);
        gridShader.SetBuffer(updateGridKernel, "gridSumsBuffer", gridSumsBuffer);

        gridShader.SetBuffer(clearGridKernel, "gridOffsetBuffer", gridOffsetBufferIn);

        gridShader.SetBuffer(prefixSumKernel, "gridOffsetBuffer", gridOffsetBuffer);
        gridShader.SetBuffer(prefixSumKernel, "gridOffsetBufferIn", gridOffsetBufferIn);
        gridShader.SetBuffer(prefixSumKernel, "gridSumsBuffer", gridSumsBuffer2);

        gridShader.SetBuffer(addSumsKernel, "gridOffsetBuffer", gridOffsetBuffer);

        gridShader.SetBuffer(rearrangeBoidsKernel, "gridBuffer", gridBuffer);
        gridShader.SetBuffer(rearrangeBoidsKernel, "gridOffsetBuffer", gridOffsetBuffer);
        gridShader.SetBuffer(rearrangeBoidsKernel, "boids", boidBuffer);
        gridShader.SetBuffer(rearrangeBoidsKernel, "boidsOut", boidBufferOut);

        gridShader.SetFloat("gridCellSize", gridCellSize);
        gridShader.SetInt("gridDimY", gridDimY);
        gridShader.SetInt("gridDimX", gridDimX);
        gridShader.SetInt("gridTotalCells", gridTotalCells);
        gridShader.SetInt("blocks", blocks);

        boidShader.SetBuffer(updateBoidsKernel, "gridOffsetBuffer", gridOffsetBuffer);
        boidShader.SetFloat("gridCellSize", gridCellSize);
        boidShader.SetInt("gridDimY", gridDimY);
        boidShader.SetInt("gridDimX", gridDimX);

        // Setup UI
        if (lineFormationButton != null)
            lineFormationButton.onClick.AddListener(() => SetFormation(FormationType.Line));
        if (squareFormationButton != null)
            squareFormationButton.onClick.AddListener(() => SetFormation(FormationType.Square));
        if (columnFormationButton != null)
            columnFormationButton.onClick.AddListener(() => SetFormation(FormationType.Column));
        if (wedgeFormationButton != null)
            wedgeFormationButton.onClick.AddListener(() => SetFormation(FormationType.Wedge));
        
        UpdateBattalionBuffer();
        
        Debug.Log($"BattalionData size: {Marshal.SizeOf<BattalionData>()} bytes");
        Debug.Log($"Buffer stride: {battalionBuffer.stride} bytes");
        
        Debug.Log("TacticalBoids2D.Start() - Initialization complete");
    }
    
    void VerifyAndSetComputeBuffers()
    {
        // Check boidBuffer
        if (boidBuffer == null || !boidBuffer.IsValid())
        {
            Debug.LogError("boidBuffer is null or invalid. Creating a new one...");
            boidBuffer = new ComputeBuffer(numBoids, 20); // 5 float2s (pos, vel, target) + 1 int
        }

        // Check boidBufferOut
        if (boidBufferOut == null || !boidBufferOut.IsValid())
        {
            Debug.LogError("boidBufferOut is null or invalid. Creating a new one...");
            boidBufferOut = new ComputeBuffer(numBoids, 20);
        }

        // Check gridBuffer
        if (gridBuffer == null || !gridBuffer.IsValid())
        {
            Debug.LogError("gridBuffer is null or invalid. Creating a new one...");
            gridBuffer = new ComputeBuffer(numBoids, 8);
        }

        // Check gridOffsetBuffer
        if (gridOffsetBuffer == null || !gridOffsetBuffer.IsValid())
        {
            Debug.LogError("gridOffsetBuffer is null or invalid. Creating a new one..." + gridTotalCells);
            gridOffsetBuffer = new ComputeBuffer(gridTotalCells, 4);
        }

        // Check gridOffsetBufferIn
        if (gridOffsetBufferIn == null || !gridOffsetBufferIn.IsValid())
        {
            Debug.LogError("gridOffsetBufferIn is null or invalid. Creating a new one...");
            gridOffsetBufferIn = new ComputeBuffer(gridTotalCells, 4);
        }

        // Check gridSumsBuffer
        if (gridSumsBuffer == null || !gridSumsBuffer.IsValid())
        {
            Debug.LogError("gridSumsBuffer is null or invalid. Creating a new one...");
            gridSumsBuffer = new ComputeBuffer(blocks, 4);
        }

        // Check gridSumsBuffer2
        if (gridSumsBuffer2 == null || !gridSumsBuffer2.IsValid())
        {
            Debug.LogError("gridSumsBuffer2 is null or invalid. Creating a new one...");
            gridSumsBuffer2 = new ComputeBuffer(blocks, 4);
        }

        // Check battalionBuffer
        if (battalionBuffer == null || !battalionBuffer.IsValid())
        {
            Debug.LogError("battalionBuffer is null or invalid. Creating a new one...");
            int structSize = Marshal.SizeOf<BattalionData>();
            Debug.Log($"Creating battalion buffer with stride: {structSize} bytes");
            battalionBuffer = new ComputeBuffer(
                Mathf.Max(1, battalionManager.battalions.Count),
                structSize
            );
            UpdateBattalionBuffer();
        }

        // Set all buffers for each kernel
        try {
            // Boid shader buffers
            boidShader.SetBuffer(updateBoidsKernel, "boidsIn", boidBufferOut);
            boidShader.SetBuffer(updateBoidsKernel, "boidsOut", boidBuffer);
            boidShader.SetBuffer(updateBoidsKernel, "gridOffsetBuffer", gridOffsetBuffer);
            boidShader.SetBuffer(updateBoidsKernel, "battalions", battalionBuffer);
            
            // Grid shader buffers for updateGridKernel
            gridShader.SetBuffer(updateGridKernel, "boids", boidBuffer);
            gridShader.SetBuffer(updateGridKernel, "gridBuffer", gridBuffer);
            gridShader.SetBuffer(updateGridKernel, "gridOffsetBuffer", gridOffsetBufferIn);
            
            // Grid shader buffers for clearGridKernel
            gridShader.SetBuffer(clearGridKernel, "gridOffsetBuffer", gridOffsetBufferIn);
            
            // Grid shader buffers for prefixSumKernel
            gridShader.SetBuffer(prefixSumKernel, "gridOffsetBuffer", gridOffsetBuffer);
            gridShader.SetBuffer(prefixSumKernel, "gridOffsetBufferIn", gridOffsetBufferIn);
            gridShader.SetBuffer(prefixSumKernel, "gridSumsBuffer", gridSumsBuffer2);
            
            // Grid shader buffers for addSumsKernel
            gridShader.SetBuffer(addSumsKernel, "gridOffsetBuffer", gridOffsetBuffer);
            
            // Grid shader buffers for rearrangeBoidsKernel
            gridShader.SetBuffer(rearrangeBoidsKernel, "gridBuffer", gridBuffer);
            gridShader.SetBuffer(rearrangeBoidsKernel, "gridOffsetBuffer", gridOffsetBuffer);
            gridShader.SetBuffer(rearrangeBoidsKernel, "boids", boidBuffer);
            gridShader.SetBuffer(rearrangeBoidsKernel, "boidsOut", boidBufferOut);
            
            Debug.Log("Successfully set all compute shader buffers");
        }
        catch (System.Exception e) {
            Debug.LogError("Error setting compute shader buffers: " + e.Message);
        }
    }
    
    // Add a check for sumBlocksKernel buffer setting
    void UpdateGridSumsBuffers(bool swap)
    {
        try {
            gridShader.SetBuffer(sumBlocksKernel, "gridSumsBufferIn", swap ? gridSumsBuffer : gridSumsBuffer2);
            gridShader.SetBuffer(sumBlocksKernel, "gridSumsBuffer", swap ? gridSumsBuffer2 : gridSumsBuffer);
        }
        catch (System.Exception e) {
            Debug.LogError("Error setting sumBlocks buffers: " + e.Message);
        }
    }

// Add a check for addSumsKernel buffer setting
    void UpdateAddSumsBuffer(bool swap)
    {
        try {
            gridShader.SetBuffer(addSumsKernel, "gridSumsBufferIn", swap ? gridSumsBuffer : gridSumsBuffer2);
        }
        catch (System.Exception e) {
            Debug.LogError("Error setting addSums buffer: " + e.Message);
        }
    }



    // Update is called once per frame
    void Update()
    {
        fpsText.text = "FPS: " + (int)(1 / Time.smoothDeltaTime);
        UpdateSelectionText();
        HandleInput();
        VerifyAndSetComputeBuffers();

        if (useGpu)
        {
            boidShader.SetFloat("deltaTime", Time.deltaTime);

            // Update battalion buffer with any changes
            UpdateBattalionBuffer();

            // Clear indices
            gridShader.Dispatch(clearGridKernel, blocks, 1, 1);

            // Populate grid
            gridShader.Dispatch(updateGridKernel, Mathf.CeilToInt(numBoids / blockSize), 1, 1);

            // Generate Offsets (Prefix Sum)
            // Offsets in each block
            gridShader.Dispatch(prefixSumKernel, blocks, 1, 1);

            // Offsets for sums of blocks
            bool swap = false;
            for (int d = 1; d < blocks; d *= 2)
            {
                UpdateGridSumsBuffers(swap);
                gridShader.SetInt("d", d);
                gridShader.Dispatch(sumBlocksKernel, Mathf.CeilToInt(blocks / blockSize), 1, 1);
                swap = !swap;
            }

            // Apply offsets of sums to each block
            UpdateAddSumsBuffer(swap);
            gridShader.Dispatch(addSumsKernel, blocks, 1, 1);

            // Rearrange boids
            gridShader.Dispatch(rearrangeBoidsKernel, Mathf.CeilToInt(numBoids / blockSize), 1, 1);

            // Compute boid behaviours
            boidShader.Dispatch(updateBoidsKernel, Mathf.CeilToInt(numBoids / blockSize), 1, 1);
        }
        else // CPU
        {
            // Spatial grid
            ClearGrid();
            UpdateGrid();
            GenerateGridOffsets();
            RearrangeBoids();

            // Read boids from GPU if needed (for selection)
            if (battalionManager.selectedBattalions.Count > 0 || Input.GetMouseButton(0))
            {
                var readback = AsyncGPUReadback.Request(boidBuffer);
                readback.WaitForCompletion();
                readback.GetData<Boid>().CopyTo(boids);
            }

            for (int i = 0; i < numBoids; i++)
            {
                var boid = boidsTemp[i];
                MergedBehaviours(ref boid);
                LimitSpeed(ref boid);
                KeepInBounds(ref boid);

                // Update boid position
                boid.pos += boid.vel * Time.deltaTime;
                boids[i] = boid;
            }

            // Send data to gpu buffer
            boidBuffer.SetData(boids);
        }

        // Actually draw the boids with battalion colors
        DrawBattalions();
    }

    // Draw boids with different colors by battalion
    // void DrawBattalions()
    // {
    //     foreach (Battalion battalion in battalionManager.battalions)
    //     {
    //         // Set color for this battalion
    //         rp.matProps.SetColor("_Color", battalion.isSelected ? Color.white : battalion.color);
    //         rp.matProps.SetBuffer("boids", boidBuffer);
    //
    //         // Draw just this battalion's boids
    //         Graphics.RenderPrimitives(rp, MeshTopology.Triangles, 3, battalion.count, battalion.startIndex);
    //     }
    // }
    void DrawBattalions()
    {
        // Set up the render parameters once
        rp.matProps.SetBuffer("boids", boidBuffer);
    
        // Render all boids in one call
        Graphics.RenderPrimitives(rp, MeshTopology.Triangles, numBoids * 3);
    }
    
    void UpdateBattalionBuffer()
    {
        if (battalionManager == null || battalionManager.battalions == null || battalionManager.battalions.Count == 0)
        {
            Debug.LogWarning("No battalions to update!");
            return;
        }

        try {
            // Get the exact size using Marshal
            int structSize = Marshal.SizeOf<BattalionData>();
            Debug.Log($"BattalionData struct size: {structSize} bytes");
            
            // Make sure buffer has correct size
            if (battalionBuffer == null || !battalionBuffer.IsValid() || battalionBuffer.stride != structSize)
            {
                if (battalionBuffer != null && battalionBuffer.IsValid())
                    battalionBuffer.Release();
                    
                battalionBuffer = new ComputeBuffer(
                    Mathf.Max(1, battalionManager.battalions.Count),
                    structSize
                );
                
                Debug.Log($"Created new buffer with stride: {battalionBuffer.stride}");
                
                // Need to reassign buffer to shader
                boidShader.SetBuffer(updateBoidsKernel, "battalions", battalionBuffer);
            }
            
            BattalionData[] battalionDataArray = new BattalionData[battalionManager.battalions.Count];
            for (int i = 0; i < battalionManager.battalions.Count; i++)
            {
                Battalion battalion = battalionManager.battalions[i];
                BattalionData data = new BattalionData
                {
                    id = battalion.id,
                    startIndex = battalion.startIndex,
                    count = battalion.count,
                    padding1 = 0,
                    targetPosX = battalion.targetPosition.x,
                    targetPosY = battalion.targetPosition.y,
                    formationSizeX = battalion.formationSize.x,
                    formationSizeY = battalion.formationSize.y,
                    formationType = (int)battalion.formationType,
                    padding2 = 0
                };
                battalionDataArray[i] = data;
            }
        
            Debug.Log($"Setting data for {battalionDataArray.Length} battalions. First battalion: ID={battalionDataArray[0].id}, Start={battalionDataArray[0].startIndex}, Count={battalionDataArray[0].count}");
            
            battalionBuffer.SetData(battalionDataArray);
        }
        catch (System.Exception e) {
            Debug.LogError("Error updating battalion buffer: " + e.Message + "\n" + e.StackTrace);
        }
    }
    
    void UpdateSelectionText()
{
    if (selectionText != null)
    {
        if (battalionManager.selectedBattalions.Count > 0)
        {
            selectionText.text = $"Selected: {battalionManager.selectedBattalions.Count} battalion(s)\n";
            int totalBoids = 0;
            foreach (var battalion in battalionManager.selectedBattalions)
            {
                totalBoids += battalion.count;
            }
            selectionText.text += $"Troops: {totalBoids}";
        }
        else
        {
            selectionText.text = "No battalions selected";
        }
    }
}

// Set formation for selected battalions
void SetFormation(FormationType formationType)
{
    battalionManager.SetFormation(formationType);
}

// Handle mouse and keyboard input
void HandleInput()
{
    // Get mouse position in world space
    Vector2 mousePos = Camera.main.ScreenToWorldPoint(Input.mousePosition);
    currentMousePos = mousePos;

    // Box selection
    if (Input.GetMouseButtonDown(0))
    {
        dragStart = mousePos;
        isDragging = true;
        showSelectionBox = false;

        // Single click battalion selection
        float2[] positions = new float2[numBoids];
        // Get positions from GPU if needed
        if (useGpu)
        {
            var readback = AsyncGPUReadback.Request(boidBuffer);
            readback.WaitForCompletion();
            var boidsData = readback.GetData<Boid>();
            for (int i = 0; i < numBoids; i++)
            {
                positions[i] = boidsData[i].pos;
            }
        }
        else
        {
            for (int i = 0; i < numBoids; i++)
            {
                positions[i] = boids[i].pos;
            }
        }

        battalionManager.SelectBattalionAt(mousePos, positions, Input.GetKey(KeyCode.LeftShift));
    }
    else if (Input.GetMouseButton(0)) // Dragging
    {
        // Show selection box after threshold
        if (isDragging && Vector2.Distance(dragStart, mousePos) > 0.1f)
        {
            showSelectionBox = true;
        }
    }
    else if (Input.GetMouseButtonUp(0)) // Mouse released
    {
        isDragging = false;
        showSelectionBox = false;
        
        // If it was a click (not much dragging), we already handled it in MouseButtonDown
        // Otherwise, handle box selection here
        if (Vector2.Distance(dragStart, mousePos) > 0.1f)
        {
            // Box selection logic would go here
            // For now we'll just use the single click selection
        }
    }

    // Right click to move selected battalions
    if (Input.GetMouseButtonDown(1) && battalionManager.selectedBattalions.Count > 0)
    {
        float2 targetPos = new float2(mousePos.x, mousePos.y);
        battalionManager.MoveSelectedBattalions(targetPos);
    }

    // Keyboard shortcuts for formations
    if (Input.GetKeyDown(KeyCode.L))
        SetFormation(FormationType.Line);
    else if (Input.GetKeyDown(KeyCode.S))
        SetFormation(FormationType.Square);
    else if (Input.GetKeyDown(KeyCode.C))
        SetFormation(FormationType.Column);
    else if (Input.GetKeyDown(KeyCode.W))
        SetFormation(FormationType.Wedge);
    else if (Input.GetKeyDown(KeyCode.R))
        SetFormation(FormationType.Scattered);
    }

    
    void MergedBehaviours(ref Boid boid)
    {
        float2 center = float2.zero;
        float2 battalionCenter = float2.zero;
        float2 close = float2.zero;
        float2 avgVel = float2.zero;
        float2 target = float2.zero;
        int neighbours = 0;
        int battalionNeighbours = 0;

        // Get this boid's battalion
        Battalion battalion = null;
        foreach (Battalion b in battalionManager.battalions)
        {
            if (boid.battalionId == b.id)
            {
                battalion = b;
                break;
            }
        }

        // If we have a battalion, set the target
        if (battalion != null)
        {
            target = battalion.targetPosition;

            // Calculate position in formation
            float2 formationPos = battalionManager.GetBoidFormationPosition(
                Array.IndexOf(boids.ToArray(), boid),
                battalion.targetPosition,
                battalion.formationType,
                battalion.formationSize
            );

            target = formationPos;
        }

        var gridXY = GetGridLocation(boid);
        int gridCell = GetGridIDbyLoc(gridXY);

        for (int y = gridCell - gridDimX; y <= gridCell + gridDimX; y += gridDimX)
        {
            int start = gridOffsets[y - 2];
            int end = gridOffsets[y + 1];
            for (int i = start; i < end; i++)
            {
                Boid other = boidsTemp[i];
                var diff = boid.pos - other.pos;
                var distanceSq = math.dot(diff, diff);
                if (distanceSq > 0 && distanceSq < visualRangeSq)
                {
                    bool sameBattalion = boid.battalionId == other.battalionId;

                    // Apply separation regardless of battalion
                    if (distanceSq < minDistanceSq)
                    {
                        close += diff / distanceSq;
                    }

                    // Stronger cohesion within same battalion
                    if (sameBattalion)
                    {
                        battalionCenter += other.pos;
                        battalionNeighbours++;
                    }

                    // General flocking with all neighbors
                    center += other.pos;
                    avgVel += other.vel;
                    neighbours++;
                }
            }
        }

        // Battalion cohesion - stronger attraction to battalion members
        if (battalionNeighbours > 0)
        {
            battalionCenter /= battalionNeighbours;
            boid.vel += (battalionCenter - boid.pos) * (battalionCohesionFactor * Time.deltaTime);
        }

        // General flocking behaviors
        if (neighbours > 0)
        {
            center /= neighbours;
            avgVel /= neighbours;

            boid.vel += (center - boid.pos) * (cohesionFactor * Time.deltaTime);
            boid.vel += (avgVel - boid.vel) * (alignmentFactor * Time.deltaTime);
        }

        // Move toward target
        if (math.length(target - boid.pos) > 0.1f)
        {
            boid.vel += math.normalize(target - boid.pos) * targetFactor * Time.deltaTime;
        }

        boid.vel += close * (separationFactor * Time.deltaTime);
    }

    void LimitSpeed(ref Boid boid)
    {
        var speed = math.length(boid.vel);
        var clampedSpeed = Mathf.Clamp(speed, minSpeed, maxSpeed);
        boid.vel *= clampedSpeed / speed;
    }

    // Keep boids on screen
    void KeepInBounds(ref Boid boid)
    {
        if (Mathf.Abs(boid.pos.x) > xBound)
        {
            boid.vel.x -= Mathf.Sign(boid.pos.x) * Time.deltaTime * turnSpeed;
        }

        if (Mathf.Abs(boid.pos.y) > yBound)
        {
            boid.vel.y -= Mathf.Sign(boid.pos.y) * Time.deltaTime * turnSpeed;
        }
    }

    int GetGridID(Boid boid)
      {
        int gridX = Mathf.FloorToInt(boid.pos.x / gridCellSize + gridDimX / 2);
        int gridY = Mathf.FloorToInt(boid.pos.y / gridCellSize + gridDimY / 2);
        return (gridDimX * gridY) + gridX;
      }

  int GetGridIDbyLoc(int2 cell)
  {
    return (gridDimX * cell.y) + cell.x;
  }

  int2 GetGridLocation(Boid boid)
  {
    int gridX = Mathf.FloorToInt(boid.pos.x / gridCellSize + gridDimX / 2);
    int gridY = Mathf.FloorToInt(boid.pos.y / gridCellSize + gridDimY / 2);
    return new int2(gridX, gridY);
  }

  void ClearGrid()
  {
    for (int i = 0; i < gridTotalCells; i++)
    {
      gridOffsets[i] = 0;
    }
  }

  void UpdateGrid()
  {
    for (int i = 0; i < numBoids; i++)
    {
      int id = GetGridID(boids[i]);
      var boidGrid = grid[i];
      boidGrid.x = id;
      boidGrid.y = gridOffsets[id];
      grid[i] = boidGrid;
      gridOffsets[id]++;
    }
  }

  void GenerateGridOffsets()
  {
    for (int i = 1; i < gridTotalCells; i++)
    {
      gridOffsets[i] += gridOffsets[i - 1];
    }
  }

  void RearrangeBoids()
  {
    for (int i = 0; i < numBoids; i++)
    {
      int gridID = grid[i].x;
      int cellOffset = grid[i].y;
      int index = gridOffsets[gridID] - 1 - cellOffset;
      boidsTemp[index] = boids[i];
    }
  }

  public void SliderChange(float val)
  {
    numBoids = (int)Mathf.Pow(2, val);
    var limit = useGpu ? gpuLimit : cpuLimit;
    if (numBoids > limit)
    {
      numBoids = limit;
    }
    OnDestroy();
    Start();
  }

  public void ModeChange()
  {
    useGpu = !useGpu;
    modeButton.image.color = useGpu ? Color.green : Color.red;
    modeButton.GetComponentInChildren<Text>().text = useGpu ? "GPU" : "CPU";
    numSlider.maxValue = Mathf.Log(useGpu ? gpuLimit : cpuLimit, 2);

    // WebGPU doesn't like readbacks at the moment
    if (SystemInfo.graphicsDeviceType == GraphicsDeviceType.WebGPU) return;

    if (useGpu) return;
    var readback = AsyncGPUReadback.Request(boidBuffer);
    readback.WaitForCompletion();
    readback.GetData<Boid>().CopyTo(boids);
  }

  public void SwitchTo3D()
  {
    UnityEngine.SceneManagement.SceneManager.LoadScene("Boids3DScene");
  }

  void OnDestroy()
  {
      if (boids.IsCreated)
      {
          boids.Dispose();
          boidsTemp.Dispose();
      }

      if (grid.IsCreated)
      {
          grid.Dispose();
          gridOffsets.Dispose();
      }

      if (boidBuffer != null && boidBuffer.IsValid()) 
          boidBuffer.Release();
      if (boidBufferOut != null && boidBufferOut.IsValid()) 
          boidBufferOut.Release();
      if (gridBuffer != null && gridBuffer.IsValid()) 
          gridBuffer.Release();
      if (gridOffsetBuffer != null && gridOffsetBuffer.IsValid()) 
          gridOffsetBuffer.Release();
      if (gridOffsetBufferIn != null && gridOffsetBufferIn.IsValid()) 
          gridOffsetBufferIn.Release();
      if (gridSumsBuffer != null && gridSumsBuffer.IsValid()) 
          gridSumsBuffer.Release();
      if (gridSumsBuffer2 != null && gridSumsBuffer2.IsValid()) 
          gridSumsBuffer2.Release();
      if (trianglePositions != null && trianglePositions.IsValid()) 
          trianglePositions.Release();
      if (battalionBuffer != null && battalionBuffer.IsValid()) 
          battalionBuffer.Release();
  }

  Vector2[] GetTriangleVerts()
  {
    return new Vector2[] {
      new(-.4f, -.5f),
      new(0, .5f),
      new(.4f, -.5f),
    };
  }
}
