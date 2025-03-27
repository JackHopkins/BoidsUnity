using UnityEngine;
using UnityEngine.UI;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine.Rendering;
using System;
using System.Collections.Generic;

struct Boid
{
  public float2 pos;
  public float2 vel;
  public uint team;
}

struct ObstacleData
{
    public float2 pos;
    public float radius;
    public float strength;
}

struct QuadNode
{
    public float2 center;
    public float size;      // Half-width of the node
    public uint childIndex; // Index of first child (other children are at childIndex+1, +2, +3)
    public uint startIndex; // Start index for boids in this node
    public uint count;      // Number of boids in this node
    public uint flags;      // Bit flags: 1=leaf, 2=active, etc.
}

public class Main2D : MonoBehaviour
{
  const float blockSize = 1024f;
  const int NODE_LEAF = 1;
  const int NODE_ACTIVE = 2;

  [Header("Performance")]
  [SerializeField] int numBoids = 500;
  bool useGpu = false;

  [Header("Settings")]
  [SerializeField] float maxSpeed = 2;
  [SerializeField] float edgeMargin = .5f;
  [SerializeField] float visualRange = .5f;
  float visualRangeSq => visualRange * visualRange;
  [SerializeField] float minDistance = 0.15f;
  float minDistanceSq => minDistance * minDistance;
  [SerializeField] float cohesionFactor = 2;
  [SerializeField] float separationFactor = 1;
  [SerializeField] float alignmentFactor = 5;

  [Header("Obstacles")]
  [SerializeField] private List<Obstacle2D> obstacles = new List<Obstacle2D>();
  [SerializeField] private float obstacleAvoidanceWeight = 5f;
  [SerializeField] ComputeBuffer obstacleBuffer;
  [SerializeField] int maxObstacles = 10;

  [Header("Prefabs")]
  [SerializeField] Text fpsText;
  [SerializeField] Text boidText;
  [SerializeField] Slider numSlider;
  [SerializeField] Button modeButton;
  [SerializeField] ComputeShader boidShader;
  [SerializeField] ComputeShader gridShader;
  [SerializeField] Material boidMat;
  Vector2[] triangleVerts;
  GraphicsBuffer trianglePositions;

  [Header("Quad-Tree Settings")]
  [SerializeField] private bool useQuadTree = false;
  [SerializeField] private int maxQuadTreeDepth = 8;
  [SerializeField] private int maxBoidsPerNode = 32;
  [SerializeField] private int initialQuadTreeSize = 500; // Initial size of the simulation area
  [SerializeField] private ComputeShader quadTreeShader;

  [Header("Team Settings")]
  [SerializeField] float teamRatio = 0.5f; // Ratio of boids in team 0 vs team 1
  [SerializeField] float intraTeamCohesionMultiplier = 2.5f; // Stronger cohesion within same team
  [SerializeField] float interTeamRepulsionMultiplier = 2.5f; // Stronger repulsion between different teams
  [SerializeField] Color team0Color = Color.blue;
  [SerializeField] Color team1Color = Color.red;

  float minSpeed;
  float turnSpeed;

  NativeArray<Boid> boids;
  NativeArray<Boid> boidsTemp;

  int updateBoidsKernel, generateBoidsKernel;
  int updateGridKernel, clearGridKernel, prefixSumKernel, sumBlocksKernel, addSumsKernel, rearrangeBoidsKernel;
  int blocks;
  int countBoidsKernel, sumNodeCountsKernel, updateNodeCountsKernel, redistributeBoidsKernel, clearNodeCountsKernel;

  ComputeBuffer boidBuffer;
  ComputeBuffer boidBufferOut;
  ComputeBuffer gridBuffer;
  ComputeBuffer gridOffsetBuffer;
  ComputeBuffer gridOffsetBufferIn;
  ComputeBuffer gridSumsBuffer;
  ComputeBuffer gridSumsBuffer2;

  private ComputeBuffer quadNodesBuffer;
  private ComputeBuffer nodeCountBuffer;       // Counter for nodes
  private ComputeBuffer boidIndicesBuffer;     // For storing sorted boid indices
  private ComputeBuffer activeNodesBuffer;     // List of active nodes
  private ComputeBuffer activeNodeCountBuffer; // Counter for active nodes
  private ComputeBuffer nodeCountsBuffer; // Dedicated buffer for atomic counting


  private int clearQuadTreeKernel;
  private int insertBoidsKernel;
  private int buildActiveNodesKernel;
  private int sortBoidsKernel;
  private int subdivideNodesKernel;

  private const int MaxQuadNodes = 16384; // Maximum number of quad-tree nodes

  // Index is particle ID, x value is position flattened to 1D array, y value is grid cell offset
  NativeArray<int2> grid;
  NativeArray<int> gridOffsets;
  int gridDimY, gridDimX, gridTotalCells;
  float gridCellSize;

  float xBound, yBound;
  RenderParams rp;

  readonly int cpuLimit = 1 << 16;
  readonly int gpuLimit = (int)blockSize * 65535;

  void Awake()
  {
    numSlider.maxValue = Mathf.Log(useGpu ? gpuLimit : cpuLimit, 2);
    triangleVerts = GetTriangleVerts();
  }

	// Add this method to initialize obstacles in Start() after existing initialization
  private void InitializeObstacles()
  {
    // Find all obstacles in the scene if not set in inspector
    if (obstacles.Count == 0)
    {
        obstacles.AddRange(FindObjectsOfType<Obstacle2D>());
    }
    
    // Create the obstacle buffer for GPU processing
    if (useGpu)
    {
        // Make sure we have at least one element in the buffer to avoid errors
        obstacleBuffer = new ComputeBuffer(Mathf.Max(1, obstacles.Count), 16); // 16 bytes per obstacle
        var obstacleData = new ObstacleData[Mathf.Max(1, obstacles.Count)];
        
        for (int i = 0; i < obstacles.Count; i++)
        {
            obstacleData[i] = new ObstacleData
            {
                pos = new float2(obstacles[i].transform.position.x, obstacles[i].transform.position.y),
                radius = obstacles[i].repulsionRadius,
                strength = obstacles[i].repulsionStrength
            };
        }
        
        obstacleBuffer.SetData(obstacleData);
        
        // Set the obstacle buffer to the compute shader
        boidShader.SetBuffer(updateBoidsKernel, "obstacles", obstacleBuffer);
        boidShader.SetInt("numObstacles", obstacles.Count);
        boidShader.SetFloat("obstacleAvoidanceWeight", obstacleAvoidanceWeight);
    }
  }

private void UpdateQuadTree()
{
    if (!useQuadTree || quadTreeShader == null) return;

    // Debug output
    if (Time.frameCount % 120 == 0) {
        Debug.Log($"Frame {Time.frameCount}: Updating quad tree");
        
        uint[] nodeCountDebug = new uint[1];
        nodeCountBuffer.GetData(nodeCountDebug);
        Debug.Log($"Current node count: {nodeCountDebug[0]}");
    }

    // Step 1: Clear and initialize quad-tree
    quadTreeShader.Dispatch(clearQuadTreeKernel, 1, 1, 1);
    
    // Clear node counts before we start
    quadTreeShader.Dispatch(clearNodeCountsKernel, Mathf.CeilToInt(MaxQuadNodes / blockSize), 1, 1);

    // Step 2: Activate the root node
    int activateRootKernel = quadTreeShader.FindKernel("ActivateRoot");
    quadTreeShader.Dispatch(activateRootKernel, 1, 1, 1);

    // Step 3: Insert boids into root node (updates nodeCounts buffer)
    quadTreeShader.SetBuffer(insertBoidsKernel, "boids", boidBuffer);
    quadTreeShader.Dispatch(insertBoidsKernel, Mathf.CeilToInt(numBoids / blockSize), 1, 1);

    // Step 4: Update the quadNodes with counts from nodeCounts buffer
    quadTreeShader.Dispatch(updateNodeCountsKernel, Mathf.CeilToInt(MaxQuadNodes / blockSize), 1, 1);

    // Debug - check if root has boids
    if (Time.frameCount % 120 == 0) {
        QuadNode[] rootNode = new QuadNode[1];
        quadNodesBuffer.GetData(rootNode, 0, 0, 1);
        Debug.Log($"Root node count: {rootNode[0].count}, flags: {rootNode[0].flags}");
    }

    // Step 5: Iterative subdivision and redistribution
    for (int i = 0; i < maxQuadTreeDepth; i++)
    {
        // Subdivide nodes that need it
        quadTreeShader.Dispatch(subdivideNodesKernel, Mathf.CeilToInt(MaxQuadNodes / blockSize), 1, 1);
        
        // Redistribute boids to child nodes
        quadTreeShader.Dispatch(redistributeBoidsKernel, Mathf.CeilToInt(numBoids / blockSize), 1, 1);
        
        // Update node counts
        quadTreeShader.Dispatch(updateNodeCountsKernel, Mathf.CeilToInt(MaxQuadNodes / blockSize), 1, 1);
        
        // Debug check - only periodically to avoid spam
        if (Time.frameCount % 120 == 0 && i == maxQuadTreeDepth - 1) {
            uint[] activeNodeCountDebug = new uint[1];
            activeNodeCountBuffer.GetData(activeNodeCountDebug);
            Debug.Log($"Active nodes after iteration {i}: {activeNodeCountDebug[0]}");
            
            // Check if we have child nodes
            if (activeNodeCountDebug[0] > 1) {
                uint[] firstFewActiveNodes = new uint[Math.Min(5, (int)activeNodeCountDebug[0])];
                activeNodesBuffer.GetData(firstFewActiveNodes, 0, 0, firstFewActiveNodes.Length);
                
                string nodeStr = string.Join(", ", firstFewActiveNodes);
                Debug.Log($"First few active nodes: {nodeStr}");
                
                // Check a sample of nodes
                QuadNode[] sampleNodes = new QuadNode[firstFewActiveNodes.Length];
                for (int n = 0; n < firstFewActiveNodes.Length; n++) {
                    uint nodeIdx = firstFewActiveNodes[n];
                    quadNodesBuffer.GetData(sampleNodes, (int)nodeIdx, (int)nodeIdx, 1);
                }
                
                // Log sample node info
                for (int n = 0; n < sampleNodes.Length; n++) {
                    Debug.Log($"Node {firstFewActiveNodes[n]}: Count={sampleNodes[n].count}, " +
                             $"StartIndex={sampleNodes[n].startIndex}, Flags={sampleNodes[n].flags}");
                }
            }
        }
    }

    // Step 6: Sort boids according to the quadtree
    quadTreeShader.Dispatch(sortBoidsKernel, Mathf.CeilToInt(numBoids / blockSize), 1, 1);

    // Swap buffers
    var temp = boidBuffer;
    boidBuffer = boidBufferOut;
    boidBufferOut = temp;

    // Update shader buffers
    boidShader.SetBuffer(updateBoidsKernel, "boidsIn", boidBuffer);
    boidShader.SetBuffer(updateBoidsKernel, "boidsOut", boidBufferOut);

    // Use quadtree for boid calculations
    boidShader.SetInt("useQuadTree", 1);
    boidShader.Dispatch(updateBoidsKernel, Mathf.CeilToInt(numBoids / blockSize), 1, 1);
}

  // Start is called before the first frame update
  void Start()
  {
    // Zoom camera based on number of boids
    Camera.main.orthographicSize = Mathf.Max(2, Mathf.Sqrt(numBoids) / 10 + edgeMargin);
    Camera.main.transform.position = new Vector3(0, 0, -10);
    GetComponent<MoveCamera2D>().Start();

    boidText.text = "Boids: " + numBoids;
    xBound = Camera.main.orthographicSize * Camera.main.aspect - edgeMargin;
    yBound = Camera.main.orthographicSize - edgeMargin;
    turnSpeed = maxSpeed * 3;
    minSpeed = maxSpeed * 0.75f;


    // Get kernel IDs
    updateBoidsKernel = boidShader.FindKernel("UpdateBoids");
    generateBoidsKernel = boidShader.FindKernel("GenerateBoids");
    updateGridKernel = gridShader.FindKernel("UpdateGrid");
    clearGridKernel = gridShader.FindKernel("ClearGrid");
    prefixSumKernel = gridShader.FindKernel("PrefixSum");
    sumBlocksKernel = gridShader.FindKernel("SumBlocks");
    addSumsKernel = gridShader.FindKernel("AddSums");
    rearrangeBoidsKernel = gridShader.FindKernel("RearrangeBoids");
    countBoidsKernel = quadTreeShader.FindKernel("CountBoids");
    sumNodeCountsKernel = quadTreeShader.FindKernel("SumNodeCounts");

  
	InitializeObstacles();

    // Setup compute buffer
    boidBuffer = new ComputeBuffer(numBoids, 20);
    boidBufferOut = new ComputeBuffer(numBoids, 20);
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

	boidShader.SetFloat("teamRatio", teamRatio);
	boidShader.SetFloat("intraTeamCohesionMultiplier", intraTeamCohesionMultiplier);
	boidShader.SetFloat("interTeamRepulsionMultiplier", interTeamRepulsionMultiplier);
    

	// Set up team colours
	boidMat.SetColor("_Team0Color", team0Color);
	boidMat.SetColor("_Team1Color", team1Color);

    // Generate boids on GPU if over CPU limit
    if (numBoids <= cpuLimit)
    {
      // Populate initial boids
      boids = new NativeArray<Boid>(numBoids, Allocator.Persistent);
      boidsTemp = new NativeArray<Boid>(numBoids, Allocator.Persistent);
      for (int i = 0; i < numBoids; i++)
      {
        var pos = new float2(UnityEngine.Random.Range(-xBound, xBound), UnityEngine.Random.Range(-yBound, yBound));
        var vel = new float2(UnityEngine.Random.Range(-maxSpeed, maxSpeed), UnityEngine.Random.Range(-maxSpeed, maxSpeed));
        var boid = new Boid();
        boid.pos = pos;
        boid.vel = vel;
		boid.team = (uint)(i < numBoids * teamRatio ? 0 : 1); // Assign team based on ratio
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

	if (quadTreeShader != null)
    {
		Debug.LogWarning("Quad Tree Shader is not null.");
        // Find kernels
        clearQuadTreeKernel = quadTreeShader.FindKernel("ClearQuadTree");
        insertBoidsKernel = quadTreeShader.FindKernel("InsertBoids");
        buildActiveNodesKernel = quadTreeShader.FindKernel("BuildActiveNodes");
        sortBoidsKernel = quadTreeShader.FindKernel("SortBoids");
		subdivideNodesKernel = quadTreeShader.FindKernel("SubdivideNodes");
        
        clearNodeCountsKernel = quadTreeShader.FindKernel("ClearNodeCounts");
        redistributeBoidsKernel = quadTreeShader.FindKernel("RedistributeBoids");
        updateNodeCountsKernel = quadTreeShader.FindKernel("UpdateNodeCounts");


        
        // Create buffers
        quadNodesBuffer = new ComputeBuffer(MaxQuadNodes, 28); // 28 bytes per node
        nodeCountBuffer = new ComputeBuffer(1, 4);
        boidIndicesBuffer = new ComputeBuffer(numBoids * 2, 4);
        activeNodesBuffer = new ComputeBuffer(MaxQuadNodes, 4);
        activeNodeCountBuffer = new ComputeBuffer(1, 4);
        
        // Set initial parameters
        quadTreeShader.SetInt("maxDepth", maxQuadTreeDepth);
        quadTreeShader.SetInt("maxBoidsPerNode", maxBoidsPerNode);

        quadTreeShader.SetInt("numBoids", numBoids);
        quadTreeShader.SetFloat("worldSize", initialQuadTreeSize);
		    

        
		uint[] initialNodeCount = new uint[] { 1 }; // Start with 1 node (the root)
		nodeCountBuffer.SetData(initialNodeCount);

		uint[] initialActiveCount = new uint[] { 1 }; // Start with 1 active node (the root)
		activeNodeCountBuffer.SetData(initialActiveCount);

		QuadNode[] initialNodes = new QuadNode[MaxQuadNodes];
        // Initialize root
        initialNodes[0] = new QuadNode { 
            center = new float2(0, 0),
            size = initialQuadTreeSize,
            childIndex = 0,
            startIndex = 0,
            count = 0,
            flags = NODE_LEAF | NODE_ACTIVE
        };
        quadNodesBuffer.SetData(initialNodes);

        uint[] initialCounts = new uint[] { 1 }; // Start with 1 node
        nodeCountBuffer.SetData(initialCounts);

        uint[] initialActive = new uint[] { 1 }; // Start with 1 active node
        activeNodeCountBuffer.SetData(initialActive);

        uint[] initialActiveNodes = new uint[MaxQuadNodes];
        initialActiveNodes[0] = 0; // First active node is the root
        activeNodesBuffer.SetData(initialActiveNodes);

        nodeCountsBuffer = new ComputeBuffer(MaxQuadNodes, 4); // 4 bytes (uint) per node


        // Set buffers
        quadTreeShader.SetBuffer(countBoidsKernel, "boids", boidBuffer);
        quadTreeShader.SetBuffer(countBoidsKernel, "boidIndices", boidIndicesBuffer);

        quadTreeShader.SetBuffer(sumNodeCountsKernel, "quadNodes", quadNodesBuffer);
        quadTreeShader.SetBuffer(sumNodeCountsKernel, "boidIndices", boidIndicesBuffer);
        quadTreeShader.SetInt("numBoids", numBoids);
        
        // Set buffers
		quadTreeShader.SetBuffer(subdivideNodesKernel, "quadNodes", quadNodesBuffer);
		quadTreeShader.SetBuffer(subdivideNodesKernel, "nodeCount", nodeCountBuffer);
		quadTreeShader.SetBuffer(subdivideNodesKernel, "activeNodes", activeNodesBuffer);
		quadTreeShader.SetBuffer(subdivideNodesKernel, "activeNodeCount", activeNodeCountBuffer);

        quadTreeShader.SetBuffer(clearQuadTreeKernel, "quadNodes", quadNodesBuffer);
        quadTreeShader.SetBuffer(clearQuadTreeKernel, "nodeCount", nodeCountBuffer);
		quadTreeShader.SetBuffer(clearQuadTreeKernel, "activeNodes", activeNodesBuffer);
        quadTreeShader.SetBuffer(clearQuadTreeKernel, "activeNodeCount", activeNodeCountBuffer);
        
        quadTreeShader.SetBuffer(insertBoidsKernel, "boids", boidBuffer);
        quadTreeShader.SetBuffer(insertBoidsKernel, "quadNodes", quadNodesBuffer);
        quadTreeShader.SetBuffer(insertBoidsKernel, "nodeCount", nodeCountBuffer);
		quadTreeShader.SetBuffer(insertBoidsKernel, "boidIndices", boidIndicesBuffer);
        
        quadTreeShader.SetBuffer(buildActiveNodesKernel, "quadNodes", quadNodesBuffer);
        quadTreeShader.SetBuffer(buildActiveNodesKernel, "nodeCount", nodeCountBuffer);
        quadTreeShader.SetBuffer(buildActiveNodesKernel, "activeNodes", activeNodesBuffer);
        quadTreeShader.SetBuffer(buildActiveNodesKernel, "activeNodeCount", activeNodeCountBuffer);
        
        quadTreeShader.SetBuffer(sortBoidsKernel, "boids", boidBuffer);
        quadTreeShader.SetBuffer(sortBoidsKernel, "boidsOut", boidBufferOut);
        quadTreeShader.SetBuffer(sortBoidsKernel, "quadNodes", quadNodesBuffer);
        quadTreeShader.SetBuffer(sortBoidsKernel, "boidIndices", boidIndicesBuffer);
		quadTreeShader.SetBuffer(sortBoidsKernel, "nodeCount", nodeCountBuffer);
            
        quadTreeShader.SetBuffer(redistributeBoidsKernel, "boids", boidBuffer);
        quadTreeShader.SetBuffer(redistributeBoidsKernel, "boidIndices", boidIndicesBuffer);
        quadTreeShader.SetBuffer(redistributeBoidsKernel, "quadNodes", quadNodesBuffer);

        quadTreeShader.SetBuffer(updateNodeCountsKernel, "nodeCounts", nodeCountsBuffer);
        quadTreeShader.SetBuffer(updateNodeCountsKernel, "quadNodes", quadNodesBuffer);
        quadTreeShader.SetBuffer(updateNodeCountsKernel, "nodeCount", nodeCountBuffer);

        quadTreeShader.SetBuffer(clearNodeCountsKernel, "nodeCount", nodeCountsBuffer);
    
    
   

        // Set boid shader to use quad-tree
        boidShader.SetBuffer(updateBoidsKernel, "quadNodes", quadNodesBuffer);
        boidShader.SetBuffer(updateBoidsKernel, "activeNodes", activeNodesBuffer);
        boidShader.SetBuffer(updateBoidsKernel, "boidIndices", boidIndicesBuffer);
		boidShader.SetBuffer(updateBoidsKernel, "nodeCount", nodeCountBuffer);
        boidShader.SetInt("useQuadTree", useQuadTree ? 1 : 0);

    

		uint[] debugNodeCount = new uint[1];
		nodeCountBuffer.GetData(debugNodeCount);
		Debug.Log($"Initial node count: {debugNodeCount[0]}");

		uint[] debugActiveCount = new uint[1];
		activeNodeCountBuffer.GetData(debugActiveCount);
		Debug.Log($"Initial active node count: {debugActiveCount[0]}");
    }
		
	Debug.Log($"Max boids per node: {maxBoidsPerNode}");

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
    gridDimX = Mathf.FloorToInt(xBound * 2 / gridCellSize) + 30;
    gridDimY = Mathf.FloorToInt(yBound * 2 / gridCellSize) + 30;
    gridTotalCells = gridDimX * gridDimY;

    // Don't generate grid on CPU if over CPU limit
    if (numBoids <= cpuLimit)
    {
      grid = new NativeArray<int2>(numBoids, Allocator.Persistent);
      gridOffsets = new NativeArray<int>(gridTotalCells, Allocator.Persistent);
    }

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


  }
	
private void UpdateObstacleData()
{
    if (useGpu && obstacles.Count > 0)
    {
        var obstacleData = new ObstacleData[obstacles.Count];
        
        for (int i = 0; i < obstacles.Count; i++)
        {
            if (obstacles[i] != null)
            {
                obstacleData[i] = new ObstacleData
                {
                    pos = new float2(obstacles[i].transform.position.x, obstacles[i].transform.position.y),
                    radius = obstacles[i].repulsionRadius,
                    strength = obstacles[i].repulsionStrength
                };
            }
        }
        
        obstacleBuffer.SetData(obstacleData);
    }
}

private void CheckBuffersValid()
{
    Debug.Log("Checking buffers:");
    Debug.Log($"quadNodesBuffer: {(quadNodesBuffer != null ? "Valid" : "NULL")}");
    Debug.Log($"nodeCountBuffer: {(nodeCountBuffer != null ? "Valid" : "NULL")}");
    Debug.Log($"boidIndicesBuffer: {(boidIndicesBuffer != null ? "Valid" : "NULL")}");
    Debug.Log($"activeNodesBuffer: {(activeNodesBuffer != null ? "Valid" : "NULL")}");
    Debug.Log($"activeNodeCountBuffer: {(activeNodeCountBuffer != null ? "Valid" : "NULL")}");
    Debug.Log($"boidBuffer: {(boidBuffer != null ? "Valid" : "NULL")}");
    Debug.Log($"boidBufferOut: {(boidBufferOut != null ? "Valid" : "NULL")}");
}

// Add this method to Main2D.cs to debug the quad-tree
void OnDrawGizmos()
{
    if (!Application.isPlaying || !useQuadTree) return;
    
    // Read the entire quad tree structure
    QuadNode[] allNodes = new QuadNode[MaxQuadNodes];
    quadNodesBuffer.GetData(allNodes);
    
    // Read node count to know how many nodes exist
    uint[] counts = new uint[1];
    nodeCountBuffer.GetData(counts);
    uint nodeCount = counts[0];

	uint[] activeNodeCount = new uint[1];
	activeNodeCountBuffer.GetData(activeNodeCount);
	Debug.Log($"Active node count: {activeNodeCount[0]}");

	if (activeNodeCount[0] > 0) {
    	uint[] activeNodeData = new uint[Mathf.Min(10, (int)activeNodeCount[0])];
    	activeNodesBuffer.GetData(activeNodeData);
    	Debug.Log($"First active node: {activeNodeData[0]}");
	}
    
    Debug.Log($"Total quad tree nodes: {nodeCount}");
    
    // Visualize all nodes up to a certain depth
    DrawQuadNodeWithInfo(allNodes, 0, 0, Color.green);
}

void DrawQuadNodeWithInfo(QuadNode[] allNodes, uint nodeIndex, int depth, Color color)
{
    if (depth > 4 || nodeIndex >= allNodes.Length) return;
    
    QuadNode node = allNodes[nodeIndex];
    
    // Draw this node
    Gizmos.color = color;
    Vector3 center = new Vector3(node.center.x, node.center.y, 0);
    Vector3 size = new Vector3(node.size * 2, node.size * 2, 0.1f);
    Gizmos.DrawWireCube(center, size);
    
    // Display node info
    bool isLeaf = (node.flags & 1) != 0;
    int boidCount = (int)node.count;
    
    // Color code based on number of boids
    if (boidCount > 0)
    {
        // Use a color gradient based on boid count
        float intensity = Mathf.Min(1.0f, boidCount / (float)maxBoidsPerNode);
        Color fillColor = new Color(intensity, 0, 1-intensity, 0.2f);
        Gizmos.color = fillColor;
        Gizmos.DrawCube(center, size);
        
        // Show node counts as text in the scene
        UnityEditor.Handles.Label(center, $"Node {nodeIndex}: {boidCount} boids\nLeaf: {isLeaf}");
    }
    
    // Draw children if not a leaf
    if (!isLeaf && node.childIndex > 0)
    {
        // Alternate colors for child nodes
        Color childColor = new Color(color.r * 0.8f, color.g * 0.8f, color.b * 0.8f);
        for (uint i = 0; i < 4; i++)
        {
            DrawQuadNodeWithInfo(allNodes, node.childIndex + i, depth + 1, childColor);
        }
    }
}

  // Update is called once per frame
  void Update()
  {
    fpsText.text = "FPS: " + (int)(1 / Time.smoothDeltaTime);

    if (useGpu)
    {
        boidShader.SetFloat("deltaTime", Time.deltaTime);
        UpdateObstacleData();
        
        if (useQuadTree && quadTreeShader != null)
        {	
			UpdateQuadTree();
        }
        else
        {
            boidShader.SetFloat("deltaTime", Time.deltaTime);
    
            // Can be called here to update the positions (CALL SPARINGLY)
            UpdateObstacleData();
			
			boidShader.SetInt("useQuadTree", 0);
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
                gridShader.SetBuffer(sumBlocksKernel, "gridSumsBufferIn", swap ? gridSumsBuffer : gridSumsBuffer2);
                gridShader.SetBuffer(sumBlocksKernel, "gridSumsBuffer", swap ? gridSumsBuffer2 : gridSumsBuffer);
                gridShader.SetInt("d", d);
                gridShader.Dispatch(sumBlocksKernel, Mathf.CeilToInt(blocks / blockSize), 1, 1);
                swap = !swap;
            }

            // Apply offsets of sums to each block
            gridShader.SetBuffer(addSumsKernel, "gridSumsBufferIn", swap ? gridSumsBuffer : gridSumsBuffer2);
            gridShader.Dispatch(addSumsKernel, blocks, 1, 1);


            // Rearrange boids
            gridShader.Dispatch(rearrangeBoidsKernel, Mathf.CeilToInt(numBoids / blockSize), 1, 1);

            // Compute boid behaviours
            boidShader.Dispatch(updateBoidsKernel, Mathf.CeilToInt(numBoids / blockSize), 1, 1);
        }
    }
    else // CPU
    {
      // Spatial grid
      ClearGrid();
      UpdateGrid();
      GenerateGridOffsets();
      RearrangeBoids();

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

    // Actually draw the boids
    Graphics.RenderPrimitives(rp, MeshTopology.Triangles, numBoids * 3);
  }

  // In Main2D.cs, MergedBehaviours method
	void MergedBehaviours(ref Boid boid)
	{
    float2 center = float2.zero;
    float2 close = float2.zero;
    float2 avgVel = float2.zero;
    int sameTeamNeighbours = 0;
    int otherTeamNeighbours = 0;

    var gridXY = GetGridLocation(boid);
    int gridCell = GetGridIDbyLoc(gridXY);

    for (int y = gridCell - gridDimX; y <= gridCell + gridDimX; y += gridDimX)
    {
        // Check bounds to avoid index errors
        if (y < 0 || y >= gridOffsets.Length - 2) continue;
        
        int start = gridOffsets[y - 2];
        int end = gridOffsets[y + 1];
        for (int i = start; i < end; i++)
        {
            Boid other = boidsTemp[i];
            var diff = boid.pos - other.pos;
            var distanceSq = math.dot(diff, diff);
            if (distanceSq > 0 && distanceSq < visualRangeSq)
            {
                bool sameTeam = boid.team == other.team;
                
                if (distanceSq < minDistanceSq)
                {
                    float repulsionStrength = sameTeam ? 1.0f : interTeamRepulsionMultiplier;
                    close += diff / distanceSq * repulsionStrength;
                }
                
                if (sameTeam)
                {
                    center += other.pos;
                    avgVel += other.vel;
                    sameTeamNeighbours++;
                }
                else
                {
                    otherTeamNeighbours++;
                }
            }
        }
    }

    if (sameTeamNeighbours > 0)
    {
        center /= sameTeamNeighbours;
        avgVel /= sameTeamNeighbours;

        // Apply stronger cohesion with same team
        boid.vel += (center - boid.pos) * (cohesionFactor * intraTeamCohesionMultiplier * Time.deltaTime);
        boid.vel += (avgVel - boid.vel) * (alignmentFactor * Time.deltaTime);
    }

    boid.vel += close * (separationFactor * Time.deltaTime);
    
    // Add obstacle avoidance
    float2 obstacleForce = float2.zero;
    
    if (obstacles != null && obstacles.Count > 0)
    {
        foreach (var obstacle in obstacles)
        {
            if (obstacle != null)
            {
                obstacleForce += obstacle.GetRepulsionForce(boid.pos);
            }
        }
        
        boid.vel += obstacleForce * obstacleAvoidanceWeight * Time.deltaTime;
    }
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
    // First check if native arrays are created before trying to dispose them
    if (boids.IsCreated)
    {
        boids.Dispose();
    }
    
    if (boidsTemp.IsCreated)
    {
        boidsTemp.Dispose();
    }

    if (grid.IsCreated)
    {
        grid.Dispose();
    }
    
    if (gridOffsets.IsCreated)
    {
        gridOffsets.Dispose();
    }

    // Release all compute buffers if they exist
    SafeReleaseBuffer(ref boidBuffer);
    SafeReleaseBuffer(ref boidBufferOut);
    SafeReleaseBuffer(ref gridBuffer);
    SafeReleaseBuffer(ref gridOffsetBuffer);
    SafeReleaseBuffer(ref gridOffsetBufferIn);
    SafeReleaseBuffer(ref gridSumsBuffer);
    SafeReleaseBuffer(ref gridSumsBuffer2);
    SafeReleaseBuffer(ref obstacleBuffer);

	// Clean up quad-tree resources
	SafeReleaseBuffer(ref quadNodesBuffer);
	SafeReleaseBuffer(ref nodeCountBuffer);
	SafeReleaseBuffer(ref boidIndicesBuffer);
	SafeReleaseBuffer(ref activeNodesBuffer);
	SafeReleaseBuffer(ref activeNodeCountBuffer);

    trianglePositions.Release();
}

// Add this helper method for safely releasing compute buffers
private void SafeReleaseBuffer(ref ComputeBuffer buffer)
{
    if (buffer != null)
    {
        buffer.Release();
        buffer = null;
    }
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


