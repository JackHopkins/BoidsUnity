using UnityEngine;
using UnityEngine.UI;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine.Rendering;
using System;
using System.Collections.Generic;

namespace BoidsUnity
{
    public class Main2D : MonoBehaviour
    {
        const float blockSize = 256f;
        
        [Header("Performance")]
        [SerializeField] int numBoids = 500;
        bool useGpu = false;
        
        [Header("Settings")]
        [SerializeField] float maxSpeed = 2;
        [SerializeField] float edgeMargin = .5f;
        [SerializeField] float visualRange = .5f;
        [SerializeField] float minDistance = 0.15f;
        [SerializeField] float cohesionFactor = 2;
        [SerializeField] float separationFactor = 1;
        [SerializeField] float alignmentFactor = 5;
        
        [Header("Obstacles")]
        [SerializeField] private List<Obstacle2D> obstacles = new List<Obstacle2D>();
        [SerializeField] private float obstacleAvoidanceWeight = 5f;
        [SerializeField] int maxObstacles = 10;
        
        [Header("Quad-Tree Settings")]
        [SerializeField] private bool useQuadTree = false;
        [SerializeField] private int maxQuadTreeDepth = 8;
        [SerializeField] private int maxBoidsPerNode = 32;
        [SerializeField] private int initialQuadTreeSize = 500; 
        [SerializeField] private bool drawQuadTreeGizmos = false;
        [SerializeField] private int gizmoDetailLevel = 2;
        
        [Header("Team Settings")]
        [SerializeField] float teamRatio = 0.5f;
        [SerializeField] float intraTeamCohesionMultiplier = 2.5f;
        [SerializeField] float interTeamRepulsionMultiplier = 2.5f;
        [SerializeField] Color team0Color = Color.blue;
        [SerializeField] Color team1Color = Color.red;
        
        [Header("Prefabs")]
        [SerializeField] Text fpsText;
        [SerializeField] Text boidText;
        [SerializeField] Slider numSlider;
        [SerializeField] Button modeButton;
        [SerializeField] ComputeShader boidShader;
        [SerializeField] ComputeShader gridShader;
        [SerializeField] ComputeShader quadTreeShader;
        [SerializeField] Material boidMat;
        
        // Component managers
        private BoidBehaviorManager boidBehaviorManager;
        private GridManager gridManager;
        private QuadTreeManager quadTreeManager;
        private ObstacleManager obstacleManager;
        private BoidRenderer boidRenderer;
        
        // Boid data
        private NativeArray<Boid> boids;
        private NativeArray<Boid> boidsTemp;
        private ComputeBuffer boidBuffer;
        private ComputeBuffer boidBufferOut;
        
        // Compute shader kernels
        int updateBoidsKernel, generateBoidsKernel;
        
        // Bounds
        float xBound, yBound;
        
        // Limits
        readonly int cpuLimit = 1 << 16;
        readonly int gpuLimit = (int)blockSize * 65535;
        
        void Awake()
        {
            numSlider.maxValue = Mathf.Log(useGpu ? gpuLimit : cpuLimit, 2);
        }
        
        void Start()
        {
            // Initialize camera
            Camera.main.orthographicSize = Mathf.Max(2, Mathf.Sqrt(numBoids) / 10 + edgeMargin);
            Camera.main.transform.position = new Vector3(0, 0, -10);
            GetComponent<MoveCamera2D>().Start();
            
            // Initialize UI
            boidText.text = "Boids: " + numBoids;
            
            // Calculate bounds
            xBound = Camera.main.orthographicSize * Camera.main.aspect - edgeMargin;
            yBound = Camera.main.orthographicSize - edgeMargin;
            
            // Create and initialize component managers
            InitializeManagers();
            
            // Get kernel IDs
            updateBoidsKernel = boidShader.FindKernel("UpdateBoids");
            generateBoidsKernel = boidShader.FindKernel("GenerateBoids");
            
            // Initialize boid buffers
            InitializeBoidBuffers();
            
            // Initialize component subsystems
            obstacleManager.InitializeObstacles(boidShader, updateBoidsKernel, useGpu);
            gridManager.Initialize(numBoids, visualRange, xBound, yBound, boidBuffer, boidBufferOut);
            quadTreeManager.Initialize(numBoids, boidBuffer, boidBufferOut, boidShader);
            boidRenderer.Initialize(boidBuffer, team0Color, team1Color);
            
            // Set boid shader parameters
            SetBoidShaderParameters();
        }
        
        private void InitializeManagers()
        {
            // Create managers
            boidBehaviorManager = new BoidBehaviorManager
            {
                maxSpeed = maxSpeed,
                edgeMargin = edgeMargin,
                visualRange = visualRange,
                minDistance = minDistance,
                cohesionFactor = cohesionFactor,
                separationFactor = separationFactor,
                alignmentFactor = alignmentFactor,
                teamRatio = teamRatio,
                intraTeamCohesionMultiplier = intraTeamCohesionMultiplier,
                interTeamRepulsionMultiplier = interTeamRepulsionMultiplier,
                team0Color = team0Color,
                team1Color = team1Color,
                obstacleAvoidanceWeight = obstacleAvoidanceWeight,
                obstacles = obstacles
            };
            boidBehaviorManager.InitializeBounds(Camera.main.orthographicSize, edgeMargin);
            
            gridManager = new GridManager(gridShader, boidBehaviorManager);
            
            quadTreeManager = new QuadTreeManager(quadTreeShader)
            {
                useQuadTree = useQuadTree,
                maxQuadTreeDepth = maxQuadTreeDepth,
                maxBoidsPerNode = maxBoidsPerNode,
                initialQuadTreeSize = initialQuadTreeSize,
                drawQuadTreeGizmos = drawQuadTreeGizmos,
                gizmoDetailLevel = gizmoDetailLevel
            };
            
            obstacleManager = new ObstacleManager(obstacleAvoidanceWeight, maxObstacles);
            
            boidRenderer = new BoidRenderer(boidMat);
        }
        
        private void InitializeBoidBuffers()
        {
            // Setup compute buffers
            boidBuffer = new ComputeBuffer(numBoids, 20);
            boidBufferOut = new ComputeBuffer(numBoids, 20);
            
            // Generate boids on CPU or GPU based on limit
            if (numBoids <= cpuLimit)
            {
                // Populate initial boids on CPU
                boids = new NativeArray<Boid>(numBoids, Allocator.Persistent);
                boidsTemp = new NativeArray<Boid>(numBoids, Allocator.Persistent);
                
                for (int i = 0; i < numBoids; i++)
                {
                    var pos = new float2(UnityEngine.Random.Range(-xBound, xBound), UnityEngine.Random.Range(-yBound, yBound));
                    var vel = new float2(UnityEngine.Random.Range(-maxSpeed, maxSpeed), UnityEngine.Random.Range(-maxSpeed, maxSpeed));
                    var boid = new Boid
                    {
                        pos = pos,
                        vel = vel,
                        team = (uint)(i < numBoids * teamRatio ? 0 : 1) // Assign team based on ratio
                    };
                    boids[i] = boid;
                }
                boidBuffer.SetData(boids);
            }
            else
            {
                // Generate boids on GPU
                boidShader.SetBuffer(generateBoidsKernel, "boidsOut", boidBuffer);
                boidShader.SetInt("randSeed", UnityEngine.Random.Range(0, int.MaxValue));
                boidShader.Dispatch(generateBoidsKernel, Mathf.CeilToInt(numBoids / blockSize), 1, 1);
            }
        }
        
        private void SetBoidShaderParameters()
        {
            boidShader.SetBuffer(updateBoidsKernel, "boidsIn", boidBufferOut);
            boidShader.SetBuffer(updateBoidsKernel, "boidsOut", boidBuffer);
            boidShader.SetInt("numBoids", numBoids);
            boidShader.SetFloat("maxSpeed", maxSpeed);
            boidShader.SetFloat("minSpeed", boidBehaviorManager.minSpeed);
            boidShader.SetFloat("edgeMargin", edgeMargin);
            boidShader.SetFloat("visualRangeSq", boidBehaviorManager.visualRangeSq);
            boidShader.SetFloat("minDistanceSq", boidBehaviorManager.minDistanceSq);
            boidShader.SetFloat("turnSpeed", boidBehaviorManager.turnSpeed);
            boidShader.SetFloat("xBound", xBound);
            boidShader.SetFloat("yBound", yBound);
            boidShader.SetFloat("cohesionFactor", cohesionFactor);
            boidShader.SetFloat("separationFactor", separationFactor);
            boidShader.SetFloat("alignmentFactor", alignmentFactor);
            boidShader.SetFloat("teamRatio", teamRatio);
            boidShader.SetFloat("intraTeamCohesionMultiplier", intraTeamCohesionMultiplier);
            boidShader.SetFloat("interTeamRepulsionMultiplier", interTeamRepulsionMultiplier);
            
            // Set grid parameters to boid shader
            boidShader.SetBuffer(updateBoidsKernel, "gridOffsetBuffer", gridManager.gridOffsetBuffer);
            boidShader.SetFloat("gridCellSize", gridManager.gridCellSize);
            boidShader.SetInt("gridDimY", gridManager.gridDimY);
            boidShader.SetInt("gridDimX", gridManager.gridDimX);
        }
        
        void Update()
        {
            // Update FPS
            fpsText.text = "FPS: " + (int)(1 / Time.smoothDeltaTime);
            
            if (useGpu)
            {
                // Update obstacles if needed
                obstacleManager.UpdateObstacleData(useGpu);
                
                // Set delta time for physics simulation
                boidShader.SetFloat("deltaTime", Time.deltaTime);
                
                if (useQuadTree)
                {
                    // Update using quadtree
                    quadTreeManager.UpdateQuadTree(numBoids, boidBuffer, boidBufferOut, boidShader);
                    
                    // Swap buffers (if not already done by quadtree manager)
                    var temp = boidBuffer;
                    boidBuffer = boidBufferOut;
                    boidBufferOut = temp;
                }
                else
                {
                    // Update using spatial grid
                    boidShader.SetInt("useQuadTree", 0);
                    gridManager.UpdateGridGPU(numBoids, boidBuffer, boidBufferOut);
                    
                    // Compute boid behaviors
                    boidShader.Dispatch(updateBoidsKernel, Mathf.CeilToInt(numBoids / blockSize), 1, 1);
                    
                    // Swap buffers
                    var temp = boidBuffer;
                    boidBuffer = boidBufferOut;
                    boidBufferOut = temp;
                }
            }
            else // CPU mode
            {
                // Spatial grid
                gridManager.ClearGrid();
                gridManager.UpdateGrid(boids, numBoids);
                gridManager.GenerateGridOffsets();
                gridManager.RearrangeBoids(boids, boidsTemp, numBoids);
                
                // Process each boid on CPU
                for (int i = 0; i < numBoids; i++)
                {
                    var boid = boidsTemp[i];
                    boidBehaviorManager.MergedBehaviours(ref boid, boidsTemp, gridManager.GetGridOffsets(), gridManager.gridDimX, gridManager.gridDimY, gridManager.gridCellSize);
                    boidBehaviorManager.LimitSpeed(ref boid);
                    boidBehaviorManager.KeepInBounds(ref boid);
                    
                    // Update position
                    boid.pos += boid.vel * Time.deltaTime;
                    boids[i] = boid;
                }
                
                // Send updated positions to GPU for rendering
                boidBuffer.SetData(boids);
            }
            
            // Render the boids
            boidRenderer.RenderBoids(numBoids);
        }
        
        // Called from UI slider
        public void SliderChange(float val)
        {
            numBoids = (int)Mathf.Pow(2, val);
            var limit = useGpu ? gpuLimit : cpuLimit;
            if (numBoids > limit)
            {
                numBoids = limit;
            }
            
            RestartSimulation(numBoids);
        }
        
        // Called from UI button
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
        
        // Public methods for UI to call
        public void RestartSimulation(int newBoidCount)
        {
            CleanupResources();
            numBoids = newBoidCount;
            Start();
        }
        
        public void SetGpuMode(bool useGpuMode)
        {
            if (this.useGpu == useGpuMode) return;
            
            this.useGpu = useGpuMode;
            RestartSimulation(numBoids);
        }
        
        void OnDrawGizmos()
        {
            if (quadTreeManager != null)
            {
                quadTreeManager.OnDrawGizmos(transform);
            }
        }
        
        void CleanupResources()
        {
            // Clean up native arrays
            if (boids.IsCreated) boids.Dispose();
            if (boidsTemp.IsCreated) boidsTemp.Dispose();
            
            // Clean up component managers
            gridManager?.Cleanup();
            quadTreeManager?.Cleanup();
            obstacleManager?.Cleanup();
            boidRenderer?.Cleanup();
            
            // Clean up compute buffers
            boidBuffer?.Release();
            boidBufferOut?.Release();
        }
        
        void OnDestroy()
        {
            CleanupResources();
        }
    }
}