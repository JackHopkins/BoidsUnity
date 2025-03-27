using UnityEngine;
using UnityEngine.UI;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine.Rendering;
using System;
using System.Collections.Generic;

// Add this struct right after the Boid struct
struct ObstacleData
{
    public float2 pos;
    public float radius;
    public float strength;
}


struct Boid
{
  public float2 pos;
  public float2 vel;
  public uint team;
}

public class Main2D : MonoBehaviour
{
  const float blockSize = 1024f;

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

  [Header("Team Settings")]
  [SerializeField] float teamRatio = 0.5f; // Ratio of boids in team 0 vs team 1
  [SerializeField] float intraTeamCohesionMultiplier = 1.5f; // Stronger cohesion within same team
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

  ComputeBuffer boidBuffer;
  ComputeBuffer boidBufferOut;
  ComputeBuffer gridBuffer;
  ComputeBuffer gridOffsetBuffer;
  ComputeBuffer gridOffsetBufferIn;
  ComputeBuffer gridSumsBuffer;
  ComputeBuffer gridSumsBuffer2;

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


  // Update is called once per frame
  void Update()
  {
    fpsText.text = "FPS: " + (int)(1 / Time.smoothDeltaTime);

    if (useGpu)
    {
      boidShader.SetFloat("deltaTime", Time.deltaTime);
	
	  // Can be called here to update the positions (CALL SPARINGLY)
	  UpdateObstacleData();

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


