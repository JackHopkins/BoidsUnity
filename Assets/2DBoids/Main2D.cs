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
  const float blockSize = 256f;
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

  [Header("Debug Settings")]
  [SerializeField] private bool drawQuadTreeGizmos = false;
  [SerializeField] private int gizmoDetailLevel = 2; // Controls how many levels to draw


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
  int countBoidsKernel, sumNodeCountsKernel, updateNodeCountsKernel, clearNodeCountsKernel;

  private bool useUnifiedKernel = true; // Set to true to use the unified approach

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
  private ComputeBuffer subdivDebugBuffer;

  private int recountBoidsKernel;
  private int clearQuadTreeKernel;
  private int insertBoidsKernel;
  //private int buildActiveNodesKernel;
  private int sortBoidsKernel;
  //private int subdivideNodesKernel;
  private int initializeTreeKernel;
  private int buildUnifiedKernel;
  private int subdivideAndRedistributeKernel;
  private bool printQuadTreeDebugInfo = true;
  private int debugPrintInterval = 60; // Print every 60 frames


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
    
// Add this method to your Main2D class
private void DebugPrintQuadTreeInfo()
{
    if (!printQuadTreeDebugInfo || Time.frameCount % debugPrintInterval != 0)
        return;
    
    // Don't try to visualize if the buffer hasn't been created yet
    if (quadNodesBuffer == null || nodeCountBuffer == null)
        return;
    
    // Create a diagnostic buffer to check subdivision conditions
    ComputeBuffer diagBuffer = new ComputeBuffer(10, 4);
    
    // Initialize with zeros to detect if values are being written
    uint[] initialData = new uint[10];
    diagBuffer.SetData(initialData);
    
    // Find the diagnostic kernel
    int diagKernel = quadTreeShader.FindKernel("DiagnoseSubdivision");
    
    // Set all necessary buffers and parameters
    quadTreeShader.SetBuffer(diagKernel, "quadNodes", quadNodesBuffer);
    quadTreeShader.SetBuffer(diagKernel, "nodeCount", nodeCountBuffer);
    quadTreeShader.SetBuffer(diagKernel, "diagData", diagBuffer);
    quadTreeShader.SetInt("maxBoidsPerNode", maxBoidsPerNode);
    
    // Dispatch the kernel
    quadTreeShader.Dispatch(diagKernel, 1, 1, 1);
    
    // Read back the diagnostic data
    uint[] diagData = new uint[10];
    diagBuffer.GetData(diagData);
    
    // Extract condition results
    bool isLeaf = diagData[0] == 1;
    bool countExceedsThreshold = diagData[1] == 1;
    bool hasSpaceForChildren = diagData[2] == 1;
    bool sizeAboveThreshold = diagData[3] == 1;
    
    // Extract actual values
    uint flags = diagData[4];
    uint count = diagData[5];
    uint threshold = diagData[6];
    uint currentNodeCount = diagData[7];
    uint size = diagData[8];
    
    // Read node count and active node count from buffers
    uint[] counts = new uint[1];
    nodeCountBuffer.GetData(counts);
    int totalNodeCount = (int)counts[0];
    
    uint[] activeCounts = new uint[1];
    activeNodeCountBuffer.GetData(activeCounts);
    int activeNodeCount = (int)activeCounts[0];
    
    // Avoid reading too many nodes
    if (totalNodeCount <= 0 || totalNodeCount > MaxQuadNodes) {
        Debug.Log($"Invalid node count: {totalNodeCount}");
        diagBuffer.Release();
        return;
    }
    
    // Read node data - only enough for our display
    int maxNodesToShow = Mathf.Min(totalNodeCount, 50);
    QuadNode[] allNodes = new QuadNode[maxNodesToShow];
    quadNodesBuffer.GetData(allNodes, 0, 0, maxNodesToShow);
    
    // Read active node indices if available
    int[] activeIndices = new int[activeNodeCount];
    if (activeNodeCount > 0 && activeNodeCount < MaxQuadNodes) {
        uint[] activeNodesData = new uint[activeNodeCount];
        activeNodesBuffer.GetData(activeNodesData, 0, 0, activeNodeCount);
        for (int i = 0; i < activeNodeCount; i++) {
            activeIndices[i] = (int)activeNodesData[i];
        }
    }
    
    // Start building our report
    System.Text.StringBuilder sb = new System.Text.StringBuilder();
    
    // Section 1: Subdivision diagnostic
    sb.AppendLine("=== SUBDIVISION DIAGNOSTIC ===");
    sb.AppendLine($"Root node - Flags: {flags}, Count: {count}, Size: {size}");
    sb.AppendLine($"System - maxBoidsPerNode: {threshold}, Current node count: {currentNodeCount}");
    sb.AppendLine("\nCondition checks:");
    sb.AppendLine($"1. Is Leaf Node: {(isLeaf ? "✓ YES" : "✗ NO")}");
    sb.AppendLine($"2. Count > Threshold: {(countExceedsThreshold ? "✓ YES" : "✗ NO")} ({count} vs {threshold})");
    sb.AppendLine($"3. Has Space for Children: {(hasSpaceForChildren ? "✓ YES" : "✗ NO")} ({currentNodeCount} vs {MaxQuadNodes - 4})");
    sb.AppendLine($"4. Size Above Minimum: {(sizeAboveThreshold ? "✓ YES" : "✗ NO")} ({size} vs 2.0)");
    
    // Overall subdivision decision
    bool shouldSubdivide = isLeaf && countExceedsThreshold && hasSpaceForChildren && sizeAboveThreshold;
    sb.AppendLine($"\nFINAL RESULT: Should Subdivide = {(shouldSubdivide ? "YES" : "NO")}");
    
    // Identify specific blocking condition
    if (!shouldSubdivide) {
        sb.AppendLine("\nBLOCKING CONDITIONS:");
        if (!isLeaf) sb.AppendLine("- Node is not marked as a leaf node");
        if (!countExceedsThreshold) sb.AppendLine("- Boid count does not exceed threshold");
        if (!hasSpaceForChildren) sb.AppendLine("- Not enough space for child nodes");
        if (!sizeAboveThreshold) sb.AppendLine("- Node size is too small");
    }
    
    // Section 2: Quad Tree Structure
    sb.AppendLine("\n=== QUAD TREE STRUCTURE ===");
    sb.AppendLine($"Total nodes: {totalNodeCount}, Active nodes: {activeNodeCount}");
    sb.AppendLine("\nQuad Tree Nodes:");
    sb.AppendLine("NodeIdx\tIsLeaf\tIsActive\tBoidCount\tSize\tCenter\tChildIdx");
    sb.AppendLine("----------------------------------------------------------------------");
    
    // Add info for each node
    for (int i = 0; i < maxNodesToShow; i++) {
        QuadNode node = allNodes[i];
        bool nodeIsLeaf = (node.flags & NODE_LEAF) != 0;
        bool isActive = (node.flags & NODE_ACTIVE) != 0;
        bool isInActiveList = System.Array.IndexOf(activeIndices, i) >= 0;
        
        string activeStatus = isActive ? "Yes" : "No";
        if (isInActiveList && !isActive) activeStatus = "Listed";
        if (!isInActiveList && isActive) activeStatus = "Flagged";
        
        sb.AppendLine($"{i}\t{(nodeIsLeaf ? "Yes" : "No")}\t{activeStatus}\t{node.count}\t{node.size}\t({node.center.x:F1}, {node.center.y:F1})\t{node.childIndex}");
        
        // Also print child info if this node has children
        if (!nodeIsLeaf && node.childIndex > 0 && node.childIndex < maxNodesToShow) {
            for (int c = 0; c < 4; c++) {
                int childIdx = (int)node.childIndex + c;
                if (childIdx < maxNodesToShow) {
                    QuadNode child = allNodes[childIdx];
                    sb.AppendLine($"  └─{childIdx}\t{((child.flags & NODE_LEAF) != 0 ? "Yes" : "No")}\t{((child.flags & NODE_ACTIVE) != 0 ? "Yes" : "No")}\t{child.count}\t{child.size}\t({child.center.x:F1}, {child.center.y:F1})\t{child.childIndex}");
                }
            }
        }
    }
    
    if (totalNodeCount > maxNodesToShow) {
        sb.AppendLine($"... and {totalNodeCount - maxNodesToShow} more nodes");
    }
    
    // Section 3: Boid distribution
    sb.AppendLine("\n=== BOID DISTRIBUTION ===");
    
    // Print summary of boid distribution
    int[] countHistogram = new int[10]; // Count nodes with 0, 1-5, 6-10, etc. boids
    int maxBoidsInNode = 0;
    int nodesWithBoids = 0;
    int totalBoidsInNodes = 0;
    
    for (int i = 0; i < maxNodesToShow; i++) {
        int boidCount = (int)allNodes[i].count;
        totalBoidsInNodes += boidCount;
        
        if (boidCount > 0) {
            nodesWithBoids++;
            maxBoidsInNode = Mathf.Max(maxBoidsInNode, boidCount);
        }
        
        if (boidCount == 0) countHistogram[0]++;
        else if (boidCount <= 5) countHistogram[1]++;
        else if (boidCount <= 10) countHistogram[2]++;
        else if (boidCount <= 20) countHistogram[3]++;
        else if (boidCount <= 50) countHistogram[4]++;
        else if (boidCount <= 100) countHistogram[5]++;
        else if (boidCount <= 200) countHistogram[6]++;
        else countHistogram[7]++;
    }
    
    sb.AppendLine($"Total boids in nodes: {totalBoidsInNodes} (should match numBoids: {numBoids})");
    sb.AppendLine($"Nodes with boids: {nodesWithBoids} of {totalNodeCount}");
    sb.AppendLine($"Max boids in a single node: {maxBoidsInNode} (threshold: {maxBoidsPerNode})");
    sb.AppendLine($"Empty nodes: {countHistogram[0]}");
    sb.AppendLine($"Nodes with 1-5 boids: {countHistogram[1]}");
    sb.AppendLine($"Nodes with 6-10 boids: {countHistogram[2]}");
    sb.AppendLine($"Nodes with 11-20 boids: {countHistogram[3]}");
    sb.AppendLine($"Nodes with 21-50 boids: {countHistogram[4]}");
    sb.AppendLine($"Nodes with 51-100 boids: {countHistogram[5]}");
    sb.AppendLine($"Nodes with 101-200 boids: {countHistogram[6]}");
    sb.AppendLine($"Nodes with >200 boids: {countHistogram[7]}");
    
    // Log the complete report
    Debug.Log(sb.ToString());
    
    // Clean up
    diagBuffer.Release();
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

    // OPTIMIZATION 1: Only rebuild quadtree every N frames
    // This dramatically reduces GPU overhead when boids move slowly
    if (Time.frameCount % 3 != 0) {
        // Just run the simulation using the existing quadtree
        boidShader.SetBuffer(updateBoidsKernel, "boidsIn", boidBuffer);
        boidShader.SetBuffer(updateBoidsKernel, "boidsOut", boidBufferOut);
        boidShader.SetFloat("deltaTime", Time.deltaTime);
        boidShader.SetInt("useQuadTree", 1);
        boidShader.Dispatch(updateBoidsKernel, Mathf.CeilToInt(numBoids / blockSize), 1, 1);
        
        // Swap buffers
        var temp = boidBuffer;
        boidBuffer = boidBufferOut;
        boidBufferOut = temp;
        return;
    }

    // First, completely clear the quadtree and node counts
    // This is critical to avoid stale data between frames
    int clearNodeCountsKernel = quadTreeShader.FindKernel("ClearNodeCounts");
    quadTreeShader.SetBuffer(clearNodeCountsKernel, "nodeCounts", nodeCountsBuffer);
    quadTreeShader.Dispatch(clearNodeCountsKernel, Mathf.CeilToInt(MaxQuadNodes / 256f), 1, 1);

    // Approach 1: Use a single unified kernel (most efficient, but may not work on all platforms)
    if (useUnifiedKernel && buildUnifiedKernel >= 0)
    {
        // CRITICAL FIX: We need to make sure the root is a leaf node again before rebuilding
        // Use a special kernel just for fixing the root node
        int resetRootKernel = quadTreeShader.FindKernel("ResetRootNode");
        if (resetRootKernel >= 0) {
            quadTreeShader.SetBuffer(resetRootKernel, "quadNodes", quadNodesBuffer);
            quadTreeShader.SetFloat("worldSize", initialQuadTreeSize);
            quadTreeShader.Dispatch(resetRootKernel, 1, 1, 1);
        }

        // Set all needed buffers
        quadTreeShader.SetBuffer(buildUnifiedKernel, "boids", boidBuffer);
        quadTreeShader.SetBuffer(buildUnifiedKernel, "boidsOut", boidBufferOut);
        quadTreeShader.SetBuffer(buildUnifiedKernel, "quadNodes", quadNodesBuffer);
        quadTreeShader.SetBuffer(buildUnifiedKernel, "nodeCount", nodeCountBuffer);
        quadTreeShader.SetBuffer(buildUnifiedKernel, "activeNodes", activeNodesBuffer);
        quadTreeShader.SetBuffer(buildUnifiedKernel, "activeNodeCount", activeNodeCountBuffer);
        quadTreeShader.SetBuffer(buildUnifiedKernel, "boidIndices", boidIndicesBuffer);
        quadTreeShader.SetBuffer(buildUnifiedKernel, "nodeCounts", nodeCountsBuffer);
        quadTreeShader.SetInt("maxBoidsPerNode", maxBoidsPerNode);
        quadTreeShader.SetInt("numBoids", numBoids);
        quadTreeShader.SetFloat("worldSize", initialQuadTreeSize);

        // Single dispatch for the entire tree building process
        quadTreeShader.Dispatch(buildUnifiedKernel, Mathf.Max(1, Mathf.CeilToInt(numBoids / 256f)), 1, 1);
        
        // Now run the simulation using the built quadtree
        boidShader.SetBuffer(updateBoidsKernel, "boidsIn", boidBuffer);
        boidShader.SetBuffer(updateBoidsKernel, "boidsOut", boidBufferOut);
        boidShader.SetFloat("deltaTime", Time.deltaTime);
        boidShader.SetInt("useQuadTree", 1);
        boidShader.Dispatch(updateBoidsKernel, Mathf.CeilToInt(numBoids / 256f), 1, 1);
    }
    // Approach 2: Use fewer but separate kernel dispatches
    else
    {
        // CRITICAL FIX: Reset the tree completely with a proper root node
        quadTreeShader.SetBuffer(initializeTreeKernel, "quadNodes", quadNodesBuffer);
        quadTreeShader.SetBuffer(initializeTreeKernel, "nodeCount", nodeCountBuffer);
        quadTreeShader.SetBuffer(initializeTreeKernel, "activeNodes", activeNodesBuffer);
        quadTreeShader.SetBuffer(initializeTreeKernel, "activeNodeCount", activeNodeCountBuffer);
        quadTreeShader.SetBuffer(initializeTreeKernel, "nodeCounts", nodeCountsBuffer);
        quadTreeShader.SetFloat("worldSize", initialQuadTreeSize);
        quadTreeShader.Dispatch(initializeTreeKernel, 1, 1, 1);

        // Insert boids into quadtree - make sure to set numBoids for the shader
        quadTreeShader.SetBuffer(insertBoidsKernel, "boids", boidBuffer);
        quadTreeShader.SetBuffer(insertBoidsKernel, "boidIndices", boidIndicesBuffer);
        quadTreeShader.SetBuffer(insertBoidsKernel, "nodeCounts", nodeCountsBuffer);
        quadTreeShader.SetInt("numBoids", numBoids);
        quadTreeShader.Dispatch(insertBoidsKernel, Mathf.CeilToInt(numBoids / 256f), 1, 1);
        
        // Update node counts
        quadTreeShader.SetBuffer(updateNodeCountsKernel, "quadNodes", quadNodesBuffer);
        quadTreeShader.SetBuffer(updateNodeCountsKernel, "nodeCounts", nodeCountsBuffer);
        quadTreeShader.Dispatch(updateNodeCountsKernel, Mathf.CeilToInt(MaxQuadNodes / 256f), 1, 1);
        
        // OPTIMIZATION 3: Use fewer iterations with combined subdivide and redistribute
        if (subdivideAndRedistributeKernel >= 0) {
            // If we have the combined kernel, use it for fewer dispatches
            int iterations = Mathf.Min(3, maxQuadTreeDepth);
            for (int i = 0; i < iterations; i++) {
                quadTreeShader.SetBuffer(subdivideAndRedistributeKernel, "boids", boidBuffer);
                quadTreeShader.SetBuffer(subdivideAndRedistributeKernel, "quadNodes", quadNodesBuffer);
                quadTreeShader.SetBuffer(subdivideAndRedistributeKernel, "nodeCount", nodeCountBuffer);
                quadTreeShader.SetBuffer(subdivideAndRedistributeKernel, "activeNodes", activeNodesBuffer);
                quadTreeShader.SetBuffer(subdivideAndRedistributeKernel, "activeNodeCount", activeNodeCountBuffer);
                quadTreeShader.SetBuffer(subdivideAndRedistributeKernel, "boidIndices", boidIndicesBuffer);
                quadTreeShader.SetBuffer(subdivideAndRedistributeKernel, "nodeCounts", nodeCountsBuffer);
                quadTreeShader.SetInt("numBoids", numBoids);
                
                // Dispatch with enough threads for both operations
                int threadGroups = Mathf.Max(
                    Mathf.CeilToInt(numBoids / 256f),
                    Mathf.CeilToInt(MaxQuadNodes / 256f)
                );
                quadTreeShader.Dispatch(subdivideAndRedistributeKernel, threadGroups, 1, 1);
                
                // Clear and update node counts after each iteration
                quadTreeShader.Dispatch(clearNodeCountsKernel, Mathf.CeilToInt(MaxQuadNodes / 256f), 1, 1);
                
                // CRITICAL FIX: We need to recount all boids after redistribution
                
                if (recountBoidsKernel >= 0) {
                    quadTreeShader.SetBuffer(recountBoidsKernel, "boidIndices", boidIndicesBuffer);
                    quadTreeShader.SetBuffer(recountBoidsKernel, "nodeCounts", nodeCountsBuffer);
                    quadTreeShader.SetInt("numBoids", numBoids);
                    quadTreeShader.Dispatch(recountBoidsKernel, Mathf.CeilToInt(numBoids / 256f), 1, 1);
                }
                
                // Update node counts after recounting
                quadTreeShader.Dispatch(updateNodeCountsKernel, Mathf.CeilToInt(MaxQuadNodes / 256f), 1, 1);
            }
        }
        
        // After updating node counts, apply a final redistribution
        int forceRedistributeKernel = quadTreeShader.FindKernel("ForceRedistributeRootBoids");
        quadTreeShader.SetBuffer(forceRedistributeKernel, "boids", boidBuffer);
        quadTreeShader.SetBuffer(forceRedistributeKernel, "boidIndices", boidIndicesBuffer);
        quadTreeShader.SetBuffer(forceRedistributeKernel, "quadNodes", quadNodesBuffer);
        quadTreeShader.SetBuffer(forceRedistributeKernel, "nodeCounts", nodeCountsBuffer);
        quadTreeShader.SetInt("numBoids", numBoids);
        quadTreeShader.Dispatch(forceRedistributeKernel, Mathf.CeilToInt(numBoids / 256f), 1, 1);

        // CRITICAL FIX: Recount boids once more after final redistribution
        //int recountBoidsKernel = quadTreeShader.FindKernel("RecountBoids");
        if (recountBoidsKernel >= 0) {
            quadTreeShader.SetBuffer(recountBoidsKernel, "boidIndices", boidIndicesBuffer);
            quadTreeShader.SetBuffer(recountBoidsKernel, "nodeCounts", nodeCountsBuffer);
            quadTreeShader.SetInt("numBoids", numBoids);
            quadTreeShader.Dispatch(recountBoidsKernel, Mathf.CeilToInt(numBoids / 256f), 1, 1);
        }

        // Update node counts again
        quadTreeShader.Dispatch(updateNodeCountsKernel, Mathf.CeilToInt(MaxQuadNodes / 256f), 1, 1);
        
        // Sort boids according to quadtree
        quadTreeShader.SetBuffer(sortBoidsKernel, "boids", boidBuffer);
        quadTreeShader.SetBuffer(sortBoidsKernel, "boidsOut", boidBufferOut);
        quadTreeShader.SetBuffer(sortBoidsKernel, "boidIndices", boidIndicesBuffer);
        quadTreeShader.SetInt("numBoids", numBoids);
        quadTreeShader.Dispatch(sortBoidsKernel, Mathf.CeilToInt(numBoids / 256f), 1, 1);
        
        // Swap buffers to get sorted boids
        var temp = boidBuffer;
        boidBuffer = boidBufferOut;
        boidBufferOut = temp;
        
        // Run the boid update
        boidShader.SetBuffer(updateBoidsKernel, "boidsIn", boidBuffer);
        boidShader.SetBuffer(updateBoidsKernel, "boidsOut", boidBufferOut);
        boidShader.SetFloat("deltaTime", Time.deltaTime);
        boidShader.SetInt("useQuadTree", 1);
        boidShader.Dispatch(updateBoidsKernel, Mathf.CeilToInt(numBoids / 256f), 1, 1);
    }
    
    // Final buffer swap after boid update
    var finalTemp = boidBuffer;
    boidBuffer = boidBufferOut;
    boidBufferOut = finalTemp;

    DebugPrintQuadTreeInfo();
}

private void AnalyzeQuadTreeBoidDistribution()
{
    if (!useQuadTree || quadNodesBuffer == null || boidIndicesBuffer == null)
        return;
    
    // Step 1: Read basic information
    uint[] counts = new uint[1];
    nodeCountBuffer.GetData(counts);
    int nodeCount = (int)counts[0];
    
    // Step 2: Read the node data
    QuadNode[] allNodes = new QuadNode[nodeCount];
    quadNodesBuffer.GetData(allNodes, 0, 0, nodeCount);
    
    // Step 3: Read a sample of boid-to-node assignments to verify distribution
    int sampleSize = Mathf.Min(numBoids, 1000);
    uint[] boidNodeAssignments = new uint[sampleSize * 2];
    boidIndicesBuffer.GetData(boidNodeAssignments, 0, 0, sampleSize * 2);
    
    // Step 4: Count boids in each node by examining assignments
    Dictionary<uint, int> actualBoidsPerNode = new Dictionary<uint, int>();
    for (int i = 0; i < sampleSize; i++)
    {
        uint nodeIdx = boidNodeAssignments[i * 2 + 1];
        if (!actualBoidsPerNode.ContainsKey(nodeIdx))
            actualBoidsPerNode[nodeIdx] = 0;
        actualBoidsPerNode[nodeIdx]++;
    }
    
    // Step 5: Compare with reported counts
    Debug.Log($"=== DETAILED BOID DISTRIBUTION ANALYSIS ===");
    Debug.Log($"Sampling {sampleSize} of {numBoids} boids");
    Debug.Log("Node\tReported\tSampled\tDifference");
    Debug.Log("---------------------------------------");
    
    int totalSampled = 0;
    foreach (var kvp in actualBoidsPerNode)
    {
        uint nodeIdx = kvp.Key;
        int sampledCount = kvp.Value;
        totalSampled += sampledCount;
        
        // Get the reported count from the node structure
        int reportedCount = nodeIdx < nodeCount ? (int)allNodes[nodeIdx].count : 0;
        
        // Calculate scaled reported count based on sample size
        float scaledReported = reportedCount * (sampleSize / (float)numBoids);
        
        Debug.Log($"{nodeIdx}\t{reportedCount}\t{sampledCount}\t{sampledCount - scaledReported:F1}");
    }
    
    Debug.Log($"Total sampled: {totalSampled} (should be {sampleSize})");
    
    // Step 6: Analyze boid indices to check for any anomalies
    HashSet<uint> uniqueBoidIndices = new HashSet<uint>();
    for (int i = 0; i < sampleSize; i++)
    {
        uniqueBoidIndices.Add(boidNodeAssignments[i * 2]);
    }
    
    Debug.Log($"Unique boid indices: {uniqueBoidIndices.Count} (should be {sampleSize})");
    if (uniqueBoidIndices.Count < sampleSize)
    {
        Debug.LogWarning("Some boids appear multiple times in the assignment list!");
    }
    
    // Step 7: Validate the quadtree structure
    Debug.Log("\n=== QUADTREE STRUCTURE VALIDATION ===");
    
    // Check if root node is properly set
    bool rootIsLeaf = (allNodes[0].flags & NODE_LEAF) != 0;
    bool rootHasChildren = allNodes[0].childIndex > 0 && allNodes[0].childIndex < nodeCount;
    bool rootChildrenAreLeaves = true;
    
    if (rootHasChildren)
    {
        for (int i = 0; i < 4; i++)
        {
            int childIdx = (int)allNodes[0].childIndex + i;
            if (childIdx < nodeCount)
            {
                bool isLeaf = (allNodes[childIdx].flags & NODE_LEAF) != 0;
                if (!isLeaf) rootChildrenAreLeaves = false;
            }
        }
    }
    
    Debug.Log($"Root node is leaf: {rootIsLeaf}");
    Debug.Log($"Root has valid children: {rootHasChildren}");
    if (rootHasChildren)
    {
        Debug.Log($"Root's children are leaves: {rootChildrenAreLeaves}");
    }
    
    // Step 8: Check for orphaned nodes or invalid child indices
    int nodesWithInvalidChildren = 0;
    for (int i = 0; i < nodeCount; i++)
    {
        if ((allNodes[i].flags & NODE_LEAF) == 0 && allNodes[i].childIndex > 0)
        {
            if (allNodes[i].childIndex >= nodeCount || allNodes[i].childIndex + 3 >= nodeCount)
            {
                nodesWithInvalidChildren++;
                Debug.LogWarning($"Node {i} has invalid child index: {allNodes[i].childIndex}");
            }
        }
    }
    
    Debug.Log($"Nodes with invalid children: {nodesWithInvalidChildren}");
}

private void DebugSubdivisionIssues()
{
    // Create debug buffer if it doesn't exist
    if (subdivDebugBuffer == null)
    {
        subdivDebugBuffer = new ComputeBuffer(10, 4); // 10 uint values
    }
    
    // Find debug kernel
    int debugSubdivKernel = quadTreeShader.FindKernel("DebugSubdivision");
    
    // Set buffers
    quadTreeShader.SetBuffer(debugSubdivKernel, "quadNodes", quadNodesBuffer);
    quadTreeShader.SetBuffer(debugSubdivKernel, "nodeCount", nodeCountBuffer);
    quadTreeShader.SetBuffer(debugSubdivKernel, "activeNodeCount", activeNodeCountBuffer);
    
    quadTreeShader.SetInt("maxBoidsPerNode", maxBoidsPerNode);
    quadTreeShader.SetBuffer(debugSubdivKernel, "subdivDebug", subdivDebugBuffer);

    // Dispatch
    quadTreeShader.Dispatch(debugSubdivKernel, 1, 1, 1);
    
    // Read back data
    uint[] debugData = new uint[10];
    subdivDebugBuffer.GetData(debugData);
    
    // Analyze results
    Debug.Log($"Subdivision Debug:");
    Debug.Log($"Root flags: {debugData[0]} (Has LEAF bit: {(debugData[7] == 1 ? "Yes" : "No")})");
    Debug.Log($"Root count: {debugData[1]} (Exceeds threshold of {debugData[5]}: {(debugData[8] == 1 ? "Yes" : "No")})");
    Debug.Log($"Node count: {debugData[2]} (Room for children: {(debugData[9] == 1 ? "Yes" : "No")})");
    Debug.Log($"Root size: {debugData[4]}");
    Debug.Log($"Active nodes: {debugData[6]}");
    
    // Check if ALL conditions pass
    bool shouldSubdivide = debugData[7] == 1 && debugData[8] == 1 && debugData[9] == 1;
    Debug.Log($"Should root subdivide? {(shouldSubdivide ? "YES" : "NO - subdivision blocked")}");
    
    // Read actual parameters
    uint[] maxBoidsPerNodeArr = new uint[1];
    ComputeBuffer tempBuffer = new ComputeBuffer(1, 4);
    quadTreeShader.SetBuffer(debugSubdivKernel, "maxBoidsPerNodeCheck", tempBuffer);
    quadTreeShader.Dispatch(debugSubdivKernel, 1, 1, 1);
    tempBuffer.GetData(maxBoidsPerNodeArr);
    tempBuffer.Release();

    ComputeBuffer testBuffer = new ComputeBuffer(1, 4);
    int testKernel = quadTreeShader.FindKernel("TestMaxBoidsPerNode");
    quadTreeShader.SetBuffer(testKernel, "testBuffer", testBuffer);
    quadTreeShader.SetInt("maxBoidsPerNode", maxBoidsPerNode);
    quadTreeShader.Dispatch(testKernel, 1, 1, 1);
    
    uint[] result = new uint[1];
    testBuffer.GetData(result);
    Debug.Log($"Direct test - maxBoidsPerNode in shader: {result[0]}");
    
    testBuffer.Release();
    
    Debug.Log($"maxBoidsPerNode value in shader: {maxBoidsPerNodeArr[0]} (C# value: {maxBoidsPerNode})");
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
    initializeTreeKernel = quadTreeShader.FindKernel("InitializeQuadTree");
    buildUnifiedKernel = quadTreeShader.FindKernel("BuildQuadtreeUnified");
    subdivideAndRedistributeKernel = quadTreeShader.FindKernel("SubdivideAndRedistribute"); 

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
        recountBoidsKernel = quadTreeShader.FindKernel("RecountBoids");
        clearQuadTreeKernel = quadTreeShader.FindKernel("ClearQuadTree");
        insertBoidsKernel = quadTreeShader.FindKernel("InsertBoids");
        //buildActiveNodesKernel = quadTreeShader.FindKernel("BuildActiveNodes");
        sortBoidsKernel = quadTreeShader.FindKernel("SortBoids");
		//subdivideNodesKernel = quadTreeShader.FindKernel("SubdivideNodes");
        
        clearNodeCountsKernel = quadTreeShader.FindKernel("ClearNodeCounts");
        //redistributeBoidsKernel = quadTreeShader.FindKernel("RedistributeBoids");
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

        if (initializeTreeKernel >= 0)
        {
            quadTreeShader.SetBuffer(initializeTreeKernel, "quadNodes", quadNodesBuffer);
            quadTreeShader.SetBuffer(initializeTreeKernel, "nodeCount", nodeCountBuffer);
            quadTreeShader.SetBuffer(initializeTreeKernel, "activeNodes", activeNodesBuffer);
            quadTreeShader.SetBuffer(initializeTreeKernel, "activeNodeCount", activeNodeCountBuffer);
            quadTreeShader.SetBuffer(initializeTreeKernel, "nodeCounts", nodeCountsBuffer);
        }
        
        if (buildUnifiedKernel >= 0)
        {
            quadTreeShader.SetBuffer(buildUnifiedKernel, "quadNodes", quadNodesBuffer);
            quadTreeShader.SetBuffer(buildUnifiedKernel, "nodeCount", nodeCountBuffer);
            quadTreeShader.SetBuffer(buildUnifiedKernel, "activeNodes", activeNodesBuffer);
            quadTreeShader.SetBuffer(buildUnifiedKernel, "activeNodeCount", activeNodeCountBuffer);
            quadTreeShader.SetBuffer(buildUnifiedKernel, "boidIndices", boidIndicesBuffer);
            quadTreeShader.SetBuffer(buildUnifiedKernel, "nodeCounts", nodeCountsBuffer);
            quadTreeShader.SetBuffer(buildUnifiedKernel, "boids", boidBuffer);
            quadTreeShader.SetBuffer(buildUnifiedKernel, "boidsOut", boidBufferOut);
            quadTreeShader.SetInt("numBoids", numBoids);
            quadTreeShader.SetInt("maxDepth", maxQuadTreeDepth);
            quadTreeShader.SetInt("maxBoidsPerNode", maxBoidsPerNode);
            quadTreeShader.SetFloat("worldSize", initialQuadTreeSize);
        }
        
        if (subdivideAndRedistributeKernel >= 0)
        {
            quadTreeShader.SetBuffer(subdivideAndRedistributeKernel, "quadNodes", quadNodesBuffer);
            quadTreeShader.SetBuffer(subdivideAndRedistributeKernel, "nodeCount", nodeCountBuffer);
            quadTreeShader.SetBuffer(subdivideAndRedistributeKernel, "activeNodes", activeNodesBuffer);
            quadTreeShader.SetBuffer(subdivideAndRedistributeKernel, "activeNodeCount", activeNodeCountBuffer);
            quadTreeShader.SetBuffer(subdivideAndRedistributeKernel, "boidIndices", boidIndicesBuffer);
            quadTreeShader.SetBuffer(subdivideAndRedistributeKernel, "nodeCounts", nodeCountsBuffer);
            quadTreeShader.SetBuffer(subdivideAndRedistributeKernel, "boids", boidBuffer);
            quadTreeShader.SetInt("maxBoidsPerNode", maxBoidsPerNode);
        }
        
        // Set buffers
		//quadTreeShader.SetBuffer(subdivideNodesKernel, "quadNodes", quadNodesBuffer);
		//quadTreeShader.SetBuffer(subdivideNodesKernel, "nodeCount", nodeCountBuffer);
		//quadTreeShader.SetBuffer(subdivideNodesKernel, "activeNodes", activeNodesBuffer);
		//quadTreeShader.SetBuffer(subdivideNodesKernel, "activeNodeCount", activeNodeCountBuffer);

        quadTreeShader.SetBuffer(clearQuadTreeKernel, "quadNodes", quadNodesBuffer);
        quadTreeShader.SetBuffer(clearQuadTreeKernel, "nodeCount", nodeCountBuffer);
		quadTreeShader.SetBuffer(clearQuadTreeKernel, "activeNodes", activeNodesBuffer);
        quadTreeShader.SetBuffer(clearQuadTreeKernel, "activeNodeCount", activeNodeCountBuffer);
        
        quadTreeShader.SetBuffer(insertBoidsKernel, "boids", boidBuffer);
        quadTreeShader.SetBuffer(insertBoidsKernel, "quadNodes", quadNodesBuffer);
        quadTreeShader.SetBuffer(insertBoidsKernel, "nodeCount", nodeCountBuffer);
		quadTreeShader.SetBuffer(insertBoidsKernel, "boidIndices", boidIndicesBuffer);
        
        //quadTreeShader.SetBuffer(buildActiveNodesKernel, "quadNodes", quadNodesBuffer);
        //quadTreeShader.SetBuffer(buildActiveNodesKernel, "nodeCount", nodeCountBuffer);
        //quadTreeShader.SetBuffer(buildActiveNodesKernel, "activeNodes", activeNodesBuffer);
        //quadTreeShader.SetBuffer(buildActiveNodesKernel, "activeNodeCount", activeNodeCountBuffer);
        
        quadTreeShader.SetBuffer(sortBoidsKernel, "boids", boidBuffer);
        quadTreeShader.SetBuffer(sortBoidsKernel, "boidsOut", boidBufferOut);
        quadTreeShader.SetBuffer(sortBoidsKernel, "quadNodes", quadNodesBuffer);
        quadTreeShader.SetBuffer(sortBoidsKernel, "boidIndices", boidIndicesBuffer);
		quadTreeShader.SetBuffer(sortBoidsKernel, "nodeCount", nodeCountBuffer);
            
        //quadTreeShader.SetBuffer(redistributeBoidsKernel, "boids", boidBuffer);
        //quadTreeShader.SetBuffer(redistributeBoidsKernel, "boidIndices", boidIndicesBuffer);
        //quadTreeShader.SetBuffer(redistributeBoidsKernel, "quadNodes", quadNodesBuffer);

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
    // Only draw if:
    // 1. We're in the editor
    // 2. The game is playing
    // 3. Visualization is explicitly enabled
    // 4. Quadtree is being used
    if (!Application.isPlaying || !drawQuadTreeGizmos || !useQuadTree || quadNodesBuffer == null) 
        return;
    
    // Don't try to visualize if the buffer hasn't been created yet
    if (quadNodesBuffer == null || quadNodesBuffer.count == 0)
        return;
    
    // Limit how many nodes we'll read back
    uint[] counts = new uint[1];
    nodeCountBuffer.GetData(counts);
    int nodeCount = (int)Mathf.Min(counts[0], MaxQuadNodes);
    
    // If there are no nodes or too many, skip visualization
    if (nodeCount <= 0 || nodeCount > 1000) 
        return;
    
    // Read just enough nodes, not the entire buffer
    QuadNode[] allNodes = new QuadNode[nodeCount];
    quadNodesBuffer.GetData(allNodes, 0, 0, nodeCount);
    
    // Just visualize the root node and immediate children
    DrawQuadNodeWithInfo(allNodes, 0, 0, Color.green);
}

void DrawQuadNodeWithInfo(QuadNode[] allNodes, uint nodeIndex, int depth, Color color)
{
    // Early out if we've reached our detail limit
    if (depth > gizmoDetailLevel || nodeIndex >= allNodes.Length) 
        return;
    
    QuadNode node = allNodes[nodeIndex];
    
    // Draw this node
    Gizmos.color = color;
    Vector3 center = new Vector3(node.center.x, node.center.y, 0);
    Vector3 size = new Vector3(node.size * 2, node.size * 2, 0.1f);
    Gizmos.DrawWireCube(center, size);
    
    // Display node info based on boid count
    int boidCount = (int)node.count;
    
    // Only draw filled boxes for nodes with boids
    if (boidCount > 0)
    {
        // Use a color gradient based on boid count
        float intensity = Mathf.Min(1.0f, boidCount / (float)maxBoidsPerNode);
        Color fillColor = new Color(intensity, 0, 1-intensity, 0.2f);
        Gizmos.color = fillColor;
        Gizmos.DrawCube(center, size);
        
        // Only draw labels for important nodes to reduce clutter
        if (boidCount > maxBoidsPerNode / 2 || depth <= 1)
        {
            // Use handles for text in the editor only
            #if UNITY_EDITOR
            UnityEditor.Handles.Label(center, $"Node {nodeIndex}: {boidCount}");
            #endif
        }
    }
    
    // Draw children if not a leaf and has valid children
    if ((node.flags & NODE_LEAF) == 0 && node.childIndex > 0 && node.childIndex < allNodes.Length)
    {
        // Alternate colors for child nodes
        Color childColor = new Color(color.r * 0.8f, color.g * 0.8f, color.b * 0.8f);
        
        // Draw each child, with bounds checking
        for (uint i = 0; i < 4; i++)
        {
            uint childIdx = node.childIndex + i;
            if (childIdx < allNodes.Length)
            {
                DrawQuadNodeWithInfo(allNodes, childIdx, depth + 1, childColor);
            }
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


