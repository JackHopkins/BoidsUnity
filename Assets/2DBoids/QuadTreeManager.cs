using UnityEngine;
using Unity.Mathematics;
using System.Text;
using System.Collections.Generic;

namespace BoidsUnity
{
    public class QuadTreeManager
    {
        // Constants
        private const int NODE_LEAF = 1;
        private const int NODE_ACTIVE = 2;
        private const int MaxQuadNodes = 65536; // Increased from 16384 for high boid counts
        
        // Settings
        public bool useQuadTree = false;
        public int maxQuadTreeDepth = 8;
        public int maxBoidsPerNode = 32;
        public int initialQuadTreeSize = 500;
        
        // Debug settings
        public bool drawQuadTreeGizmos = false;
        public int gizmoDetailLevel = 2;
        private bool printQuadTreeDebugInfo = true;
        private int debugPrintInterval = 60;
        
        // Compute shader and related fields
        private ComputeShader quadTreeShader;
        private bool useUnifiedKernel = true;
        
        // Buffers
        private ComputeBuffer quadNodesBuffer;
        private ComputeBuffer nodeCountBuffer;
        private ComputeBuffer boidIndicesBuffer;
        private ComputeBuffer activeNodesBuffer;
        private ComputeBuffer activeNodeCountBuffer;
        private ComputeBuffer nodeCountsBuffer;
        private ComputeBuffer subdivDebugBuffer;
        
        // Incremental update buffers
        private ComputeBuffer boidHistoryBuffer;
        private ComputeBuffer movedBoidIndicesBuffer;
        private ComputeBuffer movedBoidCountBuffer;
        
        // Incremental update settings
        private bool useIncrementalUpdates = true;
        private int fullRebuildInterval = 120; // Full rebuild every N frames (increased from 60)
        private int highBoidCountRebuildInterval = 180; // Even less frequent rebuilds for high counts
        private const int HighBoidCountThreshold = 25000; // Matches shader constant
        
        // Kernel IDs
        private int recountBoidsKernel;
        private int clearQuadTreeKernel;
        private int insertBoidsKernel;
        private int sortBoidsKernel;
        private int initializeTreeKernel;
        private int buildUnifiedKernel;
        private int subdivideAndRedistributeKernel;
        private int clearNodeCountsKernel;
        private int updateNodeCountsKernel;
        
        // Incremental update kernels
        private int trackMovedBoidsKernel;
        private int incrementalUpdateKernel;
        private int repairTreeStructureKernel;
        private int collapseNodesKernel;
        private int initializeBoidHistoryKernel;
        
        public QuadTreeManager(ComputeShader shader)
        {
            quadTreeShader = shader;
            
            // Get kernel IDs for standard operations
            recountBoidsKernel = quadTreeShader.FindKernel("RecountBoids");
            clearQuadTreeKernel = quadTreeShader.FindKernel("ClearQuadTree");
            insertBoidsKernel = quadTreeShader.FindKernel("InsertBoids");
            sortBoidsKernel = quadTreeShader.FindKernel("SortBoids");
            clearNodeCountsKernel = quadTreeShader.FindKernel("ClearNodeCounts");
            updateNodeCountsKernel = quadTreeShader.FindKernel("UpdateNodeCounts");
            initializeTreeKernel = quadTreeShader.FindKernel("InitializeQuadTree");
            buildUnifiedKernel = quadTreeShader.FindKernel("BuildQuadtreeUnified");
            subdivideAndRedistributeKernel = quadTreeShader.FindKernel("SubdivideAndRedistribute");
            
            // Get kernel IDs for incremental updates
            trackMovedBoidsKernel = quadTreeShader.FindKernel("TrackMovedBoids");
            incrementalUpdateKernel = quadTreeShader.FindKernel("IncrementalQuadTreeUpdate");
            repairTreeStructureKernel = quadTreeShader.FindKernel("RepairTreeStructure");
            collapseNodesKernel = quadTreeShader.FindKernel("CollapseNodes");
            initializeBoidHistoryKernel = quadTreeShader.FindKernel("InitializeBoidHistory");
        }
        
        public void Initialize(int numBoids, ComputeBuffer boidBuffer, ComputeBuffer boidBufferOut, ComputeShader boidShader)
        {
            if (quadTreeShader == null) return;
            
            // Create standard buffers
            quadNodesBuffer = new ComputeBuffer(MaxQuadNodes, 28); // 28 bytes per node
            nodeCountBuffer = new ComputeBuffer(1, 4);
            boidIndicesBuffer = new ComputeBuffer(numBoids * 2, 4);
            activeNodesBuffer = new ComputeBuffer(MaxQuadNodes, 4);
            activeNodeCountBuffer = new ComputeBuffer(1, 4);
            nodeCountsBuffer = new ComputeBuffer(MaxQuadNodes, 4);
            
            // Create incremental update buffers
            boidHistoryBuffer = new ComputeBuffer(numBoids, 28); // BoidWithHistory struct (pos, prevPos, vel, team, nodeIndex)
            movedBoidIndicesBuffer = new ComputeBuffer(numBoids, 4);
            movedBoidCountBuffer = new ComputeBuffer(1, 4);
            
            // Initialize moved boid count to 0
            uint[] initialMovedCount = new uint[] { 0 };
            movedBoidCountBuffer.SetData(initialMovedCount);
            
            // Set initial parameters - adjust maxBoidsPerNode for high boid counts
            int adjustedMaxBoidsPerNode = numBoids > HighBoidCountThreshold ? maxBoidsPerNode * 4 : maxBoidsPerNode;
            
            quadTreeShader.SetInt("maxDepth", maxQuadTreeDepth);
            quadTreeShader.SetInt("maxBoidsPerNode", adjustedMaxBoidsPerNode);
            quadTreeShader.SetInt("numBoids", numBoids);
            quadTreeShader.SetFloat("worldSize", initialQuadTreeSize);
            
            // Initialize root node
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

            uint[] initialActiveNodes = new uint[MaxQuadNodes];
            initialActiveNodes[0] = 0; // First active node is the root
            activeNodesBuffer.SetData(initialActiveNodes);
            
            // Set up the buffer bindings between quadtree and boid shader
            boidShader.SetBuffer(boidShader.FindKernel("UpdateBoids"), "quadNodes", quadNodesBuffer);
            boidShader.SetBuffer(boidShader.FindKernel("UpdateBoids"), "activeNodes", activeNodesBuffer);
            boidShader.SetBuffer(boidShader.FindKernel("UpdateBoids"), "boidIndices", boidIndicesBuffer);
            boidShader.SetBuffer(boidShader.FindKernel("UpdateBoids"), "nodeCount", nodeCountBuffer);
            boidShader.SetInt("useQuadTree", useQuadTree ? 1 : 0);
            
            // Set up all kernel buffers
            SetupKernelBuffers(boidBuffer, boidBufferOut);
            
            // Initialize boid history for tracking movement
            quadTreeShader.SetBuffer(initializeBoidHistoryKernel, "boids", boidBuffer);
            quadTreeShader.SetBuffer(initializeBoidHistoryKernel, "boidHistory", boidHistoryBuffer);
            quadTreeShader.SetBuffer(initializeBoidHistoryKernel, "movedBoidCount", movedBoidCountBuffer);
            quadTreeShader.SetInt("numBoids", numBoids);
            
            ProfilingUtility.BeginSample("InitializeBoidHistory");
            quadTreeShader.Dispatch(initializeBoidHistoryKernel, Mathf.CeilToInt(numBoids / 256f), 1, 1);
            ProfilingUtility.EndSample("InitializeBoidHistory");
        }
        
        private void SetupKernelBuffers(ComputeBuffer boidBuffer, ComputeBuffer boidBufferOut)
        {
            // Set buffers for standard kernels
            quadTreeShader.SetBuffer(recountBoidsKernel, "boidIndices", boidIndicesBuffer);
            quadTreeShader.SetBuffer(recountBoidsKernel, "nodeCounts", nodeCountsBuffer);
            
            quadTreeShader.SetBuffer(clearQuadTreeKernel, "quadNodes", quadNodesBuffer);
            quadTreeShader.SetBuffer(clearQuadTreeKernel, "nodeCount", nodeCountBuffer);
            quadTreeShader.SetBuffer(clearQuadTreeKernel, "activeNodes", activeNodesBuffer);
            quadTreeShader.SetBuffer(clearQuadTreeKernel, "activeNodeCount", activeNodeCountBuffer);
            
            quadTreeShader.SetBuffer(insertBoidsKernel, "boids", boidBuffer);
            quadTreeShader.SetBuffer(insertBoidsKernel, "quadNodes", quadNodesBuffer);
            quadTreeShader.SetBuffer(insertBoidsKernel, "nodeCount", nodeCountBuffer);
            quadTreeShader.SetBuffer(insertBoidsKernel, "boidIndices", boidIndicesBuffer);
            quadTreeShader.SetBuffer(insertBoidsKernel, "nodeCounts", nodeCountsBuffer);
            
            quadTreeShader.SetBuffer(sortBoidsKernel, "boids", boidBuffer);
            quadTreeShader.SetBuffer(sortBoidsKernel, "boidsOut", boidBufferOut);
            quadTreeShader.SetBuffer(sortBoidsKernel, "quadNodes", quadNodesBuffer);
            quadTreeShader.SetBuffer(sortBoidsKernel, "boidIndices", boidIndicesBuffer);
            quadTreeShader.SetBuffer(sortBoidsKernel, "nodeCount", nodeCountBuffer);
            
            quadTreeShader.SetBuffer(updateNodeCountsKernel, "nodeCounts", nodeCountsBuffer);
            quadTreeShader.SetBuffer(updateNodeCountsKernel, "quadNodes", quadNodesBuffer);
            quadTreeShader.SetBuffer(updateNodeCountsKernel, "nodeCount", nodeCountBuffer);
            
            quadTreeShader.SetBuffer(clearNodeCountsKernel, "nodeCounts", nodeCountsBuffer);
            
            // Set buffers for unified kernel if available
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
            }
            
            // Set buffers for subdivideAndRedistribute kernel if available
            if (subdivideAndRedistributeKernel >= 0)
            {
                quadTreeShader.SetBuffer(subdivideAndRedistributeKernel, "quadNodes", quadNodesBuffer);
                quadTreeShader.SetBuffer(subdivideAndRedistributeKernel, "nodeCount", nodeCountBuffer);
                quadTreeShader.SetBuffer(subdivideAndRedistributeKernel, "activeNodes", activeNodesBuffer);
                quadTreeShader.SetBuffer(subdivideAndRedistributeKernel, "activeNodeCount", activeNodeCountBuffer);
                quadTreeShader.SetBuffer(subdivideAndRedistributeKernel, "boidIndices", boidIndicesBuffer);
                quadTreeShader.SetBuffer(subdivideAndRedistributeKernel, "nodeCounts", nodeCountsBuffer);
                quadTreeShader.SetBuffer(subdivideAndRedistributeKernel, "boids", boidBuffer);
            }
            
            // Set buffers for initialize tree kernel
            if (initializeTreeKernel >= 0)
            {
                quadTreeShader.SetBuffer(initializeTreeKernel, "quadNodes", quadNodesBuffer);
                quadTreeShader.SetBuffer(initializeTreeKernel, "nodeCount", nodeCountBuffer);
                quadTreeShader.SetBuffer(initializeTreeKernel, "activeNodes", activeNodesBuffer);
                quadTreeShader.SetBuffer(initializeTreeKernel, "activeNodeCount", activeNodeCountBuffer);
                quadTreeShader.SetBuffer(initializeTreeKernel, "nodeCounts", nodeCountsBuffer);
            }
            
            // Set buffers for incremental update kernels
            if (trackMovedBoidsKernel >= 0)
            {
                quadTreeShader.SetBuffer(trackMovedBoidsKernel, "boids", boidBuffer);
                quadTreeShader.SetBuffer(trackMovedBoidsKernel, "boidHistory", boidHistoryBuffer);
                quadTreeShader.SetBuffer(trackMovedBoidsKernel, "movedBoidIndices", movedBoidIndicesBuffer);
                quadTreeShader.SetBuffer(trackMovedBoidsKernel, "movedBoidCount", movedBoidCountBuffer);
            }
            
            if (incrementalUpdateKernel >= 0)
            {
                quadTreeShader.SetBuffer(incrementalUpdateKernel, "boids", boidBuffer);
                quadTreeShader.SetBuffer(incrementalUpdateKernel, "boidHistory", boidHistoryBuffer);
                quadTreeShader.SetBuffer(incrementalUpdateKernel, "movedBoidIndices", movedBoidIndicesBuffer);
                quadTreeShader.SetBuffer(incrementalUpdateKernel, "movedBoidCount", movedBoidCountBuffer);
                quadTreeShader.SetBuffer(incrementalUpdateKernel, "quadNodes", quadNodesBuffer);
                quadTreeShader.SetBuffer(incrementalUpdateKernel, "nodeCounts", nodeCountsBuffer);
                quadTreeShader.SetBuffer(incrementalUpdateKernel, "boidIndices", boidIndicesBuffer);
            }
            
            if (repairTreeStructureKernel >= 0)
            {
                quadTreeShader.SetBuffer(repairTreeStructureKernel, "quadNodes", quadNodesBuffer);
                quadTreeShader.SetBuffer(repairTreeStructureKernel, "nodeCount", nodeCountBuffer);
                quadTreeShader.SetBuffer(repairTreeStructureKernel, "nodeCounts", nodeCountsBuffer);
                quadTreeShader.SetBuffer(repairTreeStructureKernel, "activeNodes", activeNodesBuffer);
                quadTreeShader.SetBuffer(repairTreeStructureKernel, "activeNodeCount", activeNodeCountBuffer);
            }
            
            if (collapseNodesKernel >= 0)
            {
                quadTreeShader.SetBuffer(collapseNodesKernel, "quadNodes", quadNodesBuffer);
                quadTreeShader.SetBuffer(collapseNodesKernel, "nodeCount", nodeCountBuffer);
                quadTreeShader.SetBuffer(collapseNodesKernel, "nodeCounts", nodeCountsBuffer);
                quadTreeShader.SetBuffer(collapseNodesKernel, "activeNodes", activeNodesBuffer);
                quadTreeShader.SetBuffer(collapseNodesKernel, "activeNodeCount", activeNodeCountBuffer);
                quadTreeShader.SetBuffer(collapseNodesKernel, "boidIndices", boidIndicesBuffer);
                quadTreeShader.SetBuffer(collapseNodesKernel, "boidHistory", boidHistoryBuffer);
            }
        }
        
        public void UpdateQuadTree(int numBoids, ComputeBuffer boidBuffer, ComputeBuffer boidBufferOut, ComputeShader boidShader)
        {
            if (!useQuadTree || quadTreeShader == null) return;

            // Set shared parameters
            quadTreeShader.SetInt("numBoids", numBoids);
            
            // Check if we should do a full rebuild or incremental update
            // For high boid counts, use a longer rebuild interval
            int rebuildInterval = numBoids > HighBoidCountThreshold ? highBoidCountRebuildInterval : fullRebuildInterval;
            bool doFullRebuild = !useIncrementalUpdates || Time.frameCount % rebuildInterval == 0;

            if (!doFullRebuild) {
                // INCREMENTAL UPDATE APPROACH
                ProfilingUtility.BeginSample("IncrementalQuadTreeUpdate");
                
                // 1. Track which boids have moved significantly
                ProfilingUtility.BeginSample("TrackMovedBoids");
                quadTreeShader.SetBuffer(trackMovedBoidsKernel, "boids", boidBuffer);
                quadTreeShader.SetBuffer(trackMovedBoidsKernel, "boidHistory", boidHistoryBuffer);
                quadTreeShader.SetBuffer(trackMovedBoidsKernel, "movedBoidIndices", movedBoidIndicesBuffer);
                quadTreeShader.SetBuffer(trackMovedBoidsKernel, "movedBoidCount", movedBoidCountBuffer);
                quadTreeShader.SetFloat("deltaTime", Time.deltaTime);
                
                TimedDispatch(
                    quadTreeShader,
                    trackMovedBoidsKernel,
                    Mathf.CeilToInt(numBoids / 256f), 1, 1,
                    "TrackMovedBoids",
                    numBoids
                );
                ProfilingUtility.EndSample("TrackMovedBoids");
                
                // Read back the moved boid count to determine if we need to continue with incremental update
                uint[] movedCount = new uint[1];
                movedBoidCountBuffer.GetData(movedCount);
                
                // Occasionally log how many boids needed updating (less frequently for high boid counts)
                int logInterval = numBoids > HighBoidCountThreshold ? 300 : 60;
                if (Time.frameCount % logInterval == 0) {
                    float percentMoved = (movedCount[0] / (float)numBoids) * 100f;
                    Debug.Log($"Incremental update: {movedCount[0]}/{numBoids} boids ({percentMoved:F1}%) near boundary");
                }
                
                if (movedCount[0] > 0) {
                    // Clear node counts for accurate recounting
                    ProfilingUtility.BeginSample("ClearNodeCounts");
                    quadTreeShader.Dispatch(clearNodeCountsKernel, Mathf.CeilToInt(MaxQuadNodes / 256f), 1, 1);
                    ProfilingUtility.EndSample("ClearNodeCounts");
                    
                    // 2. Update the quadtree for boids that moved
                    ProfilingUtility.BeginSample("UpdateMovedBoids");
                    quadTreeShader.SetBuffer(incrementalUpdateKernel, "boids", boidBuffer);
                    quadTreeShader.SetBuffer(incrementalUpdateKernel, "boidHistory", boidHistoryBuffer);
                    quadTreeShader.SetBuffer(incrementalUpdateKernel, "movedBoidIndices", movedBoidIndicesBuffer);
                    quadTreeShader.SetBuffer(incrementalUpdateKernel, "movedBoidCount", movedBoidCountBuffer);
                    quadTreeShader.SetBuffer(incrementalUpdateKernel, "quadNodes", quadNodesBuffer);
                    quadTreeShader.SetBuffer(incrementalUpdateKernel, "nodeCounts", nodeCountsBuffer);
                    quadTreeShader.SetBuffer(incrementalUpdateKernel, "boidIndices", boidIndicesBuffer);
                    
                    // Use enough thread groups to cover all moved boids
                    TimedDispatch(
                        quadTreeShader,
                        incrementalUpdateKernel,
                        Mathf.CeilToInt(movedCount[0] / 256f), 1, 1,
                        "IncrementalUpdate",
                        (int)movedCount[0]
                    );
                    ProfilingUtility.EndSample("UpdateMovedBoids");

                    // 3. Update node counts from atomic counters
                    ProfilingUtility.BeginSample("UpdateNodeCounts");
                    quadTreeShader.Dispatch(updateNodeCountsKernel, Mathf.CeilToInt(MaxQuadNodes / 256f), 1, 1);
                    ProfilingUtility.EndSample("UpdateNodeCounts");
                    
                    // 4. Check if nodes need subdivision or merging
                    ProfilingUtility.BeginSample("RepairTreeStructure");
                    int adjustedMaxBoidsPerNode = numBoids > HighBoidCountThreshold ? maxBoidsPerNode * 4 : maxBoidsPerNode;
                    quadTreeShader.SetInt("maxBoidsPerNode", adjustedMaxBoidsPerNode);
                    
                    TimedDispatch(
                        quadTreeShader,
                        repairTreeStructureKernel,
                        Mathf.CeilToInt(MaxQuadNodes / 256f), 1, 1,
                        "RepairTreeStructure",
                        numBoids
                    );
                    ProfilingUtility.EndSample("RepairTreeStructure");
                    
                    // Check for active nodes that need processing
                    uint[] activeCount = new uint[1];
                    activeNodeCountBuffer.GetData(activeCount);
                    
                    if (activeCount[0] > 0) {
                        // If there are active nodes, process them (subdivide or collapse)
                        // First try to subdivide nodes that need it
                        if (subdivideAndRedistributeKernel >= 0) {
                            ProfilingUtility.BeginSample("SubdivideNodes");
                            quadTreeShader.Dispatch(subdivideAndRedistributeKernel, 
                                Mathf.CeilToInt(activeCount[0] / 256f), 1, 1);
                            ProfilingUtility.EndSample("SubdivideNodes");
                        }
                        
                        // Then try to collapse nodes that are nearly empty
                        if (collapseNodesKernel >= 0) {
                            ProfilingUtility.BeginSample("CollapseNodes");
                            TimedDispatch(
                                quadTreeShader,
                                collapseNodesKernel,
                                Mathf.CeilToInt(activeCount[0] / 256f), 1, 1,
                                "CollapseNodes",
                                numBoids
                            );
                            ProfilingUtility.EndSample("CollapseNodes");
                        }
                    }
                }
                
                // Sort boids based on the quadtree
                ProfilingUtility.BeginSample("SortBoids");
                quadTreeShader.SetBuffer(sortBoidsKernel, "boids", boidBuffer);
                quadTreeShader.SetBuffer(sortBoidsKernel, "boidsOut", boidBufferOut);
                
                TimedDispatch(
                    quadTreeShader,
                    sortBoidsKernel,
                    Mathf.CeilToInt(numBoids / 256f), 1, 1,
                    "SortBoidsGPU",
                    numBoids
                );
                ProfilingUtility.EndSample("SortBoids");
                
                // Run the boid update using the maintained quadtree
                boidShader.SetBuffer(boidShader.FindKernel("UpdateBoids"), "boidsIn", boidBuffer);
                boidShader.SetBuffer(boidShader.FindKernel("UpdateBoids"), "boidsOut", boidBufferOut);
                boidShader.SetFloat("deltaTime", Time.deltaTime);
                boidShader.SetInt("useQuadTree", 1);
                
                ProfilingUtility.BeginSample("UpdateBoidKernel");
                TimedDispatch(
                    boidShader,
                    boidShader.FindKernel("UpdateBoids"),
                    Mathf.CeilToInt(numBoids / 256f), 1, 1,
                    "UpdateBoidKernelIncremental",
                    numBoids
                );
                ProfilingUtility.EndSample("UpdateBoidKernel");
                
                ProfilingUtility.EndSample("IncrementalQuadTreeUpdate");
                return;
            }

            // Full rebuild approach (used periodically to prevent any potential degradation)
            // First, completely clear the quadtree and node counts
            quadTreeShader.SetBuffer(clearNodeCountsKernel, "nodeCounts", nodeCountsBuffer);
            ProfilingUtility.BeginSample("ClearNodeCounts");
            quadTreeShader.Dispatch(clearNodeCountsKernel, Mathf.CeilToInt(MaxQuadNodes / 256f), 1, 1);
            ProfilingUtility.EndSample("ClearNodeCounts");

            // Approach 1: Use a single unified kernel (most efficient, but may not work on all platforms)
            if (useUnifiedKernel && buildUnifiedKernel >= 0)
            {
                // Reset the root node
                int resetRootKernel = quadTreeShader.FindKernel("ResetRootNode");
                if (resetRootKernel >= 0) {
                    quadTreeShader.SetBuffer(resetRootKernel, "quadNodes", quadNodesBuffer);
                    quadTreeShader.SetFloat("worldSize", initialQuadTreeSize);
                    
                    ProfilingUtility.BeginSample("ResetRootNode");
                    quadTreeShader.Dispatch(resetRootKernel, 1, 1, 1);
                    ProfilingUtility.EndSample("ResetRootNode");
                }

                // Set all needed buffers
                quadTreeShader.SetBuffer(buildUnifiedKernel, "boids", boidBuffer);
                quadTreeShader.SetBuffer(buildUnifiedKernel, "boidsOut", boidBufferOut);
                
                // Adjust maxBoidsPerNode for high boid counts to reduce subdivisions
                int adjustedMaxBoidsPerNode = numBoids > HighBoidCountThreshold ? maxBoidsPerNode * 4 : maxBoidsPerNode;
                quadTreeShader.SetInt("maxBoidsPerNode", adjustedMaxBoidsPerNode);
                
                quadTreeShader.SetInt("numBoids", numBoids);
                quadTreeShader.SetFloat("worldSize", initialQuadTreeSize);

                // Single dispatch for the entire tree building process
                ProfilingUtility.BeginSample("BuildQuadtreeUnified");
                
                // Use a simple approach - ensure GL.Flush to make timing more accurate
                System.Diagnostics.Stopwatch sw = System.Diagnostics.Stopwatch.StartNew();
                
                // Dispatch the kernel
                quadTreeShader.Dispatch(buildUnifiedKernel, Mathf.Max(1, Mathf.CeilToInt(numBoids / 256f)), 1, 1);
                
                // Force GPU to finish work before stopping timer
                GL.Flush();
                
                sw.Stop();
                double ms = sw.Elapsed.TotalMilliseconds;
                
                // Record this timing for the profiler
                ProfilingUtility.RecordManualTiming("BuildQuadtreeUnified", ms);
                
                // Log occasional timing for debugging
                if (Time.frameCount % 60 == 0)
                {
                    Debug.Log($"BuildQuadtreeUnified timing: {ms:F3} ms");
                }
                
                ProfilingUtility.EndSample("BuildQuadtreeUnified");
                
                // Run the simulation using the built quadtree
                boidShader.SetBuffer(boidShader.FindKernel("UpdateBoids"), "boidsIn", boidBuffer);
                boidShader.SetBuffer(boidShader.FindKernel("UpdateBoids"), "boidsOut", boidBufferOut);
                boidShader.SetFloat("deltaTime", Time.deltaTime);
                boidShader.SetInt("useQuadTree", 1);
                
                ProfilingUtility.BeginSample("UpdateBoidKernel");
                
                // Use the helper method to time and profile the dispatch
                TimedDispatch(
                    boidShader, 
                    boidShader.FindKernel("UpdateBoids"), 
                    Mathf.CeilToInt(numBoids / 256f), 1, 1, 
                    "UpdateBoidKernelGPU", 
                    numBoids
                );
                
                ProfilingUtility.EndSample("UpdateBoidKernel");
            }
            // Approach 2: Use separate kernel dispatches
            else
            {
                // Reset the tree completely with a proper root node
                quadTreeShader.SetFloat("worldSize", initialQuadTreeSize);
                
                ProfilingUtility.BeginSample("InitializeQuadTree");
                quadTreeShader.Dispatch(initializeTreeKernel, 1, 1, 1);
                ProfilingUtility.EndSample("InitializeQuadTree");

                // Insert boids into quadtree
                quadTreeShader.SetInt("numBoids", numBoids);
                
                ProfilingUtility.BeginSample("InsertBoids");
                quadTreeShader.Dispatch(insertBoidsKernel, Mathf.CeilToInt(numBoids / 256f), 1, 1);
                ProfilingUtility.EndSample("InsertBoids");
                
                // Update node counts
                ProfilingUtility.BeginSample("UpdateNodeCounts");
                quadTreeShader.Dispatch(updateNodeCountsKernel, Mathf.CeilToInt(MaxQuadNodes / 256f), 1, 1);
                ProfilingUtility.EndSample("UpdateNodeCounts");
                
                // Use fewer iterations with combined subdivide and redistribute
                if (subdivideAndRedistributeKernel >= 0) {
                    int iterations = Mathf.Min(3, maxQuadTreeDepth);
                    for (int i = 0; i < iterations; i++) {
                        quadTreeShader.SetInt("numBoids", numBoids);
                        quadTreeShader.SetInt("maxBoidsPerNode", maxBoidsPerNode);
                        
                        // Dispatch with enough threads for both operations
                        int threadGroups = Mathf.Max(
                            Mathf.CeilToInt(numBoids / 256f),
                            Mathf.CeilToInt(MaxQuadNodes / 256f)
                        );
                        
                        ProfilingUtility.BeginSample($"SubdivideAndRedistribute_I{i}");
                        quadTreeShader.Dispatch(subdivideAndRedistributeKernel, threadGroups, 1, 1);
                        ProfilingUtility.EndSample($"SubdivideAndRedistribute_I{i}");
                        
                        // Clear and update node counts after each iteration
                        ProfilingUtility.BeginSample($"ClearNodeCounts_I{i}");
                        quadTreeShader.Dispatch(clearNodeCountsKernel, Mathf.CeilToInt(MaxQuadNodes / 256f), 1, 1);
                        ProfilingUtility.EndSample($"ClearNodeCounts_I{i}");
                        
                        // Recount all boids after redistribution
                        if (recountBoidsKernel >= 0) {
                            ProfilingUtility.BeginSample($"RecountBoids_I{i}");
                            quadTreeShader.Dispatch(recountBoidsKernel, Mathf.CeilToInt(numBoids / 256f), 1, 1);
                            ProfilingUtility.EndSample($"RecountBoids_I{i}");
                        }
                        
                        // Update node counts after recounting
                        ProfilingUtility.BeginSample($"UpdateNodeCounts_I{i}");
                        quadTreeShader.Dispatch(updateNodeCountsKernel, Mathf.CeilToInt(MaxQuadNodes / 256f), 1, 1);
                        ProfilingUtility.EndSample($"UpdateNodeCounts_I{i}");
                    }
                }
                
                // After updating node counts, apply a final redistribution
                int forceRedistributeKernel = quadTreeShader.FindKernel("ForceRedistributeRootBoids");
                quadTreeShader.SetBuffer(forceRedistributeKernel, "boids", boidBuffer);
                quadTreeShader.SetBuffer(forceRedistributeKernel, "boidIndices", boidIndicesBuffer);
                quadTreeShader.SetBuffer(forceRedistributeKernel, "quadNodes", quadNodesBuffer);
                quadTreeShader.SetBuffer(forceRedistributeKernel, "nodeCounts", nodeCountsBuffer);
                quadTreeShader.SetInt("numBoids", numBoids);
                
                ProfilingUtility.BeginSample("ForceRedistributeRootBoids");
                quadTreeShader.Dispatch(forceRedistributeKernel, Mathf.CeilToInt(numBoids / 256f), 1, 1);
                ProfilingUtility.EndSample("ForceRedistributeRootBoids");

                // Recount boids once more after final redistribution
                if (recountBoidsKernel >= 0) {
                    ProfilingUtility.BeginSample("RecountBoids_Final");
                    quadTreeShader.Dispatch(recountBoidsKernel, Mathf.CeilToInt(numBoids / 256f), 1, 1);
                    ProfilingUtility.EndSample("RecountBoids_Final");
                }

                // Update node counts again
                ProfilingUtility.BeginSample("UpdateNodeCounts_Final");
                quadTreeShader.Dispatch(updateNodeCountsKernel, Mathf.CeilToInt(MaxQuadNodes / 256f), 1, 1);
                ProfilingUtility.EndSample("UpdateNodeCounts_Final");
                
                // Sort boids according to quadtree
                quadTreeShader.SetInt("numBoids", numBoids);
                
                ProfilingUtility.BeginSample("SortBoids");
                
                // Use the helper method to time and profile the dispatch
                TimedDispatch(
                    quadTreeShader, 
                    sortBoidsKernel, 
                    Mathf.CeilToInt(numBoids / 256f), 1, 1, 
                    "SortBoidsGPU", 
                    numBoids
                );
                
                ProfilingUtility.EndSample("SortBoids");
                
                // Run the boid update
                boidShader.SetBuffer(boidShader.FindKernel("UpdateBoids"), "boidsIn", boidBuffer);
                boidShader.SetBuffer(boidShader.FindKernel("UpdateBoids"), "boidsOut", boidBufferOut);
                boidShader.SetFloat("deltaTime", Time.deltaTime);
                boidShader.SetInt("useQuadTree", 1);
                
                ProfilingUtility.BeginSample("UpdateBoidKernel");
                
                // Use the helper method to time and profile the dispatch
                TimedDispatch(
                    boidShader, 
                    boidShader.FindKernel("UpdateBoids"), 
                    Mathf.CeilToInt(numBoids / 256f), 1, 1, 
                    "UpdateBoidKernelMultiDisp", 
                    numBoids
                );
                
                ProfilingUtility.EndSample("UpdateBoidKernel");
            }
            
            if (printQuadTreeDebugInfo && Time.frameCount % debugPrintInterval == 0) {
                DebugPrintQuadTreeInfo();
            }
        }
        
        public void DebugPrintQuadTreeInfo()
        {
            if (!printQuadTreeDebugInfo || quadNodesBuffer == null || nodeCountBuffer == null)
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
            
            // Read node count and active node count from buffers
            uint[] counts = new uint[1];
            nodeCountBuffer.GetData(counts);
            int totalNodeCount = (int)counts[0];
            
            uint[] activeCounts = new uint[1];
            activeNodeCountBuffer.GetData(activeCounts);
            int activeNodeCount = (int)activeCounts[0];
            
            // Read node data - only enough for our display
            int maxNodesToShow = Mathf.Min(totalNodeCount, 50);
            QuadNode[] allNodes = new QuadNode[maxNodesToShow];
            quadNodesBuffer.GetData(allNodes, 0, 0, maxNodesToShow);
            
            // Build diagnostic report
            StringBuilder sb = new StringBuilder();
            sb.AppendLine("=== QUAD TREE STRUCTURE ===");
            sb.AppendLine($"Total nodes: {totalNodeCount}, Active nodes: {activeNodeCount}");
            
            // Only log up to a reasonable number of nodes to avoid spamming the console
            if (maxNodesToShow > 0) {
                int nodeToShow = Mathf.Min(10, maxNodesToShow);
                sb.AppendLine("\nFirst few nodes:");
                for (int i = 0; i < nodeToShow; i++) {
                    QuadNode node = allNodes[i];
                    bool isLeaf = (node.flags & NODE_LEAF) != 0;
                    bool isActive = (node.flags & NODE_ACTIVE) != 0;
                    sb.AppendLine($"Node {i}: Count={node.count}, Size={node.size}, Center=({node.center.x:F1},{node.center.y:F1}), Leaf={isLeaf}, Active={isActive}");
                }
            }
            
            // Log statistics about boid distribution
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
                
                // Update histogram
                if (boidCount == 0) countHistogram[0]++;
                else if (boidCount <= 5) countHistogram[1]++;
                else if (boidCount <= 10) countHistogram[2]++;
                else if (boidCount <= 20) countHistogram[3]++;
                else if (boidCount <= 50) countHistogram[4]++;
                else countHistogram[5]++;
            }
            
            sb.AppendLine("\n=== BOID DISTRIBUTION ===");
            sb.AppendLine($"Max boids in a single node: {maxBoidsInNode} (threshold: {maxBoidsPerNode})");
            sb.AppendLine($"Nodes with boids: {nodesWithBoids} of {totalNodeCount}");
            sb.AppendLine($"Empty nodes: {countHistogram[0]}");
            sb.AppendLine($"Nodes with 1-5 boids: {countHistogram[1]}");
            sb.AppendLine($"Nodes with 6-10 boids: {countHistogram[2]}");
            sb.AppendLine($"Nodes with 11-20 boids: {countHistogram[3]}");
            sb.AppendLine($"Nodes with 21-50 boids: {countHistogram[4]}");
            sb.AppendLine($"Nodes with >50 boids: {countHistogram[5]}");
            
            Debug.Log(sb.ToString());
            
            // Clean up
            diagBuffer.Release();
        }
        
        public void OnDrawGizmos(Transform parent)
        {
            // Only draw if the game is playing, visualization is enabled, and quadtree is being used
            if (!Application.isPlaying || !drawQuadTreeGizmos || !useQuadTree || quadNodesBuffer == null) 
                return;
            
            // Get node count
            uint[] counts = new uint[1];
            nodeCountBuffer.GetData(counts);
            int nodeCount = (int)Mathf.Min(counts[0], MaxQuadNodes);
            
            // If there are no nodes or too many, skip visualization
            if (nodeCount <= 0 || nodeCount > 1000) 
                return;
            
            // Read node data
            QuadNode[] allNodes = new QuadNode[nodeCount];
            quadNodesBuffer.GetData(allNodes, 0, 0, nodeCount);
            
            // Just visualize the root node and immediate children
            DrawQuadNodeWithInfo(allNodes, 0, 0, Color.green);
        }
        
        private void DrawQuadNodeWithInfo(QuadNode[] allNodes, uint nodeIndex, int depth, Color color)
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
            
            // Draw division lines for this node to show quadrants
            if (depth <= gizmoDetailLevel - 1)
            {
                Gizmos.color = new Color(color.r, color.g, color.b, 0.3f);
                Gizmos.DrawLine(
                    new Vector3(center.x - node.size, center.y, 0),
                    new Vector3(center.x + node.size, center.y, 0));
                Gizmos.DrawLine(
                    new Vector3(center.x, center.y - node.size, 0),
                    new Vector3(center.x, center.y + node.size, 0));
            }
            
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
                
                // Add label for editor display
                #if UNITY_EDITOR
                if (boidCount > maxBoidsPerNode / 2 || depth <= 1) {
                    UnityEditor.Handles.Label(center, $"Node {nodeIndex}: {boidCount}");
                }
                #endif
            }
            
            // Draw children if not a leaf and has valid children
            if ((node.flags & NODE_LEAF) == 0 && node.childIndex > 0 && node.childIndex < allNodes.Length)
            {
                // Alternate colors for child nodes
                Color childColor = new Color(color.r * 0.8f, color.g * 0.8f, color.b * 0.8f);
                
                // Draw connecting lines from parent to each child
                for (uint i = 0; i < 4; i++)
                {
                    uint childIdx = node.childIndex + i;
                    if (childIdx < allNodes.Length)
                    {
                        QuadNode childNode = allNodes[childIdx];
                        
                        // Calculate expected child center position based on quadrant
                        float childSize = node.size * 0.5f;
                        float offsetX = (i & 1) != 0 ? -childSize : childSize;
                        float offsetY = (i & 2) != 0 ? -childSize : childSize;
                        Vector3 expectedChildCenter = new Vector3(
                            node.center.x + offsetX,
                            node.center.y + offsetY,
                            0);
                        
                        // Check if child position matches expected position
                        Vector3 actualChildCenter = new Vector3(childNode.center.x, childNode.center.y, 0);
                        float discrepancy = Vector3.Distance(expectedChildCenter, actualChildCenter);
                        
                        // If there's significant discrepancy, highlight it
                        if (discrepancy > 0.01f)
                        {
                            Gizmos.color = Color.red;
                            Gizmos.DrawLine(center, actualChildCenter);
                            
                            // Draw expected position as a dashed outline
                            Gizmos.color = Color.yellow;
                            Gizmos.DrawWireSphere(expectedChildCenter, childSize * 0.1f);
                            
                            #if UNITY_EDITOR
                            UnityEditor.Handles.color = Color.red;
                            UnityEditor.Handles.Label(
                                Vector3.Lerp(center, actualChildCenter, 0.5f), 
                                "Misaligned");
                            #endif
                        }
                        else
                        {
                            // Draw thin connection line from parent to child
                            Gizmos.color = new Color(0.7f, 0.7f, 0.7f, 0.3f);
                            Gizmos.DrawLine(center, actualChildCenter);
                        }
                        
                        // Continue recursive drawing
                        DrawQuadNodeWithInfo(allNodes, childIdx, depth + 1, childColor);
                    }
                }
            }
        }
        
        // Helper method to time a GPU compute dispatch with profiling
        private void TimedDispatch(ComputeShader shader, int kernelId, int threadGroupsX, int threadGroupsY, int threadGroupsZ, string profileName, int boidsCount)
        {
            // Use stopwatch for precise timing
            System.Diagnostics.Stopwatch sw = System.Diagnostics.Stopwatch.StartNew();
            
            // Dispatch the compute shader
            shader.Dispatch(kernelId, threadGroupsX, threadGroupsY, threadGroupsZ);
            
            // Only force GPU to complete for profiling if below high boid count threshold
            // This is a critical optimization - GL.Flush() is very expensive with high boid counts
            bool isHighBoidCount = boidsCount > HighBoidCountThreshold;
            
            if (!isHighBoidCount || Time.frameCount % 300 == 0) {
                GL.Flush();
            }
            
            sw.Stop();
            double ms = sw.Elapsed.TotalMilliseconds;
            
            // Record timing data only if we forced a flush or for low boid counts
            if (!isHighBoidCount || Time.frameCount % 300 == 0) {
                ProfilingUtility.RecordManualTiming(profileName, ms);
            }
            
            // Log for high boid counts but only occasionally
            if (isHighBoidCount && Time.frameCount % 300 == 0)
            {
                Debug.Log($"{profileName} with {boidsCount} boids took {ms:F2}ms");
            }
        }
        
        public void Cleanup()
        {
            // Release standard buffers
            quadNodesBuffer?.Release();
            nodeCountBuffer?.Release();
            boidIndicesBuffer?.Release();
            activeNodesBuffer?.Release();
            activeNodeCountBuffer?.Release();
            nodeCountsBuffer?.Release();
            subdivDebugBuffer?.Release();
            
            // Release incremental update buffers
            boidHistoryBuffer?.Release();
            movedBoidIndicesBuffer?.Release();
            movedBoidCountBuffer?.Release();
        }
    }
}
