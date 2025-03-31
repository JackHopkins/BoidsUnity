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
        private const int MaxQuadNodes = 16384;
        
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
        
        public QuadTreeManager(ComputeShader shader)
        {
            quadTreeShader = shader;
            
            // Get kernel IDs
            recountBoidsKernel = quadTreeShader.FindKernel("RecountBoids");
            clearQuadTreeKernel = quadTreeShader.FindKernel("ClearQuadTree");
            insertBoidsKernel = quadTreeShader.FindKernel("InsertBoids");
            sortBoidsKernel = quadTreeShader.FindKernel("SortBoids");
            clearNodeCountsKernel = quadTreeShader.FindKernel("ClearNodeCounts");
            updateNodeCountsKernel = quadTreeShader.FindKernel("UpdateNodeCounts");
            initializeTreeKernel = quadTreeShader.FindKernel("InitializeQuadTree");
            buildUnifiedKernel = quadTreeShader.FindKernel("BuildQuadtreeUnified");
            subdivideAndRedistributeKernel = quadTreeShader.FindKernel("SubdivideAndRedistribute");
        }
        
        public void Initialize(int numBoids, ComputeBuffer boidBuffer, ComputeBuffer boidBufferOut, ComputeShader boidShader)
        {
            if (quadTreeShader == null) return;
            
            // Create buffers
            quadNodesBuffer = new ComputeBuffer(MaxQuadNodes, 28); // 28 bytes per node
            nodeCountBuffer = new ComputeBuffer(1, 4);
            boidIndicesBuffer = new ComputeBuffer(numBoids * 2, 4);
            activeNodesBuffer = new ComputeBuffer(MaxQuadNodes, 4);
            activeNodeCountBuffer = new ComputeBuffer(1, 4);
            nodeCountsBuffer = new ComputeBuffer(MaxQuadNodes, 4);
            
            // Set initial parameters
            quadTreeShader.SetInt("maxDepth", maxQuadTreeDepth);
            quadTreeShader.SetInt("maxBoidsPerNode", maxBoidsPerNode);
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
            
            SetupKernelBuffers(boidBuffer, boidBufferOut);
        }
        
        private void SetupKernelBuffers(ComputeBuffer boidBuffer, ComputeBuffer boidBufferOut)
        {
            // Set buffers for various kernels
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
        }
        
        public void UpdateQuadTree(int numBoids, ComputeBuffer boidBuffer, ComputeBuffer boidBufferOut, ComputeShader boidShader)
        {
            if (!useQuadTree || quadTreeShader == null) return;

            // OPTIMIZATION 1: Only rebuild quadtree every N frames
            // This dramatically reduces GPU overhead when boids move slowly
            if (Time.frameCount % 3 != 0) {
                // Just run the simulation using the existing quadtree
                boidShader.SetBuffer(boidShader.FindKernel("UpdateBoids"), "boidsIn", boidBuffer);
                boidShader.SetBuffer(boidShader.FindKernel("UpdateBoids"), "boidsOut", boidBufferOut);
                boidShader.SetFloat("deltaTime", Time.deltaTime);
                boidShader.SetInt("useQuadTree", 1);
                boidShader.Dispatch(boidShader.FindKernel("UpdateBoids"), Mathf.CeilToInt(numBoids / 256f), 1, 1);
                return;
            }

            // First, completely clear the quadtree and node counts
            quadTreeShader.SetBuffer(clearNodeCountsKernel, "nodeCounts", nodeCountsBuffer);
            quadTreeShader.Dispatch(clearNodeCountsKernel, Mathf.CeilToInt(MaxQuadNodes / 256f), 1, 1);

            // Approach 1: Use a single unified kernel (most efficient, but may not work on all platforms)
            if (useUnifiedKernel && buildUnifiedKernel >= 0)
            {
                // Reset the root node
                int resetRootKernel = quadTreeShader.FindKernel("ResetRootNode");
                if (resetRootKernel >= 0) {
                    quadTreeShader.SetBuffer(resetRootKernel, "quadNodes", quadNodesBuffer);
                    quadTreeShader.SetFloat("worldSize", initialQuadTreeSize);
                    quadTreeShader.Dispatch(resetRootKernel, 1, 1, 1);
                }

                // Set all needed buffers
                quadTreeShader.SetBuffer(buildUnifiedKernel, "boids", boidBuffer);
                quadTreeShader.SetBuffer(buildUnifiedKernel, "boidsOut", boidBufferOut);
                quadTreeShader.SetInt("maxBoidsPerNode", maxBoidsPerNode);
                quadTreeShader.SetInt("numBoids", numBoids);
                quadTreeShader.SetFloat("worldSize", initialQuadTreeSize);

                // Single dispatch for the entire tree building process
                quadTreeShader.Dispatch(buildUnifiedKernel, Mathf.Max(1, Mathf.CeilToInt(numBoids / 256f)), 1, 1);
                
                // Run the simulation using the built quadtree
                boidShader.SetBuffer(boidShader.FindKernel("UpdateBoids"), "boidsIn", boidBuffer);
                boidShader.SetBuffer(boidShader.FindKernel("UpdateBoids"), "boidsOut", boidBufferOut);
                boidShader.SetFloat("deltaTime", Time.deltaTime);
                boidShader.SetInt("useQuadTree", 1);
                boidShader.Dispatch(boidShader.FindKernel("UpdateBoids"), Mathf.CeilToInt(numBoids / 256f), 1, 1);
            }
            // Approach 2: Use separate kernel dispatches
            else
            {
                // Reset the tree completely with a proper root node
                quadTreeShader.SetFloat("worldSize", initialQuadTreeSize);
                quadTreeShader.Dispatch(initializeTreeKernel, 1, 1, 1);

                // Insert boids into quadtree
                quadTreeShader.SetInt("numBoids", numBoids);
                quadTreeShader.Dispatch(insertBoidsKernel, Mathf.CeilToInt(numBoids / 256f), 1, 1);
                
                // Update node counts
                quadTreeShader.Dispatch(updateNodeCountsKernel, Mathf.CeilToInt(MaxQuadNodes / 256f), 1, 1);
                
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
                        quadTreeShader.Dispatch(subdivideAndRedistributeKernel, threadGroups, 1, 1);
                        
                        // Clear and update node counts after each iteration
                        quadTreeShader.Dispatch(clearNodeCountsKernel, Mathf.CeilToInt(MaxQuadNodes / 256f), 1, 1);
                        
                        // Recount all boids after redistribution
                        if (recountBoidsKernel >= 0) {
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

                // Recount boids once more after final redistribution
                if (recountBoidsKernel >= 0) {
                    quadTreeShader.Dispatch(recountBoidsKernel, Mathf.CeilToInt(numBoids / 256f), 1, 1);
                }

                // Update node counts again
                quadTreeShader.Dispatch(updateNodeCountsKernel, Mathf.CeilToInt(MaxQuadNodes / 256f), 1, 1);
                
                // Sort boids according to quadtree
                quadTreeShader.SetInt("numBoids", numBoids);
                quadTreeShader.Dispatch(sortBoidsKernel, Mathf.CeilToInt(numBoids / 256f), 1, 1);
                
                // Run the boid update
                boidShader.SetBuffer(boidShader.FindKernel("UpdateBoids"), "boidsIn", boidBuffer);
                boidShader.SetBuffer(boidShader.FindKernel("UpdateBoids"), "boidsOut", boidBufferOut);
                boidShader.SetFloat("deltaTime", Time.deltaTime);
                boidShader.SetInt("useQuadTree", 1);
                boidShader.Dispatch(boidShader.FindKernel("UpdateBoids"), Mathf.CeilToInt(numBoids / 256f), 1, 1);
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
        
        public void Cleanup()
        {
            quadNodesBuffer?.Release();
            nodeCountBuffer?.Release();
            boidIndicesBuffer?.Release();
            activeNodesBuffer?.Release();
            activeNodeCountBuffer?.Release();
            nodeCountsBuffer?.Release();
            subdivDebugBuffer?.Release();
        }
    }
}
