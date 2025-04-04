using UnityEngine;
using Unity.Mathematics;
using Unity.Collections;

namespace BoidsUnity
{
    public class GridManager
    {
        // Grid dimensions and properties
        public int gridDimX;
        public int gridDimY;
        public int gridTotalCells;
        public float gridCellSize;
        public int blocks;
        
        // Officer grid dimensions (4x smaller on each side, 16x smaller in total)
        public int officerGridDimX;
        public int officerGridDimY;
        public int officerGridTotalCells;
        public float officerGridCellSize;
        public int officerBlocks;
        
        // CPU-side data structures for the grid
        private NativeArray<int2> grid;
        private NativeArray<int> gridOffsets;
        
        // GPU buffers for the grid
        public ComputeBuffer gridBuffer;
        public ComputeBuffer gridOffsetBuffer;
        public ComputeBuffer gridOffsetBufferIn;
        public ComputeBuffer gridSumsBuffer;
        public ComputeBuffer gridSumsBuffer2;
        
        // GPU buffers for the officer grid
        public ComputeBuffer officerGridBuffer;
        public ComputeBuffer officerGridOffsetBuffer;
        public ComputeBuffer officerGridOffsetBufferIn;
        public ComputeBuffer officerGridSumsBuffer;
        public ComputeBuffer officerGridSumsBuffer2;
        
        // Kernel IDs
        private int updateGridKernel;
        private int clearGridKernel;
        private int prefixSumKernel;
        private int sumBlocksKernel;
        private int addSumsKernel;
        private int rearrangeBoidsKernel;
        
        // Officer grid kernel IDs
        private int officerPrefixSumKernel;
        private int officerSumBlocksKernel;
        private int officerAddSumsKernel;
        
        private ComputeShader gridShader;
        private BoidBehaviorManager behaviorManager;
        
        public GridManager(ComputeShader shader, BoidBehaviorManager boidManager)
        {
            gridShader = shader;
            behaviorManager = boidManager;
            
            // Get kernel IDs for regular grid
            updateGridKernel = gridShader.FindKernel("UpdateGrid");
            clearGridKernel = gridShader.FindKernel("ClearGrid");
            prefixSumKernel = gridShader.FindKernel("PrefixSum");
            sumBlocksKernel = gridShader.FindKernel("SumBlocks");
            addSumsKernel = gridShader.FindKernel("AddSums");
            rearrangeBoidsKernel = gridShader.FindKernel("RearrangeBoids");
            
            // Get kernel IDs for officer grid
            officerPrefixSumKernel = gridShader.FindKernel("OfficerPrefixSum");
            officerSumBlocksKernel = gridShader.FindKernel("OfficerSumBlocks");
            officerAddSumsKernel = gridShader.FindKernel("OfficerAddSums");
        }
        
        public void Initialize(int numBoids, float visualRange, float xBound, float yBound, ComputeBuffer boidBuffer, ComputeBuffer boidBufferOut)
        {
            // Set up the regular spatial grid
            gridCellSize = visualRange;
            gridDimX = Mathf.FloorToInt(xBound * 2 / gridCellSize) + 30;
            gridDimY = Mathf.FloorToInt(yBound * 2 / gridCellSize) + 30;
            gridTotalCells = gridDimX * gridDimY;
            blocks = Mathf.CeilToInt(gridTotalCells / 256f);
            
            // Set up the officer grid (4x smaller on each dimension)
            officerGridCellSize = visualRange * 4f;
            officerGridDimX = Mathf.FloorToInt(xBound * 2 / officerGridCellSize) + 8; // 8 instead of 30/4 for safety padding
            officerGridDimY = Mathf.FloorToInt(yBound * 2 / officerGridCellSize) + 8;
            officerGridTotalCells = officerGridDimX * officerGridDimY;
            officerBlocks = Mathf.CeilToInt(officerGridTotalCells / 256f);
            
            // Create CPU-side arrays if needed (for CPU mode)
            grid = new NativeArray<int2>(numBoids, Allocator.Persistent);
            gridOffsets = new NativeArray<int>(gridTotalCells, Allocator.Persistent);
            
            // Create regular grid GPU buffers
            gridBuffer = new ComputeBuffer(numBoids, 8);
            gridOffsetBuffer = new ComputeBuffer(gridTotalCells, 4);
            gridOffsetBufferIn = new ComputeBuffer(gridTotalCells, 4);
            gridSumsBuffer = new ComputeBuffer(blocks, 4);
            gridSumsBuffer2 = new ComputeBuffer(blocks, 4);
            
            // Create officer grid GPU buffers
            officerGridBuffer = new ComputeBuffer(numBoids, 8);
            officerGridOffsetBuffer = new ComputeBuffer(officerGridTotalCells, 4);
            officerGridOffsetBufferIn = new ComputeBuffer(officerGridTotalCells, 4);
            officerGridSumsBuffer = new ComputeBuffer(officerBlocks, 4);
            officerGridSumsBuffer2 = new ComputeBuffer(officerBlocks, 4);
            
            // Set up shader parameters - regular grid
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

            // Set up officer grid buffers
            gridShader.SetBuffer(updateGridKernel, "officerGridBuffer", officerGridBuffer);
            gridShader.SetBuffer(updateGridKernel, "officerGridOffsetBuffer", officerGridOffsetBufferIn);
            
            gridShader.SetBuffer(clearGridKernel, "officerGridOffsetBuffer", officerGridOffsetBufferIn);
            
            gridShader.SetBuffer(officerPrefixSumKernel, "officerGridOffsetBuffer", officerGridOffsetBuffer);
            gridShader.SetBuffer(officerPrefixSumKernel, "officerGridOffsetBufferIn", officerGridOffsetBufferIn);
            gridShader.SetBuffer(officerPrefixSumKernel, "officerGridSumsBuffer", officerGridSumsBuffer2);
            
            gridShader.SetBuffer(officerSumBlocksKernel, "officerGridSumsBuffer", officerGridSumsBuffer);
            gridShader.SetBuffer(officerSumBlocksKernel, "officerGridSumsBufferIn", officerGridSumsBuffer2);
            
            gridShader.SetBuffer(officerAddSumsKernel, "officerGridOffsetBuffer", officerGridOffsetBuffer);
            gridShader.SetBuffer(officerAddSumsKernel, "officerGridSumsBufferIn", officerGridSumsBuffer);

            // Regular grid dimensions/properties
            gridShader.SetFloat("gridCellSize", gridCellSize);
            gridShader.SetInt("gridDimY", gridDimY);
            gridShader.SetInt("gridDimX", gridDimX);
            gridShader.SetInt("gridTotalCells", gridTotalCells);
            gridShader.SetInt("blocks", blocks);
            
            // Officer grid dimensions/properties
            gridShader.SetFloat("officerGridCellSize", officerGridCellSize);
            gridShader.SetInt("officerGridDimY", officerGridDimY);
            gridShader.SetInt("officerGridDimX", officerGridDimX);
            gridShader.SetInt("officerGridTotalCells", officerGridTotalCells);
            gridShader.SetInt("officerBlocks", officerBlocks);
        }
        
        public void UpdateGridGPU(int numBoids, ComputeBuffer boidBuffer, ComputeBuffer boidBufferOut)
        {   
            // Clear indices for both regular and officer grids
            ProfilingUtility.BeginSample("ClearGridIndices");
            gridShader.Dispatch(clearGridKernel, Mathf.Max(blocks, officerBlocks), 1, 1);
            ProfilingUtility.EndSample("ClearGridIndices");

            // Populate both grids
            ProfilingUtility.BeginSample("PopulateGrid");
            gridShader.Dispatch(updateGridKernel, Mathf.CeilToInt(numBoids / 256f), 1, 1);
            ProfilingUtility.EndSample("PopulateGrid");

            // REGULAR GRID PROCESSING
            // Generate Offsets (Prefix Sum)
            // Offsets in each block
            ProfilingUtility.BeginSample("PrefixSumBlocks");
            gridShader.Dispatch(prefixSumKernel, blocks, 1, 1);
            ProfilingUtility.EndSample("PrefixSumBlocks");

            // Offsets for sums of blocks
            ProfilingUtility.BeginSample("PrefixSumHierarchical");
            bool swap = false;
            for (int d = 1; d < blocks; d *= 2)
            {
                gridShader.SetBuffer(sumBlocksKernel, "gridSumsBufferIn", swap ? gridSumsBuffer : gridSumsBuffer2);
                gridShader.SetBuffer(sumBlocksKernel, "gridSumsBuffer", swap ? gridSumsBuffer2 : gridSumsBuffer);
                gridShader.SetInt("d", d);
                gridShader.Dispatch(sumBlocksKernel, Mathf.CeilToInt(blocks / 256f), 1, 1);
                swap = !swap;
            }
            ProfilingUtility.EndSample("PrefixSumHierarchical");

            // Apply offsets of sums to each block
            ProfilingUtility.BeginSample("ApplySums");
            gridShader.SetBuffer(addSumsKernel, "gridSumsBufferIn", swap ? gridSumsBuffer : gridSumsBuffer2);
            gridShader.Dispatch(addSumsKernel, blocks, 1, 1);
            ProfilingUtility.EndSample("ApplySums");

            // OFFICER GRID PROCESSING
            // Generate Offsets (Prefix Sum) for officer grid
            ProfilingUtility.BeginSample("OfficerPrefixSumBlocks");
            gridShader.Dispatch(officerPrefixSumKernel, officerBlocks, 1, 1);
            ProfilingUtility.EndSample("OfficerPrefixSumBlocks");

            // Offsets for sums of blocks in officer grid
            ProfilingUtility.BeginSample("OfficerPrefixSumHierarchical");
            bool officerSwap = false;
            for (int d = 1; d < officerBlocks; d *= 2)
            {
                gridShader.SetBuffer(officerSumBlocksKernel, "officerGridSumsBufferIn", officerSwap ? officerGridSumsBuffer : officerGridSumsBuffer2);
                gridShader.SetBuffer(officerSumBlocksKernel, "officerGridSumsBuffer", officerSwap ? officerGridSumsBuffer2 : officerGridSumsBuffer);
                gridShader.SetInt("d", d);
                gridShader.Dispatch(officerSumBlocksKernel, Mathf.CeilToInt(officerBlocks / 256f), 1, 1);
                officerSwap = !officerSwap;
            }
            ProfilingUtility.EndSample("OfficerPrefixSumHierarchical");

            // Apply offsets of sums to each officer block
            ProfilingUtility.BeginSample("OfficerApplySums");
            gridShader.SetBuffer(officerAddSumsKernel, "officerGridSumsBufferIn", officerSwap ? officerGridSumsBuffer : officerGridSumsBuffer2);
            gridShader.Dispatch(officerAddSumsKernel, officerBlocks, 1, 1);
            ProfilingUtility.EndSample("OfficerApplySums");

            // Rearrange boids
            ProfilingUtility.BeginSample("RearrangeBoids");
            
            // Time this dispatch specifically
            System.Diagnostics.Stopwatch sw = System.Diagnostics.Stopwatch.StartNew();
            
            gridShader.Dispatch(rearrangeBoidsKernel, Mathf.CeilToInt(numBoids / 256f), 1, 1);
            
            // Force GPU to complete work
            GL.Flush();
            
            sw.Stop();
            double ms = sw.Elapsed.TotalMilliseconds;
            
            // Record timing for high boid counts
            if (numBoids > 10000)
            {
                ProfilingUtility.RecordManualTiming("RearrangeBoidsGPU", ms);
                
                if (numBoids > 25000 && Time.frameCount % 120 == 0)
                {
                    Debug.Log($"Rearranging {numBoids} boids took {ms:F2}ms");
                }
            }
            
            ProfilingUtility.EndSample("RearrangeBoids");
        }
        
        // CPU-side grid functions
        public int GetGridID(Boid boid)
        {
            int gridX = Mathf.FloorToInt(boid.pos.x / gridCellSize + gridDimX / 2);
            int gridY = Mathf.FloorToInt(boid.pos.y / gridCellSize + gridDimY / 2);
            return (gridDimX * gridY) + gridX;
        }

        public void ClearGrid()
        {
            for (int i = 0; i < gridTotalCells; i++)
            {
                gridOffsets[i] = 0;
            }
        }

        public void UpdateGrid(NativeArray<Boid> boids, int numBoids)
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

        public void GenerateGridOffsets()
        {
            for (int i = 1; i < gridTotalCells; i++)
            {
                gridOffsets[i] += gridOffsets[i - 1];
            }
        }

        public void RearrangeBoids(NativeArray<Boid> boids, NativeArray<Boid> boidsTemp, int numBoids)
        {
            for (int i = 0; i < numBoids; i++)
            {
                int gridID = grid[i].x;
                int cellOffset = grid[i].y;
                int index = gridOffsets[gridID] - 1 - cellOffset;
                boidsTemp[index] = boids[i];
            }
        }
        
        public void Cleanup()
        {
            if (grid.IsCreated) grid.Dispose();
            if (gridOffsets.IsCreated) gridOffsets.Dispose();
            
            // Release regular grid buffers
            gridBuffer?.Release();
            gridOffsetBuffer?.Release();
            gridOffsetBufferIn?.Release();
            gridSumsBuffer?.Release();
            gridSumsBuffer2?.Release();
            
            // Release officer grid buffers
            officerGridBuffer?.Release();
            officerGridOffsetBuffer?.Release();
            officerGridOffsetBufferIn?.Release();
            officerGridSumsBuffer?.Release();
            officerGridSumsBuffer2?.Release();
        }
        
        public int[] GetGridOffsets()
        {
            return gridOffsets.ToArray();
        }
    }
}
