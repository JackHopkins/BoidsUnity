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
        
        // CPU-side data structures for the grid
        private NativeArray<int2> grid;
        private NativeArray<int> gridOffsets;
        
        // GPU buffers for the grid
        public ComputeBuffer gridBuffer;
        public ComputeBuffer gridOffsetBuffer;
        public ComputeBuffer gridOffsetBufferIn;
        public ComputeBuffer gridSumsBuffer;
        public ComputeBuffer gridSumsBuffer2;
        
        // Kernel IDs
        private int updateGridKernel;
        private int clearGridKernel;
        private int prefixSumKernel;
        private int sumBlocksKernel;
        private int addSumsKernel;
        private int rearrangeBoidsKernel;
        
        private ComputeShader gridShader;
        private BoidBehaviorManager behaviorManager;
        
        public GridManager(ComputeShader shader, BoidBehaviorManager boidManager)
        {
            gridShader = shader;
            behaviorManager = boidManager;
            
            // Get kernel IDs
            updateGridKernel = gridShader.FindKernel("UpdateGrid");
            clearGridKernel = gridShader.FindKernel("ClearGrid");
            prefixSumKernel = gridShader.FindKernel("PrefixSum");
            sumBlocksKernel = gridShader.FindKernel("SumBlocks");
            addSumsKernel = gridShader.FindKernel("AddSums");
            rearrangeBoidsKernel = gridShader.FindKernel("RearrangeBoids");
        }
        
        public void Initialize(int numBoids, float visualRange, float xBound, float yBound, ComputeBuffer boidBuffer, ComputeBuffer boidBufferOut)
        {
            // Set up the spatial grid
            gridCellSize = visualRange;
            gridDimX = Mathf.FloorToInt(xBound * 2 / gridCellSize) + 30;
            gridDimY = Mathf.FloorToInt(yBound * 2 / gridCellSize) + 30;
            gridTotalCells = gridDimX * gridDimY;
            blocks = Mathf.CeilToInt(gridTotalCells / 256f);
            
            // Create CPU-side arrays if needed (for CPU mode)
            grid = new NativeArray<int2>(numBoids, Allocator.Persistent);
            gridOffsets = new NativeArray<int>(gridTotalCells, Allocator.Persistent);
            
            // Create GPU buffers
            gridBuffer = new ComputeBuffer(numBoids, 8);
            gridOffsetBuffer = new ComputeBuffer(gridTotalCells, 4);
            gridOffsetBufferIn = new ComputeBuffer(gridTotalCells, 4);
            gridSumsBuffer = new ComputeBuffer(blocks, 4);
            gridSumsBuffer2 = new ComputeBuffer(blocks, 4);
            
            // Set up shader parameters
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
        }
        
        public void UpdateGridGPU(int numBoids, ComputeBuffer boidBuffer, ComputeBuffer boidBufferOut)
        {   
            // Clear indices
            gridShader.Dispatch(clearGridKernel, blocks, 1, 1);

            // Populate grid
            gridShader.Dispatch(updateGridKernel, Mathf.CeilToInt(numBoids / 256f), 1, 1);

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
                gridShader.Dispatch(sumBlocksKernel, Mathf.CeilToInt(blocks / 256f), 1, 1);
                swap = !swap;
            }

            // Apply offsets of sums to each block
            gridShader.SetBuffer(addSumsKernel, "gridSumsBufferIn", swap ? gridSumsBuffer : gridSumsBuffer2);
            gridShader.Dispatch(addSumsKernel, blocks, 1, 1);

            // Rearrange boids
            gridShader.Dispatch(rearrangeBoidsKernel, Mathf.CeilToInt(numBoids / 256f), 1, 1);
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
            
            gridBuffer?.Release();
            gridOffsetBuffer?.Release();
            gridOffsetBufferIn?.Release();
            gridSumsBuffer?.Release();
            gridSumsBuffer2?.Release();
        }
        
        public int[] GetGridOffsets()
        {
            return gridOffsets.ToArray();
        }
    }
}
