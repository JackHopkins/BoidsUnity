using UnityEngine;
using Unity.Mathematics;
using System;
using Unity.Collections;

namespace BoidsUnity
{
    struct FrustumData {
        public float2 center;
        public float width;
        public float height;
        public float margin;
        public float padding; // Padding to ensure 24-byte total size
    };
    public class LODManager
    {
        // LOD settings
        private bool useLOD = true;
        public int metaGridDimX { get; private set; }
        public int metaGridDimY { get; private set; }
        public int metaGridTotalCells { get; private set; }
        public float metaGridCellSize { get; private set; }
        private int blocks;
        
        // Frustum data
        private FrustumData frustumData;
        
        // Compute shader and kernels
        private ComputeShader lodShader;
        private int updateFrustumKernel;
        private int createMetaBoidsKernel;
        private int expandMetaBoidsKernel;
        private int mergeBoidsToMetaKernel;
        private int splitMetaToBoidsKernel;
        
        // GPU buffers for LOD system
        public ComputeBuffer metaBoidBuffer;         // Stores meta-boids
        public ComputeBuffer metaBoidCountBuffer;    // Counts of meta-boids per cell
        public ComputeBuffer frustumDataBuffer;      // Camera frustum data
        public ComputeBuffer frustumGridBuffer;      // Grid cells inside/outside frustum
        
        public LODManager(ComputeShader shader)
        {
            if (shader == null)
            {
                Debug.LogError("LOD Shader is null! Make sure to assign it in the inspector.");
                return;
            }
            
            lodShader = shader;
            
            // Get kernel IDs - add logging to debug
            try {
                // Find all kernels by name
                updateFrustumKernel = lodShader.FindKernel("UpdateFrustum");
                createMetaBoidsKernel = lodShader.FindKernel("CreateMetaBoids");
                expandMetaBoidsKernel = lodShader.FindKernel("ExpandMetaBoids");
                mergeBoidsToMetaKernel = lodShader.FindKernel("MergeBoidsToMeta");
                splitMetaToBoidsKernel = lodShader.FindKernel("SplitMetaToBoids");
                
                // Log kernel indices
                Debug.Log($"LOD Kernel Indices: UpdateFrustum({updateFrustumKernel}), CreateMetaBoids({createMetaBoidsKernel}), " + 
                          $"ExpandMetaBoids({expandMetaBoidsKernel}), MergeBoidsToMeta({mergeBoidsToMetaKernel}), " + 
                          $"SplitMetaToBoids({splitMetaToBoidsKernel})");
                
                // Verify shader properties (resource bindings)
                VerifyShaderProperties();
            }
            catch (Exception e) {
                Debug.LogError($"Error finding kernels: {e.Message}");
            }
        }
        
        private void VerifyShaderProperties()
        {
            // This helps identify missing buffer bindings
            try {
                // Get thread count for first kernel to verify it exists
                uint x, y, z;
                lodShader.GetKernelThreadGroupSizes(updateFrustumKernel, out x, out y, out z);
                Debug.Log($"UpdateFrustum kernel thread group size: ({x}, {y}, {z})");
                
                // List all kernels and their properties
                string[] properties = new string[] {
                    "boids", "boidsOut", "metaBoids", "metaBoidCounts", 
                    "frustumData", "frustumGrid"
                };
                
                foreach (string prop in properties)
                {
                    Debug.Log($"Checking if shader uses property: {prop} - Result: {HasProperty(prop)}");
                }
            }
            catch (Exception e) {
                Debug.LogError($"Error verifying shader properties: {e.Message}");
            }
        }
        
        private bool HasProperty(string name)
        {
            // This is an approximate way to check if a property exists in the compute shader
            try {
                lodShader.SetFloat(name + "_test_property", 0);
                return false; // If it doesn't throw, it's not a buffer property
            }
            catch {
                return true; // Most likely exists as a buffer
            }
        }

        public void Initialize(int numBoids, float baseGridCellSize, float xBound, float yBound, Camera camera, ComputeBuffer boidBuffer, ComputeBuffer boidBufferOut)
        {
            // Set up the meta spatial grid (coarser than the regular grid)
            metaGridCellSize = baseGridCellSize * 4f; // Coarser grid for meta-boids
            metaGridDimX = Mathf.FloorToInt(xBound * 2 / metaGridCellSize) + 10;
            metaGridDimY = Mathf.FloorToInt(yBound * 2 / metaGridCellSize) + 10;
            metaGridTotalCells = metaGridDimX * metaGridDimY;
            blocks = Mathf.CeilToInt(metaGridTotalCells / 256f);
            
            // Initialize frustum data
            float frustumMargin = 3f * baseGridCellSize; // Margin for transitions
            frustumData = new FrustumData
            {
                center = new float2(camera.transform.position.x, camera.transform.position.y),
                width = camera.orthographicSize * 2f * camera.aspect,
                height = camera.orthographicSize * 2f,
                margin = frustumMargin
            };
            
            // Create GPU buffers
            int maxMetaBoids = Mathf.Max(numBoids / 20, 1000); // Estimate max number of meta-boids
            metaBoidBuffer = new ComputeBuffer(maxMetaBoids, 32); // sizeof(MetaBoid) - 8 bytes per float2 (x2) + 4 bytes per uint (x2) + 4 bytes per float (x2)
            metaBoidCountBuffer = new ComputeBuffer(metaGridTotalCells, 4);
            frustumDataBuffer = new ComputeBuffer(1, 24); // sizeof(FrustumData) - 8 bytes for float2 + 4 bytes per float (x4)
            frustumGridBuffer = new ComputeBuffer(metaGridTotalCells, 4);
            
            // Upload initial frustum data
            frustumDataBuffer.SetData(new FrustumData[] { frustumData });
            
            // Set up shader parameters
            SetShaderParameters(numBoids, boidBuffer, boidBufferOut);
        }
        
        private void SetShaderParameters(int numBoids, ComputeBuffer boidBuffer, ComputeBuffer boidBufferOut)
        {
            // Set common parameters
            lodShader.SetInt("numBoids", numBoids);
            lodShader.SetInt("metaGridDimX", metaGridDimX);
            lodShader.SetInt("metaGridDimY", metaGridDimY);
            lodShader.SetInt("metaGridTotalCells", metaGridTotalCells);
            lodShader.SetFloat("metaGridCellSize", metaGridCellSize);
            
            try {
                // First set all buffers explicitly for each kernel
                Debug.Log("Setting UpdateFrustum kernel buffers...");
                lodShader.SetBuffer(updateFrustumKernel, "frustumData", frustumDataBuffer);
                lodShader.SetBuffer(updateFrustumKernel, "frustumGrid", frustumGridBuffer);
                
                Debug.Log("Setting CreateMetaBoids kernel buffers...");
                lodShader.SetBuffer(createMetaBoidsKernel, "boids", boidBuffer);
                lodShader.SetBuffer(createMetaBoidsKernel, "frustumData", frustumDataBuffer);
                lodShader.SetBuffer(createMetaBoidsKernel, "frustumGrid", frustumGridBuffer);
                lodShader.SetBuffer(createMetaBoidsKernel, "metaBoids", metaBoidBuffer);
                lodShader.SetBuffer(createMetaBoidsKernel, "metaBoidCounts", metaBoidCountBuffer);
                
                Debug.Log("Setting ExpandMetaBoids kernel buffers...");
                lodShader.SetBuffer(expandMetaBoidsKernel, "metaBoids", metaBoidBuffer);
                lodShader.SetBuffer(expandMetaBoidsKernel, "metaBoidCounts", metaBoidCountBuffer);
                lodShader.SetBuffer(expandMetaBoidsKernel, "frustumGrid", frustumGridBuffer);
                
                Debug.Log("Setting MergeBoidsToMeta kernel buffers...");
                lodShader.SetBuffer(mergeBoidsToMetaKernel, "boids", boidBuffer);
                lodShader.SetBuffer(mergeBoidsToMetaKernel, "boidsOut", boidBufferOut);
                lodShader.SetBuffer(mergeBoidsToMetaKernel, "frustumGrid", frustumGridBuffer);
                lodShader.SetBuffer(mergeBoidsToMetaKernel, "metaBoids", metaBoidBuffer);
                lodShader.SetBuffer(mergeBoidsToMetaKernel, "metaBoidCounts", metaBoidCountBuffer);
                
                Debug.Log("Setting SplitMetaToBoids kernel buffers...");
                lodShader.SetBuffer(splitMetaToBoidsKernel, "boids", boidBuffer);
                lodShader.SetBuffer(splitMetaToBoidsKernel, "boidsOut", boidBufferOut);
                lodShader.SetBuffer(splitMetaToBoidsKernel, "frustumGrid", frustumGridBuffer);
                lodShader.SetBuffer(splitMetaToBoidsKernel, "metaBoids", metaBoidBuffer);
                lodShader.SetBuffer(splitMetaToBoidsKernel, "metaBoidCounts", metaBoidCountBuffer);
                lodShader.SetBuffer(splitMetaToBoidsKernel, "frustumData", frustumDataBuffer);
                
                Debug.Log("All kernel buffers set successfully!");
            }
            catch (Exception e) {
                Debug.LogError($"Error setting kernel buffers: {e.Message}");
            }
        }
        
        public void UpdateLOD(int numBoids, ComputeBuffer boidBuffer, ComputeBuffer boidBufferOut, Camera camera, float deltaTime)
        {
            // Skip if LOD is not enabled
            if (!useLOD) return;
            
            try {
                // Start timing
                float startTime = Time.realtimeSinceStartup;
                
                // Update frustum data from camera
                UpdateFrustumData(camera);
                
                // Ensure all required buffers are set for the kernels we'll use
                SafeSetKernelBuffers(numBoids, boidBuffer, boidBufferOut);
                
                // Update time parameter
                lodShader.SetFloat("deltaTime", deltaTime);
                
                // 1. Update the frustum grid
                ProfilingUtility.BeginSample("UpdateFrustumGrid");
                lodShader.Dispatch(updateFrustumKernel, Mathf.CeilToInt(metaGridTotalCells / 256f), 1, 1);
                ProfilingUtility.EndSample("UpdateFrustumGrid");
                
                // 2. Use working kernels for LOD
                ProfilingUtility.BeginSample("MergeBoidsToMeta");
                lodShader.Dispatch(mergeBoidsToMetaKernel, Mathf.CeilToInt(numBoids / 256f), 1, 1);
                ProfilingUtility.EndSample("MergeBoidsToMeta");
                
                // 3. Apply transitions at edges of frustum
                ProfilingUtility.BeginSample("SplitMetaToBoids");
                lodShader.Dispatch(splitMetaToBoidsKernel, Mathf.CeilToInt(metaGridTotalCells / 256f), 1, 1);
                ProfilingUtility.EndSample("SplitMetaToBoids");
                
                // Calculate and log performance metrics
                float elapsedMs = (Time.realtimeSinceStartup - startTime) * 1000f;
                
                // Record timing data
                ProfilingUtility.RecordManualTiming("LOD_TotalTime", elapsedMs);
                
                // Log every 120 frames for high boid counts
                if (numBoids > 10000 && Time.frameCount % 120 == 0)
                {
                    Debug.Log($"LOD processing for {numBoids} boids took {elapsedMs:F2}ms");
                }
            }
            catch (Exception e) {
                Debug.LogError($"Error in UpdateLOD: {e.Message}");
                
                // If there's an error, ensure boids keep moving
                SafeCopyBoidBuffer(boidBuffer, boidBufferOut, numBoids);
            }
        }
        
        private void SafeSetKernelBuffers(int numBoids, ComputeBuffer boidBuffer, ComputeBuffer boidBufferOut)
        {
            // Focus only on the working kernels
            
            // UpdateFrustum kernel buffers
            lodShader.SetBuffer(updateFrustumKernel, "frustumData", frustumDataBuffer);
            lodShader.SetBuffer(updateFrustumKernel, "frustumGrid", frustumGridBuffer);
            lodShader.SetBuffer(updateFrustumKernel, "boids", boidBuffer);           // Required for kernel validation
            lodShader.SetBuffer(updateFrustumKernel, "boidsOut", boidBufferOut);     // Required for kernel validation
            lodShader.SetBuffer(updateFrustumKernel, "metaBoids", metaBoidBuffer);   // Required for kernel validation
            lodShader.SetBuffer(updateFrustumKernel, "metaBoidCounts", metaBoidCountBuffer); // Required for kernel validation
            
            // MergeBoidsToMeta kernel buffers
            lodShader.SetBuffer(mergeBoidsToMetaKernel, "boids", boidBuffer);
            lodShader.SetBuffer(mergeBoidsToMetaKernel, "boidsOut", boidBufferOut);
            lodShader.SetBuffer(mergeBoidsToMetaKernel, "frustumGrid", frustumGridBuffer);
            lodShader.SetBuffer(mergeBoidsToMetaKernel, "metaBoids", metaBoidBuffer);
            lodShader.SetBuffer(mergeBoidsToMetaKernel, "metaBoidCounts", metaBoidCountBuffer);
            lodShader.SetBuffer(mergeBoidsToMetaKernel, "frustumData", frustumDataBuffer); // May be needed
            
            // SplitMetaToBoids kernel buffers
            lodShader.SetBuffer(splitMetaToBoidsKernel, "boids", boidBuffer);
            lodShader.SetBuffer(splitMetaToBoidsKernel, "boidsOut", boidBufferOut);
            lodShader.SetBuffer(splitMetaToBoidsKernel, "frustumGrid", frustumGridBuffer);
            lodShader.SetBuffer(splitMetaToBoidsKernel, "metaBoids", metaBoidBuffer);
            lodShader.SetBuffer(splitMetaToBoidsKernel, "metaBoidCounts", metaBoidCountBuffer);
            lodShader.SetBuffer(splitMetaToBoidsKernel, "frustumData", frustumDataBuffer);
        }
        
        private void SafeCopyBoidBuffer(ComputeBuffer source, ComputeBuffer dest, int numBoids)
        {
            try {
                // Direct buffer copy fallback in case the compute shader fails
                Boid[] tempBoids = new Boid[numBoids];
                source.GetData(tempBoids);
                dest.SetData(tempBoids);
            }
            catch (Exception e) {
                Debug.LogError($"Error in fallback boid buffer copy: {e.Message}");
            }
        }
        
        private void ProcessBoidTransitions(int numBoids, ComputeBuffer boidBuffer, ComputeBuffer boidBufferOut, float deltaTime)
        {
            // This method is no longer used in the simplified implementation
            // It was replaced with direct buffer operations in the UpdateLOD method
        }
        
        private void UpdateFrustumData(Camera camera)
        {
            // Create a new frustum data struct to ensure correct memory layout
            var updatedFrustum = new FrustumData {
                center = new float2(camera.transform.position.x, camera.transform.position.y),
                width = camera.orthographicSize * 2f * camera.aspect,
                height = camera.orthographicSize * 2f,
                margin = frustumData.margin,
                padding = 0 // Ensure padding is set
            };
            
            // Store for future use
            frustumData = updatedFrustum;
            
            // Upload updated frustum data
            frustumDataBuffer.SetData(new FrustumData[] { updatedFrustum });
        }
        
        public void SetUseLOD(bool useLOD)
        {
            this.useLOD = useLOD;
        }
        
        public void Cleanup()
        {
            metaBoidBuffer?.Release();
            metaBoidCountBuffer?.Release();
            frustumDataBuffer?.Release();
            frustumGridBuffer?.Release();
        }
    }
}