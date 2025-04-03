using UnityEngine;
using Unity.Mathematics;

namespace BoidsUnity
{
    public class GridRenderer : MonoBehaviour
    {
        [Header("Render Settings")]
        [SerializeField] public Material gridMaterial;
        [SerializeField] private Color gridColor = new Color(0.2f, 0.2f, 0.2f, 0.3f);
        [SerializeField] [Range(0.001f, 0.01f)] private float lineWidth = 0.005f;
        [SerializeField] private bool showGrid = true;
        
        // Grid properties
        private int gridDimX;
        private int gridDimY;
        private float gridCellSize;
        private float2 gridOrigin;
        
        // Mesh components 
        private Mesh gridMesh;
        private Material gridMaterialInstance;
        
        // Buffer for grid visualization
        private ComputeBuffer gridLinesBuffer;
        private int maxLineVertices = 10000; // Adjust based on max grid size
        
        public void Initialize(int dimX, int dimY, float cellSize, float xBound, float yBound)
        {
            // Store grid dimensions
            gridDimX = dimX;
            gridDimY = dimY;
            gridCellSize = cellSize;
            
            // Calculate grid origin (bottom-left corner)
            gridOrigin = new float2(-gridDimX * gridCellSize / 2, -gridDimY * gridCellSize / 2);
            
            // Create material instance
            if (gridMaterial != null)
            {
                gridMaterialInstance = new Material(gridMaterial);
                gridMaterialInstance.SetColor("_GridColor", gridColor);
                gridMaterialInstance.SetFloat("_LineWidth", lineWidth);
            }
            else
            {
                Debug.LogWarning("Grid material is null. Grid visualization will be disabled.");
                showGrid = false;
            }
            
            // Create the grid mesh
            CreateGridMesh();
        }
        
        private void CreateGridMesh()
        {
            try
            {
                // Safety checks
                if (gridDimX <= 0 || gridDimY <= 0)
                {
                    Debug.LogError($"Invalid grid dimensions: {gridDimX}x{gridDimY}");
                    return;
                }
                
                // Calculate number of vertices needed
                int horzLineCount = Mathf.Min(gridDimY + 1, 500); // Cap to prevent huge meshes
                int vertLineCount = Mathf.Min(gridDimX + 1, 500); // Cap to prevent huge meshes
                int totalLines = horzLineCount + vertLineCount;
                int totalVertices = totalLines * 2; // 2 vertices per line
                
                // Warn about large grid mesh
                if (totalVertices > 5000)
                {
                    Debug.LogWarning($"Creating large grid mesh with {totalVertices} vertices. This might impact performance.");
                }
                
                // Create vertex arrays
                Vector3[] vertices = new Vector3[totalVertices];
                int[] indices = new int[totalVertices];
                
                // Get the grid spacing, potentially skipping cells for large grids
                float horzSkip = (float)gridDimY / (horzLineCount - 1);
                float vertSkip = (float)gridDimX / (vertLineCount - 1);
                
                // Set vertices for horizontal lines
                for (int i = 0; i < horzLineCount; i++)
                {
                    float yIdx = i * horzSkip;
                    float y = gridOrigin.y + yIdx * gridCellSize;
                    
                    // Start point
                    vertices[i * 2] = new Vector3(gridOrigin.x, y, 0);
                    // End point
                    vertices[i * 2 + 1] = new Vector3(gridOrigin.x + gridDimX * gridCellSize, y, 0);
                    
                    // Indices (just use the vertex index since we're drawing lines)
                    indices[i * 2] = i * 2;
                    indices[i * 2 + 1] = i * 2 + 1;
                }
                
                // Set vertices for vertical lines
                int offset = horzLineCount * 2;
                for (int i = 0; i < vertLineCount; i++)
                {
                    float xIdx = i * vertSkip;
                    float x = gridOrigin.x + xIdx * gridCellSize;
                    
                    // Start point
                    vertices[offset + i * 2] = new Vector3(x, gridOrigin.y, 0);
                    // End point
                    vertices[offset + i * 2 + 1] = new Vector3(x, gridOrigin.y + gridDimY * gridCellSize, 0);
                    
                    // Indices
                    indices[offset + i * 2] = offset + i * 2;
                    indices[offset + i * 2 + 1] = offset + i * 2 + 1;
                }
                
                // Create mesh
                if (gridMesh == null)
                {
                    gridMesh = new Mesh();
                }
                else
                {
                    gridMesh.Clear();
                }
                
                gridMesh.name = "GridMesh";
                gridMesh.vertices = vertices;
                gridMesh.SetIndices(indices, MeshTopology.Lines, 0);
            }
            catch (System.Exception e)
            {
                Debug.LogError($"Error creating grid mesh: {e.Message}");
                showGrid = false; // Disable grid if mesh creation fails
            }
        }
        
        public void UpdateGrid(int dimX, int dimY, float cellSize, float xBound, float yBound)
        {
            // Only rebuild if dimensions change
            if (dimX != gridDimX || dimY != gridDimY || cellSize != gridCellSize)
            {
                gridDimX = dimX;
                gridDimY = dimY;
                gridCellSize = cellSize;
                
                // Recalculate origin
                gridOrigin = new float2(-gridDimX * gridCellSize / 2, -gridDimY * gridCellSize / 2);
                
                // Recreate mesh with new dimensions
                CreateGridMesh();
            }
        }
        
        public void SetGridColor(Color color)
        {
            gridColor = color;
            if (gridMaterialInstance != null)
            {
                gridMaterialInstance.SetColor("_GridColor", gridColor);
            }
        }
        
        public void SetLineWidth(float width)
        {
            lineWidth = width;
            if (gridMaterialInstance != null)
            {
                gridMaterialInstance.SetFloat("_LineWidth", lineWidth);
            }
        }
        
        public void SetVisibility(bool visible)
        {
            showGrid = visible;
        }
        
        public void Render()
        {
            // Check for valid components
            if (!showGrid) return;
            if (gridMesh == null)
            {
                Debug.LogWarning("Grid mesh is null. Cannot render grid.");
                return;
            }
            if (gridMaterialInstance == null)
            {
                Debug.LogWarning("Grid material instance is null. Cannot render grid.");
                return;
            }
            
            try
            {
                // Draw the mesh with our grid material
                Graphics.DrawMesh(gridMesh, Matrix4x4.identity, gridMaterialInstance, 0);
            }
            catch (System.Exception e)
            {
                Debug.LogError($"Error drawing grid mesh: {e.Message}");
                showGrid = false; // Disable grid if rendering fails
            }
        }
        
        public void Cleanup()
        {
            if (gridMesh != null)
            {
                if (Application.isPlaying)
                {
                    Destroy(gridMesh);
                }
                else
                {
                    DestroyImmediate(gridMesh);
                }
            }
            
            if (gridMaterialInstance != null)
            {
                if (Application.isPlaying)
                {
                    Destroy(gridMaterialInstance);
                }
                else
                {
                    DestroyImmediate(gridMaterialInstance);
                }
            }
            
            gridLinesBuffer?.Release();
        }
    }
}