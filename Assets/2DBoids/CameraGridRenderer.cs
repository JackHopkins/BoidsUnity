using UnityEngine;

namespace BoidsUnity
{
    [RequireComponent(typeof(Camera))]
    public class CameraGridRenderer : MonoBehaviour
    {
        public bool showRegularGrid = true;
        public bool showOfficerGrid = true;
        public Color regularGridColor = new Color(0.3f, 0.3f, 0.3f, 0.7f);
        public Color officerGridColor = new Color(0.7f, 0.2f, 0.2f, 0.7f);
        
        private Material lineMaterial;
        private GridManager gridManager;
        private bool initialized = false;
        
        // Grid data
        private float regularCellSize;
        private float officerCellSize;
        private int regularGridDimX, regularGridDimY;
        private int officerGridDimX, officerGridDimY;
        
        void Awake()
        {
            // Create a material that will render the grid lines
            Shader shader = Shader.Find("Hidden/Internal-Colored");
            lineMaterial = new Material(shader);
            lineMaterial.hideFlags = HideFlags.HideAndDontSave;
        }
        
        public void Initialize(GridManager manager)
        {
            gridManager = manager;
            
            if (gridManager == null)
            {
                Debug.LogError("CameraGridRenderer: GridManager reference is null");
                return;
            }
            
            // Cache grid dimensions
            regularCellSize = gridManager.gridCellSize;
            officerCellSize = gridManager.officerGridCellSize;
            regularGridDimX = gridManager.gridDimX;
            regularGridDimY = gridManager.gridDimY;
            officerGridDimX = gridManager.officerGridDimX;
            officerGridDimY = gridManager.officerGridDimY;
            
            initialized = true;
            Debug.Log($"CameraGridRenderer initialized with regular grid {regularGridDimX}x{regularGridDimY}, cell size {regularCellSize}");
            Debug.Log($"Officer grid {officerGridDimX}x{officerGridDimY}, cell size {officerCellSize}");
        }
        
        void OnPostRender()
        {
            if (!initialized || !enabled)
                return;
            
            // Print a debug message to confirm this method is being called
            Debug.Log($"OnPostRender called for camera {GetComponent<Camera>().name}");
            
            if (showRegularGrid)
                DrawGrid(regularGridDimX, regularGridDimY, regularCellSize, regularGridColor);
                
            if (showOfficerGrid)
                DrawGrid(officerGridDimX, officerGridDimY, officerCellSize, officerGridColor);
        }
        
        // Also implement OnRenderObject as a fallback
        void OnRenderObject()
        {
            if (!initialized || !enabled)
                return;
            
            // Only draw for the main camera
            if (Camera.current != Camera.main)
                return;
                
            Debug.Log($"OnRenderObject called for camera {Camera.current.name}");
            
            if (showRegularGrid)
                DrawGridAlternative(regularGridDimX, regularGridDimY, regularCellSize, regularGridColor);
                
            if (showOfficerGrid)
                DrawGridAlternative(officerGridDimX, officerGridDimY, officerCellSize, officerGridColor);
        }
        
        // Alternative drawing method that might work if OnPostRender doesn't
        void DrawGridAlternative(int dimX, int dimY, float cellSize, Color color)
        {
            // Calculate world dimensions
            float worldWidth = dimX * cellSize;
            float worldHeight = dimY * cellSize;
            float startX = -worldWidth/2;
            float startY = -worldHeight/2;
            
            // Set up material
            lineMaterial.SetPass(0);
            
            GL.PushMatrix();
            GL.MultMatrix(transform.localToWorldMatrix);
            GL.Begin(GL.LINES);
            GL.Color(color);
            
            // Draw horizontal lines
            for (int y = 0; y <= dimY; y++)
            {
                float posY = startY + y * cellSize;
                GL.Vertex3(startX, posY, 0);
                GL.Vertex3(startX + worldWidth, posY, 0);
            }
            
            // Draw vertical lines
            for (int x = 0; x <= dimX; x++)
            {
                float posX = startX + x * cellSize;
                GL.Vertex3(posX, startY, 0);
                GL.Vertex3(posX, startY + worldHeight, 0);
            }
            
            GL.End();
            GL.PopMatrix();
        }
        
        void DrawGrid(int dimX, int dimY, float cellSize, Color color)
        {
            GL.PushMatrix();
            
            lineMaterial.SetPass(0);
            GL.LoadPixelMatrix();
            GL.Begin(GL.LINES);
            
            // Convert from world space to screen space
            float gridWorldWidth = dimX * cellSize;
            float gridWorldHeight = dimY * cellSize;
            
            // Center in screen coordinates
            Camera cam = GetComponent<Camera>();
            Vector3 worldTopLeft = new Vector3(-gridWorldWidth/2, gridWorldHeight/2, 0);
            Vector3 worldBottomRight = new Vector3(gridWorldWidth/2, -gridWorldHeight/2, 0);
            
            Vector3 screenTopLeft = cam.WorldToScreenPoint(worldTopLeft);
            Vector3 screenBottomRight = cam.WorldToScreenPoint(worldBottomRight);
            
            float screenWidth = screenBottomRight.x - screenTopLeft.x;
            float screenHeight = screenTopLeft.y - screenBottomRight.y;
            
            float cellScreenWidth = screenWidth / dimX;
            float cellScreenHeight = screenHeight / dimY;
            
            GL.Color(color);
            
            // Draw horizontal lines
            for (int y = 0; y <= dimY; y++)
            {
                float yPos = screenTopLeft.y - y * cellScreenHeight;
                GL.Vertex3(screenTopLeft.x, yPos, 0);
                GL.Vertex3(screenBottomRight.x, yPos, 0);
            }
            
            // Draw vertical lines
            for (int x = 0; x <= dimX; x++)
            {
                float xPos = screenTopLeft.x + x * cellScreenWidth;
                GL.Vertex3(xPos, screenTopLeft.y, 0);
                GL.Vertex3(xPos, screenBottomRight.y, 0);
            }
            
            GL.End();
            GL.PopMatrix();
        }
        
        public void ToggleRegularGrid()
        {
            showRegularGrid = !showRegularGrid;
        }
        
        public void ToggleOfficerGrid()
        {
            showOfficerGrid = !showOfficerGrid;
        }
        
        void OnDestroy()
        {
            if (lineMaterial != null)
                DestroyImmediate(lineMaterial);
        }
    }
}