using UnityEngine;
using System.Collections.Generic;

namespace BoidsUnity
{
    public class SimpleGridRenderer : MonoBehaviour
    {
        public bool showRegularGrid = true;
        public bool showOfficerGrid = true;
        public Color regularGridColor = new Color(0.3f, 0.3f, 0.3f, 0.7f);
        public Color officerGridColor = new Color(0.7f, 0.2f, 0.2f, 0.7f);
        
        // Line renderers
        private GameObject regularGridHolder;
        private GameObject officerGridHolder;
        private List<LineRenderer> regularGridLines = new List<LineRenderer>();
        private List<LineRenderer> officerGridLines = new List<LineRenderer>();
        
        // Grid info
        private GridManager gridManager;
        private float regularCellSize;
        private float officerCellSize;
        private int regularGridDimX, regularGridDimY;
        private int officerGridDimX, officerGridDimY;
        
        private bool initialized = false;
        
        public void Initialize(GridManager manager)
        {
            gridManager = manager;
            
            if (gridManager == null)
            {
                Debug.LogError("SimpleGridRenderer: GridManager reference is null");
                return;
            }
            
            // Cache grid dimensions
            regularCellSize = gridManager.gridCellSize;
            officerCellSize = gridManager.officerGridCellSize;
            regularGridDimX = gridManager.gridDimX;
            regularGridDimY = gridManager.gridDimY;
            officerGridDimX = gridManager.officerGridDimX;
            officerGridDimY = gridManager.officerGridDimY;
            
            // Create line renderers
            CreateGridLines();
            
            initialized = true;
            Debug.Log($"SimpleGridRenderer initialized");
        }
        
        void CreateGridLines()
        {
            // Clean up existing holders
            if (regularGridHolder != null)
                Destroy(regularGridHolder);
                
            if (officerGridHolder != null)
                Destroy(officerGridHolder);
                
            regularGridLines.Clear();
            officerGridLines.Clear();
            
            // Create new holders
            regularGridHolder = new GameObject("RegularGridLines");
            regularGridHolder.transform.parent = transform;
            regularGridHolder.transform.localPosition = Vector3.zero;
            
            officerGridHolder = new GameObject("OfficerGridLines");
            officerGridHolder.transform.parent = transform;
            officerGridHolder.transform.localPosition = Vector3.zero;
            
            // Create regular grid lines
            CreateGridForHolder(
                regularGridHolder, 
                regularGridDimX, 
                regularGridDimY, 
                regularCellSize, 
                regularGridColor, 
                regularGridLines
            );
            
            // Create officer grid lines
            CreateGridForHolder(
                officerGridHolder, 
                officerGridDimX, 
                officerGridDimY, 
                officerCellSize, 
                officerGridColor, 
                officerGridLines
            );
            
            // Set initial visibility
            regularGridHolder.SetActive(showRegularGrid);
            officerGridHolder.SetActive(showOfficerGrid);
        }
        
        void CreateGridForHolder(GameObject holder, int dimX, int dimY, float cellSize, Color color, List<LineRenderer> lines)
        {
            // Calculate total width and height
            float gridWidth = dimX * cellSize;
            float gridHeight = dimY * cellSize;
            
            // Calculate start position (bottom-left corner of grid)
            float startX = -gridWidth / 2;
            float startY = -gridHeight / 2;
            
            // Create horizontal lines
            for (int y = 0; y <= dimY; y++)
            {
                GameObject lineObj = new GameObject($"HLine_{y}");
                lineObj.transform.parent = holder.transform;
                
                LineRenderer line = lineObj.AddComponent<LineRenderer>();
                SetupLineRenderer(line, color);
                
                line.positionCount = 2;
                line.SetPosition(0, new Vector3(startX, startY + y * cellSize, -0.1f));
                line.SetPosition(1, new Vector3(startX + gridWidth, startY + y * cellSize, -0.1f));
                
                lines.Add(line);
            }
            
            // Create vertical lines
            for (int x = 0; x <= dimX; x++)
            {
                GameObject lineObj = new GameObject($"VLine_{x}");
                lineObj.transform.parent = holder.transform;
                
                LineRenderer line = lineObj.AddComponent<LineRenderer>();
                SetupLineRenderer(line, color);
                
                line.positionCount = 2;
                line.SetPosition(0, new Vector3(startX + x * cellSize, startY, -0.1f));
                line.SetPosition(1, new Vector3(startX + x * cellSize, startY + gridHeight, -0.1f));
                
                lines.Add(line);
            }
        }
        
        void SetupLineRenderer(LineRenderer line, Color color)
        {
            line.startWidth = 0.05f;
            line.endWidth = 0.05f;
            line.startColor = color;
            line.endColor = color;
            line.material = new Material(Shader.Find("Sprites/Default"));
            line.useWorldSpace = true;
            line.receiveShadows = false;
            line.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
        }
        
        public void ToggleRegularGrid()
        {
            showRegularGrid = !showRegularGrid;
            if (regularGridHolder != null)
                regularGridHolder.SetActive(showRegularGrid);
        }
        
        public void ToggleOfficerGrid()
        {
            showOfficerGrid = !showOfficerGrid;
            if (officerGridHolder != null)
                officerGridHolder.SetActive(showOfficerGrid);
        }
        
        public void SetVisualizationActive(bool active)
        {
            enabled = active;
            if (regularGridHolder != null)
                regularGridHolder.SetActive(active && showRegularGrid);
                
            if (officerGridHolder != null)
                officerGridHolder.SetActive(active && showOfficerGrid);
        }
        
        private void OnDestroy()
        {
            if (regularGridHolder != null)
                Destroy(regularGridHolder);
                
            if (officerGridHolder != null)
                Destroy(officerGridHolder);
        }
    }
}