using UnityEngine;
using System.Collections.Generic;
using Unity.Mathematics;

// Represents a single battalion of boids
[System.Serializable]
public class Battalion
{
    public int id;                      // Unique identifier
    public int startIndex;              // Start index in boid array
    public int count;                   // Number of boids in this battalion
    public Color color;                 // Color for visual distinction
    public float2 targetPosition;       // Target position for this battalion
    public float2 formationSize;        // Size of the formation (width, height)
    public bool isSelected;             // Whether this battalion is currently selected
    public FormationType formationType; // Current formation type

    // Returns whether a world position is within the battalion's bounds (for selection)
    public bool ContainsPosition(float2 position, float2[] boidPositions)
    {
        // Calculate battalion center based on average of boid positions
        float2 center = float2.zero;
        for (int i = 0; i < count; i++)
        {
            center += boidPositions[startIndex + i];
        }
        center /= count;
    
        // Simple radius-based selection
        float selectionRadius = Mathf.Sqrt(count) * 0.5f;
        return math.distance(position, center) <= selectionRadius;
    }
}

// Possible formation types
public enum FormationType
{
    Square,
    Line,
    Column,
    Wedge,
    Scattered
}

// Battalion manager class
public class BattalionManager : MonoBehaviour
{
    [Header("Battalion Settings")]
    public int battalionSize = 1000;
    public int numBattalions = 5;
    public float formationSpread = 0.1f;
    public Color[] battalionColors;

    // List of all battalions
    public List<Battalion> battalions = new List<Battalion>();
    
    // Currently selected battalions
    public List<Battalion> selectedBattalions = new List<Battalion>();

    // Initialize battalions based on total boid count
    public void InitializeBattalions(int totalBoids)
    {
        battalions.Clear();
        selectedBattalions.Clear();

        // Calculate actual battalion size (may be smaller for the last battalion)
        int actualBattalionCount = Mathf.CeilToInt((float)totalBoids / battalionSize);
        numBattalions = Mathf.Min(actualBattalionCount, numBattalions);

        // Create default colors if not provided
        if (battalionColors == null || battalionColors.Length == 0)
        {
            battalionColors = new Color[] {
                Color.red, Color.blue, Color.green, Color.yellow, Color.cyan,
                Color.magenta, new Color(1, 0.5f, 0), new Color(0.5f, 0, 1)
            };
        }

        // Create battalions
        for (int i = 0; i < numBattalions; i++)
        {
            int startIdx = i * battalionSize;
            int count = Mathf.Min(battalionSize, totalBoids - startIdx);
            
            if (count <= 0)
                break;

            // Create battalion
            Battalion battalion = new Battalion
            {
                id = i,
                startIndex = startIdx,
                count = count,
                color = battalionColors[i % battalionColors.Length],
                targetPosition = new float2(
                    UnityEngine.Random.Range(-5f, 5f),
                    UnityEngine.Random.Range(-5f, 5f)
                ),
                formationSize = new float2(2f, 2f),
                isSelected = false,
                formationType = FormationType.Square
            };

            battalions.Add(battalion);
        }
    }

    // Select a single battalion at world position
    public void SelectBattalionAt(Vector2 worldPos, float2[] boidPositions, bool addToSelection = false)
    {
        if (!addToSelection)
        {
            // Deselect all battalions
            foreach (Battalion b in selectedBattalions)
            {
                b.isSelected = false;
            }
            selectedBattalions.Clear();
        }

        // Find battalion under cursor
        foreach (Battalion battalion in battalions)
        {
            if (battalion.ContainsPosition(worldPos, boidPositions))
            {
                battalion.isSelected = true;
                if (!selectedBattalions.Contains(battalion))
                {
                    selectedBattalions.Add(battalion);
                }
                break;
            }
        }
    }

    // Get the target position for a specific boid in its battalion
    public float2 GetBoidFormationPosition(int boidIndex, float2 battalionTarget, FormationType formation, float2 formationSize)
    {
        // Find which battalion this boid belongs to
        Battalion battalion = null;
        int localIndex = 0;
        
        foreach (Battalion b in battalions)
        {
            if (boidIndex >= b.startIndex && boidIndex < b.startIndex + b.count)
            {
                battalion = b;
                localIndex = boidIndex - b.startIndex;
                break;
            }
        }

        if (battalion == null)
            return battalionTarget;

        float2 offset = float2.zero;
        float boidSpread = formationSpread;
        int rows, cols;

        // Calculate formation positions based on type
        switch (formation)
        {
            case FormationType.Square:
                // Form a square/rectangular grid
                cols = Mathf.CeilToInt(Mathf.Sqrt(battalion.count));
                rows = Mathf.CeilToInt((float)battalion.count / cols);
                int row = localIndex / cols;
                int col = localIndex % cols;
                
                offset = new float2(
                    (col - cols/2) * boidSpread * formationSize.x,
                    (row - rows/2) * boidSpread * formationSize.y
                );
                break;

            case FormationType.Line:
                // Form a horizontal line
                offset = new float2(
                    (localIndex - battalion.count/2) * boidSpread * formationSize.x,
                    0
                );
                break;

            case FormationType.Column:
                // Form a vertical column
                offset = new float2(
                    0,
                    (localIndex - battalion.count/2) * boidSpread * formationSize.y
                );
                break;

            case FormationType.Wedge:
                // Form a wedge/triangle
                int wedgeWidth = Mathf.CeilToInt(Mathf.Sqrt(battalion.count * 2));
                int currentRow = 0;
                int currentRowStart = 0;
                int rowWidth = 1;

                // Find which row this boid belongs to in the wedge
                while (currentRowStart + rowWidth <= localIndex)
                {
                    currentRowStart += rowWidth;
                    currentRow++;
                    rowWidth++;
                }

                int posInRow = localIndex - currentRowStart;
                offset = new float2(
                    (posInRow - rowWidth/2.0f) * boidSpread * formationSize.x,
                    -currentRow * boidSpread * formationSize.y
                );
                break;

            case FormationType.Scattered:
                // Random scatter within battalion bounds
                System.Random random = new System.Random(boidIndex + battalion.id * 1000);
                float randomOffsetX = ((float)random.NextDouble() - 0.5f) * formationSize.x;
                float randomOffsetY = ((float)random.NextDouble() - 0.5f) * formationSize.y;
                offset = new float2(randomOffsetX, randomOffsetY);
                break;
        }

        return battalionTarget + offset;
    }

    public float2 GetFormationPositionSimplified(int boidIndex, Battalion battalion)
    {
        float2 offset = float2.zero;
        float boidSpread = formationSpread;
        int localIndex = boidIndex - battalion.startIndex;
        
        if (localIndex < 0 || localIndex >= battalion.count)
            return battalion.targetPosition; // Boid doesn't belong to this battalion
        
        // Calculate formation positions based on type
        switch (battalion.formationType)
        {
            case FormationType.Square:
                // Form a square/rectangular grid
                int cols = Mathf.CeilToInt(Mathf.Sqrt(battalion.count));
                int row = localIndex / cols;
                int col = localIndex % cols;
                
                offset = new float2(
                    (col - cols/2.0f) * boidSpread * battalion.formationSize.x,
                    (row - cols/2.0f) * boidSpread * battalion.formationSize.y
                );
                break;

            case FormationType.Line:
                // Form a horizontal line
                offset = new float2(
                    (localIndex - battalion.count/2.0f) * boidSpread * battalion.formationSize.x,
                    0
                );
                break;

            case FormationType.Column:
                // Form a vertical column
                offset = new float2(
                    0,
                    (localIndex - battalion.count/2.0f) * boidSpread * battalion.formationSize.y
                );
                break;

            case FormationType.Wedge:
                // Form a wedge/triangle
                int wedgeWidth = Mathf.CeilToInt(Mathf.Sqrt(battalion.count * 2));
                int currentRow = 0;
                int currentRowStart = 0;
                int rowWidth = 1;

                while (currentRowStart + rowWidth <= localIndex)
                {
                    currentRowStart += rowWidth;
                    currentRow++;
                    rowWidth++;
                }

                int posInRow = localIndex - currentRowStart;
                offset = new float2(
                    (posInRow - rowWidth/2.0f) * boidSpread * battalion.formationSize.x,
                    -currentRow * boidSpread * battalion.formationSize.y
                );
                break;

            case FormationType.Scattered:
                // Use a deterministic pseudo-random distribution
                UnityEngine.Random.InitState(boidIndex + battalion.id * 1000);
                offset = new float2(
                    (UnityEngine.Random.value - 0.5f) * battalion.formationSize.x,
                    (UnityEngine.Random.value - 0.5f) * battalion.formationSize.y
                );
                break;
        }

        return battalion.targetPosition + offset;
    }


    // Issue a move command to selected battalions
    public void MoveSelectedBattalions(float2 targetPosition)
    {
        foreach (Battalion battalion in selectedBattalions)
        {
            battalion.targetPosition = targetPosition;
        }
    }

    // Change formation type for selected battalions
    public void SetFormation(FormationType formationType)
    {
        foreach (Battalion battalion in selectedBattalions)
        {
            battalion.formationType = formationType;
        }
    }
}