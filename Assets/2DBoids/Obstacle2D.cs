using UnityEngine;
using Unity.Mathematics;

[RequireComponent(typeof(SpriteRenderer))]
public class Obstacle2D : MonoBehaviour
{
    [Header("Repulsion Settings")]
    [Tooltip("How strongly this obstacle repels boids")]
    public float repulsionStrength = 5f;
    
    [Tooltip("Distance at which boids start to be repelled")]
    public float repulsionRadius = 2f;
    
    [Tooltip("Debug visualization")]
    public bool showDebugRadius = true;
    
    private SpriteRenderer spriteRenderer;
    private Color originalColor;
    
    void Start()
    {
        spriteRenderer = GetComponent<SpriteRenderer>();
        if (spriteRenderer != null)
        {
            originalColor = spriteRenderer.color;
            
            // Set the sprite's size to match the repulsion radius
            transform.localScale = new Vector3(repulsionRadius * 2, repulsionRadius * 2, 1);
        }
    }
    
    void OnDrawGizmos()
    {
        if (showDebugRadius)
        {
            Gizmos.color = new Color(1f, 0.2f, 0.2f, 0.3f);
            Gizmos.DrawSphere(transform.position, repulsionRadius);
        }
    }
    
    // Method for CPU-based repulsion calculation
    public float2 GetRepulsionForce(float2 boidPosition)
    {
        float2 obstaclePos = new float2(transform.position.x, transform.position.y);
        float2 direction = boidPosition - obstaclePos;
        float distance = math.length(direction);
        
        if (distance < repulsionRadius)
        {
            // Normalize direction
            float2 normalizedDir = direction / distance;
            
            // Scale force inversely with distance (closer = stronger)
            float forceMagnitude = repulsionStrength * (1.0f - (distance / repulsionRadius));
            return normalizedDir * forceMagnitude;
        }
        
        return float2.zero;
    }
}