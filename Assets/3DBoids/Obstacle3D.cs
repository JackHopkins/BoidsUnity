using UnityEngine;
using Unity.Mathematics;

[RequireComponent(typeof(MeshRenderer))]
public class Obstacle3D : MonoBehaviour
{
    [Header("Repulsion Settings")]
    [Tooltip("How strongly this obstacle repels boids")]
    public float repulsionStrength = 5f;
    
    [Tooltip("Distance at which boids start to be repelled")]
    public float repulsionRadius = 2f;
    
    [Tooltip("Debug visualization")]
    public bool showDebugRadius = true;
    
    private MeshRenderer meshRenderer;
    private Color originalColor;
    
    void Start()
    {
        meshRenderer = GetComponent<MeshRenderer>();
        if (meshRenderer != null)
        {
            if (meshRenderer.material.HasProperty("_Color"))
            {
                originalColor = meshRenderer.material.color;
            }
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
    public float3 GetRepulsionForce(float3 boidPosition)
    {
        float3 obstaclePos = new float3(transform.position.x, transform.position.y, transform.position.z);
        float3 direction = boidPosition - obstaclePos;
        float distance = math.length(direction);
        
        if (distance < repulsionRadius)
        {
            // Normalize direction
            float3 normalizedDir = direction / distance;
            
            // Scale force inversely with distance (closer = stronger)
            float forceMagnitude = repulsionStrength * (1.0f - (distance / repulsionRadius));
            return normalizedDir * forceMagnitude;
        }
        
        return float3.zero;
    }
}