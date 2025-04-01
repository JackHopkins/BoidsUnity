using UnityEngine;
using UnityEngine.Rendering;

namespace BoidsUnity
{
    public class BoidRenderer
    {
        private Material boidMaterial;
        private Vector2[] triangleVerts;
        private GraphicsBuffer trianglePositions;
        private RenderParams renderParams;
        
        public BoidRenderer(Material material)
        {
            boidMaterial = material;
            triangleVerts = GetTriangleVerts();
        }
        
        public void Initialize(ComputeBuffer boidBuffer, Color team0Color, Color team1Color)
        {
            // Set render params
            renderParams = new RenderParams(boidMaterial);
            renderParams.matProps = new MaterialPropertyBlock();
            renderParams.matProps.SetBuffer("boids", boidBuffer);
            renderParams.worldBounds = new Bounds(Vector3.zero, Vector3.one * 3000);
            
            // Set up team colors
            boidMaterial.SetColor("_Team0Color", team0Color);
            boidMaterial.SetColor("_Team1Color", team1Color);
            
            // Set up triangle positions
            trianglePositions = new GraphicsBuffer(GraphicsBuffer.Target.Structured, 3, 8);
            trianglePositions.SetData(triangleVerts);
            renderParams.matProps.SetBuffer("_Positions", trianglePositions);
        }
        
        public void RenderBoids(int numBoids)
        {
            ProfilingUtility.BeginSample("RenderBoids");
            
            // Time the GPU rendering operation
            System.Diagnostics.Stopwatch sw = System.Diagnostics.Stopwatch.StartNew();
            
            // Render triangles (3 per boid)
            Graphics.RenderPrimitives(renderParams, MeshTopology.Triangles, numBoids * 3);
            
            // Force completion before timing
            GL.Flush();
            
            sw.Stop();
            double ms = sw.Elapsed.TotalMilliseconds;
            
            // Record timing stats
            ProfilingUtility.RecordManualTiming("RenderBoidsGPU", ms);
            
            // Occasional logging for high boid counts
            if (numBoids > 25000 && Time.frameCount % 60 == 0)
            {
                Debug.Log($"Rendering {numBoids} boids took {ms:F2}ms");
            }
            
            ProfilingUtility.EndSample("RenderBoids");
        }
        
        public void Cleanup()
        {
            trianglePositions?.Release();
        }
        
        private Vector2[] GetTriangleVerts()
        {
            return new Vector2[] {
                new(-0.4f, -0.5f),
                new(0, 0.5f),
                new(0.4f, -0.5f),
            };
        }
    }
}
