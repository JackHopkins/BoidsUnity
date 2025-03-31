using UnityEngine;
using Unity.Mathematics;
using System.Collections.Generic;

namespace BoidsUnity
{
    public class ObstacleManager
    {
        private List<Obstacle2D> obstacles = new List<Obstacle2D>();
        private float obstacleAvoidanceWeight = 5f;
        private ComputeBuffer obstacleBuffer;
        private int maxObstacles = 10;
        
        public ObstacleManager(float avoidanceWeight = 5f, int maxObstacles = 10)
        {
            this.obstacleAvoidanceWeight = avoidanceWeight;
            this.maxObstacles = maxObstacles;
        }
        
        public void FindAllObstacles()
        {
            obstacles.Clear();
            obstacles.AddRange(GameObject.FindObjectsOfType<Obstacle2D>());
        }
        
        public void InitializeObstacles(ComputeShader boidShader, int updateBoidsKernel, bool useGpu)
        {
            // Find all obstacles in the scene if not set in inspector
            if (obstacles.Count == 0)
            {
                FindAllObstacles();
            }
            
            // Create the obstacle buffer for GPU processing
            if (useGpu)
            {
                // Make sure we have at least one element in the buffer to avoid errors
                obstacleBuffer = new ComputeBuffer(Mathf.Max(1, obstacles.Count), 16); // 16 bytes per obstacle
                var obstacleData = new ObstacleData[Mathf.Max(1, obstacles.Count)];
                
                for (int i = 0; i < obstacles.Count; i++)
                {
                    obstacleData[i] = new ObstacleData
                    {
                        pos = new float2(obstacles[i].transform.position.x, obstacles[i].transform.position.y),
                        radius = obstacles[i].repulsionRadius,
                        strength = obstacles[i].repulsionStrength
                    };
                }
                
                obstacleBuffer.SetData(obstacleData);
                
                // Set the obstacle buffer to the compute shader
                boidShader.SetBuffer(updateBoidsKernel, "obstacles", obstacleBuffer);
                boidShader.SetInt("numObstacles", obstacles.Count);
                boidShader.SetFloat("obstacleAvoidanceWeight", obstacleAvoidanceWeight);
            }
        }
        
        public void UpdateObstacleData(bool useGpu)
        {
            if (useGpu && obstacles.Count > 0)
            {
                var obstacleData = new ObstacleData[obstacles.Count];
                
                for (int i = 0; i < obstacles.Count; i++)
                {
                    if (obstacles[i] != null)
                    {
                        obstacleData[i] = new ObstacleData
                        {
                            pos = new float2(obstacles[i].transform.position.x, obstacles[i].transform.position.y),
                            radius = obstacles[i].repulsionRadius,
                            strength = obstacles[i].repulsionStrength
                        };
                    }
                }
                
                obstacleBuffer.SetData(obstacleData);
            }
        }
        
        public List<Obstacle2D> GetObstacles()
        {
            return obstacles;
        }
        
        public void Cleanup()
        {
            obstacleBuffer?.Release();
        }
    }
}
