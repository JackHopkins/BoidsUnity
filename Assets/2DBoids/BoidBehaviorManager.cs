using UnityEngine;
using Unity.Mathematics;
using Unity.Collections;
using System.Collections.Generic;

namespace BoidsUnity
{
    public class BoidBehaviorManager
    {
        // Movement settings
        public float maxSpeed = 2f;
        public float minSpeed;
        public float turnSpeed;
        public float edgeMargin = 0.5f;
        
        // Behavior settings
        public float visualRange = 0.5f;
        public float minDistance = 0.15f;
        public float visualRangeSq => visualRange * visualRange;
        public float minDistanceSq => minDistance * minDistance;
        public float cohesionFactor = 2f;
        public float separationFactor = 1f;
        public float alignmentFactor = 5f;
        
        // Team settings
        public float teamRatio = 0.5f;
        public float intraTeamCohesionMultiplier = 2.5f;
        public float interTeamRepulsionMultiplier = 2.5f;
        public Color team0Color = Color.blue;
        public Color team1Color = Color.red;
        
        // Obstacle settings
        public float obstacleAvoidanceWeight = 5f;
        public List<Obstacle2D> obstacles = new List<Obstacle2D>();
        
        // Bounds
        public float xBound;
        public float yBound;
        
        public BoidBehaviorManager()
        {
            minSpeed = maxSpeed * 0.75f;
            turnSpeed = maxSpeed * 3f;
        }

        public void InitializeBounds(float cameraSize, float margin)
        {
            xBound = cameraSize * Camera.main.aspect - margin;
            yBound = cameraSize - margin;
        }

        public void MergedBehaviours(ref Boid boid, NativeArray<Boid> boidsTemp, int[] gridOffsets, int gridDimX, int gridDimY, float gridCellSize)
        {
            float2 center = float2.zero;
            float2 close = float2.zero;
            float2 avgVel = float2.zero;
            int sameTeamNeighbours = 0;
            int otherTeamNeighbours = 0;

            var gridXY = GetGridLocation(boid, gridCellSize, gridDimX, gridDimY);
            int gridCell = GetGridIDbyLoc(gridXY, gridDimX);

            for (int y = gridCell - gridDimX; y <= gridCell + gridDimX; y += gridDimX)
            {
                // Check bounds to avoid index errors
                if (y < 0 || y >= gridOffsets.Length - 2) continue;
                
                int start = gridOffsets[y - 2];
                int end = gridOffsets[y + 1];
                for (int i = start; i < end; i++)
                {
                    Boid other = boidsTemp[i];
                    var diff = boid.pos - other.pos;
                    var distanceSq = math.dot(diff, diff);
                    if (distanceSq > 0 && distanceSq < visualRangeSq)
                    {
                        bool sameTeam = boid.team == other.team;
                        
                        if (distanceSq < minDistanceSq)
                        {
                            float repulsionStrength = sameTeam ? 1.0f : interTeamRepulsionMultiplier;
                            close += diff / distanceSq * repulsionStrength;
                        }
                        
                        if (sameTeam)
                        {
                            center += other.pos;
                            avgVel += other.vel;
                            sameTeamNeighbours++;
                        }
                        else
                        {
                            otherTeamNeighbours++;
                        }
                    }
                }
            }

            if (sameTeamNeighbours > 0)
            {
                center /= sameTeamNeighbours;
                avgVel /= sameTeamNeighbours;

                // Apply stronger cohesion with same team
                boid.vel += (center - boid.pos) * (cohesionFactor * intraTeamCohesionMultiplier * Time.deltaTime);
                boid.vel += (avgVel - boid.vel) * (alignmentFactor * Time.deltaTime);
            }

            boid.vel += close * (separationFactor * Time.deltaTime);
            
            // Add obstacle avoidance
            float2 obstacleForce = float2.zero;
            
            if (obstacles != null && obstacles.Count > 0)
            {
                foreach (var obstacle in obstacles)
                {
                    if (obstacle != null)
                    {
                        obstacleForce += obstacle.GetRepulsionForce(boid.pos);
                    }
                }
                
                boid.vel += obstacleForce * obstacleAvoidanceWeight * Time.deltaTime;
            }
        }

        public void LimitSpeed(ref Boid boid)
        {
            var speed = math.length(boid.vel);
            var clampedSpeed = Mathf.Clamp(speed, minSpeed, maxSpeed);
            boid.vel *= clampedSpeed / speed;
        }

        public void KeepInBounds(ref Boid boid)
        {
            if (Mathf.Abs(boid.pos.x) > xBound)
            {
                boid.vel.x -= Mathf.Sign(boid.pos.x) * Time.deltaTime * turnSpeed;
            }
            if (Mathf.Abs(boid.pos.y) > yBound)
            {
                boid.vel.y -= Mathf.Sign(boid.pos.y) * Time.deltaTime * turnSpeed;
            }
        }

        // Helper methods for grid calculations
        public int2 GetGridLocation(Boid boid, float gridCellSize, int gridDimX, int gridDimY)
        {
            int gridX = Mathf.FloorToInt(boid.pos.x / gridCellSize + gridDimX / 2);
            int gridY = Mathf.FloorToInt(boid.pos.y / gridCellSize + gridDimY / 2);
            return new int2(gridX, gridY);
        }

        public int GetGridIDbyLoc(int2 cell, int gridDimX)
        {
            return (gridDimX * cell.y) + cell.x;
        }
    }
}
