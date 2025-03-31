using UnityEngine;
using Unity.Mathematics;

namespace BoidsUnity
{
    // Defining structs for the boid simulation
    public struct Boid
    {
        public float2 pos;
        public float2 vel;
        public uint team;
    }

    public struct ObstacleData
    {
        public float2 pos;
        public float radius;
        public float strength;
    }

    public struct QuadNode
    {
        public float2 center;
        public float size;      // Half-width of the node
        public uint childIndex; // Index of first child (other children are at childIndex+1, +2, +3)
        public uint startIndex; // Start index for boids in this node
        public uint count;      // Number of boids in this node
        public uint flags;      // Bit flags: 1=leaf, 2=active, etc.
    }
}
