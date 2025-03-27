Shader "Unlit/boidShader" {
  Properties {
    _Team0Color ("Team 0 Color", Color) = (0, 0, 1, 1)
    _Team1Color ("Team 1 Color", Color) = (1, 0, 0, 1)
    _Scale ("Scale", Float) = 0.1
  }
  SubShader {
    Pass {
      CGPROGRAM
      #pragma vertex vert
      #pragma fragment frag

      struct Boid {
        float2 pos;
        float2 vel;
        uint team;
      };
      struct v2f {
        float4 position : SV_POSITION;
        uint team : TEXCOORD0;
      };

      void rotate2D(inout float2 v, float2 vel) {
        float2 dir = normalize(vel);
        v = float2(v.x * dir.y + v.y * dir.x, v.y * dir.y - v.x * dir.x);
      }

      float4 _Team0Color;
      float4 _Team1Color;
      //float4 _Colour;
      float _Scale;
      StructuredBuffer<Boid> boids;
      StructuredBuffer<float2> _Positions;

      v2f vert(uint vertexID : SV_VertexID) {
        v2f o;
        uint instanceID = vertexID / 3;
        Boid boid = boids[instanceID];
        float2 pos = _Positions[vertexID - instanceID * 3];
        rotate2D(pos, boid.vel);
        o.position = UnityObjectToClipPos(float4(pos * _Scale + boid.pos.xy, 0, 0));
        o.team = boid.team;
        return o;
      }

      fixed4 frag(v2f i) : SV_Target {
        return i.team == 0 ? _Team0Color : _Team1Color;
      }
      ENDCG
    }
  }
}
