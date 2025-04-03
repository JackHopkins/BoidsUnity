Shader "Unlit/boidShader" {
  Properties {
    _Team0Color ("Team 0 Color", Color) = (0, 0, 1, 1)
    _Team1Color ("Team 1 Color", Color) = (1, 0, 0, 1)
    _OfficerDarkening ("Officer Darkening", Range(0.4, 0.8)) = 0.6
    _Scale ("Scale", Float) = 0.1
    _OfficerScale ("Scale", Float) = 0.2
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
        uint status;
      };
      struct v2f {
        float4 position : SV_POSITION;
        uint team : TEXCOORD0;
        uint status : TEXCOORD1;
      };

      void rotate2D(inout float2 v, float2 vel) {
        float2 dir = normalize(vel);
        v = float2(v.x * dir.y + v.y * dir.x, v.y * dir.y - v.x * dir.x);
      }

      float4 _Team0Color;
      float4 _Team1Color;
      float _OfficerDarkening;
      float _Scale;
      float _OfficerScale;
      
      StructuredBuffer<Boid> boids;
      StructuredBuffer<float2> _Positions;

      v2f vert(uint vertexID : SV_VertexID) {
        v2f o;
        uint instanceID = vertexID / 3;
        Boid boid = boids[instanceID];
        float2 pos = _Positions[vertexID - instanceID * 3];
        rotate2D(pos, boid.vel);
        // Scale officers differently
        if (boid.status == 0) {
          o.position = UnityObjectToClipPos(float4(pos * _Scale + boid.pos.xy, 0, 0));
        } else
        {
          o.position = UnityObjectToClipPos(float4(pos * _OfficerScale + boid.pos.xy, 0, 0));
        }
        o.team = boid.team;
        o.status = boid.status;
        return o;
      }

      fixed4 frag(v2f i) : SV_Target {
        fixed4 teamColor = i.team == 0 ? _Team0Color : _Team1Color;
        // Apply darker color for officers
        return i.status == 1 ? teamColor * _OfficerDarkening : teamColor;
      }
      ENDCG
    }
  }
}
