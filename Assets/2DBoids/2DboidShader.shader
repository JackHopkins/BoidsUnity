Shader "Unlit/BattalionBoidShader" {
  Properties {
    _Color ("Color", Color) = (1, 1, 0, 1)
    _Scale ("Scale", Float) = 0.1
  }
  SubShader {
    Tags { "RenderType"="Opaque" }
    LOD 100

    Pass {
      CGPROGRAM
      #pragma vertex vert
      #pragma fragment frag
      #include "UnityCG.cginc"

      struct Boid {
        float2 pos;
        float2 vel;
        float2 target;
        int battalionId;
      };

      void rotate2D(inout float2 v, float2 vel) {
        float2 dir = normalize(vel);
        v = float2(v.x * dir.y + v.y * dir.x, v.y * dir.y - v.x * dir.x);
      }

      float4 _Color;
      float _Scale;
      StructuredBuffer<Boid> boids;
      StructuredBuffer<float2> _Positions;

      struct appdata {
        uint vertexID : SV_VertexID;
      };

      struct v2f {
        float4 pos : SV_POSITION;
        float3 color : COLOR;
        float battalionId : TEXCOORD0;
      };

      v2f vert(appdata v) {
        uint instanceID = v.vertexID / 3;
        Boid boid = boids[instanceID];
        float2 pos = _Positions[v.vertexID - instanceID * 3];
        rotate2D(pos, boid.vel);
        v2f o;
        o.pos = UnityObjectToClipPos(float4(pos * _Scale + boid.pos.xy, 0, 1));
        
        // Pass the battalion ID to the fragment shader
        o.color = _Color.rgb;
        o.battalionId = boid.battalionId;
        return o;
      }
      
      fixed4 frag(v2f i) : SV_Target {
        // Use the passed color from the vertex shader (which is from the MaterialPropertyBlock)
        return fixed4(i.color, 1.0);
      }
      ENDCG
    }
  }
}