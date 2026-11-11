// RT64 - Adds the half-resolution point light pass (RS64LightsPS with halfParams.x) to the full-resolution frame

#include "shared/rt64_rs64_lights_cb.h"

ConstantBuffer<RS64LightsCB> gLights : register(b0);
Texture2D<float> gDepth : register(t1);
Texture2D<float4> gHalf : register(t2);

// The four nearest half-resolution texels, those at this pixel's distance (alpha code) counting most and sky texels not at all.
float4 PSMain(in float4 pos : SV_Position, in float2 uv : TEXCOORD0) : SV_TARGET {
    const int2 px = int2(pos.xy);
    const float depth = gDepth.Load(int3(px, 0));
    if (depth >= 0.9980f) {
        return float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    const float2 rel = (pos.xy - gLights.viewport.xy) / gLights.viewport.zw;
    const float4 v4 = mul(gLights.viewFromScreen, float4(rel.x * 2.0f - 1.0f, 1.0f - rel.y * 2.0f, depth, 1.0f));
    const float code = length(v4.xyz / v4.w) / 64.0f;
    const int2 lo = int2(gLights.viewport.xy) >> 1;
    const int2 hi = max((int2(gLights.viewport.xy + gLights.viewport.zw) >> 1) - 1, lo);
    const int2 i0 = px >> 1;
    const float2 f = float2(px & 1) * 0.5f;
    float3 sum = float3(0.0f, 0.0f, 0.0f);
    float wsum = 0.0f;
    [unroll]
    for (int k = 0; k < 4; k++) {
        const int2 o2 = int2(k & 1, k >> 1);
        const float4 h = gHalf.Load(int3(clamp(i0 + o2, lo, hi), 0));
        if (h.a >= 60000.0f) {
            continue;
        }

        const float d = abs(h.a - code) / max(code, 1.0e-3f);
        const float bilinear = (o2.x ? f.x : 1.0f - f.x) * (o2.y ? f.y : 1.0f - f.y);
        const float w = (bilinear + 0.01f) / (1.0e-4f + d * d);
        sum += w * h.rgb;
        wsum += w;
    }

    return float4((wsum > 0.0f) ? (sum / wsum) : float3(0.0f, 0.0f, 0.0f), 0.0f);
}
