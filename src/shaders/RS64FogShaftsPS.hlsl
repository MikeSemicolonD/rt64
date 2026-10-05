//
// RT64
//

#include "shared/rt64_rs64_fog_shafts_cb.h"
#include "RS64CutoutAlpha.hlsli"

ConstantBuffer<RS64FogShaftsCB> gFog : register(b1, space3);
Texture2D<float> gDepth : register(t2, space3);

// Mirrors rs64lights::fogFactor (RSPProcessCS vertex fog).
float fogFactor(float3 q) {
    const float4 c = mul(gFog.viewProj, float4(q, 1.0f));
    const float a = (c.w > 1e-6f) ? ((max(c.z, 0.0f) / c.w) * gFog.fogMul + gFog.fogOffset) : gFog.fogOffset;
    return saturate(a / 255.0f);
}

// Mirrors rs64lights::interleavedNoise.
float interleavedNoise(float2 xy) {
    return frac(52.9829189f * frac(dot(xy, float2(0.06711056f, 0.00583715f))));
}

// Same two rays as RS64ShadowsPS: drawn casters (0x01), then static terrain (0x02) from a view-distance-scaled start.
float sunVisible(float3 q) {
    RayDesc ray;
    ray.Origin = q;
    ray.Direction = gFog.sunDir.xyz;
    ray.TMin = gFog.tMin;
    ray.TMax = gFog.tMax;
    float hitT;
    uint id;
    return rs64SunHit(ray, gFog.cutoutDrawCount, hitT, id) ? 0.0f : 1.0f;
}

float4 PSMain(in float4 pos : SV_Position, in float2 uv : TEXCOORD0) : SV_TARGET {
    const bool debug = (gFog.debugMode != 0);
    // Sky pin and cleared depth: the game never fogs the sky, so it gets no in-scatter.
    const float depth = gDepth.Load(int3(pos.xy, 0));
    if (depth >= gFog.skyDepth) {
        return debug ? float4(0.0f, 0.0f, 0.0f, 1.0f) : float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    const float2 rel = (pos.xy - gFog.viewport.xy) / gFog.viewport.zw;
    const float4 s = float4(rel.x * 2.0f - 1.0f, 1.0f - rel.y * 2.0f, depth, 1.0f);
    const float4 v4 = mul(gFog.viewFromScreen, s);
    if (abs(v4.w) < 1e-20f) {
        return debug ? float4(0.0f, 0.0f, 0.0f, 1.0f) : float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    const float3 p = v4.xyz / v4.w;
    const float fEnd = fogFactor(p);
    if (fEnd <= (1.0f / 512.0f)) {
        return debug ? float4(0.0f, 0.0f, 0.0f, 1.0f) : float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    // Mirrors rs64lights::fogSteps + litInScatter; a step where the fog does not grow adds nothing, so its ray is skipped.
    const float jitter = interleavedNoise(pos.xy);
    const uint n = gFog.steps;
    float fPrev = 0.0f;
    float lit = 0.0f;
    for (uint i = 0; i < n; i++) {
        const float t = (i + 1 == n) ? 1.0f : (float(i) + 1.0f - jitter) / float(n);
        const float3 q = p * t;
        const float f = fogFactor(q);
        const float df = f - fPrev;
        if (df > 0.0f) {
            // Mirrors rs64lights::fogVisT.
            lit += sunVisible(p * (t * (1.0f - gFog.visBias))) * df;
        }
        fPrev = f;
    }

    if (debug) {
        const float m = lit / fEnd;
        return float4(m, m, m, 1.0f);
    }
    return float4(gFog.strength * gFog.fogColor.rgb * lit, 0.0f);
}
