// RT64 - Sun shafts: stepped in-scatter through the fog along each view ray

#include "shared/rt64_rs64_fog_shafts_cb.h"
#include "shared/rt64_rsp_fog.h"
#include "RS64Trace.hlsli"

ConstantBuffer<RS64FogShaftsCB> gFog : register(b1, space3);
Texture2D<float> gDepth : register(t2, space3);

// Mirrors rs64lights::fogFactor (RSPProcessCS vertex fog).
float fogFactor(float3 q) {
    const float4 c = mul(gFog.viewProj, float4(q, 1.0f));
    const RSPFog fog = { gFog.fogMul, gFog.fogOffset, gFog.fogTail, gFog.fogColor.w };
    const float a = (c.w > 1e-6f) ? (rs64FogDepth(gFog.viewProj, q, fog) * gFog.fogMul + gFog.fogOffset) : gFog.fogOffset;
    return rs64FogCurve(a / 255.0f, gFog.fogTail);
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

float3 viewPoint(float2 xy, float depth) {
    const float2 rel = (xy - gFog.viewport.xy) / gFog.viewport.zw;
    const float4 v4 = mul(gFog.viewFromScreen, float4(rel.x * 2.0f - 1.0f, 1.0f - rel.y * 2.0f, depth, 1.0f));
    return v4.xyz / ((abs(v4.w) < 1e-20f) ? 1e-20f : v4.w);
}

// Length of the haze march through a full-resolution pixel: to the receiver, capped at hazeRange (the sky takes the full range).
float hazeDist(float2 xy, float depth) {
    return (depth >= gFog.skyDepth) ? gFog.hazeRange : min(length(viewPoint(xy, depth)), gFog.hazeRange);
}

#if defined(RS64_HAZE)

// Level haze, independent of the game's fog: sunlit air along the view ray, brightest toward the sun. Drawn at half resolution, march length in alpha.
float4 PSMain(in float4 pos : SV_Position, in float2 uv : TEXCOORD0) : SV_TARGET {
    const int2 px = min(int2(pos.xy) * 2, int2(gFog.viewport.xy + gFog.viewport.zw) - 1);
    const float2 xy = float2(px) + 0.5f;
    const float depth = gDepth.Load(int3(px, 0));
    const float3 dir = normalize(viewPoint(xy, 0.5f));
    const float dist = hazeDist(xy, depth);
    const float jitter = interleavedNoise(pos.xy);
    // Every haze step traces (the fog pass skips steps where its fog does not grow), so it takes half the steps; the jitter hides the banding.
    const uint n = max(gFog.steps / 2u, 4u);
    float lit = 0.0f;
    for (uint i = 0; i < n; i++) {
        lit += sunVisible(dir * (dist * (float(i) + jitter) / float(n)));
    }

    const float g = 0.6f;
    const float hg = (1.0f - g * g) / pow(1.0f + g * g - 2.0f * g * dot(dir, gFog.sunDir.xyz), 1.5f);
    const float phase = 0.25f + 0.75f * hg * (1.0f - g) * (1.0f - g) * (1.0f - g) / (1.0f - g * g);
    const float scatter = 1.0f - exp(-gFog.hazeDensity * dist * lit / float(n));
    return float4(gFog.hazeColor.rgb * phase * scatter, dist);
}

#elif defined(RS64_HAZE_UPSAMPLE)

Texture2D<float4> gColor : register(t12, space3);

// Adds the half-resolution haze to the frame: the four nearest half texels, those with this pixel's march length counting most.
float4 PSMain(in float4 pos : SV_Position, in float2 uv : TEXCOORD0) : SV_TARGET {
    const int2 px = int2(pos.xy);
    const float dist = hazeDist(pos.xy, gDepth.Load(int3(px, 0)));
    const int2 lo = int2(gFog.viewport.xy) >> 1;
    const int2 hi = max((int2(gFog.viewport.xy + gFog.viewport.zw) >> 1) - 1, lo);
    const int2 i0 = (px - 1) >> 1;
    const float2 f = float2((px - 1) & 1) * 0.5f + 0.25f;
    float3 sum = float3(0.0f, 0.0f, 0.0f);
    float wsum = 0.0f;
    [unroll]
    for (int k = 0; k < 4; k++) {
        const int2 o2 = int2(k & 1, k >> 1);
        const float4 h = gColor.Load(int3(clamp(i0 + o2, lo, hi), 0));
        const float d = abs(h.a - dist) / max(dist, 1.0f);
        const float bilinear = (o2.x ? f.x : 1.0f - f.x) * (o2.y ? f.y : 1.0f - f.y);
        const float w = (bilinear + 0.01f) / (1.0e-4f + d * d);
        sum += w * h.rgb;
        wsum += w;
    }

    return float4((wsum > 0.0f) ? (sum / wsum) : float3(0.0f, 0.0f, 0.0f), 0.0f);
}

#else

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
    float tPrev = 0.0f;
    float lit = 0.0f;
    for (uint i = 0; i < n; i++) {
        const float t = (i + 1 == n) ? 1.0f : (float(i) + 1.0f - jitter) / float(n);
        const float3 q = p * t;
        const float f = fogFactor(q);
        const float df = f - fPrev;
        if (df > 0.0f) {
            // Mirrors rs64lights::fogVisT, sampled at the step's middle. The last step, next to the receiver, carries most of the game's fog and counts as lit: its rays only traced the shape of the rock into the fog.
            lit += ((i + 1 == n) ? 1.0f : sunVisible(p * (0.5f * (tPrev + t) * (1.0f - gFog.visBias)))) * df;
        }
        fPrev = f;
        tPrev = t;
    }

    if (debug) {
        const float m = lit / fEnd;
        return float4(m, m, m, 1.0f);
    }
    // The game's fog already assumes lit air; the pass subtracts the share the sun doesn't reach, so shafts are shadowed fog, not extra glow.
    return float4(gFog.strength * gFog.fogColor.rgb * max(fEnd - lit, 0.0f), 0.0f);
}

#endif
