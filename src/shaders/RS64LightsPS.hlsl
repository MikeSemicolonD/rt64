//
// RT64
//

#include "shared/rt64_rs64_lights_cb.h"

#ifdef RS64_LIGHT_SHADOWS
#include "RS64CutoutAlpha.hlsli"
#include "RS64TerrainNormal.hlsli"
ConstantBuffer<RS64LightsCB> gLights : register(b1, space3);
Texture2D<float> gDepth : register(t2, space3);
#else
ConstantBuffer<RS64LightsCB> gLights : register(b0);
Texture2D<float> gDepth : register(t1);
#endif

#ifdef RS64_LIGHT_SHADOWS

// Mirrors rs64lights::lightRaySpan; casters, cutouts and static terrain in one query (rs64SunHit).
bool lightVisible(float3 p, float3 n, float3 toLight, float dist, float radius, float shadowStart) {
    const float tMax = dist - max(gLights.shadowParams.x * radius, shadowStart);
    if (tMax <= gLights.tMin) {
        return true;
    }
    const float viewDist = length(p);
    RayDesc ray;
    ray.Origin = p + n * (gLights.normalBias * viewDist);
    ray.Direction = toLight / dist;
    ray.TMin = gLights.tMin;
    ray.TMax = tMax;
    float hitT;
    uint id;
    return !rs64SunHit(ray, uint(gLights.shadowParams.y), hitT, id);
}
#endif

float3 viewFromPixel(float2 px, float depth) {
    const float2 rel = (px - gLights.viewport.xy) / gLights.viewport.zw;
    const float4 s = float4(rel.x * 2.0f - 1.0f, 1.0f - rel.y * 2.0f, depth, 1.0f);
    const float4 v = mul(gLights.viewFromScreen, s);
    return v.xyz / v.w;
}

float4 PSMain(in float4 pos : SV_Position, in float2 uv : TEXCOORD0) : SV_TARGET {
    const float depth = gDepth.Load(int3(pos.xy, 0));
    const float3 p = viewFromPixel(pos.xy, depth);
    float3 n = normalize(cross(ddy(p), ddx(p)));
    if (dot(n, p) > 0.0f) {
        n = -n;
    }

    // Sky and cleared depth: the game pins the sky dome at depth 0x7FBF (~0.99805), just below the clear value.
    if (depth >= 0.9980f) {
        return float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

#ifdef RS64_LIGHT_SHADOWS
    // Static terrain: the smooth heightfield normal instead of the per-facet one (shadowParams.z = terrain normals on).
    float3 nT;
    if ((gLights.shadowParams.z != 0.0f) && rs64TerrainNormal(p, length(p), gLights.shadowParams.w, nT)) {
        n = (dot(nT, p) > 0.0f) ? -nT : nT;
    }
#endif

    if (gLights.debugMode == 1) {
        const float d = saturate(length(p) / gLights.debugRange);
        return float4(d, d, d, 1.0f);
    }

    float3 sum = float3(0.0f, 0.0f, 0.0f);
#ifdef RS64_LIGHT_SHADOWS
    // The 4 nearest in-range lights get a shadow ray (mirrors rs64lights::nearestLights); the rest stay unshadowed.
    uint ranked[4] = { 0, 0, 0, 0 };
    float rankedDist[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    uint rankCount = 0;
    for (uint r = 0; r < gLights.lightCount; r++) {
        const float d = length(gLights.lightPosRadius[r].xyz - p);
        if (d >= gLights.lightPosRadius[r].w) {
            continue;
        }
        uint slot = rankCount;
        while ((slot > 0) && (rankedDist[slot - 1] > d)) {
            slot--;
        }
        if (slot >= 4) {
            continue;
        }
        for (uint j = min(rankCount, 3u); j > slot; j--) {
            ranked[j] = ranked[j - 1];
            rankedDist[j] = rankedDist[j - 1];
        }
        ranked[slot] = r;
        rankedDist[slot] = d;
        rankCount = min(rankCount + 1, 4u);
    }
    if (gLights.debugMode == 3) {
        if (rankCount == 0) {
            return float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        const uint k = ranked[0];
        const float3 toK = gLights.lightPosRadius[k].xyz - p;
        const bool vis = lightVisible(p, n, toK, length(toK), gLights.lightPosRadius[k].w, gLights.lightParams[k].y);
        return vis ? float4(0.0f, 1.0f, 0.0f, 0.6f) : float4(1.0f, 0.0f, 0.0f, 0.6f);
    }
#endif
    for (uint i = 0; i < gLights.lightCount; i++) {
        const float3 toLight = gLights.lightPosRadius[i].xyz - p;
        const float dist = length(toLight);
        const float radius = gLights.lightPosRadius[i].w;
        if (dist >= radius) {
            continue;
        }

        if (gLights.debugMode == 2) {
            return float4(1.0f, 0.0f, 1.0f, 0.6f);
        }

        float vis = 1.0f;
#ifdef RS64_LIGHT_SHADOWS
        for (uint s = 0; s < rankCount; s++) {
            if (ranked[s] == i) {
                vis = lightVisible(p, n, toLight, dist, radius, gLights.lightParams[i].y) ? 1.0f : 0.0f;
            }
        }
#endif
        const float att = pow(saturate(1.0f - dist / radius), gLights.lightParams[i].x);
        const float ndl = saturate((dot(n, toLight / max(dist, 1e-4f)) + gLights.wrap) / (1.0f + gLights.wrap));
        sum += gLights.lightColor[i].rgb * gLights.lightColor[i].w * att * ndl * vis;
    }

    if (gLights.debugMode == 2) {
        return float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    return float4(sum * gLights.gain, 0.0f);
}
