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
    const bool sky = depth >= 0.9980f;

    // Glowing panel faces (their own alpha-blended draw): seen from the front, where the view ray meets one before the scene.
    if (gLights.debugMode == RS64_LIGHTS_PANEL_FACES) {
        float3 glow = float3(0.0f, 0.0f, 0.0f);
        float cover = 0.0f;
        // The game's depths sit near 1 (a mid value can unproject behind the camera), so the view ray comes from one in that range.
        const float3 eye = normalize(viewFromPixel(pos.xy, min(depth, 0.997f)));
        const float sceneDist = sky ? 1e30f : length(p);
        for (uint g = 0; g < gLights.lightCount; g++) {
            const float3 ru = gLights.lightRectU[g].xyz;
            const float3 rv = gLights.lightRectV[g].xyz;
            const float3 face = gLights.lightSpot[g].xyz;
            const float facing = dot(eye, face);
            if ((dot(ru, ru) <= 0.0f) || (facing >= 0.0f)) {
                continue;
            }

            const float3 c = gLights.lightPosRadius[g].xyz;
            const float t = dot(c, face) / facing;
            const float3 h = eye * t - c;
            if ((t > 0.0f) && (t < sceneDist) && (abs(dot(h, ru)) <= dot(ru, ru)) && (abs(dot(h, rv)) <= dot(rv, rv))) {
                glow += gLights.lightColor[g].rgb * gLights.lightColor[g].w;
                cover = 1.0f;
            }
        }

        return float4(saturate(glow), cover);
    }

    if (sky) {
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
        // A panel lights from its nearest point, one-sided along its facing.
        const float3 ru = gLights.lightRectU[i].xyz;
        const float3 rv = gLights.lightRectV[i].xyz;
        const bool panel = dot(ru, ru) > 0.0f;
        float3 lp = gLights.lightPosRadius[i].xyz;
        if (panel) {
            const float3 rel = p - lp;
            lp += ru * clamp(dot(rel, ru) / dot(ru, ru), -1.0f, 1.0f) + rv * clamp(dot(rel, rv) / dot(rv, rv), -1.0f, 1.0f);
        }

        const float3 toLight = lp - p;
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
        // Spot cone as RT64's Lights.hlsli: lit where dot(surface -> light, beam) <= max cosine, fading out from the falloff cosine to it.
        float spot = 1.0f;
        const float maxCos = gLights.lightParams[i].z, fallCos = gLights.lightParams[i].w;
        if (panel) {
            spot = saturate(dot(-toLight / max(dist, 1e-4f), gLights.lightSpot[i].xyz));
        }
        else if (maxCos < 1.0f) {
            const float d = dot(toLight / max(dist, 1e-4f), gLights.lightSpot[i].xyz);
            spot = (d > maxCos) ? 0.0f : ((maxCos - fallCos) > 1e-5f ? 1.0f - saturate((d - fallCos) / (maxCos - fallCos)) : 1.0f);
        }

        float spec = 0.0f;
        if (gLights.specParams.x > 0.0f) {
            const float3 l = toLight / max(dist, 1e-4f);
            const float3 h = normalize(l - normalize(p));
            spec = gLights.specParams.x * pow(saturate(dot(n, h)), gLights.specParams.y) * saturate(dot(n, l) * 4.0f);
        }

        sum += gLights.lightColor[i].rgb * gLights.lightColor[i].w * att * (ndl + spec) * vis * spot;
    }

    if (gLights.debugMode == 2) {
        return float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    return float4(sum * gLights.gain, 0.0f);
}
