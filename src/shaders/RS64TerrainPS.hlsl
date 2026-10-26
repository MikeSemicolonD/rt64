//
// RT64
//

#include "RS64Terrain.hlsli"

float3 texel(uint layer, int2 p) {
    p = clamp(p, 0, 63);
    return unpackRGBA8(gTexels[layer * 4096 + p.y * 64 + p.x]).rgb;
}

// Water strength of the view-angle tint toward the sky (fog colour) and of the sun glint.
static const float kWaterFresnel = 0.6f;
static const float kWaterGlint = 0.6f;
static const float kWaterGlintPower = 48.0f;

// Cells past the map edge over which the water there fades fully into the fog.
static const float kEdgeFogCells = 16.0f;

// The terrain combiner (TEXEL0 * SHADE) and the blender's fog (shade alpha toward the fog colour); the tile texture is clamped bilinear.
// The water draw also tints toward the sky at grazing angles and adds the sun's glint, in camera space.
float4 PSMain(in float4 vPosition : SV_POSITION, in float4 vShade : COLOR0, in float2 vUV : TEXCOORD0, nointerpolation in uint vLayer : LAYER,
    in float3 vViewPos : TEXCOORD1, nointerpolation in float3 vNormal : TEXCOORD2, nointerpolation in float3 vSunDir : TEXCOORD3, in float2 vLattice : TEXCOORD4, in float2 vEdge : TEXCOORD5) : SV_TARGET {
    const float2 st = vUV * 64.0f - 0.5f;
    const int2 p0 = int2(floor(st));
    const float2 t = st - float2(p0);
    float3 tex = lerp(lerp(texel(vLayer, p0), texel(vLayer, p0 + int2(1, 0)), t.x), lerp(texel(vLayer, p0 + int2(0, 1)), texel(vLayer, p0 + int2(1, 1)), t.x), t.y);

#ifdef RS64_TERRAIN_PACKS

    // A texture pack's replacement for this layer, once the texture cache has streamed it in.
    const float2 uvDx = ddx(vUV), uvDy = ddy(vUV);
    const uint packStart = uint(gFrame[2].w);
    const float packIndex = (packStart > 0) ? gFrame[packStart + vLayer / 4][vLayer % 4] : -1.0f;
    if (packIndex >= 0.0f) {
        tex = gTextures[NonUniformResourceIndex(uint(packIndex))].SampleGrad(gSampler, vUV, uvDx, uvDy).rgb;
    }

#endif

    if (gTerrain.debug == 1) {
        return float4(vShade.rgb, 1.0f);
    }

    float3 color = tex * vShade.rgb;
    float alpha = 1.0f;
    float fog = vShade.a;
    if (gTerrain.water) {
        const float3 v = normalize(-vViewPos);
        float3 n = normalize(vNormal);
        float3 l = normalize(vSunDir);

        // Seen from below the surface's open-air side: flip both, so a sign slip cannot hide the effect.
        if (dot(n, v) < 0.0f) {
            n = -n;
            l = -l;
        }

        const float fresnel = pow(1.0f - saturate(dot(n, v)), 5.0f);
        const float glint = pow(saturate(dot(reflect(-l, n), v)), kWaterGlintPower);

        // Past the map edge (vEdge.x): opaque, the water look as far as its edge cell has it (vEdge.y), and fog that adds a fade by distance past the edge
        // and (from two cells out) toward the far plane, so it meets the map with the same fog. In the map: the shoreline fade and the game's fog.
        const bool outside = (vEdge.x > 0.5f);
        const float waterLook = outside ? vEdge.y : 1.0f;
        color = lerp(color, lerp(color, gTerrain.fogColor.rgb, fresnel * kWaterFresnel) + (gFrame[1].rgb / 255.0f) * glint * kWaterGlint, waterLook);
        alpha = outside ? 1.0f : waterShoreAlpha(vLattice);
        fog = terrainFog(vViewPos);
        if (outside) {
            const float2 cells = vLattice * 0.25f;
            const float2 past = max(max(-cells, cells - float2(gTerrain.mapWidth, gTerrain.mapHeight)), 0.0f);
            const float d = max(past.x, past.y);
            fog = max(fog, max(smoothstep(0.0f, kEdgeFogCells, d), farPlaneFade(vViewPos) * saturate(d * 0.5f)));
        }
    }

    if (gTerrain.debug == 2) {
        return float4(fog, gTerrain.water ? alpha : 0.0f, (gTerrain.water && (vEdge.x > 0.5f)) ? 1.0f : 0.0f, 1.0f);
    }

    return float4(lerp(color, gTerrain.fogColor.rgb, fog), alpha);
}
