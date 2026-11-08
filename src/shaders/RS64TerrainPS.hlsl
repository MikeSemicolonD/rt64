// RT64 - Whole-map terrain texturing and shading

#include "RS64Terrain.hlsli"
#include "RS64WaterRipple.hlsli"

float3 texel(uint layer, int2 p) {
    p = clamp(p, 0, 63);
    return unpackRGBA8(gTexels[layer * 4096 + p.y * 64 + p.x]).rgb;
}

float3 texelWrapped(uint layer, int2 p) {
    p &= 63;
    return unpackRGBA8(gTexels[layer * 4096 + p.y * 64 + p.x]).rgb;
}

// The layer bilinear at map point m (map units, 8 per base texel) repeated `repeat` times per cell.
float3 texelRepeated(uint layer, float2 m, float repeat) {
    const float2 s = m * (repeat / 8.0f) - 0.5f;
    const int2 p0 = int2(floor(s));
    const float2 t = s - float2(p0);
    return lerp(lerp(texelWrapped(layer, p0), texelWrapped(layer, p0 + int2(1, 0)), t.x), lerp(texelWrapped(layer, p0 + int2(0, 1)), texelWrapped(layer, p0 + int2(1, 1)), t.x), t.y);
}

// Water detail where the base texture is magnified (up close): the layer again at a power-of-2 repeat whose texels stay near one a pixel, blended
// between neighbouring repeats so no band shows. Its difference from the base is added at up to kWaterDetail (over 1: the textures are low contrast).
static const float kWaterDetail = 2.5f;
static const float kWaterDetailMaxLod = 4.0f;

// Water strength of the view-angle tint toward the sky (fog colour), and of the sun glint: a sharp sparkle lobe over a broad sheen.
static const float kWaterFresnel = 0.6f;
static const float kWaterGlint = 1.2f;
static const float kWaterGlintPower = 320.0f;
static const float kWaterSheen = 0.2f;
static const float kWaterSheenPower = 32.0f;

// Cells past the map edge over which the water there fades fully into the fog.
static const float kEdgeFogCells = 16.0f;

// With fogClip, the fog amount from which land fades out into the sky behind it, gone at full fog.
static const float kFogClipFadeStart = 0.75f;

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
        // Lattice steps are a quarter cell (128 map units).
        const float2 m = vLattice * 128.0f;
        const float footprint = max(length(ddx(m)), length(ddy(m)));

        // lod = log2 of the base magnification (8 map units a texel); repeats 2^(k-1) and 2^k keep detail texels at half to one a pixel.
        const float lod = min(log2(8.0f / max(footprint, 1e-3f)), kWaterDetailMaxLod);
        const float detailWeight = kWaterDetail * saturate(lod);
        if (detailWeight > 0.0f) {
            const float k = floor(lod);
            const float3 detail = lerp(texelRepeated(vLayer, m, exp2(k - 1.0f)), texelRepeated(vLayer, m, exp2(k)), lod - k);
            color = saturate(tex + (detail - tex) * detailWeight) * vShade.rgb;        }

        const float3 nMap = rs64RippleNormal(m, gTerrain.rippleTime, footprint, gTerrain.rippleStrength);
        float3 n = normalize(mul((float3x3)gWorldTransforms[gTerrain.worldIndex], nMap));
        float3 l = normalize(vSunDir);

        // Seen from below the surface's open-air side: flip both, so a sign slip cannot hide the effect.
        if (dot(n, v) < 0.0f) {
            n = -n;
            l = -l;
        }

        const float fresnel = pow(1.0f - saturate(dot(n, v)), 5.0f);
        const float rv = saturate(dot(reflect(-l, n), v));
        const float glint = pow(rv, kWaterGlintPower) + (kWaterSheen / kWaterGlint) * pow(rv, kWaterSheenPower);

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
    else if (vEdge.y > 0.0f) {
        // Ground the drawn water fades out over (a seabed bump just under the surface) takes the water's view tint, or it showed through as a dark patch.
        const float3 v = normalize(-vViewPos);
        float3 n = normalize(vNormal);
        if (dot(n, v) < 0.0f) {
            n = -n;
        }

        const float fresnel = pow(1.0f - saturate(dot(n, v)), 5.0f);
        color = lerp(color, gTerrain.fogColor.rgb, fresnel * kWaterFresnel * vEdge.y);
    }

    if (gTerrain.fogClip) {
        alpha *= 1.0f - smoothstep(kFogClipFadeStart, 1.0f, fog);
        if (alpha <= 0.0f) {
            discard;
        }
    }

    if (gTerrain.debug == 2) {
        return float4(fog, gTerrain.water ? alpha : 0.0f, (gTerrain.water && (vEdge.x > 0.5f)) ? 1.0f : 0.0f, 1.0f);
    }

    return float4(lerp(color, gTerrain.fogColor.rgb, fog), alpha);
}
