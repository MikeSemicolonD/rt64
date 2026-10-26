//
// RT64
//

#include "shared/rt64_rs64_terrain_cb.h"
#include "shared/rt64_rsp_fog.h"
#include "shared/rt64_rsp_viewport.h"

[[vk::push_constant]] ConstantBuffer<RS64TerrainCB> gTerrain : register(b0, space0);
StructuredBuffer<float4x4> gWorldTransforms : register(t1, space0);
StructuredBuffer<float4x4> gViewProjTransforms : register(t2, space0);
StructuredBuffer<RSPViewport> gViewports : register(t3, space0);
StructuredBuffer<RSPFog> gFog : register(t4, space0);
// Lattice of (4 * width + 1) x (4 * height + 1) samples: heights in record units (s8 << 4) and packed RGBA8 vertex colours.
StructuredBuffer<int> gHeights : register(t5, space0);
StructuredBuffer<uint> gColors : register(t6, space0);
// Per cell: its texture layer. Texels: 4096 packed RGBA8 per layer (64 x 64).
StructuredBuffer<uint> gCellLayers : register(t7, space0);
StructuredBuffer<uint> gTexels : register(t8, space0);
// Per frame: [0] ambient rgb, cell scale; [1] sun rgb, water level (s8 units); [2] sun direction, w = where the texture pack block starts (0 none);
// [3..] each texture layer's current layer (animated textures' frame), 4 per float4; then each layer's pack replacement (texture cache index, -1 none).
StructuredBuffer<float4> gFrame : register(t9, space0);
// Per frame: [0, chunkCount) each chunk's LOD (0xFF = culled), then the chunk lists of the per-LOD draws.
StructuredBuffer<uint> gChunks : register(t10, space0);
SamplerState gSampler : register(s11, space0);

#ifdef RS64_TERRAIN_PACKS
// Set 1: the raster's bindless texture cache, for texture pack replacements.
Texture2D<float4> gTextures[8192] : register(t0, space1);
#endif

float4 unpackRGBA8(uint c) {
    return float4(c & 0xFF, (c >> 8) & 0xFF, (c >> 16) & 0xFF, (c >> 24) & 0xFF) / 255.0f;
}

// Height of a lattice sample in record units: the ground (the seabed under water; the water is its own flat mesh).
float latticeHeight(int2 p, uint latticeW, uint latticeH) {
    p = clamp(p, int2(0, 0), int2(latticeW - 1, latticeH - 1));
    return float(gHeights[p.y * latticeW + p.x]);
}

float4 catmullRomWeights(float t) {
    const float t2 = t * t, t3 = t2 * t;
    return float4(-0.5f * t3 + t2 - 0.5f * t, 1.5f * t3 - 2.5f * t2 + 1.0f, -1.5f * t3 + 2.0f * t2 + 0.5f * t, 0.5f * t3 - 0.5f * t2);
}

// Catmull-Rom through the lattice samples: passes through every sample, no creases along the sample lines.
float smoothHeight(int2 i0, float2 t, uint latticeW, uint latticeH) {
    const float4 wx = catmullRomWeights(t.x), wy = catmullRomWeights(t.y);
    float h = 0.0f;
    for (int y = 0; y < 4; y++) {
        const float row = wx.x * latticeHeight(i0 + int2(-1, y - 1), latticeW, latticeH) + wx.y * latticeHeight(i0 + int2(0, y - 1), latticeW, latticeH)
                        + wx.z * latticeHeight(i0 + int2(1, y - 1), latticeW, latticeH) + wx.w * latticeHeight(i0 + int2(2, y - 1), latticeW, latticeH);
        h += wy[y] * row;
    }

    return h;
}

// Surface height at a lattice coordinate (4 samples per cell).
float surfaceHeight(float2 f, uint latticeW, uint latticeH) {
    const int2 i0 = int2(min(uint2(max(f, 0.0f)), uint2(latticeW - 2, latticeH - 2)));
    const float2 t = f - float2(i0);
    if (gTerrain.smooth) {
        return smoothHeight(i0, t, latticeW, latticeH);
    }

    return lerp(lerp(latticeHeight(i0, latticeW, latticeH), latticeHeight(i0 + int2(1, 0), latticeW, latticeH), t.x),
                lerp(latticeHeight(i0 + int2(0, 1), latticeW, latticeH), latticeHeight(i0 + int2(1, 1), latticeW, latticeH), t.x), t.y);
}

// The blender's fog at camera-space point v (RSPProcessCS vertex fog).
float terrainFog(float3 v) {
    if (gTerrain.fogIndex == 0) {
        return 0.0f;
    }

    const RSPFog fog = gFog[gTerrain.fogIndex - 1];
    const float4x4 proj = gViewProjTransforms[gTerrain.viewProjIndex];
    const float4 clip = mul(proj, float4(v, 1.0f));
    return (clip.w != 0.0f) ? saturate(((max(clip.z, 0) / clip.w) * fog.mul + fog.offset) / 255.0f) : 0.0f;
}

// The drawn water's lift over the water level (record units; heights grow downward). Mirrors rs64lights::TerrainWaterLift.
static const float kWaterLift = 16.0f;

// The drawn water's opacity at lattice coordinate f: opaque over ground at or under the water level (an ocean level's flat sea is often ground at
// the level under a wave texture), fading out as the ground rises to the lifted surface, so a shore meets the water with no hard line.
float waterShoreAlpha(float2 f) {
    const uint latticeW = gTerrain.mapWidth * 4 + 1, latticeH = gTerrain.mapHeight * 4 + 1;
    return smoothstep(-kWaterLift, 0.0f, surfaceHeight(clamp(f, 0.0f, float2(latticeW - 1, latticeH - 1)), latticeW, latticeH) - gFrame[1].w * 16.0f);
}

// 0 to 1 over the second half of the way to the far plane along camera-space point v's view ray (s v reaches clip z = w at sFar).
float farPlaneFade(float3 v) {
    const float4x4 proj = gViewProjTransforms[gTerrain.viewProjIndex];
    const float denom = dot(proj[2].xyz - proj[3].xyz, v);
    const float sFar = (denom > 0.0f) ? (proj[3].w - proj[2].w) / denom : 0.0f;
    return (sFar > 0.0f) ? smoothstep(0.5f, 1.0f, 1.0f / sFar) : 0.0f;
}
