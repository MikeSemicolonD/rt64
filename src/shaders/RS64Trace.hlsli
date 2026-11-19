// RT64 - Ray-query helpers shared by the RS64 traced passes: bindings, cutout alpha tests, hit colours and sun rays

#pragma once

// Sets 0-2 are RT64's raster common and bindless texture sets (TextureSampler.hlsli); RS64 traced resources live in set 3.
#include "TextureSampler.hlsli"
#include "shared/rt64_color_combiner.h"
#include "shared/rt64_rs64_cutout.h"

RaytracingAccelerationStructure gScene : register(t3, space3);
StructuredBuffer<RS64CutoutDraw> gCutoutDraws : register(t4, space3);
ByteAddressBuffer gCutoutIndices : register(t5, space3);
ByteAddressBuffer gTexCoords : register(t6, space3);
// RSPProcessCS's shaded vertex colours (float4 per vertex, the fog in alpha when the draw fogs).
ByteAddressBuffer gShadedCols : register(t13, space3);
// Hit colour tables (rs64lights::HitColorEntry: first BLAS triangle, rgba8; a = 255 emissive) for the opaque casters (instance 0) and emissive draws (7).
StructuredBuffer<uint2> gHitColors : register(t9, space3);
StructuredBuffer<uint2> gEmissiveColors : register(t10, space3);

// Mirrors rs64lights::cutoutDrawFor.
uint rs64CutoutDrawFor(uint prim, uint count) {
    if (count == 0) {
        return 0;
    }
    uint lo = 0;
    uint hi = count - 1;
    while (lo < hi) {
        const uint mid = (lo + hi + 1) / 2;
        if (gCutoutDraws[mid].firstTriangle <= prim) {
            lo = mid;
        }
        else {
            hi = mid - 1;
        }
    }
    return lo;
}

float2 rs64TexCoord(uint vertexIndex) {
    return asfloat(gTexCoords.Load2(vertexIndex * 8));
}

// Translucent cutouts (flags bit 0: foot beams, foam) are hits only while this is set, for water reflection rays; they never cast.
static bool gRs64TranslucentRays = false;
// A translucent draw blocks a water reflection ray from this combined alpha.
static const float kTranslucentAlphaMin = 0.2f;

// Texels 0 and 1 (the draw's next tile; texel 0 again unless both is set) at a cutout hit, white for a texture the draw does not use. One sampler
// in a loop: every inlined copy of sampleTexture in the ray query loops grew the shader until DXC overflowed.
void rs64CutoutTexels(RS64CutoutDraw d, uint prim, float2 bary, bool both, out float4 texel0, out float4 texel1) {
    const RenderParams rp = DynamicRenderParams[d.instanceIndex];
    const uint i0 = gCutoutIndices.Load((prim * 3 + 0) * 4);
    const uint i1 = gCutoutIndices.Load((prim * 3 + 1) * 4);
    const uint i2 = gCutoutIndices.Load((prim * 3 + 2) * 4);
    const float2 uv = rs64TexCoord(i0) * (1.0f - bary.x - bary.y) + rs64TexCoord(i1) * bary.x + rs64TexCoord(i2) * bary.y;
    const OtherMode otherMode = { rp.omL, rp.omH };
    float4 texels[2] = { float4(1.0f, 1.0f, 1.0f, 1.0f), float4(1.0f, 1.0f, 1.0f, 1.0f) };
    [loop]
    for (uint k = 0; k < (both ? 2u : 1u); k++) {
        const bool second = (k == 1);
        if (!(second ? renderFlagUsesTexture1(rp.flags) : renderFlagUsesTexture0(rp.flags))) {
            continue;
        }
        RDPTile rdpTile = RDPTiles[d.tileIndex + k];
        if (!renderFlagDynamicTiles(rp.flags)) {
            rdpTile.cms = second ? renderCMS1(rp.flags) : renderCMS0(rp.flags);
            rdpTile.cmt = second ? renderCMT1(rp.flags) : renderCMT0(rp.flags);
            rdpTile.nativeSampler = second ? renderFlagNativeSampler1(rp.flags) : renderFlagNativeSampler0(rp.flags);
        }
        const GPUTile gpuTile = GPUTiles[d.tileIndex + k];
        // Mirrors rs64lights::cutoutMipGradient: gradients that land on mip 0 (zero gradients give log2(0) = NaN on mipmapped replacements).
        const float2 g = 1.18920712f / gpuTile.tcScale;
        texels[k] = sampleTexture(otherMode, rp.flags, uv, float2(g.x, 0.0f), float2(0.0f, g.y), rdpTile, gpuTile, false);
    }
    texel0 = texels[0];
    texel1 = both ? texels[1] : texels[0];
}

// A translucent cutout's colour and opacity at a hit as its colour combiner makes them, from both textures (texel 0 again in 1-cycle, as RasterPS)
// and the draw's average vertex colour as shade (flags bits 8-31 rgb, 2-7 alpha). Effect sprites take their shape from a second tile's alpha mask
// over an opaque colour tile: texel 0 alone reflected them as solid dark rectangles.
bool rs64TwoTexels(RS64CutoutDraw d) {
    const OtherMode otherMode = { DynamicRenderParams[d.instanceIndex].omL, DynamicRenderParams[d.instanceIndex].omH };
    return ((d.flags & 3u) != 0u) && (otherMode.cycleType() != G_CYC_1CYCLE);
}

// The draw's average vertex colour from its cutout flags.
float4 rs64CutoutShade(RS64CutoutDraw d) {
    return float4(float3((d.flags >> 8) & 0xFFu, (d.flags >> 16) & 0xFFu, (d.flags >> 24) & 0xFFu) / 255.0f, float((d.flags >> 2) & 0x3Fu) / 63.0f);
}

float4 rs64TranslucentCombined(RS64CutoutDraw d, float4 texel0, float4 texel1, float4 shade) {
    const RenderParams rp = DynamicRenderParams[d.instanceIndex];
    const ColorCombiner colorCombiner = { rp.ccL, rp.ccH };
    const OtherMode otherMode = { rp.omL, rp.omH };
    ColorCombiner::Inputs ccInputs;
    ccInputs.otherMode = otherMode;
    ccInputs.alphaOnly = false;
    ccInputs.texVal0 = texel0;
    ccInputs.texVal1 = texel1;
    ccInputs.primColor = instanceRDPParams[d.instanceIndex].primColor;
    ccInputs.shadeColor = shade;
    ccInputs.envColor = instanceRDPParams[d.instanceIndex].envColor;
    ccInputs.keyCenter = instanceRDPParams[d.instanceIndex].keyCenter;
    ccInputs.keyScale = instanceRDPParams[d.instanceIndex].keyScale;
    ccInputs.lodFraction = 0.0f;
    ccInputs.primLodFrac = instanceRDPParams[d.instanceIndex].primLOD.x;
    ccInputs.noise = 0.5f;
    ccInputs.K4 = (instanceRDPParams[d.instanceIndex].convertK[4] / 255.0f);
    ccInputs.K5 = (instanceRDPParams[d.instanceIndex].convertK[5] / 255.0f);
    float4 combined;
    float alphaCompare;
    colorCombiner.run(ccInputs, combined, alphaCompare);
    return combined;
}

// Texel-0 alpha at the hit against the draw's alpha-compare threshold (mirrors rs64lights::cutoutThreshold); untextured cutouts block.
// A translucent cutout tests its combined alpha instead.
bool rs64CutoutBlocks(uint prim, float2 bary, uint count) {
    const RS64CutoutDraw d = gCutoutDraws[rs64CutoutDrawFor(prim, count)];
    const bool translucent = ((d.flags & 1u) != 0u);
    if ((translucent && !gRs64TranslucentRays) || ((d.flags & 2u) != 0u)) {
        return false;
    }

    float4 texel0;
    float4 texel1;
    rs64CutoutTexels(d, prim, bary, rs64TwoTexels(d), texel0, texel1);
    if (translucent) {
        return rs64TranslucentCombined(d, texel0, texel1, rs64CutoutShade(d)).a >= kTranslucentAlphaMin;
    }

    const OtherMode otherMode = { DynamicRenderParams[d.instanceIndex].omL, DynamicRenderParams[d.instanceIndex].omH };
    const float threshold = (otherMode.alphaCompare() == G_AC_THRESHOLD) ? instanceRDPParams[d.instanceIndex].blendColor.a : 0.5f;
    return texel0.a >= threshold;
}

// A translucent cutout hit's combined colour and opacity; a = 0 otherwise.
float4 rs64TranslucentHit(uint instanceId, uint prim, float2 bary, uint count) {
    if ((instanceId != 2) || (count == 0)) {
        return float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    const RS64CutoutDraw d = gCutoutDraws[rs64CutoutDrawFor(prim, count)];
    if ((d.flags & 1u) == 0u) {
        return float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    float4 texel0;
    float4 texel1;
    rs64CutoutTexels(d, prim, bary, rs64TwoTexels(d), texel0, texel1);
    const float4 c = rs64TranslucentCombined(d, texel0, texel1, rs64CutoutShade(d));
    return float4(saturate(c.rgb), max(saturate(c.a), 1.0e-3f));
}

// The sky dome in camera-space direction d as the game textures it: a ray from the camera meets the dome's traced triangles (cutouts flagged 2).
// Only the flag is tested in the query loop and the texture is sampled once, at the hit.
bool rs64TracedSky(float3 d, uint count, out float3 rgb) {
    rgb = float3(0.0f, 0.0f, 0.0f);
    if (count == 0) {
        return false;
    }

    RayDesc r;
    r.Origin = float3(0.0f, 0.0f, 0.0f);
    r.Direction = d;
    r.TMin = 0.0f;
    r.TMax = 1.0e30f;
    RayQuery<RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> q;
    q.TraceRayInline(gScene, RAY_FLAG_NONE, 0x04, r);
    while (q.Proceed()) {
        if ((q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE) && ((gCutoutDraws[rs64CutoutDrawFor(q.CandidatePrimitiveIndex(), count)].flags & 2u) != 0u)) {
            q.CommitNonOpaqueTriangleHit();
        }
    }
    if (q.CommittedStatus() != COMMITTED_TRIANGLE_HIT) {
        return false;
    }

    const uint prim = q.CommittedPrimitiveIndex();
    const float2 bary = q.CommittedTriangleBarycentrics();
    const RS64CutoutDraw sky = gCutoutDraws[rs64CutoutDrawFor(prim, count)];
    float4 texel0;
    float4 texel1;
    rs64CutoutTexels(sky, prim, bary, rs64TwoTexels(sky), texel0, texel1);

    // The hit's own shaded colour: the dome's vertex colours carry its horizon haze, which one average for the draw lost.
    const float4 c0 = asfloat(gShadedCols.Load4(gCutoutIndices.Load((prim * 3 + 0) * 4) * 16));
    const float4 c1 = asfloat(gShadedCols.Load4(gCutoutIndices.Load((prim * 3 + 1) * 4) * 16));
    const float4 c2 = asfloat(gShadedCols.Load4(gCutoutIndices.Load((prim * 3 + 2) * 4) * 16));
    const float4 shade = saturate(c0 * (1.0f - bary.x - bary.y) + c1 * bary.x + c2 * bary.y);
    rgb = saturate(rs64TranslucentCombined(sky, texel0, texel1, shade).rgb);

    // The blender's fog (first cycle fog colour x shade alpha), as the game draws the dome.
    const uint blender = DynamicRenderParams[sky.instanceIndex].omL >> 16;
    if ((((blender >> 14) & 3u) == 3u) && (((blender >> 10) & 3u) == 2u)) {
        rgb = lerp(rgb, instanceRDPParams[sky.instanceIndex].fogColor.rgb, shade.a);
    }
    return true;
}

// TLAS instance id of the last committed caster hit (2 = cutout BLAS); read by the shadow pass's debug mode 3.
static uint gRs64HitInstance = 0;

// Drawn casters (0x01) plus cutouts (0x04, alpha-tested); the first committed hit ends the search.
bool rs64CasterHit(RayDesc ray, uint count, out float hitT) {
    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> q;
    q.TraceRayInline(gScene, RAY_FLAG_NONE, (count > 0) ? 0x05 : 0x01, ray);
    while (q.Proceed()) {
        if ((q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE) && rs64CutoutBlocks(q.CandidatePrimitiveIndex(), q.CandidateTriangleBarycentrics(), count)) {
            q.CommitNonOpaqueTriangleHit();
        }
    }
    const bool hit = (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT);
    hitT = hit ? q.CommittedRayT() : ray.TMax;
    gRs64HitInstance = hit ? q.CommittedInstanceID() : 0;
    return hit;
}

// a = 1 emissive (tag 255), -1 floor (tag 253, rs64lights::markFloor), 0 otherwise.
float4 rs64UnpackHit(uint rgba) {
    const uint tag = rgba >> 24;
    return float4(float(rgba & 0xFFu), float((rgba >> 8) & 0xFFu), float((rgba >> 16) & 0xFFu), 0.0f) / 255.0f + float4(0.0f, 0.0f, 0.0f, (tag == 255u) ? 1.0f : ((tag == 253u) ? -1.0f : 0.0f));
}

uint rs64HitEntry(uint prim, uint count, bool emissive) {
    uint lo = 0;
    uint hi = count - 1;
    while (lo < hi) {
        const uint mid = (lo + hi + 1) / 2;
        const uint first = emissive ? gEmissiveColors[mid].x : gHitColors[mid].x;
        if (first <= prim) {
            lo = mid;
        }
        else {
            hi = mid - 1;
        }
    }
    return lo;
}

// Static terrain hit colours (rs64lights::terrainHitColors): header (triangles per ground cell, first water triangle or 0, ground cells, drawn water
// surface y in map space), then rgba8 per ground cell and per water quad. A zero header (the placeholder) means no table.
ByteAddressBuffer gTerrainColors : register(t11, space3);

bool rs64TerrainCellColor(uint prim, out float3 rgb) {
    rgb = float3(0.0f, 0.0f, 0.0f);
    const uint per = gTerrainColors.Load(0);
    if (per == 0) {
        return false;
    }

    const uint waterPrim = gTerrainColors.Load(4);
    const uint slot = ((waterPrim > 0) && (prim >= waterPrim)) ? (gTerrainColors.Load(8) + (prim - waterPrim) / 2) : (prim / per);
    rgb = rs64UnpackHit(gTerrainColors.Load(16 + slot * 4)).rgb;
    return true;
}

// How much of the drawn water covers a static terrain hit at map-space height y (heights grow downward): its shore fade over the lift, as
// waterShoreAlpha in RS64Terrain.hlsli; 0 for ground above the drawn surface.
float rs64WaterCover(float y, float lift) {
    return (gTerrainColors.Load(4) > 0) ? smoothstep(0.0f, lift, y - asfloat(gTerrainColors.Load(12))) : 0.0f;
}

// Mirrors rs64lights::hitColorFor; rgb = hit colour, a = 1 when emissive. hitParams: x hit count, y emissive count, z terrain colour, w caster average
// (cutout and history hits use the average); static terrain hits take their cell's colour when the table is bound.
float4 rs64HitColor(uint instanceId, uint prim, uint4 hitParams) {
    float3 cell;
    if ((instanceId == 1) && rs64TerrainCellColor(prim, cell)) {
        return float4(cell, 0.0f);
    }
    if ((instanceId == 0) && (hitParams.x > 0)) {
        return rs64UnpackHit(gHitColors[rs64HitEntry(prim, hitParams.x, false)].y);
    }
    if ((instanceId == 7) && (hitParams.y > 0)) {
        return rs64UnpackHit(gEmissiveColors[rs64HitEntry(prim, hitParams.y, true)].y);
    }
    return rs64UnpackHit((instanceId == 1) ? hitParams.z : hitParams.w);
}

// Barycentrics of the last rs64ClosestHit hit, for sampling a translucent cutout there.
static float2 gRs64HitBary = float2(0.0f, 0.0f);

// Closest hit for bounce and reflection rays: casters, terrain (back faces culled), cutouts (alpha-tested) and emissive draws.
bool rs64ClosestHit(RayDesc ray, uint cutoutCount, out float hitT, out uint instanceId, out uint prim) {
    RayQuery<RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> q;
    q.TraceRayInline(gScene, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, (cutoutCount > 0) ? 0x87 : 0x83, ray);
    while (q.Proceed()) {
        if ((q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE) && rs64CutoutBlocks(q.CandidatePrimitiveIndex(), q.CandidateTriangleBarycentrics(), cutoutCount)) {
            q.CommitNonOpaqueTriangleHit();
        }
    }
    const bool hit = (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT);
    hitT = hit ? q.CommittedRayT() : ray.TMax;
    instanceId = hit ? q.CommittedInstanceID() : 0;
    prim = hit ? q.CommittedPrimitiveIndex() : 0;
    gRs64HitBary = hit ? q.CommittedTriangleBarycentrics() : float2(0.0f, 0.0f);
    return hit;
}

// Mirrors rs64lights::schlick.
float rs64Schlick(float cosTheta, float f0) {
    const float m = 1.0f - saturate(cosTheta);
    return f0 + (1.0f - f0) * m * m * m * m * m;
}

// Mirrors rs64lights::roughReflectDir: mirror of v about n, offset like sunDiskDir ray 1 of 2 inside a cone of tan tanCone.
float3 rs64RoughReflect(float3 v, float3 n, float tanCone, float rot) {
    const float3 m = normalize(v - 2.0f * dot(v, n) * n);
    const float3 seed = (abs(m.x) < 0.9f) ? float3(1.0f, 0.0f, 0.0f) : float3(0.0f, 1.0f, 0.0f);
    const float3 t = normalize(cross(m, seed));
    const float3 b = cross(m, t);
    const float r = sqrt(1.5f / 2.0f) * tanCone;
    const float th = 2.39996323f + 6.28318531f * rot;
    return normalize(m + (cos(th) * r) * t + (sin(th) * r) * b);
}

// Mirrors rs64lights::cosineHemisphereDir.
float3 rs64CosineDir(float3 n, uint i, uint count, float rot) {
    const float3 seed = (abs(n.x) < 0.9f) ? float3(1.0f, 0.0f, 0.0f) : float3(0.0f, 1.0f, 0.0f);
    const float3 t = normalize(cross(n, seed));
    const float3 b = cross(n, t);
    const float r = sqrt((float(i) + 0.5f) / float(max(count, 1u)));
    const float th = 2.39996323f * float(i) + 6.28318531f * rot;
    return normalize(t * (cos(th) * r) + b * (sin(th) * r) + n * sqrt(max(0.0f, 1.0f - r * r)));
}

// Sun/light visibility against drawn casters (0x01), static terrain (0x02) and cutouts (0x04, alpha-tested) in one traversal. Back faces are culled
// for terrain only (the other instances disable culling): a receiver below the static surface sees its underside. instanceId: 0 drawn, 1 terrain, 2 cutout.
bool rs64SunHitMasked(RayDesc ray, uint cutoutCount, uint mask, out float hitT, out uint instanceId) {
    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> q;
    q.TraceRayInline(gScene, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, mask & ((cutoutCount > 0) ? 0x07u : 0x03u), ray);
    while (q.Proceed()) {
        if ((q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE) && rs64CutoutBlocks(q.CandidatePrimitiveIndex(), q.CandidateTriangleBarycentrics(), cutoutCount)) {
            q.CommitNonOpaqueTriangleHit();
        }
    }
    const bool hit = (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT);
    hitT = hit ? q.CommittedRayT() : ray.TMax;
    instanceId = hit ? q.CommittedInstanceID() : 0;
    gRs64HitInstance = instanceId;
    return hit;
}

bool rs64SunHit(RayDesc ray, uint cutoutCount, out float hitT, out uint instanceId) {
    return rs64SunHitMasked(ray, cutoutCount, 0xFFu, hitT, instanceId);
}

// For a static terrain receiver: the terrain itself blocks only from terrainSkip along the ray. The coarse heightfield's own bumps a few metres off
// shadowed it in saw-tooth spikes under a low sun; mountains across a valley and every other caster still shadow it.
bool rs64TerrainBlocksPastSkip(RayDesc ray, uint cutoutCount, float terrainSkip, out float hitT, out uint instanceId) {
    RayQuery<RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES | RAY_FLAG_FORCE_OPAQUE> q;
    q.TraceRayInline(gScene, RAY_FLAG_NONE, 0x02u, ray);
    while (q.Proceed()) {
    }

    hitT = ray.TMax;
    instanceId = 0;
    if (q.CommittedStatus() != COMMITTED_TRIANGLE_HIT) {
        return false;
    }

    const float t = q.CommittedRayT();
    instanceId = q.CommittedInstanceID();
    if (q.CommittedTriangleFrontFace()) {
        hitT = t;
        return true;
    }

    // Leaving through a back face means the ray entered the terrain inside the skip. Underground for another skip length is a hill, not a bump it skimmed.
    if ((t - ray.TMin) >= terrainSkip) {
        hitT = ray.TMin;
        return true;
    }

    RayDesc past = ray;
    past.TMin = t;
    return rs64SunHitMasked(past, cutoutCount, 0x02u, hitT, instanceId);
}

bool rs64SunHitTerrainReceiver(RayDesc ray, uint cutoutCount, float terrainSkip, out float hitT, out uint instanceId) {
    float t0;
    uint id0;
    const bool other = rs64SunHitMasked(ray, cutoutCount, 0x05u, t0, id0);
    RayDesc distant = ray;
    distant.TMin = max(ray.TMin, terrainSkip);
    float t1 = ray.TMax;
    uint id1 = 0;
    const bool terrain = (distant.TMin < distant.TMax) && rs64TerrainBlocksPastSkip(distant, cutoutCount, terrainSkip, t1, id1);
    hitT = other ? (terrain ? min(t0, t1) : t0) : t1;
    instanceId = (other && (!terrain || (t0 <= t1))) ? id0 : id1;
    gRs64HitInstance = instanceId;
    return other || terrain;
}
