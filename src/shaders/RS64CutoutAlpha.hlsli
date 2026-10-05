//
// RT64
//

#pragma once

// Sets 0-2 are RT64's raster common and bindless texture sets (TextureSampler.hlsli); RS64 traced resources live in set 3.
#include "TextureSampler.hlsli"
#include "shared/rt64_rs64_cutout.h"

RaytracingAccelerationStructure gScene : register(t3, space3);
StructuredBuffer<RS64CutoutDraw> gCutoutDraws : register(t4, space3);
ByteAddressBuffer gCutoutIndices : register(t5, space3);
ByteAddressBuffer gTexCoords : register(t6, space3);
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

// Texel-0 alpha at the hit against the draw's alpha-compare threshold (mirrors rs64lights::cutoutThreshold); untextured cutouts block.
bool rs64CutoutBlocks(uint prim, float2 bary, uint count) {
    const RS64CutoutDraw d = gCutoutDraws[rs64CutoutDrawFor(prim, count)];
    const RenderParams rp = DynamicRenderParams[d.instanceIndex];
    if (!renderFlagUsesTexture0(rp.flags)) {
        return true;
    }
    const uint i0 = gCutoutIndices.Load((prim * 3 + 0) * 4);
    const uint i1 = gCutoutIndices.Load((prim * 3 + 1) * 4);
    const uint i2 = gCutoutIndices.Load((prim * 3 + 2) * 4);
    const float2 uv = rs64TexCoord(i0) * (1.0f - bary.x - bary.y) + rs64TexCoord(i1) * bary.x + rs64TexCoord(i2) * bary.y;
    const OtherMode otherMode = { rp.omL, rp.omH };
    RDPTile rdpTile = RDPTiles[d.tileIndex];
    if (!renderFlagDynamicTiles(rp.flags)) {
        rdpTile.cms = renderCMS0(rp.flags);
        rdpTile.cmt = renderCMT0(rp.flags);
        rdpTile.nativeSampler = renderFlagNativeSampler0(rp.flags);
    }
    const GPUTile gpuTile = GPUTiles[d.tileIndex];
    // Mirrors rs64lights::cutoutMipGradient: gradients that land on mip 0 (zero gradients give log2(0) = NaN on mipmapped replacements).
    const float2 g = 1.18920712f / gpuTile.tcScale;
    const float4 texel = sampleTexture(otherMode, rp.flags, uv, float2(g.x, 0.0f), float2(0.0f, g.y), rdpTile, gpuTile, false);
    const float threshold = (otherMode.alphaCompare() == G_AC_THRESHOLD) ? instanceRDPParams[d.instanceIndex].blendColor.a : 0.5f;
    return texel.a >= threshold;
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

// Mirrors rs64lights::hitColorFor; rgb = hit colour, a = 1 when emissive. hitParams: x hit count, y emissive count, z terrain colour, w caster average
// (cutout and history hits use the average).
float4 rs64HitColor(uint instanceId, uint prim, uint4 hitParams) {
    if ((instanceId == 0) && (hitParams.x > 0)) {
        return rs64UnpackHit(gHitColors[rs64HitEntry(prim, hitParams.x, false)].y);
    }
    if ((instanceId == 7) && (hitParams.y > 0)) {
        return rs64UnpackHit(gEmissiveColors[rs64HitEntry(prim, hitParams.y, true)].y);
    }
    return rs64UnpackHit((instanceId == 1) ? hitParams.z : hitParams.w);
}

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
bool rs64SunHit(RayDesc ray, uint cutoutCount, out float hitT, out uint instanceId) {
    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> q;
    q.TraceRayInline(gScene, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, (cutoutCount > 0) ? 0x07 : 0x03, ray);
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
