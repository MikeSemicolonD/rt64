// RT64 - Browser test of RS64SoftRayQuery: traces gRays against the bound scene and writes each closest hit to gOut.
// Ray (12 words): origin, tmin, direction, tmax, flags, mask, mode (1: commit only even candidates), count (ray 0 only).
// Hit (8 words): status, t, primitive, instance id, barycentrics, front face, object-space origin x.

ByteAddressBuffer gScene : register(t0);
ByteAddressBuffer gRays : register(t1);
RWByteAddressBuffer gOut : register(u2);

#define RS64_SRQ_SCENE gScene
#include "RS64SoftRayQuery.hlsli"

[numthreads(64, 1, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    if (id.x >= gRays.Load(44)) {
        return;
    }

    const uint at = id.x * 48;
    RayDesc ray;
    ray.Origin = asfloat(gRays.Load3(at));
    ray.TMin = asfloat(gRays.Load(at + 12));
    ray.Direction = asfloat(gRays.Load3(at + 16));
    ray.TMax = asfloat(gRays.Load(at + 28));
    const uint flags = gRays.Load(at + 32);
    const uint mask = gRays.Load(at + 36);
    const uint mode = gRays.Load(at + 40);

    RS64SoftRayQuery<RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> q;
    q.TraceRayInline(gScene, flags, mask, ray);
    while (q.Proceed()) {
        if ((q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE) && ((mode == 0) || ((q.CandidatePrimitiveIndex() % 2) == 0))) {
            q.CommitNonOpaqueTriangleHit();
        }
    }

    const uint outAt = id.x * 32;
    const bool hit = q.CommittedStatus() == COMMITTED_TRIANGLE_HIT;
    gOut.Store4(outAt, uint4(q.CommittedStatus(), asuint(hit ? q.CommittedRayT() : 0.0f), hit ? q.CommittedPrimitiveIndex() : 0, hit ? q.CommittedInstanceID() : 0));
    gOut.Store4(outAt + 16, uint4(asuint(hit ? q.CommittedTriangleBarycentrics() : float2(0.0f, 0.0f)), (hit && q.CommittedTriangleFrontFace()) ? 1u : 0u, asuint(hit ? q.CommittedObjectRayOrigin().x : 0.0f)));
}
