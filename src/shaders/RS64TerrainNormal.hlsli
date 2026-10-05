//
// RT64
//

#pragma once

// Static terrain mesh (TLAS instance id 1, mask 0x02): its index buffer and welded smooth vertex normals (float3, map space).
ByteAddressBuffer gTerrainIndices : register(t7, space3);
ByteAddressBuffer gTerrainNormals : register(t8, space3);

float3 rs64TerrainVertexNormal(uint vertexIndex) {
    return asfloat(gTerrainNormals.Load3(vertexIndex * 12));
}

// Mirrors rs64lights::cofactorNormal: normal through the instance's 3x4 transform (inverse-transpose).
float3 rs64CofactorNormal(float3x4 m, float3 n) {
    const float3 a = m[0].xyz;
    const float3 b = m[1].xyz;
    const float3 c = m[2].xyz;
    const float3 bc = cross(b, c);
    const float s = (dot(a, bc) < 0.0f) ? -1.0f : 1.0f;
    return s * normalize(float3(dot(bc, n), dot(cross(c, a), n), dot(cross(a, b), n)));
}

// Smooth heightfield normal (camera space) when the static terrain lies at this pixel's view distance (within window * dist); false for anything else.
bool rs64TerrainNormal(float3 p, float dist, float window, out float3 n) {
    n = float3(0.0f, 0.0f, 0.0f);
    RayDesc ray;
    ray.Origin = float3(0.0f, 0.0f, 0.0f);
    ray.Direction = p / dist;
    ray.TMin = dist * (1.0f - window);
    ray.TMax = dist * (1.0f + window);
    RayQuery<RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES | RAY_FLAG_FORCE_OPAQUE> q;
    q.TraceRayInline(gScene, RAY_FLAG_NONE, 0x02, ray);
    q.Proceed();
    if (q.CommittedStatus() != COMMITTED_TRIANGLE_HIT) {
        return false;
    }
    const uint prim = q.CommittedPrimitiveIndex();
    const float2 bary = q.CommittedTriangleBarycentrics();
    const uint i0 = gTerrainIndices.Load((prim * 3 + 0) * 4);
    const uint i1 = gTerrainIndices.Load((prim * 3 + 1) * 4);
    const uint i2 = gTerrainIndices.Load((prim * 3 + 2) * 4);
    const float3 nm = rs64TerrainVertexNormal(i0) * (1.0f - bary.x - bary.y) + rs64TerrainVertexNormal(i1) * bary.x + rs64TerrainVertexNormal(i2) * bary.y;
    n = rs64CofactorNormal(q.CommittedObjectToWorld3x4(), nm);
    return true;
}
