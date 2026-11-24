// RT64 - Software inline ray queries for WebGPU, which has no ray tracing: the RayQuery methods the RS64 traced passes use, walking
// plume's AS heap layout (contrib/plume/plume_webgpu_as.h). The includer defines RS64_SRQ_SCENE as the bound ByteAddressBuffer, whose
// first byte is the TLAS. Traversal order follows tinybvh's traverse_bvh2.cl (MIT): nearer child first.
// TraceRayInline runs the whole traversal and Proceed() returns false: the WGSL converter loses values carried through a caller's
// Proceed() loop in a large shader. So there are no candidates, and non-opaque geometry is skipped (the browser's scene has none: cutouts
// need bindless textures).

#pragma once

// DXR's values, defined here so the soft query works on any shader target.
#define RAY_FLAG_NONE 0x00
#define RAY_FLAG_FORCE_OPAQUE 0x01
#define RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH 0x04
#define RAY_FLAG_CULL_BACK_FACING_TRIANGLES 0x10
#define RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES 0x200
#define CANDIDATE_NON_OPAQUE_TRIANGLE 0
#define COMMITTED_NOTHING 0
#define COMMITTED_TRIANGLE_HIT 1

struct RS64RayDesc {
    float3 Origin;
    float TMin;
    float3 Direction;
    float TMax;
};

#define RayDesc RS64RayDesc

static const uint RS64_SRQ_LEAF = 0x80000000u;
static const uint RS64_SRQ_STACK = 64;

float3 rs64SrqLoad3(uint byteOffset) {
    return asfloat(RS64_SRQ_SCENE.Load3(byteOffset));
}

// Slab test against [tmin, best]; the entry distance, or a negative value for a miss.
float rs64SrqSlab(float3 lo, float3 hi, float3 org, float3 invDir, float tmin, float best) {
    const float3 t0 = (lo - org) * invDir;
    const float3 t1 = (hi - org) * invDir;
    const float3 tn = min(t0, t1);
    const float3 tf = max(t0, t1);
    const float enter = max(max(tn.x, tn.y), max(tn.z, tmin));
    const float leave = min(min(tf.x, tf.y), min(tf.z, best));
    return (enter <= leave) ? enter : -1.0f;
}

template<uint Flags>
struct RS64SoftRayQuery {
    uint status;
    float comT;
    float2 comBary;
    uint comPrim;
    uint comInstanceID;
    uint comInstanceAt;
    bool comFront;
    float3 comObjOrg;
    float3 comObjDir;

    void TraceRayInline(ByteAddressBuffer scene, uint rayFlags, uint instanceMask, RayDesc ray) {
        const uint flags = Flags | rayFlags;
        const uint mask = instanceMask & 0xFFu;
        const bool firstHit = (flags & RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH) != 0;
        const uint instanceCount = RS64_SRQ_SCENE.Load(0);
        uint stack[RS64_SRQ_STACK];
        bool stop = false;
        status = COMMITTED_NOTHING;
        comT = ray.TMax;
        comBary = float2(0.0f, 0.0f);
        comPrim = 0;
        comInstanceID = 0;
        comInstanceAt = 0;
        comFront = false;
        comObjOrg = float3(0.0f, 0.0f, 0.0f);
        comObjDir = float3(0.0f, 0.0f, 0.0f);

        for (uint i = 0; (i < instanceCount) && !stop; i++) {
            const uint at = 16 + i * 64;
            const uint4 tail = RS64_SRQ_SCENE.Load4(at + 48);
            const uint4 header = RS64_SRQ_SCENE.Load4(tail.w);
            const bool opaque = ((header.z & 1u) != 0) || ((flags & RAY_FLAG_FORCE_OPAQUE) != 0);
            uint top = 0;
            if (((tail.x & mask) != 0) && (header.x != 0) && opaque) {
                stack[0] = (header.y == 0) ? RS64_SRQ_LEAF : 0u;
                top = 1;
            }

            const float4 r0 = asfloat(RS64_SRQ_SCENE.Load4(at));
            const float4 r1 = asfloat(RS64_SRQ_SCENE.Load4(at + 16));
            const float4 r2 = asfloat(RS64_SRQ_SCENE.Load4(at + 32));
            const float3 org = float3(dot(r0.xyz, ray.Origin) + r0.w, dot(r1.xyz, ray.Origin) + r1.w, dot(r2.xyz, ray.Origin) + r2.w);
            const float3 dir = float3(dot(r0.xyz, ray.Direction), dot(r1.xyz, ray.Direction), dot(r2.xyz, ray.Direction));
            const float3 safe = select(abs(dir) < 1e-20f, select(dir < 0.0f, -1e-20f, 1e-20f), dir);
            const float3 invDir = 1.0f / safe;
            const uint blasBase = tail.w;
            const uint triBase = tail.w + 16 + header.y * 64;
            const bool cullBack = ((flags & RAY_FLAG_CULL_BACK_FACING_TRIANGLES) != 0) && ((tail.z & 1u) == 0);

            while ((top > 0) && !stop) {
                top--;
                const uint word = stack[top];
                if ((word & RS64_SRQ_LEAF) != 0) {
                    // Moller-Trumbore in object space; front face is DXR's (clockwise from the origin), det < 0.
                    const uint tri = triBase + (word & ~RS64_SRQ_LEAF) * 48;
                    const float3 v0 = rs64SrqLoad3(tri);
                    const float3 e1 = rs64SrqLoad3(tri + 16) - v0;
                    const float3 e2 = rs64SrqLoad3(tri + 32) - v0;
                    const float3 p = cross(dir, e2);
                    const float det = dot(e1, p);
                    const bool front = det < 0.0f;
                    const float inv = 1.0f / ((det != 0.0f) ? det : 1.0f);
                    const float3 s = org - v0;
                    const float u = dot(s, p) * inv;
                    const float3 q = cross(s, e1);
                    const float v = dot(dir, q) * inv;
                    const float t = dot(e2, q) * inv;
                    if ((det != 0.0f) && !(cullBack && !front) && (u >= 0.0f) && (v >= 0.0f) && ((u + v) <= 1.0f) && (t >= ray.TMin) && (t <= comT)) {
                        status = COMMITTED_TRIANGLE_HIT;
                        comT = t;
                        comBary = float2(u, v);
                        comPrim = RS64_SRQ_SCENE.Load(tri + 12);
                        comInstanceID = tail.y;
                        comInstanceAt = at;
                        comFront = front;
                        comObjOrg = org;
                        comObjDir = dir;
                        stop = firstHit;
                    }
                }
                else {
                    // Nearer child on top; a full stack drops the farther one (a hit there may be missed), never writes past the stack.
                    const uint node = blasBase + 16 + word * 64;
                    const uint4 a = RS64_SRQ_SCENE.Load4(node);
                    const uint4 b = RS64_SRQ_SCENE.Load4(node + 16);
                    const uint4 c = RS64_SRQ_SCENE.Load4(node + 32);
                    const uint4 d = RS64_SRQ_SCENE.Load4(node + 48);
                    const float dl = rs64SrqSlab(asfloat(a.xyz), asfloat(b.xyz), org, invDir, ray.TMin, comT);
                    const float dr = rs64SrqSlab(asfloat(c.xyz), asfloat(d.xyz), org, invDir, ray.TMin, comT);
                    const bool hitL = dl >= 0.0f;
                    const bool hitR = dr >= 0.0f;
                    const bool leftFirst = dl <= dr;
                    const uint nearer = (hitL && (!hitR || leftFirst)) ? a.w : b.w;
                    const uint farther = leftFirst ? b.w : a.w;
                    if (hitL && hitR && ((top + 2) <= RS64_SRQ_STACK)) {
                        stack[top] = farther;
                        top++;
                    }

                    if ((hitL || hitR) && (top < RS64_SRQ_STACK)) {
                        stack[top] = nearer;
                        top++;
                    }
                }
            }
        }
    }

    bool Proceed() {
        return false;
    }

    void CommitNonOpaqueTriangleHit() {
    }

    uint CandidateType() {
        return CANDIDATE_NON_OPAQUE_TRIANGLE;
    }

    uint CandidatePrimitiveIndex() {
        return 0;
    }

    float2 CandidateTriangleBarycentrics() {
        return float2(0.0f, 0.0f);
    }

    uint CommittedStatus() {
        return status;
    }

    float CommittedRayT() {
        return comT;
    }

    uint CommittedPrimitiveIndex() {
        return comPrim;
    }

    uint CommittedInstanceID() {
        return comInstanceID;
    }

    float2 CommittedTriangleBarycentrics() {
        return comBary;
    }

    bool CommittedTriangleFrontFace() {
        return comFront;
    }

    float3 CommittedObjectRayOrigin() {
        return comObjOrg;
    }

    float3 CommittedObjectRayDirection() {
        return comObjDir;
    }

    // The instance record holds world-to-object; this inverts it.
    float3x4 CommittedObjectToWorld3x4() {
        const float4 r0 = asfloat(RS64_SRQ_SCENE.Load4(comInstanceAt));
        const float4 r1 = asfloat(RS64_SRQ_SCENE.Load4(comInstanceAt + 16));
        const float4 r2 = asfloat(RS64_SRQ_SCENE.Load4(comInstanceAt + 32));
        const float3 c0 = cross(r1.xyz, r2.xyz);
        const float3 c1 = cross(r2.xyz, r0.xyz);
        const float3 c2 = cross(r0.xyz, r1.xyz);
        const float inv = 1.0f / dot(r0.xyz, c0);
        const float3 t = float3(r0.w, r1.w, r2.w);
        const float3 m0 = float3(c0.x, c1.x, c2.x) * inv;
        const float3 m1 = float3(c0.y, c1.y, c2.y) * inv;
        const float3 m2 = float3(c0.z, c1.z, c2.z) * inv;
        return float3x4(float4(m0, -dot(m0, t)), float4(m1, -dot(m1, t)), float4(m2, -dot(m2, t)));
    }
};
