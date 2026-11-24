// RT64 - Software inline ray queries for WebGPU, which has no ray tracing: the RayQuery methods the RS64 traced passes use, walking
// plume's AS heap layout (contrib/plume/plume_webgpu_as.h). The includer defines RS64_SRQ_SCENE as the bound ByteAddressBuffer, whose
// first byte is the TLAS. Traversal order follows tinybvh's traverse_bvh2.cl (MIT): nearer child first.

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

template<uint Flags>
struct RS64SoftRayQuery {
    float3 org;
    float3 dir;
    float tmin;
    float tmax;
    uint flags;
    uint mask;
    bool done;
    uint nextInstance;
    uint instanceCount;
    bool inBlas;
    uint blasBase;
    uint triBase;
    bool blasOpaque;
    bool cullBack;
    uint instanceID;
    float3 objOrg;
    float3 objDir;
    float3 invDir;
    uint stack[RS64_SRQ_STACK];
    uint top;

    float candT;
    float2 candBary;
    uint candPrim;
    bool candFront;

    uint status;
    float comT;
    float2 comBary;
    uint comPrim;
    uint comInstanceID;
    bool comFront;
    float3 comObjOrg;
    float3 comObjDir;

    void TraceRayInline(ByteAddressBuffer scene, uint rayFlags, uint instanceMask, RayDesc ray) {
        org = ray.Origin;
        dir = ray.Direction;
        tmin = ray.TMin;
        tmax = ray.TMax;
        flags = Flags | rayFlags;
        mask = instanceMask & 0xFFu;
        done = false;
        nextInstance = 0;
        instanceCount = RS64_SRQ_SCENE.Load(0);
        inBlas = false;
        top = 0;
        status = COMMITTED_NOTHING;
        comT = ray.TMax;
    }

    // Slab test against [tmin, best]; returns the entry distance, or a negative value for a miss.
    float slab(float3 lo, float3 hi, float best) {
        const float3 t0 = (lo - objOrg) * invDir;
        const float3 t1 = (hi - objOrg) * invDir;
        const float3 tn = min(t0, t1);
        const float3 tf = max(t0, t1);
        const float enter = max(max(tn.x, tn.y), max(tn.z, tmin));
        const float leave = min(min(tf.x, tf.y), min(tf.z, best));
        return (enter <= leave) ? enter : -1.0f;
    }

    // Moller-Trumbore in object space; front face is DXR's (clockwise from the origin), det < 0.
    bool hitTriangle(uint t, float best, out float hitT, out float2 bary, out bool front, out uint prim) {
        const uint at = triBase + t * 48;
        const float3 v0 = rs64SrqLoad3(at);
        const float3 e1 = rs64SrqLoad3(at + 16) - v0;
        const float3 e2 = rs64SrqLoad3(at + 32) - v0;
        prim = RS64_SRQ_SCENE.Load(at + 12);
        hitT = 0.0f;
        bary = float2(0.0f, 0.0f);
        front = false;
        const float3 p = cross(objDir, e2);
        const float det = dot(e1, p);
        if (det == 0.0f) {
            return false;
        }

        front = det < 0.0f;
        if (cullBack && !front) {
            return false;
        }

        const float inv = 1.0f / det;
        const float3 s = objOrg - v0;
        const float u = dot(s, p) * inv;
        const float3 q = cross(s, e1);
        const float v = dot(objDir, q) * inv;
        hitT = dot(e2, q) * inv;
        bary = float2(u, v);
        return (u >= 0.0f) && (v >= 0.0f) && ((u + v) <= 1.0f) && (hitT >= tmin) && (hitT <= best);
    }

    // Enters instance nextInstance when the mask lets it through and its BLAS has triangles. No early exits: the WGSL converter can't
    // follow a return or continue out of the loops it is called from.
    void tryInstance() {
        const uint at = 16 + nextInstance * 64;
        nextInstance++;
        const uint4 tail = RS64_SRQ_SCENE.Load4(at + 48);
        const uint4 header = RS64_SRQ_SCENE.Load4(tail.w);
        if (((tail.x & mask) != 0) && (header.x != 0)) {
            const float4 r0 = asfloat(RS64_SRQ_SCENE.Load4(at));
            const float4 r1 = asfloat(RS64_SRQ_SCENE.Load4(at + 16));
            const float4 r2 = asfloat(RS64_SRQ_SCENE.Load4(at + 32));
            objOrg = float3(dot(r0.xyz, org) + r0.w, dot(r1.xyz, org) + r1.w, dot(r2.xyz, org) + r2.w);
            objDir = float3(dot(r0.xyz, dir), dot(r1.xyz, dir), dot(r2.xyz, dir));
            const float3 safe = select(abs(objDir) < 1e-20f, select(objDir < 0.0f, -1e-20f, 1e-20f), objDir);
            invDir = 1.0f / safe;
            blasBase = tail.w;
            triBase = tail.w + 16 + header.y * 64;
            blasOpaque = ((header.z & 1u) != 0) || ((flags & RAY_FLAG_FORCE_OPAQUE) != 0);
            cullBack = ((flags & RAY_FLAG_CULL_BACK_FACING_TRIANGLES) != 0) && ((tail.z & 1u) == 0);
            instanceID = tail.y;
            top = 1;
            stack[0] = (header.y == 0) ? RS64_SRQ_LEAF : 0u;
            inBlas = true;
        }
    }

    void commit(float t, float2 bary, uint prim, bool front) {
        status = COMMITTED_TRIANGLE_HIT;
        comT = t;
        comBary = bary;
        comPrim = prim;
        comInstanceID = instanceID;
        comFront = front;
        comObjOrg = objOrg;
        comObjDir = objDir;
        if ((flags & RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH) != 0) {
            done = true;
        }
    }

    // Opaque hits commit here; a non-opaque one stops with Proceed() true for the caller to commit or skip.
    bool Proceed() {
        bool candidate = false;
        while (!done && !candidate) {
            if (!inBlas) {
                if (nextInstance < instanceCount) {
                    tryInstance();
                }
                else {
                    done = true;
                }
            }
            else if (top == 0) {
                inBlas = false;
            }
            else {
                const uint word = stack[top - 1];
                top--;
                if ((word & RS64_SRQ_LEAF) != 0) {
                    float t;
                    float2 bary;
                    bool front;
                    uint prim;
                    if (hitTriangle(word & ~RS64_SRQ_LEAF, comT, t, bary, front, prim)) {
                        if (blasOpaque) {
                            commit(t, bary, prim, front);
                        }
                        else {
                            candT = t;
                            candBary = bary;
                            candPrim = prim;
                            candFront = front;
                            candidate = true;
                        }
                    }
                }
                else {
                    visitNode(word);
                }
            }
        }

        return candidate;
    }

    // Pushes the children whose boxes the ray enters before the current best hit, nearer on top.
    // A full stack drops the farther child (a hit there may be missed), never writes past the stack.
    void visitNode(uint word) {
        const uint at = blasBase + 16 + word * 64;
        const uint4 a = RS64_SRQ_SCENE.Load4(at);
        const uint4 b = RS64_SRQ_SCENE.Load4(at + 16);
        const uint4 c = RS64_SRQ_SCENE.Load4(at + 32);
        const uint4 d = RS64_SRQ_SCENE.Load4(at + 48);
        const float dl = slab(asfloat(a.xyz), asfloat(b.xyz), comT);
        const float dr = slab(asfloat(c.xyz), asfloat(d.xyz), comT);
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

    void CommitNonOpaqueTriangleHit() {
        commit(candT, candBary, candPrim, candFront);
    }

    uint CandidateType() {
        return CANDIDATE_NON_OPAQUE_TRIANGLE;
    }

    uint CandidatePrimitiveIndex() {
        return candPrim;
    }

    float2 CandidateTriangleBarycentrics() {
        return candBary;
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
};
