// RT64 - VR controllers in the OpenXR grip pose's space (meters, -z along the handle towards the buttons, +y up from the knuckles): the runtime's
// own controller model from a vertex buffer (RS64_HAND_MESH), or a stand-in built from the vertex index: a handle, a head and the tracking ring.

#include "shared/rt64_rs64_vr_hands_cb.h"

[[vk::push_constant]] ConstantBuffer<RS64VrHandsCB> gHand : register(b0, space0);

#ifdef RS64_HAND_MESH

void VSMain(float3 position : POSITION, float3 normal : NORMAL, out float4 oPosition : SV_POSITION, out float3 oNormal : NORMAL, out float3 oColor : COLOR0, out float3 oStage : TEXCOORD0) {
    const float4 g = float4(position, 1.0f);
    oPosition = float4(dot(gHand.clip[0], g), dot(gHand.clip[1], g), dot(gHand.clip[2], g), dot(gHand.clip[3], g));
    oNormal = float3(dot(gHand.rotation[0].xyz, normal), dot(gHand.rotation[1].xyz, normal), dot(gHand.rotation[2].xyz, normal));
    oStage = float3(dot(gHand.rotation[0], g), dot(gHand.rotation[1], g), dot(gHand.rotation[2], g));
    oColor = gHand.color.rgb;
}

#else

static const float kPi = 3.14159265f;
static const uint kAround = 24;
static const uint kAlong = 13;
static const uint kPartVertices = kAround * kAlong * 6;

// One quad's six corners as (around, along) steps.
static const uint2 kCorners[6] = { uint2(0, 0), uint2(1, 0), uint2(1, 1), uint2(0, 0), uint2(1, 1), uint2(0, 1) };

// Rotation about x by `a` radians.
float3 tiltX(float3 v, float a) {
    const float c = cos(a), s = sin(a);
    return float3(v.x, c * v.y - s * v.z, s * v.y + c * v.z);
}

void VSMain(uint vertexId : SV_VertexID, out float4 oPosition : SV_POSITION, out float3 oNormal : NORMAL, out float3 oColor : COLOR0, out float3 oStage : TEXCOORD0) {
    const uint part = vertexId / kPartVertices;
    const uint inPart = vertexId % kPartVertices;
    const uint quad = inPart / 6;
    const uint2 corner = kCorners[inPart % 6];
    const uint i = (quad % kAround) + corner.x;
    const uint j = (quad / kAround) + corner.y;
    const float theta = 2.0f * kPi * float(i) / float(kAround);
    float3 p;
    float3 n;
    float shade = 1.0f;

    if (part == 0) {
        // Handle: a capsule along z, its two half spheres joined by the middle row (j 6 to 7).
        const float radius = 0.017f;
        const float halfLength = 0.035f;
        const bool top = (j >= 7);
        const float phi = top ? (0.5f * kPi * float(j - 7) / 6.0f) : (-0.5f * kPi + 0.5f * kPi * float(j) / 6.0f);
        n = float3(cos(phi) * cos(theta), cos(phi) * sin(theta), sin(phi));
        p = radius * n + float3(0.0f, 0.0f, (top ? halfLength : -halfLength) + 0.01f);
    }
    else if (part == 1) {
        // Head: a flattened ellipsoid where the thumb rests.
        const float3 radii = float3(0.03f, 0.014f, 0.032f);
        const float phi = -0.5f * kPi + kPi * float(j) / float(kAlong);
        const float3 s = float3(cos(phi) * cos(theta), sin(phi), cos(phi) * sin(theta));
        p = s * radii + float3(0.0f, 0.012f, -0.045f);
        n = normalize(s / radii);
    }
    else {
        // Tracking ring: a torus around the head, tilted up towards the front.
        const float major = 0.045f;
        const float minor = 0.005f;
        const float phi = 2.0f * kPi * float(j) / float(kAlong);
        const float3 local = float3((major + minor * cos(phi)) * cos(theta), minor * sin(phi), (major + minor * cos(phi)) * sin(theta));
        n = tiltX(float3(cos(phi) * cos(theta), sin(phi), cos(phi) * sin(theta)), -0.6f);
        p = tiltX(local, -0.6f) + float3(0.0f, 0.03f, -0.06f);
        shade = 1.35f;
    }

    const float4 g = float4(p, 1.0f);
    oPosition = float4(dot(gHand.clip[0], g), dot(gHand.clip[1], g), dot(gHand.clip[2], g), dot(gHand.clip[3], g));
    oNormal = float3(dot(gHand.rotation[0].xyz, n), dot(gHand.rotation[1].xyz, n), dot(gHand.rotation[2].xyz, n));
    oStage = float3(dot(gHand.rotation[0], g), dot(gHand.rotation[1], g), dot(gHand.rotation[2], g));
    oColor = gHand.color.rgb * shade;
}

#endif
