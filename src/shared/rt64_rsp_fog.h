//
// RT64
//

#pragma once

#include "shared/rt64_hlsl.h"

#ifdef HLSL_CPU
namespace interop {
#endif
    struct RSPFog {
        float mul;
        float offset;
        float tailEnd;
        // 1 in VR: fog follows the depth along the game camera's view, not along the eye's, so head turns and moves leave it where the game put it.
        float radial;
    };
#ifdef HLSL_CPU
};
#else
// Fog depth (clip z over w) of camera-space point v. In VR viewProj = P * E with E the rigid eye view and P's rows 2 and 3 (0, 0, m22, m32) and
// (0, 0, m23, 0), so row 3 is m23 * E's row 2 and row 2 is m22 * E's row 2 plus m32 in w; the game's curve A + B / z is evaluated at the game camera's depth |v.z|.
// Distance instead of depth fogged everything off the view's centre: the game's fog ramp spans a sliver of depth.
float rs64FogDepth(float4x4 viewProj, float3 v, RSPFog fog) {
    const float4 clip = mul(viewProj, float4(v, 1.0f));
    const float3 r3 = viewProj[3].xyz;
    const float m23 = length(r3);

    if ((fog.radial == 0.0f) || (m23 < 1e-12f)) {
        return (clip.w != 0.0f) ? (max(clip.z, 0) / clip.w) : 0.0f;
    }

    const float3 a3 = abs(r3);
    const uint j = (a3.x >= a3.y) ? ((a3.x >= a3.z) ? 0 : 2) : ((a3.y >= a3.z) ? 1 : 2);
    const float a = viewProj[2][j] / viewProj[3][j];
    const float m32 = viewProj[2][3] - a * viewProj[3][3];
    const float d = max(abs(v.z), 1e-6f);
    return max(a + m32 / (m23 * d), 0.0f);
}

// Mirrors rs64lights::fogCurve.
float rs64FogCurve(float x, float tailEnd) {
    const float knee = 0.85f;
    if ((tailEnd > knee) && (x > knee)) {
        x = knee + (1.0f - knee) * (x - knee) / (tailEnd - knee);
    }
    return saturate(x);
}
#endif