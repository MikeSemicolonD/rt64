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
    };
#ifdef HLSL_CPU
};
#else
// Mirrors rs64lights::fogCurve.
float rs64FogCurve(float x, float tailEnd) {
    const float knee = 0.85f;
    if ((tailEnd > knee) && (x > knee)) {
        x = knee + (1.0f - knee) * (x - knee) / (tailEnd - knee);
    }
    return saturate(x);
}
#endif