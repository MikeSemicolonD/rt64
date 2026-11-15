// RT64 - Push constants of the VR controller pass

#pragma once

#include "shared/rt64_hlsl.h"

#ifdef HLSL_CPU
namespace interop {
#endif
    // One controller in one eye (RS64VrHandsVS/PS): clip from grip space as rows, the grip's rotation rows into the tracking space for lighting, and the body colour.
    struct RS64VrHandsCB {
        float4 clip[4];
        float4 rotation[3];
        float4 color;
    };
#ifdef HLSL_CPU
};
#endif
