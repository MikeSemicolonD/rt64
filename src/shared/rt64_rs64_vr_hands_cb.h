// RT64 - Push constants and depth test parameters of the VR controller pass

#pragma once

#include "shared/rt64_hlsl.h"

#ifdef HLSL_CPU
namespace interop {
#endif
    // One controller in one eye (RS64VrHandsVS/PS): clip from grip space as rows, the grip's transform rows into the tracking space (rotation in xyz,
    // translation in w; the rotation also lights it), and the body colour.
    struct RS64VrHandsCB {
        float4 clip[4];
        float4 rotation[3];
        float4 color;
    };

    // One eye's scene depth for hiding the controllers behind it: that eye's view from the tracking space as rows (meters, OpenXR axes), its fov
    // tangents (left, right, up, down), the game projection's z and w columns, then units per meter, the viewport's depth scale and offset and 1 = on.
    struct RS64VrHandsDepthCB {
        float4 eyeFromStage[3];
        float4 tanFov;
        float4 zColumn;
        float4 wColumn;
        float4 params;
        float4 size;
    };
#ifdef HLSL_CPU
};
#endif
