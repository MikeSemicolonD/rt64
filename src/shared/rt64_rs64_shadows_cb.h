//
// RT64
//

#pragma once

#include "shared/rt64_hlsl.h"

#ifdef HLSL_CPU
namespace interop {
#endif
    struct RS64ShadowsCB {
        float4x4 viewFromScreen;
        float4 viewport;
        float4 sunDir;
        uint debugMode;
        float strength;
        float tMin;
        float tMax;
        float normalBias;
        float terrainTMinScale;
        uint cutoutDrawCount;
        uint softRays;
        float4 receiverSkip;
        // Soft shadows: tan(sun angular radius), pixels per world unit at unit distance, max blur radius in pixels, terrain receivers (0 off, 1 smooth normals, 2 smooth normals + terminator).
        float4 softParams;
        // Terrain-normal depth window (fraction of view distance), unused x3.
        float4 terrainParams;
        // Off-screen casters: view -> viewport clip (frustum test), history slot cameras in this frame's camera space (w = skip radius), slot mask (0x08 << slot).
        float4x4 screenFromView;
        float4 historyCam[4];
        uint historyMask;
        uint historyPad0;
        uint historyPad1;
        uint historyPad2;
        // AO (rays, range fraction of view distance, strength, min blur px), GI (rays, range fraction, strength, min blur px),
        // reflections (tan roughness cone, range fraction, strength, emissive gain for bounce + reflection), sun colour.
        float4 aoParams;
        float4 giParams;
        float4 reflParams;
        float4 sunColor;
        // Hit colours (patched when the scene is built): hit table count, emissive table count, terrain colour, caster average (rgba8).
        uint4 hitParams;
        // 1 sun shadows, 2 AO, 4 GI, 8 reflections.
        uint featureMask;
        uint featurePad0;
        uint featurePad1;
        uint featurePad2;
    };
#ifdef HLSL_CPU
};
#endif
