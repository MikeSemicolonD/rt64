//
// RT64
//

#pragma once

// Descriptor sets of the RS64 passes (rt64_framebuffer_renderer_rs64.cpp, the VR hands in the present queue).

#include "common/rt64_common.h"
#include "common/rt64_plume.h"

#include "rt64_sampler_library.h"

namespace RT64 {
    struct RS64LightsDescriptorSet : RenderDescriptorSetBase {
        uint32_t gLights;
        uint32_t gDepth;

        RS64LightsDescriptorSet(RenderDevice *device = nullptr) {
            builder.begin();
            gLights = builder.addConstantBuffer(0);
            gDepth = builder.addTexture(1);
            builder.end();

            if (device != nullptr) {
                create(device);
            }
        }
    };

    struct RS64VrHandsDescriptorSet : RenderDescriptorSetBase {
        uint32_t gDepth;
        uint32_t gSceneDepth;

        RS64VrHandsDescriptorSet(RenderDevice *device = nullptr) {
            builder.begin();
            gDepth = builder.addConstantBuffer(1);
            gSceneDepth = builder.addTexture(2);
            builder.end();

            if (device != nullptr) {
                create(device);
            }
        }
    };

    // Adds the half-resolution point light pass to the frame: its constant buffer (view and viewport), the depth and the half-resolution lights.
    struct RS64LightsUpsampleDescriptorSet : RenderDescriptorSetBase {
        uint32_t gLights;
        uint32_t gDepth;
        uint32_t gHalf;

        RS64LightsUpsampleDescriptorSet(RenderDevice *device = nullptr) {
            builder.begin();
            gLights = builder.addConstantBuffer(0);
            gDepth = builder.addTexture(1);
            gHalf = builder.addTexture(2);
            builder.end();

            if (device != nullptr) {
                create(device);
            }
        }
    };

    // The RS64 whole-map terrain pass: the frame's transform, viewport and fog buffers plus the static map lattice and tile textures.
    struct RS64TerrainDescriptorSet : RenderDescriptorSetBase {
        uint32_t gWorldTransforms;
        uint32_t gViewProjTransforms;
        uint32_t gViewports;
        uint32_t gFog;
        uint32_t gHeights;
        uint32_t gColors;
        uint32_t gCellLayers;
        uint32_t gTexels;
        uint32_t gFrame;
        uint32_t gChunks;
        uint32_t gSampler;

        RS64TerrainDescriptorSet(const SamplerLibrary &samplerLibrary, RenderDevice *device = nullptr) {
            builder.begin();
            gWorldTransforms = builder.addStructuredBuffer(1);
            gViewProjTransforms = builder.addStructuredBuffer(2);
            gViewports = builder.addStructuredBuffer(3);
            gFog = builder.addStructuredBuffer(4);
            gHeights = builder.addStructuredBuffer(5);
            gColors = builder.addStructuredBuffer(6);
            gCellLayers = builder.addStructuredBuffer(7);
            gTexels = builder.addStructuredBuffer(8);
            gFrame = builder.addStructuredBuffer(9);
            gChunks = builder.addStructuredBuffer(10);
            gSampler = builder.addImmutableSampler(11, samplerLibrary.linear.clampClamp.get());
            builder.end();

            if (device != nullptr) {
                create(device);
            }
        }
    };

    // Set 3 of the traced RS64 passes (sets 0-2 are the raster common and bindless texture sets).
    struct RS64TracedDescriptorSet : RenderDescriptorSetBase {
        uint32_t gParams;
        uint32_t gDepth;
        uint32_t gScene;
        uint32_t gCutoutDraws;
        uint32_t gCutoutIndices;
        uint32_t gTexCoords;
        uint32_t gTerrainIndices;
        uint32_t gTerrainNormals;
        uint32_t gHitColors;
        uint32_t gEmissiveColors;
        uint32_t gTerrainColors;
        // The frame's colour target for screen-space water reflections (shadow pass only; the other passes bind their depth here).
        uint32_t gColor;
        uint32_t gShadedCols;

        RS64TracedDescriptorSet(RenderDevice *device = nullptr) {
            builder.begin();
            gParams = builder.addConstantBuffer(1);
            gDepth = builder.addTexture(2);
            gScene = builder.addAccelerationStructure(3);
            gCutoutDraws = builder.addStructuredBuffer(4);
            gCutoutIndices = builder.addByteAddressBuffer(5);
            gTexCoords = builder.addByteAddressBuffer(6);
            gTerrainIndices = builder.addByteAddressBuffer(7);
            gTerrainNormals = builder.addByteAddressBuffer(8);
            gHitColors = builder.addStructuredBuffer(9);
            gEmissiveColors = builder.addStructuredBuffer(10);
            gTerrainColors = builder.addByteAddressBuffer(11);
            gColor = builder.addTexture(12);
            gShadedCols = builder.addByteAddressBuffer(13);
            builder.end();

            if (device != nullptr) {
                create(device);
            }
        }
    };

    struct RS64ShadowBlurDescriptorSet : RenderDescriptorSetBase {
        uint32_t gBlur;
        uint32_t gDepth;
        uint32_t gMask;
        uint32_t gIndirect;

        RS64ShadowBlurDescriptorSet(RenderDevice *device = nullptr) {
            builder.begin();
            gBlur = builder.addConstantBuffer(0);
            gDepth = builder.addTexture(1);
            gMask = builder.addTexture(2);
            gIndirect = builder.addTexture(3);
            builder.end();

            if (device != nullptr) {
                create(device);
            }
        }
    };

    struct RS64ShadowTemporalDescriptorSet : RenderDescriptorSetBase {
        uint32_t gBlur;
        uint32_t gDepth;
        uint32_t gMask;
        uint32_t gIndirect;
        uint32_t gHistMask;
        uint32_t gHistIndirect;

        RS64ShadowTemporalDescriptorSet(RenderDevice *device = nullptr) {
            builder.begin();
            gBlur = builder.addConstantBuffer(0);
            gDepth = builder.addTexture(1);
            gMask = builder.addTexture(2);
            gIndirect = builder.addTexture(3);
            gHistMask = builder.addTexture(4);
            gHistIndirect = builder.addTexture(5);
            builder.end();

            if (device != nullptr) {
                create(device);
            }
        }
    };

    struct RS64WaterMirrorDescriptorSet : RenderDescriptorSetBase {
        uint32_t gBlur;
        uint32_t gDepth;
        uint32_t gMirror;
        uint32_t gLate;

        RS64WaterMirrorDescriptorSet(RenderDevice *device = nullptr) {
            builder.begin();
            gBlur = builder.addConstantBuffer(0);
            gDepth = builder.addTexture(1);
            gMirror = builder.addTexture(3);
            gLate = builder.addTexture(4);
            builder.end();

            if (device != nullptr) {
                create(device);
            }
        }
    };
};
