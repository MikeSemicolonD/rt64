//
// RT64
//

#pragma once

#include "rt64_sampler_library.h"

namespace RT64 {
    // Android swap chains offer RGBA8 but not BGRA8.
#if defined(__ANDROID__)
    static const RenderFormat SwapChainFormat = RenderFormat::R8G8B8A8_UNORM;
#else
    static const RenderFormat SwapChainFormat = RenderFormat::B8G8R8A8_UNORM;
#endif

    struct ShaderRecord {
        std::unique_ptr<RenderPipeline> pipeline;
        std::unique_ptr<RenderPipelineLayout> pipelineLayout;
    };

    struct ShaderLibrary {
        SamplerLibrary samplerLibrary;
        bool usesHDR = false;
        bool usesHardwareResolve = false;

        // All shaders.
        ShaderRecord bicubicScaling;
        ShaderRecord boxFilter;
        ShaderRecord compose;
        ShaderRecord debug;
        ShaderRecord fbChangesClear;
        ShaderRecord fbChangesDrawColor;
        ShaderRecord rs64Lights;
        ShaderRecord rs64LightsDebug;
        ShaderRecord rs64Shadows;
        ShaderRecord rs64ShadowsDebug;
        ShaderRecord rs64ShadowsSoft;
        ShaderRecord rs64ShadowBlurH;
        ShaderRecord rs64ShadowBlurV;
        ShaderRecord rs64ShadowBlurVDebug;
        ShaderRecord rs64LightsShadowed;
        ShaderRecord rs64LightsShadowedDebug;
        ShaderRecord rs64FogShafts;
        ShaderRecord rs64FogShaftsDebug;
        ShaderRecord fbChangesDrawDepth;
        ShaderRecord fbReadAnyChanges;
        ShaderRecord fbReadAnyFull;
        ShaderRecord fbReinterpret;
        ShaderRecord fbWriteColor;
        ShaderRecord fbWriteDepth;
        ShaderRecord fbWriteDepthMS;
        ShaderRecord gaussianFilterRGB3x3;
        ShaderRecord histogramAverage;
        ShaderRecord histogramClear;
        ShaderRecord histogramSet;
        ShaderRecord idle;
        ShaderRecord im3dLine;
        ShaderRecord im3dPoint;
        ShaderRecord im3dTriangle;
        ShaderRecord luminanceHistogram;
        ShaderRecord postProcess;
        ShaderRecord rspModify;
        ShaderRecord rspProcess;
        ShaderRecord rspSmoothNormal;
        ShaderRecord rspVertexTestZ;
        ShaderRecord rspVertexTestZMS;
        ShaderRecord rspWorld;
        ShaderRecord rtCopyColorToDepth;
        ShaderRecord rtCopyDepthToColor;
        ShaderRecord rtCopyColorToDepthMS;
        ShaderRecord rtCopyDepthToColorMS;
        ShaderRecord textureDecode;
        ShaderRecord textureCopy;
        ShaderRecord textureResolve;
        ShaderRecord videoInterfaceLinear;
        ShaderRecord videoInterfaceNearest;
        ShaderRecord videoInterfacePixel;

        ShaderLibrary(bool usesHDR, bool usesHardwareResolve);
        ~ShaderLibrary();
        void setupCommonShaders(RenderInterface *rhi, RenderDevice *device);
        void setupMultisamplingShaders(RenderInterface *rhi, RenderDevice *device, const RenderMultisampling &multisampling);
    };
};
