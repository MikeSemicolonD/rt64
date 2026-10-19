//
// RT64
//

#pragma once

#include "rt64_shader_common.h"

#include <condition_variable>
#include <cstdlib>
#include <mutex>
#include <thread>

#include "re-spirv/re-spirv.h"

#include "plume_render_interface.h"
#include "shared/rt64_blender.h"
#include "shared/rt64_color_combiner.h"
#include "shared/rt64_other_mode.h"

#include "rt64_shader_library.h"
#include "rt64_shader_compiler.h"

// Require the use of sample locations for MSAA.
#define SAMPLE_LOCATIONS_REQUIRED 1

namespace RT64 {
    // D3D11, and GPUs whose per-stage texture limit can't hold the bindless texture array (Raspberry Pi), bind each draw's tiles to 8 fixed
    // slots and have no ubershader. ROGUESQ_TEXTURE_SLOTS=1 forces it for testing.
    inline bool UsesTextureSlots(bool d3d11, const RenderDeviceCapabilities &capabilities) {
        static const bool forced = []() { const char *e = std::getenv("ROGUESQ_TEXTURE_SLOTS"); return (e != nullptr) && (e[0] == '1'); }();
        return d3d11 || forced || (capabilities.maxTexturesPerStage < 8192);
    }

    // Textures are decoded ahead of time by TextureDecodeCS when compute is available; otherwise draws sample raw TMEM.
    // ROGUESQ_GPU_TEXTURE_DECODE=0 forces raw TMEM to isolate decode problems on a new GPU.
    inline bool DecodesTexturesOnGPU(const RenderDeviceCapabilities &capabilities) {
        static const bool allowed = []() { const char *e = std::getenv("ROGUESQ_GPU_TEXTURE_DECODE"); return (e == nullptr) || (e[0] != '0'); }();
        return allowed && capabilities.computeShaders;
    }

    // Must match RT64_TMEM_SLOTS in the slot shaders: the SPIR-V ones use 4 so a draw's textures fit 16 per stage.
    inline uint32_t TMEMSlotCount(bool d3d11) {
        return d3d11 ? 8 : 4;
    }

    struct OptimizerCacheSPIRV {
        respv::Shader rasterVS;
        respv::Shader rasterVSFlat;
        respv::Shader rasterPS;
        respv::Shader rasterPSMS;
        respv::Shader rasterPSDepth;
        respv::Shader rasterPSDepthMS;
        respv::Shader rasterPSFlatDepth;
        respv::Shader rasterPSFlatDepthMS;
        respv::Shader rasterPSFlat;
        respv::Shader rasterPSFlatMS;

        void initialize(bool textureSlots);
    };

    struct PipelineCreation {
        RenderDevice *device;
        const RenderPipelineLayout *pipelineLayout;
        const RenderShader *vertexShader;
        const RenderShader *pixelShader;
        bool alphaBlend;
        bool culling;
        bool NoN;
        bool zCmp;
        bool zUpd;
        bool cvgAdd;
        bool usesHDR;
        std::vector<RenderSpecConstant> specConstants;
        RenderMultisampling multisampling;
    };

    struct RasterShaderText {
        std::string vertexShader;
        std::string pixelShader;
    };

    struct RasterShader {
        ShaderDescription desc;
        RenderDevice *device;
        std::unique_ptr<RenderPipeline> pipeline;

        RasterShader(RenderDevice *device, const ShaderDescription &desc, const RenderPipelineLayout *pipelineLayout, RenderShaderFormat shaderFormat, const RenderMultisampling &multisampling, 
            const ShaderCompiler *shaderCompiler, const OptimizerCacheSPIRV *optimizerCacheSPIRV);

        ~RasterShader();
        static RasterShaderText generateShaderText(const ShaderDescription &desc, bool multisampling);
        static std::unique_ptr<RenderPipeline> createPipeline(const PipelineCreation &c);
        static RenderMultisampling generateMultisamplingPattern(uint32_t sampleCount, bool sampleLocationsSupported);
    };

    struct RasterShaderUber {
        static const uint64_t RasterVSLibraryHash;
        static const uint64_t RasterPSLibraryHash;

        std::unique_ptr<RenderPipeline> pipelines[8];
        std::unique_ptr<RenderPipeline> postBlendDitherNoiseAddPipeline;
        std::unique_ptr<RenderPipeline> postBlendDitherNoiseSubPipeline;
        std::unique_ptr<RenderPipeline> postBlendDitherNoiseSubNegativePipeline;
        std::mutex firstPipelineMutex;
        std::condition_variable firstPipelineCondition;
        bool pipelinesCreated = false;
        std::unique_ptr<RenderPipelineLayout> pipelineLayout;
        std::vector<std::vector<PipelineCreation>> pipelineThreadCreations;
        std::vector<std::unique_ptr<std::thread>> pipelineThreads;
        std::unique_ptr<RenderShader> vertexShader;
        std::unique_ptr<RenderShader> pixelShader;

        RasterShaderUber(RenderDevice *device, RenderShaderFormat shaderFormat, const RenderMultisampling &multisampling, const ShaderLibrary *shaderLibrary, uint32_t threadCount, bool textureSlots, bool buildPipelines = true);
        ~RasterShaderUber();
        void threadCreatePipelines(uint32_t threadIndex);
        void waitForPipelineCreation();
        uint32_t pipelineStateIndex(bool zCmp, bool zUpd, bool cvgAdd) const;
        const RenderPipeline *getPipeline(bool zCmp, bool zUpd, bool cvgAdd) const;
    };
};
