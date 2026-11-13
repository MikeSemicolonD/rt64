//
// RT64
//

#pragma once

#include <atomic>

#include "common/rt64_user_configuration.h"
#include "hle/rt64_vi.h"

#include "rt64_descriptor_sets.h"
#include "rt64_shader_library.h"

namespace RT64 {
    // Fraction of the window height the presented picture is moved up by (the host sets it while an on-screen keyboard covers the bottom).
    extern std::atomic<float> presentShiftY;

    struct VIRenderer {
        std::unique_ptr<VideoInterfaceDescriptorSet> descriptorSet;
        const RenderSampler *descriptorSetSampler = nullptr;

        struct RenderParams {
            RenderDevice *device = nullptr;
            RenderCommandList *commandList = nullptr;
            RenderTexture *texture = nullptr;
            const RenderSwapChain *swapChain = nullptr;
            const ShaderLibrary *shaderLibrary = nullptr;
            RenderFormat textureFormat = RenderFormat::UNKNOWN;
            hlslpp::float2 resolutionScale;
            uint32_t downsamplingScale = 0;
            uint32_t textureWidth = 0;
            uint32_t textureHeight = 0;
            UserConfiguration::Filtering filtering = UserConfiguration::Filtering::Linear;
            const VI *vi = nullptr;
            bool removeBlackBorders = false;
            bool letterbox43 = false;
        };

        VIRenderer();
        ~VIRenderer();
        void render(const RenderParams &p);
        // letterbox43: a letterboxed frame (a cutscene's short VI region) is shown inside a 4:3 screen, as on a TV, instead of filling the window.
        static void getViewportAndScissor(const RenderSwapChain *swapChain, const VI &vi, hlslpp::float2 resolutionScale, uint32_t downsamplingScale, bool removeBlackBorders, bool letterbox43, RenderViewport &viewport, RenderRect &scissor);
    };
};