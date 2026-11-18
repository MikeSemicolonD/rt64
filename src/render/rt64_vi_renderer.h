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

    // A picture wider than the window fills its height and loses its sides instead of showing bars (the host sets it for letterboxed menus).
    extern std::atomic<bool> letterboxCrop;

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
            // Draw into a target of this size instead of the swap chain's (VR's virtual screen); 0 = the swap chain.
            uint32_t targetWidth = 0;
            uint32_t targetHeight = 0;
            // Fill the target and crop the overflow instead of fitting inside it (the window's view of a VR eye).
            bool cover = false;
        };

        VIRenderer();
        ~VIRenderer();
        void render(const RenderParams &p);
        static void getViewportAndScissor(const RenderSwapChain *swapChain, const VI &vi, hlslpp::float2 resolutionScale, uint32_t downsamplingScale, bool removeBlackBorders, RenderViewport &viewport, RenderRect &scissor);
        static void getViewportAndScissor(hlslpp::float2 fullWindow, const VI &vi, hlslpp::float2 resolutionScale, uint32_t downsamplingScale, bool removeBlackBorders, RenderViewport &viewport, RenderRect &scissor);
    };
};