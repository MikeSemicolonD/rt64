//
// RT64
//

#include "rt64_vi_renderer.h"

#include <cstdlib>

#include "shared/rt64_hlsl.h"
#include "shared/rt64_video_interface.h"

namespace RT64 {
    // VIRenderer

    VIRenderer::VIRenderer() { }

    VIRenderer::~VIRenderer() { }

    inline hlslpp::float2 computeHDSize(hlslpp::float2 sdSize, hlslpp::float2 resolutionScale, uint32_t downsamplingScale) {
        return (sdSize * resolutionScale) / float(downsamplingScale);
    }

    inline hlslpp::float2 fromSDtoHD(hlslpp::float2 coordinate, hlslpp::float2 sdSize, hlslpp::float2 hdSize) {
        const hlslpp::float2 relativeScale = hdSize / sdSize;
        return coordinate * relativeScale;
    }

    inline hlslpp::float2 fromHDtoWindow(hlslpp::float2 coordinate, hlslpp::float2 hdSize, hlslpp::float2 windowSize) {
        const hlslpp::float2 hdCenter = hdSize / 2;
        const hlslpp::float2 windowCenter = windowSize / 2;
        const hlslpp::float2 relativeCoordinate = { coordinate.x - hdCenter.x, coordinate.y - hdCenter.y };

        // Window is wider than virtual HD TV, we do pillarboxing.
        float relativeScale;
        if ((windowSize.x / windowSize.y) > (hdSize.x / hdSize.y)) {
            relativeScale = windowSize.y / hdSize.y;
        }
        // Window is taller than virtual HD TV, we do letterboxing.
        else {
            relativeScale = windowSize.x / hdSize.x;
        }

        return windowCenter + relativeCoordinate * relativeScale;
    }

    void VIRenderer::render(const RenderParams &p) {
        const ShaderRecord *shader = nullptr;
        const RenderSampler *sampler = nullptr;
        switch (p.filtering) {
        case UserConfiguration::Filtering::Nearest:
            shader = &p.shaderLibrary->videoInterfaceNearest;
            sampler = p.shaderLibrary->samplerLibrary.nearest.borderBorder.get();
            break;
        case UserConfiguration::Filtering::AntiAliasedPixelScaling:
            shader = &p.shaderLibrary->videoInterfacePixel;
            sampler = p.shaderLibrary->samplerLibrary.linear.borderBorder.get();
            break;
        case UserConfiguration::Filtering::Linear:
        default:
            shader = &p.shaderLibrary->videoInterfaceLinear;
            sampler = p.shaderLibrary->samplerLibrary.linear.borderBorder.get();
            break;
        }

        if ((descriptorSet == nullptr) || (descriptorSetSampler != sampler)) {
            descriptorSet = std::make_unique<VideoInterfaceDescriptorSet>(sampler, p.device);
            descriptorSetSampler = sampler;
        }

        descriptorSet->setTexture(descriptorSet->gInput, p.texture, RenderTextureLayout::SHADER_READ);

        RenderViewport viewport;
        RenderRect scissor;
        getViewportAndScissor(p.swapChain, *p.vi, p.resolutionScale, p.downsamplingScale, p.removeBlackBorders, viewport, scissor);
        p.commandList->setViewports(viewport);
        p.commandList->setScissors(scissor);

        // ROGUESQ_VI_FILTER=<radius>: N64 VI-style soften applied in the present pass (VideoInterfacePS).
        // Radius is in color-target texels (~resolutionScale texels == one native pixel); 0 = off (default,
        // unchanged behavior). See plans/model-texture VI-filter note; the game point-samples faithfully,
        // this reintroduces the console's whole-frame softening that RT64 otherwise omits.
        static const float s_viFilter = []() {
            const char *v = std::getenv("ROGUESQ_VI_FILTER");
            return (v && *v) ? float(std::atof(v)) : 0.0f;
        }();

        // ROGUESQ_VI_GAMMA=<g>: override the present gamma (VideoInterfacePS applies pow(color, gamma)).
        // The game leaves VI gamma_enable off (VI::gamma()==1.0), so the raw digital framebuffer displays
        // linearly and looks darker than the game did on real N64 analog/CRT output (proven: our raw FB ==
        // PJ64 raw FB; ideal-initial.PNG is brighter; gamma 1/2.2 on our FB matches it). A value < 1
        // brightens (recover shadow detail on model textures); 0/unset keeps the faithful VI gamma.
        static const float s_gamma = []() {
            const char *v = std::getenv("ROGUESQ_VI_GAMMA");
            return (v && *v) ? float(std::atof(v)) : 0.0f;
        }();

        // ROGUESQ_VI_OVERSCAN=<native pixels>: pull the right edge of the sampled region in by this
        // many native pixels so the composite never samples the render target's stale outermost
        // column (pooled targets keep prior-frame pixels there -> flickering saturated sliver on
        // dark full-frame screens). Default 1; 0 disables (A/B against the raw edge).
        static const float s_overscan = []() {
            const char *v = std::getenv("ROGUESQ_VI_OVERSCAN");
            return (v && *v) ? float(std::atof(v)) : 1.0f;
        }();

        const hlslpp::uint2 nativeSize = p.vi->fbSize();
        interop::VideoInterfaceCB pushConstants;
        pushConstants.videoResolution = computeHDSize(hlslpp::float2(nativeSize), p.resolutionScale, p.downsamplingScale);
        pushConstants.textureResolution = { float(p.textureWidth), float(p.textureHeight) };
        pushConstants.gamma = (s_gamma > 0.0f) ? s_gamma : p.vi->gamma();
        pushConstants.viFilter = s_viFilter;
        // Inset = (overscan native columns) * texels-per-native + half a texel. The +0.5 guard makes
        // the linear tap land on the last texel of the KEPT region instead of blending back into the
        // stale outer column (a plain 1-native-px inset still bleeds ~half the garbage through).
        const float texelsPerNativeX = (nativeSize.x > 0) ? (float(p.textureWidth) / float(nativeSize.x)) : 0.0f;
        pushConstants.overscan = { (s_overscan > 0.0f) ? (s_overscan * texelsPerNativeX + 0.5f) : 0.0f, 0.0f };

        p.commandList->setPipeline(shader->pipeline.get());
        p.commandList->setGraphicsPipelineLayout(shader->pipelineLayout.get());
        p.commandList->setGraphicsDescriptorSet(descriptorSet->get(), 0);
        p.commandList->setGraphicsPushConstants(0, &pushConstants);
        p.commandList->setVertexBuffers(0, nullptr, 0, nullptr);
        p.commandList->drawInstanced(3, 1, 0, 0);
    }

    void VIRenderer::getViewportAndScissor(const RenderSwapChain *swapChain, const VI &vi, hlslpp::float2 resolutionScale, uint32_t downsamplingScale, bool removeBlackBorders, RenderViewport &viewport, RenderRect &scissor) {
        // We define three different coordinate spaces to work with to translate the VI parameters into the Window.
        //
        // VideoSD: This corresponds to the SD TV Scanline space, which is what the VI natively works on.
        // 
        // VideoHD: This corresponds to what the imaginary "HD" TV would be if it supported the amount of scanlines
        // desired by the resolution scale (divided by the downsampling scale) and a wider aspect ratio.
        // 
        // Window: The native coordinate space of the device's render target.
        //
        // The provided buffer doesn't necessarily have the same dimensions that the VI will sample to display it 
        // on the screen. To work around that, the viewport the buffer will be drawn in will be expanded so only
        // the region of interest is rendered. A scissor will cut it off correctly according to the coordinates
        // specified by the VI.
        const hlslpp::float2 sdSize = removeBlackBorders ? hlslpp::float2(vi.fbSize()) : hlslpp::float2(320.0f, 240.0f);
        hlslpp::float2 hdSize = computeHDSize(sdSize, resolutionScale, downsamplingScale);
        if (removeBlackBorders) {
            hdSize.x *= vi.pixelAspect();
        }

        const hlslpp::float2 windowSize = { float(swapChain->getWidth()), float(swapChain->getHeight()) };

        // Query the VI for the current rendering area.
        hlslpp::float4 viViewRect = vi.viewRectangle() * sdSize.xyxy;
        hlslpp::float4 viCropRect = vi.cropRectangle() * sdSize.xyxy;

        // Scale all the rectangles to the space of the Window.
        hlslpp::float2 topLeftViewport = fromSDtoHD({ float(viViewRect.x), float(viViewRect.y) }, sdSize, hdSize);
        hlslpp::float2 bottomRightViewport = fromSDtoHD({ float(viViewRect.x + viViewRect.z), float(viViewRect.y + viViewRect.w) }, sdSize, hdSize);
        topLeftViewport = fromHDtoWindow(topLeftViewport, hdSize, windowSize);
        bottomRightViewport = fromHDtoWindow(bottomRightViewport, hdSize, windowSize);

        hlslpp::float2 topLeftScissor = fromSDtoHD({ float(viCropRect.x), float(viCropRect.y) }, sdSize, hdSize);
        hlslpp::float2 bottomRightScissor = fromSDtoHD({ float(viCropRect.x + viCropRect.z), float(viCropRect.y + viCropRect.w) }, sdSize, hdSize);
        topLeftScissor = fromHDtoWindow(topLeftScissor, hdSize, windowSize);
        bottomRightScissor = fromHDtoWindow(bottomRightScissor, hdSize, windowSize);

        viewport = RenderViewport(topLeftViewport.x, topLeftViewport.y, bottomRightViewport.x - topLeftViewport.x, bottomRightViewport.y - topLeftViewport.y);
        scissor = RenderRect(lround(topLeftScissor.x), lround(topLeftScissor.y), lround(bottomRightScissor.x), lround(bottomRightScissor.y));
    }
};