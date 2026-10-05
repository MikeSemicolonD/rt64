//
// RT64
//

#include "rt64_framebuffer_renderer.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_set>

#include "../include/rt64_extended_gbi.h"

#include "common/rt64_elapsed_timer.h"
#include "common/rt64_math.h"
#include "hle/rt64_color_converter.h"
#include "gbi/rt64_f3d.h"
#include "shared/rt64_framebuffer_params.h"
#include "shared/rt64_raster_params.h"
#include "shared/rt64_rs64_lights_cb.h"
#include "shared/rt64_rs64_shadows_cb.h"
#include "shared/rt64_rs64_shadow_blur_cb.h"
#include "shared/rt64_rs64_fog_shafts_cb.h"
#include "shared/rt64_rs64_cutout.h"

static_assert(sizeof(interop::RS64CutoutDraw) == sizeof(rs64lights::CutoutEntry), "cutout table layout");
#include "hle/rt64_rs64_lights.h"

#include "rt64_descriptor_sets.h"
#include "rt64_render_worker.h"

// TODO: Move to shared.

namespace interop {
    struct BicubicCB {
        uint2 InputResolution;
        uint2 OutputResolution;
    };

    struct HistogramAverageCB {
        uint pixelCount;
        float minLuminance;
        float luminanceRange;
        float timeDelta;
        float tau;
    };

    struct HistogramSetCB {
        float luminanceValue;
    };

    struct LuminanceHistogramCB {
        uint inputWidth;
        uint inputHeight;
        float minLuminance;
        float oneOverLuminanceRange;
    };

    struct TextureCB {
        uint2 TextureSize;
        float2 TexelSize;
    };
};

namespace RT64 {
    // ROGUESQ_LOG_BATCH: per-vertex renderIndex stamp collisions (a vertex stamped
    // with two different object indices). Order-safe coalescing assumes vertices are
    // not shared across GameCalls; a nonzero total here would break that assumption.
    static std::atomic<uint64_t> s_ri_collisions{0};

    // Helper functions.

    RenderRect convertFixedRect(FixedRect rect, hlslpp::float2 resScale, int32_t fbWidth, float aspectRatioScale, float extOriginPercentage, int32_t horizontalMisalignment, uint16_t leftOrigin, uint16_t rightOrigin) {
        if (!rect.isNull()) {
            auto computeOrigin = [=](uint16_t origin) {
                if (origin < G_EX_ORIGIN_NONE) {
                    return std::lround(((fbWidth * origin) / G_EX_ORIGIN_RIGHT) * extOriginPercentage + (fbWidth / 2) * (1.0f - extOriginPercentage));
                }
                else {
                    return fbWidth / 2L;
                }
            };

            auto correctMisalignment = [=](int32_t coord, uint16_t origin) {
                if (origin < G_EX_ORIGIN_NONE) {
                    return int32_t(coord - (coord % std::lround(resScale[1]))) - horizontalMisalignment;
                }
                else {
                    return coord;
                }
            };

            int32_t left = static_cast<int32_t>(std::floor((computeOrigin(leftOrigin) + (rect.left(true) - computeOrigin(leftOrigin)) * aspectRatioScale) * resScale.x));
            int32_t right = static_cast<int32_t>(std::ceil((computeOrigin(rightOrigin) + (rect.right(true) - computeOrigin(rightOrigin)) * aspectRatioScale) * resScale.x));
            int32_t top = lround(rect.top(true) * resScale.y);
            int32_t bottom = lround(rect.bottom(true) * resScale.y);
            left = correctMisalignment(left, leftOrigin);
            right = correctMisalignment(right, rightOrigin);
            return RenderRect(left, top, right, bottom);
        }
        else {
            return RenderRect(0, 0, 0, 0);
        }
    }
    
    RenderViewport convertViewportRect(FixedRect rect, hlslpp::float2 resScale, int32_t fbWidth, float aspectRatioScale, float extOriginPercentage, float horizontalMisalignment, uint16_t leftOrigin, uint16_t rightOrigin) {
        auto computeOrigin = [=](uint16_t origin) {
            if (origin < G_EX_ORIGIN_NONE) {
                return ((fbWidth * origin) / G_EX_ORIGIN_RIGHT) * extOriginPercentage + (fbWidth / 2) * (1.0f - extOriginPercentage);
            }
            else {
                return float(fbWidth) / 2;
            }
        };
        
        auto correctMisalignment = [=](float coord, uint16_t origin) {
            if (origin < G_EX_ORIGIN_NONE) {
                return (coord - std::fmod(coord, resScale[1])) - horizontalMisalignment;
            }
            else {
                return coord;
            }
        };
        
        float left = std::round((computeOrigin(leftOrigin) + (rect.left(true) - computeOrigin(leftOrigin)) * aspectRatioScale) * resScale.x);
        float right = std::round((computeOrigin(rightOrigin) + (rect.right(true) - computeOrigin(rightOrigin)) * aspectRatioScale) * resScale.x);
        float top = std::round(rect.top(true) * resScale.y);
        float bottom = std::round(rect.bottom(true) * resScale.y);
        left = correctMisalignment(left, leftOrigin);
        right = correctMisalignment(right, rightOrigin);
        return RenderViewport(left, top, right - left, bottom - top);
    }

    hlslpp::float3 viewPositionFrom(hlslpp::float4x4 viewI) {
        return viewI[3].xyz;
    }

    hlslpp::float3 viewDirectionFrom(hlslpp::float4x4 viewI) {
        return hlslpp::normalize(viewI[2].xyz);
    }

    RenderColor toRenderColor(hlslpp::float4 v) {
        return { v.x, v.y, v.z, v.w };
    }

    static RenderRect viewportScissorIntersection(const RenderViewport &viewport, const RenderRect &scissor) {
        return RenderRect{
            std::max(static_cast<int32_t>(std::floor(viewport.x)), scissor.left),
            std::max(static_cast<int32_t>(std::floor(viewport.y)), scissor.top),
            std::min(static_cast<int32_t>(std::ceil(viewport.x + viewport.width)), scissor.right),
            std::min(static_cast<int32_t>(std::ceil(viewport.y + viewport.height)), scissor.bottom)
        };
    }

    // RasterScene

    RasterScene::RasterScene() { }

    // FramebufferRenderer
    
    FramebufferRenderer::FramebufferRenderer(RenderWorker *worker, bool rtSupport, UserConfiguration::GraphicsAPI graphicsAPI, const ShaderLibrary *shaderLibrary) {
        assert(worker != nullptr);

        this->shaderLibrary = shaderLibrary;

        frameParams.frameCount = 0;
        frameParams.viewUbershaders = false;
        frameParams.ditherNoiseStrength = 1.0f;

        shaderUploader = std::make_unique<BufferUploader>(worker->device);
        descCommonSet = std::make_unique<FramebufferRendererDescriptorCommonSet>(shaderLibrary->samplerLibrary, worker->device->getCapabilities().raytracing, worker->device);

#   if RT_ENABLED
        if (rtSupport) {
            this->rtSupport = rtSupport;
            rtResources = std::make_unique<RaytracingResources>(worker, graphicsAPI);
    }
#   endif
    }

    FramebufferRenderer::~FramebufferRenderer() {
        dummyColorTargetView.reset();
        dummyDepthTargetView.reset();
        dummyColorTarget.reset();
        dummyDepthTarget.reset();
    }
    
    void FramebufferRenderer::resetFramebuffers(RenderWorker *worker, bool ubershadersVisible, float ditherNoiseStrength, const RenderMultisampling &multisampling) {
        instanceDrawCallVector.clear();
        hitGroupVector.clear();
        renderIndicesVector.clear();
        renderIndexData.clear();
        rawRenderIndexData.clear();
        rspSmoothNormalVector.clear();
        frameParams.viewUbershaders = ubershadersVisible;
        frameParams.ditherNoiseStrength = ditherNoiseStrength;
        // Diagnostic (ROGUESQ_DESC_DIAG): per-frame framebuffer-pair count. The cinematic's
        // CIMG churn inflates this; framebufferVector grows to its high-water (2 descriptor
        // sets each), which is the real CBV view-heap leak driver.
        {
            static int s_fc = -1;
            if (s_fc == -1) { const char *e = std::getenv("ROGUESQ_DESC_DIAG"); s_fc = (e && e[0] && e[0] != '0') ? 0 : -2; }
            if (s_fc >= 0) { static uint32_t s_max = 0; if (framebufferCount > s_max) { s_max = framebufferCount; fprintf(stderr, "[fbpair] frame fbPairCount=%u (new max) vecSize=%zu\n", framebufferCount, framebufferVector.size()); fflush(stderr); } }
        }
        framebufferCount = 0;

        // Create dummy color target if it hasn't been created yet.
        if (dummyColorTarget == nullptr) {
            RenderTextureDesc dummyColorDesc = RenderTextureDesc::ColorTarget(4, 4, RenderTarget::colorBufferFormat(shaderLibrary->usesHDR), multisampling);
            dummyColorTarget = worker->device->createTexture(dummyColorDesc);
            dummyColorTarget->setName("Framebuffer Renderer Color Dummy");
            dummyColorTargetView = dummyColorTarget->createTextureView(RenderTextureViewDesc::Texture2D(dummyColorDesc.format));
            dummyColorTargetTransitioned = false;
        }

        // Create dummy depth target if it hasn't been created yet.
        if (dummyDepthTarget == nullptr) {
            RenderTextureDesc dummyDepthDesc = RenderTextureDesc::DepthTarget(4, 4, RenderFormat::D32_FLOAT, multisampling);
            dummyDepthTarget = worker->device->createTexture(dummyDepthDesc);
            dummyDepthTarget->setName("Framebuffer Renderer Depth Dummy");
            dummyDepthTargetView = dummyDepthTarget->createTextureView(RenderTextureViewDesc::Texture2D(dummyDepthDesc.format));
            dummyDepthTargetTransitioned = false;
        }
    }

#if RT_ENABLED
    void FramebufferRenderer::resetRaytracing(RaytracingShaderCache *rtShaderCache, const RenderTexture *blueNoiseTexture) {
        assert(rtResources != nullptr);
        assert(rtShaderCache != nullptr);
        assert(blueNoiseTexture != nullptr);

        const int32_t stateIndex = rtShaderCache->getActiveState();
        rtState = &rtShaderCache->states[stateIndex];
        rtPipelineLayout = rtShaderCache->pipelineLayout.get();
        rtResources->resetBottomLevelAS();

        this->blueNoiseTexture = blueNoiseTexture;
    }
#endif

    void FramebufferRenderer::updateTextureCache(TextureCache *textureCache) {
        const std::unique_lock<std::mutex> textureMapLock(textureCache->textureMapMutex);
        textureCacheVersions = textureCache->textureMap.versions;
        textureCacheTextures = textureCache->textureMap.textures;
        textureCacheTextureReplacements = textureCache->textureMap.textureReplacements;
        textureCacheFreeSpaces = textureCache->textureMap.freeSpaces;
        textureCacheSize = static_cast<uint32_t>(textureCacheTextures.size());
        textureCacheGlobalVersion = textureCache->textureMap.globalVersion;
        textureCacheReplacementMapEnabled = textureCache->textureMap.replacementMapEnabled;
        dynamicTextureViewVector.clear();
        dynamicTextureBarrierVector.clear();
    }

    void FramebufferRenderer::createGPUTiles(const DrawCallTile *callTiles, uint32_t callTileCount, interop::GPUTile *dstGPUTiles, const FramebufferManager *fbManager, 
        TextureCache *textureCache, uint64_t submissionFrame)
    {
        for (uint32_t i = 0; i < callTileCount; i++) {
            const DrawCallTile &callTile = callTiles[i];
            if (!callTile.valid) {
                continue;
            }
            
            interop::GPUTile &gpuTile = dstGPUTiles[i];
            if (callTile.tileCopyUsed) {
                const auto &it = fbManager->tileCopies.find(callTile.tmemHashOrID);
                if (it != fbManager->tileCopies.end()) {
                    const FramebufferManager::TileCopy &tileCopy = it->second;
                    gpuTile.tcScale.x = static_cast<float>(tileCopy.usedWidth) / static_cast<float>(callTile.tileCopyWidth);
                    gpuTile.tcScale.y = static_cast<float>(tileCopy.usedHeight) / static_cast<float>(callTile.tileCopyHeight);
                    gpuTile.ulScale.x = tileCopy.ulScaleS ? gpuTile.tcScale.x : 1.0f;
                    gpuTile.ulScale.y = tileCopy.ulScaleT ? gpuTile.tcScale.y : 1.0f;
                    gpuTile.texelShift = tileCopy.texelShift;
                    gpuTile.texelMask = tileCopy.texelMask;
                    gpuTile.textureIndex = getTextureIndex(tileCopy);
                    gpuTile.textureDimensions = interop::float3(float(tileCopy.textureWidth), float(tileCopy.textureHeight), 1.0f);
                    gpuTile.flags.alphaIsCvg = !callTile.reinterpretTile;
                    gpuTile.flags.highRes = true;
                    gpuTile.flags.fromCopy = true;
                    gpuTile.flags.rawTMEM = false;
                    gpuTile.flags.hasMipmaps = false;
                }
            }
            else {
                // Retrieve the texture from the cache or use a blank texture if not found.
                uint32_t textureIndex = 0;
                bool textureReplaced = false;
                bool hasMipmaps = false;
                bool shiftedByHalf = false;
                textureCache->useTexture(callTile.tmemHashOrID, submissionFrame, textureIndex, gpuTile.tcScale, gpuTile.textureDimensions, textureReplaced, hasMipmaps, shiftedByHalf);

                // Describe the GPU tile for a regular texture.
                gpuTile.ulScale.x = gpuTile.tcScale.x;
                gpuTile.ulScale.y = gpuTile.tcScale.y;
                gpuTile.texelShift = { 0, 0 };
                gpuTile.texelMask = { UINT_MAX, UINT_MAX };
                gpuTile.textureIndex = textureIndex;
                gpuTile.flags.alphaIsCvg = false;
                gpuTile.flags.highRes = textureReplaced;
                gpuTile.flags.fromCopy = false;
                gpuTile.flags.rawTMEM = !textureReplaced && callTile.rawTMEM;
                gpuTile.flags.hasMipmaps = hasMipmaps;
                gpuTile.flags.shiftedByHalf = shiftedByHalf;
            }
        }
    }

    uint32_t FramebufferRenderer::getDestinationIndex() {
        uint32_t dstIndex;
        if (textureCacheFreeSpaces.empty()) {
            dstIndex = textureCacheSize++;
        }
        else {
            dstIndex = textureCacheFreeSpaces.back();
            textureCacheFreeSpaces.pop_back();
        }

        return dstIndex;
    }
    
    uint32_t FramebufferRenderer::getTextureIndex(RenderTarget *renderTarget) {
        assert(renderTarget != nullptr);

        uint32_t dstIndex = getDestinationIndex();
        dynamicTextureViewVector.emplace_back(DynamicTextureView{ renderTarget->getResolvedTexture(), dstIndex, renderTarget->getResolvedTextureView()});
        return dstIndex;
    }
    
    uint32_t FramebufferRenderer::getTextureIndex(const FramebufferManager::TileCopy &tileCopy) {
        assert(tileCopy.texture != nullptr);

        uint32_t dstIndex = getDestinationIndex();
        dynamicTextureViewVector.emplace_back(DynamicTextureView{ tileCopy.texture.get(), dstIndex, nullptr });
        dynamicTextureBarrierVector.emplace_back(RenderTextureBarrier(tileCopy.texture.get(), RenderTextureLayout::SHADER_READ));
        return dstIndex;
    }
    
    void FramebufferRenderer::updateShaderDescriptorSet(RenderWorker *worker, const DrawBuffers *drawBuffers, const OutputBuffers *outputBuffers, const bool raytracingEnabled) {
        assert(worker != nullptr);
        assert(drawBuffers != nullptr);
        
        // Size the boundless texture descriptor set at a stable ceiling (the shader's
        // UpperRange=8192) and build it ONCE, rather than rebuilding it ever-larger each
        // time textureCacheSize grows. RS64's cinematic churns many tile-copies/dynamic
        // views per frame (getDestinationIndex bumps textureCacheSize), so the old
        // ((textureCacheSize+1)*3)/2 path rebuilt the set constantly; because make_unique
        // allocates the new (larger) set before the old one is freed, each rebuild stranded
        // the old block at a higher CBV_SRV_UAV heap offset → the 64K view heap climbed to
        // exhaustion and getCPUHandleAt asserted (the cinematic descriptor-heap crash).
        // A fixed-capacity set is created once and reused (setTexture updates entries in
        // place). Opt out to the legacy growth with ROGUESQ_FBTEX_FIXED_CAP=0.
        static int s_fixed_cap = -1;
        if (s_fixed_cap < 0) { const char *e = std::getenv("ROGUESQ_FBTEX_FIXED_CAP"); s_fixed_cap = (e && e[0] == '0') ? 0 : 1; }
        const bool createSet = (descTextureSet == nullptr) || (descTextureSet->textureCacheSize < (textureCacheSize + 1));
        if (createSet) {
            const uint32_t legacyCap = ((textureCacheSize + 1) * 3) / 2;
            const uint32_t fixedCap = (textureCacheSize + 1) > (uint32_t)FramebufferRendererDescriptorTextureSet::UpperRange
                ? legacyCap : (uint32_t)FramebufferRendererDescriptorTextureSet::UpperRange;
            static int s_dd = -1;
            if (s_dd == -1) { const char *e = std::getenv("ROGUESQ_DESC_DIAG"); s_dd = (e && e[0] && e[0] != '0') ? 0 : -2; }
            if (s_dd >= 0 && s_dd < 200) { ++s_dd; fprintf(stderr, "[desc-texset] rebuild textureCacheSize=%u cap=%u\n", textureCacheSize, s_fixed_cap ? fixedCap : legacyCap); fflush(stderr); }
            descTextureSet = std::make_unique<FramebufferRendererDescriptorTextureSet>(worker->device, s_fixed_cap ? fixedCap : legacyCap);
        }

        if (createSet || (descriptorTextureReplacementMapEnabled != textureCacheReplacementMapEnabled)) {
            descriptorTextureVersions.clear();
            descriptorTextureGlobalVersion = 0;
            descriptorTextureReplacementMapEnabled = textureCacheReplacementMapEnabled;
        }

#   if RT_ENABLED
        if (raytracingEnabled) {
            descCommonSet->setBuffer(descCommonSet->posBuffer, outputBuffers->worldPosBuffer.buffer.get(), outputBuffers->worldPosBuffer.allocatedSize);
            descCommonSet->setBuffer(descCommonSet->normBuffer, outputBuffers->worldNormBuffer.buffer.get(), outputBuffers->worldNormBuffer.allocatedSize);
            descCommonSet->setBuffer(descCommonSet->velBuffer, outputBuffers->worldVelBuffer.buffer.get(), outputBuffers->worldVelBuffer.allocatedSize);
            descCommonSet->setBuffer(descCommonSet->genTexCoordBuffer, outputBuffers->genTexCoordBuffer.buffer.get(), outputBuffers->genTexCoordBuffer.allocatedSize);
            descCommonSet->setBuffer(descCommonSet->shadedColBuffer, outputBuffers->shadedColBuffer.buffer.get(), outputBuffers->shadedColBuffer.allocatedSize);
            descCommonSet->setBuffer(descCommonSet->srcFogIndices, drawBuffers->fogIndicesBuffer.get(), drawBuffers->fogIndicesBuffer.allocatedSize);
            descCommonSet->setBuffer(descCommonSet->srcLightIndices, drawBuffers->lightIndicesBuffer.get(), drawBuffers->lightIndicesBuffer.allocatedSize);
            descCommonSet->setBuffer(descCommonSet->srcLightCounts, drawBuffers->lightCountsBuffer.get(), drawBuffers->lightCountsBuffer.allocatedSize);
            descCommonSet->setBuffer(descCommonSet->indexBuffer, drawBuffers->faceIndicesBuffer.get(), drawBuffers->faceIndicesBuffer.allocatedSize);
            descCommonSet->setBuffer(descCommonSet->RSPFogVector, drawBuffers->rspFogBuffer.get(), drawBuffers->rspFogBuffer.allocatedSize, RenderBufferStructuredView(sizeof(interop::RSPFog)));
            descCommonSet->setBuffer(descCommonSet->RSPLightVector, drawBuffers->rspLightsBuffer.get(), drawBuffers->rspLightsBuffer.allocatedSize, RenderBufferStructuredView(sizeof(interop::RSPLight)));
            descCommonSet->setTexture(descCommonSet->gViewDirection, rtResources->viewDirectionTexture.get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gShadingPosition, rtResources->shadingPositionTexture.get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gShadingNormal, rtResources->shadingNormalTexture.get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gShadingSpecular, rtResources->shadingSpecularTexture.get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gDiffuse, rtResources->diffuseTexture.get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gInstanceId, rtResources->instanceIdTexture.get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gDirectLightAccum, rtResources->directLightTexture[rtResources->swapBuffers ? 1 : 0].get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gIndirectLightAccum, rtResources->indirectLightTexture[rtResources->swapBuffers ? 1 : 0].get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gReflection, rtResources->reflectionTexture.get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gRefraction, rtResources->refractionTexture.get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gTransparent, rtResources->transparentTexture.get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gFlow, rtResources->flowTexture.get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gReactiveMask, rtResources->reactiveMaskTexture.get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gLockMask, rtResources->lockMaskTexture.get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gNormalRoughness, rtResources->normalRoughnessTexture[rtResources->swapBuffers ? 1 : 0].get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gDepth, rtResources->depthTexture[rtResources->swapBuffers ? 1 : 0].get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gPrevNormalRoughness, rtResources->normalRoughnessTexture[rtResources->swapBuffers ? 0 : 1].get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gPrevDepth, rtResources->depthTexture[rtResources->swapBuffers ? 0 : 1].get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gPrevDirectLightAccum, rtResources->directLightTexture[rtResources->swapBuffers ? 0 : 1].get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gPrevIndirectLightAccum, rtResources->indirectLightTexture[rtResources->swapBuffers ? 0 : 1].get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gFilteredDirectLight, rtResources->filteredDirectLightTexture[1].get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gFilteredIndirectLight, rtResources->filteredIndirectLightTexture[1].get(), RenderTextureLayout::GENERAL);

            const uint32_t hitBufferPixelCount = rtResources->textureWidth * rtResources->textureHeight * RaytracingResources::MaxHitQueries;
            descCommonSet->setBuffer(descCommonSet->gHitVelocityDistance, rtResources->hitVelocityDistanceBuffer.get(), hitBufferPixelCount * 16, rtResources->hitVelocityDistanceBufferView.get());
            descCommonSet->setBuffer(descCommonSet->gHitColor, rtResources->hitColorBuffer.get(), hitBufferPixelCount * 4, rtResources->hitColorBufferView.get());
            descCommonSet->setBuffer(descCommonSet->gHitNormalFog, rtResources->hitNormalFogBuffer.get(), hitBufferPixelCount * 8, rtResources->hitNormalFogBufferView.get());
            descCommonSet->setBuffer(descCommonSet->gHitInstanceId, rtResources->hitInstanceIdBuffer.get(), hitBufferPixelCount * 2, rtResources->hitInstanceIdBufferView.get());
            descCommonSet->setAccelerationStructure(descCommonSet->SceneBVH, rtResources->topLevelAS.get());
            descCommonSet->setBuffer(descCommonSet->SceneLights, rtResources->lightsBuffer.get(), sizeof(interop::PointLight) * std::max(rtResources->rtParams.lightsCount, 1U), RenderBufferStructuredView(sizeof(interop::PointLight)));
            descCommonSet->setBuffer(descCommonSet->interleavedRasters, interleavedRastersBuffer.get(), sizeof(interop::InterleavedRaster) * std::max(interleavedRastersCount, 1U), RenderBufferStructuredView(sizeof(interop::InterleavedRaster)));
            descCommonSet->setTexture(descCommonSet->gBlueNoise, blueNoiseTexture, RenderTextureLayout::SHADER_READ);
            descCommonSet->setBuffer(descCommonSet->instanceExtraParams, drawBuffers->extraParamsBuffer.get(), RenderBufferStructuredView(sizeof(interop::ExtraParams)));
            descCommonSet->setBuffer(descCommonSet->RtParams, rtResources->rtParamsBuffer.get(), sizeof(interop::RaytracingParams));
        }
#   endif

        descCommonSet->setBuffer(descCommonSet->FrParams, frameParamsBuffer.get(), sizeof(interop::FrameParams));
        descCommonSet->setBuffer(descCommonSet->instanceRenderIndices, renderIndicesBuffer.get(), RenderBufferStructuredView(sizeof(interop::RenderIndices)));
        descCommonSet->setBuffer(descCommonSet->instanceRDPParams, drawBuffers->rdpParamsBuffer.get(), RenderBufferStructuredView(sizeof(interop::RDPParams)));
        descCommonSet->setBuffer(descCommonSet->RDPTiles, drawBuffers->rdpTilesBuffer.get(), RenderBufferStructuredView(sizeof(interop::RDPTile)));
        descCommonSet->setBuffer(descCommonSet->GPUTiles, drawBuffers->gpuTilesBuffer.get(), RenderBufferStructuredView(sizeof(interop::GPUTile)));
        descCommonSet->setBuffer(descCommonSet->DynamicRenderParams, drawBuffers->renderParamsBuffer.get(), RenderBufferStructuredView(sizeof(interop::RenderParams)));

        // Make sure the versions vector matches the texture cache size.
        descriptorTextureVersions.resize(textureCacheSize, 0);

        // Update texture vector with static textures from the cache and dynamic resource views.
        if (descriptorTextureGlobalVersion != textureCacheGlobalVersion) {
            const uint32_t textureVersionSize = static_cast<uint32_t>(textureCacheVersions.size());
            for (uint32_t i = 0; i < textureVersionSize; i++) {
                if (textureCacheVersions[i] == descriptorTextureVersions[i]) {
                    continue;
                }

                descriptorTextureVersions[i] = textureCacheVersions[i];
                if (textureCacheTextures[i] == nullptr) {
                    continue;
                }

                if (textureCacheReplacementMapEnabled && (textureCacheTextureReplacements[i] != nullptr)) {
                    descTextureSet->setTexture(i, textureCacheTextureReplacements[i]->texture.get(), RenderTextureLayout::SHADER_READ);
                }
                else if (textureCacheTextures[i]->texture != nullptr) {
                    descTextureSet->setTexture(i, textureCacheTextures[i]->texture.get(), RenderTextureLayout::SHADER_READ);
                }
                else {
                    descTextureSet->setTexture(i, textureCacheTextures[i]->tmem.get(), RenderTextureLayout::SHADER_READ);
                }
            }

            descriptorTextureGlobalVersion = textureCacheGlobalVersion;
        }

        for (const DynamicTextureView &dynamicView : dynamicTextureViewVector) {
            descTextureSet->setTexture(dynamicView.dstIndex, dynamicView.texture, RenderTextureLayout::SHADER_READ, dynamicView.textureView);
            descriptorTextureVersions[dynamicView.dstIndex] = 0;
        }
    }

    void FramebufferRenderer::updateRSPSmoothNormalSet(RenderWorker *worker, const DrawBuffers *drawBuffers, const OutputBuffers *outputBuffers) {
        assert(worker != nullptr);

        if (smoothDescSet == nullptr) {
            smoothDescSet = std::make_unique<RSPSmoothNormalDescriptorSet>(worker->device);
        }

        smoothDescSet->setBuffer(smoothDescSet->srcWorldPos, outputBuffers->worldPosBuffer.buffer.get(), outputBuffers->worldPosBuffer.allocatedSize, RenderBufferStructuredView(sizeof(float) * 4));
        smoothDescSet->setBuffer(smoothDescSet->srcCol, drawBuffers->normalColorBuffer.get(), drawBuffers->normalColorBuffer.allocatedSize, RenderBufferStructuredView(sizeof(uint8_t) * 4));
        smoothDescSet->setBuffer(smoothDescSet->srcFaceIndices, drawBuffers->faceIndicesBuffer.get(), drawBuffers->faceIndicesBuffer.allocatedSize, RenderBufferStructuredView(sizeof(uint32_t)));
        smoothDescSet->setBuffer(smoothDescSet->dstWorldNorm, outputBuffers->worldNormBuffer.buffer.get(), outputBuffers->worldNormBuffer.allocatedSize, RenderBufferStructuredView(sizeof(float) * 4));
    }

    void FramebufferRenderer::updateRSPVertexTestZSet(RenderWorker *worker, const DrawBuffers *drawBuffers, const OutputBuffers *outputBuffers) {
        assert(worker != nullptr);

        if (vertexTestZSet == nullptr) {
            vertexTestZSet = std::make_unique<RSPVertexTestZDescriptorSet>(worker->device);
        }
        
        vertexTestZSet->setBuffer(vertexTestZSet->screenPos, outputBuffers->screenPosBuffer.buffer.get(), RenderBufferStructuredView(sizeof(float) * 4));
        vertexTestZSet->setBuffer(vertexTestZSet->srcFaceIndices, drawBuffers->faceIndicesBuffer.get(), drawBuffers->faceIndicesBuffer.allocatedSize, RenderBufferStructuredView(sizeof(uint32_t)));
        vertexTestZSet->setBuffer(vertexTestZSet->dstFaceIndices, outputBuffers->testZIndexBuffer.buffer.get(), outputBuffers->testZIndexBuffer.allocatedSize, RenderBufferStructuredView(sizeof(uint32_t)));
    }

    void FramebufferRenderer::updateShaderViews(RenderWorker *worker, const DrawBuffers *drawBuffers, const OutputBuffers *outputBuffers, const bool raytracingEnabled) {
        updateShaderDescriptorSet(worker, drawBuffers, outputBuffers, raytracingEnabled);
        updateRSPVertexTestZSet(worker, drawBuffers, outputBuffers);

#   if RT_ENABLED
        if (raytracingEnabled) {
            updateRSPSmoothNormalSet(worker, drawBuffers, outputBuffers);
            rtResources->updateShaderSets(worker, shaderLibrary);
        }
#   endif
    }

    bool FramebufferRenderer::submitDepthAccess(RenderWorker *worker, RenderFramebufferStorage *fbStorage, bool readOnly, bool &depthState) {
        if (depthState == readOnly) {
            return false;
        }
        
        RenderFramebuffer *renderFramebuffer = readOnly ? fbStorage->colorWriteDepthRead.get() : fbStorage->colorDepthWrite.get();
        const RenderTextureLayout depthReadState = RenderTextureLayout::DEPTH_READ;
        const RenderTextureLayout depthWriteState = RenderTextureLayout::DEPTH_WRITE;
        worker->commandList->barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(fbStorage->depthTarget->texture.get(), readOnly ? depthReadState : RenderTextureLayout::DEPTH_WRITE));
        worker->commandList->setFramebuffer(renderFramebuffer);
        depthState = readOnly;
        return true;
    }
    
    void FramebufferRenderer::submitRasterScene(RenderWorker *worker, const Framebuffer &framebuffer, RenderFramebufferStorage *fbStorage, const RasterScene &rasterScene, bool &depthState) {
        InstanceDrawCall::Type previousCallType = InstanceDrawCall::Type::Unknown;
        bool previousVertexTestZ = false;
        const RenderPipeline *previousPipeline = nullptr;
        RenderRect previousScissor;
        interop::RasterParams rasterParams;
        RenderDescriptorSet *descRealFbSet = framebuffer.descRealFbSet->get();
        RenderDescriptorSet *descDummyFbSet = framebuffer.descDummyFbSet->get();

        // ROGUESQ_LOG_BATCH — batching-headroom counters (summary emitted after
        // the loop). Measures how many setPipeline/setScissor binds a perfect
        // per-scene sort could remove (savable = sequential switches - distinct).
        static const bool s_log_batch = []{
            const char *e = std::getenv("ROGUESQ_LOG_BATCH");
            return e && *e && *e != '0';
        }();
        uint32_t bDraws = 0, bPipeSw = 0, bScisSw = 0, bRuns = 0;
        std::unordered_set<const void*> bDistinctPipe;
        std::vector<RenderRect> bDistinctScis;

        auto switchToGraphicsPipeline = [&]() {
            previousCallType = InstanceDrawCall::Type::Unknown;
            previousVertexTestZ = false;
            previousPipeline = nullptr;
            previousScissor = RenderRect();
            worker->commandList->setGraphicsPipelineLayout(rendererPipelineLayout);
            worker->commandList->setGraphicsDescriptorSet(descCommonSet->get(), 0);
            worker->commandList->setGraphicsDescriptorSet(descTextureSet->get(), 1);
            worker->commandList->setGraphicsDescriptorSet(descTextureSet->get(), 2);
            worker->commandList->setGraphicsDescriptorSet(depthState ? descRealFbSet : descDummyFbSet, 3);
            worker->commandList->setViewports(framebuffer.viewport);
        };

        auto switchToDepthRead = [&]() {
            if (submitDepthAccess(worker, fbStorage, true, depthState)) {
                worker->commandList->setGraphicsDescriptorSet(descRealFbSet, 3);
            }
        };

        auto switchToDepthWrite = [&]() {
            if (submitDepthAccess(worker, fbStorage, false, depthState)) {
                worker->commandList->setGraphicsDescriptorSet(descDummyFbSet, 3);
            }
        };

        auto drawCallTriangles = [&](const InstanceDrawCall &drawCall) {
            if (drawCall.type == InstanceDrawCall::Type::IndexedTriangles) {
                worker->commandList->drawIndexedInstanced(drawCall.triangles.faceCount * 3, 1, drawCall.triangles.indexStart, 0, 0);
            }
            else {
                worker->commandList->drawInstanced(drawCall.triangles.faceCount * 3, 1, drawCall.triangles.indexStart, 0);
            }
        };

        if (fbStorage->colorTarget != nullptr) {
            switchToGraphicsPipeline();
        }
        
        // ROGUESQ_LOG_PIPELINE (stage 2.5) — count rasterScene submissions
        // and total instances reaching this for-loop. If instance count is
        // way lower than [pipe-stage 2-pushDrawCall] (added on the queue
        // side), drawCalls are being pushed to rasterScenes that never get
        // submitRasterScene'd.
        {
            static const bool log_pipe = []{
                const char *e = std::getenv("ROGUESQ_LOG_PIPELINE");
                return e && *e && *e != '0';
            }();
            if (log_pipe) {
                static std::atomic<int> n_call{0};
                static std::atomic<uint64_t> n_inst{0};
                int n = ++n_call;
                n_inst.fetch_add(rasterScene.instanceIndices.size(),
                                 std::memory_order_relaxed);
                if (n <= 5 || (n % 200) == 0) {
                    std::fprintf(stderr,
                        "[pipe-stage 2.5-rasterIter] submits=%d total-instances=%llu (this submit=%zu)\n",
                        n, (unsigned long long)n_inst.load(),
                        rasterScene.instanceIndices.size());
                    std::fflush(stderr);
                }
            }
        }

        ElapsedTimer batchTimer; // ROGUESQ_LOG_BATCH: submitRasterScene recording time
        for (size_t sceneIdx = 0; sceneIdx < rasterScene.instanceIndices.size(); ++sceneIdx) {
            const uint32_t i = rasterScene.instanceIndices[sceneIdx];
            const InstanceDrawCall &drawCall = instanceDrawCallVector[i];

            // ROGUESQ_LOG_PIPELINE (stage 2.6) — per-instance classification,
            // per-frame counted. Reset on g_pipe_frame_idx advance.
            {
                static const bool log_pipe = []{
                    const char *e = std::getenv("ROGUESQ_LOG_PIPELINE");
                    return e && *e && *e != '0';
                }();
                if (log_pipe && drawCall.type != InstanceDrawCall::Type::FillRect) {
                    extern std::atomic<uint64_t> g_pipe_frame_idx;
                    static std::atomic<uint64_t> last_frame{(uint64_t)-1};
                    static std::atomic<int> n_in_frame{0};
                    static std::atomic<int> n_total{0};
                    uint64_t cur = g_pipe_frame_idx.load(std::memory_order_relaxed);
                    uint64_t prev = last_frame.exchange(cur, std::memory_order_relaxed);
                    if (prev != cur) {
                        int finalCount = n_in_frame.exchange(0, std::memory_order_relaxed);
                        if (finalCount > 0) {
                            std::fprintf(stderr,
                                "[pipe-stage 2.6-iter] frame=%llu FINAL=%d\n",
                                (unsigned long long)prev, finalCount);
                            std::fflush(stderr);
                        }
                    }
                    int nf = ++n_in_frame;
                    int nt = ++n_total;
                    if (nf == 1 || (nf % 50) == 0 || nt <= 5) {
                        const uint32_t lo = (uint32_t)(drawCall.triangles.shaderDesc.colorCombiner.L);
                        std::fprintf(stderr,
                            "[pipe-stage 2.6-iter] frame=%llu nf=%d total=%d type=%d L=0x%08X\n",
                            (unsigned long long)cur, nf, nt, (int)drawCall.type, lo);
                        std::fflush(stderr);
                    }
                }
            }

            switch (drawCall.type) {
            case InstanceDrawCall::Type::IndexedTriangles:
            case InstanceDrawCall::Type::RawTriangles:
            case InstanceDrawCall::Type::RegularRect: {
                // ROGUESQ: skip draws with null colorTarget instead of
                // asserting. Some workloads route instances through a
                // rasterScene whose fbStorage has no allocated colorTarget
                // (the original assert killed the game on these).
                if (fbStorage->colorTarget == nullptr) {
                    static const bool log_pipe = []{
                        const char *e = std::getenv("ROGUESQ_LOG_PIPELINE");
                        return e && *e && *e != '0';
                    }();
                    if (log_pipe) {
                        static std::atomic<int> n_skip{0};
                        int n = ++n_skip;
                        if (n <= 5 || (n % 200) == 0) {
                            std::fprintf(stderr,
                                "[skip-null-rt] type=%d L=0x%08X (skipped #%d)\n",
                                (int)drawCall.type,
                                (unsigned)(drawCall.triangles.shaderDesc.colorCombiner.L),
                                n);
                            std::fflush(stderr);
                        }
                    }
                    break;
                }

                const bool typeDifferent = (drawCall.type != previousCallType);
                const bool testZDifferent = (drawCall.type == InstanceDrawCall::Type::IndexedTriangles) && (drawCall.triangles.vertexTestZ != previousVertexTestZ);
                if (typeDifferent || testZDifferent) {
                    switch (drawCall.type) {
                    case InstanceDrawCall::Type::IndexedTriangles:
                        worker->commandList->setVertexBuffers(0, indexedVertexViews.data(), uint32_t(indexedVertexViews.size()), vertexInputSlots.data());
                        worker->commandList->setIndexBuffer(drawCall.triangles.vertexTestZ ? &testZIndexBufferView : &indexBufferView);
                        previousVertexTestZ = drawCall.triangles.vertexTestZ;
                        break;
                    case InstanceDrawCall::Type::RawTriangles:
                    case InstanceDrawCall::Type::RegularRect:
                        worker->commandList->setVertexBuffers(0, rawVertexViews.data(), uint32_t(rawVertexViews.size()), vertexInputSlots.data());
                        worker->commandList->setIndexBuffer(nullptr);
                        break;
                    default:
                        assert(false && "Unknown draw call type.");
                        break;
                    };

                    previousCallType = drawCall.type;
                }

                const auto &triangles = drawCall.triangles;
                assert(triangles.pipeline != nullptr);

                // Draw calls can sometimes end up with empty scissors and cause validation errors. We just skip them.
                if (triangles.scissor.isEmpty()) {
                    // ROGUESQ_LOG_PIPELINE (stage 3) — empty-scissor drops.
                    // If this is where most attribution-period draws die,
                    // the cull is in scissor computation upstream.
                    static const bool log_pipe = []{
                        const char *e = std::getenv("ROGUESQ_LOG_PIPELINE");
                        return e && *e && *e != '0';
                    }();
                    if (log_pipe) {
                        static std::atomic<int> n_skip{0};
                        int n = ++n_skip;
                        if (n <= 5 || (n % 500) == 0) {
                            const uint32_t lo = (uint32_t)(drawCall.triangles.shaderDesc.colorCombiner.L);
                            std::fprintf(stderr,
                                "[pipe-stage 3-scissorSkip] skipped=%d L=0x%08X type=%d\n",
                                n, lo, (int)drawCall.type);
                            std::fflush(stderr);
                        }
                    }
                    continue;
                }

                // A new pass must be started if decals are required and something wrote to the depth buffer before this call.
                const interop::OtherMode otherMode = triangles.shaderDesc.otherMode;
                bool depthDecal = (otherMode.zMode() == ZMODE_DEC);
                bool depthWrite = otherMode.zUpd();
                if (depthDecal) {
                    switchToDepthRead();
                }
                else if (!depthDecal && depthWrite) {
                    switchToDepthWrite();
                }

                if (previousScissor != triangles.scissor) {
                    worker->commandList->setScissors(triangles.scissor);
                    previousScissor = triangles.scissor;
                    if (s_log_batch) ++bScisSw;
                }

                if (previousPipeline != triangles.pipeline) {
                    worker->commandList->setPipeline(triangles.pipeline);
                    previousPipeline = triangles.pipeline;
                    if (s_log_batch) ++bPipeSw;
                }

                if (s_log_batch) {
                    ++bDraws;
                    bDistinctPipe.insert(triangles.pipeline);
                    bool seenScis = false;
                    for (const RenderRect &r : bDistinctScis) {
                        if (!(r != triangles.scissor)) { seenScis = true; break; }
                    }
                    if (!seenScis) bDistinctScis.push_back(triangles.scissor);
                }

                // Coalesce a run of consecutive IndexedTriangles that share pipeline /
                // scissor / screen transform and have contiguous index ranges into a
                // single draw. Order-safe: no reordering, and same pipeline => same
                // depth/blend state (already bound above), so a merged run is exactly
                // what the per-draw sequence would have rasterized. renderIndex comes
                // from the per-vertex stream for the merged draw.
                static const bool s_no_coalesce = [](){ const char* e = std::getenv("ROGUESQ_NO_COALESCE"); return e && *e && *e != '0'; }();
                // Coalesce consecutive same-TYPE triangle draws (indexed, raw, or rect)
                // sharing pipeline/scissor/screen transform with contiguous index (or raw
                // vertex) ranges. Indexed -> drawIndexedInstanced, raw/rect -> drawInstanced;
                // both source renderIndex per-vertex (their respective slot-3 stream).
                const bool coalescable =
                    drawCall.type == InstanceDrawCall::Type::IndexedTriangles ||
                    drawCall.type == InstanceDrawCall::Type::RawTriangles ||
                    drawCall.type == InstanceDrawCall::Type::RegularRect;
                size_t runEnd = sceneIdx;
                uint32_t runFaces = triangles.faceCount;
                if (!s_no_coalesce && coalescable && !triangles.postBlendDitherNoise) {
                    uint32_t nextIndex = triangles.indexStart + triangles.faceCount * 3;
                    for (size_t j = sceneIdx + 1; j < rasterScene.instanceIndices.size(); ++j) {
                        const InstanceDrawCall &d2 = instanceDrawCallVector[rasterScene.instanceIndices[j]];
                        if (d2.type != drawCall.type) break;
                        const auto &t2 = d2.triangles;
                        if (t2.pipeline != triangles.pipeline) break;
                        if (t2.vertexTestZ != triangles.vertexTestZ) break;
                        if (t2.postBlendDitherNoise) break;
                        if (t2.indexStart != nextIndex) break;
                        if (t2.scissor.isEmpty()) break;
                        if (t2.scissor != triangles.scissor) break;
                        if (t2.screenScale.x != triangles.screenScale.x || t2.screenScale.y != triangles.screenScale.y) break;
                        if (t2.screenOffset.x != triangles.screenOffset.x || t2.screenOffset.y != triangles.screenOffset.y) break;
                        runFaces += t2.faceCount;
                        nextIndex += t2.faceCount * 3;
                        runEnd = j;
                    }
                }

                if (runEnd > sceneIdx) {
                    rasterParams.renderIndex = 0;
                    rasterParams.useVertexRenderIndex = 1; // per-vertex renderIndex spans the merged objects
                    rasterParams.screenScale = triangles.screenScale;
                    rasterParams.screenOffset = triangles.screenOffset;
                    worker->commandList->setGraphicsPushConstants(0, &rasterParams);
                    if (drawCall.type == InstanceDrawCall::Type::IndexedTriangles) {
                        worker->commandList->drawIndexedInstanced(runFaces * 3, 1, triangles.indexStart, 0, 0);
                    }
                    else {
                        worker->commandList->drawInstanced(runFaces * 3, 1, triangles.indexStart, 0);
                    }
                    if (s_log_batch) { bDraws += uint32_t(runEnd - sceneIdx); ++bRuns; }
                    sceneIdx = runEnd;
                    break;
                }

                if (s_log_batch) ++bRuns;
                rasterParams.renderIndex = i;
                rasterParams.useVertexRenderIndex = 0; // single draw: per-draw push-constant path (identical to prior behavior)
                rasterParams.screenScale = triangles.screenScale;
                rasterParams.screenOffset = triangles.screenOffset;
                worker->commandList->setGraphicsPushConstants(0, &rasterParams);

                drawCallTriangles(drawCall);

                // Simulate dither noise.
                if (triangles.postBlendDitherNoise) {
                    if (triangles.postBlendDitherNoiseNegative) {
                        worker->commandList->setPipeline(postBlendDitherNoiseSubNegativePipeline);
                    }
                    else {
                        worker->commandList->setPipeline(postBlendDitherNoiseAddPipeline);
                        drawCallTriangles(drawCall);

                        worker->commandList->setPipeline(postBlendDitherNoiseSubPipeline);
                    }

                    drawCallTriangles(drawCall);
                    previousPipeline = nullptr;
                }

                break;
            };
            case InstanceDrawCall::Type::FillRect: {
                const auto &clearRect = drawCall.clearRect;
                RenderTarget *chosenTarget = (fbStorage->colorTarget != nullptr) ? fbStorage->colorTarget : fbStorage->depthTarget;
                bool rectCoversWholeTarget = (clearRect.rect.left == 0) && (clearRect.rect.top == 0) && (uint32_t(clearRect.rect.right) == chosenTarget->width) && (uint32_t(clearRect.rect.bottom) == chosenTarget->height);

                // ROGUESQ_NO_FULLSCREEN_FILLRECT — ported from reverted commit
                // b656c1a (2026-05-13). Rogue Squadron's attribution/fade
                // transitions render text content then emit a full-screen
                // FillRect to clear/fade — but our recompile may execute the
                // FillRect at full opacity every frame, erasing the text the
                // user is supposed to see during fade-in. Mode 1 ("all" or
                // "1") skips EVERY full-screen FillRect; useful for diagnosing
                // whether content is being overwritten by spurious clears.
                static const int s_skip_fullscreen_fill = []() {
                    const char* e = std::getenv("ROGUESQ_NO_FULLSCREEN_FILLRECT");
                    if (!e || !*e || e[0] == '0') return 0;
                    if (!strcmp(e, "all") || !strcmp(e, "1")) return 1;
                    return 1; // anything else truthy = mode 1
                }();
                if (s_skip_fullscreen_fill && rectCoversWholeTarget) {
                    static std::atomic<uint64_t> s_skipped{0};
                    uint64_t n = ++s_skipped;
                    if (n <= 4 || (n % 1000) == 0) {
                        const uint32_t rtAddr = fbStorage->framebufferKey.colorTargetKey.address;
                        std::fprintf(stderr,
                            "[fill-rect] SKIP #%llu rt=0x%08X (mode=1 all-fullscreen)\n",
                            (unsigned long long)n, rtAddr);
                        std::fflush(stderr);
                    }
                    break;
                }

                const RenderRect *clearRects = rectCoversWholeTarget ? nullptr : &clearRect.rect;
                uint32_t clearRectCount = rectCoversWholeTarget ? 0 : 1;
                if (fbStorage->colorTarget != nullptr) {
                    worker->commandList->clearColor(0, clearRect.color, clearRects, clearRectCount);
                }
                else {
                    worker->commandList->clearDepth(true, clearRect.depth, clearRects, clearRectCount);
                }

                break;
            };
            case InstanceDrawCall::Type::VertexTestZ: {
                assert(testZIndexBuffer != nullptr);
                assert(fbStorage->colorTarget != nullptr);

                const interop::RSPVertexTestZCB &testZCB = drawCall.vertexTestZ;
                switchToDepthRead();

                const bool useMSAA = (fbStorage->colorTarget->multisampling.sampleCount > 0);
                const auto &rspVertexTestZ = useMSAA ? shaderLibrary->rspVertexTestZMS : shaderLibrary->rspVertexTestZ;
                worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderBufferBarrier(testZIndexBuffer, RenderBufferAccess::WRITE));
                worker->commandList->setPipeline(rspVertexTestZ.pipeline.get());
                worker->commandList->setComputePipelineLayout(rspVertexTestZ.pipelineLayout.get());
                worker->commandList->setComputePushConstants(0, &testZCB);
                worker->commandList->setComputeDescriptorSet(vertexTestZSet->get(), 0);
                worker->commandList->setComputeDescriptorSet(descRealFbSet, 1);
                worker->commandList->dispatch(1, 1, 1);
                worker->commandList->barriers(RenderBarrierStage::GRAPHICS, RenderBufferBarrier(testZIndexBuffer, RenderBufferAccess::READ));

                switchToGraphicsPipeline();
                break;
            };
            default:
                // Do nothing.
                break;
            }
        }

        if (s_log_batch && bDraws >= 100) {
            static std::atomic<int> s_bn{0};
            int n = ++s_bn;
            if (n <= 8 || (n % 20) == 0) {
                const int dPipe = (int)bDistinctPipe.size();
                const int dScis = (int)bDistinctScis.size();
                std::fprintf(stderr,
                    "[batch] draws=%u runsEmitted=%u recUs=%lld pipeSw=%u distinctPipe=%d savablePipe=%d scisSw=%u distinctScis=%d savableScis=%d riCollisions=%llu\n",
                    bDraws, bRuns, (long long)batchTimer.elapsedMicroseconds(), bPipeSw, dPipe, (int)bPipeSw - dPipe,
                    bScisSw, dScis, (int)bScisSw - dScis,
                    (unsigned long long)s_ri_collisions.load(std::memory_order_relaxed));
                std::fflush(stderr);
            }
        }

        // Mark targets for resolve.
        if (fbStorage->colorTarget != nullptr) {
            fbStorage->colorTarget->markForResolve();
        }

        if (fbStorage->depthTarget != nullptr) {
            fbStorage->depthTarget->markForResolve();
        }
    }

    void FramebufferRenderer::updateMultisampling() {
        dummyColorTargetView.reset();
        dummyDepthTargetView.reset();
        dummyColorTarget.reset();
        dummyDepthTarget.reset();
#   if RT_ENABLED
        if (rtResources != nullptr) {
            rtResources->updateMultisampling();
        }
#   endif
    }

#if RT_ENABLED
    void FramebufferRenderer::updateRaytracingScene(RenderWorker *worker, const RaytracingScene &rtScene) {
        auto &rtParams = rtResources->rtParams;
        const bool lumaActive = rtScene.presetScene.luminanceRange > 0.0f;

        // Only use jitter when an upscaler is active.
        Upscaler *upscaler = rtResources->getUpscaler(rtResources->upscalerMode);
        bool jitterActive = rtResources->upscaleActive && (upscaler != nullptr);
        if (jitterActive) {
            const int phaseCount = upscaler->getJitterPhaseCount(rtResources->textureWidth, rtScene.screenWidth);
            rtParams.pixelJitter = HaltonJitter(frameParams.frameCount, phaseCount);
        }
        else {
            rtParams.pixelJitter = { 0.0f, 0.0f };
        }

        rtParams.viewport.x = rtScene.viewport.x;
        rtParams.viewport.y = rtScene.viewport.y;
        rtParams.viewport.z = rtScene.viewport.width;
        rtParams.viewport.w = rtScene.viewport.height;

        const auto &preset = rtScene.presetScene;
        rtParams.ambientBaseColor = hlslpp::float4(preset.ambientBaseColor, 0.0f);
        rtParams.ambientNoGIColor = hlslpp::float4(preset.ambientNoGIColor, 0.0f);
        rtParams.eyeLightDiffuseColor = hlslpp::float4(preset.eyeLightDiffuseColor, 0.0f);
        rtParams.eyeLightSpecularColor = hlslpp::float4(preset.eyeLightSpecularColor, 0.0f);
        rtParams.giDiffuseStrength = preset.giDiffuseStrength;
        rtParams.giBackgroundStrength = preset.giBackgroundStrength;
        rtParams.tonemapExposure = preset.tonemapExposure;
        rtParams.tonemapWhite = preset.tonemapWhite;
        rtParams.tonemapBlack = preset.tonemapBlack;

        const auto &proj = rtScene.curProjMatrix;
        rtParams.fovRadians = fovFromProj(proj);
        rtParams.nearDist = nearPlaneFromProj(proj);
        rtParams.farDist = farPlaneFromProj(proj);

        if (isnan(rtParams.fovRadians)) {
            rtParams.fovRadians = 0.75f;
        }

        if (isnan(rtParams.nearDist)) {
            rtParams.nearDist = 1.0f;
        }

        if (isnan(rtParams.farDist)) {
            rtParams.farDist = 1000.0f;
        }

        rtParams.view = rtScene.curViewMatrix;
        rtParams.projection = rtScene.curProjMatrix;

        rtParams.viewI = hlslpp::inverse(rtParams.view);
        rtParams.projectionI = hlslpp::inverse(rtParams.projection);
        rtParams.viewProj = hlslpp::mul(rtParams.view, rtParams.projection);
        rtParams.prevViewProj = hlslpp::mul(rtScene.prevViewMatrix, rtScene.prevProjMatrix);

        // TODO: There's probably a way to compute this without calculating the FOV/Near/Far values.
        // Pinhole camera vectors to generate non-normalized ray direction.
        // TODO: Make a fake target and focal distance at the midpoint of the near/far planes
        // until the game sends that data in some way in the future.
        const float FocalDistance = (rtParams.nearDist + rtParams.farDist) / 2.0f;
        const hlslpp::float3 Up(0.0f, 1.0f, 0.0f);
        const hlslpp::float3 Pos = viewPositionFrom(rtParams.viewI);
        const hlslpp::float3 Target = Pos + viewDirectionFrom(rtParams.viewI) * FocalDistance;
        hlslpp::float3 cameraW = hlslpp::normalize(Target - Pos) * FocalDistance;
        hlslpp::float3 cameraU = hlslpp::normalize(hlslpp::cross(cameraW, Up));
        hlslpp::float3 cameraV = hlslpp::normalize(hlslpp::cross(cameraU, cameraW));
        const float ulen = FocalDistance * std::tan(rtParams.fovRadians * 0.5f);// * rtParams.aspectRatio;
        const float vlen = FocalDistance * std::tan(rtParams.fovRadians * 0.5f);
        cameraU = cameraU * ulen;
        cameraV = cameraV * vlen;
        rtParams.cameraU = hlslpp::float4(cameraU, 0.0f);
        rtParams.cameraV = hlslpp::float4(cameraV, 0.0f);
        rtParams.cameraW = hlslpp::float4(cameraW, 0.0f);

        // Enable light reprojection if denoising is enabled.
#   ifdef DI_REPROJECTION_SUPPORT
        globalParamsBufferData.diReproject = !rtResources->skipReprojection && denoiserEnabled && (globalParamsBufferData.diSamples > 0) && (rtResources->upscalerMode != UpscaleMode::DLSS) ? 1 : 0;
#   else
        rtParams.diReproject = 0;
#   endif

        rtParams.giReproject = !rtResources->skipReprojection && rtResources->denoiserEnabled && (rtParams.giSamples > 0) && (rtResources->upscalerMode != UpscaleMode::DLSS) ? 1 : 0;
        rtParams.binaryLockMask = (rtResources->upscalerMode != UpscaleMode::FSR);
        rtParams.interleavedRastersCount = interleavedRastersCount;
        
        Framebuffer &framebuffer = framebufferVector[framebufferCount - 1];
        RenderDescriptorSet *descRealDepthSet = framebuffer.descRealFbSet->get();
        RenderDescriptorSet *descriptorSets[] = { descCommonSet->get(), descTextureSet->get(), descTextureSet->get(), descRealDepthSet };
        rtResources->updateTopLevelASResources(worker, instanceDrawCallVector, rtScene.instanceIndices);
        rtResources->createShaderBindingTable(worker, rtState, descriptorSets, uint32_t(std::size(descriptorSets)), hitGroupVector);
        rtResources->updateLightsBuffer(worker, rtScene);
    }
    
    void FramebufferRenderer::submitRaytracingScene(RenderWorker *worker, RenderTarget *colorTarget, const RaytracingScene &rtScene) {
        // Unbind any render targets.
        worker->commandList->setFramebuffer(nullptr);

        // Resolve the color target if necessary before using it as the RT scene background.
        colorTarget->resolveTarget(worker, shaderLibrary);
        worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderTextureBarrier(colorTarget->getResolvedTexture(), RenderTextureLayout::SHADER_READ));

        if (rtResources->transitionOutputBuffers) {
            RenderTextureBarrier afterCreationBarriers[] = {
                RenderTextureBarrier(rtResources->viewDirectionTexture.get(), RenderTextureLayout::GENERAL),
                RenderTextureBarrier(rtResources->shadingPositionTexture.get(), RenderTextureLayout::GENERAL),
                RenderTextureBarrier(rtResources->shadingNormalTexture.get(), RenderTextureLayout::GENERAL),
                RenderTextureBarrier(rtResources->shadingSpecularTexture.get(), RenderTextureLayout::GENERAL),
                RenderTextureBarrier(rtResources->instanceIdTexture.get(), RenderTextureLayout::GENERAL),
                RenderTextureBarrier(rtResources->directLightTexture[0].get(), RenderTextureLayout::GENERAL),
                RenderTextureBarrier(rtResources->directLightTexture[1].get(), RenderTextureLayout::GENERAL),
                RenderTextureBarrier(rtResources->indirectLightTexture[0].get(), RenderTextureLayout::GENERAL),
                RenderTextureBarrier(rtResources->indirectLightTexture[1].get(), RenderTextureLayout::GENERAL),
                RenderTextureBarrier(rtResources->normalRoughnessTexture[0].get(), RenderTextureLayout::GENERAL),
                RenderTextureBarrier(rtResources->normalRoughnessTexture[1].get(), RenderTextureLayout::GENERAL),
                RenderTextureBarrier(rtResources->filteredDirectLightTexture[0].get(), RenderTextureLayout::GENERAL),
                RenderTextureBarrier(rtResources->filteredDirectLightTexture[1].get(), RenderTextureLayout::GENERAL),
                RenderTextureBarrier(rtResources->filteredIndirectLightTexture[0].get(), RenderTextureLayout::GENERAL),
                RenderTextureBarrier(rtResources->filteredIndirectLightTexture[1].get(), RenderTextureLayout::GENERAL)
            };

            worker->commandList->barriers(RenderBarrierStage::COMPUTE, afterCreationBarriers, uint32_t(std::size(afterCreationBarriers)));
            rtResources->transitionOutputBuffers = false;
        }

        // Make sure all these buffers are usable as UAVs.
        RenderTextureBarrier preDispatchBarriers[] = {
            RenderTextureBarrier(rtResources->diffuseTexture.get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(rtResources->reflectionTexture.get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(rtResources->refractionTexture.get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(rtResources->transparentTexture.get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(rtResources->flowTexture.get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(rtResources->reactiveMaskTexture.get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(rtResources->lockMaskTexture.get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(rtResources->depthTexture[0].get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(rtResources->depthTexture[1].get(), RenderTextureLayout::GENERAL)
        };

        worker->commandList->barriers(RenderBarrierStage::COMPUTE, preDispatchBarriers, uint32_t(std::size(preDispatchBarriers)));
        
        // Bind pipeline and dispatch primary rays.
        RT64_LOG_PRINTF("Dispatching primary rays");
        Framebuffer &framebuffer = framebufferVector[framebufferCount - 1];
        RenderDescriptorSet *descRealFbSet = framebuffer.descRealFbSet->get();
        rtResources->shaderBindingTableInfo.groups.rayGen.startIndex = 0;
        worker->commandList->setPipeline(rtState->pipeline.get());
        worker->commandList->setRaytracingPipelineLayout(rtPipelineLayout);
        worker->commandList->setRaytracingDescriptorSet(descCommonSet->get(), 0);
        worker->commandList->setRaytracingDescriptorSet(descTextureSet->get(), 1);
        worker->commandList->setRaytracingDescriptorSet(descTextureSet->get(), 2);
        worker->commandList->setRaytracingDescriptorSet(descRealFbSet, 3);
        worker->commandList->traceRays(rtResources->textureWidth, rtResources->textureHeight, 1, rtResources->shaderBindingTableBuffer->at(0), rtResources->shaderBindingTableInfo.groups);

        // Barriers for shading buffers before dispatching secondary rays.
        RenderTextureBarrier shadingBarriers[] = {
            RenderTextureBarrier(rtResources->instanceIdTexture.get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(rtResources->shadingPositionTexture.get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(rtResources->viewDirectionTexture.get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(rtResources->shadingNormalTexture.get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(rtResources->shadingSpecularTexture.get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(rtResources->reflectionTexture.get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(rtResources->refractionTexture.get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(rtResources->normalRoughnessTexture[rtResources->swapBuffers ? 1 : 0].get(), RenderTextureLayout::GENERAL),
        };

        worker->commandList->barriers(RenderBarrierStage::COMPUTE, shadingBarriers, uint32_t(std::size(shadingBarriers)));

        // Dispatch rays for direct light.
        RT64_LOG_PRINTF("Dispatching direct light rays");
        rtResources->shaderBindingTableInfo.groups.rayGen.startIndex = 1;
        worker->commandList->traceRays(rtResources->textureWidth, rtResources->textureHeight, 1, rtResources->shaderBindingTableBuffer->at(0), rtResources->shaderBindingTableInfo.groups);

        // Dispatch rays for indirect light.
        RT64_LOG_PRINTF("Dispatching indirect light rays");
        rtResources->shaderBindingTableInfo.groups.rayGen.startIndex = 2;
        worker->commandList->traceRays(rtResources->textureWidth, rtResources->textureHeight, 1, rtResources->shaderBindingTableBuffer->at(0), rtResources->shaderBindingTableInfo.groups);

        // Wait until indirect light is done before dispatching reflection or refraction rays.
        // TODO: This is only required to prevent simultaneous usage of the anyhit buffers.
        // This barrier can be removed if this no longer happens, resulting in less serialization of the commands.
        worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderTextureBarrier(rtResources->indirectLightTexture[rtResources->swapBuffers ? 1 : 0].get(), RenderTextureLayout::GENERAL));

        // Dispatch rays for refraction.
        RT64_LOG_PRINTF("Dispatching refraction rays");
        rtResources->shaderBindingTableInfo.groups.rayGen.startIndex = 4;
        worker->commandList->traceRays(rtResources->textureWidth, rtResources->textureHeight, 1, rtResources->shaderBindingTableBuffer->at(0), rtResources->shaderBindingTableInfo.groups);

        // Wait until refraction is done before dispatching reflection rays.
        // TODO: This is only required to prevent simultaneous usage of the anyhit buffers.
        // This barrier can be removed if this no longer happens, resulting in less serialization of the commands.
        worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderTextureBarrier(rtResources->refractionTexture.get(), RenderTextureLayout::GENERAL));

        // Reflection passes.
        int reflections = rtResources->maxReflections;
        while (reflections > 0) {
            // Dispatch rays for reflection.
            RT64_LOG_PRINTF("Dispatching reflection rays");
            rtResources->shaderBindingTableInfo.groups.rayGen.startIndex = 3;
            worker->commandList->traceRays(rtResources->textureWidth, rtResources->textureHeight, 1, rtResources->shaderBindingTableBuffer->at(0), rtResources->shaderBindingTableInfo.groups);
            reflections--;

            // Add a barrier to wait for the input UAVs to be finished if there's more passes left to be done.
            if (reflections > 0) {
                RenderTextureBarrier newInputBarriers[] = {
                    RenderTextureBarrier(rtResources->viewDirectionTexture.get(), RenderTextureLayout::GENERAL),
                    RenderTextureBarrier(rtResources->shadingNormalTexture.get(), RenderTextureLayout::GENERAL),
                    RenderTextureBarrier(rtResources->instanceIdTexture.get(), RenderTextureLayout::GENERAL),
                    RenderTextureBarrier(rtResources->reflectionTexture.get(), RenderTextureLayout::GENERAL)
                };

                worker->commandList->barriers(RenderBarrierStage::COMPUTE, newInputBarriers, uint32_t(std::size(newInputBarriers)));
            }
        }

        // Copy direct light raw buffer to the first direct filtered buffer.
        {
            RenderTexture *source = rtResources->directLightTexture[rtResources->swapBuffers ? 1 : 0].get();
            RenderTexture *dest = rtResources->filteredDirectLightTexture[1].get();

            RenderTextureBarrier beforeCopyBarriers[] = {
                RenderTextureBarrier(source, RenderTextureLayout::COPY_SOURCE),
                RenderTextureBarrier(dest, RenderTextureLayout::COPY_DEST)
            };

            worker->commandList->barriers(RenderBarrierStage::COPY, beforeCopyBarriers, uint32_t(std::size(beforeCopyBarriers)));
            worker->commandList->copyTexture(dest, source);

            RenderTextureBarrier afterCopyBarriers[] = {
                RenderTextureBarrier(source, RenderTextureLayout::GENERAL),
                RenderTextureBarrier(dest, RenderTextureLayout::SHADER_READ)
            };

            worker->commandList->barriers(RenderBarrierStage::COMPUTE, afterCopyBarriers, uint32_t(std::size(afterCopyBarriers)));
        }

        // Copy indirect light raw buffer to the first indirect filtered buffer.
        bool denoiseGI = rtResources->denoiserEnabled && (rtResources->rtParams.giSamples > 0) && (rtResources->upscalerMode != UpscaleMode::DLSS);
        {
            RenderTexture *source = rtResources->indirectLightTexture[rtResources->swapBuffers ? 1 : 0].get();
            RenderTexture *dest = rtResources->filteredIndirectLightTexture[denoiseGI ? 0 : 1].get();

            RenderTextureBarrier beforeCopyBarriers[] = {
                RenderTextureBarrier(source, RenderTextureLayout::COPY_SOURCE),
                RenderTextureBarrier(dest, RenderTextureLayout::COPY_DEST)
            };

            worker->commandList->barriers(RenderBarrierStage::COPY, beforeCopyBarriers, uint32_t(std::size(beforeCopyBarriers)));
            worker->commandList->copyTexture(dest, source);

            RenderTextureBarrier afterCopyBarriers[] = {
                RenderTextureBarrier(source, RenderTextureLayout::GENERAL),
                RenderTextureBarrier(dest, RenderTextureLayout::SHADER_READ)
            };

            worker->commandList->barriers(RenderBarrierStage::COMPUTE, afterCopyBarriers, uint32_t(std::size(afterCopyBarriers)));
        }

        // Apply a gaussian filter to the indirect light with a compute shader.
        if (denoiseGI) {
            for (int i = 0; i < 5; i++) {
                const uint32_t ThreadGroupWorkCount = 8;
                uint32_t dispatchX = (rtResources->textureWidth + ThreadGroupWorkCount - 1) / ThreadGroupWorkCount;
                uint32_t dispatchY = (rtResources->textureHeight + ThreadGroupWorkCount - 1) / ThreadGroupWorkCount;
                const ShaderRecord &gaussianFilter = shaderLibrary->gaussianFilterRGB3x3;
                interop::TextureCB textureCB;
                textureCB.TextureSize = { rtResources->textureWidth, rtResources->textureHeight };
                textureCB.TexelSize = { 1.0f / rtResources->textureWidth, 1.0f / rtResources->textureHeight };

                worker->commandList->setPipeline(gaussianFilter.pipeline.get());
                worker->commandList->setComputePipelineLayout(gaussianFilter.pipelineLayout.get());
                worker->commandList->setComputePushConstants(0, &textureCB);
                worker->commandList->setComputeDescriptorSet(rtResources->indirectFilterSets[i % 2]->get(), 0);
                worker->commandList->dispatch(dispatchX, dispatchY, 1);

                RenderTextureBarrier afterBlurBarriers[] = {
                    RenderTextureBarrier(rtResources->filteredIndirectLightTexture[(i % 2) ? 1 : 0].get(), RenderTextureLayout::GENERAL),
                    RenderTextureBarrier(rtResources->filteredIndirectLightTexture[(i % 2) ? 0 : 1].get(), RenderTextureLayout::SHADER_READ)
                };

                worker->commandList->barriers(RenderBarrierStage::COMPUTE, afterBlurBarriers, uint32_t(std::size(afterBlurBarriers)));
            }
        }

        // Compose the output buffer.
        RenderTexture *rtOutputCur = rtResources->outputTexture[rtResources->swapBuffers ? 1 : 0].get();

        // Barriers for shading buffers after rays are finished.
        RenderTextureBarrier afterDispatchBarriers[] = {
            RenderTextureBarrier(rtOutputCur, RenderTextureLayout::COLOR_WRITE),
            RenderTextureBarrier(colorTarget->texture.get(), RenderTextureLayout::COLOR_WRITE),
            RenderTextureBarrier(rtResources->diffuseTexture.get(), RenderTextureLayout::SHADER_READ),
            RenderTextureBarrier(rtResources->reflectionTexture.get(), RenderTextureLayout::SHADER_READ),
            RenderTextureBarrier(rtResources->refractionTexture.get(), RenderTextureLayout::SHADER_READ),
            RenderTextureBarrier(rtResources->transparentTexture.get(), RenderTextureLayout::SHADER_READ)
        };

        worker->commandList->barriers(RenderBarrierStage::GRAPHICS, afterDispatchBarriers, uint32_t(std::size(afterDispatchBarriers)));

        // Set the output as the current render target.
        worker->commandList->setFramebuffer(rtResources->outputFramebuffer[rtResources->swapBuffers ? 1 : 0].get());

        // Apply the scissor and viewport to the size of the output texture.
        worker->commandList->setViewports(RenderViewport(0.0f, 0.0f, float(rtResources->textureWidth), float(rtResources->textureHeight)));
        worker->commandList->setScissors(RenderRect(0, 0, rtResources->textureWidth, rtResources->textureHeight));

        // Draw the raytracing output.
        RT64_LOG_PRINTF("Composing the raytracing output");
        const ShaderRecord &composeShader = shaderLibrary->compose;
        worker->commandList->setVertexBuffers(0, nullptr, 0, nullptr);
        worker->commandList->setPipeline(composeShader.pipeline.get());
        worker->commandList->setGraphicsPipelineLayout(composeShader.pipelineLayout.get());
        worker->commandList->setGraphicsDescriptorSet(rtResources->composeSet->get(), 0);
        worker->commandList->drawInstanced(3, 1, 0, 0);

        // Switch resources to the correct states after composing the image
        RenderTextureBarrier afterComposeBarriers[] = {
            RenderTextureBarrier(rtOutputCur, RenderTextureLayout::SHADER_READ),
            RenderTextureBarrier(rtResources->filteredDirectLightTexture[1].get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(rtResources->filteredIndirectLightTexture[1].get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(rtResources->flowTexture.get(), RenderTextureLayout::SHADER_READ),
            RenderTextureBarrier(rtResources->reactiveMaskTexture.get(), RenderTextureLayout::SHADER_READ),
            RenderTextureBarrier(rtResources->lockMaskTexture.get(), RenderTextureLayout::SHADER_READ)
        };

        worker->commandList->barriers(RenderBarrierStage::GRAPHICS_AND_COMPUTE, afterComposeBarriers, uint32_t(std::size(afterComposeBarriers)));

        const bool lumaActive = rtScene.presetScene.luminanceRange > 0.0f;
        if (lumaActive) {
            const uint32_t ThreadGroupWorkRegionDim = 8;
            worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderTextureBarrier(rtResources->downscaledOutputTexture.get(), RenderTextureLayout::GENERAL));

            RT64_LOG_PRINTF("Do the downscaling shader");
            {
                // Execute the compute shader for downscaling the image.
                const ShaderRecord &bicubicScaling = shaderLibrary->bicubicScaling;
                uint32_t dispatchX = ((rtResources->textureWidth / 8) + ThreadGroupWorkRegionDim - 1) / ThreadGroupWorkRegionDim;
                uint32_t dispatchY = ((rtResources->textureHeight / 8) + ThreadGroupWorkRegionDim - 1) / ThreadGroupWorkRegionDim;
                interop::BicubicCB bicubicCB;
                bicubicCB.InputResolution = { rtResources->textureWidth, rtResources->textureHeight };
                bicubicCB.OutputResolution = { rtResources->textureWidth / 8, rtResources->textureHeight / 8 };
                worker->commandList->setPipeline(bicubicScaling.pipeline.get());
                worker->commandList->setComputePipelineLayout(bicubicScaling.pipelineLayout.get());
                worker->commandList->setComputePushConstants(0, &bicubicCB);
                worker->commandList->setComputeDescriptorSet(rtResources->downscaleSet->get(), 0);
                worker->commandList->dispatch(dispatchX, dispatchY, 1);
            }


            worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderTextureBarrier(rtResources->downscaledOutputTexture.get(), RenderTextureLayout::GENERAL));

            RT64_LOG_PRINTF("Do the luminance histogram shader");
            {
                // Execute the compute shader for the luminance histogram.
                const ShaderRecord &luminanceHistogram = shaderLibrary->luminanceHistogram;
                uint32_t dispatchX = ((rtResources->textureWidth / 8) + ThreadGroupWorkRegionDim - 1) / ThreadGroupWorkRegionDim;
                uint32_t dispatchY = ((rtResources->textureHeight / 8) + ThreadGroupWorkRegionDim - 1) / ThreadGroupWorkRegionDim;
                interop::LuminanceHistogramCB histogramCB;
                histogramCB.inputWidth = rtResources->textureWidth / 8;
                histogramCB.inputHeight = rtResources->textureHeight / 8;
                histogramCB.minLuminance = rtScene.presetScene.minLuminance;
                histogramCB.oneOverLuminanceRange = 1.0f / rtScene.presetScene.luminanceRange;
                worker->commandList->setPipeline(luminanceHistogram.pipeline.get());
                worker->commandList->setComputePipelineLayout(luminanceHistogram.pipelineLayout.get());
                worker->commandList->setComputePushConstants(0, &histogramCB);
                worker->commandList->setComputeDescriptorSet(rtResources->lumaSet->get(), 0);
                worker->commandList->dispatch(dispatchX, dispatchY, 1);
            }

            worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderTextureBarrier(rtResources->downscaledOutputTexture.get(), RenderTextureLayout::SHADER_READ));

            RT64_LOG_PRINTF("Do the luminance average shader");
            {
                // Execute the compute shader for the luminance histogram average.
                const ShaderRecord &histogramAverage = shaderLibrary->histogramAverage;
                interop::HistogramAverageCB averageCB;
                averageCB.pixelCount = (rtResources->textureWidth / 8) * (rtResources->textureHeight / 8);
                averageCB.minLuminance = rtScene.presetScene.minLuminance;
                averageCB.luminanceRange = rtScene.presetScene.luminanceRange;
                averageCB.timeDelta = rtScene.deltaTime;
                averageCB.tau = rtScene.presetScene.lumaUpdateTime;
                worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderTextureBarrier(rtResources->lumaAverageTexture.get(), RenderTextureLayout::GENERAL));
                worker->commandList->setPipeline(histogramAverage.pipeline.get());
                worker->commandList->setComputePipelineLayout(histogramAverage.pipelineLayout.get());
                worker->commandList->setComputePushConstants(0, &averageCB);
                worker->commandList->setComputeDescriptorSet(rtResources->lumaAvgSet->get(), 0);
                worker->commandList->dispatch(ThreadGroupWorkRegionDim, ThreadGroupWorkRegionDim, 1);
                worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderTextureBarrier(rtResources->lumaAverageTexture.get(), RenderTextureLayout::SHADER_READ));
            }

            RT64_LOG_PRINTF("Do the histogram clear shader");
            {
                // Execute the compute shader for clearing the luminance histogram.
                const ShaderRecord &histogramClear = shaderLibrary->histogramClear;
                worker->commandList->setPipeline(histogramClear.pipeline.get());
                worker->commandList->setComputePipelineLayout(histogramClear.pipelineLayout.get());
                worker->commandList->setComputeDescriptorSet(rtResources->lumaClearSet->get(), 0);
                worker->commandList->dispatch(ThreadGroupWorkRegionDim, ThreadGroupWorkRegionDim, 1);
            }
        }
        else {
            RT64_LOG_PRINTF("Do the histogram set shader");
            {
                // Execute the compute shader for setting the luminance value.
                const ShaderRecord &histogramSet = shaderLibrary->histogramSet;
                interop::HistogramSetCB setCB;
                setCB.luminanceValue = rtScene.presetScene.minLuminance;
                worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderTextureBarrier(rtResources->lumaAverageTexture.get(), RenderTextureLayout::GENERAL));
                worker->commandList->setPipeline(histogramSet.pipeline.get());
                worker->commandList->setComputePipelineLayout(histogramSet.pipelineLayout.get());
                worker->commandList->setComputePushConstants(0, &setCB);
                worker->commandList->setComputeDescriptorSet(rtResources->lumaSetSet->get(), 0);
                worker->commandList->dispatch(1, 1, 1);
                worker->commandList->barriers(RenderBarrierStage::GRAPHICS_AND_COMPUTE, RenderTextureBarrier(rtResources->lumaAverageTexture.get(), RenderTextureLayout::SHADER_READ));
            }
        }

        Upscaler *upscaler = rtResources->getUpscaler(rtResources->upscalerMode);
        const bool upscalerActive = rtResources->upscaleActive && (upscaler != nullptr);
        if (upscalerActive) {
            thread_local std::vector<RenderTextureBarrier> beforeBarriers;
            thread_local std::vector<RenderTextureBarrier> afterBarriers;
            beforeBarriers.clear();
            afterBarriers.clear();

            beforeBarriers.push_back(RenderTextureBarrier(rtResources->upscaledOutputTexture.get(), RenderTextureLayout::GENERAL));
            afterBarriers.push_back(RenderTextureBarrier(rtResources->upscaledOutputTexture.get(), RenderTextureLayout::SHADER_READ));
            RenderTexture *rtDepthCur = rtResources->depthTexture[rtResources->swapBuffers ? 1 : 0].get();
            if (upscaler->requiresNonShaderResourceInputs()) {
                for (RenderTexture *res : { rtOutputCur, rtResources->flowTexture.get(), rtResources->reactiveMaskTexture.get(), rtResources->lockMaskTexture.get(), rtDepthCur }) {
                    beforeBarriers.push_back(RenderTextureBarrier(res, RenderTextureLayout::SHADER_READ));
                    afterBarriers.push_back(RenderTextureBarrier(res, RenderTextureLayout::SHADER_READ));
                }
            }

            worker->commandList->barriers(RenderBarrierStage::COMPUTE, beforeBarriers);

            Upscaler::UpscaleParameters params;
            params.inRect = { 0, 0, static_cast<int>(rtResources->textureWidth), static_cast<int>(rtResources->textureHeight) };
            params.inDiffuseAlbedo = rtResources->diffuseTexture.get();
            params.inSpecularAlbedo = rtResources->shadingSpecularTexture.get();
            params.inNormalRoughness = rtResources->normalRoughnessTexture[rtResources->swapBuffers ? 1 : 0].get();
            params.inColor = rtOutputCur;
            params.inFlow = rtResources->flowTexture.get();
            params.inReactiveMask = rtResources->upscalerReactiveMask ? rtResources->reactiveMaskTexture.get() : nullptr;
            params.inLockMask = rtResources->upscalerLockMask ? rtResources->lockMaskTexture.get() : nullptr;
            params.inDepth = rtDepthCur;
            params.outColor = rtResources->upscaledOutputTexture.get();
            params.jitterX = -rtResources->rtParams.pixelJitter.x;
            params.jitterY = -rtResources->rtParams.pixelJitter.y;
            params.deltaTime = rtScene.deltaTime * 1000.0f;
            params.nearPlane = rtResources->rtParams.nearDist;
            params.farPlane = rtResources->rtParams.farDist;
            params.fovY = rtResources->rtParams.fovRadians;
            params.resetAccumulation = false; // TODO: Make this configurable via the API.
            upscaler->upscale(worker, params);

            worker->commandList->barriers(RenderBarrierStage::GRAPHICS, afterBarriers);
        }

        // Set the final render target. Apply the same scissor and viewport that was determined for the raytracing step.
        worker->commandList->setFramebuffer(colorTarget->textureFramebuffer.get());
        worker->commandList->setViewports(rtScene.viewport);
        worker->commandList->setScissors(rtScene.scissor);
        worker->commandList->setVertexBuffers(0, nullptr, 0, nullptr);

        // Draw final output.
        const ShaderRecord &postProcess = shaderLibrary->postProcess;
        worker->commandList->setPipeline(postProcess.pipeline.get());
        worker->commandList->setGraphicsPipelineLayout(postProcess.pipelineLayout.get());
        worker->commandList->setGraphicsDescriptorSet(rtResources->postProcessSet->get(), 0);
        worker->commandList->drawInstanced(3, 1, 0, 0);

        // Draw debug view on top.
        if (rtResources->rtParams.visualizationMode != interop::VisualizationMode::Final) {
            const ShaderRecord &debugShader = shaderLibrary->debug;
            worker->commandList->setPipeline(debugShader.pipeline.get());
            worker->commandList->setGraphicsPipelineLayout(debugShader.pipelineLayout.get());
            worker->commandList->setGraphicsDescriptorSet(descCommonSet->get(), 0);
            worker->commandList->setGraphicsDescriptorSet(descTextureSet->get(), 1);
            worker->commandList->setGraphicsDescriptorSet(descTextureSet->get(), 2);
            worker->commandList->setGraphicsDescriptorSet(descRealFbSet, 3);
            worker->commandList->drawInstanced(3, 1, 0, 0);
        }

        // Mark targets for resolve.
        colorTarget->markForResolve();
    }
#endif

    void FramebufferRenderer::submitRSPSmoothNormalCompute(RenderWorker *worker, const OutputBuffers *outputBuffers) {
        if (rspSmoothNormalVector.empty()) {
            return;
        }
        
        worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderBufferBarrier(outputBuffers->worldNormBuffer.buffer.get(), RenderBufferAccess::WRITE));

        const int ThreadGroupSize = 64;
        const auto &rspSmoothNormal = shaderLibrary->rspSmoothNormal;
        for (const RSPSmoothNormalGenerationCB &cb : rspSmoothNormalVector) {
            const uint32_t triangleCount = cb.indexCount / 3;
            const uint32_t dispatchCount = (triangleCount + ThreadGroupSize - 1) / ThreadGroupSize;
            worker->commandList->setPipeline(rspSmoothNormal.pipeline.get());
            worker->commandList->setComputePipelineLayout(rspSmoothNormal.pipelineLayout.get());
            worker->commandList->setComputePushConstants(0, &cb);
            worker->commandList->setComputeDescriptorSet(smoothDescSet->get(), 0);
            worker->commandList->dispatch(dispatchCount, 1, 1);
        }
    }

    void FramebufferRenderer::recordSetup(RenderWorker *worker, std::vector<BufferUploader *> bufferUploaders, RSPProcessor *rspProcessor,
        VertexProcessor *vertexProcessor, const OutputBuffers *outputBuffers, bool rtEnabled) {
        if (!dummyColorTargetTransitioned) {
            worker->commandList->barriers(RenderBarrierStage::GRAPHICS_AND_COMPUTE, RenderTextureBarrier(dummyColorTarget.get(), RenderTextureLayout::SHADER_READ));
            dummyColorTargetTransitioned = true;
        }

        if (!dummyDepthTargetTransitioned) {
            worker->commandList->barriers(RenderBarrierStage::GRAPHICS_AND_COMPUTE, RenderTextureBarrier(dummyDepthTarget.get(), RenderTextureLayout::DEPTH_READ));
            dummyDepthTargetTransitioned = true;
        }

        for (BufferUploader *uploader : bufferUploaders) {
            uploader->commandListBeforeBarriers(worker);
        }

        shaderUploader->commandListBeforeBarriers(worker);

        for (BufferUploader *uploader : bufferUploaders) {
            uploader->commandListCopyResources(worker);
        }

        shaderUploader->commandListCopyResources(worker);

        for (BufferUploader *uploader : bufferUploaders) {
            uploader->commandListAfterBarriers(worker);
        }

        if (rspProcessor != nullptr) {
            rspProcessor->recordCommandList(worker, shaderLibrary, outputBuffers);
        }

        if (vertexProcessor != nullptr) {
            vertexProcessor->recordCommandList(worker, shaderLibrary, outputBuffers);
        }

        // One full-screen pass of each kind per colour target: earlier pairs that share the target defer to the last one.
        {
            thread_local std::vector<const void *> targets;
            thread_local std::vector<uint8_t> shadowsOn, lightsOn, fogOn;
            targets.assign(framebufferCount, nullptr);
            shadowsOn.assign(framebufferCount, 0);
            lightsOn.assign(framebufferCount, 0);
            fogOn.assign(framebufferCount, 0);
            for (uint32_t f = 0; f < framebufferCount; f++) {
                targets[f] = framebufferVector[f].rs64Target;
                shadowsOn[f] = framebufferVector[f].rs64ShadowsOn ? 1 : 0;
                lightsOn[f] = framebufferVector[f].rs64LightsOn ? 1 : 0;
                fogOn[f] = framebufferVector[f].rs64FogOn ? 1 : 0;
            }
            rs64lights::lastPassPerTarget(targets.data(), reinterpret_cast<bool *>(shadowsOn.data()), framebufferCount);
            rs64lights::lastPassPerTarget(targets.data(), reinterpret_cast<bool *>(lightsOn.data()), framebufferCount);
            rs64lights::lastPassPerTarget(targets.data(), reinterpret_cast<bool *>(fogOn.data()), framebufferCount);
            for (uint32_t f = 0; f < framebufferCount; f++) {
                framebufferVector[f].rs64ShadowsOn = (shadowsOn[f] != 0);
                framebufferVector[f].rs64LightsOn = (lightsOn[f] != 0);
                framebufferVector[f].rs64FogOn = (fogOn[f] != 0);
            }
        }

        // Shadow casters were gathered by addFramebuffer; build their acceleration structures after this frame's world vertices.
        const bool rs64Dynamic = (!rs64Shadow.indices.empty() || !rs64Shadow.cutoutIndices.empty() || !rs64Shadow.craftIndices.empty()) && (vertexProcessor != nullptr) && (rs64Shadow.worldPos != nullptr);
        if (rs64Dynamic || rs64Shadow.terrainActive) {
            rs64TimingMark(worker, RS64TimeScene);
            recordRS64ShadowScene(worker, rs64Dynamic);
            rs64TimingMark(worker, RS64TimeScene);
        }
        else {
            rs64Shadow.ready = false;
            rs64Shadow.indices.clear();
            rs64Shadow.cutoutDrawCount = 0;
        }
        rs64Shadow.ranges.clear();
        rs64Shadow.cutoutRanges.clear();
        rs64Shadow.cutoutIndices.clear();
        rs64Shadow.rangeCalls.clear();
        rs64Shadow.emissiveRanges.clear();
        rs64Shadow.emissiveCalls.clear();
        rs64Shadow.craftRanges.clear();

#   if RT_ENABLED
        if (rtEnabled) {
            assert(rtResources != nullptr);
            if (!rtResources->bottomLevelASVector.empty()) {
                if (!rspSmoothNormalVector.empty()) {
                    submitRSPSmoothNormalCompute(worker, outputBuffers);
                }

                rtResources->submitBottomLevelASCreation(worker);
                rtResources->submitTopLevelASCreation(worker);
            }
        }
#   endif

        shaderUploader->commandListAfterBarriers(worker);
        pendingUploaders = bufferUploaders;

        if (!dynamicTextureBarrierVector.empty()) {
            worker->commandList->barriers(RenderBarrierStage::GRAPHICS_AND_COMPUTE, dynamicTextureBarrierVector);
        }
    }

    void FramebufferRenderer::recordFramebuffer(RenderWorker *worker, uint32_t framebufferIndex) {
        // Submit all transition barriers first.
        thread_local std::vector<RenderTextureBarrier> startBarriers;
        startBarriers.clear();

        const Framebuffer &framebuffer = framebufferVector[framebufferIndex];
        for (RenderTarget *target : framebuffer.transitionRenderTargetSet) {
            startBarriers.emplace_back(RenderTextureBarrier(target->getResolvedTexture(), RenderTextureLayout::SHADER_READ));
        }
        
        const RenderTargetDrawCall &targetDrawCall = framebuffer.renderTargetDrawCall;
        RenderTarget *colorTarget = targetDrawCall.fbStorage->colorTarget;
        RenderTarget *depthTarget = targetDrawCall.fbStorage->depthTarget;
        if (colorTarget != nullptr) {
            startBarriers.emplace_back(RenderTextureBarrier(colorTarget->texture.get(), RenderTextureLayout::COLOR_WRITE));
        }

        startBarriers.emplace_back(RenderTextureBarrier(depthTarget->texture.get(), RenderTextureLayout::DEPTH_WRITE));
        worker->commandList->barriers(RenderBarrierStage::GRAPHICS, startBarriers);

        bool depthState = false;
        worker->commandList->setFramebuffer(targetDrawCall.fbStorage->colorDepthWrite.get());
        for (const auto &pair : targetDrawCall.sceneIndices) {
#       if RT_ENABLED
            if (pair.second) {
                const auto &rtScene = targetDrawCall.rtScenes[pair.first];

                // Draw all the interleaved rasterized buffers that will be used in the render target.
                thread_local std::vector<RenderTextureBarrier> interleavedBarriers;
                interleavedBarriers.clear();
                for (uint32_t i = 0; i < interleavedRastersCount; i++) {
                    bool interleavedDepthState = false;
                    const uint32_t sceneIndex = rtScene.interleavedRasters[i].rasterSceneIndex;
                    RenderTarget *colorRenderTarget = rtResources->interleavedColorTargetVector[i].get();
                    RenderTarget *depthRenderTarget = rtResources->interleavedDepthTargetVector[i].get();
                    worker->commandList->barriers(RenderBarrierStage::GRAPHICS, {
                        RenderTextureBarrier(colorRenderTarget->texture.get(), RenderTextureLayout::COLOR_WRITE),
                        RenderTextureBarrier(depthRenderTarget->texture.get(), RenderTextureLayout::DEPTH_WRITE)
                    });

                    RenderFramebufferStorage *fbStorage = rtResources->interleavedFramebufferStorageVector[i].get();
                    worker->commandList->setFramebuffer(fbStorage->colorDepthWrite.get());
                    worker->commandList->clearColor();
                    worker->commandList->clearDepth();

                    submitDepthAccess(worker, fbStorage, false, interleavedDepthState);
                    submitRasterScene(worker, framebuffer, fbStorage, targetDrawCall.rasterScenes[sceneIndex], interleavedDepthState);

                    // Resolve the interleaved scene.
                    // TODO: Depth textures need to be thrown into a separate view vector for multisampled textures.
                    fbStorage->colorTarget->resolveTarget(worker, shaderLibrary);

                    interleavedBarriers.emplace_back(RenderTextureBarrier(colorRenderTarget->texture.get(), RenderTextureLayout::SHADER_READ));
                    interleavedBarriers.emplace_back(RenderTextureBarrier(depthRenderTarget->texture.get(), RenderTextureLayout::DEPTH_READ));
                }

                if (!interleavedBarriers.empty()) {
                    worker->commandList->barriers(RenderBarrierStage::COMPUTE, interleavedBarriers);
                }

                submitDepthAccess(worker, targetDrawCall.fbStorage, true, depthState);
                submitRaytracingScene(worker, targetDrawCall.fbStorage->colorTarget, rtScene);
            }
            else
#       endif
            {
                const RasterScene &rasterScene = targetDrawCall.rasterScenes[pair.first];
                submitDepthAccess(worker, targetDrawCall.fbStorage, false, depthState);
                submitRasterScene(worker, framebuffer, targetDrawCall.fbStorage, rasterScene, depthState);
                if (targetDrawCall.rs64LightsAfterScene == static_cast<int32_t>(pair.first)) {
                    if (framebuffer.rs64ShadowsOn && rs64Shadow.ready) {
                        rs64TimingMark(worker, RS64TimeShadows);
                        recordRS64Shadows(worker, framebuffer, depthState);
                        rs64TimingMark(worker, RS64TimeShadows);
                    }
                    if (framebuffer.rs64FogOn && rs64Shadow.ready && (rs64Shadow.tlas != nullptr)) {
                        rs64TimingMark(worker, RS64TimeFog);
                        recordRS64FogShafts(worker, framebuffer, depthState);
                        rs64TimingMark(worker, RS64TimeFog);
                    }
                    if (framebuffer.rs64LightsOn) {
                        rs64TimingMark(worker, RS64TimeLights);
                        recordRS64Lights(worker, framebuffer, depthState);
                        rs64TimingMark(worker, RS64TimeLights);
                    }
                }
            }
        }
    }

    void FramebufferRenderer::waitForUploaders() {
        shaderUploader->wait();

        for (BufferUploader *uploader : pendingUploaders) {
            uploader->wait();
        }
    }

#if RT_ENABLED
    void FramebufferRenderer::setRaytracingConfig(const RaytracingConfiguration &rtConfig, bool resolutionChanged) {
        assert(rtResources != nullptr);
        rtResources->setRaytracingConfig(rtConfig, resolutionChanged);
    }
#endif

    static rs64lights::Mat4 rs64ToMat4(const interop::float4x4 &m) {
        static_assert(sizeof(interop::float4x4) == sizeof(rs64lights::Mat4), "float4x4 layout");
        rs64lights::Mat4 r;
        memcpy(&r, &m, sizeof(r));
        return r;
    }

    // Two projections show the same 3D view when their raw view and projection matrices and their N64 viewport match (the radar shares the matrices but not the viewport).
    // ignoreProj: camera-space positions only need the same camera; the player craft is drawn under its own projection (nearer clip) in the same view.
    static bool rs64SameView(const DrawData &drawData, uint32_t a, uint32_t b, bool ignoreProj = false) {
        const size_t n = std::min({ drawData.viewTransforms.size(), drawData.projTransforms.size(), drawData.rspViewports.size() });
        if ((a >= n) || (b >= n)) {
            return false;
        }
        if (a == b) {
            return true;
        }
        const interop::RSPViewport &va = drawData.rspViewports[a];
        const interop::RSPViewport &vb = drawData.rspViewports[b];
        const bool sameViewport = (va.scale.x == vb.scale.x) && (va.scale.y == vb.scale.y) && (va.scale.z == vb.scale.z) && (va.translate.x == vb.translate.x) && (va.translate.y == vb.translate.y) && (va.translate.z == vb.translate.z);
        return sameViewport && rs64lights::sameMatrix(rs64ToMat4(drawData.viewTransforms[a]), rs64ToMat4(drawData.viewTransforms[b]), 1e-4f)
            && (ignoreProj || rs64lights::sameMatrix(rs64ToMat4(drawData.projTransforms[a]), rs64ToMat4(drawData.projTransforms[b]), 1e-4f));
    }

    void FramebufferRenderer::addFramebuffer(const DrawParams &p) {
        assert(p.fbStorage != nullptr);
        
        // Setup framebuffer pair data and descriptor set.
        const FramebufferPair &fbPair = p.curWorkload->fbPairs[p.fbPairIndex];
        interop::FramebufferParams fbParams;
        fbParams.resolution = { p.targetWidth / p.resolutionScale.x, p.targetHeight / p.resolutionScale.y };
        fbParams.resolutionScale = p.resolutionScale;
        fbParams.horizontalMisalignment = p.horizontalMisalignment;
        framebufferCount++;

        while (framebufferCount > framebufferVector.size()) {
            framebufferVector.emplace_back();
            framebufferVector.back().paramsBuffer = p.worker->device->createBuffer(RenderBufferDesc::UploadBuffer(256, RenderBufferFlag::CONSTANT));
            framebufferVector.back().descRealFbSet = std::make_unique<FramebufferRendererDescriptorFramebufferSet>(p.worker->device);
            framebufferVector.back().descDummyFbSet = std::make_unique<FramebufferRendererDescriptorFramebufferSet>(p.worker->device);
            framebufferVector.back().rs64LightsBuffer = p.worker->device->createBuffer(RenderBufferDesc::UploadBuffer(sizeof(interop::RS64LightsCB), RenderBufferFlag::CONSTANT));
            framebufferVector.back().rs64LightsSet = std::make_unique<RS64LightsDescriptorSet>(p.worker->device);
            if (p.worker->device->getCapabilities().rayQuery) {
                framebufferVector.back().rs64ShadowsBuffer = p.worker->device->createBuffer(RenderBufferDesc::UploadBuffer(sizeof(interop::RS64ShadowsCB), RenderBufferFlag::CONSTANT));
                framebufferVector.back().rs64ShadowsSet = std::make_unique<RS64TracedDescriptorSet>(p.worker->device);
                framebufferVector.back().rs64LightsShadowedSet = std::make_unique<RS64TracedDescriptorSet>(p.worker->device);
                framebufferVector.back().rs64FogBuffer = p.worker->device->createBuffer(RenderBufferDesc::UploadBuffer(sizeof(interop::RS64FogShaftsCB), RenderBufferFlag::CONSTANT));
                framebufferVector.back().rs64FogSet = std::make_unique<RS64TracedDescriptorSet>(p.worker->device);
                framebufferVector.back().rs64BlurHSet = std::make_unique<RS64ShadowBlurDescriptorSet>(p.worker->device);
                framebufferVector.back().rs64BlurVSet = std::make_unique<RS64ShadowBlurDescriptorSet>(p.worker->device);
                framebufferVector.back().rs64BlurHBuffer = p.worker->device->createBuffer(RenderBufferDesc::UploadBuffer(sizeof(interop::RS64ShadowBlurCB), RenderBufferFlag::CONSTANT));
                framebufferVector.back().rs64BlurVBuffer = p.worker->device->createBuffer(RenderBufferDesc::UploadBuffer(sizeof(interop::RS64ShadowBlurCB), RenderBufferFlag::CONSTANT));
            }
        }

        Framebuffer &framebuffer = framebufferVector[framebufferCount - 1];
        void *paramBufferBytes = framebuffer.paramsBuffer->map();
        memcpy(paramBufferBytes, &fbParams, sizeof(interop::FramebufferParams));
        framebuffer.paramsBuffer->unmap();

        RenderTexture *backgroundColorTexture = (p.fbStorage->colorTarget != nullptr) ? p.fbStorage->colorTarget->getResolvedTexture() : dummyColorTarget.get();
        RenderTextureView *backgroundColorTextureView = (p.fbStorage->colorTarget != nullptr) ? p.fbStorage->colorTarget->getResolvedTextureView() : dummyColorTargetView.get();
        framebuffer.descRealFbSet->setBuffer(framebuffer.descRealFbSet->FbParams, framebuffer.paramsBuffer.get(), sizeof(interop::FramebufferParams));
        framebuffer.descRealFbSet->setTexture(framebuffer.descRealFbSet->gBackgroundColor, backgroundColorTexture, RenderTextureLayout::SHADER_READ, backgroundColorTextureView);
        framebuffer.descRealFbSet->setTexture(framebuffer.descRealFbSet->gBackgroundDepth, p.fbStorage->depthTarget->texture.get(), RenderTextureLayout::DEPTH_READ, p.fbStorage->depthTarget->textureView.get());
        framebuffer.descDummyFbSet->setBuffer(framebuffer.descDummyFbSet->FbParams, framebuffer.paramsBuffer.get(), sizeof(interop::FramebufferParams));
        framebuffer.descDummyFbSet->setTexture(framebuffer.descDummyFbSet->gBackgroundColor, backgroundColorTexture, RenderTextureLayout::SHADER_READ, backgroundColorTextureView);
        framebuffer.descDummyFbSet->setTexture(framebuffer.descDummyFbSet->gBackgroundDepth, dummyDepthTarget.get(), RenderTextureLayout::DEPTH_READ, dummyDepthTargetView.get());

        // Store ubershader and other effect pipelines.
        const RasterShaderUber *rasterShaderUber = p.rasterShaderCache->getGPUShaderUber();
        rendererPipelineLayout = rasterShaderUber->pipelineLayout.get();
        postBlendDitherNoiseAddPipeline = rasterShaderUber->postBlendDitherNoiseAddPipeline.get();
        postBlendDitherNoiseSubPipeline = rasterShaderUber->postBlendDitherNoiseSubPipeline.get();
        postBlendDitherNoiseSubNegativePipeline = rasterShaderUber->postBlendDitherNoiseSubNegativePipeline.get();

        // Make a new render target draw call.
        RenderTargetDrawCall &targetDrawCall = framebuffer.renderTargetDrawCall;
        const DrawData &drawData = p.curWorkload->drawData;
        const DrawBuffers &drawBuffers = p.curWorkload->drawBuffers;
        const OutputBuffers &outputBuffers = p.curWorkload->outputBuffers;
        const RenderBuffer *screenPosRes = outputBuffers.screenPosBuffer.buffer.get();
        const RenderBuffer *tcRes = outputBuffers.genTexCoordBuffer.buffer.get();
        const RenderBuffer *indexRes = drawBuffers.faceIndicesBuffer.get();
        const RenderBuffer *shadedColRes = outputBuffers.shadedColBuffer.buffer.get();
        const RenderBuffer *worldPosRes = outputBuffers.worldPosBuffer.buffer.get();
        const RenderBuffer *worldNormRes = outputBuffers.worldNormBuffer.buffer.get();
        const RenderBuffer *worldVelRes = outputBuffers.worldVelBuffer.buffer.get();
        const RenderBuffer *triPosRes = drawBuffers.triPosBuffer.get();
        const RenderBuffer *triTcRes = drawBuffers.triTcBuffer.get();
        const RenderBuffer *triColRes = drawBuffers.triColorBuffer.get();
        const uint32_t PosStride = sizeof(float) * 4;
        const uint32_t ColStride = sizeof(float) * 4;
        const uint32_t WorldNormStride = sizeof(float) * 4;
        const uint32_t WorldVelStride = sizeof(float) * 4;
        const uint32_t TcStride = sizeof(float) * 2;
        const uint32_t IndexStride = sizeof(uint32_t);
        const uint32_t vertexCount = drawData.vertexCount();
        const uint32_t indexCount = uint32_t(drawData.faceIndices.size());
        const uint32_t rawTriVertexCount = drawData.rawTriVertexCount();
        vertexInputSlots[0] = RenderInputSlot(0, PosStride);
        vertexInputSlots[1] = RenderInputSlot(1, TcStride);
        vertexInputSlots[2] = RenderInputSlot(2, ColStride);
        vertexInputSlots[3] = RenderInputSlot(3, sizeof(uint32_t)); // per-vertex renderIndex
        indexedVertexViews[0] = RenderVertexBufferView(RenderBufferReference(screenPosRes), PosStride * vertexCount);
        indexedVertexViews[1] = RenderVertexBufferView(RenderBufferReference(tcRes), TcStride * vertexCount);
        indexedVertexViews[2] = RenderVertexBufferView(RenderBufferReference(shadedColRes), ColStride * vertexCount);
        // indexedVertexViews[3] / rawVertexViews[3] are set in endFramebuffers, after
        // renderIndexBuffer / rawRenderIndexBuffer are uploaded.
        indexBufferView = RenderIndexBufferView(RenderBufferReference(indexRes), IndexStride * indexCount, RenderFormat::R32_UINT);
        rawVertexViews[0] = RenderVertexBufferView(RenderBufferReference(triPosRes), PosStride * rawTriVertexCount);
        rawVertexViews[1] = RenderVertexBufferView(RenderBufferReference(triTcRes), TcStride * rawTriVertexCount);
        rawVertexViews[2] = RenderVertexBufferView(RenderBufferReference(triColRes), ColStride * rawTriVertexCount);
        testZIndexBuffer = outputBuffers.testZIndexBuffer.buffer.get();
        testZIndexBufferView = RenderIndexBufferView(testZIndexBuffer, uint32_t(outputBuffers.testZIndexBuffer.allocatedSize), RenderFormat::R32_UINT);

        RasterScene rasterScene;
        auto checkRasterScene = [&](RasterScene &rasterScene) {
            if (!rasterScene.instanceIndices.empty()) {
                uint32_t sceneIndex = static_cast<uint32_t>(targetDrawCall.rasterScenes.size());
                targetDrawCall.rasterScenes.push_back(rasterScene);
                targetDrawCall.sceneIndices.push_back({ sceneIndex, false });
                rasterScene.instanceIndices.clear();

                return true;
            }
            else {
                return false;
            }
        };

        targetDrawCall.rasterScenes.clear();
        targetDrawCall.rs64LightsAfterScene = -1;

#   if RT_ENABLED
        RaytracingScene rtScene;
        auto checkRtScene = [&](RaytracingScene &rtScene) {
            if (!rtScene.instanceIndices.empty()) {
                uint32_t sceneIndex = static_cast<uint32_t>(targetDrawCall.rtScenes.size());
                targetDrawCall.rtScenes.push_back(rtScene);
                targetDrawCall.sceneIndices.push_back({ sceneIndex, true });
                rtScene.instanceIndices.clear();

                return true;
            }
            else {
                return false;
            }
        };

        targetDrawCall.rtScenes.clear();
#   endif

        targetDrawCall.fbStorage = p.fbStorage;
        targetDrawCall.sceneIndices.clear();

        const float SimilarityPercentage = 0.1f; // TODO: Make more strict once VI ratios are in.
        const float scissorRatio = p.pixelAspect * static_cast<float>(fbPair.scissorRect.width(false, true)) / static_cast<float>(fbPair.scissorRect.height(false, true));
        const bool adjustRatio = (abs((scissorRatio / p.aspectRatioSource) - 1.0f) < SimilarityPercentage);
        const float aspectRatioScale = adjustRatio ? (p.aspectRatioTarget / p.aspectRatioSource) : 1.0f;
        // Full-width test for widescreen expansion, with one native pixel of slack (the game scissors 3D to 0..W-1 but clears 0..W).
        const auto coversFbWidth = [&](int32_t ulx, int32_t lrx) {
            const int32_t Slack = 4;
            return (ulx <= fbPair.scissorRect.ulx + Slack) && (lrx + Slack >= fbPair.scissorRect.lrx);
        };
        InstanceDrawCall instanceDrawCall;
        interop::RenderIndices renderIndices;

        // Per-vertex renderIndex stamp for draw-coalescing. Sized once per frame to
        // the global vertex counts (members are cleared in resetFramebuffers; the
        // first addFramebuffer of the frame grows them, later ones accumulate).
        if (renderIndexData.size() != drawData.vertexCount()) {
            renderIndexData.assign(drawData.vertexCount(), 0xFFFFFFFFu);
        }
        if (rawRenderIndexData.size() != drawData.rawTriVertexCount()) {
            rawRenderIndexData.assign(drawData.rawTriVertexCount(), 0xFFFFFFFFu);
        }
        uint32_t globalCallIndex = 0;
        const float wideWidth = p.fbWidth * p.resolutionScale.x;
        const float originalWidth = p.fbWidth * p.resolutionScale.y;
        const float commonHeight = float(p.targetHeight);
        framebuffer.viewport = RenderViewport(0.0f, 0.0f, wideWidth, commonHeight);
        
        const interop::float2 halfViewportSize = { framebuffer.viewport.width / 2.0f, framebuffer.viewport.height / 2.0f };
        const interop::float2 halfPixelOffset = { 1.0f / framebuffer.viewport.width, -1.0f / framebuffer.viewport.height };
        const float middleViewport = (wideWidth / 2.0f) - (originalWidth / 2.0f);
        const float extOriginPercentage = p.extAspectPercentage;
        uint32_t vertexTestZFaceIndicesStart = 0;
        int32_t vertexTestZCallIndex = -1;
        RenderViewport viewportClip;

        // Widescreen: clear the strips outside the original aspect to black on the first pass into this color image, so 2D screens that never clear get bars instead of the previous screen.
        bool firstPassForColorImage = true;
        for (uint32_t f = 0; f < p.fbPairIndex; f++) {
            if (p.curWorkload->fbPairs[f].colorImage.address == fbPair.colorImage.address) {
                firstPassForColorImage = false;
                break;
            }
        }

        if ((aspectRatioScale > 1.0f) && firstPassForColorImage && (p.fbStorage->colorTarget != nullptr) && !fbPair.scissorRect.isNull()) {
            const RenderRect full = convertFixedRect(fbPair.scissorRect, p.resolutionScale, p.fbWidth, 1.0f, extOriginPercentage, 0, G_EX_ORIGIN_NONE, G_EX_ORIGIN_NONE);
            const RenderRect inner = convertFixedRect(fbPair.scissorRect, p.resolutionScale, p.fbWidth, 1.0f / aspectRatioScale, extOriginPercentage, 0, G_EX_ORIGIN_NONE, G_EX_ORIGIN_NONE);
            const RenderRect strips[2] = { RenderRect(full.left, full.top, inner.left, full.bottom), RenderRect(inner.right, full.top, full.right, full.bottom) };
            for (const RenderRect &strip : strips) {
                if (strip.right <= strip.left) {
                    continue;
                }

                InstanceDrawCall stripClear;
                stripClear.type = InstanceDrawCall::Type::FillRect;
                stripClear.clearRect.rect = strip;
                stripClear.clearRect.color = RenderColor(0.0f, 0.0f, 0.0f, 1.0f);
                renderIndicesVector.push_back(interop::RenderIndices{});
                rasterScene.instanceIndices.push_back(static_cast<uint32_t>(instanceDrawCallVector.size()));
                instanceDrawCallVector.push_back(stripClear);
            }
        }

        const rs64lights::Config &lightsCfg = rs64lights::config();
        if (lightsCfg.shadows) {
            static bool s_capsLogged = false;
            if (!s_capsLogged) {
                s_capsLogged = true;
                const RenderDeviceCapabilities &caps = p.worker->device->getCapabilities();
                std::fprintf(stderr, "[rt-shadows] device raytracing=%d rayQuery=%d\n", (int)caps.raytracing, (int)caps.rayQuery);
            }
        }
        const bool lightsMSAA = (p.fbStorage->colorTarget != nullptr) && (p.fbStorage->colorTarget->multisampling.sampleCount > 1);
        const bool lightsWanted = lightsCfg.enabled && p.rs64Lights && (p.fbStorage->colorTarget != nullptr) && !lightsMSAA && !p.curWorkload->rs64Lights.empty();
        if (lightsCfg.enabled && lightsMSAA) {
            static bool s_warned = false;
            if (!s_warned) {
                s_warned = true;
                std::fprintf(stderr, "[rt-lights] MSAA target: light pass skipped\n");
            }
        }
        const bool shadowsWanted = rs64lights::tracePassWanted(lightsCfg.shadows, lightsCfg.ao, lightsCfg.gi, lightsCfg.reflections) && p.rs64Lights && (p.fbStorage->colorTarget != nullptr) && !lightsMSAA && p.curWorkload->rs64Sun.valid && p.worker->device->getCapabilities().rayQuery;
        const bool fogWanted = lightsCfg.fogShafts && p.rs64Lights && (p.fbStorage->colorTarget != nullptr) && !lightsMSAA && p.curWorkload->rs64Sun.valid && p.worker->device->getCapabilities().rayQuery;
        const bool lightShadowsWanted = rs64lights::sceneWanted(false, lightsWanted, lightsCfg.lightShadows, p.worker->device->getCapabilities().rayQuery, lightsMSAA);
        const bool sceneWanted = shadowsWanted || lightShadowsWanted || fogWanted;
        const bool postWanted = lightsWanted || shadowsWanted || fogWanted;
        // The main 3D view is the perspective projection with the most draws; the pass runs after its last fragment (same view and projection), so other views such as the radar stay unlit.
        uint32_t lightsProjIndex = UINT32_MAX;
        uint32_t lightsLastIndex = UINT32_MAX;
        uint32_t lightsSplitProj = UINT32_MAX;
        uint32_t lightsSplitCall = UINT32_MAX;
        if (postWanted) {
            uint32_t mostCalls = 0;
            for (uint32_t pr = 0; pr < fbPair.projectionCount; pr++) {
                const Projection &proj = fbPair.projections[pr];
                if ((proj.type == Projection::Type::Perspective) && !proj.scissorRect.isNull() && (proj.gameCallCount > mostCalls)) {
                    mostCalls = proj.gameCallCount;
                    lightsProjIndex = pr;
                }
            }
            if (lightsProjIndex != UINT32_MAX) {
                const uint32_t mainTi = fbPair.projections[lightsProjIndex].transformsIndex;
                for (uint32_t pr = 0; pr < fbPair.projectionCount; pr++) {
                    const Projection &proj = fbPair.projections[pr];
                    // The radar is a few draws in camera space under the same matrices and viewport, after the scene; tiny fragments are skipped so it stays unlit.
                    if ((proj.type == Projection::Type::Perspective) && !proj.scissorRect.isNull() && (proj.gameCallCount >= 8) && rs64SameView(drawData, proj.transformsIndex, mainTi)) {
                        lightsLastIndex = pr;
                    }
                }
                // The pass goes right after the view's last scene-geometry draw (depth-writing, not XLU or prim-depth): effect sprites drawn after it
                // (explosions, glows) are not darkened by the ground they float over.
                for (uint32_t pr = lightsProjIndex; (lightsLastIndex != UINT32_MAX) && (pr <= lightsLastIndex); pr++) {
                    const Projection &proj = fbPair.projections[pr];
                    if ((proj.type != Projection::Type::Perspective) || proj.scissorRect.isNull() || ((pr != lightsProjIndex) && !rs64SameView(drawData, proj.transformsIndex, mainTi))) {
                        continue;
                    }
                    for (uint32_t d = 0; d < proj.gameCallCount; d++) {
                        const GameCall &call = proj.gameCalls[d];
                        const interop::OtherMode &om = call.callDesc.otherMode;
                        rs64lights::CasterMode cm;
                        cm.zUpd = om.zUpd();
                        cm.primDepth = (om.zSource() == G_ZS_PRIM);
                        cm.xlu = ((om.L & ZMODE_MASK) == ZMODE_XLU);
                        cm.fillOrCopy = (om.cycleType() == G_CYC_FILL) || (om.cycleType() == G_CYC_COPY);
                        cm.extended = (call.callDesc.extendedType != DrawExtendedType::None);
                        cm.triangles = call.callDesc.triangleCount;
                        if (rs64lights::casterClass(cm) != rs64lights::CasterNone) {
                            lightsSplitProj = pr;
                            lightsSplitCall = d;
                        }
                    }
                }
            }
        }
        bool lightsSawPersp = false;
        bool lightsSplit = false;
        bool lightsHaveProj = false;
        uint32_t lightsTransformsIndex = 0;
        hlslpp::float2 lightsScreenScale = { 1.0f, 1.0f };
        hlslpp::float2 lightsScreenOffset = { 0.0f, 0.0f };
        for (uint32_t pr = 0; (pr < fbPair.projectionCount) && (globalCallIndex < p.maxGameCall); pr++) {
            const Projection &proj = fbPair.projections[pr];
            if (proj.scissorRect.isNull()) {
                continue;
            }

            if (postWanted && lightsHaveProj && !lightsSplit && (pr > lightsLastIndex)) {
                checkRasterScene(rasterScene);
                targetDrawCall.rs64LightsAfterScene = static_cast<int32_t>(targetDrawCall.rasterScenes.size()) - 1;
                lightsSplit = true;
            }
            lightsSawPersp = lightsSawPersp || (pr == lightsProjIndex);

#       if RT_ENABLED
            // TODO: Move heuristics of RT proj elsewhere?
            // TODO: Use detected scenes logic instead.
            const bool perspProj = (proj.type == Projection::Type::Perspective);
            bool rtProj = p.rtEnabled && perspProj && fbPair.depthWrite && targetDrawCall.rtScenes.empty(); // TODO: Remove the last condition once multiple heaps per RT scene are supported.

            // Make sure the matrices are compatible if we're switching to a new projection.
            bool rtProjCompatible = true;
            if (rtProj && (!rtScene.instanceIndices.empty())) {
                const float Threshold = 1e-6f;
                const float viewMatrixDiff = matrixDifference(drawData.modViewTransforms[proj.transformsIndex], rtScene.curViewMatrix);
                const float projMatrixDiff = matrixDifference(drawData.modProjTransforms[proj.transformsIndex], rtScene.curProjMatrix);
                rtProjCompatible = (viewMatrixDiff < Threshold) && (projMatrixDiff < Threshold);
            }

            // TODO: Remove this condition once multiple heaps per RT scene are supported.
            if (rtProj && !rtProjCompatible) {
                rtProj = false;
            }
#       endif
            
            auto &triangles = instanceDrawCall.triangles;
            triangles.screenScale = { 1.0f, 1.0f };
            triangles.screenOffset = halfPixelOffset;

            float projInvRatioScale = 1.0f / aspectRatioScale;
            const int16_t *viewportClipRatios = &drawData.viewportClipRatios[proj.transformsIndex * 4];
            const uint16_t viewportOrigin = drawData.viewportOrigins[proj.transformsIndex];
            if (proj.usesViewport()) {
                // The call's scissor spans the whole width of the framebuffer pair scissor. Custom origin must not be in use to be able to use the stretched viewport.
                const auto &viewport = drawData.rspViewports[proj.transformsIndex];
                FixedRect intersectionRect = proj.scissorRect.intersection(viewport.rect(viewportClipRatios));
                bool coversWholeWidth = !intersectionRect.isEmpty() && coversFbWidth(intersectionRect.ulx, intersectionRect.lrx);
                bool horizontalRatio = !intersectionRect.isEmpty() && (intersectionRect.width(true, true) > intersectionRect.height(true, true));
                bool useWideViewport = (viewportOrigin == G_EX_ORIGIN_NONE) && coversWholeWidth && horizontalRatio;                if (useWideViewport) {
                    projInvRatioScale = 1.0f;
                }
                else {
                    triangles.screenScale.x = originalWidth / wideWidth;

                    if (viewportOrigin < G_EX_ORIGIN_NONE) {
                        const float centerOffset = ((middleViewport * viewportOrigin) / G_EX_ORIGIN_CENTER) * extOriginPercentage + middleViewport * (1.0f - extOriginPercentage);
                        triangles.screenOffset.x = halfPixelOffset.x + ((centerOffset - middleViewport) / halfViewportSize.x);
                    }
                }

                viewportClip = convertViewportRect(viewport.rect(viewportClipRatios), p.resolutionScale, p.fbWidth, projInvRatioScale, extOriginPercentage, 0.0f, viewportOrigin, viewportOrigin);
            }

            if (postWanted && !lightsHaveProj && (pr == lightsProjIndex)) {
                lightsHaveProj = true;
                lightsTransformsIndex = proj.transformsIndex;
                lightsScreenScale = triangles.screenScale;
                lightsScreenOffset = triangles.screenOffset;
            }

            for (uint32_t d = 0; (d < proj.gameCallCount) && (globalCallIndex < p.maxGameCall); d++) {
                const GameCall &call = proj.gameCalls[d];
                renderIndices.instanceIndex = call.callDesc.callIndex;
                renderIndices.faceIndicesStart = call.meshDesc.faceIndicesStart;
                renderIndices.rdpTileIndex = call.callDesc.tileIndex;
                renderIndices.rdpTileCount = call.callDesc.tileCount;
                renderIndices.highlightColor = call.debuggerDesc.highlightColor;
                renderIndicesVector.push_back(renderIndices);

                uint32_t cycleType = call.callDesc.otherMode.cycleType();
                if (cycleType == G_CYC_FILL) {
                    instanceDrawCall.type = InstanceDrawCall::Type::FillRect;

                    auto &clearRect = instanceDrawCall.clearRect;
                    if (call.debuggerDesc.highlightColor > 0) {
                        clearRect.color = toRenderColor(ColorConverter::RGBA32::toRGBAF(call.debuggerDesc.highlightColor));
                    }
                    else {
                        if (p.fbStorage->colorTarget == nullptr) {
                            clearRect.depth = ColorConverter::D16::toF(call.callDesc.fillColor & 0xFFFF);
                        }
                        else if (fbPair.colorImage.siz == G_IM_SIZ_32b) {
                            clearRect.color = toRenderColor(ColorConverter::RGBA32::toRGBAF(call.callDesc.fillColor));
                        }
                        else {
                            clearRect.color = toRenderColor(ColorConverter::RGBA16::toRGBAF(call.callDesc.fillColor & 0xFFFF));
                        }
                    }

                    float invRatioScale = 1.0f / aspectRatioScale;
                    int32_t horizontalMisalignment = 0;

                    // A rect that spans the whole width of the scissor.
                    if (coversFbWidth(call.callDesc.rect.ulx, call.callDesc.rect.lrx)) {
                        invRatioScale = 1.0f;
                    }
                    // A regular rectangle that should correct its misalignment.
                    else {
                        horizontalMisalignment = int32_t(p.horizontalMisalignment);
                    }

                    clearRect.rect = convertFixedRect(call.callDesc.rect, p.resolutionScale, p.fbWidth, invRatioScale, extOriginPercentage, horizontalMisalignment, call.callDesc.rectLeftOrigin, call.callDesc.rectRightOrigin);

                    // A partial fill touching a scissor edge extends to the widened edge on that side (the game masks the horizon ring with half-width fills).
                    if (invRatioScale != 1.0f) {
                        const RenderRect full = convertFixedRect(fbPair.scissorRect, p.resolutionScale, p.fbWidth, 1.0f, extOriginPercentage, 0, G_EX_ORIGIN_NONE, G_EX_ORIGIN_NONE);
                        if (call.callDesc.rect.ulx <= fbPair.scissorRect.ulx + 4) {
                            clearRect.rect.left = full.left;
                        }

                        if (call.callDesc.rect.lrx + 4 >= fbPair.scissorRect.lrx) {
                            clearRect.rect.right = full.right;
                        }
                    }
                }
                else if (call.callDesc.extendedType != DrawExtendedType::None) {
                    switch (call.callDesc.extendedType) {
                    case DrawExtendedType::VertexTestZ:
                        instanceDrawCall.type = InstanceDrawCall::Type::VertexTestZ;
                        instanceDrawCall.vertexTestZ.vertexIndex = call.callDesc.extendedData.vertexTestZ.vertexIndex;
                        instanceDrawCall.vertexTestZ.resolutionScale = p.resolutionScale;
                        instanceDrawCall.vertexTestZ.srcIndexStart = call.meshDesc.faceIndicesStart + 3;
                        instanceDrawCall.vertexTestZ.dstIndexStart = vertexTestZFaceIndicesStart;
                        instanceDrawCall.vertexTestZ.indexCount = 0;
                        vertexTestZCallIndex = int32_t(instanceDrawCallVector.size());
                        break;
                    case DrawExtendedType::EndVertexTestZ:
                        instanceDrawCall.type = InstanceDrawCall::Type::Unknown;
                        vertexTestZCallIndex = -1;
                        break;
                    default:
                        assert(false && "Unknown extended type.");
                        break;
                    }
                }
                else {
#               if RT_ENABLED
                    if (rtProj) {
                        instanceDrawCall.type = InstanceDrawCall::Type::Raytracing;

                        if (hitGroupVector.empty()) {
                            // TODO: Support specialized shaders.
                            //const RaytracingShaderPrograms &shaderPrograms = p.ubershadersOnly ? rtState->shaderProgramsMap.find(UberShaderHash)->second : rtState->getShaderPrograms(call.shaderDesc);
                            const RaytracingShaderPrograms &shaderPrograms = rtState->shaderProgramsMap.find(UberShaderHash)->second;
                            hitGroupVector.emplace_back(shaderPrograms.surface);
                            hitGroupVector.emplace_back(shaderPrograms.shadow);
                        }

                        auto &raytracing = instanceDrawCall.raytracing;
                        raytracing.hitGroupIndex = 0;
                        raytracing.cullDisable = !call.shaderDesc.flags.culling;

                        const interop::OtherMode &otherMode = call.callDesc.otherMode;
                        raytracing.queryMask = (otherMode.zCmp() || otherMode.zUpd()) ? DepthRayQueryMask : NoDepthRayQueryMask;
                        if (drawData.extraParams[call.callDesc.callIndex].shadowCatcherFactor > 0.0f) {
                            raytracing.queryMask |= ShadowCatcherRayQueryMask;
                        }

                        const RenderBottomLevelASMesh asMesh(indexRes->at(call.meshDesc.faceIndicesStart *IndexStride), worldPosRes->at(0), RenderFormat::R32_UINT, RenderFormat::R32G32B32_FLOAT, call.callDesc.triangleCount * 3, vertexCount, PosStride, false);
                        rtResources->addBottomLevelASMesh(asMesh);

                        if (false) { // TODO: call.shaderDesc.flags.smoothNormal
                            RSPSmoothNormalGenerationCB rspSmoothNormal;
                            rspSmoothNormal.indexStart = call.meshDesc.faceIndicesStart;
                            rspSmoothNormal.indexCount = call.callDesc.triangleCount * 3;
                            rspSmoothNormalVector.push_back(rspSmoothNormal);
                        }
                    }
                    else 
#               endif
                    {
                        triangles.shaderDesc = call.shaderDesc;

                        RasterShader *gpuShader = p.ubershadersOnly ? nullptr : p.rasterShaderCache->getGPUShader(call.shaderDesc);
                        if (gpuShader != nullptr) {
                            triangles.pipeline = gpuShader->pipeline.get();
                        }
                        else {
                            const bool copyMode = (call.shaderDesc.otherMode.cycleType() == G_CYC_COPY);
                            triangles.pipeline = rasterShaderUber->getPipeline(
                                !copyMode && call.shaderDesc.otherMode.zCmp() && (call.shaderDesc.otherMode.zMode() != ZMODE_DEC),
                                !copyMode && call.shaderDesc.otherMode.zUpd(),
                                (call.shaderDesc.otherMode.cvgDst() == CVG_DST_WRAP) || (call.shaderDesc.otherMode.cvgDst() == CVG_DST_SAVE));
                        }
                        
                        triangles.faceCount = call.callDesc.triangleCount;
                        triangles.vertexTestZ = (vertexTestZCallIndex >= 0);
                        triangles.postBlendDitherNoise = false;

                        float invRatioScale = 1.0f / aspectRatioScale;
                        float horizontalMisalignment = 0.0f;
                        switch (proj.type) {
                        case Projection::Type::Perspective:
                        case Projection::Type::Orthographic: {
                            instanceDrawCall.type = InstanceDrawCall::Type::IndexedTriangles;
                            triangles.indexStart = triangles.vertexTestZ ? vertexTestZFaceIndicesStart : call.meshDesc.faceIndicesStart;
                            invRatioScale = projInvRatioScale;
                            break;
                        }
                        case Projection::Type::Rectangle: {
                            instanceDrawCall.type = InstanceDrawCall::Type::RegularRect;
                            triangles.indexStart = call.meshDesc.rawVertexStart;

                            bool tileCopiesUsed = false;
                            for (uint32_t t = 0; (t < call.callDesc.tileCount) && !tileCopiesUsed; t++) {
                                tileCopiesUsed = drawData.callTiles[call.callDesc.tileIndex + t].tileCopyUsed;
                            }

                            // The call's scissor spans the whole width of the framebuffer pair scissor. The rect must not be using extended origins.
                            const bool regularOrigins = (call.callDesc.rectLeftOrigin == G_EX_ORIGIN_NONE) && (call.callDesc.rectRightOrigin == G_EX_ORIGIN_NONE);
                            const bool coversScissorWidth = regularOrigins && coversFbWidth(call.callDesc.rect.ulx, call.callDesc.rect.lrx);
                            if (tileCopiesUsed || coversScissorWidth) {
                                invRatioScale = 1.0f;
                            }
                            else {
                                horizontalMisalignment = p.horizontalMisalignment;
                            }
                            RenderViewport viewportRect = convertViewportRect(call.callDesc.rect, p.resolutionScale, p.fbWidth, invRatioScale, extOriginPercentage, horizontalMisalignment, call.callDesc.rectLeftOrigin, call.callDesc.rectRightOrigin);
                            triangles.screenScale = { viewportRect.width / framebuffer.viewport.width, viewportRect.height / framebuffer.viewport.height };
                            triangles.screenOffset.x = halfPixelOffset.x + ((viewportRect.x + viewportRect.width / 2.0f) - halfViewportSize.x) / halfViewportSize.x;
                            triangles.screenOffset.y = halfPixelOffset.y + (halfViewportSize.y - (viewportRect.y + viewportRect.height / 2.0f)) / halfViewportSize.y;

                            if (p.postBlendNoise) {
                                // Indicate if post blend dither noise should be applied.
                                bool rgbDitherNoise = (call.shaderDesc.otherMode.rgbDither() == G_CD_NOISE);
                                triangles.postBlendDitherNoise = rgbDitherNoise && !call.shaderDesc.otherMode.zCmp() && !call.shaderDesc.otherMode.zUpd();
                                triangles.postBlendDitherNoiseNegative = p.postBlendNoiseNegative;
                            }

                            break;
                        }
                        case Projection::Type::Triangle: {
                            instanceDrawCall.type = InstanceDrawCall::Type::RawTriangles;
                            triangles.indexStart = call.meshDesc.rawVertexStart;
                            break;
                        }
                        case Projection::Type::None:
                        default:
                            break;
                        }

                        triangles.scissor = convertFixedRect(call.callDesc.scissorRect, p.resolutionScale, p.fbWidth, invRatioScale, extOriginPercentage, int32_t(horizontalMisalignment), call.callDesc.scissorLeftOrigin, call.callDesc.scissorRightOrigin);

                        bool usesViewport = (proj.type == Projection::Type::Perspective) || (proj.type == Projection::Type::Orthographic);
                        if (usesViewport) {
                            triangles.scissor = viewportScissorIntersection(viewportClip, triangles.scissor);
                        }
                        
                        if (triangles.vertexTestZ && usesViewport) {
                            instanceDrawCallVector[vertexTestZCallIndex].vertexTestZ.indexCount += call.callDesc.triangleCount * 3;
                            vertexTestZFaceIndicesStart += call.callDesc.triangleCount * 3;
                        }
                    }
                }

                // Determine to use the draw call either in the RT scene or the raster scene.
                const uint32_t instanceIndex = static_cast<uint32_t>(instanceDrawCallVector.size());

                {
                    const uint32_t ri = instanceIndex; // == renderIndex the shader uses
                    if (instanceDrawCall.type == InstanceDrawCall::Type::IndexedTriangles && !triangles.vertexTestZ) {
                        const uint32_t start = triangles.indexStart;
                        const uint32_t cnt = triangles.faceCount * 3;
                        for (uint32_t k = 0; k < cnt; k++) {
                            const uint32_t vid = drawData.faceIndices[start + k];
                            if (vid < renderIndexData.size()) {
                                if (renderIndexData[vid] != 0xFFFFFFFFu && renderIndexData[vid] != ri) {
                                    s_ri_collisions.fetch_add(1, std::memory_order_relaxed);
                                }
                                renderIndexData[vid] = ri;
                            }
                        }
                    }
                    else if (instanceDrawCall.type == InstanceDrawCall::Type::RawTriangles ||
                             instanceDrawCall.type == InstanceDrawCall::Type::RegularRect) {
                        const uint32_t start = triangles.indexStart; // rawVertexStart for the raw path
                        const uint32_t cnt = triangles.faceCount * 3;
                        for (uint32_t k = 0; k < cnt; k++) {
                            const uint32_t vid = start + k;
                            if (vid < rawRenderIndexData.size()) rawRenderIndexData[vid] = ri;
                        }
                    }
                }
#           if RT_ENABLED
                bool rtCall = instanceDrawCall.type == InstanceDrawCall::Type::Raytracing;
                if (rtCall) {
                    // If the current scene is not compatible, we submit it before the raster scene.
                    if (!rtProjCompatible) {
                        checkRtScene(rtScene);
                    }

                    const bool addedRasterScene = checkRasterScene(rasterScene);
                    if (rtScene.instanceIndices.empty()) {
                        float projRatioScale = 1.0f / aspectRatioScale;
                        float invRatioScale = 1.0f / aspectRatioScale;
                        const bool coversScissorWidth = coversFbWidth(proj.scissorRect.ulx, proj.scissorRect.lrx);
                        if (coversScissorWidth) {
                            invRatioScale = 1.0f;
                        }
                        else {
                            projRatioScale = 1.0f;
                        }

                        rtScene.curViewMatrix = drawData.modViewTransforms[proj.transformsIndex];
                        rtScene.curProjMatrix = drawData.modProjTransforms[proj.transformsIndex];
                        rtScene.prevViewMatrix = drawData.prevViewTransforms[proj.transformsIndex];
                        rtScene.prevProjMatrix = drawData.prevProjTransforms[proj.transformsIndex];

                        const auto &viewport = drawData.rspViewports[proj.transformsIndex];
                        rtScene.viewport = convertViewportRect(viewport.rect(viewportClipRatios), p.resolutionScale, p.fbWidth, invRatioScale, extOriginPercentage, 0.0f, G_EX_ORIGIN_NONE, G_EX_ORIGIN_NONE);
                        rtScene.scissor = convertFixedRect(proj.scissorRect, p.resolutionScale, p.fbWidth, invRatioScale, extOriginPercentage, 0, G_EX_ORIGIN_NONE, G_EX_ORIGIN_NONE);

                        rtScene.presetScene = p.presetScene;
                        rtScene.screenWidth = lround(static_cast<float>(p.fbWidth) * p.resolutionScale.x);
                        rtScene.screenHeight = lround(static_cast<float>(p.fbHeight) * p.resolutionScale.y);

                        if (proj.pointLightCount > 0) {
                            rtScene.pointLights = proj.pointLights.data();
                            rtScene.lightCount = proj.pointLightCount;
                        }
                        else {
                            rtScene.pointLights = nullptr;
                            rtScene.lightCount = 0;
                        }
                    }
                    else if (rtProjCompatible && addedRasterScene) {
                        targetDrawCall.sceneIndices.pop_back();
                        const uint32_t rasterSceneIndex = static_cast<uint32_t>(targetDrawCall.rasterScenes.size() - 1);
                        const auto &rasterScene = targetDrawCall.rasterScenes[rasterSceneIndex];
                        rtScene.interleavedRasters.push_back({ rasterSceneIndex, rasterScene.instanceIndices.back(), 0, 0 });
                    }

                    rtScene.instanceIndices.push_back(instanceIndex);
                }
                else 
#           endif
                {
                    rasterScene.instanceIndices.push_back(instanceIndex);
                }

                instanceDrawCallVector.push_back(instanceDrawCall);
                globalCallIndex++;
                if (postWanted && lightsHaveProj && !lightsSplit && (pr == lightsSplitProj) && (d == lightsSplitCall)) {
                    checkRasterScene(rasterScene);
                    targetDrawCall.rs64LightsAfterScene = static_cast<int32_t>(targetDrawCall.rasterScenes.size()) - 1;
                    lightsSplit = true;
                }
            }
        }

#   if RT_ENABLED
        checkRtScene(rtScene);
#   endif
        checkRasterScene(rasterScene);
        if (postWanted && lightsSawPersp && !lightsSplit) {
            targetDrawCall.rs64LightsAfterScene = static_cast<int32_t>(targetDrawCall.rasterScenes.size()) - 1;
        }
        const int32_t lightsScene = targetDrawCall.rs64LightsAfterScene;
        const bool lightsFilled = lightsWanted && (lightsScene >= 0) && lightsHaveProj && fillRS64Lights(p, framebuffer, lightsTransformsIndex, lightsScreenScale, lightsScreenOffset);
        bool shadowsFilled = false;
        bool fogFilled = false;
        framebuffer.rs64LightsShadowed = false;
        if (sceneWanted && (lightsScene >= 0) && lightsHaveProj) {
            // Casters: opaque, depth-writing, non-prim-depth triangles of the main view (prim depth = sky dome, sprites, glows).
            const uint32_t mainTi = fbPair.projections[lightsProjIndex].transformsIndex;
            // Terrain casts from the static HMP map when this frame's record-space origin solves; its drawn (geomorphing, view-limited) draws are then skipped.
            const rs64lights::TerrainFrame &tf = p.curWorkload->rs64Terrain;
            const bool terrainLerp = drawData.lerpWorldTransforms.size() == drawData.worldTransforms.size();
            const auto &terrainWorld = terrainLerp ? drawData.lerpWorldTransforms : drawData.worldTransforms;
            int32_t terrainOx = 0;
            int32_t terrainOz = 0;
            uint32_t terrainVotes = 0;
            if (lightsCfg.shadowTerrain && tf.map && !tf.mixed && (tf.transformIndex < terrainWorld.size())) {
                const bool hasPrev = (rs64Shadow.terrainOriginVersion == tf.map->version);
                terrainVotes = rs64lights::solveTerrainOrigin(*tf.map, tf.samples.data(), tf.samples.size(), terrainOx, terrainOz, hasPrev, rs64Shadow.terrainOriginX, rs64Shadow.terrainOriginZ);
            }
            rs64Shadow.terrainActive = (terrainVotes > 0);
            if (rs64Shadow.terrainActive) {
                rs64Shadow.terrainOriginVersion = tf.map->version;
                rs64Shadow.terrainOriginX = terrainOx;
                rs64Shadow.terrainOriginZ = terrainOz;
                rs64Shadow.terrainMap = tf.map;
                rs64lights::terrainInstanceTransform(rs64ToMat4(terrainWorld[tf.transformIndex]), terrainOx, terrainOz, rs64Shadow.terrainTransform);
            }
            const bool hitColors = lightsCfg.gi || lightsCfg.reflections;
            thread_local std::vector<std::pair<rs64lights::IndexRange, uint32_t>> terrainCalls;
            terrainCalls.clear();
            // Ranges accumulate over the workload's framebuffer pairs (the game can split one view across pairs); recordSetup consumes them.
            for (uint32_t pr = 0; pr < fbPair.projectionCount; pr++) {
                const Projection &proj = fbPair.projections[pr];
                if ((proj.type != Projection::Type::Perspective) || proj.scissorRect.isNull() || !rs64lights::casterFragment(pr == lightsProjIndex, rs64SameView(drawData, proj.transformsIndex, mainTi), proj.gameCallCount)) {
                    continue;
                }
                for (uint32_t d = 0; d < proj.gameCallCount; d++) {
                    const GameCall &call = proj.gameCalls[d];
                    const interop::OtherMode &om = call.callDesc.otherMode;
                    const uint32_t L = om.L;
                    rs64lights::CasterMode cm;
                    cm.zUpd = om.zUpd();
                    cm.primDepth = (om.zSource() == G_ZS_PRIM);
                    cm.xlu = ((L & ZMODE_MASK) == ZMODE_XLU);
                    cm.fillOrCopy = (om.cycleType() == G_CYC_FILL) || (om.cycleType() == G_CYC_COPY);
                    cm.extended = (call.callDesc.extendedType != DrawExtendedType::None);
                    cm.alphaCompare = (om.alphaCompare() != 0);
                    cm.cvgXAlpha = ((L & (CVG_X_ALPHA | ALPHA_CVG_SEL)) == (CVG_X_ALPHA | ALPHA_CVG_SEL));
                    cm.triangles = call.callDesc.triangleCount;
                    const bool drawnTerrain = rs64Shadow.terrainActive && (call.callDesc.minWorldMatrix == call.callDesc.maxWorldMatrix) && rs64lights::isTerrainTransform(tf, call.callDesc.minWorldMatrix);
                    const bool oneTransform = (call.callDesc.minWorldMatrix == call.callDesc.maxWorldMatrix);
                    const auto &noCast = p.curWorkload->rs64NoCast;
                    const auto &noCastCutout = p.curWorkload->rs64NoCastCutout;
                    const bool emissive = oneTransform && (std::find(noCast.begin(), noCast.end(), call.callDesc.minWorldMatrix) != noCast.end());
                    const bool glowCard = oneTransform && (std::find(noCastCutout.begin(), noCastCutout.end(), call.callDesc.minWorldMatrix) != noCastCutout.end());
                    const rs64lights::CasterClass cc = rs64lights::casterAfterDeny(rs64lights::casterClass(cm), emissive, glowCard, lightsCfg.shadowCutout);
                    if (hitColors && drawnTerrain) {
                        terrainCalls.push_back({ { call.meshDesc.faceIndicesStart, call.callDesc.triangleCount }, call.callDesc.callIndex });
                    }
                    if (hitColors && (emissive || glowCard) && !cm.fillOrCopy) {
                        rs64Shadow.emissiveRanges.push_back({ call.meshDesc.faceIndicesStart, call.callDesc.triangleCount });
                        rs64Shadow.emissiveCalls.push_back(call.callDesc.callIndex);
                    }
                    if (!drawnTerrain) {
                        const auto &craftTi = p.curWorkload->rs64Craft;
                        const bool craft = oneTransform && (std::find(craftTi.begin(), craftTi.end(), call.callDesc.minWorldMatrix) != craftTi.end());
                        if ((cc == rs64lights::CasterOpaque) && craft) {
                            rs64Shadow.craftRanges.push_back({ call.meshDesc.faceIndicesStart, call.callDesc.triangleCount });
                        }
                        else if (cc == rs64lights::CasterOpaque) {
                            rs64Shadow.ranges.push_back({ call.meshDesc.faceIndicesStart, call.callDesc.triangleCount });
                            rs64Shadow.rangeCalls.push_back(call.callDesc.callIndex);
                        }
                        else if (cc == rs64lights::CasterCutout) {
                            rs64Shadow.cutoutRanges.push_back({ { call.meshDesc.faceIndicesStart, call.callDesc.triangleCount }, call.callDesc.callIndex, call.callDesc.tileIndex });
                        }
                    }
                }
            }
            rs64Shadow.indices.clear();
            rs64lights::gatherCasterIndices(drawData.faceIndices.data(), drawData.faceIndices.size(), rs64Shadow.ranges.data(), rs64Shadow.ranges.size(), rs64Shadow.indices);
            rs64Shadow.craftIndices.clear();
            rs64lights::gatherCasterIndices(drawData.faceIndices.data(), drawData.faceIndices.size(), rs64Shadow.craftRanges.data(), rs64Shadow.craftRanges.size(), rs64Shadow.craftIndices);
            rs64Shadow.cutoutIndices.clear();
            rs64Shadow.cutoutTable.clear();
            rs64lights::buildCutoutScene(drawData.faceIndices.data(), drawData.faceIndices.size(), rs64Shadow.cutoutRanges.data(), rs64Shadow.cutoutRanges.size(), rs64Shadow.cutoutIndices, rs64Shadow.cutoutTable);
            rs64Shadow.worldPos = worldPosRes;
            rs64Shadow.texCoords = tcRes;
            rs64Shadow.texCoordsSize = outputBuffers.genTexCoordBuffer.allocatedSize;
            rs64Shadow.vertexCount = drawData.vertexCount();
            const bool sceneHasCasters = !rs64Shadow.indices.empty() || !rs64Shadow.cutoutIndices.empty() || !rs64Shadow.craftIndices.empty() || rs64Shadow.terrainActive;
            // Hangar: the largest draw (the ship, resting level) gives the world up; floor-facing receivers keep the game's painted shadow.
            rs64lights::Sun sun = p.curWorkload->rs64Sun;
            // Boot sequence: the authored key light is fixed in world space, so a moving camera does not swing the shadows.
            float viewR[9];
            if (sun.worldAuthored && (mainTi < drawData.viewTransforms.size()) && rs64lights::viewRotation(rs64ToMat4(drawData.viewTransforms[mainTi]), viewR)) {
                sun.valid = rs64lights::sunCameraDir(rs64lights::menuSunDir(), viewR, sun.dir);
            }
            if (sun.fromModel) {
                const Projection &mainProj = fbPair.projections[lightsProjIndex];
                uint32_t bestTris = 0;
                uint32_t bestTi = UINT32_MAX;
                for (uint32_t d = 0; d < mainProj.gameCallCount; d++) {
                    const GameCall &call = mainProj.gameCalls[d];
                    if (call.callDesc.triangleCount > bestTris) {
                        bestTris = call.callDesc.triangleCount;
                        bestTi = call.callDesc.minWorldMatrix;
                    }
                }
                const bool sunLerp = drawData.lerpWorldTransforms.size() == drawData.worldTransforms.size();
                const auto &sunWorld = sunLerp ? drawData.lerpWorldTransforms : drawData.worldTransforms;
                static const float kStraightDown[3] = { 0.0f, 1.0f, 0.0f };
                float up[3];
                if ((bestTi < sunWorld.size()) && rs64lights::modelSunDir(kStraightDown, rs64ToMat4(sunWorld[bestTi]), up)) {
                    sun.skip[0] = up[0];
                    sun.skip[1] = up[1];
                    sun.skip[2] = up[2];
                    sun.skip[3] = lightsCfg.hangarFloorCos;
                }
            }
            if (hitColors) {
                buildRS64HitColors(p, terrainCalls, (sun.fromModel && (sun.skip[3] > 0.0f)) ? sun.skip : nullptr);
            }
            shadowsFilled = shadowsWanted && sceneHasCasters && fillRS64Shadows(p, framebuffer, lightsTransformsIndex, lightsScreenScale, lightsScreenOffset, sun);
            if (fogWanted && sceneHasCasters) {
                // The game fogs only terrain and a few models, and terrain often draws in its own same-view projection.
                rs64lights::FogParams fogPick;
                for (uint32_t pr = 0; pr < fbPair.projectionCount; pr++) {
                    const Projection &fogProj = fbPair.projections[pr];
                    if ((fogProj.type != Projection::Type::Perspective) || !rs64SameView(drawData, fogProj.transformsIndex, mainTi)) {
                        continue;
                    }
                    for (uint32_t d = 0; d < fogProj.gameCallCount; d++) {
                        const GameCall &call = fogProj.gameCalls[d];
                        const uint32_t fi = call.meshDesc.faceIndicesStart;
                        if ((call.callDesc.triangleCount == 0) || (fi >= drawData.faceIndices.size())) {
                            continue;
                        }
                        const uint32_t v = drawData.faceIndices[fi];
                        const uint16_t fogIndex = (v < drawData.fogIndices.size()) ? drawData.fogIndices[v] : 0;
                        if ((fogIndex == 0) || (fogIndex > drawData.rspFog.size())) {
                            continue;
                        }
                        const interop::RSPFog &rf = drawData.rspFog[fogIndex - 1];
                        const interop::float4 &fc = call.callDesc.rdpParams.fogColor;
                        const float color[3] = { fc.x, fc.y, fc.z };
                        rs64lights::considerFogCall(fogPick, call.callDesc.triangleCount, rf.mul, rf.offset, color);
                    }
                }
                const rs64lights::FogParams &fog = rs64lights::workloadFog(rs64FogCache, p.curWorkload->workloadId, fogPick);
                const bool fogVisible = (lightsCfg.fogShaftsDebug != 0) || rs64lights::fogShaftsVisible(fog.color, lightsCfg.fogShaftsStrength);
                fogFilled = fogVisible && fillRS64FogShafts(p, framebuffer, lightsTransformsIndex, lightsScreenScale, lightsScreenOffset, sun, fog);
                if (lightsCfg.log) {
                    static uint32_t s_f = 0;
                    if ((++s_f % 120) == 1) {
                        std::fprintf(stderr, "[rt-fog] pick: valid=%d tris=%u mul=%.2f offset=%.2f color=(%.3f,%.3f,%.3f) filled=%d\n", (int)fog.valid, fog.tris, fog.mul, fog.offset, fog.color[0], fog.color[1], fog.color[2], (int)fogFilled);
                    }
                }
            }
            framebuffer.rs64LightsShadowed = lightsFilled && lightShadowsWanted && sceneHasCasters && (framebuffer.rs64LightsShadowedSet != nullptr);
            if (lightsCfg.log) {
                static uint32_t s_n = 0;
                if ((++s_n % 120) == 1) {
                    std::fprintf(stderr, "[rt-shadows] casters: ranges=%zu indices=%zu cutouts=%zu cutoutTris=%zu vertices=%u filled=%d sun=(%.3f,%.3f,%.3f)\n", rs64Shadow.ranges.size(), rs64Shadow.indices.size(), rs64Shadow.cutoutTable.size(), rs64Shadow.cutoutIndices.size() / 3, rs64Shadow.vertexCount, (int)shadowsFilled,
                        p.curWorkload->rs64Sun.dir[0], p.curWorkload->rs64Sun.dir[1], p.curWorkload->rs64Sun.dir[2]);
                    std::fprintf(stderr, "[rt-terrain] origin: noCast=%zu samples=%zu votes=%u ti=%u transforms=%zu mixed=%d origin=(%d,%d) active=%d lightsShadowed=%d\n", p.curWorkload->rs64NoCast.size(), tf.samples.size(), terrainVotes, tf.transformIndex, tf.transforms.size(), (int)tf.mixed, terrainOx, terrainOz, (int)rs64Shadow.terrainActive, (int)framebuffer.rs64LightsShadowed);
                }
            }
        }
        framebuffer.rs64LightsOn = lightsFilled;
        framebuffer.rs64ShadowsOn = shadowsFilled;
        framebuffer.rs64FogOn = fogFilled;
        framebuffer.rs64Target = p.fbStorage->colorTarget;
        if (!lightsFilled && !shadowsFilled && !fogFilled) {
            targetDrawCall.rs64LightsAfterScene = -1;
        }
        if (lightsCfg.log) {
            static uint32_t s_n[2] = {};
            uint32_t types[5] = {};
            for (uint32_t pr = 0; pr < fbPair.projectionCount; pr++) {
                ++types[std::min<uint32_t>(4, (uint32_t)fbPair.projections[pr].type)];
            }
            if ((types[1] > 0) && ((++s_n[(p.deltaTimeMs == 0.0f) ? 0 : 1] % 120) == 1)) {
                std::fprintf(stderr, "[rt-lights] fb: dt=%.2f projs none/persp/ortho/rect/tri=%u/%u/%u/%u/%u candidates=%zu", p.deltaTimeMs, types[0], types[1], types[2], types[3], types[4], p.curWorkload->rs64Lights.size());
                std::fprintf(stderr, " order[main=%u last=%u]:", lightsProjIndex, lightsLastIndex);
                for (uint32_t pr = 0; pr < fbPair.projectionCount; pr++) {
                    const Projection &q = fbPair.projections[pr];
                    const bool same = (lightsProjIndex != UINT32_MAX) && rs64SameView(drawData, q.transformsIndex, fbPair.projections[lightsProjIndex].transformsIndex);
                    std::fprintf(stderr, " %u:t%d:c%u:s%d:ti%u:sc(%d,%d,%d,%d)", pr, (int)q.type, q.gameCallCount, (int)same, q.transformsIndex, q.scissorRect.ulx, q.scissorRect.uly, q.scissorRect.lrx, q.scissorRect.lry);
                }
                std::fprintf(stderr, " color=%d msaa=%d main=%d split=%d proj=%d scene=%d filled=%d ti=%u modView=%zu modProj=%zu world=%zu lerp=%zu\n",
                    (int)(p.fbStorage->colorTarget != nullptr), (int)lightsMSAA, (int)lightsSawPersp, (int)lightsSplit, (int)lightsHaveProj, lightsScene, (int)lightsFilled,
                    lightsTransformsIndex, drawData.modViewTransforms.size(), drawData.modProjTransforms.size(), drawData.worldTransforms.size(), drawData.lerpWorldTransforms.size());
            }
        }
    }

    bool FramebufferRenderer::rs64ViewProj(const DrawParams &p, uint32_t transformsIndex, rs64lights::Mat4 &out) {
        const DrawData &drawData = p.curWorkload->drawData;

        // The projection processor only fills the modified transforms when it runs (interpolation or aspect adjustment); otherwise the raster pass uses the raw ones.
        const bool modified = (drawData.modViewTransforms.size() == drawData.viewTransforms.size()) && (drawData.modProjTransforms.size() == drawData.projTransforms.size());
        const auto &viewTransforms = modified ? drawData.modViewTransforms : drawData.viewTransforms;
        const auto &projTransforms = modified ? drawData.modProjTransforms : drawData.projTransforms;
        if ((transformsIndex >= viewTransforms.size()) || (transformsIndex >= projTransforms.size())) {
            return false;
        }

        out = rs64lights::mul(rs64ToMat4(viewTransforms[transformsIndex]), rs64ToMat4(projTransforms[transformsIndex]));
        return true;
    }

    bool FramebufferRenderer::rs64ViewFromScreen(const DrawParams &p, uint32_t transformsIndex, hlslpp::float2 screenScale, hlslpp::float2 screenOffset, rs64lights::Mat4 &out) {
        const DrawData &drawData = p.curWorkload->drawData;
        if (transformsIndex >= drawData.rspViewports.size()) {
            return false;
        }

        rs64lights::Mat4 viewProj;
        if (!rs64ViewProj(p, transformsIndex, viewProj)) {
            return false;
        }

        // Same resolution the raster pass maps screen positions with (FbParams.resolution).
        const interop::RSPViewport &vp = drawData.rspViewports[transformsIndex];
        rs64lights::ScreenMap map;
        map.vpScale[0] = vp.scale.x;
        map.vpScale[1] = vp.scale.y;
        map.vpScale[2] = vp.scale.z;
        map.vpTranslate[0] = vp.translate.x;
        map.vpTranslate[1] = vp.translate.y;
        map.vpTranslate[2] = vp.translate.z;
        map.res[0] = p.targetWidth / p.resolutionScale.x;
        map.res[1] = p.targetHeight / p.resolutionScale.y;
        map.screenScale[0] = screenScale.x;
        map.screenScale[1] = screenScale.y;
        map.screenOffset[0] = screenOffset.x;
        map.screenOffset[1] = screenOffset.y;
        return rs64lights::viewFromScreen(viewProj, map, out);
    }

    bool FramebufferRenderer::fillRS64Lights(const DrawParams &p, Framebuffer &framebuffer, uint32_t transformsIndex, hlslpp::float2 screenScale, hlslpp::float2 screenOffset) {
        const DrawData &drawData = p.curWorkload->drawData;
        const auto &candidates = p.curWorkload->rs64Lights;
        rs64lights::Mat4 inv;
        if (!rs64ViewFromScreen(p, transformsIndex, screenScale, screenOffset, inv)) {
            return false;
        }
        const bool lerp = drawData.lerpWorldTransforms.size() == drawData.worldTransforms.size();
        const auto &world = lerp ? drawData.lerpWorldTransforms : drawData.worldTransforms;
        thread_local std::vector<rs64lights::Resolved> resolved;
        resolved.clear();
        for (const rs64lights::Candidate &c : candidates) {
            const bool cameraSpace = (c.transformIndex == rs64lights::CameraSpace);
            if (!cameraSpace && ((c.transformIndex >= world.size()) || !rs64SameView(drawData, c.viewIndex, transformsIndex, c.modelUnits))) {
                continue;
            }
            const float v[4] = { c.local[0], c.local[1], c.local[2], 1.0f };
            float o[4] = { v[0], v[1], v[2], 1.0f };
            if (!cameraSpace) {
                rs64lights::transform(v, rs64ToMat4(world[c.transformIndex]), o);
            }
            rs64lights::Resolved r;
            r.pos[0] = o[0] / o[3];
            r.pos[1] = o[1] / o[3];
            r.pos[2] = o[2] / o[3];
            r.radius = c.radius;
            r.color[0] = c.color[0];
            r.color[1] = c.color[1];
            r.color[2] = c.color[2];
            r.intensity = c.intensity;
            r.falloff = c.falloff;
            r.shadowStart = c.shadowStart * rs64lights::config().spriteShadowStart;
            if (c.modelUnits && !cameraSpace) {
                const float s = rs64lights::transformScale(rs64ToMat4(world[c.transformIndex]));
                r.radius = c.radius * s;
                r.shadowStart = c.shadowStart * s;
            }
            resolved.push_back(r);
        }

        rs64lights::Resolved chosen[rs64lights::MaxLights];
        const size_t count = rs64lights::selectLights(resolved.data(), resolved.size(), chosen, rs64lights::MaxLights);
        if (count == 0) {
            return false;
        }
        if (rs64lights::config().log) {
            static uint32_t s_n = 0;
            if ((++s_n % 90) == 1) {
                std::fprintf(stderr, "[rt-lights] chosen %zu of %zu:", count, resolved.size());
                for (size_t i = 0; i < count && i < 6; ++i) {
                    std::fprintf(stderr, " (%.0f,%.0f,%.0f r=%.0f i=%.2f)", chosen[i].pos[0], chosen[i].pos[1], chosen[i].pos[2], chosen[i].radius, chosen[i].intensity);
                }
                std::fprintf(stderr, "\n");
            }
        }

        const rs64lights::Config &cfg = rs64lights::config();
        interop::RS64LightsCB cb = {};
        memcpy(&cb.viewFromScreen, &inv, sizeof(inv));
        cb.viewport = { framebuffer.viewport.x, framebuffer.viewport.y, framebuffer.viewport.width, framebuffer.viewport.height };
        cb.lightCount = static_cast<uint32_t>(count);
        cb.debugMode = static_cast<uint32_t>(cfg.debugMode);
        cb.gain = cfg.gain;
        cb.wrap = cfg.wrap;
        cb.debugRange = cfg.debugRange;
        cb.tMin = cfg.shadowTMin;
        cb.normalBias = cfg.shadowNormalBias;
        cb.terrainTMinScale = cfg.shadowTerrainTMin;
        cb.shadowParams = { cfg.lightShadowEnd, 0.0f, (cfg.terrainNormals && rs64Shadow.terrainNormalGpu) ? 1.0f : 0.0f, cfg.terrainNormalWindow };
        for (size_t i = 0; i < count; ++i) {
            cb.lightPosRadius[i] = { chosen[i].pos[0], chosen[i].pos[1], chosen[i].pos[2], chosen[i].radius };
            cb.lightColor[i] = { chosen[i].color[0], chosen[i].color[1], chosen[i].color[2], chosen[i].intensity };
            cb.lightParams[i] = { chosen[i].falloff, chosen[i].shadowStart, 0.0f, 0.0f };
        }

        void *bytes = framebuffer.rs64LightsBuffer->map();
        memcpy(bytes, &cb, sizeof(cb));
        framebuffer.rs64LightsBuffer->unmap();
        framebuffer.rs64LightsSet->setBuffer(framebuffer.rs64LightsSet->gLights, framebuffer.rs64LightsBuffer.get(), sizeof(interop::RS64LightsCB));
        framebuffer.rs64LightsSet->setTexture(framebuffer.rs64LightsSet->gDepth, p.fbStorage->depthTarget->texture.get(), RenderTextureLayout::DEPTH_READ, p.fbStorage->depthTarget->textureView.get());
        if (framebuffer.rs64LightsShadowedSet) {
            framebuffer.rs64LightsShadowedSet->setBuffer(framebuffer.rs64LightsShadowedSet->gParams, framebuffer.rs64LightsBuffer.get(), sizeof(interop::RS64LightsCB));
            framebuffer.rs64LightsShadowedSet->setTexture(framebuffer.rs64LightsShadowedSet->gDepth, p.fbStorage->depthTarget->texture.get(), RenderTextureLayout::DEPTH_READ, p.fbStorage->depthTarget->textureView.get());
        }
        framebuffer.rs64LightsDebug = (cfg.debugMode != 0);
        return true;
    }

    bool FramebufferRenderer::fillRS64Shadows(const DrawParams &p, Framebuffer &framebuffer, uint32_t transformsIndex, hlslpp::float2 screenScale, hlslpp::float2 screenOffset, const rs64lights::Sun &sun) {
        rs64lights::Mat4 inv;
        if (!sun.valid || !framebuffer.rs64ShadowsSet || !rs64ViewFromScreen(p, transformsIndex, screenScale, screenOffset, inv)) {
            return false;
        }

        const rs64lights::Config &cfg = rs64lights::config();
        interop::RS64ShadowsCB cb = {};
        memcpy(&cb.viewFromScreen, &inv, sizeof(inv));
        cb.viewport = { framebuffer.viewport.x, framebuffer.viewport.y, framebuffer.viewport.width, framebuffer.viewport.height };
        cb.sunDir = { sun.dir[0], sun.dir[1], sun.dir[2], 0.0f };
        cb.debugMode = static_cast<uint32_t>(cfg.shadowsDebug);
        cb.strength = cfg.shadowStrength;
        cb.tMin = cfg.shadowTMin;
        cb.tMax = cfg.shadowTMax;
        cb.normalBias = cfg.shadowNormalBias;
        cb.terrainTMinScale = cfg.shadowTerrainTMin;
        cb.receiverSkip = { sun.skip[0], sun.skip[1], sun.skip[2], sun.skip[3] };

        // Soft path (the only one that carries AO, GI and reflections): debug 0 or 5-8.
        const bool indirect = cfg.ao || cfg.gi || cfg.reflections;
        rs64lights::Mat4 viewProj;
        const bool soft = (cfg.softShadows || indirect) && ((cfg.shadowsDebug == 0) || (cfg.shadowsDebug >= 5)) && framebuffer.rs64BlurHSet && (p.fbStorage->colorTarget != nullptr) && rs64ViewProj(p, transformsIndex, viewProj);
        if (!soft && !cfg.shadows) {
            return false;
        }
        // Soft path without soft shadows (AO/GI/reflections only): one sharp sun ray.
        const float tanAngle = cfg.softShadows ? std::tan(cfg.sunAngleDeg * 3.14159265f / 180.0f) : 0.0f;
        const float maxRadius = cfg.softBlurPx * p.resolutionScale.y;
        cb.softRays = cfg.softShadows ? static_cast<uint32_t>(cfg.softRays) : 1u;
        const float res = p.resolutionScale.y;
        cb.aoParams = { float(cfg.aoRays), cfg.aoRange, cfg.aoStrength, cfg.aoBlurPx * res };
        cb.giParams = { float(cfg.giRays), cfg.giRange, cfg.giStrength, cfg.giBlurPx * res };
        cb.reflParams = { std::tan(cfg.reflRoughnessDeg * 3.14159265f / 180.0f), cfg.reflRange, cfg.reflStrength, cfg.giEmissive };
        cb.sunColor = { sun.color[0], sun.color[1], sun.color[2], 0.0f };
        cb.featureMask = (cfg.shadows ? 1u : 0u) | ((soft && cfg.ao) ? 2u : 0u) | ((soft && cfg.gi) ? 4u : 0u) | ((soft && cfg.reflections && sun.fromModel) ? 8u : 0u);
        cb.terrainParams = { cfg.terrainNormalWindow, 0.0f, 0.0f, 0.0f };
        rs64lights::Mat4 sfv;
        if (rs64lights::inverse(inv, sfv)) {
            memcpy(&cb.screenFromView, &sfv, sizeof(sfv));
        }
        cb.softParams = { tanAngle, soft ? rs64lights::pixelsPerUnit(viewProj, framebuffer.viewport.height) : 0.0f, maxRadius, (cfg.terrainNormals && rs64Shadow.terrainNormalGpu) ? (cfg.terrainTerminator ? 2.0f : 1.0f) : 0.0f };

        void *bytes = framebuffer.rs64ShadowsBuffer->map();
        memcpy(bytes, &cb, sizeof(cb));
        framebuffer.rs64ShadowsBuffer->unmap();
        framebuffer.rs64ShadowsSet->setBuffer(framebuffer.rs64ShadowsSet->gParams, framebuffer.rs64ShadowsBuffer.get(), sizeof(interop::RS64ShadowsCB));
        framebuffer.rs64ShadowsSet->setTexture(framebuffer.rs64ShadowsSet->gDepth, p.fbStorage->depthTarget->texture.get(), RenderTextureLayout::DEPTH_READ, p.fbStorage->depthTarget->textureView.get());
        framebuffer.rs64ShadowsDebug = (cfg.shadowsDebug != 0);
        framebuffer.rs64ShadowsSoft = soft;
        if (soft) {
            ensureRS64SoftTargets(p.worker, framebuffer, p.fbStorage->colorTarget->width, p.fbStorage->colorTarget->height);
            interop::RS64ShadowBlurCB bcb = {};
            memcpy(&bcb.viewFromScreen, &inv, sizeof(inv));
            bcb.viewport = cb.viewport;
            const float blurMax = std::max({ maxRadius, cfg.ao ? cb.aoParams.w : 0.0f, cfg.gi ? cb.giParams.w : 0.0f });
            bcb.params = { cfg.shadows ? cfg.shadowStrength : 0.0f, 0.02f, (cfg.shadowsDebug >= 5) ? 1.0f : 0.0f, blurMax };
            bcb.aoParams = { cfg.aoStrength, (cfg.shadowsDebug >= 5) ? float(cfg.shadowsDebug) : 0.0f, 0.0f, 0.0f };
            bcb.texelDir = { 1.0f, 0.0f };
            memcpy(framebuffer.rs64BlurHBuffer->map(), &bcb, sizeof(bcb));
            framebuffer.rs64BlurHBuffer->unmap();
            bcb.texelDir = { 0.0f, 1.0f };
            memcpy(framebuffer.rs64BlurVBuffer->map(), &bcb, sizeof(bcb));
            framebuffer.rs64BlurVBuffer->unmap();
            RS64ShadowBlurDescriptorSet &hs = *framebuffer.rs64BlurHSet;
            RS64ShadowBlurDescriptorSet &vs = *framebuffer.rs64BlurVSet;
            hs.setBuffer(hs.gBlur, framebuffer.rs64BlurHBuffer.get(), sizeof(interop::RS64ShadowBlurCB));
            hs.setTexture(hs.gDepth, p.fbStorage->depthTarget->texture.get(), RenderTextureLayout::DEPTH_READ, p.fbStorage->depthTarget->textureView.get());
            hs.setTexture(hs.gMask, framebuffer.rs64SoftMask.get(), RenderTextureLayout::SHADER_READ);
            hs.setTexture(hs.gIndirect, framebuffer.rs64SoftIndirect.get(), RenderTextureLayout::SHADER_READ);
            vs.setTexture(vs.gIndirect, framebuffer.rs64SoftIndirectPing.get(), RenderTextureLayout::SHADER_READ);
            vs.setBuffer(vs.gBlur, framebuffer.rs64BlurVBuffer.get(), sizeof(interop::RS64ShadowBlurCB));
            vs.setTexture(vs.gDepth, p.fbStorage->depthTarget->texture.get(), RenderTextureLayout::DEPTH_READ, p.fbStorage->depthTarget->textureView.get());
            vs.setTexture(vs.gMask, framebuffer.rs64SoftPing.get(), RenderTextureLayout::SHADER_READ);
        }
        return true;
    }

    // Hit colours in BLAS triangle order for the opaque casters (ranges, mirroring gatherCasterIndices' skips), the emissive draws (their own BLAS), the frame's
    // drawn terrain and the caster average: vertex-colour average x the call's render-tile TMEM average x prim.
    void FramebufferRenderer::buildRS64HitColors(const DrawParams &p, const std::vector<std::pair<rs64lights::IndexRange, uint32_t>> &terrainCalls, const float *floorUp) {
        RS64ShadowScene &s = rs64Shadow;
        const DrawData &drawData = p.curWorkload->drawData;
        const auto &fi = drawData.faceIndices;
        const auto &nc = drawData.normColBytes;
        const auto &callTex = p.curWorkload->rs64CallTex;
        auto drawColor = [&](const rs64lights::IndexRange &r, uint32_t callIndex, float out[3]) {
            float texRgb[3] = {};
            const float *tex = nullptr;
            if ((callIndex < callTex.size()) && ((float)callTex[callIndex].w > 0.0f)) {
                texRgb[0] = (float)callTex[callIndex].x;
                texRgb[1] = (float)callTex[callIndex].y;
                texRgb[2] = (float)callTex[callIndex].z;
                tex = texRgb;
            }
            float vtx[3] = { 1.0f, 1.0f, 1.0f };
            if (tex == nullptr) {
                float sum[3] = {};
                uint32_t n = 0;
                const uint32_t count = r.triangleCount * 3;
                const uint32_t step = std::max<uint32_t>(1, r.triangleCount / 16) * 3;
                for (uint32_t k = 0; (k < count) && (size_t(r.start) + k < fi.size()); k += step) {
                    const size_t v = fi[r.start + k];
                    if (v * 4 + 3 < nc.size()) {
                        sum[0] += nc[v * 4 + 0] / 255.0f;
                        sum[1] += nc[v * 4 + 1] / 255.0f;
                        sum[2] += nc[v * 4 + 2] / 255.0f;
                        ++n;
                    }
                }
                for (int k = 0; (k < 3) && (n > 0); ++k) {
                    vtx[k] = sum[k] / float(n);
                }
            }
            float prim[3] = { 1.0f, 1.0f, 1.0f };
            if (callIndex < drawData.rdpParams.size()) {
                prim[0] = (float)drawData.rdpParams[callIndex].primColor.x;
                prim[1] = (float)drawData.rdpParams[callIndex].primColor.y;
                prim[2] = (float)drawData.rdpParams[callIndex].primColor.z;
            }
            rs64lights::drawAverageColor(vtx, tex, prim, out);
        };
        // Area-weighted world-space normal sum of up to 64 of a draw's triangles (the hangar camera is unrotated, so world space is camera space).
        const auto &pos = drawData.posFloats;
        const auto &wIdx = drawData.worldIndices;
        float sumN[3] = {};
        auto drawNormalSum = [&](const rs64lights::IndexRange &r, float out[3]) {
            out[0] = out[1] = out[2] = 0.0f;
            const uint32_t step = std::max<uint32_t>(1, r.triangleCount / 64);
            for (uint32_t t = 0; t < r.triangleCount; t += step) {
                float w[3][3];
                bool ok = true;
                for (uint32_t k = 0; (k < 3) && ok; ++k) {
                    const size_t f = size_t(r.start) + t * 3 + k;
                    const size_t v = (f < fi.size()) ? fi[f] : SIZE_MAX;
                    if ((v == SIZE_MAX) || (v * 3 + 2 >= pos.size()) || (v >= wIdx.size()) || (wIdx[v] >= drawData.worldTransforms.size())) {
                        ok = false;
                        break;
                    }
                    const rs64lights::Mat4 m = rs64ToMat4(drawData.worldTransforms[wIdx[v]]);
                    for (int j = 0; j < 3; ++j) {
                        w[k][j] = pos[v * 3 + 0] * m.m[0][j] + pos[v * 3 + 1] * m.m[1][j] + pos[v * 3 + 2] * m.m[2][j] + m.m[3][j];
                    }
                }
                if (!ok) {
                    continue;
                }
                const float e1[3] = { w[1][0] - w[0][0], w[1][1] - w[0][1], w[1][2] - w[0][2] };
                const float e2[3] = { w[2][0] - w[0][0], w[2][1] - w[0][1], w[2][2] - w[0][2] };
                out[0] += e1[1] * e2[2] - e1[2] * e2[1];
                out[1] += e1[2] * e2[0] - e1[0] * e2[2];
                out[2] += e1[0] * e2[1] - e1[1] * e2[0];
            }
            return (out[0] != 0.0f) || (out[1] != 0.0f) || (out[2] != 0.0f);
        };
        auto buildTable = [&](const std::vector<rs64lights::IndexRange> &ranges, const std::vector<uint32_t> &calls, bool emissive, std::vector<rs64lights::HitColorEntry> &table, double sum[3]) {
            table.clear();
            uint32_t tri = 0;
            for (size_t r = 0; (r < ranges.size()) && (r < calls.size()); ++r) {
                const size_t count = size_t(ranges[r].triangleCount) * 3;
                if ((count == 0) || (size_t(ranges[r].start) + count > fi.size())) {
                    continue;
                }
                float c[3];
                drawColor(ranges[r], calls[r], c);
                uint32_t rgba = rs64lights::packHitColor(c[0], c[1], c[2], emissive);
                if ((floorUp != nullptr) && !emissive && drawNormalSum(ranges[r], sumN)) {
                    if (rs64lights::floorFacing(sumN, floorUp, 0.9f)) {
                        rgba = rs64lights::markFloor(rgba);
                    }
                }
                table.push_back({ tri, rgba });
                tri += ranges[r].triangleCount;
                for (int k = 0; k < 3; ++k) {
                    sum[k] += c[k];
                }
            }
        };
        double casterSum[3] = {};
        double emissiveSum[3] = {};
        buildTable(s.ranges, s.rangeCalls, false, s.hitTable, casterSum);
        buildTable(s.emissiveRanges, s.emissiveCalls, true, s.emissiveTable, emissiveSum);
        s.emissiveIndices.clear();
        rs64lights::gatherCasterIndices(fi.data(), fi.size(), s.emissiveRanges.data(), s.emissiveRanges.size(), s.emissiveIndices);
        const double n = double(std::max<size_t>(s.hitTable.size(), 1));
        s.averageColor = rs64lights::packHitColor(float(casterSum[0] / n), float(casterSum[1] / n), float(casterSum[2] / n), false);
        if (rs64lights::config().log) {
            static uint32_t s_n = 0;
            if ((++s_n % 120) < 4) {
                std::fprintf(stderr, "[rt-hit] casters=%zu emissive=%zu cutouts=%zu avg=%08X terrain=%08X:", s.hitTable.size(), s.emissiveTable.size(), s.cutoutTable.size(), s.averageColor, s.terrainColor);
                for (size_t i = 0; (i < s.hitTable.size()) && (i < 48); ++i) {
                    const uint32_t call = (i < s.rangeCalls.size()) ? s.rangeCalls[i] : 0;
                    const bool tex = (call < callTex.size()) && ((float)callTex[call].w > 0.0f);
                    std::fprintf(stderr, " %u:%08X%s(c%u)", s.hitTable[i].firstTriangle, s.hitTable[i].rgba, tex ? "t" : "", call);
                }
                std::fprintf(stderr, " | emissive:");
                for (size_t i = 0; (i < s.emissiveTable.size()) && (i < 16); ++i) {
                    const uint32_t call = (i < s.emissiveCalls.size()) ? s.emissiveCalls[i] : 0;
                    std::fprintf(stderr, " %u:%08X(c%u,%ut)", s.emissiveTable[i].firstTriangle, s.emissiveTable[i].rgba, call, (i < s.emissiveRanges.size()) ? s.emissiveRanges[i].triangleCount : 0);
                }
                std::fprintf(stderr, "\n");
            }
        }
        if (!terrainCalls.empty()) {
            double t[3] = {};
            for (const auto &tc : terrainCalls) {
                float c[3];
                drawColor(tc.first, tc.second, c);
                for (int k = 0; k < 3; ++k) {
                    t[k] += c[k];
                }
            }
            const double m = double(terrainCalls.size());
            s.terrainColor = rs64lights::packHitColor(float(t[0] / m), float(t[1] / m), float(t[2] / m), false);
        }
    }

    // Soft-shadow mask (shadow, blur radius) and ping textures at the colour target's size; recreated when it changes.
    void FramebufferRenderer::ensureRS64SoftTargets(RenderWorker *worker, Framebuffer &framebuffer, uint32_t width, uint32_t height) {
        if (framebuffer.rs64SoftMask && (framebuffer.rs64SoftW == width) && (framebuffer.rs64SoftH == height)) {
            return;
        }
        framebuffer.rs64SoftMaskFb.reset();
        framebuffer.rs64SoftPingFb.reset();
        framebuffer.rs64SoftMask = worker->device->createTexture(RenderTextureDesc::ColorTarget(width, height, RenderFormat::R16G16B16A16_FLOAT));
        framebuffer.rs64SoftPing = worker->device->createTexture(RenderTextureDesc::ColorTarget(width, height, RenderFormat::R16G16B16A16_FLOAT));
        framebuffer.rs64SoftIndirect = worker->device->createTexture(RenderTextureDesc::ColorTarget(width, height, RenderFormat::R16G16B16A16_FLOAT));
        framebuffer.rs64SoftIndirectPing = worker->device->createTexture(RenderTextureDesc::ColorTarget(width, height, RenderFormat::R16G16B16A16_FLOAT));
        const RenderTexture *maskTargets[] = { framebuffer.rs64SoftMask.get(), framebuffer.rs64SoftIndirect.get() };
        const RenderTexture *pingTargets[] = { framebuffer.rs64SoftPing.get(), framebuffer.rs64SoftIndirectPing.get() };
        framebuffer.rs64SoftMaskFb = worker->device->createFramebuffer(RenderFramebufferDesc(maskTargets, 2));
        framebuffer.rs64SoftPingFb = worker->device->createFramebuffer(RenderFramebufferDesc(pingTargets, 2));
        framebuffer.rs64SoftW = width;
        framebuffer.rs64SoftH = height;
    }

    void FramebufferRenderer::recordRS64Shadows(RenderWorker *worker, const Framebuffer &framebuffer, bool &depthState) {
        const RenderTargetDrawCall &targetDrawCall = framebuffer.renderTargetDrawCall;
        submitDepthAccess(worker, targetDrawCall.fbStorage, true, depthState);
        framebuffer.rs64ShadowsSet->setAccelerationStructure(framebuffer.rs64ShadowsSet->gScene, rs64Shadow.tlas.get());
        {
            // History slots are placed when the scene is built, after the fill.
            uint8_t *bytes = static_cast<uint8_t *>(framebuffer.rs64ShadowsBuffer->map());
            memcpy(bytes + offsetof(interop::RS64ShadowsCB, historyCam), rs64Shadow.historyCam, sizeof(rs64Shadow.historyCam));
            memcpy(bytes + offsetof(interop::RS64ShadowsCB, historyMask), &rs64Shadow.historyMask, sizeof(uint32_t));
            const uint32_t hitParams[4] = { rs64Shadow.hitCount, rs64Shadow.emissiveCount, rs64Shadow.terrainColor, rs64Shadow.averageColor };
            memcpy(bytes + offsetof(interop::RS64ShadowsCB, hitParams), hitParams, sizeof(hitParams));
            framebuffer.rs64ShadowsBuffer->unmap();
        }
        const RenderViewport &v = framebuffer.viewport;
        if (framebuffer.rs64ShadowsSoft) {
            // Trace into the mask, blur H into ping, then blur V and darken the colour target (left bound as colorWriteDepthRead, as submitDepthAccess set it).
            const RenderRect sc(int32_t(v.x), int32_t(v.y), int32_t(v.x + v.width), int32_t(v.y + v.height));
            RenderTexture *mask = framebuffer.rs64SoftMask.get();
            RenderTexture *ping = framebuffer.rs64SoftPing.get();
            RenderTexture *ind = framebuffer.rs64SoftIndirect.get();
            RenderTexture *indPing = framebuffer.rs64SoftIndirectPing.get();
            const RenderTextureBarrier toMask[] = { RenderTextureBarrier(mask, RenderTextureLayout::COLOR_WRITE), RenderTextureBarrier(ind, RenderTextureLayout::COLOR_WRITE) };
            worker->commandList->barriers(RenderBarrierStage::GRAPHICS, toMask, uint32_t(std::size(toMask)));
            worker->commandList->setFramebuffer(framebuffer.rs64SoftMaskFb.get());
            worker->commandList->clearColor(0, RenderColor(0.0f, 0.0f, 0.0f, 0.0f));
            worker->commandList->clearColor(1, RenderColor(0.0f, 0.0f, 0.0f, 0.0f));
            worker->commandList->setViewports(v);
            worker->commandList->setScissors(sc);
            worker->commandList->setPipeline(shaderLibrary->rs64ShadowsSoft.pipeline.get());
            worker->commandList->setGraphicsPipelineLayout(shaderLibrary->rs64ShadowsSoft.pipelineLayout.get());
            bindRS64Traced(worker, *framebuffer.rs64ShadowsSet, framebuffer.rs64ShadowsBuffer.get(), offsetof(interop::RS64ShadowsCB, cutoutDrawCount), false);
            worker->commandList->setVertexBuffers(0, nullptr, 0, nullptr);
            worker->commandList->drawInstanced(3, 1, 0, 0);

            const RenderTextureBarrier toPing[] = {
                RenderTextureBarrier(mask, RenderTextureLayout::SHADER_READ), RenderTextureBarrier(ind, RenderTextureLayout::SHADER_READ),
                RenderTextureBarrier(ping, RenderTextureLayout::COLOR_WRITE), RenderTextureBarrier(indPing, RenderTextureLayout::COLOR_WRITE)
            };
            worker->commandList->barriers(RenderBarrierStage::GRAPHICS, toPing, uint32_t(std::size(toPing)));
            worker->commandList->setFramebuffer(framebuffer.rs64SoftPingFb.get());
            worker->commandList->clearColor(0, RenderColor(0.0f, 0.0f, 0.0f, 0.0f));
            worker->commandList->clearColor(1, RenderColor(0.0f, 0.0f, 0.0f, 0.0f));
            worker->commandList->setViewports(v);
            worker->commandList->setScissors(sc);
            worker->commandList->setPipeline(shaderLibrary->rs64ShadowBlurH.pipeline.get());
            worker->commandList->setGraphicsPipelineLayout(shaderLibrary->rs64ShadowBlurH.pipelineLayout.get());
            worker->commandList->setGraphicsDescriptorSet(framebuffer.rs64BlurHSet->get(), 0);
            worker->commandList->drawInstanced(3, 1, 0, 0);

            const RenderTextureBarrier toCompose[] = { RenderTextureBarrier(ping, RenderTextureLayout::SHADER_READ), RenderTextureBarrier(indPing, RenderTextureLayout::SHADER_READ) };
            worker->commandList->barriers(RenderBarrierStage::GRAPHICS, toCompose, uint32_t(std::size(toCompose)));
            worker->commandList->setFramebuffer(targetDrawCall.fbStorage->colorWriteDepthRead.get());
            worker->commandList->setViewports(v);
            worker->commandList->setScissors(sc);
            const ShaderRecord &vrec = framebuffer.rs64ShadowsDebug ? shaderLibrary->rs64ShadowBlurVDebug : shaderLibrary->rs64ShadowBlurV;
            worker->commandList->setPipeline(vrec.pipeline.get());
            worker->commandList->setGraphicsPipelineLayout(vrec.pipelineLayout.get());
            worker->commandList->setGraphicsDescriptorSet(framebuffer.rs64BlurVSet->get(), 0);
            worker->commandList->drawInstanced(3, 1, 0, 0);
            return;
        }
        const ShaderRecord &record = framebuffer.rs64ShadowsDebug ? shaderLibrary->rs64ShadowsDebug : shaderLibrary->rs64Shadows;
        worker->commandList->setViewports(v);
        worker->commandList->setScissors(RenderRect(int32_t(v.x), int32_t(v.y), int32_t(v.x + v.width), int32_t(v.y + v.height)));
        worker->commandList->setPipeline(record.pipeline.get());
        worker->commandList->setGraphicsPipelineLayout(record.pipelineLayout.get());
        bindRS64Traced(worker, *framebuffer.rs64ShadowsSet, framebuffer.rs64ShadowsBuffer.get(), offsetof(interop::RS64ShadowsCB, cutoutDrawCount), false);
        worker->commandList->setVertexBuffers(0, nullptr, 0, nullptr);
        worker->commandList->drawInstanced(3, 1, 0, 0);
    }

    // Traced passes: this frame's cutout table, indices and tex-coords into set 3 (placeholders when there are none), the count patched into the
    // pass's constant buffer at record time (the fills run before recordSetup builds the table), then sets 0-2 as raster binds them.
    void FramebufferRenderer::bindRS64Traced(RenderWorker *worker, RS64TracedDescriptorSet &set, RenderBuffer *paramsBuffer, uint64_t countOffset, bool countAsFloat) {
        RS64ShadowScene &s = rs64Shadow;
        if (!rs64CutoutPlaceholder) {
            rs64CutoutPlaceholder = worker->device->createBuffer(RenderBufferDesc::UploadBuffer(64, RenderBufferFlag::STORAGE));
        }
        const bool have = (s.cutoutDrawCount > 0) && (s.texCoords != nullptr);
        const uint32_t count = have ? s.cutoutDrawCount : 0;
        if (have) {
            set.setBuffer(set.gCutoutDraws, s.cutoutTableGpu.get(), s.cutoutTableGpuCapacity, RenderBufferStructuredView(sizeof(rs64lights::CutoutEntry)));
            set.setBuffer(set.gCutoutIndices, s.cutoutIndexGpu.get(), s.cutoutIndexGpuCapacity);
            set.setBuffer(set.gTexCoords, const_cast<RenderBuffer *>(s.texCoords), s.texCoordsSize);
        }
        else {
            set.setBuffer(set.gCutoutDraws, rs64CutoutPlaceholder.get(), 64, RenderBufferStructuredView(sizeof(rs64lights::CutoutEntry)));
            set.setBuffer(set.gCutoutIndices, rs64CutoutPlaceholder.get(), 64);
            set.setBuffer(set.gTexCoords, rs64CutoutPlaceholder.get(), 64);
        }
        const RenderBufferStructuredView hitView(sizeof(rs64lights::HitColorEntry));
        if ((s.hitCount > 0) && s.hitTableGpu) {
            set.setBuffer(set.gHitColors, s.hitTableGpu.get(), s.hitTableGpuCapacity, hitView);
        }
        else {
            set.setBuffer(set.gHitColors, rs64CutoutPlaceholder.get(), 64, hitView);
        }
        if ((s.emissiveCount > 0) && s.emissiveTableGpu) {
            set.setBuffer(set.gEmissiveColors, s.emissiveTableGpu.get(), s.emissiveTableGpuCapacity, hitView);
        }
        else {
            set.setBuffer(set.gEmissiveColors, rs64CutoutPlaceholder.get(), 64, hitView);
        }
        if (s.terrainBlas && s.terrainNormalGpu) {
            set.setBuffer(set.gTerrainIndices, s.terrainIndexGpu.get(), s.terrainIndexBytes);
            set.setBuffer(set.gTerrainNormals, s.terrainNormalGpu.get(), s.terrainNormalBytes);
        }
        else {
            set.setBuffer(set.gTerrainIndices, rs64CutoutPlaceholder.get(), 64);
            set.setBuffer(set.gTerrainNormals, rs64CutoutPlaceholder.get(), 64);
        }
        uint8_t *bytes = static_cast<uint8_t *>(paramsBuffer->map());
        if (countAsFloat) {
            const float f = float(count);
            memcpy(bytes + countOffset, &f, sizeof(f));
        }
        else {
            memcpy(bytes + countOffset, &count, sizeof(count));
        }
        paramsBuffer->unmap();
        worker->commandList->setGraphicsDescriptorSet(descCommonSet->get(), 0);
        worker->commandList->setGraphicsDescriptorSet(descTextureSet->get(), 1);
        worker->commandList->setGraphicsDescriptorSet(descTextureSet->get(), 2);
        worker->commandList->setGraphicsDescriptorSet(set.get(), 3);
    }

    bool FramebufferRenderer::fillRS64FogShafts(const DrawParams &p, Framebuffer &framebuffer, uint32_t transformsIndex, hlslpp::float2 screenScale, hlslpp::float2 screenOffset, const rs64lights::Sun &sun, const rs64lights::FogParams &fog) {
        rs64lights::Mat4 inv;
        rs64lights::Mat4 viewProj;
        if (!sun.valid || !fog.valid || !framebuffer.rs64FogSet || !rs64ViewProj(p, transformsIndex, viewProj) || !rs64ViewFromScreen(p, transformsIndex, screenScale, screenOffset, inv)) {
            return false;
        }

        const rs64lights::Config &cfg = rs64lights::config();
        interop::RS64FogShaftsCB cb = {};
        memcpy(&cb.viewFromScreen, &inv, sizeof(inv));
        memcpy(&cb.viewProj, &viewProj, sizeof(viewProj));
        cb.viewport = { framebuffer.viewport.x, framebuffer.viewport.y, framebuffer.viewport.width, framebuffer.viewport.height };
        cb.sunDir = { sun.dir[0], sun.dir[1], sun.dir[2], 0.0f };
        cb.fogColor = { fog.color[0], fog.color[1], fog.color[2], 0.0f };
        cb.debugMode = static_cast<uint32_t>(cfg.fogShaftsDebug);
        cb.steps = static_cast<uint32_t>(cfg.fogShaftsSteps);
        cb.strength = cfg.fogShaftsStrength;
        cb.fogMul = fog.mul;
        cb.fogOffset = fog.offset;
        cb.tMin = cfg.shadowTMin;
        cb.tMax = cfg.shadowTMax;
        cb.terrainTMinScale = cfg.shadowTerrainTMin;
        cb.skyDepth = rs64lights::fogSkyDepth();
        cb.visBias = cfg.shadowNormalBias;

        void *bytes = framebuffer.rs64FogBuffer->map();
        memcpy(bytes, &cb, sizeof(cb));
        framebuffer.rs64FogBuffer->unmap();
        framebuffer.rs64FogSet->setBuffer(framebuffer.rs64FogSet->gParams, framebuffer.rs64FogBuffer.get(), sizeof(interop::RS64FogShaftsCB));
        framebuffer.rs64FogSet->setTexture(framebuffer.rs64FogSet->gDepth, p.fbStorage->depthTarget->texture.get(), RenderTextureLayout::DEPTH_READ, p.fbStorage->depthTarget->textureView.get());
        framebuffer.rs64FogDebug = (cfg.fogShaftsDebug != 0);
        return true;
    }

    void FramebufferRenderer::recordRS64FogShafts(RenderWorker *worker, const Framebuffer &framebuffer, bool &depthState) {
        const RenderTargetDrawCall &targetDrawCall = framebuffer.renderTargetDrawCall;
        submitDepthAccess(worker, targetDrawCall.fbStorage, true, depthState);
        framebuffer.rs64FogSet->setAccelerationStructure(framebuffer.rs64FogSet->gScene, rs64Shadow.tlas.get());
        const ShaderRecord &record = framebuffer.rs64FogDebug ? shaderLibrary->rs64FogShaftsDebug : shaderLibrary->rs64FogShafts;
        const RenderViewport &v = framebuffer.viewport;
        worker->commandList->setViewports(v);
        worker->commandList->setScissors(RenderRect(int32_t(v.x), int32_t(v.y), int32_t(v.x + v.width), int32_t(v.y + v.height)));
        worker->commandList->setPipeline(record.pipeline.get());
        worker->commandList->setGraphicsPipelineLayout(record.pipelineLayout.get());
        bindRS64Traced(worker, *framebuffer.rs64FogSet, framebuffer.rs64FogBuffer.get(), offsetof(interop::RS64FogShaftsCB, cutoutDrawCount), false);
        worker->commandList->setVertexBuffers(0, nullptr, 0, nullptr);
        worker->commandList->drawInstanced(3, 1, 0, 0);
    }

    void FramebufferRenderer::recordRS64TerrainBLAS(RenderWorker *worker) {
        RS64ShadowScene &s = rs64Shadow;
        RenderDevice *device = worker->device;
        thread_local std::vector<float> pos;
        thread_local std::vector<uint32_t> idx;
        rs64lights::buildTerrainMesh(*s.terrainMap, rs64lights::config().shadowTerrainSub, pos, idx);
        s.terrainBuiltVersion = s.terrainMap->version;
        s.terrainBlas.reset();
        s.terrainNormalGpu.reset();
        if (idx.empty()) {
            return;
        }
        const uint64_t posBytes = pos.size() * sizeof(float);
        const uint64_t idxBytes = idx.size() * sizeof(uint32_t);
        s.terrainPosBuffer = device->createBuffer(RenderBufferDesc::UploadBuffer(posBytes, RenderBufferFlag::ACCELERATION_STRUCTURE_INPUT));
        memcpy(s.terrainPosBuffer->map(), pos.data(), posBytes);
        s.terrainPosBuffer->unmap();
        s.terrainIndexBuffer = device->createBuffer(RenderBufferDesc::UploadBuffer(idxBytes, RenderBufferFlag::ACCELERATION_STRUCTURE_INPUT));
        memcpy(s.terrainIndexBuffer->map(), idx.data(), idxBytes);
        s.terrainIndexBuffer->unmap();
        const RenderBottomLevelASMesh mesh(RenderBufferReference(s.terrainIndexBuffer.get()), RenderBufferReference(s.terrainPosBuffer.get()), RenderFormat::R32_UINT, RenderFormat::R32G32B32_FLOAT, uint32_t(idx.size()), uint32_t(pos.size() / 3), sizeof(float) * 3, true);
        RenderBottomLevelASBuildInfo info;
        device->setBottomLevelASBuildInfo(info, &mesh, 1, false, true);
        s.terrainBlasBuffer = device->createBuffer(RenderBufferDesc::AccelerationStructureBuffer(info.accelerationStructureSize));
        s.terrainBlas = device->createAccelerationStructure(RenderAccelerationStructureDesc(RenderAccelerationStructureType::BOTTOM_LEVEL, RenderBufferReference(s.terrainBlasBuffer.get()), info.accelerationStructureSize));
        s.terrainScratch = device->createBuffer(RenderBufferDesc::DefaultBuffer(info.scratchSize, RenderBufferFlag::ACCELERATION_STRUCTURE_SCRATCH));
        worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderBufferBarrier(s.terrainBlasBuffer.get(), RenderBufferAccess::WRITE));
        worker->commandList->buildBottomLevelAS(s.terrainBlas.get(), RenderBufferReference(s.terrainScratch.get()), info);
        worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderBufferBarrier(s.terrainBlasBuffer.get(), RenderBufferAccess::READ));

        // Smooth normals and indices for the shadow pass's terrain receivers, GPU-local like the cutout tables.
        thread_local std::vector<float> nrm;
        rs64lights::buildTerrainNormals(pos, idx, nrm);
        s.terrainIndexBytes = idxBytes;
        s.terrainNormalBytes = nrm.size() * sizeof(float);
        s.terrainNormalBuffer = device->createBuffer(RenderBufferDesc::UploadBuffer(s.terrainNormalBytes));
        memcpy(s.terrainNormalBuffer->map(), nrm.data(), s.terrainNormalBytes);
        s.terrainNormalBuffer->unmap();
        s.terrainIndexGpu = device->createBuffer(RenderBufferDesc::DefaultBuffer(idxBytes, RenderBufferFlag::STORAGE));
        s.terrainNormalGpu = device->createBuffer(RenderBufferDesc::DefaultBuffer(s.terrainNormalBytes, RenderBufferFlag::STORAGE));
        const RenderBufferBarrier copyIn[] = { RenderBufferBarrier(s.terrainIndexGpu.get(), RenderBufferAccess::WRITE), RenderBufferBarrier(s.terrainNormalGpu.get(), RenderBufferAccess::WRITE) };
        worker->commandList->barriers(RenderBarrierStage::COPY, copyIn, uint32_t(std::size(copyIn)));
        worker->commandList->copyBufferRegion(RenderBufferReference(s.terrainIndexGpu.get()), RenderBufferReference(s.terrainIndexBuffer.get()), idxBytes);
        worker->commandList->copyBufferRegion(RenderBufferReference(s.terrainNormalGpu.get()), RenderBufferReference(s.terrainNormalBuffer.get()), s.terrainNormalBytes);
        const RenderBufferBarrier copyOut[] = { RenderBufferBarrier(s.terrainIndexGpu.get(), RenderBufferAccess::READ), RenderBufferBarrier(s.terrainNormalGpu.get(), RenderBufferAccess::READ) };
        worker->commandList->barriers(RenderBarrierStage::GRAPHICS, copyOut, uint32_t(std::size(copyOut)));
        if (rs64lights::config().log) {
            std::fprintf(stderr, "[rt-terrain] blas: map=%ux%u version=%llu tris=%zu bytes=%llu\n", s.terrainMap->width, s.terrainMap->height, (unsigned long long)s.terrainMap->version, idx.size() / 3, (unsigned long long)info.accelerationStructureSize);
        }
    }

    void FramebufferRenderer::recordRS64ShadowScene(RenderWorker *worker, bool dynamic) {
        RS64ShadowScene &s = rs64Shadow;
        RenderDevice *device = worker->device;
        auto grow = [device](std::unique_ptr<RenderBuffer> &buf, uint64_t &cap, uint64_t need, RenderBufferDesc desc) {
            if ((need <= cap) && buf) {
                return false;
            }
            cap = need + need / 2 + 256;
            desc.size = cap;
            buf = device->createBuffer(desc);
            return true;
        };

        s.retired.blas.reset();
        s.retired.blasBuffer.reset();
        if (s.terrainActive && (s.terrainBuiltVersion != s.terrainMap->version)) {
            recordRS64TerrainBLAS(worker);
        }

        RenderTopLevelASInstance instances[3 + 4 + 2];
        uint32_t instanceCount = 0;
        RenderBottomLevelASBuildInfo blasInfo;
        RenderBottomLevelASBuildInfo cutoutInfo;
        const bool opaque = dynamic && !s.indices.empty();
        s.cutoutDrawCount = 0;
        if (opaque) {
            const uint64_t indexBytes = s.indices.size() * sizeof(uint32_t);
            grow(s.indexBuffer, s.indexCapacity, indexBytes, RenderBufferDesc::UploadBuffer(indexBytes, RenderBufferFlag::ACCELERATION_STRUCTURE_INPUT));
            void *dst = s.indexBuffer->map();
            memcpy(dst, s.indices.data(), indexBytes);
            s.indexBuffer->unmap();
            const RenderBottomLevelASMesh mesh(RenderBufferReference(s.indexBuffer.get()), RenderBufferReference(s.worldPos), RenderFormat::R32_UINT, RenderFormat::R32G32B32_FLOAT, uint32_t(s.indices.size()), s.vertexCount, sizeof(float) * 4, true);
            device->setBottomLevelASBuildInfo(blasInfo, &mesh, 1, true, false);
            if (grow(s.blasBuffer, s.blasCapacity, blasInfo.accelerationStructureSize, RenderBufferDesc::AccelerationStructureBuffer(blasInfo.accelerationStructureSize)) || !s.blas) {
                s.blas = device->createAccelerationStructure(RenderAccelerationStructureDesc(RenderAccelerationStructureType::BOTTOM_LEVEL, RenderBufferReference(s.blasBuffer.get()), s.blasCapacity));
            }
            grow(s.blasScratch, s.blasScratchCapacity, blasInfo.scratchSize, RenderBufferDesc::DefaultBuffer(blasInfo.scratchSize, RenderBufferFlag::ACCELERATION_STRUCTURE_SCRATCH));
            instances[instanceCount++] = RenderTopLevelASInstance(RenderBufferReference(s.blasBuffer.get()), 0, 0x01, 0, true, RenderAffineTransform());
        }
        // Cutouts: a non-opaque BLAS hit-tested against the draw's texture alpha; left out when the tex-coord buffer is missing.
        if (dynamic && !s.cutoutIndices.empty() && (s.texCoords != nullptr)) {
            const uint64_t idxBytes = s.cutoutIndices.size() * sizeof(uint32_t);
            grow(s.cutoutIndexBuffer, s.cutoutIndexCapacity, idxBytes, RenderBufferDesc::UploadBuffer(idxBytes, RenderBufferFlag::ACCELERATION_STRUCTURE_INPUT | RenderBufferFlag::STORAGE));
            memcpy(s.cutoutIndexBuffer->map(), s.cutoutIndices.data(), idxBytes);
            s.cutoutIndexBuffer->unmap();
            const uint64_t tabBytes = s.cutoutTable.size() * sizeof(rs64lights::CutoutEntry);
            grow(s.cutoutTableBuffer, s.cutoutTableCapacity, tabBytes, RenderBufferDesc::UploadBuffer(tabBytes, RenderBufferFlag::STORAGE));
            memcpy(s.cutoutTableBuffer->map(), s.cutoutTable.data(), tabBytes);
            s.cutoutTableBuffer->unmap();
            grow(s.cutoutIndexGpu, s.cutoutIndexGpuCapacity, idxBytes, RenderBufferDesc::DefaultBuffer(idxBytes, RenderBufferFlag::STORAGE));
            grow(s.cutoutTableGpu, s.cutoutTableGpuCapacity, tabBytes, RenderBufferDesc::DefaultBuffer(tabBytes, RenderBufferFlag::STORAGE));
            s.cutoutIndexBytes = idxBytes;
            s.cutoutTableBytes = tabBytes;
            const RenderBottomLevelASMesh cmesh(RenderBufferReference(s.cutoutIndexBuffer.get()), RenderBufferReference(s.worldPos), RenderFormat::R32_UINT, RenderFormat::R32G32B32_FLOAT, uint32_t(s.cutoutIndices.size()), s.vertexCount, sizeof(float) * 4, false);
            device->setBottomLevelASBuildInfo(cutoutInfo, &cmesh, 1, true, false);
            if (grow(s.cutoutBlasBuffer, s.cutoutBlasCapacity, cutoutInfo.accelerationStructureSize, RenderBufferDesc::AccelerationStructureBuffer(cutoutInfo.accelerationStructureSize)) || !s.cutoutBlas) {
                s.cutoutBlas = device->createAccelerationStructure(RenderAccelerationStructureDesc(RenderAccelerationStructureType::BOTTOM_LEVEL, RenderBufferReference(s.cutoutBlasBuffer.get()), s.cutoutBlasCapacity));
            }
            grow(s.cutoutBlasScratch, s.cutoutBlasScratchCapacity, cutoutInfo.scratchSize, RenderBufferDesc::DefaultBuffer(cutoutInfo.scratchSize, RenderBufferFlag::ACCELERATION_STRUCTURE_SCRATCH));
            instances[instanceCount++] = RenderTopLevelASInstance(RenderBufferReference(s.cutoutBlasBuffer.get()), 2, 0x04, 0, true, RenderAffineTransform());
            s.cutoutDrawCount = uint32_t(s.cutoutTable.size());
        }
        // Emissive draws (lasers, glow and exhaust cards): seen by bounce and reflection rays (mask 0x80), never by shadow queries.
        RenderBottomLevelASBuildInfo emissiveInfo;
        const bool emissiveScene = dynamic && !s.emissiveIndices.empty() && !s.emissiveTable.empty();
        s.emissiveCount = 0;
        if (emissiveScene) {
            const uint64_t idxBytes = s.emissiveIndices.size() * sizeof(uint32_t);
            grow(s.emissiveIndexBuffer, s.emissiveIndexCapacity, idxBytes, RenderBufferDesc::UploadBuffer(idxBytes, RenderBufferFlag::ACCELERATION_STRUCTURE_INPUT));
            memcpy(s.emissiveIndexBuffer->map(), s.emissiveIndices.data(), idxBytes);
            s.emissiveIndexBuffer->unmap();
            const RenderBottomLevelASMesh emesh(RenderBufferReference(s.emissiveIndexBuffer.get()), RenderBufferReference(s.worldPos), RenderFormat::R32_UINT, RenderFormat::R32G32B32_FLOAT, uint32_t(s.emissiveIndices.size()), s.vertexCount, sizeof(float) * 4, true);
            device->setBottomLevelASBuildInfo(emissiveInfo, &emesh, 1, true, false);
            if (grow(s.emissiveBlasBuffer, s.emissiveBlasCapacity, emissiveInfo.accelerationStructureSize, RenderBufferDesc::AccelerationStructureBuffer(emissiveInfo.accelerationStructureSize)) || !s.emissiveBlas) {
                s.emissiveBlas = device->createAccelerationStructure(RenderAccelerationStructureDesc(RenderAccelerationStructureType::BOTTOM_LEVEL, RenderBufferReference(s.emissiveBlasBuffer.get()), s.emissiveBlasCapacity));
            }
            grow(s.emissiveBlasScratch, s.emissiveBlasScratchCapacity, emissiveInfo.scratchSize, RenderBufferDesc::DefaultBuffer(emissiveInfo.scratchSize, RenderBufferFlag::ACCELERATION_STRUCTURE_SCRATCH));
            instances[instanceCount++] = RenderTopLevelASInstance(RenderBufferReference(s.emissiveBlasBuffer.get()), 7, 0x80, 0, true, RenderAffineTransform());
            s.emissiveCount = uint32_t(s.emissiveTable.size());
        }
        // Craft draws cast like other drawn casters (mask 0x01) but never enter a history snapshot; hits take the caster average colour.
        RenderBottomLevelASBuildInfo craftInfo;
        const bool craftScene = dynamic && !s.craftIndices.empty();
        if (craftScene) {
            const uint64_t idxBytes = s.craftIndices.size() * sizeof(uint32_t);
            grow(s.craftIndexBuffer, s.craftIndexCapacity, idxBytes, RenderBufferDesc::UploadBuffer(idxBytes, RenderBufferFlag::ACCELERATION_STRUCTURE_INPUT));
            memcpy(s.craftIndexBuffer->map(), s.craftIndices.data(), idxBytes);
            s.craftIndexBuffer->unmap();
            const RenderBottomLevelASMesh kmesh(RenderBufferReference(s.craftIndexBuffer.get()), RenderBufferReference(s.worldPos), RenderFormat::R32_UINT, RenderFormat::R32G32B32_FLOAT, uint32_t(s.craftIndices.size()), s.vertexCount, sizeof(float) * 4, true);
            device->setBottomLevelASBuildInfo(craftInfo, &kmesh, 1, true, false);
            if (grow(s.craftBlasBuffer, s.craftBlasCapacity, craftInfo.accelerationStructureSize, RenderBufferDesc::AccelerationStructureBuffer(craftInfo.accelerationStructureSize)) || !s.craftBlas) {
                s.craftBlas = device->createAccelerationStructure(RenderAccelerationStructureDesc(RenderAccelerationStructureType::BOTTOM_LEVEL, RenderBufferReference(s.craftBlasBuffer.get()), s.craftBlasCapacity));
            }
            grow(s.craftBlasScratch, s.craftBlasScratchCapacity, craftInfo.scratchSize, RenderBufferDesc::DefaultBuffer(craftInfo.scratchSize, RenderBufferFlag::ACCELERATION_STRUCTURE_SCRATCH));
            instances[instanceCount++] = RenderTopLevelASInstance(RenderBufferReference(s.craftBlasBuffer.get()), 8, 0x01, 0, true, RenderAffineTransform());
        }
        // Hit colour tables, GPU-local like the cutout tables.
        s.hitCount = 0;
        const auto uploadTable = [&](const std::vector<rs64lights::HitColorEntry> &table, std::unique_ptr<RenderBuffer> &up, uint64_t &upCap, std::unique_ptr<RenderBuffer> &gpu, uint64_t &gpuCap, uint64_t &bytes) {
            bytes = table.size() * sizeof(rs64lights::HitColorEntry);
            grow(up, upCap, bytes, RenderBufferDesc::UploadBuffer(bytes, RenderBufferFlag::STORAGE));
            memcpy(up->map(), table.data(), bytes);
            up->unmap();
            grow(gpu, gpuCap, bytes, RenderBufferDesc::DefaultBuffer(bytes, RenderBufferFlag::STORAGE));
        };
        if (opaque && !s.hitTable.empty()) {
            uploadTable(s.hitTable, s.hitTableBuffer, s.hitTableCapacity, s.hitTableGpu, s.hitTableGpuCapacity, s.hitTableBytes);
            s.hitCount = uint32_t(s.hitTable.size());
        }
        if (emissiveScene) {
            uploadTable(s.emissiveTable, s.emissiveTableBuffer, s.emissiveTableCapacity, s.emissiveTableGpu, s.emissiveTableGpuCapacity, s.emissiveTableBytes);
        }
        if (s.terrainActive && s.terrainBlas) {
            RenderAffineTransform xf;
            memcpy(xf.m, s.terrainTransform, sizeof(xf.m));
            instances[instanceCount++] = RenderTopLevelASInstance(RenderBufferReference(s.terrainBlasBuffer.get()), 1, 0x02, 0, false, xf);
        }
        // Past frames' casters (slot i: id 3+i, mask 0x08 << i), moved from their camera into this one through map space.
        const rs64lights::Config &lcfg = rs64lights::config();
        const bool terrainNow = s.terrainActive && s.terrainBlas && (s.terrainMap != nullptr);
        s.historyMask = 0;
        for (int i = 0; terrainNow && (i < lcfg.shadowHistory); ++i) {
            RS64ShadowScene::History &h = s.history[i];
            float inv[3][4];
            if (!h.blas || (h.mapVersion != s.terrainMap->version) || !rs64lights::affineInverse(h.terrainTransform, inv)) {
                continue;
            }
            float delta[3][4];
            rs64lights::affineCompose(s.terrainTransform, inv, delta);
            RenderAffineTransform xf;
            memcpy(xf.m, delta, sizeof(xf.m));
            instances[instanceCount++] = RenderTopLevelASInstance(RenderBufferReference(h.blasBuffer.get()), 3 + i, uint8_t(0x08u << i), 0, true, xf);
            s.historyMask |= 0x08u << i;
            s.historyCam[i][0] = delta[0][3];
            s.historyCam[i][1] = delta[1][3];
            s.historyCam[i][2] = delta[2][3];
            s.historyCam[i][3] = lcfg.shadowHistorySkip;
        }
        if (lcfg.log && (lcfg.shadowHistory > 0)) {
            static uint32_t s_histLog = 0;
            if ((++s_histLog % 120) == 1) {
                std::fprintf(stderr, "[rt-shadows] history mask=0x%02X terrain=%d cam0=(%.0f,%.0f,%.0f) cam1=(%.0f,%.0f,%.0f)\n", s.historyMask, terrainNow ? 1 : 0,
                    s.historyCam[0][0], s.historyCam[0][1], s.historyCam[0][2], s.historyCam[1][0], s.historyCam[1][1], s.historyCam[1][2]);
            }
        }
        // Every interval frames this frame's opaque BLAS is kept as a history slot (it owns its geometry once built); the next frame grows a fresh one.
        const bool snapshot = opaque && terrainNow && (lcfg.shadowHistory > 0) && ((++s.historyFrame % uint32_t(lcfg.shadowHistoryInterval)) == 0);
        s.indices.clear();
        s.terrainActive = false;
        if (instanceCount == 0) {
            s.ready = false;
            return;
        }

        RenderTopLevelASBuildInfo tlasInfo;
        device->setTopLevelASBuildInfo(tlasInfo, instances, instanceCount, true, false);
        if (grow(s.tlasBuffer, s.tlasCapacity, tlasInfo.accelerationStructureSize, RenderBufferDesc::AccelerationStructureBuffer(tlasInfo.accelerationStructureSize)) || !s.tlas) {
            s.tlas = device->createAccelerationStructure(RenderAccelerationStructureDesc(RenderAccelerationStructureType::TOP_LEVEL, RenderBufferReference(s.tlasBuffer.get()), s.tlasCapacity));
        }
        grow(s.tlasScratch, s.tlasScratchCapacity, tlasInfo.scratchSize, RenderBufferDesc::DefaultBuffer(tlasInfo.scratchSize, RenderBufferFlag::ACCELERATION_STRUCTURE_SCRATCH));
        const uint64_t instBytes = tlasInfo.instancesBufferData.size();
        grow(s.instancesBuffer, s.instancesCapacity, instBytes, RenderBufferDesc::UploadBuffer(instBytes, RenderBufferFlag::ACCELERATION_STRUCTURE_INPUT));
        void *inst = s.instancesBuffer->map();
        memcpy(inst, tlasInfo.instancesBufferData.data(), instBytes);
        s.instancesBuffer->unmap();

        // Vulkan takes a barrier's source stage from the buffer's previous barrier; without these a freshly grown AS buffer orders its builds after nothing.
        worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderBufferBarrier(s.tlasBuffer.get(), RenderBufferAccess::WRITE));
        if (opaque) {
            worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderBufferBarrier(s.blasBuffer.get(), RenderBufferAccess::WRITE));
            worker->commandList->buildBottomLevelAS(s.blas.get(), RenderBufferReference(s.blasScratch.get()), blasInfo);
            worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderBufferBarrier(s.blasBuffer.get(), RenderBufferAccess::READ));
        }
        if (s.cutoutDrawCount > 0) {
            const RenderBufferBarrier copyIn[] = { RenderBufferBarrier(s.cutoutIndexGpu.get(), RenderBufferAccess::WRITE), RenderBufferBarrier(s.cutoutTableGpu.get(), RenderBufferAccess::WRITE) };
            worker->commandList->barriers(RenderBarrierStage::COPY, copyIn, uint32_t(std::size(copyIn)));
            worker->commandList->copyBufferRegion(RenderBufferReference(s.cutoutIndexGpu.get()), RenderBufferReference(s.cutoutIndexBuffer.get()), s.cutoutIndexBytes);
            worker->commandList->copyBufferRegion(RenderBufferReference(s.cutoutTableGpu.get()), RenderBufferReference(s.cutoutTableBuffer.get()), s.cutoutTableBytes);
            const RenderBufferBarrier copyOut[] = { RenderBufferBarrier(s.cutoutIndexGpu.get(), RenderBufferAccess::READ), RenderBufferBarrier(s.cutoutTableGpu.get(), RenderBufferAccess::READ) };
            worker->commandList->barriers(RenderBarrierStage::GRAPHICS, copyOut, uint32_t(std::size(copyOut)));
            worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderBufferBarrier(s.cutoutBlasBuffer.get(), RenderBufferAccess::WRITE));
            worker->commandList->buildBottomLevelAS(s.cutoutBlas.get(), RenderBufferReference(s.cutoutBlasScratch.get()), cutoutInfo);
            worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderBufferBarrier(s.cutoutBlasBuffer.get(), RenderBufferAccess::READ));
        }
        if (emissiveScene) {
            worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderBufferBarrier(s.emissiveBlasBuffer.get(), RenderBufferAccess::WRITE));
            worker->commandList->buildBottomLevelAS(s.emissiveBlas.get(), RenderBufferReference(s.emissiveBlasScratch.get()), emissiveInfo);
            worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderBufferBarrier(s.emissiveBlasBuffer.get(), RenderBufferAccess::READ));
        }
        if (craftScene) {
            worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderBufferBarrier(s.craftBlasBuffer.get(), RenderBufferAccess::WRITE));
            worker->commandList->buildBottomLevelAS(s.craftBlas.get(), RenderBufferReference(s.craftBlasScratch.get()), craftInfo);
            worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderBufferBarrier(s.craftBlasBuffer.get(), RenderBufferAccess::READ));
        }
        for (int t = 0; t < 2; ++t) {
            RenderBuffer *gpu = (t == 0) ? s.hitTableGpu.get() : s.emissiveTableGpu.get();
            RenderBuffer *up = (t == 0) ? s.hitTableBuffer.get() : s.emissiveTableBuffer.get();
            const uint64_t bytes = (t == 0) ? s.hitTableBytes : s.emissiveTableBytes;
            const bool active = (t == 0) ? (s.hitCount > 0) : (s.emissiveCount > 0);
            if (!active || (gpu == nullptr) || (up == nullptr)) {
                continue;
            }
            worker->commandList->barriers(RenderBarrierStage::COPY, RenderBufferBarrier(gpu, RenderBufferAccess::WRITE));
            worker->commandList->copyBufferRegion(RenderBufferReference(gpu), RenderBufferReference(up), bytes);
            worker->commandList->barriers(RenderBarrierStage::GRAPHICS, RenderBufferBarrier(gpu, RenderBufferAccess::READ));
        }
        worker->commandList->buildTopLevelAS(s.tlas.get(), RenderBufferReference(s.tlasScratch.get()), RenderBufferReference(s.instancesBuffer.get()), tlasInfo);
        worker->commandList->barriers(RenderBarrierStage::GRAPHICS_AND_COMPUTE, RenderBufferBarrier(s.tlasBuffer.get(), RenderBufferAccess::READ));
        s.ready = true;
        if (snapshot) {
            RS64ShadowScene::History &h = s.history[s.historyNext % uint32_t(lcfg.shadowHistory)];
            s.retired.blas = std::move(h.blas);
            s.retired.blasBuffer = std::move(h.blasBuffer);
            h.blasBuffer = std::move(s.blasBuffer);
            h.blas = std::move(s.blas);
            memcpy(h.terrainTransform, s.terrainTransform, sizeof(h.terrainTransform));
            h.mapVersion = s.terrainMap->version;
            s.blasCapacity = 0;
            ++s.historyNext;
        }
    }

    void FramebufferRenderer::rs64TimingBegin(RenderWorker *worker) {
        rs64TimingMarks = 0;
        if (!rs64lights::config().logTiming) {
            return;
        }
        if (!rs64TimingPool) {
            rs64TimingPool = worker->device->createQueryPool(uint32_t(std::size(rs64TimingLabels)));
        }
        worker->commandList->resetQueryPool(rs64TimingPool.get(), 0, uint32_t(std::size(rs64TimingLabels)));
    }

    // Called once before and once after a pass; the pair's difference is that pass's GPU time.
    void FramebufferRenderer::rs64TimingMark(RenderWorker *worker, uint8_t pass) {
        if (!rs64TimingPool || (rs64TimingMarks >= std::size(rs64TimingLabels))) {
            return;
        }
        rs64TimingLabels[rs64TimingMarks] = pass;
        worker->commandList->writeTimestamp(rs64TimingPool.get(), rs64TimingMarks);
        ++rs64TimingMarks;
    }

    // After the workload's command list has executed and been waited on.
    void FramebufferRenderer::rs64TimingReport() {
        if (!rs64TimingPool) {
            return;
        }
        if (rs64TimingMarks >= 2) {
            rs64TimingPool->queryResults();
            const uint64_t *ts = rs64TimingPool->getResults();
            for (uint32_t i = 0; (i + 1) < rs64TimingMarks; i += 2) {
                rs64TimingSum[rs64TimingLabels[i]] += double(ts[i + 1] - ts[i]) / 1000000.0;
            }
        }
        if (++rs64TimingFrames >= 120) {
            const double n = double(rs64TimingFrames);
            std::fprintf(stderr, "[rt-timing] avg ms/workload over %u: scene=%.3f shadows=%.3f fog=%.3f lights=%.3f total=%.3f\n", rs64TimingFrames,
                rs64TimingSum[RS64TimeScene] / n, rs64TimingSum[RS64TimeShadows] / n, rs64TimingSum[RS64TimeFog] / n, rs64TimingSum[RS64TimeLights] / n,
                (rs64TimingSum[RS64TimeScene] + rs64TimingSum[RS64TimeShadows] + rs64TimingSum[RS64TimeFog] + rs64TimingSum[RS64TimeLights]) / n);
            for (double &v : rs64TimingSum) {
                v = 0.0;
            }
            rs64TimingFrames = 0;
        }
    }

    void FramebufferRenderer::recordRS64Lights(RenderWorker *worker, const Framebuffer &framebuffer, bool &depthState) {
        const RenderTargetDrawCall &targetDrawCall = framebuffer.renderTargetDrawCall;
        submitDepthAccess(worker, targetDrawCall.fbStorage, true, depthState);
        // Shadowed variant when this frame's scene was built; otherwise the Phase 1 pass.
        const bool shadowed = framebuffer.rs64LightsShadowed && rs64Shadow.ready && (rs64Shadow.tlas != nullptr);
        const ShaderRecord &record = shadowed ? (framebuffer.rs64LightsDebug ? shaderLibrary->rs64LightsShadowedDebug : shaderLibrary->rs64LightsShadowed)
                                              : (framebuffer.rs64LightsDebug ? shaderLibrary->rs64LightsDebug : shaderLibrary->rs64Lights);
        const RenderViewport &v = framebuffer.viewport;
        worker->commandList->setViewports(v);
        worker->commandList->setScissors(RenderRect(int32_t(v.x), int32_t(v.y), int32_t(v.x + v.width), int32_t(v.y + v.height)));
        worker->commandList->setPipeline(record.pipeline.get());
        worker->commandList->setGraphicsPipelineLayout(record.pipelineLayout.get());
        if (shadowed) {
            framebuffer.rs64LightsShadowedSet->setAccelerationStructure(framebuffer.rs64LightsShadowedSet->gScene, rs64Shadow.tlas.get());
            bindRS64Traced(worker, *framebuffer.rs64LightsShadowedSet, framebuffer.rs64LightsBuffer.get(), offsetof(interop::RS64LightsCB, shadowParams) + sizeof(float), true);
        }
        else {
            worker->commandList->setGraphicsDescriptorSet(framebuffer.rs64LightsSet->get(), 0);
        }
        worker->commandList->setVertexBuffers(0, nullptr, 0, nullptr);
        worker->commandList->drawInstanced(3, 1, 0, 0);
    }

    void FramebufferRenderer::endFramebuffers(RenderWorker *worker, const DrawBuffers *drawBuffers, const OutputBuffers *outputBuffers, bool rtEnabled) {
        bool shaderViewRtEnabled = false;
        std::vector<BufferUploader::Upload> shaderUploads = {
            { renderIndicesVector.data(), { 0, renderIndicesVector.size() }, sizeof(interop::RenderIndices), RenderBufferFlag::STORAGE, { }, &renderIndicesBuffer},
            { &frameParams, { 0, 1 }, sizeof(interop::FrameParams), RenderBufferFlag::CONSTANT, { }, &frameParamsBuffer}
        };
        if (!renderIndexData.empty()) {
            shaderUploads.push_back({ renderIndexData.data(), { 0, renderIndexData.size() }, sizeof(uint32_t), RenderBufferFlag::VERTEX, { }, &renderIndexBuffer });
        }
        if (!rawRenderIndexData.empty()) {
            shaderUploads.push_back({ rawRenderIndexData.data(), { 0, rawRenderIndexData.size() }, sizeof(uint32_t), RenderBufferFlag::VERTEX, { }, &rawRenderIndexBuffer });
        }

#   if RT_ENABLED
        // FIXME: Add support for multiple raytracing scenes.
        Framebuffer *chosenFramebuffer = nullptr;
        RaytracingScene *chosenRtScene = nullptr;
        if (rtEnabled) {
            rtResources->updateBottomLevelASResources(worker);

            for (uint32_t i = 0; i < framebufferCount; i++) {
                RenderTargetDrawCall &targetDrawCall = framebufferVector[i].renderTargetDrawCall;
                if (!targetDrawCall.rtScenes.empty()) {
                    chosenFramebuffer = &framebufferVector[i];
                    chosenRtScene = &targetDrawCall.rtScenes[0];
                }
            }
        }

        if (chosenRtScene != nullptr) {
            if (rtResources->updateOutputBuffers) {
                rtResources->createOutputBuffers(worker, chosenRtScene->screenWidth, chosenRtScene->screenHeight);
                rtResources->updateOutputBuffers = false;
            }

            const RenderTarget *framebufferTarget = chosenFramebuffer->renderTargetDrawCall.fbStorage->colorTarget;
            interleavedRastersCount = static_cast<uint32_t>(chosenRtScene->interleavedRasters.size());
            rtResources->updateInterleavedRenderTargets(worker, chosenRtScene->screenWidth, chosenRtScene->screenHeight, interleavedRastersCount, framebufferTarget->multisampling, framebufferTarget->usesHDR);

            for (uint32_t i = 0; i < interleavedRastersCount; i++) {
                auto &intRaster = chosenRtScene->interleavedRasters[i];
                RenderTarget *colorTarget = rtResources->interleavedColorTargetVector[i].get();
                RenderTarget *depthTarget = rtResources->interleavedDepthTargetVector[i].get();
                intRaster.colorTextureIndex = getTextureIndex(colorTarget);
                intRaster.depthTextureIndex = getTextureIndex(depthTarget);
                chosenFramebuffer->transitionRenderTargetSet.emplace(colorTarget);
                chosenFramebuffer->transitionRenderTargetSet.emplace(depthTarget);
            }

            // Must have at least one element in the vector.
            if (chosenRtScene->interleavedRasters.empty()) {
                chosenRtScene->interleavedRasters.emplace_back();
            }

            shaderUploads.push_back({ &rtResources->rtParams, { 0, 1 }, sizeof(interop::RaytracingParams), RenderBufferFlag::CONSTANT, { }, &rtResources->rtParamsBuffer });
            shaderUploads.push_back({ chosenRtScene->interleavedRasters.data(), { 0, chosenRtScene->interleavedRasters.size() }, sizeof(interop::InterleavedRaster), RenderBufferFlag::STORAGE, { }, &interleavedRastersBuffer });

            updateRaytracingScene(worker, *chosenRtScene);
            shaderViewRtEnabled = true;
        }
#   endif

        shaderUploader->submit(worker, shaderUploads);

        // Slot-3 (per-vertex renderIndex) views, now that the buffers exist.
        if (!renderIndexData.empty()) {
            indexedVertexViews[3] = RenderVertexBufferView(RenderBufferReference(renderIndexBuffer.get()), uint32_t(sizeof(uint32_t) * renderIndexData.size()));
        }
        if (!rawRenderIndexData.empty()) {
            rawVertexViews[3] = RenderVertexBufferView(RenderBufferReference(rawRenderIndexBuffer.get()), uint32_t(sizeof(uint32_t) * rawRenderIndexData.size()));
        }

        updateShaderViews(worker, drawBuffers, outputBuffers, shaderViewRtEnabled);
    }

    void FramebufferRenderer::advanceFrame(bool rtEnabled) {
        frameParams.frameCount++;

#   if RT_ENABLED
        if (rtEnabled) {
            rtResources->swapBuffers = !rtResources->swapBuffers;
            rtResources->skipReprojection = false;
        }
#   endif
    }
};

/*
void RT64::View::renderIm3d() {
    if (Im3d::GetDrawListCount() > 0) {
        commandList->SetGraphicsRootSignature(worker->device->getIm3dRootSignature());

        commandList->SetDescriptorHeaps(1, &descriptorHeap);
        commandList->SetGraphicsRootDescriptorTable(0, descriptorHeap->GetGPUDescriptorHandleForHeapStart());
        commandList->RSSetViewports(1, &rtViewport);
        commandList->RSSetScissorRects(1, &rtScissor);

        unsigned int totalVertexCount = 0;
        for (Im3d::U32 i = 0, n = Im3d::GetDrawListCount(); i < n; ++i) {
            auto &drawList = Im3d::GetDrawLists()[i];
            totalVertexCount += drawList.m_vertexCount;
        }

        if (totalVertexCount > 0) {
            // Release the previous vertex buffer if it should be bigger.
            if (!im3dVertexBuffer.IsNull() && (totalVertexCount > im3dVertexCount)) {
                im3dVertexBuffer.Release();
            }

            // Create the vertex buffer if it's empty.
            const UINT vertexBufferSize = totalVertexCount * sizeof(Im3d::VertexData);
            if (im3dVertexBuffer.IsNull()) {
                CD3DX12_RESOURCE_DESC uploadBufferDesc = CD3DX12_RESOURCE_DESC::Buffer(vertexBufferSize);
                im3dVertexBuffer = worker->device->allocateResource(D3D12_HEAP_TYPE_UPLOAD, &uploadBufferDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr);
                im3dVertexCount = totalVertexCount;
                im3dVertexBufferView.BufferLocation = im3dVertexBuffer.Get()->GetGPUVirtualAddress();
                im3dVertexBufferView.StrideInBytes = sizeof(Im3d::VertexData);
                im3dVertexBufferView.SizeInBytes = vertexBufferSize;
            }

            // Copy data to vertex buffer.
            UINT8 *pDataBegin;
            CD3DX12_RANGE readRange(0, 0);
            D3D12_CHECK(im3dVertexBuffer.Get()->Map(0, &readRange, reinterpret_cast<void **>(&pDataBegin)));
            for (Im3d::U32 i = 0, n = Im3d::GetDrawListCount(); i < n; ++i) {
                auto &drawList = Im3d::GetDrawLists()[i];
                size_t copySize = sizeof(Im3d::VertexData) * drawList.m_vertexCount;
                memcpy(pDataBegin, drawList.m_vertexData, copySize);
                pDataBegin += copySize;
            }

            im3dVertexBuffer.Get()->Unmap(0, nullptr);

            unsigned int vertexOffset = 0;
            for (Im3d::U32 i = 0, n = Im3d::GetDrawListCount(); i < n; ++i) {
                auto &drawList = Im3d::GetDrawLists()[i];
                commandList->IASetVertexBuffers(0, 1, &im3dVertexBufferView);
                switch (drawList.m_primType) {
                case Im3d::DrawPrimitive_Points:
                    commandList->SetPipelineState(worker->device->getIm3dPipelineStatePoint());
                    commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_POINTLIST);
                    break;
                case Im3d::DrawPrimitive_Lines:
                    commandList->SetPipelineState(worker->device->getIm3dPipelineStateLine());
                    commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_LINELIST);
                    break;
                case Im3d::DrawPrimitive_Triangles:
                    commandList->SetPipelineState(worker->device->getIm3dPipelineStateTriangle());
                    commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                    break;
                default:
                    break;
                }

                commandList->DrawInstanced(drawList.m_vertexCount, 1, vertexOffset, 0);
                vertexOffset += drawList.m_vertexCount;
            }
        }
    }
}
*/

/*
namespace RT64 {
    class View {
    private:
        // Im3D
        AllocatedResource im3dVertexBuffer;
        D3D12_VERTEX_BUFFER_VIEW im3dVertexBufferView;
        unsigned int im3dVertexCount;
    };
};
*/