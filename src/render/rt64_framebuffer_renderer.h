//
// RT64
//

#pragma once

#include <stdint.h>

#include "common/rt64_emulator_configuration.h"
#include "common/rt64_user_configuration.h"
#include "hle/rt64_framebuffer_manager.h"
#include "hle/rt64_workload.h"
#include "preset/rt64_preset_scene.h"
#include "shared/rt64_frame_params.h"
#include "shared/rt64_gpu_tile.h"
#include "shared/rt64_render_indices.h"
#include "shared/rt64_render_params.h"

#include "rt64_buffer_uploader.h"
#include "rt64_descriptor_sets.h"
#include "rt64_framebuffer_renderer_call.h"
#include "rt64_raster_shader_cache.h"
#include "rt64_render_target.h"
#include "rt64_rsp_processor.h"
#include "rt64_vertex_processor.h"

#if RT_ENABLED
#   include "rt64_raytracing_resources.h"
#   include "rt64_raytracing_shader_cache.h"
#endif

namespace RT64 {
    struct DynamicTextureView {
        const RenderTexture *texture = nullptr;
        uint32_t dstIndex = 0;
        const RenderTextureView *textureView = nullptr;
    };

    struct RasterScene {
        std::vector<uint32_t> instanceIndices;

        RasterScene();
    };

    struct RenderTargetDrawCall {
        typedef std::pair<uint32_t, bool> SceneIndexPair;

        RenderFramebufferStorage *fbStorage = nullptr;
        std::vector<RasterScene> rasterScenes;
        std::vector<SceneIndexPair> sceneIndices;
        int32_t rs64LightsAfterScene = -1;
#   if RT_ENABLED
        std::vector<RaytracingScene> rtScenes;
#   endif
    };

    struct RSPSmoothNormalGenerationCB {
        uint32_t indexStart;
        uint32_t indexCount;
    };

    struct FramebufferRenderer {
        std::vector<uint32_t> textureCacheVersions;
        std::vector<Texture *> textureCacheTextures;
        std::vector<Texture *> textureCacheTextureReplacements;
        std::vector<uint32_t> textureCacheFreeSpaces;
        uint32_t textureCacheSize = 0;
        uint32_t textureCacheGlobalVersion = 0;
        bool textureCacheReplacementMapEnabled = false;
        std::vector<InstanceDrawCall> instanceDrawCallVector;
        std::vector<RenderPipelineProgram> hitGroupVector;
        std::vector<interop::RenderIndices> renderIndicesVector;
        // Per-vertex renderIndex (draw-coalescing): one entry per global vertex,
        // = the renderIndex of the object owning that vertex. Uploaded as vertex
        // stream 3 so a merged draw can span multiple objects. See s_ri_collisions.
        std::vector<uint32_t> renderIndexData;      // indexed-triangle vertices
        std::vector<uint32_t> rawRenderIndexData;   // raw/rect vertices
        std::vector<DynamicTextureView> dynamicTextureViewVector;
        std::vector<RenderTextureBarrier> dynamicTextureBarrierVector;
        std::unique_ptr<BufferUploader> shaderUploader;
        std::vector<RSPSmoothNormalGenerationCB> rspSmoothNormalVector;
        std::array<RenderInputSlot, 4> vertexInputSlots;
        std::array<RenderVertexBufferView, 4> indexedVertexViews;
        std::array<RenderVertexBufferView, 4> rawVertexViews;
        RenderIndexBufferView indexBufferView;
        RenderBuffer *testZIndexBuffer = nullptr;
        RenderIndexBufferView testZIndexBufferView;
        BufferPair renderIndicesBuffer;
        BufferPair renderIndexBuffer;       // upload of renderIndexData (vertex stream 3, indexed)
        BufferPair rawRenderIndexBuffer;    // upload of rawRenderIndexData (vertex stream 3, raw)
        BufferPair interleavedRastersBuffer;
        uint32_t interleavedRastersCount = 0;
        BufferPair frameParamsBuffer;
        RenderPipelineLayout *rendererPipelineLayout = nullptr;
        RenderPipeline *postBlendDitherNoiseAddPipeline = nullptr;
        RenderPipeline *postBlendDitherNoiseSubPipeline = nullptr;
        RenderPipeline *postBlendDitherNoiseSubNegativePipeline = nullptr;
        std::unique_ptr<FramebufferRendererDescriptorCommonSet> descCommonSet;
        std::unique_ptr<FramebufferRendererDescriptorTextureSet> descTextureSet;
        std::unique_ptr<RenderTexture> dummyColorTarget;
        std::unique_ptr<RenderTexture> dummyDepthTarget;
        std::unique_ptr<RenderTextureView> dummyColorTargetView;
        std::unique_ptr<RenderTextureView> dummyDepthTargetView;
        bool dummyColorTargetTransitioned = false;
        bool dummyDepthTargetTransitioned = false;
        std::vector<uint32_t> descriptorTextureVersions;
        uint32_t descriptorTextureGlobalVersion = 0;
        bool descriptorTextureReplacementMapEnabled = false;
        std::unique_ptr<RSPSmoothNormalDescriptorSet> smoothDescSet;
        std::unique_ptr<RSPVertexTestZDescriptorSet> vertexTestZSet;
        interop::FrameParams frameParams;
        const ShaderLibrary *shaderLibrary = nullptr;

#   if RT_ENABLED
        const RenderTexture *blueNoiseTexture = nullptr;
        const RaytracingState *rtState = nullptr;
        const RenderPipelineLayout *rtPipelineLayout = nullptr;
        std::unique_ptr<RaytracingResources> rtResources;
        bool rtSupport = false;
#   endif

        struct Framebuffer {
            std::unique_ptr<RenderBuffer> paramsBuffer;
            std::unique_ptr<FramebufferRendererDescriptorFramebufferSet> descRealFbSet;
            std::unique_ptr<FramebufferRendererDescriptorFramebufferSet> descDummyFbSet;
            std::set<RenderTarget *> transitionRenderTargetSet;
            RenderTargetDrawCall renderTargetDrawCall;
            RenderViewport viewport;
            std::unique_ptr<RenderBuffer> rs64LightsBuffer;
            std::unique_ptr<RS64LightsDescriptorSet> rs64LightsSet;
            bool rs64LightsDebug = false;
            std::unique_ptr<RenderBuffer> rs64ShadowsBuffer;
            std::unique_ptr<RS64TracedDescriptorSet> rs64ShadowsSet;
            bool rs64ShadowsDebug = false;
            bool rs64LightsOn = false;
            bool rs64ShadowsOn = false;
            std::unique_ptr<RS64TracedDescriptorSet> rs64LightsShadowedSet;
            bool rs64LightsShadowed = false;
            const void *rs64Target = nullptr;
            std::unique_ptr<RenderBuffer> rs64FogBuffer;
            std::unique_ptr<RS64TracedDescriptorSet> rs64FogSet;
            bool rs64FogDebug = false;
            bool rs64FogOn = false;
            std::unique_ptr<RenderTexture> rs64SoftMask;
            std::unique_ptr<RenderTexture> rs64SoftPing;
            std::unique_ptr<RenderTexture> rs64SoftIndirect;
            std::unique_ptr<RenderTexture> rs64SoftIndirectPing;
            std::unique_ptr<RenderFramebuffer> rs64SoftMaskFb;
            std::unique_ptr<RenderFramebuffer> rs64SoftPingFb;
            uint32_t rs64SoftW = 0;
            uint32_t rs64SoftH = 0;
            std::unique_ptr<RS64ShadowBlurDescriptorSet> rs64BlurHSet;
            std::unique_ptr<RS64ShadowBlurDescriptorSet> rs64BlurVSet;
            std::unique_ptr<RenderBuffer> rs64BlurHBuffer;
            std::unique_ptr<RenderBuffer> rs64BlurVBuffer;
            bool rs64ShadowsSoft = false;
        };

        // Per displayed frame: the main view's opaque casters as one mesh over worldPosBuffer (plume's Vulkan BLAS build handles one mesh only).
        struct RS64ShadowScene {
            std::vector<uint32_t> indices;
            std::vector<rs64lights::IndexRange> ranges;
            // Alpha-tested casters: a non-opaque BLAS (TLAS mask 0x04) plus the triangle -> draw table the traced passes sample through.
            std::vector<rs64lights::CutoutRange> cutoutRanges;
            std::vector<uint32_t> cutoutIndices;
            std::vector<rs64lights::CutoutEntry> cutoutTable;
            uint32_t cutoutDrawCount = 0;
            std::unique_ptr<RenderBuffer> cutoutIndexBuffer, cutoutTableBuffer, cutoutBlasBuffer, cutoutBlasScratch;
            std::unique_ptr<RenderAccelerationStructure> cutoutBlas;
            uint64_t cutoutIndexCapacity = 0, cutoutTableCapacity = 0, cutoutBlasCapacity = 0, cutoutBlasScratchCapacity = 0;
            // GPU-local copies the traced passes read per candidate hit (the upload copies stay for the BLAS build).
            std::unique_ptr<RenderBuffer> cutoutIndexGpu, cutoutTableGpu;
            uint64_t cutoutIndexGpuCapacity = 0, cutoutTableGpuCapacity = 0, cutoutIndexBytes = 0, cutoutTableBytes = 0;
            const RenderBuffer *texCoords = nullptr;
            uint64_t texCoordsSize = 0;
            const RenderBuffer *worldPos = nullptr;
            uint32_t vertexCount = 0;
            std::unique_ptr<RenderBuffer> indexBuffer, blasBuffer, tlasBuffer, blasScratch, tlasScratch, instancesBuffer;
            uint64_t indexCapacity = 0, blasCapacity = 0, tlasCapacity = 0, blasScratchCapacity = 0, tlasScratchCapacity = 0, instancesCapacity = 0;
            std::unique_ptr<RenderAccelerationStructure> blas, tlas;
            bool ready = false;
            // Static HMP terrain: one BLAS per map version in map space, instanced each frame through the terrain draw's transform.
            std::shared_ptr<const rs64lights::TerrainMap> terrainMap;
            uint64_t terrainBuiltVersion = 0;
            std::unique_ptr<RenderBuffer> terrainPosBuffer, terrainIndexBuffer, terrainBlasBuffer, terrainScratch;
            std::unique_ptr<RenderBuffer> terrainNormalBuffer, terrainIndexGpu, terrainNormalGpu;
            uint64_t terrainIndexBytes = 0, terrainNormalBytes = 0;
            std::unique_ptr<RenderAccelerationStructure> terrainBlas;
            float terrainTransform[3][4] = {};
            bool terrainActive = false;
            uint64_t terrainOriginVersion = 0;
            int32_t terrainOriginX = 0, terrainOriginZ = 0;
            // Hit colours for bounce light and reflections: per opaque caster draw (BLAS triangle order) and per emissive draw (instance 7, mask 0x80, never a shadow caster).
            std::vector<uint32_t> rangeCalls;
            std::vector<rs64lights::IndexRange> emissiveRanges;
            std::vector<uint32_t> emissiveCalls;
            std::vector<uint32_t> emissiveIndices;
            std::vector<rs64lights::HitColorEntry> hitTable, emissiveTable;
            uint32_t terrainColor = 0;
            uint32_t averageColor = 0;
            std::unique_ptr<RenderBuffer> hitTableBuffer, hitTableGpu, emissiveTableBuffer, emissiveTableGpu;
            uint64_t hitTableCapacity = 0, hitTableGpuCapacity = 0, emissiveTableCapacity = 0, emissiveTableGpuCapacity = 0, hitTableBytes = 0, emissiveTableBytes = 0;
            std::unique_ptr<RenderBuffer> emissiveIndexBuffer, emissiveBlasBuffer, emissiveBlasScratch;
            uint64_t emissiveIndexCapacity = 0, emissiveBlasCapacity = 0, emissiveBlasScratchCapacity = 0;
            std::unique_ptr<RenderAccelerationStructure> emissiveBlas;
            uint32_t hitCount = 0, emissiveCount = 0;
            // Craft draws (instance 8, mask 0x01): their own BLAS, so history snapshots of the opaque BLAS never hold a ship.
            std::vector<rs64lights::IndexRange> craftRanges;
            std::vector<uint32_t> craftIndices;
            std::unique_ptr<RenderBuffer> craftIndexBuffer, craftBlasBuffer, craftBlasScratch;
            uint64_t craftIndexCapacity = 0, craftBlasCapacity = 0, craftBlasScratchCapacity = 0;
            std::unique_ptr<RenderAccelerationStructure> craftBlas;
            // Off-screen casters: past frames' opaque BLASes, placed through the terrain transform (camera_k -> map -> camera_now).
            struct History {
                std::unique_ptr<RenderBuffer> blasBuffer;
                std::unique_ptr<RenderAccelerationStructure> blas;
                float terrainTransform[3][4] = {};
                uint64_t mapVersion = 0;
            };
            History history[4];
            // An evicted slot is still in the TLAS being recorded; freed on the next scene build.
            History retired;
            uint32_t historyNext = 0, historyFrame = 0;
            uint32_t historyMask = 0;
            float historyCam[4][4] = {};
        };

        RS64ShadowScene rs64Shadow;
        rs64lights::FogCache rs64FogCache;
        std::unique_ptr<RenderBuffer> rs64CutoutPlaceholder;

        std::vector<Framebuffer> framebufferVector;
        std::vector<BufferUploader *> pendingUploaders;
        uint32_t framebufferCount = 0;

        struct DrawParams {
            RenderWorker *worker;
            RenderFramebufferStorage *fbStorage;
            const Workload *curWorkload;
            uint32_t fbPairIndex;
            uint32_t fbWidth;
            uint32_t fbHeight;
            uint32_t targetWidth;
            uint32_t targetHeight;
            RasterShaderCache *rasterShaderCache;
            hlslpp::float2 resolutionScale;
            float aspectRatioSource;
            float pixelAspect = 1.0f;
            float aspectRatioTarget;
            float extAspectPercentage;
            float horizontalMisalignment;
            PresetScene presetScene;
            bool rtEnabled;
            uint64_t submissionFrame;
            float deltaTimeMs;
            bool ubershadersOnly;
            bool postBlendNoise;
            bool postBlendNoiseNegative;
            uint32_t maxGameCall;
            bool rs64Lights = false;
        };

        FramebufferRenderer(RenderWorker *worker, bool rtSupport, UserConfiguration::GraphicsAPI graphicsAPI, const ShaderLibrary *shaderLibrary);
        ~FramebufferRenderer();
        void resetFramebuffers(RenderWorker *worker, bool ubershadersVisible, float ditherNoiseStrength, const RenderMultisampling &multisampling);
        void updateTextureCache(TextureCache *textureCache);
        void createGPUTiles(const DrawCallTile *callTiles, uint32_t callTileCount, interop::GPUTile *dstGPUTiles, const FramebufferManager *fbManager, TextureCache *textureCache, uint64_t submissionFrame);
        uint32_t getDestinationIndex();
        uint32_t getTextureIndex(RenderTarget *renderTarget);
        uint32_t getTextureIndex(const FramebufferManager::TileCopy &tileCopy);
        void updateMultisampling();
        void updateShaderDescriptorSet(RenderWorker *worker, const DrawBuffers *drawBuffers, const OutputBuffers *outputBuffers, bool raytracingEnabled);
        void updateRSPSmoothNormalSet(RenderWorker *worker, const DrawBuffers *drawBuffers, const OutputBuffers *outputBuffers);
        void updateRSPVertexTestZSet(RenderWorker *worker, const DrawBuffers *drawBuffers, const OutputBuffers *outputBuffers);
        void updateShaderViews(RenderWorker *worker, const DrawBuffers *drawBuffers, const OutputBuffers *outputBuffers, bool raytracingEnabled);
        void submitRSPSmoothNormalCompute(RenderWorker *worker, const OutputBuffers *outputBuffers);
        bool submitDepthAccess(RenderWorker *worker, RenderFramebufferStorage *fbStorage, bool readOnly, bool &depthState);
        void submitRasterScene(RenderWorker *worker, const Framebuffer &framebuffer, RenderFramebufferStorage *fbStorage, const RasterScene &rasterScene, bool &depthState);
        void addFramebuffer(const DrawParams &p);
        bool fillRS64Lights(const DrawParams &p, Framebuffer &framebuffer, uint32_t transformsIndex, hlslpp::float2 screenScale, hlslpp::float2 screenOffset);
        bool rs64ViewFromScreen(const DrawParams &p, uint32_t transformsIndex, hlslpp::float2 screenScale, hlslpp::float2 screenOffset, rs64lights::Mat4 &out);
        bool fillRS64Shadows(const DrawParams &p, Framebuffer &framebuffer, uint32_t transformsIndex, hlslpp::float2 screenScale, hlslpp::float2 screenOffset, const rs64lights::Sun &sun);
        void recordRS64Shadows(RenderWorker *worker, const Framebuffer &framebuffer, bool &depthState);
        void ensureRS64SoftTargets(RenderWorker *worker, Framebuffer &framebuffer, uint32_t width, uint32_t height);
        void buildRS64HitColors(const DrawParams &p, const std::vector<std::pair<rs64lights::IndexRange, uint32_t>> &terrainCalls, const float *floorUp);
        bool rs64ViewProj(const DrawParams &p, uint32_t transformsIndex, rs64lights::Mat4 &out);
        bool fillRS64FogShafts(const DrawParams &p, Framebuffer &framebuffer, uint32_t transformsIndex, hlslpp::float2 screenScale, hlslpp::float2 screenOffset, const rs64lights::Sun &sun, const rs64lights::FogParams &fog);
        void recordRS64FogShafts(RenderWorker *worker, const Framebuffer &framebuffer, bool &depthState);
        void bindRS64Traced(RenderWorker *worker, RS64TracedDescriptorSet &set, RenderBuffer *paramsBuffer, uint64_t countOffset, bool countAsFloat);
        void recordRS64ShadowScene(RenderWorker *worker, bool dynamic);
        void recordRS64TerrainBLAS(RenderWorker *worker);
        void endFramebuffers(RenderWorker *worker, const DrawBuffers *drawBuffers, const OutputBuffers *outputBuffers, bool rtEnabled);
        void recordSetup(RenderWorker *worker, std::vector<BufferUploader *> bufferUploaders, RSPProcessor *rspProcessor, VertexProcessor *vertexProcessor, const OutputBuffers *outputBuffers, bool rtEnabled);
        void recordFramebuffer(RenderWorker *worker, uint32_t framebufferIndex);
        void recordRS64Lights(RenderWorker *worker, const Framebuffer &framebuffer, bool &depthState);
        void waitForUploaders();

        // GPU time of the RS64 passes (ROGUESQ_LOG_RT_TIMING=1): begin/end timestamp pairs per pass, averaged and logged every 120 workloads.
        enum RS64TimingPass : uint8_t { RS64TimeScene, RS64TimeShadows, RS64TimeFog, RS64TimeLights, RS64TimeCount };
        std::unique_ptr<RenderQueryPool> rs64TimingPool;
        uint32_t rs64TimingMarks = 0;
        uint8_t rs64TimingLabels[64] = {};
        double rs64TimingSum[RS64TimeCount] = {};
        uint32_t rs64TimingFrames = 0;
        void rs64TimingBegin(RenderWorker *worker);
        void rs64TimingMark(RenderWorker *worker, uint8_t pass);
        void rs64TimingReport();
        void advanceFrame(bool rtEnabled);

#   if RT_ENABLED
        void resetRaytracing(RaytracingShaderCache *rtShaderCache, const RenderTexture *blueNoiseTexture);
        void updateRaytracingScene(RenderWorker *worker, const RaytracingScene &rtScene);
        void submitRaytracingScene(RenderWorker *worker, RenderTarget *colorTarget, const RaytracingScene &rtScene);
        void setRaytracingConfig(const RaytracingConfiguration &rtConfig, bool resolutionChanged);
#   endif
    };
};