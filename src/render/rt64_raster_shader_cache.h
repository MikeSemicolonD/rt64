//
// RT64
//

#pragma once

#include <atomic>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include "rt64_raster_shader.h"

namespace RT64 {
    // Compiles every combination listed in comboPath into the D3D11 shader cache directory and copies the list there.
    // Returns the number of combinations that failed, or -1 if D3D11 is unavailable.
    int PrecompileD3D11Shaders(const std::string &comboPath, const std::string &cacheDirectory);

    struct RasterShaderCache {
        struct CompilationThread {
            RasterShaderCache *shaderCache;
            std::unique_ptr<std::thread> thread;
            std::atomic<bool> threadRunning;
            // One DXC compiler per thread: IDxcCompiler/IDxcUtils are not safe to use from several threads at once.
            std::unique_ptr<ShaderCompiler> shaderCompiler;

            CompilationThread(RasterShaderCache *shaderCache);
            ~CompilationThread();
            void loop();
        };

        RenderDevice *device;
        std::unique_ptr<RasterShaderUber> shaderUber;
        OptimizerCacheSPIRV optimizerCacheSPIRV;
        std::mutex submissionMutex;
        std::queue<ShaderDescription> descQueue;
        // Startup warm-up: one thread at idle priority, after every draw-requested shader; a draw that needs a queued one promotes it.
        std::queue<ShaderDescription> warmupQueue;
        std::unordered_set<uint64_t> warmupPending;
        bool warmupBusy = false;
        std::mutex descQueueMutex;
        int32_t descQueueActiveCount = 0;
        std::condition_variable descQueueChanged;
        std::unordered_map<uint64_t, bool> shaderHashes;
        std::unordered_map<uint64_t, std::unique_ptr<RasterShader>> GPUShaders;        std::mutex GPUShadersMutex;
        std::list<std::unique_ptr<CompilationThread>> compilationThreads;
        uint32_t threadCount;
        uint32_t ubershaderThreadCount;
        RenderShaderFormat shaderFormat;
        RenderMultisampling multisampling;
        bool usesHDR = false;
        std::atomic<bool> noUbershader = false;
        std::atomic<uint32_t> compiledCount = 0;
        FILE *comboFile = nullptr;
        
        RasterShaderCache(uint32_t threadCount, uint32_t ubershaderThreadCount);
        ~RasterShaderCache();
        // Without an ubershader (texture slots, or skipUbershader for GPUs too slow to compile it), draws wait for their specialized
        // pipeline and comboList (a shipped list of shader combinations) is compiled at startup.
        void setup(RenderDevice *device, RenderShaderFormat shaderFormat, const ShaderLibrary *shaderLibrary, const RenderMultisampling &multisampling, bool textureSlots, bool skipUbershader, const std::string &comboList);
        void submit(const ShaderDescription &desc, bool warmup = false);
        void waitForAll();
        void destroyAll();
        RasterShader *getGPUShader(const ShaderDescription &desc);
        RasterShaderUber *getGPUShaderUber() const;
        uint32_t shaderCount();
        void getCompileStatus(uint32_t &pending, uint32_t &compiled);
    };
};