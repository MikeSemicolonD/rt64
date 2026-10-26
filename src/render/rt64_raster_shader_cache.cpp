//
// RT64
//

#include "rt64_raster_shader_cache.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <filesystem>

#include "common/rt64_thread.h"

#ifdef PLUME_D3D11_ENABLED
#   include "plume_d3d11.h"

namespace plume {
    extern std::unique_ptr<RenderInterface> CreateD3D11Interface();
};
#endif

#define ENABLE_OPTIMIZED_SHADER_GENERATION

namespace RT64 {
    // RasterShaderCache::CompilationThread
    
    RasterShaderCache::CompilationThread::CompilationThread(RasterShaderCache *shaderCache) {
        assert(shaderCache != nullptr);

        this->shaderCache = shaderCache;

        thread = std::make_unique<std::thread>(&CompilationThread::loop, this);
        threadRunning = false;
    }

    RasterShaderCache::CompilationThread::~CompilationThread() {
        threadRunning = false;
        shaderCache->descQueueChanged.notify_all();
        thread->join();
        thread.reset(nullptr);
    }

    void RasterShaderCache::CompilationThread::loop() {
        Thread::setCurrentThreadName("RT64 Shader");

        // The shader compilation thread should have idle priority by default as the application can use the ubershader in the meantime.
        Thread::setCurrentThreadPriority(Thread::Priority::Idle);

#if defined(ENABLE_OPTIMIZED_SHADER_GENERATION) && defined(_WIN32)
        shaderCompiler = std::make_unique<ShaderCompiler>();
#endif

        threadRunning = true;
        bool priorityRaised = false;

        while (threadRunning) {
            ShaderDescription shaderDesc;
            bool fromPriorityQueue = false;
            bool fromWarmup = false;

            // Check the top of the queue or wait if it's empty.
            {
                std::unique_lock<std::mutex> queueLock(shaderCache->descQueueMutex);
                shaderCache->descQueueActiveCount--;
                shaderCache->descQueueChanged.wait(queueLock, [this]() {
                    return !threadRunning || !shaderCache->descQueue.empty() || (!shaderCache->warmupQueue.empty() && !shaderCache->warmupBusy);
                });

                shaderCache->descQueueActiveCount++;
                if (!shaderCache->descQueue.empty()) {
                    shaderDesc = shaderCache->descQueue.front();
                    shaderCache->descQueue.pop();
                    fromPriorityQueue = true;
                }
                else if (!shaderCache->warmupQueue.empty() && !shaderCache->warmupBusy) {
                    shaderDesc = shaderCache->warmupQueue.front();
                    shaderCache->warmupQueue.pop();
                    shaderCache->warmupBusy = true;
                    fromWarmup = true;
                }
            }

            // A draw that needed this shader already moved it to descQueue.
            if (fromWarmup) {
                const std::unique_lock<std::mutex> lock(shaderCache->submissionMutex);
                fromPriorityQueue = (shaderCache->warmupPending.erase(shaderDesc.hash()) > 0);
            }

            // Compile the shader at the top of the queue.
            if (fromPriorityQueue) {
                // Without an ubershader the draw is missing until its shader is ready, so idle priority would starve it; warm-up stays idle.
                const bool raise = shaderCache->noUbershader && !fromWarmup;
                if (raise != priorityRaised) {
                    Thread::setCurrentThreadPriority(raise ? Thread::Priority::Normal : Thread::Priority::Idle);
                    priorityRaised = raise;
                }

                assert((shaderCache->shaderUber != nullptr) && "Ubershader should've been created by the time a new shader is submitted to the cache.");
                const RenderPipelineLayout *uberPipelineLayout = shaderCache->shaderUber->pipelineLayout.get();
                const RenderMultisampling multisampling = shaderCache->multisampling;
                std::unique_ptr<RasterShader> newShader = std::make_unique<RasterShader>(shaderCache->device, shaderDesc, uberPipelineLayout, shaderCache->shaderFormat, multisampling, shaderCompiler.get(), &shaderCache->optimizerCacheSPIRV);

                {
                    const std::unique_lock<std::mutex> lock(shaderCache->GPUShadersMutex);
                    shaderCache->GPUShaders[shaderDesc.hash()] = std::move(newShader);
                }

                shaderCache->compiledCount++;

                bool idle;
                {
                    const std::unique_lock<std::mutex> queueLock(shaderCache->descQueueMutex);
                    idle = shaderCache->descQueue.empty() && shaderCache->warmupQueue.empty();
                }

                if (idle) {
                    shaderCache->device->savePipelineCache();
                }
            }

            if (fromWarmup) {
                {
                    const std::unique_lock<std::mutex> queueLock(shaderCache->descQueueMutex);
                    shaderCache->warmupBusy = false;
                }

                shaderCache->descQueueChanged.notify_all();
            }
        }
    }

    // Without compute shaders (feature level 10_x) textured draws sample raw TMEM, so each listed combination also needs its canDecodeTMEM variant.
    static bool rawTMEMVariant(const ShaderDescription &desc, ShaderDescription &variant) {
        if ((!desc.flags.usesTexture0 && !desc.flags.usesTexture1) || desc.flags.canDecodeTMEM) {
            return false;
        }

        variant = desc;
        variant.flags.canDecodeTMEM = 1;
        return true;
    }

    // PrecompileD3D11Shaders

    int PrecompileD3D11Shaders(const std::string &comboPath, const std::string &cacheDirectory) {
#   ifdef PLUME_D3D11_ENABLED
        std::vector<ShaderDescription> descs;
        if (FILE *file = std::fopen(comboPath.c_str(), "r")) {
            ShaderDescription desc = {};
            while (std::fscanf(file, "%X %X %X %X %X", &desc.otherMode.L, &desc.otherMode.H, &desc.colorCombiner.L, &desc.colorCombiner.H, &desc.flags.value) == 5) {
                descs.push_back(desc);
            }

            std::fclose(file);
        }

        if (descs.empty()) {
            fprintf(stderr, "[d3d11-precompile] No combinations read from %s.\n", comboPath.c_str());
            return -1;
        }

        std::unique_ptr<RenderInterface> renderInterface = CreateD3D11Interface();
        std::unique_ptr<RenderDevice> device = (renderInterface != nullptr) ? renderInterface->createDevice() : nullptr;
        if (device == nullptr) {
            fprintf(stderr, "[d3d11-precompile] Unable to create a D3D11 device.\n");
            return -1;
        }

        if (!device->getCapabilities().computeShaders) {
            const size_t listedCount = descs.size();
            for (size_t i = 0; i < listedCount; i++) {
                ShaderDescription variant;
                if (rawTMEMVariant(descs[i], variant)) {
                    descs.push_back(variant);
                }
            }
        }

        std::filesystem::create_directories(cacheDirectory);
        static_cast<D3D11Device *>(device.get())->shaderCacheDirectory = cacheDirectory;

        // RT64's fixed pipelines, as created at startup on D3D11 (no HDR, no MSAA).
        ShaderLibrary shaderLibrary(false, true);
        shaderLibrary.textureSlots = true;
        shaderLibrary.setupCommonShaders(renderInterface.get(), device.get());
        shaderLibrary.setupMultisamplingShaders(renderInterface.get(), device.get(), RenderMultisampling());
        RasterShaderUber rasterShaderUber(device.get(), RenderShaderFormat::SPIRV, RenderMultisampling(), &shaderLibrary, 1, true);

        // The cache key covers the shader code, spec constants and register base only, so an empty layout gives the same entries the game uses.
        std::unique_ptr<RenderPipelineLayout> pipelineLayout = device->createPipelineLayout(RenderPipelineLayoutDesc());
        OptimizerCacheSPIRV optimizerCache;
        optimizerCache.initialize(true);

        std::atomic<uint32_t> nextIndex = 0;
        std::atomic<uint32_t> doneCount = 0;
        std::atomic<int> failedCount = 0;
        auto worker = [&]() {
            for (uint32_t i = nextIndex++; i < descs.size(); i = nextIndex++) {
                RasterShader shader(device.get(), descs[i], pipelineLayout.get(), RenderShaderFormat::SPIRV, RenderMultisampling(), nullptr, &optimizerCache);
                if (shader.pipeline == nullptr) {
                    failedCount++;
                }

                fprintf(stderr, "[d3d11-precompile] %u / %zu\n", ++doneCount, descs.size());
            }
        };

        std::vector<std::thread> threads;
        for (uint32_t t = 0; t < std::max(std::thread::hardware_concurrency(), 1U); t++) {
            threads.emplace_back(worker);
        }

        for (std::thread &thread : threads) {
            thread.join();
        }

        const std::filesystem::path listCopy = std::filesystem::path(cacheDirectory) / "rt64_combos.txt";
        std::error_code ec;
        if (!std::filesystem::equivalent(comboPath, listCopy, ec)) {
            std::filesystem::copy_file(comboPath, listCopy, std::filesystem::copy_options::overwrite_existing, ec);
        }

        fprintf(stderr, "[d3d11-precompile] %zu combinations, %d failed, cache in %s\n", descs.size(), failedCount.load(), cacheDirectory.c_str());
        return failedCount;
#   else
        fprintf(stderr, "[d3d11-precompile] This build has no D3D11 backend.\n");
        return -1;
#   endif
    }

    // RasterShaderCache

    RasterShaderCache::RasterShaderCache(uint32_t threadCount, uint32_t ubershaderThreadCount) {
        assert(threadCount > 0);

        this->threadCount = threadCount;
        this->ubershaderThreadCount = ubershaderThreadCount;

#ifdef ENABLE_OPTIMIZED_SHADER_GENERATION
        descQueueActiveCount = threadCount;

        for (uint32_t t = 0; t < threadCount; t++) {
            compilationThreads.push_back(std::make_unique<CompilationThread>(this));
        }
#endif
    }

    // The cache the host's compile indicator reports on (one per application).
    static std::atomic<RasterShaderCache *> activeShaderCache = nullptr;

    RasterShaderCache::~RasterShaderCache() {
        RasterShaderCache *self = this;
        activeShaderCache.compare_exchange_strong(self, nullptr);
        compilationThreads.clear();
    }

    void RasterShaderCache::getCompileStatus(uint32_t &pending, uint32_t &compiled) {
        {
            const std::unique_lock<std::mutex> queueLock(descQueueMutex);
            pending = uint32_t(descQueue.size() + warmupQueue.size()) + uint32_t(std::max(descQueueActiveCount, 0));
        }

        compiled = compiledCount;
    }

    void GetShaderCompileStatus(uint32_t &pending, uint32_t &compiled, bool &drawsWait) {
        RasterShaderCache *cache = activeShaderCache;
        pending = 0;
        compiled = 0;
        drawsWait = false;
        if (cache != nullptr) {
            cache->getCompileStatus(pending, compiled);
            drawsWait = cache->noUbershader;
        }
    }

    void RasterShaderCache::setup(RenderDevice *device, RenderShaderFormat shaderFormat, const ShaderLibrary *shaderLibrary, const RenderMultisampling &multisampling, bool textureSlots, bool skipUbershader, const std::string &comboList) {
        assert(device != nullptr);

        this->device = device;
        this->shaderFormat = shaderFormat;
        this->multisampling = multisampling;

        noUbershader = textureSlots || skipUbershader;
        shaderUber = std::make_unique<RasterShaderUber>(device, shaderFormat, multisampling, shaderLibrary, ubershaderThreadCount, textureSlots, !noUbershader);
        activeShaderCache = this;
        usesHDR = shaderLibrary->usesHDR;

        // Initialize the re-spirv optimizer cache.
        if (shaderFormat == RenderShaderFormat::SPIRV) {
            optimizerCacheSPIRV.initialize(textureSlots);
        }

        // Combinations are recorded as otherMode L/H, combiner L/H, flags. Without an ubershader the lists seen in earlier sessions are
        // queued at startup, so draws find their pipelines ready (D3D11: plume's disk cache; Vulkan: the driver's) instead of popping in.
        std::string comboPath;
        bool warmShipped = false;
        if (const char *recordPath = std::getenv("ROGUESQ_RECORD_SHADER_COMBOS")) {
            comboPath = recordPath;
        }
        else if (textureSlots && (shaderFormat != RenderShaderFormat::SPIRV)) {
            const char *cacheDirectory = std::getenv("PLUME_D3D11_SHADER_CACHE");
            comboPath = std::string((cacheDirectory != nullptr) ? cacheDirectory : "d3d11_shader_cache") + "/rt64_combos.txt";
        }
        else if (noUbershader) {
            const char *cacheDirectory = (shaderFormat == RenderShaderFormat::SPIRV) ? "vulkan_shader_cache" : "d3d12_shader_cache";
            std::error_code ec;
            std::filesystem::create_directories(cacheDirectory, ec);
            comboPath = std::string(cacheDirectory) + "/rt64_combos.txt";
            warmShipped = true;
        }

        if (comboFile != nullptr) {
            std::fclose(comboFile);
            comboFile = nullptr;
        }

        const bool computeShaders = DecodesTexturesOnGPU(device->getCapabilities());
        auto submitCombo = [&](const ShaderDescription &desc) {
            ShaderDescription variant;
            if (!computeShaders && rawTMEMVariant(desc, variant)) {
                submit(variant, true);
            }

            submit(desc, true);
        };

        ShaderDescription desc = {};
        if (warmShipped) {
            const char *text = comboList.c_str();
            int consumed = 0;
            while (std::sscanf(text, "%X %X %X %X %X%n", &desc.otherMode.L, &desc.otherMode.H, &desc.colorCombiner.L, &desc.colorCombiner.H, &desc.flags.value, &consumed) == 5) {
                submitCombo(desc);
                text += consumed;
            }
        }

        const bool recording = (std::getenv("ROGUESQ_RECORD_SHADER_COMBOS") != nullptr);
        if (FILE *previous = (comboPath.empty() || recording) ? nullptr : std::fopen(comboPath.c_str(), "r")) {
            while (std::fscanf(previous, "%X %X %X %X %X", &desc.otherMode.L, &desc.otherMode.H, &desc.colorCombiner.L, &desc.colorCombiner.H, &desc.flags.value) == 5) {
                submitCombo(desc);
            }

            std::fclose(previous);
        }

        if (!comboPath.empty()) {
            comboFile = std::fopen(comboPath.c_str(), "a");
        }
    }

    void RasterShaderCache::submit(const ShaderDescription &desc, bool warmup) {
        {
            std::unique_lock<std::mutex> queueLock(submissionMutex);

            // Verify if an entry with the same hash was already submitted before.
            const uint64_t shaderHash = desc.hash();
            bool &found = shaderHashes[shaderHash];
            if (found) {
                if (warmup || (warmupPending.erase(shaderHash) == 0)) {
                    return;
                }
            }
            else {
                found = true;
                if (warmup) {
                    warmupPending.insert(shaderHash);
                }
                else if (comboFile != nullptr) {
                    std::fprintf(comboFile, "%08X %08X %08X %08X %08X\n", desc.otherMode.L, desc.otherMode.H, desc.colorCombiner.L, desc.colorCombiner.H, desc.flags.value);
                    std::fflush(comboFile);
                }
            }
        }

        // Push a new shader compilation to the queue.
        {
            const std::unique_lock<std::mutex> queueLock(descQueueMutex);
            (warmup ? warmupQueue : descQueue).push(desc);
        }

        descQueueChanged.notify_all();
    }
    
    void RasterShaderCache::waitForAll() {
        {
            std::unique_lock<std::mutex> queueLock(descQueueMutex);
            descQueue = std::queue<ShaderDescription>();
            warmupQueue = std::queue<ShaderDescription>();
        }

        bool keepWaiting = false;
        do {
            std::unique_lock<std::mutex> queueLock(descQueueMutex);
            keepWaiting = (descQueueActiveCount > 0);
        } while (keepWaiting);
    }

    void RasterShaderCache::destroyAll() {
        {
            std::unique_lock<std::mutex> lock(GPUShadersMutex);
            GPUShaders.clear();
        }

        {
            std::unique_lock<std::mutex> queueLock(submissionMutex);
            shaderHashes.clear();
        }
    }

    RasterShader *RasterShaderCache::getGPUShader(const ShaderDescription &desc) {
        const uint64_t shaderHash = desc.hash();

        const std::unique_lock<std::mutex> lock(GPUShadersMutex);
        auto shaderIt = GPUShaders.find(shaderHash);
        if (shaderIt == GPUShaders.end()) {
            return nullptr;
        }

        return shaderIt->second.get();
    }

    RasterShaderUber *RasterShaderCache::getGPUShaderUber() const {
        return shaderUber.get();
    }

    uint32_t RasterShaderCache::shaderCount() {
        std::unique_lock<std::mutex> lock(GPUShadersMutex);
        return GPUShaders.size();
    }
};
