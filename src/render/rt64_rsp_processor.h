//
// RT64
//

#pragma once

#include "hle/rt64_workload.h"

#include "rt64_descriptor_sets.h"

namespace RT64 {
    struct RSPProcessor {
        struct ProcessCB {
            uint32_t vertexStart;
            uint32_t vertexCount;
            float prevFrameWeight;
            float curFrameWeight;
        };

        struct ModifyCB {
            uint32_t modifyCount;
        };

        struct ProcessParams {
            RenderWorker *worker = nullptr;
            DrawData *drawData = nullptr;
            DrawBuffers *drawBuffers = nullptr;
            OutputBuffers *outputBuffers = nullptr;
            float prevFrameWeight = 0.0f;
            float curFrameWeight = 1.0f;
        };

        ProcessCB processCB;
        ModifyCB modifyCB;
        std::unique_ptr<RSPProcessDescriptorSet> processSet;
        std::unique_ptr<RSPModifyDescriptorSet> modifySet;

        // Devices without compute (D3D10-class) run RSPProcessCS and RSPModifyCS on the CPU from the upload buffers' contents.
        bool cpuProcessing = false;
        const DrawBuffers *cpuDrawBuffers = nullptr;
        std::vector<hlslpp::float4> cpuScreenPos;
        std::vector<hlslpp::float2> cpuTexcoords;
        std::vector<hlslpp::float4> cpuColors;
        std::unique_ptr<RenderBuffer> cpuPosUpload;
        std::unique_ptr<RenderBuffer> cpuTcUpload;
        std::unique_ptr<RenderBuffer> cpuColUpload;
        uint64_t cpuUploadCapacity = 0;

        RSPProcessor(RenderDevice *device);
        ~RSPProcessor();
        void process(const ProcessParams &p);
        void recordCommandList(RenderWorker *worker, const ShaderLibrary *shaderLibrary, const OutputBuffers *outputBuffers);
        void processOnCPU();
        void recordCPUUploads(RenderWorker *worker, const OutputBuffers *outputBuffers);
    };
};
