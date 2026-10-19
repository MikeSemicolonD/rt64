//
// RT64
//

#include "rt64_rsp_processor.h"

#include "rt64_buffer_uploader.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>

#include "shared/rt64_rsp_fog.h"
#include "shared/rt64_rsp_light.h"
#include "shared/rt64_rsp_lookat.h"
#include "shared/rt64_rsp_viewport.h"

namespace RT64 {
    // CPU copies of RSPProcessCS's helpers. The GPU reads these row-major matrices column-major, so its mul(M, v) is mul(v, M) here and vice versa.

    static hlslpp::float3 normalizeSafe(const hlslpp::float3 &v) {
        const float l = hlslpp::length(v);
        return (l > 0.0f) ? (v / l) : v;
    }

    static float computeAttenuation(float distance, const interop::RSPLight &light) {
        const float A = 1.0f / 2.0f;
        const float B = 1.0f / 16.0f;
        const float C = 1.0f / 32768.0f;
        const float D = 1.0f / 524288.0f;
        const float Epsilon = 1e-6f;
        const float attenuation = 1.0f / (A + B * light.kc + C * light.kl * distance + D * light.kq * powf(distance, 2.0f));
        return fmodf(attenuation, 1.0f + Epsilon);
    }

    static float computeNDotL(const hlslpp::float3 &norm, const hlslpp::float3 &lightDir) {
        return std::clamp(float(hlslpp::dot(norm, lightDir)) * 4.0f, 0.0f, 1.0f);
    }

    static float computeLength(const hlslpp::float3 &d) {
        return sqrtf(d.f32[0] * d.f32[0] + d.f32[1] * d.f32[1] + 2.0f * d.f32[2] * d.f32[2]);
    }

    static hlslpp::float3 computePosLight(const hlslpp::float3 &pos, const hlslpp::float3 &norm, const interop::RSPLight &light, const hlslpp::float4x4 &worldMatrix) {
        const hlslpp::float3 worldVertexPos = hlslpp::mul(hlslpp::float4(pos, 1.0f), worldMatrix).xyz;
        hlslpp::float3 worldLightDir = hlslpp::float3(light.posDir) - worldVertexPos;
        const float worldLightDist = computeLength(worldLightDir);
        if (worldLightDist > 0.0f) {
            worldLightDir /= worldLightDist;
        }

        const hlslpp::float3 localLightDir = hlslpp::mul(worldMatrix, hlslpp::float4(worldLightDir, 0.0f)).xyz;
        return computeNDotL(norm, localLightDir) * computeAttenuation(worldLightDist, light) * hlslpp::float3(light.col);
    }

    static hlslpp::float3 computeDirLight(const hlslpp::float3 &norm, const interop::RSPLight &light, const hlslpp::float4x4 &worldMatrix) {
        hlslpp::float3 localLightDir = hlslpp::mul(worldMatrix, hlslpp::float4(hlslpp::float3(light.posDir), 0.0f)).xyz;
        const float localLightLength = hlslpp::length(localLightDir);
        if (localLightLength > 0.0f) {
            localLightDir /= localLightLength;
        }

        return std::max(float(hlslpp::dot(norm, localLightDir)), 0.0f) * hlslpp::float3(light.col);
    }

    static hlslpp::float2 computeTextureGen(const hlslpp::float2 &inputUV, const hlslpp::float3 &inputNormal, const interop::RSPLookAt &lookAt, bool textureGenLinear, const hlslpp::float4x4 &worldMatrix) {
        float texgenU = hlslpp::dot(inputNormal, normalizeSafe(hlslpp::mul(worldMatrix, hlslpp::float4(hlslpp::float3(lookAt.x), 0.0f)).xyz));
        float texgenV = hlslpp::dot(inputNormal, normalizeSafe(hlslpp::mul(worldMatrix, hlslpp::float4(hlslpp::float3(lookAt.y), 0.0f)).xyz));
        texgenU = std::clamp(texgenU, -1.0f, 1.0f);
        texgenV = std::clamp(texgenV, -1.0f, 1.0f);
        if (textureGenLinear) {
            texgenU = acosf(-texgenU) * 325.94932f;
            texgenV = acosf(-texgenV) * 325.94932f;
        }
        else {
            texgenU = (texgenU + 1.0f) * 512.0f;
            texgenV = (texgenV + 1.0f) * 512.0f;
        }

        return hlslpp::float2((inputUV.f32[0] / 65536.0f) * texgenU, (inputUV.f32[1] / 65536.0f) * texgenV);
    }

    template<typename T>
    static const T *uploadedData(const BufferPair &pair) {
        return (pair.uploadBuffer != nullptr) ? static_cast<const T *>(pair.uploadBuffer->map()) : nullptr;
    }

    // RSPProcessor

    RSPProcessor::RSPProcessor(RenderDevice *device) {
        processSet = std::make_unique<RSPProcessDescriptorSet>(device);
        modifySet = std::make_unique<RSPModifyDescriptorSet>(device);
        // ROGUESQ_CPU_RSP=1 transforms vertices on the CPU even with compute, for GPUs slower than the CPU (Raspberry Pi).
        const char *cpuSetting = std::getenv("ROGUESQ_CPU_RSP");
        cpuProcessing = !device->getCapabilities().computeShaders || ((cpuSetting != nullptr) && (cpuSetting[0] == '1'));
    }

    RSPProcessor::~RSPProcessor() { }
    
    void RSPProcessor::process(const ProcessParams &p) {
        const DrawData &drawData = *p.drawData;
        const uint32_t drawVertexCount = drawData.vertexCount();
        processCB.vertexStart = uint32_t(p.outputBuffers->screenPosBuffer.computedSize / (sizeof(float) * 4));
        processCB.vertexCount = drawVertexCount - processCB.vertexStart;
        processCB.prevFrameWeight = p.prevFrameWeight;
        processCB.curFrameWeight = p.curFrameWeight;
        p.outputBuffers->screenPosBuffer.computedSize += processCB.vertexCount * sizeof(float) * 4;
        p.outputBuffers->genTexCoordBuffer.computedSize += processCB.vertexCount * sizeof(float) * 2;
        p.outputBuffers->shadedColBuffer.computedSize += processCB.vertexCount * sizeof(float) * 4;
        if (cpuProcessing) {
            cpuDrawBuffers = p.drawBuffers;
            modifyCB.modifyCount = drawData.modifyCount();
            return;
        }

        processSet->setBuffer(processSet->srcPos, p.drawBuffers->positionBuffer.get(), p.drawBuffers->positionBuffer.getView(0));
        processSet->setBuffer(processSet->srcVel, p.drawBuffers->velocityBuffer.get(), p.drawBuffers->velocityBuffer.getView(0));
        processSet->setBuffer(processSet->srcTc, p.drawBuffers->texcoordBuffer.get(), p.drawBuffers->texcoordBuffer.getView(0));
        processSet->setBuffer(processSet->srcTcVel, p.drawBuffers->texcoordVelocityBuffer.get(), p.drawBuffers->texcoordVelocityBuffer.getView(0));
        processSet->setBuffer(processSet->srcCol, p.drawBuffers->normalColorBuffer.get(), p.drawBuffers->normalColorBuffer.getView(0));
        processSet->setBuffer(processSet->srcNorm, p.drawBuffers->normalColorBuffer.get(), p.drawBuffers->normalColorBuffer.getView(1));
        processSet->setBuffer(processSet->srcViewProjIndices, p.drawBuffers->viewProjIndicesBuffer.get(), p.drawBuffers->viewProjIndicesBuffer.getView(0));
        processSet->setBuffer(processSet->srcWorldIndices, p.drawBuffers->worldIndicesBuffer.get(), p.drawBuffers->worldIndicesBuffer.getView(0));
        processSet->setBuffer(processSet->srcFogIndices, p.drawBuffers->fogIndicesBuffer.get(), p.drawBuffers->fogIndicesBuffer.getView(0));
        processSet->setBuffer(processSet->srcLightIndices, p.drawBuffers->lightIndicesBuffer.get(), p.drawBuffers->lightIndicesBuffer.getView(0));
        processSet->setBuffer(processSet->srcLightCounts, p.drawBuffers->lightCountsBuffer.get(), p.drawBuffers->lightCountsBuffer.getView(0));
        processSet->setBuffer(processSet->srcLookAtIndices, p.drawBuffers->lookAtIndicesBuffer.get(), p.drawBuffers->lookAtIndicesBuffer.getView(0));
        processSet->setBuffer(processSet->rspViewportVector, p.drawBuffers->rspViewportsBuffer.get(), RenderBufferStructuredView(sizeof(interop::RSPViewport)));
        processSet->setBuffer(processSet->rspFogVector, p.drawBuffers->rspFogBuffer.get(), RenderBufferStructuredView(sizeof(interop::RSPFog)));
        processSet->setBuffer(processSet->rspLightVector, p.drawBuffers->rspLightsBuffer.get(), RenderBufferStructuredView(sizeof(interop::RSPLight)));
        processSet->setBuffer(processSet->rspLookAtVector, p.drawBuffers->rspLookAtBuffer.get(), RenderBufferStructuredView(sizeof(interop::RSPLookAt)));
        processSet->setBuffer(processSet->viewProjTransforms, p.drawBuffers->viewProjTransformsBuffer.get(), RenderBufferStructuredView(sizeof(interop::float4x4)));
        processSet->setBuffer(processSet->worldTransforms, p.drawBuffers->worldTransformsBuffer.get(), RenderBufferStructuredView(sizeof(interop::float4x4)));
        processSet->setBuffer(processSet->dstPos, p.outputBuffers->screenPosBuffer.buffer.get(), RenderBufferStructuredView(sizeof(float) * 4));
        processSet->setBuffer(processSet->dstTc, p.outputBuffers->genTexCoordBuffer.buffer.get(), RenderBufferStructuredView(sizeof(float) * 2));
        processSet->setBuffer(processSet->dstCol, p.outputBuffers->shadedColBuffer.buffer.get(), RenderBufferStructuredView(sizeof(float) * 4));

        const uint32_t modifyCount = drawData.modifyCount();
        modifyCB.modifyCount = modifyCount;
        modifySet->setBuffer(modifySet->srcModifyPos, p.drawBuffers->modifyPosUintsBuffer.get(), p.drawBuffers->modifyPosUintsBuffer.getView(0));
        modifySet->setBuffer(modifySet->screenPos, p.outputBuffers->screenPosBuffer.buffer.get(), RenderBufferStructuredView(sizeof(float) * 4));
    }

    void RSPProcessor::processOnCPU() {
        const DrawBuffers &b = *cpuDrawBuffers;
        const float *srcPos = uploadedData<float>(b.positionBuffer);
        const float *srcVel = uploadedData<float>(b.velocityBuffer);
        const float *srcTc = uploadedData<float>(b.texcoordBuffer);
        const float *srcTcVel = uploadedData<float>(b.texcoordVelocityBuffer);
        const uint8_t *srcCol = uploadedData<uint8_t>(b.normalColorBuffer);
        const int8_t *srcNorm = uploadedData<int8_t>(b.normalColorBuffer);
        const uint16_t *srcViewProjIndices = uploadedData<uint16_t>(b.viewProjIndicesBuffer);
        const uint16_t *srcWorldIndices = uploadedData<uint16_t>(b.worldIndicesBuffer);
        const uint16_t *srcFogIndices = uploadedData<uint16_t>(b.fogIndicesBuffer);
        const uint16_t *srcLightIndices = uploadedData<uint16_t>(b.lightIndicesBuffer);
        const uint8_t *srcLightCounts = uploadedData<uint8_t>(b.lightCountsBuffer);
        const uint16_t *srcLookAtIndices = uploadedData<uint16_t>(b.lookAtIndicesBuffer);
        const interop::RSPViewport *rspViewports = uploadedData<interop::RSPViewport>(b.rspViewportsBuffer);
        const interop::RSPFog *rspFogs = uploadedData<interop::RSPFog>(b.rspFogBuffer);
        const interop::RSPLight *rspLights = uploadedData<interop::RSPLight>(b.rspLightsBuffer);
        const interop::RSPLookAt *rspLookAts = uploadedData<interop::RSPLookAt>(b.rspLookAtBuffer);
        const interop::float4x4 *viewProjTransforms = uploadedData<interop::float4x4>(b.viewProjTransformsBuffer);
        const interop::float4x4 *worldTransforms = uploadedData<interop::float4x4>(b.worldTransformsBuffer);
        const uint32_t *srcModifyPos = uploadedData<uint32_t>(b.modifyPosUintsBuffer);

        cpuScreenPos.resize(processCB.vertexCount);
        cpuTexcoords.resize(processCB.vertexCount);
        cpuColors.resize(processCB.vertexCount);
        const bool inputsValid = (srcPos != nullptr) && (srcCol != nullptr) && (srcViewProjIndices != nullptr) && (srcWorldIndices != nullptr) && (viewProjTransforms != nullptr) && (worldTransforms != nullptr) && (rspViewports != nullptr);
        if ((processCB.vertexCount > 0) && !inputsValid) {
            processCB.vertexCount = 0;
        }

        for (uint32_t v = 0; v < processCB.vertexCount; v++) {
            const uint32_t vertexOffsetIndex = processCB.vertexStart + v;
            const uint32_t viewProjIndex = srcViewProjIndices[vertexOffsetIndex];
            const hlslpp::float4x4 viewProjMat = viewProjTransforms[viewProjIndex];
            const hlslpp::float4x4 worldMat = worldTransforms[srcWorldIndices[vertexOffsetIndex]];
            const uint32_t posIndex = vertexOffsetIndex * 3;
            const uint32_t normColIndex = vertexOffsetIndex * 4;
            const hlslpp::float3 norm(srcNorm[normColIndex + 0] / 127.0f, srcNorm[normColIndex + 1] / 127.0f, srcNorm[normColIndex + 2] / 127.0f);

            // NDC position.
            const hlslpp::float3 pos(srcPos[posIndex + 0], srcPos[posIndex + 1], srcPos[posIndex + 2]);
            const hlslpp::float3 vel = (srcVel != nullptr) ? hlslpp::float3(srcVel[posIndex + 0], srcVel[posIndex + 1], srcVel[posIndex + 2]) : hlslpp::float3(0.0f, 0.0f, 0.0f);
            hlslpp::float4 tfPos = hlslpp::mul(hlslpp::float4(pos - vel * (1.0f - processCB.curFrameWeight), 1.0f), hlslpp::mul(worldMat, viewProjMat));

            // Fog.
            const uint32_t fogIndex = (srcFogIndices != nullptr) ? srcFogIndices[vertexOffsetIndex] : 0;
            float alpha;
            if ((fogIndex > 0) && (rspFogs != nullptr)) {
                const interop::RSPFog &rspFog = rspFogs[fogIndex - 1];
                const float fogAlpha = (std::max(tfPos.f32[2], 0.0f) / tfPos.f32[3]) * rspFog.mul + rspFog.offset;
                alpha = std::clamp(fogAlpha / 255.0f, 0.0f, 1.0f);
            }
            else {
                alpha = srcCol[normColIndex + 3] / 255.0f;
            }

            // Texgen.
            const uint32_t tcIndex = vertexOffsetIndex * 2;
            const uint32_t lookAtIndex = (srcLookAtIndices != nullptr) ? srcLookAtIndices[vertexOffsetIndex] : 0;
            hlslpp::float2 tc = (srcTc != nullptr) ? hlslpp::float2(srcTc[tcIndex + 0], srcTc[tcIndex + 1]) : hlslpp::float2(0.0f, 0.0f);
            if ((lookAtIndex & RSP_LOOKAT_INDEX_ENABLED) && (rspLookAts != nullptr)) {
                const bool textureGenLinear = (lookAtIndex & RSP_LOOKAT_INDEX_LINEAR) != 0;
                tc = computeTextureGen(tc, norm, rspLookAts[lookAtIndex >> RSP_LOOKAT_INDEX_SHIFT], textureGenLinear, worldMat);
            }
            else if (srcTcVel != nullptr) {
                tc -= hlslpp::float2(srcTcVel[tcIndex + 0], srcTcVel[tcIndex + 1]) * (1.0f - processCB.curFrameWeight);
            }

            // Lighting.
            const uint32_t lightIndex = (srcLightIndices != nullptr) ? srcLightIndices[vertexOffsetIndex] : 0;
            const uint32_t lightCount = (srcLightCounts != nullptr) ? srcLightCounts[vertexOffsetIndex] : 0;
            hlslpp::float3 color;
            if ((lightCount > 0) && (rspLights != nullptr)) {
                const uint32_t ambientIndex = lightIndex + lightCount - 1;
                hlslpp::float3 resultColor = rspLights[ambientIndex].col;
                for (uint32_t i = lightIndex; i < ambientIndex; i++) {
                    const interop::RSPLight &light = rspLights[i];
                    resultColor += (light.kc > 0) ? computePosLight(pos, norm, light, worldMat) : computeDirLight(norm, light, worldMat);
                }

                color = hlslpp::min(resultColor, hlslpp::float3(1.0f, 1.0f, 1.0f));
            }
            else {
                color = hlslpp::float3(srcCol[normColIndex + 0] / 255.0f, srcCol[normColIndex + 1] / 255.0f, srcCol[normColIndex + 2] / 255.0f);
            }

            // Matches the shader's guard for geometry exactly at the near clip plane.
            if (tfPos.f32[3] == 0.0f) {
                tfPos.f32[3] = 1e-6f;
            }

            // Convert to N64 screen position.
            const interop::RSPViewport &rspViewport = rspViewports[viewProjIndex];
            const float w = tfPos.f32[3];
            const hlslpp::float3 ndcPos(tfPos.f32[0] / w, tfPos.f32[1] / -w, tfPos.f32[2] / w);
            cpuScreenPos[v] = hlslpp::float4(ndcPos * hlslpp::float3(rspViewport.scale) + hlslpp::float3(rspViewport.translate), w);
            cpuTexcoords[v] = tc;
            cpuColors[v] = hlslpp::float4(color, alpha);
        }

        // RSPModifyCS: per-vertex screen position overrides.
        for (uint32_t m = 0; (m < modifyCB.modifyCount) && (srcModifyPos != nullptr); m++) {
            const uint32_t modifyOffset = m * 2;
            const bool modifyZ = (srcModifyPos[modifyOffset] & 0x1) != 0;
            const uint32_t vertexIndex = srcModifyPos[modifyOffset] >> 1;
            const uint32_t modifyValue = srcModifyPos[modifyOffset + 1];
            if ((vertexIndex < processCB.vertexStart) || (vertexIndex >= (processCB.vertexStart + processCB.vertexCount))) {
                continue;
            }

            hlslpp::float4 &screenPos = cpuScreenPos[vertexIndex - processCB.vertexStart];
            if (modifyZ) {
                screenPos.f32[2] = modifyValue / 65536.0f;
            }
            else {
                screenPos.f32[0] = int16_t((modifyValue >> 16) & 0xFFFF) / 4.0f;
                screenPos.f32[1] = int16_t(modifyValue & 0xFFFF) / 4.0f;
            }
        }
    }

    void RSPProcessor::recordCPUUploads(RenderWorker *worker, const OutputBuffers *outputBuffers) {
        const uint32_t count = processCB.vertexCount;
        if (count == 0) {
            return;
        }

        if (cpuUploadCapacity < count) {
            cpuUploadCapacity = std::max<uint64_t>(count, cpuUploadCapacity * 2);
            cpuPosUpload = worker->device->createBuffer(RenderBufferDesc::UploadBuffer(cpuUploadCapacity * sizeof(float) * 4));
            cpuTcUpload = worker->device->createBuffer(RenderBufferDesc::UploadBuffer(cpuUploadCapacity * sizeof(float) * 2));
            cpuColUpload = worker->device->createBuffer(RenderBufferDesc::UploadBuffer(cpuUploadCapacity * sizeof(float) * 4));
        }

        float *pos = static_cast<float *>(cpuPosUpload->map());
        float *tc = static_cast<float *>(cpuTcUpload->map());
        float *col = static_cast<float *>(cpuColUpload->map());
        for (uint32_t v = 0; v < count; v++) {
            for (uint32_t c = 0; c < 4; c++) {
                pos[v * 4 + c] = cpuScreenPos[v].f32[c];
                col[v * 4 + c] = cpuColors[v].f32[c];
            }

            tc[v * 2 + 0] = cpuTexcoords[v].f32[0];
            tc[v * 2 + 1] = cpuTexcoords[v].f32[1];
        }

        cpuPosUpload->unmap();
        cpuTcUpload->unmap();
        cpuColUpload->unmap();

        const uint64_t start = processCB.vertexStart;
        RenderBufferBarrier beforeBarriers[] = {
            RenderBufferBarrier(outputBuffers->screenPosBuffer.buffer.get(), RenderBufferAccess::WRITE),
            RenderBufferBarrier(outputBuffers->genTexCoordBuffer.buffer.get(), RenderBufferAccess::WRITE),
            RenderBufferBarrier(outputBuffers->shadedColBuffer.buffer.get(), RenderBufferAccess::WRITE)
        };

        RenderBufferBarrier afterBarriers[] = {
            RenderBufferBarrier(outputBuffers->screenPosBuffer.buffer.get(), RenderBufferAccess::READ),
            RenderBufferBarrier(outputBuffers->genTexCoordBuffer.buffer.get(), RenderBufferAccess::READ),
            RenderBufferBarrier(outputBuffers->shadedColBuffer.buffer.get(), RenderBufferAccess::READ)
        };

        worker->commandList->barriers(RenderBarrierStage::COPY, beforeBarriers, uint32_t(std::size(beforeBarriers)));
        worker->commandList->copyBufferRegion(outputBuffers->screenPosBuffer.buffer->at(start * sizeof(float) * 4), cpuPosUpload->at(0), count * sizeof(float) * 4);
        worker->commandList->copyBufferRegion(outputBuffers->genTexCoordBuffer.buffer->at(start * sizeof(float) * 2), cpuTcUpload->at(0), count * sizeof(float) * 2);
        worker->commandList->copyBufferRegion(outputBuffers->shadedColBuffer.buffer->at(start * sizeof(float) * 4), cpuColUpload->at(0), count * sizeof(float) * 4);
        worker->commandList->barriers(RenderBarrierStage::GRAPHICS, afterBarriers, uint32_t(std::size(afterBarriers)));
    }

    void RSPProcessor::recordCommandList(RenderWorker *worker, const ShaderLibrary *shaderLibrary, const OutputBuffers *outputBuffers) {
        const uint32_t ThreadGroupSize = 64;
        if (cpuProcessing) {
            processOnCPU();
            recordCPUUploads(worker, outputBuffers);
            return;
        }

        if (processCB.vertexCount > 0) {
            RenderBufferBarrier beforeBarriers[] = {
                RenderBufferBarrier(outputBuffers->screenPosBuffer.buffer.get(), RenderBufferAccess::WRITE),
                RenderBufferBarrier(outputBuffers->genTexCoordBuffer.buffer.get(), RenderBufferAccess::WRITE),
                RenderBufferBarrier(outputBuffers->shadedColBuffer.buffer.get(), RenderBufferAccess::WRITE)
            };

            RenderBufferBarrier afterBarriers[] = {
                RenderBufferBarrier(outputBuffers->screenPosBuffer.buffer.get(), RenderBufferAccess::READ),
                RenderBufferBarrier(outputBuffers->genTexCoordBuffer.buffer.get(), RenderBufferAccess::READ),
                RenderBufferBarrier(outputBuffers->shadedColBuffer.buffer.get(), RenderBufferAccess::READ)
            };

            const int dispatchCount = (processCB.vertexCount + ThreadGroupSize - 1) / ThreadGroupSize;
            worker->commandList->barriers(RenderBarrierStage::COMPUTE, beforeBarriers, uint32_t(std::size(beforeBarriers)));
            worker->commandList->setPipeline(shaderLibrary->rspProcess.pipeline.get());
            worker->commandList->setComputePipelineLayout(shaderLibrary->rspProcess.pipelineLayout.get());
            worker->commandList->setComputePushConstants(0, &processCB);
            worker->commandList->setComputeDescriptorSet(processSet->get(), 0);
            worker->commandList->dispatch(dispatchCount, 1, 1);
            worker->commandList->barriers(RenderBarrierStage::GRAPHICS, afterBarriers, uint32_t(std::size(afterBarriers)));
        }

        if (modifyCB.modifyCount > 0) {
            RenderBufferBarrier beforeBarrier = RenderBufferBarrier(outputBuffers->screenPosBuffer.buffer.get(), RenderBufferAccess::WRITE);
            RenderBufferBarrier afterBarrier = RenderBufferBarrier(outputBuffers->screenPosBuffer.buffer.get(), RenderBufferAccess::READ);
            const int dispatchCount = (modifyCB.modifyCount + ThreadGroupSize - 1) / ThreadGroupSize;
            worker->commandList->barriers(RenderBarrierStage::COMPUTE, beforeBarrier);
            worker->commandList->setPipeline(shaderLibrary->rspModify.pipeline.get());
            worker->commandList->setComputePipelineLayout(shaderLibrary->rspModify.pipelineLayout.get());
            worker->commandList->setComputePushConstants(0, &modifyCB);
            worker->commandList->setComputeDescriptorSet(modifySet->get(), 0);
            worker->commandList->dispatch(dispatchCount, 1, 1);
            worker->commandList->barriers(RenderBarrierStage::GRAPHICS, afterBarrier);
        }
    }
};
