//
// RT64
//

#include "Depth.hlsli"
#include "FbCommon.hlsli"

[[vk::push_constant]] ConstantBuffer<FbCommonCB> gConstants : register(b0, space0);
#ifdef RT64_WEBGPU
ByteAddressBuffer gNewInput : register(t1, space0);
ByteAddressBuffer gCurInput : register(t2, space0);
#define NEW_INPUT(i) loadUint(gNewInput, i, FbBufferBytes(gConstants.siz))
#define CUR_INPUT(i) loadUint(gCurInput, i, FbBufferBytes(gConstants.siz))
#else
Buffer<uint> gNewInput : register(t1, space0);
Buffer<uint> gCurInput : register(t2, space0);
#define NEW_INPUT(i) gNewInput[i]
#define CUR_INPUT(i) gCurInput[i]
#endif
RWStructuredBuffer<uint> gOutputCount : register(u3, space0);
// 25 is SPIR-V NonReadable: WebGPU allows these formats only as write-only storage textures.
[[vk::image_format("rgba8")]] [[vk::ext_decorate(25)]] RWTexture2D<float4> gOutputChangeColor : register(u0, space1);
[[vk::image_format("r32f")]] [[vk::ext_decorate(25)]] RWTexture2D<float> gOutputChangeDepth : register(u1, space1);
#ifdef RT64_WEBGPU
// r8ui isn't a WebGPU storage format.
[[vk::image_format("r32ui")]] [[vk::ext_decorate(25)]] RWTexture2D<uint> gOutputChangeBoolean : register(u2, space1);
#else
[[vk::image_format("r8ui")]] [[vk::ext_decorate(25)]] RWTexture2D<uint> gOutputChangeBoolean : register(u2, space1);
#endif

[numthreads(FB_COMMON_WORKGROUP_SIZE, FB_COMMON_WORKGROUP_SIZE, 1)]
void CSMain(uint2 coord : SV_DispatchThreadID) {
    if ((coord.x < gConstants.resolution.x) && (coord.y < gConstants.resolution.y)) {
        const uint bufferIndex = coord.y * gConstants.resolution.x.x + coord.x;
        const uint2 pixelCoord = gConstants.offset + coord.xy;
        if (NEW_INPUT(bufferIndex) != CUR_INPUT(bufferIndex)) {
            const uint swappedUint = EndianSwapUINT(NEW_INPUT(bufferIndex), gConstants.siz);
            if (gConstants.fmt == G_IM_FMT_DEPTH) {
                const float newDepth = Depth16ToFloat(swappedUint);
                gOutputChangeDepth[pixelCoord] = newDepth;
            }
            else {
                const float4 newColor = UINTToFloat4(swappedUint, gConstants.siz, gConstants.fmt);
                gOutputChangeColor[pixelCoord] = newColor;
            }

            gOutputChangeBoolean[pixelCoord] = 1;
            InterlockedAdd(gOutputCount[0], 1);
        }
        else {
            gOutputChangeBoolean[pixelCoord] = 0;
        }
    }
}