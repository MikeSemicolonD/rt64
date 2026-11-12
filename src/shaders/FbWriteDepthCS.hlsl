//
// RT64
//

#include "Depth.hlsli"
#include "FbCommon.hlsli"

[[vk::push_constant]] ConstantBuffer<FbCommonCB> gConstants : register(b0, space0);
#ifdef RT64_WEBGPU
RWByteAddressBuffer gOutput : register(u1, space0);
#define STORE_OUTPUT(i, v) storeUint(gOutput, i, FbBufferBytes(gConstants.siz), v)
#else
RWBuffer<uint> gOutput : register(u1, space0);
#define STORE_OUTPUT(i, v) gOutput[i] = v
#endif
#ifdef MULTISAMPLING
Texture2DMS<float> gInput : register(t0, space1);
#else
Texture2D<float> gInput : register(t0, space1);
#endif

[numthreads(FB_COMMON_WORKGROUP_SIZE, FB_COMMON_WORKGROUP_SIZE, 1)]
void CSMain(uint2 coord : SV_DispatchThreadID) {
    if ((coord.x < gConstants.resolution.x) && (coord.y < gConstants.resolution.y)) {
        uint2 offsetCoord = gConstants.offset + coord;
        uint dstIndex = offsetCoord.y * gConstants.resolution.x + offsetCoord.x;
#   ifdef MULTISAMPLING
        float inputDepth = gInput.Load(offsetCoord, 0);
#else
        float inputDepth = gInput.Load(uint3(offsetCoord, 0));
#   endif
        float z = clamp(inputDepth, 0.0f, 1.0f);
        float dz = 0.0f; // TODO
        STORE_OUTPUT(dstIndex, EndianSwapUINT16(FloatToDepth16(z, dz)));
    }
}