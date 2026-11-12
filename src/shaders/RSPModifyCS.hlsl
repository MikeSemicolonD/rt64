//
// RT64
//

#define GROUP_SIZE 64

struct RSPModifyCB {
    uint modifyCount;
};

[[vk::push_constant]] ConstantBuffer<RSPModifyCB> gConstants : register(b0);
#ifdef RT64_WEBGPU
ByteAddressBuffer srcModifyPos : register(t1);
#define MODIFY_POS(i) srcModifyPos.Load((i) * 4)
#else
Buffer<uint> srcModifyPos : register(t1);
#define MODIFY_POS(i) srcModifyPos[i]
#endif
RWStructuredBuffer<float4> screenPos : register(u2);

[numthreads(GROUP_SIZE, 1, 1)]
void CSMain(uint modifyIndex : SV_DispatchThreadID) {
    if (modifyIndex >= gConstants.modifyCount) {
        return;
    }

    const uint modifyOffset = modifyIndex * 2;
    const bool modifyZ = MODIFY_POS(modifyOffset) & 0x1;
    const uint vertexIndex = MODIFY_POS(modifyOffset) >> 1;
    const uint modifyValue = MODIFY_POS(modifyOffset + 1);
    if (modifyZ) {
        screenPos[vertexIndex].z = modifyValue / 65536.0f;
    }
    else {
        const uint extX = (modifyValue >> 16) & 0xFFFF;
        const uint extY = modifyValue & 0xFFFF;
        const int intX = int(extX) << 16 >> 16;
        const int intY = int(extY) << 16 >> 16;
        screenPos[vertexIndex].x = intX / 4.0f;
        screenPos[vertexIndex].y = intY / 4.0f;
    }
}