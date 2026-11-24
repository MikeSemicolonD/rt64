// RT64 - Browser test: copies the first 16 words of a bound acceleration structure (plume WebGPU's software AS heap) to a buffer

ByteAddressBuffer gScene : register(t0);
RWByteAddressBuffer gOut : register(u1);

[numthreads(16, 1, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    gOut.Store(id.x * 4, gScene.Load(id.x * 4));
}
