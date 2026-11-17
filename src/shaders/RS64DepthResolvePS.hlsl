// RT64 - Sample 0 of the multisampled depth target as a single-sampled R32F texture, for the RS64 passes that read depth (RS64_SINGLE_SAMPLE: a plain copy)

#ifdef RS64_SINGLE_SAMPLE
Texture2D<float> gInput : register(t1);
#else
Texture2DMS<float> gInput : register(t1);
#endif

float PSMain(in float4 pos : SV_Position, in float2 uv : TEXCOORD0) : SV_TARGET {
#ifdef RS64_SINGLE_SAMPLE
    return gInput.Load(int3(pos.xy, 0));
#else
    return gInput.Load(int2(pos.xy), 0);
#endif
}
