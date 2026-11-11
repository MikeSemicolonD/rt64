// RT64 - Sample 0 of the multisampled depth target as a single-sampled R32F texture, for the RS64 passes that read depth

Texture2DMS<float> gInput : register(t1);

float PSMain(in float4 pos : SV_Position, in float2 uv : TEXCOORD0) : SV_TARGET {
    return gInput.Load(int2(pos.xy), 0);
}
