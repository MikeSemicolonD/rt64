// RT64 - VR controller stand-in: light from above in the tracking space, opaque over a transparent layer.

float4 PSMain(float4 position : SV_POSITION, float3 normal : NORMAL, float3 color : COLOR0) : SV_TARGET {
    const float3 light = normalize(float3(0.3f, 1.0f, 0.4f));
    const float ndl = saturate(dot(normalize(normal), light));
    return float4(saturate(color * (0.35f + 0.65f * ndl)), 1.0f);
}
