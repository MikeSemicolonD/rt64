// RT64 - VR controller stand-in: light from above in the tracking space, opaque over a transparent layer, hidden where the eye's scene depth is nearer.

#include "shared/rt64_rs64_vr_hands_cb.h"

ConstantBuffer<RS64VrHandsDepthCB> gDepth : register(b1);
Texture2D<float> gSceneDepth : register(t2);

// True when the scene drawn for this eye is nearer than `stage` (a point in the tracking space).
bool behindScene(float3 stage) {
    const float4 s = float4(stage, 1.0f);
    const float3 e = float3(dot(gDepth.eyeFromStage[0], s), dot(gDepth.eyeFromStage[1], s), dot(gDepth.eyeFromStage[2], s));
    if (e.z >= 0.0f) {
        return false;
    }

    const float2 t = e.xy / -e.z;
    const float4 f = gDepth.tanFov;
    const float2 uv = float2((t.x - f.x) / (f.y - f.x), (f.z - t.y) / (f.z - f.w));
    if (any(uv < 0.0f) || any(uv >= 1.0f)) {
        return false;
    }

    // The game's camera space flips y and z and is in game units.
    const float upm = gDepth.params.x;
    const float4 g = float4(e.x * upm, -e.y * upm, -e.z * upm, 1.0f);
    const float wc = dot(g, gDepth.wColumn);
    if (wc <= 0.0f) {
        return false;
    }

    const float depth = (dot(g, gDepth.zColumn) / wc) * gDepth.params.y + gDepth.params.z;
    return depth > gSceneDepth.Load(int3(uv * gDepth.size.xy, 0));
}

float4 PSMain(float4 position : SV_POSITION, float3 normal : NORMAL, float3 color : COLOR0, float3 stage : TEXCOORD0) : SV_TARGET {
    if ((gDepth.params.w > 0.5f) && behindScene(stage)) {
        discard;
    }

    const float3 light = normalize(float3(0.3f, 1.0f, 0.4f));
    const float ndl = saturate(dot(normalize(normal), light));
    return float4(saturate(color * (0.35f + 0.65f * ndl)), 1.0f);
}
