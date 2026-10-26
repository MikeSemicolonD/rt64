//
// RT64
//

#include "shared/rt64_rs64_shadow_blur_cb.h"

ConstantBuffer<RS64ShadowBlurCB> gBlur : register(b0);
Texture2D<float> gDepth : register(t1);
Texture2D<float4> gMask : register(t2);
Texture2D<float4> gIndirect : register(t3);
Texture2D<float4> gHistMask : register(t4);
Texture2D<float4> gHistIndirect : register(t5);

struct TemporalOut {
    float4 mask : SV_TARGET0;
    float4 indirect : SV_TARGET1;
};

// Accumulates the traced AO (mask.a) and indirect light (indirect.rgb) over frames: the surface's position reprojected into the previous frame through
// the camera motion, kept when that frame saw the same surface there (its distance code matches). Shadow, radius, distance and reflection stay per frame.
TemporalOut PSMain(in float4 pos : SV_Position, in float2 uv : TEXCOORD0) {
    const int2 px = int2(pos.xy);
    TemporalOut o;
    o.mask = gMask.Load(int3(px, 0));
    o.indirect = gIndirect.Load(int3(px, 0));
    if ((gBlur.temporal.x <= 0.0f) || (o.mask.b >= 60000.0f)) {
        return o;
    }

    const float depth = gDepth.Load(int3(px, 0));
    const float2 rel = (pos.xy - gBlur.viewport.xy) / gBlur.viewport.zw;
    const float4 v4 = mul(gBlur.viewFromScreen, float4(rel.x * 2.0f - 1.0f, 1.0f - rel.y * 2.0f, depth, 1.0f));
    const float3 p = v4.xyz / v4.w;
    const float4 s4 = mul(gBlur.prevScreenFromView, float4(p, 1.0f));
    if (s4.w <= 0.0f) {
        return o;
    }

    const float2 prel = float2(s4.x / s4.w * 0.5f + 0.5f, 0.5f - s4.y / s4.w * 0.5f);
    if (any(prel < 0.0f) || any(prel >= 1.0f)) {
        return o;
    }

    // Bilinear over the four nearest history texels (a nearest read snaps between texels as the camera drifts and shimmers); each tap must be the same surface.
    const float2 f = gBlur.prevViewport.xy + prel * gBlur.prevViewport.zw - 0.5f;
    const int2 q0 = int2(floor(f));
    const float2 t = f - float2(q0);
    const int2 lo = int2(gBlur.prevViewport.xy);
    const int2 hi = int2(gBlur.prevViewport.xy + gBlur.prevViewport.zw) - 1;
    const float4 pp = mul(gBlur.prevViewFromView, float4(p, 1.0f));
    const float dPrev = length(pp.xyz / pp.w);
    float ao = 0.0f;
    float3 ind = 0.0f;
    float wsum = 0.0f;
    [unroll]
    for (int k = 0; k < 4; k++) {
        const int2 o2 = int2(k & 1, k >> 1);
        const int2 q = clamp(q0 + o2, lo, hi);
        const float w = (o2.x ? t.x : 1.0f - t.x) * (o2.y ? t.y : 1.0f - t.y);
        const float4 h = gHistMask.Load(int3(q, 0));
        const float dHist = (h.b >= 60000.0f) ? 1.0e30f : (h.b * 64.0f);
        if ((w > 0.0f) && (abs(dHist - dPrev) <= (gBlur.temporal.y * dPrev))) {
            ao += w * h.a;
            ind += w * gHistIndirect.Load(int3(q, 0)).rgb;
            wsum += w;
        }
    }
    if (wsum < 0.25f) {
        return o;
    }

    const float a = gBlur.temporal.z;
    o.mask.a = lerp(ao / wsum, o.mask.a, a);
    o.indirect.rgb = lerp(ind / wsum, o.indirect.rgb, a);
    return o;
}
