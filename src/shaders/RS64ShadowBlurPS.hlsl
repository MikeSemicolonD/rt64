//
// RT64
//

#include "shared/rt64_rs64_shadow_blur_cb.h"

ConstantBuffer<RS64ShadowBlurCB> gBlur : register(b0);
Texture2D<float> gDepth : register(t1);
Texture2D<float4> gMask : register(t2);
Texture2D<float4> gIndirect : register(t3);

// View distance from the mask's distance code (written by the soft trace: distance / 64, sky = 65000); sky reads as infinitely far.
float decodeDist(float code) {
    return (code >= 60000.0f) ? 1.0e30f : (code * 64.0f);
}

// Mirrors rs64lights::shadowDepthWeight; dc is the expected distance at the tap.
float depthWeight(float dc, float ds) {
    if ((ds >= 1.0e29f) || (dc <= 0.0f)) {
        return 0.0f;
    }
    return saturate(1.0f - abs(dc - ds) / (gBlur.params.y * dc));
}

// Mirrors rs64lights::depthSlope.
float depthSlope(float dc, float dMinus, float dPlus) {
    const bool m = dMinus < 1.0e29f;
    const bool p = dPlus < 1.0e29f;
    const float back = dc - dMinus;
    const float fwd = dPlus - dc;
    if (m && p) {
        return (abs(back) < abs(fwd)) ? back : fwd;
    }
    return m ? back : (p ? fwd : 0.0f);
}

// Mirrors rs64lights::blurTapCount.
int blurTapCount(float radiusPx) {
    return clamp(int(ceil(radiusPx * 0.5f)), 1, 16);
}

#ifdef RS64_BLUR_V
float4 PSMain(in float4 pos : SV_Position, in float2 uv : TEXCOORD0) : SV_TARGET {
#else
struct BlurOut {
    float4 mask : SV_TARGET0;
    float4 indirect : SV_TARGET1;
};

BlurOut PSMain(in float4 pos : SV_Position, in float2 uv : TEXCOORD0) {
#endif
    const int2 px = int2(pos.xy);
    const float4 c = gMask.Load(int3(px, 0));
    const float4 ci = gIndirect.Load(int3(px, 0));
    const int2 lo = int2(gBlur.viewport.xy);
    const int2 hi = int2(gBlur.viewport.xy + gBlur.viewport.zw) - 1;
    // Lit pixels carry radius 0; take the radius of any neighbour whose penumbra reaches here so the blur spreads onto both sides of the edge.
    float radius = c.g;
    const int dTaps = blurTapCount(gBlur.params.w);
    [loop]
    for (int d = 1; d <= dTaps; d++) {
        const float off = gBlur.params.w * float(d) / float(dTaps);
        [unroll]
        for (int sd = -1; sd <= 1; sd += 2) {
            const int2 q = clamp(px + int2(round(gBlur.texelDir * (off * float(sd)))), lo, hi);
            const float rq = gMask.Load(int3(q, 0)).g;
            if (rq >= off) {
                radius = max(radius, rq);
            }
        }
    }
    // Shadow (r) and AO (a) of the mask and the indirect rgb + reflection share one set of weights.
    float s = c.r;
    float ao = c.a;
    float4 ind = ci;
    const float dc = decodeDist(c.b);
    if ((radius > 0.5f) && (dc < 1.0e29f)) {
        // Taps spread over the penumbra radius, Gaussian in offset, weighted against the surface's own depth slope (ground at a grazing angle keeps its taps).
        const int2 dir = int2(gBlur.texelDir);
        const float slope = depthSlope(dc, decodeDist(gMask.Load(int3(clamp(px - dir, lo, hi), 0)).b), decodeDist(gMask.Load(int3(clamp(px + dir, lo, hi), 0)).b));
        const int taps = blurTapCount(radius);
        float2 sum = float2(c.r, c.a);
        float4 isum = ci;
        float wsum = 1.0f;
        [loop]
        for (int k = 1; k <= taps; k++) {
            const float u = float(k) / float(taps);
            const float off = radius * u;
            const float g = exp(-2.72f * u * u);
            [unroll]
            for (int sgn = -1; sgn <= 1; sgn += 2) {
                const int2 q = clamp(px + int2(round(gBlur.texelDir * (off * float(sgn)))), lo, hi);
                const float stepPx = dot(float2(q - px), gBlur.texelDir);
                const float4 tq = gMask.Load(int3(q, 0));
                const float w = g * depthWeight(dc + slope * stepPx, decodeDist(tq.b));
                sum += w * float2(tq.r, tq.a);
                isum += w * gIndirect.Load(int3(q, 0));
                wsum += w;
            }
        }
        s = sum.x / wsum;
        ao = sum.y / wsum;
        ind = isum / wsum;
    }
#ifdef RS64_BLUR_V
    // Debug views (copy blend): 5 shadow, 6 AO, 7 indirect, 8 reflection, 9 raw reflection hit colour.
    const float dbg = gBlur.aoParams.y;
    if (dbg > 8.5f) {
        return float4(ind.rgb, 1.0f);
    }
    if (dbg > 7.5f) {
        return float4(ind.aaa, 1.0f);
    }
    if (dbg > 6.5f) {
        return float4(ind.rgb, 1.0f);
    }
    if (dbg > 5.5f) {
        return float4((1.0f - ao).xxx, 1.0f);
    }
    if (dbg > 4.5f) {
        return float4((1.0f - s).xxx, 1.0f);
    }
    // Mirrors rs64lights::composeFactor (blend dst * src + dst * srcA).
    const float dark = (1.0f - gBlur.params.x * s) * (1.0f - gBlur.aoParams.x * ao);
    return float4(dark * saturate(ind.rgb), dark);
#else
    BlurOut o;
    o.mask = float4(s, radius, c.b, ao);
    o.indirect = ind;
    return o;
#endif
}
