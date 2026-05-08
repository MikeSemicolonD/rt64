//
// RT64
//

#include "shared/rt64_video_interface.h"

[[vk::push_constant]] ConstantBuffer<VideoInterfaceCB> gConstants : register(b0);
Texture2D<float4> gInput : register(t1);
SamplerState gSampler : register(s2);

// Limit texture sampling to the area the VI can sample of the texture.

float4 SampleInputRaw(float2 uv) {
    const float2 LowerRight = gConstants.videoResolution / gConstants.textureResolution;
    const float2 HalfPixel = float2(0.5f, 0.5f) / gConstants.textureResolution;
    float2 outsideBorder = step(LowerRight - HalfPixel, uv);
    float4 sampledColor = gInput.SampleLevel(gSampler, clamp(uv, HalfPixel, LowerRight - HalfPixel), 0);
    sampledColor.rgb *= max(1.0f - outsideBorder.x - outsideBorder.y, 0.0f);
    return sampledColor;
}

// Forward declaration — used by edge-aware AA below.
float Luma(float3 c);

// VI post-processing: approximates the N64 VI hardware filter chain.
// - aaMode 0/1/2 (RESAMP_*): N64 hardware re-fetches up to 7 neighbor
//   pixels and blends weighted by 3-bit coverage stored in alpha. We don't
//   preserve coverage through RT64's pipeline, so we approximate with
//   edge-aware blending: detect coverage-equivalent (luma difference
//   between center and neighbors) and blend only at edges. Interior
//   pixels of fully-opaque objects pass through unchanged — preserves
//   thin features like distant X-Wings or F5 logo edges that a uniform
//   blur would smear away.
// - aaMode 3 (NONE): single sample.
float4 SampleAA(float2 uv) {
    const uint aaMode = gConstants.viFlags & 0x3u;
    if (aaMode == 3u) {
        return SampleInputRaw(uv);
    }
    // Sample at video-pixel offsets so the blur radius scales with the
    // game's native resolution, not the upscaled render target.
    const float2 PixelStep = 1.0f / gConstants.videoResolution;
    float4 c   = SampleInputRaw(uv);
    float4 nL  = SampleInputRaw(uv + float2(-PixelStep.x, 0));
    float4 nR  = SampleInputRaw(uv + float2( PixelStep.x, 0));
    float4 nT  = SampleInputRaw(uv + float2(0, -PixelStep.y));
    float4 nB  = SampleInputRaw(uv + float2(0,  PixelStep.y));
    // Edge detection by max neighbor-vs-center luma difference. A small
    // diff means we're in a uniform region (interior) — don't blur.
    // A large diff means we're at an edge — apply full 5-tap blend.
    // Threshold scaled so that anything > ~25% luma difference is fully
    // edge-blended; smaller differences interpolate smoothly.
    float lc = Luma(c.rgb);
    float maxDiff = max(max(abs(Luma(nL.rgb) - lc), abs(Luma(nR.rgb) - lc)),
                        max(abs(Luma(nT.rgb) - lc), abs(Luma(nB.rgb) - lc)));
    float blend = saturate(maxDiff * 4.0f);
    float4 averaged = c * 0.5f + (nL + nR + nT + nB) * 0.125f;
    return lerp(c, averaged, blend);
}

// Divot filter — N64 VI applies a 3-tap horizontal "median" to suppress
// isolated single-pixel artifacts at triangle corners. Real hardware
// operates on luma, not per-channel: it picks the SAMPLE whose Y is the
// median of three. Per-channel min/max/median scrambles colors (median of
// pure R/G/B = black), suppressing legitimate high-contrast highlights
// like the X-Wing engine glow. This whole-sample-pick approach preserves
// color identity.
float Luma(float3 c) {
    return dot(c, float3(0.299f, 0.587f, 0.114f));
}

float4 PickMedianByLuma(float4 a, float4 b, float4 c) {
    float la = Luma(a.rgb), lb = Luma(b.rgb), lc = Luma(c.rgb);
    // The median of three values: whichever is between the other two.
    bool aIsMedian = ((la >= lb) && (la <= lc)) || ((la >= lc) && (la <= lb));
    bool bIsMedian = ((lb >= la) && (lb <= lc)) || ((lb >= lc) && (lb <= la));
    return aIsMedian ? a : (bIsMedian ? b : c);
}

// VI post-processing dispatcher: AA → divot → return.
//
// Divot temporarily disabled (2026-05-06): combining divot's 3 horizontal
// samples with each one being a 5-tap AA = 15 effective texture fetches
// per pixel for a fullscreen pass. Caused a GPU-driver crash
// (nvwgf2umx.dll) with garbled F5 logo just before. Keeping AA alone
// since that was visually validated. Re-enable divot under an explicit
// gate or with reduced sample count before re-investing.
float4 SampleInput(float2 uv) {
    return SampleAA(uv);
}

// 2x2 Bayer pattern values normalized to [-0.5, 0.5] then scaled by
// 1/255 — this is the magnitude N64 hardware uses to break up 5-bit
// (RGB555 framebuffer) color banding when gamma dither is enabled.
// Position-based so it's stable frame-to-frame, no temporal noise.
float GammaDitherOffset(float2 fragCoord) {
    int2 p = int2(fragCoord) & 1;
    int idx = p.y * 2 + p.x;
    // Bayer 2x2: 0,2,3,1 / 4 - 0.5 = -0.5, 0.0, +0.25, -0.25
    float t = (idx == 0) ? -0.5f
            : (idx == 1) ?  0.0f
            : (idx == 2) ? +0.25f
            :              -0.25f;
    return t / 255.0f;
}

float4 SampleInputGamma(float2 uv, float2 /*fragCoord*/) {
    // Gamma dither also temporarily disabled (2026-05-06) along with
    // divot — re-enable separately once we've validated each filter
    // independently against the cinematic without crashing.
    float4 c = pow(SampleInput(uv), gConstants.gamma);
    c.a = 1.0f;
    return c;
}

//
// Sourced from https://www.shadertoy.com/view/csX3RH
//
float4 PixelAntialiasing(float2 uv, float2 fragCoord) {
    float2 uvTexspace = uv * gConstants.videoResolution;
    float2 seam = floor(uvTexspace + 0.5f);
    uvTexspace = (uvTexspace - seam) / fwidth(uvTexspace) + seam;
    uvTexspace = clamp(uvTexspace, seam - 0.5f, seam + 0.5f);
    return SampleInputGamma(uvTexspace / gConstants.textureResolution, fragCoord);
}

float4 PSMain(in float4 pos : SV_Position, in float2 uv : TEXCOORD0) : SV_TARGET {
#ifdef PIXEL_ANTIALIASING
    return PixelAntialiasing(uv, pos.xy);
#else
    return SampleInputGamma((uv / gConstants.textureResolution) * gConstants.videoResolution, pos.xy);
#endif
}