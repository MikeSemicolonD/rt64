// RT64 - Puts what was drawn after the trace into the water's horizon mirror

#include "shared/rt64_rs64_shadow_blur_cb.h"

ConstantBuffer<RS64ShadowBlurCB> gBlur : register(b0);
Texture2D<float> gDepth : register(t1);
Texture2D<float4> gMirror : register(t3);
Texture2D<float4> gLate : register(t4);

// Horizon mirror, drawn after the view's last 3D draw: water whose traced reflection missed showed the sky found on screen in its reflected direction
// (the trace's mirror target). That spot is looked up again in the finished frame, and what is drawn there now (a craft, its explosion, smoke drawn
// after the trace) takes the sky's place: the water gets weight * (late - sky), as a far observer would see it. Their true mirror image can lie below
// the screen (an explosion near the camera's height), which the traced reflection cannot show. Two draws carry the signed difference through a UNORM
// target: this one adds its rising part, the RS64_MIRROR_NEG one subtracts its falling part.
float4 PSMain(in float4 pos : SV_Position, in float2 uv : TEXCOORD0) : SV_TARGET {
    const int2 px = int2(pos.xy);
    const float4 mi = gMirror.Load(int3(px, 0));
    if (mi.z <= 0.0f) {
        return float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    // Kept a little inside the edges, as rs64ScreenAt does: the outermost columns can hold a black or white sliver.
    const int2 margin = max(int2(gBlur.viewport.zw / 160.0f), int2(2, 2));
    const int2 lo = int2(gBlur.viewport.xy) + margin;
    const int2 hi = int2(gBlur.viewport.xy + gBlur.viewport.zw) - 1 - margin;
    const int2 m = clamp(int2(gBlur.viewport.xy + float2(mi.x * 0.5f + 0.5f, 0.5f - mi.y * 0.5f) * gBlur.viewport.zw), lo, hi);

    // A draw much nearer the camera than the camera is above the water (the player's own craft) would mirror close to the horizon, far from where it
    // could be reflected: those keep the sky.
    const float dm = gDepth.Load(int3(m, 0));
    if (dm < 0.9980f) {
        const float2 rel = (pos.xy - gBlur.viewport.xy) / gBlur.viewport.zw;
        const float4 w4 = mul(gBlur.viewFromScreen, float4(rel.x * 2.0f - 1.0f, 1.0f - rel.y * 2.0f, gDepth.Load(int3(px, 0)), 1.0f));
        const float4 q4 = mul(gBlur.viewFromScreen, float4(mi.xy, dm, 1.0f));
        const float height = abs(dot(w4.xyz / w4.w, gBlur.mirrorUp.xyz));
        if (length(q4.xyz / q4.w) < 0.5f * height) {
            return float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }

    const uint s = uint(mi.w);
    const float3 sky = float3(s & 0xFFu, (s >> 8) & 0xFFu, (s >> 16) & 0xFFu) / 255.0f;
    const float3 d = (gLate.Load(int3(m, 0)).rgb - sky) * mi.z * gBlur.mirrorUp.w;
#ifdef RS64_MIRROR_NEG
    return float4(max(-d, 0.0f), 0.0f);
#else
    return float4(max(d, 0.0f), 0.0f);
#endif
}
