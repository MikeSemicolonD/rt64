//
// RT64
//

#pragma once

// Wind ripples on water: octaves of tileable gradient noise in map space (512 units a cell), each scrolling one whole tile period every kRippleLoop
// seconds, so the wrapped clock (rs64lights::rippleClock) never jumps. Crossed sine waves beat into a regular egg-crate grid; noise doesn't.
static const float kRippleLoop = 64.0f;
static const int kRipplePeriod = 8;

float2 rs64RippleGradient(int2 p) {
    p = ((p % kRipplePeriod) + kRipplePeriod) % kRipplePeriod;
    uint h = uint(p.x) * 73856093u ^ uint(p.y) * 19349663u;
    h ^= h >> 13;
    h *= 0x5bd1e995u;
    h ^= h >> 15;
    const float a = float(h & 0xFFFFu) * (6.2831853f / 65536.0f);
    return float2(cos(a), sin(a));
}

// Gradient noise and its derivative (quintic fade) at x, in lattice cells.
float3 rs64RippleNoise(float2 x) {
    const int2 i = int2(floor(x));
    const float2 f = x - floor(x);
    const float2 u = f * f * f * (f * (f * 6.0f - 15.0f) + 10.0f);
    const float2 du = 30.0f * f * f * (f * (f - 2.0f) + 1.0f);
    const float2 ga = rs64RippleGradient(i);
    const float2 gb = rs64RippleGradient(i + int2(1, 0));
    const float2 gc = rs64RippleGradient(i + int2(0, 1));
    const float2 gd = rs64RippleGradient(i + int2(1, 1));
    const float va = dot(ga, f);
    const float vb = dot(gb, f - float2(1.0f, 0.0f));
    const float vc = dot(gc, f - float2(0.0f, 1.0f));
    const float vd = dot(gd, f - float2(1.0f, 1.0f));
    const float k = va - vb - vc + vd;
    const float v = va + u.x * (vb - va) + u.y * (vc - va) + u.x * u.y * k;
    const float2 d = ga + u.x * (gb - ga) + u.y * (gc - ga) + u.x * u.y * (ga - gb - gc + gd) + du * (u.yx * k + float2(vb, vc) - va);
    return float3(v, d);
}

// Slope (d height / d map x, z) at map point m, time t. An octave fades out while its cell is under three pixel footprints (map units) wide.
float2 rs64RippleSlope(float2 m, float t, float footprint) {
    // Cell size (map units), rotation (radians), slope amplitude, whole tile periods scrolled per loop along the rotated x (speed near sqrt(cell), as deep water).
    const float4 octaves[7] = {
        float4(560.0f, 0.4f, 0.05f, 1.0f),
        float4(310.0f, 2.1f, 0.04f, 1.0f),
        float4(170.0f, 3.9f, 0.03f, 2.0f),
        float4(95.0f, 5.3f, 0.022f, 3.0f),
        float4(50.0f, 1.2f, 0.018f, 3.0f),
        float4(28.0f, 4.6f, 0.014f, 4.0f),
        float4(15.0f, 2.9f, 0.01f, 6.0f)
    };
    float2 g = float2(0.0f, 0.0f);
    [unroll]
    for (int i = 0; i < 7; i++) {
        const float4 o = octaves[i];
        const float fade = saturate(1.0f - 3.0f * footprint / o.x);
        if (fade <= 0.0f) {
            continue;
        }

        const float c = cos(o.y);
        const float s = sin(o.y);
        const float2 r = float2(c * m.x + s * m.y, -s * m.x + c * m.y) / o.x;
        const float2 d = rs64RippleNoise(r + float2(o.w * float(kRipplePeriod) * t / kRippleLoop, 0.0f)).yz;
        // Back from the rotated frame (transpose).
        g += float2(c * d.x - s * d.y, s * d.x + c * d.y) * (o.z * fade);
    }
    return g;
}

// The rippled surface normal in map space (Y down, as the terrain's), its slope scaled by strength.
float3 rs64RippleNormal(float2 m, float t, float footprint, float strength) {
    const float2 g = rs64RippleSlope(m, t, footprint) * strength;
    return normalize(float3(-g.x, -1.0f, -g.y));
}
