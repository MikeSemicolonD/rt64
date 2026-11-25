#pragma once

// RS64 fog: the game's fog ramp, the frame's fog pick and the fog shaft samples.

#include "rt64_rs64_math.h"

namespace rs64lights {
    // Fog alpha from the linear fog value x (depth * mul + offset, over 255). With tailEnd past the knee the fog follows x to the knee, then thins
    // to full at x = tailEnd, for a fog-limited level drawn further than stock; 0 keeps the stock ramp. Mirrored by rs64FogCurve (rt64_rsp_fog.h).
    // The knee sits near full fog so terrain past the stock fog shows only faintly, not as dark silhouettes.
    inline float fogCurve(float x, float tailEnd) {
        const float knee = 0.97f;
        if ((tailEnd > knee) && (x > knee)) {
            x = knee + (1.0f - knee) * (x - knee) / (tailEnd - knee);
        }
        return std::clamp(x, 0.0f, 1.0f);
    }

    // RSPProcessCS vertex fog at a camera-space point; clip.w == 0 (the eye) takes the max(z, 0) == 0 limit. Mirrored by RS64FogShaftsPS.
    inline float fogFactor(const Mat4& viewProj, const float q[3], float mul, float offset, float tailEnd = 0.0f) {
        const float v[4] = { q[0], q[1], q[2], 1.0f };
        float c[4];
        transform(v, viewProj, c);
        const float a = (c[3] > 1e-6f) ? ((std::max(c[2], 0.0f) / c[3]) * mul + offset) : offset;
        return fogCurve(a / 255.0f, tailEnd);
    }

    // March fractions along camera->receiver: step i ends at (i + 1 - jitter) / n, the last exactly at the receiver.
    inline void fogSteps(uint32_t n, float jitter, float* t) {
        for (uint32_t i = 0; i < n; ++i) {
            t[i] = (i + 1 == n) ? 1.0f : (float(i) + 1.0f - jitter) / float(n);
        }
    }

    // Sun-visibility sample along camera->receiver, pulled toward the camera by bias so the receiver never blocks its own last step. Mirrored by RS64FogShaftsPS.
    inline float fogVisT(float t, float bias) {
        return t * (1.0f - bias);
    }

    // Depth at or past the sky pin (0x7FBE/0x7FBF of 32767) or the clear value; the game never fogs the sky. Half a tick under 0x7FBE.
    inline float fogSkyDepth() {
        return (float(0x7FBE) - 0.5f) / 32767.0f;
    }

    // Interleaved gradient noise (Jimenez): fixed per pixel, so a still view does not shimmer.
    inline float interleavedNoise(float x, float y) {
        const float a = 0.06711056f * x + 0.00583715f * y;
        const float b = 52.9829189f * (a - std::floor(a));
        return b - std::floor(b);
    }

    // Black fog or zero strength adds nothing, so the pass is skipped instead of tracing for zero.
    inline bool fogShaftsVisible(const float color[3], float strength) {
        return std::max(color[0], std::max(color[1], color[2])) * strength >= (1.0f / 255.0f);
    }

    struct FogParams {
        bool valid = false;
        uint32_t tris = 0;
        float mul = 0.0f;
        float offset = 0.0f;
        float tailEnd = 0.0f;
        float color[3] = {};
    };

    // The main view's largest fogged draw sets the fog; ties keep the first.
    inline void considerFogCall(FogParams& best, uint32_t tris, float mul, float offset, const float color[3], float tailEnd = 0.0f) {
        if ((tris == 0) || (best.valid && (tris <= best.tris))) {
            return;
        }
        best.valid = true;
        best.tris = tris;
        best.mul = mul;
        best.offset = offset;
        best.tailEnd = tailEnd;
        best.color[0] = color[0];
        best.color[1] = color[1];
        best.color[2] = color[2];
    }

    struct FogCache {
        uint64_t workloadId = UINT64_MAX;
        FogParams params;
    };

    // One fog per workload across its framebuffer pairs; a pair without a fogged main draw reuses the pick.
    inline const FogParams& workloadFog(FogCache& c, uint64_t workloadId, const FogParams& pairPick) {
        if (c.workloadId != workloadId) {
            c.workloadId = workloadId;
            c.params = FogParams{};
        }
        if (pairPick.valid && (!c.params.valid || (pairPick.tris > c.params.tris))) {
            c.params = pairPick;
        }
        return c.params;
    }

    // Insertion of the k smallest distances, ascending; mirrored by RS64LightsPS.
    inline uint32_t nearestLights(const float* dist, uint32_t count, uint32_t k, uint32_t* out) {
        uint32_t n = 0;
        for (uint32_t i = 0; i < count; ++i) {
            uint32_t pos = n;
            while ((pos > 0) && (dist[out[pos - 1]] > dist[i])) {
                --pos;
            }
            if (pos >= k) {
                continue;
            }
            const uint32_t last = std::min(n, k - 1);
            for (uint32_t j = last; j > pos; --j) {
                out[j] = out[j - 1];
            }
            out[pos] = i;
            n = std::min(n + 1, k);
        }
        return n;
    }
}
