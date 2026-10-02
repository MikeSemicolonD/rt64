#pragma once

#include <cstdint>
#include <cmath>

// Rogue Squadron's crosshair rings are texrects placed from the HUD struct (D_8010CA30); matching a texrect to its ring element lets it draw as an interpolated quad.
namespace rs64xhair {
    constexpr uint32_t kHudBase = 0x8010CA30u;
    constexpr uint32_t kElementOffsets[2] = { 0x28u, 0x58u };
    constexpr uint32_t kFlagsOffset = 0x14u;
    constexpr uint32_t kPosOffset = 0x18u;
    // A ring's texrect upper-left is the element position plus this origin.
    constexpr float kOriginX = 256.0f;
    constexpr float kOriginY = 224.0f;
    constexpr float kTolerance = 0.75f;

    struct Element { float x, y; uint32_t flags; };
    struct Rect { float ulx, uly, lrx, lry; };
    struct TexSize { uint32_t w, h; };

    // On-screen length of a texture span drawn at a 5.10 step.
    inline float drawnSize(uint32_t texels, int32_t step) {
        return (step > 0) ? (float)texels * 1024.0f / (float)step : 0.0f;
    }

    // An edge may also sit at the clip edge when the game clamped the rect there.
    inline bool axisMatches(float ul, float lr, float expUl, float expLr, float clipUl, float clipLr) {
        const bool ulOk = std::fabs(ul - expUl) <= kTolerance;
        const bool lrOk = std::fabs(lr - expLr) <= kTolerance;
        const bool ulClamped = (ul <= clipUl + kTolerance) && (expUl < ul);
        const bool lrClamped = (lr >= clipLr - kTolerance) && (expLr > lr);
        return (ulOk || lrOk) && (ulOk || ulClamped) && (lrOk || lrClamped);
    }

    inline bool matches(const Element &e, const Rect &r, float w, float h, const Rect &clip) {
        if (((e.flags & 1u) == 0) || (w <= 0.0f) || (h <= 0.0f)) {
            return false;
        }
        const float ex = e.x + kOriginX, ey = e.y + kOriginY;
        return axisMatches(r.ulx, r.lrx, ex, ex + w, clip.ulx, clip.lrx) && axisMatches(r.uly, r.lry, ey, ey + h, clip.uly, clip.lry);
    }

    // Index of the ring element (0 outer, 1 inner) this texrect draws, or -1.
    inline int ringOf(const Element els[2], const Rect &r, TexSize tex, int32_t dsdx, int32_t dtdy, const Rect &clip) {
        const float w = drawnSize(tex.w, dsdx), h = drawnSize(tex.h, dtdy);
        for (int i = 0; i < 2; i++) {
            if (matches(els[i], r, w, h, clip)) {
                return i;
            }
        }
        return -1;
    }
}
