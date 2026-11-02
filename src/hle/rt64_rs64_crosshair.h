#pragma once

#include <cstdint>
#include <cmath>

// Rogue Squadron's crosshair rings are texrects placed from the HUD struct (D_8010CA30); matching a texrect to its ring element lets it draw as an interpolated quad.
namespace rs64xhair {
    constexpr uint32_t kHudBase = 0x8010CA30u;
    // Outer ring, inner ring, Y-wing bomb sight.
    constexpr int kElementCount = 3;
    constexpr uint32_t kElementOffsets[kElementCount] = { 0x28u, 0x58u, 0x88u };
    constexpr uint32_t kFlagsOffset = 0x14u;
    constexpr uint32_t kPosOffset = 0x18u;
    // A ring's texrect upper-left is the element position plus this origin.
    constexpr float kOriginX = 256.0f;
    constexpr float kOriginY = 224.0f;
    constexpr float kTolerance = 0.75f;

    struct Element { float x, y; uint32_t flags; };
    struct Rect { float ulx, uly, lrx, lry; };
    struct TexSize { uint32_t w, h; };

    // Tile size of each element's texture; the elements stay flagged outside the flight HUD (the results-screen iris).
    constexpr TexSize kElementTex[kElementCount] = { { 32, 32 }, { 16, 16 }, { 32, 20 } };

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

    // Index of the element (0 outer, 1 inner, 2 bomb sight) this texrect draws, or -1.
    inline int ringOf(const Element els[kElementCount], const Rect &r, TexSize tex, int32_t dsdx, int32_t dtdy, const Rect &clip) {
        const float w = drawnSize(tex.w, dsdx), h = drawnSize(tex.h, dtdy);
        // The bomb sight is a 20x20 image in a 32x20 tile and is drawn tile-height wide.
        const float wSquare = drawnSize(tex.h, dsdx);
        for (int i = 0; i < kElementCount; i++) {
            if ((tex.w != kElementTex[i].w) || (tex.h != kElementTex[i].h))
            {
                continue;
            }

            if (matches(els[i], r, w, h, clip) || matches(els[i], r, wSquare, h, clip)) {
                return i;
            }
        }
        return -1;
    }
}
