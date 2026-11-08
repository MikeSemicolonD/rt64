#pragma once

#include <atomic>
#include <cstdint>

// HUD placement in widescreen, 0 = in the original 4:3 area, 1 = at the screen edges ("hudPlacement" in roguesq_video.json). Set by the host,
// read by the workload queue as the extended-GBI origin percentage.
extern std::atomic<float> g_rs64_hud_placement;

// Rogue Squadron's 2D HUD elements anchored to a screen side. A host hook puts a marker command at the start of each element's display-list chunk;
// the F5 profile then draws that element's texrects with left or right extended-GBI origins, so they slide toward the edge without stretching.
namespace rs64hud {
    // Unused by the game's ucode; w0 low byte = Side.
    constexpr uint8_t kMarkerOpcode = 0x60;

    enum Side : uint8_t { None = 0, Left = 1, Right = 2 };

    // drawSubtitleText element flag set on font text (subtitles, messages, menus), which stays where the game puts it.
    constexpr uint32_t kTextFlag = 0x20u;

    // The crosshair rings and bomb sight are elements of the double-buffered HUD struct D_8010CA30[2] (0x278 each); they follow the aim anywhere on screen.
    constexpr uint32_t kAimElementsBegin = 0x8010CA30u;
    constexpr uint32_t kAimElementsEnd = kAimElementsBegin + 2u * 0x278u;

    // A rect as tall as the scissor (the mission-end wipe panel) is a screen transition, never an edge-anchored element. All values in quarter pixels.
    inline bool spansHeight(int32_t uly, int32_t lry, int32_t scissorUly, int32_t scissorLry) {
        return (uly <= scissorUly + 4) && (lry + 4 >= scissorLry);
    }

    // item is the element's address, x its upper-left from the screen centre in native pixels, screenWidth the game's view width (512). The middle third stays put.
    inline Side classify(uint32_t item, float x, uint32_t flags, float screenWidth) {
        if ((flags & kTextFlag) || ((item >= kAimElementsBegin) && (item < kAimElementsEnd))) {
            return None;
        }

        const float edge = screenWidth / 6.0f;
        return (x < -edge) ? Left : (x > edge) ? Right : None;
    }

    // Native pixels an edge-anchored element moves outward: how far the extended origin sits from the centre-anchored position.
    inline float shiftNative(float fbWidth, float placement, float aspectScale) {
        if (aspectScale <= 1.0f || placement <= 0.0f) {
            return 0.0f;
        }

        return fbWidth * 0.5f * placement * (aspectScale - 1.0f);
    }

    inline uint32_t markerW0(Side side) {
        return (uint32_t(kMarkerOpcode) << 24) | uint32_t(side);
    }
}
