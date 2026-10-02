#pragma once

#include <cstdint>

// Tow cable segments are ring slots that change role as the cable pays out, so each segment gets an id by its position from the ship end, not by its slot.
namespace rs64cable {
    constexpr uint32_t kSlotStride = 0xF8;
    constexpr uint32_t kInstanceOffset = 0x40;
    // The scene node traverseSceneGraphRecursive maps for a segment is 0x0C into its mesh instance.
    constexpr uint32_t kSceneNodeOffset = 0x0C;
    // The head index is a u8; vanilla uses 60 slots, the long-tow-cable mod 160.
    constexpr uint32_t kMaxSlots = 255;

    inline uint32_t sceneNodeOf(uint32_t instance) {
        return instance + kSceneNodeOffset;
    }

    // State block at 0x8010B6E8: phase (0 off, 1 fired, 2 paying out, 3 released, 4 dropping), head slot, segment count.
    struct CableState {
        uint8_t phase = 0;
        uint8_t head = 0;
        uint8_t count = 0;

        static CableState fromWord(uint32_t word) {
            return CableState{ uint8_t(word >> 24), uint8_t(word >> 16), uint8_t(word >> 8) };
        }
    };

    // Ring slot of a mesh instance inside the pool, or -1.
    inline int slotIndex(uint32_t pool, uint32_t instance) {
        if ((pool == 0) || (instance < pool + kInstanceOffset)) {
            return -1;
        }
        const uint32_t rel = instance - pool - kInstanceOffset;
        if (((rel % kSlotStride) != 0) || ((rel / kSlotStride) >= kMaxSlots)) {
            return -1;
        }
        return int(rel / kSlotStride);
    }

    // Whether the renderer draws this slot: (head - i) mod N for i < count. A ring that has wrapped is full (count == N).
    inline bool isDrawnSlot(int index, CableState s) {
        if ((s.phase == 0) || (s.count == 0) || (index < 0)) {
            return false;
        }
        if (uint32_t(s.head) + 1 >= s.count) {
            return (index <= s.head) && (uint32_t(index) + s.count > s.head);
        }
        return index < s.count;
    }

    struct CableRoles {
        uint32_t nodes[kMaxSlots] = {};
        uint32_t n = 0;
        uint32_t generation = 0;
        CableState last;
        // Frames since the last cable submit; 0 = a submit already happened this frame.
        uint32_t framesSinceSubmit = 2;

        void onSubmit(uint32_t node, CableState s) {
            if (framesSinceSubmit != 0) {
                const bool gap = framesSinceSubmit > 1;
                const bool refired = (last.phase >= 3) && (s.phase <= 2);
                if (gap || refired || (s.count < last.count)) {
                    generation++;
                }
                n = 0;
                framesSinceSubmit = 0;
            }
            last = s;
            if (n < kMaxSlots) {
                nodes[n++] = node;
            }
        }

        void onFrame() {
            if (framesSinceSubmit < 2) {
                framesSinceSubmit++;
            }
        }

        uint32_t aliasFor(uint32_t node) const {
            if (framesSinceSubmit > 1) {
                return 0;
            }
            for (uint32_t i = 0; i < n; ++i) {
                if (nodes[i] == node) {
                    return 0x10000000u | ((generation & 0xFFFFFu) << 8) | i;
                }
            }
            return 0;
        }
    };
}
