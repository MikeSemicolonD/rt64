//
// RT64
//

#include "rt64_framebuffer_storage.h"

#include <cassert>
#include <cstring>

namespace RT64 {
    // FramebufferStorage

    FramebufferStorage::FramebufferStorage() {
        reset();
    }

    void FramebufferStorage::reset() {
        // A frame's copies are dead once it resets, so a buffer that a heavier scene (an earlier level) grew is given back after a stretch of lighter frames.
        constexpr uint32_t ShrinkAfterResets = 600;
        recentPeak = std::max(recentPeak, rdramUsed);
        if (++resetsSincePeak >= ShrinkAfterResets) {
            if (rdramData.size() > size_t(recentPeak) * 2) {
                std::vector<uint8_t>((size_t(recentPeak) * 3) / 2, 0).swap(rdramData);
            }

            recentPeak = 0;
            resetsSincePeak = 0;
        }

        rdramUsed = 0;
        handleVector.clear();
    }

    void FramebufferStorage::store(uint32_t fbPairIndex, uint32_t address, const uint8_t *data, uint32_t size) {
        uint32_t dstIndex = rdramUsed;
        rdramUsed += size;
        if (rdramUsed > rdramData.size()) {
            const uint32_t newSize = (rdramUsed * 3) / 2;
            rdramData.resize(newSize, 0);
        }

        memcpy(rdramData.data() + dstIndex, data, size);

        Handle handle;
        handle.fbPairIndex = fbPairIndex;
        handle.address = address;
        handle.rdramIndex = dstIndex;
        handle.size = size;
        handleVector.emplace_back(handle);
    }

    const FramebufferStorage::Handle *FramebufferStorage::get(uint32_t maxFbPairIndex, uint32_t address) const {
        const FramebufferStorage::Handle *maxHandle = nullptr;
        for (const auto &handle : handleVector) {
            if (handle.fbPairIndex > maxFbPairIndex) {
                continue;
            }

            if (handle.address != address) {
                continue;
            }

            maxHandle = &handle;
        }

        return maxHandle;
    }

    const uint8_t *FramebufferStorage::getRDRAM(const Handle &handle) const {
        assert((handle.rdramIndex + handle.size) <= rdramData.size());
        return rdramData.data() + handle.rdramIndex;
    }
};