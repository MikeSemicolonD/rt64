//
// RT64
//

#include "rt64_framebuffer_storage.h"

#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {
    static int rt64_alloc_log_enabled() {
        static int s = -1;
        if (s < 0) {
#ifdef _MSC_VER
#  pragma warning(push)
#  pragma warning(disable:4996)
#endif
#if defined(__clang__)
#  pragma clang diagnostic push
#  pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
            const char *e = std::getenv("ROGUESQ_LOG_RT64_ALLOC");
#if defined(__clang__)
#  pragma clang diagnostic pop
#endif
#ifdef _MSC_VER
#  pragma warning(pop)
#endif
            s = (e && e[0] == '1') ? 1 : 0;
        }
        return s;
    }
    static uint64_t rt64_alloc_now_ms() {
        using namespace std::chrono;
        return (uint64_t)duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
    }
}

namespace RT64 {
    // FramebufferStorage

    FramebufferStorage::FramebufferStorage() {
        reset();
    }

    void FramebufferStorage::reset() {
        rdramUsed = 0;
        handleVector.clear();
    }

    void FramebufferStorage::store(uint32_t fbPairIndex, uint32_t address, const uint8_t *data, uint32_t size) {
        uint32_t dstIndex = rdramUsed;
        rdramUsed += size;
        if (rdramUsed > rdramData.size()) {
            const size_t oldCap = rdramData.size();
            const uint32_t newSize = (rdramUsed * 3) / 2;
            rdramData.resize(newSize, 0);
            if (rt64_alloc_log_enabled() && (newSize - oldCap) >= (1u << 20)) {
                fprintf(stderr, "[rt64-alloc] rdramData: oldCap=%zu newCap=%u dN=%u rdramUsed=%u fbPairIndex=%u addr=0x%08X size=%u ms=%llu\n",
                    oldCap, (unsigned)newSize, (unsigned)(newSize - (uint32_t)oldCap),
                    (unsigned)rdramUsed, (unsigned)fbPairIndex, (unsigned)address, (unsigned)size,
                    (unsigned long long)rt64_alloc_now_ms());
                fflush(stderr);
            }
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