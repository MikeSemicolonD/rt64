#pragma once

#include <atomic>
#include <cstdint>

// Set by the host hook when the game zeroes a new mode's first buffer: its physical address (0 = none) and the last workload id at that moment.
extern std::atomic<uint32_t> g_rs64_cpu_cleared_fb;
extern std::atomic<uint64_t> g_rs64_cpu_cleared_wid;

// Buffer the present-side gate is currently holding black (0 = released); the host menu redirect stays off while it is set.
extern std::atomic<uint32_t> g_rs64_transition_gate_fb;

// Id of the last workload the gfx thread built, published for the host hook.
extern std::atomic<uint64_t> g_rs64_last_workload_id;

// Rogue Squadron mode-change blanking: the game shows a zero-height VI mode, then a CPU-zeroed buffer RT64 never sees.
namespace rs64transition {
    constexpr int kGateCap = 60;

    inline bool isBlankVStart(uint32_t vStartReg) {
        return ((vStartReg >> 16) & 0x3FFu) == (vStartReg & 0x3FFu);
    }

    // The VI origin sits up to ~2 rows past the buffer base.
    inline bool viShowsBuffer(uint32_t viAddr, uint32_t fb) {
        viAddr &= 0x00FFFFFFu;
        fb &= 0x00FFFFFFu;
        return (viAddr >= fb) && (viAddr - fb < 0x1000u);
    }

    struct Gate {
        uint32_t fb = 0;
        uint64_t armWorkloadId = 0;
        int presents = 0;

        // armWorkloadId is the last workload built before the clear; only later workloads count as drawing the new buffer.
        void arm(uint32_t clearedFb, uint64_t workloadId) {
            fb = clearedFb & 0x00FFFFFFu;
            armWorkloadId = workloadId;
            presents = 0;
        }

        uint32_t armedFb() const {
            return fb;
        }

        // True while the VI shows the zeroed buffer and no workload after the clear has drawn into it.
        // drawnWorkloadId is the id of a workload that drew into the armed buffer, or 0.
        bool shouldBlack(uint32_t viAddr, uint64_t drawnWorkloadId) {
            if (fb == 0) {
                return false;
            }

            const bool drawnAfterClear = (drawnWorkloadId > armWorkloadId);
            if (drawnAfterClear || !viShowsBuffer(viAddr, fb) || (presents >= kGateCap)) {
                fb = 0;
                return false;
            }

            ++presents;
            return true;
        }
    };
}
