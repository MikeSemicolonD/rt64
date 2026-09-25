//
// RT64
//

#include "rt64_interpreter.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <chrono>
#include <atomic>

//#define DUMP_DISPLAY_LISTS

namespace RT64 {
    static FILE *displayListFp = nullptr;

    // F5 DL-walker desync trace (ROGUESQ_DESYNC_TRACE): a ring of the last opcodes
    // dispatched, with each command's dl pointer so the byte-deltas reveal per-op
    // consumed lengths. The F5 GBI calls rt64_f5_desync_dump() when it reads a
    // garbage setColorImage (data word parsed as a command) so we can see WHICH
    // preceding op mis-advanced the cursor (the desync source).
    namespace {
        struct F5RingEntry { uint8_t op; uint32_t w0; uint32_t w1; uint64_t dlptr; };
        F5RingEntry g_f5ring[96] = {};
        uint32_t g_f5ring_pos = 0;
        int g_f5ring_on = -1;
    }
    // ROGUESQ_LOG_WALK_PROFILE: per-opcode timing inside the F5 walk; dumps command count and
    // the hottest opcodes whenever one walk exceeds ~20ms, to attribute the hitch spikes.
    static bool rt64_walk_profile_enabled() {
        static int on = -1;
        if (on < 0) { const char *e = std::getenv("ROGUESQ_LOG_WALK_PROFILE"); on = (e && e[0] && e[0] != '0') ? 1 : 0; }
        return on == 1;
    }
    bool rt64_f5_desync_enabled() {
        if (g_f5ring_on < 0) { const char *e = std::getenv("ROGUESQ_DESYNC_TRACE"); g_f5ring_on = (e && e[0] && e[0] != '0') ? 1 : 0; }
        return g_f5ring_on == 1;
    }
    void rt64_f5_desync_record(uint8_t op, uint32_t w0, uint32_t w1, const void *dl) {
        F5RingEntry &e = g_f5ring[g_f5ring_pos % 96];
        e.op = op; e.w0 = w0; e.w1 = w1; e.dlptr = (uint64_t)(uintptr_t)dl;
        g_f5ring_pos++;
    }
    extern "C" void rt64_f5_desync_dump(const char *why, uint32_t badW1) {
        if (!rt64_f5_desync_enabled()) return;
        static int s_dumps = 0;
        if (s_dumps >= 12) return;
        ++s_dumps;
        std::fprintf(stderr, "[desync #%d] %s badW1=0x%08X — last %d ops (op w0 w1 | bytesConsumed):\n",
                     s_dumps, why, badW1, 24);
        uint32_t start = (g_f5ring_pos >= 24) ? (g_f5ring_pos - 24) : 0;
        for (uint32_t i = start; i < g_f5ring_pos; i++) {
            const F5RingEntry &e = g_f5ring[i % 96];
            int64_t consumed = -1;
            if (i + 1 <= g_f5ring_pos - 1) { const F5RingEntry &n = g_f5ring[(i + 1) % 96]; consumed = (int64_t)(n.dlptr - e.dlptr); }
            std::fprintf(stderr, "  @%08llX op=0x%02X w0=0x%08X w1=0x%08X bytes=%lld\n", (unsigned long long)(0x80000000ull | e.dlptr), e.op, e.w0, e.w1, (long long)consumed);
        }
        std::fflush(stderr);
    }

    // Interpreter

    Interpreter::Interpreter() {
        state = nullptr;
        hleGBI = nullptr;
        extendedFunction = gbiManager.getExtendedFunction();
    }

    void Interpreter::setup(State *state) {
        this->state = state;
    }

    void Interpreter::loadUCodeGBI(uint32_t textAddress, uint32_t dataAddress, bool resetFromTask) {
        if (!resetFromTask) {
            state->flush();
        }

        const uint32_t AddressMask = 0xFFFFF8;
        const uint32_t maskedTextAddress = textAddress & AddressMask;
        const uint32_t maskedDataAddress = dataAddress & AddressMask;
        if ((UCode.textAddress != maskedTextAddress) || (UCode.dataAddress != maskedDataAddress)) {
            hleGBI = gbiManager.getGBIForUCode(state->RDRAM, maskedTextAddress, maskedDataAddress);
            if (hleGBI != nullptr) {
                state->rsp->setGBI(hleGBI);
            }

            UCode.textAddress = maskedTextAddress;
            UCode.dataAddress = maskedDataAddress;
        }

        if (hleGBI != nullptr) {
            GBIReset resetFunction = resetFromTask ? hleGBI->resetFromTask : hleGBI->resetFromLoad;
            if (resetFunction != nullptr) {
                resetFunction(state);
            }
        }
    }

    void Interpreter::processRDPLists(uint32_t dlStartAdddress, DisplayList *dlStart, DisplayList *dlEnd) {
        state->dlCpuProfiler.start();

        // Update the state with the current display list address.
        state->displayListAddress = dlStartAdddress;
        state->displayListCounter++;

        // Check RDRAM if required.
        state->checkRDRAM();

        GBI *rdpGBI = state->rdp->gbi;
        constexpr unsigned int opCodeMask = 0x3F;

        // Run the command interpreter.
        assert(rdpGBI != nullptr);
        DisplayList *dl = dlStart;
        uint8_t opCode;
        GBIFunction func;
        uint32_t cmdLength;
        size_t pendingCommandRemainingBytes = state->rdp->pendingCommandRemainingBytes;

        if (dlStart >= dlEnd) {
            state->dlCpuProfiler.end();
            return;
        }

        if (pendingCommandRemainingBytes != 0) {
            // Copy the remaining command bytes from the current displaylist
            uint32_t toCopy = (uint32_t)std::min(pendingCommandRemainingBytes, (uintptr_t)dlEnd - (uintptr_t)dl);
            memcpy(state->rdp->pendingCommandBuffer.data() + state->rdp->pendingCommandCurrentBytes, dl, toCopy);

            // Modify start to skip the copied bytes
            dl = (DisplayList *)(toCopy + (uintptr_t)dl);

            // Check if we've copied all of the bytes of the command into the buffer
            if (pendingCommandRemainingBytes == toCopy) {
                // All bytes have been copied, so run the completed command
                DisplayList *pendingCommand = (DisplayList *)state->rdp->pendingCommandBuffer.data();
                opCode = (pendingCommand->w0 >> 24) & opCodeMask;
                func = rdpGBI->map[opCode];

                if (func != nullptr) {
                    func(state, &pendingCommand);
                }
                else {
                    RT64_LOG_PRINTF("DL Parser ran into an unknown RDP opCode: %u / 0x%X", opCode, opCode);
                }

                state->rdp->pendingCommandCurrentBytes = 0;
                state->rdp->pendingCommandRemainingBytes = 0;
            }
            // Not all of the bytes were copied, so adjust RDP state accordingly and exit.
            else {
                state->rdp->pendingCommandCurrentBytes += toCopy;
                state->rdp->pendingCommandRemainingBytes -= toCopy;
                state->dlCpuProfiler.end();
                return;
            }
        }

        // Create a dummy pointer and pass that, since displaylist pointer incrementing is handled differently in LLE.
        DisplayList *dummy;
        while ((dl != nullptr) && ((dlEnd == nullptr) || (dl < dlEnd))) {
            opCode = (dl->w0 >> 24) & opCodeMask;

            if ((extendedOpCode != 0) && (opCode == extendedOpCode)) {
                dummy = dl;
                extendedFunction(state, &dl);
                cmdLength = 1;
            }
            else {
                func = rdpGBI->map[opCode];
                cmdLength = state->rdp->commandWordLengths[opCode];

#       ifdef DUMP_DISPLAY_LISTS
                RT64_LOG_PRINTF("0x%08X 0x%08X", dl->w0, dl->w1);
#       endif

                // Check if this command is unfinished and store the partial contents if so.
                if (dl + cmdLength > dlEnd) {
                    uint32_t toCopy = (uint32_t)((uintptr_t)dlEnd - (uintptr_t)dl);
                    memcpy(state->rdp->pendingCommandBuffer.data(), dl, toCopy);
                    state->rdp->pendingCommandCurrentBytes = toCopy;
                    state->rdp->pendingCommandRemainingBytes = cmdLength * sizeof(DisplayList) - toCopy;
                    break;
                }

                if (func != nullptr) {
                    dummy = dl;
                    func(state, &dummy);
                }
                else {
                    RT64_LOG_PRINTF("DL Parser ran into an unknown RDP opCode: %u / 0x%X", opCode, opCode);
                }
            }

            if (dl != nullptr) {
                dl += cmdLength;
            }
        }

        state->dlCpuProfiler.end();
    }

    void Interpreter::processDisplayLists(uint32_t dlStartAdddress, DisplayList *dlStart) {
        assert(hleGBI != nullptr);

        state->dlCpuProfiler.start();

        // Update the state with the current display list address.
        state->displayListAddress = dlStartAdddress;
        state->displayListCounter++;

        // Check RDRAM if required.
        state->checkRDRAM();

        // Run the command interpreter.
        DisplayList *dl = dlStart;
        uint8_t opCode;
        GBIFunction func;
        const bool desyncTrace = rt64_f5_desync_enabled();
        const bool walkProf = rt64_walk_profile_enabled();
        uint64_t prof_us[256] = {}; uint32_t prof_n[256] = {}; uint64_t prof_cmds = 0;
        const auto prof_t0 = walkProf ? std::chrono::high_resolution_clock::now()
                                      : std::chrono::high_resolution_clock::time_point{};
        while (dl != nullptr) {
            opCode = (dl->w0 >> 24);

            if (desyncTrace) rt64_f5_desync_record(opCode, dl->w0, dl->w1, reinterpret_cast<const void *>(reinterpret_cast<const uint8_t *>(dl) - state->RDRAM));   // RDRAM offset, not host pointer

            const auto cmd_t0 = walkProf ? std::chrono::high_resolution_clock::now()
                                         : std::chrono::high_resolution_clock::time_point{};
            if ((extendedOpCode != 0) && (opCode == extendedOpCode)) {
                extendedFunction(state, &dl);
            }
            else {
                func = hleGBI->map[opCode];

#       ifdef DUMP_DISPLAY_LISTS
                RT64_LOG_PRINTF("0x%08X 0x%08X", dl->w0, dl->w1);
#       endif

                if (func != nullptr) {
                    func(state, &dl);
                }
                else {
                    RT64_LOG_PRINTF("DL Parser ran into an unknown opCode (GBI %u): %u / 0x%X", uint32_t(hleGBI->ucode), opCode, opCode);
                }
            }
            if (walkProf) {
                prof_us[opCode] += (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::high_resolution_clock::now() - cmd_t0).count();
                prof_n[opCode]++; prof_cmds++;
            }

            if (dl != nullptr) {
                dl++;
            }
        }

        if (walkProf) {
            const uint64_t total_us = (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::high_resolution_clock::now() - prof_t0).count();
            // Every 300 walks: cumulative per-opcode share, for comparing configurations.
            {
                static uint64_t s_us[256] = {}, s_n[256] = {}, s_total = 0, s_walks = 0;
                for (int o = 0; o < 256; ++o) { s_us[o] += prof_us[o]; s_n[o] += prof_n[o]; }
                s_total += total_us;
                if (++s_walks % 300 == 0) {
                    char buf[512]; int off = snprintf(buf, sizeof buf, "[walksum] walks=%llu avg=%.2fms top:", (unsigned long long)s_walks, s_total / 1000.0 / s_walks);
                    uint64_t tmp[256]; std::memcpy(tmp, s_us, sizeof tmp);
                    for (int k = 0; k < 5; ++k) {
                        int best = -1; uint64_t bestv = 0;
                        for (int o = 0; o < 256; ++o) if (tmp[o] > bestv) { bestv = tmp[o]; best = o; }
                        if (best < 0) break;
                        off += snprintf(buf + off, sizeof buf - off, " 0x%02X=%.2fms/n%.0f", best, bestv / 1000.0 / s_walks, double(s_n[best]) / s_walks);
                        tmp[best] = 0;
                    }
                    fprintf(stderr, "%s\n", buf); fflush(stderr);
                }
            }
            if (total_us > 20000) {
                char buf[512]; int off = snprintf(buf, sizeof buf,
                    "[walkprof] %.1fms cmds=%llu top:", total_us / 1000.0, (unsigned long long)prof_cmds);
                for (int k = 0; k < 4; ++k) {
                    int best = -1; uint64_t bestv = 0;
                    for (int o = 0; o < 256; ++o) if (prof_us[o] > bestv) { bestv = prof_us[o]; best = o; }
                    if (best < 0 || bestv == 0) break;
                    off += snprintf(buf + off, sizeof buf - off, " 0x%02X=%.1fms/n%u", best, bestv / 1000.0, prof_n[best]);
                    prof_us[best] = 0;
                }
                fprintf(stderr, "%s\n", buf); fflush(stderr);
            }
        }

        state->dlCpuProfiler.end();
    }
};
