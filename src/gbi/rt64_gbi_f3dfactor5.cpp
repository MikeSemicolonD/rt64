//
// RT64
//
// HLE GBI module for Factor 5's custom RSP microcode (Rogue Squadron 64,
// Battle for Naboo, Indiana Jones and the Infernal Machine).
//
// Inherits the F3DEX dispatch table and overrides only the opcodes Factor 5
// reuses for its own purposes. The module exists in RT64 (rather than as a
// game-side patch) for the same reason the other gbi_*.cpp variants do —
// each ucode family lives in one place so RT64 owns the "what does opcode
// 0xB5 mean for ucode X" decision.
//
// Recovered + cleaned from MikeSemicolonD/rt64@bbf45b7 (2026-05-03), which
// bundled this with extensive runtime diagnostics. Diagnostic-only code
// (RDRAM/DL history capture for offline RE) lives separately and is not
// required for runtime correctness.
//

#include "rt64_gbi_f3dfactor5.h"

#include "hle/rt64_state.h"
#include "hle/rt64_rdp.h"

#include "rt64_gbi_f3dex.h"
#include "rt64_gbi_f3d.h"
#include "rt64_gbi_rdp.h"

#include "shared/rt64_f3d_defines.h"

#include <cstdio>
#include <cstdlib>
#include <unordered_map>
#include <unordered_set>

namespace RT64 {
    namespace GBI_F3DFACTOR5 {
        // ROGUESQ_LOG_GBI defaults to OFF (logs spam stderr at thousands of
        // lines/sec, locks up the ImGui inspector after a few seconds).
        // Set ROGUESQ_LOG_GBI=1 to re-enable the per-handler diagnostic logs.
        static bool gbi_log_enabled() {
            static bool s_enabled = []() {
                const char* v = std::getenv("ROGUESQ_LOG_GBI");
                return v && v[0] && v[0] != '0';
            }();
            return s_enabled;
        }
    }
}

namespace RT64 {
    namespace GBI_F3DFACTOR5 {

        // op 0x80: Factor 5 chunk header (metadata, not control flow).
        // Payload: w0=next_chunk_addr, w1=prev_chunk_addr (back-pointer).
        // Chunks are 0x108-byte blocks packed contiguously; the parent DL
        // walks the chain via standard G_DL (op 0x06). Header is a no-op
        // for HLE.
        //
        // CAUTION: tried calling state->fullSync() here as a batch flush —
        // crashed with "vector subscript out of range" after 3 invocations.
        // fullSync from inside a mid-DL handler is unsafe because the
        // workload-cursor advance leaves indexed structures in a transitional
        // state. Don't reintroduce without auditing the workload-indexed call
        // paths first.
        void op_noop(State *state, DisplayList **dl) {
            // no-op
        }

        // op 0xB5: Factor 5 chunk/DL terminator. Each 0x108-byte chunk ends
        // with op 0xB5; treating as no-op + relying on op 0xB8 (standard F3D
        // G_ENDDL) for real return is what unblocked text rendering in the
        // 2026-05-03 fork. RDRAM-bounds exit in the interpreter loop catches
        // chunk runaways.

        // op 0xB4 / op 0xBF: 16-byte commands (cmd word + 8-byte payload).
        // Drift detector showed every drift-into-ASCII transition was
        // preceded by op_B4 / op_BF dispatched at standard 8-byte stride.
        // Consume the extra word so the dispatch loop's dl++ aligns to the
        // next real command.
        void op_consume16(State *state, DisplayList **dl) {
            (*dl)++;  // skip the 8-byte payload
        }

        // op 0x06 strict G_DL filter. Standard F3D G_DL has w0=0x06000000
        // (opcode + branch flag at bit 16, rest zero). Factor 5 emits
        // commands sharing first byte 0x06 but packing data into w0's low
        // 24 bits (e.g. w0=0x060A07C0). Treating those as G_DLs lets runDl
        // dispatch w1 as a target address — corrupts the interpreter state.
        // Filter: only delegate to F3D::runDl if w0's payload bits (excluding
        // branch flag at bit 16) are zero.
        void op_06_strict_dl(State *state, DisplayList **dl) {
            const uint32_t w0Payload = (*dl)->w0 & 0x00FEFFFF;
            static int s_pass = 0, s_skip = 0;
            if (w0Payload == 0) {
                if (gbi_log_enabled() && (++s_pass <= 8)) {
                    std::fprintf(stderr,
                        "[gbi-f5] G_DL pass #%d w0=0x%08X w1=0x%08X (target)\n",
                        s_pass, (*dl)->w0, (*dl)->w1);
                    std::fflush(stderr);
                }
                GBI_F3D::runDl(state, dl);
            }
            else {
                if (gbi_log_enabled() && (++s_skip <= 8)) {
                    std::fprintf(stderr,
                        "[gbi-f5] G_DL skip #%d w0=0x%08X w1=0x%08X (opcode reuse)\n",
                        s_skip, (*dl)->w0, (*dl)->w1);
                    std::fflush(stderr);
                }
            }
        }

        // op 0xFF (G_SETCIMG) filter. Factor 5 emits one G_SETCIMG with a
        // bogus payload (w0=0xFFF00F0F, w1=0) immediately before the overlay
        // TEXRECT batch. That zeroes RDP::colorImage.address; subsequent
        // TEXRECTs render onto a null target. Real SETCIMGs always carry a
        // non-zero w1.
        //
        // Also reject when fmt field (w0 bits 23-21) is invalid (>4). Standard
        // G_IM_FMT values: RGBA=0, YUV=1, CI=2, IA=3, I=4. fmt 5,6,7 only
        // appear when garbage (e.g. RGBA8888 pixel = 0xFFFFFFFF) is parsed
        // as a SETCIMG cmd. Without this gate, downstream allocators get fed
        // bogus fmt+siz+width and crash with zero-size buffer asserts in
        // D3D12MemoryAllocator.
        // Permissive moveMem (op 0x03). F3D's default asserts on any subcode
        // outside the standard set (viewport/lookat/L0-L7/matrix1). Factor 5's
        // ucode emits subcodes the stock F3D handler does not know; rather
        // than crashing, log each unknown subcode once via the env-gated
        // ROGUESQ_LOG_GBI=1 channel and treat as a no-op so the dispatch loop
        // keeps moving.
        void moveMem_permissive(State *state, DisplayList **dl) {
            const uint32_t subcode = (*dl)->p0(16, 8);
            switch (subcode) {
                case F3D_G_MV_VIEWPORT:
                case F3D_G_MV_MATRIX_1:
                case F3D_G_MV_L0: case F3D_G_MV_L1: case F3D_G_MV_L2:
                case F3D_G_MV_L3: case F3D_G_MV_L4: case F3D_G_MV_L5:
                case F3D_G_MV_L6: case F3D_G_MV_L7:
                case F3D_G_MV_LOOKATX: case F3D_G_MV_LOOKATY:
                    GBI_F3D::moveMem(state, dl);
                    return;
                default: {
                    static std::unordered_set<uint32_t> s_seen;
                    if (gbi_log_enabled() && s_seen.insert(subcode).second) {
                        std::fprintf(stderr,
                            "[gbi-f5] moveMem unknown subcode=0x%02X w0=0x%08X w1=0x%08X (no-op)\n",
                            subcode, (*dl)->w0, (*dl)->w1);
                        std::fflush(stderr);
                    }
                    return;
                }
            }
        }

        // setOtherMode + setScissor logging. Pass-through. We want to see
        // what cycle/blend/render-mode state Factor 5 sets up before
        // texrects, plus what scissor rectangle is active.
        void setOtherModeH_logged(State *state, DisplayList **dl) {
            static int s_count = 0;
            if (gbi_log_enabled() && (++s_count <= 4)) {
                std::fprintf(stderr,
                    "[gbi-f5] setOtherModeH #%d shift=%u length=%u w1=0x%08X\n",
                    s_count, (*dl)->p0(8, 8), (*dl)->p0(0, 8), (*dl)->w1);
                std::fflush(stderr);
            }
            GBI_F3D::setOtherModeH(state, dl);
        }

        void setOtherModeL_logged(State *state, DisplayList **dl) {
            static int s_count = 0;
            if (gbi_log_enabled() && (++s_count <= 4)) {
                std::fprintf(stderr,
                    "[gbi-f5] setOtherModeL #%d shift=%u length=%u w1=0x%08X\n",
                    s_count, (*dl)->p0(8, 8), (*dl)->p0(0, 8), (*dl)->w1);
                std::fflush(stderr);
            }
            GBI_F3D::setOtherModeL(state, dl);
        }

        void setRDPOtherMode_logged(State *state, DisplayList **dl) {
            static int s_count = 0;
            if (gbi_log_enabled() && (++s_count <= 4)) {
                std::fprintf(stderr,
                    "[gbi-f5] setRDPOtherMode #%d w0=0x%08X w1=0x%08X\n",
                    s_count, (*dl)->w0, (*dl)->w1);
                std::fflush(stderr);
            }
            GBI_RDP::setOtherMode(state, dl);
        }

        void setScissor_logged(State *state, DisplayList **dl) {
            static int s_count = 0;
            if (gbi_log_enabled() && (++s_count <= 4)) {
                std::fprintf(stderr,
                    "[gbi-f5] setScissor #%d ulx=%u uly=%u lrx=%u lry=%u mode=%u\n",
                    s_count,
                    (*dl)->p0(12, 12), (*dl)->p0(0, 12),
                    (*dl)->p1(12, 12), (*dl)->p1(0, 12),
                    (*dl)->p1(24, 2));
                std::fflush(stderr);
            }
            GBI_RDP::setScissor(state, dl);
        }

        // setTile + setCombine logging. Pure pass-through; we just want to
        // see what tile setup and combiner configs Factor 5 is establishing
        // before the texrects fire on tile 0.
        void setTile_logged(State *state, DisplayList **dl) {
            static int s_count = 0;
            if (gbi_log_enabled() && (++s_count <= 12)) {
                const uint8_t tile = (*dl)->p1(24, 3);
                const uint8_t fmt = (*dl)->p0(21, 3);
                const uint8_t siz = (*dl)->p0(19, 2);
                const uint16_t line = (*dl)->p0(9, 9);
                const uint16_t tmem = (*dl)->p0(0, 9);
                const uint8_t palette = (*dl)->p1(20, 4);
                std::fprintf(stderr,
                    "[gbi-f5] setTile #%d tile=%u fmt=%u siz=%u line=%u tmem=%u palette=%u\n",
                    s_count, tile, fmt, siz, line, tmem, palette);
                std::fflush(stderr);
            }
            GBI_RDP::setTile(state, dl);
        }

        void setCombine_logged(State *state, DisplayList **dl) {
            static int s_count = 0;
            if (gbi_log_enabled() && (++s_count <= 8)) {
                std::fprintf(stderr,
                    "[gbi-f5] setCombine #%d w0=0x%08X w1=0x%08X\n",
                    s_count, (*dl)->w0, (*dl)->w1);
                std::fflush(stderr);
            }
            GBI_RDP::setCombine(state, dl);
        }

        // Texrect guard. RT64 asserts in State::loadDrawState (rt64_state.cpp:262)
        // when cycleType==G_CYC_COPY but the bound tile is undefined (line==0).
        // The post-assert path treats it as `valid=false` and falls through, so
        // the safe move is to skip the draw entirely when this combination
        // would trip the assert.
        static bool texrect_copy_undefined_tile(State *state, DisplayList **dl) {
            const uint8_t tile = (*dl)[0].p1(24, 3);
            if (state->rdp->otherMode.cycleType() == G_CYC_COPY &&
                state->rdp->tiles[tile & 7].line == 0) {
                static int s_skipped = 0;
                if (gbi_log_enabled() && (++s_skipped <= 4)) {
                    std::fprintf(stderr,
                        "[gbi-f5] skip texrect: copy mode w/ undefined tile=%u\n",
                        tile);
                    std::fflush(stderr);
                }
                return true;
            }
            return false;
        }

        // fullSync (op 0xE9) tracker — Factor 5 may emit fullSync mid-DL,
        // which would advance the workload cursor and be why our post-DL
        // probe reads an empty workload (the populated one already moved).
        void fullSync_logged(State *state, DisplayList **dl) {
            static int s_count = 0;
            if (gbi_log_enabled() && (++s_count <= 8)) {
                std::fprintf(stderr,
                    "[gbi-f5] fullSync #%d (dl-emitted) w0=0x%08X w1=0x%08X\n",
                    s_count, (*dl)->w0, (*dl)->w1);
                std::fflush(stderr);
            }
            GBI_RDP::fullSync(state, dl);
        }

        // fillRect (op 0xF6) — solid color fills. Useful both as a draw-op
        // sanity check (do any fillRects fire? are they non-degenerate?) and
        // as a clue about whether Factor 5 is clearing the framebuffer per
        // frame (which would explain why subsequent texrects don't show).
        void fillRect_logged(State *state, DisplayList **dl) {
            static int s_count = 0;
            if (gbi_log_enabled() && (++s_count <= 8)) {
                std::fprintf(stderr,
                    "[gbi-f5] fillRect #%d ulx=%u uly=%u lrx=%u lry=%u\n",
                    s_count,
                    (*dl)->p1(12, 12), (*dl)->p1(0, 12),
                    (*dl)->p0(12, 12), (*dl)->p0(0, 12));
                std::fflush(stderr);
            }
            GBI_RDP::fillRect(state, dl);
        }

        void texrectLLE_guarded(State *state, DisplayList **dl) {
            if (texrect_copy_undefined_tile(state, dl)) {
                (*dl)++;  // skip the 8-byte texcoord follow-up to keep dispatch aligned
                return;
            }
            // Track per-active-CIMG-address texrect counts to see which fbs
            // actually receive draws. ROGUESQ_LOG_CIMG=1 enables.
            {
                static std::unordered_map<uint32_t, uint32_t> s_drawsPerFb;
                static int s_total = 0;
                static bool s_log = []() {
                    const char* v = std::getenv("ROGUESQ_LOG_CIMG");
                    return v && v[0] && v[0] != '0';
                }();
                if (s_log) {
                    s_drawsPerFb[state->rdp->colorImage.address]++;
                    if ((++s_total & 0x3F) == 0) {
                        std::fprintf(stderr, "[gbi-f5] texrect-per-fb (total=%d):", s_total);
                        for (const auto& kv : s_drawsPerFb) {
                            std::fprintf(stderr, " 0x%06X=%u", kv.first, kv.second);
                        }
                        std::fprintf(stderr, "\n");
                        std::fflush(stderr);
                    }
                }
            }
            static int s_count = 0;
            int n = ++s_count;
            if (gbi_log_enabled() && n <= 4) {
                int wc = state->ext.workloadQueue->writeCursor;
                const auto& wl = state->ext.workloadQueue->workloads[wc];
                std::fprintf(stderr,
                    "[gbi-f5] texrectLLE #%d PRE  cursor=%d fbPair=%u game=%u colorImg.changed=%d\n",
                    n, wc, wl.fbPairCount, wl.gameCallCount, state->rdp->colorImage.changed);
                std::fflush(stderr);
            }
            GBI_RDP::texrectLLE(state, dl);
            if (gbi_log_enabled() && n <= 4) {
                int wc = state->ext.workloadQueue->writeCursor;
                const auto& wl = state->ext.workloadQueue->workloads[wc];
                std::fprintf(stderr,
                    "[gbi-f5] texrectLLE #%d POST cursor=%d fbPair=%u game=%u triCount=%u\n",
                    n, wc, wl.fbPairCount, wl.gameCallCount, state->drawCall.triangleCount);
                std::fflush(stderr);
            }
        }

        void texrectFlipLLE_guarded(State *state, DisplayList **dl) {
            if (texrect_copy_undefined_tile(state, dl)) {
                (*dl)++;
                return;
            }
            GBI_RDP::texrectFlipLLE(state, dl);
        }

        // loadTLUT / loadTile / loadBlock guards.
        //
        // Replays in RDP::loadTLUTOperation / loadTileOperation compute
        // rowCount = ((lrt >> 2) - (ult >> 2)) + 1, wordsPerRow similarly. If
        // lr < ul, the subtraction underflows uint32_t and the inner loop
        // reads gigabytes past the texture base → AV.
        //
        // Original strategy: reject the load entirely. That kept us crash-free
        // but also threw out real texture loads (Factor 5 emits valid lr/ul
        // values that look superficially like the bogus ones), leaving tile 0
        // with empty TMEM → texrects sample black → black screen.
        //
        // New strategy: CLAMP. Swap lr/ul if inverted; cap region size.
        // Real loads land with intact bounds, opcode-reuse cases get a
        // minimal degenerate load that stores tile metadata but reads ~no
        // RDRAM. We do this by calling state->rdp->loadXxx directly with
        // clamped values rather than delegating to GBI_RDP which would
        // re-parse from the raw DL bytes.

        // Returns true if the load is so absurd we drop it; otherwise fills
        // out *uls/*ult/*lrs/*lrt clamped to a safe region (≤256x256 texels).
        static bool clamp_load_subscripts(const DisplayList *dl,
                                          uint16_t *uls_out, uint16_t *ult_out,
                                          uint16_t *lrs_out, uint16_t *lrt_out) {
            uint16_t uls = dl->p0(12, 12);
            uint16_t ult = dl->p0(0, 12);
            uint16_t lrs = dl->p1(12, 12);
            uint16_t lrt = dl->p1(0, 12);
            // Swap inverted bounds rather than dropping: if Factor 5 emits a
            // backwards range, treating it forward gives us *some* texels.
            if (lrs < uls) std::swap(uls, lrs);
            if (lrt < ult) std::swap(ult, lrt);
            // Cap span to 1024 (256 texels in 10.2 fixed-point) to keep load
            // size bounded.
            const uint16_t MaxSpan = 1024;
            if (lrs - uls > MaxSpan) lrs = uls + MaxSpan;
            if (lrt - ult > MaxSpan) lrt = ult + MaxSpan;
            *uls_out = uls;
            *ult_out = ult;
            *lrs_out = lrs;
            *lrt_out = lrt;
            return false;
        }

        void loadTLUT_guarded(State *state, DisplayList **dl) {
            const uint8_t tile = (*dl)->p1(24, 3);
            uint16_t uls, ult, lrs, lrt;
            clamp_load_subscripts(*dl, &uls, &ult, &lrs, &lrt);
            static int s_count = 0;
            if (gbi_log_enabled() && (++s_count <= 8)) {
                std::fprintf(stderr,
                    "[gbi-f5] loadTLUT #%d tile=%u uls=%u ult=%u lrs=%u lrt=%u (raw w0=0x%08X w1=0x%08X)\n",
                    s_count, tile, uls, ult, lrs, lrt, (*dl)->w0, (*dl)->w1);
                std::fflush(stderr);
            }
            state->rdp->loadTLUT(tile, uls, ult, lrs, lrt);
        }

        void loadTile_guarded(State *state, DisplayList **dl) {
            const uint8_t tile = (*dl)->p1(24, 3);
            uint16_t uls, ult, lrs, lrt;
            clamp_load_subscripts(*dl, &uls, &ult, &lrs, &lrt);
            static int s_count = 0;
            if (gbi_log_enabled() && (++s_count <= 8)) {
                std::fprintf(stderr,
                    "[gbi-f5] loadTile #%d tile=%u uls=%u ult=%u lrs=%u lrt=%u\n",
                    s_count, tile, uls, ult, lrs, lrt);
                std::fflush(stderr);
            }
            state->rdp->loadTile(tile, uls, ult, lrs, lrt);
        }

        void loadBlock_guarded(State *state, DisplayList **dl) {
            // loadBlock has different fields: uls/ult are texture origin,
            // lrs is endpoint, dxt is row stride.
            const uint8_t tile = (*dl)->p1(24, 3);
            uint16_t uls = (*dl)->p0(12, 12);
            uint16_t ult = (*dl)->p0(0, 12);
            uint16_t lrs = (*dl)->p1(12, 12);
            const uint16_t dxt = (*dl)->p1(0, 12);
            if (lrs < uls) std::swap(uls, lrs);
            // Cap the block size at 2048 texels (max valid in 12-bit field).
            if (lrs - uls > 2048) lrs = uls + 2048;
            static int s_count = 0;
            if (gbi_log_enabled() && (++s_count <= 8)) {
                std::fprintf(stderr,
                    "[gbi-f5] loadBlock #%d tile=%u uls=%u ult=%u lrs=%u dxt=%u\n",
                    s_count, tile, uls, ult, lrs, dxt);
                std::fflush(stderr);
            }
            state->rdp->loadBlock(tile, uls, ult, lrs, dxt);
        }

        // setTextureImage filter. Factor 5 sometimes emits SET_TEXTURE_IMAGE
        // (op 0xFD) with payload bytes that are not a valid RDRAM address —
        // anything from KSEG1-flagged pointers to outright garbage. The
        // address gets stored in RDP::texture.address, then the next loadTile
        // / loadBlock / loadTLUT reads `RDRAM[address + offset]` and AVs.
        //
        // RT64 already masks via RDP::maskAddress (& 0xFFFFFF), so anything
        // above 16MB is folded down. The 8MB game RDRAM may be even smaller
        // (4MB without ExpansionPak), but we accept up to 16MB conservatively.
        // Reject when the masked address is zero (no real texture lives there)
        // or when w1 has the high garbage-marker bits set (e.g. all-ones).
        void setTextureImage_filtered(State *state, DisplayList **dl) {
            const uint32_t w1 = (*dl)->w1;
            // All-ones / mostly-ones is a known Factor 5 garbage pattern.
            if (w1 == 0xFFFFFFFFu) return;
            // After RDP::maskAddress folds KSEG bits, address < 0x1000 is
            // almost certainly bogus (low RDRAM is libultra/PIF area).
            const uint32_t masked = w1 & 0x00FFFFFFu;
            if (masked < 0x1000u) return;
            GBI_RDP::setTextureImage(state, dl);
        }

        void setColorImage_filtered(State *state, DisplayList **dl) {
            const uint32_t w0 = (*dl)->w0;
            const uint32_t w1 = (*dl)->w1;
            if (w1 == 0) return;
            const uint32_t fmt = (w0 >> 21) & 0x7;
            const uint32_t siz = (w0 >> 19) & 0x3;
            if (fmt > 4) return;

            // Per-address accept counter. Periodically dumps a table of
            // every legit-shaped setCIMG address and how many times it was
            // emitted. Helps see whether VI's sampled fb actually gets
            // setCIMG'd at all (presence) and how often vs. scratch fbs
            // (frequency). ROGUESQ_LOG_CIMG=1 to enable.
            {
                static std::unordered_map<uint32_t, uint32_t> s_counts;
                static int s_total = 0;
                static bool s_log = []() {
                    const char* v = std::getenv("ROGUESQ_LOG_CIMG");
                    return v && v[0] && v[0] != '0';
                }();
                if (s_log) {
                    s_counts[w1]++;
                    if ((++s_total & 0x3FF) == 0) {
                        std::fprintf(stderr, "[gbi-f5] setCIMG counts (total=%d):", s_total);
                        for (const auto& kv : s_counts) {
                            std::fprintf(stderr, " 0x%08X=%u", kv.first, kv.second);
                        }
                        std::fprintf(stderr, "\n");
                        std::fflush(stderr);
                    }
                }
            }

            // Reject combinations that crash NativeTarget::resolveFromRDRAM's
            // readback asserts (rt64_native_target.cpp:167-176). Supported:
            // RGBA16/RGBA32, CI8, I8, DEPTH16. Everything else aborts.
            const bool unsupported =
                (siz == 0 /* 4b */) ||
                (fmt == G_IM_FMT_RGBA && siz == G_IM_SIZ_8b) ||
                (fmt == G_IM_FMT_IA) ||
                (fmt == G_IM_FMT_CI && siz != G_IM_SIZ_8b) ||
                (fmt == G_IM_FMT_I && siz != G_IM_SIZ_8b);
            if (unsupported) {
                if (gbi_log_enabled()) {
                    std::fprintf(stderr,
                        "[gbi-f5] setCIMG reject fmt=%u siz=%u w0=0x%08X w1=0x%08X\n",
                        fmt, siz, w0, w1);
                    std::fflush(stderr);
                }
                return;
            }

            GBI_F3D::setColorImage(state, dl);
        }

        void setup(GBI *gbi) {
            // Inherit all F3DEX defaults first, then override the opcodes
            // Factor 5 reuses for its own meanings.
            GBI_F3DEX::setup(gbi);

            // 0x02: Factor 5-specific (constant payload 0x028001C0/0x01FF0000).
            //       TODO: identify. Treat as no-op until the RSP-side recompile
            //       reverse-engineers it.
            gbi->map[0x02] = &op_noop;

            // 0x03: G_MOVEMEM. Factor 5 emits subcodes outside F3D's standard
            //       set. Use a permissive variant that no-ops unknown subcodes
            //       (with optional log) instead of asserting.
            gbi->map[F3D_G_MOVEMEM] = &moveMem_permissive;

            // 0x06: G_DL — but Factor 5 reuses byte 0x06 for non-DL purposes.
            //       Strict filter delegates only when payload looks like a real
            //       G_DL (no extra payload bits set).
            gbi->map[0x06] = &op_06_strict_dl;

            // 0x80: Factor 5 chunk header (metadata, no-op for HLE).
            gbi->map[0x80] = &op_noop;

            // 0xB0: F3DEX G_BRANCH_Z. Factor 5 emits packed data here that
            //       crashes the inherited handler's vertex-cache index check.
            gbi->map[0xB0] = &op_noop;

            // 0xB2: F3DEX G_MODIFYVTX. Factor 5's payload yields out-of-range
            //       vertex indices that trip the RSP::modifyVertex assert
            //       ("Vertex index is not valid. DL is possibly corrupted").
            //       Same opcode-reuse pattern as 0xB0/0xB5 — no-op for HLE.
            gbi->map[0xB2] = &op_noop;

            // 0xAF: F3DEX G_LOAD_UCODE. The inherited handler hashes whatever
            //       bytes the args point at and crashes when the lookup misses
            //       (hleGBI becomes nullptr → next opcode dereferences null).
            //       Factor 5 is monolithic — it never legitimately swaps to a
            //       child ucode mid-DL; treat any 0xAF as opcode reuse and
            //       no-op so the registered F3DFACTOR5 GBI stays selected.
            gbi->map[0xAF] = &op_noop;

            // 0xB4 / 0xBF: 16-byte payload commands. Consume the extra word.
            gbi->map[0xB4] = &op_consume16;
            gbi->map[0xBF] = &op_consume16;

            // 0xB5: Factor 5 chunk/DL terminator marker. Treated as no-op
            //       since op 0xB8 (standard G_ENDDL, inherited from F3D)
            //       is the real "return from DL".
            gbi->map[0xB5] = &op_noop;

            // 0xFC/0xF5: G_SETCOMBINE / G_SETTILE. Pass-through with logs
            //       so we can see what tile + combiner state Factor 5 is
            //       building up before texrects fire.
            gbi->map[0xFC] = &setCombine_logged;
            gbi->map[0xF5] = &setTile_logged;

            // 0xE9: G_RDPFULLSYNC. Pass-through with log so we can see if
            //       Factor 5 ever emits a fullSync mid-DL (which would
            //       advance writeCursor and make our post-DL probe see
            //       a fresh empty workload instead of the populated one).
            gbi->map[0xE9] = &fullSync_logged;

            // 0xF6: G_FILLRECT — solid-color rectangle. Pass-through with log.
            gbi->map[0xF6] = &fillRect_logged;

            // 0xB9/0xBA/0xED: setOtherMode L/H, setScissor. Pass-through
            //       with logs so we can see what cycle/blend/scissor state
            //       is active when texrects render. F3D-style setOtherMode
            //       L/H is what Factor 5 uses for legitimate state changes.
            gbi->map[0xB9] = &setOtherModeL_logged;
            gbi->map[0xBA] = &setOtherModeH_logged;
            gbi->map[0xED] = &setScissor_logged;

            // 0xEF: G_RDPSETOTHERMODE. Factor 5 reuses byte 0xEF for non-
            //       command purposes (observed: pixel-pattern bytes like
            //       w0=0xEFEFEFFF w1=0x515151FF). The inherited handler
            //       forwards w0/w1 directly into RDP::setOtherMode which
            //       then corrupts cycleType/rendermode/blender to all-ones,
            //       making texrects render with broken combiner output and
            //       producing the persistent black screen. F3D-style 0xB9/
            //       0xBA setOtherMode L/H is the path Factor 5 uses for
            //       real state changes; 0xEF is opcode reuse → no-op.
            gbi->map[0xEF] = &op_noop;

            // 0xFD: G_SETTIMG. Factor 5 sometimes emits 0xFFFFFFFF or
            //       sub-1KB-offset addresses; reject those before they're
            //       stored in RDP::texture and dereferenced by loadTile/TLUT.
            gbi->map[0xFD] = &setTextureImage_filtered;

            // 0xF0/0xF3/0xF4: G_LOADTLUT, G_LOADBLOCK, G_LOADTILE. Reject
            //       when subscripts would underflow / exceed reasonable size
            //       (Factor 5 occasionally emits ops misparsed as loads with
            //       lr < ul, which makes the deferred replay AV in RDRAM).
            gbi->map[0xF0] = &loadTLUT_guarded;
            gbi->map[0xF3] = &loadBlock_guarded;
            gbi->map[0xF4] = &loadTile_guarded;

            // 0xFF: G_SETCIMG with Factor 5-specific filtering for bogus
            //       payloads (w1=0 marker, invalid fmt fields).
            gbi->map[0xFF] = &setColorImage_filtered;

            // 0xE4 / 0xE5: Factor 5 emits TEXRECTs in LLE format (16 bytes:
            //       TEXRECT + one RDPHALF follow-up). The default F3DEX HLE
            //       texrect handler reads 24 bytes and consumes the *next*
            //       TEXRECT as garbage follow-up. Use LLE variants wrapped in
            //       a copy-mode-undefined-tile guard (texrect_copy_undefined_tile)
            //       so the assert in State::loadDrawState doesn't trip when
            //       Factor 5 emits texrects against unconfigured tiles.
            gbi->map[0xE4] = &texrectLLE_guarded;
            gbi->map[0xE5] = &texrectFlipLLE_guarded;
        }

    }  // namespace GBI_F3DFACTOR5
}  // namespace RT64
