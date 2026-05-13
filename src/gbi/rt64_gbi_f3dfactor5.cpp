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
        // ROGUESQ_LOG_GBI=1 enables per-handler diagnostic logs (off by default
        // because they lock up the ImGui inspector at thousands of lines/sec).
        static bool gbi_log_enabled() {
            static bool s_enabled = []() {
                const char* v = std::getenv("ROGUESQ_LOG_GBI");
                return v && v[0] && v[0] != '0';
            }();
            return s_enabled;
        }

        // ROGUESQ_HLE_FORCE_VISIBLE=1 — override fillColor/combiner/primColor
        // to known-rendering magenta values to test GPU output independently
        // of game state.
        static bool force_visible_enabled() {
            static bool s = []() {
                const char* v = std::getenv("ROGUESQ_HLE_FORCE_VISIBLE");
                return v && v[0] && v[0] != '0';
            }();
            return s;
        }

        // G_CC_PRIMITIVE mux: c0/c1 = (0,0,0,PRIM), a0/a1 = (0,0,0,PRIM).
        static constexpr uint32_t FORCED_COMB_W0 = 0x00FFFEE7u;
        static constexpr uint32_t FORCED_COMB_W1 = 0x771F77F8u;

        // Force magenta-opaque fillColor (RGBA5551 0xF80F packed twice).
        void setFillColor_overridden(State *state, DisplayList **dl) {
            if (force_visible_enabled()) {
                state->rdp->setFillColor(0xF80FF80F);
                return;
            }
            GBI_RDP::setFillColor(state, dl);
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

        // op 0x01/0x02/0x03 experimental HLE handlers
        // (ROGUESQ_HLE_OP02_EXPERIMENTAL=1).
        //
        // Decoded from F3DFACTOR5 RSP ucode (see
        // project_attribution_op02_breakthrough_2026_05_13.md):
        //   - op 0x01: payload (w0_low=count_or_size, w1=ram_src_addr).
        //              Likely "DMA from RAM into a DMEM scratch buffer."
        //   - op 0x03: payload constant 0x03800000 / 0x00000000. Likely
        //              "select fixed-DMEM matrix index + setup register state."
        //   - op 0x02: payload constant 0x028001C0 / 0x01FF0000. Loops over
        //              vertex pairs, calls matrix×vec multiply (func_4001F14)
        //              + perspective divide (func_4001F60), writes 16-byte
        //              RDP triangle commands to the RDP output buffer.
        //
        // The experimental handler is a first-cut implementation that:
        //   (a) Saves op 0x01's referenced RAM address for later inspection
        //   (b) When op 0x02 fires, logs the saved state + DMEM-equivalent
        //       data the original RSP would consume
        //   (c) Emits a debug-visible fillRect covering the attribution-text
        //       region with primitive color, proving the dispatch path is now
        //       producing visible output. NOT a correct text render yet —
        //       the actual matrix-vertex math + glyph-triangle emission is
        //       follow-up work.
        //   (d) Sets a flag picked up by fillRect_logged to optionally skip
        //       the subsequent black-clear so the debug output isn't wiped.
        static bool op02_experimental_enabled() {
            static bool s = []() {
                const char* v = std::getenv("ROGUESQ_HLE_OP02_EXPERIMENTAL");
                return v && v[0] && v[0] != '0';
            }();
            return s;
        }

        // Cross-handler state (per-DL-submission lifetime). Reset by
        // op 0x80 (chunk header) or first occurrence inside a DL.
        static thread_local uint32_t s_op01_last_ram = 0;
        static thread_local uint32_t s_op01_last_w0 = 0;
        static thread_local uint32_t s_op03_last_w0 = 0;
        static thread_local int s_op02_count_this_dl = 0;
        static thread_local bool s_op02_fired_recently = false;

        void op_01_experimental(State *state, DisplayList **dl) {
            const uint32_t w0 = (*dl)->w0;
            const uint32_t w1 = (*dl)->w1;
            s_op01_last_ram = w1;
            s_op01_last_w0 = w0;
            // Decode w0 fields: byte 1 = ?, byte 2 = ?, byte 3 = count_or_size
            const uint32_t b1 = (w0 >> 16) & 0xFF;
            const uint32_t b2 = (w0 >> 8) & 0xFF;
            const uint32_t b3 = w0 & 0xFF;
            static int s_count = 0;
            if (gbi_log_enabled() && (++s_count <= 8)) {
                std::fprintf(stderr,
                    "[gbi-f5 op01 #%d] w0=0x%08X (b1=%u b2=%u b3=%u) w1=0x%08X (ram src)\n",
                    s_count, w0, b1, b2, b3, w1);
                std::fflush(stderr);
            }
        }

        void op_03_experimental(State *state, DisplayList **dl) {
            const uint32_t w0 = (*dl)->w0;
            const uint32_t w1 = (*dl)->w1;
            s_op03_last_w0 = w0;
            static int s_count = 0;
            if (gbi_log_enabled() && (++s_count <= 8)) {
                std::fprintf(stderr,
                    "[gbi-f5 op03 #%d] w0=0x%08X w1=0x%08X\n",
                    s_count, w0, w1);
                std::fflush(stderr);
            }
        }

        // Matrix-vertex math for op 0x02. Reads the perspective matrix
        // loaded from RAM 0x80700000 (verified to be a 4×4 projection
        // matrix per project_attribution_op02_breakthrough_2026_05_13.md).
        // Loads vertex data from RAM-pointed buffer, applies matrix
        // transform + perspective divide, emits screen-space triangles.
        //
        // The math is faithful but the INPUT data sources (vertex pool,
        // op count, primitive layout) are still uncertain — current cut
        // uses placeholder vertex data based on the alternating
        // 0x80700000/0x80710000 payload addresses we observed.
        struct N64Matrix4x4 {
            float m[4][4];  // row-major, [row][col]
        };

        // Decode an N64 fixed-point 4×4 matrix from RDRAM at offset `addr`.
        // Layout: 32 bytes of int16 integer parts, then 32 bytes of uint16
        // fractional parts; combined as (int << 16) | frac → s15.16 fixed.
        static N64Matrix4x4 decode_n64_matrix(const uint8_t* rdram, uint32_t addr) {
            N64Matrix4x4 mat = {};
            const uint32_t phys = addr & 0x00FFFFFF;
            auto load_be16 = [&](uint32_t off) -> uint16_t {
                if (phys + off + 1 >= 0x800000) return 0;
                uint8_t hi = rdram[(phys + off) ^ 3];
                uint8_t lo = rdram[(phys + off + 1) ^ 3];
                return (hi << 8) | lo;
            };
            for (int row = 0; row < 4; ++row) {
                for (int col = 0; col < 4; ++col) {
                    int idx = row * 4 + col;
                    uint16_t ipart = load_be16(idx * 2);
                    uint16_t fpart = load_be16(32 + idx * 2);
                    int32_t combined = ((int32_t)(int16_t)ipart << 16) | fpart;
                    mat.m[row][col] = (float)combined / 65536.0f;
                }
            }
            return mat;
        }

        // Multiply a 4×4 matrix by a 4D vector (M × v → result).
        static void mat_mul_vec(const N64Matrix4x4& m, const float v[4], float out[4]) {
            for (int i = 0; i < 4; ++i) {
                out[i] = m.m[i][0] * v[0] + m.m[i][1] * v[1] +
                         m.m[i][2] * v[2] + m.m[i][3] * v[3];
            }
        }

        // Apply perspective divide (1/w on x, y, z).
        static void perspective_divide(float v[4]) {
            if (v[3] != 0.0f) {
                const float inv_w = 1.0f / v[3];
                v[0] *= inv_w;
                v[1] *= inv_w;
                v[2] *= inv_w;
            }
        }

        void op_02_experimental(State *state, DisplayList **dl) {
            const uint32_t w0 = (*dl)->w0;
            const uint32_t w1 = (*dl)->w1;
            s_op02_count_this_dl++;
            s_op02_fired_recently = true;

            const uint32_t cimg = state->rdp->colorImage.address;
            const auto& prim = state->rdp->primColorStack[state->rdp->primColorStackSize - 1];

            static int s_count = 0;
            int n = ++s_count;
            if (gbi_log_enabled() && n <= 12) {
                std::fprintf(stderr,
                    "[gbi-f5 op02 #%d] w0=0x%08X w1=0x%08X "
                    "cimg=0x%06X primRGBA=(%.2f %.2f %.2f %.2f) "
                    "last_op01_ram=0x%08X last_op03_w0=0x%08X\n",
                    n, w0, w1, cimg,
                    (float)prim.x, (float)prim.y, (float)prim.z, (float)prim.w,
                    s_op01_last_ram, s_op03_last_w0);
                std::fflush(stderr);

                // On first op_02 fire, decode + log the perspective
                // matrix from the last-seen op 0x01 RAM address. This
                // proves we can read the same matrix the real RSP uses.
                if (n == 1 && s_op01_last_ram != 0) {
                    auto mat = decode_n64_matrix(state->RDRAM, s_op01_last_ram);
                    std::fprintf(stderr, "[gbi-f5 op02 matrix from 0x%08X]:\n", s_op01_last_ram);
                    for (int r = 0; r < 4; ++r) {
                        std::fprintf(stderr, "  %9.4f %9.4f %9.4f %9.4f\n",
                            mat.m[r][0], mat.m[r][1], mat.m[r][2], mat.m[r][3]);
                    }
                    // Sanity-test the math: transform a known vertex
                    // through the matrix + perspective divide.
                    float v[4] = { 0.0f, 0.0f, -1.0f, 1.0f };  // unit point in front of camera
                    float out[4];
                    mat_mul_vec(mat, v, out);
                    std::fprintf(stderr, "  test vertex (0,0,-1,1) -> (%.4f, %.4f, %.4f, %.4f)\n",
                        out[0], out[1], out[2], out[3]);
                    perspective_divide(out);
                    std::fprintf(stderr, "  after /w: (%.4f, %.4f, %.4f, %.4f)\n",
                        out[0], out[1], out[2], out[3]);
                    std::fflush(stderr);
                }
            }
            // No drawing here — the actual visible-output emission happens
            // inside fillRect_op02_aware. See note there for why the math
            // result isn't emitted yet (RT64 deferred-state replay model
            // means we'd need to use drawTris which has its own visibility
            // issues to resolve before plumbing in the real math result).
        }

        // Wrapped fillRect_logged variant — when op 0x02 fired recently
        // in the same DL and the about-to-fire fillRect would clear the
        // entire screen with the canonical 0x00010001 black-with-alpha
        // color, override fillColor to bright green BEFORE the clear
        // runs. RT64's deferred-render pipeline captures the fillColor
        // at fillRect enqueue time, so the original F6 fillRect's
        // coords (full lo-res screen) get filled with our green
        // instead of black. Visual result: bright-green band during
        // attribution period — proving op 0x02 path is reachable.
        //
        // The "glyph pattern" experiment (post-clear small rects) didn't
        // produce visible output — likely RT64's deferred-render sort
        // order or coverage-based pixel masking suppresses the small
        // rects when they overlap a same-frame clear. Faithful glyph
        // shapes require the full op 0x02 vertex pipeline (or LLE).
        void fillRect_logged(State *state, DisplayList **dl);  // fwd-decl
        void fillRect_op02_aware(State *state, DisplayList **dl) {
            if (op02_experimental_enabled() && s_op02_fired_recently) {
                const uint32_t fillRaw = state->rdp->fillColorStack[state->rdp->fillColorStackSize - 1];
                if (fillRaw == 0x00010001u) {
                    // ROGUESQ_HLE_OP02_SKIP_CLEAR=1: drop the canonical
                    // 0x00010001 fillRect during attribution entirely (no
                    // green override, no clear at all). If the game is
                    // CPU-painting or otherwise emitting real attribution
                    // content into the same fb, skipping the clear lets
                    // that content survive to the present step.
                    static int s_skip_clear = -1;
                    if (s_skip_clear < 0) {
                        const char* v = std::getenv("ROGUESQ_HLE_OP02_SKIP_CLEAR");
                        s_skip_clear = (v && *v && v[0] != '0') ? 1 : 0;
                    }
                    if (s_skip_clear) {
                        static int s_skip_log = 0;
                        if (gbi_log_enabled() && (++s_skip_log <= 4)) {
                            std::fprintf(stderr,
                                "[gbi-f5 fillRect-skip #%d] op02 recently fired; "
                                "DROPPING canonical-black fillRect (cimg=0x%06X)\n",
                                s_skip_log, state->rdp->colorImage.address);
                            std::fflush(stderr);
                        }
                        // Don't advance *dl — the dispatch loop does that.
                        // We just don't call state->rdp->fillRect, so no
                        // FILL_RECTANGLE reaches the deferred RDP.
                        return;
                    }
                    static int s_replace = 0;
                    if (gbi_log_enabled() && (++s_replace <= 4)) {
                        std::fprintf(stderr,
                            "[gbi-f5 fillRect-replace #%d] op02 recently fired; "
                            "overriding clear fillColor to green (cimg=0x%06X)\n",
                            s_replace, state->rdp->colorImage.address);
                        std::fflush(stderr);
                    }
                    // Override stack-top fillColor to bright green BEFORE
                    // calling fillRect_logged. The original F6 fillRect
                    // command will then enqueue with our color.
                    state->rdp->setFillColor(0x07C107C1);
                    fillRect_logged(state, dl);
                    // Probe: try a single triangle via drawTris to test
                    // whether the API is usable for the eventual matrix-
                    // vertex pipeline output. ROGUESQ_HLE_OP02_DRAW_TRI_TEST=1
                    // enables this single test triangle on the green background.
                    static int s_tri_test = -1;
                    if (s_tri_test < 0) {
                        const char* v = std::getenv("ROGUESQ_HLE_OP02_DRAW_TRI_TEST");
                        s_tri_test = (v && *v && v[0] != '0') ? 1 : 0;
                    }
                    if (s_tri_test) {
                        // drawTris path proven working (cycle FILL→1CYCLE,
                        // G_CC_PRIMITIVE combiner, white prim color). The
                        // previous bitmap-font test rendered readable
                        // "LucasArts" / "Factor 5" text but that was our
                        // own font, NOT the game's actual attribution
                        // glyphs. Don't render placeholder text here —
                        // attribution rendering should use the game's
                        // actual vertex/glyph data once we trace its
                        // source. For now, no drawTris emit; user sees
                        // bright-green attribution screen (proves the
                        // green fillRect override path works) while the
                        // real implementation work continues.
                    }
                    return;
                }
            }
            fillRect_logged(state, dl);
        }


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
            // Strip coverage-related otherModeL bits (AA_EN / CVG_X_ALPHA /
            // ALPHA_CVG_SEL) — Factor 5's combiner outputs alpha=0, which
            // combined with these bits makes RT64 discard pixels as zero-cvg.
            static const uint32_t s_strip_mask = []() {
                uint32_t mask = 0;
                auto check = [&](const char* env, uint32_t bits) {
                    const char* v = std::getenv(env);
                    if (v && v[0] && v[0] != '0') mask |= bits;
                };
                check("ROGUESQ_HLE_NO_AA", 1u << 14);
                check("ROGUESQ_HLE_NO_CVGA", (1u << 23) | (1u << 24));
                check("ROGUESQ_HLE_FORCE_OPAQUE",
                      (1u << 14) | (1u << 23) | (1u << 24));
                return mask;
            }();
            if (s_strip_mask) {
                const uint32_t shift = (*dl)->p0(8, 8);
                const uint32_t length = (*dl)->p0(0, 8);
                // setOtherModeL writes `length+1` bits at position `shift`.
                // Only mask bits this write actually covers.
                const uint32_t bitsCovered =
                    ((length + 1 >= 32) ? 0xFFFFFFFFu
                                        : ((1u << (length + 1)) - 1)) << shift;
                (*dl)->w1 &= ~(s_strip_mask & bitsCovered);
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

        // FORCED_COMB_W0/W1 moved earlier (see file top).

        void setCombine_logged(State *state, DisplayList **dl) {
            // Track every distinct (w0, w1) pair the game emits for setCombine.
            // First 8 calls also get logged unconditionally so we can see
            // chronology. ROGUESQ_LOG_GBI=1 enables.
            if (gbi_log_enabled()) {
                static std::unordered_set<uint64_t> s_seen;
                static int s_count = 0;
                const uint64_t key = (uint64_t((*dl)->w0) << 32) | uint64_t((*dl)->w1);
                const bool firstSeen = s_seen.insert(key).second;
                if (++s_count <= 8 || firstSeen) {
                    std::fprintf(stderr,
                        "[gbi-f5] setCombine #%d%s w0=0x%08X w1=0x%08X\n",
                        s_count, firstSeen ? " (NEW)" : "",
                        (*dl)->w0, (*dl)->w1);
                    std::fflush(stderr);
                }
            }
            // Force solid-prim combiner under ROGUESQ_HLE_FORCE_VISIBLE.
            if (force_visible_enabled()) {
                state->rdp->setCombine(
                    (uint64_t(FORCED_COMB_W1) << 32) | uint64_t(FORCED_COMB_W0));
                // Also force prim color to magenta opaque so the combiner
                // produces visible pixels.
                state->rdp->setPrimColor(0, 0xFF, 0xFF00FFFFu);
                return;
            }

            // ROGUESQ_HLE_FORCE_COMB — like FORCE_VISIBLE but does NOT
            // bypass texrects to fillRects. Keeps the texrect dispatch
            // path intact, only swaps the combiner to constant-white.
            // Use to test: do texrects produce visible pixels when the
            // combiner output is forced to white (i.e. sampling+blender
            // pipeline works), vs. is something else upstream killing
            // them? If white pixels appear at attribution glyph
            // positions, the original combiner (or its texture
            // sampling) is the bug. If still black, the texrect
            // dispatch path itself drops the work.
            static bool s_force_comb = []() {
                const char* v = std::getenv("ROGUESQ_HLE_FORCE_COMB");
                return v && v[0] && v[0] != '0';
            }();
            if (s_force_comb) {
                state->rdp->setCombine(
                    (uint64_t(FORCED_COMB_W1) << 32) | uint64_t(FORCED_COMB_W0));
                state->rdp->setPrimColor(0, 0xFF, 0xFFFFFFFFu);
                return;
            }

            // ROGUESQ_HLE_FORCE_PRIM_OUTPUT — port of the LLE-era
            // PARTICLE_VISIBLE_DEBUG fix (reverted commit e718774 in lib/rt64).
            // For the cinematic-text mux family (0xFC11FE23 and variants —
            // attribution glyphs + N64 logo + explosion sprites), rewrite
            // both color and alpha cycle 1 D fields to force output =
            // PRIMITIVE color, alpha = ONE. Bypasses texture sampling
            // entirely so we can distinguish "pipeline works, sampling
            // broken" from "deeper bug".
            //
            //   color D cycle 1 (H bits 17-15) ← 3 (C_PRIMITIVE)
            //   alpha D cycle 1 (H bits 11-9)  ← 6 (A_ONE)
            //   (also rewrite cycle 2 D bits for 2-cycle muxes)
            //
            // Default off; set ROGUESQ_HLE_FORCE_PRIM_OUTPUT=1 to enable.
            // If attribution shows prim-colored shapes where text should
            // be, the gap is in texture sampling. If still black, the
            // gap is in the blender/render-target pipeline.
            static bool s_force_prim = []() {
                const char* v = std::getenv("ROGUESQ_HLE_FORCE_PRIM_OUTPUT");
                return v && v[0] && v[0] != '0';
            }();
            if (s_force_prim) {
                const uint32_t w0 = (*dl)->w0;
                const uint32_t w1 = (*dl)->w1;
                const uint32_t lowL = w0;
                const bool is_cinematic_mux =
                    (lowL == 0xFC11FE23u) || (lowL == 0xFC11E623u) ||
                    (lowL == 0xFC119623u) || (lowL == 0xFC127FFFu) ||
                    (lowL == 0xFC127E24u);
                if (is_cinematic_mux) {
                    // Rewrite H bits to force PRIMITIVE color + ONE alpha output.
                    uint32_t patched_w1 = w1;
                    patched_w1 = (patched_w1 & ~(0x7u << 15)) | (3u << 15); // color D c1
                    patched_w1 = (patched_w1 & ~(0x7u << 6 )) | (3u << 6 ); // color D c2
                    patched_w1 = (patched_w1 & ~(0x7u << 9 )) | (6u << 9 ); // alpha D c1
                    patched_w1 = (patched_w1 & ~(0x7u << 0 )) | (6u << 0 ); // alpha D c2
                    if (gbi_log_enabled()) {
                        static int s_n = 0;
                        if (++s_n <= 4) {
                            std::fprintf(stderr,
                                "[gbi-f5] force-prim rewrite w0=0x%08X w1=0x%08X -> w1=0x%08X\n",
                                w0, w1, patched_w1);
                            std::fflush(stderr);
                        }
                    }
                    // Force prim color to a bright known value so the
                    // forced-prim output is unmistakable on screen.
                    state->rdp->setPrimColor(0, 0xFF, 0x00FFFFFFu); // bright cyan
                    state->rdp->setCombine(
                        (uint64_t(patched_w1) << 32) | uint64_t(w0));
                    return;
                }
            }

            // Factor 5 alpha-all-zero patch (2026-05-13).
            //
            // Attribution-screen DLs emit setCombine with alpha cycle-1 fields
            // A/B/C/D all encoding index 7. SDK only defines index 7 for the
            // C field (G_ACMUX_0); for A/B/D it's an "uninitialized" value
            // RT64 maps to A_ZERO via the alphaInputABD default case. Combined
            // with the rendermode whose blender A_M1 sources CC alpha
            // (G_BL_A_IN), the result is alpha=0 and every pixel multiplies
            // by 0 in the blender → pure black output.
            //
            // We detect the all-7s pattern and rewrite the D field to index 6
            // (= A_ONE for alphaInputABD). New alpha equation:
            //   (A - B) * C + D = (0 - 0) * 0 + 1 = 1
            // The blender sees alpha=1, color reaches the framebuffer.
            //
            // Gated default-on; disable with ROGUESQ_HLE_PATCH_ALPHA=0 if it
            // breaks anything (e.g. legitimate fully-transparent draws).
            static bool s_patch_enabled = []() {
                const char* v = std::getenv("ROGUESQ_HLE_PATCH_ALPHA");
                return !(v && v[0] == '0');  // default on
            }();

            uint32_t w0 = (*dl)->w0;
            uint32_t w1 = (*dl)->w1;
            if (s_patch_enabled) {
                const uint32_t alphaA = (w0 >> 12) & 0x7;
                const uint32_t alphaB = (w1 >> 12) & 0x7;
                const uint32_t alphaC = (w0 >>  9) & 0x7;
                const uint32_t alphaD = (w1 >>  9) & 0x7;
                if (alphaA == 7 && alphaB == 7 && alphaC == 7 && alphaD == 7) {
                    const uint32_t patched_w1 = (w1 & ~(0x7u << 9)) | (0x6u << 9);
                    static int s_patched = 0;
                    if (gbi_log_enabled() && (++s_patched <= 4)) {
                        std::fprintf(stderr,
                            "[gbi-f5] alpha-patch setCombine w0=0x%08X w1=0x%08X -> w1=0x%08X (force alpha=1)\n",
                            w0, w1, patched_w1);
                        std::fflush(stderr);
                    }
                    state->rdp->setCombine(
                        (uint64_t(patched_w1) << 32) | uint64_t(w0));
                    return;
                }
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

        // setFillColor_overridden moved earlier (see file top).

        // fillRect (op 0xF6) — solid color fills. Logs draw geometry.
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
            // Track distinct (colorImg, combL, combH) at texrect time. This
            // shows which combiner Factor 5 ACTUALLY uses to render to each
            // fb (not the stale initial setup). ROGUESQ_LOG_GBI=1.
            if (gbi_log_enabled()) {
                struct Key { uint32_t fb, l, h; bool operator==(const Key& o) const { return fb==o.fb && l==o.l && h==o.h; } };
                struct KeyHash { size_t operator()(const Key& k) const { return std::hash<uint64_t>()((uint64_t(k.fb) << 32) ^ (uint64_t(k.l) << 16) ^ uint64_t(k.h)); } };
                static std::unordered_set<Key, KeyHash> s_seen;
                const auto& comb = state->rdp->colorCombinerStack[state->rdp->colorCombinerStackSize - 1];
                Key k{state->rdp->colorImage.address, comb.L, comb.H};
                if (s_seen.insert(k).second) {
                    std::fprintf(stderr,
                        "[gbi-f5] texrect-uses fb=0x%06X combL=0x%08X combH=0x%08X otherL=0x%08X\n",
                        k.fb, k.l, k.h, state->rdp->otherMode.L);
                    std::fflush(stderr);
                }
            }
            static int s_count = 0;
            int n = ++s_count;
            if (gbi_log_enabled() && n <= 4) {
                // Log the per-texrect render state so we can see whether
                // RT64 receives well-formed inputs:
                //   - colorImage: where the draw is targeted
                //   - prim/env/fill color: combiner inputs
                //   - combiner mux: how alpha is computed
                //   - othermode L: blend / cycle / coverage / alpha bits
                const auto& prim = state->rdp->primColorStack[state->rdp->primColorStackSize - 1];
                const auto& env = state->rdp->envColorStack[state->rdp->envColorStackSize - 1];
                const uint32_t fillRaw = state->rdp->fillColorStack[state->rdp->fillColorStackSize - 1];
                const auto& comb = state->rdp->colorCombinerStack[state->rdp->colorCombinerStackSize - 1];
                std::fprintf(stderr,
                    "[gbi-f5] texrect #%d colorImg=0x%06X "
                    "prim=(%.2f %.2f %.2f %.2f) env=(%.2f %.2f %.2f %.2f) fillRaw=0x%08X "
                    "combL=0x%08X combH=0x%08X otherL=0x%08X otherH=0x%08X\n",
                    n,
                    state->rdp->colorImage.address,
                    (float)prim.x, (float)prim.y, (float)prim.z, (float)prim.w,
                    (float)env.x, (float)env.y, (float)env.z, (float)env.w,
                    fillRaw,
                    comb.L, comb.H,
                    state->rdp->otherMode.L,
                    state->rdp->otherMode.H);
                std::fflush(stderr);
            }
            // ROGUESQ_HLE_FORCE_FILLRECT — convert every texrect into a
            // fillRect at the same coords. Pure existence test: if magenta
            // appears, RT64 IS presenting; just our texrect path is wrong.
            if (force_visible_enabled()) {
                state->rdp->setFillColor(0xF80FF80F);  // RGBA5551 magenta opaque
                // Use the original texrect bounds. drawTexRect cycles to
                // 1cycle and emits 2 tris. Force-fillRect is simpler: just
                // call drawRect bypassing texture sampling.
                int32_t ulx = (*dl)[0].p1(12, 12);
                int32_t uly = (*dl)[0].p1(0, 12);
                int32_t lrx = (*dl)[0].p0(12, 12);
                int32_t lry = (*dl)[0].p0(0, 12);
                state->rdp->fillRect(ulx, uly, lrx, lry);
                (*dl)++;  // consume the LLE texrect follow-up word
                return;
            }
            GBI_RDP::texrectLLE(state, dl);
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
            // Track distinct SETTIMG source addresses to find missing fb-as-
            // texture composite ops (e.g. hi-res scratch → lo-res VI fb).
            if (gbi_log_enabled()) {
                static std::unordered_map<uint32_t, uint32_t> s_seen;
                const uint32_t count = ++s_seen[w1];
                if (count == 1 || count == 8 || (count & 0xFF) == 0) {
                    std::fprintf(stderr,
                        "[gbi-f5] setTextureImage addr=0x%08X count=%u w0=0x%08X\n",
                        w1, count, (*dl)->w0);
                    std::fflush(stderr);
                    // Dump 32 bytes of texture source data to check for
                    // all-zero (asset never loaded) vs. real content.
                    if (count == 1) {
                        const uint32_t phys = w1 & 0x00FFFFFFu;
                        if (phys < 0x800000u) {
                            uint8_t* rdram = state->fromRDRAM(0);
                            if (rdram != nullptr) {
                                std::fprintf(stderr, "  bytes:");
                                int nonzero = 0;
                                for (int i = 0; i < 32; ++i) {
                                    uint8_t b = rdram[(phys + i) ^ 3];
                                    std::fprintf(stderr, " %02X", b);
                                    if (b) nonzero++;
                                }
                                std::fprintf(stderr, "  (nonzero=%d/32)\n", nonzero);
                                std::fflush(stderr);
                            }
                        }
                    }
                }
            }
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

            // After setColorImage, log the final CIMG width/fmt/siz for the
            // VI-sampled buffer addrs so we can verify they match VI's
            // expectations (needed for PresentEarly matcher).
            {
                static bool s_log = []() {
                    const char* v = std::getenv("ROGUESQ_LOG_CIMG");
                    return v && v[0] && v[0] != '0';
                }();
                if (s_log) {
                    const uint32_t addr = state->rdp->colorImage.address;
                    if (addr == 0x62B800 || addr == 0x695C00 || addr == 0x795C00) {
                        static std::unordered_map<uint64_t, uint32_t> s_seenDims;
                        uint64_t key = (uint64_t(addr) << 32) |
                                       (uint64_t(state->rdp->colorImage.width) << 16) |
                                       (uint64_t(state->rdp->colorImage.fmt) << 8) |
                                       uint64_t(state->rdp->colorImage.siz);
                        if (s_seenDims.insert({key, 1}).second) {
                            std::fprintf(stderr,
                                "[gbi-f5] visFb-CIMG addr=0x%06X width=%u fmt=%u siz=%u\n",
                                addr, state->rdp->colorImage.width,
                                state->rdp->colorImage.fmt, state->rdp->colorImage.siz);
                            std::fflush(stderr);
                        }
                    }
                }
            }
        }

        void setup(GBI *gbi) {
            // Inherit all F3DEX defaults first, then override the opcodes
            // Factor 5 reuses for its own meanings.
            GBI_F3DEX::setup(gbi);

            // 0x01/0x02/0x03: Factor 5's custom asset-load + vertex-transform
            //       + triangle-emit family. Decoded from RSP ucode at IMEM
            //       0x14F0 (op 0x02), 0x1504 (op 0x01 falls into shared
            //       body), 0x14D4 (op 0x03 prefix). When experimental
            //       handlers are enabled, route them to op_*_experimental;
            //       otherwise keep the original no-ops + permissive 0x03.
            //       See project_attribution_op02_breakthrough_2026_05_13.md.
            if (op02_experimental_enabled()) {
                gbi->map[0x01] = &op_01_experimental;
                gbi->map[0x02] = &op_02_experimental;
                gbi->map[0x03] = &op_03_experimental;
            } else {
                gbi->map[0x02] = &op_noop;
                // 0x03: G_MOVEMEM. Factor 5 emits subcodes outside F3D's
                //       standard set. Use a permissive variant that no-ops
                //       unknown subcodes (with optional log) instead of
                //       asserting.
                gbi->map[F3D_G_MOVEMEM] = &moveMem_permissive;
            }

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
            //
            // 2026-05-13 — Verified by user test (ROGUESQ_HLE_TRI_PASSTHROUGH=1)
            // that letting F3DEX's standard 0xBF handler (G_TRI1) take over
            // produces garbled graphics during N64 logo. So 0xBF in Factor 5's
            // variant is NOT standard G_TRI1 — the original drift-detector
            // diagnosis was correct. Keep as 16-byte consumer.
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

            // 0xF6: G_FILLRECT — solid-color rectangle. Use op02-aware
            //       wrapper so that when ROGUESQ_HLE_OP02_EXPERIMENTAL=1
            //       fires op_02 and that produces visible output, the
            //       subsequent black-clear (fillColor=0x00010001) is
            //       suppressed instead of wiping the experimental output.
            //       Otherwise the wrapper just falls through to fillRect_logged.
            gbi->map[0xF6] = &fillRect_op02_aware;

            // 0xF7: G_SETFILLCOLOR — override under ROGUESQ_HLE_FORCE_VISIBLE.
            gbi->map[0xF7] = &setFillColor_overridden;

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
