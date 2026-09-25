//
// RT64 — F3DFACTOR5 RDP-state handlers
//
// The Factor 5 GBI module's RDP-state / raster handlers (othermode, scissor,
// tile, combine, texrect, load-block/tile/TLUT, set-texture/color-image),
// split out of rt64_gbi_f3dfactor5.cpp for readability. These wrap the base
// GBI_F3D/GBI_RDP handlers with Factor-5-specific logging, guards, and CI4
// reinterpretation. Used by BOTH the software-T&L and native render paths
// (this is where the cinematic's textures get bound), so it stays regardless
// of which geometry path wins.
//
// First increment: the self-contained othermode/scissor setters. The rest of
// the block (setTile/setCombine/texrect/load*/set*Image + the shared CI4-track
// state s_ci4_* + clamp_load_subscripts) follows — that cluster is internally
// coupled but externally clean, so it moves here wholesale.
//

#include "rt64_gbi_f3dfactor5_internal.h"

#include "hle/rt64_state.h"
#include "hle/rt64_rdp.h"

#include "rt64_gbi_f3d.h"
#include "rt64_gbi_rdp.h"

#include "common/rt64_diag_bounds.h"

extern "C" void rt64_f5_desync_dump(const char *why, uint32_t badW1);
extern "C" volatile unsigned g_f5_heur[4];

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>

namespace RT64 {
    namespace GBI_F3DFACTOR5 {

        // Cross-module globals the moved handlers read (g_current_scene, g_mempak_hob_base,
        // g_scene_obj_count, g_most_drawn_fb, g_explosion_hold, s_cart_pass) are declared in the
        // internal header.

        // ROGUESQ_LOG_CI4_TMEM diag state: remember the last CI4 TLUT load (its tile-descriptor
        // slot + the tmem word it deposited the palette at) so the loadBlock/setTile trace can
        // print the palette-bank<->TLUT-tmem mapping for a target model texture. Diag only.
        static int s_ci4tm_tlut_tile = -1;
        static int s_ci4tm_tlut_tmem = -1;
        static int s_ci4tm_arm = 0;          // >0: a target block loaded, next setTile logs the render tile

        // ROGUESQ_LOG_CI4_TMEM=1 enables; ROGUESQ_CI4_TMEM_SRC=0xADDR overrides the target
        // texture source (default 0x555490, the dominant gameplay CI4 hull texture).
        static bool ci4tm_on() {
            static int s = -1;
            if (s < 0) { const char* v = std::getenv("ROGUESQ_LOG_CI4_TMEM"); s = (v && *v && v[0] != '0') ? 1 : 0; }
            return s == 1;
        }
        static uint32_t ci4tm_target() {
            static uint32_t t = 0xFFFFFFFFu;
            if (t == 0xFFFFFFFFu) { const char* v = std::getenv("ROGUESQ_CI4_TMEM_SRC"); t = (v && *v) ? (uint32_t)strtoul(v, nullptr, 0) : 0x555490u; }
            return t & 0x00FFFFFFu;
        }

        void setOtherModeH_logged(State *state, DisplayList **dl) {
            static int s_count = 0;
            if (gbi_log_enabled() && (++s_count <= 4)) {
                std::fprintf(stderr,
                    "[gbi-f5] setOtherModeH #%d shift=%u length=%u w1=0x%08X\n",
                    s_count, (*dl)->p0(8, 8), (*dl)->p0(0, 8), (*dl)->w1);
                std::fflush(stderr);
            }
            // ROGUESQ_LOG_OTHERMODE=1: dedicated (non-LOG_GBI) trace of every setOtherModeH write —
            // size/off/data and whether it TARGETS the textfilt field (bits 12-13), plus the resulting
            // textfilt after the write. Diagnoses why model surfaces render point when hardware filters
            // bilinearly: does the game ever write bilerp, and does F5's shift/length decode land it?
            {
                static int s_om = -1;
                if (s_om == -1) { const char* v = std::getenv("ROGUESQ_LOG_OTHERMODE"); s_om = (v && *v && v[0] != '0') ? 0 : -2; }
                if (s_om >= 0 && s_om < 200) {
                    ++s_om;
                    const uint32_t size = (*dl)->p0(0, 8), off = (*dl)->p0(8, 8), data = (*dl)->w1;
                    const bool touchesTF = (off <= 12) && (off + size > 12);   // covers bit 12 (textfilt)
                    GBI_F3D::setOtherModeH(state, dl);
                    std::fprintf(stderr,
                        "[othermode] size=%u off=%u data=0x%08X touchesTEXTFILT=%d => H=0x%08X textfilt=%u\n",
                        size, off, data, (int)touchesTF, state->rdp->otherMode.H,
                        (state->rdp->otherMode.H >> 12) & 3);
                    std::fflush(stderr);
                    return;
                }
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
            // ROGUESQ_LOG_OTHERMODE: the FULL othermode setter (G_RDPSETOTHERMODE) is the baseline that
            // carries textfilt. Log the raw w0/w1 and the resulting textfilt so we can see whether the
            // game's baseline requests bilerp (bits 12-13) and whether we decode it.
            {
                static int s_om = -1;
                if (s_om == -1) { const char* v = std::getenv("ROGUESQ_LOG_OTHERMODE"); s_om = (v && *v && v[0] != '0') ? 0 : -2; }
                if (s_om >= 0 && s_om < 40) {
                    ++s_om;
                    GBI_RDP::setOtherMode(state, dl);
                    std::fprintf(stderr,
                        "[othermode-FULL] w0=0x%08X w1=0x%08X => H=0x%08X textfilt=%u textlut=%u cyc=%u\n",
                        (*dl)->w0, (*dl)->w1, state->rdp->otherMode.H,
                        (state->rdp->otherMode.H >> 12) & 3, (state->rdp->otherMode.H >> 14) & 3,
                        (state->rdp->otherMode.H >> 20) & 3);
                    std::fflush(stderr);
                    return;
                }
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

        void setTile_logged(State *state, DisplayList **dl) {
            // ROGUESQ_LOG_CI4_TMEM: after the target CI4 block loads (s_ci4tm_arm), log the next few
            // render-tile setTiles so we can compare the render tile's palette BANK against where
            // loadTLUT actually deposited the palette (s_ci4tm_tlut_tmem). A bank mismatch => wrong
            // palette; a match with a still-streaky image => the pre-swizzled-source read.
            if (s_ci4tm_arm > 0) {
                --s_ci4tm_arm;
                std::fprintf(stderr,
                    "[ci4-tmem] RENDER-TILE tile=%u fmt=%u siz=%u line=%u tmem=%u pal(bank)=%u masks=%u maskt=%u cms=%u cmt=%u  (TLUT deposited at tmem=%d)\n",
                    (*dl)->p1(24, 3), (*dl)->p0(21, 3), (*dl)->p0(19, 2), (*dl)->p0(9, 9),
                    (*dl)->p0(0, 9), (*dl)->p1(20, 4), (*dl)->p1(4, 4) /*masks*/, (*dl)->p1(14, 4) /*maskt*/,
                    (*dl)->p1(8, 2) /*cms*/, (*dl)->p1(18, 2) /*cmt*/, s_ci4tm_tlut_tmem);
                std::fflush(stderr);
            }
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
            // ROGUESQ_CI4_FROM_I4=1: Factor 5 declares the cinematic render tile as
            // I4 (fmt=4 siz=0) but loads a 16-color TLUT and does the palette lookup
            // via the 0xFC11FE23 combiner. RT64's pipeline sees I4 → decodes as
            // intensity, ignoring the TLUT → garbled/blocky color. When a CI4-sized
            // TLUT loaded just before this render-tile setup, rewrite fmt I4 -> CI
            // (fmt bits w0[23:21] = 2) so RT64 applies the palette. Scoped by the
            // recent-TLUT flag to avoid touching genuine I4 (font) textures.
            {
                static int s_ci4 = -1;
                if (s_ci4 < 0) { const char* v = std::getenv("ROGUESQ_CI4_FROM_I4"); s_ci4 = (v && *v && v[0] != '0') ? 1 : 0; }
                const uint8_t rtile = (*dl)->p1(24, 3);
                const uint8_t rfmt  = (*dl)->p0(21, 3);
                const uint8_t rsiz  = (*dl)->p0(19, 2);
                if (s_ci4 && s_ci4_tlut_recent > 0) {
                    // DIAG: show every setTile inside the TLUT window so we can see the
                    // cinematic's actual render-tile params (not gated by LOG_GBI).
                    static int s_dl = 0;
                    if (++s_dl <= 24) { std::fprintf(stderr, "[ci4-diag] in-TLUT-window setTile tile=%u fmt=%u siz=%u recent=%d\n", rtile, rfmt, rsiz, s_ci4_tlut_recent); std::fflush(stderr); }
                    // Reinterpret a 4-bit render tile (any tile) declared as I -> CI so
                    // RT64 applies the loaded palette.
                    if (rfmt == 4 && rsiz == 0) {
                        (*dl)->w0 = ((*dl)->w0 & ~(0x7u << 21)) | (0x2u << 21);  // fmt I(4) -> CI(2)
                        static int s_rl = 0;
                        if (++s_rl <= 6) { std::fprintf(stderr, "[gbi-f5] reinterpret render tile %u I4 -> CI4 (TLUT-active)\n", rtile); std::fflush(stderr); }
                    }
                    // ROGUESQ_CI4_TILESIZE=1: synthesize a setTileSize for the render
                    // tile. TMEM-dump proved the load is correct, so the garble is
                    // RT64 mapping tex-coords with a wrong W×H. Derive W from the tile
                    // line (line 64-bit words -> 16 texels/word for 4b), H from the
                    // loaded block. Emit AFTER the game's setTile so it takes effect.
                    static int s_ts = -1;
                    if (s_ts < 0) { const char* v = std::getenv("ROGUESQ_CI4_TILESIZE"); s_ts = (v && *v && v[0] != '0') ? 1 : 0; }
                    if (s_ts && rfmt == 4 && rsiz == 0 && s_ci4_last_block_words > 0) {
                        const uint16_t rline = (*dl)->p0(9, 9);
                        int W = rline > 0 ? (int)rline * 16 : 32;          // 4b: 16 texels per 64-bit word
                        int H = (s_ci4_last_block_words * 4) / (W > 0 ? W : 32); // 16b words *4 = CI4 texels
                        if (H < 1) H = 1;
                        // Apply to the game's setTile first, then override size.
                        GBI_RDP::setTile(state, dl);
                        state->rdp->setTileSize(rtile, 0, 0, (uint16_t)((W - 1) << 2), (uint16_t)((H - 1) << 2));
                        static int s_tl = 0;
                        if (++s_tl <= 6) { std::fprintf(stderr, "[gbi-f5] synth setTileSize tile=%u W=%d H=%d (blkwords=%d line=%u)\n", rtile, W, H, s_ci4_last_block_words, rline); std::fflush(stderr); }
                        if (s_ci4_tlut_recent > 0) --s_ci4_tlut_recent;
                        return;  // already issued setTile
                    }
                }
            }
            if (s_ci4_tlut_recent > 0) --s_ci4_tlut_recent;  // bound the CI4 window
            GBI_RDP::setTile(state, dl);
        }

        void setCombine_logged(State *state, DisplayList **dl) {
            // Track every distinct (w0, w1) pair the game emits for setCombine.
            // First 8 calls also get logged unconditionally so we can see
            // chronology. ROGUESQ_LOG_GBI=1 enables.
            if (gbi_log_enabled()) {
                static rt64diag::BoundedSet<uint64_t> s_seen;
                static int s_count = 0;
                const uint64_t key = (uint64_t((*dl)->w0) << 32) | uint64_t((*dl)->w1);
                const bool firstSeen = s_seen.insert(key);
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
                    // Default: rewrite D to index 6 (A_ONE) → alpha = (0-0)*0+1 = 1
                    // so opaque content reaches the framebuffer.
                    //
                    // Attribution exception: the all-7s combiner there draws CI4
                    // glyph text whose letter SHAPE lives in the texture's alpha
                    // coverage. Forcing alpha=1 fills each glyph quad solid →
                    // white squares (user-confirmed 2026-05-21). Instead rewrite
                    // D to index 1 (TEXEL0): alpha = (A-B)*0 + TEXEL0 = TEXEL0,
                    // so the glyph's own coverage cuts out the letter shape.
                    const uint32_t newD = (g_current_scene == F5_SCENE_ATTRIBUTION) ? 0x1u : 0x6u;
                    const uint32_t patched_w1 = (w1 & ~(0x7u << 9)) | (newD << 9);
                    static int s_patched = 0;
                    if (gbi_log_enabled() && (++s_patched <= 4)) {
                        std::fprintf(stderr,
                            "[gbi-f5] alpha-patch setCombine w0=0x%08X w1=0x%08X -> w1=0x%08X (D=%u, attrib=%d)\n",
                            w0, w1, patched_w1, newD, (g_current_scene == F5_SCENE_ATTRIBUTION));
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

        // fullSync_logged stays in core (geometry: render_scene_objects + s_native_primed).

        // fillRect (op 0xF6) — solid color fills. Logs draw geometry.
        void fillRect_logged(State *state, DisplayList **dl) {
            if (state->rdp->colorImage.address == 0x76A000u) {
                ++s_attrib_fills_frame;
                // De-flicker: attribution alternates a FULL (~286 glyphs) and a
                // PARTIAL (~131) frame; both clear 0x76A000, so the partial frame
                // wipes the full text → flicker. On the partial frame (i.e. the
                // previous frame was full), skip the clear so the partial glyphs
                // accumulate onto the retained full frame and the display stays
                // complete. The full frames still clear+redraw, so the target is
                // always re-established (avoids the all-frames-skip → black bug).
                // ROGUESQ_ATTRIB_DEFLICKER=0 disables.
                static int s_deflicker = -1;
                if (s_deflicker < 0) {
                    const char* e = std::getenv("ROGUESQ_ATTRIB_DEFLICKER");
                    s_deflicker = (e && e[0] == '0') ? 0 : 1;
                }
                if (s_deflicker && g_current_scene == F5_SCENE_ATTRIBUTION && s_attrib_prev_glyphs >= 200) {
                    ++g_f5_heur[3];
                    return;  // partial frame: keep the prior full frame's content
                }
            }
            static int s_count = 0;
            if (gbi_log_enabled() && (++s_count <= 8)) {
                std::fprintf(stderr,
                    "[gbi-f5] fillRect #%d ulx=%u uly=%u lrx=%u lry=%u\n",
                    s_count,
                    (*dl)->p1(12, 12), (*dl)->p1(0, 12),
                    (*dl)->p0(12, 12), (*dl)->p0(0, 12));
                std::fflush(stderr);
            }
            // ROGUESQ_FILL_WHITE_TO_BLACK=1: the game clears the presented buffer
            // (0x76A000) to WHITE (0xFFFCFFFC) then renders the scene on top; our
            // HLE no-ops the op_02 scene geometry, so only the white survives = white
            // screen. Force the white clear to black so any faint sprite/geometry that
            // DOES render becomes visible. Diagnostic for the white-bg symptom.
            {
                static int s_w2b = -1;
                if (s_w2b < 0) { const char* v = std::getenv("ROGUESQ_FILL_WHITE_TO_BLACK"); s_w2b = (v && *v && v[0] != '0') ? 1 : 0; }
                if (s_w2b) {
                    const uint32_t fc = state->rdp->fillColorStack[state->rdp->fillColorStackSize - 1];
                    if (fc == 0xFFFCFFFCu || fc == 0xFFFEFFFEu || fc == 0xFFFFFFFFu) {
                        state->rdp->setFillColor(0x00010001u);  // RGBA5551 black, alpha=1
                    }
                }
            }
            // DIAG (ROGUESQ_LOG_FILL=1): sample the actual clear color + rect size +
            // target buffer across the whole run, to see what the cinematic/logo
            // buffers are cleared to (chasing the white-bg-not-black symptom).
            { static int s_fd = -1, s_fc = 0; if (s_fd == -1) { const char* v = std::getenv("ROGUESQ_LOG_FILL"); s_fd = (v && *v && v[0] != '0') ? 0 : -2; }
              if (s_fd >= 0) { ++s_fc;
                const uint32_t lrx = (*dl)->p0(12, 12), lry = (*dl)->p0(0, 12);
                const bool big = (lrx >= 0x800 /* >=512 in 10.2 fixed (lrx/4>=512?) */) || (lry >= 0x800);
                if ((big || (s_fc & 31) == 0) && s_fd < 200) { ++s_fd;
                  const uint32_t fc = state->rdp->fillColorStack[state->rdp->fillColorStackSize - 1];
                  std::fprintf(stderr, "[fill-diag] cimg=0x%06X fillColor=0x%08X rect=(%u,%u)-(%u,%u)\n",
                    state->rdp->colorImage.address, fc,
                    (*dl)->p1(12,12), (*dl)->p1(0,12), lrx, lry); std::fflush(stderr); } } }
            GBI_RDP::fillRect(state, dl);
        }

        void texrectLLE_guarded(State *state, DisplayList **dl) {
            // Track per-color-buffer draw ACTIVITY with exponential decay → publish the
            // most-active buffer as g_most_drawn_fb. This is the buffer the game is
            // actively rendering into NOW (phase-adaptive: title buffer during the title,
            // explosion buffer 0x290000 during the explosion). The present queue (mode 4)
            // presents it, so offscreen-rendered content reaches the screen.
            {
                const uint32_t cimg = state->rdp->colorImage.address & 0x00FFFFFFu;
                if (cimg >= 0x100000u) {  // plausible fb
                    static std::unordered_map<uint32_t, float> s_act;
                    static std::unordered_map<uint32_t, uint32_t> s_actw;  // cimg -> its color-image width
                    static int s_tick = 0;
                    float &a = s_act[cimg];
                    a += 1.0f;
                    s_actw[cimg] = state->rdp->colorImage.width;
                    if ((++s_tick & 0xFF) == 0) {  // decay every 256 texrects → favor recent
                        for (auto &kv : s_act) kv.second *= 0.5f;
                    }
                    // Update the published most-active fb (+ its width) when this buffer leads. The
                    // width lets the host present path (rt64_render_context menu fix) set VI_WIDTH to
                    // match, so a 512-wide menu buffer isn't scanned as 640/1024 -> black. Published
                    // here (gfx thread, owns the map) so the present thread never iterates it.
                    static uint32_t s_bestFb = 0; static float s_bestAct = 0;
                    if (a >= s_bestAct) { s_bestAct = a; s_bestFb = cimg; g_most_drawn_fb = cimg; g_most_drawn_fb_width = state->rdp->colorImage.width; }
                    else if (cimg == s_bestFb) { s_bestAct = a; }  // track the leader's decay
                    // Stamp the leader's last texrect so the present-side menu fix can tell a live
                    // 512-wide menu buffer from a stale one (the mission crawl draws only tris).
                    if (cimg == s_bestFb) {
                        g_most_drawn_fb_ms = (unsigned long long)std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now().time_since_epoch()).count();
                    }
                    // Periodically re-scan for the true max (leader may have decayed below another).
                    if ((s_tick & 0x3F) == 0) {
                        uint32_t mf = 0; float mx = 0;
                        for (auto &kv : s_act) if (kv.second > mx) { mx = kv.second; mf = kv.first; }
                        if (mf) { s_bestFb = mf; s_bestAct = mx; g_most_drawn_fb = mf; g_most_drawn_fb_width = s_actw[mf]; }
                    }
                }
            }
            // ROGUESQ_MEASURE_FB=1: decisive mesh-vs-billboard test. If a texrect is
            // consuming a flipbook-range texture (0x4Cxxxx-0x52xxxx), the explosion is
            // 2D BILLBOARDS (texrects), and we log its on-screen px size. If this never
            // fires while the flipbook loads, the explosion is triangle/mesh geometry.
            {
                static int s_mf = -1;
                if (s_mf == -1) { const char* v = std::getenv("ROGUESQ_MEASURE_FB"); s_mf = (v && *v && v[0] != '0') ? 0 : -2; }
                const uint32_t tsrc = state->rdp->texture.address & 0x00FFFFFFu;
                if (s_mf >= 0 && s_mf < 24 && tsrc >= 0x4C0000u && tsrc < 0x520000u) {
                    ++s_mf;
                    const int32_t ulx = (*dl)[0].p1(12, 12), uly = (*dl)[0].p1(0, 12);
                    const int32_t lrx = (*dl)[0].p0(12, 12), lry = (*dl)[0].p0(0, 12);
                    const uint8_t rtile = (*dl)[0].p1(24, 3);
                    const int32_t s  = (*dl)[1].p0(16, 16), t = (*dl)[1].p0(0, 16);
                    const int32_t ds = (*dl)[1].p1(16, 16), dtv = (*dl)[1].p1(0, 16);
                    const auto& T = state->rdp->tiles[rtile];
                    const auto& prim = state->rdp->primColorStack[state->rdp->primColorStackSize - 1];
                    const auto& env  = state->rdp->envColorStack[state->rdp->envColorStackSize - 1];
                    const auto& comb = state->rdp->colorCombinerStack[state->rdp->colorCombinerStackSize - 1];
                    std::fprintf(stderr, "[measure-fb] FLIPBOOK texrect tex=0x%06X cimg=0x%06X px=(%d,%d %dx%d) tile=%u st=(%d,%d) dsdt=(%d,%d) | tileFmt=%u siz=%u line=%u th=%u tw=%u | prim=(%.2f %.2f %.2f %.2f) env=(%.2f %.2f %.2f %.2f) combL=0x%08X combH=0x%08X otherL=0x%08X\n",
                        tsrc, state->rdp->colorImage.address & 0x00FFFFFFu,
                        ulx >> 2, uly >> 2, (lrx - ulx) >> 2, (lry - uly) >> 2,
                        rtile, s, t, ds, dtv, T.fmt, T.siz, T.line, T.uls, T.lrs,
                        (float)prim.x, (float)prim.y, (float)prim.z, (float)prim.w,
                        (float)env.x, (float)env.y, (float)env.z, (float)env.w,
                        comb.L, comb.H, state->rdp->otherMode.L);
                    std::fflush(stderr);
                }
            }
            // ROGUESQ_FX_PROBE: RGBA32 (fmt0/siz3) effect texrects — the animated sprite
            // billboards if they take the texrect path. Format-keyed, address-independent.
            {
                static int s_fxt = -1;
                if (s_fxt == -1) { const char* v = std::getenv("ROGUESQ_FX_PROBE"); s_fxt = (v && *v && v[0] != '0') ? 0 : -2; }
                if (s_fxt >= 0) {
                    const uint8_t rtile = (*dl)[0].p1(24, 3);
                    const auto& T = state->rdp->tiles[rtile];
                    {
                        static uint32_t s_dt = 0; ++s_dt;
                        const auto& prim = state->rdp->primColorStack[state->rdp->primColorStackSize - 1];
                        const auto& comb = state->rdp->colorCombinerStack[state->rdp->colorCombinerStackSize - 1];
                        // Census of texrect TYPES: dedup by (fmt,siz,comb,otherL) so each kind of
                        // texrect prints once. The animated billboard sprites are whichever small
                        // rects flipbook their source texture; identify by fmt/siz + on-screen size.
                        static std::unordered_set<uint64_t> s_seent;
                        uint64_t key = ((uint64_t)T.fmt << 60) ^ ((uint64_t)T.siz << 56)
                                     ^ ((uint64_t)comb.L << 4) ^ ((uint64_t)state->rdp->otherMode.L << 24);
                        if (s_seent.size() < 80 && s_seent.insert(key).second) {
                            const int32_t ulx = (*dl)[0].p1(12, 12), uly = (*dl)[0].p1(0, 12);
                            const int32_t lrx = (*dl)[0].p0(12, 12), lry = (*dl)[0].p0(0, 12);
                            std::fprintf(stderr,
                                "[fx-rect] #%u tex=%06X fmt=%u siz=%u cimg=%06X px=(%d,%d %dx%d) prim=(%.2f %.2f %.2f %.2f) combL=%08X combH=%08X otherL=%08X otherH=%08X\n",
                                s_dt, state->rdp->texture.address & 0x00FFFFFFu, T.fmt, T.siz,
                                state->rdp->colorImage.address & 0x00FFFFFFu,
                                ulx >> 2, uly >> 2, (lrx - ulx) >> 2, (lry - uly) >> 2,
                                (float)prim.x, (float)prim.y, (float)prim.z, (float)prim.w,
                                comb.L, comb.H, state->rdp->otherMode.L, state->rdp->otherMode.H);
                            std::fflush(stderr);
                        }
                    }
                }
            }
            // ROGUESQ_DECODE_PPM=1: manually decode the flipbook CI4 (source bytes +
            // TMEM TLUT) to an RGB PPM file. Ground truth: if MY decode is a coherent
            // flame, the data+palette are good and RT64's sampler is the only bug.
            {
                static int s_pp = -1;
                if (s_pp == -1) { const char* v = std::getenv("ROGUESQ_DECODE_PPM"); s_pp = (v && *v && v[0] != '0') ? 0 : -2; }
                const uint32_t tsrc = state->rdp->texture.address & 0x00FFFFFFu;
                const bool isScratch = (tsrc >= 0x713000u && tsrc < 0x71B000u);  // de-swizzled scratch
                if (s_pp >= 0 && s_pp < 6 && ((tsrc >= 0x4C0000u && tsrc < 0x520000u) || isScratch)) {
                    const int W = 64, H = 64;
                    const uint8_t* ram = state->RDRAM;
                    // Read the 16-color palette from its RDRAM SOURCE (captured at
                    // loadTLUT; TMEM is deferred). 16 RGBA16 entries, big-endian.
                    uint16_t pal[16];
                    for (int i = 0; i < 16; ++i) {
                        uint32_t pa = (s_last_tlut_src + (uint32_t)i * 2) & 0x00FFFFFFu;
                        pal[i] = (uint16_t)((ram[(pa) ^ 3] << 8) | ram[(pa + 1) ^ 3]);
                    }
                    char path[256]; std::snprintf(path, sizeof(path), "E:/Projects/RogueSquadron64Recomp/flip_%d_%06X.ppm", s_pp, tsrc);
                    FILE* fp = std::fopen(path, "wb");
                    if (fp) {
                        std::fprintf(fp, "P6\n%d %d\n255\n", W, H);
                        // ROGUESQ_DECODE_SWAP=1: apply the N64 odd-row 32-bit word
                        // interleave (XOR byte offset with 4 on odd rows) to un-swizzle
                        // a TMEM-formatted source. Tests whether that yields a clean frame.
                        static int s_sw = -1;
                        if (s_sw < 0) { const char* v = std::getenv("ROGUESQ_DECODE_SWAP"); s_sw = (v && *v && v[0] != '0') ? 1 : 0; }
                        for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) {
                            uint32_t byteoff = (uint32_t)(y * W + x) / 2;
                            // Scratch is already de-swizzled — read it linearly. For the
                            // raw source, apply the swap when ROGUESQ_DECODE_SWAP=1.
                            if (!isScratch && s_sw && (y & 1)) byteoff ^= 4;
                            uint8_t b = ram[(tsrc + byteoff) ^ 3];
                            int idx = (x & 1) ? (b & 0xF) : (b >> 4);
                            uint16_t rgba = pal[idx];
                            uint8_t r = (uint8_t)(((rgba >> 11) & 0x1F) << 3);
                            uint8_t g = (uint8_t)(((rgba >> 6) & 0x1F) << 3);
                            uint8_t bl = (uint8_t)(((rgba >> 1) & 0x1F) << 3);
                            std::fputc(r, fp); std::fputc(g, fp); std::fputc(bl, fp);
                        }
                        std::fclose(fp);
                        std::fprintf(stderr, "[decode-ppm] wrote %s (palSrc=0x%06X pal[0,1,15]=%04X,%04X,%04X)\n",
                            path, s_last_tlut_src, pal[0], pal[1], pal[15]);
                        std::fflush(stderr);
                    }
                    ++s_pp;
                }
            }
            if (texrect_copy_undefined_tile(state, dl)) {
                (*dl)++;  // skip the 8-byte texcoord follow-up to keep dispatch aligned
                return;
            }
            // Track per-active-CIMG-address texrect counts to see which fbs
            // actually receive draws. ROGUESQ_LOG_CIMG=1 enables.
            {
                static rt64diag::BoundedMap<uint32_t, uint32_t> s_drawsPerFb;
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
                static rt64diag::BoundedSet<Key, KeyHash> s_seen;
                const auto& comb = state->rdp->colorCombinerStack[state->rdp->colorCombinerStackSize - 1];
                Key k{state->rdp->colorImage.address, comb.L, comb.H};
                if (s_seen.insert(k)) {
                    std::fprintf(stderr,
                        "[gbi-f5] texrect-uses fb=0x%06X combL=0x%08X combH=0x%08X otherL=0x%08X\n",
                        k.fb, k.l, k.h, state->rdp->otherMode.L);
                    std::fflush(stderr);
                }
            }
            if (state->rdp->colorImage.address == 0x76A000u) {
                ++s_attrib_glyphs_frame;
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
            // Compact per-texrect coordinate dump (ROGUESQ_LOG_GBI=1, first 48).
            // rect bounds in word0; LLE texcoord step (s,t,dsdx,dtdy) in word1.
            // Used to diagnose clipped glyphs: if (lry-uly) is tiny or dtdy is
            // huge, only the top sliver of each glyph maps. Coords are 10.2
            // fixed-point (>>2 = pixels); dsdx/dtdy are s5.10.
            if (gbi_log_enabled()) {
                static int s_tc = 0;
                if (++s_tc <= 48) {
                    const int32_t ulx = (*dl)[0].p1(12, 12), uly = (*dl)[0].p1(0, 12);
                    const int32_t lrx = (*dl)[0].p0(12, 12), lry = (*dl)[0].p0(0, 12);
                    const uint32_t tile = (*dl)[0].p1(24, 3);
                    const int32_t s  = (*dl)[1].p0(16, 16), t = (*dl)[1].p0(0, 16);
                    const int32_t ds = (*dl)[1].p1(16, 16), dt = (*dl)[1].p1(0, 16);
                    std::fprintf(stderr,
                        "[gbi-f5] texrect-xy #%d fb=0x%06X tile=%u rect=(%d,%d)-(%d,%d) px=(%d,%d %dx%d) st=(%d,%d) dsdt=(%d,%d)\n",
                        s_tc, state->rdp->colorImage.address, tile,
                        ulx, uly, lrx, lry,
                        ulx >> 2, uly >> 2, (lrx - ulx) >> 2, (lry - uly) >> 2,
                        s, t, ds, dt);
                    std::fflush(stderr);
                }
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
            // ROGUESQ_FIRE_MAGENTA=1: existence test for the explosion fire. Convert
            // every texrect targeting a CINEMATIC buffer (not the 0x76A000 menu) into
            // a magenta fillRect. If magenta appears where the fire should be, the
            // texrect coords + presented-buffer targeting are CORRECT and the problem
            // is texture decode/blend (root cause C). If nothing shows, the fire
            // texrects render into a non-presented offscreen buffer (root cause B).
            {
                static int s_fm = -1;
                if (s_fm < 0) { const char* v = std::getenv("ROGUESQ_FIRE_MAGENTA"); s_fm = (v && *v && v[0] != '0') ? 1 : 0; }
                const uint32_t cimg = state->rdp->colorImage.address & 0x00FFFFFFu;
                if (s_fm && cimg != 0x76A000u && cimg >= 0x200000u) {
                    state->rdp->setFillColor(0xF80FF80F);  // RGBA5551 magenta opaque
                    int32_t ulx = (*dl)[0].p1(12, 12), uly = (*dl)[0].p1(0, 12);
                    int32_t lrx = (*dl)[0].p0(12, 12), lry = (*dl)[0].p0(0, 12);
                    state->rdp->fillRect(ulx, uly, lrx, lry);
                    (*dl)++;  // consume the LLE texrect follow-up word
                    return;
                }
            }
            // Explosion-buffer texrects: measure or override the blend.
            // ROGUESQ_FIRE_PROBE=1 logs the REAL blender (otherMode.L) + combiner the
            // game set, so we honor it instead of guessing. ROGUESQ_FIRE_ADDITIVE=1
            // forces an additive blender — but additive over the WHITE cinematic bg
            // washes to white (confirmed), so it's off by default now.
            {
                const uint32_t cimg = state->rdp->colorImage.address & 0x00FFFFFFu;
                const bool fireBuf = (cimg >= 0x200000u && cimg != 0x76A000u);
                static int s_fprobe = -1;
                if (s_fprobe < 0) { const char* v = std::getenv("ROGUESQ_FIRE_PROBE"); s_fprobe = (v && *v && v[0] != '0') ? 1 : 0; }
                if (s_fprobe) {
                    int ci = state->rdp->colorCombinerStackSize - 1;
                    uint32_t cL = (ci >= 0) ? state->rdp->colorCombinerStack[ci].L : 0;
                    uint32_t cH = (ci >= 0) ? state->rdp->colorCombinerStack[ci].H : 0;
                    // Key distinct (cimg, otherL, otherH, comb) tuples so a single
                    // config (e.g. attribution text) takes one slot and we see variety.
                    static std::unordered_set<uint64_t> s_seen;
                    uint64_t key = (uint64_t)cimg ^ ((uint64_t)state->rdp->otherMode.L << 8)
                                 ^ ((uint64_t)cL << 20) ^ ((uint64_t)cH << 40);
                    const char* ph =
                        (cimg == 0x66A000 || cimg == 0x5D4000) ? "ATTRIB" :
                        (cimg == 0x6DD000 || cimg == 0x6BA000) ? "N64LOGO" :
                        (cimg == 0x62B800 || cimg == 0x695C00) ? "CINEMATIC" :
                        (cimg == 0x290000 || cimg == 0x795C00 || cimg == 0x240000) ? "OFFSCREEN" : "OTHER";
                    if (s_seen.size() < 80 && s_seen.insert(key).second) {
                        std::fprintf(stderr,
                            "[fire-probe] %s cimg=0x%06X otherL=0x%08X otherH=0x%08X comb=0x%08X%08X\n",
                            ph, cimg, state->rdp->otherMode.L, state->rdp->otherMode.H, cH, cL);
                        std::fflush(stderr);
                    }
                }
                static int s_fadd = -1;
                if (s_fadd < 0) { const char* v = std::getenv("ROGUESQ_FIRE_ADDITIVE"); s_fadd = (v && *v && v[0] != '0') ? 1 : 0; }
                if (s_fadd && fireBuf) {
                    state->rdp->setOtherMode(0x00584040u, 0x0C084000u);
                }
            }
            // ROGUESQ_TEXRECT_PROBE=1: log each medal/insignia texrect (fmt4 src 0x62xxxx) that
            // actually REACHES RT64's draw, with the real colorImage target. Distinguishes
            // "interpreter never reaches it" (no log) from "drawn to a non-menu buffer" (log w/ odd cimg).
            {
                static const bool s_trp = []{ const char* v = std::getenv("ROGUESQ_TEXRECT_PROBE"); return v && *v && v[0] != '0'; }();
                if (s_trp) {
                    const uint32_t tsrc = state->rdp->texture.address & 0x00FFFFFFu;
                    if (tsrc >= 0x620000u && tsrc < 0x628000u) {
                        static int s_tn = 0;
                        // Per-frame medal-draw count: log once per displayListCounter so a run shows
                        // whether the medals draw EVERY frame (gap==1) or intermittently (gap>>1).
                        static uint32_t s_lastDl = 0xFFFFFFFFu; static int s_perFrame = 0;
                        if (state->displayListCounter != s_lastDl) {
                            if (s_lastDl != 0xFFFFFFFFu && s_tn < 200)
                                std::fprintf(stderr, "[texrect-frame] dl#%u medalRects=%d cimg=0x%08X\n", s_lastDl, s_perFrame, state->rdp->colorImage.address);
                            s_lastDl = state->displayListCounter; s_perFrame = 0;
                        }
                        ++s_perFrame;
                        if (++s_tn <= 12) {
                            const int32_t ulx = (*dl)[0].p1(12,12)>>2, uly = (*dl)[0].p1(0,12)>>2;
                            const int32_t lrx = (*dl)[0].p0(12,12)>>2, lry = (*dl)[0].p0(0,12)>>2;
                            std::fprintf(stderr, "[texrect-probe] #%d dl#%u DREW medal tex=0x%06X -> cimg=0x%08X w=%u px=(%d,%d)-(%d,%d)\n",
                                s_tn, (unsigned)state->displayListCounter, tsrc, state->rdp->colorImage.address, state->rdp->colorImage.width, ulx, uly, lrx, lry);
                            std::fflush(stderr);
                        }
                    }
                }
            }
            // ROGUESQ_MEDAL_FORCE=1: decisive visibility test — draw each medal/insignia texrect
            // (fmt4 0x62xxxx) as a bright OPAQUE green fillRect at its own coords. If green blocks
            // appear where the medals belong, the geometry/target/present is fine and the medal
            // texture/blend is the bug. If nothing shows, the layer is overwritten / not presented.
            {
                static const bool s_mforce = []{ const char* v = std::getenv("ROGUESQ_MEDAL_FORCE"); return v && *v && v[0] != '0'; }();
                if (s_mforce) {
                    const uint32_t tsrc = state->rdp->texture.address & 0x00FFFFFFu;
                    if (tsrc >= 0x620000u && tsrc < 0x628000u) {
                        state->rdp->setFillColor(0x07C107C1u);  // RGBA5551 bright green, opaque
                        int32_t ulx = (*dl)[0].p1(12,12), uly = (*dl)[0].p1(0,12);
                        int32_t lrx = (*dl)[0].p0(12,12), lry = (*dl)[0].p0(0,12);
                        state->rdp->fillRect(ulx, uly, lrx, lry);
                        (*dl)++;  // consume the LLE texrect follow-up word
                        return;
                    }
                }
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
            // A TLUT load with <=16 colors ((lrs>>2)+1) marks CI4 intent.
            if (((lrs >> 2) + 1) <= 16) { s_ci4_tlut_recent = 4; s_last_tlut_src = state->rdp->texture.address & 0x00FFFFFFu; }
            static int s_count = 0;
            if (gbi_log_enabled() && (++s_count <= 8)) {
                std::fprintf(stderr,
                    "[gbi-f5] loadTLUT #%d tile=%u uls=%u ult=%u lrs=%u lrt=%u (raw w0=0x%08X w1=0x%08X)\n",
                    s_count, tile, uls, ult, lrs, lrt, (*dl)->w0, (*dl)->w1);
                std::fflush(stderr);
            }
            state->rdp->loadTLUT(tile, uls, ult, lrs, lrt);
            if (ci4tm_on()) { s_ci4tm_tlut_tile = tile; s_ci4tm_tlut_tmem = state->rdp->tiles[tile].tmem; }
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
            // Split the draw call on a TMEM reload. RS64's skybox reloads ~40 distinct RGBA32 tiles into
            // the SAME tile/TMEM address within one draw call; because the render tile index never changes,
            // RT64 never splits the call, so every tile hashes the FINAL emulated-TMEM state and the whole
            // dome binds ONE texture (flat sky). Flushing pending geometry before the reload commits the
            // prior tile's faces (with the loads so far) as their own draw call, and marking the texture
            // dirty forces the next face to re-associate the new load and re-hash the tile. flush() no-ops
            // when no triangles are pending, so a load-pair-before-a-draw (e.g. TLUT+block) never
            // over-splits. Default on; ROGUESQ_F5_NO_TILE_SPLIT=1 disables. See plans/skybox-not-rendering-plan.md.
            static int s_split = -1;
            if (s_split < 0) { const char* e = std::getenv("ROGUESQ_F5_NO_TILE_SPLIT"); s_split = (e && e[0] == '1') ? 0 : 1; }
            if (s_split) state->flush();
            state->rdp->loadTile(tile, uls, ult, lrs, lrt);
            if (s_split) state->updateDrawStatusAttribute(DrawAttribute::Texture);
        }

        void loadBlock_guarded(State *state, DisplayList **dl) {
            // loadBlock has different fields: uls/ult are texture origin,
            // lrs is endpoint, dxt is row stride.
            const uint8_t tile = (*dl)->p1(24, 3);
            uint16_t uls = (*dl)->p0(12, 12);
            uint16_t ult = (*dl)->p0(0, 12);
            uint16_t lrs = (*dl)->p1(12, 12);
            uint16_t dxt = (*dl)->p1(0, 12);
            // ROGUESQ_CI4_FIXDXT=<hex>: the cinematic flipbook CI4 loadBlocks pass
            // dxt=0 (no per-row interleave). RT64's tile sampler applies the standard
            // odd-row 32-bit swap, so a dxt=0 load samples back as garbled noise.
            // Override dxt with the proper interleave value (default 0x400 = 2048/2
            // words-per-row for a 32-texel CI4 row) while a CI4 TLUT is active.
            {
                static int s_fixdxt = -1;
                if (s_fixdxt == -1) { const char* v = std::getenv("ROGUESQ_CI4_FIXDXT"); s_fixdxt = (v && *v) ? (int)strtoul(v, nullptr, 0) : -2; }
                // Scope to the FLIPBOOK range only (0x4Cxxxx-0x52xxxx) so the logo
                // (also CI4, renders clean with dxt=0) is untouched. 0x200 = the
                // odd-row interleave for a 64-wide CI4 loaded as 16b (2048/4 words).
                const uint32_t srcDx = state->rdp->texture.address & 0x00FFFFFFu;
                const bool fbRange = (srcDx >= 0x4C0000u && srcDx < 0x520000u);
                if (s_fixdxt >= 0 && fbRange && dxt == 0) {
                    dxt = (uint16_t)(s_fixdxt ? s_fixdxt : 0x200);
                }
            }
            if (lrs < uls) std::swap(uls, lrs);
            // Cap the block size at 2048 texels (max valid in 12-bit field).
            if (lrs - uls > 2048) lrs = uls + 2048;
            // Remember the block size (16b words loaded) for setTileSize synthesis.
            if (s_ci4_tlut_recent > 0) s_ci4_last_block_words = (int)(lrs - uls) + 1;
            static int s_count = 0;
            if (gbi_log_enabled() && (++s_count <= 8)) {
                std::fprintf(stderr,
                    "[gbi-f5] loadBlock #%d tile=%u uls=%u ult=%u lrs=%u dxt=%u\n",
                    s_count, tile, uls, ult, lrs, dxt);
                std::fflush(stderr);
            }
            // ROGUESQ_FX_PROBE: for the RGBA32 effect-sprite textures, log the image siz/fmt
            // (SETTIMG) vs the LOAD-tile siz/fmt. loadBlockOperation only splits into both TMEM
            // banks when the LOAD tile is siz3/fmt0; a 16b load idiom leaves the alpha bank empty.
            {
                static int s_fxl = -1;
                if (s_fxl == -1) { const char* v = std::getenv("ROGUESQ_FX_PROBE"); s_fxl = (v && *v && v[0] != '0') ? 0 : -2; }
                const uint32_t srcFx = state->rdp->texture.address & 0x00FFFFFFu;
                const bool rgbaImg = (state->rdp->texture.siz == 3 && state->rdp->texture.fmt == 0);
                if (s_fxl >= 0 && s_fxl < 24 && (rgbaImg || (srcFx >= 0x550000u && srcFx < 0x560000u))) {
                    ++s_fxl;
                    const LoadTile &lt = state->rdp->tiles[tile];
                    std::fprintf(stderr,
                        "[fx-load] src=0x%06X imgFmt=%u imgSiz=%u | loadtile[tile=%u fmt=%u siz=%u line=%u tmem=%u] uls=%u lrs=%u dxt=%u words=%d rgba32split=%d\n",
                        srcFx, state->rdp->texture.fmt, state->rdp->texture.siz,
                        tile, lt.fmt, lt.siz, lt.line, lt.tmem, uls, lrs, dxt, (int)(lrs - uls) + 1,
                        (lt.siz == 3 && lt.fmt == 0) ? 1 : 0);
                    std::fflush(stderr);
                }
            }
            // ROGUESQ_DUMP_TEX=1: when a CI4 flipbook frame loads, dump the SOURCE
            // bytes from RDRAM (texture.address) to measure whether the streamed
            // texture data is coherent CI4 (structured, non-zero) or garbage/unloaded
            // (the real cause of the flipbook noise, given the base CI4 decode works).
            {
                static int s_dt = -1;
                if (s_dt == -1) { const char* v = std::getenv("ROGUESQ_DUMP_TEX"); s_dt = (v && *v && v[0] != '0') ? 0 : -2; }
                const uint32_t srcChk = state->rdp->texture.address & 0x00FFFFFFu;
                const bool isFlipbook = (srcChk >= 0x4C0000u && srcChk < 0x520000u);
                if (s_dt >= 0 && s_dt < 16 && s_ci4_tlut_recent > 0 && isFlipbook) {
                    ++s_dt;
                    const uint32_t src = srcChk;
                    const uint8_t* ram = state->RDRAM;
                    int nz = 0; char hex[3 * 48 + 1]; int hp = 0;
                    for (int i = 0; i < 48; ++i) {
                        uint8_t b = ram[(src + i) ^ 3];
                        if (b) ++nz;
                        hp += std::snprintf(hex + hp, sizeof(hex) - hp, "%02X ", b);
                    }
                    std::fprintf(stderr, "[tex-dump] src=0x%08X fmt=%u siz=%u nz=%d/48 bytes: %s\n",
                        state->rdp->texture.address, state->rdp->texture.fmt, state->rdp->texture.siz, nz, hex);
                    std::fflush(stderr);
                }
            }
            // Split the draw call on TMEM reload (skybox one-texture bug; see loadTile_guarded).
            static int s_split = -1;
            if (s_split < 0) { const char* e = std::getenv("ROGUESQ_F5_NO_TILE_SPLIT"); s_split = (e && e[0] == '1') ? 0 : 1; }
            if (s_split) state->flush();
            state->rdp->loadBlock(tile, uls, ult, lrs, dxt);
            if (s_split) state->updateDrawStatusAttribute(DrawAttribute::Texture);
            // ROGUESQ_LOG_CI4_TMEM: full load->tile->TLUT->TMEM trace for the target CI4 model
            // texture (default 0x555490). Answers the standing open question: is the streak a
            // pre-swizzled source RT64 reads linearly, or a wrong palette bank? Prints the block
            // params, the load-tile descriptor, where loadTLUT deposited the palette, and the
            // first two CI4 rows of the RDRAM source vs the loaded TMEM (a swizzle shows as
            // swapped 32-bit halves on the odd row). Forces the (deferred) load to inspect TMEM.
            if (ci4tm_on()) {
                const uint32_t src = state->rdp->texture.address & 0x00FFFFFFu;
                if (src == ci4tm_target()) {
                    const LoadTile &lt = state->rdp->tiles[tile];
                    state->rdp->loadBlockOperation(state->rdp->tiles[tile], state->rdp->texture, false);
                    std::fprintf(stderr,
                        "[ci4-tmem] BLOCK src=0x%06X tile=%u dxt=%u imgsiz=%u words=%d | loadtile[fmt=%u siz=%u line=%u tmem=%u pal=%u masks=%u maskt=%u cms=%u cmt=%u] | lastTLUT[tile=%d tmem=%d src=0x%06X]\n",
                        src, tile, dxt, state->rdp->texture.siz, (int)(lrs - uls) + 1,
                        lt.fmt, lt.siz, lt.line, lt.tmem, lt.palette, lt.masks, lt.maskt, lt.cms, lt.cmt,
                        s_ci4tm_tlut_tile, s_ci4tm_tlut_tmem, s_last_tlut_src);
                    // 64-wide CI4 => 32 bytes/row. Dump 2 rows (64 bytes) of RDRAM source with the
                    // proven ^3 (32-bit big-endian) convention used elsewhere in this file, and TMEM
                    // as raw 64-bit words (8 words = 64 bytes, same convention as [tmem-dump]). A
                    // pre-swizzled source shows as odd-row (row1) 32-bit halves swapped between them.
                    const uint8_t* ram = state->RDRAM;
                    char sh[3 * 32 + 1]; int hp = 0;
                    for (int i = 0; i < 32; ++i) hp += std::snprintf(sh + hp, sizeof(sh) - hp, "%02X ", ram[(src + i) ^ 3]);
                    std::fprintf(stderr, "[ci4-tmem]   SRC  row0: %s\n", sh);
                    hp = 0; for (int i = 0; i < 32; ++i) hp += std::snprintf(sh + hp, sizeof(sh) - hp, "%02X ", ram[(src + 32 + i) ^ 3]);
                    std::fprintf(stderr, "[ci4-tmem]   SRC  row1: %s\n", sh);
                    std::fprintf(stderr, "[ci4-tmem]   TMEM w0..7: %016llX %016llX %016llX %016llX %016llX %016llX %016llX %016llX\n",
                        (unsigned long long)state->rdp->TMEM[0], (unsigned long long)state->rdp->TMEM[1],
                        (unsigned long long)state->rdp->TMEM[2], (unsigned long long)state->rdp->TMEM[3],
                        (unsigned long long)state->rdp->TMEM[4], (unsigned long long)state->rdp->TMEM[5],
                        (unsigned long long)state->rdp->TMEM[6], (unsigned long long)state->rdp->TMEM[7]);
                    s_ci4tm_arm = 3;  // log the next few setTiles (the render tile that binds this)
                    std::fflush(stderr);
                }
            }
            // ROGUESQ_DUMP_TEX: after a flipbook loadBlock, FORCE the (normally
            // deferred) load to execute and dump TMEM[0..3], so we can compare the
            // loaded TMEM to the coherent source bytes. If TMEM matches source -> the
            // load is fine and the garble is in SAMPLING (RT64 internal). If TMEM is
            // scrambled -> the LOAD is the bug (fixable here).
            {
                static int s_tm = -1;
                if (s_tm == -1) { const char* v = std::getenv("ROGUESQ_DUMP_TEX"); s_tm = (v && *v && v[0] != '0') ? 0 : -2; }
                const uint32_t srcChk = state->rdp->texture.address & 0x00FFFFFFu;
                if (s_tm >= 0 && s_tm < 8 && s_ci4_tlut_recent > 0 &&
                    srcChk >= 0x4C0000u && srcChk < 0x520000u) {
                    ++s_tm;
                    state->rdp->loadBlockOperation(state->rdp->tiles[tile], state->rdp->texture, false);
                    std::fprintf(stderr, "[tmem-dump] src=0x%06X TMEM[0..3]: %016llX %016llX %016llX %016llX\n",
                        srcChk,
                        (unsigned long long)state->rdp->TMEM[0], (unsigned long long)state->rdp->TMEM[1],
                        (unsigned long long)state->rdp->TMEM[2], (unsigned long long)state->rdp->TMEM[3]);
                    std::fflush(stderr);
                }
            }
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
            if (w1 == 0xFFFFFFFFu) {
                if (gbi_log_enabled()) {
                    static uint32_t s_rej_ff = 0;
                    if ((++s_rej_ff & 0xFF) == 1) {
                        std::fprintf(stderr, "[gbi-f5] setTIMG reject 0xFFFFFFFF count=%u\n", s_rej_ff);
                        std::fflush(stderr);
                    }
                }
                return;
            }
            // After RDP::maskAddress folds KSEG bits, address < 0x1000 is
            // almost certainly bogus (low RDRAM is libultra/PIF area).
            const uint32_t masked = w1 & 0x00FFFFFFu;
            if (masked < 0x1000u) {
                if (gbi_log_enabled()) {
                    static uint32_t s_rej_lo = 0;
                    if ((++s_rej_lo & 0xFF) == 1) {
                        std::fprintf(stderr, "[gbi-f5] setTIMG reject lo addr=0x%08X count=%u\n", w1, s_rej_lo);
                        std::fflush(stderr);
                    }
                }
                return;
            }
            // ROGUESQ_CI4_DESWIZZLE=1: the cinematic CI4 mosaic textures (0x4Cxxxx-
            // 0x52xxxx, e.g. the N64-logo tiles) are stored TMEM-SWIZZLED (odd rows
            // have their 32-bit halves swapped). RT64 doesn't apply the matching swap
            // -> streaky garble. PROVEN (decoded-pixel-diff): copying with byteoff^=4
            // on odd rows (32 B/row, 64-wide CI4) yields a clean image. Pre-swizzle the
            // source into a scratch ring (guest 0x80713000, 16x2KB, in the safe gap
            // between modelview matrix 0x80710040 and DL 0x80720000) and redirect the
            // texture there so RT64 loads de-swizzled (clean) data.
            {
                static int s_dsw = -1;
                if (s_dsw < 0) { const char* v = std::getenv("ROGUESQ_CI4_DESWIZZLE"); s_dsw = (v && *v && v[0] != '0') ? 1 : 0; }
                // De-swizzle is applied ONLY to the explosion FIRE CI4 tiles (~0x53xxxx-
                // 0x57xxxx). The N64-logo CI4 mosaic (0x4Cxxxx-0x52xxxx) is NOT de-swizzled
                // (removed per user — it looked wrong on the logo; RT64 handles it natively).
                // Explosion-hold trigger: when a fire CI4 tile loads, arm the cine-pace crawl.
                if (masked >= 0x530000u && masked < 0x580000u && s_ci4_tlut_recent > 0) {
                    g_explosion_hold = 90;  // frames to crawl (~30s linger at 3fps; re-arms while fire loads)
                }
                if (s_dsw && masked >= 0x530000u && masked < 0x580000u && s_ci4_tlut_recent > 0) {
                    static int s_slot = 0;
                    const uint32_t SCRATCH_BASE = 0x00713000u;
                    const int SLOTS = 22, SLOT_SZ = 0x800;  // 2KB/64x64 CI4; 22*2KB ends <0x71E000 (F5 vertex scratch)
                    uint32_t dst = SCRATCH_BASE + (uint32_t)(s_slot % SLOTS) * SLOT_SZ;
                    s_slot++;
                    uint8_t* ram = state->RDRAM;
                    for (uint32_t i = 0; i < (uint32_t)SLOT_SZ; ++i) {
                        uint32_t srcI = ((i / 32u) & 1u) ? (i ^ 4u) : i;   // odd-row 32-bit swap
                        ram[(dst + i) ^ 3] = ram[(masked + srcI) ^ 3];
                    }
                    (*dl)->w1 = (w1 & 0xFF000000u) | dst;   // redirect to de-swizzled scratch
                    if (gbi_log_enabled()) {
                        static int s_dl = 0;
                        if (++s_dl <= 6) { std::fprintf(stderr, "[gbi-f5] deswizzle CI4 0x%06X -> 0x%06X\n", masked, dst); std::fflush(stderr); }
                    }
                }
            }
            // Track distinct SETTIMG source addresses to find missing fb-as-
            // texture composite ops (e.g. hi-res scratch → lo-res VI fb).
            if (gbi_log_enabled()) {
                static rt64diag::BoundedMap<uint32_t, uint32_t> s_seen;
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
            // DL-walker desync probe: a real setColorImage address is KSEG0 (0x80) or physical
            // (0x00) high byte. A garbage high byte (0xFF/0xFC/0xDB...) means the walker read a
            // DATA word as a command — dump the preceding op ring to find the mis-consuming op.
            {
                const uint8_t hi = (uint8_t)(w1 >> 24);
                if (hi != 0x80 && hi != 0x00) {
                    rt64_f5_desync_dump("setCIMG garbage-high-byte", w1);
                }
            }
            const uint32_t fmt = (w0 >> 21) & 0x7;
            const uint32_t siz = (w0 >> 19) & 0x3;
            if (fmt > 4) return;
            // Reject implausibly-low color-image addresses (e.g. the garbage 0x000EFF
            // the cinematic emits): RT64's FramebufferManager builds a bad target from
            // them and derefs null in recordOperations (AV reading 0x10). Real frame-
            // buffers live high in RDRAM; keep the previous valid color image instead.
            if ((w1 & 0x00FFFFFFu) < 0x100000u) {
                if (gbi_log_enabled()) {
                    static int s_lo = 0;
                    if (++s_lo <= 8) { std::fprintf(stderr, "[gbi-f5] setCIMG reject low-addr w1=0x%08X\n", w1); std::fflush(stderr); }
                }
                return;
            }
            // Reject color images PAST the 8 MB RDRAM (>= 0x800000). The cinematic emits garbage
            // SETCIMGs at high addrs (0xBC0000/0xFA0000/0xFC0000) that the low-addr + width checks
            // miss; each registers a fresh out-of-RDRAM framebuffer -> the registry floods (3->8+)
            // -> CBV/tile-copy wassert/abort in the render thread (the nondeterministic "mode-2"
            // crash). A real framebuffer can't live past RDRAM. Same out-of-RDRAM class as the
            // op_b5 DL-walk bound. Keep the previous valid color image instead.
            if ((w1 & 0x00FFFFFFu) >= 0x800000u) {
                if (gbi_log_enabled()) {
                    static int s_hi = 0;
                    if (++s_hi <= 8) { std::fprintf(stderr, "[gbi-f5] setCIMG reject high-addr (past RDRAM) w1=0x%08X\n", w1); std::fflush(stderr); }
                }
                return;
            }
            // 2026-09-07: dump the command ring for the phantom-CIMG forms that DO pass the high-byte
            // check (width 1, or an address outside the game's fb window) — these are data words the
            // 8-byte walk reached; the ring shows which preceding op mis-consumed its length.
            {
                const uint32_t cw = (w0 & 0xFFFu) + 1;
                const uint32_t ca = w1 & 0x00FFFFFFu;
                if (cw <= 1u || ca < 0x400000u || ca >= 0x800000u) rt64_f5_desync_dump("setCIMG w<=1 or out-of-window", w1);
            }
            // Reject garbage-WIDTH color images (fb-registry showed widths 3477/3924).
            // RT64 builds a huge/bad render target and createTileCopyRecord derefs null
            // (AV reading 0x80, rt64_render_target.cpp setupColorFramebuffer). Real fbs
            // are <=640 (cinematic) / <=1024. Keep the previous valid color image.
            {
                // Garbage images inside the framebuffer window (2026-09-08): stale-walk words such as
                // `FF...` at 0x760000 with width 1, or odd addresses with width 3/10, were registered and
                // RT64's write-back then overwrote the game's heap free list (LucasArts-reveal crash).
                // Real framebuffers here are 320..640 wide and 64-byte aligned.
                const uint32_t cimgWidth0 = (w0 & 0xFFFu) + 1;
                static const bool s_strict = [](){ const char* e = std::getenv("ROGUESQ_F5_CIMG_STRICT"); return !(e && e[0] == '0'); }();
                if (s_strict && (cimgWidth0 < 16u || ((w1 & 0x3Fu) != 0u))) {
                    if (gbi_log_enabled()) {
                        static int s_gw = 0;
                        if (++s_gw <= 8) { std::fprintf(stderr, "[gbi-f5] setCIMG reject garbage width=%u addr=0x%08X\n", cimgWidth0, w1); std::fflush(stderr); }
                    }
                    return;
                }
            }
            {
                const uint32_t cimgWidth = (w0 & 0xFFFu) + 1;
                if (cimgWidth > 1024u) {
                    if (gbi_log_enabled()) {
                        static int s_bw = 0;
                        if (++s_bw <= 8) { std::fprintf(stderr, "[gbi-f5] setCIMG reject bad-width=%u w0=0x%08X w1=0x%08X\n", cimgWidth, w0, w1); std::fflush(stderr); }
                    }
                    return;
                }
            }

            // PHASE MARKER (ROGUESQ_FIRE_PROBE=1): classify the draw-target buffer
            // into a named boot phase and log a timestamped/frame-counted line on each
            // transition, so probe output can be anchored to attribution vs N64 logo vs
            // cinematic instead of guessing. Buffer families from prior RE notes.
            {
                static int s_pm = -1;
                if (s_pm < 0) { const char* v = std::getenv("ROGUESQ_FIRE_PROBE"); s_pm = (v && *v && v[0] != '0') ? 1 : 0; }
                if (s_pm) {
                    const uint32_t a = w1 & 0x00FFFFFFu;
                    auto classify = [](uint32_t a) -> const char* {
                        if (a == 0x66A000 || a == 0x5D4000) return "ATTRIBUTION";
                        if (a == 0x6DD000 || a == 0x6BA000) return "BOOT/N64LOGO";
                        if (a == 0x62B800 || a == 0x695C00) return "CINEMATIC";
                        if (a == 0x290000 || a == 0x795C00 || a == 0x240000) return "OFFSCREEN-CONTENT";
                        return "OTHER";
                    };
                    const char* ph = classify(a);
                    static const char* s_lastPh = nullptr;
                    static uint32_t s_lastAddr = 0;
                    static auto s_t0 = std::chrono::steady_clock::now();
                    static long s_frame = 0;
                    s_frame++;
                    if (ph != s_lastPh || a != s_lastAddr) {
                        double secs = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - s_t0).count();
                        std::fprintf(stderr,
                            "[PHASE] t=%.1fs frame=%ld -> %s (cimg=0x%06X)\n",
                            secs, s_frame, ph, a);
                        std::fflush(stderr);
                        s_lastPh = ph; s_lastAddr = a;
                    }
                }
            }

            // Raw per-call dump of the first 40 setCIMG (ROGUESQ_LOG_CIMG=1) to
            // see the exact color-image address the game emits — used to check
            // whether the attribution buffer is 0x66A000 vs a bit-20-corrupted
            // 0x76A000.
            if (gbi_log_enabled()) {
                static int s_raw = 0;
                if (++s_raw <= 40) {
                    std::fprintf(stderr,
                        "[gbi-f5] setCIMG-raw #%d w0=0x%08X w1=0x%08X (addr=0x%08X fmt=%u siz=%u width=%u)\n",
                        s_raw, w0, w1, w1, fmt, siz, ((w0 & 0xFFF) + 1));
                    std::fflush(stderr);
                }
            }

            // Per-address accept counter. Periodically dumps a table of
            // every legit-shaped setCIMG address and how many times it was
            // emitted. Helps see whether VI's sampled fb actually gets
            // setCIMG'd at all (presence) and how often vs. scratch fbs
            // (frequency). ROGUESQ_LOG_CIMG=1 to enable.
            {
                static rt64diag::BoundedMap<uint32_t, uint32_t> s_counts;
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

            // ROGUESQ_REDIRECT_SCRATCH=1: the cinematic renders the explosion into a
            // SCRATCH framebuffer 0x795C00 (640x264) then copies it to the displayed
            // buffer 0x62B800/0x695C00 via a CPU/RSP op our HLE doesn't reproduce, so
            // the explosion never reaches the screen. Redirect setColorImage(0x795C00)
            // -> 0x62B800 so RT64 renders the explosion DIRECTLY into a presented
            // buffer. Existence test for whether the content is real + renderable.
            {
                static int s_rs = -1;
                if (s_rs < 0) { const char* v = std::getenv("ROGUESQ_REDIRECT_SCRATCH"); s_rs = (v && *v && v[0] != '0') ? 1 : 0; }
                if (s_rs && (w1 & 0x00FFFFFFu) == 0x795C00u) {
                    (*dl)->w1 = (w1 & 0xFF000000u) | 0x0062B800u;
                    if (gbi_log_enabled()) {
                        static int s_rl = 0;
                        if (++s_rl <= 6) { std::fprintf(stderr, "[gbi-f5] redirect scratch 0x795C00 -> 0x62B800\n"); std::fflush(stderr); }
                    }
                }
            }

            // Attribution per-frame glyph-draw count (see s_attrib_glyphs_frame).
            // Logs how many glyph texrects landed in 0x76A000 in the frame that
            // just ended — to see if text persists or stops after the fade-in.
            if (w1 == 0x8076A000u) {
                if (gbi_log_enabled()) {
                    std::fprintf(stderr, "[gbi-f5 attrib-frame #%d] prev-frame glyphs=%d fills=%d\n",
                        ++s_attrib_frame_idx, s_attrib_glyphs_frame, s_attrib_fills_frame);
                    std::fflush(stderr);
                }
                s_attrib_prev_glyphs = s_attrib_glyphs_frame;  // for the de-flicker
                s_attrib_glyphs_frame = 0;
                s_attrib_fills_frame = 0;
                s_cart_pass = 0;  // new attribution frame → reset cartridge material pass
            }

            // Flush accumulated scene-graph models into the CURRENT (old) color
            // image BEFORE switching targets. The cinematic is multi-pass: nodes
            // accumulate while buffer A is current, then a setColorImage switches
            // to B. Without this flush they'd all dump into a later pass's fullSync
            // (was: pending=256 leaking across buffers). Now each pass's models
            // land in their own buffer.
            // Per-buffer flush (ROGUESQ_GR_PERBUF=1): render accumulated models into
            // the CURRENT (old) color image before switching, so each multi-pass
            // target gets its own models instead of all leaking into a later fullSync
            // (was: pending=256). CORRECT for buffer assignment, but it makes RT64
            // start tracking/reading-back the cinematic intermediates → trips the
            // RGBA8-readback assert (rt64_native_target.cpp:168). Off by default; the
            // generic path keeps fullSync-only flush (stabler) until the offscreen→
            // display composite is solved. Only flush into a plausible FB (>=0x200000):
            // the game briefly sets garbage CIMGs (0x1410A6 = matpool corruption) that
            // would scribble into the matrix pool and stall the loop.
            // PHASE D: the per-buffer render_scene_objects flush (ROGUESQ_GR_PERBUF) and the
            // cartridge-transform validation hook were DELETED with the host-side model
            // renderers — models render via the interpreted DL stream (docs/f5-model-dl-spec.md).
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
                        static rt64diag::BoundedSet<uint64_t> s_seenDims;
                        uint64_t key = (uint64_t(addr) << 32) |
                                       (uint64_t(state->rdp->colorImage.width) << 16) |
                                       (uint64_t(state->rdp->colorImage.fmt) << 8) |
                                       uint64_t(state->rdp->colorImage.siz);
                        if (s_seenDims.insert(key)) {
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


        // Force magenta-opaque fillColor (RGBA5551 0xF80F packed twice) under FORCE_VISIBLE.
        void setFillColor_overridden(State *state, DisplayList **dl) {
            if (force_visible_enabled()) {
                state->rdp->setFillColor(0xF80FF80F);
                return;
            }
            GBI_RDP::setFillColor(state, dl);
        }

    } // namespace GBI_F3DFACTOR5
} // namespace RT64
