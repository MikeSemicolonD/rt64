//
// RT64 — F3DFACTOR5 internal interface
//
// Shared declarations for the F3DFACTOR5 HLE GBI module after it was split
// across files for readability. Holds the small math/byte-read helpers and
// vertex struct used by both the live software-T&L pipeline and the (default-
// off) diagnostics, plus the cross-module function declarations. Each module
// (.cpp) opens `namespace RT64::GBI_F3DFACTOR5` and defines its portion.
//
#pragma once

#include <cstdint>

#include "rt64_gbi.h"        // DisplayList
#include "hle/rt64_state.h"  // State (+ rdp)

// --- Shared game/host globals (extern "C") --------------------------------------------
// Declared ONCE here so every F5 GBI module (core/model/rdpstate/...) shares one
// declaration instead of re-declaring them. Defined in upstream_compat.cpp / the
// recompiled funcs, EXCEPT g_most_drawn_fb + g_explosion_hold which are defined in the
// GBI core (rt64_gbi_f3dfactor5.cpp).
extern "C" {
    // Screen / overlay phase (set by upstream_compat.cpp hooks).
    extern volatile int g_active_overlay;       // 1=menu 2=cinematic 0=gameplay -1=none
    extern volatile int g_current_scene;        // menuOverlayInit menu-screen id (F5Scene; 9=attribution)
    // Loaded Expansion-Pak cartridge model + its real CI4 textures (set by load_hmt_and_hob).
    extern volatile unsigned g_mempak_hob_base; // cartridge HOB mesh base (0 until loaded)
    extern volatile unsigned g_mempak_hmt_base; // cartridge HMT (textures+palettes) base
    extern volatile int  g_cart_tex_ready;      // "cartridge loaded" gate (g_cart_*_pix/pal removed)
    // Per-frame scene graph (set by processSceneNode, funcs_4.c — see project_scene_object_list).
    extern volatile unsigned g_current_meshdef; // meshdef of the model being drawn
    extern volatile unsigned g_scene_node_count;// bumps per scene node traversed
    extern float    g_current_matrix[16];       // projection-baked MVP (row-major 3x4)
    extern unsigned g_scene_md[256];            // per-frame scene objects: meshdef ptrs
    extern float    g_scene_mtx[256][16];       //   their MVPs
    extern volatile int g_scene_obj_count;
    // Framebuffer / phase tracking.
    extern volatile unsigned g_last_swap_fb;    // VI-presented fb (osViSwapBuffer_recomp)
    extern volatile unsigned g_op_bf_count;     // explosion-bloom tri counter (fb-dump trigger)
    extern volatile unsigned g_most_drawn_fb;   // heaviest-drawn color image (def in core)
    extern volatile unsigned g_most_drawn_fb_width;  // its color-image width (def in core)
    extern volatile unsigned long long g_most_drawn_fb_ms;  // steady-clock ms of its last texrect (def in core)
    extern volatile int g_explosion_hold;       // explosion-phase crawl countdown (def in core)
}

// Known game menu-screen ids (g_current_scene = menuOverlayInit's action arg). Grows as we map
// screens; g_current_scene holds the raw id even when no enum name exists for it yet, so the
// renderer can dispatch per scene (g_current_scene == F5_SCENE_ATTRIBUTION) instead of a per-screen bool.
enum F5Scene { F5_SCENE_NONE = -1, F5_SCENE_ATTRIBUTION = 9 };

namespace RT64 {
    namespace GBI_F3DFACTOR5 {

        // N64 fixed-point matrix: 32 bytes int16 integer parts, then 32 bytes uint16 frac, (int<<16)|frac = s15.16. Row-major so hlslpp::mul matches RT64 (rt64_rsp.cpp).
        inline hlslpp::float4x4 decode_n64_f4x4(const uint8_t* rdram, uint32_t addr) {
            const uint32_t phys = addr & 0x00FFFFFF;
            auto be16 = [&](uint32_t off) -> uint16_t {
                if (phys + off + 1 >= 0x800000) return (uint16_t)0;
                return (uint16_t)((rdram[(phys + off) ^ 3] << 8) | rdram[(phys + off + 1) ^ 3]);
            };
            float f[16];
            for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) {
                int i = r * 4 + c; uint16_t ip = be16(i * 2), fp = be16(32 + i * 2);
                f[i] = (float)(((int32_t)(int16_t)ip << 16) | fp) / 65536.0f;
            }
            return hlslpp::float4x4(f[0],f[1],f[2],f[3], f[4],f[5],f[6],f[7],
                                    f[8],f[9],f[10],f[11], f[12],f[13],f[14],f[15]);
        }

        // Row-vector × matrix (N64/F3D convention): out = v · M. Thin hlslpp adapter.
        inline void f5_xform_row(const hlslpp::float4x4& m, const float v[4], float out[4]) {
            hlslpp::store(hlslpp::mul(hlslpp::float4(v[0],v[1],v[2],v[3]), m), out);
        }
        // Matrix × column-vector: out = M · v. Thin hlslpp adapter (op_02 paths).
        inline void f5_xform_col(const hlslpp::float4x4& m, const float v[4], float out[4]) {
            hlslpp::store(hlslpp::mul(m, hlslpp::float4(v[0],v[1],v[2],v[3])), out);
        }

        // Apply perspective divide (1/w on x, y, z).
        inline void perspective_divide(float v[4]) {
            if (v[3] != 0.0f) { const float iw = 1.0f / v[3]; v[0]*=iw; v[1]*=iw; v[2]*=iw; }
        }

        // Read a big-endian s16 / u16 / u32 from RDRAM at KSEG0 vaddr.
        inline int rd_be_s16(const uint8_t* ram, uint32_t vaddr) {
            uint32_t p = vaddr & 0x00FFFFFF;
            return (int16_t)(((uint16_t)ram[p ^ 3] << 8) | ram[(p + 1) ^ 3]);
        }
        inline uint32_t rd_be_u16(const uint8_t* ram, uint32_t vaddr) {
            uint32_t p = vaddr & 0x00FFFFFF;
            return ((uint32_t)ram[p ^ 3] << 8) | ram[(p + 1) ^ 3];
        }
        inline uint32_t rd_be_u32(const uint8_t* ram, uint32_t vaddr) {
            return (rd_be_u16(ram, vaddr) << 16) | rd_be_u16(ram, vaddr + 2);
        }

        // Unified-pipeline screen-space vertex cache entry.
        struct F5UVert { float sx, sy, sz, invw, cw; uint8_t r, g, b, a; uint8_t valid; };

        // --- PHASE D (docs/f5-model-dl-spec.md): the host-side model renderers and
        //     texture loaders (rt64_gbi_f5_model.cpp / rt64_gbi_f5_textures.cpp —
        //     render_one_model, render_scene_objects, render_cartridge_general,
        //     render_op02_generic, f5_build_global_mats/hmt_mats, f5_load_*) were
        //     DELETED. Models render through the interpreted F5 DL stream: op_01
        //     matrices, op_04/0x14 vertex batches, 0x13/0xBF/0xB4 tris/quads, and
        //     cached per-material sub-DLs (pure standard RDP ops) via G_DL.

        // ROGUESQ_LOG_HUD_PROBE: logs one draw (screen centre and size in N64 pixels) on a few sampled frames.
        void f5_hud_probe(State *state, const char *tag, float cx, float cy, float w, float h);
        // Draws a crosshair-ring texrect (dl points at its 16 bytes) as an interpolated quad; false leaves it to the texrect path.
        bool f5_crosshair_quad(State *state, const DisplayList *dl);

        // --- RDP-state / raster handlers (rt64_gbi_f5_rdpstate.cpp): Factor-5 wrappers
        //     around the base GBI_F3D/GBI_RDP handlers (logging, guards, CI4 reinterpret).
        //     setup() in the core maps these onto the opcode table.
        void setOtherModeH_logged(State *state, DisplayList **dl);
        void setOtherModeL_logged(State *state, DisplayList **dl);
        void setRDPOtherMode_logged(State *state, DisplayList **dl);
        void setScissor_logged(State *state, DisplayList **dl);
        void setTile_logged(State *state, DisplayList **dl);
        void setCombine_logged(State *state, DisplayList **dl);
        void setFillColor_overridden(State *state, DisplayList **dl);
        void fillRect_logged(State *state, DisplayList **dl);  // fillRect_op02_aware stays in core (op_02-coupled)
        void texrectLLE_guarded(State *state, DisplayList **dl);
        void texrectFlipLLE_guarded(State *state, DisplayList **dl);
        void loadTLUT_guarded(State *state, DisplayList **dl);
        void loadTile_guarded(State *state, DisplayList **dl);
        void loadBlock_guarded(State *state, DisplayList **dl);
        void setTextureImage_filtered(State *state, DisplayList **dl);
        void setColorImage_filtered(State *state, DisplayList **dl);
        // CI4-track state shared across the rdpstate handlers (loadTLUT/loadBlock write,
        // setTile/setTextureImage read) — extern during the incremental split.
        extern int s_ci4_tlut_recent;
        extern uint32_t s_last_tlut_src;
        // FORCE_VISIBLE debug gate (def in core during the split) + its forced
        // G_CC_PRIMITIVE-mux combiner words, shared by setFillColor/setCombine/texrect.
        bool force_visible_enabled();
        inline constexpr uint32_t FORCED_COMB_W0 = 0x00FFFEE7u;  // c0/c1=(0,0,0,PRIM)
        inline constexpr uint32_t FORCED_COMB_W1 = 0x771F77F8u;  // a0/a1=(0,0,0,PRIM)
        // Attribution de-flicker counters (def in core during the split): texrectLLE counts
        // glyphs + fillRect counts fills; setColorImage reads them at each 0x76A000 frame boundary.
        extern int s_attrib_glyphs_frame;
        extern int s_attrib_fills_frame;
        extern int s_attrib_frame_idx;
        extern int s_attrib_prev_glyphs;
        extern thread_local int s_cart_pass;  // cartridge render-pass counter (def in core)

        // --- Shared core gates/helpers (defined in rt64_gbi_f3dfactor5.cpp), used
        //     across modules: the ROGUESQ_LOG_GBI gate + the unified-pipeline MVP composer.
        bool gbi_log_enabled();
        // Compose the cinematic/cartridge MVP from the op_01 matrix ring; fills Mout[12] =
        // columns 0,1,3 of the row-major MVP (render_one_model's clip x/y/w layout). False if
        // fewer than 2 distinct matrices are available yet.
        bool f5_compose_mvp(const uint8_t* ram, float Mout[12]);
        // op_02's per-vertex RGBA color buffer (set by op_02, read by render_op04_unified).
        extern thread_local uint32_t s_last_op02_colorbuf, s_last_op02_colorcnt;

    } // namespace GBI_F3DFACTOR5
} // namespace RT64
