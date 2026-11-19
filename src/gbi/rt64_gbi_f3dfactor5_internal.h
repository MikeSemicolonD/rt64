// RT64 - F3DFACTOR5 internal interface
// Math and byte-read helpers, the vertex struct and cross-file declarations for the module's .cpp files, which each open namespace RT64::GBI_F3DFACTOR5.
#pragma once

#include <cstdint>

#include "rt64_gbi.h"        // DisplayList
#include "hle/rt64_state.h"  // State (+ rdp)

#include "common/rt64_rs64_env.h"
#include "hle/rt64_rs64_host.h"

extern "C" {
    // Explosion-phase crawl countdown (defined in the core).
    extern volatile int g_explosion_hold;
    // Dumps the recent F5 command ring (rt64_interpreter.cpp).
    void rt64_f5_desync_dump(const char *why, uint32_t badW1);
}

// Known game menu-screen ids (g_current_scene = menuOverlayInit's action arg). Grows as we map
// screens; g_current_scene holds the raw id even when no enum name exists for it yet, so the
// renderer can dispatch per scene (g_current_scene == F5_SCENE_ATTRIBUTION) instead of a per-screen bool.
enum F5Scene { F5_SCENE_NONE = -1, F5_SCENE_ATTRIBUTION = 9 };

namespace RT64 {
    namespace GBI_F3DFACTOR5 {

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

        // ROGUESQ_LOG_HUD_PROBE: logs one draw (screen centre and size in N64 pixels) on a few sampled frames.
        void f5_hud_probe(State *state, const char *tag, float cx, float cy, float w, float h);
        // Draws a crosshair-ring texrect (dl points at its 16 bytes) as an interpolated quad; false leaves it to the texrect path.
        bool f5_crosshair_quad(State *state, const DisplayList *dl);
        bool f5_credits_glyph_quad(State *state, const DisplayList *dl);
        // VR: draws a HUD element's texrect as a quad at the radar's depth in front of the game camera; false leaves it to the texrect path.
        bool f5_vr_hud_quad(State *state, const DisplayList *dl);
        // VR in missions: a full-height wipe texrect as one quad at the HUD depth, grown past the screen edges it touches; false leaves it a texrect.
        bool f5_vr_wipe_quad(State *state, const DisplayList *dl);
        // VR outside missions: a backdrop or menu texrect as a quad where the game camera faces; false leaves it to the texrect path.
        bool f5_vr_scene_quad(State *state, const DisplayList *dl);
        // The task whose scene clear fill was stretched to the whole target (VR).
        extern uint64_t g_vr_fill_task;

        // --- RDP-state / raster handlers (rt64_gbi_f5_rdpstate.cpp): Factor-5 wrappers
        //     around the base GBI_F3D/GBI_RDP handlers (logging, guards, CI4 reinterpret).
        //     setup() in the core maps these onto the opcode table.
        void setOtherModeH_logged(State *state, DisplayList **dl);
        void setOtherModeL_logged(State *state, DisplayList **dl);
        void setRDPOtherMode_logged(State *state, DisplayList **dl);
        void setScissor_logged(State *state, DisplayList **dl);
        void setTile_logged(State *state, DisplayList **dl);
        void setCombine_logged(State *state, DisplayList **dl);
        void fillRect_logged(State *state, DisplayList **dl);
        void texrectLLE_guarded(State *state, DisplayList **dl);
        // RS64 HUD marker (rs64hud::kMarkerOpcode): the screen side the next texrects of this 2D element anchor to.
        void op_60_hud_side(State *state, DisplayList **dl);
        // RS64 light for a texture rectangle drawn as a 3D glow (the game's blinking lamps): centre and half width in native pixels.
        void f5_lamp_light(State *state, float cx, float cy, float halfW);
        void texrectFlipLLE_guarded(State *state, DisplayList **dl);
        void loadTLUT_guarded(State *state, DisplayList **dl);
        void loadTile_guarded(State *state, DisplayList **dl);
        void loadBlock_guarded(State *state, DisplayList **dl);
        void setTextureImage_filtered(State *state, DisplayList **dl);
        // A 24-bit texture address in the texture pool as the game laid it out (language mods move the pool; rs64_f5_set_texture_pool_move).
        uint32_t f5_unmoved_address(uint32_t masked);
        void setColorImage_filtered(State *state, DisplayList **dl);
        // CI4-track state shared across the rdpstate handlers (loadTLUT/loadBlock write, setTile/setTextureImage read), defined in the core.
        extern int s_ci4_tlut_recent;
        extern uint32_t s_last_tlut_src;
        // Attribution de-flicker counters, defined in the core: texrectLLE counts glyphs and fillRect counts fills; setColorImage reads them at each 0x76A000 frame boundary.
        extern int s_attrib_glyphs_frame;
        extern int s_attrib_fills_frame;
        extern int s_attrib_frame_idx;
        extern int s_attrib_prev_glyphs;
        extern thread_local int s_cart_pass;  // cartridge render-pass counter (def in core)

        // The ROGUESQ_LOG_GBI gate (defined in the core).
        bool gbi_log_enabled();
        // op_02's per-vertex RGBA color buffer (set by op_02, read by the face emitters).
        extern thread_local uint32_t s_last_op02_colorbuf, s_last_op02_colorcnt;

    } // namespace GBI_F3DFACTOR5
} // namespace RT64
