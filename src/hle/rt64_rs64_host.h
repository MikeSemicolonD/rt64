#pragma once

// RS64 host boundary: what the game host defines for RT64, and the hooks and counters RT64 defines for the host. The VR callbacks are in rt64_rs64_vr.h.
// toml_hooks.c, compiled as C with the recompiled code, declares the three F5 hooks it calls in rs64_toml_hooks.h.

#include <atomic>
#include <cstdint>

extern "C" {
    // Defined by the host.

    int rs64_static_terrain_mode(void);
    int rs64_terrain_fog_clip(void);
    int rs64_terrain_tessellation(void);
    int rs64_terrain_textures(uint32_t level, const uint8_t **bank, uint32_t *bankSize, const uint8_t **ids, uint32_t *idsSize);
    // The level's mission sun colour (normalized), read from its hmp; 0 when unknown.
    int rs64_level_sun_color(uint32_t level, float *rgb);
    float rs64_draw_dist(void);
    float rs64_fog_dist(void);
    float rs64_fog_tail_end(void);
    uint32_t rs64_level_fog_rgb(void);
    int rs64_level_has_clouds(void);
    int rs64_credits_active(void);

    // ultramodern: tell the game this graphics task is done before its GPU work finishes.
    void rs64_gfx_complete_early(void);

    // Active overlay: 1 menu, 2 cinematic, 0 gameplay, -1 none.
    extern volatile int g_active_overlay;
    // menuOverlayInit's menu-screen id (F5Scene; 9 = attribution).
    extern volatile int g_current_scene;
    extern volatile unsigned g_op_bf_count;

    // Defined by RT64.

    // 1 while the game shows a 2D screen (tasks under 64 3D faces, held for 10 tasks before it changes).
    int rs64_f5_screen2d(void);
    void rs64_f5_set_texture_pool_move(uint32_t orig, uint32_t moved, uint32_t size);
    void rs64_f5_map_node(uint32_t ringDst, uint32_t nodeId);
    void rs64_f5_map_sprite(uint32_t dlAddr, uint32_t face);
    void rs64_f5_cable_submit(uint32_t pool, uint32_t meshInstance, uint32_t state);

    // The heaviest-drawn color image, its width and the steady-clock ms of its last texrect.
    extern volatile unsigned g_most_drawn_fb;
    extern volatile unsigned g_most_drawn_fb_width;
    extern volatile unsigned long long g_most_drawn_fb_ms;

    extern volatile unsigned g_f5_task_hops;
    extern volatile unsigned g_f5_task_faces;
    // Heuristic firing counts (ROGUESQ_LOG_HEURISTICS): 0 = op 0x0A seen, 1 = op_01 shape rule disagrees with byte1 bit0, 2 = B4 filler guard fired, 3 = attribution de-flicker skip.
    extern volatile unsigned g_f5_heur[4];

    // Per-frame workload counts for the F5 profiler HUD's RDP slots, summed in State::flush and read-and-cleared by the host.
    extern std::atomic<uint32_t> g_rs64_frame_tris;
    extern std::atomic<uint32_t> g_rs64_frame_draws;
    extern std::atomic<uint32_t> g_rs64_frame_texloads;
}
