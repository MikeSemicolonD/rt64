#pragma once

// Dynamic point lights for the F5 profile: candidates recorded at draw time, resolved per displayed frame.

#include "rt64_rs64_math.h"

namespace rs64lights {
    constexpr uint32_t MaxLights = 64;

    // Set while an interpolated in-between frame renders: the soft shadow pass reuses the real frame's trace, reprojected, instead of tracing again.
    inline bool g_reuseTrace = false;

    // KindLampMark: not a light, where one of the game's blinking lamps (a 2D texture rectangle) is drawn this frame, for lamp presets.
    enum Kind : uint8_t { KindNone = 0, KindSprite = 1, KindLaser = 2, KindMesh = 3, KindLampMark = 4 };

    // Default light ranks for the 32 slots, higher kept first; a light preset's "priority" overrides them (the built-in presets: engine exhausts 7,
    // the pickup 6, the torpedo 1, lasers 0).
    constexpr int PriorityLaser = 0;
    constexpr int PriorityLevel = 2;
    constexpr int PriorityMesh = 3;
    constexpr int PrioritySprite = 5;

    struct Candidate {
        uint32_t transformIndex = 0;
        uint32_t viewIndex = 0;
        float local[3] = {};
        float color[3] = {};
        float radius = 0.0f;
        float intensity = 0.0f;
        float falloff = 2.0f;
        uint8_t kind = KindNone;
        // Pure light (laser bolts): its mesh never casts. Physical emitters (torpedoes) still do.
        bool emissive = false;
        // Occluders closer than this to the light are ignored (an explosion sprite's own extent: wreckage inside the blast).
        float shadowStart = 0.0f;
        // radius and shadowStart are in the model's units (scaled by its transform when resolved), not camera units.
        bool modelUnits = false;
        // Light-preset extras (RT64 PointLight): beam direction (local, like the position) and cone cosines as RT64 defines them (lit where
        // dot(surface -> light, beam) <= spotMaxCos; 1 = no cone); flicker 0-1 dims it by a smooth random signal keyed by seed.
        float dir[3] = { 0.0f, -1.0f, 0.0f };
        float spotMaxCos = 1.0f;
        float spotFalloffCos = 1.0f;
        float flicker = 0.0f;
        uint32_t seed = 0;
        // A KindLampMark: local = native screen (x, y), screenHalf = the lamp's half width in native pixels, color / intensity = its PRIM.
        float screenHalf = 0.0f;
        // A lamp preset's light: kept only while a lamp mark covers its spot on screen, taking the mark's colour and fade.
        bool lampGated = false;
        // A laser bolt's light (laserBolt mesh key): capped in selectLights.
        bool projectile = false;
        // Rank for the light slots, as Resolved.
        int priority = PriorityMesh;
    };

    struct Resolved {
        float pos[3] = {};
        float radius = 0.0f;
        float color[3] = {};
        float intensity = 0.0f;
        float falloff = 2.0f;
        float shadowStart = 0.0f;
        uint8_t kind = KindNone;
        // An engine exhaust light (a mesh light in model units).
        bool engine = false;
        // As Candidate, with dir in camera space.
        float dir[3] = { 0.0f, -1.0f, 0.0f };
        // A glowing panel's half-width and half-height axes in camera space (it faces along dir); zero = a point light.
        float rectU[3] = {};
        float rectV[3] = {};
        float spotMaxCos = 1.0f;
        float spotFalloffCos = 1.0f;
        float flicker = 0.0f;
        uint32_t seed = 0;
        bool lampGated = false;
        // A laser bolt's light: at most a few slots (selectLights).
        bool projectile = false;
        // Rank for the light slots (Priority*), higher kept first.
        int priority = 0;
    };

    // The water ripple clock: wall seconds wrapped at RS64WaterRipple.hlsli's kRippleLoop (every wave loops whole in it), so it keeps float precision.
    inline float rippleClock() {
        static const auto s_start = std::chrono::steady_clock::now();
        return float(std::fmod(std::chrono::duration<double>(std::chrono::steady_clock::now() - s_start).count(), 64.0));
    }

    // Flicker gain at time t (seconds): 1 - flicker x smooth value noise at 12 steps a second, each light on its own sequence (seed).
    inline float flickerGain(float flicker, uint32_t seed, double t) {
        if (!(flicker > 0.0f)) {
            return 1.0f;
        }
        const auto rnd = [seed](uint64_t k) {
            uint64_t x = (k + 1) * 0x9E3779B97F4A7C15ull ^ (uint64_t(seed) << 17);
            x = (x ^ (x >> 31)) * 0xBF58476D1CE4E5B9ull;
            return float((x ^ (x >> 29)) & 0xFFFFFF) / float(0xFFFFFF);
        };
        const double u = t * 12.0;
        const uint64_t k = uint64_t(u);
        const float f = float(u - double(k)), s = f * f * (3.0f - 2.0f * f);
        const float n = rnd(k) * (1.0f - s) + rnd(k + 1) * s;
        return 1.0f - std::min(flicker, 1.0f) * n;
    }

    // PRIM colour of an 0xBD sprite (RRGGBBAA). Dark smoke and faded sprites give no light.
    inline bool spriteLight(uint32_t primRGBA, float cameraSize, float radiusScale, float maxRadius, Candidate& out) {
        const float r = (float)((primRGBA >> 24) & 0xFFu) / 255.0f;
        const float g = (float)((primRGBA >> 16) & 0xFFu) / 255.0f;
        const float b = (float)((primRGBA >> 8) & 0xFFu) / 255.0f;
        const float a = (float)(primRGBA & 0xFFu) / 255.0f;
        const float lum = 0.2126f * r + 0.7152f * g + 0.0722f * b;
        const float intensity = ((lum - 0.35f) / 0.65f) * a;
        if (!(intensity > 0.05f) || !(cameraSize > 0.0f)) {
            return false;
        }
        const float peak = std::max(r, std::max(g, b));
        out = Candidate{};
        out.color[0] = r / peak;
        out.color[1] = g / peak;
        out.color[2] = b / peak;
        out.radius = std::min(cameraSize * radiusScale, maxRadius);
        out.intensity = std::min(intensity, 1.0f);
        out.shadowStart = cameraSize;
        out.kind = KindSprite;
        return true;
    }

    // The object name in mesh-registry keys: the HOB object's, except where two files give different models one name (the game loads one at a
    // time, by name). pl_crafts/rmobil's ground vehicle is an object "vwing" with 7 nodes; the V-wing has 3. Mirrored by the RS64 Explorer.
    inline const char* meshKeyName(const char* name, uint32_t nodes) {
        return ((std::strcmp(name, "vwing") == 0) && (nodes == 7)) ? "rmobil" : name;
    }

    // Flying craft models (player craft, wingmen, enemy fighters and shuttles): they cast every frame but a stale history copy would be a trail.
    inline bool movingCraft(const char* meshKey) {
        if (meshKey == nullptr) {
            return false;
        }
        const char* hash = std::strchr(meshKey, '#');
        if (hash == nullptr) {
            return false;
        }
        static const char* const kCraft[] = { "xwing", "ywing", "awing", "vwing", "snowspeeder", "falcon", "tie_inter", "tie_fighter", "tie_d", "imp_shuttle" };
        const size_t len = size_t(hash - meshKey);
        // Wingmen use their own models, "wm" + craft + "wng" (wmxwng, wmvwng).
        if ((len == 6) && (std::strncmp(meshKey, "wm", 2) == 0) && (std::strncmp(meshKey + 3, "wng", 3) == 0)) {
            return true;
        }

        for (const char* c : kCraft) {
            if ((std::strlen(c) == len) && (std::strncmp(c, meshKey, len) == 0)) {
                return true;
            }
        }
        return false;
    }

    // Laser bolts (meshKey "<hob object name>#s<set>.p<part>.<index>"): pure light, never casters. Their lights are F1 Lights-tab mesh presets.
    inline bool laserBolt(const char* meshKey) {
        static const char* const kLasers[] = { "red_laser", "green_laser", "ion_laser" };
        if (meshKey == nullptr) {
            return false;
        }
        for (const char* name : kLasers) {
            const size_t n = std::strlen(name);
            if ((std::strncmp(meshKey, name, n) == 0) && (meshKey[n] == '#')) {
                return true;
            }
        }
        return false;
    }

    // Candidate whose local position is already in camera space (no transform, any view).
    constexpr uint32_t CameraSpace = UINT32_MAX;

    // Briefing-room hologram: the menu camera is fixed, so the light sits at an authored camera-space point (the room origin, inside the beam).
    inline void menuHoloLight(const float pos[3], float radius, float intensity, float falloff, Candidate& out) {
        out = Candidate{};
        out.transformIndex = CameraSpace;
        out.local[0] = pos[0];
        out.local[1] = pos[1];
        out.local[2] = pos[2];
        out.color[0] = 0.35f;
        out.color[1] = 0.6f;
        out.color[2] = 1.0f;
        out.radius = radius;
        out.intensity = intensity;
        out.falloff = falloff;
        out.kind = KindMesh;
    }

    // Level-select preview image (RGBA16 texrects fading in over the hologram): texel colour weighted by texel and fade alpha, gathered over one frame
    // (active while the briefing room is up) and lit the next. weight / samples is how much of the image is showing.
    struct MenuImageGlow {
        bool active = false;
        double rgb[3] = {};
        double weight = 0.0;
        double samples = 0.0;
    };

    inline MenuImageGlow& menuImageGlow() {
        static MenuImageGlow g;
        return g;
    }

    // Adds an RGBA16 big-endian texel (rdram is byte-swapped per word) seen at fade alpha `fade`.
    inline void menuImageSample(MenuImageGlow& g, uint16_t texel, float fade) {
        const double a = (texel & 1) ? fade : 0.0;
        g.rgb[0] += a * ((texel >> 11) & 31) / 31.0;
        g.rgb[1] += a * ((texel >> 6) & 31) / 31.0;
        g.rgb[2] += a * ((texel >> 1) & 31) / 31.0;
        g.weight += a;
        g.samples += 1.0;
    }

    // The image's light: its average colour at the hologram, as strong as the image is visible. False when nothing showed.
    inline bool menuImageLight(const MenuImageGlow& g, const float pos[3], float radius, float intensity, float falloff, Candidate& out) {
        if (!(g.weight > 0.0) || !(intensity > 0.0f)) {
            return false;
        }

        menuHoloLight(pos, radius, intensity * float(g.weight / g.samples), falloff, out);
        for (int c = 0; c < 3; ++c) {
            out.color[c] = float(g.rgb[c] / g.weight);
        }

        return true;
    }

    // Native screen (x, y) -> final NDC, the space projectToScreen outputs.
    inline void screenToFinalNdc(const ScreenMap& s, float x, float y, float out[2]) {
        out[0] = ((x - s.res[0] * 0.5f) / (s.res[0] * 0.5f)) * s.screenScale[0] + s.screenOffset[0];
        out[1] = ((y - s.res[1] * 0.5f) / (s.res[1] * -0.5f)) * s.screenScale[1] + s.screenOffset[1];
    }

    // Gates lamp lights by lamp marks (both in final NDC): a light keeps only when the nearest mark within reach of it (the mark's half
    // width plus slack, per axis scale) covers it, and takes that mark's colour and fade. True = keep. Unit-tested in rs64_lights_test.
    inline bool lampMatch(const float light[2], const float (*marks)[2], const float* reach, size_t count, size_t& best) {
        float bestD = 0.0f;
        bool found = false;
        for (size_t i = 0; i < count; ++i) {
            const float dx = (light[0] - marks[i][0]) / reach[i * 2 + 0], dy = (light[1] - marks[i][1]) / reach[i * 2 + 1];
            const float d = dx * dx + dy * dy;
            if ((d <= 1.0f) && (!found || (d < bestD))) {
                bestD = d;
                best = i;
                found = true;
            }
        }
        return found;
    }

    inline void projectToScreen(const Mat4& viewProj, const ScreenMap& s, const float view[3], float out[3]) {
        const float v[4] = { view[0], view[1], view[2], 1.0f };
        float c[4];
        transform(v, viewProj, c);
        const float ndc[3] = { c[0] / c[3], -c[1] / c[3], c[2] / c[3] };
        const float sx = ndc[0] * s.vpScale[0] + s.vpTranslate[0];
        const float sy = ndc[1] * s.vpScale[1] + s.vpTranslate[1];
        const float rx = (sx - s.res[0] * 0.5f) / (s.res[0] * 0.5f);
        const float ry = (sy - s.res[1] * 0.5f) / (s.res[1] * -0.5f);
        out[0] = rx * s.screenScale[0] + s.screenOffset[0];
        out[1] = ry * s.screenScale[1] + s.screenOffset[1];
        out[2] = ndc[2] * s.vpScale[2] + s.vpTranslate[2];
    }

    // Uniform scale of a row-vector transform (length of its x row).
    inline float transformScale(const Mat4& m) {
        return std::sqrt(m.m[0][0] * m.m[0][0] + m.m[0][1] * m.m[0][1] + m.m[0][2] * m.m[0][2]);
    }

    // One light per spot: a part drawn in several vertex loads emits its exhaust lights once per load, which tripled them and overflowed MaxLights
    // (the nearest-first cut then changed frame to frame). Lights within `eps` of each other with the same colour keep the strongest. An engine's
    // animated glow sprite also makes a sprite light that pulses and drops out; within half an exhaust light's reach, the exhaust light stands for it.
    // Returns the count.
    inline size_t dedupeLights(Resolved* lights, size_t n, float eps) {
        size_t out = 0;
        for (size_t i = 0; i < n; ++i) {
            bool merged = false;
            for (size_t j = 0; (j < n) && !merged && (lights[i].kind == KindSprite); ++j) {
                const Resolved& m = lights[j];
                const float dx = m.pos[0] - lights[i].pos[0], dy = m.pos[1] - lights[i].pos[1], dz = m.pos[2] - lights[i].pos[2];
                merged = m.engine && ((dx * dx + dy * dy + dz * dz) <= (0.25f * m.radius * m.radius));
            }
            for (size_t j = 0; (j < out) && !merged; ++j) {
                Resolved& k = lights[j];
                const float dx = k.pos[0] - lights[i].pos[0], dy = k.pos[1] - lights[i].pos[1], dz = k.pos[2] - lights[i].pos[2];
                const bool sameColor = (k.color[0] == lights[i].color[0]) && (k.color[1] == lights[i].color[1]) && (k.color[2] == lights[i].color[2]);
                // The r_pow pickup emits one copy per part, a few units apart: a fixed eps merged them only some frames, so its light doubled and halved (flicker).
                const float reach = std::max(eps, 0.01f * std::max(k.radius, lights[i].radius));
                if (sameColor && ((dx * dx + dy * dy + dz * dz) <= (reach * reach))) {
                    if (lights[i].intensity > k.intensity) {
                        k = lights[i];
                    }
                    merged = true;
                }
            }
            if (!merged) {
                lights[out++] = lights[i];
            }
        }
        return out;
    }

    // Drops lights whose reach is under minReach of their distance from the camera (a distant ship's 50-unit exhaust lights a few pixels), so they
    // don't take slots from nearer lights. Returns the count.
    inline size_t dropFarLights(Resolved* lights, size_t n, float minReach) {
        size_t out = 0;
        for (size_t i = 0; i < n; ++i) {
            const Resolved& r = lights[i];
            const float d2 = r.pos[0] * r.pos[0] + r.pos[1] * r.pos[1] + r.pos[2] * r.pos[2];
            if ((r.radius * r.radius) >= (minReach * minReach * d2)) {
                lights[out++] = r;
            }
        }
        return out;
    }

    // First-pass share of the light slots for each default priority, so one busy kind (a battle's explosions) can't leave the others dark; other
    // priorities have none and only fill what is left. Indexed by priority 0-7; roguesq_video.json's rayTracing.lightLimits sets them.
    inline size_t g_priorityShare[8] = { 4, 2, 4, 6, 0, 10, 4, 12 };

    inline size_t priorityShare(int priority) {
        return ((priority >= 0) && (priority < 8)) ? g_priorityShare[priority] : 0;
    }

    // In pools by priority (higher first; within one priority, projectiles last), nearest light edge to the camera (origin) first; lights too faint
    // to change a pixel (a burnt-out explosion frame) are dropped. Each pool first takes up to its share (priorityShare), then the slots left go to the pools' remaining lights in priority
    // order. Projectile lights (laser bolts, fired in pairs by every ship) take at most maxProjectiles slots in all. Within each pool, when lights
    // are left out, the chosen ones fade out as their edge distance nears the first one left out (over a quarter of their radius), so a light
    // swapping across the cut is dark: no pop.
    inline size_t selectLights(const Resolved* in, size_t n, Resolved* out, size_t maxOut, bool cutFade = true, size_t maxProjectiles = SIZE_MAX) {
        std::map<int, std::vector<std::pair<float, size_t>>, std::greater<int>> pools;
        for (size_t i = 0; i < n; ++i) {
            if (!(in[i].intensity >= (1.0f / 255.0f)) || !(in[i].radius > 0.0f)) continue;
            const float d = std::sqrt(in[i].pos[0] * in[i].pos[0] + in[i].pos[1] * in[i].pos[1] + in[i].pos[2] * in[i].pos[2]);
            pools[in[i].priority * 2 + (in[i].projectile ? 0 : 1)].emplace_back(d - in[i].radius, i);
        }

        // How many each pool takes: its share first, then the slots left in priority order.
        std::map<int, size_t, std::greater<int>> takes;
        size_t used = 0;
        size_t projectiles = 0;
        for (int pass = 0; pass < 2; ++pass) {
            for (auto& [key, o] : pools) {
                const bool projectile = (key & 1) == 0;
                size_t& take = takes[key];
                size_t want = (pass == 0) ? std::min(o.size(), priorityShare(key >> 1)) : (o.size() - take);
                want = std::min(want, maxOut - used);
                if (projectile) {
                    want = std::min(want, maxProjectiles - std::min(projectiles, maxProjectiles));
                    projectiles += want;
                }
                take += want;
                used += want;
            }
        }

        size_t count = 0;
        for (auto& [key, o] : pools) {
            const size_t take = takes[key];
            const bool cut = cutFade && (o.size() > take);
            std::partial_sort(o.begin(), o.begin() + (cut ? take + 1 : take), o.end());
            for (size_t i = 0; i < take; ++i) {
                Resolved& r = out[count++];
                r = in[o[i].second];
                if (cut) {
                    r.intensity *= std::clamp((o[take].first - o[i].first) / (0.25f * r.radius), 0.0f, 1.0f);
                }
            }
        }
        return count;
    }

    // One texel of a tile in TMEM (4 KB, N64 byte order; odd-row word swaps only permute texels) as RGBA 0-1. fmt/siz are G_IM_FMT/G_IM_SIZ;
    // CI uses the RGBA16 TLUT in upper TMEM (8-byte entries). RGBA32, YUV and 4/8-bit RGBA are not handled (false).
    inline bool tmemTexel(const uint8_t* tmem, uint8_t fmt, uint8_t siz, uint32_t tmemWord, uint32_t lineWords, uint8_t palette, uint32_t x, uint32_t y, float c[4]) {
        if ((siz > 2) || (fmt == 1) || ((fmt == 0) && (siz != 2))) {
            return false;
        }

        auto rgba16 = [](uint16_t v, float o[4]) {
            o[0] = float((v >> 11) & 0x1F) / 31.0f;
            o[1] = float((v >> 6) & 0x1F) / 31.0f;
            o[2] = float((v >> 1) & 0x1F) / 31.0f;
            o[3] = float(v & 1);
        };
        const uint32_t bits = 4u << siz;
        const uint32_t bitOff = x * bits;
        const uint32_t byte = ((tmemWord + y * lineWords) * 8 + bitOff / 8) & 0xFFF;
        uint32_t v = 0;
        if (bits == 4) {
            v = (bitOff & 4) ? (tmem[byte] & 0xF) : (tmem[byte] >> 4);
        }
        else if (bits == 8) {
            v = tmem[byte];
        }
        else {
            v = (uint32_t(tmem[byte]) << 8) | tmem[(byte + 1) & 0xFFF];
        }

        if (fmt == 2) {
            const uint32_t idx = (bits == 4) ? ((uint32_t(palette) << 4) | v) : (v & 0xFF);
            const uint32_t e = (0x800 + idx * 8) & 0xFFF;
            rgba16(uint16_t((uint32_t(tmem[e]) << 8) | tmem[(e + 1) & 0xFFF]), c);
        }
        else if (fmt == 0) {
            rgba16(uint16_t(v), c);
        }
        else if (fmt == 3) {
            const float i = (bits == 4) ? float(v >> 1) / 7.0f : (bits == 8) ? float(v >> 4) / 15.0f : float(v >> 8) / 255.0f;
            const float a = (bits == 4) ? float(v & 1) : (bits == 8) ? float(v & 0xF) / 15.0f : float(v & 0xFF) / 255.0f;
            c[0] = c[1] = c[2] = i;
            c[3] = a;
        }
        else {
            const float i = (bits == 4) ? float(v) / 15.0f : (bits == 8) ? float(v) / 255.0f : float(v >> 8) / 255.0f;
            c[0] = c[1] = c[2] = c[3] = i;
        }
        return true;
    }

    // A tile's place in TMEM, its size in texels and its texcoord shifts (G_TX_SHIFT: 1-10 divide by 2^n, 11-15 multiply by 2^(16-n)); a mask tile
    // is sampled at the colour tile's texel through its own shift.
    struct TmemTile {
        uint8_t fmt;
        uint8_t siz;
        uint32_t tmemWord;
        uint32_t lineWords;
        uint8_t palette;
        uint32_t width;
        uint32_t height;
        uint8_t shiftS = 0;
        uint8_t shiftT = 0;
    };

    inline uint32_t tileShift(uint32_t c, uint8_t shift) {
        return (shift == 0) ? c : (shift <= 10) ? (c >> shift) : (c << (16 - shift));
    }

    // Mean visible brightness of a sprite frame: luminance of the colour tile x the coverage, which comes from the mask tile when the sprite
    // has one (the game's sprites: colour TEXEL0, alpha TEXEL1 = an I4/I8 mask sampled through its tile shift, half size) and from the colour tile's own alpha
    // otherwise. Up to 8x8 samples. Negative when the tile can't be read. A burnt-out explosion frame reads ~0; a full fireball ~0.15.
    inline float spriteEmission(const uint8_t* tmem, const TmemTile& color, const TmemTile* mask) {
        const uint32_t w = color.width;
        const uint32_t h = color.height;
        const uint32_t mw = mask ? mask->width : 0;
        const uint32_t mh = mask ? mask->height : 0;
        if ((w == 0) || (h == 0) || (w > 1024) || (h > 1024) || (mask && ((mw == 0) || (mh == 0) || (mw > 1024) || (mh > 1024))))
        {
            return -1.0f;
        }

        const uint32_t stepX = std::max<uint32_t>(1, w / 8);
        const uint32_t stepY = std::max<uint32_t>(1, h / 8);
        double sum = 0.0;
        uint32_t n = 0;
        for (uint32_t y = 0; y < h; y += stepY) {
            for (uint32_t x = 0; x < w; x += stepX) {
                float c[4] = {};
                if (!tmemTexel(tmem, color.fmt, color.siz, color.tmemWord, color.lineWords, color.palette, x, y, c)) {
                    return -1.0f;
                }

                float a = c[3];
                if (mask) {
                    float m[4] = {};
                    const uint32_t mx = std::min(tileShift(x, mask->shiftS), mw - 1);
                    const uint32_t my = std::min(tileShift(y, mask->shiftT), mh - 1);
                    if (!tmemTexel(tmem, mask->fmt, mask->siz, mask->tmemWord, mask->lineWords, mask->palette, mx, my, m)) {
                        return -1.0f;
                    }

                    a = m[3];
                }

                sum += (0.2126f * c[0] + 0.7152f * c[1] + 0.0722f * c[2]) * a;
                ++n;
            }
        }
        return (n > 0) ? float(sum / n) : -1.0f;
    }

    // A sprite light's strength for a frame of this brightness: full at ref and above. Unreadable frames (< 0) and ref 0 keep full strength.
    inline float spriteEmitWeight(float emission, float ref) {
        if ((emission < 0.0f) || !(ref > 0.0f)) {
            return 1.0f;
        }

        return std::min(emission / ref, 1.0f);
    }

    // Pixels that differ between two framebuffer images. Unchanged runs are skipped a block at a time, as nearly all of a frame's pixels match.
    inline uint32_t changedPixels(const uint8_t* a, const uint8_t* b, uint32_t bytes, uint32_t pixelBytes) {
        const uint32_t blockBytes = pixelBytes * 32;
        uint32_t changed = 0;
        for (uint32_t block = 0; block < bytes; block += blockBytes) {
            const uint32_t end = std::min(block + blockBytes, bytes);
            if (std::memcmp(a + block, b + block, end - block) == 0) {
                continue;
            }
            for (uint32_t i = block; (i + pixelBytes) <= end; i += pixelBytes) {
                changed += (std::memcmp(a + i, b + i, pixelBytes) != 0) ? 1u : 0u;
            }
        }
        return changed;
    }

    // Alpha-weighted average colour of a tile in TMEM (see tmemTexel). coverage = mean alpha; false when nothing is visible.
    inline bool tmemAverage(const uint8_t* tmem, uint8_t fmt, uint8_t siz, uint32_t tmemWord, uint32_t lineWords, uint32_t width, uint32_t height, uint8_t palette, float out[3], float& coverage) {
        if ((siz > 3) || (fmt == 1) || ((siz == 3) && (fmt != 0)) || (width == 0) || (height == 0)) {
            return false;
        }

        // RGBA32 keeps red and green in the low half of TMEM and blue and alpha at the same offset in the high half.
        const auto rgba32 = [&](uint32_t x, uint32_t y, float c[4]) {
            const uint32_t half = ((tmemWord + y * lineWords) * 8 + x * 2) & 0x7FF;
            c[0] = float(tmem[half]) / 255.0f;
            c[1] = float(tmem[(half + 1) & 0x7FF]) / 255.0f;
            c[2] = float(tmem[half | 0x800]) / 255.0f;
            c[3] = float(tmem[((half + 1) & 0x7FF) | 0x800]) / 255.0f;
        };

        // At most 8x8 samples: it runs per draw call while recording.
        const uint32_t stepX = std::max<uint32_t>(1, width / 8);
        const uint32_t stepY = std::max<uint32_t>(1, height / 8);
        double sum[3] = {};
        double alpha = 0.0;
        uint32_t n = 0;
        for (uint32_t y = 0; y < height; y += stepY) {
            for (uint32_t x = 0; x < width; x += stepX) {
                float c[4] = {};
                if (siz == 3) {
                    rgba32(x, y, c);
                }
                else if (!tmemTexel(tmem, fmt, siz, tmemWord, lineWords, palette, x, y, c)) {
                    return false;
                }

                sum[0] += c[0] * c[3];
                sum[1] += c[1] * c[3];
                sum[2] += c[2] * c[3];
                alpha += c[3];
                ++n;
            }
        }
        coverage = (n > 0) ? float(alpha / n) : 0.0f;
        if (!(alpha > 1e-6)) {
            return false;
        }
        for (int k = 0; k < 3; ++k) {
            out[k] = float(sum[k] / alpha);
        }
        return true;
    }
}
