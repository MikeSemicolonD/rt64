#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

// Dynamic point lights for the F5 profile: candidates recorded at draw time, resolved per displayed frame.
namespace rs64lights {
    constexpr uint32_t MaxLights = 32;

    enum Kind : uint8_t { KindNone = 0, KindSprite = 1, KindLaser = 2, KindMesh = 3 };

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
    };

    // Row-vector convention (v * M), matching hlslpp::mul(float4, float4x4).
    struct Mat4 {
        float m[4][4];
    };

    inline Mat4 identity() {
        Mat4 r{};
        for (int i = 0; i < 4; ++i) {
            r.m[i][i] = 1.0f;
        }
        return r;
    }

    inline Mat4 mul(const Mat4& a, const Mat4& b) {
        Mat4 r{};
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                float s = 0.0f;
                for (int k = 0; k < 4; ++k) {
                    s += a.m[i][k] * b.m[k][j];
                }
                r.m[i][j] = s;
            }
        }
        return r;
    }

    inline void transform(const float v[4], const Mat4& a, float out[4]) {
        for (int j = 0; j < 4; ++j) {
            out[j] = v[0] * a.m[0][j] + v[1] * a.m[1][j] + v[2] * a.m[2][j] + v[3] * a.m[3][j];
        }
    }

    // Element-wise equality relative to the larger magnitude (at least 1).
    inline bool sameMatrix(const Mat4& a, const Mat4& b, float relTol) {
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                const float scale = std::max(1.0f, std::max(std::fabs(a.m[i][j]), std::fabs(b.m[i][j])));
                if (!(std::fabs(a.m[i][j] - b.m[i][j]) <= relTol * scale)) {
                    return false;
                }
            }
        }
        return true;
    }

    inline bool inverse(const Mat4& a, Mat4& out) {
        double w[4][8];
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                w[i][j] = a.m[i][j];
                w[i][j + 4] = (i == j) ? 1.0 : 0.0;
            }
        }
        for (int c = 0; c < 4; ++c) {
            int piv = c;
            for (int r = c + 1; r < 4; ++r) {
                if (std::fabs(w[r][c]) > std::fabs(w[piv][c])) piv = r;
            }
            if (std::fabs(w[piv][c]) < 1e-12) {
                return false;
            }
            if (piv != c) {
                for (int j = 0; j < 8; ++j) std::swap(w[piv][j], w[c][j]);
            }
            const double d = w[c][c];
            for (int j = 0; j < 8; ++j) w[c][j] /= d;
            for (int r = 0; r < 4; ++r) {
                if (r == c) continue;
                const double f = w[r][c];
                for (int j = 0; j < 8; ++j) w[r][j] -= f * w[c][j];
            }
        }
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                out.m[i][j] = (float)w[i][j + 4];
            }
        }
        return true;
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

    // Player-craft engine glows: the glow cards' centres in their part's model space (from ROGUESQ_DUMP_MESH_VERTS), lit just behind the card.
    // X-wing nozzles ride on the S-foil parts p1/p2 (two each), the Y-wing's are both in p0.3, the A-wing's are the nozzle rings closing its engine parts p4/p6.
    constexpr uint32_t MaxExhaustLights = 5;

    struct ExhaustEntry {
        const char* key;
        uint32_t count;
        float pos[MaxExhaustLights][3];
        float rgb[3];
    };

    inline const ExhaustEntry* exhaustTable(size_t& n) {
        static const ExhaustEntry kExhaust[] = {
            { "xwing#s0.p1.2", 2, { { -1093.0f, -622.0f, -4828.0f }, { 1093.0f, 622.0f, -4828.0f } }, { 1.0f, 0.45f, 0.4f } },
            { "xwing#s0.p2.2", 2, { { -1093.0f, 622.0f, -4828.0f }, { 1093.0f, -622.0f, -4828.0f } }, { 1.0f, 0.45f, 0.4f } },
            { "ywing#s0.p0.3", 2, { { -2393.0f, 62.0f, -3097.0f }, { 2393.0f, 62.0f, -3097.0f } }, { 1.0f, 0.45f, 0.4f } },
            { "vwing#s0.p0.3", 2, { { -808.0f, -46.0f, -1962.0f }, { 808.0f, -46.0f, -1962.0f } }, { 1.0f, 0.45f, 0.4f } },
            { "awing#s0.p4.0", 1, { { 883.0f, 107.0f, -2870.0f } }, { 1.0f, 0.6f, 0.35f } },
            { "awing#s0.p6.0", 1, { { -883.0f, 107.0f, -2870.0f } }, { 1.0f, 0.6f, 0.35f } },
            // Falcon: the exhaust strip follows the rear hull arc (radius ~7740 about the disc centre).
            { "falcon#s0.p0.0", 5, { { 0.0f, 12.0f, -7740.0f }, { -1500.0f, 12.0f, -7593.0f }, { 1500.0f, 12.0f, -7593.0f }, { -3000.0f, 12.0f, -7135.0f }, { 3000.0f, 12.0f, -7135.0f } }, { 0.6f, 0.75f, 1.0f } },
        };
        n = sizeof(kExhaust) / sizeof(kExhaust[0]);
        return kExhaust;
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

    // Inline (per-frame) draws are identified by a part's node mesh, "<name>#s<set>.p<part>.0"; its exhaust entry is the glow card in that part.
    inline const char* exhaustKeyForPart(const char* meshKey) {
        if (meshKey == nullptr) {
            return nullptr;
        }
        const char* dot = std::strrchr(meshKey, '.');
        if (dot == nullptr) {
            return nullptr;
        }
        const size_t prefix = size_t(dot - meshKey) + 1;
        size_t n = 0;
        const ExhaustEntry* t = exhaustTable(n);
        for (size_t i = 0; i < n; ++i) {
            if ((std::strncmp(t[i].key, meshKey, prefix) == 0) && (std::strchr(t[i].key + prefix, '.') == nullptr)) {
                return t[i].key;
            }
        }
        return nullptr;
    }

    inline uint32_t exhaustLights(const char* meshKey, float radius, float intensity, Candidate* out, uint32_t cap) {
        constexpr float kBehind = 200.0f;
        if (meshKey == nullptr) {
            return 0;
        }
        size_t count = 0;
        const ExhaustEntry* table = exhaustTable(count);
        for (size_t t = 0; t < count; ++t) {
            const ExhaustEntry& e = table[t];
            if (std::strcmp(meshKey, e.key) != 0) {
                continue;
            }
            uint32_t n = 0;
            for (uint32_t i = 0; (i < e.count) && (n < cap); ++i, ++n) {
                out[n] = Candidate{};
                out[n].local[0] = e.pos[i][0];
                out[n].local[1] = e.pos[i][1];
                out[n].local[2] = e.pos[i][2] - kBehind;
                out[n].color[0] = e.rgb[0];
                out[n].color[1] = e.rgb[1];
                out[n].color[2] = e.rgb[2];
                out[n].radius = radius;
                out[n].intensity = intensity;
                out[n].kind = KindMesh;
                out[n].shadowStart = radius;
                out[n].modelUnits = true;
            }
            return n;
        }
        return 0;
    }

    // meshKey is a mesh-registry key: "<hob object name>#s<set>.p<part>.<index>".
    inline bool laserLight(const char* meshKey, float radius, Candidate& out) {
        struct Entry {
            const char* name;
            float rgb[3];
        };
        static const Entry kLasers[] = {
            { "red_laser", { 1.0f, 0.15f, 0.1f } },
            { "green_laser", { 0.2f, 1.0f, 0.25f } },
            { "ion_laser", { 0.35f, 0.55f, 1.0f } },
        };
        if (meshKey == nullptr) {
            return false;
        }
        for (const Entry& e : kLasers) {
            const size_t n = std::strlen(e.name);
            if ((std::strncmp(meshKey, e.name, n) == 0) && (meshKey[n] == '#')) {
                out = Candidate{};
                out.color[0] = e.rgb[0];
                out.color[1] = e.rgb[1];
                out.color[2] = e.rgb[2];
                out.radius = radius;
                out.intensity = 1.0f;
                out.kind = KindLaser;
                out.emissive = true;
                return true;
            }
        }
        return false;
    }

    // Upgrade pickup (mesh `r_pow`, npcPowerUpUpdate): the mission-select ring's warm yellow from the translucent core; the solid frame casts. Zero intensity = none.
    inline bool pickupLight(const char* meshKey, float radius, float intensity, Candidate& out) {
        static const char kName[] = "r_pow";
        const size_t n = sizeof(kName) - 1;
        if ((meshKey == nullptr) || !(intensity > 0.0f) || (std::strncmp(meshKey, kName, n) != 0) || (meshKey[n] != '#')) {
            return false;
        }
        out = Candidate{};
        out.color[0] = 1.0f;
        out.color[1] = 0.8f;
        out.color[2] = 0.35f;
        out.radius = radius;
        out.intensity = intensity;
        out.kind = KindLaser;
        out.emissive = false;
        return true;
    }

    // Proton torpedo: a red-orange glow riding on the torpedo mesh, carried by the laser path; the body is physical and casts (its trail is depth-write-off sprites).
    inline bool torpedoLight(const char* meshKey, float radius, Candidate& out) {
        static const char kName[] = "ph_torp";
        const size_t n = sizeof(kName) - 1;
        if ((meshKey == nullptr) || (std::strncmp(meshKey, kName, n) != 0) || (meshKey[n] != '#')) {
            return false;
        }
        out = Candidate{};
        out.color[0] = 1.0f;
        out.color[1] = 0.4f;
        out.color[2] = 0.2f;
        out.radius = radius;
        out.intensity = 1.0f;
        out.kind = KindLaser;
        return true;
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

    // Warm glow of the table's lit ring: `count` camera-space lights on a circle in the table plane (axisX, axisD), lifted along `up`. Zero intensity = none.
    inline uint32_t menuRingLights(const float center[3], const float axisX[3], const float axisD[3], const float up[3], float ringRadius, float lift, uint32_t count, float radius, float intensity, Candidate* out, uint32_t maxOut) {
        if (!(intensity > 0.0f)) {
            return 0;
        }
        const uint32_t n = std::min(count, maxOut);
        for (uint32_t i = 0; i < n; ++i) {
            const float a = 6.2831853f * float(i) / float(count);
            const float ca = std::cos(a), sa = std::sin(a);
            Candidate& c = out[i];
            c = Candidate{};
            c.transformIndex = CameraSpace;
            for (int k = 0; k < 3; ++k) {
                c.local[k] = center[k] + ringRadius * (ca * axisX[k] + sa * axisD[k]) + lift * up[k];
            }
            c.color[0] = 1.0f;
            c.color[1] = 0.8f;
            c.color[2] = 0.35f;
            c.radius = radius;
            c.intensity = intensity;
            c.kind = KindMesh;
        }
        return n;
    }

    // RT64's vertex path: clip = v * viewProj; ndc = (x/w, -y/w, z/w); screen = ndc * vpScale + vpTranslate;
    // raster ndc = (screen.xy - res/2) / (res.x/2, -res.y/2); final = raster ndc * screenScale + screenOffset; depth = screen.z.
    struct ScreenMap {
        float vpScale[3];
        float vpTranslate[3];
        float res[2];
        float screenScale[2];
        float screenOffset[2];
    };

    // The post-divide steps are affine per axis, so they fold into one matrix applied in clip space.
    inline Mat4 clipToScreen(const ScreenMap& s) {
        const float hx = s.res[0] * 0.5f;
        const float hy = s.res[1] * 0.5f;
        Mat4 t{};
        t.m[0][0] = s.vpScale[0] / hx * s.screenScale[0];
        t.m[3][0] = (s.vpTranslate[0] - hx) / hx * s.screenScale[0] + s.screenOffset[0];
        t.m[1][1] = s.vpScale[1] / hy * s.screenScale[1];
        t.m[3][1] = -(s.vpTranslate[1] - hy) / hy * s.screenScale[1] + s.screenOffset[1];
        t.m[2][2] = s.vpScale[2];
        t.m[3][2] = s.vpTranslate[2];
        t.m[3][3] = 1.0f;
        return t;
    }

    // Maps (final ndc x, final ndc y, depth, 1) back to view space (divide by w after the multiply).
    inline bool viewFromScreen(const Mat4& viewProj, const ScreenMap& s, Mat4& out) {
        return inverse(mul(viewProj, clipToScreen(s)), out);
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
                if (sameColor && ((dx * dx + dy * dy + dz * dz) <= (eps * eps))) {
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

    // Nearest light edge to the camera (origin) first; zero-intensity lights are dropped.
    inline size_t selectLights(const Resolved* in, size_t n, Resolved* out, size_t maxOut) {
        std::vector<std::pair<float, size_t>> order;
        order.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            if (!(in[i].intensity > 0.0f) || !(in[i].radius > 0.0f)) continue;
            const float d = std::sqrt(in[i].pos[0] * in[i].pos[0] + in[i].pos[1] * in[i].pos[1] + in[i].pos[2] * in[i].pos[2]);
            order.emplace_back(d - in[i].radius, i);
        }
        const size_t count = std::min(order.size(), maxOut);
        std::partial_sort(order.begin(), order.begin() + count, order.end());
        for (size_t i = 0; i < count; ++i) {
            out[i] = in[order[i].second];
        }
        return count;
    }

    struct Sun {
        float dir[3] = {};
        float color[3] = {};
        bool valid = false;
        // Hangar: the renderer derives the world up from the view's largest draw (the ship) and skips floor-facing receivers.
        bool fromModel = false;
        // Boot sequence: dir is the authored light in camera space until the renderer rotates it by the frame's view.
        bool worldAuthored = false;
        float skip[4] = {};
        // Level sun: its world direction (levelSunWorld), which the renderer turns to camera space through the frame's terrain transform; the RDRAM
        // camera matrix the GBI uses is not always the live one (Calamari: the sun stayed fixed to the camera).
        bool levelWorld = false;
        float world[3] = {};
    };

    // The level light points toward the sun with y flipped relative to the y-down world (Tatooine's sun meshes); R maps world to camera as out = R * v.
    // A sun stored below the horizon (level 6; the game never shades with it) is mirrored above it.
    // Rotation of a row-vector view matrix (v * M) as sunCameraDir's R (out = R * v), each axis normalised.
    inline bool viewRotation(const Mat4& view, float R[9]) {
        for (int i = 0; i < 3; ++i) {
            const float len = std::sqrt(view.m[i][0] * view.m[i][0] + view.m[i][1] * view.m[i][1] + view.m[i][2] * view.m[i][2]);
            if (!(len > 1e-6f)) {
                return false;
            }
            for (int j = 0; j < 3; ++j) {
                R[j * 3 + i] = view.m[i][j] / len;
            }
        }
        return true;
    }

    // minTanElev lifts a sun lower than atan(minTanElev) above the horizon to that elevation.
    inline bool sunCameraDir(const float worldL[3], const float viewR[9], float out[3], float minTanElev = 0.0f) {
        const float horizontal = std::sqrt(worldL[0] * worldL[0] + worldL[2] * worldL[2]);
        const float w[3] = { worldL[0], -std::max(std::fabs(worldL[1]), horizontal * minTanElev), worldL[2] };
        float v[3];
        for (int i = 0; i < 3; ++i) {
            v[i] = viewR[i * 3 + 0] * w[0] + viewR[i * 3 + 1] * w[1] + viewR[i * 3 + 2] * w[2];
        }
        const float len = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        if (!(len > 1e-6f)) {
            return false;
        }
        for (int i = 0; i < 3; ++i) {
            out[i] = v[i] / len;
        }
        return true;
    }

    // Level light record (0x80136E20) to the sun's world direction: its horizontal part points away from the sun the baked terrain shading uses (Tatooine, level 6), so turn it 180 degrees about the vertical.
    inline void levelSunWorld(const float rec[3], float out[3]) {
        out[0] = -rec[0];
        out[1] = rec[1];
        out[2] = -rec[2];
    }

    // Sun given in a model's frame (e.g. "straight down" for a ship resting on the hangar floor), mapped to camera space through its row-vector modelview.
    inline bool modelSunDir(const float modelL[3], const Mat4& model, float out[3]) {
        float R[9];
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                R[i * 3 + j] = model.m[j][i];
            }
        }
        return sunCameraDir(modelL, R, out);
    }

    // The up direction most of the scene agrees on: each candidate (xyz per entry) scores the weight of every candidate within acos(cosTol) of it.
    // Hangar: floor, walls, platform and parked ships share it; a tilted prop or the bobbing selected ship is outvoted.
    inline bool consensusUp(const float* ups, const uint32_t* weights, size_t n, float cosTol, float out[3]) {
        uint64_t bestScore = 0;
        size_t best = n;
        for (size_t i = 0; i < n; ++i) {
            uint64_t score = 0;
            for (size_t j = 0; j < n; ++j) {
                const float d = ups[i * 3] * ups[j * 3] + ups[i * 3 + 1] * ups[j * 3 + 1] + ups[i * 3 + 2] * ups[j * 3 + 2];
                score += (d > cosTol) ? weights[j] : 0;
            }
            if (score > bestScore) {
                bestScore = score;
                best = i;
            }
        }
        if (best == n) {
            return false;
        }

        for (int k = 0; k < 3; ++k) {
            out[k] = ups[best * 3 + k];
        }

        return true;
    }

    // Camera motion since the previous frame (row vectors: previous camera space = current * out), voted over static objects: each candidate is an
    // object's inverse(current) * previous, weighted by its triangles; a moving object (the selected ship, a fighter) disagrees with the rest.
    // Rotation within rotTol per entry, translation within posTol + 0.002 x its length. False when no candidate carries half the weight.
    inline bool consensusCameraDelta(const Mat4* deltas, const uint32_t* weights, size_t n, float rotTol, float posTol, Mat4& out) {
        auto agree = [&](const Mat4& a, const Mat4& b) {
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    if (std::fabs(a.m[r][c] - b.m[r][c]) > rotTol) {
                        return false;
                    }
                }
            }
            const float len = std::sqrt(a.m[3][0] * a.m[3][0] + a.m[3][1] * a.m[3][1] + a.m[3][2] * a.m[3][2]);
            for (int c = 0; c < 3; ++c) {
                if (std::fabs(a.m[3][c] - b.m[3][c]) > (posTol + 0.002f * len)) {
                    return false;
                }
            }
            return true;
        };
        uint64_t total = 0;
        uint64_t bestScore = 0;
        size_t best = n;
        for (size_t i = 0; i < n; ++i) {
            total += weights[i];
            uint64_t score = 0;
            for (size_t j = 0; j < n; ++j) {
                score += agree(deltas[i], deltas[j]) ? weights[j] : 0;
            }
            if (score > bestScore) {
                bestScore = score;
                best = i;
            }
        }
        if ((best == n) || ((bestScore * 2) < total)) {
            return false;
        }

        out = deltas[best];

        return true;
    }

    // Hangar floor: receivers whose normal is within acos(skip.w) of the world up (skip.xyz, camera space) keep the game's painted shadow; w == 0 disables. Mirrored by RS64ShadowsPS.
    inline bool receiverSkipped(const float n[3], const float skip[4]) {
        return (skip[3] > 0.0f) && ((n[0] * skip[0] + n[1] * skip[1] + n[2] * skip[2]) > skip[3]);
    }

    struct IndexRange {
        uint32_t start;
        uint32_t triangleCount;
    };

    inline size_t gatherCasterIndices(const uint32_t* faceIndices, size_t faceIndexCount, const IndexRange* ranges, size_t rangeCount, std::vector<uint32_t>& out) {
        size_t added = 0;
        for (size_t r = 0; r < rangeCount; ++r) {
            const size_t count = size_t(ranges[r].triangleCount) * 3;
            if ((count == 0) || (size_t(ranges[r].start) + count > faceIndexCount)) continue;
            out.insert(out.end(), faceIndices + ranges[r].start, faceIndices + ranges[r].start + count);
            added += count;
        }
        return added;
    }

    enum CasterClass : uint8_t { CasterNone = 0, CasterOpaque = 1, CasterCutout = 2 };

    struct CasterMode {
        bool zUpd = false;
        bool primDepth = false;
        bool xlu = false;
        bool fillOrCopy = false;
        bool extended = false;
        bool alphaCompare = false;
        bool cvgXAlpha = false;
        uint32_t triangles = 0;
    };

    // Depth-writing, scene-depth, non-XLU triangles cast; alpha-compare or coverage x alpha ones only through the alpha test.
    inline CasterClass casterClass(const CasterMode& m) {
        if (!m.zUpd || m.primDepth || m.xlu || m.fillOrCopy || m.extended || (m.triangles == 0)) {
            return CasterNone;
        }
        return (m.alphaCompare || m.cvgXAlpha) ? CasterCutout : CasterOpaque;
    }

    // Reference for the threshold RS64CutoutAlpha.hlsli computes itself: raster discards alpha < blend alpha for G_AC_THRESHOLD; dither and coverage x alpha use the midpoint.
    inline float cutoutThreshold(bool thresholdCompare, float blendAlpha) {
        return thresholdCompare ? blendAlpha : 0.5f;
    }

    // Laser bolts (noCast) never cast; glow cards (noCastCutout) lose only their cutout faces; with cutouts off no cutout casts.
    inline CasterClass casterAfterDeny(CasterClass cc, bool noCast, bool noCastCutout, bool cutoutOn) {
        if (noCast) {
            return CasterNone;
        }
        if ((cc == CasterCutout) && (noCastCutout || !cutoutOn)) {
            return CasterNone;
        }
        return cc;
    }

    // Per-axis UV gradient for the cutout alpha fetch: TextureSampler's mip = 0.5*log2(|g*tcScale|^2) - 0.25 lands on 0 (zero gradients give NaN). Mirrored by RS64CutoutAlpha.hlsli.
    inline float cutoutMipGradient(float tcScale) {
        return 1.18920712f / tcScale;
    }

    struct CutoutRange {
        IndexRange range;
        uint32_t instanceIndex = 0;
        uint32_t tileIndex = 0;
    };

    struct CutoutEntry {
        uint32_t firstTriangle;
        uint32_t instanceIndex;
        uint32_t tileIndex;
        uint32_t pad;
    };

    // Cutout BLAS indices plus one table entry per kept draw (first triangle in the BLAS, call index, RDP tile); ranges past the face buffer are dropped.
    inline size_t buildCutoutScene(const uint32_t* faceIndices, size_t faceIndexCount, const CutoutRange* ranges, size_t n, std::vector<uint32_t>& indices, std::vector<CutoutEntry>& table) {
        size_t added = 0;
        for (size_t r = 0; r < n; ++r) {
            const size_t count = size_t(ranges[r].range.triangleCount) * 3;
            if ((count == 0) || (size_t(ranges[r].range.start) + count > faceIndexCount)) {
                continue;
            }
            table.push_back({ uint32_t(indices.size() / 3), ranges[r].instanceIndex, ranges[r].tileIndex, 0 });
            indices.insert(indices.end(), faceIndices + ranges[r].range.start, faceIndices + ranges[r].range.start + count);
            added += count;
        }
        return added;
    }

    // Last entry whose first triangle is <= prim; mirrored by RS64CutoutAlpha.hlsli.
    inline uint32_t cutoutDrawFor(uint32_t prim, const CutoutEntry* table, size_t n) {
        if ((n == 0) || (prim < table[0].firstTriangle)) {
            return UINT32_MAX;
        }
        size_t lo = 0;
        size_t hi = n - 1;
        while (lo < hi) {
            const size_t mid = (lo + hi + 1) / 2;
            if (table[mid].firstTriangle <= prim) {
                lo = mid;
            }
            else {
                hi = mid - 1;
            }
        }
        return uint32_t(lo);
    }

    // Self-lit surfaces (the briefing room's light strips): emitters for GI and reflections, never casters.
    inline bool emissiveSurface(const char* meshKey) {
        static const char kName[] = "lights";
        const size_t n = sizeof(kName) - 1;
        return (meshKey != nullptr) && (std::strncmp(meshKey, kName, n) == 0) && (meshKey[n] == '#');
    }

    // Light-source glow cards are cutouts that must never cast.
    inline bool cutoutDenied(const char* meshKey) {
        static const char* const kDenied[] = { "i_ltsrc_hi", "i_lnd_hi" };
        if (meshKey == nullptr) {
            return false;
        }
        for (const char* name : kDenied) {
            const size_t n = std::strlen(name);
            if ((std::strncmp(meshKey, name, n) == 0) && (meshKey[n] == '#')) {
                return true;
            }
        }
        return false;
    }

    // HMP terrain (descriptor 0x80136DC0): W*H cells, each a 512-unit tile of 5x5 s8 heights (y = h << 4, base 0).
    constexpr int32_t TerrainTileSpan = 512;
    // The traced water surface's depth under the water level, heightmap units: at the level, just under the drawn water (lifted 2 record units), so
    // a pixel's terrain lookup finds it. Heights grow downward: water is where the ground is at or above the level (selectGridCellTileVariant 0x80046030).
    constexpr int32_t TerrainWaterDip = 0;

    // Inputs of the game's terrain vertex lighting: descriptor 0x80136DC0 ambient (+0x3C), cell scale (+0x10) and sun (light object at +0x48).
    struct TerrainShade {
        float ambient[3] = {};
        float sunRgb[3] = {};
        float sunDir[3] = {};
        float cellScale = 12.8f;
    };

    struct TerrainMap {
        uint32_t width = 0, height = 0;
        std::vector<int8_t> heights;
        std::vector<uint8_t> present;
        std::vector<int32_t> tileCell;
        float heightScale = 1.0f;
        uint64_t version = 0;
        // Cells that changed since map baseVersion (runtime edits; inclusive, with the one-cell border their normals and colours reach); baseVersion 0 = a new map.
        uint64_t baseVersion = 0;
        uint32_t dirtyX0 = 0, dirtyZ0 = 0, dirtyX1 = 0, dirtyZ1 = 0;
        // 25 RGB vertex colours per cell (row-major, like heights), filled only for the static terrain renderer.
        std::vector<uint8_t> colors;
        // Per cell: the tile's texture id (tile +0, an index into the level's lv_X_TEX table). Level id (gCurrentLevel) at capture.
        std::vector<uint16_t> texIds;
        uint32_t level = 0;
        // Per cell: 0 ground, 1 water (tile flag 0x08), 2 shore (0x10); the water surface itself animates and is drawn from TerrainPassFrame.
        std::vector<uint8_t> waterKind;
        // Water level at capture ([0x801375C0], heightmap units); ground below it shows the wave tiles instead.
        int32_t waterLevel = 0;
        TerrainShade shade;
    };

    // A cell's 25 vertex colours as the game lit them this frame (its record's colours, every light included), harvested for the terrain pass.
    struct TerrainCellColors {
        uint32_t cell = 0;
        uint8_t rgb[75] = {};
    };

    // Whole-map terrain pass for one workload: the anchor game call the GBI emitted at the first terrain record and the record-space origin (in cells).
    struct TerrainPassFrame {
        std::shared_ptr<const TerrainMap> map;
        uint32_t anchorCall = UINT32_MAX;
        int32_t originX = 0, originZ = 0;
        int32_t offsetX = 0, offsetZ = 0;
        // This frame's water level (heightmap units).
        int32_t waterLevel = 0;
        std::vector<TerrainCellColors> harvest;
        // The game's task counter (30 a second), the clock for the texture animations.
        uint64_t task = 0;
    };

    // A texture animation from lv_X_TEX: after the plain id -> bank entries come blocks of (id, frame count, frames per second, banks...), ending at 0xFFFF.
    struct TerrainTexAnim {
        uint32_t id = 0;
        uint32_t fps = 0;
        std::vector<uint32_t> banks;
    };

    // The table's own layout has no count of plain entries: blocks start at the first index at or above minFirst from which they parse cleanly to the 0xFFFF end.
    inline std::vector<TerrainTexAnim> terrainTextureAnims(const uint8_t* ids, uint32_t idsSize, uint32_t bankEntries, uint32_t minFirst) {
        const uint32_t n = idsSize / 4;
        auto at = [&](uint32_t i) {
            return (uint32_t(ids[i * 4]) << 8) | ids[i * 4 + 1];
        };
        for (uint32_t first = minFirst; first < n; ++first) {
            std::vector<TerrainTexAnim> anims;
            uint32_t i = first;
            bool ok = true;
            while (ok && (i < n) && (at(i) != 0xFFFF)) {
                const uint32_t count = (i + 2 < n) ? at(i + 1) : 0;
                ok = (at(i) < first) && (count > 0) && (at(i + 2) > 0) && (i + 3 + count <= n);
                if (ok) {
                    TerrainTexAnim &a = anims.emplace_back();
                    a.id = at(i);
                    a.fps = at(i + 2);
                    for (uint32_t k = 0; ok && (k < count); ++k) {
                        a.banks.push_back(at(i + 3 + k));
                        ok = a.banks.back() < bankEntries;
                    }
                    i += 3 + count;
                }
            }
            if (ok && (i < n)) {
                return anims;
            }
        }
        return {};
    }

    // CI4 64x64 + 16 RGBA5551 bank entry (level/tex, 2080 bytes; odd rows TMEM word-swapped) to 4096 RGBA8 texels, as the explorer's decodeTerrainTex.
    constexpr uint32_t TerrainTexBytes = 2080;
    inline void decodeTerrainTexture(const uint8_t* e, uint32_t out[4096]) {
        auto c5 = [](uint32_t v) {
            return (v << 3) | (v >> 2);
        };
        for (uint32_t i = 0; i < 4096; ++i) {
            const uint32_t src = ((i >> 6) & 1) ? (i ^ 8) : i;
            const uint8_t byte = e[src >> 1];
            const uint32_t idx = (src & 1) ? (byte & 15) : (byte >> 4);
            const uint32_t v = (uint32_t(e[2048 + idx * 2]) << 8) | e[2048 + idx * 2 + 1];
            out[i] = c5((v >> 11) & 31) | (c5((v >> 6) & 31) << 8) | (c5((v >> 1) & 31) << 16) | ((v & 1) ? 0xFF000000u : 0u);
        }
    }

    struct TerrainSample {
        int32_t x = 0, z = 0;
        uint32_t tile = 0;
    };

    struct TerrainFrame {
        std::shared_ptr<const TerrainMap> map;
        std::vector<TerrainSample> samples;
        uint32_t transformIndex = UINT32_MAX;
        std::vector<uint32_t> transforms;
        bool mixed = false;
        // The whole-map terrain pass knows the record-space origin exactly (in solveTerrainOrigin's terms); it replaces the vote, which open water starves.
        bool exact = false;
        int32_t exactX = 0, exactZ = 0;
    };

    // The game reloads the terrain root matrix mid-list under new transform indices; only a different matrix marks the frame mixed.
    inline void noteTerrainTransform(TerrainFrame& t, uint32_t ti, bool sameAsFirst) {
        if (t.transformIndex == UINT32_MAX) {
            t.transformIndex = ti;
            t.transforms.push_back(ti);
            return;
        }
        if (std::find(t.transforms.begin(), t.transforms.end(), ti) != t.transforms.end()) {
            return;
        }
        if (sameAsFirst) {
            t.transforms.push_back(ti);
        }
        else {
            t.mixed = true;
        }
    }

    inline bool isTerrainTransform(const TerrainFrame& t, uint32_t ti) {
        return std::find(t.transforms.begin(), t.transforms.end(), ti) != t.transforms.end();
    }

    inline uint64_t fnv1a64(const uint8_t* p, size_t n, uint64_t h = 0xcbf29ce484222325ull) {
        for (size_t i = 0; i < n; ++i) {
            h = (h ^ p[i]) * 1099511628211ull;
        }
        return h;
    }

    // tileCell[t] = the only cell using tile t, -1 if none, -2 if several.
    inline void buildTileCells(const uint16_t* tileOfCell, size_t cellCount, std::vector<int32_t>& out) {
        out.clear();
        for (size_t c = 0; c < cellCount; ++c) {
            const uint16_t t = tileOfCell[c];
            if (t >= out.size()) {
                out.resize(size_t(t) + 1, -1);
            }
            out[t] = (out[t] == -1) ? int32_t(c) : -2;
        }
    }

    // Same lattice, truncation and winding as f5_tile_grid for an unmorphed tile, so drawn near tiles coincide with it.
    inline void terrainTileMesh(const int8_t h[25], int sub, int32_t x0, int32_t z0, float heightScale, std::vector<float>& pos, std::vector<uint32_t>& idx) {
        const int N = 4 * sub;
        const uint32_t base = uint32_t(pos.size() / 3);
        float H[25];
        for (int k = 0; k < 25; ++k) {
            H[k] = (float)((int32_t)h[k] << 4);
        }
        for (int rr = 0; rr <= N; ++rr) {
            for (int cc = 0; cc <= N; ++cc) {
                const float fx = (float)cc / sub, fy = (float)rr / sub;
                const int xi = std::min((int)fx, 3), yi = std::min((int)fy, 3);
                const float tx = fx - xi, ty = fy - yi;
                const float hv = (H[yi * 5 + xi] * (1 - tx) + H[yi * 5 + xi + 1] * tx) * (1 - ty) + (H[(yi + 1) * 5 + xi] * (1 - tx) + H[(yi + 1) * 5 + xi + 1] * tx) * ty;
                const int32_t y = std::clamp((int32_t)(hv * heightScale), -32768, 32767);
                pos.push_back((float)(x0 + (int32_t)(fx * 128.0f)));
                pos.push_back((float)y);
                pos.push_back((float)(z0 + (int32_t)(fy * 128.0f)));
            }
        }
        for (int r = 0; r < N; ++r) {
            for (int c = 0; c < N; ++c) {
                const uint32_t a = base + uint32_t(r * (N + 1) + c), b = a + 1, cc = a + uint32_t(N + 1), d = cc + 1;
                const uint32_t tri[6] = { a, d, cc, a, b, d };
                idx.insert(idx.end(), tri, tri + 6);
            }
        }
    }

    // Cells past each map edge the water draw covers (RS64TerrainVS): past the game's terrain reach (64 cells around the camera at most), so an
    // ocean level's sea fogs out instead of ending in view.
    constexpr int32_t TerrainWaterMargin = 96;
    // How far (cells) the water's outer ring reaches past the margin, past any far plane (RS64TerrainVS kWaterFarCells).
    constexpr int32_t TerrainWaterFarCells = 4096;

    // The water draw's cells, row-major over the map and its margin: water and shore cells in the map, and every margin cell (so the map
    // never ends in a void); none on a map without water. fn(col, row, map cell), past the edge the nearest edge cell.
    template <typename F>
    inline void forEachTerrainWaterCell(const TerrainMap& m, F&& fn) {
        const int32_t w = int32_t(m.width), h = int32_t(m.height);
        const bool anyWater = std::any_of(m.waterKind.begin(), m.waterKind.end(), [](uint8_t k) { return k != 0; });
        if ((w == 0) || (h == 0) || !anyWater) {
            return;
        }

        for (int32_t row = -TerrainWaterMargin; row < h + TerrainWaterMargin; ++row) {
            for (int32_t col = -TerrainWaterMargin; col < w + TerrainWaterMargin; ++col) {
                const bool outside = (col < 0) || (row < 0) || (col >= w) || (row >= h);
                const size_t c = size_t(std::clamp(row, 0, h - 1)) * m.width + size_t(std::clamp(col, 0, w - 1));
                if (outside || ((c < m.waterKind.size()) && (m.waterKind[c] != 0))) {
                    fn(col, row, c);
                }
            }
        }
    }

    inline void buildTerrainMesh(const TerrainMap& m, int sub, std::vector<float>& pos, std::vector<uint32_t>& idx) {
        pos.clear();
        idx.clear();
        for (uint32_t row = 0; row < m.height; ++row) {
            for (uint32_t col = 0; col < m.width; ++col) {
                const size_t c = size_t(row) * m.width + col;
                if ((c >= m.present.size()) || !m.present[c] || ((c + 1) * 25 > m.heights.size())) {
                    continue;
                }
                terrainTileMesh(&m.heights[c * 25], sub, int32_t(col) * TerrainTileSpan, int32_t(row) * TerrainTileSpan, m.heightScale, pos, idx);
            }
        }

        // Water and shore cells also get a flat surface just under the drawn water (flat at the level), so traced rays meet water instead of the
        // seabed without the water surface shadowing itself.
        const float y = float((m.waterLevel + TerrainWaterDip) << 4) * m.heightScale;
        forEachTerrainWaterCell(m, [&](int32_t col, int32_t row, size_t) {
            const uint32_t a = uint32_t(pos.size() / 3);
            // The outer ring reaches TerrainWaterFarCells further, as the drawn water does.
            const float reach = float(TerrainWaterFarCells * TerrainTileSpan);
            const int32_t lastCol = int32_t(m.width) + TerrainWaterMargin - 1, lastRow = int32_t(m.height) + TerrainWaterMargin - 1;
            const float x0 = float(col * TerrainTileSpan) - ((col == -TerrainWaterMargin) ? reach : 0.0f), x1 = float((col + 1) * TerrainTileSpan) + ((col == lastCol) ? reach : 0.0f);
            const float z0 = float(row * TerrainTileSpan) - ((row == -TerrainWaterMargin) ? reach : 0.0f), z1 = float((row + 1) * TerrainTileSpan) + ((row == lastRow) ? reach : 0.0f);
            const float corners[12] = { x0, y, z0, x1, y, z0, x0, y, z1, x1, y, z1 };
            pos.insert(pos.end(), corners, corners + 12);
            const uint32_t tri[6] = { a, a + 3, a + 2, a, a + 1, a + 3 };
            idx.insert(idx.end(), tri, tri + 6);
        });
    }

    // The drawn water surface's lift over the water level (record units; heights grow downward), as kWaterLift in RS64Terrain.hlsli; the water fades
    // out over it as a shore rises.
    constexpr float TerrainWaterLift = 16.0f;

    // Hit colours for buildTerrainMesh's triangles, in its cell order: header (triangles per ground cell, first water triangle or 0, ground cells, the
    // drawn water surface's map-space y as float bits), then one rgba8 per ground cell and per water quad: the cell's texture average (texAvg, rgb8 per
    // cell) times its average vertex colour.
    inline void terrainHitColors(const TerrainMap& m, int sub, const std::vector<uint32_t>& texAvg, std::vector<uint32_t>& out) {
        const uint32_t per = uint32_t(2 * (4 * sub) * (4 * sub));
        out.assign(4, 0);
        auto cellColor = [&](size_t c) {
            uint32_t rgb = 0;
            for (int ch = 0; ch < 3; ++ch) {
                uint32_t sum = 0;
                for (int k = 0; (k < 25) && ((c + 1) * 75 <= m.colors.size()); ++k) {
                    sum += m.colors[c * 75 + k * 3 + ch];
                }
                const uint32_t shade = ((c + 1) * 75 <= m.colors.size()) ? (sum / 25) : 255;
                const uint32_t tex = (c < texAvg.size()) ? ((texAvg[c] >> (8 * ch)) & 0xFF) : 255;
                rgb |= ((tex * shade + 127) / 255) << (8 * ch);
            }
            return rgb;
        };

        uint32_t ground = 0;
        for (size_t c = 0; c < size_t(m.width) * m.height; ++c) {
            if ((c >= m.present.size()) || !m.present[c] || ((c + 1) * 25 > m.heights.size())) {
                continue;
            }
            out.push_back(cellColor(c));
            ++ground;
        }

        bool water = false;
        forEachTerrainWaterCell(m, [&](int32_t, int32_t, size_t c) {
            out.push_back(cellColor(c));
            water = true;
        });

        out[0] = per;
        out[1] = water ? ground * per : 0;
        out[2] = ground;
        const float surfaceY = (float(m.waterLevel * 16) - TerrainWaterLift) * m.heightScale;
        memcpy(&out[3], &surfaceY, sizeof(float));
    }

    // A level light from the HMP lighting block, as the terrain lighting reads it (rgb +0, type byte +0xE, pos +0x10, radius +0x1C).
    struct TerrainLight {
        float rgb[3] = {};
        float pos[3] = {};
        float radius = 0.0f;
        bool box = false;
    };

    inline int8_t terrainHeight(const TerrainMap& m, uint32_t col, uint32_t row, int k) {
        return m.heights[(size_t(row) * m.width + col) * 25 + size_t(k)];
    }

    // Central differences over the height lattice, as emitGridCellTileStrip 0x80049814 builds them; past the map edge the cell's own tile stands in.
    inline void terrainVertexNormal(const TerrainMap& m, uint32_t col, uint32_t row, int px, int py, float cellScale, float out[3]) {

#if defined(__clang__)
#pragma clang fp contract(off)
#endif

        const uint32_t l = (col > 0) ? col - 1 : col;
        const uint32_t r = (col + 1 < m.width) ? col + 1 : col;
        const uint32_t u = (row > 0) ? row - 1 : row;
        const uint32_t d = (row + 1 < m.height) ? row + 1 : row;
        const int hx0 = (px > 0) ? terrainHeight(m, col, row, py * 5 + px - 1) : terrainHeight(m, l, row, py * 5 + 3);
        const int hx1 = (px < 4) ? terrainHeight(m, col, row, py * 5 + px + 1) : terrainHeight(m, r, row, py * 5 + 1);
        const int hz0 = (py > 0) ? terrainHeight(m, col, row, (py - 1) * 5 + px) : terrainHeight(m, col, u, 15 + px);
        const int hz1 = (py < 4) ? terrainHeight(m, col, row, (py + 1) * 5 + px) : terrainHeight(m, col, d, 5 + px);
        const float ny = 256.0f / cellScale;
        const float nx = float(hx0 - hx1);
        const float nz = float(hz0 - hz1);
        const float len = std::sqrt(nx * nx + ny * ny + nz * nz);

        out[0] = nx / len;
        out[1] = ny / len;
        out[2] = nz / len;
    }

    // transformWorldPosToGridSpace 0x8004A6C0 (misnamed): ambient + sun * N.L, plus box and point lights; p = (col + px/4, height, row + py/4).
    inline void terrainVertexColor(const TerrainShade& s, const float n[3], const float p[3], const TerrainLight* lights, size_t count, uint8_t out[3]) {

#if defined(__clang__)
#pragma clang fp contract(off)
#endif

        float c[3] = { s.ambient[0], s.ambient[1], s.ambient[2] };
        const float ndl = n[0] * s.sunDir[0] + n[1] * s.sunDir[1] + n[2] * s.sunDir[2];

        if (0.0f < ndl) {
            for (int i = 0; i < 3; ++i) {
                c[i] = s.ambient[i] + s.sunRgb[i] * ndl;
            }
        }

        for (size_t k = 0; k < count; ++k) {
            const TerrainLight& L = lights[k];
            float scale = 1.0f;

            if (L.box) {
                const float dx = std::fabs(p[0] - L.pos[0]);
                const float dz = std::fabs(p[2] - L.pos[2]);

                if ((L.radius <= dx) || (L.radius <= dz)) {
                    continue;
                }

                scale = std::sqrt(1.0f - (((dz < dx) ? dx : dz) / L.radius));
            }
            else {
                const float dx = p[0] - L.pos[0];
                const float dy = p[1] - L.pos[1];
                const float dz = p[2] - L.pos[2];
                const float nd = n[0] * dx + n[1] * dy + n[2] * dz;
                const float d2 = dx * dx + dy * dy + dz * dz;

                if (!(0.01f < nd) || !(d2 < L.radius * L.radius)) {
                    continue;
                }

                if (!(d2 < 1e-4f)) {
                    scale = nd * ((1.0f / std::sqrt(d2)) - (1.0f / L.radius));
                }
            }

            for (int i = 0; i < 3; ++i) {
                c[i] = c[i] + L.rgb[i] * scale;
            }
        }

        for (int i = 0; i < 3; ++i) {
            out[i] = (c[i] >= 256.0f) ? 255 : ((c[i] < 0.0f) ? 0 : uint8_t(c[i]));
        }
    }

    // The 25 colours of one cell; matches the game's cell+0x54 block (verified bit-exact on PJ64 Tatooine goldens, lights untested).
    inline void terrainCellColors(const TerrainMap& m, uint32_t col, uint32_t row, const TerrainShade& s, const TerrainLight* lights, size_t count, uint8_t out[25 * 3]) {

#if defined(__clang__)
#pragma clang fp contract(off)
#endif

        const float hscale = s.cellScale * (1.0f / 256.0f) * 0.5f;

        for (int k = 0; k < 25; ++k) {
            const int px = k % 5, py = k / 5;
            float n[3];

            terrainVertexNormal(m, col, row, px, py, s.cellScale, n);

            const float p[3] = { float(col) + float(px) * 0.25f, float(terrainHeight(m, col, row, k)) * hscale, float(row) + float(py) * 0.25f };

            terrainVertexColor(s, n, p, lights, count, out + k * 3);
        }
    }

    // Area-weighted vertex normals (+y up), welded across tile seams by integer (x, z) so the shadow pass sees a smooth heightfield instead of facets.
    inline void buildTerrainNormals(const std::vector<float>& pos, const std::vector<uint32_t>& idx, std::vector<float>& out) {
        const size_t verts = pos.size() / 3;
        std::unordered_map<uint64_t, uint32_t> weld;
        weld.reserve(verts);
        std::vector<uint32_t> slot(verts);
        for (size_t v = 0; v < verts; ++v) {
            const uint64_t key = (uint64_t(uint32_t(int32_t(pos[v * 3]))) << 32) | uint32_t(int32_t(pos[v * 3 + 2]));
            slot[v] = weld.emplace(key, uint32_t(weld.size())).first->second;
        }
        std::vector<float> acc(weld.size() * 3, 0.0f);
        for (size_t t = 0; t + 2 < idx.size(); t += 3) {
            const float* p0 = &pos[idx[t] * 3];
            const float* p1 = &pos[idx[t + 1] * 3];
            const float* p2 = &pos[idx[t + 2] * 3];
            const float e1[3] = { p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2] };
            const float e2[3] = { p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2] };
            float n[3] = { e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0] };
            if (n[1] < 0.0f) {
                n[0] = -n[0];
                n[1] = -n[1];
                n[2] = -n[2];
            }
            for (int k = 0; k < 3; ++k) {
                float* a = &acc[size_t(slot[idx[t + k]]) * 3];
                a[0] += n[0];
                a[1] += n[1];
                a[2] += n[2];
            }
        }
        out.resize(verts * 3);
        for (size_t v = 0; v < verts; ++v) {
            const float* a = &acc[size_t(slot[v]) * 3];
            const float l = std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
            out[v * 3 + 0] = (l > 0.0f) ? a[0] / l : 0.0f;
            out[v * 3 + 1] = (l > 0.0f) ? a[1] / l : 1.0f;
            out[v * 3 + 2] = (l > 0.0f) ? a[2] / l : 0.0f;
        }
    }

    // Normal through a 3x4 instance transform: inverse-transpose via the cofactor rows, normalized. Mirrored by RS64TerrainNormal.hlsli.
    inline void cofactorNormal(const float m[3][4], const float n[3], float out[3]) {
        const float a[3] = { m[0][0], m[0][1], m[0][2] };
        const float b[3] = { m[1][0], m[1][1], m[1][2] };
        const float c[3] = { m[2][0], m[2][1], m[2][2] };
        const float bc[3] = { b[1] * c[2] - b[2] * c[1], b[2] * c[0] - b[0] * c[2], b[0] * c[1] - b[1] * c[0] };
        const float ca[3] = { c[1] * a[2] - c[2] * a[1], c[2] * a[0] - c[0] * a[2], c[0] * a[1] - c[1] * a[0] };
        const float ab[3] = { a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0] };
        const float det = a[0] * bc[0] + a[1] * bc[1] + a[2] * bc[2];
        const float s = (det < 0.0f) ? -1.0f : 1.0f;
        float o[3] = { bc[0] * n[0] + bc[1] * n[1] + bc[2] * n[2], ca[0] * n[0] + ca[1] * n[1] + ca[2] * n[2], ab[0] * n[0] + ab[1] * n[1] + ab[2] * n[2] };
        const float l = std::sqrt(o[0] * o[0] + o[1] * o[1] + o[2] * o[2]);
        out[0] = (l > 0.0f) ? s * o[0] / l : 0.0f;
        out[1] = (l > 0.0f) ? s * o[1] / l : 0.0f;
        out[2] = (l > 0.0f) ? s * o[2] / l : 0.0f;
    }

    // Terrain receivers: away from the sun is fully shadowed, fading in over ndl in [0, width]; blocked = traced fraction. Mirrored by RS64ShadowsPS.
    inline float terminatorShadow(float blocked, float ndl, float width) {
        return 1.0f - (1.0f - blocked) * std::clamp(ndl / width, 0.0f, 1.0f);
    }

    // Identity of the HMP map: tile-index array, tile table bytes (heights may land after the index array during a load), table address, size.
    inline uint64_t terrainMapKey(const uint16_t* tiles, size_t cells, const uint8_t* table, size_t tableBytes, uint32_t tab, uint32_t w, uint32_t h) {
        const uint32_t key[3] = { tab, w, h };
        uint64_t k = fnv1a64(reinterpret_cast<const uint8_t*>(tiles), cells * sizeof(uint16_t));
        k = fnv1a64(table, tableBytes, k);
        return fnv1a64(reinterpret_cast<const uint8_t*>(key), sizeof(key), k);
    }

    // Each sample whose tile maps to one cell votes for that cell's record-space offset; a strict majority of at least two wins,
    // or any votes that all agree with the previous frame's origin; with no votes at all (only flat far tiles drawn) the previous origin carries over.
    inline uint32_t solveTerrainOrigin(const TerrainMap& m, const TerrainSample* s, size_t n, int32_t& ox, int32_t& oz, bool hasPrev = false, int32_t prevX = 0, int32_t prevZ = 0) {
        std::map<std::pair<int32_t, int32_t>, uint32_t> votes;
        uint32_t total = 0;
        for (size_t i = 0; i < n; ++i) {
            if ((m.width == 0) || (s[i].tile >= m.tileCell.size()) || (m.tileCell[s[i].tile] < 0)) {
                continue;
            }
            const int32_t cell = m.tileCell[s[i].tile];
            const int32_t col = cell % int32_t(m.width), row = cell / int32_t(m.width);
            ++votes[{ col * TerrainTileSpan - s[i].x, row * TerrainTileSpan - s[i].z }];
            ++total;
        }
        uint32_t best = 0;
        for (const auto& v : votes) {
            if (v.second > best) {
                best = v.second;
                ox = v.first.first;
                oz = v.first.second;
            }
        }
        if ((best >= 2) && (best * 2 > total)) {
            return best;
        }
        if (hasPrev && (best > 0) && (best == total) && (ox == prevX) && (oz == prevZ)) {
            return best;
        }
        if (hasPrev && (total == 0)) {
            ox = prevX;
            oz = prevZ;
            return 1;
        }
        return 0;
    }

    // Row-major 3x4 (TLAS layout) for map-space vertices: camera = (map - (ox, 0, oz)) * recordToCamera.
    inline void terrainInstanceTransform(const Mat4& recordToCamera, int32_t ox, int32_t oz, float out[3][4]) {
        const auto& m = recordToCamera.m;
        for (int j = 0; j < 3; ++j) {
            out[j][0] = m[0][j];
            out[j][1] = m[1][j];
            out[j][2] = m[2][j];
            out[j][3] = m[3][j] - (float)ox * m[0][j] - (float)oz * m[2][j];
        }
    }

    inline float envFloat(const char* name, float def) {
        const char* e = std::getenv(name);
        return (e && *e) ? (float)std::atof(e) : def;
    }

    struct Config {
        std::atomic<bool> enabled;
        int debugMode;
        float gain;
        float wrap;
        float spriteRadius;
        float spriteRadiusMax;
        float laserRadius;
        float debugRange;
        bool log;
        std::atomic<bool> shadows;
        int shadowsDebug;
        float shadowStrength;
        float shadowTMin;
        float shadowTMax;
        float shadowNormalBias;
        bool shadowTerrain;
        int shadowTerrainSub;
        float shadowTerrainTMin;
        bool lightShadows;
        float lightShadowEnd;
        float menuHoloRadius;
        float menuHoloGain;
        float menuHoloFalloff;
        float menuRingGain;
        float menuImageGain;
        float temporalAlpha;
        float hangarFloorCos;
        float torpedoRadius;
        std::atomic<bool> fogShafts;
        int fogShaftsSteps;
        float fogShaftsStrength;
        int fogShaftsDebug;
        bool shadowCutout;
        float pickupGain;
        float pickupRadius;
        std::atomic<bool> softShadows;
        int softRays;
        float sunAngleDeg;
        float softBlurPx;
        bool terrainNormals;
        bool terrainTerminator;
        float spriteShadowStart;
        float terrainNormalWindow;
        int shadowHistory;
        int shadowHistoryInterval;
        float shadowHistorySkip;
        bool logTiming;
        float exhaustGain;
        float exhaustRadius;
        std::atomic<bool> ao;
        int aoRays;
        float aoRange;
        float aoBlurPx;
        float aoStrength;
        std::atomic<bool> gi;
        int giRays;
        float giRange;
        float giBlurPx;
        float giStrength;
        float giEmissive;
        std::atomic<bool> reflections;
        float reflRoughnessDeg;
        float reflRange;
        float reflStrength;
        // A level sun lower than this (degrees) is lifted to it: a near-horizon sun throws long shadows and the terminator blackens gentle slopes.
        float sunMinElevationDeg;
    };

    // ROGUESQ_RT_LIGHTS=1 enables the pass; the rest are tuning and debug knobs.
    inline Config& mutableConfig() {
        static Config c = {
            envFloat("ROGUESQ_RT_LIGHTS", 0.0f) != 0.0f,
            (int)envFloat("ROGUESQ_RT_LIGHTS_DEBUG", 0.0f),
            envFloat("ROGUESQ_RT_LIGHTS_GAIN", 1.0f),
            envFloat("ROGUESQ_RT_LIGHTS_WRAP", 0.3f),
            envFloat("ROGUESQ_RT_LIGHTS_SPRITE_RADIUS", 3.0f),
            envFloat("ROGUESQ_RT_LIGHTS_SPRITE_RADIUS_MAX", 1200.0f),
            envFloat("ROGUESQ_RT_LIGHTS_LASER_RADIUS", 250.0f),
            envFloat("ROGUESQ_RT_LIGHTS_DEBUG_RANGE", 4000.0f),
            envFloat("ROGUESQ_LOG_RT_LIGHTS", 0.0f) != 0.0f,
            envFloat("ROGUESQ_RT_SHADOWS", 0.0f) != 0.0f,
            (int)envFloat("ROGUESQ_RT_SHADOWS_DEBUG", 0.0f),
            envFloat("ROGUESQ_RT_SHADOWS_STRENGTH", 0.45f),
            envFloat("ROGUESQ_RT_SHADOWS_TMIN", 2.0f),
            envFloat("ROGUESQ_RT_SHADOWS_TMAX", 30000.0f),
            envFloat("ROGUESQ_RT_SHADOWS_BIAS", 0.002f),
            envFloat("ROGUESQ_RT_SHADOWS_TERRAIN", 1.0f) != 0.0f,
            std::clamp((int)envFloat("ROGUESQ_RT_SHADOWS_TERRAIN_SUB", 2.0f), 1, 4),
            envFloat("ROGUESQ_RT_SHADOWS_TERRAIN_TMIN", 0.0f),
            envFloat("ROGUESQ_RT_LIGHTS_SHADOWS", 1.0f) != 0.0f,
            envFloat("ROGUESQ_RT_LIGHTS_SHADOW_END", 0.1f),
            envFloat("ROGUESQ_RT_MENU_HOLO_RADIUS", 20000.0f),
            envFloat("ROGUESQ_RT_MENU_HOLO_GAIN", 1.0f),
            envFloat("ROGUESQ_RT_MENU_HOLO_FALLOFF", 1.0f),
            envFloat("ROGUESQ_RT_MENU_RING_GAIN", 3.0f),
            envFloat("ROGUESQ_RT_MENU_IMAGE_GAIN", 2.0f),
            std::clamp(envFloat("ROGUESQ_RT_TEMPORAL", 0.2f), 0.0f, 1.0f),
            envFloat("ROGUESQ_RT_HANGAR_FLOOR_COS", 0.995f),
            envFloat("ROGUESQ_RT_LIGHTS_TORPEDO_RADIUS", 750.0f),
            envFloat("ROGUESQ_RT_FOG_SHAFTS", 0.0f) != 0.0f,
            std::clamp((int)envFloat("ROGUESQ_RT_FOG_SHAFTS_STEPS", 12.0f), 4, 32),
            envFloat("ROGUESQ_RT_FOG_SHAFTS_STRENGTH", 0.35f),
            (int)envFloat("ROGUESQ_RT_FOG_SHAFTS_DEBUG", 0.0f),
            envFloat("ROGUESQ_RT_SHADOWS_CUTOUT", 1.0f) != 0.0f,
            envFloat("ROGUESQ_RT_LIGHTS_PICKUP_GAIN", 3.0f),
            envFloat("ROGUESQ_RT_LIGHTS_PICKUP_RADIUS", 1500.0f),
            envFloat("ROGUESQ_RT_SHADOWS_SOFT", 1.0f) != 0.0f,
            std::clamp((int)envFloat("ROGUESQ_RT_SHADOWS_SOFT_RAYS", 4.0f), 1, 16),
            envFloat("ROGUESQ_RT_SHADOWS_SUN_ANGLE", 1.0f),
            std::clamp(envFloat("ROGUESQ_RT_SHADOWS_SOFT_BLUR", 8.0f), 0.0f, 32.0f),
            envFloat("ROGUESQ_RT_SHADOWS_TERRAIN_NORMALS", 1.0f) != 0.0f,
            envFloat("ROGUESQ_RT_SHADOWS_TERMINATOR", 1.0f) != 0.0f,
            envFloat("ROGUESQ_RT_LIGHTS_SPRITE_SHADOW_START", 1.0f),
            std::clamp(envFloat("ROGUESQ_RT_SHADOWS_TERRAIN_NORMAL_WINDOW", 0.03f), 0.0f, 0.5f),
            std::clamp((int)envFloat("ROGUESQ_RT_SHADOWS_HISTORY", 2.0f), 0, 4),
            std::max((int)envFloat("ROGUESQ_RT_SHADOWS_HISTORY_INTERVAL", 8.0f), 1),
            envFloat("ROGUESQ_RT_SHADOWS_HISTORY_SKIP", 0.0f),
            envFloat("ROGUESQ_LOG_RT_TIMING", 0.0f) != 0.0f,
            envFloat("ROGUESQ_RT_LIGHTS_EXHAUST_GAIN", 0.8f),
            envFloat("ROGUESQ_RT_LIGHTS_EXHAUST_RADIUS", 4000.0f),
            envFloat("ROGUESQ_RT_AO", 0.0f) != 0.0f,
            std::clamp((int)envFloat("ROGUESQ_RT_AO_RAYS", 2.0f), 1, 8),
            envFloat("ROGUESQ_RT_AO_RANGE", 0.04f),
            envFloat("ROGUESQ_RT_AO_BLUR", 4.0f),
            envFloat("ROGUESQ_RT_AO_STRENGTH", 0.5f),
            envFloat("ROGUESQ_RT_GI", 0.0f) != 0.0f,
            std::clamp((int)envFloat("ROGUESQ_RT_GI_RAYS", 1.0f), 1, 4),
            envFloat("ROGUESQ_RT_GI_RANGE", 0.25f),
            envFloat("ROGUESQ_RT_GI_BLUR", 8.0f),
            envFloat("ROGUESQ_RT_GI_STRENGTH", 1.5f),
            envFloat("ROGUESQ_RT_GI_EMISSIVE", 1.0f),
            envFloat("ROGUESQ_RT_REFLECTIONS", 0.0f) != 0.0f,
            envFloat("ROGUESQ_RT_REFL_ROUGHNESS", 20.0f),
            envFloat("ROGUESQ_RT_REFL_RANGE", 0.1f),
            envFloat("ROGUESQ_RT_REFL_STRENGTH", 3.0f),
            std::clamp(envFloat("ROGUESQ_RT_SUN_MIN_ELEVATION", 25.0f), 0.0f, 89.0f),
        };
        return c;
    }

    inline const Config& config() {
        return mutableConfig();
    }

    // Feature switches the settings menu can flip at runtime; a set environment variable pins its switch (the env var wins).
    enum class Feature { Lights, Shadows, SoftShadows, FogShafts, AmbientOcclusion, GlobalIllumination, Reflections };

    inline const char* featureEnv(Feature f) {
        switch (f) {
            case Feature::Lights: return "ROGUESQ_RT_LIGHTS";
            case Feature::Shadows: return "ROGUESQ_RT_SHADOWS";
            case Feature::SoftShadows: return "ROGUESQ_RT_SHADOWS_SOFT";
            case Feature::AmbientOcclusion: return "ROGUESQ_RT_AO";
            case Feature::GlobalIllumination: return "ROGUESQ_RT_GI";
            case Feature::Reflections: return "ROGUESQ_RT_REFLECTIONS";
            default: return "ROGUESQ_RT_FOG_SHAFTS";
        }
    }

    inline bool featurePinned(Feature f) {
        const char* e = std::getenv(featureEnv(f));
        return (e != nullptr) && (*e != '\0');
    }

    inline bool feature(Feature f) {
        const Config& c = config();
        switch (f) {
            case Feature::Lights: return c.enabled;
            case Feature::Shadows: return c.shadows;
            case Feature::SoftShadows: return c.softShadows;
            case Feature::AmbientOcclusion: return c.ao;
            case Feature::GlobalIllumination: return c.gi;
            case Feature::Reflections: return c.reflections;
            default: return c.fogShafts;
        }
    }

    // Returns false (no change) when the env var pins it.
    inline bool setFeature(Feature f, bool on) {
        if (featurePinned(f)) {
            return false;
        }
        Config& c = mutableConfig();
        switch (f) {
            case Feature::Lights: c.enabled = on; break;
            case Feature::Shadows: c.shadows = on; break;
            case Feature::SoftShadows: c.softShadows = on; break;
            case Feature::AmbientOcclusion: c.ao = on; break;
            case Feature::GlobalIllumination: c.gi = on; break;
            case Feature::Reflections: c.reflections = on; break;
            default: c.fogShafts = on; break;
        }
        return true;
    }

    // Debug views in one cycle (F9 / Shift+F9): off, the trace views (ROGUESQ_RT_SHADOWS_DEBUG 1-10), then the light views (ROGUESQ_RT_LIGHTS_DEBUG 1-3).
    struct DebugView {
        int shadows;
        int lights;
        const char* name;
    };

    inline const DebugView* debugViews(size_t& count) {
        static const DebugView kViews[] = {
            { 0, 0, "off" },
            { 1, 0, "shadow mask" },
            { 2, 0, "shadow mask + self-hits" },
            { 3, 0, "shadow mask + cutout casters" },
            { 4, 0, "shadow casters" },
            { 5, 0, "soft shadow" },
            { 6, 0, "ambient occlusion" },
            { 7, 0, "indirect light (GI + reflections)" },
            { 8, 0, "reflection strength" },
            { 9, 0, "reflection hit colour" },
            { 10, 0, "reflection hit instance" },
            { 0, 1, "view distance" },
            { 0, 2, "light reach" },
            { 0, 3, "light visibility" },
        };
        count = sizeof(kViews) / sizeof(kViews[0]);
        return kViews;
    }

    // Steps the live debug view by dir (+1 / -1) from the one in use; returns it.
    inline const DebugView& stepDebugView(int dir) {
        size_t n = 0;
        const DebugView* views = debugViews(n);
        Config& c = mutableConfig();
        size_t cur = 0;
        for (size_t i = 0; i < n; ++i) {
            if ((views[i].shadows == c.shadowsDebug) && (views[i].lights == c.debugMode)) {
                cur = i;
            }
        }
        const size_t next = (cur + n + size_t(dir < 0 ? n - 1 : 1)) % n;
        c.shadowsDebug = views[next].shadows;
        c.debugMode = views[next].lights;
        return views[next];
    }

    // A framebuffer pair's own main fragment always casts; other same-view fragments need 8+ draws (the trailing radar has ~4).
    inline bool casterFragment(bool isMain, bool sameView, uint32_t calls) {
        return isMain || (sameView && (calls >= 8));
    }

    // The game may split one view over several framebuffer pairs on the same target; only the last keeps its full-screen pass so pixels are not shaded twice.
    inline void lastPassPerTarget(const void* const* targets, bool* on, size_t n) {
        for (size_t i = 0; i < n; ++i) {
            if (!on[i]) {
                continue;
            }
            for (size_t j = i + 1; j < n; ++j) {
                if (on[j] && (targets[j] == targets[i])) {
                    on[i] = false;
                    break;
                }
            }
        }
    }

    inline bool parseVec3(const char* s, float out[3]) {
        if ((s == nullptr) || (*s == '\0')) {
            return false;
        }
        float v[3];
        if (std::sscanf(s, "%f,%f,%f", &v[0], &v[1], &v[2]) != 3) {
            return false;
        }
        out[0] = v[0];
        out[1] = v[1];
        out[2] = v[2];
        return true;
    }

    enum MenuScene : int { MenuSceneNone = 0, MenuSceneMissionSelect = 1, MenuSceneHangar = 2 };

    // Overlay mapped at 0x800A5130 (0 mission, 1 menu, 2 cinematic, -1 unknown), set by the host's loadOverlay hook.
    inline std::atomic<int>& activeOverlay() {
        static std::atomic<int> s_overlay{ -1 };
        return s_overlay;
    }

    // Menu 3D scenes with no game light: no mission scene root (0x80138D20), menu data set (0x800CE730), account menu (id byte 0x800CE734 == 1);
    // the menu overlay's screen (u16 0x800CFF50) is 1 for mission select, 0 for the hangar, 2 for name entry. After an aborted mission the menu overlay
    // runs with no menu data and a stale mission root, so with menuOverlay set and menuData 0 the screen alone decides.
    inline MenuScene menuScene(uint32_t missionRoot, uint32_t menuData, uint8_t menuId, uint16_t menuScreen, bool menuOverlay = false) {
        const bool afterAbort = menuOverlay && (menuData == 0);
        if (!afterAbort && ((missionRoot != 0) || ((menuData & 0xFF800000u) != 0x80000000u) || (menuData == 0x80000000u) || (menuId != 1))) {
            return MenuSceneNone;
        }
        return (menuScreen == 1) ? MenuSceneMissionSelect : ((menuScreen == 0) ? MenuSceneHangar : MenuSceneNone);
    }

    // Boot sequence (N64 logo, attribution screens): no level light, no mission loaded and no menu scene; lit by the authored key light.
    inline bool introScene(bool levelSun, uint32_t missionRoot, MenuScene menu) {
        return !levelSun && (missionRoot == 0) && (menu == MenuSceneNone);
    }

    // Hangar key light in camera space (stable while the camera orbits; the floor keeps the game's painted shadow); ROGUESQ_RT_HANGAR_SUN=x,y,z overrides.
    inline const float* hangarSunDir() {
        static const struct Dir {
            float v[3] = { 0.3f, 0.85f, 0.45f };
            Dir() {
                parseVec3(std::getenv("ROGUESQ_RT_HANGAR_SUN"), v);
            }
        } s_dir;
        return s_dir.v;
    }

    // Authored key light for the briefing room, in the level-light convention (sunCameraDir); ROGUESQ_RT_MENU_SUN=x,y,z overrides.
    inline const float* menuSunDir() {
        static const struct Dir {
            float v[3] = { 0.3f, 0.85f, 0.45f };
            Dir() {
                parseVec3(std::getenv("ROGUESQ_RT_MENU_SUN"), v);
            }
        } s_dir;
        return s_dir.v;
    }

    // Hologram light position in the briefing room's fixed camera space (the shared room origin); ROGUESQ_RT_MENU_HOLO=x,y,z overrides.
    inline const float* menuHoloPos() {
        static const struct Pos {
            float v[3] = { 0.0f, 22.0f, 11068.0f };
            Pos() {
                parseVec3(std::getenv("ROGUESQ_RT_MENU_HOLO"), v);
            }
        } s_pos;
        return s_pos.v;
    }

    // The GBI captures (terrain, no-cast lasers) and the world-vertex pass feed the scene for sun shadows, shadowed lights and fog shafts.
    inline bool sceneCaptureEnabled(const Config& c) {
        return c.shadows || (c.enabled && c.lightShadows) || c.fogShafts || c.ao || c.gi || c.reflections;
    }

    // Laser/torpedo mesh lookup: lights need the emitters, and every scene feature needs the bolts kept out of the casters.
    inline bool emitterLookupEnabled(const Config& c) {
        return c.enabled || sceneCaptureEnabled(c);
    }

    inline bool sceneWanted(bool shadowsWanted, bool lightsWanted, bool lightShadows, bool rayQuery, bool msaa) {
        return shadowsWanted || (lightsWanted && lightShadows && rayQuery && !msaa);
    }

    // Shadow ray toward a point light, mirrored by RS64LightsPS: ends shadowEnd*radius short of the light so a light inside the ground is not blocked by it.
    inline bool lightRaySpan(float dist, float radius, float shadowEnd, float tMin, float viewDist, float terrainTMinScale, float& tMax, float& terrainTMin, bool& traceTerrain, float shadowStart = 0.0f) {
        tMax = dist - std::max(shadowEnd * radius, shadowStart);
        if (tMax <= tMin) {
            return false;
        }
        terrainTMin = std::max(tMin, terrainTMinScale * viewDist);
        traceTerrain = terrainTMin < tMax;
        return true;
    }

    // RSPProcessCS vertex fog at a camera-space point; clip.w == 0 (the eye) takes the max(z, 0) == 0 limit. Mirrored by RS64FogShaftsPS.
    inline float fogFactor(const Mat4& viewProj, const float q[3], float mul, float offset) {
        const float v[4] = { q[0], q[1], q[2], 1.0f };
        float c[4];
        transform(v, viewProj, c);
        const float a = (c[3] > 1e-6f) ? ((std::max(c[2], 0.0f) / c[3]) * mul + offset) : offset;
        return std::clamp(a / 255.0f, 0.0f, 1.0f);
    }

    // March fractions along camera->receiver: step i ends at (i + 1 - jitter) / n, the last exactly at the receiver.
    inline void fogSteps(uint32_t n, float jitter, float* t) {
        for (uint32_t i = 0; i < n; ++i) {
            t[i] = (i + 1 == n) ? 1.0f : (float(i) + 1.0f - jitter) / float(n);
        }
    }

    // Sun-visibility sample along camera->receiver, pulled toward the camera by bias so the receiver never blocks its own last step. Mirrored by RS64FogShaftsPS.
    inline float fogVisT(float t, float bias) {
        return t * (1.0f - bias);
    }

    // Depth at or past the sky pin (0x7FBE/0x7FBF of 32767) or the clear value; the game never fogs the sky. Half a tick under 0x7FBE.
    inline float fogSkyDepth() {
        return (float(0x7FBE) - 0.5f) / 32767.0f;
    }

    // Interleaved gradient noise (Jimenez): fixed per pixel, so a still view does not shimmer.
    inline float interleavedNoise(float x, float y) {
        const float a = 0.06711056f * x + 0.00583715f * y;
        const float b = 52.9829189f * (a - std::floor(a));
        return b - std::floor(b);
    }

    // Alpha-weighted average colour of a tile in TMEM (4 KB, N64 byte order; odd-row word swaps only permute texels). fmt/siz are G_IM_FMT/G_IM_SIZ;
    // CI uses the RGBA16 TLUT in upper TMEM (8-byte entries). RGBA32 and YUV are not handled. coverage = mean alpha; false when nothing is visible.
    inline bool tmemAverage(const uint8_t* tmem, uint8_t fmt, uint8_t siz, uint32_t tmemWord, uint32_t lineWords, uint32_t width, uint32_t height, uint8_t palette, float out[3], float& coverage) {
        if ((siz > 2) || (fmt == 1) || (width == 0) || (height == 0)) {
            return false;
        }
        auto rgba16 = [](uint16_t v, float c[4]) {
            c[0] = float((v >> 11) & 0x1F) / 31.0f;
            c[1] = float((v >> 6) & 0x1F) / 31.0f;
            c[2] = float((v >> 1) & 0x1F) / 31.0f;
            c[3] = float(v & 1);
        };
        // At most 8x8 samples: it runs per draw call while recording.
        const uint32_t stepX = std::max<uint32_t>(1, width / 8);
        const uint32_t stepY = std::max<uint32_t>(1, height / 8);
        const uint32_t bits = 4u << siz;
        double sum[3] = {};
        double alpha = 0.0;
        uint32_t n = 0;
        for (uint32_t y = 0; y < height; y += stepY) {
            for (uint32_t x = 0; x < width; x += stepX) {
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
                float c[4] = {};
                if (fmt == 2) {
                    const uint32_t idx = (bits == 4) ? ((uint32_t(palette) << 4) | v) : (v & 0xFF);
                    const uint32_t e = (0x800 + idx * 8) & 0xFFF;
                    rgba16(uint16_t((uint32_t(tmem[e]) << 8) | tmem[(e + 1) & 0xFFF]), c);
                }
                else if (fmt == 0) {
                    if (bits != 16) {
                        return false;
                    }
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

    // Ray i of n toward a sun disk of angular radius atan(tanAngle): concentric golden-angle spiral rotated by rot (0-1); n <= 1 is the sun itself. Mirrored by RS64ShadowsPS.
    inline void sunDiskDir(const float sun[3], float tanAngle, uint32_t i, uint32_t n, float rot, float out[3]) {
        if (n <= 1) {
            out[0] = sun[0];
            out[1] = sun[1];
            out[2] = sun[2];
            return;
        }
        const float seed[3] = { (std::fabs(sun[0]) < 0.9f) ? 1.0f : 0.0f, (std::fabs(sun[0]) < 0.9f) ? 0.0f : 1.0f, 0.0f };
        float t[3] = { sun[1] * seed[2] - sun[2] * seed[1], sun[2] * seed[0] - sun[0] * seed[2], sun[0] * seed[1] - sun[1] * seed[0] };
        const float tl = std::sqrt(t[0] * t[0] + t[1] * t[1] + t[2] * t[2]);
        t[0] /= tl;
        t[1] /= tl;
        t[2] /= tl;
        const float b[3] = { sun[1] * t[2] - sun[2] * t[1], sun[2] * t[0] - sun[0] * t[2], sun[0] * t[1] - sun[1] * t[0] };
        const float r = std::sqrt((float(i) + 0.5f) / float(n)) * tanAngle;
        const float th = 2.39996323f * float(i) + 6.28318531f * rot;
        const float c = std::cos(th) * r;
        const float s = std::sin(th) * r;
        const float d[3] = { sun[0] + c * t[0] + s * b[0], sun[1] + c * t[1] + s * b[1], sun[2] + c * t[2] + s * b[2] };
        const float dl = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
        out[0] = d[0] / dl;
        out[1] = d[1] / dl;
        out[2] = d[2] / dl;
    }

    // Ray i of count over the cosine-weighted hemisphere about unit n (concentric golden-angle spiral rotated by rot). Mirrored by RS64CutoutAlpha.hlsli.
    inline void cosineHemisphereDir(const float n[3], uint32_t i, uint32_t count, float rot, float out[3]) {
        const float seed[3] = { (std::fabs(n[0]) < 0.9f) ? 1.0f : 0.0f, (std::fabs(n[0]) < 0.9f) ? 0.0f : 1.0f, 0.0f };
        float t[3] = { n[1] * seed[2] - n[2] * seed[1], n[2] * seed[0] - n[0] * seed[2], n[0] * seed[1] - n[1] * seed[0] };
        const float tl = std::sqrt(t[0] * t[0] + t[1] * t[1] + t[2] * t[2]);
        t[0] /= tl;
        t[1] /= tl;
        t[2] /= tl;
        const float b[3] = { n[1] * t[2] - n[2] * t[1], n[2] * t[0] - n[0] * t[2], n[0] * t[1] - n[1] * t[0] };
        const float r = std::sqrt((float(i) + 0.5f) / float(std::max(count, 1u)));
        const float th = 2.39996323f * float(i) + 6.28318531f * rot;
        const float c = std::cos(th) * r;
        const float s = std::sin(th) * r;
        const float h = std::sqrt(std::max(0.0f, 1.0f - r * r));
        for (int k = 0; k < 3; ++k) {
            out[k] = t[k] * c + b[k] * s + n[k] * h;
        }
        const float l = std::sqrt(out[0] * out[0] + out[1] * out[1] + out[2] * out[2]);
        out[0] /= l;
        out[1] /= l;
        out[2] /= l;
    }

    // AO contribution of a hit at t within range: 1 at contact, 0 at the range edge.
    inline float aoWeight(float t, float range) {
        return std::clamp(1.0f - t / range, 0.0f, 1.0f);
    }

    // Schlick Fresnel.
    inline float schlick(float cosTheta, float f0) {
        const float m = 1.0f - std::clamp(cosTheta, 0.0f, 1.0f);
        return f0 + (1.0f - f0) * m * m * m * m * m;
    }

    // Mirror of view direction v about unit n, offset within a cone of tan tanCone (sunDiskDir ray 1 of 2, rotated by rot).
    inline void roughReflectDir(const float v[3], const float n[3], float tanCone, float rot, float out[3]) {
        const float d = v[0] * n[0] + v[1] * n[1] + v[2] * n[2];
        float m[3] = { v[0] - 2.0f * d * n[0], v[1] - 2.0f * d * n[1], v[2] - 2.0f * d * n[2] };
        const float l = std::sqrt(m[0] * m[0] + m[1] * m[1] + m[2] * m[2]);
        m[0] /= l;
        m[1] /= l;
        m[2] /= l;
        sunDiskDir(m, tanCone, 1, 2, rot, out);
    }

    // Hit colour per caster draw (first BLAS triangle, rgba8 linear colour; a = 255 marks an emissive draw). Looked up by RS64CutoutAlpha.hlsli.
    struct HitColorEntry {
        uint32_t firstTriangle;
        uint32_t rgba;
    };

    inline uint32_t packHitColor(float r, float g, float b, bool emissive) {
        auto q = [](float c) { return uint32_t(std::lround(std::clamp(c, 0.0f, 1.0f) * 255.0f)); };
        return (uint32_t(emissive ? 255u : 254u) << 24) | (q(b) << 16) | (q(g) << 8) | q(r);
    }

    // Floor draws (tag 253) keep their colour; the deck's reflection ray skips them so the deck does not reflect itself.
    inline uint32_t markFloor(uint32_t rgba) {
        return (rgba & 0x00FFFFFFu) | (253u << 24);
    }

    // A draw whose area-weighted normal sum lies along the floor's up (either winding) is floor.
    inline bool floorFacing(const float normalSum[3], const float up[3], float cosMin) {
        const float len = std::sqrt(normalSum[0] * normalSum[0] + normalSum[1] * normalSum[1] + normalSum[2] * normalSum[2]);
        const float upLen = std::sqrt(up[0] * up[0] + up[1] * up[1] + up[2] * up[2]);
        if (!(len > 0.0f) || !(upLen > 0.0f)) {
            return false;
        }
        return std::fabs(normalSum[0] * up[0] + normalSum[1] * up[1] + normalSum[2] * up[2]) / (len * upLen) > cosMin;
    }

    inline size_t hitColorFor(uint32_t prim, const HitColorEntry* table, size_t n) {
        if ((n == 0) || (prim < table[0].firstTriangle)) {
            return SIZE_MAX;
        }
        size_t lo = 0;
        size_t hi = n - 1;
        while (lo < hi) {
            const size_t mid = (lo + hi + 1) / 2;
            if (table[mid].firstTriangle <= prim) {
                lo = mid;
            }
            else {
                hi = mid - 1;
            }
        }
        return lo;
    }

    // Average colour of a draw: texture average (nullptr = untextured: vertex average) x prim (ignored when near black, e.g. a combiner that does not use it).
    // A textured draw's prelit vertex colour is baked shading (~0.1 on models), not albedo.
    inline void drawAverageColor(const float vtx[3], const float* tex, const float prim[3], float out[3]) {
        const bool usePrim = std::max(prim[0], std::max(prim[1], prim[2])) >= 0.05f;
        for (int k = 0; k < 3; ++k) {
            float c = ((tex != nullptr) ? tex[k] : vtx[k]) * (usePrim ? prim[k] : 1.0f);
            out[k] = std::isfinite(c) ? std::clamp(c, 0.0f, 1.0f) : 0.0f;
        }
    }

    // Shadow-pass composite, blend dst * src + dst * srcA = dst * dark * (1 + indirect): a = darkening from sun shadow and AO, rgb = dark x indirect clamped to 1
    // (the colour target may be UNORM). Mirrored by RS64ShadowBlurPS.
    inline void composeFactor(float shadow, float ao, const float indirect[3], float shadowStrength, float aoStrength, float out[4]) {
        const float dark = (1.0f - shadowStrength * shadow) * (1.0f - aoStrength * ao);
        for (int k = 0; k < 3; ++k) {
            out[k] = dark * std::clamp(indirect[k], 0.0f, 1.0f);
        }
        out[3] = dark;
    }

    // The shadow trace pass runs for any of its features.
    inline bool tracePassWanted(bool shadows, bool ao, bool gi, bool reflections) {
        return shadows || ao || gi || reflections;
    }

    // Reflective receivers: the hangar deck (the floor test receiverSkipped uses); none outside the hangar.
    inline bool reflectionReceiver(const float skip[4], const float n[3]) {
        return receiverSkipped(n, skip);
    }

    // Screen penumbra in pixels for a blocker meanHitT away from a receiver at view distance dist; pixelsPerUnit is at unit distance.
    inline float penumbraPixels(float meanHitT, float tanAngle, float dist, float pixelsPerUnit, float maxPx) {
        if (!(meanHitT > 0.0f) || !(dist > 0.0f)) {
            return 0.0f;
        }
        return std::clamp(meanHitT * tanAngle * pixelsPerUnit / dist, 0.0f, maxPx);
    }

    // Pixels per world unit at unit distance: length of the clip-y column (independent of camera orientation) times half the viewport height.
    inline float pixelsPerUnit(const Mat4& viewProj, float viewportHeight) {
        const auto& m = viewProj.m;
        return std::sqrt(m[0][1] * m[0][1] + m[1][1] * m[1][1] + m[2][1] * m[2][1]) * viewportHeight * 0.5f;
    }

    // View-distance change per pixel along the blur direction: the smaller one-sided step (so a depth edge on one side does not tilt the plane); sky sides (>= 1e29) are ignored. Mirrored by RS64ShadowBlurPS.
    inline float depthSlope(float dc, float dMinus, float dPlus) {
        const bool m = dMinus < 1.0e29f;
        const bool p = dPlus < 1.0e29f;
        const float back = dc - dMinus;
        const float fwd = dPlus - dc;
        if (m && p) {
            return (std::fabs(back) < std::fabs(fwd)) ? back : fwd;
        }
        return m ? back : (p ? fwd : 0.0f);
    }

    // 3x4 affine (TLAS layout, out = M * [v, 1]) inverse; false when singular.
    inline bool affineInverse(const float m[3][4], float out[3][4]) {
        const float a = m[0][0], b = m[0][1], c = m[0][2];
        const float d = m[1][0], e = m[1][1], f = m[1][2];
        const float g = m[2][0], h = m[2][1], k = m[2][2];
        const float A = e * k - f * h, B = -(d * k - f * g), C = d * h - e * g;
        const float det = a * A + b * B + c * C;
        if (!(std::fabs(det) > 1e-12f)) {
            return false;
        }
        const float r[3][3] = {
            { A / det, -(b * k - c * h) / det, (b * f - c * e) / det },
            { B / det, (a * k - c * g) / det, -(a * f - c * d) / det },
            { C / det, -(a * h - b * g) / det, (a * e - b * d) / det },
        };
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                out[i][j] = r[i][j];
            }
            out[i][3] = -(r[i][0] * m[0][3] + r[i][1] * m[1][3] + r[i][2] * m[2][3]);
        }
        return true;
    }

    // out = a o b (apply b, then a).
    inline void affineCompose(const float a[3][4], const float b[3][4], float out[3][4]) {
        float r[3][4];
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 4; ++j) {
                r[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j] + ((j == 3) ? a[i][3] : 0.0f);
            }
        }
        std::memcpy(out, r, sizeof(r));
    }

    // Ray parameter where o + t d leaves the view frustum's side planes (|x|, |y| <= w, w > 0) under row-vector screenFromView; 0 if o is outside, huge if it never leaves. Mirrored by RS64ShadowsPS.
    inline float frustumExitT(const Mat4& sfv, const float o[3], const float d[3]) {
        float c0[4], cd[4];
        for (int j = 0; j < 4; ++j) {
            c0[j] = o[0] * sfv.m[0][j] + o[1] * sfv.m[1][j] + o[2] * sfv.m[2][j] + sfv.m[3][j];
            cd[j] = d[0] * sfv.m[0][j] + d[1] * sfv.m[1][j] + d[2] * sfv.m[2][j];
        }
        const float a[5] = { c0[3] - c0[0], c0[3] + c0[0], c0[3] - c0[1], c0[3] + c0[1], c0[3] };
        const float b[5] = { cd[3] - cd[0], cd[3] + cd[0], cd[3] - cd[1], cd[3] + cd[1], cd[3] };
        float t = 1.0e30f;
        for (int i = 0; i < 5; ++i) {
            if (a[i] < 0.0f) {
                return 0.0f;
            }
            if (b[i] < 0.0f) {
                t = std::min(t, -a[i] / b[i]);
            }
        }
        return t;
    }

    // Unit-direction ray vs sphere: entry/exit parameters (t0 may be negative); false on a miss.
    inline bool raySphereSpan(const float o[3], const float d[3], const float c[3], float r, float& t0, float& t1) {
        const float oc[3] = { o[0] - c[0], o[1] - c[1], o[2] - c[2] };
        const float b = oc[0] * d[0] + oc[1] * d[1] + oc[2] * d[2];
        const float q = oc[0] * oc[0] + oc[1] * oc[1] + oc[2] * oc[2] - r * r;
        const float disc = b * b - q;
        if (disc < 0.0f) {
            return false;
        }
        const float s = std::sqrt(disc);
        t0 = -b - s;
        t1 = -b + s;
        return true;
    }

    // Blur taps per side: one per two pixels of radius, 1-16. Mirrored by RS64ShadowBlurPS.
    inline int blurTapCount(float radiusPx) {
        return std::clamp((int)std::ceil(radiusPx * 0.5f), 1, 16);
    }

    // Blur tap weight: 1 at the centre's view distance, 0 beyond tolerance * dCenter, so shadows never bleed across depth edges or onto sky. Mirrored by RS64ShadowBlurPS.
    inline float shadowDepthWeight(float dCenter, float dSample, float tolerance) {
        if (!std::isfinite(dSample) || !(dCenter > 0.0f)) {
            return 0.0f;
        }
        return std::clamp(1.0f - std::fabs(dCenter - dSample) / (tolerance * dCenter), 0.0f, 1.0f);
    }

    // Fog the sun reaches: sum of vis_i * max(fog_i - fog_{i-1}, 0), fog_{-1} = 0 (the shader skips steps where fog drops). All visible gives the game's fog at the receiver.
    inline float litInScatter(const float* fog, const float* vis, uint32_t n) {
        float lit = 0.0f;
        float prev = 0.0f;
        for (uint32_t i = 0; i < n; ++i) {
            lit += vis[i] * std::max(fog[i] - prev, 0.0f);
            prev = fog[i];
        }
        return lit;
    }

    // Black fog or zero strength adds nothing, so the pass is skipped instead of tracing for zero.
    inline bool fogShaftsVisible(const float color[3], float strength) {
        return std::max(color[0], std::max(color[1], color[2])) * strength >= (1.0f / 255.0f);
    }

    struct FogParams {
        bool valid = false;
        uint32_t tris = 0;
        float mul = 0.0f;
        float offset = 0.0f;
        float color[3] = {};
    };

    // The main view's largest fogged draw sets the fog; ties keep the first.
    inline void considerFogCall(FogParams& best, uint32_t tris, float mul, float offset, const float color[3]) {
        if ((tris == 0) || (best.valid && (tris <= best.tris))) {
            return;
        }
        best.valid = true;
        best.tris = tris;
        best.mul = mul;
        best.offset = offset;
        best.color[0] = color[0];
        best.color[1] = color[1];
        best.color[2] = color[2];
    }

    struct FogCache {
        uint64_t workloadId = UINT64_MAX;
        FogParams params;
    };

    // One fog per workload across its framebuffer pairs; a pair without a fogged main draw reuses the pick.
    inline const FogParams& workloadFog(FogCache& c, uint64_t workloadId, const FogParams& pairPick) {
        if (c.workloadId != workloadId) {
            c.workloadId = workloadId;
            c.params = FogParams{};
        }
        if (pairPick.valid && (!c.params.valid || (pairPick.tris > c.params.tris))) {
            c.params = pairPick;
        }
        return c.params;
    }

    // Insertion of the k smallest distances, ascending; mirrored by RS64LightsPS.
    inline uint32_t nearestLights(const float* dist, uint32_t count, uint32_t k, uint32_t* out) {
        uint32_t n = 0;
        for (uint32_t i = 0; i < count; ++i) {
            uint32_t pos = n;
            while ((pos > 0) && (dist[out[pos - 1]] > dist[i])) {
                --pos;
            }
            if (pos >= k) {
                continue;
            }
            const uint32_t last = std::min(n, k - 1);
            for (uint32_t j = last; j > pos; --j) {
                out[j] = out[j - 1];
            }
            out[pos] = i;
            n = std::min(n + 1, k);
        }
        return n;
    }
}
