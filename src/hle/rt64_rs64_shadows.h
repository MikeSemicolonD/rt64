#pragma once

// RS64 ray-traced passes: the sun, which draws cast shadows, and the per-pixel trace (soft shadows, AO, bounce light, reflections).

#include "rt64_rs64_math.h"

namespace rs64lights {
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

    // A level preset's haze: sunlit air near the camera that shows shafts on its own, whatever the game's fog. amount = the share of light a
    // fully sunlit ray of length range scatters; color < 0 = the sun's colour.
    struct Haze {
        float amount = 0.0f;
        float range = 0.0f;
        float color[3] = { -1.0f, -1.0f, -1.0f };
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

    // A triangle wholly outside the view that only casts (never drawn): its vertices, the other mode it was submitted under, its transform and view.
    struct CastOnlyTri {
        uint32_t idx[3];
        uint32_t otherModeL;
        uint32_t otherModeH;
        uint32_t transform;
        uint32_t view;
    };

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

    // Reference for the threshold RS64Trace.hlsli computes itself: raster discards alpha < blend alpha for G_AC_THRESHOLD; dither and coverage x alpha use the midpoint.
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

    // Per-axis UV gradient for the cutout alpha fetch: TextureSampler's mip = 0.5*log2(|g*tcScale|^2) - 0.25 lands on 0 (zero gradients give NaN). Mirrored by RS64Trace.hlsli.
    inline float cutoutMipGradient(float tcScale) {
        return 1.18920712f / tcScale;
    }

    // Cutout flags: a translucent draw (a World Devastator's foot beam) is alpha-tested for water reflection rays only, and never casts.
    constexpr uint32_t CutoutTranslucent = 1u;
    // The sky dome: only water reflection rays looking for the sky off screen meet it (RS64Trace.hlsli rs64TracedSky).
    constexpr uint32_t CutoutSky = 2u;

    // Average vertex colour of up to 16 of a draw's triangles' first vertices: rgb8 in bits 8-31 of the cutout flags, alpha in bits 2-7 (6 bits).
    // The shade input of a translucent draw's colour combiner in reflections.
    inline uint32_t packCutoutShade(const std::vector<uint32_t> &faceIndices, const std::vector<uint8_t> &normColBytes, uint32_t start, uint32_t triangleCount) {
        uint32_t sum[4] = {};
        uint32_t n = 0;
        const uint32_t step = std::max<uint32_t>(1, triangleCount / 16) * 3;
        for (uint32_t k = 0; (k < triangleCount * 3) && (size_t(start) + k < faceIndices.size()); k += step) {
            const size_t v = faceIndices[start + k];
            if (v * 4 + 3 < normColBytes.size()) {
                sum[0] += normColBytes[v * 4 + 0];
                sum[1] += normColBytes[v * 4 + 1];
                sum[2] += normColBytes[v * 4 + 2];
                sum[3] += normColBytes[v * 4 + 3];
                ++n;
            }
        }

        if (n == 0) {
            return 0xFFFFFFFCu;
        }

        return ((sum[0] / n) << 8) | ((sum[1] / n) << 16) | ((sum[2] / n) << 24) | (((sum[3] / n) >> 2) << 2);
    }

    struct CutoutRange {
        IndexRange range;
        uint32_t instanceIndex = 0;
        uint32_t tileIndex = 0;
        uint32_t flags = 0;
    };

    struct CutoutEntry {
        uint32_t firstTriangle;
        uint32_t instanceIndex;
        uint32_t tileIndex;
        uint32_t flags;
    };

    // Cutout BLAS indices plus one table entry per kept draw (first triangle in the BLAS, call index, RDP tile); ranges past the face buffer are dropped.
    inline size_t buildCutoutScene(const uint32_t* faceIndices, size_t faceIndexCount, const CutoutRange* ranges, size_t n, std::vector<uint32_t>& indices, std::vector<CutoutEntry>& table) {
        size_t added = 0;
        for (size_t r = 0; r < n; ++r) {
            const size_t count = size_t(ranges[r].range.triangleCount) * 3;
            if ((count == 0) || (size_t(ranges[r].range.start) + count > faceIndexCount)) {
                continue;
            }
            table.push_back({ uint32_t(indices.size() / 3), ranges[r].instanceIndex, ranges[r].tileIndex, ranges[r].flags });
            indices.insert(indices.end(), faceIndices + ranges[r].range.start, faceIndices + ranges[r].range.start + count);
            added += count;
        }
        return added;
    }

    // Last entry whose first triangle is <= prim; mirrored by RS64Trace.hlsli.
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

    // Ray i of count over the cosine-weighted hemisphere about unit n (concentric golden-angle spiral rotated by rot). Mirrored by RS64Trace.hlsli.
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

    // Hit colour per caster draw (first BLAS triangle, rgba8 linear colour; a = 255 marks an emissive draw). Looked up by RS64Trace.hlsli.
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

    // One sky dome triangle for skyBands: its draw's average colour and the sine of its centroid's elevation above the horizon.
    struct SkySample {
        float rgb[3];
        float elevation;
    };

    // The sky a missed water reflection sees, from the dome's triangles: the average near the horizon (below ~15 degrees) and overhead (above ~45 degrees).
    // A band with no triangles takes the whole dome's average. False when there is no dome.
    inline bool skyBands(const SkySample* samples, size_t count, float horizon[3], float zenith[3]) {
        constexpr float HorizonMax = 0.26f;
        constexpr float ZenithMin = 0.7f;
        double low[3] = {}, high[3] = {}, total[3] = {};
        uint32_t nLow = 0, nHigh = 0;
        for (size_t i = 0; i < count; ++i) {
            const SkySample& s = samples[i];
            for (int k = 0; k < 3; ++k) {
                total[k] += s.rgb[k];
                low[k] += (s.elevation < HorizonMax) ? s.rgb[k] : 0.0;
                high[k] += (s.elevation > ZenithMin) ? s.rgb[k] : 0.0;
            }
            nLow += (s.elevation < HorizonMax) ? 1 : 0;
            nHigh += (s.elevation > ZenithMin) ? 1 : 0;
        }

        if (count == 0) {
            return false;
        }

        for (int k = 0; k < 3; ++k) {
            const float avg = float(total[k] / double(count));
            horizon[k] = (nLow > 0) ? float(low[k] / double(nLow)) : avg;
            zenith[k] = (nHigh > 0) ? float(high[k] / double(nHigh)) : avg;
        }

        return true;
    }

    // Shadow-pass composite, blend dst * src + dst * srcA = dst * dark * (1 + indirect): a = darkening from sun shadow and AO, rgb = dark x indirect clamped to 1
    // (the colour target is UNORM). On water (waterF > 0) rgb is 0 and a keeps dark x (1 - F); the additive water draw adds indirect (R x F). Mirrored by RS64ShadowBlurPS.
    inline void composeFactor(float shadow, float ao, const float indirect[3], float waterF, float shadowStrength, float aoStrength, float out[4]) {
        const float dark = (1.0f - shadowStrength * shadow) * (1.0f - aoStrength * ao);
        for (int k = 0; k < 3; ++k) {
            out[k] = (waterF > 0.0f) ? 0.0f : dark * std::clamp(indirect[k], 0.0f, 1.0f);
        }
        out[3] = dark * ((waterF > 0.0f) ? std::clamp(1.0f - waterF, 0.0f, 1.0f) : 1.0f);
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

    // The unit normal of the source's y = const plane under a 3x4 affine (TLAS layout): cofactor * (0, 1, 0) = column 2 x column 0, either sign.
    inline bool affinePlaneNormalY(const float m[3][4], float out[3]) {
        const float c0[3] = { m[0][0], m[1][0], m[2][0] };
        const float c2[3] = { m[0][2], m[1][2], m[2][2] };
        const float n[3] = { c2[1] * c0[2] - c2[2] * c0[1], c2[2] * c0[0] - c2[0] * c0[2], c2[0] * c0[1] - c2[1] * c0[0] };
        const float len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        if (len <= 0.0f) {
            return false;
        }

        for (int i = 0; i < 3; ++i) {
            out[i] = n[i] / len;
        }

        return true;
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
}
