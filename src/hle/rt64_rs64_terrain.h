#pragma once

// The game's HMP terrain map, rebuilt as one mesh for the whole-map terrain pass and the traced terrain casters.

#include "rt64_rs64_math.h"

namespace rs64lights {
    // HMP terrain (descriptor 0x80136DC0): W*H cells, each a 512-unit tile of 5x5 s8 heights (y = h << 4, base 0).
    constexpr int32_t TerrainTileSpan = 512;
    // The drawn water surface's lift over the water level (record units; heights grow downward), as kWaterLift in RS64Terrain.hlsli; the water fades
    // out over it as a shore rises. Water is where the ground is at or above the level (selectGridCellTileVariant 0x80046030).
    constexpr float TerrainWaterLift = 16.0f;
    // How far under the drawn water the traced water sits: a pixel's terrain lookup (a few % of its distance) must reach it from a camera skimming the sea.
    constexpr float TerrainWaterTraceGap = 1.0f;

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
        // The clock for the texture animations, in 30 Hz ticks of game time.
        uint64_t task = 0;
        // Land fades out toward the fog end (RS64TerrainCB::fogClip).
        bool fogClip = false;
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
    // Cell words: low 16 bits texture layer, bits 16-17 water kind (0 ground, 1 water, 2 shore). Sets bits 18+ to the water draw's layer.
    // Shore tiles paint sand and water into one texture, as do a few cells flagged open water, and showed as square patches, so those take a nearby texture open water commonly uses, or the map's main one.
    inline void terrainWaterLayers(std::vector<uint32_t>& cells, const std::vector<uint8_t>& kinds, int32_t width, int32_t height) {
        constexpr uint32_t RareShare = 8;
        std::unordered_map<uint32_t, uint32_t> openCells;
        uint32_t mostOpen = 0, mainOpen = 0;
        for (size_t c = 0; (c < cells.size()) && (c < kinds.size()); ++c) {
            if (kinds[c] == 1) {
                const uint32_t layer = cells[c] & 0xFFFFu;
                const uint32_t n = ++openCells[layer];
                if (n > mostOpen) {
                    mostOpen = n;
                    mainOpen = layer;
                }
            }
        }

        const auto commonOpen = [&](uint32_t layer) {
            const auto it = openCells.find(layer);
            return (it != openCells.end()) && ((it->second * RareShare) >= mostOpen);
        };

        const auto nearbyOpen = [&](int32_t col, int32_t row, uint32_t own) {
            for (int32_t r = 1; r <= 3; ++r) {
                for (int32_t dz = -r; dz <= r; ++dz) {
                    for (int32_t dx = -r; dx <= r; ++dx) {
                        const int32_t x = col + dx, z = row + dz;
                        if ((std::max(std::abs(dx), std::abs(dz)) != r) || (x < 0) || (z < 0) || (x >= width) || (z >= height)) {
                            continue;
                        }

                        const size_t n = size_t(z) * width + x;
                        if ((n < kinds.size()) && (kinds[n] == 1) && commonOpen(cells[n] & 0xFFFFu)) {
                            return cells[n] & 0xFFFFu;
                        }
                    }
                }
            }

            return (mostOpen > 0) ? mainOpen : own;
        };

        for (size_t c = 0; c < cells.size(); ++c) {
            const uint32_t own = cells[c] & 0xFFFFu;
            const uint8_t kind = (c < kinds.size()) ? kinds[c] : uint8_t(0);
            const bool borrow = (kind == 2) || ((kind == 1) && !commonOpen(own));
            cells[c] |= (borrow ? nearbyOpen(int32_t(c % width), int32_t(c / width), own) : own) << 18;
        }
    }

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

        // Water and shore cells also get a flat surface just under the drawn water, so traced rays meet water instead of the seabed without the
        // water surface shadowing itself.
        const float waterH = float(m.waterLevel * 16) - TerrainWaterLift;
        const int32_t latticeW = int32_t(m.width) * 4, latticeH = int32_t(m.height) * 4;

        // Ground height (map units, before heightScale) at lattice point (fx, fz), clamped to the map; the water level where no cell is loaded.
        const auto latticeHeight = [&](int32_t fx, int32_t fz) {
            fx = std::clamp(fx, 0, latticeW);
            fz = std::clamp(fz, 0, latticeH);
            const int32_t col = std::min(fx / 4, int32_t(m.width) - 1), row = std::min(fz / 4, int32_t(m.height) - 1);
            const size_t c = size_t(row) * m.width + size_t(col);
            if ((c >= m.present.size()) || !m.present[c] || ((c + 1) * 25 > m.heights.size())) {
                return waterH;
            }

            return float(int32_t(m.heights[c * 25 + size_t((fz - row * 4) * 5 + (fx - col * 4))]) * 16);
        };

        forEachTerrainWaterCell(m, [&](int32_t col, int32_t row, size_t c) {
            const uint32_t a = uint32_t(pos.size() / 3);
            // The outer ring reaches TerrainWaterFarCells further, as the drawn water does.
            const float reach = float(TerrainWaterFarCells * TerrainTileSpan);
            const int32_t lastCol = int32_t(m.width) + TerrainWaterMargin - 1, lastRow = int32_t(m.height) + TerrainWaterMargin - 1;
            const float x0 = float(col * TerrainTileSpan) - ((col == -TerrainWaterMargin) ? reach : 0.0f), x1 = float((col + 1) * TerrainTileSpan) + ((col == lastCol) ? reach : 0.0f);
            const float z0 = float(row * TerrainTileSpan) - ((row == -TerrainWaterMargin) ? reach : 0.0f), z1 = float((row + 1) * TerrainTileSpan) + ((row == lastRow) ? reach : 0.0f);

            // Past the edge the terrain pass carries the edge cell on outward at its edge's height (RS64TerrainVS), the water's or the ground's above it;
            // the traced surface follows it, or the camera ray's depth window missed it at most angles and nothing out there reflected.
            const bool outside = (col < 0) || (row < 0) || (col >= int32_t(m.width)) || (row >= int32_t(m.height));
            const bool water = (c < m.waterKind.size()) && (m.waterKind[c] != 0);
            const auto cornerY = [&](int32_t kx, int32_t kz) {
                const float h = outside ? latticeHeight((col + kx) * 4, (row + kz) * 4) : waterH;
                return ((outside && !water) ? h : std::min(waterH, h)) + TerrainWaterTraceGap;
            };

            const float y00 = cornerY(0, 0) * m.heightScale, y10 = cornerY(1, 0) * m.heightScale, y01 = cornerY(0, 1) * m.heightScale, y11 = cornerY(1, 1) * m.heightScale;
            const float corners[12] = { x0, y00, z0, x1, y10, z0, x0, y01, z1, x1, y11, z1 };
            pos.insert(pos.end(), corners, corners + 12);
            const uint32_t tri[6] = { a, a + 3, a + 2, a, a + 1, a + 3 };
            idx.insert(idx.end(), tri, tri + 6);
        });
    }

    // Hit colours for buildTerrainMesh's triangles, in its cell order: header (triangles per ground cell, first water triangle or 0, ground cells, the
    // drawn water surface's map-space y as float bits), then one rgba8 per ground cell and per water quad: the cell's texture average (texAvg, rgb8 per
    // cell) times its average vertex colour.
    // buildTerrainMesh's first water triangle (after every ground cell's triangles), or 0 without water.
    inline uint32_t terrainWaterPrim(const TerrainMap& m, int sub) {
        uint32_t ground = 0;
        for (size_t c = 0; c < size_t(m.width) * m.height; ++c) {
            ground += ((c < m.present.size()) && m.present[c] && ((c + 1) * 25 <= m.heights.size())) ? 1u : 0u;
        }

        bool water = false;
        forEachTerrainWaterCell(m, [&](int32_t, int32_t, size_t) { water = true; });
        return water ? ground * uint32_t(2 * (4 * sub) * (4 * sub)) : 0u;
    }

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

        forEachTerrainWaterCell(m, [&](int32_t, int32_t, size_t c) {
            out.push_back(cellColor(c));
        });

        out[0] = per;
        out[1] = terrainWaterPrim(m, sub);
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

    // Area-weighted vertex normals (+y up), welded across tile seams by integer (x, y, z) so the shadow pass sees a smooth heightfield instead of facets.
    // The height is in the key: the water surface shares the ground's grid, and welding a slope to the flat water under it tilted the slope's normal up
    // (a lit diamond in a self-shadowed hillside).
    inline void buildTerrainNormals(const std::vector<float>& pos, const std::vector<uint32_t>& idx, std::vector<float>& out) {
        const size_t verts = pos.size() / 3;
        std::unordered_map<uint64_t, uint32_t> weld;
        weld.reserve(verts);
        std::vector<uint32_t> slot(verts);
        for (size_t v = 0; v < verts; ++v) {
            const int32_t c[3] = { int32_t(pos[v * 3]), int32_t(pos[v * 3 + 1]), int32_t(pos[v * 3 + 2]) };
            const uint64_t key = fnv1a64(reinterpret_cast<const uint8_t*>(c), sizeof(c));
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
}
