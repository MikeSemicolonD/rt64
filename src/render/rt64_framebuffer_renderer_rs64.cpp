//
// RT64
//

// The RS64 passes of the framebuffer renderer: ray-traced lights, shadows and fog shafts, the water mirror and the whole-map terrain.

#include "rt64_framebuffer_renderer.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_set>

#include "../include/rt64_extended_gbi.h"

#include "common/rt64_elapsed_timer.h"
#include "common/rt64_math.h"
#include "xxHash/xxh3.h"
#include "common/rt64_tmem_hasher.h"
#include "hle/rt64_color_converter.h"
#include "gbi/rt64_f3d.h"
#include "render/rt64_raster_shader.h"
#include "shared/rt64_framebuffer_params.h"
#include "shared/rt64_raster_params.h"
#include "shared/rt64_rs64_lights_cb.h"
#include "shared/rt64_rs64_shadows_cb.h"
#include "shared/rt64_rs64_shadow_blur_cb.h"
#include "shared/rt64_rs64_fog_shafts_cb.h"
#include "shared/rt64_rs64_cutout.h"

#include "common/rt64_rs64_env.h"
#include "hle/rt64_rs64_lights.h"
#include "hle/rt64_rs64_host.h"
#include "hle/rt64_rs64_vr.h"

#include "rt64_descriptor_sets.h"
#include "rt64_render_worker.h"

static_assert(sizeof(interop::RS64CutoutDraw) == sizeof(rs64lights::CutoutEntry), "cutout table layout");

// RS64 whole-map terrain pass: cells per chunk edge (one instance, one LOD and one frustum test per chunk).
static constexpr uint32_t RS64TerrainChunkCells = 8;

namespace RT64 {
    static rs64lights::Mat4 rs64ToMat4(const interop::float4x4 &m) {
        static_assert(sizeof(interop::float4x4) == sizeof(rs64lights::Mat4), "float4x4 layout");
        rs64lights::Mat4 r;
        memcpy(&r, &m, sizeof(r));
        return r;
    }

    // Whether the traced passes have terrain normals this frame. The constants are filled before the scene build makes them for a new map, so without the
    // pending build a map's first frame traced no terrain or water (the opening Calamari cutscene lost its reflections for a frame).
    template <typename Scene>
    static bool rs64TerrainNormalsThisFrame(const Scene &s) {
        return s.terrainNormalGpu || (s.terrainActive && s.terrainMap && (s.terrainBuiltVersion != s.terrainMap->version));
    }

    // Two projections show the same 3D view when their raw view and projection matrices and their N64 viewport match (the radar shares the matrices but not the viewport).
    // ignoreProj: camera-space positions only need the same camera; the player craft is drawn under its own projection (nearer clip) in the same view.
    static bool rs64SameView(const DrawData &drawData, uint32_t a, uint32_t b, bool ignoreProj = false) {
        const size_t n = std::min({ drawData.viewTransforms.size(), drawData.projTransforms.size(), drawData.rspViewports.size() });
        if ((a >= n) || (b >= n)) {
            return false;
        }
        if (a == b) {
            return true;
        }
        const interop::RSPViewport &va = drawData.rspViewports[a];
        const interop::RSPViewport &vb = drawData.rspViewports[b];
        const bool sameViewport = (va.scale.x == vb.scale.x) && (va.scale.y == vb.scale.y) && (va.scale.z == vb.scale.z) && (va.translate.x == vb.translate.x) && (va.translate.y == vb.translate.y) && (va.translate.z == vb.translate.z);
        return sameViewport && rs64lights::sameMatrix(rs64ToMat4(drawData.viewTransforms[a]), rs64ToMat4(drawData.viewTransforms[b]), 1e-4f)
            && (ignoreProj || rs64lights::sameMatrix(rs64ToMat4(drawData.projTransforms[a]), rs64ToMat4(drawData.projTransforms[b]), 1e-4f));
    }

    static rs64lights::CasterMode rs64CasterMode(const interop::OtherMode &om, bool extended, uint32_t triangles) {
        rs64lights::CasterMode cm;
        cm.zUpd = om.zUpd();
        cm.primDepth = (om.zSource() == G_ZS_PRIM);
        cm.xlu = ((om.L & ZMODE_MASK) == ZMODE_XLU);
        cm.fillOrCopy = (om.cycleType() == G_CYC_FILL) || (om.cycleType() == G_CYC_COPY);
        cm.extended = extended;
        cm.alphaCompare = (om.alphaCompare() != 0);
        cm.cvgXAlpha = ((om.L & (CVG_X_ALPHA | ALPHA_CVG_SEL)) == (CVG_X_ALPHA | ALPHA_CVG_SEL));
        cm.triangles = triangles;
        return cm;
    }

    FramebufferRenderer::RS64Passes FramebufferRenderer::planRS64Passes(const DrawParams &p, Framebuffer &framebuffer) {
        const FramebufferPair &fbPair = p.curWorkload->fbPairs[p.fbPairIndex];
        const DrawData &drawData = p.curWorkload->drawData;
        RS64Passes passes;

        const rs64lights::Config &lightsCfg = rs64lights::config();
        if (lightsCfg.shadows) {
            static bool s_capsLogged = false;
            if (!s_capsLogged) {
                s_capsLogged = true;
                const RenderDeviceCapabilities &caps = p.worker->device->getCapabilities();
                std::fprintf(stderr, "[rt-shadows] device raytracing=%d rayQuery=%d\n", (int)caps.raytracing, (int)caps.rayQuery);
            }
        }
        passes.msaa = (p.fbStorage->colorTarget != nullptr) && (p.fbStorage->colorTarget->multisampling.sampleCount > 1);
        passes.lightsWanted = lightsCfg.enabled && p.rs64Lights && (p.fbStorage->colorTarget != nullptr) && !(p.curWorkload->rs64Lights.empty() && p.curWorkload->pointLights.empty());
        // A new view over the world (the cockpit) traces sun shadows only, onto its own pixels (the world under it is cleared depth), with its geometry
        // joining the casters; the world's terrain origin and hit colours stay as the world's pass set them.
        passes.newView = framebuffer.rs64.newView;
        passes.shadowsWanted = (passes.newView ? lightsCfg.shadows.load(std::memory_order_relaxed) : rs64lights::tracePassWanted(lightsCfg.shadows, lightsCfg.ao, lightsCfg.gi, lightsCfg.reflections)) && p.rs64Lights && (p.fbStorage->colorTarget != nullptr) && p.curWorkload->rs64Sun.valid && p.worker->device->getCapabilities().rayQuery;
        passes.fogWanted = !passes.newView && lightsCfg.fogShafts && p.rs64Lights && (p.fbStorage->colorTarget != nullptr) && p.curWorkload->rs64Sun.valid && p.worker->device->getCapabilities().rayQuery;
        passes.lightShadowsWanted = !passes.newView && rs64lights::sceneWanted(false, passes.lightsWanted, lightsCfg.lightShadows, p.worker->device->getCapabilities().rayQuery);
        passes.sceneWanted = passes.shadowsWanted || passes.lightShadowsWanted || passes.fogWanted;
        passes.postWanted = passes.lightsWanted || passes.shadowsWanted || passes.fogWanted;
        framebuffer.rs64.msaa = passes.msaa;
        if (passes.postWanted && passes.msaa) {
            prepareRS64MSAA(p, framebuffer);
        }

        // The main 3D view is the perspective projection with the most draws; the pass runs after its last fragment (same view and projection), so other views such as the radar stay unlit.
        if (passes.postWanted) {
            // Ranked by scene-geometry draws (depth-writing casters), then by draws.
            uint64_t best = 0;
            for (uint32_t pr = 0; pr < fbPair.projectionCount; pr++) {
                const Projection &proj = fbPair.projections[pr];
                if ((proj.type != Projection::Type::Perspective) || proj.scissorRect.isNull() || (proj.gameCallCount == 0)) {
                    continue;
                }
                uint64_t casters = 0;
                for (uint32_t d = 0; d < proj.gameCallCount; d++) {
                    const GameCall &call = proj.gameCalls[d];
                    const rs64lights::CasterMode cm = rs64CasterMode(call.callDesc.otherMode, call.callDesc.extendedType != DrawExtendedType::None, call.callDesc.triangleCount);
                    casters += (rs64lights::casterClass(cm) != rs64lights::CasterNone) ? 1 : 0;
                }
                const uint64_t score = (casters << 32) | proj.gameCallCount;
                if (score > best) {
                    best = score;
                    passes.projIndex = pr;
                }
            }
            // A view with no scene geometry has nothing to light (the pause menu's emblem over the held mission frame, which already carries its passes).
            if ((best >> 32) == 0) {
                passes.projIndex = UINT32_MAX;
            }
            if (passes.projIndex != UINT32_MAX) {
                const uint32_t mainTi = fbPair.projections[passes.projIndex].transformsIndex;
                for (uint32_t pr = 0; pr < fbPair.projectionCount; pr++) {
                    const Projection &proj = fbPair.projections[pr];
                    // The radar is a few draws in camera space under the same matrices and viewport, after the scene; tiny fragments are skipped so it stays unlit.
                    if ((proj.type == Projection::Type::Perspective) && !proj.scissorRect.isNull() && (proj.gameCallCount >= 8) && rs64SameView(drawData, proj.transformsIndex, mainTi)) {
                        passes.lastIndex = pr;
                    }
                }
                // The pass goes right after the view's last scene-geometry draw (depth-writing, not XLU or prim-depth): effect sprites drawn after it
                // (explosions, glows) are not darkened by the ground they float over.
                for (uint32_t pr = passes.projIndex; (passes.lastIndex != UINT32_MAX) && (pr <= passes.lastIndex); pr++) {
                    const Projection &proj = fbPair.projections[pr];
                    if ((proj.type != Projection::Type::Perspective) || proj.scissorRect.isNull() || ((pr != passes.projIndex) && !rs64SameView(drawData, proj.transformsIndex, mainTi))) {
                        continue;
                    }
                    for (uint32_t d = 0; d < proj.gameCallCount; d++) {
                        const GameCall &call = proj.gameCalls[d];
                        const rs64lights::CasterMode cm = rs64CasterMode(call.callDesc.otherMode, call.callDesc.extendedType != DrawExtendedType::None, call.callDesc.triangleCount);
                        if (rs64lights::casterClass(cm) != rs64lights::CasterNone) {
                            passes.splitProj = pr;
                            passes.splitCall = d;
                        }
                    }
                }
            }
        }

        return passes;
    }

    void FramebufferRenderer::finishRS64Passes(const DrawParams &p, Framebuffer &framebuffer, const RS64Passes &passes) {
        const FramebufferPair &fbPair = p.curWorkload->fbPairs[p.fbPairIndex];
        const DrawData &drawData = p.curWorkload->drawData;
        const rs64lights::Config &lightsCfg = rs64lights::config();
        RenderTargetDrawCall &targetDrawCall = framebuffer.renderTargetDrawCall;

        if (passes.postWanted && passes.sawPersp && !passes.split) {
            targetDrawCall.rs64LightsAfterScene = static_cast<int32_t>(targetDrawCall.rasterScenes.size()) - 1;
        }

        if (!passes.mirrorSplit) {
            targetDrawCall.rs64MirrorAfterScene = static_cast<int32_t>(targetDrawCall.rasterScenes.size()) - 1;
        }
        const int32_t lightsScene = targetDrawCall.rs64LightsAfterScene;
        const bool lightsFilled = passes.lightsWanted && (lightsScene >= 0) && passes.haveProj && fillRS64Lights(p, framebuffer, passes.transformsIndex, passes.screenScale, passes.screenOffset);
        bool shadowsFilled = false;
        bool fogFilled = false;
        framebuffer.rs64.lightsShadowed = false;
        if (passes.sceneWanted && (lightsScene >= 0) && passes.haveProj) {
            fillRS64ScenePasses(p, framebuffer, passes, lightsFilled, shadowsFilled, fogFilled);
        }

        framebuffer.rs64.lightsOn = lightsFilled;
        framebuffer.rs64.shadowsOn = shadowsFilled;
        framebuffer.rs64.fogOn = fogFilled;
        framebuffer.rs64.target = p.fbStorage->colorTarget;
        if (!lightsFilled && !shadowsFilled && !fogFilled) {
            targetDrawCall.rs64LightsAfterScene = -1;
        }

        if (!shadowsFilled) {
            framebuffer.rs64.mirrorOn = false;
        }
        if (lightsCfg.log) {
            static uint32_t s_n[2] = {};
            uint32_t types[5] = {};
            for (uint32_t pr = 0; pr < fbPair.projectionCount; pr++) {
                ++types[std::min<uint32_t>(4, (uint32_t)fbPair.projections[pr].type)];
            }
            if ((types[1] > 0) && ((++s_n[(p.deltaTimeMs == 0.0f) ? 0 : 1] % 120) == 1)) {
                std::fprintf(stderr, "[rt-lights] fb: dt=%.2f projs none/persp/ortho/rect/tri=%u/%u/%u/%u/%u candidates=%zu", p.deltaTimeMs, types[0], types[1], types[2], types[3], types[4], p.curWorkload->rs64Lights.size());
                std::fprintf(stderr, " order[main=%u last=%u]:", passes.projIndex, passes.lastIndex);
                for (uint32_t pr = 0; pr < fbPair.projectionCount; pr++) {
                    const Projection &q = fbPair.projections[pr];
                    const bool same = (passes.projIndex != UINT32_MAX) && rs64SameView(drawData, q.transformsIndex, fbPair.projections[passes.projIndex].transformsIndex);
                    std::fprintf(stderr, " %u:t%d:c%u:s%d:ti%u:sc(%d,%d,%d,%d)", pr, (int)q.type, q.gameCallCount, (int)same, q.transformsIndex, q.scissorRect.ulx, q.scissorRect.uly, q.scissorRect.lrx, q.scissorRect.lry);
                }
                std::fprintf(stderr, " color=%d msaa=%d main=%d split=%d proj=%d scene=%d filled=%d ti=%u modView=%zu modProj=%zu world=%zu lerp=%zu\n",
                    (int)(p.fbStorage->colorTarget != nullptr), (int)passes.msaa, (int)passes.sawPersp, (int)passes.split, (int)passes.haveProj, lightsScene, (int)lightsFilled,
                    passes.transformsIndex, drawData.modViewTransforms.size(), drawData.modProjTransforms.size(), drawData.worldTransforms.size(), drawData.lerpWorldTransforms.size());
            }
        }
    }

    void FramebufferRenderer::fillRS64ScenePasses(const DrawParams &p, Framebuffer &framebuffer, const RS64Passes &passes, bool lightsFilled, bool &shadowsFilled, bool &fogFilled) {
        const FramebufferPair &fbPair = p.curWorkload->fbPairs[p.fbPairIndex];
        const DrawData &drawData = p.curWorkload->drawData;
        const OutputBuffers &outputBuffers = p.curWorkload->outputBuffers;
        const RenderBuffer *worldPosRes = outputBuffers.worldPosBuffer.buffer.get();
        const RenderBuffer *tcRes = outputBuffers.genTexCoordBuffer.buffer.get();
        const rs64lights::Config &lightsCfg = rs64lights::config();

        // Casters: opaque, depth-writing, non-prim-depth triangles of the main view (prim depth = sky dome, sprites, glows).
        const uint32_t mainTi = fbPair.projections[passes.projIndex].transformsIndex;
        // Terrain casts from the static HMP map when this frame's record-space origin solves; its drawn (geomorphing, view-limited) draws are then skipped.
        const rs64lights::TerrainFrame &tf = p.curWorkload->rs64Terrain;
        const bool terrainLerp = drawData.lerpWorldTransforms.size() == drawData.worldTransforms.size();
        const auto &terrainWorld = terrainLerp ? drawData.lerpWorldTransforms : drawData.worldTransforms;
        int32_t terrainOx = 0;
        int32_t terrainOz = 0;
        uint32_t terrainVotes = 0;
        if (!passes.newView) {
            rs64Shadow.worldViewTi = mainTi;
        }

        if (!passes.newView && lightsCfg.shadowTerrain && tf.map && !tf.mixed && (tf.transformIndex < terrainWorld.size())) {
            const bool hasPrev = (rs64Shadow.terrainOriginVersion == tf.map->version);
            if (tf.exact) {
                terrainOx = tf.exactX;
                terrainOz = tf.exactZ;
                terrainVotes = 1;
            }
            else {
                terrainVotes = rs64lights::solveTerrainOrigin(*tf.map, tf.samples.data(), tf.samples.size(), terrainOx, terrainOz, hasPrev, rs64Shadow.terrainOriginX, rs64Shadow.terrainOriginZ);
            }
        }
        if (!passes.newView) {
            rs64Shadow.terrainActive = (terrainVotes > 0);
        }

        if (!passes.newView && rs64Shadow.terrainActive) {
            rs64Shadow.terrainOriginVersion = tf.map->version;
            rs64Shadow.terrainOriginX = terrainOx;
            rs64Shadow.terrainOriginZ = terrainOz;
            rs64Shadow.terrainMap = tf.map;
            // The trace pass's constants are filled before the scene build sets this for a new map, so a level's first frame had no water.
            if (rs64Shadow.terrainBuiltVersion != tf.map->version) {
                rs64Shadow.terrainWaterPrim = rs64lights::terrainWaterPrim(*tf.map, lightsCfg.shadowTerrainSub);
            }

            rs64lights::terrainInstanceTransform(rs64ToMat4(terrainWorld[tf.transformIndex]), terrainOx, terrainOz, rs64Shadow.terrainTransform);
        }
        const bool hitColors = !passes.newView && (lightsCfg.gi || lightsCfg.reflections || (rs64Shadow.terrainWaterPrim > 0));
        thread_local std::vector<std::pair<rs64lights::IndexRange, uint32_t>> terrainCalls;
        terrainCalls.clear();
        // Ranges accumulate over the workload's framebuffer pairs (the game can split one view across pairs); recordSetup consumes them.
        for (uint32_t pr = 0; pr < fbPair.projectionCount; pr++) {
            const Projection &proj = fbPair.projections[pr];
            if ((proj.type != Projection::Type::Perspective) || proj.scissorRect.isNull() || !rs64lights::casterFragment(pr == passes.projIndex, rs64SameView(drawData, proj.transformsIndex, mainTi), proj.gameCallCount)) {
                continue;
            }
            for (uint32_t d = 0; d < proj.gameCallCount; d++) {
                const GameCall &call = proj.gameCalls[d];
                const rs64lights::CasterMode cm = rs64CasterMode(call.callDesc.otherMode, call.callDesc.extendedType != DrawExtendedType::None, call.callDesc.triangleCount);
                const bool drawnTerrain = rs64Shadow.terrainActive && (call.callDesc.minWorldMatrix == call.callDesc.maxWorldMatrix) && rs64lights::isTerrainTransform(tf, call.callDesc.minWorldMatrix);
                const bool oneTransform = (call.callDesc.minWorldMatrix == call.callDesc.maxWorldMatrix);
                const auto &noCast = p.curWorkload->rs64NoCast;
                const auto &noCastCutout = p.curWorkload->rs64NoCastCutout;
                const auto &emissiveXlu = p.curWorkload->rs64EmissiveXlu;
                const bool emissive = oneTransform && ((std::find(noCast.begin(), noCast.end(), call.callDesc.minWorldMatrix) != noCast.end())
                    || (cm.xlu && (std::find(emissiveXlu.begin(), emissiveXlu.end(), call.callDesc.minWorldMatrix) != emissiveXlu.end())));
                const bool glowCard = oneTransform && (std::find(noCastCutout.begin(), noCastCutout.end(), call.callDesc.minWorldMatrix) != noCastCutout.end());
                const rs64lights::CasterClass cc = rs64lights::casterAfterDeny(rs64lights::casterClass(cm), emissive, glowCard, lightsCfg.shadowCutout);
                if (hitColors && drawnTerrain) {
                    terrainCalls.push_back({ { call.meshDesc.faceIndicesStart, call.callDesc.triangleCount }, call.callDesc.callIndex });
                }
                // The sky dome: the only cull=BOTH geometry pinned to a flat depth (isDomeTri in rt64_rsp.cpp).
                const uint32_t cullBoth = call.callDesc.cullBothMask;
                if (hitColors && cm.primDepth && (cullBoth != 0) && ((call.callDesc.geometryMode & cullBoth) == cullBoth) && !cm.fillOrCopy) {
                    rs64Shadow.skyRanges.push_back({ call.meshDesc.faceIndicesStart, call.callDesc.triangleCount });
                    rs64Shadow.skyCalls.push_back(call.callDesc.callIndex);
                    // Traced too, so water reflects the dome where it is off screen (looking down) instead of a flat haze.
                    if (lightsCfg.reflections && !passes.newView) {
                        const uint32_t shade = rs64lights::packCutoutShade(p.curWorkload->drawData.faceIndices, p.curWorkload->drawData.normColBytes, call.meshDesc.faceIndicesStart, call.callDesc.triangleCount);
                        rs64Shadow.cutoutRanges.push_back({ { call.meshDesc.faceIndicesStart, call.callDesc.triangleCount }, call.callDesc.callIndex, call.callDesc.tileIndex, rs64lights::CutoutSky | shade });
                    }
                    if (!rs64Shadow.skyWorldValid && oneTransform && (call.callDesc.minWorldMatrix < terrainWorld.size())) {
                        rs64Shadow.skyWorld = rs64ToMat4(terrainWorld[call.callDesc.minWorldMatrix]);
                        rs64Shadow.skyWorldValid = true;
                    }
                }

                if (!passes.newView && (emissive || glowCard) && !cm.fillOrCopy) {
                    rs64Shadow.emissiveRanges.push_back({ call.meshDesc.faceIndicesStart, call.callDesc.triangleCount });
                    rs64Shadow.emissiveCalls.push_back(call.callDesc.callIndex);
                }

                // Translucent world draws (a World Devastator's foot beams, foam) join the cutouts flagged translucent: water reflections alpha-test them.
                const bool translucent = !passes.newView && lightsCfg.reflections && cm.xlu && !cm.primDepth && !cm.fillOrCopy && (cc == rs64lights::CasterNone) && !emissive && !glowCard;
                if (translucent && !drawnTerrain) {
                    const uint32_t shade = rs64lights::packCutoutShade(p.curWorkload->drawData.faceIndices, p.curWorkload->drawData.normColBytes, call.meshDesc.faceIndicesStart, call.callDesc.triangleCount);
                    rs64Shadow.cutoutRanges.push_back({ { call.meshDesc.faceIndicesStart, call.callDesc.triangleCount }, call.callDesc.callIndex, call.callDesc.tileIndex, rs64lights::CutoutTranslucent | shade });
                }
                if (!drawnTerrain) {
                    const auto &craftTi = p.curWorkload->rs64Craft;
                    const bool craft = oneTransform && (std::find(craftTi.begin(), craftTi.end(), call.callDesc.minWorldMatrix) != craftTi.end());
                    if ((cc == rs64lights::CasterOpaque) && craft) {
                        rs64Shadow.craftRanges.push_back({ call.meshDesc.faceIndicesStart, call.callDesc.triangleCount });
                    }
                    else if (cc == rs64lights::CasterOpaque) {
                        rs64Shadow.ranges.push_back({ call.meshDesc.faceIndicesStart, call.callDesc.triangleCount });
                        rs64Shadow.rangeCalls.push_back(call.callDesc.callIndex);
                    }
                    else if (cc == rs64lights::CasterCutout) {
                        rs64Shadow.cutoutRanges.push_back({ { call.meshDesc.faceIndicesStart, call.callDesc.triangleCount }, call.callDesc.callIndex, call.callDesc.tileIndex });
                    }
                }
            }
        }
        rs64Shadow.indices.clear();
        rs64lights::gatherCasterIndices(drawData.faceIndices.data(), drawData.faceIndices.size(), rs64Shadow.ranges.data(), rs64Shadow.ranges.size(), rs64Shadow.indices);
        rs64Shadow.craftIndices.clear();
        rs64lights::gatherCasterIndices(drawData.faceIndices.data(), drawData.faceIndices.size(), rs64Shadow.craftRanges.data(), rs64Shadow.craftRanges.size(), rs64Shadow.craftIndices);

        // Triangles kept only for casting (wholly outside the view): classed like a drawn call; opaque ones in the main view cast (terrain casts from its own map).
        for (const rs64lights::CastOnlyTri &t : p.curWorkload->rs64CastOnly) {
            interop::OtherMode om;
            om.L = t.otherModeL;
            om.H = t.otherModeH;
            const rs64lights::CasterMode cm = rs64CasterMode(om, false, 1);
            const auto &noCast = p.curWorkload->rs64NoCast;
            const auto &glow = p.curWorkload->rs64NoCastCutout;
            const auto &craftTi = p.curWorkload->rs64Craft;
            const bool denied = std::find(noCast.begin(), noCast.end(), t.transform) != noCast.end();
            const bool glowCard = std::find(glow.begin(), glow.end(), t.transform) != glow.end();
            const bool terrain = rs64Shadow.terrainActive && rs64lights::isTerrainTransform(tf, t.transform);
            const bool worldView = (rs64Shadow.worldViewTi != UINT32_MAX) && rs64SameView(drawData, t.view, rs64Shadow.worldViewTi);
            if (terrain || (rs64lights::casterAfterDeny(rs64lights::casterClass(cm), denied, glowCard, lightsCfg.shadowCutout) != rs64lights::CasterOpaque) || !(rs64SameView(drawData, t.view, mainTi) || worldView))
            {
                continue;
            }

            auto &dst = (std::find(craftTi.begin(), craftTi.end(), t.transform) != craftTi.end()) ? rs64Shadow.craftIndices : rs64Shadow.indices;
            dst.insert(dst.end(), t.idx, t.idx + 3);
        }
        rs64Shadow.cutoutIndices.clear();
        rs64Shadow.cutoutTable.clear();
        rs64lights::buildCutoutScene(drawData.faceIndices.data(), drawData.faceIndices.size(), rs64Shadow.cutoutRanges.data(), rs64Shadow.cutoutRanges.size(), rs64Shadow.cutoutIndices, rs64Shadow.cutoutTable);
        rs64Shadow.worldPos = worldPosRes;
        rs64Shadow.texCoords = tcRes;
        rs64Shadow.texCoordsSize = outputBuffers.genTexCoordBuffer.allocatedSize;
        rs64Shadow.shadedCols = outputBuffers.shadedColBuffer.buffer.get();
        rs64Shadow.shadedColsSize = outputBuffers.shadedColBuffer.allocatedSize;
        rs64Shadow.vertexCount = drawData.vertexCount();
        const bool sceneHasCasters = !rs64Shadow.indices.empty() || !rs64Shadow.cutoutIndices.empty() || !rs64Shadow.craftIndices.empty() || rs64Shadow.terrainActive;
        const rs64lights::Sun sun = resolveRS64Sun(p, passes, mainTi, terrainWorld, tf);

        // Emissive draws need their table without GI or reflections too: the shadow pass finds them on screen to leave them unshadowed.
        if (hitColors || !rs64Shadow.emissiveRanges.empty()) {
            buildRS64HitColors(p, terrainCalls, (sun.fromModel && (sun.skip[3] > 0.0f)) ? sun.skip : nullptr);
        }
        // The frame's fog, for the shafts and for fogging what water reflects. The game fogs only terrain and a few models, and terrain often draws
        // in its own same-view projection.
        rs64lights::FogParams fogPick;
        if ((passes.fogWanted || passes.shadowsWanted) && sceneHasCasters) {
            fogPick = pickRS64Fog(p, mainTi);
        }

        const rs64lights::FogParams &fog = rs64lights::workloadFog(rs64FogCache, p.curWorkload->workloadId, fogPick);
        shadowsFilled = passes.shadowsWanted && sceneHasCasters && fillRS64Shadows(p, framebuffer, passes.transformsIndex, passes.screenScale, passes.screenOffset, sun, fog);
        if (passes.fogWanted && sceneHasCasters) {
            const bool fogDebug = (lightsCfg.fogShaftsDebug != 0);
            const bool fogVisible = fog.valid && (fogDebug || rs64lights::fogShaftsVisible(fog.color, lightsCfg.fogShaftsStrength));
            const rs64lights::Haze &haze = p.curWorkload->rs64Haze;
            const bool hazeVisible = !fogDebug && (haze.amount > 0.0f) && (haze.range > 0.0f) && (lightsCfg.hazeStrength > 0.0f);
            fogFilled = (fogVisible || hazeVisible) && fillRS64FogShafts(p, framebuffer, passes.transformsIndex, passes.screenScale, passes.screenOffset, sun, fog, fogVisible, hazeVisible);
            if (lightsCfg.log) {
                static uint32_t s_f = 0;
                if ((++s_f % 120) == 1) {
                    std::fprintf(stderr, "[rt-fog] pick: valid=%d tris=%u mul=%.2f offset=%.2f color=(%.3f,%.3f,%.3f) filled=%d haze=%.2f/%.0f\n", (int)fog.valid, fog.tris, fog.mul, fog.offset, fog.color[0], fog.color[1], fog.color[2], (int)fogFilled, hazeVisible ? haze.amount : 0.0f, haze.range);
                }
            }
        }
        framebuffer.rs64.lightsShadowed = lightsFilled && passes.lightShadowsWanted && sceneHasCasters && (framebuffer.rs64.lightsShadowedSet != nullptr);
        if (lightsCfg.log) {
            static uint32_t s_n = 0;
            if ((++s_n % 120) == 1) {
                std::fprintf(stderr, "[rt-shadows] casters: ranges=%zu indices=%zu cutouts=%zu cutoutTris=%zu vertices=%u filled=%d sun=(%.3f,%.3f,%.3f)\n", rs64Shadow.ranges.size(), rs64Shadow.indices.size(), rs64Shadow.cutoutTable.size(), rs64Shadow.cutoutIndices.size() / 3, rs64Shadow.vertexCount, (int)shadowsFilled,
                    sun.dir[0], sun.dir[1], sun.dir[2]);
                std::fprintf(stderr, "[rt-terrain] origin: noCast=%zu samples=%zu votes=%u ti=%u transforms=%zu mixed=%d origin=(%d,%d) active=%d lightsShadowed=%d\n", p.curWorkload->rs64NoCast.size(), tf.samples.size(), terrainVotes, tf.transformIndex, tf.transforms.size(), (int)tf.mixed, terrainOx, terrainOz, (int)rs64Shadow.terrainActive, (int)framebuffer.rs64.lightsShadowed);
            }
        }
    }

    rs64lights::Sun FramebufferRenderer::resolveRS64Sun(const DrawParams &p, const RS64Passes &passes, uint32_t mainTi, const std::vector<interop::float4x4> &terrainWorld, const rs64lights::TerrainFrame &tf) {
        const FramebufferPair &fbPair = p.curWorkload->fbPairs[p.fbPairIndex];
        const DrawData &drawData = p.curWorkload->drawData;
        const rs64lights::Config &lightsCfg = rs64lights::config();

        // Hangar: the largest draw (the ship, resting level) gives the world up; floor-facing receivers keep the game's painted shadow.
        rs64lights::Sun sun = p.curWorkload->rs64Sun;
        // Boot sequence: the authored key light is fixed in world space, so a moving camera does not swing the shadows.
        float viewR[9];
        if (sun.worldAuthored && (mainTi < drawData.viewTransforms.size()) && rs64lights::viewRotation(rs64ToMat4(drawData.viewTransforms[mainTi]), viewR)) {
            sun.valid = rs64lights::sunCameraDir(rs64lights::menuSunDir(), viewR, sun.dir);
        }

        // Level sun: through the (interpolated) terrain transform, the same frame the terrain casters use, so it stays fixed to the world.
        // Boot sequence passes with no terrain drawn keep the last terrain rotation; the authored sun points elsewhere, so the lit faces of the 3D logo flipped.
        static float s_introViewR[9];
        static bool s_introViewKnown = false;
        const float sunMinTanElev = std::tan(lightsCfg.sunMinElevationDeg * 3.14159265f / 180.0f);

        if (!sun.levelWorld) {
            s_introViewKnown = false;
        }

        if (sun.levelWorld && rs64Shadow.terrainActive && rs64lights::viewRotation(rs64ToMat4(terrainWorld[tf.transformIndex]), viewR)) {
            sun.valid = rs64lights::sunCameraDir(sun.world, viewR, sun.dir, sunMinTanElev);
            memcpy(s_introViewR, viewR, sizeof(viewR));
            s_introViewKnown = true;
        }
        else if (sun.levelWorld && sun.worldAuthored && s_introViewKnown) {
            sun.valid = rs64lights::sunCameraDir(sun.world, s_introViewR, sun.dir, sunMinTanElev);
        }

        if (sun.fromModel) {
            // World up by triangle-weighted vote over the main view's draws: the single largest draw alternates between tilted props in the fly-in.
            const Projection &mainProj = fbPair.projections[passes.projIndex];
            const bool sunLerp = drawData.lerpWorldTransforms.size() == drawData.worldTransforms.size();
            const auto &sunWorld = sunLerp ? drawData.lerpWorldTransforms : drawData.worldTransforms;
            static const float kStraightDown[3] = { 0.0f, 1.0f, 0.0f };
            thread_local std::vector<float> ups;
            thread_local std::vector<uint32_t> upWeights;
            ups.clear();
            upWeights.clear();
            for (uint32_t d = 0; d < mainProj.gameCallCount; d++) {
                const GameCall &call = mainProj.gameCalls[d];
                const uint32_t ti = call.callDesc.minWorldMatrix;
                float u[3];
                if ((call.callDesc.triangleCount > 0) && (ti == call.callDesc.maxWorldMatrix) && (ti < sunWorld.size()) && rs64lights::modelSunDir(kStraightDown, rs64ToMat4(sunWorld[ti]), u)) {
                    ups.insert(ups.end(), u, u + 3);
                    upWeights.push_back(call.callDesc.triangleCount);
                }
            }
            float up[3];
            if (rs64lights::consensusUp(ups.data(), upWeights.data(), upWeights.size(), 0.9994f, up)) {
                sun.skip[0] = up[0];
                sun.skip[1] = up[1];
                sun.skip[2] = up[2];
                sun.skip[3] = lightsCfg.hangarFloorCos;
            }
        }

        return sun;
    }

    rs64lights::FogParams FramebufferRenderer::pickRS64Fog(const DrawParams &p, uint32_t mainTi) {
        const FramebufferPair &fbPair = p.curWorkload->fbPairs[p.fbPairIndex];
        const DrawData &drawData = p.curWorkload->drawData;
        rs64lights::FogParams fogPick;

        for (uint32_t pr = 0; pr < fbPair.projectionCount; pr++) {
            const Projection &fogProj = fbPair.projections[pr];
            if ((fogProj.type != Projection::Type::Perspective) || !rs64SameView(drawData, fogProj.transformsIndex, mainTi)) {
                continue;
            }
            for (uint32_t d = 0; d < fogProj.gameCallCount; d++) {
                const GameCall &call = fogProj.gameCalls[d];
                const uint32_t fi = call.meshDesc.faceIndicesStart;
                if ((call.callDesc.triangleCount == 0) || (fi >= drawData.faceIndices.size())) {
                    continue;
                }
                const uint32_t v = drawData.faceIndices[fi];
                const uint16_t fogIndex = (v < drawData.fogIndices.size()) ? drawData.fogIndices[v] : 0;
                if ((fogIndex == 0) || (fogIndex > drawData.rspFog.size())) {
                    continue;
                }
                const interop::RSPFog &rf = drawData.rspFog[fogIndex - 1];
                const interop::float4 &fc = call.callDesc.rdpParams.fogColor;
                const float color[3] = { fc.x, fc.y, fc.z };
                rs64lights::considerFogCall(fogPick, call.callDesc.triangleCount, rf.mul, rf.offset, color, rf.tailEnd);
            }
        }

        return fogPick;
    }

    // The whole-map terrain pass draws at this call: its transforms, fog and the frame's terrain origin.
    void FramebufferRenderer::setRS64TerrainAnchor(const DrawParams &p, const GameCall &call, interop::float2 resolution) {
        const DrawData &drawData = p.curWorkload->drawData;
        const rs64lights::TerrainPassFrame &tp = p.curWorkload->rs64TerrainPass;
        const uint32_t v = drawData.faceIndices[call.meshDesc.faceIndicesStart];
        rs64TerrainMap = tp.map;
        rs64TerrainFrame = tp;
        rs64TerrainCB.worldIndex = drawData.worldIndices[v];
        rs64TerrainCB.viewProjIndex = drawData.viewProjIndices[v];

        // The modified view-projection when the projection processor ran (interpolation, widening, VR eyes), as rs64ViewProj picks it.
        const bool modViewProj = (drawData.modViewProjTransforms.size() == drawData.viewProjTransforms.size());
        const auto &terrainViewProj = modViewProj ? drawData.modViewProjTransforms : drawData.viewProjTransforms;
        rs64TerrainMvp = hlslpp::mul(hlslpp::float4x4(drawData.worldTransforms[rs64TerrainCB.worldIndex]), hlslpp::float4x4(terrainViewProj[rs64TerrainCB.viewProjIndex]));
        rs64TerrainCB.fogIndex = drawData.fogIndices[v];
        rs64TerrainCB.originX = tp.originX;
        rs64TerrainCB.originZ = tp.originZ;
        rs64TerrainCB.offsetX = float(tp.offsetX);
        rs64TerrainCB.offsetZ = float(tp.offsetZ);
        rs64TerrainCB.fogColor = drawData.rdpParams[call.callDesc.callIndex].fogColor;
        rs64TerrainCB.resolution = resolution;
        rs64TerrainCB.heightScale = tp.map ? tp.map->heightScale : 1.0f;
        rs64TerrainCB.fogClip = tp.fogClip ? 1u : 0u;
        rs64TerrainCB.rippleTime = rs64lights::rippleClock();
        rs64TerrainCB.rippleStrength = rs64lights::config().waterRipple;
    }

    bool FramebufferRenderer::rs64ViewProj(const DrawParams &p, uint32_t transformsIndex, rs64lights::Mat4 &out) {
        const DrawData &drawData = p.curWorkload->drawData;

        // The projection processor only fills the modified transforms when it runs (interpolation or aspect adjustment); otherwise the raster pass uses the raw ones.
        const bool modified = (drawData.modViewTransforms.size() == drawData.viewTransforms.size()) && (drawData.modProjTransforms.size() == drawData.projTransforms.size());
        const auto &viewTransforms = modified ? drawData.modViewTransforms : drawData.viewTransforms;
        const auto &projTransforms = modified ? drawData.modProjTransforms : drawData.projTransforms;
        if ((transformsIndex >= viewTransforms.size()) || (transformsIndex >= projTransforms.size())) {
            return false;
        }

        out = rs64lights::mul(rs64ToMat4(viewTransforms[transformsIndex]), rs64ToMat4(projTransforms[transformsIndex]));
        return true;
    }

    bool FramebufferRenderer::rs64ViewFromScreen(const DrawParams &p, uint32_t transformsIndex, hlslpp::float2 screenScale, hlslpp::float2 screenOffset, rs64lights::Mat4 &out, rs64lights::ScreenMap *mapOut) {
        const DrawData &drawData = p.curWorkload->drawData;
        if (transformsIndex >= drawData.rspViewports.size()) {
            return false;
        }

        rs64lights::Mat4 viewProj;
        if (!rs64ViewProj(p, transformsIndex, viewProj)) {
            return false;
        }

        // Same resolution the raster pass maps screen positions with (FbParams.resolution).
        const interop::RSPViewport &vp = drawData.rspViewports[transformsIndex];
        rs64lights::ScreenMap map;
        map.vpScale[0] = vp.scale.x;
        map.vpScale[1] = vp.scale.y;
        map.vpScale[2] = vp.scale.z;
        map.vpTranslate[0] = vp.translate.x;
        map.vpTranslate[1] = vp.translate.y;
        map.vpTranslate[2] = vp.translate.z;
        map.res[0] = p.targetWidth / p.resolutionScale.x;
        map.res[1] = p.targetHeight / p.resolutionScale.y;
        map.screenScale[0] = screenScale.x;
        map.screenScale[1] = screenScale.y;
        map.screenOffset[0] = screenOffset.x;
        map.screenOffset[1] = screenOffset.y;
        if (mapOut != nullptr) {
            *mapOut = map;
        }

        return rs64lights::viewFromScreen(viewProj, map, out);
    }

    bool FramebufferRenderer::fillRS64Lights(const DrawParams &p, Framebuffer &framebuffer, uint32_t transformsIndex, hlslpp::float2 screenScale, hlslpp::float2 screenOffset) {
        const DrawData &drawData = p.curWorkload->drawData;
        const auto &candidates = p.curWorkload->rs64Lights;
        rs64lights::Mat4 inv;
        rs64lights::ScreenMap screenMap;
        if (!rs64ViewFromScreen(p, transformsIndex, screenScale, screenOffset, inv, &screenMap)) {
            return false;
        }
        const bool lerp = drawData.lerpWorldTransforms.size() == drawData.worldTransforms.size();
        const auto &world = lerp ? drawData.lerpWorldTransforms : drawData.worldTransforms;
        thread_local std::vector<rs64lights::Resolved> resolved;
        resolved.clear();
        // The game's lamps drawn this frame (2D rectangles), in final NDC with their reach (half width plus slack, per axis).
        thread_local std::vector<std::array<float, 2>> lampMarks;
        thread_local std::vector<float> lampReach;
        thread_local std::vector<const rs64lights::Candidate *> lampMarkOf;
        lampMarks.clear();
        lampReach.clear();
        lampMarkOf.clear();
        // Lamp rectangles are 2D and come in the game's next graphics task, so they are taken from the shared store (rs64lights::lampMarksFor).
        thread_local std::vector<rs64lights::Candidate> lampMarkSet;
        rs64lights::lampMarksFor(p.curWorkload->workloadId, 3, lampMarkSet);
        for (const rs64lights::Candidate &c : lampMarkSet) {
            float n[2];
            rs64lights::screenToFinalNdc(screenMap, c.local[0], c.local[1], n);
            lampMarks.push_back({ n[0], n[1] });
            const float slack = c.screenHalf * 1.5f + 2.0f;
            lampReach.push_back(slack / (screenMap.res[0] * 0.5f) * screenMap.screenScale[0]);
            lampReach.push_back(slack / (screenMap.res[1] * 0.5f) * screenMap.screenScale[1]);
            lampMarkOf.push_back(&c);
        }

        for (const rs64lights::Candidate &c : candidates) {
            const bool cameraSpace = (c.transformIndex == rs64lights::CameraSpace);
            if (!cameraSpace && ((c.transformIndex >= world.size()) || !rs64SameView(drawData, c.viewIndex, transformsIndex, c.modelUnits))) {
                continue;
            }
            const float v[4] = { c.local[0], c.local[1], c.local[2], 1.0f };
            float o[4] = { v[0], v[1], v[2], 1.0f };
            if (!cameraSpace) {
                rs64lights::transform(v, rs64ToMat4(world[c.transformIndex]), o);
            }
            rs64lights::Resolved r;
            r.pos[0] = o[0] / o[3];
            r.pos[1] = o[1] / o[3];
            r.pos[2] = o[2] / o[3];
            r.radius = c.radius;
            r.color[0] = c.color[0];
            r.color[1] = c.color[1];
            r.color[2] = c.color[2];
            r.intensity = c.intensity;
            r.falloff = c.falloff;
            r.shadowStart = c.shadowStart * rs64lights::config().spriteShadowStart;
            r.kind = c.kind;
            r.engine = (c.kind == rs64lights::KindMesh) && c.modelUnits;
            r.spotMaxCos = c.spotMaxCos;
            r.spotFalloffCos = c.spotFalloffCos;
            r.flicker = c.flicker;
            r.seed = c.seed;
            if (cameraSpace) {
                std::copy(c.dir, c.dir + 3, r.dir);
            }
            else {
                rs64lights::rotateDir(c.dir, rs64ToMat4(world[c.transformIndex]), r.dir);
            }
            if (c.modelUnits && !cameraSpace) {
                const float s = rs64lights::transformScale(rs64ToMat4(world[c.transformIndex]));
                r.radius = c.radius * s;
                r.shadowStart = c.shadowStart * s;
            }
            r.lampGated = c.lampGated;
            r.projectile = c.projectile;
            r.priority = c.priority;
            resolved.push_back(r);
        }

        // Lamp presets' lights shine only while a lamp is drawn on their spot, in its colour and fade (blink, facing).
        rs64lights::Mat4 lampViewProj;
        const bool lampProj = rs64ViewProj(p, transformsIndex, lampViewProj);
        const size_t lampLights = std::count_if(resolved.begin(), resolved.end(), [](const rs64lights::Resolved &r) { return r.lampGated; });
        const size_t beforeGate = resolved.size();
        thread_local std::vector<std::array<float, 3>> lampProjected;
        lampProjected.clear();
        resolved.erase(std::remove_if(resolved.begin(), resolved.end(), [&](rs64lights::Resolved &r) {
            if (!r.lampGated) {
                return false;
            }

            float s[3];
            size_t best = 0;
            if (!lampProj || lampMarks.empty()) {
                return true;
            }

            rs64lights::projectToScreen(lampViewProj, screenMap, r.pos, s);
            lampProjected.push_back({ s[0], s[1], s[2] });
            if ((s[2] < 0.0f) || (s[2] > 1.0f) || !rs64lights::lampMatch(s, reinterpret_cast<const float (*)[2]>(lampMarks.data()), lampReach.data(), lampMarks.size(), best)) {
                return true;
            }

            // The lamp's own colour at the preset's strength (its colour's peak).
            const rs64lights::Candidate &m = *lampMarkOf[best];
            const float strength = std::max(r.color[0], std::max(r.color[1], r.color[2]));
            for (int j = 0; j < 3; ++j) {
                r.color[j] = m.color[j] * strength;
            }

            r.intensity *= m.intensity;
            return false;
        }), resolved.end());

        // Every frame with lamps on screen for the first 100, then one in 300.
        if (rs64lights::config().log && !lampMarks.empty()) {
            static uint32_t s_lampLog = 0;
            if ((s_lampLog++ < 100) || ((s_lampLog % 300) == 0)) {
                fprintf(stderr, "[rt-lights] lamps: %zu drawn, %zu lamp lights, %zu lit (res %.0fx%.0f scale %.3f,%.3f offset %.3f,%.3f)\n", lampMarks.size(), lampLights,
                    lampLights - (beforeGate - resolved.size()), screenMap.res[0], screenMap.res[1], screenMap.screenScale[0], screenMap.screenScale[1], screenMap.screenOffset[0], screenMap.screenOffset[1]);
                // Each of the first marks with the nearest projected lamp light, both in final NDC: what the match compares.
                for (size_t i = 0; (i < lampMarks.size()) && (i < 4); ++i) {
                    float bestD = 1e9f;
                    std::array<float, 3> near3 = { 0.0f, 0.0f, 0.0f };
                    for (const auto &q : lampProjected) {
                        const float d = std::hypot(q[0] - lampMarks[i][0], q[1] - lampMarks[i][1]);
                        if (d < bestD) {
                            bestD = d;
                            near3 = q;
                        }
                    }

                    fprintf(stderr, "[rt-lights]   lamp px (%.1f, %.1f) half %.1f -> ndc (%.3f, %.3f) reach (%.3f, %.3f); nearest light ndc (%.3f, %.3f) z %.3f, %zu projected\n",
                        lampMarkOf[i]->local[0], lampMarkOf[i]->local[1], lampMarkOf[i]->screenHalf, lampMarks[i][0], lampMarks[i][1], lampReach[i * 2], lampReach[i * 2 + 1],
                        near3[0], near3[1], near3[2], lampProjected.size());
                }
            }
        }

        // RT64 light presets (inspector Lights tab): level map space (the terrain pass's frame) in a level, camera space elsewhere; skipped while a level's origin is unknown.
        const rs64lights::TerrainFrame &tf = p.curWorkload->rs64Terrain;
        // A front-end scene (rs64PresetScene) is camera space even with a terrain map (the briefing room's hologram draws the level's terrain).
        const bool inLevel = (tf.map != nullptr) && (p.curWorkload->rs64PresetScene == -1);
        const bool originKnown = inLevel && !tf.mixed && (tf.transformIndex < world.size()) && (tf.exact || (rs64Shadow.terrainOriginVersion == tf.map->version));
        float mapToCamera[3][4] = {};
        float mapScale = 1.0f;
        rs64lights::Mat4 presetToCamera = rs64lights::identity();
        if (originKnown) {
            const rs64lights::Mat4 recordToCamera = rs64ToMat4(world[tf.transformIndex]);
            rs64lights::terrainInstanceTransform(recordToCamera, tf.exact ? tf.exactX : rs64Shadow.terrainOriginX, tf.exact ? tf.exactZ : rs64Shadow.terrainOriginZ, mapToCamera);
            mapScale = rs64lights::transformScale(recordToCamera);
            for (int j = 0; j < 3; ++j) {
                for (int k = 0; k < 4; ++k) {
                    presetToCamera.m[k][j] = mapToCamera[j][k];
                }
            }
        }
        // Gizmo views for the F1 Lights tab: a space -> camera transform to space -> final NDC, the camera's place in the space, a depth to unproject at.
        rs64lights::Mat4 cameraToNdc;
        const bool haveNdc = rs64lights::inverse(inv, cameraToNdc);
        const auto gizmoView = [&](const rs64lights::Mat4 &spaceToCamera, bool yDown) {
            rs64lights::PresetView v;
            rs64lights::Mat4 cameraFromSpace;
            if (!haveNdc || !rs64lights::inverse(spaceToCamera, cameraFromSpace)) {
                return v;
            }
            v.toNdc = rs64lights::mul(spaceToCamera, cameraToNdc);
            v.valid = rs64lights::inverse(v.toNdc, v.fromNdc);
            v.yDown = yDown;
            for (int j = 0; j < 3; ++j) {
                v.cameraPos[j] = cameraFromSpace.m[3][j];
            }
            v.depth = drawData.rspViewports[transformsIndex].translate.z;
            return v;
        };
        if (!inLevel || originKnown) {
            rs64lights::publishPresetView(gizmoView(presetToCamera, originKnown), rs64lights::PresetLevel);
        }
        // The picked mesh preset's part: its model space (the game's models are Y-down), when it was drawn in this view.
        const uint32_t pickTi = p.curWorkload->rs64MeshPickTransform;
        const bool pickHere = (pickTi < world.size()) && rs64SameView(drawData, p.curWorkload->rs64MeshPickView, transformsIndex, true);
        rs64lights::publishPresetView(pickHere ? gizmoView(rs64ToMat4(world[pickTi]), true) : rs64lights::PresetView{}, rs64lights::PresetMesh);
        for (const interop::PointLight &l : p.curWorkload->pointLights) {
            if (inLevel && !originKnown) {
                break;
            }
            const float m[3] = { l.position.x, l.position.y, l.position.z };
            rs64lights::Resolved r;
            for (int j = 0; j < 3; ++j) {
                r.pos[j] = originKnown ? (mapToCamera[j][0] * m[0] + mapToCamera[j][1] * m[1] + mapToCamera[j][2] * m[2] + mapToCamera[j][3]) : m[j];
            }
            r.radius = l.attenuationRadius * mapScale;
            r.color[0] = l.diffuseColor.x;
            r.color[1] = l.diffuseColor.y;
            r.color[2] = l.diffuseColor.z;
            r.intensity = 1.0f;
            r.falloff = l.attenuationExponent;
            r.shadowStart = l.shadowOffset * mapScale;
            r.kind = rs64lights::KindMesh;
            r.priority = rs64lights::PriorityLevel;
            const float d[3] = { l.direction.x, l.direction.y, l.direction.z };
            for (int j = 0; j < 3; ++j) {
                r.dir[j] = originKnown ? (mapToCamera[j][0] * d[0] + mapToCamera[j][1] * d[1] + mapToCamera[j][2] * d[2]) : d[j];
            }
            const float dirLen = std::sqrt(r.dir[0] * r.dir[0] + r.dir[1] * r.dir[1] + r.dir[2] * r.dir[2]);
            for (float &v : r.dir) {
                v = (dirLen > 1e-12f) ? v / dirLen : v;
            }
            r.spotMaxCos = l.spotMaxCosine;
            r.spotFalloffCos = l.spotFalloffCosine;

            // A panel: width along the horizontal axis across its facing (X when it faces straight up or down), height across both.
            if ((l.rectSize.x > 0.0f) && (l.rectSize.y > 0.0f) && (dirLen > 1e-12f)) {
                const float dl = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
                const float n[3] = { d[0] / dl, d[1] / dl, d[2] / dl };
                float u[3] = { -n[2], 0.0f, n[0] };
                const float ul = std::sqrt(u[0] * u[0] + u[2] * u[2]);

                if (ul < 1e-4f) {
                    u[0] = 1.0f;
                    u[2] = 0.0f;
                }
                else {
                    u[0] /= ul;
                    u[2] /= ul;
                }

                const float v[3] = { n[1] * u[2] - n[2] * u[1], n[2] * u[0] - n[0] * u[2], n[0] * u[1] - n[1] * u[0] };
                const float hw = l.rectSize.x * 0.5f;
                const float hh = l.rectSize.y * 0.5f;

                for (int j = 0; j < 3; ++j) {
                    const float a[3] = { u[0] * hw, u[1] * hw, u[2] * hw };
                    const float b[3] = { v[0] * hh, v[1] * hh, v[2] * hh };
                    r.rectU[j] = originKnown ? (mapToCamera[j][0] * a[0] + mapToCamera[j][1] * a[1] + mapToCamera[j][2] * a[2]) : a[j];
                    r.rectV[j] = originKnown ? (mapToCamera[j][0] * b[0] + mapToCamera[j][1] * b[1] + mapToCamera[j][2] * b[2]) : b[j];
                }

                r.spotMaxCos = 1.0f;
            }

            r.flicker = l.flickerIntensity;
            // A level light's flicker sequence follows its place (stable while it isn't moved).
            r.seed = uint32_t(rs64lights::fnv1a64(reinterpret_cast<const uint8_t *>(m), sizeof(m)));
            resolved.push_back(r);
        }

        resolved.resize(rs64lights::dedupeLights(resolved.data(), resolved.size(), 1.0f));
        resolved.resize(rs64lights::dropFarLights(resolved.data(), resolved.size(), rs64lights::config().minLightReach));
        rs64lights::Resolved chosen[rs64lights::MaxLights];
        const size_t count = rs64lights::selectLights(resolved.data(), resolved.size(), chosen, rs64lights::MaxLights, rs64lights::config().cutFade, size_t(rs64lights::config().maxProjectileLights));
        rs64lights::publishActiveLights(chosen, count, resolved.size());
        if (count == 0) {
            return false;
        }
        if (rs64lights::config().log) {
            static uint32_t s_n = 0;
            if ((++s_n % 90) == 1) {
                const auto isProjectile = [](const rs64lights::Resolved &r) { return r.projectile; };
                std::fprintf(stderr, "[rt-lights] chosen %zu of %zu (lasers %zu of %zu; by priority", count, resolved.size(), size_t(std::count_if(chosen, chosen + count, isProjectile)),
                    size_t(std::count_if(resolved.begin(), resolved.end(), isProjectile)));
                std::map<int, std::pair<int, int>, std::greater<int>> byPriority;
                for (const rs64lights::Resolved &r : resolved) {
                    byPriority[r.priority].second++;
                }
                for (size_t i = 0; i < count; ++i) {
                    byPriority[chosen[i].priority].first++;
                }
                for (const auto &[pr, n] : byPriority) {
                    std::fprintf(stderr, " %d:%d/%d", pr, n.first, n.second);
                }
                std::fprintf(stderr, "):");
                for (size_t i = 0; i < count && i < 6; ++i) {
                    std::fprintf(stderr, " (%.0f,%.0f,%.0f r=%.0f i=%.2f)", chosen[i].pos[0], chosen[i].pos[1], chosen[i].pos[2], chosen[i].radius, chosen[i].intensity);
                }
                std::fprintf(stderr, "\n");
            }
        }

        const rs64lights::Config &cfg = rs64lights::config();
        interop::RS64LightsCB cb = {};
        memcpy(&cb.viewFromScreen, &inv, sizeof(inv));
        cb.viewport = { framebuffer.viewport.x, framebuffer.viewport.y, framebuffer.viewport.width, framebuffer.viewport.height };
        cb.lightCount = static_cast<uint32_t>(count);
        cb.debugMode = static_cast<uint32_t>(cfg.debugMode);
        cb.gain = cfg.gain;
        cb.wrap = cfg.wrap;
        cb.debugRange = cfg.debugRange;
        cb.tMin = cfg.shadowTMin;
        cb.normalBias = cfg.shadowNormalBias;
        cb.terrainTMinScale = cfg.shadowTerrainTMin;
        cb.shadowParams = { cfg.lightShadowEnd, 0.0f, (cfg.terrainNormals && rs64TerrainNormalsThisFrame(rs64Shadow)) ? 1.0f : 0.0f, cfg.terrainNormalWindow };
        cb.specParams = { cfg.specular, cfg.specularPower, 0.0f, 0.0f };

        // ROGUESQ_RT_LIGHTS_HALF=0 shades every pixel; the debug views always do.
        static const bool s_lightsHalf = rs64env::on("ROGUESQ_RT_LIGHTS_HALF", true);
        framebuffer.rs64.lightsHalfOn = s_lightsHalf && (cfg.debugMode == 0) && (framebuffer.rs64.lightsUpsampleSet != nullptr);
        cb.halfParams = { framebuffer.rs64.lightsHalfOn ? 1.0f : 0.0f, 0.0f, 0.0f, 0.0f };
        if (framebuffer.rs64.lightsHalfOn) {
            const RenderTarget *ct = p.fbStorage->colorTarget;
            const uint32_t hw = (ct->width + 1) / 2;
            const uint32_t hh = (ct->height + 1) / 2;
            framebuffer.rs64.lightsHalf.ensure(p.worker->device, hw, hh, RenderFormat::R16G16B16A16_FLOAT);

            RS64LightsUpsampleDescriptorSet &us = *framebuffer.rs64.lightsUpsampleSet;
            us.setBuffer(us.gLights, framebuffer.rs64.lightsBuffer.get(), sizeof(interop::RS64LightsCB));
            bindRS64Depth(p, framebuffer, us, us.gDepth);
            us.setTexture(us.gHalf, framebuffer.rs64.lightsHalf.texture.get(), RenderTextureLayout::SHADER_READ);
        }

        // Flicker runs on wall time, so it keeps its pace whatever the frame rate.
        static const auto s_flickerStart = std::chrono::steady_clock::now();
        const double flickerT = std::chrono::duration<double>(std::chrono::steady_clock::now() - s_flickerStart).count();
        for (size_t i = 0; i < count; ++i) {
            const rs64lights::Resolved &c = chosen[i];
            cb.lightPosRadius[i] = { c.pos[0], c.pos[1], c.pos[2], c.radius };
            cb.lightColor[i] = { c.color[0], c.color[1], c.color[2], c.intensity * rs64lights::flickerGain(c.flicker, c.seed, flickerT) };
            cb.lightParams[i] = { c.falloff, c.shadowStart, c.spotMaxCos, c.spotFalloffCos };
            cb.lightSpot[i] = { c.dir[0], c.dir[1], c.dir[2], 0.0f };
            cb.lightRectU[i] = { c.rectU[0], c.rectU[1], c.rectU[2], 0.0f };
            cb.lightRectV[i] = { c.rectV[0], c.rectV[1], c.rectV[2], 0.0f };
        }

        void *bytes = framebuffer.rs64.lightsBuffer->map();
        memcpy(bytes, &cb, sizeof(cb));
        framebuffer.rs64.lightsBuffer->unmap();
        framebuffer.rs64.lightsSet->setBuffer(framebuffer.rs64.lightsSet->gLights, framebuffer.rs64.lightsBuffer.get(), sizeof(interop::RS64LightsCB));
        bindRS64Depth(p, framebuffer, *framebuffer.rs64.lightsSet, framebuffer.rs64.lightsSet->gDepth);
        if (framebuffer.rs64.lightsShadowedSet) {
            framebuffer.rs64.lightsShadowedSet->setBuffer(framebuffer.rs64.lightsShadowedSet->gParams, framebuffer.rs64.lightsBuffer.get(), sizeof(interop::RS64LightsCB));
            bindRS64Depth(p, framebuffer, *framebuffer.rs64.lightsShadowedSet, framebuffer.rs64.lightsShadowedSet->gDepth);
            bindRS64Depth(p, framebuffer, *framebuffer.rs64.lightsShadowedSet, framebuffer.rs64.lightsShadowedSet->gColor);
        }
        framebuffer.rs64.lightsDebug = (cfg.debugMode != 0);

        framebuffer.rs64.panels = (cfg.debugMode == 0) && std::any_of(chosen, chosen + count, [](const rs64lights::Resolved &r) { return (r.rectU[0] != 0.0f) || (r.rectU[1] != 0.0f) || (r.rectU[2] != 0.0f); });

        if (framebuffer.rs64.panels) {
            cb.debugMode = RS64_LIGHTS_PANEL_FACES;
            cb.halfParams = { 0.0f, 0.0f, 0.0f, 0.0f };
            bytes = framebuffer.rs64.panelsBuffer->map();
            memcpy(bytes, &cb, sizeof(cb));
            framebuffer.rs64.panelsBuffer->unmap();
            framebuffer.rs64.panelsSet->setBuffer(framebuffer.rs64.panelsSet->gLights, framebuffer.rs64.panelsBuffer.get(), sizeof(interop::RS64LightsCB));
            bindRS64Depth(p, framebuffer, *framebuffer.rs64.panelsSet, framebuffer.rs64.panelsSet->gDepth);
        }

        return true;
    }

    bool FramebufferRenderer::fillRS64Shadows(const DrawParams &p, Framebuffer &framebuffer, uint32_t transformsIndex, hlslpp::float2 screenScale, hlslpp::float2 screenOffset, const rs64lights::Sun &sun, const rs64lights::FogParams &fog) {
        rs64lights::Mat4 inv;
        if (!sun.valid || !framebuffer.rs64.shadowsSet || !rs64ViewFromScreen(p, transformsIndex, screenScale, screenOffset, inv)) {
            return false;
        }

        rs64lights::publishSun(sun);
        const rs64lights::Config &cfg = rs64lights::config();
        interop::RS64ShadowsCB cb = {};
        memcpy(&cb.viewFromScreen, &inv, sizeof(inv));
        cb.viewport = { framebuffer.viewport.x, framebuffer.viewport.y, framebuffer.viewport.width, framebuffer.viewport.height };
        cb.sunDir = { sun.dir[0], sun.dir[1], sun.dir[2], 0.0f };
        cb.debugMode = static_cast<uint32_t>(cfg.shadowsDebug);
        cb.strength = cfg.shadowStrength;
        cb.tMin = cfg.shadowTMin;
        cb.tMax = cfg.shadowTMax;
        cb.sunTMax = sun.fromModel ? std::min(cfg.shadowTMax, cfg.hangarShadowRange) : cfg.shadowTMax;
        cb.normalBias = cfg.shadowNormalBias;
        cb.terrainTMinScale = cfg.shadowTerrainTMin;
        cb.receiverSkip = { sun.skip[0], sun.skip[1], sun.skip[2], sun.skip[3] };

        // Soft path (the only one that carries AO, GI and reflections): debug 0 or 5-8.
        const bool indirect = cfg.ao || cfg.gi || cfg.reflections;
        rs64lights::Mat4 viewProj;
        const bool soft = (cfg.softShadows || indirect) && ((cfg.shadowsDebug == 0) || (cfg.shadowsDebug >= 5)) && framebuffer.rs64.blurHSet && (p.fbStorage->colorTarget != nullptr) && rs64ViewProj(p, transformsIndex, viewProj);
        if (!soft && !cfg.shadows) {
            return false;
        }
        // Soft path without soft shadows (AO/GI/reflections only): one sharp sun ray.
        const float tanAngle = cfg.softShadows ? std::tan(cfg.sunAngleDeg * 3.14159265f / 180.0f) : 0.0f;
        const float maxRadius = cfg.softBlurPx * p.resolutionScale.y;
        cb.softRays = cfg.softShadows ? static_cast<uint32_t>(cfg.softRays) : 1u;
        const float res = p.resolutionScale.y;
        // Ranges are fractions of each pixel's view distance; the hangar is a fixed-size room the camera zooms through (about 5000 to 1200 units), so
        // there they are absolute (negative = world units), anchored at the settled view so zooming in doesn't shorten the rays and brighten the ship.
        constexpr float kHangarViewDist = 1200.0f;
        const float rangeScale = sun.fromModel ? -kHangarViewDist : 1.0f;
        cb.aoParams = { float(cfg.aoRays), cfg.aoRange * rangeScale, cfg.aoStrength, cfg.aoBlurPx * res };
        cb.giParams = { float(cfg.giRays), cfg.giRange * rangeScale, cfg.giStrength, cfg.giBlurPx * res };
        cb.reflParams = { std::tan(cfg.reflRoughnessDeg * 3.14159265f / 180.0f), cfg.reflRange * rangeScale, cfg.reflStrength, cfg.giEmissive };
        cb.sunColor = { sun.color[0], sun.color[1], sun.color[2], 0.0f };
        cb.featureMask = (cfg.shadows ? 1u : 0u) | ((soft && cfg.ao) ? 2u : 0u) | ((soft && cfg.gi) ? 4u : 0u) | ((soft && cfg.reflections && sun.fromModel) ? 8u : 0u);
        cb.waterPrim = rs64Shadow.terrainWaterPrim;
        cb.terrainMapWidth = rs64Shadow.terrainMap ? rs64Shadow.terrainMap->width : 0u;
        cb.terrainMapHeight = rs64Shadow.terrainMap ? rs64Shadow.terrainMap->height : 0u;
        cb.rippleTime = rs64lights::rippleClock();
        cb.rippleStrength = cfg.waterRipple;
        cb.fogParams = { fog.mul, fog.offset, fog.valid ? 1.0f : 0.0f, fog.tailEnd };
        // w: VR, so the fog follows the game camera's depth like the vertex fog (rs64FogDepth).
        cb.fogColor = { fog.color[0], fog.color[1], fog.color[2], (rs64_vr_active() != 0) ? 1.0f : 0.0f };
        const hlslpp::float4 &skyH = rs64Shadow.skyHorizon, &skyZ = rs64Shadow.skyZenith;
        cb.skyHorizon = { (float)skyH.x, (float)skyH.y, (float)skyH.z, cfg.waterShadow };
        cb.skyZenith = { (float)skyZ.x, (float)skyZ.y, (float)skyZ.z, ((float)skyZ.w > 0.0f) ? cfg.reflSky : 0.0f };
        cb.skyDome = { (float)rs64Shadow.skyDome.x, (float)rs64Shadow.skyDome.y, (float)rs64Shadow.skyDome.z, (float)rs64Shadow.skyDome.w };
        cb.featureMask |= (soft && (cb.waterPrim > 0)) ? 16u : 0u;
        cb.featureMask |= cfg.modelTerminator ? 32u : 0u;
        if (framebuffer.rs64.newView) {
            cb.featureMask &= (1u | 32u);
        }
        // The temporal pass upsamples, so the half-resolution trace needs it on.
        // Not in the hangar: its shadows fall close to the camera on large flat surfaces, where the lower resolution bands.
        const bool halfTrace = soft && cfg.softHalfRes && !sun.fromModel && (cfg.temporalAlpha > 0.0f) && (cfg.temporalAlpha < 1.0f);
        cb.featureMask |= halfTrace ? 64u : 0u;
        cb.terrainParams = { cfg.terrainNormalWindow, cfg.reflWaterBlurPx * res, cfg.terminatorFlatLo, std::max(cfg.terminatorFlatHi, cfg.terminatorFlatLo + 0.01f) };
        rs64lights::Mat4 sfv;
        if (rs64lights::inverse(inv, sfv)) {
            memcpy(&cb.screenFromView, &sfv, sizeof(sfv));
        }
        cb.softParams = { tanAngle, soft ? rs64lights::pixelsPerUnit(viewProj, framebuffer.viewport.height) : 0.0f, maxRadius, (cfg.terrainNormals && rs64TerrainNormalsThisFrame(rs64Shadow)) ? (cfg.terrainTerminator ? 2.0f : 1.0f) : 0.0f };

        void *bytes = framebuffer.rs64.shadowsBuffer->map();
        memcpy(bytes, &cb, sizeof(cb));
        framebuffer.rs64.shadowsBuffer->unmap();
        framebuffer.rs64.shadowsSet->setBuffer(framebuffer.rs64.shadowsSet->gParams, framebuffer.rs64.shadowsBuffer.get(), sizeof(interop::RS64ShadowsCB));
        bindRS64Depth(p, framebuffer, *framebuffer.rs64.shadowsSet, framebuffer.rs64.shadowsSet->gDepth);
        if (soft) {
            framebuffer.rs64.shadowsSet->setTexture(framebuffer.rs64.shadowsSet->gColor, p.fbStorage->colorTarget->getResolvedTexture(), RenderTextureLayout::SHADER_READ);
        }
        else {
            bindRS64Depth(p, framebuffer, *framebuffer.rs64.shadowsSet, framebuffer.rs64.shadowsSet->gColor);
        }

        framebuffer.rs64.shadowsDebug = (cfg.shadowsDebug != 0);
        framebuffer.rs64.shadowsSoft = soft;
        if (soft) {
            ensureRS64SoftTargets(p.worker, framebuffer, p.fbStorage->colorTarget->width, p.fbStorage->colorTarget->height, halfTrace);
            interop::RS64ShadowBlurCB bcb = {};
            memcpy(&bcb.viewFromScreen, &inv, sizeof(inv));
            bcb.viewport = cb.viewport;
            const float blurMax = std::max({ maxRadius, cfg.ao ? cb.aoParams.w : 0.0f, cfg.gi ? cb.giParams.w : 0.0f, ((cb.featureMask & 16u) != 0u) ? cb.terrainParams.y : 0.0f });
            bcb.params = { cfg.shadows ? cfg.shadowStrength : 0.0f, 0.02f, (cfg.shadowsDebug >= 5) ? 1.0f : 0.0f, blurMax };
            bcb.aoParams = { cfg.aoStrength, (cfg.shadowsDebug >= 5) ? float(cfg.shadowsDebug) : 0.0f, 0.0f, 0.0f };
            fillRS64Temporal(p, framebuffer, transformsIndex, inv, bcb);
            bcb.temporal.w = halfTrace ? 1.0f : 0.0f;
            float waterUp[3];
            framebuffer.rs64.mirrorOn = (cfg.reflMirror > 0.0f) && !framebuffer.rs64.newView && (cfg.shadowsDebug == 0) && rs64Shadow.terrainActive
                && (rs64Shadow.terrainWaterPrim > 0) && rs64lights::affinePlaneNormalY(rs64Shadow.terrainTransform, waterUp);
            if (framebuffer.rs64.mirrorOn) {
                bcb.mirrorUp = { waterUp[0], waterUp[1], waterUp[2], cfg.reflMirror };
            }

            bcb.texelDir = { 1.0f, 0.0f };
            memcpy(framebuffer.rs64.blurHBuffer->map(), &bcb, sizeof(bcb));
            framebuffer.rs64.blurHBuffer->unmap();
            bcb.texelDir = { 0.0f, 1.0f };
            memcpy(framebuffer.rs64.blurVBuffer->map(), &bcb, sizeof(bcb));
            framebuffer.rs64.blurVBuffer->unmap();
            RS64ShadowBlurDescriptorSet &hs = *framebuffer.rs64.blurHSet;
            RS64ShadowBlurDescriptorSet &vs = *framebuffer.rs64.blurVSet;
            hs.setBuffer(hs.gBlur, framebuffer.rs64.blurHBuffer.get(), sizeof(interop::RS64ShadowBlurCB));
            bindRS64Depth(p, framebuffer, hs, hs.gDepth);
            const uint32_t w = framebuffer.rs64.histTarget;
            hs.setTexture(hs.gMask, framebuffer.rs64.temporalOn ? framebuffer.rs64.histMask[w].get() : framebuffer.rs64.softMask.get(), RenderTextureLayout::SHADER_READ);
            hs.setTexture(hs.gIndirect, framebuffer.rs64.temporalOn ? framebuffer.rs64.histIndirect[w].get() : framebuffer.rs64.softIndirect.get(), RenderTextureLayout::SHADER_READ);
            if (framebuffer.rs64.temporalOn) {
                RS64ShadowTemporalDescriptorSet &ts = *framebuffer.rs64.temporalSet;
                ts.setBuffer(ts.gBlur, framebuffer.rs64.blurHBuffer.get(), sizeof(interop::RS64ShadowBlurCB));
                bindRS64Depth(p, framebuffer, ts, ts.gDepth);
                ts.setTexture(ts.gMask, framebuffer.rs64.softMask.get(), RenderTextureLayout::SHADER_READ);
                ts.setTexture(ts.gIndirect, framebuffer.rs64.softIndirect.get(), RenderTextureLayout::SHADER_READ);
                ts.setTexture(ts.gHistMask, framebuffer.rs64.histMask[w ^ 1].get(), RenderTextureLayout::SHADER_READ);
                ts.setTexture(ts.gHistIndirect, framebuffer.rs64.histIndirect[w ^ 1].get(), RenderTextureLayout::SHADER_READ);
            }
            vs.setTexture(vs.gIndirect, framebuffer.rs64.softIndirectPing.get(), RenderTextureLayout::SHADER_READ);
            vs.setBuffer(vs.gBlur, framebuffer.rs64.blurVBuffer.get(), sizeof(interop::RS64ShadowBlurCB));
            bindRS64Depth(p, framebuffer, vs, vs.gDepth);
            vs.setTexture(vs.gMask, framebuffer.rs64.softPing.get(), RenderTextureLayout::SHADER_READ);
            if (framebuffer.rs64.mirrorOn) {
                const RenderTarget *ct = p.fbStorage->colorTarget;
                if (!framebuffer.rs64.mirrorLate || (framebuffer.rs64.mirrorW != ct->width) || (framebuffer.rs64.mirrorH != ct->height) || (framebuffer.rs64.mirrorFormat != ct->format)) {
                    framebuffer.rs64.mirrorLate = p.worker->device->createTexture(RenderTextureDesc::ColorTarget(ct->width, ct->height, ct->format));
                    framebuffer.rs64.mirrorW = ct->width;
                    framebuffer.rs64.mirrorH = ct->height;
                    framebuffer.rs64.mirrorFormat = ct->format;
                }

                RS64WaterMirrorDescriptorSet &ms = *framebuffer.rs64.mirrorSet;
                ms.setBuffer(ms.gBlur, framebuffer.rs64.blurVBuffer.get(), sizeof(interop::RS64ShadowBlurCB));
                bindRS64Depth(p, framebuffer, ms, ms.gDepth);
                ms.setTexture(ms.gMirror, framebuffer.rs64.softMirror.get(), RenderTextureLayout::SHADER_READ);
                ms.setTexture(ms.gLate, framebuffer.rs64.mirrorLate.get(), RenderTextureLayout::SHADER_READ);
            }
        }
        else {
            framebuffer.rs64.mirrorOn = false;
        }
        return true;
    }

    // Temporal AO / indirect for this slot: the camera motion since its previous frame, voted over the main view's objects by matrix id
    // (rs64lights::consensusCameraDelta), and the history set the pass writes this frame. No motion found = no history this frame.
    void FramebufferRenderer::fillRS64Temporal(const DrawParams &p, Framebuffer &framebuffer, uint32_t transformsIndex, const rs64lights::Mat4 &viewFromScreen, interop::RS64ShadowBlurCB &bcb) {
        const rs64lights::Config &cfg = rs64lights::config();
        framebuffer.rs64.temporalOn = (cfg.temporalAlpha > 0.0f) && (cfg.temporalAlpha < 1.0f);
        framebuffer.rs64.reuseTrace = false;
        if (!framebuffer.rs64.temporalOn) {
            framebuffer.rs64.histValid = false;
            return;
        }

        const DrawData &drawData = p.curWorkload->drawData;
        const FramebufferPair &fbPair = p.curWorkload->fbPairs[p.fbPairIndex];
        const bool lerp = drawData.lerpWorldTransforms.size() == drawData.worldTransforms.size();
        const auto &world = lerp ? drawData.lerpWorldTransforms : drawData.worldTransforms;
        thread_local std::unordered_map<uint32_t, rs64lights::Mat4> objects;
        thread_local std::unordered_set<uint32_t> ambiguous;
        thread_local std::vector<std::pair<uint32_t, uint32_t>> calls;
        thread_local std::unordered_map<uint32_t, size_t> callOf;
        objects.clear();
        ambiguous.clear();
        calls.clear();
        callOf.clear();
        for (uint32_t pr = 0; pr < fbPair.projectionCount; pr++) {
            const Projection &proj = fbPair.projections[pr];
            if ((proj.type != Projection::Type::Perspective) || proj.scissorRect.isNull() || !rs64SameView(drawData, proj.transformsIndex, transformsIndex)) {
                continue;
            }
            for (uint32_t d = 0; d < proj.gameCallCount; d++) {
                const auto &desc = proj.gameCalls[d].callDesc;
                const uint32_t ti = desc.minWorldMatrix;
                if ((ti != desc.maxWorldMatrix) || (ti >= world.size()) || (ti >= drawData.worldTransformGroups.size()) || (desc.triangleCount == 0)) {
                    continue;
                }
                const uint32_t id = drawData.transformGroups[drawData.worldTransformGroups[ti]].matrixId;
                if ((id == G_EX_ID_AUTO) || (id == G_EX_ID_IGNORE)) {
                    continue;
                }
                const rs64lights::Mat4 m = rs64ToMat4(world[ti]);
                const auto ins = objects.emplace(id, m);
                if (!ins.second && !rs64lights::sameMatrix(ins.first->second, m, 1e-3f)) {
                    ambiguous.insert(id);
                }

                // One vote per object, weighted by all its triangles: its calls' deltas are identical, and per-call votes made the all-pairs vote quadratic in draw calls.
                const auto slot = callOf.emplace(id, calls.size());
                if (slot.second) {
                    calls.emplace_back(id, desc.triangleCount);
                }
                else {
                    calls[slot.first->second].second += desc.triangleCount;
                }
            }
        }

        thread_local std::vector<rs64lights::Mat4> deltas;
        thread_local std::vector<uint32_t> weights;
        deltas.clear();
        weights.clear();
        for (const auto &c : calls) {
            const auto prev = framebuffer.rs64.prevObjects.find(c.first);
            rs64lights::Mat4 curInv;
            if ((prev == framebuffer.rs64.prevObjects.end()) || ambiguous.count(c.first) || !rs64lights::inverse(objects[c.first], curInv)) {
                continue;
            }
            deltas.push_back(rs64lights::mul(curInv, prev->second));
            weights.push_back(c.second);
        }

        rs64lights::Mat4 delta;
        const bool moved = framebuffer.rs64.histValid && rs64lights::consensusCameraDelta(deltas.data(), weights.data(), deltas.size(), 0.002f, 2.0f, delta);

        // An interpolated frame reprojects the real frame's history without tracing (camera unchanged if no motion is found) and leaves the
        // history, objects and camera of that real frame for the next one.
        framebuffer.rs64.reuseTrace = rs64lights::g_reuseTrace && framebuffer.rs64.histValid;
        if (framebuffer.rs64.reuseTrace) {
            if (!moved) {
                delta = rs64lights::identity();
            }

            const rs64lights::Mat4 prevScreen = rs64lights::mul(delta, framebuffer.rs64.prevScreenFromView);
            memcpy(&bcb.prevScreenFromView, &prevScreen, sizeof(prevScreen));
            memcpy(&bcb.prevViewFromView, &delta, sizeof(delta));
            bcb.prevViewport = { framebuffer.rs64.prevViewport[0], framebuffer.rs64.prevViewport[1], framebuffer.rs64.prevViewport[2], framebuffer.rs64.prevViewport[3] };
            bcb.temporal = { 1.0f, 0.03f, cfg.temporalAlpha, 0.0f };
            bcb.aoParams.z = 1.0f;
            framebuffer.rs64.histTarget = framebuffer.rs64.histWrite ^ 1;
            return;
        }

        if (moved) {
            const rs64lights::Mat4 prevScreen = rs64lights::mul(delta, framebuffer.rs64.prevScreenFromView);
            memcpy(&bcb.prevScreenFromView, &prevScreen, sizeof(prevScreen));
            memcpy(&bcb.prevViewFromView, &delta, sizeof(delta));
            bcb.prevViewport = { framebuffer.rs64.prevViewport[0], framebuffer.rs64.prevViewport[1], framebuffer.rs64.prevViewport[2], framebuffer.rs64.prevViewport[3] };
        }
        bcb.temporal = { moved ? 1.0f : 0.0f, 0.03f, cfg.temporalAlpha, 0.0f };
        if (cfg.log) {
            static uint32_t s_n = 0;
            if ((++s_n % 120) == 1) {
                std::fprintf(stderr, "[rt-temporal] objects=%zu matched=%zu history=%d\n", objects.size(), deltas.size(), moved ? 1 : 0);
            }
        }

        // This frame becomes the next frame's history.
        framebuffer.rs64.histWrite ^= 1;
        framebuffer.rs64.histTarget = framebuffer.rs64.histWrite;
        for (uint32_t id : ambiguous) {
            objects.erase(id);
        }
        framebuffer.rs64.prevObjects.swap(objects);
        rs64lights::inverse(viewFromScreen, framebuffer.rs64.prevScreenFromView);
        framebuffer.rs64.prevViewport[0] = framebuffer.viewport.x;
        framebuffer.rs64.prevViewport[1] = framebuffer.viewport.y;
        framebuffer.rs64.prevViewport[2] = framebuffer.viewport.width;
        framebuffer.rs64.prevViewport[3] = framebuffer.viewport.height;
        framebuffer.rs64.histValid = true;
    }

    // Hit colours in BLAS triangle order for the opaque casters (ranges, mirroring gatherCasterIndices' skips), the emissive draws (their own BLAS), the frame's
    // drawn terrain and the caster average: vertex-colour average x the call's render-tile TMEM average x prim.
    void FramebufferRenderer::buildRS64HitColors(const DrawParams &p, const std::vector<std::pair<rs64lights::IndexRange, uint32_t>> &terrainCalls, const float *floorUp) {
        RS64ShadowScene &s = rs64Shadow;
        const DrawData &drawData = p.curWorkload->drawData;
        const auto &fi = drawData.faceIndices;
        const auto &nc = drawData.normColBytes;
        const auto &callTex = p.curWorkload->rs64CallTex;
        auto drawColor = [&](const rs64lights::IndexRange &r, uint32_t callIndex, float out[3]) {
            float texRgb[3] = {};
            const float *tex = nullptr;
            const uint32_t flags = (callIndex < callTex.size()) ? uint32_t((float)callTex[callIndex].w) : 2u;
            if ((flags & 1u) != 0) {
                texRgb[0] = (float)callTex[callIndex].x;
                texRgb[1] = (float)callTex[callIndex].y;
                texRgb[2] = (float)callTex[callIndex].z;
                tex = texRgb;
            }
            float vtx[3] = { 1.0f, 1.0f, 1.0f };
            if (tex == nullptr) {
                float sum[3] = {};
                uint32_t n = 0;
                const uint32_t count = r.triangleCount * 3;
                const uint32_t step = std::max<uint32_t>(1, r.triangleCount / 16) * 3;
                for (uint32_t k = 0; (k < count) && (size_t(r.start) + k < fi.size()); k += step) {
                    const size_t v = fi[r.start + k];
                    if (v * 4 + 3 < nc.size()) {
                        sum[0] += nc[v * 4 + 0] / 255.0f;
                        sum[1] += nc[v * 4 + 1] / 255.0f;
                        sum[2] += nc[v * 4 + 2] / 255.0f;
                        ++n;
                    }
                }
                for (int k = 0; (k < 3) && (n > 0); ++k) {
                    vtx[k] = sum[k] / float(n);
                }
            }
            float prim[3] = { 1.0f, 1.0f, 1.0f };
            if (((flags & 2u) != 0) && (callIndex < drawData.rdpParams.size())) {
                prim[0] = (float)drawData.rdpParams[callIndex].primColor.x;
                prim[1] = (float)drawData.rdpParams[callIndex].primColor.y;
                prim[2] = (float)drawData.rdpParams[callIndex].primColor.z;
            }
            rs64lights::drawAverageColor(vtx, tex, prim, out);
        };
        // Area-weighted world-space normal sum of up to 64 of a draw's triangles (the hangar camera is unrotated, so world space is camera space).
        const auto &pos = drawData.posFloats;
        const auto &wIdx = drawData.worldIndices;
        float sumN[3] = {};
        auto drawNormalSum = [&](const rs64lights::IndexRange &r, float out[3]) {
            out[0] = out[1] = out[2] = 0.0f;
            const uint32_t step = std::max<uint32_t>(1, r.triangleCount / 64);
            for (uint32_t t = 0; t < r.triangleCount; t += step) {
                float w[3][3];
                bool ok = true;
                for (uint32_t k = 0; (k < 3) && ok; ++k) {
                    const size_t f = size_t(r.start) + t * 3 + k;
                    const size_t v = (f < fi.size()) ? fi[f] : SIZE_MAX;
                    if ((v == SIZE_MAX) || (v * 3 + 2 >= pos.size()) || (v >= wIdx.size()) || (wIdx[v] >= drawData.worldTransforms.size())) {
                        ok = false;
                        break;
                    }
                    const rs64lights::Mat4 m = rs64ToMat4(drawData.worldTransforms[wIdx[v]]);
                    for (int j = 0; j < 3; ++j) {
                        w[k][j] = pos[v * 3 + 0] * m.m[0][j] + pos[v * 3 + 1] * m.m[1][j] + pos[v * 3 + 2] * m.m[2][j] + m.m[3][j];
                    }
                }
                if (!ok) {
                    continue;
                }
                const float e1[3] = { w[1][0] - w[0][0], w[1][1] - w[0][1], w[1][2] - w[0][2] };
                const float e2[3] = { w[2][0] - w[0][0], w[2][1] - w[0][1], w[2][2] - w[0][2] };
                out[0] += e1[1] * e2[2] - e1[2] * e2[1];
                out[1] += e1[2] * e2[0] - e1[0] * e2[2];
                out[2] += e1[0] * e2[1] - e1[1] * e2[0];
            }
            return (out[0] != 0.0f) || (out[1] != 0.0f) || (out[2] != 0.0f);
        };
        auto buildTable = [&](const std::vector<rs64lights::IndexRange> &ranges, const std::vector<uint32_t> &calls, bool emissive, std::vector<rs64lights::HitColorEntry> &table, double sum[3]) {
            table.clear();
            uint32_t tri = 0;
            for (size_t r = 0; (r < ranges.size()) && (r < calls.size()); ++r) {
                const size_t count = size_t(ranges[r].triangleCount) * 3;
                if ((count == 0) || (size_t(ranges[r].start) + count > fi.size())) {
                    continue;
                }
                float c[3];
                drawColor(ranges[r], calls[r], c);
                uint32_t rgba = rs64lights::packHitColor(c[0], c[1], c[2], emissive);
                if ((floorUp != nullptr) && !emissive && drawNormalSum(ranges[r], sumN)) {
                    if (rs64lights::floorFacing(sumN, floorUp, 0.9f)) {
                        rgba = rs64lights::markFloor(rgba);
                    }
                }
                table.push_back({ tri, rgba });
                tri += ranges[r].triangleCount;
                for (int k = 0; k < 3; ++k) {
                    sum[k] += c[k];
                }
            }
        };
        double casterSum[3] = {};
        double emissiveSum[3] = {};
        buildTable(s.ranges, s.rangeCalls, false, s.hitTable, casterSum);
        buildTable(s.emissiveRanges, s.emissiveCalls, true, s.emissiveTable, emissiveSum);

        // The dome is world-aligned in its model space with up along -Y (heights grow downward), so a triangle's elevation needs no transform.
        thread_local std::vector<rs64lights::SkySample> skySamples;
        skySamples.clear();
        for (size_t r = 0; (r < s.skyRanges.size()) && (r < s.skyCalls.size()); ++r) {
            rs64lights::SkySample sample;
            drawColor(s.skyRanges[r], s.skyCalls[r], sample.rgb);
            for (uint32_t t = 0; t < s.skyRanges[r].triangleCount; ++t) {
                float c[3] = {};
                bool ok = true;
                for (uint32_t k = 0; (k < 3) && ok; ++k) {
                    const size_t f = size_t(s.skyRanges[r].start) + t * 3 + k;
                    const size_t v = (f < fi.size()) ? fi[f] : SIZE_MAX;
                    ok = (v != SIZE_MAX) && (v * 3 + 2 < pos.size());
                    for (int j = 0; (j < 3) && ok; ++j) {
                        c[j] += pos[v * 3 + j];
                    }
                }

                const float len = std::sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]);
                if (ok && (len > 0.0f)) {
                    sample.elevation = -c[1] / len;
                    skySamples.push_back(sample);
                }
            }
        }

        // The dome as a sphere in camera space: its draw's translation and its mean vertex distance (model space) times the draw's scale. The game slides
        // the dome down as the camera climbs, so a reflected direction meets it at a different point than a direction at infinity would.
        double domeLen = 0.0;
        size_t domeVerts = 0;
        for (size_t r = 0; r < s.skyRanges.size(); ++r) {
            for (uint32_t f = 0; f < s.skyRanges[r].triangleCount * 3; ++f) {
                const size_t fx = size_t(s.skyRanges[r].start) + f;
                const size_t v = (fx < fi.size()) ? fi[fx] : SIZE_MAX;
                if ((v != SIZE_MAX) && (v * 3 + 2 < pos.size())) {
                    domeLen += std::sqrt(double(pos[v * 3]) * pos[v * 3] + double(pos[v * 3 + 1]) * pos[v * 3 + 1] + double(pos[v * 3 + 2]) * pos[v * 3 + 2]);
                    ++domeVerts;
                }
            }
        }

        s.skyDome = hlslpp::float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (s.skyWorldValid && (domeVerts > 0)) {
            const float radius = float(domeLen / double(domeVerts)) * rs64lights::transformScale(s.skyWorld);
            s.skyDome = hlslpp::float4(s.skyWorld.m[3][0], s.skyWorld.m[3][1], s.skyWorld.m[3][2], radius);
        }

        float skyH[3] = {}, skyZ[3] = {};
        const bool sky = rs64lights::skyBands(skySamples.data(), skySamples.size(), skyH, skyZ);
        // The game culls the dome while the camera looks down (a death spiral): the last frame's bands keep the water reflecting the sky.
        if (sky || (s.skyZenith.w <= 0.0f) || !s.terrainActive) {
            s.skyHorizon = hlslpp::float4(skyH[0], skyH[1], skyH[2], sky ? 1.0f : 0.0f);
            s.skyZenith = hlslpp::float4(skyZ[0], skyZ[1], skyZ[2], sky ? 1.0f : 0.0f);
        }
        s.emissiveIndices.clear();
        rs64lights::gatherCasterIndices(fi.data(), fi.size(), s.emissiveRanges.data(), s.emissiveRanges.size(), s.emissiveIndices);
        const double n = double(std::max<size_t>(s.hitTable.size(), 1));
        s.averageColor = rs64lights::packHitColor(float(casterSum[0] / n), float(casterSum[1] / n), float(casterSum[2] / n), false);
        if (rs64lights::config().log) {
            static uint32_t s_n = 0;
            if ((++s_n % 120) < 4) {
                std::fprintf(stderr, "[rt-hit] casters=%zu emissive=%zu cutouts=%zu avg=%08X terrain=%08X:", s.hitTable.size(), s.emissiveTable.size(), s.cutoutTable.size(), s.averageColor, s.terrainColor);
                for (size_t i = 0; (i < s.hitTable.size()) && (i < 48); ++i) {
                    const uint32_t call = (i < s.rangeCalls.size()) ? s.rangeCalls[i] : 0;
                    const bool tex = (call < callTex.size()) && ((uint32_t((float)callTex[call].w) & 1u) != 0);
                    std::fprintf(stderr, " %u:%08X%s(c%u)", s.hitTable[i].firstTriangle, s.hitTable[i].rgba, tex ? "t" : "", call);
                }
                std::fprintf(stderr, " | emissive:");
                for (size_t i = 0; (i < s.emissiveTable.size()) && (i < 16); ++i) {
                    const uint32_t call = (i < s.emissiveCalls.size()) ? s.emissiveCalls[i] : 0;
                    std::fprintf(stderr, " %u:%08X(c%u,%ut)", s.emissiveTable[i].firstTriangle, s.emissiveTable[i].rgba, call, (i < s.emissiveRanges.size()) ? s.emissiveRanges[i].triangleCount : 0);
                }
                std::fprintf(stderr, "\n");
            }
        }
        if (!terrainCalls.empty()) {
            double t[3] = {};
            for (const auto &tc : terrainCalls) {
                float c[3];
                drawColor(tc.first, tc.second, c);
                for (int k = 0; k < 3; ++k) {
                    t[k] += c[k];
                }
            }
            const double m = double(terrainCalls.size());
            s.terrainColor = rs64lights::packHitColor(float(t[0] / m), float(t[1] / m), float(t[2] / m), false);
        }
    }

    // Soft-shadow mask (shadow, blur radius) and ping textures at the colour target's size; recreated when it changes.
    void FramebufferRenderer::ensureRS64SoftTargets(RenderWorker *worker, Framebuffer &framebuffer, uint32_t width, uint32_t height, bool halfTrace) {
        if (framebuffer.rs64.softMask && (framebuffer.rs64.softW == width) && (framebuffer.rs64.softH == height) && (framebuffer.rs64.softHalf == halfTrace)) {
            return;
        }
        const uint32_t traceW = halfTrace ? ((width + 1) / 2) : width;
        const uint32_t traceH = halfTrace ? ((height + 1) / 2) : height;
        framebuffer.rs64.softHalf = halfTrace;
        framebuffer.rs64.softMaskFb.reset();
        framebuffer.rs64.softPingFb.reset();
        framebuffer.rs64.softMask = worker->device->createTexture(RenderTextureDesc::ColorTarget(traceW, traceH, RenderFormat::R16G16B16A16_FLOAT));
        framebuffer.rs64.softPing = worker->device->createTexture(RenderTextureDesc::ColorTarget(width, height, RenderFormat::R16G16B16A16_FLOAT));
        framebuffer.rs64.softIndirect = worker->device->createTexture(RenderTextureDesc::ColorTarget(traceW, traceH, RenderFormat::R16G16B16A16_FLOAT));
        framebuffer.rs64.softIndirectPing = worker->device->createTexture(RenderTextureDesc::ColorTarget(width, height, RenderFormat::R16G16B16A16_FLOAT));
        framebuffer.rs64.softMirror = worker->device->createTexture(RenderTextureDesc::ColorTarget(traceW, traceH, RenderFormat::R32G32B32A32_FLOAT));
        const RenderTexture *maskTargets[] = { framebuffer.rs64.softMask.get(), framebuffer.rs64.softIndirect.get(), framebuffer.rs64.softMirror.get() };
        const RenderTexture *pingTargets[] = { framebuffer.rs64.softPing.get(), framebuffer.rs64.softIndirectPing.get() };
        framebuffer.rs64.softMaskFb = worker->device->createFramebuffer(RenderFramebufferDesc(maskTargets, 3));
        framebuffer.rs64.softPingFb = worker->device->createFramebuffer(RenderFramebufferDesc(pingTargets, 2));
        for (uint32_t k = 0; k < 2; k++) {
            framebuffer.rs64.histFb[k].reset();
            framebuffer.rs64.histMask[k] = worker->device->createTexture(RenderTextureDesc::ColorTarget(width, height, RenderFormat::R16G16B16A16_FLOAT));
            framebuffer.rs64.histIndirect[k] = worker->device->createTexture(RenderTextureDesc::ColorTarget(width, height, RenderFormat::R16G16B16A16_FLOAT));
            const RenderTexture *histTargets[] = { framebuffer.rs64.histMask[k].get(), framebuffer.rs64.histIndirect[k].get() };
            framebuffer.rs64.histFb[k] = worker->device->createFramebuffer(RenderFramebufferDesc(histTargets, 2));
        }
        framebuffer.rs64.histValid = false;
        framebuffer.rs64.softW = width;
        framebuffer.rs64.softH = height;
    }

    void FramebufferRenderer::recordRS64Shadows(RenderWorker *worker, const Framebuffer &framebuffer, bool &depthState) {
        const RenderTargetDrawCall &targetDrawCall = framebuffer.renderTargetDrawCall;
        submitDepthAccess(worker, targetDrawCall.fbStorage, true, depthState);
        framebuffer.rs64.shadowsSet->setAccelerationStructure(framebuffer.rs64.shadowsSet->gScene, rs64Shadow.tlas.get());
        {
            // History slots are placed when the scene is built, after the fill.
            uint8_t *bytes = static_cast<uint8_t *>(framebuffer.rs64.shadowsBuffer->map());
            memcpy(bytes + offsetof(interop::RS64ShadowsCB, historyCam), rs64Shadow.historyCam, sizeof(rs64Shadow.historyCam));
            // A new view (the cockpit) skips the history slots: earlier frames' world casters sweep past it and flickered across the dash.
            const uint32_t historyMask = framebuffer.rs64.newView ? 0u : rs64Shadow.historyMask;
            memcpy(bytes + offsetof(interop::RS64ShadowsCB, historyMask), &historyMask, sizeof(uint32_t));
            const uint32_t hitParams[4] = { rs64Shadow.hitCount, rs64Shadow.emissiveCount, rs64Shadow.terrainColor, rs64Shadow.averageColor };
            memcpy(bytes + offsetof(interop::RS64ShadowsCB, hitParams), hitParams, sizeof(hitParams));
            framebuffer.rs64.shadowsBuffer->unmap();
        }
        const RenderViewport &v = framebuffer.viewport;
        if (framebuffer.rs64.shadowsSoft) {
            // Trace into the mask, blur H into ping, then blur V and darken the colour target (left bound as colorWriteDepthRead, as submitDepthAccess set it).
            const RenderRect sc(int32_t(v.x), int32_t(v.y), int32_t(v.x + v.width), int32_t(v.y + v.height));
            RenderTexture *mask = framebuffer.rs64.softMask.get();
            RenderTexture *ping = framebuffer.rs64.softPing.get();
            RenderTexture *ind = framebuffer.rs64.softIndirect.get();
            RenderTexture *indPing = framebuffer.rs64.softIndirectPing.get();
            // The trace reads the colour target for screen-space water reflections; it is written again by the compose.
            RenderTexture *color = targetDrawCall.fbStorage->colorTarget->getResolvedTexture();
            if (!framebuffer.rs64.reuseTrace) {
                rs64TimingMark(worker, RS64TimeTrace);
                const RenderTextureBarrier toMask[] = {
                    RenderTextureBarrier(mask, RenderTextureLayout::COLOR_WRITE), RenderTextureBarrier(ind, RenderTextureLayout::COLOR_WRITE), RenderTextureBarrier(color, RenderTextureLayout::SHADER_READ),
                    RenderTextureBarrier(framebuffer.rs64.softMirror.get(), RenderTextureLayout::COLOR_WRITE)
                };
                worker->commandList->barriers(RenderBarrierStage::GRAPHICS, toMask, uint32_t(std::size(toMask)));
                worker->commandList->setFramebuffer(framebuffer.rs64.softMaskFb.get());
                worker->commandList->clearColor(0, RenderColor(0.0f, 0.0f, 0.0f, 0.0f));
                worker->commandList->clearColor(1, RenderColor(0.0f, 0.0f, 0.0f, 0.0f));
                worker->commandList->clearColor(2, RenderColor(0.0f, 0.0f, 0.0f, 0.0f));
                if (framebuffer.rs64.softHalf) {
                    worker->commandList->setViewports(RenderViewport(v.x / 2.0f, v.y / 2.0f, v.width / 2.0f, v.height / 2.0f, v.minDepth, v.maxDepth));
                    worker->commandList->setScissors(RenderRect(int32_t(v.x) / 2, int32_t(v.y) / 2, (int32_t(v.x + v.width) + 1) / 2, (int32_t(v.y + v.height) + 1) / 2));
                }
                else {
                    worker->commandList->setViewports(v);
                    worker->commandList->setScissors(sc);
                }
                worker->commandList->setPipeline(shaderLibrary->rs64ShadowsSoft.pipeline.get());
                worker->commandList->setGraphicsPipelineLayout(shaderLibrary->rs64ShadowsSoft.pipelineLayout.get());
                bindRS64Traced(worker, *framebuffer.rs64.shadowsSet, framebuffer.rs64.shadowsBuffer.get(), offsetof(interop::RS64ShadowsCB, cutoutDrawCount), false);
                worker->commandList->setVertexBuffers(0, nullptr, 0, nullptr);
                worker->commandList->drawInstanced(3, 1, 0, 0);
                rs64TimingMark(worker, RS64TimeTrace);
            }

            // Accumulate AO and indirect light into this frame's history set, which the H blur then reads.
            if (framebuffer.rs64.temporalOn) {
                rs64TimingMark(worker, RS64TimeTemporal);
                const uint32_t w = framebuffer.rs64.histTarget;
                RenderTexture *histMask = framebuffer.rs64.histMask[w].get();
                RenderTexture *histInd = framebuffer.rs64.histIndirect[w].get();
                const RenderTextureBarrier toHist[] = {
                    RenderTextureBarrier(mask, RenderTextureLayout::SHADER_READ), RenderTextureBarrier(ind, RenderTextureLayout::SHADER_READ),
                    RenderTextureBarrier(framebuffer.rs64.histMask[w ^ 1].get(), RenderTextureLayout::SHADER_READ), RenderTextureBarrier(framebuffer.rs64.histIndirect[w ^ 1].get(), RenderTextureLayout::SHADER_READ),
                    RenderTextureBarrier(histMask, RenderTextureLayout::COLOR_WRITE), RenderTextureBarrier(histInd, RenderTextureLayout::COLOR_WRITE)
                };
                worker->commandList->barriers(RenderBarrierStage::GRAPHICS, toHist, uint32_t(std::size(toHist)));
                worker->commandList->setFramebuffer(framebuffer.rs64.histFb[w].get());
                worker->commandList->clearColor(0, RenderColor(0.0f, 0.0f, 0.0f, 0.0f));
                worker->commandList->clearColor(1, RenderColor(0.0f, 0.0f, 0.0f, 0.0f));
                worker->commandList->setViewports(v);
                worker->commandList->setScissors(sc);
                worker->commandList->setPipeline(shaderLibrary->rs64ShadowTemporal.pipeline.get());
                worker->commandList->setGraphicsPipelineLayout(shaderLibrary->rs64ShadowTemporal.pipelineLayout.get());
                worker->commandList->setGraphicsDescriptorSet(framebuffer.rs64.temporalSet->get(), 0);
                worker->commandList->drawInstanced(3, 1, 0, 0);
                mask = histMask;
                ind = histInd;
                rs64TimingMark(worker, RS64TimeTemporal);
            }

            rs64TimingMark(worker, RS64TimeBlur);
            const RenderTextureBarrier toPing[] = {
                RenderTextureBarrier(mask, RenderTextureLayout::SHADER_READ), RenderTextureBarrier(ind, RenderTextureLayout::SHADER_READ),
                RenderTextureBarrier(ping, RenderTextureLayout::COLOR_WRITE), RenderTextureBarrier(indPing, RenderTextureLayout::COLOR_WRITE)
            };
            worker->commandList->barriers(RenderBarrierStage::GRAPHICS, toPing, uint32_t(std::size(toPing)));
            worker->commandList->setFramebuffer(framebuffer.rs64.softPingFb.get());
            worker->commandList->clearColor(0, RenderColor(0.0f, 0.0f, 0.0f, 0.0f));
            worker->commandList->clearColor(1, RenderColor(0.0f, 0.0f, 0.0f, 0.0f));
            worker->commandList->setViewports(v);
            worker->commandList->setScissors(sc);
            worker->commandList->setPipeline(shaderLibrary->rs64ShadowBlurH.pipeline.get());
            worker->commandList->setGraphicsPipelineLayout(shaderLibrary->rs64ShadowBlurH.pipelineLayout.get());
            worker->commandList->setGraphicsDescriptorSet(framebuffer.rs64.blurHSet->get(), 0);
            worker->commandList->drawInstanced(3, 1, 0, 0);

            const RenderTextureBarrier toCompose[] = {
                RenderTextureBarrier(ping, RenderTextureLayout::SHADER_READ), RenderTextureBarrier(indPing, RenderTextureLayout::SHADER_READ),
                RenderTextureBarrier(targetDrawCall.fbStorage->colorTarget->texture.get(), RenderTextureLayout::COLOR_WRITE)
            };
            worker->commandList->barriers(RenderBarrierStage::GRAPHICS, toCompose, uint32_t(std::size(toCompose)));
            worker->commandList->setFramebuffer(targetDrawCall.fbStorage->colorWriteDepthRead.get());
            worker->commandList->setViewports(v);
            worker->commandList->setScissors(sc);
            const ShaderRecord &vrec = framebuffer.rs64.shadowsDebug ? shaderLibrary->rs64ShadowBlurVDebug : shaderLibrary->rs64ShadowBlurV;
            worker->commandList->setPipeline(vrec.pipeline.get());
            worker->commandList->setGraphicsPipelineLayout(vrec.pipelineLayout.get());
            worker->commandList->setGraphicsDescriptorSet(framebuffer.rs64.blurVSet->get(), 0);
            worker->commandList->drawInstanced(3, 1, 0, 0);

            if (!framebuffer.rs64.shadowsDebug) {
                worker->commandList->setPipeline(shaderLibrary->rs64ShadowBlurWater.pipeline.get());
                worker->commandList->setGraphicsPipelineLayout(shaderLibrary->rs64ShadowBlurWater.pipelineLayout.get());
                worker->commandList->setGraphicsDescriptorSet(framebuffer.rs64.blurVSet->get(), 0);
                worker->commandList->drawInstanced(3, 1, 0, 0);
            }

            rs64TimingMark(worker, RS64TimeBlur);
            return;
        }
        const ShaderRecord &record = framebuffer.rs64.shadowsDebug ? shaderLibrary->rs64ShadowsDebug : shaderLibrary->rs64Shadows;
        worker->commandList->setViewports(v);
        worker->commandList->setScissors(RenderRect(int32_t(v.x), int32_t(v.y), int32_t(v.x + v.width), int32_t(v.y + v.height)));
        worker->commandList->setPipeline(record.pipeline.get());
        worker->commandList->setGraphicsPipelineLayout(record.pipelineLayout.get());
        bindRS64Traced(worker, *framebuffer.rs64.shadowsSet, framebuffer.rs64.shadowsBuffer.get(), offsetof(interop::RS64ShadowsCB, cutoutDrawCount), false);
        worker->commandList->setVertexBuffers(0, nullptr, 0, nullptr);
        worker->commandList->drawInstanced(3, 1, 0, 0);
    }

    // Traced passes: this frame's cutout table, indices and tex-coords into set 3 (placeholders when there are none), the count patched into the
    // pass's constant buffer at record time (the fills run before recordSetup builds the table), then sets 0-2 as raster binds them.
    void FramebufferRenderer::bindRS64Traced(RenderWorker *worker, RS64TracedDescriptorSet &set, RenderBuffer *paramsBuffer, uint64_t countOffset, bool countAsFloat) {
        RS64ShadowScene &s = rs64Shadow;
        if (!rs64CutoutPlaceholder) {
            rs64CutoutPlaceholder = worker->device->createBuffer(RenderBufferDesc::UploadBuffer(64, RenderBufferFlag::STORAGE));
            memset(rs64CutoutPlaceholder->map(), 0, 64);
            rs64CutoutPlaceholder->unmap();
        }
        const bool have = (s.cutoutDrawCount > 0) && (s.texCoords != nullptr);
        const uint32_t count = have ? s.cutoutDrawCount : 0;
        if (have) {
            set.setBuffer(set.gCutoutDraws, s.cutoutTableGpu.get(), s.cutoutTableGpuCapacity, RenderBufferStructuredView(sizeof(rs64lights::CutoutEntry)));
            set.setBuffer(set.gCutoutIndices, s.cutoutIndexGpu.get(), s.cutoutIndexGpuCapacity);
            set.setBuffer(set.gTexCoords, const_cast<RenderBuffer *>(s.texCoords), s.texCoordsSize);
        }
        else {
            set.setBuffer(set.gCutoutDraws, rs64CutoutPlaceholder.get(), 64, RenderBufferStructuredView(sizeof(rs64lights::CutoutEntry)));
            set.setBuffer(set.gCutoutIndices, rs64CutoutPlaceholder.get(), 64);
            set.setBuffer(set.gTexCoords, rs64CutoutPlaceholder.get(), 64);
        }
        if (have && (s.shadedCols != nullptr)) {
            set.setBuffer(set.gShadedCols, const_cast<RenderBuffer *>(s.shadedCols), s.shadedColsSize);
        }
        else {
            set.setBuffer(set.gShadedCols, rs64CutoutPlaceholder.get(), 64);
        }
        const RenderBufferStructuredView hitView(sizeof(rs64lights::HitColorEntry));
        if ((s.hitCount > 0) && s.hitTableGpu) {
            set.setBuffer(set.gHitColors, s.hitTableGpu.get(), s.hitTableGpuCapacity, hitView);
        }
        else {
            set.setBuffer(set.gHitColors, rs64CutoutPlaceholder.get(), 64, hitView);
        }
        if ((s.emissiveCount > 0) && s.emissiveTableGpu) {
            set.setBuffer(set.gEmissiveColors, s.emissiveTableGpu.get(), s.emissiveTableGpuCapacity, hitView);
        }
        else {
            set.setBuffer(set.gEmissiveColors, rs64CutoutPlaceholder.get(), 64, hitView);
        }
        if (s.terrainBlas && s.terrainNormalGpu) {
            set.setBuffer(set.gTerrainIndices, s.terrainIndexGpu.get(), s.terrainIndexBytes);
            set.setBuffer(set.gTerrainNormals, s.terrainNormalGpu.get(), s.terrainNormalBytes);
        }
        else {
            set.setBuffer(set.gTerrainIndices, rs64CutoutPlaceholder.get(), 64);
            set.setBuffer(set.gTerrainNormals, rs64CutoutPlaceholder.get(), 64);
        }

        // The placeholder's zero header means no table: terrain hits keep the level's average colour.
        if (s.terrainBlas && s.terrainColorGpu) {
            set.setBuffer(set.gTerrainColors, s.terrainColorGpu.get(), s.terrainColorBytes);
        }
        else {
            set.setBuffer(set.gTerrainColors, rs64CutoutPlaceholder.get(), 64);
        }
        uint8_t *bytes = static_cast<uint8_t *>(paramsBuffer->map());
        if (countAsFloat) {
            const float f = float(count);
            memcpy(bytes + countOffset, &f, sizeof(f));
        }
        else {
            memcpy(bytes + countOffset, &count, sizeof(count));
        }
        paramsBuffer->unmap();
        worker->commandList->setGraphicsDescriptorSet(descCommonSet->get(), 0);
        worker->commandList->setGraphicsDescriptorSet(descTextureSet->get(), 1);
        worker->commandList->setGraphicsDescriptorSet(descTextureSet->get(), 2);
        worker->commandList->setGraphicsDescriptorSet(set.get(), 3);
    }

    bool FramebufferRenderer::fillRS64FogShafts(const DrawParams &p, Framebuffer &framebuffer, uint32_t transformsIndex, hlslpp::float2 screenScale, hlslpp::float2 screenOffset, const rs64lights::Sun &sun, const rs64lights::FogParams &fog, bool fogOn, bool hazeOn) {
        rs64lights::Mat4 inv;
        rs64lights::Mat4 viewProj;
        if (!sun.valid || !framebuffer.rs64.fogSet || !rs64ViewProj(p, transformsIndex, viewProj) || !rs64ViewFromScreen(p, transformsIndex, screenScale, screenOffset, inv)) {
            return false;
        }

        const rs64lights::Config &cfg = rs64lights::config();
        interop::RS64FogShaftsCB cb = {};
        memcpy(&cb.viewFromScreen, &inv, sizeof(inv));
        memcpy(&cb.viewProj, &viewProj, sizeof(viewProj));
        cb.viewport = { framebuffer.viewport.x, framebuffer.viewport.y, framebuffer.viewport.width, framebuffer.viewport.height };
        cb.sunDir = { sun.dir[0], sun.dir[1], sun.dir[2], 0.0f };
        cb.fogColor = { fog.color[0], fog.color[1], fog.color[2], (rs64_vr_active() != 0) ? 1.0f : 0.0f };
        cb.debugMode = static_cast<uint32_t>(cfg.fogShaftsDebug);
        cb.steps = static_cast<uint32_t>(cfg.fogShaftsSteps);
        cb.strength = cfg.fogShaftsStrength;
        cb.fogMul = fog.mul;
        cb.fogOffset = fog.offset;
        cb.fogTail = fog.tailEnd;
        cb.tMin = cfg.shadowTMin;
        cb.tMax = cfg.shadowTMax;
        cb.terrainTMinScale = cfg.shadowTerrainTMin;
        cb.skyDepth = rs64lights::fogSkyDepth();
        cb.visBias = cfg.shadowNormalBias;

        const rs64lights::Haze &haze = p.curWorkload->rs64Haze;
        const float *hazeColor = (haze.color[0] >= 0.0f) ? haze.color : sun.color;
        cb.hazeColor = { hazeColor[0] * cfg.hazeStrength, hazeColor[1] * cfg.hazeStrength, hazeColor[2] * cfg.hazeStrength, 0.0f };
        cb.hazeRange = haze.range;
        cb.hazeDensity = hazeOn ? (-std::log(1.0f - haze.amount) / haze.range) : 0.0f;

        void *bytes = framebuffer.rs64.fogBuffer->map();
        memcpy(bytes, &cb, sizeof(cb));
        framebuffer.rs64.fogBuffer->unmap();
        framebuffer.rs64.fogSet->setBuffer(framebuffer.rs64.fogSet->gParams, framebuffer.rs64.fogBuffer.get(), sizeof(interop::RS64FogShaftsCB));
        bindRS64Depth(p, framebuffer, *framebuffer.rs64.fogSet, framebuffer.rs64.fogSet->gDepth);
        if (hazeOn) {
            const RenderTarget *ct = p.fbStorage->colorTarget;
            const uint32_t hw = (ct->width + 1) / 2;
            const uint32_t hh = (ct->height + 1) / 2;
            framebuffer.rs64.hazeHalf.ensure(p.worker->device, hw, hh, RenderFormat::R16G16B16A16_FLOAT);

            framebuffer.rs64.fogSet->setTexture(framebuffer.rs64.fogSet->gColor, framebuffer.rs64.hazeHalf.texture.get(), RenderTextureLayout::SHADER_READ);
        }
        else {
            bindRS64Depth(p, framebuffer, *framebuffer.rs64.fogSet, framebuffer.rs64.fogSet->gColor);
        }

        framebuffer.rs64.fogDebug = (cfg.fogShaftsDebug != 0);
        framebuffer.rs64.fogShade = fogOn;
        framebuffer.rs64.hazeOn = hazeOn;
        return true;
    }

    void FramebufferRenderer::prepareRS64MSAA(const DrawParams &p, Framebuffer &framebuffer) {
        const RenderTarget *dt = p.fbStorage->depthTarget;
        framebuffer.rs64.depthCopy.ensure(p.worker->device, dt->width, dt->height, RenderFormat::R32_FLOAT);

        framebuffer.rs64.depthResolveSet->setTexture(framebuffer.rs64.depthResolveSet->gInput, dt->texture.get(), RenderTextureLayout::DEPTH_READ, dt->textureView.get());
    }

    void FramebufferRenderer::bindRS64Depth(const DrawParams &p, const Framebuffer &framebuffer, RenderDescriptorSetBase &set, uint32_t slot) {
        if (framebuffer.rs64.msaa && framebuffer.rs64.depthCopy.texture) {
            set.setTexture(slot, framebuffer.rs64.depthCopy.texture.get(), RenderTextureLayout::SHADER_READ);
        }
        else {
            set.setTexture(slot, p.fbStorage->depthTarget->texture.get(), RenderTextureLayout::DEPTH_READ, p.fbStorage->depthTarget->textureView.get());
        }
    }

    // Copies sample 0 of the MSAA depth (and resolves the colour when a pass reads it) for the passes that follow, then rebinds the target read-only.
    void FramebufferRenderer::recordRS64MSAAResolve(RenderWorker *worker, const Framebuffer &framebuffer, bool resolveColor, bool &depthState) {
        RenderFramebufferStorage *fbStorage = framebuffer.renderTargetDrawCall.fbStorage;
        RenderTexture *copy = framebuffer.rs64.depthCopy.texture.get();
        if (copy == nullptr) {
            return;
        }

        submitDepthAccess(worker, fbStorage, true, depthState);
        worker->commandList->barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(copy, RenderTextureLayout::COLOR_WRITE));
        worker->commandList->setFramebuffer(framebuffer.rs64.depthCopy.framebuffer.get());
        worker->commandList->setViewports(RenderViewport(0.0f, 0.0f, float(framebuffer.rs64.depthCopy.width), float(framebuffer.rs64.depthCopy.height)));
        worker->commandList->setScissors(RenderRect(0, 0, int32_t(framebuffer.rs64.depthCopy.width), int32_t(framebuffer.rs64.depthCopy.height)));
        worker->commandList->setPipeline(shaderLibrary->rs64DepthResolve.pipeline.get());
        worker->commandList->setGraphicsPipelineLayout(shaderLibrary->rs64DepthResolve.pipelineLayout.get());
        worker->commandList->setGraphicsDescriptorSet(framebuffer.rs64.depthResolveSet->get(), 0);
        worker->commandList->setVertexBuffers(0, nullptr, 0, nullptr);
        worker->commandList->drawInstanced(3, 1, 0, 0);

        RenderTarget *colorTarget = fbStorage->colorTarget;
        if (resolveColor) {
            // Marked again after: the frame's own resolve must still run once the rest of the frame is drawn.
            colorTarget->markForResolve();
            colorTarget->resolveTarget(worker, shaderLibrary);
            colorTarget->markForResolve();
        }

        const RenderTextureBarrier toPasses[] = {
            RenderTextureBarrier(copy, RenderTextureLayout::SHADER_READ), RenderTextureBarrier(colorTarget->getResolvedTexture(), RenderTextureLayout::SHADER_READ),
            RenderTextureBarrier(colorTarget->texture.get(), RenderTextureLayout::COLOR_WRITE)
        };

        worker->commandList->barriers(RenderBarrierStage::GRAPHICS, toPasses, uint32_t(std::size(toPasses)));
        worker->commandList->setFramebuffer(fbStorage->colorWriteDepthRead.get());
    }

    // Horizon mirror: copies the colour target after the view's last draw, then swaps the sky missed water rays reflected for what is drawn there now.
    void FramebufferRenderer::recordRS64Mirror(RenderWorker *worker, const Framebuffer &framebuffer, bool &depthState) {

        const RenderTargetDrawCall &targetDrawCall = framebuffer.renderTargetDrawCall;
        if (framebuffer.rs64.msaa) {
            recordRS64MSAAResolve(worker, framebuffer, true, depthState);
        }

        RenderTexture *color = targetDrawCall.fbStorage->colorTarget->getResolvedTexture();
        RenderTexture *copy = framebuffer.rs64.mirrorLate.get();
        const RenderTextureBarrier toCopy[] = { RenderTextureBarrier(color, RenderTextureLayout::COPY_SOURCE), RenderTextureBarrier(copy, RenderTextureLayout::COPY_DEST) };

        worker->commandList->barriers(RenderBarrierStage::COPY, toCopy, uint32_t(std::size(toCopy)));
        worker->commandList->copyTexture(copy, color);

        const RenderTextureBarrier toDraw[] = {
            RenderTextureBarrier(targetDrawCall.fbStorage->colorTarget->texture.get(), RenderTextureLayout::COLOR_WRITE), RenderTextureBarrier(copy, RenderTextureLayout::SHADER_READ),
            RenderTextureBarrier(framebuffer.rs64.softMirror.get(), RenderTextureLayout::SHADER_READ)
        };

        worker->commandList->barriers(RenderBarrierStage::GRAPHICS, toDraw, uint32_t(std::size(toDraw)));
        submitDepthAccess(worker, targetDrawCall.fbStorage, true, depthState);
        worker->commandList->setFramebuffer(targetDrawCall.fbStorage->colorWriteDepthRead.get());

        const RenderViewport &v = framebuffer.viewport;

        worker->commandList->setViewports(v);
        worker->commandList->setScissors(RenderRect(int32_t(v.x), int32_t(v.y), int32_t(v.x + v.width), int32_t(v.y + v.height)));
        worker->commandList->setVertexBuffers(0, nullptr, 0, nullptr);

        for (const ShaderRecord *rec : { &shaderLibrary->rs64WaterMirror, &shaderLibrary->rs64WaterMirrorNeg }) {
            worker->commandList->setPipeline(rec->pipeline.get());
            worker->commandList->setGraphicsPipelineLayout(rec->pipelineLayout.get());
            worker->commandList->setGraphicsDescriptorSet(framebuffer.rs64.mirrorSet->get(), 0);
            worker->commandList->drawInstanced(3, 1, 0, 0);
        }
    }

    void FramebufferRenderer::recordRS64FogShafts(RenderWorker *worker, const Framebuffer &framebuffer, bool &depthState) {
        const RenderTargetDrawCall &targetDrawCall = framebuffer.renderTargetDrawCall;
        submitDepthAccess(worker, targetDrawCall.fbStorage, true, depthState);
        framebuffer.rs64.fogSet->setAccelerationStructure(framebuffer.rs64.fogSet->gScene, rs64Shadow.tlas.get());
        const RenderViewport &v = framebuffer.viewport;
        worker->commandList->setViewports(v);
        worker->commandList->setScissors(RenderRect(int32_t(v.x), int32_t(v.y), int32_t(v.x + v.width), int32_t(v.y + v.height)));
        worker->commandList->setVertexBuffers(0, nullptr, 0, nullptr);
        auto draw = [&](const ShaderRecord &record) {
            worker->commandList->setPipeline(record.pipeline.get());
            worker->commandList->setGraphicsPipelineLayout(record.pipelineLayout.get());
            bindRS64Traced(worker, *framebuffer.rs64.fogSet, framebuffer.rs64.fogBuffer.get(), offsetof(interop::RS64FogShaftsCB, cutoutDrawCount), false);
            worker->commandList->drawInstanced(3, 1, 0, 0);
        };

        if (framebuffer.rs64.fogShade) {
            draw(framebuffer.rs64.fogDebug ? shaderLibrary->rs64FogShaftsDebug : shaderLibrary->rs64FogShafts);
        }

        if (framebuffer.rs64.hazeOn && (framebuffer.rs64.hazeHalf.framebuffer != nullptr)) {
            RenderTexture *halfTex = framebuffer.rs64.hazeHalf.texture.get();
            const RenderTextureBarrier toHalf[] = { RenderTextureBarrier(halfTex, RenderTextureLayout::COLOR_WRITE) };
            worker->commandList->barriers(RenderBarrierStage::GRAPHICS, toHalf, uint32_t(std::size(toHalf)));
            worker->commandList->setFramebuffer(framebuffer.rs64.hazeHalf.framebuffer.get());
            worker->commandList->clearColor(0, RenderColor(0.0f, 0.0f, 0.0f, 0.0f));
            worker->commandList->setViewports(RenderViewport(v.x / 2.0f, v.y / 2.0f, v.width / 2.0f, v.height / 2.0f, v.minDepth, v.maxDepth));
            worker->commandList->setScissors(RenderRect(int32_t(v.x) / 2, int32_t(v.y) / 2, (int32_t(v.x + v.width) + 1) / 2, (int32_t(v.y + v.height) + 1) / 2));
            draw(shaderLibrary->rs64HazeShafts);

            RenderTexture *color = targetDrawCall.fbStorage->colorTarget->texture.get();
            const RenderTextureBarrier toUpsample[] = { RenderTextureBarrier(halfTex, RenderTextureLayout::SHADER_READ), RenderTextureBarrier(color, RenderTextureLayout::COLOR_WRITE) };
            worker->commandList->barriers(RenderBarrierStage::GRAPHICS, toUpsample, uint32_t(std::size(toUpsample)));
            worker->commandList->setFramebuffer(targetDrawCall.fbStorage->colorWriteDepthRead.get());
            worker->commandList->setViewports(v);
            worker->commandList->setScissors(RenderRect(int32_t(v.x), int32_t(v.y), int32_t(v.x + v.width), int32_t(v.y + v.height)));
            draw(shaderLibrary->rs64HazeUpsample);
        }
    }

    // A terrain tile as the game loads it: the bank entry's CI4 texels at TMEM 0 and its 16 RGBA5551 colours as palette 15 (each 4 times, as a
    // TLUT load stores them), with the game's load tile, so RT64 hashes and decodes it like the game's own upload (texture packs match).
    static const LoadTile RS64TerrainLoadTile = { G_IM_FMT_CI, G_IM_SIZ_4b, 4, 0, 15, 0, 0, 6, 6, 0, 0, 0, 0, 252, 252 };

    static uint64_t rs64TerrainTMEM(const uint8_t *entry, std::array<uint8_t, 4096> &tmem) {
        tmem.fill(0);
        memcpy(tmem.data(), entry, 2048);
        for (uint32_t i = 0; i < 16; ++i) {
            for (uint32_t k = 0; k < 4; ++k) {
                memcpy(&tmem[0xF80 + i * 8 + k * 2], entry + 2048 + i * 2, 2);
            }
        }

        return TMEMHasher::hash(tmem.data(), RS64TerrainLoadTile, 64, 64, G_TT_RGBA16, TMEMHasher::CurrentHashVersion);
    }

    // Texture packs for the terrain pass: a layer the loaded pack replaces goes to the texture cache under RT64's own hash (the cache
    // finds and streams the replacement), then each frame takes the replacement's texture index (the pass samples it in RS64TerrainPS).
    void FramebufferRenderer::rs64TerrainTextures(TextureCache *textureCache, uint64_t submissionFrame) {
        RS64TerrainGPU &g = rs64TerrainGpu;
        g.layerReplaced.assign(g.layerHash.size(), UINT32_MAX);
        if (textureSlots || !textureCacheReplacementMapEnabled || g.layerHash.empty()) {
            return;
        }

        const uint32_t generation = textureCache->replacementGeneration.load();
        if (generation != g.replacementGeneration) {
            g.replacementGeneration = generation;
            g.layersQueued = 0;
        }

        uint32_t index = 0;
        interop::float2 scale;
        interop::float3 dimensions;
        bool replaced = false, mipmaps = false, shifted = false;
        for (size_t l = g.layersQueued; l < g.layerHash.size(); ++l) {
            if (textureCache->hasReplacement(g.layerHash[l]) && !textureCache->useTexture(g.layerHash[l], submissionFrame, index)) {
                textureCache->queueGPUUploadTMEM(g.layerHash[l], submissionFrame, g.layerTMEM[l].data(), int(g.layerTMEM[l].size()), 64, 64, G_TT_RGBA16, RS64TerrainLoadTile, true);
            }
        }

        g.layersQueued = g.layerHash.size();

        for (size_t l = 0; l < g.layerHash.size(); ++l) {
            if (textureCache->useTexture(g.layerHash[l], submissionFrame, index, scale, dimensions, replaced, mipmaps, shifted) && replaced) {
                g.layerReplaced[l] = index;
            }
        }
    }

    // Uploads the map for the terrain pass when its version changes: the (4W+1)x(4H+1) height and colour lattice, each cell's texture layer and
    // the tile textures (ROM level/tex entries the level's lv_X_TEX table names; one white layer when the level has no table).
    void FramebufferRenderer::recordRS64TerrainSetup(RenderWorker *worker) {
        const rs64lights::TerrainMap &m = *rs64TerrainMap;
        if ((rs64TerrainGpu.version == m.version) || (m.width == 0) || (m.height == 0) || (m.colors.size() < size_t(m.width) * m.height * 75) || (m.texIds.size() < size_t(m.width) * m.height)) {
            return;
        }

        const uint32_t latticeW = m.width * 4 + 1, latticeH = m.height * 4 + 1;
        thread_local std::vector<int32_t> heights;
        thread_local std::vector<uint32_t> colors, cellLayers, texels;
        heights.assign(size_t(latticeW) * latticeH, 0);
        colors.assign(size_t(latticeW) * latticeH, 0xFFFFFFFFu);
        for (uint32_t row = 0; row < m.height; ++row) {
            for (uint32_t col = 0; col < m.width; ++col) {
                const size_t c = size_t(row) * m.width + col;
                for (uint32_t k = 0; k < 25; ++k) {
                    const size_t l = size_t(row * 4 + k / 5) * latticeW + col * 4 + k % 5;
                    const uint8_t *rgb = &m.colors[(c * 25 + k) * 3];
                    heights[l] = int32_t(m.heights[c * 25 + k]) * 16;
                    colors[l] = uint32_t(rgb[0]) | (uint32_t(rgb[1]) << 8) | (uint32_t(rgb[2]) << 16) | 0xFF000000u;
                }
            }
        }

        const uint32_t chunksX = (m.width + RS64TerrainChunkCells - 1) / RS64TerrainChunkCells, chunksZ = (m.height + RS64TerrainChunkCells - 1) / RS64TerrainChunkCells;
        rs64TerrainGpu.chunkMinH.assign(size_t(chunksX) * chunksZ, INT32_MAX);
        rs64TerrainGpu.chunkMaxH.assign(size_t(chunksX) * chunksZ, INT32_MIN);
        for (uint32_t row = 0; row < latticeH; ++row) {
            for (uint32_t col = 0; col < latticeW; ++col) {
                const size_t k = size_t(std::min(row / (4 * RS64TerrainChunkCells), chunksZ - 1)) * chunksX + std::min(col / (4 * RS64TerrainChunkCells), chunksX - 1);
                const int32_t h = heights[size_t(row) * latticeW + col];
                rs64TerrainGpu.chunkMinH[k] = std::min(rs64TerrainGpu.chunkMinH[k], h);
                rs64TerrainGpu.chunkMaxH[k] = std::max(rs64TerrainGpu.chunkMaxH[k], h);
            }
        }

        // Low 16 bits: texture layer; bits 16-17: water kind (0 ground, 1 water, 2 shore); bits 18+: the water draw's layer.
        cellLayers.assign(size_t(m.width) * m.height, 0);
        for (size_t c = 0; (c < cellLayers.size()) && (c < m.waterKind.size()); ++c) {
            cellLayers[c] = uint32_t(m.waterKind[c]) << 16;
        }

        // A runtime edit of the map this setup built last (its baseVersion) keeps the texture layers: only a texture new to the map decodes.
        const bool update = (m.baseVersion != 0) && (m.baseVersion == rs64TerrainGpu.version);
        if (!update) {
            texels.clear();
            rs64TerrainGpu.anims.clear();
            rs64TerrainGpu.layerOf.clear();
            rs64TerrainGpu.layerTMEM.clear();
            rs64TerrainGpu.layerHash.clear();
            rs64TerrainGpu.layerReplaced.clear();
            rs64TerrainGpu.layersQueued = 0;
        }

        const uint8_t *bank = nullptr, *ids = nullptr;
        uint32_t bankSize = 0, idsSize = 0;
        if (rs64_terrain_textures(m.level, &bank, &bankSize, &ids, &idsSize)) {
            // Keyed by bank, or 0x10000 | id for an animated id so its frames never move a still texture that shares the first bank.
            std::unordered_map<uint32_t, uint32_t> &layerOf = rs64TerrainGpu.layerOf;
            auto layer = [&](uint32_t key, uint32_t b) {
                auto it = layerOf.find(key);
                if (it == layerOf.end()) {
                    it = layerOf.emplace(key, uint32_t(texels.size() / 4096)).first;
                    texels.resize(texels.size() + 4096);
                    rs64lights::decodeTerrainTexture(bank + size_t(b) * rs64lights::TerrainTexBytes, &texels[texels.size() - 4096]);
                    rs64TerrainGpu.layerHash.push_back(rs64TerrainTMEM(bank + size_t(b) * rs64lights::TerrainTexBytes, rs64TerrainGpu.layerTMEM.emplace_back()));
                }

                return it->second;
            };

            const uint32_t usedIds = m.texIds.empty() ? 0 : (*std::max_element(m.texIds.begin(), m.texIds.end()) + 1);
            std::unordered_map<uint32_t, const rs64lights::TerrainTexAnim *> animOf;
            const std::vector<rs64lights::TerrainTexAnim> anims = rs64lights::terrainTextureAnims(ids, idsSize, bankSize / rs64lights::TerrainTexBytes, usedIds);
            for (const rs64lights::TerrainTexAnim &a : anims) {
                animOf[a.id] = &a;
            }

            for (size_t c = 0; c < cellLayers.size(); ++c) {
                const uint32_t t = m.texIds[c];
                if ((t * 4 + 2) > idsSize) {
                    continue;
                }

                const uint32_t b = (uint32_t(ids[t * 4]) << 8) | ids[t * 4 + 1];
                if ((b + 1) * rs64lights::TerrainTexBytes > bankSize) {
                    continue;
                }

                const auto anim = animOf.find(t);
                if (anim == animOf.end()) {
                    cellLayers[c] |= layer(b, b);
                    continue;
                }

                const bool fresh = !layerOf.count(0x10000u | t);
                const uint32_t l = layer(0x10000u | t, b);
                cellLayers[c] |= l;
                if (fresh) {
                    RS64TerrainGPU::Anim &ga = rs64TerrainGpu.anims.emplace_back();
                    ga.layer = l;
                    ga.fps = anim->second->fps;
                    for (uint32_t fb : anim->second->banks) {
                        ga.frames.push_back(layer(fb, fb));
                    }
                }
            }
        }

        rs64lights::terrainWaterLayers(cellLayers, m.waterKind, int32_t(m.width), int32_t(m.height));

        if (texels.empty()) {
            texels.assign(4096, 0xFFFFFFFFu);
        }

        RenderDevice *device = worker->device;
        RS64TerrainGPU &g = rs64TerrainGpu;
        g.heightsBytes = heights.size() * sizeof(int32_t);
        g.colorsBytes = colors.size() * sizeof(uint32_t);
        g.cellLayersBytes = cellLayers.size() * sizeof(uint32_t);
        g.texelsBytes = texels.size() * sizeof(uint32_t);
        g.layers = uint32_t(texels.size() / 4096);
        const uint64_t offsets[3] = { 0, g.heightsBytes, g.heightsBytes + g.cellLayersBytes };
        g.staging = device->createBuffer(RenderBufferDesc::UploadBuffer(offsets[2] + g.texelsBytes));
        uint8_t *dst = reinterpret_cast<uint8_t *>(g.staging->map());
        memcpy(dst + offsets[0], heights.data(), g.heightsBytes);
        memcpy(dst + offsets[1], cellLayers.data(), g.cellLayersBytes);
        memcpy(dst + offsets[2], texels.data(), g.texelsBytes);
        g.staging->unmap();
        g.heights = device->createBuffer(RenderBufferDesc::DefaultBuffer(g.heightsBytes, RenderBufferFlag::STORAGE));
        g.cellLayers = device->createBuffer(RenderBufferDesc::DefaultBuffer(g.cellLayersBytes, RenderBufferFlag::STORAGE));
        g.texels = device->createBuffer(RenderBufferDesc::DefaultBuffer(g.texelsBytes, RenderBufferFlag::STORAGE));

        // Colours stay CPU-writable: each frame takes the game's own colours of the cells it drew (recordRS64TerrainFrame).
        g.colors = device->createBuffer(RenderBufferDesc::UploadBuffer(g.colorsBytes, RenderBufferFlag::STORAGE));
        memcpy(g.colors->map(), colors.data(), g.colorsBytes);
        g.colors->unmap();
        RenderBuffer *bufs[3] = { g.heights.get(), g.cellLayers.get(), g.texels.get() };
        const uint64_t sizes[3] = { g.heightsBytes, g.cellLayersBytes, g.texelsBytes };
        RenderBufferBarrier copyIn[3], copyOut[3];
        for (int i = 0; i < 3; ++i) {
            copyIn[i] = RenderBufferBarrier(bufs[i], RenderBufferAccess::WRITE);
            copyOut[i] = RenderBufferBarrier(bufs[i], RenderBufferAccess::READ);
        }

        worker->commandList->barriers(RenderBarrierStage::COPY, copyIn, 3);
        for (int i = 0; i < 3; ++i) {
            worker->commandList->copyBufferRegion(RenderBufferReference(bufs[i]), RenderBufferReference(g.staging.get(), offsets[i]), sizes[i]);
        }

        worker->commandList->barriers(RenderBarrierStage::GRAPHICS, copyOut, 3);
        if (g.set == nullptr) {
            g.set = std::make_unique<RS64TerrainDescriptorSet>(shaderLibrary->samplerLibrary, device);
        }

        g.set->setBuffer(g.set->gHeights, g.heights.get(), g.heightsBytes, RenderBufferStructuredView(sizeof(int32_t)));
        g.set->setBuffer(g.set->gColors, g.colors.get(), g.colorsBytes, RenderBufferStructuredView(sizeof(uint32_t)));
        g.set->setBuffer(g.set->gCellLayers, g.cellLayers.get(), g.cellLayersBytes, RenderBufferStructuredView(sizeof(uint32_t)));
        g.set->setBuffer(g.set->gTexels, g.texels.get(), g.texelsBytes, RenderBufferStructuredView(sizeof(uint32_t)));
        g.version = m.version;
        const size_t waterCells = std::count(m.waterKind.begin(), m.waterKind.end(), uint8_t(1)), shoreCells = std::count(m.waterKind.begin(), m.waterKind.end(), uint8_t(2));
        std::fprintf(stderr, "[static-terrain] gpu map %ux%u level %u version %llu: %u texture layers (%zu animated), %zu water and %zu shore cells\n", m.width, m.height,
            m.level, (unsigned long long)m.version, g.layers, g.anims.size(), waterCells, shoreCells);
    }

    // Per frame: the map's lighting inputs and this frame's water surface (wave tiles, water level), as float4s for RS64TerrainVS gFrame.
    void FramebufferRenderer::recordRS64TerrainFrame(RenderWorker *worker) {
        RS64TerrainGPU &g = rs64TerrainGpu;
        if ((g.set == nullptr) || !rs64TerrainMap) {
            return;
        }

        // ROGUESQ_F5_STATIC_TERRAIN_SMOOTH=0 keeps the bilinear (creased) surface; ROGUESQ_F5_STATIC_TERRAIN_DEBUG=1 draws vertex colours only.
        static const uint32_t s_smooth = rs64env::on("ROGUESQ_F5_STATIC_TERRAIN_SMOOTH", true) ? 1u : 0u;
        static const uint32_t s_debug = uint32_t(rs64env::integer("ROGUESQ_F5_STATIC_TERRAIN_DEBUG"));
        rs64TerrainCB.sub = uint32_t(rs64_terrain_tessellation());
        rs64TerrainCB.smooth = s_smooth;
        rs64TerrainCB.debug = s_debug;
        rs64TerrainCB.chunkCells = RS64TerrainChunkCells;
        rs64TerrainCB.mapWidth = rs64TerrainMap->width;
        rs64TerrainCB.mapHeight = rs64TerrainMap->height;
        rs64TerrainCB.chunksX = (rs64TerrainMap->width + RS64TerrainChunkCells - 1) / RS64TerrainChunkCells;
        rs64TerrainCB.chunkCount = rs64TerrainCB.chunksX * ((rs64TerrainMap->height + RS64TerrainChunkCells - 1) / RS64TerrainChunkCells);

        // Header, then each texture layer's current layer (4 per float4).
        // After the layer remap: each layer's texture pack replacement (texture cache index, -1 none); gFrame[2].w is where that block starts, 0 = none.
        const uint32_t layerFloat4s = (g.layers + 3) / 4;
        const bool anyReplaced = std::any_of(g.layerReplaced.begin(), g.layerReplaced.end(), [](uint32_t i) { return i != UINT32_MAX; });
        g.anyReplaced = anyReplaced;
        const uint32_t FrameFloat4s = 3 + layerFloat4s * (anyReplaced ? 2 : 1);
        if ((g.frame == nullptr) || (g.frameFloat4s < FrameFloat4s)) {
            g.frame = worker->device->createBuffer(RenderBufferDesc::UploadBuffer(FrameFloat4s * 16, RenderBufferFlag::STORAGE));
            g.frameFloat4s = FrameFloat4s;
        }

        const rs64lights::TerrainShade &s = rs64TerrainMap->shade;
        float *f = reinterpret_cast<float *>(g.frame->map());
        const float header[12] = { s.ambient[0], s.ambient[1], s.ambient[2], s.cellScale, s.sunRgb[0], s.sunRgb[1], s.sunRgb[2], float(rs64TerrainFrame.waterLevel),
            s.sunDir[0], s.sunDir[1], s.sunDir[2], anyReplaced ? float(3 + layerFloat4s) : 0.0f };
        memcpy(f, header, sizeof(header));
        float *remap = f + 12;
        for (uint32_t l = 0; l < g.layers; ++l) {
            remap[l] = float(l);
        }

        for (const RS64TerrainGPU::Anim &a : g.anims) {
            remap[a.layer] = float(a.frames[(rs64TerrainFrame.task * a.fps / 30) % a.frames.size()]);
        }

        if (anyReplaced) {
            float *replaced = f + (3 + layerFloat4s) * 4;
            for (uint32_t l = 0; l < layerFloat4s * 4; ++l) {
                replaced[l] = ((l < g.layerReplaced.size()) && (g.layerReplaced[l] != UINT32_MAX)) ? float(g.layerReplaced[l]) : -1.0f;
            }
        }

        g.frame->unmap();
        g.set->setBuffer(g.set->gFrame, g.frame.get(), FrameFloat4s * 16, RenderBufferStructuredView(16));

        // The game's colours of the cells it drew this frame replace the ported ones (both are the same where no extra light reaches the cell).
        if ((g.colors != nullptr) && (g.version == rs64TerrainMap->version) && !rs64TerrainFrame.harvest.empty()) {
            const uint32_t latticeW = rs64TerrainMap->width * 4 + 1;
            uint32_t *lattice = reinterpret_cast<uint32_t *>(g.colors->map());
            for (const rs64lights::TerrainCellColors &hc : rs64TerrainFrame.harvest) {
                const uint32_t col = hc.cell % rs64TerrainMap->width, row = hc.cell / rs64TerrainMap->width;
                for (uint32_t k = 0; k < 25; ++k) {
                    const uint8_t *rgb = &hc.rgb[k * 3];
                    lattice[size_t(row * 4 + k / 5) * latticeW + col * 4 + k % 5] = uint32_t(rgb[0]) | (uint32_t(rgb[1]) << 8) | (uint32_t(rgb[2]) << 16) | 0xFF000000u;
                }
            }

            g.colors->unmap();
        }

        // Chunk culling and LOD: a chunk is dropped when its box lies behind the camera or past a side plane (with a wide side guard for the widescreen
        // expansion), never for distance: past the far plane the depth clamp hides it per pixel, so a distant chunk only drops to a coarser LOD.
        // Its LOD grows by one per doubling of view depth past ROGUESQ_F5_STATIC_TERRAIN_LOD_NEAR cells (0 = full detail everywhere).
        static const float s_lodNear = rs64env::number("ROGUESQ_F5_STATIC_TERRAIN_LOD_NEAR", 8.0f);
        const rs64lights::TerrainMap &m = *rs64TerrainMap;
        const uint32_t chunksX = (m.width + RS64TerrainChunkCells - 1) / RS64TerrainChunkCells, chunksZ = (m.height + RS64TerrainChunkCells - 1) / RS64TerrainChunkCells;
        const uint32_t chunkCount = chunksX * chunksZ;
        if (g.chunkMinH.size() != chunkCount) {
            return;
        }

        const uint32_t sub = rs64TerrainCB.sub;
        uint32_t maxLod = 0;
        while (((1u << (maxLod + 1)) <= 4 * sub) && (maxLod + 1 < RS64TerrainGPU::MaxLods)) {
            ++maxLod;
        }

        thread_local std::vector<uint32_t> lods, lists[RS64TerrainGPU::MaxLods];
        lods.assign(chunkCount, 0xFFu);
        for (auto &l : lists) {
            l.clear();
        }

        const float hs = rs64TerrainCB.heightScale;
        constexpr float Guard = 3.0f;
        const auto boxCulled = [&](float x0, float z0, float span, float y0, float y1) {
            int outside[5] = {};
            for (int c = 0; c < 8; ++c) {
                const hlslpp::float4 p = hlslpp::mul(hlslpp::float4(x0 + ((c & 1) ? span : 0.0f), (c & 2) ? y1 : y0, z0 + ((c & 4) ? span : 0.0f), 1.0f), rs64TerrainMvp);
                const float px = p.x, py = p.y, pw = p.w;
                // A box up to a span behind the camera plane stays: interpolated frames draw this list with a blended camera that can see it.
                outside[0] += (pw <= -span);
                outside[1] += (px > Guard * pw);
                outside[2] += (px < -Guard * pw);
                outside[3] += (py > Guard * pw);
                outside[4] += (py < -Guard * pw);
            }

            return std::any_of(std::begin(outside), std::end(outside), [](int n) { return n == 8; });
        };

        for (uint32_t cz = 0; cz < chunksZ; ++cz) {
            for (uint32_t cx = 0; cx < chunksX; ++cx) {
                const uint32_t k = cz * chunksX + cx;
                const float x0 = (float(cx * RS64TerrainChunkCells) - rs64TerrainCB.originX) * 512.0f + rs64TerrainCB.offsetX;
                const float z0 = (float(cz * RS64TerrainChunkCells) - rs64TerrainCB.originZ) * 512.0f + rs64TerrainCB.offsetZ;
                const float span = float(RS64TerrainChunkCells) * 512.0f;
                const float y0 = std::min(float(g.chunkMinH[k]), float(rs64TerrainFrame.waterLevel) * 16.0f) * hs, y1 = std::max(float(g.chunkMaxH[k]), float(rs64TerrainFrame.waterLevel) * 16.0f) * hs;
                if (boxCulled(x0, z0, span, y0, y1)) {
                    continue;
                }

                const hlslpp::float4 c = hlslpp::mul(hlslpp::float4(x0 + span * 0.5f, (y0 + y1) * 0.5f, z0 + span * 0.5f, 1.0f), rs64TerrainMvp);
                const float depthCells = std::max(float(c.w), 0.0f) / 512.0f;
                uint32_t lod = 0;
                if ((s_lodNear > 0.0f) && (depthCells >= s_lodNear)) {
                    lod = std::min(maxLod, 1u + uint32_t(std::floor(std::log2(depthCells / s_lodNear))));
                }

                lods[k] = lod;
                lists[lod].push_back(k);
            }
        }

        thread_local std::vector<uint32_t> packed;
        packed.assign(lods.begin(), lods.end());
        for (uint32_t l = 0; l < RS64TerrainGPU::MaxLods; ++l) {
            g.lodBase[l] = uint32_t(packed.size());
            g.lodCount[l] = uint32_t(lists[l].size());
            packed.insert(packed.end(), lists[l].begin(), lists[l].end());
        }

        // The water draw in chunk-sized blocks over the map plus its margin. A block inside the map with no water draws nothing; the perimeter reaches past any far plane, so it always stays.
        const int32_t margin = rs64lights::TerrainWaterMargin;
        const uint32_t blocksX = (m.width + 2 * margin + RS64TerrainChunkCells - 1) / RS64TerrainChunkCells;
        const uint32_t blocksZ = (m.height + 2 * margin + RS64TerrainChunkCells - 1) / RS64TerrainChunkCells;
        const float waterH = float(rs64TerrainFrame.waterLevel) * 16.0f;
        const float groundMin = float(*std::min_element(g.chunkMinH.begin(), g.chunkMinH.end())), groundMax = float(*std::max_element(g.chunkMaxH.begin(), g.chunkMaxH.end()));
        const float waterY0 = (std::min(groundMin, waterH) - 16.0f) * hs, waterY1 = (std::max(groundMax, waterH) + 16.0f) * hs;
        g.waterBase = uint32_t(packed.size());
        for (uint32_t bz = 0; bz < blocksZ; ++bz) {
            for (uint32_t bx = 0; bx < blocksX; ++bx) {
                const int32_t col = int32_t(bx * RS64TerrainChunkCells) - margin, row = int32_t(bz * RS64TerrainChunkCells) - margin;
                const bool perimeter = (bx == 0) || (bz == 0) || (bx == blocksX - 1) || (bz == blocksZ - 1);
                const bool insideMap = (col >= 0) && (row >= 0) && ((col + RS64TerrainChunkCells) <= m.width) && ((row + RS64TerrainChunkCells) <= m.height);
                bool hasWater = !insideMap;
                for (uint32_t z = 0; insideMap && !hasWater && (z < RS64TerrainChunkCells); ++z) {
                    for (uint32_t x = 0; (x < RS64TerrainChunkCells) && !hasWater; ++x) {
                        const size_t cell = size_t(row + z) * m.width + col + x;
                        hasWater = (cell < m.waterKind.size()) && (m.waterKind[cell] != 0);
                    }
                }

                if (!hasWater) {
                    continue;
                }

                const float blockX = (float(col) - rs64TerrainCB.originX) * 512.0f + rs64TerrainCB.offsetX;
                const float blockZ = (float(row) - rs64TerrainCB.originZ) * 512.0f + rs64TerrainCB.offsetZ;
                if (perimeter || !boxCulled(blockX, blockZ, float(RS64TerrainChunkCells) * 512.0f, waterY0, waterY1)) {
                    packed.push_back(bz * blocksX + bx);
                }
            }
        }

        g.waterCount = uint32_t(packed.size()) - g.waterBase;

        // Sized for the most any frame can list, so the buffer is not replaced while earlier frames still read it.
        const uint32_t chunksWorstCase = chunkCount * 2 + blocksX * blocksZ;
        if ((g.chunks == nullptr) || (g.chunksCapacity < chunksWorstCase)) {
            g.chunksCapacity = chunksWorstCase;
            g.chunks = worker->device->createBuffer(RenderBufferDesc::UploadBuffer(g.chunksCapacity * sizeof(uint32_t), RenderBufferFlag::STORAGE));
        }

        memcpy(g.chunks->map(), packed.data(), packed.size() * sizeof(uint32_t));
        g.chunks->unmap();
        g.set->setBuffer(g.set->gChunks, g.chunks.get(), g.chunksCapacity * sizeof(uint32_t), RenderBufferStructuredView(sizeof(uint32_t)));
    }

    // The visible chunks, one instanced draw per LOD (chunkCells * 4 * sub >> lod quads per chunk edge), at the anchor call's place in the scene
    // with its scissor and screen mapping.
    void FramebufferRenderer::recordRS64Terrain(RenderWorker *worker, const Framebuffer &framebuffer, const InstanceDrawCall &anchor) {
        const RS64TerrainGPU &g = rs64TerrainGpu;
        // ROGUESQ_F5_STATIC_TERRAIN_CHECK: a pass that draws no land (the map vanishes for that frame) says why.
        static const bool s_check = rs64env::on("ROGUESQ_F5_STATIC_TERRAIN_CHECK");
        if (s_check) {
            uint32_t land = 0;
            for (uint32_t l = 0; l < RS64TerrainGPU::MaxLods; ++l) {
                land += g.lodCount[l];
            }

            if (!rs64TerrainMap || (g.version != (rs64TerrainMap ? rs64TerrainMap->version : 0)) || (g.set == nullptr) || (g.chunks == nullptr) || anchor.triangles.scissor.isEmpty() || (land == 0)) {
                std::fprintf(stderr, "[static-terrain] no land drawn: map=%d version gpu=%llu map=%llu set=%d chunks=%d scissorEmpty=%d landChunks=%u origin=%d,%d offset=%.0f,%.0f\n",
                    (int)(rs64TerrainMap != nullptr), (unsigned long long)g.version, (unsigned long long)(rs64TerrainMap ? rs64TerrainMap->version : 0), (int)(g.set != nullptr),
                    (int)(g.chunks != nullptr), (int)anchor.triangles.scissor.isEmpty(), land, rs64TerrainCB.originX, rs64TerrainCB.originZ, rs64TerrainCB.offsetX, rs64TerrainCB.offsetZ);
            }
        }

        if (!rs64TerrainMap || (g.version != rs64TerrainMap->version) || (g.set == nullptr) || (g.chunks == nullptr) || anchor.triangles.scissor.isEmpty()) {
            return;
        }

        interop::RS64TerrainCB cb = rs64TerrainCB;
        cb.screenScale = anchor.triangles.screenScale;
        cb.screenOffset = anchor.triangles.screenOffset;
        // A texture pack replaced some layers: the variant that also samples the texture cache (set 1).
        const bool packs = g.anyReplaced && (shaderLibrary->rs64TerrainPacks.pipeline != nullptr) && (descTextureSet != nullptr);
        const ShaderRecord &record = packs ? shaderLibrary->rs64TerrainPacks : shaderLibrary->rs64Terrain;
        const ShaderRecord &blended = packs ? shaderLibrary->rs64TerrainWaterPacks : shaderLibrary->rs64TerrainWater;
        worker->commandList->setPipeline((cb.fogClip ? blended : record).pipeline.get());
        worker->commandList->setGraphicsPipelineLayout(record.pipelineLayout.get());
        worker->commandList->setGraphicsDescriptorSet(g.set->get(), 0);
        if (packs) {
            worker->commandList->setGraphicsDescriptorSet(descTextureSet->get(), 1);
        }

        // Without this the map was depth-tested against earlier frames' depth: where the ground lay farther than before (a camera rising or
        // looking down) it failed, and old frames showed through.
        if (anchor.rs64TerrainClearDepth) {
            worker->commandList->clearDepth(true, 1.0f, &anchor.triangles.scissor, 1);
        }

        worker->commandList->setViewports(framebuffer.viewport);
        worker->commandList->setScissors(anchor.triangles.scissor);
        worker->commandList->setVertexBuffers(0, nullptr, 0, nullptr);
        worker->commandList->setIndexBuffer(nullptr);
        for (uint32_t l = 0; l < RS64TerrainGPU::MaxLods; ++l) {
            const uint32_t quads = (RS64TerrainChunkCells * 4 * cb.sub) >> l;
            if ((g.lodCount[l] == 0) || (quads == 0)) {
                continue;
            }

            cb.lod = l;
            cb.chunkListBase = g.lodBase[l];
            worker->commandList->setGraphicsPushConstants(0, &cb);
            worker->commandList->drawInstanced(quads * quads * 6, g.lodCount[l], 0, 0);
        }

        // Runs on every level: besides any water it carries the map's edge cells out past the edge, so a land-only map fogs out instead of ending in a void.
        cb.water = 1;
        cb.waterMargin = uint32_t(rs64lights::TerrainWaterMargin);
        cb.chunkListBase = g.waterBase;
        worker->commandList->setPipeline((packs ? shaderLibrary->rs64TerrainWaterPacks : shaderLibrary->rs64TerrainWater).pipeline.get());
        worker->commandList->setGraphicsPushConstants(0, &cb);
        worker->commandList->drawInstanced(RS64TerrainChunkCells * RS64TerrainChunkCells * 6, g.waterCount, 0, 0);
    }

    void FramebufferRenderer::recordRS64TerrainBLAS(RenderWorker *worker) {
        RS64ShadowScene &s = rs64Shadow;
        RenderDevice *device = worker->device;
        thread_local std::vector<float> pos;
        thread_local std::vector<uint32_t> idx;
        rs64lights::buildTerrainMesh(*s.terrainMap, rs64lights::config().shadowTerrainSub, pos, idx);
        s.terrainBuiltVersion = s.terrainMap->version;

        // Each cell's texture average, from the level's ROM texture bank (one decode per bank entry).
        const rs64lights::TerrainMap &tm = *s.terrainMap;
        thread_local std::vector<uint32_t> texAvg, hitColors;
        texAvg.assign(size_t(tm.width) * tm.height, 0x808080u);
        const uint8_t *bank = nullptr, *ids = nullptr;
        uint32_t bankSize = 0, idsSize = 0;
        if (rs64_terrain_textures(tm.level, &bank, &bankSize, &ids, &idsSize)) {
            std::unordered_map<uint32_t, uint32_t> avgOf;
            uint32_t texels[4096];
            for (size_t c = 0; (c < texAvg.size()) && (c < tm.texIds.size()); ++c) {
                const uint32_t t = tm.texIds[c];
                if ((t * 4 + 2) > idsSize) {
                    continue;
                }

                const uint32_t b = (uint32_t(ids[t * 4]) << 8) | ids[t * 4 + 1];
                if ((b + 1) * rs64lights::TerrainTexBytes > bankSize) {
                    continue;
                }

                auto it = avgOf.find(b);
                if (it == avgOf.end()) {
                    rs64lights::decodeTerrainTexture(bank + size_t(b) * rs64lights::TerrainTexBytes, texels);
                    uint32_t sum[3] = {};
                    for (uint32_t texel : texels) {
                        for (int ch = 0; ch < 3; ++ch) {
                            sum[ch] += (texel >> (8 * ch)) & 0xFF;
                        }
                    }

                    it = avgOf.emplace(b, (sum[0] / 4096) | ((sum[1] / 4096) << 8) | ((sum[2] / 4096) << 16)).first;
                }

                texAvg[c] = it->second;
            }
        }

        rs64lights::terrainHitColors(tm, rs64lights::config().shadowTerrainSub, texAvg, hitColors);
        s.terrainWaterPrim = hitColors[1];
        s.terrainColorBytes = hitColors.size() * sizeof(uint32_t);
        s.terrainColorBuffer = device->createBuffer(RenderBufferDesc::UploadBuffer(s.terrainColorBytes));
        memcpy(s.terrainColorBuffer->map(), hitColors.data(), s.terrainColorBytes);
        s.terrainColorBuffer->unmap();
        s.terrainColorGpu = device->createBuffer(RenderBufferDesc::DefaultBuffer(s.terrainColorBytes, RenderBufferFlag::STORAGE));
        s.terrainBlas.reset();
        s.terrainNormalGpu.reset();
        if (idx.empty()) {
            return;
        }
        const uint64_t posBytes = pos.size() * sizeof(float);
        const uint64_t idxBytes = idx.size() * sizeof(uint32_t);
        s.terrainPosBuffer = device->createBuffer(RenderBufferDesc::UploadBuffer(posBytes));
        memcpy(s.terrainPosBuffer->map(), pos.data(), posBytes);
        s.terrainPosBuffer->unmap();
        s.terrainIndexBuffer = device->createBuffer(RenderBufferDesc::UploadBuffer(idxBytes));
        memcpy(s.terrainIndexBuffer->map(), idx.data(), idxBytes);
        s.terrainIndexBuffer->unmap();

        // The BLAS builds from GPU-local copies: on Vulkan a build straight from the upload buffers came out empty (no ray hit the terrain).
        s.terrainPosGpu = device->createBuffer(RenderBufferDesc::DefaultBuffer(posBytes, RenderBufferFlag::ACCELERATION_STRUCTURE_INPUT));
        s.terrainIndexGpu = device->createBuffer(RenderBufferDesc::DefaultBuffer(idxBytes, RenderBufferFlag::STORAGE | RenderBufferFlag::ACCELERATION_STRUCTURE_INPUT));
        const RenderBufferBarrier inputsIn[] = { RenderBufferBarrier(s.terrainPosGpu.get(), RenderBufferAccess::WRITE), RenderBufferBarrier(s.terrainIndexGpu.get(), RenderBufferAccess::WRITE) };
        worker->commandList->barriers(RenderBarrierStage::COPY, inputsIn, uint32_t(std::size(inputsIn)));
        worker->commandList->copyBufferRegion(RenderBufferReference(s.terrainPosGpu.get()), RenderBufferReference(s.terrainPosBuffer.get()), posBytes);
        worker->commandList->copyBufferRegion(RenderBufferReference(s.terrainIndexGpu.get()), RenderBufferReference(s.terrainIndexBuffer.get()), idxBytes);
        const RenderBufferBarrier inputsOut[] = { RenderBufferBarrier(s.terrainPosGpu.get(), RenderBufferAccess::READ), RenderBufferBarrier(s.terrainIndexGpu.get(), RenderBufferAccess::READ) };
        worker->commandList->barriers(RenderBarrierStage::COMPUTE, inputsOut, uint32_t(std::size(inputsOut)));

        const RenderBottomLevelASMesh mesh(RenderBufferReference(s.terrainIndexGpu.get()), RenderBufferReference(s.terrainPosGpu.get()), RenderFormat::R32_UINT, RenderFormat::R32G32B32_FLOAT, uint32_t(idx.size()), uint32_t(pos.size() / 3), sizeof(float) * 3, true);
        RenderBottomLevelASBuildInfo info;
        device->setBottomLevelASBuildInfo(info, &mesh, 1, false, true);
        s.terrainBlasBuffer = device->createBuffer(RenderBufferDesc::AccelerationStructureBuffer(info.accelerationStructureSize));
        s.terrainBlas = device->createAccelerationStructure(RenderAccelerationStructureDesc(RenderAccelerationStructureType::BOTTOM_LEVEL, RenderBufferReference(s.terrainBlasBuffer.get()), info.accelerationStructureSize));
        s.terrainScratch = device->createBuffer(RenderBufferDesc::DefaultBuffer(info.scratchSize, RenderBufferFlag::ACCELERATION_STRUCTURE_SCRATCH));
        worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderBufferBarrier(s.terrainBlasBuffer.get(), RenderBufferAccess::WRITE));
        worker->commandList->buildBottomLevelAS(s.terrainBlas.get(), RenderBufferReference(s.terrainScratch.get()), info);
        worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderBufferBarrier(s.terrainBlasBuffer.get(), RenderBufferAccess::READ));

        // Smooth normals and indices for the shadow pass's terrain receivers, GPU-local like the cutout tables.
        thread_local std::vector<float> nrm;
        rs64lights::buildTerrainNormals(pos, idx, nrm);
        s.terrainIndexBytes = idxBytes;
        s.terrainNormalBytes = nrm.size() * sizeof(float);
        s.terrainNormalBuffer = device->createBuffer(RenderBufferDesc::UploadBuffer(s.terrainNormalBytes));
        memcpy(s.terrainNormalBuffer->map(), nrm.data(), s.terrainNormalBytes);
        s.terrainNormalBuffer->unmap();
        s.terrainNormalGpu = device->createBuffer(RenderBufferDesc::DefaultBuffer(s.terrainNormalBytes, RenderBufferFlag::STORAGE));
        const RenderBufferBarrier copyIn[] = { RenderBufferBarrier(s.terrainNormalGpu.get(), RenderBufferAccess::WRITE), RenderBufferBarrier(s.terrainColorGpu.get(), RenderBufferAccess::WRITE) };
        worker->commandList->barriers(RenderBarrierStage::COPY, copyIn, uint32_t(std::size(copyIn)));
        worker->commandList->copyBufferRegion(RenderBufferReference(s.terrainNormalGpu.get()), RenderBufferReference(s.terrainNormalBuffer.get()), s.terrainNormalBytes);
        worker->commandList->copyBufferRegion(RenderBufferReference(s.terrainColorGpu.get()), RenderBufferReference(s.terrainColorBuffer.get()), s.terrainColorBytes);
        const RenderBufferBarrier copyOut[] = { RenderBufferBarrier(s.terrainIndexGpu.get(), RenderBufferAccess::READ), RenderBufferBarrier(s.terrainNormalGpu.get(), RenderBufferAccess::READ),
            RenderBufferBarrier(s.terrainColorGpu.get(), RenderBufferAccess::READ) };
        worker->commandList->barriers(RenderBarrierStage::GRAPHICS, copyOut, uint32_t(std::size(copyOut)));
        if (rs64lights::config().log) {
            std::fprintf(stderr, "[rt-terrain] blas: map=%ux%u version=%llu tris=%zu bytes=%llu\n", s.terrainMap->width, s.terrainMap->height, (unsigned long long)s.terrainMap->version, idx.size() / 3, (unsigned long long)info.accelerationStructureSize);
        }
    }

    void FramebufferRenderer::recordRS64ShadowScene(RenderWorker *worker, bool dynamic) {
        RS64ShadowScene &s = rs64Shadow;
        RenderDevice *device = worker->device;
        auto grow = [device](std::unique_ptr<RenderBuffer> &buf, uint64_t &cap, uint64_t need, RenderBufferDesc desc) {
            if ((need <= cap) && buf) {
                return false;
            }
            cap = need + need / 2 + 256;
            desc.size = cap;
            buf = device->createBuffer(desc);
            return true;
        };

        s.retired.blas.reset();
        s.retired.blasBuffer.reset();
        if (s.terrainActive && (s.terrainBuiltVersion != s.terrainMap->version)) {
            recordRS64TerrainBLAS(worker);
        }

        RenderTopLevelASInstance instances[3 + 4 + 2];
        uint32_t instanceCount = 0;
        RenderBottomLevelASBuildInfo blasInfo;
        RenderBottomLevelASBuildInfo cutoutInfo;
        const bool opaque = dynamic && !s.indices.empty();
        s.cutoutDrawCount = 0;
        if (opaque) {
            const uint64_t indexBytes = s.indices.size() * sizeof(uint32_t);
            grow(s.indexBuffer, s.indexCapacity, indexBytes, RenderBufferDesc::UploadBuffer(indexBytes, RenderBufferFlag::ACCELERATION_STRUCTURE_INPUT));
            void *dst = s.indexBuffer->map();
            memcpy(dst, s.indices.data(), indexBytes);
            s.indexBuffer->unmap();
            const RenderBottomLevelASMesh mesh(RenderBufferReference(s.indexBuffer.get()), RenderBufferReference(s.worldPos), RenderFormat::R32_UINT, RenderFormat::R32G32B32_FLOAT, uint32_t(s.indices.size()), s.vertexCount, sizeof(float) * 4, true);
            device->setBottomLevelASBuildInfo(blasInfo, &mesh, 1, true, false);
            if (grow(s.blasBuffer, s.blasCapacity, blasInfo.accelerationStructureSize, RenderBufferDesc::AccelerationStructureBuffer(blasInfo.accelerationStructureSize)) || !s.blas) {
                s.blas = device->createAccelerationStructure(RenderAccelerationStructureDesc(RenderAccelerationStructureType::BOTTOM_LEVEL, RenderBufferReference(s.blasBuffer.get()), s.blasCapacity));
            }
            grow(s.blasScratch, s.blasScratchCapacity, blasInfo.scratchSize, RenderBufferDesc::DefaultBuffer(blasInfo.scratchSize, RenderBufferFlag::ACCELERATION_STRUCTURE_SCRATCH));
            instances[instanceCount++] = RenderTopLevelASInstance(RenderBufferReference(s.blasBuffer.get()), 0, 0x01, 0, true, RenderAffineTransform());
        }
        // Cutouts: a non-opaque BLAS hit-tested against the draw's texture alpha; left out when the tex-coord buffer is missing.
        if (dynamic && !s.cutoutIndices.empty() && (s.texCoords != nullptr)) {
            const uint64_t idxBytes = s.cutoutIndices.size() * sizeof(uint32_t);
            grow(s.cutoutIndexBuffer, s.cutoutIndexCapacity, idxBytes, RenderBufferDesc::UploadBuffer(idxBytes, RenderBufferFlag::ACCELERATION_STRUCTURE_INPUT | RenderBufferFlag::STORAGE));
            memcpy(s.cutoutIndexBuffer->map(), s.cutoutIndices.data(), idxBytes);
            s.cutoutIndexBuffer->unmap();
            const uint64_t tabBytes = s.cutoutTable.size() * sizeof(rs64lights::CutoutEntry);
            grow(s.cutoutTableBuffer, s.cutoutTableCapacity, tabBytes, RenderBufferDesc::UploadBuffer(tabBytes, RenderBufferFlag::STORAGE));
            memcpy(s.cutoutTableBuffer->map(), s.cutoutTable.data(), tabBytes);
            s.cutoutTableBuffer->unmap();
            grow(s.cutoutIndexGpu, s.cutoutIndexGpuCapacity, idxBytes, RenderBufferDesc::DefaultBuffer(idxBytes, RenderBufferFlag::STORAGE));
            grow(s.cutoutTableGpu, s.cutoutTableGpuCapacity, tabBytes, RenderBufferDesc::DefaultBuffer(tabBytes, RenderBufferFlag::STORAGE));
            s.cutoutIndexBytes = idxBytes;
            s.cutoutTableBytes = tabBytes;
            const RenderBottomLevelASMesh cmesh(RenderBufferReference(s.cutoutIndexBuffer.get()), RenderBufferReference(s.worldPos), RenderFormat::R32_UINT, RenderFormat::R32G32B32_FLOAT, uint32_t(s.cutoutIndices.size()), s.vertexCount, sizeof(float) * 4, false);
            device->setBottomLevelASBuildInfo(cutoutInfo, &cmesh, 1, true, false);
            if (grow(s.cutoutBlasBuffer, s.cutoutBlasCapacity, cutoutInfo.accelerationStructureSize, RenderBufferDesc::AccelerationStructureBuffer(cutoutInfo.accelerationStructureSize)) || !s.cutoutBlas) {
                s.cutoutBlas = device->createAccelerationStructure(RenderAccelerationStructureDesc(RenderAccelerationStructureType::BOTTOM_LEVEL, RenderBufferReference(s.cutoutBlasBuffer.get()), s.cutoutBlasCapacity));
            }
            grow(s.cutoutBlasScratch, s.cutoutBlasScratchCapacity, cutoutInfo.scratchSize, RenderBufferDesc::DefaultBuffer(cutoutInfo.scratchSize, RenderBufferFlag::ACCELERATION_STRUCTURE_SCRATCH));
            instances[instanceCount++] = RenderTopLevelASInstance(RenderBufferReference(s.cutoutBlasBuffer.get()), 2, 0x04, 0, true, RenderAffineTransform());
            s.cutoutDrawCount = uint32_t(s.cutoutTable.size());
        }
        // Emissive draws (lasers, glow and exhaust cards): seen by bounce and reflection rays (mask 0x80), never by shadow queries.
        RenderBottomLevelASBuildInfo emissiveInfo;
        const bool emissiveScene = dynamic && !s.emissiveIndices.empty() && !s.emissiveTable.empty();
        s.emissiveCount = 0;
        if (emissiveScene) {
            const uint64_t idxBytes = s.emissiveIndices.size() * sizeof(uint32_t);
            grow(s.emissiveIndexBuffer, s.emissiveIndexCapacity, idxBytes, RenderBufferDesc::UploadBuffer(idxBytes, RenderBufferFlag::ACCELERATION_STRUCTURE_INPUT));
            memcpy(s.emissiveIndexBuffer->map(), s.emissiveIndices.data(), idxBytes);
            s.emissiveIndexBuffer->unmap();
            const RenderBottomLevelASMesh emesh(RenderBufferReference(s.emissiveIndexBuffer.get()), RenderBufferReference(s.worldPos), RenderFormat::R32_UINT, RenderFormat::R32G32B32_FLOAT, uint32_t(s.emissiveIndices.size()), s.vertexCount, sizeof(float) * 4, true);
            device->setBottomLevelASBuildInfo(emissiveInfo, &emesh, 1, true, false);
            if (grow(s.emissiveBlasBuffer, s.emissiveBlasCapacity, emissiveInfo.accelerationStructureSize, RenderBufferDesc::AccelerationStructureBuffer(emissiveInfo.accelerationStructureSize)) || !s.emissiveBlas) {
                s.emissiveBlas = device->createAccelerationStructure(RenderAccelerationStructureDesc(RenderAccelerationStructureType::BOTTOM_LEVEL, RenderBufferReference(s.emissiveBlasBuffer.get()), s.emissiveBlasCapacity));
            }
            grow(s.emissiveBlasScratch, s.emissiveBlasScratchCapacity, emissiveInfo.scratchSize, RenderBufferDesc::DefaultBuffer(emissiveInfo.scratchSize, RenderBufferFlag::ACCELERATION_STRUCTURE_SCRATCH));
            instances[instanceCount++] = RenderTopLevelASInstance(RenderBufferReference(s.emissiveBlasBuffer.get()), 7, 0x80, 0, true, RenderAffineTransform());
            s.emissiveCount = uint32_t(s.emissiveTable.size());
        }
        // Craft draws cast like other drawn casters (mask 0x01) but never enter a history snapshot; hits take the caster average colour.
        RenderBottomLevelASBuildInfo craftInfo;
        const bool craftScene = dynamic && !s.craftIndices.empty();
        if (craftScene) {
            const uint64_t idxBytes = s.craftIndices.size() * sizeof(uint32_t);
            grow(s.craftIndexBuffer, s.craftIndexCapacity, idxBytes, RenderBufferDesc::UploadBuffer(idxBytes, RenderBufferFlag::ACCELERATION_STRUCTURE_INPUT));
            memcpy(s.craftIndexBuffer->map(), s.craftIndices.data(), idxBytes);
            s.craftIndexBuffer->unmap();
            const RenderBottomLevelASMesh kmesh(RenderBufferReference(s.craftIndexBuffer.get()), RenderBufferReference(s.worldPos), RenderFormat::R32_UINT, RenderFormat::R32G32B32_FLOAT, uint32_t(s.craftIndices.size()), s.vertexCount, sizeof(float) * 4, true);
            device->setBottomLevelASBuildInfo(craftInfo, &kmesh, 1, true, false);
            if (grow(s.craftBlasBuffer, s.craftBlasCapacity, craftInfo.accelerationStructureSize, RenderBufferDesc::AccelerationStructureBuffer(craftInfo.accelerationStructureSize)) || !s.craftBlas) {
                s.craftBlas = device->createAccelerationStructure(RenderAccelerationStructureDesc(RenderAccelerationStructureType::BOTTOM_LEVEL, RenderBufferReference(s.craftBlasBuffer.get()), s.craftBlasCapacity));
            }
            grow(s.craftBlasScratch, s.craftBlasScratchCapacity, craftInfo.scratchSize, RenderBufferDesc::DefaultBuffer(craftInfo.scratchSize, RenderBufferFlag::ACCELERATION_STRUCTURE_SCRATCH));
            instances[instanceCount++] = RenderTopLevelASInstance(RenderBufferReference(s.craftBlasBuffer.get()), 8, 0x01, 0, true, RenderAffineTransform());
        }
        // Hit colour tables, GPU-local like the cutout tables.
        s.hitCount = 0;
        const auto uploadTable = [&](const std::vector<rs64lights::HitColorEntry> &table, std::unique_ptr<RenderBuffer> &up, uint64_t &upCap, std::unique_ptr<RenderBuffer> &gpu, uint64_t &gpuCap, uint64_t &bytes) {
            bytes = table.size() * sizeof(rs64lights::HitColorEntry);
            grow(up, upCap, bytes, RenderBufferDesc::UploadBuffer(bytes, RenderBufferFlag::STORAGE));
            memcpy(up->map(), table.data(), bytes);
            up->unmap();
            grow(gpu, gpuCap, bytes, RenderBufferDesc::DefaultBuffer(bytes, RenderBufferFlag::STORAGE));
        };
        if (opaque && !s.hitTable.empty()) {
            uploadTable(s.hitTable, s.hitTableBuffer, s.hitTableCapacity, s.hitTableGpu, s.hitTableGpuCapacity, s.hitTableBytes);
            s.hitCount = uint32_t(s.hitTable.size());
        }
        if (emissiveScene) {
            uploadTable(s.emissiveTable, s.emissiveTableBuffer, s.emissiveTableCapacity, s.emissiveTableGpu, s.emissiveTableGpuCapacity, s.emissiveTableBytes);
        }
        if (s.terrainActive && s.terrainBlas) {
            RenderAffineTransform xf;
            memcpy(xf.m, s.terrainTransform, sizeof(xf.m));
            instances[instanceCount++] = RenderTopLevelASInstance(RenderBufferReference(s.terrainBlasBuffer.get()), 1, 0x02, 0, false, xf);
        }
        // Past frames' casters (slot i: id 3+i, mask 0x08 << i), moved from their camera into this one through map space.
        const rs64lights::Config &lcfg = rs64lights::config();
        const bool terrainNow = s.terrainActive && s.terrainBlas && (s.terrainMap != nullptr);
        s.historyMask = 0;
        for (int i = 0; terrainNow && (i < lcfg.shadowHistory); ++i) {
            RS64ShadowScene::History &h = s.history[i];
            float inv[3][4];
            if (!h.blas || (h.mapVersion != s.terrainMap->version) || !rs64lights::affineInverse(h.terrainTransform, inv)) {
                continue;
            }
            float delta[3][4];
            rs64lights::affineCompose(s.terrainTransform, inv, delta);
            RenderAffineTransform xf;
            memcpy(xf.m, delta, sizeof(xf.m));
            instances[instanceCount++] = RenderTopLevelASInstance(RenderBufferReference(h.blasBuffer.get()), 3 + i, uint8_t(0x08u << i), 0, true, xf);
            s.historyMask |= 0x08u << i;
            s.historyCam[i][0] = delta[0][3];
            s.historyCam[i][1] = delta[1][3];
            s.historyCam[i][2] = delta[2][3];
            s.historyCam[i][3] = lcfg.shadowHistorySkip;
        }
        if (lcfg.log && (lcfg.shadowHistory > 0)) {
            static uint32_t s_histLog = 0;
            if ((++s_histLog % 120) == 1) {
                std::fprintf(stderr, "[rt-shadows] history mask=0x%02X terrain=%d cam0=(%.0f,%.0f,%.0f) cam1=(%.0f,%.0f,%.0f)\n", s.historyMask, terrainNow ? 1 : 0,
                    s.historyCam[0][0], s.historyCam[0][1], s.historyCam[0][2], s.historyCam[1][0], s.historyCam[1][1], s.historyCam[1][2]);
            }
        }
        // Every interval frames this frame's opaque BLAS is kept as a history slot (it owns its geometry once built); the next frame grows a fresh one.
        const bool snapshot = opaque && terrainNow && (lcfg.shadowHistory > 0) && ((++s.historyFrame % uint32_t(lcfg.shadowHistoryInterval)) == 0);
        s.indices.clear();
        s.terrainActive = false;
        if (instanceCount == 0) {
            s.ready = false;
            return;
        }

        RenderTopLevelASBuildInfo tlasInfo;
        device->setTopLevelASBuildInfo(tlasInfo, instances, instanceCount, true, false);
        if (grow(s.tlasBuffer, s.tlasCapacity, tlasInfo.accelerationStructureSize, RenderBufferDesc::AccelerationStructureBuffer(tlasInfo.accelerationStructureSize)) || !s.tlas) {
            s.tlas = device->createAccelerationStructure(RenderAccelerationStructureDesc(RenderAccelerationStructureType::TOP_LEVEL, RenderBufferReference(s.tlasBuffer.get()), s.tlasCapacity));
        }
        grow(s.tlasScratch, s.tlasScratchCapacity, tlasInfo.scratchSize, RenderBufferDesc::DefaultBuffer(tlasInfo.scratchSize, RenderBufferFlag::ACCELERATION_STRUCTURE_SCRATCH));
        const uint64_t instBytes = tlasInfo.instancesBufferData.size();
        grow(s.instancesBuffer, s.instancesCapacity, instBytes, RenderBufferDesc::UploadBuffer(instBytes, RenderBufferFlag::ACCELERATION_STRUCTURE_INPUT));
        void *inst = s.instancesBuffer->map();
        memcpy(inst, tlasInfo.instancesBufferData.data(), instBytes);
        s.instancesBuffer->unmap();

        // Vulkan takes a barrier's source stage from the buffer's previous barrier; without these a freshly grown AS buffer orders its builds after nothing.
        worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderBufferBarrier(s.tlasBuffer.get(), RenderBufferAccess::WRITE));
        if (opaque) {
            worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderBufferBarrier(s.blasBuffer.get(), RenderBufferAccess::WRITE));
            worker->commandList->buildBottomLevelAS(s.blas.get(), RenderBufferReference(s.blasScratch.get()), blasInfo);
            worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderBufferBarrier(s.blasBuffer.get(), RenderBufferAccess::READ));
        }
        if (s.cutoutDrawCount > 0) {
            const RenderBufferBarrier copyIn[] = { RenderBufferBarrier(s.cutoutIndexGpu.get(), RenderBufferAccess::WRITE), RenderBufferBarrier(s.cutoutTableGpu.get(), RenderBufferAccess::WRITE) };
            worker->commandList->barriers(RenderBarrierStage::COPY, copyIn, uint32_t(std::size(copyIn)));
            worker->commandList->copyBufferRegion(RenderBufferReference(s.cutoutIndexGpu.get()), RenderBufferReference(s.cutoutIndexBuffer.get()), s.cutoutIndexBytes);
            worker->commandList->copyBufferRegion(RenderBufferReference(s.cutoutTableGpu.get()), RenderBufferReference(s.cutoutTableBuffer.get()), s.cutoutTableBytes);
            const RenderBufferBarrier copyOut[] = { RenderBufferBarrier(s.cutoutIndexGpu.get(), RenderBufferAccess::READ), RenderBufferBarrier(s.cutoutTableGpu.get(), RenderBufferAccess::READ) };
            worker->commandList->barriers(RenderBarrierStage::GRAPHICS, copyOut, uint32_t(std::size(copyOut)));
            worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderBufferBarrier(s.cutoutBlasBuffer.get(), RenderBufferAccess::WRITE));
            worker->commandList->buildBottomLevelAS(s.cutoutBlas.get(), RenderBufferReference(s.cutoutBlasScratch.get()), cutoutInfo);
            worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderBufferBarrier(s.cutoutBlasBuffer.get(), RenderBufferAccess::READ));
        }
        if (emissiveScene) {
            worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderBufferBarrier(s.emissiveBlasBuffer.get(), RenderBufferAccess::WRITE));
            worker->commandList->buildBottomLevelAS(s.emissiveBlas.get(), RenderBufferReference(s.emissiveBlasScratch.get()), emissiveInfo);
            worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderBufferBarrier(s.emissiveBlasBuffer.get(), RenderBufferAccess::READ));
        }
        if (craftScene) {
            worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderBufferBarrier(s.craftBlasBuffer.get(), RenderBufferAccess::WRITE));
            worker->commandList->buildBottomLevelAS(s.craftBlas.get(), RenderBufferReference(s.craftBlasScratch.get()), craftInfo);
            worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderBufferBarrier(s.craftBlasBuffer.get(), RenderBufferAccess::READ));
        }
        for (int t = 0; t < 2; ++t) {
            RenderBuffer *gpu = (t == 0) ? s.hitTableGpu.get() : s.emissiveTableGpu.get();
            RenderBuffer *up = (t == 0) ? s.hitTableBuffer.get() : s.emissiveTableBuffer.get();
            const uint64_t bytes = (t == 0) ? s.hitTableBytes : s.emissiveTableBytes;
            const bool active = (t == 0) ? (s.hitCount > 0) : (s.emissiveCount > 0);
            if (!active || (gpu == nullptr) || (up == nullptr)) {
                continue;
            }
            worker->commandList->barriers(RenderBarrierStage::COPY, RenderBufferBarrier(gpu, RenderBufferAccess::WRITE));
            worker->commandList->copyBufferRegion(RenderBufferReference(gpu), RenderBufferReference(up), bytes);
            worker->commandList->barriers(RenderBarrierStage::GRAPHICS, RenderBufferBarrier(gpu, RenderBufferAccess::READ));
        }
        worker->commandList->buildTopLevelAS(s.tlas.get(), RenderBufferReference(s.tlasScratch.get()), RenderBufferReference(s.instancesBuffer.get()), tlasInfo);
        worker->commandList->barriers(RenderBarrierStage::GRAPHICS_AND_COMPUTE, RenderBufferBarrier(s.tlasBuffer.get(), RenderBufferAccess::READ));
        s.ready = true;
        if (snapshot) {
            RS64ShadowScene::History &h = s.history[s.historyNext % uint32_t(lcfg.shadowHistory)];
            s.retired.blas = std::move(h.blas);
            s.retired.blasBuffer = std::move(h.blasBuffer);
            h.blasBuffer = std::move(s.blasBuffer);
            h.blas = std::move(s.blas);
            memcpy(h.terrainTransform, s.terrainTransform, sizeof(h.terrainTransform));
            h.mapVersion = s.terrainMap->version;
            s.blasCapacity = 0;
            ++s.historyNext;
        }
    }

    void FramebufferRenderer::rs64TimingBegin(RenderWorker *worker) {
        rs64TimingMarks = 0;
        if (!rs64lights::config().logTiming) {
            return;
        }
        if (!rs64TimingPool) {
            rs64TimingPool = worker->device->createQueryPool(uint32_t(std::size(rs64TimingLabels)));
        }
        worker->commandList->resetQueryPool(rs64TimingPool.get(), 0, uint32_t(std::size(rs64TimingLabels)));
    }

    // Called once before and once after a pass; the pair's difference is that pass's GPU time.
    void FramebufferRenderer::rs64TimingMark(RenderWorker *worker, uint8_t pass) {
        if (!rs64TimingPool || (rs64TimingMarks >= std::size(rs64TimingLabels))) {
            return;
        }
        rs64TimingLabels[rs64TimingMarks] = pass;
        worker->commandList->writeTimestamp(rs64TimingPool.get(), rs64TimingMarks);
        ++rs64TimingMarks;
    }

    // After the workload's command list has executed and been waited on.
    void FramebufferRenderer::rs64TimingReport() {
        if (!rs64TimingPool) {
            return;
        }
        if (rs64TimingMarks >= 2) {
            rs64TimingPool->queryResults();
            const uint64_t *ts = rs64TimingPool->getResults();
            for (uint32_t i = 0; (i + 1) < rs64TimingMarks; i += 2) {
                rs64TimingSum[rs64TimingLabels[i]] += double(ts[i + 1] - ts[i]) / 1000000.0;
            }
        }
        if (++rs64TimingFrames >= 120) {
            const double n = double(rs64TimingFrames);
            const double shadows = rs64TimingSum[RS64TimeShadows] + rs64TimingSum[RS64TimeTrace] + rs64TimingSum[RS64TimeTemporal] + rs64TimingSum[RS64TimeBlur];
            std::fprintf(stderr, "[rt-timing] avg ms/workload over %u: scene=%.3f shadows=%.3f (trace=%.3f temporal=%.3f blur=%.3f) fog=%.3f lights=%.3f total=%.3f\n", rs64TimingFrames,
                rs64TimingSum[RS64TimeScene] / n, shadows / n, rs64TimingSum[RS64TimeTrace] / n, rs64TimingSum[RS64TimeTemporal] / n, rs64TimingSum[RS64TimeBlur] / n,
                rs64TimingSum[RS64TimeFog] / n, rs64TimingSum[RS64TimeLights] / n,
                (rs64TimingSum[RS64TimeScene] + shadows + rs64TimingSum[RS64TimeFog] + rs64TimingSum[RS64TimeLights]) / n);
            for (double &v : rs64TimingSum) {
                v = 0.0;
            }
            rs64TimingFrames = 0;
        }
    }

    void FramebufferRenderer::recordRS64Lights(RenderWorker *worker, const Framebuffer &framebuffer, bool &depthState) {
        const RenderTargetDrawCall &targetDrawCall = framebuffer.renderTargetDrawCall;
        submitDepthAccess(worker, targetDrawCall.fbStorage, true, depthState);
        // Shadowed variant when this frame's scene was built; otherwise the Phase 1 pass.
        const bool shadowed = framebuffer.rs64.lightsShadowed && rs64Shadow.ready && (rs64Shadow.tlas != nullptr);
        const ShaderRecord &record = shadowed ? (framebuffer.rs64.lightsDebug ? shaderLibrary->rs64LightsShadowedDebug : shaderLibrary->rs64LightsShadowed)
                                              : (framebuffer.rs64.lightsDebug ? shaderLibrary->rs64LightsDebug : shaderLibrary->rs64Lights);
        const RenderViewport &v = framebuffer.viewport;
        const RenderRect sc(int32_t(v.x), int32_t(v.y), int32_t(v.x + v.width), int32_t(v.y + v.height));
        const bool half = framebuffer.rs64.lightsHalfOn && (framebuffer.rs64.lightsHalf.framebuffer != nullptr);
        const ShaderRecord &pass = !half ? record : (shadowed ? shaderLibrary->rs64LightsShadowedHalf : shaderLibrary->rs64LightsHalf);
        if (half) {
            RenderTexture *halfTex = framebuffer.rs64.lightsHalf.texture.get();
            const RenderTextureBarrier toHalf[] = { RenderTextureBarrier(halfTex, RenderTextureLayout::COLOR_WRITE) };
            worker->commandList->barriers(RenderBarrierStage::GRAPHICS, toHalf, uint32_t(std::size(toHalf)));
            worker->commandList->setFramebuffer(framebuffer.rs64.lightsHalf.framebuffer.get());
            worker->commandList->clearColor(0, RenderColor(0.0f, 0.0f, 0.0f, 65000.0f));
            worker->commandList->setViewports(RenderViewport(v.x / 2.0f, v.y / 2.0f, v.width / 2.0f, v.height / 2.0f, v.minDepth, v.maxDepth));
            worker->commandList->setScissors(RenderRect(int32_t(v.x) / 2, int32_t(v.y) / 2, (int32_t(v.x + v.width) + 1) / 2, (int32_t(v.y + v.height) + 1) / 2));
        }
        else {
            worker->commandList->setViewports(v);
            worker->commandList->setScissors(sc);
        }
        worker->commandList->setPipeline(pass.pipeline.get());
        worker->commandList->setGraphicsPipelineLayout(pass.pipelineLayout.get());
        if (shadowed) {
            framebuffer.rs64.lightsShadowedSet->setAccelerationStructure(framebuffer.rs64.lightsShadowedSet->gScene, rs64Shadow.tlas.get());
            bindRS64Traced(worker, *framebuffer.rs64.lightsShadowedSet, framebuffer.rs64.lightsBuffer.get(), offsetof(interop::RS64LightsCB, shadowParams) + sizeof(float), true);
        }
        else {
            worker->commandList->setGraphicsDescriptorSet(framebuffer.rs64.lightsSet->get(), 0);
        }
        worker->commandList->setVertexBuffers(0, nullptr, 0, nullptr);
        worker->commandList->drawInstanced(3, 1, 0, 0);

        // Add the half-resolution lights to the frame, matching each pixel to the half texels at its distance.
        if (half) {
            RenderTexture *halfTex = framebuffer.rs64.lightsHalf.texture.get();
            RenderTexture *color = targetDrawCall.fbStorage->colorTarget->texture.get();
            const RenderTextureBarrier toUpsample[] = { RenderTextureBarrier(halfTex, RenderTextureLayout::SHADER_READ), RenderTextureBarrier(color, RenderTextureLayout::COLOR_WRITE) };
            worker->commandList->barriers(RenderBarrierStage::GRAPHICS, toUpsample, uint32_t(std::size(toUpsample)));
            worker->commandList->setFramebuffer(targetDrawCall.fbStorage->colorWriteDepthRead.get());
            worker->commandList->setViewports(v);
            worker->commandList->setScissors(sc);
            worker->commandList->setPipeline(shaderLibrary->rs64LightsUpsample.pipeline.get());
            worker->commandList->setGraphicsPipelineLayout(shaderLibrary->rs64LightsUpsample.pipelineLayout.get());
            worker->commandList->setGraphicsDescriptorSet(framebuffer.rs64.lightsUpsampleSet->get(), 0);
            worker->commandList->drawInstanced(3, 1, 0, 0);
        }

        if (framebuffer.rs64.panels) {
            worker->commandList->setPipeline(shaderLibrary->rs64LightsDebug.pipeline.get());
            worker->commandList->setGraphicsPipelineLayout(shaderLibrary->rs64LightsDebug.pipelineLayout.get());
            worker->commandList->setGraphicsDescriptorSet(framebuffer.rs64.panelsSet->get(), 0);
            worker->commandList->drawInstanced(3, 1, 0, 0);
        }
    }

    void FramebufferRenderer::RS64Target::ensure(RenderDevice *device, uint32_t w, uint32_t h, RenderFormat format) {
        if (texture && (width == w) && (height == h)) {
            return;
        }

        framebuffer.reset();
        texture = device->createTexture(RenderTextureDesc::ColorTarget(w, h, format));
        const RenderTexture *targets[] = { texture.get() };
        framebuffer = device->createFramebuffer(RenderFramebufferDesc(targets, 1));
        width = w;
        height = h;
    }

    void FramebufferRenderer::RS64FramebufferState::create(RenderDevice *device) {
        lightsBuffer = device->createBuffer(RenderBufferDesc::UploadBuffer(sizeof(interop::RS64LightsCB), RenderBufferFlag::CONSTANT));
        lightsSet = std::make_unique<RS64LightsDescriptorSet>(device);
        lightsUpsampleSet = std::make_unique<RS64LightsUpsampleDescriptorSet>(device);
        depthResolveSet = std::make_unique<RenderTargetCopyDescriptorSet>(device);
        panelsBuffer = device->createBuffer(RenderBufferDesc::UploadBuffer(sizeof(interop::RS64LightsCB), RenderBufferFlag::CONSTANT));
        panelsSet = std::make_unique<RS64LightsDescriptorSet>(device);
        if (device->getCapabilities().rayQuery) {
            shadowsBuffer = device->createBuffer(RenderBufferDesc::UploadBuffer(sizeof(interop::RS64ShadowsCB), RenderBufferFlag::CONSTANT));
            shadowsSet = std::make_unique<RS64TracedDescriptorSet>(device);
            lightsShadowedSet = std::make_unique<RS64TracedDescriptorSet>(device);
            fogBuffer = device->createBuffer(RenderBufferDesc::UploadBuffer(sizeof(interop::RS64FogShaftsCB), RenderBufferFlag::CONSTANT));
            fogSet = std::make_unique<RS64TracedDescriptorSet>(device);
            blurHSet = std::make_unique<RS64ShadowBlurDescriptorSet>(device);
            blurVSet = std::make_unique<RS64ShadowBlurDescriptorSet>(device);
            temporalSet = std::make_unique<RS64ShadowTemporalDescriptorSet>(device);
            mirrorSet = std::make_unique<RS64WaterMirrorDescriptorSet>(device);
            blurHBuffer = device->createBuffer(RenderBufferDesc::UploadBuffer(sizeof(interop::RS64ShadowBlurCB), RenderBufferFlag::CONSTANT));
            blurVBuffer = device->createBuffer(RenderBufferDesc::UploadBuffer(sizeof(interop::RS64ShadowBlurCB), RenderBufferFlag::CONSTANT));
        }
    }
};
