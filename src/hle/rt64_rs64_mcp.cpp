// RT64 - MCP bridge requests that read RT64 state, run on the gfx thread

#include "rt64_rs64_mcp.h"

#include <algorithm>
#include <atomic>
#include <deque>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <stdexcept>

#include "rt64_state.h"
#include "rt64_application.h"
#include "rt64_rs64_lights.h"
#include "render/rt64_raster_shader.h"

using nlohmann::json;

namespace RT64 {
    namespace rs64mcp {
        typedef json (*Handler)(State &state, const json &params);

        struct Request {
            uint32_t tag;
            json id;
            std::string method;
            json params;
        };

        static std::mutex queueMutex;
        static std::deque<Request> inbox;
        static std::deque<std::pair<uint32_t, json>> outbox;
        static std::atomic<int> pendingPause{ -1 };
        static std::atomic<bool> pendingReset{ false };
        static std::atomic<bool> pausedMirror{ false };
        static std::atomic<uint64_t> drains{ 0 };

        static json handlePause(State &state, const json &params) {
            state.debuggerInspector.paused = params.value("on", true);
            return { { "paused", state.debuggerInspector.paused } };
        }

        // Frozen frame: redraw it as the in-between frame at this weight (0 = previous frame, 1 = this one), as the F1 inspector's slider does.
        static json handleInterpWeight(State &state, const json &params) {
            float &w = state.debuggerInspector.renderer.interpolationWeight;
            if (params.contains("weight") && params["weight"].is_number()) {
                w = std::clamp(params["weight"].get<float>(), 0.0f, 1.0f);
            }

            return { { "weight", w } };
        }

        static json handleStatus(State &state, const json &) {
            const DebuggerInspector &inspector = state.debuggerInspector;
            return {
                { "paused", inspector.paused }, { "workload_counter", state.workloadCounter }, { "framebuffer_index", inspector.renderer.framebufferIndex },
                { "draw_index", inspector.renderer.globalDrawCallIndex }, { "free_cam", inspector.camera.enabled },
                { "texture_dumping", !state.dumpingTexturesDirectory.empty() }
            };
        }

        static std::string hex(uint64_t v) {
            char buf[24];
            snprintf(buf, sizeof(buf), "0x%llX", (unsigned long long)v);
            return buf;
        }

        static const char *projectionTypeName(Projection::Type type) {
            switch (type) {
            case Projection::Type::Perspective:
                return "perspective";
            case Projection::Type::Orthographic:
                return "orthographic";
            case Projection::Type::Rectangle:
                return "rectangle";
            case Projection::Type::Triangle:
                return "triangle";
            default:
                return "none";
            }
        }

        // The frozen frame: the workload RT64 re-submits while paused. Errors unless paused, so its arrays aren't being written.
        static Workload &frozenWorkload(State &state) {
            if (!state.debuggerInspector.paused) {
                throw std::runtime_error("not paused; use pause_at first");
            }

            return state.ext.workloadQueue->workloads[state.ext.workloadQueue->previousWriteCursor()];
        }

        static uint32_t indexParam(const json &params, const char *name, uint32_t count) {
            if (!params.contains(name) || !params[name].is_number_unsigned() || (params[name].get<uint64_t>() >= count)) {
                throw std::runtime_error(std::string(name) + " must be an integer in [0, " + std::to_string(count) + ")");
            }

            return params[name].get<uint32_t>();
        }

        static json matrixJson(const interop::float4x4 &m) {
            json rows = json::array();
            for (int i = 0; i < 4; i++) {
                rows.push_back({ m[i][0], m[i][1], m[i][2], m[i][3] });
            }

            return rows;
        }

        static json transformGroupJson(const TransformGroup &g) {
            return {
                { "matrix_id", hex(g.matrixId) }, { "position", g.positionInterpolation }, { "rotation", g.rotationInterpolation },
                { "scale", g.scaleInterpolation }, { "skew", g.skewInterpolation }, { "perspective", g.perspectiveInterpolation },
                { "vertex", g.vertexInterpolation }, { "texcoord", g.texcoordInterpolation }, { "tile", g.tileInterpolation },
                { "look_at", g.lookAtInterpolation }, { "ordering", g.ordering }, { "aspect", g.aspectMode }, { "editable", g.editable }
            };
        }

        static json handleFrameSummary(State &state, const json &) {
            const Workload &workload = frozenWorkload(state);
            json fbPairs = json::array();
            uint32_t draws = 0;
            uint32_t triangles = 0;
            uint32_t syncs = 0;
            for (uint32_t f = 0; f < workload.fbPairCount; f++) {
                const FramebufferPair &fbPair = workload.fbPairs[f];
                json projections = json::array();
                for (uint32_t p = 0; p < fbPair.projectionCount; p++) {
                    const Projection &proj = fbPair.projections[p];
                    projections.push_back({ { "index", p }, { "type", projectionTypeName(proj.type) }, { "draws", proj.gameCallCount } });
                    draws += proj.gameCallCount;
                    for (uint32_t d = 0; d < proj.gameCallCount; d++) {
                        triangles += proj.gameCalls[d].callDesc.triangleCount;
                    }
                }

                if (fbPair.syncRequired || (f == workload.fbPairCount - 1)) {
                    syncs++;
                }

                fbPairs.push_back({
                    { "index", f }, { "color_addr", hex(fbPair.colorImage.address) }, { "depth_addr", hex(fbPair.depthImage.address) }, { "dl_addr", hex(fbPair.displayListAddress) },
                    { "width", fbPair.colorImage.width }, { "fmt", fbPair.colorImage.fmt }, { "siz", fbPair.colorImage.siz }, { "projections", projections }
                });
            }

            return { { "draws", draws }, { "triangles", triangles }, { "fb_pairs", fbPairs }, { "rdram_syncs", syncs } };
        }

        static json drawRow(const Workload &workload, uint32_t f, uint32_t p, uint32_t d) {
            const DrawCall &call = workload.fbPairs[f].projections[p].gameCalls[d].callDesc;
            const DrawData &drawData = workload.drawData;
            json hashes = json::array();
            for (uint32_t t = 0; t < call.tileCount; t++) {
                hashes.push_back(hex(drawData.callTiles[call.tileIndex + t].tmemHashOrID));
            }

            json row = {
                { "fb", f }, { "proj", p }, { "call", d }, { "tris", call.triangleCount }, { "tiles", call.tileCount }, { "tmem_hashes", hashes },
                { "combiner", { hex(call.colorCombiner.L), hex(call.colorCombiner.H) } }, { "other_mode", { hex(call.otherMode.L), hex(call.otherMode.H) } },
                { "geometry_mode", hex(call.geometryMode) }, { "min_world", call.minWorldMatrix }, { "max_world", call.maxWorldMatrix },
                { "min_world_addr", hex(drawData.worldTransformPhysicalAddresses[call.minWorldMatrix]) },
                { "prim", { call.rdpParams.primColor[0], call.rdpParams.primColor[1], call.rdpParams.primColor[2], call.rdpParams.primColor[3] } }
            };

            // The light-preset key of the model part drawn with this transform, for mesh and emissive presets.
            const auto key = workload.rs64MeshKeys.find(call.minWorldMatrix);
            if (key != workload.rs64MeshKeys.end()) {
                row["mesh_key"] = key->second;
            }

            return row;
        }

        static json handleListDraws(State &state, const json &params) {
            const Workload &workload = frozenWorkload(state);
            const uint32_t offset = params.value("offset", 0u);
            const uint32_t limit = params.value("limit", 500u);
            const bool oneFb = params.contains("fb");
            const uint32_t fbOnly = oneFb ? indexParam(params, "fb", workload.fbPairCount) : 0;
            json rows = json::array();
            uint32_t total = 0;
            for (uint32_t f = 0; f < workload.fbPairCount; f++) {
                if (oneFb && (f != fbOnly)) {
                    continue;
                }

                const FramebufferPair &fbPair = workload.fbPairs[f];
                const bool oneProj = params.contains("proj");
                const uint32_t projOnly = oneProj ? indexParam(params, "proj", fbPair.projectionCount) : 0;
                for (uint32_t p = 0; p < fbPair.projectionCount; p++) {
                    if (oneProj && (p != projOnly)) {
                        continue;
                    }

                    for (uint32_t d = 0; d < fbPair.projections[p].gameCallCount; d++, total++) {
                        if ((total >= offset) && (rows.size() < limit)) {
                            rows.push_back(drawRow(workload, f, p, d));
                        }
                    }
                }
            }

            return { { "total", total }, { "rows", rows } };
        }

        static json loadJson(const LoadOperation &op) {
            json j = { { "type", op.type }, { "texture", op.texture }, { "tile", op.tile } };
            switch (op.type) {
            case LoadOperation::Type::Tile:
                j["operation"] = { { "tile", op.operationTile.tile }, { "uls", op.operationTile.uls }, { "ult", op.operationTile.ult }, { "lrs", op.operationTile.lrs }, { "lrt", op.operationTile.lrt } };
                break;
            case LoadOperation::Type::Block:
                j["operation"] = { { "tile", op.operationBlock.tile }, { "uls", op.operationBlock.uls }, { "ult", op.operationBlock.ult }, { "lrs", op.operationBlock.lrs }, { "dxt", op.operationBlock.dxt } };
                break;
            case LoadOperation::Type::TLUT:
                j["operation"] = { { "tile", op.operationTLUT.tile }, { "uls", op.operationTLUT.uls }, { "ult", op.operationTLUT.ult }, { "lrs", op.operationTLUT.lrs }, { "lrt", op.operationTLUT.lrt } };
                break;
            }

            return j;
        }

        static json vertexList(const DrawData &drawData, const GameCall &gameCall, bool indexed) {
            const DrawCall &call = gameCall.callDesc;
            json verts = json::array();
            if (indexed) {
                const uint32_t end = gameCall.meshDesc.faceIndicesStart + call.triangleCount * 3;
                for (uint32_t i = gameCall.meshDesc.faceIndicesStart; i < end; i++) {
                    const uint32_t v = drawData.faceIndices[i];
                    verts.push_back({
                        { "index", v }, { "pos", { drawData.posFloats[v * 3 + 0], drawData.posFloats[v * 3 + 1], drawData.posFloats[v * 3 + 2] } },
                        { "tc", { drawData.tcFloats[v * 2 + 0], drawData.tcFloats[v * 2 + 1] } },
                        { "norm_col", { drawData.normColBytes[v * 4 + 0], drawData.normColBytes[v * 4 + 1], drawData.normColBytes[v * 4 + 2], drawData.normColBytes[v * 4 + 3] } },
                        { "screen", { drawData.posScreen[v][0], drawData.posScreen[v][1], drawData.posScreen[v][2] } }
                    });
                }

                return verts;
            }

            const uint32_t end = gameCall.meshDesc.rawVertexStart + call.triangleCount * 3;
            for (uint32_t v = gameCall.meshDesc.rawVertexStart; v < end; v++) {
                verts.push_back({
                    { "pos", { drawData.triPosFloats[v * 4 + 0], drawData.triPosFloats[v * 4 + 1], drawData.triPosFloats[v * 4 + 2], drawData.triPosFloats[v * 4 + 3] } },
                    { "tc", { drawData.triTcFloats[v * 2 + 0], drawData.triTcFloats[v * 2 + 1] } },
                    { "color", { drawData.triColorFloats[v * 4 + 0], drawData.triColorFloats[v * 4 + 1], drawData.triColorFloats[v * 4 + 2], drawData.triColorFloats[v * 4 + 3] } }
                });
            }

            return verts;
        }

        static json handleGetDraw(State &state, const json &params) {
            const Workload &workload = frozenWorkload(state);
            const DrawData &drawData = workload.drawData;
            const uint32_t f = indexParam(params, "fb", workload.fbPairCount);
            const uint32_t p = indexParam(params, "proj", workload.fbPairs[f].projectionCount);
            const Projection &proj = workload.fbPairs[f].projections[p];
            const uint32_t d = indexParam(params, "call", proj.gameCallCount);
            const GameCall &gameCall = proj.gameCalls[d];
            const DrawCall &call = gameCall.callDesc;
            json j = drawRow(workload, f, p, d);
            j["projection_type"] = projectionTypeName(proj.type);

            json changed = json::array();
            const DrawStatus status(call.drawStatusChanges);
            for (uint32_t i = 0; i < uint32_t(DrawAttribute::Count); i++) {
                if (status.isChanged(DrawAttribute(i))) {
                    changed.push_back(DrawCall::attributeName(DrawAttribute(i)));
                }
            }

            j["attributes_changed"] = changed;
            if (proj.type == Projection::Type::Rectangle) {
                j["rect"] = { { "ulx", call.rect.ulx }, { "uly", call.rect.uly }, { "lrx", call.rect.lrx }, { "lry", call.rect.lry }, { "dsdx", call.rectDsdx }, { "dtdy", call.rectDtdy },
                    { "left_origin", call.rectLeftOrigin }, { "right_origin", call.rectRightOrigin } };
            }

            if (proj.usesViewport()) {
                auto matrixInfo = [&](uint32_t m) {
                    return json{
                        { "index", m }, { "segmented_addr", hex(drawData.worldTransformSegmentedAddresses[m]) }, { "physical_addr", hex(drawData.worldTransformPhysicalAddresses[m]) },
                        { "group", transformGroupJson(drawData.transformGroups[drawData.worldTransformGroups[m]]) }, { "matrix", matrixJson(drawData.worldTransforms[m]) }
                    };
                };

                j["min_world_matrix"] = matrixInfo(call.minWorldMatrix);
                j["max_world_matrix"] = matrixInfo(call.maxWorldMatrix);
                j["face_indices_start"] = gameCall.meshDesc.faceIndicesStart;
            }

            json loads = json::array();
            for (uint32_t l = 0; l < call.loadCount; l++) {
                loads.push_back(loadJson(drawData.loadOperations[call.loadIndex + l]));
            }

            json tiles = json::array();
            for (uint32_t t = 0; t < call.tileCount; t++) {
                const DrawCallTile &tile = drawData.callTiles[call.tileIndex + t];
                json tj = { { "load_tile", tile.loadTile }, { "sample_width", tile.sampleWidth }, { "sample_height", tile.sampleHeight }, { "line_width", tile.lineWidth }, { "tlut", tile.tlut } };
                if (tile.validTexcoords()) {
                    tj["min_texcoord"] = { tile.minTexcoord.x, tile.minTexcoord.y };
                    tj["max_texcoord"] = { tile.maxTexcoord.x, tile.maxTexcoord.y };
                }

                tj[tile.tileCopyUsed ? "tile_copy_id" : "tmem_hash"] = hex(tile.tmemHashOrID);
                tiles.push_back(tj);
            }

            const auto &rdp = call.rdpParams;
            const int32_t *K = rdp.convertK;
            j["rdp"] = {
                { "prim_color", { rdp.primColor[0], rdp.primColor[1], rdp.primColor[2], rdp.primColor[3] } }, { "prim_lod", { rdp.primLOD[0], rdp.primLOD[1] } },
                { "prim_depth", { rdp.primDepth[0], rdp.primDepth[1] } }, { "env_color", { rdp.envColor[0], rdp.envColor[1], rdp.envColor[2], rdp.envColor[3] } },
                { "fog_color", { rdp.fogColor[0], rdp.fogColor[1], rdp.fogColor[2], rdp.fogColor[3] } }, { "blend_color", { rdp.blendColor[0], rdp.blendColor[1], rdp.blendColor[2], rdp.blendColor[3] } },
                { "convert_k", { K[0], K[1], K[2], K[3], K[4], K[5] } }, { "key_center", { rdp.keyCenter[0], rdp.keyCenter[1], rdp.keyCenter[2] } },
                { "key_scale", { rdp.keyScale[0], rdp.keyScale[1], rdp.keyScale[2] } },
                { "scissor", { call.scissorRect.ulx, call.scissorRect.uly, call.scissorRect.lrx, call.scissorRect.lry } }, { "scissor_mode", call.scissorMode }
            };

            // The call's generated raster shaders, as the Debugger's "Dump Pixel/Vertex Shader" buttons write them.
            if (params.contains("shader_dir") && params["shader_dir"].is_string()) {
                const std::filesystem::path dir = std::filesystem::u8path(params["shader_dir"].get<std::string>());
                const std::string base = "call_" + std::to_string(f) + "_" + std::to_string(p) + "_" + std::to_string(d);
                const RasterShaderText text = RasterShader::generateShaderText(gameCall.shaderDesc, true);
                std::filesystem::create_directories(dir);
                std::ofstream(dir / (base + ".ps.hlsl")) << text.pixelShader;
                std::ofstream(dir / (base + ".vs.hlsl")) << text.vertexShader;
                j["shader_files"] = { { "pixel", (dir / (base + ".ps.hlsl")).u8string() }, { "vertex", (dir / (base + ".vs.hlsl")).u8string() } };
            }

            j["loads"] = loads;
            j["tile_list"] = tiles;
            j["fill_color"] = hex(call.fillColor);
            j["texture_on"] = call.textureOn;
            j["texture_tile"] = call.textureTile;
            j["texture_levels"] = call.textureLevels;
            if (params.value("vertices", false)) {
                j["vertices"] = vertexList(drawData, gameCall, proj.usesViewport());
            }

            return j;
        }

        // Writes the tile's TMEM bytes to `path` and its decode parameters to `path`.json (tools/debug/mcp/tmem_decode.py reads both).
        static json handleDumpTmem(State &state, const json &params) {
            const Workload &workload = frozenWorkload(state);
            const uint32_t f = indexParam(params, "fb", workload.fbPairCount);
            const uint32_t p = indexParam(params, "proj", workload.fbPairs[f].projectionCount);
            const Projection &proj = workload.fbPairs[f].projections[p];
            const uint32_t d = indexParam(params, "call", proj.gameCallCount);
            const DrawCall &call = proj.gameCalls[d].callDesc;
            const uint32_t t = indexParam(params, "tile", call.tileCount);
            const DrawCallTile &tile = workload.drawData.callTiles[call.tileIndex + t];
            if (tile.tileCopyUsed) {
                throw std::runtime_error("tile is a framebuffer copy; use view_framebuffer");
            }

            if (!params.contains("path") || !params["path"].is_string()) {
                throw std::runtime_error("path (string) required");
            }

            uint32_t textureIndex = 0;
            const Texture *texture = state.ext.textureCache->useTexture(tile.tmemHashOrID, workload.submissionFrame, textureIndex) ? state.ext.textureCache->getTexture(textureIndex) : nullptr;
            if ((texture == nullptr) || texture->bytesTMEM.empty()) {
                throw std::runtime_error("texture " + hex(tile.tmemHashOrID) + " is not in the texture cache");
            }

            const std::filesystem::path path = std::filesystem::u8path(params["path"].get<std::string>());
            std::filesystem::create_directories(path.parent_path());
            std::ofstream o(path, std::ios::binary);
            o.write(reinterpret_cast<const char *>(texture->bytesTMEM.data()), texture->bytesTMEM.size());
            if (!o) {
                throw std::runtime_error("could not write " + path.u8string());
            }

            const LoadTile &lt = tile.loadTile;
            const json meta = {
                { "fmt", lt.fmt }, { "siz", lt.siz }, { "line", lt.line }, { "tmem", lt.tmem }, { "palette", lt.palette }, { "tlut", tile.tlut },
                { "uls", lt.uls }, { "ult", lt.ult }, { "lrs", lt.lrs }, { "lrt", lt.lrt }, { "masks", lt.masks }, { "maskt", lt.maskt }, { "cms", lt.cms }, { "cmt", lt.cmt },
                { "width", tile.sampleWidth }, { "height", tile.sampleHeight }, { "tmem_hash", hex(tile.tmemHashOrID) }
            };

            const std::filesystem::path jsonPath = path.u8string() + ".json";
            std::ofstream(jsonPath) << meta.dump(2);
            return { { "path", path.u8string() }, { "json_path", jsonPath.u8string() }, { "bytes", texture->bytesTMEM.size() }, { "meta", meta } };
        }

        static json handleTextureDumping(State &state, const json &params) {
            if (params.contains("dir") && params["dir"].is_string()) {
                state.dumpingTexturesDirectory = std::filesystem::u8path(params["dir"].get<std::string>());
                std::filesystem::create_directories(state.dumpingTexturesDirectory);
            }
            else {
                state.dumpingTexturesDirectory.clear();
            }

            return { { "dir", state.dumpingTexturesDirectory.empty() ? json() : json(state.dumpingTexturesDirectory.u8string()) } };
        }

        // Shows one framebuffer pair's color (or depth) image in place of the frame, as the F1 Debugger's "View Framebuffer" slider does; -1 turns it off.
        static json handleViewFramebuffer(State &state, const json &params) {
            const Workload &workload = frozenWorkload(state);
            DebuggerRenderer &renderer = state.debuggerInspector.renderer;
            const int index = params.value("index", -1);
            if ((index < -1) || (index >= int(workload.fbPairCount))) {
                throw std::runtime_error("index must be -1 or in [0, " + std::to_string(workload.fbPairCount) + ")");
            }

            renderer.framebufferIndex = index;
            renderer.framebufferDepth = params.value("depth", false);
            if (index >= 0) {
                const FramebufferPair &fbPair = workload.fbPairs[index];
                renderer.framebufferAddress = renderer.framebufferDepth ? fbPair.depthImage.address : fbPair.colorImage.address;
            }

            return { { "index", index }, { "depth", renderer.framebufferDepth }, { "address", hex(renderer.framebufferAddress) } };
        }

        // Draws the frame only up to one global draw call, as the F1 Debugger's "View Draw Call" slider does; -1 turns it off.
        static json handleViewDraw(State &state, const json &params) {
            const Workload &workload = frozenWorkload(state);
            const int index = params.value("index", -1);
            if ((index < -1) || (index > int(workload.gameCallCount))) {
                throw std::runtime_error("index must be -1 or in [0, " + std::to_string(workload.gameCallCount) + "]");
            }

            state.debuggerInspector.renderer.globalDrawCallIndex = index;
            return { { "index", index }, { "draws", workload.gameCallCount } };
        }

        static json rowJson(const hlslpp::float4x4 &m, int row, int count) {
            float v[4];
            hlslpp::store(m[row], v);
            return json(std::vector<float>(v, v + count));
        }

        static hlslpp::float3 float3Param(const json &params, const char *name) {
            const json &v = params[name];
            if (!v.is_array() || (v.size() != 3) || !v[0].is_number() || !v[1].is_number() || !v[2].is_number()) {
                throw std::runtime_error(std::string(name) + " must be [x, y, z]");
            }

            return hlslpp::float3(v[0].get<float>(), v[1].get<float>(), v[2].get<float>());
        }

        static json handleGetCamera(State &state, const json &) {
            const DebuggerCamera &camera = state.debuggerInspector.camera;
            json view = json::array();
            for (int i = 0; i < 4; i++) {
                for (float f : rowJson(camera.viewMatrix, i, 4)) {
                    view.push_back(f);
                }
            }

            return {
                { "enabled", camera.enabled }, { "position", rowJson(camera.invViewMatrix, 3, 3) }, { "forward", rowJson(camera.invViewMatrix, 2, 3) },
                { "near", camera.nearPlane }, { "far", camera.farPlane }, { "fov", camera.fov }, { "view", view }
            };
        }

        // RT64's free camera on the frozen frame: seeded from the first perspective projection, then placed (position + target), moved ([right, up, forward]) or rotated ([yaw, pitch] radians).
        static json handleFreeCam(State &state, const json &params) {
            DebuggerCamera &camera = state.debuggerInspector.camera;
            if (!params.value("enabled", true)) {
                camera.enabled = false;
                return handleGetCamera(state, params);
            }

            const Workload &workload = frozenWorkload(state);
            if (!camera.enabled) {
                bool seeded = false;
                for (uint32_t f = 0; (f < workload.fbPairCount) && !seeded; f++) {
                    for (uint32_t p = 0; (p < workload.fbPairs[f].projectionCount) && !seeded; p++) {
                        if (workload.fbPairs[f].projections[p].type == Projection::Type::Perspective) {
                            state.debuggerInspector.enableFreeCamera(workload, f, p);
                            seeded = true;
                        }
                    }
                }

                if (!seeded) {
                    throw std::runtime_error("the frozen frame has no perspective projection");
                }
            }

            // The camera looks along +row 2 of invView, which lookAtPerspective points away from its focus, so the focus is mirrored through the position.
            if (params.contains("position") || params.contains("target")) {
                const hlslpp::float3 position = float3Param(params, "position");
                state.cameraController.lookAtPerspective(camera, position, position * 2.0f - float3Param(params, "target"));
            }

            // Same signs as the free camera's keys (D right, E up, W forward).
            if (params.contains("move")) {
                const hlslpp::float3 m = float3Param(params, "move");
                state.cameraController.movePerspective(camera, hlslpp::float3(-m.x, m.y, -m.z));
            }

            if (params.contains("rotate")) {
                const json &r = params["rotate"];
                if (!r.is_array() || (r.size() != 2) || !r[0].is_number() || !r[1].is_number()) {
                    throw std::runtime_error("rotate must be [yaw, pitch]");
                }

                state.cameraController.rotatePerspective(camera, r[0].get<float>(), r[1].get<float>());
            }

            return handleGetCamera(state, params);
        }

        static std::string nameParam(const json &params) {
            if (!params.contains("name") || !params["name"].is_string() || params["name"].get<std::string>().empty()) {
                throw std::runtime_error("name (non-empty string) required");
            }

            return params["name"].get<std::string>();
        }

        // RT64's light presets (the F1 Lights tab, roguesq_lights.json schema); the light pass picks edits up on the next drawn frame.
        static json handleLightsList(State &state, const json &) {
            json presets = json::object();
            for (const auto &it : state.lightsLibrary.presetMap) {
                json preset;
                it.second.writeJson(preset);
                presets[it.first] = preset;
            }

            return { { "level", state.lightsLibraryInspector.currentLevel }, { "presets", presets } };
        }

        // The lights the last light pass shone (camera space, after the 32-slot selection) and the sun the last shadow pass traced.
        static json handleLightsActive(State &, const json &) {
            static const char *const kKinds[] = { "none", "sprite", "laser", "mesh", "lamp_mark" };
            const rs64lights::ActiveLights a = rs64lights::activeLights();
            json lights = json::array();
            for (const rs64lights::Resolved &r : a.lights) {
                json l = {
                    { "kind", (r.kind < 5) ? kKinds[r.kind] : "?" }, { "priority", r.priority }, { "pos", { r.pos[0], r.pos[1], r.pos[2] } },
                    { "radius", r.radius }, { "color", { r.color[0], r.color[1], r.color[2] } }, { "intensity", r.intensity }, { "falloff", r.falloff },
                    { "shadow_start", r.shadowStart }
                };
                if (r.engine) l["engine"] = true;
                if (r.projectile) l["projectile"] = true;
                if (r.lampGated) l["lamp_gated"] = true;
                if (r.flicker > 0.0f) l["flicker"] = r.flicker;
                if ((r.rectU[0] != 0.0f) || (r.rectU[1] != 0.0f) || (r.rectU[2] != 0.0f)) {
                    l["panel"] = { { "half_u", { r.rectU[0], r.rectU[1], r.rectU[2] } }, { "half_v", { r.rectV[0], r.rectV[1], r.rectV[2] } }, { "facing", { r.dir[0], r.dir[1], r.dir[2] } } };
                }
                if (r.spotMaxCos < 1.0f) {
                    l["spot"] = { { "dir", { r.dir[0], r.dir[1], r.dir[2] } }, { "max_cos", r.spotMaxCos }, { "falloff_cos", r.spotFalloffCos } };
                }
                lights.push_back(l);
            }

            json sun = nullptr;
            if (a.sunTraced) {
                const rs64lights::Sun &s = a.sun;
                sun = {
                    { "valid", s.valid }, { "dir", { s.dir[0], s.dir[1], s.dir[2] } }, { "color", { s.color[0], s.color[1], s.color[2] } },
                    { "from_model", s.fromModel }, { "floor_up", { s.skip[0], s.skip[1], s.skip[2] } }, { "level_world", s.levelWorld },
                    { "strength", rs64lights::config().shadowStrength }
                };
            }

            return { { "lights", lights }, { "candidates", a.candidates }, { "sun", sun } };
        }

        static json handleLightsSet(State &state, const json &params) {
            const std::string name = nameParam(params);
            if (!params.contains("preset") || !params["preset"].is_object()) {
                throw std::runtime_error("preset (object) required");
            }

            PresetLights preset;
            bool parsed = false;
            try {
                parsed = preset.readJson(params["preset"]);
            }
            catch (const std::exception &e) {
                throw std::runtime_error(std::string("malformed preset: ") + e.what());
            }

            if (!parsed) {
                throw std::runtime_error("malformed preset: needs a lights array");
            }

            state.lightsLibrary.presetMap[name] = preset;
            return { { "name", name } };
        }

        static json handleLightsRemove(State &state, const json &params) {
            return { { "removed", state.lightsLibrary.presetMap.erase(nameParam(params)) > 0 } };
        }

        static json handleLightsSave(State &state, const json &params) {
            if (!params.contains("path") || !params["path"].is_string()) {
                throw std::runtime_error("path (string) required");
            }

            const std::filesystem::path path = std::filesystem::u8path(params["path"].get<std::string>());
            if (!state.lightsLibrary.save(path)) {
                throw std::runtime_error("could not write " + path.u8string());
            }

            return { { "path", path.u8string() } };
        }

        // One series of the Render tab's Frametimes plot (ms), oldest first; unwritten slots (0) are skipped.
        static json profilerJson(const ProfilingTimer &timer, bool withSamples) {
            std::vector<double> samples;
            const size_t size = timer.size();
            for (size_t i = 0; i < size; i++) {
                const double v = timer.data()[(timer.index() + i) % size];
                if (v > 0.0) {
                    samples.push_back(v);
                }
            }

            json j = { { "count", samples.size() } };
            if (!samples.empty()) {
                std::vector<double> sorted = samples;
                std::sort(sorted.begin(), sorted.end());
                double sum = 0.0;
                for (double v : sorted) {
                    sum += v;
                }

                j["min"] = sorted.front();
                j["avg"] = sum / sorted.size();
                j["max"] = sorted.back();
                j["p99"] = sorted[std::min(sorted.size() - 1, size_t(sorted.size() * 0.99))];
            }

            if (withSamples) {
                j["samples"] = samples;
            }

            return j;
        }

        static json handleFrameTimes(State &state, const json &params) {
            const bool withSamples = params.value("samples", false);
            return {
                { "present", profilerJson(state.ext.presentQueue->presentProfiler, withSamples) },
                { "renderer_cpu", profilerJson(state.ext.workloadQueue->rendererCPUProfiler, withSamples) },
                { "renderer_gpu", profilerJson(state.ext.workloadQueue->rendererGPUProfiler, withSamples) },
                { "matching", profilerJson(state.ext.workloadQueue->matchingProfiler, withSamples) },
                { "workload", profilerJson(state.ext.workloadQueue->workloadProfiler, withSamples) },
                { "dl_api", profilerJson(*state.ext.dlApiProfiler, withSamples) },
                { "dl_cpu", profilerJson(state.dlCpuProfiler, withSamples) },
                { "update_screen_api", profilerJson(*state.ext.screenApiProfiler, withSamples) },
                { "vi_changed", profilerJson(state.viChangedProfiler, withSamples) },
                { "update_screen_cpu", profilerJson(state.screenCpuProfiler, withSamples) }
            };
        }

        // The draw calls under a framebuffer pixel (native coordinates, as vi_info's fb_size), front-most first: the F1 right-click picker.
        static json handlePick(State &state, const json &params) {
            Workload &workload = frozenWorkload(state);
            if (!params.contains("x") || !params.contains("y") || !params["x"].is_number() || !params["y"].is_number()) {
                throw std::runtime_error("x and y (numbers, native framebuffer pixels) required");
            }

            json rows = json::array();
            const hlslpp::float2 cursor(params["x"].get<float>(), params["y"].get<float>());
            for (const DebuggerInspector::CallIndices &c : state.debuggerInspector.pickCalls(workload, cursor)) {
                rows.push_back(drawRow(workload, c.fbPairIndex, c.projectionIndex, c.drawCallIndex));
            }

            return { { "draws", rows } };
        }

        // Tints one draw call of the frozen frame on screen (RT64's debugger highlight), or clears every tint.
        static json handleHighlight(State &state, const json &params) {
            Workload &workload = frozenWorkload(state);
            if (params.value("clear", false)) {
                for (uint32_t f = 0; f < workload.fbPairCount; f++) {
                    for (uint32_t p = 0; p < workload.fbPairs[f].projectionCount; p++) {
                        for (uint32_t d = 0; d < workload.fbPairs[f].projections[p].gameCallCount; d++) {
                            workload.fbPairs[f].projections[p].gameCalls[d].debuggerDesc.highlightColor = 0;
                        }
                    }
                }

                return { { "cleared", true } };
            }

            const uint32_t f = indexParam(params, "fb", workload.fbPairCount);
            const uint32_t p = indexParam(params, "proj", workload.fbPairs[f].projectionCount);
            Projection &proj = workload.fbPairs[f].projections[p];
            const uint32_t d = indexParam(params, "call", proj.gameCallCount);
            const uint32_t color = params.value("color", 0xFF00FFA0u);
            proj.gameCalls[d].debuggerDesc.highlightColor = color;
            return { { "fb", f }, { "proj", p }, { "call", d }, { "color", hex(color) } };
        }

        static json handleFrameWarnings(State &state, const json &) {
            json warnings = json::array();
            for (const CommandWarning &w : frozenWorkload(state).commandWarnings) {
                json j = { { "message", w.message } };
                switch (w.indexType) {
                case CommandWarning::IndexType::LoadIndex:
                    j["load"] = w.load.index;
                    break;
                case CommandWarning::IndexType::TileIndex:
                    j["tile"] = w.tile.index;
                    break;
                case CommandWarning::IndexType::CallIndex:
                    j["call"] = { { "fb", w.call.fbPairIndex }, { "proj", w.call.projIndex }, { "call", w.call.callIndex } };
                    break;
                }

                warnings.push_back(j);
            }

            return { { "warnings", warnings } };
        }

        static json handleSpriteCommands(State &state, const json &) {
            json commands = json::array();
            for (const SpriteCommand &s : frozenWorkload(state).spriteCommands) {
                commands.push_back({ { "fb", s.fbPairIndex }, { "proj", s.projIndex }, { "call", s.callIndex }, { "calls", s.callCount }, { "replacement_hash", hex(s.replacementHash) } });
            }

            return { { "commands", commands } };
        }

        // The Debugger tab's Fog / Lights / LookAt / Modify sections for the frozen frame.
        static json handleRspState(State &state, const json &) {
            const DrawData &drawData = frozenWorkload(state).drawData;
            json fog = json::array();
            for (const interop::RSPFog &f : drawData.rspFog) {
                fog.push_back({ { "mul", f.mul }, { "offset", f.offset } });
            }

            json lights = json::array();
            for (const interop::RSPLight &l : drawData.rspLights) {
                lights.push_back({
                    { "pos_dir", { l.posDir.x, l.posDir.y, l.posDir.z } }, { "col", { l.col.x, l.col.y, l.col.z } }, { "colc", { l.colc.x, l.colc.y, l.colc.z } },
                    { "kc", l.kc }, { "kl", l.kl }, { "kq", l.kq }
                });
            }

            json lookAt = json::array();
            for (const interop::RSPLookAt &l : drawData.rspLookAt) {
                lookAt.push_back({ { "x", { l.x.x, l.x.y, l.x.z } }, { "y", { l.y.x, l.y.y, l.y.z } } });
            }

            json modify = json::array();
            for (size_t m = 0; m < drawData.modifyCount(); m++) {
                modify.push_back({ { "z", (drawData.modifyPosUints[m * 2] & 0x1) != 0 }, { "index", drawData.modifyPosUints[m * 2] >> 1 }, { "value", drawData.modifyPosUints[m * 2 + 1] } });
            }

            return { { "fog", fog }, { "lights", lights }, { "look_at", lookAt }, { "modify", modify } };
        }

        static json handleViInfo(State &state, const json &) {
            const VI &vi = state.lastScreenVI;
            const hlslpp::uint2 size = vi.fbSize();
            return {
                { "status", hex(vi.status.word) }, { "origin", hex(vi.origin) }, { "width", vi.width }, { "intr", vi.intr }, { "v_current_line", vi.vCurrentLine },
                { "burst", hex(vi.burst.word) }, { "v_sync", vi.vSync }, { "h_sync", hex(vi.hSync.word) }, { "leap", hex(vi.leap.word) },
                { "h_region", hex(vi.hRegion.word) }, { "v_region", hex(vi.vRegion.word) }, { "v_burst", hex(vi.vBurst.word) },
                { "x_transform", hex(vi.xTransform.word) }, { "y_transform", hex(vi.yTransform.word) },
                { "fb_address", hex(vi.fbAddress()) }, { "fb_size", { uint32_t(size.x), uint32_t(size.y) } }, { "visible", vi.visible() }, { "pixel_aspect", vi.pixelAspect() }
            };
        }

        static json enhancementJson(const EnhancementConfiguration &e) {
            return {
                { "presentationMode", int(e.presentation.mode) }, { "removeBlackBorders", e.presentation.removeBlackBorders },
                { "fixRectLR", e.rect.fixRectLR }, { "scaleLOD", e.textureLOD.scale }
            };
        }

        static json handleGetConfig(State &state, const json &) {
            return { { "user", *state.ext.userConfig }, { "enhancement", enhancementJson(*state.ext.enhancementConfig) } };
        }

        // Applies the Configuration tab's settings the way the F1 tab does; save=true also writes them to RT64's config file.
        static json handleSetConfig(State &state, const json &params) {
            if (params.contains("user")) {
                json merged = *state.ext.userConfig;
                for (const auto &it : params["user"].items()) {
                    if (!merged.contains(it.key())) {
                        throw std::runtime_error("unknown setting: " + it.key());
                    }

                    merged[it.key()] = it.value();
                }

                UserConfiguration &userConfig = *state.ext.userConfig;
                const uint32_t previousMSAACount = userConfig.msaaSampleCount();
                from_json(merged, userConfig);
                userConfig.validate();
                state.ext.sharedQueueResources->setUserConfig(userConfig, true);

                // As the inspector does: a new sample count needs every multisampled pipeline and target rebuilt.
                if (userConfig.msaaSampleCount() != previousMSAACount) {
                    state.ext.app->updateMultisampling();
                }
                state.configurationSaveQueued = state.configurationSaveQueued || params.value("save", false);
            }

            if (params.contains("enhancement")) {
                EnhancementConfiguration &e = *state.ext.enhancementConfig;
                const json &p = params["enhancement"];
                for (const auto &it : p.items()) {
                    if (it.key() == "presentationMode") {
                        e.presentation.mode = EnhancementConfiguration::Presentation::Mode(it.value().get<int>());
                    }
                    else if (it.key() == "removeBlackBorders") {
                        e.presentation.removeBlackBorders = it.value().get<bool>();
                    }
                    else if (it.key() == "fixRectLR") {
                        e.rect.fixRectLR = it.value().get<bool>();
                    }
                    else if (it.key() == "scaleLOD") {
                        e.textureLOD.scale = it.value().get<bool>();
                    }
                    else {
                        throw std::runtime_error("unknown setting: " + it.key());
                    }
                }

                state.ext.sharedQueueResources->setEnhancementConfig(e);
            }

            return handleGetConfig(state, params);
        }

        // The RS64 ray-tracing passes (F9's switches); an env var that pins a feature wins and is reported.
        static json handleRtFeatures(State &, const json &params) {
            static const std::map<std::string, rs64lights::Feature> features = {
                { "lights", rs64lights::Feature::Lights }, { "shadows", rs64lights::Feature::Shadows }, { "soft_shadows", rs64lights::Feature::SoftShadows },
                { "fog_shafts", rs64lights::Feature::FogShafts }, { "ao", rs64lights::Feature::AmbientOcclusion }, { "gi", rs64lights::Feature::GlobalIllumination },
                { "reflections", rs64lights::Feature::Reflections }
            };

            if (params.contains("set")) {
                for (const auto &it : params["set"].items()) {
                    auto f = features.find(it.key());
                    if (f == features.end()) {
                        throw std::runtime_error("unknown feature: " + it.key());
                    }

                    rs64lights::setFeature(f->second, it.value().get<bool>());
                }
            }

            json j = json::object();
            json pinned = json::array();
            for (const auto &it : features) {
                j[it.first] = rs64lights::feature(it.second);
                if (rs64lights::featurePinned(it.second)) {
                    pinned.push_back(it.first);
                }
            }

            j["pinned_by_env"] = pinned;
            return j;
        }

        // F4 (texture-pack replacement) and F3 (ViewRDRAM) as settable flags.
        static json handleRenderToggles(State &state, const json &params) {
            bool &replacement = state.ext.textureCache->textureMap.replacementMapEnabled;
            std::atomic<bool> &viewRDRAM = state.ext.presentQueue->viewRDRAM;
            replacement = params.value("texture_replacement", replacement);
            viewRDRAM = params.value("view_rdram", viewRDRAM.load());
            return { { "texture_replacement", replacement }, { "view_rdram", viewRDRAM.load() } };
        }

        // The RT debug views F9 / Shift+F9 cycle (rs64lights::debugViews): step by +-n, or pick one by name; returns the one in use and all names.
        static json handleRtDebugView(State &, const json &params) {
            size_t n = 0;
            const rs64lights::DebugView *views = rs64lights::debugViews(n);
            if (params.contains("name")) {
                const std::string want = params["name"].get<std::string>();
                size_t found = n;
                for (size_t i = 0; i < n; ++i) {
                    if (want == views[i].name) {
                        found = i;
                    }
                }

                if (found == n) {
                    throw std::runtime_error("unknown RT debug view: " + want);
                }

                rs64lights::mutableConfig().shadowsDebug = views[found].shadows;
                rs64lights::mutableConfig().debugMode = views[found].lights;
            }
            else if (params.contains("step")) {
                const int step = params["step"].get<int>();
                for (int k = 0; k < std::abs(step); ++k) {
                    rs64lights::stepDebugView(step < 0 ? -1 : 1);
                }
            }

            const rs64lights::Config &c = rs64lights::config();
            json names = json::array();
            std::string current = "custom";
            for (size_t i = 0; i < n; ++i) {
                names.push_back(views[i].name);
                if ((views[i].shadows == c.shadowsDebug) && (views[i].lights == c.debugMode)) {
                    current = views[i].name;
                }
            }

            return { { "view", current }, { "views", names } };
        }

        static const std::map<std::string, Handler> &handlers() {
            static const std::map<std::string, Handler> table = {
                { "rt64_pause", handlePause },
                { "rt64_status", handleStatus },
                { "interp_weight", handleInterpWeight },
                { "frame_summary", handleFrameSummary },
                { "list_draws", handleListDraws },
                { "get_draw", handleGetDraw },
                { "dump_tmem", handleDumpTmem },
                { "texture_dumping", handleTextureDumping },
                { "view_framebuffer", handleViewFramebuffer },
                { "view_draw", handleViewDraw },
                { "free_cam", handleFreeCam },
                { "get_camera", handleGetCamera },
                { "lights_list", handleLightsList },
                { "lights_active", handleLightsActive },
                { "lights_set", handleLightsSet },
                { "lights_remove", handleLightsRemove },
                { "lights_save", handleLightsSave },
                { "frame_times", handleFrameTimes },
                { "frame_warnings", handleFrameWarnings },
                { "pick", handlePick },
                { "highlight", handleHighlight },
                { "sprite_commands", handleSpriteCommands },
                { "rsp_state", handleRspState },
                { "vi_info", handleViInfo },
                { "get_config", handleGetConfig },
                { "set_config", handleSetConfig },
                { "rt_features", handleRtFeatures },
                { "render_toggles", handleRenderToggles },
                { "rt_debug_view", handleRtDebugView },
            };

            return table;
        }

        bool has(const std::string &method) {
            return handlers().count(method) > 0;
        }

        void post(uint32_t tag, const json &id, const std::string &method, const json &params) {
            std::lock_guard<std::mutex> lock(queueMutex);
            inbox.push_back({ tag, id, method, params });
        }

        bool takeReply(uint32_t &tag, json &reply) {
            std::lock_guard<std::mutex> lock(queueMutex);
            if (outbox.empty()) {
                return false;
            }

            tag = outbox.front().first;
            reply = std::move(outbox.front().second);
            outbox.pop_front();
            return true;
        }

        void drain(State &state) {
            if (pendingReset.exchange(false)) {
                state.debuggerInspector.renderer.framebufferIndex = -1;
                state.debuggerInspector.renderer.globalDrawCallIndex = -1;
                state.debuggerInspector.camera.enabled = false;
                state.dumpingTexturesDirectory.clear();
            }

            const int pause = pendingPause.exchange(-1);
            if (pause >= 0) {
                state.debuggerInspector.paused = (pause != 0);
            }

            std::deque<Request> requests;
            {
                std::lock_guard<std::mutex> lock(queueMutex);
                requests.swap(inbox);
            }

            for (Request &req : requests) {
                json reply = { { "id", req.id } };
                try {
                    reply["result"] = handlers().at(req.method)(state, req.params.is_object() ? req.params : json::object());
                    reply["ok"] = true;
                }
                catch (const std::exception &e) {
                    reply["ok"] = false;
                    reply["error"] = e.what();
                }

                std::lock_guard<std::mutex> lock(queueMutex);
                outbox.emplace_back(req.tag, std::move(reply));
            }

            pausedMirror = state.debuggerInspector.paused;
            drains++;
        }

        void requestPause(bool on) {
            pendingPause = on ? 1 : 0;
        }

        void requestReset() {
            pendingReset = true;
        }

        bool paused() {
            return pausedMirror;
        }

        uint64_t drainCount() {
            return drains;
        }
    };
};
