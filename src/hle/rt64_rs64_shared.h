#pragma once

// State the render thread shares with the inspector and the host: lamp marks, preset views, the active lights and the preset reload queues.

#include "rt64_rs64_point_lights.h"
#include "rt64_rs64_shadows.h"

namespace rs64lights {
    // The game draws its lamps in their own graphics task, after the 3D scene's, so lamp marks reach the light pass through this store:
    // marks build up per workload; a later workload drawing lamps publishes the finished set.
    struct LampMarks {
        uint64_t workloadId = 0;
        std::vector<Candidate> marks;
    };

    inline std::mutex& lampMarksMutex() {
        static std::mutex m;
        return m;
    }

    inline LampMarks& lampMarksBuilding() {
        static LampMarks s;
        return s;
    }

    inline LampMarks& lampMarksPublished() {
        static LampMarks s;
        return s;
    }

    inline void addLampMark(uint64_t workloadId, const Candidate& mark) {
        std::lock_guard<std::mutex> lock(lampMarksMutex());
        LampMarks& b = lampMarksBuilding();
        if (b.workloadId != workloadId) {
            // Swapped, not moved: both sets keep their capacity instead of regrowing every workload.
            if (!b.marks.empty()) {
                std::swap(lampMarksPublished(), b);
            }
            b.workloadId = workloadId;
            b.marks.clear();
        }
        b.marks.push_back(mark);
    }

    // The lamp marks for a workload into out: the finished set when it is from within maxAge workloads of it, else the one still building.
    inline void lampMarksFor(uint64_t workloadId, uint64_t maxAge, std::vector<Candidate>& out) {
        std::lock_guard<std::mutex> lock(lampMarksMutex());
        const auto recent = [&](const LampMarks& s) { return !s.marks.empty() && ((s.workloadId > workloadId ? s.workloadId - workloadId : workloadId - s.workloadId) <= maxAge); };
        out.clear();
        if (recent(lampMarksPublished())) {
            out.assign(lampMarksPublished().marks.begin(), lampMarksPublished().marks.end());
        }
        else if (recent(lampMarksBuilding())) {
            out.assign(lampMarksBuilding().marks.begin(), lampMarksBuilding().marks.end());
        }
    }

    // The light pass's view in a preset space, published each frame for the F1 Lights tab's Im3d gizmos: slot PresetLevel is map space in a level,
    // camera space elsewhere; slot PresetMesh is the model space of one drawn part of the picked mesh preset. toNdc maps a point (row vector)
    // to the final NDC the light pass draws in; depth is a screen depth inside the view, for unprojecting; yDown: the space's up is -y.
    struct PresetView {
        bool valid = false;
        bool yDown = false;
        Mat4 toNdc{};
        Mat4 fromNdc{};
        float cameraPos[3] = {};
        float depth = 0.0f;
    };

    enum PresetSlot { PresetLevel = 0, PresetMesh = 1 };

    inline std::mutex& presetViewMutex() {
        static std::mutex m;
        return m;
    }

    inline PresetView& presetViewShared(int slot) {
        static PresetView v[2];
        return v[slot];
    }

    inline void publishPresetView(const PresetView& v, int slot) {
        std::lock_guard<std::mutex> lock(presetViewMutex());
        presetViewShared(slot) = v;
    }

    inline PresetView presetView(int slot) {
        std::lock_guard<std::mutex> lock(presetViewMutex());
        return presetViewShared(slot);
    }

    // What the last light pass drew (camera space) and the sun the last shadow pass traced, for the MCP's lights_active.
    struct ActiveLights {
        std::vector<Resolved> lights;
        size_t candidates = 0;
        Sun sun;
        bool sunTraced = false;
    };

    inline std::mutex& activeLightsMutex() {
        static std::mutex m;
        return m;
    }

    inline ActiveLights& activeLightsShared() {
        static ActiveLights a;
        return a;
    }

    inline void publishActiveLights(const Resolved* chosen, size_t count, size_t candidates) {
        std::lock_guard<std::mutex> lock(activeLightsMutex());
        activeLightsShared().lights.assign(chosen, chosen + count);
        activeLightsShared().candidates = candidates;
    }

    inline void publishSun(const Sun& sun) {
        std::lock_guard<std::mutex> lock(activeLightsMutex());
        activeLightsShared().sun = sun;
        activeLightsShared().sunTraced = true;
    }

    inline Sun publishedSun() {
        std::lock_guard<std::mutex> lock(activeLightsMutex());
        return activeLightsShared().sun;
    }

    // Whether a sphere (camera space, looking down +z) can shadow a visible pixel: the line from it away from the sun must enter the view (z >= 0, |x| <= slopeX * z, |y| <= slopeY * z).
    inline bool casterReachesView(const float p[3], float r, const float away[3], float slopeX, float slopeY) {
        const float rx = r * std::sqrt(1.0f + slopeX * slopeX);
        const float ry = r * std::sqrt(1.0f + slopeY * slopeY);
        const float planes[5][4] = {
            { 0.0f, 0.0f, 1.0f, r },
            { -1.0f, 0.0f, slopeX, rx },
            { 1.0f, 0.0f, slopeX, rx },
            { 0.0f, -1.0f, slopeY, ry },
            { 0.0f, 1.0f, slopeY, ry }
        };
        float tNear = 0.0f;
        float tFar = 1e30f;
        for (const auto& n : planes) {
            const float start = n[0] * p[0] + n[1] * p[1] + n[2] * p[2] + n[3];
            const float slope = n[0] * away[0] + n[1] * away[1] + n[2] * away[2];
            if (slope == 0.0f) {
                if (start < 0.0f) {
                    return false;
                }
                continue;
            }
            const float t = -start / slope;
            if (slope > 0.0f) {
                tNear = std::max(tNear, t);
            }
            else {
                tFar = std::min(tFar, t);
            }
        }
        return tNear <= tFar;
    }

    inline ActiveLights activeLights() {
        std::lock_guard<std::mutex> lock(activeLightsMutex());
        return activeLightsShared();
    }

    // The view the inspector's frame uses, picked before Im3d's frame and read when its draw lists are drawn (both on RT64's state thread).
    inline PresetView& inspectorView() {
        static PresetView v;
        return v;
    }

    // Light-preset library JSON read by the host (enabled mods, then roguesq_lights.json), merged into RT64's light library on its own thread at the next frame.
    inline std::mutex& presetTextMutex() {
        static std::mutex m;
        return m;
    }

    inline std::vector<std::string>& presetTextQueue() {
        static std::vector<std::string> q;
        return q;
    }

    // Set with a queued set of files that replaces the library (a reload) rather than merging into it.
    inline bool& presetTextReplaces() {
        static bool replaces = false;
        return replaces;
    }

    // Every light file in override order; the library is emptied before they merge at the next frame.
    inline void queuePresetReload(std::vector<std::string> texts) {
        std::lock_guard<std::mutex> lock(presetTextMutex());
        presetTextQueue() = std::move(texts);
        presetTextReplaces() = true;
    }

    inline std::vector<std::string> takePresetTexts(bool &replaces) {
        std::lock_guard<std::mutex> lock(presetTextMutex());
        replaces = std::exchange(presetTextReplaces(), false);
        return std::exchange(presetTextQueue(), {});
    }

    // Presets a mod sets while the game runs (host API light_set), by name, each one preset's JSON; applied after the light files on every change and reload.
    inline std::mutex& runtimePresetMutex() {
        static std::mutex m;
        return m;
    }

    inline std::map<std::string, std::string>& runtimePresets() {
        static std::map<std::string, std::string> m;
        return m;
    }

    inline uint64_t& runtimePresetVersion() {
        static uint64_t v = 0;
        return v;
    }

    // An empty json removes the preset.
    inline void setRuntimePreset(const std::string &name, std::string json) {
        std::lock_guard<std::mutex> lock(runtimePresetMutex());
        if (json.empty()) {
            runtimePresets().erase(name);
        }
        else {
            runtimePresets()[name] = std::move(json);
        }

        runtimePresetVersion()++;
    }

    // The presets when they changed since seen (or force), else false.
    inline bool takeRuntimePresets(uint64_t &seen, bool force, std::map<std::string, std::string> &out) {
        std::lock_guard<std::mutex> lock(runtimePresetMutex());
        if (!force && (seen == runtimePresetVersion())) {
            return false;
        }

        seen = runtimePresetVersion();
        out = runtimePresets();
        return true;
    }

    // The host's reader for the files above (set by the host when it first reads them), for the F1 Lights tab's Reload light files.
    inline std::function<void()>& presetReloader() {
        static std::function<void()> f;
        return f;
    }

    // Preset-space point under final NDC (x, y) at the view's depth.
    inline void presetUnproject(const PresetView& v, float x, float y, float out[3]) {
        const float in[4] = { x, y, v.depth, 1.0f };
        float o[4];
        transform(in, v.fromNdc, o);
        for (int i = 0; i < 3; ++i) {
            out[i] = o[i] / o[3];
        }
    }
}
