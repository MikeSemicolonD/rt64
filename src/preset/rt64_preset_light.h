//
// RT64
//

#pragma once

#include <set>

#include "common/rt64_common.h"
#include "shared/rt64_point_light.h"

#include "rt64_preset.h"
#include "rt64_preset_inspector.h"

namespace RT64 {
    class View;

    struct PresetLights : public PresetBase {
        // The enabled status is stored in the light itself instead since more
        // often that not most of them will be enabled, so storing them in a
        // separate list would lead to unnecessary overhead.
        struct Light {
            interop::PointLight description;
            bool enabled;
        };

        std::map<std::string, Light> lightMap;

        // RS64: the level (gCurrentLevel) whose map space the lights are in; NoLevel = outside missions (camera space), AnyLevel = everywhere.
        static constexpr int NoLevel = -1;
        static constexpr int AnyLevel = -2;
        // Front-end scenes, inside NoLevel: the mission-select briefing room and the craft-select hangar.
        static constexpr int BriefingRoom = -3;
        static constexpr int Hangar = -4;
        int level = AnyLevel;

        // Whether the preset lights a frame whose level (or front-end scene) is current.
        bool appliesTo(int current) const {
            return (level == AnyLevel) || (level == current) || ((level == NoLevel) && (current <= BriefingRoom));
        }

        // RS64: mesh-registry keys, '|'-separated ("red_laser", "xwing#s0.p1|wmxwng#s0.p1"): light positions are in the matched part's space
        // and follow every instance. Craft variants (player / wingman models) share the model space, so one preset lists the parts on the same joint.
        std::string mesh;
        // RS64: a mesh preset's radius and shadow offset in camera units, not the part's model units (laser bolts are drawn at varying scales).
        bool cameraUnits = false;
        // RS64: a mesh preset on one of the game's blinking lamps (a glw hardpoint): each light shines only while a lamp is drawn at its
        // spot on screen this frame, in that lamp's colour and fade, so it blinks and turns with it.
        bool lamp = false;

        // RS64: effect-sprite keys, '|'-separated ("spr:<texture name>"). A matching sprite's automatic light is replaced by this
        // preset's enabled lights at the sprite (radius = attenuation radius x the sprite's size, brightness following its fade); none = no light.
        std::string sprite;

        // A key matches one of the mesh's keys exactly or extends it at a '#' or '.' boundary ("red_laser" matches "red_laser#s2.p0.0").
        static bool meshMatches(const std::string &mesh, const std::string &key) {
            for (size_t start = 0; start < mesh.size();) {
                const size_t end = std::min(mesh.find('|', start), mesh.size());
                const size_t n = end - start;
                if ((n > 0) && (key.compare(0, n, mesh, start, n) == 0) && ((key.size() == n) || (key[n] == '#') || (key[n] == '.'))) {
                    return true;
                }

                start = end + 1;
            }

            return false;
        }

        virtual bool readJson(const json &jsonObj) override;
        virtual bool writeJson(json &jsonObj) const override;
    };

    struct PresetLightsLibrary : public PresetLibrary<PresetLights> { };

    struct PresetLightsLibraryInspector : public PresetLibraryInspector<PresetLights, PresetLightsLibrary> {
        std::string selectedLightName = "";
        bool skipGizmos = false;
        // RS64: level of the frame being inspected, stamped on new presets.
        int currentLevel = PresetLights::NoLevel;
        // RS64: mesh keys drawn this frame (filled by the F5 GBI), offered by the Mesh picker.
        std::set<std::string> seenMeshes;
        // RS64: effect-sprite keys drawn this frame with their PRIM colour (RRGGBBAA), offered by the Sprite picker.
        std::map<std::string, uint32_t> seenSprites;
        // RS64: the Im3d view this frame is the picked mesh preset's part (one of its drawn instances), so its lights can show and drag.
        bool meshViewValid = false;

        void inspectLibrary(PresetLightsLibrary &library, RenderWindow window);
        void inspectSelection(PresetLightsLibrary &library);
    };
};