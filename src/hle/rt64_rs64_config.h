#pragma once

// RS64 light-pass settings (ROGUESQ_RT_* and the in-game toggles), debug views and the front-end scene checks.

#include "../common/rt64_rs64_env.h"
#include "rt64_rs64_point_lights.h"

namespace rs64lights {
    inline float envFloat(const char* name, float def) {
        return rs64env::number(name, def);
    }

    struct Config {
        std::atomic<bool> enabled;
        int debugMode;
        float gain;
        float wrap;
        float spriteRadius;
        float spriteRadiusMax;
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
        float menuImageGain;
        float temporalAlpha;
        bool softHalfRes;
        float hangarFloorCos;
        std::atomic<bool> fogShafts;
        int fogShaftsSteps;
        float fogShaftsStrength;
        int fogShaftsDebug;
        bool shadowCutout;
        std::atomic<bool> softShadows;
        int softRays;
        float sunAngleDeg;
        float softBlurPx;
        bool terrainNormals;
        bool terrainTerminator;
        // The terminator fades in between these flatness values (|heightfield normal . up|): a cliff's normal swings column by column and saw-tooths.
        float terminatorFlatLo;
        float terminatorFlatHi;
        // Model faces turned away from the sun darken like a cast shadow: their baked shading follows a fixed light, so a shadow cast on a ship was darker than its own dark side.
        bool modelTerminator;
        float spriteShadowStart;
        float terrainNormalWindow;
        int shadowHistory;
        int shadowHistoryInterval;
        float shadowHistorySkip;
        bool logTiming;
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
        // Strength of the sky a water reflection ray sees when it hits nothing (0 = water keeps its drawn colour there).
        float reflSky;
        // Blur radius (pixels at native resolution) of water reflections: one jittered ray per pixel speckles at reflection edges.
        float reflWaterBlurPx;
        // Strength of the horizon mirror: water shows what is drawn after the trace (effect sprites, late craft) where it lies mirrored about the horizon.
        float reflMirror;
        // Cast shadow strength on water (1 = as on ground): what the water reflects stays lit in a shadow.
        float waterShadow;
        // Slope strength of the water's wind ripples (0 = flat water).
        float waterRipple;
        // A level sun lower than this (degrees) is lifted to it: a near-horizon sun throws long shadows and the terminator blackens gentle slopes.
        float sunMinElevationDeg;
        // Sprite frame brightness (spriteEmission) that lights at full strength; dimmer frames scale their light down. 0 = off.
        float spriteEmitRef;
        // selectLights fades lights near the light-count cut (A/B switch).
        bool cutFade;
        // Laser bolt lights get at most this many of the slots other lights leave.
        int maxProjectileLights;
        // Hangar sun shadows count only blockers this close (game units): the ship's own parts, not the hangar's walls.
        float hangarShadowRange;
        // dropFarLights' radius / distance floor.
        float minLightReach;
        // Blinn-Phong highlight strength (0 = off) and exponent for the light pass.
        float specular;
        float specularPower;
        // Scale on every level preset's haze brightness (0 = no haze).
        float hazeStrength;
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
            envFloat("ROGUESQ_RT_MENU_IMAGE_GAIN", 2.0f),
            std::clamp(envFloat("ROGUESQ_RT_TEMPORAL", 0.2f), 0.0f, 1.0f),
            envFloat("ROGUESQ_RT_HALF_RES", 1.0f) != 0.0f,
            envFloat("ROGUESQ_RT_HANGAR_FLOOR_COS", 0.995f),
            envFloat("ROGUESQ_RT_FOG_SHAFTS", 0.0f) != 0.0f,
            std::clamp((int)envFloat("ROGUESQ_RT_FOG_SHAFTS_STEPS", 12.0f), 4, 32),
            envFloat("ROGUESQ_RT_FOG_SHAFTS_STRENGTH", 0.35f),
            (int)envFloat("ROGUESQ_RT_FOG_SHAFTS_DEBUG", 0.0f),
            envFloat("ROGUESQ_RT_SHADOWS_CUTOUT", 1.0f) != 0.0f,
            envFloat("ROGUESQ_RT_SHADOWS_SOFT", 1.0f) != 0.0f,
            std::clamp((int)envFloat("ROGUESQ_RT_SHADOWS_SOFT_RAYS", 4.0f), 1, 16),
            envFloat("ROGUESQ_RT_SHADOWS_SUN_ANGLE", 4.0f),
            std::clamp(envFloat("ROGUESQ_RT_SHADOWS_SOFT_BLUR", 24.0f), 0.0f, 32.0f),
            envFloat("ROGUESQ_RT_SHADOWS_TERRAIN_NORMALS", 1.0f) != 0.0f,
            envFloat("ROGUESQ_RT_SHADOWS_TERMINATOR", 1.0f) != 0.0f,
            std::clamp(envFloat("ROGUESQ_RT_SHADOWS_TERMINATOR_FLAT_LO", 0.3f), 0.0f, 1.0f),
            std::clamp(envFloat("ROGUESQ_RT_SHADOWS_TERMINATOR_FLAT_HI", 0.9f), 0.0f, 1.0f),
            envFloat("ROGUESQ_RT_SHADOWS_MODEL_TERMINATOR", 1.0f) != 0.0f,
            envFloat("ROGUESQ_RT_LIGHTS_SPRITE_SHADOW_START", 1.0f),
            std::clamp(envFloat("ROGUESQ_RT_SHADOWS_TERRAIN_NORMAL_WINDOW", 0.05f), 0.0f, 0.5f),
            std::clamp((int)envFloat("ROGUESQ_RT_SHADOWS_HISTORY", 2.0f), 0, 4),
            std::max((int)envFloat("ROGUESQ_RT_SHADOWS_HISTORY_INTERVAL", 8.0f), 1),
            envFloat("ROGUESQ_RT_SHADOWS_HISTORY_SKIP", 0.0f),
            envFloat("ROGUESQ_LOG_RT_TIMING", 0.0f) != 0.0f,
            envFloat("ROGUESQ_RT_AO", 0.0f) != 0.0f,
            std::clamp((int)envFloat("ROGUESQ_RT_AO_RAYS", 2.0f), 1, 8),
            envFloat("ROGUESQ_RT_AO_RANGE", 0.04f),
            envFloat("ROGUESQ_RT_AO_BLUR", 4.0f),
            envFloat("ROGUESQ_RT_AO_STRENGTH", 0.5f),
            envFloat("ROGUESQ_RT_GI", 0.0f) != 0.0f,
            std::clamp((int)envFloat("ROGUESQ_RT_GI_RAYS", 1.0f), 1, 4),
            envFloat("ROGUESQ_RT_GI_RANGE", 0.25f),
            envFloat("ROGUESQ_RT_GI_BLUR", 8.0f),
            envFloat("ROGUESQ_RT_GI_STRENGTH", 0.3f),
            envFloat("ROGUESQ_RT_GI_EMISSIVE", 1.0f),
            envFloat("ROGUESQ_RT_REFLECTIONS", 0.0f) != 0.0f,
            envFloat("ROGUESQ_RT_REFL_ROUGHNESS", 20.0f),
            envFloat("ROGUESQ_RT_REFL_RANGE", 0.1f),
            envFloat("ROGUESQ_RT_REFL_STRENGTH", 3.0f),
            std::max(envFloat("ROGUESQ_RT_REFL_SKY", 1.0f), 0.0f),
            std::max(envFloat("ROGUESQ_RT_REFL_WATER_BLUR", 3.0f), 0.0f),
            std::clamp(envFloat("ROGUESQ_RT_REFL_MIRROR", 0.0f), 0.0f, 1.0f),
            std::clamp(envFloat("ROGUESQ_RT_WATER_SHADOW", 0.5f), 0.0f, 1.0f),
            std::clamp(envFloat("ROGUESQ_RT_WATER_RIPPLE", 1.0f), 0.0f, 4.0f),
            std::clamp(envFloat("ROGUESQ_RT_SUN_MIN_ELEVATION", 25.0f), 0.0f, 89.0f),
            std::max(envFloat("ROGUESQ_RT_LIGHTS_SPRITE_EMIT_REF", 0.12f), 0.0f),
            envFloat("ROGUESQ_RT_LIGHTS_CUT_FADE", 1.0f) != 0.0f,
            std::clamp((int)envFloat("ROGUESQ_RT_LIGHTS_PROJECTILE_MAX", 8.0f), 0, (int)MaxLights),
            envFloat("ROGUESQ_RT_SHADOWS_HANGAR_RANGE", 500.0f),
            std::max(envFloat("ROGUESQ_RT_LIGHTS_MIN_REACH", 0.01f), 0.0f),
            std::max(envFloat("ROGUESQ_RT_LIGHTS_SPEC", 0.5f), 0.0f),
            std::clamp(envFloat("ROGUESQ_RT_LIGHTS_SPEC_POWER", 32.0f), 1.0f, 512.0f),
            std::max(envFloat("ROGUESQ_RT_HAZE_STRENGTH", 1.0f), 0.0f),
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
        return rs64env::str(featureEnv(f)) != nullptr;
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

    // Returns false (no change) when ROGUESQ_RT_LIGHTS_PROJECTILE_MAX pins it.
    inline bool setMaxProjectileLights(int n) {
        if (rs64env::str("ROGUESQ_RT_LIGHTS_PROJECTILE_MAX") != nullptr) {
            return false;
        }
        mutableConfig().maxProjectileLights = std::clamp(n, 0, (int)MaxLights);
        return true;
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
    // A pair that starts a new view (newView, after a depth clear) ends the earlier view, which keeps its own last pass.
    inline void lastPassPerTarget(const void* const* targets, const bool* newView, bool* on, size_t n) {
        for (size_t i = 0; i < n; ++i) {
            if (!on[i]) {
                continue;
            }
            for (size_t j = i + 1; j < n; ++j) {
                if (targets[j] != targets[i]) {
                    continue;
                }

                if (newView[j]) {
                    break;
                }

                if (on[j]) {
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

    // No level light this frame, but the same level's mission is still on (or the pause menu's scene is up): the level's last sun stays.
    inline bool keepMissionSun(bool levelSun, uint32_t missionRoot, bool inMission, MenuScene menu, bool sameLevel) {
        return !levelSun && sameLevel && (menu == MenuSceneNone) && ((missionRoot != 0) || inMission);
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

    // Where the briefing room's preview-image glow sits, in its fixed camera space (the hologram, the shared room origin), and its reach and
    // falloff; the hologram's own light is the built-in preset briefing_hologram at the same place.
    inline const float* menuHoloPos() {
        static const float v[3] = { 0.0f, 22.0f, 11068.0f };
        return v;
    }
    constexpr float MenuHoloRadius = 20000.0f;
    constexpr float MenuHoloFalloff = 1.0f;

    // The GBI captures (terrain, no-cast lasers) and the world-vertex pass feed the scene for sun shadows, shadowed lights and fog shafts.
    inline bool sceneCaptureEnabled(const Config& c) {
        return c.shadows || (c.enabled && c.lightShadows) || c.fogShafts || c.ao || c.gi || c.reflections;
    }

    // Laser/torpedo mesh lookup: lights need the emitters, and every scene feature needs the bolts kept out of the casters.
    inline bool emitterLookupEnabled(const Config& c) {
        return c.enabled || sceneCaptureEnabled(c);
    }

    inline bool sceneWanted(bool shadowsWanted, bool lightsWanted, bool lightShadows, bool rayQuery) {
        return shadowsWanted || (lightsWanted && lightShadows && rayQuery);
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
}
