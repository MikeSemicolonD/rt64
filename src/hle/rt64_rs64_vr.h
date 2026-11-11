#pragma once

// VR eye matrices for RS64. Game camera space (what the F5 modelviews produce and RT64's view slot sees) is x right, y down, z forward
// (guScale(1, 1, -1) before guPerspective; the F5 viewport flips y). OpenXR space is x right, y up, z back: the same axes turned 180 degrees about x.

#include "rt64_rs64_lights.h"

#include <cmath>

namespace rs64vr {
    using rs64lights::Mat4;

    // Same layout as XrPosef: orientation quaternion, then position in meters, OpenXR axes.
    struct Pose {
        float qx, qy, qz, qw;
        float px, py, pz;
    };

    // Same layout as XrFovf, radians; left and down are negative.
    struct Fov {
        float left, right, up, down;
    };

    // y and z components of the axis flip between OpenXR and game camera space.
    constexpr float kGameYSign = -1.0f;
    constexpr float kGameZSign = -1.0f;

    // Both eyes as OpenXR located them for one rendered frame; the same values go back in the projection layer.
    struct EyeViews {
        Pose eye[2];
        Fov fov[2];
        bool valid = false;
    };

    // View matrix (row vectors) that moves game camera space into the eye's space, keeping game axes and units.
    inline Mat4 eyeView(const Pose& eye, float unitsPerMeter) {
        const float x = eye.qx, y = eye.qy, z = eye.qz, w = eye.qw;
        const float r[3][3] = {
            { 1.0f - 2.0f * (y * y + z * z), 2.0f * (x * y - z * w), 2.0f * (x * z + y * w) },
            { 2.0f * (x * y + z * w), 1.0f - 2.0f * (x * x + z * z), 2.0f * (y * z - x * w) },
            { 2.0f * (x * z - y * w), 2.0f * (y * z + x * w), 1.0f - 2.0f * (x * x + y * y) }
        };
        const float f[3] = { 1.0f, kGameYSign, kGameZSign };
        const float p[3] = { eye.px, eye.py, eye.pz };

        // The inverse eye transform in OpenXR space, conjugated by the flip: rotation F R F, translation -F R^T p.
        Mat4 m = rs64lights::identity();
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                m.m[i][j] = f[i] * r[i][j] * f[j];
            }
        }

        for (int j = 0; j < 3; ++j) {
            const float rtp = r[0][j] * p[0] + r[1][j] * p[1] + r[2][j] * p[2];
            m.m[3][j] = -f[j] * rtp * unitsPerMeter;
        }

        return m;
    }

    // Head turned left by `degrees` about the up axis, no eye offset: the fake head turn for cull testing.
    inline Mat4 yawView(float degrees) {
        const float half = degrees * 3.14159265f / 360.0f;
        const Pose pose{ 0.0f, std::sin(half), 0.0f, std::cos(half), 0.0f, 0.0f, 0.0f };
        return eyeView(pose, 1.0f);
    }

    // A head turned left by `yawDegrees` and up by `pitchDegrees` (pitch in the head's frame), at (x, y, z) meters in OpenXR axes.
    inline Pose headPose(float yawDegrees, float pitchDegrees, float x, float y, float z) {
        const float hy = yawDegrees * 3.14159265f / 360.0f;
        const float hp = pitchDegrees * 3.14159265f / 360.0f;
        const float cy = std::cos(hy), sy = std::sin(hy), cp = std::cos(hp), sp = std::sin(hp);
        return Pose{ cy * sp, sy * cp, -sy * sp, cy * cp, x, y, z };
    }

    // The view for a head turned left by `yawDegrees` and up by `pitchDegrees`, no eye offset.
    inline Mat4 headView(float yawDegrees, float pitchDegrees) {
        return eyeView(headPose(yawDegrees, pitchDegrees, 0.0f, 0.0f, 0.0f), 1.0f);
    }

    // The game's projection with its x/y terms replaced by an asymmetric eye fov; the z and w columns (depth, fog) stay the game's.
    inline Mat4 eyeProjection(const Mat4& gameProj, const Fov& fov) {
        const float tl = std::tan(fov.left), tr = std::tan(fov.right);
        const float tu = std::tan(fov.up), td = std::tan(fov.down);
        const float wz = gameProj.m[2][3];
        const float sx = gameProj.m[0][0] < 0.0f ? -1.0f : 1.0f;
        const float sy = gameProj.m[1][1] < 0.0f ? -1.0f : 1.0f;

        Mat4 m = gameProj;
        for (int r = 0; r < 4; ++r) {
            m.m[r][0] = 0.0f;
            m.m[r][1] = 0.0f;
        }

        m.m[0][0] = sx * 2.0f * wz / (tr - tl);
        m.m[2][0] = -sx * (tr + tl) * wz / (tr - tl);
        m.m[1][1] = sy * 2.0f * wz / (tu - td);
        m.m[2][1] = -sy * (tu + td) * wz / (tu - td);
        return m;
    }
}

namespace plume {
    struct RenderTexture;
}

// Host side of VR (src/video/vr_openxr.cpp through rt64_render_context.cpp); every call is safe without a session.
extern "C" {
    // 1 while an OpenXR session renders: RT64 draws a second eye and submits to the headset.
    int rs64_vr_session(void);

    // 1 while culls and the HUD must work in every direction: a session or the fake head turn (ROGUESQ_VR_FAKE_YAW).
    int rs64_vr_active(void);

    // The fake head's pose (OpenXR axes, meters) while ROGUESQ_VR_FAKE_YAW is set; 0 = it is off.
    int rs64_vr_fake_head(rs64vr::Pose *pose);

    // 1 in a mission: only the mission HUD becomes quads; menus keep their texrect effects.
    int rs64_vr_hud_quads(void);

    // Headset rate RT64 interpolates to (0 = no session) and the per-eye image size.
    uint32_t rs64_vr_target_rate(void);
    void rs64_vr_eye_size(uint32_t *width, uint32_t *height);

    // Game units per meter for the eye offset.
    float rs64_vr_units_per_meter(void);

    // Workload thread: both eye poses predicted `framesAhead` headset frames past the last xrWaitFrame. 0 = none.
    int rs64_vr_locate(uint32_t framesAhead, rs64vr::EyeViews *out);

    // Present thread, once per headset frame: xrWaitFrame + xrBeginFrame. 1 = render, 0 = submit nothing, -1 = no session.
    int rs64_vr_begin_frame(void);

    // Present thread: the eye's swapchain image, acquired and waited on, in COLOR_WRITE layout; null without a session.
    plume::RenderTexture *rs64_vr_acquire_eye(int eye, uint32_t *width, uint32_t *height);

    // Present thread: releases acquired images and ends the frame with `views` (null = no layers).
    void rs64_vr_end_frame(const rs64vr::EyeViews *views);
}
