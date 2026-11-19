#pragma once

// RS64 light-pass maths: 4x4 row-vector matrices and the vertex path's screen mapping.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rs64lights {
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

    // A direction (row vector, w = 0) through m, normalised; zero length stays as given.
    inline void rotateDir(const float in[3], const Mat4& m, float out[3]) {
        const float v[4] = { in[0], in[1], in[2], 0.0f };
        float o[4];
        transform(v, m, o);
        const float len = std::sqrt(o[0] * o[0] + o[1] * o[1] + o[2] * o[2]);
        for (int i = 0; i < 3; ++i) {
            out[i] = (len > 1e-12f) ? o[i] / len : in[i];
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
}
