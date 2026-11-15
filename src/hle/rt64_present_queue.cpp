//
// RT64
//

#include "rt64_present_queue.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#include "common/rt64_thread.h"
#include "rhi/rt64_render_hooks.h"

#include "rt64_rs64_mcp.h"
#include "rt64_rs64_transition.h"
#include "rt64_rs64_vr.h"
#include "rt64_workload_queue.h"
#include "shared/rt64_rs64_vr_hands_cb.h"

// Most draw-active color buffer, published by the F3DFACTOR5 GBI module. Present
// mode 4 presents it so offscreen-rendered content (e.g. the cinematic explosion
// in 0x290000) reaches the screen.
extern "C" volatile unsigned g_most_drawn_fb;

// ROGUESQ_LOG_SWAPCHAIN=1: surface suspend/resume and swapchain resize events (the Vulkan backend logs acquire/present results).
static bool rs64LogSwapChain() {
    static const bool enabled = []() {
        const char *v = std::getenv("ROGUESQ_LOG_SWAPCHAIN");
        return (v != nullptr) && (v[0] != '\0') && (v[0] != '0');
    }();
    return enabled;
}

std::atomic<uint32_t> g_rs64_cpu_cleared_fb{ 0 };
std::atomic<uint64_t> g_rs64_cpu_cleared_wid{ 0 };
std::atomic<uint32_t> g_rs64_transition_gate_fb{ 0 };
std::atomic<uint64_t> g_rs64_last_workload_id{ 0 };

namespace RT64 {
    // ROGUESQ_SCREENSHOT_EVERY=<seconds>: saves the game's color buffer to screenshots/shot_NNN.png, at most ROGUESQ_SCREENSHOT_MAX (default 20) per run.
    struct RS64Screenshot {
        std::unique_ptr<RenderBuffer> buffer;
        uint64_t bufferSize = 0;
        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t outWidth = 0;
        uint32_t outHeight = 0;
        uint32_t rowWidth = 0;
        float pixelAspect = 1.0f;
        bool rgba8 = true;
        bool requested = false;
        int taken = 0;
        std::chrono::steady_clock::time_point last = std::chrono::steady_clock::now();
    };

    static RS64Screenshot rs64Screenshot;

    // On-demand capture (rs64CaptureFrame, the MCP screenshot): the present thread takes the request, writes the PNG and signals.
    static std::mutex rs64CaptureMutex;
    static std::condition_variable rs64CaptureCondition;
    static std::string rs64CapturePath;
    static uint64_t rs64CaptureRequested = 0;
    static uint64_t rs64CaptureTaken = 0;
    static uint64_t rs64CaptureWritten = 0;
    static RS64CaptureResult rs64CaptureLast;

    RS64CaptureResult rs64CaptureFrame(const std::string &path, int timeoutMs) {
        std::unique_lock<std::mutex> lock(rs64CaptureMutex);
        rs64CapturePath = path;
        const uint64_t want = ++rs64CaptureRequested;
        if (!rs64CaptureCondition.wait_for(lock, std::chrono::milliseconds(timeoutMs), [want]() { return rs64CaptureWritten >= want; })) {
            rs64CaptureRequested = rs64CaptureWritten;
            return {};
        }

        return rs64CaptureLast;
    }

    static bool rs64QueueScreenshot(RenderDevice *device, RenderCommandList *commandList, const VIRenderer::RenderParams &params) {
        static const double every = []() { const char *v = std::getenv("ROGUESQ_SCREENSHOT_EVERY"); return (v != nullptr) ? std::atof(v) : 0.0; }();
        static const int maxShots = []() { const char *v = std::getenv("ROGUESQ_SCREENSHOT_MAX"); return (v != nullptr) ? std::max(std::atoi(v), 1) : 20; }();
        RS64Screenshot &s = rs64Screenshot;
        const bool rgba8 = (params.textureFormat == RenderFormat::R8G8B8A8_UNORM);
        if (!rgba8 && (params.textureFormat != RenderFormat::R16G16B16A16_UNORM)) {
            return false;
        }

        bool requested = false;
        {
            const std::scoped_lock lock(rs64CaptureMutex);
            if (rs64CaptureRequested > rs64CaptureTaken) {
                rs64CaptureTaken = rs64CaptureRequested;
                requested = true;
            }
        }

        const auto now = std::chrono::steady_clock::now();
        if (!requested && ((every <= 0.0) || (s.taken >= maxShots) || (std::chrono::duration<double>(now - s.last).count() < every))) {
            return false;
        }

        s.requested = requested;
        s.last = now;
        s.rgba8 = rgba8;
        s.width = params.textureWidth;
        s.height = params.textureHeight;
        s.rowWidth = (s.width + 63) & ~63U;

        // Keep only what the VI pass shows: the target's last native column and row hold stale pixels (see the VI overscan inset).
        const hlslpp::uint2 nativeSize = (params.vi != nullptr) && (params.downsamplingScale > 0) ? params.vi->fbSize() : hlslpp::uint2(0, 0);
        const float hdW = float(nativeSize.x) * float(params.resolutionScale.x) / float(params.downsamplingScale);
        const float hdH = float(nativeSize.y) * float(params.resolutionScale.y) / float(params.downsamplingScale);
        const float cropW = (nativeSize.x > 1) ? (hdW - hdW / float(nativeSize.x)) : hdW;
        const float cropH = (nativeSize.y > 1) ? (hdH - hdH / float(nativeSize.y)) : hdH;
        // The VI pass widens each framebuffer pixel by this on screen (VIRenderer::getViewportAndScissor), so the PNG does too.
        s.pixelAspect = (params.vi != nullptr) && params.removeBlackBorders ? params.vi->pixelAspect() : 1.0f;
        s.outWidth = (cropW >= 1.0f) ? std::min(uint32_t(cropW), s.width) : s.width;
        s.outHeight = (cropH >= 1.0f) ? std::min(uint32_t(cropH), s.height) : s.height;

        const uint64_t size = uint64_t(s.rowWidth) * s.height * (rgba8 ? 4 : 8);
        if (s.bufferSize < size) {
            s.buffer = device->createBuffer(RenderBufferDesc::ReadbackBuffer(size));
            s.bufferSize = size;
        }

        commandList->barriers(RenderBarrierStage::COPY, RenderTextureBarrier(params.texture, RenderTextureLayout::COPY_SOURCE));
        commandList->copyTextureRegion(RenderTextureCopyLocation::PlacedFootprint(s.buffer.get(), params.textureFormat, s.width, s.height, 1, s.rowWidth), RenderTextureCopyLocation::Subresource(params.texture));
        commandList->barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(params.texture, RenderTextureLayout::SHADER_READ));
        return true;
    }

    // VR headset frame paths, logged every 2 s: fresh eyes, the last eyes again (and the check that failed), kept alive while the game presents nothing, no render asked, wait or begin failed.
    struct RS64VrStats {
        uint32_t fresh = 0, screen = 0, fadeHeld = 0, noColor = 0, noEyes = 0, noRight = 0, noViews = 0, submitFail = 0, keepAlive = 0, noRender = 0, failed = 0;
        Timestamp stamp;
    };

    static RS64VrStats rs64VrStats;
    // The poses of the eyes last copied to the headset; a repeated frame shows those images at these poses.
    static rs64vr::EyeViews rs64VrLastViews;
    // The last fresh frame went to the virtual screen, so a repeat shows the screen again.
    static bool rs64VrLastScreen = false;

    // Fade through black when the headset changes between the virtual screen and the 3D view: the old image dims out, then the new one comes up.
    static constexpr std::chrono::milliseconds kVrFadeHalf(150);
    static bool rs64VrFading = false;
    static Timestamp rs64VrFadeStamp;

    // Sets this headset frame's brightness. newIsScreen is the view a fresh frame would show (startsFade false when there is none);
    // true back means hold the old image instead, while it fades out.
    static bool rs64VrFade(bool startsFade, bool newIsScreen) {
        const bool shownSomething = rs64VrLastViews.valid || rs64VrLastScreen;
        if (!rs64VrFading && startsFade && shownSomething && (newIsScreen != rs64VrLastScreen)) {
            rs64VrFading = true;
            rs64VrFadeStamp = Timer::current();
        }

        bool holdOld = false;
        float brightness = 1.0f;
        if (rs64VrFading) {
            const auto elapsed = Timer::current() - rs64VrFadeStamp;
            const float half = float(std::chrono::duration_cast<std::chrono::microseconds>(kVrFadeHalf).count());
            const float t = float(std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count()) / half;
            if (t < 1.0f) {
                holdOld = true;
                brightness = 1.0f - t;
            }
            else if (t < 2.0f) {
                brightness = t - 1.0f;
            }
            else {
                rs64VrFading = false;
            }
        }

        rs64_vr_set_brightness(brightness);
        return holdOld;
    }

    // Ends a headset frame with the last image again: the virtual screen or the eyes at their poses.
    static void rs64VrRepeat() {
        if (rs64VrLastScreen) {
            rs64_vr_end_frame_screen();
        }
        else {
            rs64_vr_end_frame(&rs64VrLastViews);
        }
    }

    static void rs64VrLogStats() {
        RS64VrStats &s = rs64VrStats;
        if (!rs64vr::logEnabled()) {
            return;
        }

        if ((s.stamp == Timestamp()) || (Timer::current() - s.stamp >= std::chrono::seconds(2))) {
            if (!(s.stamp == Timestamp())) {
                fprintf(stderr, "[vr] 2s: fresh %u, screen %u, held for fade %u, repeat (no color %u, no eyes %u, no right %u, no views %u, submit failed %u), kept alive %u, no render %u, failed %u, rate %u Hz\n",
                    s.fresh, s.screen, s.fadeHeld, s.noColor, s.noEyes, s.noRight, s.noViews, s.submitFail, s.keepAlive, s.noRender, s.failed, rs64_vr_target_rate());
            }

            const Timestamp now = Timer::current();
            s = RS64VrStats();
            s.stamp = now;
        }
    }

    // Row-major 4x4 for column vectors.
    struct RS64VrMat {
        float m[4][4] = {};
    };

    static RS64VrMat rs64VrMul(const RS64VrMat &a, const RS64VrMat &b) {
        RS64VrMat r;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                for (int k = 0; k < 4; k++) {
                    r.m[i][j] += a.m[i][k] * b.m[k][j];
                }
            }
        }

        return r;
    }

    // The pose as a transform into the tracking space, or (inverse) out of it.
    static RS64VrMat rs64VrPoseMatrix(const rs64vr::Pose &p, bool inverse) {
        const float x = p.qx, y = p.qy, z = p.qz, w = p.qw;
        const float r[3][3] = {
            { 1.0f - 2.0f * (y * y + z * z), 2.0f * (x * y - z * w), 2.0f * (x * z + y * w) },
            { 2.0f * (x * y + z * w), 1.0f - 2.0f * (x * x + z * z), 2.0f * (y * z - x * w) },
            { 2.0f * (x * z - y * w), 2.0f * (y * z + x * w), 1.0f - 2.0f * (x * x + y * y) }
        };
        const float t[3] = { p.px, p.py, p.pz };
        RS64VrMat out;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                out.m[i][j] = inverse ? r[j][i] : r[i][j];
            }

            out.m[i][3] = inverse ? -(r[0][i] * t[0] + r[1][i] * t[1] + r[2][i] * t[2]) : t[i];
        }

        out.m[3][3] = 1.0f;
        return out;
    }

    // OpenXR's asymmetric projection, depth 0 at the near plane and 1 at the far one; clip y is up (the vertex shaders build with -fvk-invert-y).
    static RS64VrMat rs64VrProjection(const rs64vr::Fov &fov) {
        constexpr float kNear = 0.05f;
        constexpr float kFar = 100.0f;
        const float tl = std::tan(fov.left), tr = std::tan(fov.right);
        const float tu = std::tan(fov.up), td = std::tan(fov.down);
        RS64VrMat p;
        p.m[0][0] = 2.0f / (tr - tl);
        p.m[0][2] = (tr + tl) / (tr - tl);
        p.m[1][1] = 2.0f / (tu - td);
        p.m[1][2] = (tu + td) / (tu - td);
        p.m[2][2] = -kFar / (kFar - kNear);
        p.m[2][3] = -kFar * kNear / (kFar - kNear);
        p.m[3][2] = -1.0f;
        return p;
    }

    // VR: both controllers into the controller layer's images at this headset frame's eye poses; the frame then ends with that layer on top.
    static void rs64VrDrawHands(RenderDevice *device, RenderWorker *worker, const ShaderLibrary *shaderLibrary) {
        constexpr uint32_t kHandVertices = 3 * 24 * 13 * 6;
        rs64vr::EyeViews views;
        rs64vr::Pose hands[2] = {};
        int valid[2] = {};
        if (!rs64_vr_hands_frame(&views, hands, valid)) {
            return;
        }

        RenderTexture *images[2] = {};
        uint32_t width = 0;
        uint32_t height = 0;
        for (int e = 0; e < 2; e++) {
            images[e] = rs64_vr_acquire_hands(e, &width, &height);
            if (images[e] == nullptr) {
                return;
            }
        }

        static std::unique_ptr<RenderTexture> depth;
        static uint32_t depthWidth = 0;
        static uint32_t depthHeight = 0;
        static std::unordered_map<RenderTexture *, std::unique_ptr<RenderFramebuffer>> framebuffers;

        // The runtime's controller models, uploaded once each when they load.
        static std::unique_ptr<RenderBuffer> models[2];
        static uint32_t modelVertices[2] = {};
        for (int h = 0; h < 2; h++) {
            const float *vertices = nullptr;
            uint32_t count = 0;
            if ((models[h] == nullptr) && rs64_vr_hand_model(h, &vertices, &count) && (count > 0)) {
                const size_t size = size_t(count) * sizeof(float) * 6;
                models[h] = device->createBuffer(RenderBufferDesc::VertexBuffer(size, RenderHeapType::UPLOAD));
                std::memcpy(models[h]->map(), vertices, size);
                models[h]->unmap();
                modelVertices[h] = count;
            }
        }
        if ((depth == nullptr) || (depthWidth != width) || (depthHeight != height)) {
            framebuffers.clear();
            depth = device->createTexture(RenderTextureDesc::DepthTarget(width, height, RenderFormat::D32_FLOAT));
            depth->setName("VR Controllers Depth");
            depthWidth = width;
            depthHeight = height;
        }

        RenderCommandList *commandList = worker->commandList.get();
        commandList->begin();
        for (int e = 0; e < 2; e++) {
            std::unique_ptr<RenderFramebuffer> &framebuffer = framebuffers[images[e]];
            if (framebuffer == nullptr) {
                const RenderTexture *color = images[e];
                framebuffer = device->createFramebuffer(RenderFramebufferDesc(&color, 1, depth.get()));
            }

            commandList->barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(images[e], RenderTextureLayout::COLOR_WRITE));
            commandList->barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(depth.get(), RenderTextureLayout::DEPTH_WRITE));
            commandList->setFramebuffer(framebuffer.get());
            commandList->clearColor(0, RenderColor(0.0f, 0.0f, 0.0f, 0.0f));
            commandList->clearDepth();
            commandList->setViewports(RenderViewport(0.0f, 0.0f, float(width), float(height)));
            commandList->setScissors(RenderRect(0, 0, int32_t(width), int32_t(height)));

            const RS64VrMat viewProj = rs64VrMul(rs64VrProjection(views.fov[e]), rs64VrPoseMatrix(views.eye[e], true));
            for (int h = 0; h < 2; h++) {
                if (!valid[h]) {
                    continue;
                }

                const RS64VrMat grip = rs64VrPoseMatrix(hands[h], false);
                const RS64VrMat clip = rs64VrMul(viewProj, grip);
                interop::RS64VrHandsCB cb;
                for (int r = 0; r < 4; r++) {
                    cb.clip[r] = { clip.m[r][0], clip.m[r][1], clip.m[r][2], clip.m[r][3] };
                }

                for (int r = 0; r < 3; r++) {
                    cb.rotation[r] = { grip.m[r][0], grip.m[r][1], grip.m[r][2], 0.0f };
                }

                cb.color = { 0.16f, 0.16f, 0.18f, 1.0f };
                if (models[h] != nullptr) {
                    const RenderVertexBufferView view(RenderBufferReference(models[h].get()), modelVertices[h] * uint32_t(sizeof(float) * 6));
                    const RenderInputSlot slot(0, sizeof(float) * 6);
                    commandList->setPipeline(shaderLibrary->rs64VrHandsMesh.pipeline.get());
                    commandList->setGraphicsPipelineLayout(shaderLibrary->rs64VrHandsMesh.pipelineLayout.get());
                    commandList->setGraphicsPushConstants(0, &cb);
                    commandList->setVertexBuffers(0, &view, 1, &slot);
                    commandList->drawInstanced(modelVertices[h], 1, 0, 0);
                }
                else {
                    commandList->setPipeline(shaderLibrary->rs64VrHands.pipeline.get());
                    commandList->setGraphicsPipelineLayout(shaderLibrary->rs64VrHands.pipelineLayout.get());
                    commandList->setGraphicsPushConstants(0, &cb);
                    commandList->drawInstanced(kHandVertices, 1, 0, 0);
                }
            }
        }

        commandList->end();
        worker->execute();
        worker->wait();
    }

    // VR: one headset frame showing the last eyes again, while the game presents nothing (loads, holds); without it the runtime shows its loading screen.
    static void rs64VrKeepAlive(RenderDevice *device, RenderWorker *worker, const ShaderLibrary *shaderLibrary) {
        const int render = rs64_vr_begin_frame();
        rs64VrFade(false, false);
        if (render == 1) {
            rs64VrDrawHands(device, worker, shaderLibrary);
            rs64VrRepeat();
            rs64VrStats.keepAlive++;
        }
        else if (render == 0) {
            rs64_vr_end_frame(nullptr);
            rs64VrStats.noRender++;
        }
        else {
            rs64VrStats.failed++;
        }

        rs64VrLogStats();
    }

    // VR: raw copies of both eye targets into the headset's swapchain images, then the frame ends with the poses they were rendered at.
    // The targets hold display-encoded RGBA8 and the images are sRGB, so the bytes go across unchanged.
    static bool rs64SubmitEyes(RenderWorker *worker, const ShaderLibrary *shaderLibrary, RenderTarget *left, RenderTarget *right, const rs64vr::EyeViews &views) {
        RenderTarget *targets[2] = { left, right };
        RenderTexture *images[2] = {};
        uint32_t width = 0;
        uint32_t height = 0;
        for (int e = 0; e < 2; e++) {
            images[e] = rs64_vr_acquire_eye(e, &width, &height);
        }

        if ((images[0] == nullptr) || (images[1] == nullptr)) {
            return false;
        }

        static uint32_t s_loggedSize = 0;
        const uint32_t sizeKey = (left->width << 16) ^ left->height;
        if (rs64vr::logEnabled() && (sizeKey != s_loggedSize)) {
            s_loggedSize = sizeKey;
            fprintf(stderr, "[vr] eye target %ux%u (right %ux%u), swapchain %ux%u\n", left->width, left->height, right->width, right->height, width, height);
        }

        RenderCommandList *commandList = worker->commandList.get();
        commandList->begin();
        for (int e = 0; e < 2; e++) {
            targets[e]->resolveTarget(worker, shaderLibrary);
            RenderTexture *source = targets[e]->getResolvedTexture();
            const RenderBox box(0, 0, int32_t(std::min(targets[e]->width, width)), int32_t(std::min(targets[e]->height, height)));
            commandList->barriers(RenderBarrierStage::COPY, RenderTextureBarrier(source, RenderTextureLayout::COPY_SOURCE));
            commandList->barriers(RenderBarrierStage::COPY, RenderTextureBarrier(images[e], RenderTextureLayout::COPY_DEST));
            commandList->copyTextureRegion(RenderTextureCopyLocation::Subresource(images[e]), RenderTextureCopyLocation::Subresource(source), 0, 0, 0, &box);
            commandList->barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(source, RenderTextureLayout::SHADER_READ));
            commandList->barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(images[e], RenderTextureLayout::COLOR_WRITE));
        }

        commandList->end();
        worker->execute();
        worker->wait();
        rs64_vr_end_frame(&views);
        return true;
    }

    static void rs64WriteScreenshot() {
        RS64Screenshot &s = rs64Screenshot;
        const uint8_t *src = static_cast<const uint8_t *>(s.buffer->map());
        const uint32_t fw = s.outWidth;
        const uint32_t oh = s.outHeight;
        std::vector<uint8_t> raw(size_t(fw) * oh * 4);
        for (uint32_t y = 0; y < oh; y++) {
            for (uint32_t x = 0; x < fw; x++) {
                uint8_t *dst = &raw[(size_t(y) * fw + x) * 4];
                if (s.rgba8) {
                    const uint8_t *p = &src[(size_t(y) * s.rowWidth + x) * 4];
                    dst[0] = p[0];
                    dst[1] = p[1];
                    dst[2] = p[2];
                }
                else {
                    // High byte of each little-endian 16-bit channel.
                    const uint8_t *p = &src[(size_t(y) * s.rowWidth + x) * 8];
                    dst[0] = p[1];
                    dst[1] = p[3];
                    dst[2] = p[5];
                }

                dst[3] = 255;
            }
        }

        // Stretch the rows to the width the screen shows them at, blending the two nearest source pixels.
        const uint32_t ow = std::max(uint32_t(std::lround(double(fw) * s.pixelAspect)), 1U);
        std::vector<uint8_t> rgba(size_t(ow) * oh * 4);
        for (uint32_t x = 0; x < ow; x++) {
            const double sx = std::clamp((double(x) + 0.5) * double(fw) / double(ow) - 0.5, 0.0, double(fw - 1));
            const uint32_t x0 = uint32_t(sx);
            const uint32_t x1 = std::min(x0 + 1, fw - 1);
            const double t = sx - double(x0);
            for (uint32_t y = 0; y < oh; y++) {
                const uint8_t *a = &raw[(size_t(y) * fw + x0) * 4];
                const uint8_t *b = &raw[(size_t(y) * fw + x1) * 4];
                uint8_t *dst = &rgba[(size_t(y) * ow + x) * 4];
                for (int c = 0; c < 4; c++) {
                    dst[c] = uint8_t(std::lround(double(a[c]) * (1.0 - t) + double(b[c]) * t));
                }
            }
        }

        s.buffer->unmap();

        if (s.requested) {
            std::unordered_set<uint32_t> colours;
            for (size_t i = 0; (i < rgba.size()) && (colours.size() < 65536); i += 4 * 7) {
                uint32_t c;
                std::memcpy(&c, &rgba[i], 4);
                colours.insert(c);
            }

            const std::scoped_lock lock(rs64CaptureMutex);
            const std::filesystem::path out(rs64CapturePath);
            std::error_code ec;
            if (out.has_parent_path()) {
                std::filesystem::create_directories(out.parent_path(), ec);
            }

            const bool ok = stbi_write_png(rs64CapturePath.c_str(), int(ow), int(oh), 4, rgba.data(), int(ow) * 4) != 0;
            rs64CaptureLast = { ok, ow, oh, uint32_t(colours.size()) };
            rs64CaptureWritten = rs64CaptureTaken;
            rs64CaptureCondition.notify_all();
            return;
        }

        std::error_code ec;
        std::filesystem::create_directories("screenshots", ec);
        char path[64];
        std::snprintf(path, sizeof(path), "screenshots/shot_%03d.png", s.taken);
        stbi_write_png(path, int(ow), int(oh), 4, rgba.data(), int(ow) * 4);
        s.taken++;
        std::fprintf(stderr, "[screenshot] %s (%ux%u)\n", path, ow, oh);
    }

    // PresentQueue

    PresentQueue::PresentQueue() {
        reset();
    }

    PresentQueue::~PresentQueue() {
        presentThreadRunning = false;
        cursorCondition.notify_all();

        if (presentThread != nullptr) {
            presentThread->join();
            delete presentThread;
        }

        presentIdCondition.notify_all();
    }

    void PresentQueue::reset() {
        threadCursor = 0;
        writeCursor = 0;
        barrierCursor = 0;
        presentId = 0;
    }

    void PresentQueue::advanceToNextPresent() {
        int nextWriteCursor = (writeCursor + 1) % presents.size();

        // Stall the thread until the barrier is lifted if we're trying to write on a present being used by the GPU.
        bool waitForBarrier;
        do {
            const std::scoped_lock lock(cursorMutex);
            waitForBarrier = (nextWriteCursor == barrierCursor);
        } while (waitForBarrier);

        // Modify the cursor and notify anything waiting on the queue.
        {
            const std::scoped_lock lock(cursorMutex);
            writeCursor = nextWriteCursor;
        }

        cursorCondition.notify_all();
    }

    void PresentQueue::repeatLastPresent() {
        {
            const std::scoped_lock lock(cursorMutex);
            threadCursor = previousWriteCursor();
        }

        cursorCondition.notify_all();
    }

    uint32_t PresentQueue::previousWriteCursor() const {
        if (writeCursor > 0) {
            return writeCursor - 1;
        }
        else {
            return uint32_t(presents.size()) - 1;
        }
    }

    void PresentQueue::waitForIdle() {
        std::unique_lock<std::mutex> threadLock(threadMutex);
    }

    void PresentQueue::waitForPresentId(uint64_t waitId) {
        std::unique_lock<std::mutex> presentLock(presentIdMutex);
        presentIdCondition.wait(presentLock, [&]() {
            return (waitId <= presentId) || !presentThreadRunning;
        });
    }

    void PresentQueue::setup(const External &ext) {
        this->ext = ext;

        viRenderer = std::make_unique<VIRenderer>();

        presentThreadRunning = true;
        presentThread = new std::thread(&PresentQueue::threadLoop, this);
    }

    bool PresentQueue::rs64SubmitScreen(RenderTarget *colorTarget, const VI &vi, UserConfiguration::Filtering filtering, bool removeBlackBorders, bool letterbox43) {
        uint32_t width = 0;
        uint32_t height = 0;
        RenderTexture *image = rs64_vr_acquire_screen(&width, &height);
        if (image == nullptr) {
            return false;
        }

        if ((vrScreenTexture == nullptr) || (vrScreenWidth != width) || (vrScreenHeight != height)) {
            vrScreenFramebuffer.reset();
            vrScreenTexture = ext.device->createTexture(RenderTextureDesc::ColorTarget(width, height, SwapChainFormat));
            vrScreenTexture->setName("VR Screen");
            const RenderTexture *screenTarget = vrScreenTexture.get();
            vrScreenFramebuffer = ext.device->createFramebuffer(RenderFramebufferDesc(&screenTarget, 1));
            vrScreenWidth = width;
            vrScreenHeight = height;
        }

        RenderCommandList *commandList = ext.presentGraphicsWorker->commandList.get();
        commandList->begin();

        // The same VI pass the window uses (crop, letterbox, pixel aspect), sized to the virtual screen.
        VIRenderer::RenderParams params;
        params.device = ext.device;
        params.commandList = commandList;
        params.swapChain = ext.swapChain;
        params.shaderLibrary = ext.shaderLibrary;
        params.textureFormat = colorTarget->format;
        params.resolutionScale = colorTarget->resolutionScale;
        params.downsamplingScale = 1;
        params.filtering = filtering;
        params.vi = &vi;
        params.removeBlackBorders = removeBlackBorders;
        params.letterbox43 = letterbox43;
        params.targetWidth = width;
        params.targetHeight = height;
        if (colorTarget->downsampleMultiplier > 1) {
            colorTarget->downsampleTarget(ext.presentGraphicsWorker, ext.shaderLibrary);
            params.texture = colorTarget->downsampledTexture.get();
            params.textureWidth = colorTarget->width / colorTarget->downsampleMultiplier;
            params.textureHeight = colorTarget->height / colorTarget->downsampleMultiplier;
            params.downsamplingScale = colorTarget->downsampleMultiplier;
        }
        else {
            colorTarget->resolveTarget(ext.presentGraphicsWorker, ext.shaderLibrary);
            params.texture = colorTarget->getResolvedTexture();
            params.textureWidth = colorTarget->width;
            params.textureHeight = colorTarget->height;
        }

        commandList->barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(vrScreenTexture.get(), RenderTextureLayout::COLOR_WRITE));
        commandList->setFramebuffer(vrScreenFramebuffer.get());
        commandList->clearColor();
        commandList->barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(params.texture, RenderTextureLayout::SHADER_READ));
        viRenderer->render(params);

        // Display-encoded BGRA8 into the B8G8R8A8 sRGB image, bytes unchanged, as the eyes do.
        const RenderBox box(0, 0, int32_t(width), int32_t(height));
        commandList->barriers(RenderBarrierStage::COPY, RenderTextureBarrier(vrScreenTexture.get(), RenderTextureLayout::COPY_SOURCE));
        commandList->barriers(RenderBarrierStage::COPY, RenderTextureBarrier(image, RenderTextureLayout::COPY_DEST));
        commandList->copyTextureRegion(RenderTextureCopyLocation::Subresource(image), RenderTextureCopyLocation::Subresource(vrScreenTexture.get()), 0, 0, 0, &box);
        commandList->barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(image, RenderTextureLayout::COLOR_WRITE));
        commandList->end();
        ext.presentGraphicsWorker->execute();
        ext.presentGraphicsWorker->wait();
        rs64_vr_end_frame_screen();
        return true;
    }

    void PresentQueue::threadPresent(const Present &present, bool &swapChainValid) {
        FramebufferManager &fbManager = ext.sharedResources->framebufferManager;
        RenderTargetManager &targetManager = ext.sharedResources->renderTargetManager;
        hlslpp::float2 resolutionScale;
        EnhancementConfiguration::Presentation::Mode presentationMode;
        bool removeBlackBorders;
        bool letterbox43;
        UserConfiguration::RefreshRate refreshRate;
        UserConfiguration::Filtering filtering;
        bool fxaa;
        uint32_t viOriginalRate;
        uint32_t targetRate;
        {
            std::scoped_lock<std::mutex> configurationLock(ext.sharedResources->configurationMutex);
            resolutionScale = ext.sharedResources->resolutionScale;
            presentationMode = ext.sharedResources->enhancementConfig.presentation.mode;
            removeBlackBorders = ext.sharedResources->enhancementConfig.presentation.removeBlackBorders;
            letterbox43 = (ext.sharedResources->userConfig.aspectRatio == UserConfiguration::AspectRatio::Original);
            refreshRate = ext.sharedResources->userConfig.refreshRate;
            filtering = ext.sharedResources->userConfig.filtering;
            fxaa = ext.sharedResources->userConfig.fxaa;
            viOriginalRate = ext.sharedResources->viOriginalRate;
            targetRate = ext.sharedResources->targetRate;
        }

        RenderTarget *colorTarget = nullptr;
        int32_t framesToPresent = 1;
        bool lockedWorkloadMutex = false;
        InterpolatedFrameCounters &frameCounters = ext.sharedResources->interpolatedFrames[ext.sharedResources->interpolatedFramesIndex];
        const bool usingMSAA = frameCounters.msaa;
        const uint32_t vrSet = ext.sharedResources->interpolatedFramesIndex;

        // TODO: There's a possible race condition interactions that can happen while the workload
        // queue is rendering extra frames and the present event is processed while it's generating
        // interpolated frames. When the framebuffer manager or the render target manager maps are
        // modified while the present queue is retrieving the framebuffer or the target. These can
        // likely be solved by locking the access to the managers during modification.
        
        // Perform any external write operations indicated by the event.
        if (!present.fbOperations.empty()) {
            const std::scoped_lock lock(screenFbChangePoolMutex);
            {
                RenderWorkerExecution workerExecution(ext.presentGraphicsWorker);
                fbManager.performOperations(ext.presentGraphicsWorker, &screenFbChangePool, nullptr, ext.shaderLibrary, nullptr,
                    present.fbOperations, targetManager, resolutionScale, 0, 0, nullptr);
            }
        }

        // Present the VI specified by the event.
        // Attempt to find the matching framebuffer for the VI based on the origin address.
        // If that fails, we look at the shared storage.
        if (present.screenVI.visible()) {
            Framebuffer *viFb = nullptr;
            if (!viewRDRAM) {
                viFb = fbManager.find(present.screenVI.fbAddress());
            }

            // VI_FOLLOW_DRAW (2026-05-13, ported from reverted commit e718774).
            //
            // Rogue Squadron's attribution period has VI sampling a lo-res fb
            // (0x6BA000/0x6DD000, never drawn to) while the game renders to a
            // hi-res scratch fb (0x62B800). Result without fix: VI presents
            // an empty buffer = pure black, even though there's valid
            // attribution-text content rendered at 0x62B800.
            //
            // Workaround: when VI's chosen fb is NOT in the recently-written
            // color-image set, present the most-recently-written color fb
            // instead. The present worker composites at the rt's native
            // dimensions, bypassing VI's stale width register entirely.
            //
            // ROGUESQ_VI_FOLLOW_DRAW=0 disables (VI lookup only).
            if (!viewRDRAM) {
                static const bool s_follow = []() {
                    const char *e = std::getenv("ROGUESQ_VI_FOLLOW_DRAW");
                    return !(e && *e == '0');
                }();
                const uint32_t viAddr = present.screenVI.fbAddress();
                if (s_follow && !ext.sharedResources->colorImageAddressVector.empty())
                {
                    const auto &vec = ext.sharedResources->colorImageAddressVector;
                    // Override only when VI's fb is stale.
                    bool viInRecent = false;
                    for (uint32_t a : vec) {
                        if (a == viAddr) { viInRecent = true; break; }
                    }
                    if (!viInRecent) {
                        // Pick the first vec entry that's NOT the VI address.
                        // For Rogue Squadron attribution, VI cycles between
                        // lo-res fbs that only receive fillRect clears,
                        // while the actual text content lands in a different
                        // (hi-res) fb. Front of vec may be the lo-res clear
                        // (most recent setColorImage), so we skip it and
                        // pick the next distinct addr.
                        uint32_t recentAddr = 0;
                        bool found = false;
                        for (uint32_t a : vec) {
                            if (a != viAddr) {
                                recentAddr = a;
                                found = true;
                                break;
                            }
                        }
                        if (found) {
                            Framebuffer *recentFb = fbManager.find(recentAddr);
                            if (recentFb != nullptr) {
                                viFb = recentFb;
                                static std::atomic<uint64_t> ovr{0};
                                uint64_t v = ++ovr;
                                if (v <= 5 || (v % 60) == 0) {
                                    std::fprintf(stderr,
                                        "[vi-follow-draw] #%llu viAddr=0x%08X "
                                        "-> override to recentAddr=0x%08X (setSize=%zu)\n",
                                        (unsigned long long)v,
                                        viAddr, recentAddr, vec.size());
                                    std::fflush(stderr);
                                }
                            }
                        }
                    }
                }
            }

            Framebuffer *presentFb = viFb;

            // The first buffer of a new video mode is CPU-zeroed (black on hardware) but RT64 still holds an older target there.
            static const bool s_tr_gate = []() {
                const char *v = std::getenv("ROGUESQ_NO_TRANSITION_BLANK");
                return !(v && v[0] && v[0] != '0');
            }();
            static rs64transition::Gate s_trGate;
            const uint32_t trArmed = g_rs64_cpu_cleared_fb.exchange(0, std::memory_order_acq_rel);
            bool transitionBlack = false;
            uint64_t trDrawnWid = 0;
            if (s_tr_gate) {
                if (trArmed != 0) {
                    s_trGate.arm(trArmed, g_rs64_cpu_cleared_wid.load(std::memory_order_acquire));
                }

                // The vector is rebuilt by the workload thread without a lock here; only trust a hit when its workload id is unchanged across the scan.
                if (s_trGate.armedFb() != 0) {
                    const uint64_t widBefore = ext.sharedResources->colorImageWorkloadId.load(std::memory_order_acquire);
                    bool hit = false;
                    for (uint32_t a : ext.sharedResources->colorImageAddressVector) {
                        if (rs64transition::viShowsBuffer(a, s_trGate.armedFb())) {
                            hit = true;
                            break;
                        }
                    }

                    const uint64_t widAfter = ext.sharedResources->colorImageWorkloadId.load(std::memory_order_acquire);
                    if (hit && (widBefore == widAfter)) {
                        trDrawnWid = widAfter;
                    }
                }

                transitionBlack = s_trGate.shouldBlack(present.screenVI.fbAddress(), trDrawnWid);
                g_rs64_transition_gate_fb.store(s_trGate.armedFb(), std::memory_order_release);
            }

            // A zero-height VI (the game's mode-change blank, vStart == vEnd) gives fbSize (0,0); the VI pass would divide by it and paint the whole swapchain.
            static const bool s_zeroSizeDraw = []() {
                const char *v = std::getenv("ROGUESQ_VI_ZERO_SIZE_DRAW");
                return v && v[0] && v[0] != '0';
            }();
            const hlslpp::uint2 trFbSize = present.screenVI.fbSize();
            const bool zeroSizeVI = (trFbSize.x == 0) || (trFbSize.y == 0);
            if (zeroSizeVI && !s_zeroSizeDraw) {
                transitionBlack = true;
            }

            // ROGUESQ_LOG_TRANSITION=1: every present for 120 presents after each mode-change buffer clear.
            static const bool s_tr_log = []() {
                const char *v = std::getenv("ROGUESQ_LOG_TRANSITION");
                return v && v[0] && v[0] != '0';
            }();
            if (s_tr_log) {
                static uint32_t s_lastArm = 0;
                static int s_left = 0;
                if (trArmed != 0) {
                    s_lastArm = trArmed;
                    s_left = 120;
                }

                if (zeroSizeVI) {
                    std::fprintf(stderr, "[transition] zero-size VI present vi=0x%08X w=%u vRegion=0x%08X hRegion=0x%08X presentWid=%llu drawn=%d\n",
                        present.screenVI.fbAddress(), (unsigned)present.screenVI.width, (unsigned)present.screenVI.vRegion.word, (unsigned)present.screenVI.hRegion.word,
                        (unsigned long long)present.workloadId, s_zeroSizeDraw ? 1 : 0);
                    std::fflush(stderr);
                }

                // Presents before a clear go to a ring that is dumped when the clear arrives.
                constexpr int kRing = 160;
                static std::string s_ring[kRing];
                static int s_head = 0;
                if (trArmed != 0) {
                    std::fprintf(stderr, "[transition] ---- last %d presents before clear 0x%06X ----\n", kRing, trArmed);
                    for (int i = 0; i < kRing; i++) {
                        const std::string &e = s_ring[(s_head + i) % kRing];
                        if (!e.empty()) {
                            std::fprintf(stderr, "%s\n", e.c_str());
                        }
                    }

                    std::fprintf(stderr, "[transition] ---- end ring ----\n");
                }

                const long long ms = (long long)(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count() % 100000);
                char line[320];
                std::snprintf(line, sizeof(line), "[transition] present t=%05lld vi=0x%08X w=%u size=%ux%u chosen=0x%08X lastWrite=%d branch=%s armed=0x%06X black=%d armWid=%llu drawnWid=%llu presentWid=%llu",
                    ms, present.screenVI.fbAddress(), (unsigned)present.screenVI.width, (unsigned)trFbSize.x, (unsigned)trFbSize.y, (presentFb != nullptr) ? presentFb->addressStart : 0u,
                    (presentFb != nullptr) ? (int)presentFb->lastWriteType : -1, (viFb != nullptr) ? "A" : "B", s_lastArm, transitionBlack ? 1 : 0,
                    (unsigned long long)s_trGate.armWorkloadId, (unsigned long long)trDrawnWid, (unsigned long long)present.workloadId);
                if (s_left > 0) {
                    --s_left;
                    std::fprintf(stderr, "%s\n", line);
                    std::fflush(stderr);
                }
                else {
                    s_ring[s_head] = line;
                    s_head = (s_head + 1) % kRing;
                }
            }

            // Show the framebuffer the debugger has requested instead.
            if (present.debuggerFramebuffer.view) {
                Framebuffer *candidateFb = fbManager.find(present.debuggerFramebuffer.address);
                if (candidateFb != nullptr) {
                    presentFb = candidateFb;
                }
            }
            
            if (transitionBlack) {
                // colorTarget stays null, so the swapchain is cleared to black.
            }
            else if ((presentFb != nullptr) && (viFb != nullptr)) {
                for (uint32_t colorAddress : ext.sharedResources->colorImageAddressVector) {
                    Framebuffer *colorFb = fbManager.find(colorAddress);
                    if (colorFb == nullptr) {
                        continue;
                    }

                    // Always default to interpolation being disabled for all modified framebuffers.
                    colorFb->interpolationEnabled = false;
                    
                    // When the skip buffering option is on, we check the video history to find if any of the framebuffers that
                    // were drawn in this frame have been previously used for presentation. This is ignored when the debugger
                    // has forced viewing a particular framebuffer.
                    if (!present.debuggerFramebuffer.view && (presentationMode == EnhancementConfiguration::Presentation::Mode::SkipBuffering)) {
                        for (size_t h = 0; h < viHistory.history.size(); h++) {
                            const VIHistory::Present &entry = viHistory.history[h];
                            if ((colorFb->addressStart == entry.vi.fbAddress()) && (colorFb->width == entry.fbWidth) && (colorFb->siz == entry.vi.fbSiz()) && entry.vi.compatibleWith(present.screenVI)) {
                                presentFb = colorFb;
                                break;
                            }
                        }
                    }

                    // Present early (or games that behave like it) will make it so that the presented image is a color image
                    // that the workload modified. We run a basic check to see if that holds true to indicate it was presented
                    // so interpolation is possible.
                    if (colorFb == presentFb) {
                        presentFb->interpolationEnabled = true;
                        break;
                    }
                }

                if (presentFb->interpolationEnabled) {
                    framesToPresent = frameCounters.count;
                }
                else {
                    lockedWorkloadMutex = true;
                    ext.sharedResources->workloadMutex.lock();
                }

                RenderTargetKey colorTargetKey(presentFb->addressStart, presentFb->width, presentFb->siz, Framebuffer::Type::Color);
                colorTarget = &targetManager.get(colorTargetKey, true);
                static const bool s_depthAsColor = []() {
                    const char *v = std::getenv("ROGUESQ_PRESENT_DEPTH_AS_COLOR");
                    return v && v[0] && v[0] != '0';
                }();
                if (!colorTarget->isEmpty()) {
                    // A buffer last written as depth is a stale Z buffer at a reused address; show black, not Z-as-RGBA16 white.
                    if (!s_depthAsColor && (presentFb->lastWriteType == Framebuffer::Type::Depth)) {
                        colorTarget = nullptr;
                    }
                    // If a depth framebuffer is about to be shown, convert it to color.
                    else if (presentFb->isLastWriteDifferent(Framebuffer::Type::Color)) {
                        RenderTargetKey otherColorTargetKey(presentFb->addressStart, presentFb->width, presentFb->siz, presentFb->lastWriteType);
                        RenderTarget &otherColorTarget = targetManager.get(otherColorTargetKey, true);
                        if (!otherColorTarget.isEmpty()) {
                            const FixedRect &r = presentFb->lastWriteRect;
                            RenderWorkerExecution workerExecution(ext.presentGraphicsWorker);
                            colorTarget->copyFromTarget(ext.presentGraphicsWorker, &otherColorTarget, r.left(false), r.top(false), r.width(false, true), r.height(false, true), ext.shaderLibrary);
                        }
                    }
                }
                else {
                    colorTarget = nullptr;
                }

                if (!present.paused && (viHistory.top().vi != present.screenVI)) {
                    viHistory.pushVI(present.screenVI, viFb->width);
                }
            }
            else {
                uint32_t fbAddress = present.screenVI.fbAddress();

                // Use a scratch framebuffer to upload the RAM to the render target.
                hlslpp::uint2 fbSize = present.screenVI.fbSize();
                scratchFb.addressStart = fbAddress;
                scratchFb.width = fbSize.x;
                scratchFb.height = fbSize.y;
                scratchFb.siz = present.screenVI.fbSiz();

                lockedWorkloadMutex = true;
                ext.sharedResources->workloadMutex.lock();

                RenderTargetKey colorTargetKey(fbAddress, scratchFb.width, scratchFb.siz, Framebuffer::Type::Color);
                colorTarget = &targetManager.get(colorTargetKey, true);
                colorTarget->resize(ext.presentGraphicsWorker, scratchFb.width, scratchFb.height);
                colorTarget->resolutionScale = { 1.0f, 1.0f };
                colorTarget->downsampleMultiplier = 1;

                scratchFb.nativeTarget.resetBufferHistory();

                {
                    RenderWorkerExecution workerExecution(ext.presentGraphicsWorker);
                    colorTarget->clearColorTarget(ext.presentGraphicsWorker);
                    FramebufferChange *colorFbChange = scratchFb.readChangeFromBytes(ext.presentGraphicsWorker, scratchFbChangePool, Framebuffer::Type::Color,
                        G_IM_FMT_RGBA, present.storage.data(), 0, scratchFb.height, ext.shaderLibrary);

                    if (colorFbChange != nullptr) {
                        colorTarget->copyFromChanges(ext.presentGraphicsWorker, *colorFbChange, scratchFb.width, scratchFb.height, 0, ext.shaderLibrary);
                    }
                }

                scratchFbChangePool.reset();

                if (!present.paused && (viHistory.top().vi != present.screenVI)) {
                    viHistory.pushVI(present.screenVI, fbSize.x);
                }
            }
        }

        // Create the framebuffers if necessary.
        if (swapChainFramebuffers.empty()) {
            uint32_t textureCount = ext.swapChain->getTextureCount();
            swapChainFramebuffers.resize(textureCount);
            for (uint32_t i = 0; i < textureCount; i++) {
                const RenderTexture *swapChainTexture = ext.swapChain->getTexture(i);
                swapChainFramebuffers[i] = ext.device->createFramebuffer(RenderFramebufferDesc(&swapChainTexture, 1));
            }
        }
        
        for (int32_t i = 0; i < framesToPresent; i++) {
            uint32_t frameCountersNextPresented = 0;
            if (((framesToPresent > 1) || frameCounters.vr) && (usingMSAA || (i > 0))) {
                // Stall until the interpolated color target is available.
                const uint32_t targetIndex = usingMSAA ? i : (i - 1);
                std::unique_lock<std::mutex> interpolatedLock(ext.sharedResources->interpolatedMutex);
                ext.sharedResources->interpolatedCondition.wait(interpolatedLock, [&]() {
                    return (frameCounters.available > targetIndex) || ((frameCounters.available == targetIndex) && frameCounters.skipped);
                });

                // Do not present any more frames after this one after reaching the last available frame if the workload was skipped.
                if ((frameCounters.available == targetIndex) && frameCounters.skipped) {
                    framesToPresent = std::min(int(frameCounters.available), i + 1);
                    frameCountersNextPresented = frameCounters.count;
                }
                else {
                    frameCountersNextPresented = frameCounters.presented + 1;
                }

                if (i < framesToPresent) {
                    uint32_t targetIndex = usingMSAA ? i : (i - 1);
                    colorTarget = ext.sharedResources->interpolatedColorTargets[targetIndex].get();
                }
                else {
                    colorTarget = nullptr;
                }
            }
            else if (framesToPresent == 1) {
                frameCountersNextPresented = frameCounters.count;
            }

            // VR: xrWaitFrame paces this loop. A frame with no new eyes shows the last ones again (the layer keeps each swapchain's last
            // released image); ending a frame with no layers would clear the display.
            const bool vrSession = (rs64_vr_session() != 0);
            if (vrSession && (i < framesToPresent)) {
                RS64VrStats &vs = rs64VrStats;
                const int vrRender = rs64_vr_begin_frame();
                if (vrRender == 1) {
                    rs64VrDrawHands(ext.device, ext.presentGraphicsWorker, ext.shaderLibrary);
                }

                std::vector<std::unique_ptr<RenderTarget>> &vrRight = ext.sharedResources->vrRightTargets[vrSet];
                std::vector<rs64vr::EyeViews> &vrViews = ext.sharedResources->vrTargetViews[vrSet];
                // Only a workload that rendered both eyes into its own targets has a pair; the vectors keep older frames' eyes otherwise.
                const bool haveEyes = frameCounters.vr && (colorTarget != nullptr) && (uint32_t(i) < vrRight.size()) && (uint32_t(i) < vrViews.size()) && vrViews[i].valid;
                const bool vrFresh = (vrRender == 1) && (colorTarget != nullptr) && (frameCounters.vrScreen || haveEyes);
                const bool vrHoldOld = rs64VrFade(vrFresh, frameCounters.vrScreen);
                if ((vrRender == 1) && vrHoldOld) {
                    rs64VrRepeat();
                    vs.fadeHeld++;
                }
                else if ((vrRender == 1) && frameCounters.vrScreen && (colorTarget != nullptr) && rs64SubmitScreen(colorTarget, present.screenVI, filtering, removeBlackBorders, letterbox43)) {
                    rs64VrLastScreen = true;
                    vs.screen++;
                }
                else if ((vrRender == 1) && !frameCounters.vrScreen && haveEyes && rs64SubmitEyes(ext.presentGraphicsWorker, ext.shaderLibrary, colorTarget, vrRight[i].get(), vrViews[i])) {
                    rs64VrLastViews = vrViews[i];
                    rs64VrLastScreen = false;
                    vs.fresh++;
                }
                else if (vrRender == 1) {
                    rs64VrRepeat();
                    if (colorTarget == nullptr) vs.noColor++;
                    else if (!frameCounters.vr) vs.noEyes++;
                    else if (uint32_t(i) >= vrRight.size()) vs.noRight++;
                    else if ((uint32_t(i) >= vrViews.size()) || !vrViews[i].valid) vs.noViews++;
                    else vs.submitFail++;
                }
                else if (vrRender == 0) {
                    rs64_vr_end_frame(nullptr);
                    vs.noRender++;
                }
                else {
                    vs.failed++;
                }

                rs64VrLogStats();
            }

            // The window mirrors the left eye at most 30 times a second, so a 60 Hz FIFO swapchain never holds back the headset.
            static Timestamp vrWindowTimestamp;
            const bool vrWindowDue = !vrSession || (vrWindowTimestamp == Timestamp()) || (Timer::current() - vrWindowTimestamp >= std::chrono::milliseconds(33));

            uint32_t swapChainIndex = 0;
            const bool presentFrame = (i < framesToPresent) && swapChainValid && vrWindowDue;
            if (presentFrame) {
                swapChainValid = ext.swapChain->acquireTexture(acquiredSemaphore.get(), &swapChainIndex);
                vrWindowTimestamp = Timer::current();
            }

            if (presentFrame && swapChainValid) {
                // Draw the framebuffer with the VI renderer.
                RenderTexture *swapChainTexture = ext.swapChain->getTexture(swapChainIndex);
                RenderFramebuffer *swapChainFramebuffer = swapChainFramebuffers[swapChainIndex].get();
                RenderCommandList *commandList = ext.presentGraphicsWorker->commandList.get();
                commandList->begin();
                commandList->barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(swapChainTexture, RenderTextureLayout::COLOR_WRITE));
                
                VIRenderer::RenderParams renderParams;
                if (colorTarget != nullptr) {
                    renderParams.device = ext.device;
                    renderParams.commandList = commandList;
                    renderParams.swapChain = ext.swapChain;
                    renderParams.shaderLibrary = ext.shaderLibrary;
                    renderParams.textureFormat = colorTarget->format;
                    renderParams.resolutionScale = colorTarget->resolutionScale;
                    renderParams.downsamplingScale = 1;
                    renderParams.filtering = filtering;
                    renderParams.vi = &present.screenVI;
                    renderParams.removeBlackBorders = removeBlackBorders;
                    renderParams.letterbox43 = letterbox43;

                    const bool useDownsampling = (colorTarget->downsampleMultiplier > 1);
                    if (useDownsampling) {
                        colorTarget->downsampleTarget(ext.presentGraphicsWorker, ext.shaderLibrary);
                        renderParams.texture = colorTarget->downsampledTexture.get();
                        renderParams.textureWidth = colorTarget->width / colorTarget->downsampleMultiplier;
                        renderParams.textureHeight = colorTarget->height / colorTarget->downsampleMultiplier;
                        renderParams.downsamplingScale = colorTarget->downsampleMultiplier;
                    }
                    else {
                        colorTarget->resolveTarget(ext.presentGraphicsWorker, ext.shaderLibrary);
                        renderParams.texture = colorTarget->getResolvedTexture();
                        renderParams.textureWidth = colorTarget->width;
                        renderParams.textureHeight = colorTarget->height;
                    }
                }
                
                const bool useFxaa = fxaa && (renderParams.texture != nullptr);
                if (useFxaa) {
                    const uint32_t width = ext.swapChain->getWidth();
                    const uint32_t height = ext.swapChain->getHeight();
                    if ((fxaaTexture == nullptr) || (fxaaWidth != width) || (fxaaHeight != height)) {
                        fxaaFramebuffer.reset();
                        fxaaTexture = ext.device->createTexture(RenderTextureDesc::ColorTarget(width, height, SwapChainFormat));
                        fxaaTexture->setName("FXAA Input");
                        const RenderTexture *fxaaColorTarget = fxaaTexture.get();
                        fxaaFramebuffer = ext.device->createFramebuffer(RenderFramebufferDesc(&fxaaColorTarget, 1));
                        fxaaWidth = width;
                        fxaaHeight = height;
                    }

                    commandList->barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(fxaaTexture.get(), RenderTextureLayout::COLOR_WRITE));
                    commandList->setFramebuffer(fxaaFramebuffer.get());
                }
                else {
                    commandList->setFramebuffer(swapChainFramebuffer);
                }

                commandList->clearColor();

                bool screenshotQueued = false;
                if (renderParams.texture != nullptr) {
                    commandList->barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(renderParams.texture, RenderTextureLayout::SHADER_READ));
                    viRenderer->render(renderParams);
                }

                if (useFxaa) {
                    if (fxaaDescriptorSet == nullptr) {
                        fxaaDescriptorSet = std::make_unique<VideoInterfaceDescriptorSet>(ext.shaderLibrary->samplerLibrary.linear.clampClamp.get(), ext.device);
                    }

                    fxaaDescriptorSet->setTexture(fxaaDescriptorSet->gInput, fxaaTexture.get(), RenderTextureLayout::SHADER_READ);
                    const float fxaaConstants[4] = { 1.0f / float(fxaaWidth), 1.0f / float(fxaaHeight), 0.0f, 0.0f };
                    const ShaderRecord &fxaaShader = ext.shaderLibrary->fxaa;
                    commandList->barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(fxaaTexture.get(), RenderTextureLayout::SHADER_READ));
                    commandList->setFramebuffer(swapChainFramebuffer);
                    commandList->setViewports(RenderViewport(0.0f, 0.0f, float(fxaaWidth), float(fxaaHeight)));
                    commandList->setScissors(RenderRect(0, 0, int32_t(fxaaWidth), int32_t(fxaaHeight)));
                    commandList->setPipeline(fxaaShader.pipeline.get());
                    commandList->setGraphicsPipelineLayout(fxaaShader.pipelineLayout.get());
                    commandList->setGraphicsDescriptorSet(fxaaDescriptorSet->get(), 0);
                    commandList->setGraphicsPushConstants(0, fxaaConstants);
                    commandList->setVertexBuffers(0, nullptr, 0, nullptr);
                    commandList->drawInstanced(3, 1, 0, 0);
                }

                RenderHookDraw *drawHook = GetRenderHookDraw();
                if (drawHook != nullptr) {
                    drawHook(commandList, swapChainFramebuffer);
                }

                {
                    const std::scoped_lock lock(inspectorMutex);
                    if (inspector != nullptr) {
                        inspector->draw(commandList);
                    }

                    // After every draw: the copy ends the render pass, and ImGui records into it directly.
                    if (renderParams.texture != nullptr) {
                        screenshotQueued = rs64QueueScreenshot(ext.device, commandList, renderParams);
                    }

                    commandList->barriers(RenderBarrierStage::NONE, RenderTextureBarrier(swapChainTexture, RenderTextureLayout::PRESENT));
                    commandList->end();
                    const RenderCommandList *commandList = ext.presentGraphicsWorker->commandList.get();
                    RenderCommandSemaphore *waitSemaphore = acquiredSemaphore.get();
                    RenderCommandSemaphore *signalSemaphore = drawSemaphores[swapChainIndex].get();
                    ext.presentGraphicsWorker->commandQueue->executeCommandLists(&commandList, 1, &waitSemaphore, 1, &signalSemaphore, 1, ext.presentGraphicsWorker->commandFence.get());
                    ext.presentGraphicsWorker->wait();
                }

                if (screenshotQueued) {
                    rs64WriteScreenshot();
                }
            }

            if (lockedWorkloadMutex) {
                ext.sharedResources->workloadMutex.unlock();
                lockedWorkloadMutex = false;
            }
            
            if (frameCountersNextPresented > 0) {
                {
                    std::unique_lock<std::mutex> interpolatedLock(ext.sharedResources->interpolatedMutex);
                    frameCounters.presented = frameCountersNextPresented;
                }

                ext.sharedResources->interpolatedCondition.notify_all();
            }

            // As soon as we're done with the first render target, we notify the workload queue it can proceed.
            if (i == 0) {
                notifyPresentId(present);
            }

            if (presentFrame && swapChainValid) {
                // Wait until the approximate time the next present should be at the current intended rate.
                if (!vrSession && (presentTimestamp != Timestamp()) && (targetRate > 0) && (targetRate > viOriginalRate)) {
                    Timer::preciseSleepUntil(presentTimestamp + std::chrono::nanoseconds(1'000'000'000 / targetRate));
                }

                if (presentWaitEnabled) {
                    ext.swapChain->wait();
                }

                RenderCommandSemaphore *waitSemaphore = drawSemaphores[swapChainIndex].get();
                presentTimestamp = Timer::current();
                swapChainValid = ext.swapChain->present(swapChainIndex, &waitSemaphore, 1);
                presentProfiler.logAndRestart();
            }
        }
    }

    void PresentQueue::suspendSurface() {
        // threadMutex is held for a whole present, so taking it waits out any frame still using the swap chain.
        std::unique_lock<std::mutex> threadLock(threadMutex);
        if (surfaceSuspended) {
            return;
        }
        if (rs64LogSwapChain()) {
            std::fprintf(stderr, "[swapchain] surface suspended\n");
        }
        surfaceSuspended = true;
        ext.presentGraphicsWorker->commandList->begin();
        ext.presentGraphicsWorker->commandList->end();
        ext.presentGraphicsWorker->execute();
        ext.presentGraphicsWorker->wait();
        swapChainFramebuffers.clear();
        ext.swapChain->setWindow({});
    }

    void PresentQueue::resumeSurface(RenderWindow renderWindow) {
        std::unique_lock<std::mutex> threadLock(threadMutex);
        if (!surfaceSuspended) {
            return;
        }
        if (rs64LogSwapChain()) {
            std::fprintf(stderr, "[swapchain] surface resumed\n");
        }
        ext.swapChain->setWindow(renderWindow);
        swapChainFramebuffers.clear();
        surfaceRestored = true;
        surfaceSuspended = false;
    }

    void PresentQueue::skipInterpolation() {
        {
            std::unique_lock<std::mutex> interpolatedLock(ext.sharedResources->interpolatedMutex);
            InterpolatedFrameCounters &frameCounters = ext.sharedResources->interpolatedFrames[ext.sharedResources->interpolatedFramesIndex];
            frameCounters.presented = frameCounters.count;
        }

        ext.sharedResources->interpolatedCondition.notify_all();
    }

    void PresentQueue::notifyPresentId(const Present &present) {
        {
            std::scoped_lock<std::mutex> cursorLock(presentIdMutex);
            presentId = present.presentId;
        }

        presentIdCondition.notify_all();
    }
    
    void PresentQueue::threadAdvanceBarrier() {
        std::scoped_lock<std::mutex> cursorLock(cursorMutex);
        barrierCursor = (barrierCursor + 1) % presents.size();
    }

    void PresentQueue::threadLoop() {
        Thread::setCurrentThreadName("RT64 Present");

        // Create the semaphore the acquire method will use.
        acquiredSemaphore = ext.device->createCommandSemaphore();

        // Create as many semaphores to signal as textures there are.
        while (drawSemaphores.size() < ext.swapChain->getTextureCount()) {
            drawSemaphores.emplace_back(ext.device->createCommandSemaphore());
        }

        // Since the swap chain might not need a resize right away, detect present wait.
        presentWaitEnabled = ext.device->getCapabilities().presentWait;

        int processCursor = -1;
        bool skipPresent = false;
        uint32_t displayTimingRate = UINT32_MAX;
        const bool displayTiming = ext.device->getCapabilities().displayTiming;
        bool swapChainValid = !ext.swapChain->needsResize();
        while (presentThreadRunning) {
            {
                std::unique_lock<std::mutex> cursorLock(cursorMutex);
                const auto presentReady = [&]() {
                    return (writeCursor != threadCursor) || !presentThreadRunning;
                };

                // VR: past 50 ms without a present the headset gets the last eyes again each frame (xrWaitFrame paces the loop) until the game presents.
                auto keepAliveWait = std::chrono::milliseconds(50);
                while (!presentReady()) {
                    if ((rs64_vr_session() == 0) || !(rs64VrLastViews.valid || rs64VrLastScreen)) {
                        cursorCondition.wait(cursorLock, presentReady);
                    }
                    else if (!cursorCondition.wait_for(cursorLock, keepAliveWait, presentReady)) {
                        cursorLock.unlock();
                        rs64VrKeepAlive(ext.device, ext.presentGraphicsWorker, ext.shaderLibrary);
                        cursorLock.lock();
                        keepAliveWait = std::chrono::milliseconds(0);
                    }
                }

                if (presentThreadRunning) {
                    processCursor = threadCursor;
                    threadCursor = (threadCursor + 1) % presents.size();
                    skipPresent = (writeCursor != threadCursor);
                }
            }

            if (processCursor >= 0) {
                std::unique_lock<std::mutex> threadLock(threadMutex);
                const bool suspended = surfaceSuspended;
                if (surfaceRestored) {
                    surfaceRestored = false;
                    swapChainValid = false;
                }
                const bool needsResize = !suspended && (ext.swapChain->needsResize() || !swapChainValid);
                if (needsResize) {
                    if (rs64LogSwapChain()) {
                        std::fprintf(stderr, "[swapchain] resize: needsResize=%d swapChainValid=%d\n", ext.swapChain->needsResize() ? 1 : 0, swapChainValid ? 1 : 0);
                    }

                    ext.presentGraphicsWorker->commandList->begin();
                    ext.presentGraphicsWorker->commandList->end();
                    ext.presentGraphicsWorker->execute();
                    ext.presentGraphicsWorker->wait();

                    // D3D11 render target views keep the back buffers alive, so they must go before the resize.
                    swapChainFramebuffers.clear();
                    swapChainValid = ext.swapChain->resize();

                    if (swapChainValid) {
                        ext.sharedResources->setSwapChainSize(ext.swapChain->getWidth(), ext.swapChain->getHeight());
                        
                        // Texture count could've changed after resize, so new semaphores are needed.
                        while (drawSemaphores.size() < ext.swapChain->getTextureCount()) {
                            drawSemaphores.emplace_back(ext.device->createCommandSemaphore());
                        }
                    }
                }

                if (!suspended && (needsResize || ext.appWindow->detectWindowMoved())) {
                    ext.appWindow->detectRefreshRate();
                    ext.sharedResources->setSwapChainRate(std::min(ext.appWindow->getRefreshRate(), displayTimingRate));
                }

                if (displayTiming && !suspended && !ext.swapChain->isEmpty()) {
                    uint32_t newDisplayTimingRate = ext.swapChain->getRefreshRate();
                    if (newDisplayTimingRate == 0) {
                        newDisplayTimingRate = UINT32_MAX;
                    }

                    if (newDisplayTimingRate != displayTimingRate) {
                        ext.sharedResources->setSwapChainRate(std::min(ext.appWindow->getRefreshRate(), newDisplayTimingRate));
                        displayTimingRate = newDisplayTimingRate;
                    }
                }

                skipPresent = skipPresent || suspended || ext.swapChain->isEmpty();

                Present &present = presents[processCursor];
                ext.workloadQueue->waitForWorkloadId(present.workloadId);

                if (!presentThreadRunning) {
                    continue;
                }

                if (skipPresent) {
                    skipInterpolation();
                    notifyPresentId(present);
                }
                else {
                    threadPresent(present, swapChainValid);
                }

                if (!present.paused) {
                    if (!present.fbOperations.empty()) {
                        const std::scoped_lock lock(screenFbChangePoolMutex);
                        screenFbChangePool.release(present.fbOperations.front().writeChanges.id);
                        present.fbOperations.clear();
                    }

                    threadAdvanceBarrier();
                }

                processCursor = -1;
            }
        }

        // Transition the active swap chain render target out of the present state to avoid live references to the resource.
        uint32_t swapChainIndex = 0;
        if (!ext.swapChain->isEmpty() && ext.swapChain->acquireTexture(acquiredSemaphore.get(), &swapChainIndex)) {
            RenderTexture *swapChainTexture = ext.swapChain->getTexture(swapChainIndex);
            ext.presentGraphicsWorker->commandList->begin();
            ext.presentGraphicsWorker->commandList->barriers(RenderBarrierStage::NONE, RenderTextureBarrier(swapChainTexture, RenderTextureLayout::COLOR_WRITE));
            ext.presentGraphicsWorker->commandList->end();

            const RenderCommandList *commandList = ext.presentGraphicsWorker->commandList.get();
            RenderCommandSemaphore *waitSemaphore = acquiredSemaphore.get();
            ext.presentGraphicsWorker->commandQueue->executeCommandLists(&commandList, 1, &waitSemaphore, 1, nullptr, 0, ext.presentGraphicsWorker->commandFence.get());
            ext.presentGraphicsWorker->wait();
        }
    }
};
