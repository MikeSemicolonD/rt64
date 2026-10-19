//
// RT64
//

#pragma once

#include "common/rt64_plume.h"

namespace RT64 {
    using RenderHookInit = void(RenderInterface *rhi, RenderDevice *device);
    using RenderHookDraw = void(RenderCommandList *list, RenderFramebuffer *swapChainFramebuffer);
    using RenderHookDeinit = void();

    RenderHookInit *GetRenderHookInit();
    RenderHookDraw *GetRenderHookDraw();
    RenderHookDeinit *GetRenderHookDeinit();

    void SetRenderHooks(RenderHookInit *init, RenderHookDraw *draw, RenderHookDeinit *deinit);

    // Host ImGui hook: called each frame between ImGui::NewFrame and ImGui::Render
    // (inside the inspector frame), so the host can emit its own ImGui windows.
    using RenderHookImgui = void();
    RenderHookImgui *GetRenderHookImgui();
    void SetRenderHookImgui(RenderHookImgui *imgui);

    // Defined in rt64_raster_shader_cache.cpp: builds the D3D11 shader cache for a combination list, returns the failure count or -1.
    int PrecompileD3D11Shaders(const std::string &comboPath, const std::string &cacheDirectory);

    // Defined in rt64_raster_shader_cache.cpp: raster shaders queued or compiling, compiled so far, and whether draws are skipped until
    // their shader is ready (no ubershader, i.e. D3D11/D3D10) rather than drawn with the ubershader.
    void GetShaderCompileStatus(uint32_t &pending, uint32_t &compiled, bool &drawsWait);
};
