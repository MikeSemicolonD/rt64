//
// RT64
//

#include "rt64_inspector.h"

#include <algorithm>
#include <iomanip>

#include "hle/rt64_rs64_lights.h"
#include "im3d/im3d.h"
#include "im3d/im3d_math.h"
#include "imgui/imgui.h"
#include "imgui/imgui_internal.h"
#include "implot/implot.h"

// Volk must be included before the ImGui Vulkan backend.
#include "plume_vulkan.h"

#include "imgui/imgui_impl_sdl2_custom.h"
#include "imgui/backends/imgui_impl_vulkan.h"

#if defined(_WIN32)
#   include "imgui/backends/imgui_impl_dx12.h"
#   include "imgui/backends/imgui_impl_win32.h"
#   include "utf8conv/utf8conv.h"
#endif

#if defined(_WIN32)
#   include "plume_d3d12.h"
#endif

#ifdef PLUME_WEBGPU_ENABLED
#   include "imgui/backends/imgui_impl_wgpu.h"
#   include "plume_webgpu.h"
#endif

#ifdef PLUME_D3D11_ENABLED
#   include "imgui/backends/imgui_impl_dx11.h"
#   include "plume_d3d11.h"
#endif

static std::string IniFilenameUTF8;

#ifdef _WIN32
extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
#endif

static void checkVulkanResult(VkResult res) {
    if (res == 0) {
        return;
    }

    fprintf(stderr, "ImGui Vulkan Backend failed with error code 0x%X.\n", res);
}

namespace RT64 {
    struct VulkanContext {
        VkDevice device = VK_NULL_HANDLE;
        VkRenderPass renderPass = VK_NULL_HANDLE;
        VkDescriptorPool descriptorPool = VK_NULL_HANDLE;

        ~VulkanContext() {
            if (renderPass != VK_NULL_HANDLE) {
                vkDestroyRenderPass(device, renderPass, nullptr);
            }

            if (descriptorPool != VK_NULL_HANDLE) {
                vkDestroyDescriptorPool(device, descriptorPool, nullptr);
            }
        }
    };

    // Inspector

    Inspector::Inspector(RenderDevice *device, const RenderSwapChain *swapChain, UserConfiguration::GraphicsAPI graphicsAPI, SDL_Window *sdlWindow) {
        assert(device != nullptr);
        assert(swapChain != nullptr);

        this->device = device;
        this->swapChain = swapChain;
        this->graphicsAPI = graphicsAPI;
        this->sdlWindow = sdlWindow;

        IMGUI_CHECKVERSION();

        ImGui::CreateContext();
        ImPlot::CreateContext();
        ImGui::StyleColorsDark();

        if (sdlWindow != nullptr) {
            ImGui_ImplSDL2_InitForOther(sdlWindow);
        }
        else {
#       ifdef _WIN32
            RenderWindow renderWindow = swapChain->getWindow();
            ImGui_ImplWin32_Init(renderWindow);
#       endif
        }

        switch (graphicsAPI) {
        case UserConfiguration::GraphicsAPI::D3D12: {
#       ifdef _WIN32
            D3D12Device *interfaceDevice = static_cast<D3D12Device *>(device);
            RenderDescriptorRange descriptorRange(RenderDescriptorRangeType::TEXTURE, 0, 1);
            descriptorSet = interfaceDevice->createDescriptorSet(RenderDescriptorSetDesc(&descriptorRange, 1));

            D3D12DescriptorSet *interfaceDescriptorSet = static_cast<D3D12DescriptorSet *>(descriptorSet.get());
            const D3D12SwapChain *interfaceSwapChain = static_cast<const D3D12SwapChain *>(swapChain);
            const D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle = interfaceDevice->viewHeapAllocator->getCPUHandleAt(interfaceDescriptorSet->viewAllocation.offset);
            const D3D12_GPU_DESCRIPTOR_HANDLE gpuHandle = interfaceDevice->viewHeapAllocator->getGPUHandleAt(interfaceDescriptorSet->viewAllocation.offset);
            ImGui_ImplDX12_Init(interfaceDevice->d3d, 2, interfaceSwapChain->nativeFormat, interfaceDevice->viewHeapAllocator->heap, cpuHandle, gpuHandle);
#       else
            assert(false && "Unsupported Graphics API.");
            return;
#       endif
            break;
        }
        case UserConfiguration::GraphicsAPI::Vulkan: {
#       ifdef PLUME_WEBGPU_ENABLED
            // The browser's "Vulkan" is WebGPU; device objects are made on the main thread, which owns the device.
            const WebGPUDevice *webDevice = static_cast<const WebGPUDevice *>(device);
            const WebGPUSwapChain *webSwapChain = static_cast<const WebGPUSwapChain *>(swapChain);
            webgpuRunOnMain([&]() {
                ImGui_ImplWGPU_InitInfo initInfo;
                initInfo.Device = webDevice->device;
                initInfo.NumFramesInFlight = 3;
                initInfo.RenderTargetFormat = webSwapChain->format;
                ImGui_ImplWGPU_Init(&initInfo);
                ImGui_ImplWGPU_CreateDeviceObjects();
            });

            break;
#       endif
            VulkanDevice *interfaceDevice = static_cast<VulkanDevice *>(device);
            const VulkanSwapChain *interfaceSwapChain = static_cast<const VulkanSwapChain *>(swapChain);
            ImGui_ImplVulkan_LoadFunctions([](const char *functionName, void *vulkanInstance) {
                return vkGetInstanceProcAddr(*(reinterpret_cast<VkInstance *>(vulkanInstance)), functionName);
            }, &interfaceDevice->renderInterface->instance);

            std::unordered_map<VkDescriptorType, uint32_t> typeCounts;
            typeCounts[VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER] = 1;
            
            vulkanContext = std::make_unique<VulkanContext>();
            vulkanContext->device = interfaceDevice->vk;
            vulkanContext->renderPass = VulkanGraphicsPipeline::createRenderPass(interfaceDevice, &interfaceSwapChain->pickedSurfaceFormat.format, 1, VK_FORMAT_UNDEFINED, VK_SAMPLE_COUNT_1_BIT);
            vulkanContext->descriptorPool = VulkanDescriptorSet::createDescriptorPool(interfaceDevice, typeCounts, false);

            ImGui_ImplVulkan_InitInfo initInfo = {};
            initInfo.Instance = interfaceDevice->renderInterface->instance;
            initInfo.PhysicalDevice = interfaceDevice->physicalDevice;
            initInfo.Device = vulkanContext->device;
            initInfo.QueueFamily = interfaceSwapChain->commandQueue->familyIndex;
            initInfo.Queue = interfaceSwapChain->commandQueue->queue->vk;
            initInfo.DescriptorPool = vulkanContext->descriptorPool;
            initInfo.RenderPass = vulkanContext->renderPass;
            initInfo.Subpass = 0;
            initInfo.MinImageCount = 2;
            initInfo.ImageCount = 2;
            initInfo.CheckVkResultFn = &checkVulkanResult;

            ImGui_ImplVulkan_Init(&initInfo);
            break;
        }
#   ifdef PLUME_D3D11_ENABLED
        case UserConfiguration::GraphicsAPI::D3D11: {
            D3D11Device *interfaceDevice = static_cast<D3D11Device *>(device);
            ImGui_ImplDX11_Init(interfaceDevice->d3d.Get(), interfaceDevice->immediateContext.Get());
            break;
        }
#   endif
        default:
            assert(false && "Unknown Graphics API.");
            return;
        }
    }

    Inspector::~Inspector() {
        switch (graphicsAPI) {
        case UserConfiguration::GraphicsAPI::D3D12: {
#       ifdef _WIN32
            ImGui_ImplDX12_Shutdown();
#       else
            assert(false && "Unsupported Graphics API.");
#       endif
            break;
        }
        case UserConfiguration::GraphicsAPI::Vulkan: {
#       ifdef PLUME_WEBGPU_ENABLED
            webgpuRunOnMain([]() { ImGui_ImplWGPU_Shutdown(); });
            break;
#       endif
            ImGui_ImplVulkan_Shutdown();
            vulkanContext.reset(nullptr);
            break;
        }
#   ifdef PLUME_D3D11_ENABLED
        case UserConfiguration::GraphicsAPI::D3D11: {
            std::scoped_lock lock(static_cast<D3D11Device *>(device)->contextMutex);
            ImGui_ImplDX11_Shutdown();
            break;
        }
#   endif
        default:
            assert(false && "Unknown Graphics API.");
            break;
        }

        if (sdlWindow != nullptr) {
            ImGui_ImplSDL2_Shutdown();
        }
        else {
#       ifdef _WIN32
            ImGui_ImplWin32_Shutdown();
#       endif
        }

        ImPlot::DestroyContext();
        ImGui::DestroyContext();
    }

    void Inspector::setIniPath(const std::filesystem::path &path) {
        ImGuiIO &io = ImGui::GetIO();
        IniFilenameUTF8 = path.u8string();
        io.IniFilename = IniFilenameUTF8.c_str();
    }
    
    void Inspector::newFrame(RenderWorker *worker) {
        assert(worker != nullptr);

        frameMutex.lock();
        
        if (sdlWindow != nullptr) {
            ImGui_ImplSDL2_NewFrame();
        }
        else {
#       ifdef _WIN32
            ImGui_ImplWin32_NewFrame();
#       endif
        }

        switch (graphicsAPI) {
        case UserConfiguration::GraphicsAPI::D3D12: {
#       ifdef _WIN32
            ImGui_ImplDX12_NewFrame();
#       else
            assert(false && "Unsupported Graphics API.");
#       endif
            break;
        }
        case UserConfiguration::GraphicsAPI::Vulkan: {
#       ifdef PLUME_WEBGPU_ENABLED
            // ImGui_ImplWGPU_NewFrame only creates device objects, done at init on the main thread.
            break;
#       endif
            ImGui_ImplVulkan_NewFrame();
            break;
        }
#   ifdef PLUME_D3D11_ENABLED
        case UserConfiguration::GraphicsAPI::D3D11: {
            ImGui_ImplDX11_NewFrame();
            break;
        }
#   endif
        default:
            assert(false && "Unknown Graphics API.");
            break;
        }

        ImGui::NewFrame();
        Im3d::NewFrame();
    }

    void Inspector::endFrame() {
        // Positions saved in the ini can fall outside a smaller window, so top-level windows keep their title bar on screen.
        const ImVec2 displaySize = ImGui::GetIO().DisplaySize;
        for (ImGuiWindow *window : ImGui::GetCurrentContext()->Windows) {
            if ((window->Flags & ImGuiWindowFlags_ChildWindow) || !window->Active || (displaySize.x <= 0.0f) || (displaySize.y <= 0.0f)) {
                continue;
            }

            const float keepVisible = 48.0f;
            ImVec2 clamped = window->Pos;
            clamped.x = std::clamp(clamped.x, keepVisible - window->Size.x, std::max(displaySize.x - keepVisible, 0.0f));
            clamped.y = std::clamp(clamped.y, 0.0f, std::max(displaySize.y - ImGui::GetFrameHeight(), 0.0f));
            if ((clamped.x != window->Pos.x) || (clamped.y != window->Pos.y)) {
                ImGui::SetWindowPos(window, clamped);
            }
        }

        // RS64: RT64's Im3d renderer is unported, so its draw lists (light spheres, gizmos) go through ImGui's background list, projected with the light pass's view.
        Im3d::EndFrame();
        const rs64lights::PresetView presetView = rs64lights::inspectorView();
        if (presetView.valid) {
            ImDrawList *drawList = ImGui::GetBackgroundDrawList();
            auto project = [&](const Im3d::VertexData &v, ImVec2 &out) {
                const float in[4] = { v.m_positionSize.x, v.m_positionSize.y, v.m_positionSize.z, 1.0f };
                float clip[4];
                rs64lights::transform(in, presetView.toNdc, clip);
                if (!(clip[3] > 1e-4f)) {
                    return false;
                }

                out = ImVec2((clip[0] / clip[3] + 1.0f) * 0.5f * displaySize.x, (1.0f - clip[1] / clip[3]) * 0.5f * displaySize.y);
                return true;
            };
            auto color = [](const Im3d::VertexData &v) {
                const uint32_t c = v.m_color;
                return IM_COL32(c >> 24, (c >> 16) & 0xFF, (c >> 8) & 0xFF, c & 0xFF);
            };

            for (Im3d::U32 l = 0; l < Im3d::GetDrawListCount(); l++) {
                const Im3d::DrawList &list = Im3d::GetDrawLists()[l];
                const Im3d::VertexData *v = list.m_vertexData;
                ImVec2 p[3];
                if (list.m_primType == Im3d::DrawPrimitive_Triangles) {
                    for (Im3d::U32 i = 0; i + 2 < list.m_vertexCount; i += 3) {
                        if (project(v[i], p[0]) && project(v[i + 1], p[1]) && project(v[i + 2], p[2])) {
                            drawList->AddTriangleFilled(p[0], p[1], p[2], color(v[i]));
                        }
                    }
                }
                else if (list.m_primType == Im3d::DrawPrimitive_Lines) {
                    for (Im3d::U32 i = 0; i + 1 < list.m_vertexCount; i += 2) {
                        if (project(v[i], p[0]) && project(v[i + 1], p[1])) {
                            drawList->AddLine(p[0], p[1], color(v[i]), std::max(v[i].m_positionSize.w, 1.0f));
                        }
                    }
                }
                else {
                    for (Im3d::U32 i = 0; i < list.m_vertexCount; i++) {
                        if (project(v[i], p[0])) {
                            drawList->AddCircleFilled(p[0], std::max(v[i].m_positionSize.w * 0.5f, 1.0f), color(v[i]));
                        }
                    }
                }
            }
        }

        ImGui::Render();
        frameMutex.unlock();
    }

    void Inspector::draw(RenderCommandList *commandList) {
        const std::lock_guard<std::mutex> frameLock(frameMutex);
        ImDrawData *drawData = ImGui::GetDrawData();
        if (drawData != nullptr) {
            switch (graphicsAPI) {
            case UserConfiguration::GraphicsAPI::D3D12: {
#       ifdef _WIN32
                D3D12CommandList *interfaceCommandList = static_cast<D3D12CommandList *>(commandList);
                interfaceCommandList->checkDescriptorHeaps();
                ImGui_ImplDX12_RenderDrawData(drawData, interfaceCommandList->d3d);
#       else
                assert(false && "Unsupported Graphics API.");
#       endif
                break;
            }
            case UserConfiguration::GraphicsAPI::Vulkan: {
#           ifdef PLUME_WEBGPU_ENABLED
                // Commands replay on the main thread at submission, after ImGui may have started the next frame, so the draw lists are copied.
                std::shared_ptr<ImDrawData> webCopy(new ImDrawData(*drawData), [](ImDrawData *data) {
                    for (ImDrawList *drawList : data->CmdLists) {
                        IM_DELETE(drawList);
                    }

                    delete data;
                });

                for (ImDrawList *&drawList : webCopy->CmdLists) {
                    drawList = drawList->CloneOutput();
                }

                webgpuRecordRenderPassJob(commandList, [webCopy](WGPURenderPassEncoder pass) {
                    ImGui_ImplWGPU_RenderDrawData(webCopy.get(), pass);
                });

                break;
#           endif
                VulkanCommandList *interfaceCommandList = static_cast<VulkanCommandList *>(commandList);
                ImGui_ImplVulkan_RenderDrawData(drawData, interfaceCommandList->vk);
                break;
            }
#       ifdef PLUME_D3D11_ENABLED
            case UserConfiguration::GraphicsAPI::D3D11: {
                // D3D11 commands replay at submission, after ImGui may have started the next frame, so the draw lists are copied.
                std::shared_ptr<ImDrawData> drawCopy(new ImDrawData(*drawData), [](ImDrawData *data) {
                    for (ImDrawList *drawList : data->CmdLists) {
                        IM_DELETE(drawList);
                    }

                    delete data;
                });

                for (ImDrawList *&drawList : drawCopy->CmdLists) {
                    drawList = drawList->CloneOutput();
                }

                static_cast<D3D11CommandList *>(commandList)->commands.emplace_back([drawCopy](D3D11Context &c) {
                    c.applyFramebuffer();
                    ImGui_ImplDX11_RenderDrawData(drawCopy.get());
                    c.graphicsPipelineDirty = true;
                    c.graphics.bindingsDirty = true;
                    c.framebufferDirty = true;
                });

                break;
            }
#       endif
            default:
                assert(false && "Unknown Graphics API.");
                break;
            }
        }
    }

#ifdef _WIN32
    bool Inspector::handleMessage(UINT msg, WPARAM wParam, LPARAM lParam) {
        return ImGui_ImplWin32_WndProcHandler(swapChain->getWindow(), msg, wParam, lParam);
    }
#endif

    bool Inspector::handleSdlEvent(SDL_Event *event) {
        assert((sdlWindow != nullptr) && "SDL Events shouldn't be handled if SDL was not used.");

        bool processed = ImGui_ImplSDL2_ProcessEvent(event);
        if (processed) {
            if ((event->type == SDL_KEYDOWN) || (event->type == SDL_KEYUP)) {
                return ImGui::GetIO().WantCaptureKeyboard;
            }
            else if ((event->type == SDL_MOUSEMOTION) || (event->type == SDL_MOUSEBUTTONDOWN) || (event->type == SDL_MOUSEBUTTONUP) || (event->type == SDL_MOUSEWHEEL)) {
                return ImGui::GetIO().WantCaptureMouse;
            }
        }
        
        return false;
    }
};
