//
// RT64
//

#include "rt64_camera_controller.h"

#include <cmath>

#include "common/rt64_math.h"
#include "imgui/imgui.h"

namespace RT64 {
    CameraController::CameraController() {
        lastCursorPos = { 0, 0 };
    }

    void CameraController::moveCursor(DebuggerCamera &camera, const hlslpp::int2 cursorPos, const hlslpp::int2 windowSize) {
        const bool middleMouseButton = ImGui::IsMouseDown(ImGuiMouseButton_Middle);
        /*
        // Automatically enable camera control if the user is pressing the middle mouse button over the viewport.
        const uint32_t freeCamIndex = freeCam.sceneIndex;
        const bool validIndex = (freeCamIndex < gameFrame.perspectiveScenes.size());
        if (!freeCam.enabled && validIndex && !ImGui::GetIO().WantCaptureMouse && middleMouseButton) {
            gameFrame.enableFreeCamera(freeCamIndex);
        }
        */
        float cameraSpeed = (camera.farPlane - camera.nearPlane) / 5.0f;

        // RS64: planes read from a non-standard projection can collapse (near ~= far), which froze pan and dolly; fall back to the game's scale.
        const float MinCameraSpeed = 5000.0f;
        if (!std::isfinite(cameraSpeed) || (cameraSpeed < MinCameraSpeed)) {
            cameraSpeed = MinCameraSpeed;
        }

        const ImGuiIO &io = ImGui::GetIO();
        if (camera.enabled && !io.WantCaptureMouse) {
            bool leftAlt = ImGui::IsKeyDown(ImGuiKey_LeftAlt);
            bool leftCtrl = ImGui::IsKeyDown(ImGuiKey_LeftCtrl);
            const hlslpp::float2 delta = hlslpp::float2(cursorPos - lastCursorPos) / hlslpp::float2(windowSize);
            if (middleMouseButton) {
                if (leftCtrl) {
                    movePerspective(camera, { 0.0f, 0.0f, (delta.x + delta.y) * cameraSpeed });
                }
                else if (leftAlt) {
                    const float CameraRotationSpeed = 5.0f;
                    rotatePerspective(camera, delta.x * CameraRotationSpeed, -delta.y * CameraRotationSpeed);
                }
                else {
                    movePerspective(camera, { delta.x * cameraSpeed, delta.y * cameraSpeed, 0.0f });
                }
            }

            // RS64: the wheel dollies like Ctrl + middle drag, a tenth of the drag speed per notch.
            if (io.MouseWheel != 0.0f) {
                movePerspective(camera, { 0.0f, 0.0f, -io.MouseWheel * cameraSpeed * 0.1f });
            }
        }

        // RS64: WASD flies (Q/E down/up, Shift x4) at cameraSpeed per second, unless an ImGui field has the keyboard.
        if (camera.enabled && !io.WantCaptureKeyboard) {
            const float step = cameraSpeed * io.DeltaTime * (ImGui::IsKeyDown(ImGuiKey_LeftShift) ? 4.0f : 1.0f);
            const float side = float(ImGui::IsKeyDown(ImGuiKey_A)) - float(ImGui::IsKeyDown(ImGuiKey_D));
            const float up = float(ImGui::IsKeyDown(ImGuiKey_E)) - float(ImGui::IsKeyDown(ImGuiKey_Q));
            const float forward = float(ImGui::IsKeyDown(ImGuiKey_S)) - float(ImGui::IsKeyDown(ImGuiKey_W));
            if ((side != 0.0f) || (up != 0.0f) || (forward != 0.0f)) {
                movePerspective(camera, { side * step, up * step, forward * step });
            }
        }

        lastCursorPos = cursorPos;
    }

    void CameraController::movePerspective(DebuggerCamera &camera, hlslpp::float3 translation) {
        hlslpp::float4x4 translationMatrix = matrixTranslation(translation);
        camera.viewMatrix = mul(camera.viewMatrix, translationMatrix);
        camera.invViewMatrix = hlslpp::inverse(camera.viewMatrix);
    }

    void CameraController::rotatePerspective(DebuggerCamera &camera, float yaw, float pitch) {
        hlslpp::float3x3 yaw3x3 = matrixRotationY(yaw);
        hlslpp::float3x3 pitch3x3 = matrixRotationZ(pitch);
        hlslpp::float3x3 inv3x3 = extract3x3(camera.invViewMatrix);
        inv3x3 = hlslpp::mul(inv3x3, yaw3x3);
        inv3x3 = hlslpp::mul(pitch3x3, inv3x3);
        camera.invViewMatrix[0].xyz = inv3x3[0];
        camera.invViewMatrix[1].xyz = inv3x3[1];
        camera.invViewMatrix[2].xyz = inv3x3[2];
        camera.viewMatrix = hlslpp::inverse(camera.invViewMatrix);
    }

    void CameraController::lookAtPerspective(DebuggerCamera &camera, hlslpp::float3 position, hlslpp::float3 focus) {
        const hlslpp::float3 UpVector = hlslpp::float3(0.0f, 1.0f, 0.0f);
        const hlslpp::float3 viewForwardDir = hlslpp::normalize(position - focus);
        const hlslpp::float3 viewSideDir = hlslpp::normalize(hlslpp::cross(UpVector, viewForwardDir));
        const hlslpp::float3 viewUpDir = hlslpp::normalize(hlslpp::cross(viewForwardDir, viewSideDir));
        camera.invViewMatrix[0] = hlslpp::float4(viewSideDir, 0.0f);
        camera.invViewMatrix[1] = hlslpp::float4(viewUpDir, 0.0f);
        camera.invViewMatrix[2] = hlslpp::float4(viewForwardDir, 0.0f);
        camera.invViewMatrix[3] = hlslpp::float4(position, 1.0f);
        camera.viewMatrix = hlslpp::inverse(camera.invViewMatrix);
    }
};