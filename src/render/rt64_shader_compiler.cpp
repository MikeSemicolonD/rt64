
//;
// RT64
//

#if defined(_WIN32)

#include "rt64_shader_compiler.h"

#include <mutex>

#include "common/rt64_common.h"

namespace RT64 {
    // ShaderCompiler

    ShaderCompiler::ShaderCompiler() {
        HRESULT res = DxcCreateInstance(CLSID_DxcCompiler, __uuidof(IDxcCompiler), (void **)(&dxcCompiler));
        if (FAILED(res)) {
            fprintf(stderr, "DxcCreateInstance(DxcCompiler) failed with error code 0x%lX.\n", res);
            return;
        }

        res = DxcCreateInstance(CLSID_DxcUtils, __uuidof(IDxcUtils), (void **)(&dxcUtils));
        if (FAILED(res)) {
            fprintf(stderr, "DxcCreateInstance(DxcUtils) failed with error code 0x%lX.\n", res);
            return;
        }
    }

    ShaderCompiler::~ShaderCompiler() {
        if (dxcCompiler != nullptr) {
            dxcCompiler->Release();
        }

        if (dxcUtils != nullptr) {
            dxcUtils->Release();
        }
    }

    // dxcompiler 1.7.2308 now and then faults inside Link (and once at build time inside Compile) on input that compiles fine on a retry. These guards turn
    // an access violation into a failed call, so the variant is dropped instead of the game; they hold no C++ objects, which an SEH unwind would skip.
    static HRESULT guardedCompile(IDxcCompiler *compiler, IDxcBlob *text, LPCWSTR entry, LPCWSTR profile, LPCWSTR *arguments, UINT32 argumentCount, IDxcOperationResult **result) {
        __try {
            return compiler->Compile(text, L"", entry, profile, arguments, argumentCount, nullptr, 0, nullptr, result);
        }
        __except ((GetExceptionCode() == EXCEPTION_ACCESS_VIOLATION) ? EXCEPTION_EXECUTE_HANDLER : EXCEPTION_CONTINUE_SEARCH) {
            *result = nullptr;
            return E_FAIL;
        }
    }

    static HRESULT guardedLink(IDxcLinker *linker, LPCWSTR entry, LPCWSTR profile, const LPCWSTR *libraryNames, UINT32 libraryCount, IDxcOperationResult **result) {
        __try {
            return linker->Link(entry, profile, libraryNames, libraryCount, nullptr, 0, result);
        }
        __except ((GetExceptionCode() == EXCEPTION_ACCESS_VIOLATION) ? EXCEPTION_EXECUTE_HANDLER : EXCEPTION_CONTINUE_SEARCH) {
            *result = nullptr;
            return E_FAIL;
        }
    }

    static void checkResultForError(IDxcOperationResult *result) {
        HRESULT resultCode;
        result->GetStatus(&resultCode);
        if (FAILED(resultCode)) {
            IDxcBlobEncoding *error;
            HRESULT hr = result->GetErrorBuffer(&error);
            if (FAILED(hr)) {
                throw std::runtime_error("Failed to get shader compiler error");
            }

            // Convert error blob to a string.
            std::vector<char> infoLog(error->GetBufferSize() + 1);
            memcpy(infoLog.data(), error->GetBufferPointer(), error->GetBufferSize());
            infoLog[error->GetBufferSize()] = 0;

            RT64_LOG_PRINTF("Shader compilation error: %s\n", infoLog.data());
            fprintf(stderr, "[shader-error] %s\n", infoLog.data()); fflush(stderr);
            throw std::runtime_error("Shader compilation error: " + std::string(infoLog.data()));
        }
    }

    void ShaderCompiler::compile(const std::string &shaderCode, const std::wstring &entryName, const std::wstring &profile,
        RenderShaderFormat shaderFormat, IDxcBlob **shaderBlob) const
    {
        IDxcBlobEncoding *textBlob = nullptr;
        HRESULT res = dxcUtils->CreateBlobFromPinned((LPBYTE)shaderCode.c_str(), (uint32_t)shaderCode.size(), DXC_CP_ACP, &textBlob);
        if (FAILED(res)) {
            fprintf(stderr, "CreateBlobFromPinned failed with error code 0x%lX.\n", res);
            return;
        }

        std::vector<LPCWSTR> arguments;
        arguments.push_back(L"-Qstrip_debug");

        switch (shaderFormat) {
        case RenderShaderFormat::DXIL:
            arguments.push_back(L"-Qstrip_reflect");
            break;
        case RenderShaderFormat::SPIRV:
            arguments.push_back(L"-spirv");
            arguments.push_back(L"-fvk-use-dx-layout");

            if (profile.find(L"vs") != std::wstring::npos) {
                arguments.push_back(L"-fvk-invert-y");
            }
            else if (profile.find(L"lib") != std::wstring::npos) {
                arguments.push_back(L"-fspv-target-env=vulkan1.1spirv1.4");
                arguments.push_back(L"-fspv-extension=SPV_KHR_ray_tracing");
            }

            break;
        default:
            assert(false && "Unknown shader format.");
            return;
        }

        IDxcOperationResult *result = nullptr;
        guardedCompile(dxcCompiler, textBlob, entryName.c_str(), profile.c_str(), arguments.data(), (UINT32)(arguments.size()), &result);
        if (result == nullptr) {
            textBlob->Release();
            fprintf(stderr, "[shader-error] dxcompiler faulted in Compile; the variant is dropped\n");
            throw std::runtime_error("dxcompiler faulted in Compile");
        }

        checkResultForError(result);
        result->GetResult(shaderBlob);
        textBlob->Release();
    }

    void ShaderCompiler::link(const std::wstring &entryName, const std::wstring &profile, IDxcBlob **libraryBlobs,
        const wchar_t **libraryBlobNames, uint32_t libraryBlobCount, IDxcBlob **shaderBlob) const 
    {
        // dxcompiler 1.7.2308 crashed with several compile threads in Link at once.
        static std::mutex linkMutex;
        std::lock_guard<std::mutex> lock(linkMutex);

        IDxcLinker *dxcLinker = nullptr;
        HRESULT res = DxcCreateInstance(CLSID_DxcLinker, __uuidof(IDxcLinker), (void **)(&dxcLinker));
        if (FAILED(res)) {
            fprintf(stderr, "DxcCreateInstance(DxcLinker) failed with error code 0x%lX.\n", res);
            return;
        }

        for (uint32_t i = 0; i < libraryBlobCount; i++) {
            res = dxcLinker->RegisterLibrary(libraryBlobNames[i], libraryBlobs[i]);
            if (FAILED(res)) {
                fprintf(stderr, "RegisterLibrary failed with error code 0x%lX.\n", res);
                return;
            }
        }

        IDxcOperationResult *result = nullptr;
        guardedLink(dxcLinker, entryName.c_str(), profile.c_str(), libraryBlobNames, libraryBlobCount, &result);
        if (result == nullptr) {
            dxcLinker->Release();
            fprintf(stderr, "[shader-error] dxcompiler faulted in Link; the variant is dropped\n");
            throw std::runtime_error("dxcompiler faulted in Link");
        }

        checkResultForError(result);
        result->GetResult(shaderBlob);
        dxcLinker->Release();
    }
};

#endif