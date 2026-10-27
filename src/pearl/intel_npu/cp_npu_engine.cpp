#include "cp_npu_engine.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <system_error>

#include <openvino/runtime/properties.hpp>
#include <openvino/runtime/remote_context.hpp>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#include <unistd.h>
#endif

#ifndef CP_INTEL_NPU_SOURCE_KERNEL_DIR
#define CP_INTEL_NPU_SOURCE_KERNEL_DIR ""
#endif

namespace fs = std::filesystem;

namespace cp_npu {
namespace {

void set_env(const char* name, const std::string& value) {
#ifdef _WIN32
    _putenv_s(name, value.c_str());
#else
    setenv(name, value.c_str(), 1);
#endif
}

fs::path exe_dir() {
#ifdef _WIN32
    wchar_t buf[MAX_PATH];
    const DWORD n = GetModuleFileNameW(nullptr, buf, MAX_PATH);
    return n ? fs::path(std::wstring(buf, n)).parent_path() : fs::current_path();
#else
    std::error_code ec;
    const fs::path p = fs::read_symlink("/proc/self/exe", ec);
    return ec ? fs::current_path() : p.parent_path();
#endif
}

/* Directory of the loaded OpenVINO runtime, where the plugin and the patched compiler live. */
fs::path openvino_dir() {
#ifdef _WIN32
    HMODULE mod = GetModuleHandleW(L"openvino.dll");
    if (!mod)
        return {};
    wchar_t buf[MAX_PATH];
    const DWORD n = GetModuleFileNameW(mod, buf, MAX_PATH);
    return n ? fs::path(std::wstring(buf, n)).parent_path() : fs::path();
#else
    Dl_info info{};
    if (dladdr(reinterpret_cast<void*>(&set_compiler_environment), &info) && info.dli_fname)
        return fs::path(info.dli_fname).parent_path();
    return {};
#endif
}

struct Fnv64 {
    uint64_t h = 1469598103934665603ull;
    void update(const void* data, size_t size) {
        const auto* p = static_cast<const uint8_t*>(data);
        for (size_t i = 0; i < size; ++i)
            h = (h ^ p[i]) * 1099511628211ull;
    }
    void update(const std::string& s) { update(s.data(), s.size()); }
};

/* OpenVINO's cache key covers the model and the config but not the SHAVE kernel binaries, the
 * compiler library or the int32 switch, so they select the subdirectory. */
std::string cache_subdir(const std::string& root, const std::string& kernel_dir) {
    Fnv64 h;
    std::vector<fs::path> elfs;
    std::error_code ec;
    for (const auto& e : fs::directory_iterator(kernel_dir, ec))
        if (e.path().extension() == ".elf")
            elfs.push_back(e.path());
    std::sort(elfs.begin(), elfs.end());
    for (const auto& p : elfs) {
        std::ifstream f(p, std::ios::binary);
        const std::string bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        h.update(p.filename().string());
        h.update(bytes);
    }
    const fs::path ov = openvino_dir();
#ifdef _WIN32
    const fs::path compiler = ov / "openvino_intel_npu_compiler.dll";
#else
    const fs::path compiler = ov / "libopenvino_intel_npu_compiler.so";
#endif
    if (!ov.empty() && fs::exists(compiler, ec))
        h.update(std::to_string(fs::file_size(compiler, ec)) + ":" +
                 std::to_string(fs::last_write_time(compiler, ec).time_since_epoch().count()));
    const char* i32 = std::getenv("NPU_EXPERIMENTAL_I32_DPU_OUTPUT");
    h.update(i32 ? i32 : "");
    char hex[17];
    std::snprintf(hex, sizeof(hex), "%016llx", static_cast<unsigned long long>(h.h));
    const fs::path dir = fs::path(root) / std::string(hex, 12);
    fs::create_directories(dir, ec);
    return dir.string();
}

} // namespace

void set_compiler_environment(const std::string& kernel_dir) {
    set_env("NPU_EXPERIMENTAL_I32_DPU_OUTPUT", "1");
    if (!std::getenv("NPU_EXPERIMENTAL_SHAVE_KERNEL_DIR"))
        set_env("NPU_EXPERIMENTAL_SHAVE_KERNEL_DIR", kernel_dir);
}

std::string default_kernel_dir() {
    std::error_code ec;
    fs::path local = exe_dir() / "kernels" / "intel_npu";
    if (fs::exists(local / "tile_xor_acc.3720xx.elf", ec))
        return local.make_preferred().string();
    return fs::path(CP_INTEL_NPU_SOURCE_KERNEL_DIR).make_preferred().string();
}

std::string default_cache_root() {
    return (exe_dir() / "npu_cache").make_preferred().string();
}

int Engine::init(const LaunchShape& shape, const std::string& kernel_dir, const std::string& cache_root,
                 int requests, std::string* err) {
    try {
        shutdown();
        m_shape = shape;
        set_compiler_environment(kernel_dir);
        std::error_code ec;
        if (!fs::exists(fs::path(kernel_dir) / "tile_xor_acc.3720xx.elf", ec)) {
            *err = "TileXorAcc kernel not found in " + kernel_dir;
            return -1;
        }
        /* The cache key needs the loaded runtime's directory: query a property first. */
        (void)m_core.get_available_devices();
        if (!cache_root.empty()) {
            m_cache_dir = cache_subdir(cache_root, kernel_dir);
            m_core.set_property(ov::cache_dir(m_cache_dir));
        }
        auto model = build_launch_model(shape, &m_word_index);
        const ov::AnyMap config{{"NPU_COMPILER_TYPE", "PLUGIN"}};
        const auto t0 = std::chrono::steady_clock::now();
        m_compiled = m_core.compile_model(model, "NPU", config);
        m_compile_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        for (int i = 0; i < requests; ++i)
            m_requests.push_back(m_compiled.create_infer_request());
        m_ready = true;
        return 0;
    } catch (const std::exception& e) {
        *err = e.what();
        shutdown();
        return -1;
    }
}

void Engine::shutdown() {
    m_requests.clear();
    m_compiled = ov::CompiledModel();
    m_ready = false;
}

ov::Tensor Engine::host_tensor(int index) {
    const auto& port = m_compiled.input(size_t(index));
    if (m_zero_copy) {
        try {
            return m_compiled.get_context().create_host_tensor(port.get_element_type(), port.get_shape());
        } catch (const std::exception&) {
            m_zero_copy = false;
        }
    }
    return ov::Tensor(port.get_element_type(), port.get_shape());
}

std::string Engine::device_name() {
    try {
        return m_core.get_property("NPU", ov::device::full_name);
    } catch (const std::exception&) {
        return "NPU";
    }
}

} // namespace cp_npu
