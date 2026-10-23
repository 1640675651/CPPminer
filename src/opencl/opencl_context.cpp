#include "opencl_context.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <limits.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {

void log_cl_error(const char *what, cl_int err) {
    std::fprintf(stderr, "OpenCL %s failed (%d): %s\n", what, err,
                 OpenClContext::error_string(err).c_str());
}

#ifndef CL_DEVICE_INTEGER_DOT_PRODUCT_CAPABILITIES_KHR
#define CL_DEVICE_INTEGER_DOT_PRODUCT_CAPABILITIES_KHR 0x1073
#endif
#ifndef CL_DEVICE_INTEGER_DOT_PRODUCT_INPUT_4x8BIT_KHR
#define CL_DEVICE_INTEGER_DOT_PRODUCT_INPUT_4x8BIT_KHR (1 << 1)
#endif
#ifndef CL_DEVICE_INTEGER_DOT_PRODUCT_INPUT_4x8BIT_PACKED_KHR
#define CL_DEVICE_INTEGER_DOT_PRODUCT_INPUT_4x8BIT_PACKED_KHR (1 << 0)
#endif
#ifndef CL_DEVICE_INTEGER_DOT_PRODUCT_ACCELERATION_PROPERTIES_4x8BIT_PACKED_KHR
#define CL_DEVICE_INTEGER_DOT_PRODUCT_ACCELERATION_PROPERTIES_4x8BIT_PACKED_KHR 0x1075
#endif

typedef struct _cl_device_integer_dot_product_acceleration_properties_khr {
    cl_bool signed_accelerated;
    cl_bool unsigned_accelerated;
    cl_bool mixed_signedness_accelerated;
    cl_bool accumulating_saturating_signed_accelerated;
    cl_bool accumulating_saturating_unsigned_accelerated;
    cl_bool accumulating_saturating_mixed_signedness_accelerated;
} cl_device_integer_dot_product_acceleration_properties_khr;

bool extension_enabled(cl_device_id device, const char *ext_name) {
    size_t nbytes = 0;
    if (clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, 0, nullptr, &nbytes) != CL_SUCCESS ||
        nbytes == 0) {
        return false;
    }
    std::vector<char> buf(nbytes);
    if (clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, nbytes, buf.data(), nullptr) !=
        CL_SUCCESS) {
        return false;
    }
    return std::strstr(buf.data(), ext_name) != nullptr;
}

bool device_supports_integer_dot_product(cl_device_id device, bool *hw_accel_out) {
    if (hw_accel_out) {
        *hw_accel_out = false;
    }
    if (!extension_enabled(device, "cl_khr_integer_dot_product")) {
        return false;
    }
    cl_bitfield caps = 0;
    const cl_int err = clGetDeviceInfo(device, CL_DEVICE_INTEGER_DOT_PRODUCT_CAPABILITIES_KHR,
                                       sizeof(caps), &caps, nullptr);
    const bool has_caps =
            err == CL_SUCCESS &&
            ((caps & CL_DEVICE_INTEGER_DOT_PRODUCT_INPUT_4x8BIT_KHR) != 0 ||
             (caps & CL_DEVICE_INTEGER_DOT_PRODUCT_INPUT_4x8BIT_PACKED_KHR) != 0);
    if (!has_caps) {
        if (err != CL_SUCCESS) {
            char ver[256] = {};
            if (clGetDeviceInfo(device, CL_DEVICE_VERSION, sizeof(ver), ver, nullptr) !=
                CL_SUCCESS) {
                return false;
            }
            int major = 0;
            int minor = 0;
            if (std::sscanf(ver, "OpenCL %d.%d", &major, &minor) != 2) {
                return false;
            }
            if (major < 3) {
                return false;
            }
        } else {
            return false;
        }
    }
    cl_device_integer_dot_product_acceleration_properties_khr accel = {};
    if (clGetDeviceInfo(device,
                        CL_DEVICE_INTEGER_DOT_PRODUCT_ACCELERATION_PROPERTIES_4x8BIT_PACKED_KHR,
                        sizeof(accel), &accel, nullptr) == CL_SUCCESS &&
        hw_accel_out) {
        *hw_accel_out = accel.accumulating_saturating_signed_accelerated == CL_TRUE;
    }
    return true;
}

bool device_is_discrete_gpu(cl_device_id device, cl_device_type type) {
    if (type != CL_DEVICE_TYPE_GPU) {
        return false;
    }
    cl_bool unified = CL_TRUE;
    if (clGetDeviceInfo(device, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(unified), &unified,
                        nullptr) != CL_SUCCESS) {
        return false;
    }
    return unified == CL_FALSE;
}

void append_devices_of_type(cl_platform_id plat, int platform_index, const char *pname,
                            cl_device_type type, std::vector<OclDeviceInfo> *out) {
    cl_uint n_devices = 0;
    if (clGetDeviceIDs(plat, type, 0, nullptr, &n_devices) != CL_SUCCESS || n_devices == 0) {
        return;
    }
    std::vector<cl_device_id> devices(n_devices);
    if (clGetDeviceIDs(plat, type, n_devices, devices.data(), nullptr) != CL_SUCCESS) {
        return;
    }
    for (cl_uint di = 0; di < n_devices; ++di) {
        cl_device_id dev = devices[di];
        char dname[256] = {};
        char vname[256] = {};
        clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(dname), dname, nullptr);
        clGetDeviceInfo(dev, CL_DEVICE_VENDOR, sizeof(vname), vname, nullptr);
        OclDeviceInfo info;
        info.platform_index = platform_index;
        info.device_index = static_cast<int>(di);
        info.platform = plat;
        info.device = dev;
        info.type = type;
        info.platform_name = pname;
        info.device_name = dname;
        info.vendor_name = vname;
        info.discrete = device_is_discrete_gpu(dev, type);
        info.integer_dot_product = device_supports_integer_dot_product(dev, &info.integer_dot_product_hw);
        out->push_back(info);
    }
}

int candidate_rank(const OclDeviceInfo &d) {
    if (d.type == CL_DEVICE_TYPE_GPU && d.discrete) {
        return 0;
    }
    if (d.type == CL_DEVICE_TYPE_GPU) {
        return 1;
    }
    return 2; /* CPU / other */
}

} // namespace

OpenClContext::~OpenClContext() {
    release_xfer_();
    if (program) {
        clReleaseProgram(program);
        program = nullptr;
    }
    if (queue) {
        clReleaseCommandQueue(queue);
        queue = nullptr;
    }
    if (context) {
        clReleaseContext(context);
        context = nullptr;
    }
    device = nullptr;
    platform = nullptr;
}

std::string OpenClContext::error_string(cl_int err) {
    switch (err) {
    case CL_SUCCESS:
        return "CL_SUCCESS";
    case CL_DEVICE_NOT_FOUND:
        return "CL_DEVICE_NOT_FOUND";
    case CL_INVALID_VALUE:
        return "CL_INVALID_VALUE";
    case CL_INVALID_CONTEXT:
        return "CL_INVALID_CONTEXT";
    case CL_INVALID_COMMAND_QUEUE:
        return "CL_INVALID_COMMAND_QUEUE";
    case CL_INVALID_MEM_OBJECT:
        return "CL_INVALID_MEM_OBJECT";
    case CL_INVALID_PROGRAM:
        return "CL_INVALID_PROGRAM";
    case CL_INVALID_KERNEL:
        return "CL_INVALID_KERNEL";
    case CL_INVALID_WORK_GROUP_SIZE:
        return "CL_INVALID_WORK_GROUP_SIZE";
    case CL_INVALID_WORK_DIMENSION:
        return "CL_INVALID_WORK_DIMENSION";
    case CL_BUILD_PROGRAM_FAILURE:
        return "CL_BUILD_PROGRAM_FAILURE";
    default:
        return "cl_error_" + std::to_string(err);
    }
}

std::vector<OclDeviceInfo> OpenClContext::enumerate_devices(int platform_filter) {
    std::vector<OclDeviceInfo> out;

    cl_uint n_platforms = 0;
    if (clGetPlatformIDs(0, nullptr, &n_platforms) != CL_SUCCESS || n_platforms == 0) {
        return out;
    }
    std::vector<cl_platform_id> platforms(n_platforms);
    if (clGetPlatformIDs(n_platforms, platforms.data(), nullptr) != CL_SUCCESS) {
        return out;
    }

    std::vector<OclDeviceInfo> gpus;
    std::vector<OclDeviceInfo> cpus;
    for (cl_uint pi = 0; pi < n_platforms; ++pi) {
        if (platform_filter >= 0 && static_cast<int>(pi) != platform_filter) {
            continue;
        }
        char pname[256] = {};
        clGetPlatformInfo(platforms[pi], CL_PLATFORM_NAME, sizeof(pname), pname, nullptr);
        append_devices_of_type(platforms[pi], static_cast<int>(pi), pname, CL_DEVICE_TYPE_GPU,
                               &gpus);
        append_devices_of_type(platforms[pi], static_cast<int>(pi), pname, CL_DEVICE_TYPE_CPU,
                               &cpus);
    }

    /* Prefer discrete GPUs, then iGPUs; CPU only if no GPUs exist. */
    std::stable_sort(gpus.begin(), gpus.end(),
                     [](const OclDeviceInfo &a, const OclDeviceInfo &b) {
                         return candidate_rank(a) < candidate_rank(b);
                     });

    if (!gpus.empty()) {
        out = std::move(gpus);
    } else {
        out = std::move(cpus);
    }

    for (size_t i = 0; i < out.size(); ++i) {
        out[i].flat_index = static_cast<int>(i);
    }
    return out;
}

int OpenClContext::list_devices(int platform_filter) {
    const std::vector<OclDeviceInfo> devices = enumerate_devices(platform_filter);
    if (devices.empty()) {
        std::printf("[ocl] no OpenCL devices found\n");
#if defined(__ANDROID__)
        std::printf("[ocl] Termux/Android hint: use the same LD_LIBRARY_PATH that makes clinfo work\n");
        std::printf("[ocl] (system + system_ext + vendor + SoC EGL/VNDK dirs; see docs/termux_note.md)\n");
        std::printf("[ocl] Requires a CP_STATIC_LIBCXX build so system libc++ does not break the binary\n");
#endif
        return 0;
    }

    std::printf("[ocl] OpenCL devices (use --devices N; default prefers discrete GPU):\n");
    for (const OclDeviceInfo &d : devices) {
        const char *kind = (d.type == CL_DEVICE_TYPE_CPU) ? "CPU"
                           : d.discrete                   ? "discrete GPU"
                                                          : "integrated GPU";
        std::printf("  [%d] %s\n", d.flat_index, d.device_name.c_str());
        std::printf("      platform[%d]=%s  %s%s\n", d.platform_index,
                    d.platform_name.c_str(), kind,
                    d.integer_dot_product
                            ? (d.integer_dot_product_hw ? "  int-dot" : "  int-dot (sw)")
                            : "");
    }
    return static_cast<int>(devices.size());
}

static int g_queue_throttle = 0;   /* 0 no hint, 1 high, 2 med, 3 low (cp_ocl_set_queue_throttle) */

extern "C" void cp_ocl_set_queue_throttle(int level) {
    g_queue_throttle = level;
}

#ifndef CL_VERSION_2_0
/* OpenCL 2.0 entry point (exported by the ICD loader); the 1.2 headers do not declare it. */
extern "C" CL_API_ENTRY cl_command_queue CL_API_CALL clCreateCommandQueueWithProperties(
        cl_context, cl_device_id, const cl_ulong *, cl_int *);
#endif

cl_command_queue OpenClContext::create_queue(cl_context context, cl_device_id device,
                                             cl_command_queue_properties props, cl_int *err) {
    if (g_queue_throttle >= 1 && g_queue_throttle <= 3) {
        size_t len = 0;
        std::string ext;
        if (clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, 0, nullptr, &len) == CL_SUCCESS && len > 0) {
            ext.resize(len);
            if (clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, len, &ext[0], nullptr) != CL_SUCCESS) {
                ext.clear();
            }
        }
        if (ext.find("cl_khr_throttle_hints") != std::string::npos) {
            /* CL_QUEUE_THROTTLE_KHR (0x1097): HIGH 1 << 0, MED 1 << 1, LOW 1 << 2. */
            const cl_ulong level = 1u << (g_queue_throttle - 1);
            const cl_ulong qp[] = {CL_QUEUE_PROPERTIES, props, 0x1097, level, 0};
            cl_int e = CL_SUCCESS;
            cl_command_queue q = clCreateCommandQueueWithProperties(context, device, qp, &e);
            if (q && e == CL_SUCCESS) {
                if (err) *err = CL_SUCCESS;
                return q;
            }
            std::fprintf(stderr, "[ocl] queue throttle hint rejected (%d); using a plain queue\n", e);
        }
    }
#ifdef CL_VERSION_2_0
    const cl_queue_properties plain[] = {CL_QUEUE_PROPERTIES, props, 0};
    return clCreateCommandQueueWithProperties(context, device, plain, err);
#else
    return clCreateCommandQueue(context, device, props, err);
#endif
}

bool OpenClContext::init(const OclDeviceInfo &pick) {
    platform = pick.platform;
    device = pick.device;
    platform_name = pick.platform_name;
    device_name = pick.device_name;
    vendor_name = pick.vendor_name;
    device_flat_index = pick.flat_index;
    discrete_gpu = pick.discrete;
    has_integer_dot_product = pick.integer_dot_product;
    has_integer_dot_product_hw = pick.integer_dot_product_hw;

    size_t max_wg = 0;
    if (clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_wg), &max_wg,
                        nullptr) == CL_SUCCESS &&
        max_wg > 0) {
        max_work_group_size = max_wg;
    }

    cl_int err = CL_SUCCESS;
    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (!context || err != CL_SUCCESS) {
        log_cl_error("clCreateContext", err);
        return false;
    }

    /* Nothing reads CL_PROFILING_COMMAND_* events, so keep the queue plain: with
       profiling enabled the runtime timestamps every command. CP_OCL_QUEUE_PROFILING=1
       turns it back on for ad-hoc event timing. */
    const char *prof_env = std::getenv("CP_OCL_QUEUE_PROFILING");
    const bool profiling = prof_env && prof_env[0] && prof_env[0] != '0';
    const cl_command_queue_properties qprops = profiling ? CL_QUEUE_PROFILING_ENABLE : 0;
    queue = create_queue(context, device, qprops, &err);
    if (!queue || err != CL_SUCCESS) {
        log_cl_error("clCreateCommandQueue", err);
        return false;
    }

    return true;
}

bool OpenClContext::init(int device_index, int platform_filter) {
    const std::vector<OclDeviceInfo> candidates = enumerate_devices(platform_filter);
    if (candidates.empty()) {
        std::fprintf(stderr, "No OpenCL GPU or CPU devices found\n");
        return false;
    }

    if (device_index < 0 || device_index >= static_cast<int>(candidates.size())) {
        std::fprintf(stderr,
                     "[ocl] invalid --devices %d (valid: 0..%d). Available:\n",
                     device_index, static_cast<int>(candidates.size()) - 1);
        list_devices(platform_filter);
        return false;
    }

    return init(candidates[static_cast<size_t>(device_index)]);
}

bool OpenClContext::build_program_from_source(const char *source, const char *build_options,
                                              bool quiet) {
    if (!context || !device || source == nullptr) {
        return false;
    }
    if (program) {
        clReleaseProgram(program);
        program = nullptr;
    }

    cl_int err = CL_SUCCESS;
    const char *srcs[] = {source};
    const size_t lens[] = {std::strlen(source)};
    program = clCreateProgramWithSource(context, 1, srcs, lens, &err);
    if (!program || err != CL_SUCCESS) {
        log_cl_error("clCreateProgramWithSource", err);
        return false;
    }

    err = clBuildProgram(program, 1, &device, build_options, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        if (!quiet) {
            size_t log_size = 0;
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
            std::vector<char> log(std::max(log_size, size_t{1}));
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log.size(), log.data(),
                                  nullptr);
            std::fprintf(stderr, "OpenCL build log:\n%s\n", log.data());
            log_cl_error("clBuildProgram", err);
        }
        clReleaseProgram(program);
        program = nullptr;
        return false;
    }
    dump_program_binary_(program);
    return true;
}

/* CP_OCL_DUMP_BIN=<path>: write CL_PROGRAM_BINARIES of every successfully built
   program (first to <path>, later ones to <path>.1, <path>.2, ...). On AMD the
   blob is an ELF code object: inspect with
   llvm-objdump -d --mcpu=gfx1103 <path>. Debug aid only. */
void OpenClContext::dump_program_binary_(cl_program prog) {
    const char *dump_path = std::getenv("CP_OCL_DUMP_BIN");
    if (!dump_path || !dump_path[0] || !prog) {
        return;
    }
    static int dump_index = 0;
    size_t bin_size = 0;
    if (clGetProgramInfo(prog, CL_PROGRAM_BINARY_SIZES, sizeof(bin_size), &bin_size, nullptr) !=
                CL_SUCCESS ||
        bin_size == 0) {
        return;
    }
    std::vector<unsigned char> bin(bin_size);
    unsigned char *bins[] = {bin.data()};
    if (clGetProgramInfo(prog, CL_PROGRAM_BINARIES, sizeof(bins), bins, nullptr) != CL_SUCCESS) {
        return;
    }
    std::string path = dump_path;
    if (dump_index > 0) {
        path += "." + std::to_string(dump_index);
    }
    ++dump_index;
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char *>(bin.data()),
              static_cast<std::streamsize>(bin.size()));
    std::fprintf(stderr, "[ocl] dumped program binary (%zu B) to %s\n", bin.size(),
                 path.c_str());
}

bool OpenClContext::build_program_from_file(const char *cl_path, const char *build_options,
                                            bool quiet) {
    const std::string source = read_text_file(cl_path);
    if (source.empty()) {
        if (!quiet) {
            std::fprintf(stderr, "Failed to read OpenCL source: %s\n", cl_path);
        }
        return false;
    }
    return build_program_from_source(source.c_str(), build_options, quiet);
}

#if !defined(_WIN32) && !defined(__APPLE__)
bool OpenClContext::probe_build(const char *source, const char *build_options) {
    if (!context || !device || source == nullptr) {
        return false;
    }

    /* The NVIDIA OpenCL driver is not fork-safe after context init: the child's
     * clBuildProgram always fails there. NVIDIA reports build errors normally, so
     * build in-process instead. */
    char vendor[128] = {0};
    if (clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(vendor) - 1, vendor, nullptr) ==
                CL_SUCCESS &&
        std::strstr(vendor, "NVIDIA") != nullptr) {
        return true;
    }

    cl_int probe_err = CL_SUCCESS;
    const char *srcs[] = {source};
    const size_t lens[] = {std::strlen(source)};
    cl_program probe_prog =
            clCreateProgramWithSource(context, 1, srcs, lens, &probe_err);
    if (!probe_prog || probe_err != CL_SUCCESS) {
        log_cl_error("clCreateProgramWithSource (probe)", probe_err);
        return false;
    }

    const pid_t pid = fork();
    if (pid < 0) {
        clReleaseProgram(probe_prog);
        std::fprintf(stderr, "OpenCL build probe: fork failed\n");
        return false;
    }
    if (pid == 0) {
        /* Let abort()/assert kill only this child with a normal signal exit. */
        struct sigaction sa = {};
        sa.sa_handler = SIG_DFL;
        sigaction(SIGABRT, &sa, nullptr);
        sigaction(SIGTRAP, &sa, nullptr);
        const cl_int err =
                clBuildProgram(probe_prog, 1, &device, build_options, nullptr, nullptr);
        _exit(err == CL_SUCCESS ? 0 : 1);
    }

    int status = 0;
    waitpid(pid, &status, 0);
    clReleaseProgram(probe_prog);

    if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
        return true;
    }
    if (WIFSIGNALED(status)) {
        std::fprintf(stderr,
                     "OpenCL build probe crashed (signal %d); skipping this build\n",
                     WTERMSIG(status));
    } else {
        std::fprintf(stderr,
                     "OpenCL build probe failed (exit %d); skipping this build\n",
                     WIFEXITED(status) ? WEXITSTATUS(status) : -1);
    }
    return false;
}
#else
bool OpenClContext::probe_build(const char * /*source*/, const char * /*build_options*/) {
    /* Windows: no fork. Apple: OpenCL is not fork-safe after context init.
       Rely on normal clBuildProgram error returns in both cases. */
    return true;
}
#endif

bool OpenClContext::safe_build_program_from_source(const char *source,
                                                   const char *build_options, bool quiet) {
    if (!probe_build(source, build_options)) {
        return false;
    }
    return build_program_from_source(source, build_options, quiet);
}

bool OpenClContext::safe_build_program_from_file(const char *cl_path,
                                                 const char *build_options, bool quiet) {
    const std::string source = read_text_file(cl_path);
    if (source.empty()) {
        std::fprintf(stderr, "Failed to read OpenCL source: %s\n", cl_path);
        return false;
    }
    return safe_build_program_from_source(source.c_str(), build_options, quiet);
}

cl_kernel OpenClContext::create_kernel(const char *name) const {
    cl_int err = CL_SUCCESS;
    cl_kernel kernel = clCreateKernel(program, name, &err);
    if (!kernel || err != CL_SUCCESS) {
        log_cl_error("clCreateKernel", err);
        return nullptr;
    }
    return kernel;
}

static constexpr size_t kXferStageBytes = 64 * 1024;

// cl_intel_unified_shared_memory entry points (declared here: the bundled CL headers may predate it).
typedef cl_bitfield cp_cl_mem_properties_intel;
typedef void *(CL_API_CALL *cp_clHostMemAllocINTEL_fn)(cl_context, const cp_cl_mem_properties_intel *,
                                                       size_t, cl_uint, cl_int *);
typedef cl_int(CL_API_CALL *cp_clMemBlockingFreeINTEL_fn)(cl_context, void *);
typedef cl_int(CL_API_CALL *cp_clSetKernelArgMemPointerINTEL_fn)(cl_kernel, cl_uint, const void *);

static const char *kXferSrc = R"CLC(
__kernel void cp_xfer_copy(__global const uchar *src, ulong soff, __global uchar *dst, ulong doff,
                           uint n) {
    const uint i = get_global_id(0);
    if (i < n) dst[doff + i] = src[soff + i];
}
)CLC";

bool OpenClContext::init_xfer_kernels_() const {
    size_t ext_len = 0;
    if (clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, 0, nullptr, &ext_len) != CL_SUCCESS || !ext_len) {
        return false;
    }
    std::string ext(ext_len, '\0');
    if (clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, ext_len, &ext[0], nullptr) != CL_SUCCESS ||
        ext.find("cl_intel_unified_shared_memory") == std::string::npos) {
        return false;
    }
    auto alloc_fn = reinterpret_cast<cp_clHostMemAllocINTEL_fn>(
            clGetExtensionFunctionAddressForPlatform(platform, "clHostMemAllocINTEL"));
    usm_free_fn_ = clGetExtensionFunctionAddressForPlatform(platform, "clMemBlockingFreeINTEL");
    usm_arg_fn_ = clGetExtensionFunctionAddressForPlatform(platform, "clSetKernelArgMemPointerINTEL");
    if (!alloc_fn || !usm_free_fn_ || !usm_arg_fn_) {
        return false;
    }
    cl_int err = CL_SUCCESS;
    xfer_host_ = alloc_fn(context, nullptr, kXferStageBytes, 0, &err);
    if (!xfer_host_ || err != CL_SUCCESS) {
        xfer_host_ = nullptr;
        return false;
    }
    xfer_prog_ = clCreateProgramWithSource(context, 1, &kXferSrc, nullptr, &err);
    if (!xfer_prog_ || err != CL_SUCCESS ||
        clBuildProgram(xfer_prog_, 1, &device, "", nullptr, nullptr) != CL_SUCCESS) {
        return false;
    }
    xfer_up_ = clCreateKernel(xfer_prog_, "cp_xfer_copy", &err);
    if (!xfer_up_ || err != CL_SUCCESS) return false;
    xfer_down_ = clCreateKernel(xfer_prog_, "cp_xfer_copy", &err);
    if (!xfer_down_ || err != CL_SUCCESS) return false;
    xfer_dd_ = clCreateKernel(xfer_prog_, "cp_xfer_copy", &err);
    if (!xfer_dd_ || err != CL_SUCCESS) return false;
    auto set_ptr = reinterpret_cast<cp_clSetKernelArgMemPointerINTEL_fn>(usm_arg_fn_);
    return set_ptr(xfer_up_, 0, xfer_host_) == CL_SUCCESS &&
           set_ptr(xfer_down_, 2, xfer_host_) == CL_SUCCESS;
}

void OpenClContext::release_xfer_() const {
    if (xfer_up_) clReleaseKernel(xfer_up_);
    if (xfer_down_) clReleaseKernel(xfer_down_);
    if (xfer_dd_) clReleaseKernel(xfer_dd_);
    xfer_dd_ = nullptr;
    if (xfer_prog_) clReleaseProgram(xfer_prog_);
    xfer_up_ = xfer_down_ = nullptr;
    xfer_prog_ = nullptr;
    if (xfer_host_ && usm_free_fn_ && context) {
        reinterpret_cast<cp_clMemBlockingFreeINTEL_fn>(usm_free_fn_)(context, xfer_host_);
    }
    xfer_host_ = nullptr;
}

bool OpenClContext::stage_small_xfer_(cl_mem buf, size_t bytes, bool device_write) const {
    if (bytes == 0 || bytes > kXferStageBytes || !context || !queue || !buf) {
        return false;
    }
    if (!xfer_kernels_on_()) {
        return false;
    }
    // A kernel must not write a CL_MEM_READ_ONLY buffer nor read a CL_MEM_WRITE_ONLY one: use the
    // transfer call for those.
    cl_mem_flags flags = 0;
    if (clGetMemObjectInfo(buf, CL_MEM_FLAGS, sizeof(flags), &flags, nullptr) != CL_SUCCESS ||
        (flags & (device_write ? CL_MEM_READ_ONLY : CL_MEM_WRITE_ONLY))) {
        return false;
    }
    return true;
}

bool OpenClContext::xfer_kernels_on_() const {
    if (!context || !queue) {
        return false;
    }
    if (xfer_stage_on_ < 0) {
        std::lock_guard<std::mutex> lock(*xfer_mu_);
        if (xfer_stage_on_ < 0) {
            const char *v = std::getenv("CP_OCL_XFER_STAGE");
            const bool intel = vendor_name.find("Intel") != std::string::npos ||
                               platform_name.find("Intel") != std::string::npos;
            bool on = (v && v[0]) ? v[0] != '0' : (discrete_gpu && intel);
            if (on && !init_xfer_kernels_()) {
                std::fprintf(stderr, "[ocl] small-transfer kernel path unavailable; using transfer calls\n");
                release_xfer_();
                on = false;
            }
            xfer_stage_on_ = on ? 1 : 0;
        }
    }
    return xfer_stage_on_ == 1;
}

bool OpenClContext::copy_buffer(cl_mem src, size_t src_offset, cl_mem dst, size_t dst_offset,
                                size_t bytes) const {
    if (bytes == 0) {
        return true;
    }
    bool kernel_path = bytes <= (size_t(16) << 20) && xfer_kernels_on_();
    if (kernel_path) {
        cl_mem_flags sf = 0, df = 0;
        kernel_path = clGetMemObjectInfo(src, CL_MEM_FLAGS, sizeof(sf), &sf, nullptr) == CL_SUCCESS &&
                      clGetMemObjectInfo(dst, CL_MEM_FLAGS, sizeof(df), &df, nullptr) == CL_SUCCESS &&
                      !(sf & CL_MEM_WRITE_ONLY) && !(df & CL_MEM_READ_ONLY);
    }
    if (!kernel_path) {
        const cl_int err = clEnqueueCopyBuffer(queue, src, dst, src_offset, dst_offset, bytes, 0,
                                               nullptr, nullptr);
        if (err != CL_SUCCESS) {
            log_cl_error("clEnqueueCopyBuffer", err);
            return false;
        }
        return true;
    }
    std::lock_guard<std::mutex> lock(*xfer_mu_);
    const cl_ulong soff = src_offset, doff = dst_offset;
    const cl_uint n = static_cast<cl_uint>(bytes);
    cl_int err = clSetKernelArg(xfer_dd_, 0, sizeof(cl_mem), &src);
    if (err == CL_SUCCESS) err = clSetKernelArg(xfer_dd_, 1, sizeof(soff), &soff);
    if (err == CL_SUCCESS) err = clSetKernelArg(xfer_dd_, 2, sizeof(cl_mem), &dst);
    if (err == CL_SUCCESS) err = clSetKernelArg(xfer_dd_, 3, sizeof(doff), &doff);
    if (err == CL_SUCCESS) err = clSetKernelArg(xfer_dd_, 4, sizeof(n), &n);
    if (err == CL_SUCCESS) {
        const size_t lws = 64, gws = (bytes + lws - 1) / lws * lws;
        err = clEnqueueNDRangeKernel(queue, xfer_dd_, 1, nullptr, &gws, &lws, 0, nullptr, nullptr);
    }
    if (err != CL_SUCCESS) {
        log_cl_error("kernel copy", err);
        return false;
    }
    return true;
}

// One copy kernel launch (64-wide work-groups), completed with clFinish.
static cl_int xfer_launch(cl_command_queue queue, cl_kernel k, size_t bytes, cl_event *ev) {
    const size_t lws = 64, gws = (bytes + lws - 1) / lws * lws;
    cl_int err = clEnqueueNDRangeKernel(queue, k, 1, nullptr, &gws, &lws, 0, nullptr, ev);
    if (err == CL_SUCCESS) err = clFinish(queue);
    return err;
}

// CP_OCL_XFER_WATCH=<ms> (debug): report transfers slower than <ms> with the buffer's size and flags
// and the event timestamps (queued -> submit -> start -> end; GPU times need CP_OCL_QUEUE_PROFILING=1).
static double xfer_watch_ms() {
    static const double ms = [] {
        const char *v = std::getenv("CP_OCL_XFER_WATCH");
        return (v && v[0]) ? std::atof(v) : 0.0;
    }();
    return ms;
}

static void xfer_watch_report(const char *op, cl_mem buf, cl_event ev, size_t bytes, size_t offset,
                              std::chrono::steady_clock::time_point t0) {
    const double sec =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (sec * 1e3 > xfer_watch_ms()) {
        cl_ulong q = 0, sb = 0, st = 0, en = 0;
        const bool prof =
                ev &&
                clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_QUEUED, sizeof(q), &q, nullptr) == CL_SUCCESS &&
                clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_SUBMIT, sizeof(sb), &sb, nullptr) == CL_SUCCESS &&
                clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(st), &st, nullptr) == CL_SUCCESS &&
                clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(en), &en, nullptr) == CL_SUCCESS;
        size_t buf_size = 0;
        cl_mem_flags buf_flags = 0;
        clGetMemObjectInfo(buf, CL_MEM_SIZE, sizeof(buf_size), &buf_size, nullptr);
        clGetMemObjectInfo(buf, CL_MEM_FLAGS, sizeof(buf_flags), &buf_flags, nullptr);
        std::fprintf(stderr, "[ocl] SLOW %s %zu B @%zu (buffer %zu B, flags 0x%llx): host %.3f s", op,
                     bytes, offset, buf_size, static_cast<unsigned long long>(buf_flags), sec);
        if (prof)
            std::fprintf(stderr, "; gpu queued->submit %.3f s, submit->start %.3f s, start->end %.3f s\n",
                         double(sb - q) * 1e-9, double(st - sb) * 1e-9, double(en - st) * 1e-9);
        else
            std::fprintf(stderr, "\n");
        std::fflush(stderr);
    }
    if (ev) clReleaseEvent(ev);
}

bool OpenClContext::write_buffer(cl_mem buf, const void *host, size_t bytes, size_t offset) const {
    const bool watch = xfer_watch_ms() > 0;
    cl_event ev = nullptr;
    const auto t0 = std::chrono::steady_clock::now();
    if (stage_small_xfer_(buf, bytes, /*device_write=*/true)) {
        std::lock_guard<std::mutex> lock(*xfer_mu_);
        std::memcpy(xfer_host_, host, bytes);
        const cl_ulong soff = 0, doff = offset;
        const cl_uint n = static_cast<cl_uint>(bytes);
        cl_int err = clSetKernelArg(xfer_up_, 1, sizeof(soff), &soff);
        if (err == CL_SUCCESS) err = clSetKernelArg(xfer_up_, 2, sizeof(cl_mem), &buf);
        if (err == CL_SUCCESS) err = clSetKernelArg(xfer_up_, 3, sizeof(doff), &doff);
        if (err == CL_SUCCESS) err = clSetKernelArg(xfer_up_, 4, sizeof(n), &n);
        if (err == CL_SUCCESS) err = xfer_launch(queue, xfer_up_, bytes, watch ? &ev : nullptr);
        if (err != CL_SUCCESS) {
            log_cl_error("kernel write", err);
            if (ev) clReleaseEvent(ev);
            return false;
        }
        if (watch) xfer_watch_report("write(kernel)", buf, ev, bytes, offset, t0);
        return true;
    }
    const cl_int err = clEnqueueWriteBuffer(queue, buf, CL_TRUE, offset, bytes, host, 0, nullptr,
                                            watch ? &ev : nullptr);
    if (err != CL_SUCCESS) {
        log_cl_error("clEnqueueWriteBuffer", err);
        if (ev) clReleaseEvent(ev);
        return false;
    }
    if (watch) xfer_watch_report("write", buf, ev, bytes, offset, t0);
    return true;
}

bool OpenClContext::read_buffer(cl_mem buf, void *host, size_t bytes, size_t offset) const {
    const bool watch = xfer_watch_ms() > 0;
    cl_event ev = nullptr;
    const auto t0 = std::chrono::steady_clock::now();
    if (stage_small_xfer_(buf, bytes, /*device_write=*/false)) {
        std::lock_guard<std::mutex> lock(*xfer_mu_);
        const cl_ulong soff = offset, doff = 0;
        const cl_uint n = static_cast<cl_uint>(bytes);
        cl_int err = clSetKernelArg(xfer_down_, 0, sizeof(cl_mem), &buf);
        if (err == CL_SUCCESS) err = clSetKernelArg(xfer_down_, 1, sizeof(soff), &soff);
        if (err == CL_SUCCESS) err = clSetKernelArg(xfer_down_, 3, sizeof(doff), &doff);
        if (err == CL_SUCCESS) err = clSetKernelArg(xfer_down_, 4, sizeof(n), &n);
        if (err == CL_SUCCESS) err = xfer_launch(queue, xfer_down_, bytes, watch ? &ev : nullptr);
        if (err != CL_SUCCESS) {
            log_cl_error("kernel read", err);
            if (ev) clReleaseEvent(ev);
            return false;
        }
        std::memcpy(host, xfer_host_, bytes);
        if (watch) xfer_watch_report("read(kernel)", buf, ev, bytes, offset, t0);
        return true;
    }
    const cl_int err = clEnqueueReadBuffer(queue, buf, CL_TRUE, offset, bytes, host, 0, nullptr,
                                           watch ? &ev : nullptr);
    if (err != CL_SUCCESS) {
        log_cl_error("clEnqueueReadBuffer", err);
        if (ev) clReleaseEvent(ev);
        return false;
    }
    if (watch) xfer_watch_report("read", buf, ev, bytes, offset, t0);
    return true;
}

cl_mem OpenClContext::alloc_buffer(size_t bytes, cl_mem_flags flags) const {
    cl_int err = CL_SUCCESS;
    cl_mem buf = clCreateBuffer(context, flags, bytes, nullptr, &err);
    if (!buf || err != CL_SUCCESS) {
        std::fprintf(stderr, "OpenCL clCreateBuffer(%zu bytes) failed (%d): %s\n", bytes, err,
                     error_string(err).c_str());
        return nullptr;
    }
    return buf;
}

std::string read_text_file(const char *path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return {};
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::string exe_directory(const char *argv0) {
    if (argv0 == nullptr || argv0[0] == '\0') {
        return ".";
    }
#ifdef _WIN32
    char path[MAX_PATH];
    DWORD n = GetModuleFileNameA(nullptr, path, MAX_PATH);
    if (n == 0 || n >= MAX_PATH) {
        return ".";
    }
    std::string s(path, path + n);
    const size_t slash = s.find_last_of("\\/");
    if (slash == std::string::npos) {
        return ".";
    }
    return s.substr(0, slash);
#else
    char path[PATH_MAX];
    const ssize_t n = readlink("/proc/self/exe", path, sizeof(path) - 1);
    if (n <= 0) {
        return ".";
    }
    path[n] = '\0';
    std::string s(path);
    const size_t slash = s.find_last_of('/');
    if (slash == std::string::npos) {
        return ".";
    }
    return s.substr(0, slash);
#endif
}
