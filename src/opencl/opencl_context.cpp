#include "opencl_context.hpp"

#include <algorithm>
#include <cstdio>
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
#include <unistd.h>
#endif

namespace {

void log_cl_error(const char *what, cl_int err) {
    std::fprintf(stderr, "OpenCL %s failed (%d): %s\n", what, err,
                 OpenClContext::error_string(err).c_str());
}

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
        clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(dname), dname, nullptr);
        OclDeviceInfo info;
        info.platform_index = platform_index;
        info.device_index = static_cast<int>(di);
        info.platform = plat;
        info.device = dev;
        info.type = type;
        info.platform_name = pname;
        info.device_name = dname;
        info.discrete = device_is_discrete_gpu(dev, type);
        info.integer_dot_product = extension_enabled(dev, "cl_khr_integer_dot_product");
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
                    d.integer_dot_product ? "  int-dot" : "");
    }
    return static_cast<int>(devices.size());
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

    const OclDeviceInfo &pick = candidates[static_cast<size_t>(device_index)];
    platform = pick.platform;
    device = pick.device;
    platform_name = pick.platform_name;
    device_name = pick.device_name;
    device_flat_index = pick.flat_index;
    discrete_gpu = pick.discrete;
    has_integer_dot_product = pick.integer_dot_product;

    cl_int err = CL_SUCCESS;
    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (!context || err != CL_SUCCESS) {
        log_cl_error("clCreateContext", err);
        return false;
    }

#ifdef CL_VERSION_2_0
    cl_queue_properties props[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    queue = clCreateCommandQueueWithProperties(context, device, props, &err);
#else
    queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
#endif
    if (!queue || err != CL_SUCCESS) {
        log_cl_error("clCreateCommandQueue", err);
        return false;
    }

    return true;
}

bool OpenClContext::build_program_from_source(const char *source,
                                              const char *build_options) {
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
        size_t log_size = 0;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> log(std::max(log_size, size_t{1}));
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log.size(), log.data(),
                              nullptr);
        std::fprintf(stderr, "OpenCL build log:\n%s\n", log.data());
        log_cl_error("clBuildProgram", err);
        return false;
    }
    return true;
}

bool OpenClContext::build_program_from_file(const char *cl_path,
                                            const char *build_options) {
    const std::string source = read_text_file(cl_path);
    if (source.empty()) {
        std::fprintf(stderr, "Failed to read OpenCL source: %s\n", cl_path);
        return false;
    }
    return build_program_from_source(source.c_str(), build_options);
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

bool OpenClContext::write_buffer(cl_mem buf, const void *host, size_t bytes) const {
    const cl_int err =
            clEnqueueWriteBuffer(queue, buf, CL_TRUE, 0, bytes, host, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        log_cl_error("clEnqueueWriteBuffer", err);
        return false;
    }
    return true;
}

bool OpenClContext::read_buffer(cl_mem buf, void *host, size_t bytes) const {
    const cl_int err =
            clEnqueueReadBuffer(queue, buf, CL_TRUE, 0, bytes, host, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        log_cl_error("clEnqueueReadBuffer", err);
        return false;
    }
    return true;
}

cl_mem OpenClContext::alloc_buffer(size_t bytes, cl_mem_flags flags) const {
    cl_int err = CL_SUCCESS;
    cl_mem buf = clCreateBuffer(context, flags, bytes, nullptr, &err);
    if (!buf || err != CL_SUCCESS) {
        log_cl_error("clCreateBuffer", err);
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
