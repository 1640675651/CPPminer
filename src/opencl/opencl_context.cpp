#include "opencl_context.hpp"

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

bool OpenClContext::init(int device_index) {
    cl_uint n_platforms = 0;
    if (clGetPlatformIDs(0, nullptr, &n_platforms) != CL_SUCCESS || n_platforms == 0) {
        std::fprintf(stderr, "No OpenCL platforms found\n");
        return false;
    }

    std::vector<cl_platform_id> platforms(n_platforms);
    if (clGetPlatformIDs(n_platforms, platforms.data(), nullptr) != CL_SUCCESS) {
        return false;
    }

    struct Candidate {
        cl_platform_id platform;
        cl_device_id device;
        cl_device_type type;
        std::string platform_name;
        std::string device_name;
        bool dot;
    };
    std::vector<Candidate> candidates;

    for (cl_platform_id plat : platforms) {
        char pname[256] = {};
        clGetPlatformInfo(plat, CL_PLATFORM_NAME, sizeof(pname), pname, nullptr);

        cl_uint n_devices = 0;
        if (clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 0, nullptr, &n_devices) != CL_SUCCESS ||
            n_devices == 0) {
            continue;
        }
        std::vector<cl_device_id> devices(n_devices);
        if (clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, n_devices, devices.data(), nullptr) !=
            CL_SUCCESS) {
            continue;
        }
        for (cl_device_id dev : devices) {
            char dname[256] = {};
            clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(dname), dname, nullptr);
            candidates.push_back(
                    {plat, dev, CL_DEVICE_TYPE_GPU, pname, dname,
                     extension_enabled(dev, "cl_khr_integer_dot_product")});
        }
    }

    if (candidates.empty()) {
        for (cl_platform_id plat : platforms) {
            char pname[256] = {};
            clGetPlatformInfo(plat, CL_PLATFORM_NAME, sizeof(pname), pname, nullptr);
            cl_uint n_devices = 0;
            if (clGetDeviceIDs(plat, CL_DEVICE_TYPE_CPU, 0, nullptr, &n_devices) !=
                        CL_SUCCESS ||
                n_devices == 0) {
                continue;
            }
            std::vector<cl_device_id> devices(n_devices);
            if (clGetDeviceIDs(plat, CL_DEVICE_TYPE_CPU, n_devices, devices.data(), nullptr) !=
                CL_SUCCESS) {
                continue;
            }
            for (cl_device_id dev : devices) {
                char dname[256] = {};
                clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(dname), dname, nullptr);
                candidates.push_back(
                        {plat, dev, CL_DEVICE_TYPE_CPU, pname, dname,
                         extension_enabled(dev, "cl_khr_integer_dot_product")});
            }
        }
    }

    if (candidates.empty()) {
        std::fprintf(stderr, "No OpenCL GPU or CPU devices found\n");
        return false;
    }

    int pick = 0;
    if (device_index >= 0 && device_index < static_cast<int>(candidates.size())) {
        pick = device_index;
    }

    platform = candidates[static_cast<size_t>(pick)].platform;
    device = candidates[static_cast<size_t>(pick)].device;
    platform_name = candidates[static_cast<size_t>(pick)].platform_name;
    device_name = candidates[static_cast<size_t>(pick)].device_name;
    has_integer_dot_product = candidates[static_cast<size_t>(pick)].dot;

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
