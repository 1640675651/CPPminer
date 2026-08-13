#pragma once

#define CL_TARGET_OPENCL_VERSION 120
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <cstddef>
#include <string>
#include <vector>

/* Flat enumeration entry (GPU discrete → GPU integrated → CPU). */
struct OclDeviceInfo {
    int flat_index = 0;
    int platform_index = 0;
    int device_index = 0; /* index within platform+type query order */
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    cl_device_type type = CL_DEVICE_TYPE_GPU;
    std::string platform_name;
    std::string device_name;
    std::string vendor_name;
    bool discrete = false; /* GPU with CL_DEVICE_HOST_UNIFIED_MEMORY == false */
    bool integer_dot_product = false;
};

struct OpenClContext {
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;

    std::string device_name;
    std::string platform_name;
    std::string vendor_name;
    int device_flat_index = -1;
    bool discrete_gpu = false;
    bool has_integer_dot_product = false;

    ~OpenClContext();

    /* Enumerate devices. platform_filter < 0 → all platforms. */
    static std::vector<OclDeviceInfo> enumerate_devices(int platform_filter = -1);

    /* Print enumerated devices to stdout; returns count. */
    static int list_devices(int platform_filter = -1);

    /* Select by flat index from enumerate_devices(). Fails if out of range. */
    bool init(int device_index = 0, int platform_filter = -1);
    bool build_program_from_file(const char *cl_path, const char *build_options = "");
    bool build_program_from_source(const char *source, const char *build_options = "");

    /* Probe clBuildProgram in a child process first. Drivers that abort() inside
       the compiler (e.g. Beignet on __builtin_amdgcn_sdot4) kill only the child;
       the parent returns false and can fall back to another build. Does not
       modify this->program. */
    bool probe_build(const char *source, const char *build_options = "");

    /* probe_build() then build_program_from_source/file if the probe survives. */
    bool safe_build_program_from_source(const char *source, const char *build_options = "");
    bool safe_build_program_from_file(const char *cl_path, const char *build_options = "");

    cl_kernel create_kernel(const char *name) const;

    bool write_buffer(cl_mem buf, const void *host, size_t bytes) const;
    bool read_buffer(cl_mem buf, void *host, size_t bytes) const;
    cl_mem alloc_buffer(size_t bytes, cl_mem_flags flags) const;

    static std::string error_string(cl_int err);
};

std::string read_text_file(const char *path);
std::string exe_directory(const char *argv0);
