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

struct OpenClContext {
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;

    std::string device_name;
    std::string platform_name;
    bool has_integer_dot_product = false;

    ~OpenClContext();

    bool init(int device_index = -1);
    bool build_program_from_file(const char *cl_path, const char *build_options = "");
    bool build_program_from_source(const char *source, const char *build_options = "");
    cl_kernel create_kernel(const char *name) const;

    bool write_buffer(cl_mem buf, const void *host, size_t bytes) const;
    bool read_buffer(cl_mem buf, void *host, size_t bytes) const;
    cl_mem alloc_buffer(size_t bytes, cl_mem_flags flags) const;

    static std::string error_string(cl_int err);
};

std::string read_text_file(const char *path);
std::string exe_directory(const char *argv0);
