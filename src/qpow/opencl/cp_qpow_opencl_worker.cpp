#include "cp_qpow_opencl_worker.h"

#include "cp_job_ctrl.h"
#include "cp_pool.h"
#include "opencl_context.hpp"
#include "qpow/miner.hpp"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

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

OpenClContext g_ctx;
cl_kernel g_kernel = nullptr;
cl_mem g_buf_results = nullptr;
cl_mem g_buf_midstate = nullptr;
cl_mem g_buf_start = nullptr;
cl_mem g_buf_target = nullptr;
cl_mem g_buf_dispatch = nullptr;
int g_ready = 0;
int g_platform_filter = -1;
uint32_t g_batch_size = 1000000u;
uint32_t g_compute_units = 1;

std::string directory_of_exe()
{
#ifdef _WIN32
    char path[MAX_PATH];
    DWORD n = GetModuleFileNameA(nullptr, path, MAX_PATH);
    if(n == 0 || n >= MAX_PATH) return ".";
    std::string s(path, path + n);
    size_t slash = s.find_last_of("\\/");
    return slash == std::string::npos ? "." : s.substr(0, slash);
#else
    char path[PATH_MAX];
    ssize_t n = readlink("/proc/self/exe", path, sizeof(path) - 1);
    if(n <= 0) return ".";
    path[n] = '\0';
    std::string s(path);
    size_t slash = s.find_last_of('/');
    return slash == std::string::npos ? "." : s.substr(0, slash);
#endif
}

std::string resolve_qpow_kernel_path()
{
    const std::string base = directory_of_exe();
#ifdef _WIN32
    return base + "\\kernels\\qpow_mining.cl";
#else
    return base + "/kernels/qpow_mining.cl";
#endif
}

void be64_to_le_u32s(const uint8_t be[64], uint32_t le[16])
{
    uint8_t raw[64];
    for(int i = 0; i < 64; i++) raw[i] = be[63 - i];
    for(int i = 0; i < 16; i++){
        le[i] = (uint32_t)raw[i * 4 + 0]
              | ((uint32_t)raw[i * 4 + 1] << 8)
              | ((uint32_t)raw[i * 4 + 2] << 16)
              | ((uint32_t)raw[i * 4 + 3] << 24);
    }
}

void le_u32s_to_be64(const uint32_t le[16], uint8_t be[64])
{
    uint8_t raw[64];
    for(int i = 0; i < 16; i++){
        raw[i * 4 + 0] = (uint8_t)(le[i] & 0xff);
        raw[i * 4 + 1] = (uint8_t)((le[i] >> 8) & 0xff);
        raw[i * 4 + 2] = (uint8_t)((le[i] >> 16) & 0xff);
        raw[i * 4 + 3] = (uint8_t)((le[i] >> 24) & 0xff);
    }
    for(int i = 0; i < 64; i++) be[i] = raw[63 - i];
}

void midstate_to_u32_le(const uint64_t mid[12], uint32_t out[24])
{
    for(int i = 0; i < 12; i++){
        out[2 * i] = (uint32_t)(mid[i] & 0xffffffffull);
        out[2 * i + 1] = (uint32_t)(mid[i] >> 32);
    }
}

/* Max +1 increments before low-256 wraps (carry into high half / midstate invalid). */
uint32_t headroom_low256(const uint32_t start_le[16])
{
    /* rem = ~low + 1 (2's complement distance to 2^256) */
    uint32_t rem[8];
    uint32_t carry = 1;
    for(int i = 0; i < 8; i++){
        uint64_t s = (uint64_t)(~start_le[i] & 0xffffffffu) + carry;
        rem[i] = (uint32_t)s;
        carry = (uint32_t)(s >> 32);
    }
    for(int i = 7; i >= 1; i--){
        if(rem[i] != 0) return 0xffffffffu;
    }
    return rem[0] == 0 ? 0xffffffffu : rem[0];
}

void release_buffers()
{
    auto rel = [](cl_mem& m){
        if(m){ clReleaseMemObject(m); m = nullptr; }
    };
    rel(g_buf_results);
    rel(g_buf_midstate);
    rel(g_buf_start);
    rel(g_buf_target);
    rel(g_buf_dispatch);
    if(g_kernel){
        clReleaseKernel(g_kernel);
        g_kernel = nullptr;
    }
}

int cancelled()
{
    return cp_job_should_cancel() || cp_pool_conn_lost();
}

} // namespace

extern "C" int cp_qpow_opencl_worker_list_devices(void)
{
    return OpenClContext::list_devices(g_platform_filter);
}

extern "C" int cp_qpow_opencl_worker_init(int* devices, int ndev)
{
    cp_qpow_opencl_worker_shutdown();

    int device_index = 0;
    if(devices && ndev > 0){
        device_index = devices[0];
        if(ndev > 1){
            fprintf(stderr,
                    "[qpow-ocl] warning: Quantus OpenCL uses a single device; "
                    "ignoring --devices after %d\n",
                    device_index);
        }
    }

    if(!g_ctx.init(device_index, g_platform_filter)){
        fprintf(stderr, "[qpow-ocl] OpenCL context init failed\n");
        return -1;
    }

    cl_uint cu = 1;
    clGetDeviceInfo(g_ctx.device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, nullptr);
    g_compute_units = cu ? cu : 1;

    const std::string kpath = resolve_qpow_kernel_path();
    if(!g_ctx.safe_build_program_from_file(kpath.c_str(), "")){
        fprintf(stderr, "[qpow-ocl] kernel build failed (%s)\n", kpath.c_str());
        return -1;
    }

    g_kernel = g_ctx.create_kernel("mining_main");
    if(!g_kernel){
        fprintf(stderr, "[qpow-ocl] create_kernel mining_main failed\n");
        return -1;
    }

    const size_t results_bytes = (1 + 16 + 16) * sizeof(uint32_t);
    g_buf_results = g_ctx.alloc_buffer(results_bytes, CL_MEM_READ_WRITE);
    g_buf_midstate = g_ctx.alloc_buffer(24 * sizeof(uint32_t), CL_MEM_READ_ONLY);
    g_buf_start = g_ctx.alloc_buffer(16 * sizeof(uint32_t), CL_MEM_READ_ONLY);
    g_buf_target = g_ctx.alloc_buffer(16 * sizeof(uint32_t), CL_MEM_READ_ONLY);
    g_buf_dispatch = g_ctx.alloc_buffer(4 * sizeof(uint32_t), CL_MEM_READ_ONLY);
    if(!g_buf_results || !g_buf_midstate || !g_buf_start || !g_buf_target || !g_buf_dispatch){
        fprintf(stderr, "[qpow-ocl] buffer alloc failed\n");
        release_buffers();
        return -1;
    }

    cl_int err = 0;
    err |= clSetKernelArg(g_kernel, 0, sizeof(cl_mem), &g_buf_results);
    err |= clSetKernelArg(g_kernel, 1, sizeof(cl_mem), &g_buf_midstate);
    err |= clSetKernelArg(g_kernel, 2, sizeof(cl_mem), &g_buf_start);
    err |= clSetKernelArg(g_kernel, 3, sizeof(cl_mem), &g_buf_target);
    err |= clSetKernelArg(g_kernel, 4, sizeof(cl_mem), &g_buf_dispatch);
    if(err != CL_SUCCESS){
        fprintf(stderr, "[qpow-ocl] set kernel args failed (%d)\n", err);
        release_buffers();
        return -1;
    }

    g_ready = 1;
    printf("[qpow-ocl] device[%d]: %s (%s) CUs=%u batch=%u\n",
           g_ctx.device_flat_index, g_ctx.device_name.c_str(),
           g_ctx.discrete_gpu ? "discrete" : "integrated",
           (unsigned)g_compute_units, (unsigned)g_batch_size);
    fflush(stdout);
    return 0;
}

extern "C" void cp_qpow_opencl_worker_shutdown(void)
{
    release_buffers();
    if(g_ctx.program){
        clReleaseProgram(g_ctx.program);
        g_ctx.program = nullptr;
    }
    if(g_ctx.queue){
        clReleaseCommandQueue(g_ctx.queue);
        g_ctx.queue = nullptr;
    }
    if(g_ctx.context){
        clReleaseContext(g_ctx.context);
        g_ctx.context = nullptr;
    }
    g_ctx.device = nullptr;
    g_ctx.platform = nullptr;
    g_ctx.device_flat_index = -1;
    g_ready = 0;
}

extern "C" int cp_qpow_opencl_worker_is_ready(void)
{
    return g_ready;
}

extern "C" int cp_qpow_opencl_worker_search(
    const uint8_t header[32],
    const uint8_t target_be[64],
    const uint8_t start_be[64],
    uint64_t count,
    uint8_t out_nonce_be[64],
    uint8_t out_hash_be[64],
    uint64_t* out_hashes)
{
    if(!g_ready || !out_hashes || count == 0) return CP_QPOW_OCL_ERROR;
    *out_hashes = 0;

    uint32_t target_le[16];
    be64_to_le_u32s(target_be, target_le);
    if(!g_ctx.write_buffer(g_buf_target, target_le, sizeof(target_le)))
        return CP_QPOW_OCL_ERROR;

    uint8_t cur_be[64];
    memcpy(cur_be, start_be, 64);
    uint64_t done = 0;

    while(done < count){
        if(cancelled()){
            *out_hashes = done;
            return CP_QPOW_OCL_CANCELLED;
        }

        uint32_t start_le[16];
        be64_to_le_u32s(cur_be, start_le);

        uint64_t remaining = count - done;
        uint32_t batch = g_batch_size;
        if((uint64_t)batch > remaining) batch = (uint32_t)remaining;

        uint32_t room = headroom_low256(start_le);
        if(room == 0){
            /* Advance high half by rebuilding midstate path: bump BE high via +2^256
               is not needed if host always keeps salt in high; treat as error. */
            fprintf(stderr, "[qpow-ocl] nonce low-256 exhausted\n");
            *out_hashes = done;
            return CP_QPOW_OCL_ERROR;
        }
        if(batch > room) batch = room;

        uint64_t mid[12];
        qpow::mining_midstate(header, cur_be, mid);
        uint32_t mid_u32[24];
        midstate_to_u32_le(mid, mid_u32);

        if(!g_ctx.write_buffer(g_buf_midstate, mid_u32, sizeof(mid_u32)))
            return CP_QPOW_OCL_ERROR;
        if(!g_ctx.write_buffer(g_buf_start, start_le, sizeof(start_le)))
            return CP_QPOW_OCL_ERROR;

        uint32_t zeros[33] = {0};
        if(!g_ctx.write_buffer(g_buf_results, zeros, sizeof(zeros)))
            return CP_QPOW_OCL_ERROR;

        /* One nonce per thread (pad global to local size). Matches wgpu occupancy
         * better than CU×8 under-dispatch with long per-thread loops. */
        const uint32_t local = 256;
        const uint32_t nonces_per_thread = 1;
        uint32_t num_wg = (batch + local - 1) / local;
        if(num_wg < 1) num_wg = 1;
        uint32_t total_threads = num_wg * local;

        uint32_t dispatch[4] = {
            total_threads, nonces_per_thread, batch, 0
        };
        if(!g_ctx.write_buffer(g_buf_dispatch, dispatch, sizeof(dispatch)))
            return CP_QPOW_OCL_ERROR;

        size_t global = (size_t)total_threads;
        size_t local_sz = local;
        cl_int err = clEnqueueNDRangeKernel(g_ctx.queue, g_kernel, 1, nullptr,
                                            &global, &local_sz, 0, nullptr, nullptr);
        if(err != CL_SUCCESS){
            fprintf(stderr, "[qpow-ocl] enqueue failed (%d)\n", err);
            return CP_QPOW_OCL_ERROR;
        }

        uint32_t results[33];
        if(!g_ctx.read_buffer(g_buf_results, results, sizeof(results)))
            return CP_QPOW_OCL_ERROR;

        done += batch;
        *out_hashes = done;

        if(results[0] != 0){
            le_u32s_to_be64(results + 1, out_nonce_be);
            /* hash_le layout matches wgpu results[17..]; convert to BE hash bytes */
            le_u32s_to_be64(results + 17, out_hash_be);
            return CP_QPOW_OCL_OK_FOUND;
        }

        /* Advance BE start by batch */
        uint8_t tmp[64];
        memcpy(tmp, cur_be, 64);
        /* add batch into BE nonce = add into LE then convert back */
        uint32_t next_le[16];
        memcpy(next_le, start_le, sizeof(next_le));
        uint64_t add = batch;
        for(int i = 0; i < 16 && add; i++){
            uint64_t s = (uint64_t)next_le[i] + (add & 0xffffffffull);
            next_le[i] = (uint32_t)s;
            add = (add >> 32) + (s >> 32);
        }
        le_u32s_to_be64(next_le, cur_be);
        (void)tmp;
    }

    return CP_QPOW_OCL_OK_EXHAUSTED;
}
