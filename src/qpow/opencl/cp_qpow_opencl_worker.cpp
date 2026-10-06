#include "cp_qpow_opencl_worker.h"

#include "cp_job_ctrl.h"
#include "cp_pool.h"
#include "opencl_context.hpp"
#include "qpow/miner.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
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

/* Host side of kernels/qpow_mining.cl. Only the last big-endian nonce word varies inside a
 * launch: the host folds the midstate, the other nonce words, the first linear layer and the
 * first round constants into a 12-element `pre` vector, and re-hashes every candidate the
 * kernel reports with the reference Poseidon2 before returning it. */

namespace {

typedef uint32_t u32;
typedef uint64_t u64;

OpenClContext g_ctx;
cl_kernel g_kernel = nullptr;
cl_mem g_buf_out = nullptr;   /* [0] = candidate count, [1..15] = nonce index */
cl_mem g_buf_pre = nullptr;   /* 12 x u64 */
int g_ready = 0;
int g_platform_filter = -1;
u32 g_batch_req = 0;          /* 0 = automatic */
u32 g_launch = 1u << 20;      /* nonces per launch */
size_t g_local = 64;
int g_mul = 0;
std::string g_kernel_path;

constexpr u32 k_max_candidates = 15;

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

void be_add(uint8_t n[64], u64 v)
{
    for(int i = 63; i >= 0 && v; i--){
        v += n[i];
        n[i] = (uint8_t)v;
        v >>= 8;
    }
}

u32 counter_word(const uint8_t nonce[64])
{
    return ((u32)nonce[60] << 24) | ((u32)nonce[61] << 16) | ((u32)nonce[62] << 8) | nonce[63];
}

/* Kernel input for nonces nonce .. nonce + count - 1 (counter word must not wrap). */
void make_pre(const u64 mid[12], const uint8_t nonce[64], u64 pre[12])
{
    u64 s[12];
    memcpy(s, mid, sizeof(s));
    for(int i = 0; i < 7; i++){
        u32 w;
        memcpy(&w, nonce + 32 + 4 * i, 4);
        s[i] = qpow::gf_add(s[i], (u64)w);
    }
    qpow::ext_layer(s);
    for(int i = 0; i < 12; i++)
        pre[i] = qpow::gf_canon(qpow::gf_add(s[i], qpow::RC_INITIAL[0][i]));
}

/* What the kernel computes for one nonce: canonical s[0] after the second permutation. */
u64 cpu_out0(const uint8_t header[32], const uint8_t nonce[64])
{
    u64 s[12];
    qpow::mining_midstate(header, nonce, s);
    qpow::absorb32(s, nonce + 32);
    qpow::permute(s);
    s[0] = qpow::gf_add(s[0], 1);
    s[1] = qpow::gf_add(s[1], 1);
    qpow::permute(s);
    return qpow::gf_canon(s[0]);
}

void release_kernel()
{
    if(g_kernel){ clReleaseKernel(g_kernel); g_kernel = nullptr; }
    if(g_ctx.program){ clReleaseProgram(g_ctx.program); g_ctx.program = nullptr; }
}

void release_buffers()
{
    auto rel = [](cl_mem& m){
        if(m){ clReleaseMemObject(m); m = nullptr; }
    };
    rel(g_buf_out);
    rel(g_buf_pre);
    release_kernel();
}

bool build_variant(int mul)
{
    release_kernel();
    char opts[64];
    snprintf(opts, sizeof(opts), "-DQV_MUL=%d", mul);
    if(!g_ctx.safe_build_program_from_file(g_kernel_path.c_str(), opts)) return false;
    g_kernel = g_ctx.create_kernel("qpow_scan");
    return g_kernel != nullptr;
}

/* Enqueue one scan and wait for it. dump: optional per-nonce output (self-test). */
bool run_scan(const u64 pre[12], u64 t0, u32 w0, u32 count, size_t local, cl_mem dump, u32 out[16])
{
    const u32 zero[16] = {};
    if(!g_ctx.write_buffer(g_buf_out, zero, sizeof(zero))) return false;
    if(!g_ctx.write_buffer(g_buf_pre, pre, 12 * sizeof(u64))) return false;
    cl_int err = CL_SUCCESS;
    err |= clSetKernelArg(g_kernel, 0, sizeof(cl_mem), &g_buf_out);
    err |= clSetKernelArg(g_kernel, 1, sizeof(cl_mem), dump ? &dump : nullptr);
    err |= clSetKernelArg(g_kernel, 2, sizeof(cl_mem), &g_buf_pre);
    err |= clSetKernelArg(g_kernel, 3, sizeof(u64), &t0);
    err |= clSetKernelArg(g_kernel, 4, sizeof(u32), &w0);
    err |= clSetKernelArg(g_kernel, 5, sizeof(u32), &count);
    if(err != CL_SUCCESS){
        fprintf(stderr, "[qpow-ocl] set kernel args failed (%d)\n", err);
        return false;
    }
    const size_t global = ((size_t)count + local - 1) / local * local;
    err = clEnqueueNDRangeKernel(g_ctx.queue, g_kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr);
    if(err != CL_SUCCESS){
        fprintf(stderr, "[qpow-ocl] enqueue failed (%d)\n", err);
        return false;
    }
    return g_ctx.read_buffer(g_buf_out, out, 16 * sizeof(u32));
}

/* Hash 4096 nonces with the current program and compare a spread of them with the
 * reference code. */
bool self_test(size_t local)
{
    uint8_t header[32], nonce[64];
    for(int i = 0; i < 32; i++) header[i] = (uint8_t)(i * 37 + 11);
    for(int i = 0; i < 64; i++) nonce[i] = (uint8_t)(i * 91 + 5);
    nonce[60] = 0x10;
    nonce[63] = 0xF0;
    const u32 n = 4096;
    u64 mid[12], pre[12];
    qpow::mining_midstate(header, nonce, mid);
    make_pre(mid, nonce, pre);
    cl_mem dump = g_ctx.alloc_buffer(n * sizeof(u64), CL_MEM_READ_WRITE);
    if(!dump) return false;
    u32 out[16];
    std::vector<u64> got(n);
    bool ok = run_scan(pre, 0, counter_word(nonce), n, local, dump, out) &&
              g_ctx.read_buffer(dump, got.data(), n * sizeof(u64));
    clReleaseMemObject(dump);
    if(!ok) return false;
    const u32 idx[] = {0, 1, 2, 15, 16, 255, 256, 1000, 2047, 4095};
    for(u32 i : idx){
        uint8_t nn[64];
        memcpy(nn, nonce, 64);
        be_add(nn, i);
        const u64 ref = cpu_out0(header, nn);
        if(got[i] != ref){
            fprintf(stderr, "[qpow-ocl] self-test mismatch (mul=%d) at nonce +%u: got %016llx, want %016llx\n",
                    g_mul, i, (unsigned long long)got[i], (unsigned long long)ref);
            return false;
        }
    }
    return true;
}

/* Hash rate of the current program for one work-group size: a short run sizes the probe to
 * about 50 ms, then the best of three probes counts. */
double probe_rate(size_t local)
{
    u64 pre[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    u32 out[16];
    auto timed = [&](u32 n) -> double {
        auto t0 = std::chrono::steady_clock::now();
        if(!run_scan(pre, 0, 0, n, local, nullptr, out)) return 0;
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        return s > 0 ? n / s : 0;
    };
    const double est = timed(1u << 18);
    if(est <= 0) return 0;
    double n = est * 0.05;
    n = n < 65536.0 ? 65536.0 : (n > 268435456.0 ? 268435456.0 : n);
    double best = 0;
    for(int rep = 0; rep < 3; rep++){
        const double r = timed((u32)n);
        if(r > best) best = r;
    }
    return best;
}

int cancelled()
{
    return cp_job_should_cancel() || cp_pool_conn_lost();
}

} // namespace

extern "C" void cp_qpow_opencl_worker_set_batch_size(uint32_t batch)
{
    g_batch_req = batch;
}

extern "C" uint32_t cp_qpow_opencl_worker_batch_size(void)
{
    return g_launch;
}

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

    g_kernel_path = resolve_qpow_kernel_path();
    g_buf_out = g_ctx.alloc_buffer(16 * sizeof(u32), CL_MEM_READ_WRITE);
    g_buf_pre = g_ctx.alloc_buffer(12 * sizeof(u64), CL_MEM_READ_ONLY);
    if(!g_buf_out || !g_buf_pre){
        fprintf(stderr, "[qpow-ocl] buffer alloc failed\n");
        release_buffers();
        return -1;
    }

    /* Pick the product variant and work-group size that run fastest on this device: the
     * best one differs by vendor (NVIDIA: 64-bit mul_hi; Intel/AMD: 32x32+64 multiply-add
     * chain). CP_QPOW_OCL_MUL=1|2|3 forces one. Every candidate must pass the self-test. */
    std::vector<int> muls = {3, 1};
    if(const char* e = getenv("CP_QPOW_OCL_MUL")){
        const int m = atoi(e);
        if(m >= 1 && m <= 3) muls = {m};
    }
    const size_t locals[] = {64, 256};
    int best_mul = 0;
    size_t best_local = 64;
    double best_rate = 0;
    for(int m : muls){
        g_mul = m;
        if(!build_variant(m)){
            fprintf(stderr, "[qpow-ocl] kernel build failed (%s, mul=%d)\n", g_kernel_path.c_str(), m);
            continue;
        }
        size_t max_wg = 0;
        clGetKernelWorkGroupInfo(g_kernel, g_ctx.device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(max_wg), &max_wg, nullptr);
        if(!self_test(max_wg >= 64 ? 64 : max_wg)) continue;
        for(size_t l : locals){
            if(l > max_wg) continue;
            const double r = probe_rate(l);
            printf("[qpow-ocl] probe mul=%d local=%zu: %.2f MH/s\n", m, l, r / 1e6);
            if(r > best_rate){
                best_rate = r;
                best_mul = m;
                best_local = l;
            }
        }
    }
    if(!best_mul){
        fprintf(stderr, "[qpow-ocl] no kernel variant passed the self-test, not mining on this device\n");
        release_buffers();
        return -1;
    }
    g_mul = best_mul;
    g_local = best_local;
    if(muls.size() > 1 && !build_variant(best_mul)){
        fprintf(stderr, "[qpow-ocl] kernel rebuild failed\n");
        release_buffers();
        return -1;
    }
    /* Start near 100 ms per launch; search() keeps it there unless --batch-size is set. */
    g_launch = g_batch_req ? g_batch_req : (u32)(best_rate * 0.1 > (1u << 16) ? best_rate * 0.1 : (1u << 16));

    g_ready = 1;
    printf("[qpow-ocl] device[%d]: %s (%s) CUs=%u kernel mul=%d local=%zu batch=%s self-test ok\n",
           g_ctx.device_flat_index, g_ctx.device_name.c_str(),
           g_ctx.discrete_gpu ? "discrete" : "integrated",
           (unsigned)cu, g_mul, g_local, g_batch_req ? "fixed" : "auto");
    fflush(stdout);
    return 0;
}

extern "C" void cp_qpow_opencl_worker_shutdown(void)
{
    release_buffers();
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

    u64 t0 = 0;
    for(int i = 0; i < 8; i++) t0 = (t0 << 8) | target_be[i];

    uint8_t cur[64];
    memcpy(cur, start_be, 64);
    uint8_t mid_high[32];
    u64 mid[12];
    bool have_mid = false;
    uint64_t done = 0;

    while(done < count){
        if(cancelled()){
            *out_hashes = done;
            return CP_QPOW_OCL_CANCELLED;
        }
        /* One launch; it must not cross a wrap of the counter word. */
        const u32 w0 = counter_word(cur);
        uint64_t n = g_launch;
        if(n > count - done) n = count - done;
        const uint64_t room = 0x100000000ull - w0;
        if(n > room) n = room;

        if(!have_mid || memcmp(mid_high, cur, 32) != 0){
            memcpy(mid_high, cur, 32);
            qpow::mining_midstate(header, cur, mid);
            have_mid = true;
        }
        u64 pre[12];
        make_pre(mid, cur, pre);
        u32 out[16];
        const auto ts = std::chrono::steady_clock::now();
        if(!run_scan(pre, t0, w0, (u32)n, g_local, nullptr, out)){
            *out_hashes = done;
            return CP_QPOW_OCL_ERROR;
        }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ts).count();
        if(!g_batch_req && n == g_launch && ms > 0){
            double next = (double)g_launch * 100.0 / ms;
            next = next < 65536.0 ? 65536.0 : (next > 1073741824.0 ? 1073741824.0 : next);
            g_launch = (u32)(0.5 * g_launch + 0.5 * next);
        }

        /* Lowest valid nonce of the launch: the caller resumes right after it, so later
         * shares of the same launch are found again on the next call. */
        const u32 nc = out[0] < k_max_candidates ? out[0] : k_max_candidates;
        u32 best = 0xFFFFFFFFu;
        for(u32 k = 0; k < nc; k++){
            const u32 idx = out[1 + k];
            if(idx >= best) continue;
            uint8_t nonce[64], hash[64];
            memcpy(nonce, cur, 64);
            be_add(nonce, idx);
            u64 m[12];
            qpow::mining_midstate(header, nonce, m);
            if(qpow::hash_if_valid(m, nonce + 32, target_be, hash)){
                best = idx;
                memcpy(out_nonce_be, nonce, 64);
                memcpy(out_hash_be, hash, 64);
            }
        }
        be_add(cur, n);
        done += n;
        *out_hashes = done;
        if(best != 0xFFFFFFFFu) return CP_QPOW_OCL_OK_FOUND;
    }

    return CP_QPOW_OCL_OK_EXHAUSTED;
}
