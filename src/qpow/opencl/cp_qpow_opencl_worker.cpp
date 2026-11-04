#include "cp_qpow_opencl_worker.h"

#include "cp_job_ctrl.h"
#include "cp_pool.h"
#include "opencl_context.hpp"
#include "qpow/miner.hpp"
#include "cp_api.h"
#include "qpow/nonce_line.hpp"

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

/* Host side of kernels/qpow_mining.cl. A launch covers consecutive positions t of one nonce
 * line (qpow/nonce_line.hpp): the host folds the midstate, the line base, the first linear layer
 * and ten of the first round's S-boxes into 14 launch constants, and re-hashes every candidate
 * the kernel reports with the reference Poseidon2 before returning it. */

namespace {

typedef uint32_t u32;
typedef uint64_t u64;

OpenClContext g_ctx;
cl_kernel g_kernel = nullptr;
cl_mem g_buf_out = nullptr;   /* [0] = candidate count, [1..15] = nonce index */
cl_mem g_buf_pre = nullptr;   /* nonce_line::kParams x u64 */
cl_mem g_buf_out2 = nullptr;  /* second slot of the search pipeline */
cl_mem g_buf_pre2 = nullptr;
int g_ready = 0;
int g_platform_filter = -1;
u32 g_batch_req = 0;          /* 0 = automatic */
u32 g_launch = 1u << 20;      /* nonces per launch */
size_t g_local = 64;
int g_mul = 0;
int g_red = 1;
int g_npw = 1;   /* nonces per work-item (kernel option -DQV_NPW=N via CP_QPOW_OCL_OPTS) */
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

constexpr int k_params = qpow::nonce_line::kParams;

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
    rel(g_buf_out2);
    rel(g_buf_pre2);
    release_kernel();
}

/* Set once the device compiler rejected the default add-with-overflow carry: Intel UHD 770
 * lowers __builtin_add_overflow on ulong to __spirv_IAddCarry, which its backend cannot build.
 * Every later build then uses the compare form (-DQV_OVF=0). */
bool g_ovf_compare = false;

bool build_variant(int mul, int red)
{
    release_kernel();
    /* CP_QPOW_OCL_OPTS: extra kernel build options (variants without a rebuild) */
    std::string opts = "-DQV_MUL=" + std::to_string(mul) + " -DQV_RED=" + std::to_string(red);
    /* gfx12 inline assembly (QV_RED=3) is only valid on RDNA4. There two nonces per work-item
     * also pay off (R9700: 204.8 -> 211.3 MH/s with the kernel's gfx12 defaults). */
    const bool gfx12 = g_ctx.device_name.compare(0, 5, "gfx12") == 0;
    if(gfx12) opts += " -DQV_GFX12=1";
    if(const char* x = getenv("CP_QPOW_OCL_OPTS")){
        if(x[0]){ opts += " "; opts += x; }
    }
    if(gfx12 && opts.find("QV_NPW=") == std::string::npos) opts += " -DQV_NPW=2";
    g_npw = 1;
    if(const char* p = strstr(opts.c_str(), "QV_NPW=")){
        const int v = atoi(p + 7);
        if(v >= 1 && v <= 4) g_npw = v;
    }
    if(g_ovf_compare) opts += " -DQV_OVF=0";
    if(!g_ctx.safe_build_program_from_file(g_kernel_path.c_str(), opts.c_str())){
        if(g_ovf_compare || opts.find("QV_OVF=") != std::string::npos) return false;
        fprintf(stderr, "[qpow-ocl] retrying the build with the compare-based carry (-DQV_OVF=0)\n");
        opts += " -DQV_OVF=0";
        if(!g_ctx.safe_build_program_from_file(g_kernel_path.c_str(), opts.c_str())) return false;
        g_ovf_compare = true;
    }
    g_kernel = g_ctx.create_kernel("qpow_scan");
    return g_kernel != nullptr;
}

/* Enqueue one scan and wait for it. dump: optional per-nonce output (self-test). */
bool run_scan(const u64 pk[k_params], u64 t0, u32 tb, u32 count, size_t local, cl_mem dump, u32 out[16])
{
    const u32 zero[16] = {};
    if(!g_ctx.write_buffer(g_buf_out, zero, sizeof(zero))) return false;
    if(!g_ctx.write_buffer(g_buf_pre, pk, k_params * sizeof(u64))) return false;
    cl_int err = CL_SUCCESS;
    err |= clSetKernelArg(g_kernel, 0, sizeof(cl_mem), &g_buf_out);
    err |= clSetKernelArg(g_kernel, 1, sizeof(cl_mem), dump ? &dump : nullptr);
    err |= clSetKernelArg(g_kernel, 2, sizeof(cl_mem), &g_buf_pre);
    err |= clSetKernelArg(g_kernel, 3, sizeof(u64), &t0);
    err |= clSetKernelArg(g_kernel, 4, sizeof(u32), &tb);
    err |= clSetKernelArg(g_kernel, 5, sizeof(u32), &count);
    if(err != CL_SUCCESS){
        fprintf(stderr, "[qpow-ocl] set kernel args failed (%d)\n", err);
        return false;
    }
    const size_t items = ((size_t)count + g_npw - 1) / g_npw;
    const size_t global = (items + local - 1) / local * local;
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
    uint8_t header[32], ctr[64];
    for(int i = 0; i < 32; i++) header[i] = (uint8_t)(i * 37 + 11);
    for(int i = 0; i < 64; i++) ctr[i] = (uint8_t)(i * 91 + 5);
    ctr[60] = 0x13;  /* t starts at 0x3000FF0: crosses byte boundaries, stays below 2^26 */
    ctr[61] = 0x00;
    ctr[62] = 0x0F;
    ctr[63] = 0xF0;
    const u32 n = 4096;
    u64 mid[12], pk[k_params];
    qpow::mining_midstate(header, ctr, mid);
    qpow::nonce_line::launch_params(mid, ctr, pk);
    cl_mem dump = g_ctx.alloc_buffer(n * sizeof(u64), CL_MEM_READ_WRITE);
    if(!dump) return false;
    u32 out[16];
    std::vector<u64> got(n);
    bool ok = run_scan(pk, 0, qpow::nonce_line::t_of(ctr), n, local, dump, out) &&
              g_ctx.read_buffer(dump, got.data(), n * sizeof(u64));
    clReleaseMemObject(dump);
    if(!ok) return false;
    /* 128 samples: an unstable clock that corrupts a few hashes in a thousand still shows up.
     * The inputs are fixed, so the fast reduction's rare misses (QV_WRED_FAST, about 2 hashes
     * in a million) either hit these samples on every run or never; they hit none. Any
     * mismatch therefore rejects the variant: tolerating one would only let a slightly
     * broken variant or clock through, and its wrong hashes would silently lose shares. */
    for(u32 i = 0; i < n; i += 32){
        uint8_t c[64], nn[64];
        memcpy(c, ctr, 64);
        be_add(c, i);
        qpow::nonce_line::map(c, nn);
        const u64 ref = cpu_out0(header, nn);
        if(got[i] != ref){
            fprintf(stderr, "[qpow-ocl] self-test mismatch (mul=%d red=%d) at nonce +%u: got %016llx, want %016llx\n",
                    g_mul, g_red, i, (unsigned long long)got[i], (unsigned long long)ref);
            return false;
        }
    }
    return true;
}

/* Hash rate of the current program for one work-group size: a short run sizes the probe to
 * about 50 ms, then the best of three probes counts. */
double probe_rate(size_t local)
{
    u64 pk[k_params] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
    u32 out[16];
    auto timed = [&](u32 n) -> double {
        auto t0 = std::chrono::steady_clock::now();
        if(!run_scan(pk, 0, 0, n, local, nullptr, out)) return 0;
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

/* Nonces per search() call: one launch. */
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
    g_buf_pre = g_ctx.alloc_buffer(k_params * sizeof(u64), CL_MEM_READ_ONLY);
    g_buf_out2 = g_ctx.alloc_buffer(16 * sizeof(u32), CL_MEM_READ_WRITE);
    g_buf_pre2 = g_ctx.alloc_buffer(k_params * sizeof(u64), CL_MEM_READ_ONLY);
    if(!g_buf_out || !g_buf_pre || !g_buf_out2 || !g_buf_pre2){
        fprintf(stderr, "[qpow-ocl] buffer alloc failed\n");
        release_buffers();
        return -1;
    }

    /* Pick the kernel variant and work-group size that run fastest on this device; the best
     * differs by vendor. First the 64x64 product (NVIDIA: 64-bit mul_hi; Intel/AMD: 32x32+64
     * multiply-add chain) at work-group sizes 64 and 256, then the reduction for that product
     * (Intel: the signed form; AMD RDNA4: the inline-assembly form 3, which builds as form 1
     * elsewhere). CP_QPOW_OCL_MUL=1|2|3 and CP_QPOW_OCL_RED=1|2|3 force one.
     * Every candidate must pass the self-test. */
    std::vector<int> muls = {3, 1};
    if(const char* e = getenv("CP_QPOW_OCL_MUL")){
        const int m = atoi(e);
        if(m >= 1 && m <= 3) muls = {m};
    }
    std::vector<int> reds = {2, 3};
    int red_forced = 0;
    if(const char* e = getenv("CP_QPOW_OCL_RED")){
        const int r = atoi(e);
        if(r >= 1 && r <= 3){ red_forced = r; reds.clear(); }
    }
    const size_t locals[] = {64, 256};
    int best_mul = 0, best_red = red_forced ? red_forced : 1;
    size_t best_local = 64;
    double best_rate = 0;
    auto try_variant = [&](int m, int r, const size_t* ls, int nls) {
        g_mul = m;
        g_red = r;
        if(!build_variant(m, r)){
            fprintf(stderr, "[qpow-ocl] kernel build failed (%s, mul=%d red=%d)\n", g_kernel_path.c_str(), m, r);
            return;
        }
        size_t max_wg = 0;
        clGetKernelWorkGroupInfo(g_kernel, g_ctx.device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(max_wg), &max_wg, nullptr);
        if(!self_test(max_wg >= 64 ? 64 : max_wg)) return;
        for(int i = 0; i < nls; i++){
            const size_t l = ls[i];
            if(l > max_wg) continue;
            const double rate = probe_rate(l);
            printf("[qpow-ocl] probe mul=%d red=%d local=%zu: %.2f MH/s\n", m, r, l, rate / 1e6);
            if(rate > best_rate){
                best_rate = rate;
                best_mul = m;
                best_red = r;
                best_local = l;
            }
        }
    };
    for(int m : muls) try_variant(m, best_red, locals, 2);
    if(best_mul){
        const size_t l = best_local;
        for(int r : reds) try_variant(best_mul, r, &l, 1);
    }
    if(!best_mul){
        fprintf(stderr, "[qpow-ocl] no kernel variant passed the self-test, not mining on this device\n");
        release_buffers();
        return -1;
    }
    g_local = best_local;
    if((g_mul != best_mul || g_red != best_red) && !build_variant(best_mul, best_red)){
        fprintf(stderr, "[qpow-ocl] kernel rebuild failed\n");
        release_buffers();
        return -1;
    }
    g_mul = best_mul;
    g_red = best_red;
    /* Start near 100 ms per launch; search() keeps it there unless --batch-size is set. */
    g_launch = g_batch_req ? g_batch_req : (u32)(best_rate * 0.1 > (1u << 16) ? best_rate * 0.1 : (1u << 16));

    g_ready = 1;
    cp_api_add_device(g_ctx.device_name.c_str(), OpenClContext::pci_bus_id(g_ctx.device).c_str());
    printf("[qpow-ocl] device[%d]: %s (%s) CUs=%u kernel mul=%d red=%d local=%zu batch=%s self-test ok\n",
           g_ctx.device_flat_index, g_ctx.device_name.c_str(),
           g_ctx.discrete_gpu ? "discrete" : "integrated",
           (unsigned)cu, g_mul, g_red, g_local, g_batch_req ? "fixed" : "auto");
    fflush(stdout);
    return 0;
}

namespace {
void discard();
} // namespace

extern "C" void cp_qpow_opencl_worker_shutdown(void)
{
    discard();
    if(g_ctx.queue) clFinish(g_ctx.queue); /* launches still queued use the buffers */
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

namespace {

/* One in-flight launch of the search pipeline. */
struct Slot {
    cl_mem out = nullptr, pre_buf = nullptr;
    uint8_t base[64];
    u64 pk[k_params];
    u32 out_host[16];
    u32 n = 0;
    cl_event done_ev = nullptr;
};

const u32 k_zero16[16] = {};

/* Enqueue a launch without waiting: constants upload, scan, and the read-back of the candidates. */
bool enqueue_slot(Slot& sl, u64 t0, size_t local)
{
    cl_int err = clEnqueueWriteBuffer(g_ctx.queue, sl.out, CL_FALSE, 0, sizeof(k_zero16), k_zero16, 0, nullptr, nullptr);
    err |= clEnqueueWriteBuffer(g_ctx.queue, sl.pre_buf, CL_FALSE, 0, sizeof(sl.pk), sl.pk, 0, nullptr, nullptr);
    const u32 tb = qpow::nonce_line::t_of(sl.base);
    err |= clSetKernelArg(g_kernel, 0, sizeof(cl_mem), &sl.out);
    err |= clSetKernelArg(g_kernel, 1, sizeof(cl_mem), nullptr);
    err |= clSetKernelArg(g_kernel, 2, sizeof(cl_mem), &sl.pre_buf);
    err |= clSetKernelArg(g_kernel, 3, sizeof(u64), &t0);
    err |= clSetKernelArg(g_kernel, 4, sizeof(u32), &tb);
    err |= clSetKernelArg(g_kernel, 5, sizeof(u32), &sl.n);
    const size_t items = ((size_t)sl.n + g_npw - 1) / g_npw;
    const size_t global = (items + local - 1) / local * local;
    if(err == CL_SUCCESS)
        err = clEnqueueNDRangeKernel(g_ctx.queue, g_kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr);
    if(err == CL_SUCCESS)
        err = clEnqueueReadBuffer(g_ctx.queue, sl.out, CL_FALSE, 0, sizeof(sl.out_host), sl.out_host, 0, nullptr, &sl.done_ev);
    if(err != CL_SUCCESS){
        fprintf(stderr, "[qpow-ocl] enqueue failed (%d)\n", err);
        return false;
    }
    clFlush(g_ctx.queue);
    return true;
}

/* The search pipeline outlives search(): two launches stay queued between calls, and the next
 * call collects the older one if it continues the same search. */
struct Pipe {
    bool active = false;
    int head = 0;            /* slot of the older launch */
    Slot slot[2];
    uint8_t header[32];
    uint8_t target[64];
    u64 t0 = 0;
    uint8_t expect[64];      /* start_be that continues the search: base of the older launch */
    uint8_t next[64];        /* counter the next queued launch starts at */
    uint8_t mid_high[32];    /* midstate cache for the launch parameters */
    u64 mid[12];
    bool have_mid = false;
    std::chrono::steady_clock::time_point t_win; /* rate window of the automatic launch size */
    uint64_t measured = 0;
};
Pipe g_pipe;

/* Plan the next launch at g_pipe.next into slot si and queue it. A launch must not cross a wrap
 * of the line position t (k stays fixed). Returns false on an enqueue error. */
bool enqueue_launch(int si)
{
    Pipe& q = g_pipe;
    Slot& sl = q.slot[si];
    sl.out = si ? g_buf_out2 : g_buf_out;
    sl.pre_buf = si ? g_buf_pre2 : g_buf_pre;
    const uint64_t room = qpow::nonce_line::kTSpan - qpow::nonce_line::t_of(q.next);
    const uint64_t n = g_launch < room ? g_launch : room;
    memcpy(sl.base, q.next, 64);
    if(!q.have_mid || memcmp(q.mid_high, q.next, 32) != 0){
        memcpy(q.mid_high, q.next, 32);
        qpow::mining_midstate(q.header, q.next, q.mid);
        q.have_mid = true;
    }
    qpow::nonce_line::launch_params(q.mid, q.next, sl.pk);
    sl.n = (u32)n;
    be_add(q.next, n);
    return enqueue_slot(sl, q.t0, g_local);
}

/* Forget the queued launches. No wait: the in-order queue runs the next launches after them. */
void discard()
{
    for(Slot& sl : g_pipe.slot){
        if(sl.done_ev){ clReleaseEvent(sl.done_ev); sl.done_ev = nullptr; }
        sl.n = 0;
    }
    g_pipe.active = false;
}

} // namespace

extern "C" int cp_qpow_opencl_worker_search(
    const uint8_t header[32],
    const uint8_t target_be[64],
    const uint8_t start_be[64],
    uint8_t out_nonces_be[][64],
    int max_shares,
    int* out_nshares,
    uint64_t* out_hashes)
{
    if(!g_ready || !out_nshares || !out_hashes) return CP_QPOW_OCL_ERROR;
    *out_nshares = 0;
    *out_hashes = 0;
    Pipe& q = g_pipe;
    if(cancelled()){
        discard();
        return CP_QPOW_OCL_CANCELLED;
    }
    /* The queued launches belong to the call that continues them: same job and target, starting
     * where the last call stopped. Anything else (a new job, a fee switch) starts over. */
    if(q.active && (memcmp(q.header, header, 32) != 0 || memcmp(q.target, target_be, 64) != 0 ||
                    memcmp(q.expect, start_be, 64) != 0))
        discard();
    if(!q.active){
        memcpy(q.header, header, 32);
        memcpy(q.target, target_be, 64);
        q.t0 = 0;
        for(int i = 0; i < 8; i++) q.t0 = (q.t0 << 8) | target_be[i];
        memcpy(q.expect, start_be, 64);
        memcpy(q.next, start_be, 64);
        q.have_mid = false;
        q.head = 0;
        q.active = true;
        q.t_win = std::chrono::steady_clock::now();
        q.measured = 0;
        /* Two launches in flight: the next one is queued before the current one's candidates
         * are checked on the host, so the device does not idle between launches. */
        if(!enqueue_launch(0) || !enqueue_launch(1)){
            discard();
            return CP_QPOW_OCL_ERROR;
        }
    }

    /* Collect the older launch. */
    const int si = q.head;
    Slot& sl = q.slot[si];
    if(clWaitForEvents(1, &sl.done_ev) != CL_SUCCESS){
        discard();
        return CP_QPOW_OCL_ERROR;
    }
    clReleaseEvent(sl.done_ev);
    sl.done_ev = nullptr;
    /* Automatic launch size: ~100 ms of work, from the rate over a window of about a second.
     * The time between two completions is no measure: some drivers (Intel) report both slots of
     * the pipeline done within a millisecond, which made the size jump a hundredfold. */
    q.measured += sl.n;
    const auto now = std::chrono::steady_clock::now();
    const double s = std::chrono::duration<double>(now - q.t_win).count();
    if(!g_batch_req && s > 0.05){
        double next = (double)q.measured / s * 0.1;
        next = next < 65536.0 ? 65536.0 : (next > 1073741824.0 ? 1073741824.0 : next);
        g_launch = (u32)(0.5 * g_launch + 0.5 * next);
    }
    if(s > 1.0){
        q.t_win = now;
        q.measured = 0;
    }

    /* Every valid candidate of the launch is a share to report. */
    const u32 nc = sl.out_host[0] < k_max_candidates ? sl.out_host[0] : k_max_candidates;
    for(u32 k = 0; k < nc && *out_nshares < max_shares; k++){
        uint8_t c[64], nonce[64], hash[64];
        memcpy(c, sl.base, 64);
        be_add(c, sl.out_host[1 + k]);
        qpow::nonce_line::map(c, nonce);
        u64 m[12];
        qpow::mining_midstate(header, nonce, m);
        if(qpow::hash_if_valid(m, nonce + 32, target_be, hash))
            memcpy(out_nonces_be[(*out_nshares)++], nonce, 64);
    }
    *out_hashes = sl.n;
    be_add(q.expect, sl.n);
    q.head ^= 1;
    /* Refill the slot, so two launches stay queued until the next call. If that fails, this
     * launch still counts; the next call starts over and reports the error. */
    if(!enqueue_launch(si)) discard();
    return *out_nshares ? CP_QPOW_OCL_OK_FOUND : CP_QPOW_OCL_OK_EXHAUSTED;
}
