#include "cp_opencl_prep_profile.h"

#include "case32_layout.hpp"
#include "case33_ocl_prep.hpp"
#include "cp_config.h"
#include "cp_noise.h"
#include "cp_opencl_worker.h"
#include "cp_util.h"
#include "opencl_context.hpp"

#include <cstdio>
#include <cstring>

namespace {

struct PrepTimes {
    double random_a = 0.0;
    double random_b = 0.0;
    double hash_a = 0.0;
    double hash_b = 0.0;
    double noise_seeds = 0.0;
    double fusion_b = 0.0;
    double fusion_a = 0.0;
    double total = 0.0;
};

struct ZeroBAttemptTimes {
    double random_a = 0.0;
    double hash_a = 0.0;
    double a_key = 0.0;
    double fusion_a = 0.0;
    double total = 0.0;
};

void add_prep(PrepTimes *acc, const PrepTimes &t) {
    acc->random_a += t.random_a;
    acc->random_b += t.random_b;
    acc->hash_a += t.hash_a;
    acc->hash_b += t.hash_b;
    acc->noise_seeds += t.noise_seeds;
    acc->fusion_b += t.fusion_b;
    acc->fusion_a += t.fusion_a;
    acc->total += t.total;
}

void scale_prep(PrepTimes *t, double inv) {
    t->random_a *= inv;
    t->random_b *= inv;
    t->hash_a *= inv;
    t->hash_b *= inv;
    t->noise_seeds *= inv;
    t->fusion_b *= inv;
    t->fusion_a *= inv;
    t->total *= inv;
}

void add_attempt(ZeroBAttemptTimes *acc, const ZeroBAttemptTimes &t) {
    acc->random_a += t.random_a;
    acc->hash_a += t.hash_a;
    acc->a_key += t.a_key;
    acc->fusion_a += t.fusion_a;
    acc->total += t.total;
}

void scale_attempt(ZeroBAttemptTimes *t, double inv) {
    t->random_a *= inv;
    t->hash_a *= inv;
    t->a_key *= inv;
    t->fusion_a *= inv;
    t->total *= inv;
}

uint64_t pearl_seed_to_u64(const uint8_t *seed, int seed_len) {
    uint64_t s = 0;
    for (int i = 0; i < seed_len; i++) {
        s ^= static_cast<uint64_t>(seed[i]) << ((i & 7) * 8);
    }
    return s;
}

bool run_full_prep(Case33OclPrep *prep, cl_mem d_b_sig, cl_mem d_b_pre, cl_mem d_a_pre,
                   const uint8_t job_key[32], int m, int n, int blocks_k, int macro_rows,
                   int macro_cols, uint64_t rng_seed, PrepTimes *out) {
    const size_t szAp = static_cast<size_t>(m) * static_cast<size_t>(K_DIM);
    const size_t szBpT = static_cast<size_t>(n) * static_cast<size_t>(K_DIM);
    const size_t pad_a = (szAp + 1023) / 1024 * 1024;
    const size_t pad_b = (szBpT + 1023) / 1024 * 1024;

    uint8_t hash_a[32], hash_b[32], b_seed[32], a_key[32];
    PrepTimes t{};
    double t0 = cp_now_sec();

    if (!prep->align_gen_random(prep->signal_A_buffer(), rng_seed, 0, static_cast<int>(szAp))) {
        return false;
    }
    t.random_a = cp_now_sec() - t0;

    t0 = cp_now_sec();
    if (!prep->align_gen_random(d_b_sig, rng_seed, 1, static_cast<int>(szBpT))) {
        return false;
    }
    t.random_b = cp_now_sec() - t0;

    t0 = cp_now_sec();
    if (!prep->align_keyed_hash(prep->signal_A_buffer(), szAp, pad_a, job_key, hash_a)) {
        return false;
    }
    t.hash_a = cp_now_sec() - t0;

    t0 = cp_now_sec();
    if (!prep->align_keyed_hash(d_b_sig, szBpT, pad_b, job_key, hash_b)) {
        return false;
    }
    t.hash_b = cp_now_sec() - t0;

    t0 = cp_now_sec();
    pearl_derive_noise_seeds(job_key, hash_a, hash_b, static_cast<uint32_t>(m),
                             static_cast<uint32_t>(n), 0, b_seed, a_key);
    t.noise_seeds = cp_now_sec() - t0;

    t0 = cp_now_sec();
    if (!prep->prepare_job_b(d_b_pre, b_seed, n, K_DIM, blocks_k, macro_cols)) {
        return false;
    }
    t.fusion_b = cp_now_sec() - t0;

    if (!prep->write_noise_seed(a_key)) {
        return false;
    }

    t0 = cp_now_sec();
    if (!prep->fused_prepack_a(d_a_pre, m, K_DIM, blocks_k, macro_rows)) {
        return false;
    }
    t.fusion_a = cp_now_sec() - t0;

    t.total = t.random_a + t.random_b + t.hash_a + t.hash_b + t.noise_seeds + t.fusion_b +
              t.fusion_a;
    *out = t;
    return true;
}

bool run_zero_b_attempt(Case33OclPrep *prep, cl_mem d_a_pre, const uint8_t b_noise_seed[32],
                        const uint8_t job_key[32], int m, int blocks_k, int macro_rows,
                        uint64_t rng_seed, ZeroBAttemptTimes *out) {
    const size_t szAp = static_cast<size_t>(m) * static_cast<size_t>(K_DIM);
    const size_t pad_a = (szAp + 1023) / 1024 * 1024;

    uint8_t hash_a[32], a_key[32];
    ZeroBAttemptTimes t{};
    double t0 = cp_now_sec();

    if (!prep->align_gen_random(prep->signal_A_buffer(), rng_seed, 0, static_cast<int>(szAp))) {
        return false;
    }
    t.random_a = cp_now_sec() - t0;

    t0 = cp_now_sec();
    if (!prep->align_keyed_hash(prep->signal_A_buffer(), szAp, pad_a, job_key, hash_a)) {
        return false;
    }
    t.hash_a = cp_now_sec() - t0;

    t0 = cp_now_sec();
    pearl_a_noise_seed_from_hash(b_noise_seed, hash_a, static_cast<uint32_t>(m), 0, a_key);
    if (!prep->write_noise_seed(a_key)) {
        return false;
    }
    t.a_key = cp_now_sec() - t0;

    t0 = cp_now_sec();
    if (!prep->fused_prepack_a(d_a_pre, m, K_DIM, blocks_k, macro_rows)) {
        return false;
    }
    t.fusion_a = cp_now_sec() - t0;

    t.total = t.random_a + t.hash_a + t.a_key + t.fusion_a;
    *out = t;
    return true;
}

} /* namespace */

extern "C" int cp_opencl_run_prep_profile(int device_index, int m, int n, int warmup, int runs) {
    if (m <= 0 || n <= 0 || runs < 1) {
        return -1;
    }
    if (warmup < 0) {
        warmup = 0;
    }

    cp_opencl_configure_tile_for_worker(device_index);

    const int blocks_k = K_DIM / case32::kKR;
    const int macro_rows = m / case32::kMacroM;
    const int macro_cols = n / case32::kMacroN;
    const size_t szAp = static_cast<size_t>(m) * static_cast<size_t>(K_DIM);
    const size_t szBpT = static_cast<size_t>(n) * static_cast<size_t>(K_DIM);
    const size_t b_pre_bytes = static_cast<size_t>(macro_cols) * static_cast<size_t>(blocks_k) *
                               static_cast<size_t>(case32::kMacroKbBlockB);
    const size_t a_pre_bytes = static_cast<size_t>(macro_rows) * static_cast<size_t>(blocks_k) *
                               static_cast<size_t>(case32::kMacroKbBlockA);

    uint8_t job_key[32];
    uint8_t ab_seed[8];
    for (int i = 0; i < 32; i++) {
        job_key[i] = static_cast<uint8_t>((i * 11 + 7) & 0xff);
    }
    ab_seed[0] = 0x90;
    ab_seed[1] = 0x78;
    ab_seed[2] = 0x56;
    ab_seed[3] = 0x34;
    ab_seed[4] = 0x12;
    ab_seed[5] = 0xee;
    ab_seed[6] = 0xff;
    ab_seed[7] = 0xc0;

    OpenClContext ocl;
    Case33OclPrep prep;
    cl_mem d_b_sig = nullptr;
    cl_mem d_b_pre = nullptr;
    cl_mem d_a_pre = nullptr;

    if (!ocl.init(device_index)) {
        fprintf(stderr, "[ocl-prep] OpenCL init failed\n");
        return -1;
    }

    printf("[ocl-prep] m=%d n=%d k=%d (device %d: %s)\n", m, n, K_DIM, device_index,
           ocl.device_name.c_str());
    printf("[ocl-prep] matrices A %.1f MiB  B %.1f MiB  prepack %.1f MiB\n",
           static_cast<double>(szAp) / (1024.0 * 1024.0),
           static_cast<double>(szBpT) / (1024.0 * 1024.0),
           static_cast<double>(a_pre_bytes + b_pre_bytes) / (1024.0 * 1024.0));
    printf("[ocl-prep] warmup=%d timed=%d\n", warmup, runs);
    fflush(stdout);

    if (!prep.init(&ocl, cp_ocl_kernel_dir()) || !prep.ensure_buffers(m, n, K_DIM)) {
        fprintf(stderr, "[ocl-prep] prep init failed\n");
        return -1;
    }

    d_b_sig = ocl.alloc_buffer(szBpT, CL_MEM_READ_WRITE);
    d_b_pre = ocl.alloc_buffer(b_pre_bytes, CL_MEM_READ_WRITE);
    d_a_pre = ocl.alloc_buffer(a_pre_bytes, CL_MEM_READ_WRITE);
    if (!d_b_sig || !d_b_pre || !d_a_pre) {
        fprintf(stderr, "[ocl-prep] OOM device buffers\n");
        return -1;
    }

    const uint64_t rng_base = pearl_seed_to_u64(ab_seed, static_cast<int>(sizeof(ab_seed)));

    PrepTimes full_sum{};
    for (int i = 0; i < warmup + runs; i++) {
        const uint64_t rng_seed = rng_base ^ (static_cast<uint64_t>(i) * 0x9E3779B97F4A7C15ULL);
        PrepTimes t{};
        if (!run_full_prep(&prep, d_b_sig, d_b_pre, d_a_pre, job_key, m, n, blocks_k, macro_rows,
                           macro_cols, rng_seed, &t)) {
            fprintf(stderr, "[ocl-prep] full pipeline failed on iter %d\n", i);
            return -1;
        }
        if (i >= warmup) {
            add_prep(&full_sum, t);
        }
    }
    scale_prep(&full_sum, 1.0 / static_cast<double>(runs));

    printf("[ocl-prep] full pipeline (random A/B + keyed hash + noise fusion):\n");
    printf("[ocl-prep]   random A        %.3fs\n", full_sum.random_a);
    printf("[ocl-prep]   random B        %.3fs\n", full_sum.random_b);
    printf("[ocl-prep]   keyed hash A    %.3fs\n", full_sum.hash_a);
    printf("[ocl-prep]   keyed hash B    %.3fs\n", full_sum.hash_b);
    printf("[ocl-prep]   noise seeds     %.3fs\n", full_sum.noise_seeds);
    printf("[ocl-prep]   fusion B        %.3fs\n", full_sum.fusion_b);
    printf("[ocl-prep]   fusion A        %.3fs\n", full_sum.fusion_a);
    printf("[ocl-prep]   total           %.3fs\n", full_sum.total);
    fflush(stdout);

    uint8_t b_seed[32];
    double t0 = cp_now_sec();
    pearl_b_noise_seed_from_bt(job_key, nullptr, n, K_DIM, 0, b_seed);
    const double b_seed_cpu = cp_now_sec() - t0;

    t0 = cp_now_sec();
    if (!prep.prepare_job_b(d_b_pre, b_seed, n, K_DIM, blocks_k, macro_cols)) {
        fprintf(stderr, "[ocl-prep] zero-B job prep failed\n");
        return -1;
    }
    const double job_b_fusion = cp_now_sec() - t0;

    ZeroBAttemptTimes attempt_sum{};
    for (int i = 0; i < warmup + runs; i++) {
        const uint64_t rng_seed = rng_base ^ (static_cast<uint64_t>(i + 1000) * 0x9E3779B97F4A7C15ULL);
        ZeroBAttemptTimes t{};
        if (!run_zero_b_attempt(&prep, d_a_pre, b_seed, job_key, m, blocks_k, macro_rows, rng_seed,
                                &t)) {
            fprintf(stderr, "[ocl-prep] zero-B attempt failed on iter %d\n", i);
            return -1;
        }
        if (i >= warmup) {
            add_attempt(&attempt_sum, t);
        }
    }
    scale_attempt(&attempt_sum, 1.0 / static_cast<double>(runs));

    printf("[ocl-prep] zero-B mining path (production default):\n");
    printf("[ocl-prep]   job: b_seed CPU      %.3fs (once per job)\n", b_seed_cpu);
    printf("[ocl-prep]   job: B fusion         %.3fs (once per job)\n", job_b_fusion);
    printf("[ocl-prep]   attempt: random A    %.3fs (per nonce)\n", attempt_sum.random_a);
    printf("[ocl-prep]   attempt: hash A      %.3fs\n", attempt_sum.hash_a);
    printf("[ocl-prep]   attempt: a_key CPU   %.3fs\n", attempt_sum.a_key);
    printf("[ocl-prep]   attempt: fusion A     %.3fs\n", attempt_sum.fusion_a);
    printf("[ocl-prep]   attempt total         %.3fs (per nonce)\n", attempt_sum.total);
    fflush(stdout);

    return 0;
}
