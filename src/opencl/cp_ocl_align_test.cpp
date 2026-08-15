#include "cp_opencl_align.h"

#include "case32_layout.hpp"
#include "case32_prepack.hpp"
#include "case33_ocl_prep.hpp"
#include "cp_config.h"
#include "cp_noise.h"
#include "cp_opencl_worker.h"
#include "cp_util.h"
#include "opencl_context.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

int compare_digest(const char *label, const uint8_t a[32], const uint8_t b[32]) {
    if (memcmp(a, b, 32) == 0) {
        return 0;
    }
    fprintf(stderr, "[align-test-prod] %s mismatch\n", label);
    fprintf(stderr, "  gpu: ");
    for (int i = 0; i < 32; i++) {
        fprintf(stderr, "%02x", a[i]);
    }
    fprintf(stderr, "\n  cpu: ");
    for (int i = 0; i < 32; i++) {
        fprintf(stderr, "%02x", b[i]);
    }
    fprintf(stderr, "\n");
    return -1;
}

int compare_prepack_samples(const char *label, const int8_t *gpu, const int8_t *cpu, size_t bytes,
                            int macro_rows, int macro_cols, int blocks_k, bool is_a) {
    static const int sample_macro[] = {0, 1, 17, 64, 256, 512};
    const int macro_count = is_a ? macro_rows : macro_cols;
    const size_t macro_kb_block = is_a ? static_cast<size_t>(case32::kMacroKbBlockA)
                                         : static_cast<size_t>(case32::kMacroKbBlockB);

    for (size_t si = 0; si < sizeof(sample_macro) / sizeof(sample_macro[0]); si++) {
        const int im = sample_macro[si];
        if (im >= macro_count) {
            continue;
        }
        const int kb_step = blocks_k > 1 ? blocks_k - 1 : 1;
        for (int kb = 0; kb < blocks_k; kb += kb_step) {
            const size_t off = (static_cast<size_t>(im) * static_cast<size_t>(blocks_k) +
                                static_cast<size_t>(kb)) *
                               macro_kb_block;
            if (off + macro_kb_block > bytes) {
                fprintf(stderr, "[align-test-prod] %s sample offset out of range\n", label);
                return -1;
            }
            if (memcmp(gpu + off, cpu + off, macro_kb_block) != 0) {
                size_t diff_at = 0;
                for (size_t b = 0; b < macro_kb_block; b++) {
                    if (gpu[off + b] != cpu[off + b]) {
                        diff_at = b;
                        break;
                    }
                }
                fprintf(stderr,
                        "[align-test-prod] %s prepack mismatch at macro=%d kb=%d byte +%zu (gpu=%d cpu=%d)\n",
                        label, im, kb, diff_at, (int)gpu[off + diff_at], (int)cpu[off + diff_at]);
                return -1;
            }
        }
    }
    return 0;
}

} /* namespace */

extern "C" int cp_opencl_run_alignment_tests(int device_index, int m, int n) {
    cp_opencl_configure_tile_for_worker(device_index);

    const size_t szAp = static_cast<size_t>(m) * static_cast<size_t>(K_DIM);
    const size_t szBpT = static_cast<size_t>(n) * static_cast<size_t>(K_DIM);
    const size_t pad_a = (szAp + 1023) / 1024 * 1024;
    const size_t pad_b = (szBpT + 1023) / 1024 * 1024;
    const int blocks_k = K_DIM / case32::kKR;
    const int macro_rows = m / case32::kMacroM;
    const int macro_cols = n / case32::kMacroN;
    const uint64_t rng_seed = 0xC0FFEE1234567890ULL;
    const size_t a_pre_bytes = static_cast<size_t>(macro_rows) * static_cast<size_t>(blocks_k) *
                               static_cast<size_t>(case32::kMacroKbBlockA);
    const size_t b_pre_bytes = static_cast<size_t>(macro_cols) * static_cast<size_t>(blocks_k) *
                               static_cast<size_t>(case32::kMacroKbBlockB);

    uint8_t job_key[32];
    uint8_t hash_a_gpu[32], hash_b_gpu[32];
    uint8_t hash_a_cpu[32], hash_b_cpu[32];
    uint8_t b_seed_gpu[32], a_key_gpu[32];
    uint8_t b_seed_cpu[32], a_key_cpu[32];
    uint8_t ab_seed[8];

    int8_t *h_A = nullptr;
    int8_t *h_Bt = nullptr;
    int rc = -1;
    double t0 = 0.0;

    OpenClContext ocl;
    Case33OclPrep prep;
    cl_mem d_Bt = nullptr;
    cl_mem d_a_pre = nullptr;
    cl_mem d_b_pre = nullptr;

    for (int i = 0; i < 32; i++) {
        job_key[i] = static_cast<uint8_t>((i * 11 + 7) & 0xff);
    }
    /* Little-endian encoding of rng_seed for pearl_seed_to_u64 (must match align_gen_random). */
    ab_seed[0] = 0x90;
    ab_seed[1] = 0x78;
    ab_seed[2] = 0x56;
    ab_seed[3] = 0x34;
    ab_seed[4] = 0x12;
    ab_seed[5] = 0xee;
    ab_seed[6] = 0xff;
    ab_seed[7] = 0xc0;

    printf("[align-test-prod] OpenCL vs CPU m=%d n=%d k=%d (device %d)\n", m, n, K_DIM,
           device_index);
    fflush(stdout);

    if (!ocl.init(device_index)) {
        fprintf(stderr, "[align-test-prod] OpenCL init failed\n");
        goto done;
    }
    printf("[align-test-prod] device: %s\n", ocl.device_name.c_str());
    fflush(stdout);

    if (!prep.init(&ocl, cp_ocl_kernel_dir())) {
        fprintf(stderr, "[align-test-prod] prep init failed\n");
        goto done;
    }
    if (!prep.ensure_buffers(m, n, K_DIM)) {
        fprintf(stderr, "[align-test-prod] ensure_buffers failed\n");
        goto done;
    }

    d_Bt = ocl.alloc_buffer(szBpT, CL_MEM_READ_WRITE);
    d_a_pre = ocl.alloc_buffer(a_pre_bytes, CL_MEM_READ_WRITE);
    d_b_pre = ocl.alloc_buffer(b_pre_bytes, CL_MEM_READ_WRITE);
    if (!d_Bt || !d_a_pre || !d_b_pre) {
        fprintf(stderr, "[align-test-prod] OOM device buffers\n");
        goto done;
    }

    t0 = cp_now_sec();
    if (!prep.align_gen_random(prep.signal_A_buffer(), rng_seed, 0, static_cast<int>(szAp)) ||
        !prep.align_gen_random(d_Bt, rng_seed, 1, static_cast<int>(szBpT))) {
        fprintf(stderr, "[align-test-prod] random A,B gen failed\n");
        goto done;
    }
    printf("[align-test-prod] random A,B gen %.1fs\n", cp_now_sec() - t0);
    fflush(stdout);

    t0 = cp_now_sec();
    if (!prep.align_keyed_hash(prep.signal_A_buffer(), szAp, pad_a, job_key, hash_a_gpu) ||
        !prep.align_keyed_hash(d_Bt, szBpT, pad_b, job_key, hash_b_gpu)) {
        fprintf(stderr, "[align-test-prod] GPU keyed hash failed\n");
        goto done;
    }
    printf("[align-test-prod] GPU keyed hash %.1fs\n", cp_now_sec() - t0);
    fflush(stdout);

    h_A = static_cast<int8_t *>(malloc(szAp));
    h_Bt = static_cast<int8_t *>(malloc(szBpT));
    if (!h_A || !h_Bt) {
        fprintf(stderr, "[align-test-prod] OOM host matrix buffers\n");
        goto done;
    }

    t0 = cp_now_sec();
    printf("[align-test-prod] D2H A (%.1f MiB)...\n", static_cast<double>(szAp) / (1024.0 * 1024.0));
    fflush(stdout);
    if (!prep.read_A_sig(h_A, szAp) || !ocl.read_buffer(d_Bt, h_Bt, szBpT)) {
        fprintf(stderr, "[align-test-prod] D2H failed\n");
        goto done;
    }
    printf("[align-test-prod] D2H done %.1fs\n", cp_now_sec() - t0);
    fflush(stdout);

    t0 = cp_now_sec();
    pearl_keyed_digest_int8(h_A, szAp, job_key, hash_a_cpu);
    pearl_keyed_digest_int8(h_Bt, szBpT, job_key, hash_b_cpu);
    printf("[align-test-prod] CPU keyed digest %.1fs\n", cp_now_sec() - t0);
    fflush(stdout);

    if (compare_digest("hash_a", hash_a_gpu, hash_a_cpu) != 0) {
        goto done;
    }
    if (compare_digest("hash_b", hash_b_gpu, hash_b_cpu) != 0) {
        goto done;
    }
    printf("[align-test-prod] GPU/CPU matrix hash OK\n");
    fflush(stdout);

    pearl_derive_noise_seeds(job_key, hash_a_gpu, hash_b_gpu, static_cast<uint32_t>(m),
                             static_cast<uint32_t>(n), 0, b_seed_gpu, a_key_gpu);
    pearl_commitment_seeds(job_key, h_A, h_Bt, m, n, K_DIM, 0, b_seed_cpu, a_key_cpu);
    if (compare_digest("b_noise_seed", b_seed_gpu, b_seed_cpu) != 0) {
        goto done;
    }
    if (compare_digest("a_noise_seed", a_key_gpu, a_key_cpu) != 0) {
        goto done;
    }
    printf("[align-test-prod] noise seeds OK\n");
    fflush(stdout);

    {
        uint8_t gpu_digest[32], cpu_digest[32];
        if (!prep.align_run_get_random_hash(0, a_key_gpu, 0, 1, gpu_digest)) {
            fprintf(stderr, "[align-test-prod] GPU get_random_hash launch failed\n");
            goto done;
        }
        pearl_get_random_hash(0, PEARL_SEED_LABEL_A, a_key_gpu, 1, cpu_digest);
        if (memcmp(gpu_digest, cpu_digest, 32) != 0) {
            fprintf(stderr, "[align-test-prod] GPU get_random_hash(0) mismatch\n");
            compare_digest("perm_hash0", gpu_digest, cpu_digest);
            goto done;
        }
        printf("[align-test-prod] GPU get_random_hash spot check OK\n");
        fflush(stdout);
    }

    if (!prep.align_build_perm_pairs(0, a_key_gpu, K_DIM)) {
        fprintf(stderr, "[align-test-prod] GPU perm pairs build failed\n");
        goto done;
    }
    {
        const size_t pair_words = static_cast<size_t>(K_DIM) * 2u;
        std::vector<uint32_t> gpu_pairs(pair_words);
        std::vector<uint32_t> cpu_pairs(pair_words);
        if (!prep.align_read_perm_pairs(gpu_pairs.data(), pair_words)) {
            fprintf(stderr, "[align-test-prod] read perm pairs failed\n");
            goto done;
        }
        pearl_build_perm_pairs_a(a_key_gpu, K_DIM, R_RANK, cpu_pairs.data());
        if (memcmp(gpu_pairs.data(), cpu_pairs.data(), pair_words * sizeof(uint32_t)) != 0) {
            for (size_t wi = 0; wi < pair_words; wi++) {
                if (gpu_pairs[wi] != cpu_pairs[wi]) {
                    fprintf(stderr,
                            "[align-test-prod] perm pairs A mismatch at word %zu (col %zu)\n", wi,
                            wi / 2);
                    break;
                }
            }
            goto done;
        }
    }
    printf("[align-test-prod] perm pairs A OK\n");
    fflush(stdout);

    if (!prep.align_build_perm_pairs(1, b_seed_gpu, K_DIM)) {
        fprintf(stderr, "[align-test-prod] GPU perm pairs B build failed\n");
        goto done;
    }
    {
        const size_t pair_words = static_cast<size_t>(K_DIM) * 2u;
        std::vector<uint32_t> gpu_pairs_b(pair_words);
        std::vector<uint32_t> cpu_pairs_b(pair_words);
        if (!prep.align_read_perm_pairs(gpu_pairs_b.data(), pair_words)) {
            fprintf(stderr, "[align-test-prod] read perm pairs B failed\n");
            goto done;
        }
        pearl_build_perm_pairs_b(b_seed_gpu, K_DIM, R_RANK, cpu_pairs_b.data());
        if (memcmp(gpu_pairs_b.data(), cpu_pairs_b.data(), pair_words * sizeof(uint32_t)) != 0) {
            fprintf(stderr, "[align-test-prod] perm pairs B mismatch\n");
            goto done;
        }
    }
    printf("[align-test-prod] perm pairs B OK\n");
    fflush(stdout);

    if (!prep.prepare_job_b(d_b_pre, b_seed_gpu, n, K_DIM, blocks_k, macro_cols)) {
        fprintf(stderr, "[align-test-prod] GPU fused prepack B failed\n");
        goto done;
    }
    {
        std::vector<int8_t> b_noisy(szBpT);
        std::vector<int8_t> b_pre_cpu(b_pre_bytes);
        if (pearl_build_noisy_b(n, K_DIM, R_RANK, b_seed_gpu, nullptr, b_noisy.data()) != 0) {
            fprintf(stderr, "[align-test-prod] CPU noisy B failed\n");
            goto done;
        }
        case32::prepack_b_coalesced(b_noisy.data(), n, K_DIM, blocks_k, macro_cols, &b_pre_cpu);
        std::vector<int8_t> b_pre_gpu(b_pre_bytes);
        if (!ocl.read_buffer(d_b_pre, b_pre_gpu.data(), b_pre_bytes)) {
            fprintf(stderr, "[align-test-prod] D2H B prepack failed\n");
            goto done;
        }
        {
            size_t diff_count = 0;
            size_t first_diff = 0;
            for (size_t i = 0; i < b_pre_bytes; i++) {
                if (b_pre_gpu[i] != b_pre_cpu[i]) {
                    if (diff_count == 0) {
                        first_diff = i;
                    }
                    diff_count++;
                }
            }
            if (diff_count > 0) {
                fprintf(stderr,
                        "[align-test-prod] B prepack full compare: %zu/%zu bytes differ (first @%zu "
                        "gpu=%d cpu=%d)\n",
                        diff_count, b_pre_bytes, first_diff, (int)b_pre_gpu[first_diff],
                        (int)b_pre_cpu[first_diff]);
                goto done;
            }
        }
        if (compare_prepack_samples("B", b_pre_gpu.data(), b_pre_cpu.data(), b_pre_bytes,
                                    macro_rows, macro_cols, blocks_k, false) != 0) {
            goto done;
        }
    }
    printf("[align-test-prod] fused prepack B samples OK\n");
    fflush(stdout);

    if (!prep.prepare_attempt_a(d_a_pre, ab_seed, static_cast<int>(sizeof(ab_seed)), job_key,
                                b_seed_gpu, m, K_DIM, blocks_k, macro_rows, 0, a_key_gpu)) {
        fprintf(stderr, "[align-test-prod] GPU fused prepack A failed\n");
        goto done;
    }
    {
        std::vector<int8_t> a_noisy(szAp);
        std::vector<int8_t> a_pre_cpu(a_pre_bytes);
        if (pearl_build_noisy_a(m, K_DIM, R_RANK, a_key_cpu, h_A, a_noisy.data()) != 0) {
            fprintf(stderr, "[align-test-prod] CPU noisy A failed\n");
            goto done;
        }
        case32::prepack_a_coalesced(a_noisy.data(), m, K_DIM, blocks_k, macro_rows, false,
                                    &a_pre_cpu);
        std::vector<int8_t> a_pre_gpu(a_pre_bytes);
        if (!ocl.read_buffer(d_a_pre, a_pre_gpu.data(), a_pre_bytes)) {
            fprintf(stderr, "[align-test-prod] D2H A prepack failed\n");
            goto done;
        }
        if (compare_prepack_samples("A", a_pre_gpu.data(), a_pre_cpu.data(), a_pre_bytes,
                                    macro_rows, macro_cols, blocks_k, true) != 0) {
            goto done;
        }
    }
    printf("[align-test-prod] fused prepack A samples OK\n");
    fflush(stdout);

    rc = 0;

done:
    if (d_Bt) {
        clReleaseMemObject(d_Bt);
    }
    if (d_a_pre) {
        clReleaseMemObject(d_a_pre);
    }
    if (d_b_pre) {
        clReleaseMemObject(d_b_pre);
    }
    free(h_A);
    free(h_Bt);
    if (rc == 0) {
        printf("[align-test-prod] OpenCL pipeline OK\n");
        fflush(stdout);
    }
    return rc;
}
