#include "cp_cpu_worker.h"



#include "cp_config.h"

#include "cp_jackpot.hpp"

#include "cp_job_ctrl.h"

#include "cp_noise.h"

#include "cp_util.h"

#include "gemm/case33_gemm_xor.hpp"



#include <atomic>

#include <chrono>

#include <cstdio>

#include <cstdlib>

#include <cstring>

#include <thread>

#include <vector>



namespace {



struct ZeroBCache {

    uint8_t job_key[32]{};

    int m = 0;

    int n = 0;

    int ready = 0;

    uint8_t b_noise_seed[32]{};

    std::vector<int8_t> B_sig;

    std::vector<int8_t> B_noisy;

} g_zero_b;



static int zero_b_cache_matches(const uint8_t job_key[32], int m, int n)

{

    return g_zero_b.ready && g_zero_b.m == m && g_zero_b.n == n

        && memcmp(g_zero_b.job_key, job_key, 32) == 0;

}



static int zero_b_prepare_job(const uint8_t job_key[32], int m, int n)

{

    const size_t szB = (size_t)n * (size_t)K_DIM;

    g_zero_b.B_sig.assign(szB, 0);

    g_zero_b.B_noisy.resize(szB);

    memcpy(g_zero_b.job_key, job_key, 32);

    g_zero_b.m = m;

    g_zero_b.n = n;



    pearl_b_noise_seed_from_bt(job_key, g_zero_b.B_sig.data(), n, K_DIM,
                               g_zero_b.b_noise_seed);



    if(pearl_build_noisy_b(n, K_DIM, R_RANK, g_zero_b.b_noise_seed,

                          g_zero_b.B_sig.data(), g_zero_b.B_noisy.data()) != 0){

        g_zero_b.ready = 0;

        return cp_job_should_cancel() ? -1 : -2;

    }



    g_zero_b.ready = 1;

    return 0;

}



static int zero_b_prepare_attempt(

    const uint8_t* ab_seed, int ab_seed_len,

    const uint8_t job_key[32],

    int m, int n,

    int8_t* h_A_sig, int8_t* h_Bt_sig,

    std::vector<int8_t>& A_sig,

    std::vector<int8_t>& A_noisy,

    uint8_t a_key_out[32])

{

    if(!zero_b_cache_matches(job_key, m, n)){

        if(zero_b_prepare_job(job_key, m, n) != 0)

            return -1;

    }



    const size_t szA = (size_t)m * (size_t)K_DIM;

    const size_t szB = (size_t)n * (size_t)K_DIM;

    A_sig.resize(szA);

    A_noisy.resize(szA);



    if(pearl_generate_random_a(ab_seed, ab_seed_len, m, K_DIM, A_sig.data()) != 0)

        return -1;



    if(h_A_sig) memcpy(h_A_sig, A_sig.data(), szA);

    if(h_Bt_sig) memset(h_Bt_sig, 0, szB);



    uint8_t b_seed_out[32];

    pearl_commitment_seeds(job_key, A_sig.data(), g_zero_b.B_sig.data(),

                           m, n, K_DIM, b_seed_out, a_key_out);



    if(pearl_build_noisy_a(m, K_DIM, R_RANK, a_key_out,

                           A_sig.data(), A_noisy.data()) != 0){

        return cp_job_should_cancel() ? -1 : -2;

    }



    return 0;

}



} /* namespace */



extern "C" int cp_cpu_worker_handles_matrix_prep(void)

{

    return 1;

}



extern "C" void cp_cpu_worker_begin_job(const uint8_t job_key[32], int m, int n)

{

    g_zero_b.ready = 0;

    if(zero_b_prepare_job(job_key, m, n) == 0){

        printf("[cpu] zero-B: cached noisy B for job (signal B^T = 0)\n");

        fflush(stdout);

    }

}



extern "C" void cp_cpu_worker_init(void)

{

    printf("[cpu] Case 3.3 fused GEMM+XOR worker (contiguous 8x16 tiles, zero-B)\n");

    fflush(stdout);

}



extern "C" void cp_cpu_worker_shutdown(void)

{

    g_zero_b.ready = 0;

    g_zero_b.B_sig.clear();

    g_zero_b.B_noisy.clear();

}



extern "C" int cp_cpu_worker_mine_attempt(

    const uint8_t* ab_seed, int ab_seed_len,

    const uint8_t job_key[32],

    const uint32_t pool_tgt[8],

    int m, int n,

    int cpu_matrices,

    const int8_t* h_A_noisy, const int8_t* h_B_noisy,

    const uint8_t* a_key,

    int8_t* h_A_sig, int8_t* h_Bt_sig,

    int* out_t_rows, int* out_t_cols,

    uint64_t* out_tiles_scanned)

{

    (void)cpu_matrices;



    if(out_tiles_scanned) *out_tiles_scanned = 0;

    if(out_t_rows) *out_t_rows = -1;

    if(out_t_cols) *out_t_cols = -1;



    const size_t szA = (size_t)m * (size_t)K_DIM;

    std::vector<int8_t> A_local;

    std::vector<int8_t> A_noisy_local;

    const int8_t* A = h_A_noisy;

    const int8_t* B = h_B_noisy;

    uint8_t a_key_local[32];

    const uint8_t* scan_key = a_key;



    if(!A || !B || !scan_key){

        if(zero_b_prepare_attempt(ab_seed, ab_seed_len, job_key, m, n,

                                  h_A_sig, h_Bt_sig,

                                  A_local, A_noisy_local, a_key_local) != 0){

            return cp_job_should_cancel() ? -1 : 0;

        }

        A = A_noisy_local.data();

        B = g_zero_b.B_noisy.data();

        scan_key = a_key_local;

    }



    if(m % Case33GemmXor::kMacroM != 0 || n % Case33GemmXor::kMacroN != 0){

        fprintf(stderr, "[cpu] m,n must be multiples of %dx%d (got %dx%d)\n",

                Case33GemmXor::kMacroM, Case33GemmXor::kMacroN, m, n);

        return -1;

    }



    Case33GemmXor gemm;

    gemm.set_int8_mode(Case32Int8Mode::FastU8S8);

    if(!gemm.init(m, n, K_DIM, A, B)){

        fprintf(stderr, "[cpu] Case3.3 init failed (backend=%s)\n", gemm.backend());

        return -1;

    }



    uint32_t bound[8];

    cp_scale_jackpot_target(pool_tgt, bound);

    uint32_t a_key8[8];

    memcpy(a_key8, scan_key, 32);



    const int row_parts = cp_pp_num_row_parts(m, 1);

    const int col_parts = cp_pp_num_col_parts(n, 1);

    const int total_tiles = row_parts * col_parts;

    const double scan_t0 = cp_now_sec();



    printf("[cpu] plain_proof scan %dx%d hash tiles, difficulty scaled by %d\n",

           row_parts, col_parts, PP_HASH_H * PP_HASH_W * K_DIM);

    fflush(stdout);



    std::atomic<int> found{0};

    std::atomic<uint64_t> tiles{0};

    std::atomic<int> hit_rows{-1};

    std::atomic<int> hit_cols{-1};

    std::atomic<bool> scan_done{false};



    std::thread progress_thread([&]() {

        uint64_t last_tiles = 0;

        double last_report = scan_t0;

        while(!scan_done.load(std::memory_order_relaxed)){

            std::this_thread::sleep_for(std::chrono::milliseconds(250));

            if(scan_done.load(std::memory_order_relaxed)) break;

            if(found.load(std::memory_order_relaxed)) break;



            const uint64_t cur = tiles.load(std::memory_order_relaxed);

            const double now = cp_now_sec();

            if(cur == last_tiles && now - last_report < 1.0) continue;

            if(now - last_report < 1.0 && cur - last_tiles < 4096) continue;



            double scan_sec = now - scan_t0;

            if(scan_sec < 1e-9) scan_sec = 1e-9;

            int row_done = col_parts > 0

                ? (int)((cur + (uint64_t)col_parts - 1) / (uint64_t)col_parts)

                : 0;

            if(row_done > row_parts) row_done = row_parts;



            char mac_buf[32];

            cp_pp_fmt_mac_rate(cp_pp_mac_rate_from_tiles(cur, scan_sec),

                               mac_buf, sizeof(mac_buf));

            printf("[cpu] plain_proof progress: row parts %d/%d tiles %llu/%d (%.1f%%) %s\n",

                   row_done, row_parts,

                   (unsigned long long)cur, total_tiles,

                   total_tiles > 0 ? 100.0 * (double)cur / (double)total_tiles : 0.0,

                   mac_buf);

            fflush(stdout);

            last_tiles = cur;

            last_report = now;

        }

    });



    const bool ok = gemm.scan_tiles(

        [&](const uint32_t* milestone_xor, int t_rows, int t_cols) -> bool {

            tiles.fetch_add(1, std::memory_order_relaxed);

            if(found.load(std::memory_order_relaxed)) return false;

            if(cp_jackpot::tile_beats_target(milestone_xor, Case33GemmXor::kNumMilestones,

                                             a_key8, bound)){

                int expected = 0;

                if(found.compare_exchange_strong(expected, 1)){

                    hit_rows.store(t_rows, std::memory_order_relaxed);

                    hit_cols.store(t_cols, std::memory_order_relaxed);

                }

                return false;

            }

            return true;

        },

        []() -> bool { return cp_job_should_cancel() != 0; });



    scan_done.store(true, std::memory_order_relaxed);

    if(progress_thread.joinable()) progress_thread.join();



    if(out_tiles_scanned) *out_tiles_scanned = tiles.load();



    if(cp_job_should_cancel()) return -1;

    if(!ok && found.load() == 0) return -1;



    if(found.load()){

        const int tr = hit_rows.load();

        const int tc = hit_cols.load();

        printf("[cpu] plain_proof SHARE t_rows=%d t_cols=%d\n", tr, tc);

        fflush(stdout);

        if(out_t_rows) *out_t_rows = tr;

        if(out_t_cols) *out_t_cols = tc;

        return 1;

    }

    (void)szA;

    return 0;

}

