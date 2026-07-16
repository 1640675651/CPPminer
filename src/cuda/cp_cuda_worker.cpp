#include "cp_cuda_worker.h"
#include "cp_gpu.h"

extern "C" void cp_cuda_worker_init(int* devices, int ndev)
{
    cp_gpu_init(devices, ndev);
}

extern "C" void cp_cuda_worker_shutdown(void)
{
    cp_gpu_shutdown();
}

extern "C" void cp_cuda_worker_set_contiguous_tiles(int on)
{
    cp_gpu_set_contiguous_tiles(on);
}

extern "C" void cp_cuda_worker_set_period_gemm(int on)
{
    cp_gpu_set_period_gemm(on);
}

extern "C" void cp_cuda_worker_set_period_batch(int batch)
{
    cp_gpu_set_period_batch(batch);
}

extern "C" void cp_cuda_worker_set_row_period_batch(int batch)
{
    cp_gpu_set_row_period_batch(batch);
}

extern "C" void cp_cuda_worker_set_col_period_batch(int batch)
{
    cp_gpu_set_col_period_batch(batch);
}

extern "C" void cp_cuda_worker_set_step_major_ap(int on)
{
    cp_gpu_set_step_major_ap(on);
}

extern "C" void cp_cuda_worker_set_cutlass_fused(int on)
{
    cp_gpu_set_cutlass_fused(on);
}

extern "C" int cp_cuda_worker_mine_attempt(
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
    return cp_gpu_mine_attempt(
        ab_seed, ab_seed_len, job_key, pool_tgt, m, n, cpu_matrices,
        h_A_noisy, h_B_noisy, a_key, h_A_sig, h_Bt_sig,
        out_t_rows, out_t_cols, out_tiles_scanned);
}

extern "C" int cp_cuda_worker_fetch_share_signals(int8_t* h_A_sig, int8_t* h_Bt_sig)
{
    return cp_gpu_fetch_share_signals(h_A_sig, h_Bt_sig);
}
