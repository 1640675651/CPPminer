#include "cp_worker.h"
#include "cp_noise.h"
#include "cp_proof.h"

#include <stdio.h>
#include <string.h>

#if defined(CP_ENABLE_CPU) && CP_ENABLE_CPU
#include "cp_cpu_worker.h"
#endif
#if defined(CP_ENABLE_CUDA) && CP_ENABLE_CUDA
#include "cp_cuda_worker.h"
#include "cp_gpu.h"
#endif
#if defined(CP_ENABLE_OPENCL) && CP_ENABLE_OPENCL
#include "cp_opencl_worker.h"
#endif

static CpBackendId g_backend = CP_BACKEND_NONE;

extern "C" int cp_worker_has_cpu(void)
{
#if defined(CP_ENABLE_CPU) && CP_ENABLE_CPU
    return 1;
#else
    return 0;
#endif
}

extern "C" int cp_worker_has_cuda(void)
{
#if defined(CP_ENABLE_CUDA) && CP_ENABLE_CUDA
    return 1;
#else
    return 0;
#endif
}

extern "C" int cp_worker_has_opencl(void)
{
#if defined(CP_ENABLE_OPENCL) && CP_ENABLE_OPENCL
    return 1;
#else
    return 0;
#endif
}

static CpBackendId default_backend(void)
{
#if defined(CP_ENABLE_CUDA) && CP_ENABLE_CUDA
    return CP_BACKEND_CUDA;
#elif defined(CP_ENABLE_CPU) && CP_ENABLE_CPU
    return CP_BACKEND_CPU;
#elif defined(CP_ENABLE_OPENCL) && CP_ENABLE_OPENCL
    return CP_BACKEND_OPENCL;
#else
    return CP_BACKEND_NONE;
#endif
}

extern "C" int cp_worker_select(CpBackendId id)
{
    if(id == CP_BACKEND_NONE) id = default_backend();
    switch(id){
#if defined(CP_ENABLE_CPU) && CP_ENABLE_CPU
    case CP_BACKEND_CPU:
        g_backend = CP_BACKEND_CPU;
        return 0;
#endif
#if defined(CP_ENABLE_CUDA) && CP_ENABLE_CUDA
    case CP_BACKEND_CUDA:
        g_backend = CP_BACKEND_CUDA;
        return 0;
#endif
#if defined(CP_ENABLE_OPENCL) && CP_ENABLE_OPENCL
    case CP_BACKEND_OPENCL:
        g_backend = CP_BACKEND_OPENCL;
        return 0;
#endif
    default:
        fprintf(stderr, "[worker] backend %d not built into this binary\n", (int)id);
        return -1;
    }
}

extern "C" CpBackendId cp_worker_backend_id(void)
{
    if(g_backend == CP_BACKEND_NONE)
        g_backend = default_backend();
    return g_backend;
}

extern "C" const char* cp_worker_backend_name(void)
{
    switch(cp_worker_backend_id()){
    case CP_BACKEND_CPU: return "cpu";
    case CP_BACKEND_CUDA: return "cuda";
    case CP_BACKEND_OPENCL: return "opencl";
    default: return "none";
    }
}

extern "C" void cp_worker_init(int* devices, int ndev)
{
    if(g_backend == CP_BACKEND_NONE)
        g_backend = default_backend();
    switch(g_backend){
#if defined(CP_ENABLE_CPU) && CP_ENABLE_CPU
    case CP_BACKEND_CPU:
        (void)devices; (void)ndev;
        cp_cpu_worker_init();
        return;
#endif
#if defined(CP_ENABLE_CUDA) && CP_ENABLE_CUDA
    case CP_BACKEND_CUDA:
        cp_cuda_worker_init(devices, ndev);
        return;
#endif
#if defined(CP_ENABLE_OPENCL) && CP_ENABLE_OPENCL
    case CP_BACKEND_OPENCL:
        cp_opencl_worker_init(devices, ndev);
        return;
#endif
    default:
        fprintf(stderr, "[worker] no backend available\n");
        break;
    }
}

extern "C" void cp_worker_set_ocl_platform(int platform_index)
{
#if defined(CP_ENABLE_OPENCL) && CP_ENABLE_OPENCL
    cp_opencl_worker_set_platform(platform_index);
#else
    (void)platform_index;
#endif
}

extern "C" void cp_worker_set_ocl_tile(int mr, int nr)
{
#if defined(CP_ENABLE_OPENCL) && CP_ENABLE_OPENCL
    cp_opencl_worker_set_tile(mr, nr);
#else
    (void)mr;
    (void)nr;
#endif
}

extern "C" void cp_worker_configure_ocl_tile(int device_index)
{
#if defined(CP_ENABLE_OPENCL) && CP_ENABLE_OPENCL
    cp_opencl_configure_tile_for_worker(device_index);
#else
    (void)device_index;
#endif
}

extern "C" int cp_worker_list_devices(void)
{
    if(g_backend == CP_BACKEND_NONE)
        g_backend = default_backend();
    switch(g_backend){
#if defined(CP_ENABLE_OPENCL) && CP_ENABLE_OPENCL
    case CP_BACKEND_OPENCL:
        return cp_opencl_worker_list_devices();
#endif
#if defined(CP_ENABLE_CUDA) && CP_ENABLE_CUDA
    case CP_BACKEND_CUDA:
        return cp_gpu_list_devices();
#endif
    case CP_BACKEND_CPU:
        printf("[cpu] host CPU backend (no device list)\n");
        return 0;
    default:
        fprintf(stderr, "[worker] no backend available for --list-devices\n");
        return 0;
    }
}

extern "C" void cp_worker_shutdown(void)
{
    switch(g_backend){
#if defined(CP_ENABLE_CPU) && CP_ENABLE_CPU
    case CP_BACKEND_CPU: cp_cpu_worker_shutdown(); break;
#endif
#if defined(CP_ENABLE_CUDA) && CP_ENABLE_CUDA
    case CP_BACKEND_CUDA: cp_cuda_worker_shutdown(); break;
#endif
#if defined(CP_ENABLE_OPENCL) && CP_ENABLE_OPENCL
    case CP_BACKEND_OPENCL: cp_opencl_worker_shutdown(); break;
#endif
    default: break;
    }
}

extern "C" void cp_worker_apply_backend_defaults(void)
{
    const int layout = cp_worker_default_tile_layout();
    const int contiguous =
        (layout == CP_TILE_LAYOUT_CONTIGUOUS || layout == CP_TILE_LAYOUT_CONTIGUOUS_8x8);
    pearl_set_contiguous_tiles(contiguous);
#if defined(CP_ENABLE_CUDA) && CP_ENABLE_CUDA
    if(cp_worker_backend_id() == CP_BACKEND_CUDA)
        cp_cuda_worker_set_contiguous_tiles(contiguous);
#endif
#if defined(CP_ENABLE_OPENCL) && CP_ENABLE_OPENCL
    (void)contiguous;
#endif
}

extern "C" int cp_worker_uses_contiguous_tiles(void)
{
    const int layout = cp_worker_default_tile_layout();
    return layout == CP_TILE_LAYOUT_CONTIGUOUS || layout == CP_TILE_LAYOUT_CONTIGUOUS_8x8;
}

extern "C" void cp_worker_set_period_gemm(int on)
{
#if defined(CP_ENABLE_CUDA) && CP_ENABLE_CUDA
    if(cp_worker_backend_id() == CP_BACKEND_CUDA)
        cp_cuda_worker_set_period_gemm(on);
#else
    (void)on;
#endif
}

extern "C" void cp_worker_set_period_batch(int batch)
{
#if defined(CP_ENABLE_CUDA) && CP_ENABLE_CUDA
    if(cp_worker_backend_id() == CP_BACKEND_CUDA)
        cp_cuda_worker_set_period_batch(batch);
#endif
#if defined(CP_ENABLE_OPENCL) && CP_ENABLE_OPENCL
    if(cp_worker_backend_id() == CP_BACKEND_OPENCL)
        cp_opencl_worker_set_macro_batch(batch);
#endif
    (void)batch;
}

extern "C" void cp_worker_set_row_period_batch(int batch)
{
#if defined(CP_ENABLE_CUDA) && CP_ENABLE_CUDA
    if(cp_worker_backend_id() == CP_BACKEND_CUDA)
        cp_cuda_worker_set_row_period_batch(batch);
#else
    (void)batch;
#endif
}

extern "C" void cp_worker_set_col_period_batch(int batch)
{
#if defined(CP_ENABLE_CUDA) && CP_ENABLE_CUDA
    if(cp_worker_backend_id() == CP_BACKEND_CUDA)
        cp_cuda_worker_set_col_period_batch(batch);
#endif
#if defined(CP_ENABLE_OPENCL) && CP_ENABLE_OPENCL
    if(cp_worker_backend_id() == CP_BACKEND_OPENCL)
        cp_opencl_worker_set_macro_batch(batch);
#endif
    (void)batch;
}

extern "C" void cp_worker_set_step_major_ap(int on)
{
#if defined(CP_ENABLE_CUDA) && CP_ENABLE_CUDA
    if(cp_worker_backend_id() == CP_BACKEND_CUDA)
        cp_cuda_worker_set_step_major_ap(on);
#else
    (void)on;
#endif
}

extern "C" void cp_worker_set_cutlass_fused(int on)
{
#if defined(CP_ENABLE_CUDA) && CP_ENABLE_CUDA
    if(cp_worker_backend_id() == CP_BACKEND_CUDA)
        cp_cuda_worker_set_cutlass_fused(on);
#else
    (void)on;
#endif
}

extern "C" void cp_worker_set_prepack_mode(CpPrepackMode mode)
{
#if defined(CP_ENABLE_CPU) && CP_ENABLE_CPU
    if(cp_worker_backend_id() == CP_BACKEND_CPU)
        cp_cpu_worker_set_prepack_mode(mode);
#else
    (void)mode;
#endif
}

extern "C" void cp_worker_set_inplace_prepack(int on)
{
    cp_worker_set_prepack_mode(on ? CP_PREPACK_REUSE : CP_PREPACK_SEPARATE);
}

extern "C" void cp_worker_set_simd_isa(CpSimdIsa isa)
{
#if defined(CP_ENABLE_CPU) && CP_ENABLE_CPU
    if(cp_worker_backend_id() == CP_BACKEND_CPU)
        cp_cpu_worker_set_simd_isa(isa);
#else
    (void)isa;
#endif
}

extern "C" int cp_worker_prefers_host_matrices(void)
{
    return cp_worker_backend_id() == CP_BACKEND_CPU;
}

extern "C" int cp_worker_worker_handles_matrix_prep(void)
{
#if defined(CP_ENABLE_CPU) && CP_ENABLE_CPU
    if(cp_worker_backend_id() == CP_BACKEND_CPU)
        return cp_cpu_worker_handles_matrix_prep();
#endif
#if defined(CP_ENABLE_OPENCL) && CP_ENABLE_OPENCL
    if(cp_worker_backend_id() == CP_BACKEND_OPENCL)
        return cp_opencl_worker_handles_matrix_prep();
#endif
    return 0;
}

extern "C" void cp_worker_begin_job(const uint8_t job_key[32], int m, int n)
{
#if defined(CP_ENABLE_CPU) && CP_ENABLE_CPU
    if(cp_worker_backend_id() == CP_BACKEND_CPU)
        cp_cpu_worker_begin_job(job_key, m, n);
#endif
#if defined(CP_ENABLE_OPENCL) && CP_ENABLE_OPENCL
    if(cp_worker_backend_id() == CP_BACKEND_OPENCL)
        cp_opencl_worker_begin_job(job_key, m, n);
#endif
}

extern "C" int cp_worker_default_tile_layout(void)
{
    if(cp_worker_backend_id() == CP_BACKEND_CPU)
        return CP_TILE_LAYOUT_CONTIGUOUS;
#if defined(CP_ENABLE_OPENCL) && CP_ENABLE_OPENCL
    if(cp_worker_backend_id() == CP_BACKEND_OPENCL) {
        if(cp_opencl_hash_tile_mr() == 4 && cp_opencl_hash_tile_w() == 8)
            return CP_TILE_LAYOUT_CONTIGUOUS_4x8;
        if(cp_opencl_hash_tile_w() == 8)
            return CP_TILE_LAYOUT_CONTIGUOUS_8x8;
        return CP_TILE_LAYOUT_CONTIGUOUS;
    }
#endif
    return CP_TILE_LAYOUT_SCATTERED;
}

extern "C" int cp_worker_mine_attempt(
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
    switch(cp_worker_backend_id()){
#if defined(CP_ENABLE_CPU) && CP_ENABLE_CPU
    case CP_BACKEND_CPU:
        return cp_cpu_worker_mine_attempt(
            ab_seed, ab_seed_len, job_key, pool_tgt, m, n, cpu_matrices,
            h_A_noisy, h_B_noisy, a_key, h_A_sig, h_Bt_sig,
            out_t_rows, out_t_cols, out_tiles_scanned);
#endif
#if defined(CP_ENABLE_CUDA) && CP_ENABLE_CUDA
    case CP_BACKEND_CUDA:
        return cp_cuda_worker_mine_attempt(
            ab_seed, ab_seed_len, job_key, pool_tgt, m, n, cpu_matrices,
            h_A_noisy, h_B_noisy, a_key, h_A_sig, h_Bt_sig,
            out_t_rows, out_t_cols, out_tiles_scanned);
#endif
#if defined(CP_ENABLE_OPENCL) && CP_ENABLE_OPENCL
    case CP_BACKEND_OPENCL:
        return cp_opencl_worker_mine_attempt(
            ab_seed, ab_seed_len, job_key, pool_tgt, m, n, cpu_matrices,
            h_A_noisy, h_B_noisy, a_key, h_A_sig, h_Bt_sig,
            out_t_rows, out_t_cols, out_tiles_scanned);
#endif
    default:
        fprintf(stderr, "[worker] mine_attempt: no backend\n");
        return -1;
    }
}

extern "C" int cp_worker_fetch_share_signals(int8_t* h_A_sig, int8_t* h_Bt_sig)
{
    switch(cp_worker_backend_id()){
#if defined(CP_ENABLE_CPU) && CP_ENABLE_CPU
    case CP_BACKEND_CPU:
        (void)h_A_sig;
        (void)h_Bt_sig;
        return 0; /* already on host */
#endif
#if defined(CP_ENABLE_CUDA) && CP_ENABLE_CUDA
    case CP_BACKEND_CUDA:
        return cp_cuda_worker_fetch_share_signals(h_A_sig, h_Bt_sig);
#endif
#if defined(CP_ENABLE_OPENCL) && CP_ENABLE_OPENCL
    case CP_BACKEND_OPENCL:
        return cp_opencl_worker_fetch_share_signals(h_A_sig, h_Bt_sig);
#endif
    default:
        fprintf(stderr, "[worker] fetch_share_signals: no backend\n");
        return -1;
    }
}
