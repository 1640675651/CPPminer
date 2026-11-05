#include "cp_backend_setup.h"
#include "cp_algo.h"
#include "cp_config.h"
#include "cp_fee.h"
#include "cp_noise.h"
#include "cp_proof.h"
#include "cp_qpow_mine.h"
#include "cp_state.h"
#include "cp_util.h"
#include "cp_worker.h"
#if defined(CP_ENABLE_OPENCL) && CP_ENABLE_OPENCL
#include "cp_qpow_opencl_worker.h"
#endif
#if defined(CP_ENABLE_CUDA) && CP_ENABLE_CUDA
#include "cp_qpow_cuda_worker.h"
#endif

#if defined(CP_ENABLE_CPU) && CP_ENABLE_CPU
#include "cp_cpu_affinity.h"
#endif

#if defined(CP_ENABLE_ONEDNN) && CP_ENABLE_ONEDNN
#include "cp_onednn_worker.h"
#endif

#include <stdio.h>
#include <stdlib.h>

int cp_list_devices(const CpOptions* o)
{
    int n = 0;
    if(o->backend_sel == CP_BACKEND_NONE){
#if defined(CP_ENABLE_CUDA) && CP_ENABLE_CUDA
        if(cp_worker_has_cuda()){
            if(cp_worker_select(CP_BACKEND_CUDA) != 0) return 1;
            n += cp_worker_list_devices();
        }
#endif
#if defined(CP_ENABLE_OPENCL) && CP_ENABLE_OPENCL
        if(cp_worker_has_opencl()){
            if(o->ocl_platform >= 0)
                cp_worker_set_ocl_platform(o->ocl_platform);
            if(cp_worker_select(CP_BACKEND_OPENCL) != 0) return 1;
            n += cp_worker_list_devices();
        }
#endif
#if defined(CP_ENABLE_ONEDNN) && CP_ENABLE_ONEDNN
        if(cp_worker_has_onednn()){
            if(o->ocl_platform >= 0)
                cp_worker_set_onednn_platform(o->ocl_platform);
            if(cp_worker_select(CP_BACKEND_ONEDNN) != 0) return 1;
            n += cp_worker_list_devices();
        }
#endif
#if defined(CP_ENABLE_WGPU) && CP_ENABLE_WGPU
        if(cp_worker_has_wgpu()){
            if(cp_worker_select(CP_BACKEND_WGPU) != 0) return 1;
            n += cp_worker_list_devices();
        }
#endif
        if(n <= 0){
            printf("[list-devices] no CUDA/OpenCL/OneDNN/wgpu backends in this build\n");
            return 1;
        }
        return 0;
    }
    if(cp_worker_select(o->backend_sel) != 0) return 1;
#if defined(CP_ENABLE_OPENCL) && CP_ENABLE_OPENCL
    if(o->ocl_platform >= 0)
        cp_worker_set_ocl_platform(o->ocl_platform);
#endif
#if defined(CP_ENABLE_ONEDNN) && CP_ENABLE_ONEDNN
    if(o->ocl_platform >= 0)
        cp_worker_set_onednn_platform(o->ocl_platform);
#endif
#if defined(CP_ENABLE_ONEDNN) && CP_ENABLE_ONEDNN
    if(o->ocl_platform >= 0)
        cp_worker_set_onednn_platform(o->ocl_platform);
#endif
    n = cp_worker_list_devices();
    return n > 0 ? 0 : 1;
}

int cp_apply_backend_options(CpOptions* o)
{
    if(cp_worker_select(o->backend_sel) != 0) return 1;

#if defined(CP_ENABLE_OPENCL) && CP_ENABLE_OPENCL
    if(o->ocl_platform >= 0)
        cp_worker_set_ocl_platform(o->ocl_platform);
    if(o->ocl_tile_mr > 0)
        cp_worker_set_ocl_tile(o->ocl_tile_mr, o->ocl_tile_nr);
    if(o->ocl_macro_m > 0)
        cp_worker_set_ocl_macro(o->ocl_macro_m, o->ocl_macro_n);
    if(o->ocl_issue_mode != 0)
        cp_worker_set_ocl_issue_mode(o->ocl_issue_mode);
    if(o->ocl_dot_policy != 0)
        cp_worker_set_ocl_dot_policy(o->ocl_dot_policy);
    if(o->ocl_cpm_type != 0)
        cp_worker_set_ocl_cpm_type(o->ocl_cpm_type);
    if(o->ocl_lds)
        cp_worker_set_ocl_lds(1);
    if(o->ocl_double_buff)
        cp_worker_set_ocl_double_buff(1);
#endif
#if (defined(CP_ENABLE_OPENCL) && CP_ENABLE_OPENCL) || (defined(CP_ENABLE_ONEDNN) && CP_ENABLE_ONEDNN)
    /* Default off on both backends. low saves a CPU core on oneDNN, but the A770 driver's sleeping wait
     * can miss its wake-up and stall a scan for ~70 s (arc-ocl-stall); on the OpenCL backend it also
     * cost 3.6% to the wake-up latency. */
    cp_ocl_set_queue_throttle(o->ocl_throttle);
#endif
#if defined(CP_ENABLE_WGPU) && CP_ENABLE_WGPU
    if(o->wgpu_lds >= 0)
        cp_worker_set_wgpu_lds(o->wgpu_lds);
    if(o->wgpu_tile_mr > 0)
        cp_worker_set_wgpu_tile(o->wgpu_tile_mr, o->wgpu_tile_nr);
    if(o->wgpu_macro_m > 0)
        cp_worker_set_wgpu_macro(o->wgpu_macro_m, o->wgpu_macro_n);
#endif

    if(cp_worker_backend_id() == CP_BACKEND_CUDA){
        if(o->cutlass_fused < 0) o->cutlass_fused = 1;
        cp_worker_set_cuda_mma(o->cuda_mma);
    } else {
        o->cutlass_fused = 0;
    }
    /* Case 10 (CUTLASS default) needs contiguous K (row-major Ap/BpT).
     * Step-major falls back to Case 9 wind_down; cuBLAS period / Case 7.2 packing. */
    if(o->step_major_ap < 0)
        o->step_major_ap = o->cutlass_fused ? 0 : 1;

    cp_worker_apply_backend_defaults();
    return 0;
}

int cp_quantus_backend_init(CpOptions* o)
{
    printf("[mode] algo=%s\n", cp_algo_name(o->algo_sel));
    fflush(stdout);
    /* Quantus launch batch: --batch-size, else automatic (CUDA, OpenCL) or 1e6 nonces. */
    {
        uint32_t qbatch = (cp_worker_backend_id() == CP_BACKEND_CUDA ||
                          cp_worker_backend_id() == CP_BACKEND_OPENCL) ? 0u : 1000000u;
        if(o->batch_size_set){
            if(o->batch_size < 1) o->batch_size = 1;
            qbatch = (uint32_t)o->batch_size;
        }
        cp_worker_set_period_batch((int)qbatch);
        if(qbatch)
            printf("[mode] batch-size: %u nonces/launch\n", qbatch);
        else
            printf("[mode] batch-size: auto\n");
        fflush(stdout);
    }
    if(cp_worker_backend_id() == CP_BACKEND_WGPU){
        /* No --devices → auto (all mining adapters). Explicit → those indices. */
        if(o->devices_specified)
            cp_worker_init(o->devs, o->ndev);
        else
            cp_worker_init(NULL, 0);
        if(!cp_worker_is_ready()){
            fprintf(stderr, "wgpu backend init failed\n");
            return 1;
        }
    }
#if defined(CP_ENABLE_OPENCL) && CP_ENABLE_OPENCL
    else if(cp_worker_backend_id() == CP_BACKEND_OPENCL){
        if(!o->ndev){ o->devs[0] = 0; o->ndev = 1; }
        if(cp_qpow_opencl_worker_init(o->devs, o->ndev) != 0){
            fprintf(stderr, "quantus opencl backend init failed\n");
            return 1;
        }
    }
#endif
#if defined(CP_ENABLE_CUDA) && CP_ENABLE_CUDA
    else if(cp_worker_backend_id() == CP_BACKEND_CUDA){
        if(!o->ndev){ o->devs[0] = 0; o->ndev = 1; }
        if(cp_qpow_cuda_worker_init(o->devs, o->ndev) != 0){
            fprintf(stderr, "quantus cuda backend init failed\n");
            return 1;
        }
    }
#endif
    if(cp_worker_backend_id() == CP_BACKEND_CPU){
#if defined(CP_ENABLE_CPU) && CP_ENABLE_CPU
        /* Pin the OpenMP pool physical cores first, then SMT siblings (one
         * thread per logical CPU), so the --simd auto scalar/AVX2 split pairs
         * one of each per core. */
        if(cp_cpu_affinity_init() == 0)
            cp_cpu_affinity_bind_openmp_pool();
        printf("[cpu] affinity: %s\n", cp_cpu_affinity_summary());
        fflush(stdout);
#endif
        if(cp_qpow_set_simd_isa(o->simd_isa) != 0)
            return 1;
    }
    return 0;
}

void cp_quantus_backend_shutdown(void)
{
    if(cp_worker_backend_id() == CP_BACKEND_WGPU)
        cp_worker_shutdown();
#if defined(CP_ENABLE_OPENCL) && CP_ENABLE_OPENCL
    else if(cp_worker_backend_id() == CP_BACKEND_OPENCL)
        cp_qpow_opencl_worker_shutdown();
#endif
#if defined(CP_ENABLE_CUDA) && CP_ENABLE_CUDA
    else if(cp_worker_backend_id() == CP_BACKEND_CUDA)
        cp_qpow_cuda_worker_shutdown();
#endif
}

int cp_pearl_backend_setup(CpOptions* o)
{
    cp_worker_apply_backend_defaults();
    cp_worker_set_period_gemm(!o->no_period_gemm);
#if defined(CP_ENABLE_OPENCL) && CP_ENABLE_OPENCL
    if(cp_worker_backend_id() == CP_BACKEND_OPENCL
       && o->batch_size == CP_PERIOD_BATCH_DEFAULT){
        o->batch_size = CP_MACRO_BATCH_DEFAULT;
    }
    /* Resolve tile (incl. broadcast auto 4x8) before mode banner / fee tile counts. */
    if(cp_worker_backend_id() == CP_BACKEND_OPENCL){
        cp_worker_configure_ocl_tile(o->devs[0]);
        cp_worker_apply_backend_defaults();
    }
#endif
#if defined(CP_ENABLE_WGPU) && CP_ENABLE_WGPU
    if(cp_worker_backend_id() == CP_BACKEND_WGPU
       && cp_worker_algo() == 0
       && o->batch_size == CP_PERIOD_BATCH_DEFAULT){
        o->batch_size = CP_MACRO_BATCH_DEFAULT;
    }
#endif
#if defined(CP_ENABLE_ONEDNN) && CP_ENABLE_ONEDNN
    if(cp_worker_backend_id() == CP_BACKEND_ONEDNN){
        /* Unset batches: the worker picks them once it knows the kernel. */
        if(!o->batch_size_set){
            o->batch_size = CP_ONEDNN_PERIOD_BATCH_DEFAULT;
        }
        if(!o->row_period_batch_set){
            o->row_period_batch = CP_ONEDNN_PERIOD_BATCH_DEFAULT;
        }
        cp_onednn_worker_set_auto_batch(!o->batch_size_set, !o->row_period_batch_set);
    }
#endif
    if(o->fused_jackpot < 0)
        o->fused_jackpot = (cp_worker_backend_id() == CP_BACKEND_ONEDNN) ? 1 : 0;
    cp_worker_set_period_batch(o->batch_size);
    cp_worker_set_row_period_batch(o->row_period_batch);
    if(cp_worker_backend_id() == CP_BACKEND_HEXAGON){
        /* Launch = row x col 128x128 macro blocks; the shared defaults mean "not set". */
        if(o->batch_size == CP_PERIOD_BATCH_DEFAULT)
            o->batch_size = CP_HEXAGON_LAUNCH_MACROS_DEFAULT;
        if(o->row_period_batch == CP_ROW_PERIOD_BATCH_DEFAULT)
            o->row_period_batch = CP_HEXAGON_LAUNCH_MACROS_DEFAULT;
        cp_worker_set_hexagon_launch(o->row_period_batch, o->batch_size);
        cp_worker_set_hexagon_fused_jackpot(o->fused_jackpot);
    }
    if(cp_worker_backend_id() == CP_BACKEND_INTEL_NPU){
        /* Launch = row x col 128x128 macro blocks; unset means the measured default. */
        cp_worker_set_intel_npu_launch(o->row_period_batch_set ? o->row_period_batch : 0,
                                       o->batch_size_set ? o->batch_size : 0);
        cp_worker_set_intel_npu_fused_jackpot(o->fused_jackpot);
    }
#if defined(CP_ENABLE_ONEDNN) && CP_ENABLE_ONEDNN
    /* Kernel select + JIT before mode banner so hash tile / proof layout match gemmstone.
     * Period batch must be set before init (backend banner + scan loop read batch at init). */
    if(cp_worker_backend_id() == CP_BACKEND_ONEDNN){
        cp_onednn_worker_set_fused_jackpot(o->fused_jackpot);
        if(o->onednn_layout){
            cp_onednn_worker_set_gemm_layout(o->onednn_layout);
        }
        cp_onednn_worker_init(o->devs, o->ndev);
        /* The jackpot mode in use (Xe2 falls back to the separate jackpot). */
        o->fused_jackpot = cp_onednn_worker_fused_jackpot();
        cp_worker_apply_backend_defaults();
        /* Init may raise unset batches for systolic kernels. */
        o->batch_size = cp_onednn_worker_col_period_batch();
        o->row_period_batch = cp_onednn_worker_row_period_batch();
    }
#endif
    cp_worker_set_step_major_ap(o->step_major_ap);
    cp_worker_set_cutlass_fused(o->cutlass_fused);
    pearl_set_cutlass_fused(o->cutlass_fused);
    g_cutlass_fused = o->cutlass_fused;
    cp_worker_set_prepack_mode(o->prepack_mode);
    if(cp_worker_set_simd_isa(o->simd_isa) != 0)
        return 1;
    return 0;
}

void cp_print_pearl_mode(const CpOptions* o)
{
    double host_mib = ((double)g_m_active * K_DIM + (double)g_n_active * K_DIM)
                    / (1024.0 * 1024.0);
    const int contiguous = cp_worker_uses_contiguous_tiles();
    const int tile_layout = cp_worker_default_tile_layout();
    int row_parts = cp_pp_num_row_parts(g_m_active, contiguous);
    int col_parts = cp_pp_num_col_parts(g_n_active, contiguous);
    const char *tile_layout_name =
        o->cutlass_fused ? "CUTLASS MMA lane 8x8 interleaved (128x128 CTA)"
        : (tile_layout == CP_TILE_LAYOUT_CONTIGUOUS_16x16) ? "contiguous 16x16 blocks"
        : (tile_layout == CP_TILE_LAYOUT_CONTIGUOUS_4x64) ? "contiguous 4x64 blocks"
        : (tile_layout == CP_TILE_LAYOUT_CONTIGUOUS_4x8) ? "contiguous 4x8 blocks"
        : (tile_layout == CP_TILE_LAYOUT_CONTIGUOUS_8x8) ? "contiguous 8x8 blocks"
        : (tile_layout == CP_TILE_LAYOUT_CONTIGUOUS) ? "contiguous 8x16 blocks"
        : "BzMiner periodic scattered 8x16";
    printf("[mode] backend=%s\n", cp_worker_backend_name());
    printf("[mode] plain_proof m=%d n=%d k=%d r=%d (--m %d --n %d)\n",
           g_m_active, g_n_active, K_DIM, R_RANK,
           g_m_active / CP_MATRIX_UNIT, g_n_active / CP_MATRIX_UNIT);
    printf("[mode] tile layout: %s\n", tile_layout_name);
    if(cp_worker_backend_id() == CP_BACKEND_CPU){
        /* Host signal is A only: zero B^T is proven without a host buffer. */
        const double sig_mib = (double)g_m_active * K_DIM / (1024.0 * 1024.0);
        printf("[mode] scan: fused GEMM + XOR + host jackpot\n");
        if(o->prepack_mode == CP_PREPACK_FUSED)
            printf("[mode] matrix steady: ~%.0f MiB signal A + scan buffers (fused prepack)\n",
                   sig_mib + host_mib);
        else if(o->prepack_mode == CP_PREPACK_REUSE)
            printf("[mode] matrix steady: ~%.0f MiB signal A + scan buffers (reuse prepack)\n",
                   sig_mib + host_mib);
        else
            printf("[mode] matrix peak: ~%.0f MiB host signal A + ~%.0f MiB prepack\n",
                   sig_mib, host_mib * 2.0);
    } else if(cp_worker_backend_id() == CP_BACKEND_OPENCL){
        const int tiles_per_macro =
            (tile_layout == CP_TILE_LAYOUT_CONTIGUOUS_4x8) ? (128 / 4) * (128 / 8)
            : (tile_layout == CP_TILE_LAYOUT_CONTIGUOUS_8x8) ? (128 / 8) * (128 / 8)
            : (128 / 8) * (128 / 16);
        printf("[mode] scan: OpenCL fused GEMM + XOR + device jackpot\n");
        printf("[mode] macro batch: %d (%d hash tiles/launch, --batch-size)\n",
               o->batch_size, o->batch_size * tiles_per_macro);
        printf("[mode] noisy B cached on GPU per job\n");
    } else if(cp_worker_backend_id() == CP_BACKEND_WGPU && cp_worker_algo() == 0){
        const int wgpu_macro = o->wgpu_macro_m > 0 ? o->wgpu_macro_m : 128;
        const int hash_mr = cp_pp_hash_tile_h();
        const int hash_w = cp_pp_hash_tile_w();
        printf("[mode] scan: wgpu fused GEMM + XOR + device jackpot (hash %dx%d, macro %dx%d)\n",
               hash_mr, hash_w, wgpu_macro, wgpu_macro);
        printf("[mode] macro batch: %d (%d hash tiles/launch, --batch-size)\n",
               o->batch_size, o->batch_size * (wgpu_macro / hash_mr) * (wgpu_macro / hash_w));
        printf("[mode] noisy B cached on GPU per job\n");
    } else if(cp_worker_backend_id() == CP_BACKEND_ONEDNN){
        /* oneDNN row/col period-batch is in hash tiles (see Case33GemmOnednn scan). */
        const double panel_tiles =
                (double)o->row_period_batch * (double)o->batch_size;
        if(o->fused_jackpot){
            printf("[mode] scan: oneDNN fused GEMM + in-reg XOR/BLAKE3 + GPU jackpot\n");
            printf("[mode] period batch: row=%d col=%d\n", o->row_period_batch, o->batch_size);
        } else {
            const int tile_xor_words = K_DIM / R_RANK;
            printf("[mode] scan: oneDNN Case 5 GEMM + device fold/BLAKE jackpot (batched enqueue)\n");
            printf("[mode] period batch: row=%d col=%d (~%.1f MiB tile_xor/panel on GPU)\n",
                   o->row_period_batch, o->batch_size,
                   panel_tiles * (double)tile_xor_words * (double)sizeof(uint32_t)
                           / (1024.0 * 1024.0));
        }
        {
            const char *layout_msg = o->onednn_layout;
            if(!layout_msg){
                const char *env_layout = getenv("CASE5_GEMM_LAYOUT");
                layout_msg = (env_layout && env_layout[0]) ? env_layout : "TN";
            }
            printf("[mode] device layout %s on Intel GPU\n", layout_msg);
        }
    } else if(cp_worker_backend_id() == CP_BACKEND_HEXAGON){
        const int lrows = o->row_period_batch * 128 < g_m_active ? o->row_period_batch * 128
                                                               : g_m_active;
        const int lcols = o->batch_size * 128 < g_n_active ? o->batch_size * 128 : g_n_active;
        printf("[mode] scan: cDSP HVX fused GEMM + XOR (4x64 hash tiles) + %s\n",
               o->fused_jackpot ? "DSP jackpot (--fused-jackpot)" : "host jackpot");
        printf("[mode] DSP launch: %dx%d macro blocks of 128x128 = %dx%d "
               "(--row-period-batch x --batch-size), %d hash tiles\n",
               lrows / 128, lcols / 128, lrows, lcols, (lrows / 4) * (lcols / 64));
        printf("[mode] noisy B packed in DSP memory; each row panel of A packed once\n");
    } else if(cp_worker_backend_id() == CP_BACKEND_INTEL_NPU){
        printf("[mode] scan: Intel NPU fused int8 GEMM (DPU) + 16x16 tile XOR (SHAVE) + %s\n",
               o->fused_jackpot ? "BLAKE3 jackpot on the SHAVEs (--fused-jackpot)" : "host jackpot");
        int lrows = 0, lcols = 0;
        cp_worker_intel_npu_launch_shape(&lrows, &lcols);
        printf("[mode] NPU launch: %dx%d macro blocks of 128x128 = %dx%d (K=%d, --row-period-batch x "
               "--batch-size), %d launches per attempt\n", lrows / 128, lcols / 128, lrows, lcols, K_DIM,
               (g_m_active / lrows) * (g_n_active / lcols));
        printf("[mode] noisy A and B built and packed on the host\n");
    } else if(cp_worker_backend_id() == CP_BACKEND_CUDA){
        if(o->cutlass_fused){
            printf("[mode] proof rows/cols: 8 A + 8 B^T (interleaved 4x4)\n");
            printf("[mode] scan: CUTLASS Case 10 fused GEMM + inline XOR jackpot\n");
        } else {
            printf("[mode] scan: %s\n",
                   (contiguous || o->no_period_gemm) ? "per-tile kernel"
                                                  : "period GEMM + batched jackpot");
        }
        if(!g_cpu_matrix_gen){
            printf("[mode] zero-B: no host A/B (proofs from device sub-roots); ~1.5 GiB VRAM "
                   "(A_sig + noisy A/B, no d_Bt_sig)\n");
        }
    } else if(o->cutlass_fused){
        printf("[mode] proof rows/cols: 8 A + 8 B^T (interleaved 4x4)\n");
        printf("[mode] scan: CUTLASS Case 10 fused GEMM + inline XOR jackpot\n");
    } else {
        printf("[mode] scan: %s\n",
               (contiguous || o->no_period_gemm) ? "per-tile kernel"
                                              : "period GEMM + batched jackpot");
    }
#if defined(CP_ENABLE_CUDA) && CP_ENABLE_CUDA
    if(cp_worker_backend_id() == CP_BACKEND_CUDA
       && !contiguous && !o->no_period_gemm){
        printf("[mode] Ap/BpT layout: %s (lda=%d)\n",
               o->step_major_ap ? "step-major panels" : "row-major strided",
               o->step_major_ap ? R_RANK : K_DIM);
        if(o->cutlass_fused){
            printf("[mode] jackpot: fused in GEMM kernel (no tile_xor / C_hist)\n");
            printf("[mode] period batch: row=%d col=%d\n",
                   o->row_period_batch, o->batch_size);
        } else {
            printf("[mode] jackpot: separate XOR kernel (period GEMM)\n");
            printf("[mode] period batch: row=%d col=%d (~%.0f MiB C_hist/GPU)\n",
                   o->row_period_batch, o->batch_size,
                   (double)o->row_period_batch * (double)o->batch_size
                   * (double)(K_DIM / R_RANK)
                   * (double)PP_ROW_PERIOD * (double)PP_COL_PERIOD
                   * (double)sizeof(int32_t) / (1024.0 * 1024.0));
        }
    }
#endif
    printf("[mode] hash_tiles=%dx%d (%d total)\n",
           row_parts, col_parts, row_parts * col_parts);
    {
        const double a_mib = cp_worker_supports_share_witness()
                ? 0.0 : (double)g_m_active * K_DIM / (1024.0 * 1024.0);
        const double bt_mib = cp_worker_needs_host_bt()
                ? (double)g_n_active * K_DIM / (1024.0 * 1024.0) : 0.0;
        if(a_mib + bt_mib > 0.0)
            printf("[mode] host~%.0f MiB (signal%s%s)\n", a_mib + bt_mib,
                   a_mib > 0.0 ? " A" : "", bt_mib > 0.0 ? " B^T" : "");
        else
            printf("[mode] host signal: none (device witness proofs)\n");
    }
    printf("[mode] matrix gen: %s\n",
           (g_cpu_matrix_gen || cp_worker_prefers_host_matrices())
               ? "host BLAKE3 + noise"
               : (cp_worker_worker_handles_matrix_prep()
                      ? "zero-B (B once/job, A per nonce)"
                      : "device random + commitment/noise"));
    printf("[mode] verify=%d dry_run=%d max_nonce=%d mock=%d cert_version=%u%s\n",
           g_plain_verify, g_dry_run, g_max_nonce, g_mock,
           (unsigned)g_cert_version,
           g_cert_version_forced ? " (forced)" : "");
    fflush(stdout);
    cp_fee_print_mode();
}
