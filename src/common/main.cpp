#include "cp_config.h"
#include "cp_algo.h"
#include "cp_backend_setup.h"
#include "cp_diagnostics.h"
#include "cp_fee.h"
#include "cp_api.h"
#include "cp_mine.h"
#include "cp_options.h"
#include "cp_pool_session.h"
#include "cp_qpow_mine.h"
#include "cp_state.h"
#include "cp_util.h"
#include "cp_worker.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char** argv)
{
#ifdef __MINGW32__
    /* MinGW/MSYS2 only: buffer stdout before anything prints.
     *
     * MinGW builds use the MinGW printf (__USE_MINGW_ANSI_STDIO=1), which
     * emits each %-conversion as its own write to the stream. The MS CRT
     * leaves a console stdout unbuffered, so every fragment of a status line
     * becomes a separate WriteConsole and the log crawls out piece by piece.
     * MSVC builds are not affected (the CRT printf writes a call at once) and
     * other platforms keep their default buffering.
     *
     * On Win32, _IOLBF behaves as _IOFBF (full buffering), so prompt output
     * relies on the fflush(stdout) that follows every log line. */
    setvbuf(stdout, NULL, _IOLBF, 8192);
#endif

    CpOptions opt;
    {
        const int rc = cp_options_parse(argc, argv, &opt);
        if(rc >= 0) return rc;
    }

    if(opt.algo_sel == CP_ALGO_QUANTUS){
        if(opt.backend_sel == CP_BACKEND_NONE)
            opt.backend_sel = CP_BACKEND_CPU;
        if(!cp_algo_supports(opt.algo_sel, opt.backend_sel)){
            char qb[64];
            cp_algo_format_backends(CP_ALGO_QUANTUS, qb, (int)sizeof(qb));
            fprintf(stderr,
                    "--algo quantus does not support this --backend "
                    "(supported in this build: %s)\n", qb);
            return 1;
        }
        if(opt.simd_test)
            return cp_qpow_simd_test();
        if(!opt.pool_specified && !g_mock && !opt.list_devices){
            fprintf(stderr, "--pool required for --algo quantus (no default host)\n");
            return 1;
        }
    } else if(!opt.list_devices && opt.backend_sel != CP_BACKEND_NONE &&
              !cp_algo_supports(opt.algo_sel, opt.backend_sel)){
        char pb[64];
        cp_algo_format_backends(opt.algo_sel, pb, (int)sizeof(pb));
        fprintf(stderr,
                "--backend not available for --algo %s (supported: %s)\n",
                cp_algo_name(opt.algo_sel), pb);
        return 1;
    }

    cp_worker_set_algo((int)opt.algo_sel);

    if(opt.list_devices)
        return cp_list_devices(&opt);

    if(cp_apply_backend_options(&opt) != 0) return 1;

    {
        const int rc = cp_run_test_modes(&opt);
        if(rc >= 0) return rc;
    }

    /* Runs that never reach a pool: --mock, and Pearl's --simd-test / --prepack-test, which
     * exit after backend setup (Quantus --simd-test already exited above). */
    const int offline = g_mock ||
        (opt.algo_sel == CP_ALGO_PEARL && (opt.simd_test || opt.prepack_test));
    if(!opt.wallet){
        if(offline){
            opt.wallet = "mock-wallet";
        } else {
            fprintf(stderr, "--wallet required\n");
            return 1;
        }
    }
    if(!opt.ndev){ opt.devs[0] = 0; opt.ndev = 1; }

    if(g_mock){
        /* Offline self-test: no pool submit; always verify the first share. */
        g_dry_run = 1;
        g_plain_verify = 1;
    }

    strncpy(wallet_global, opt.wallet, sizeof(wallet_global) - 1);
    wallet_global[sizeof(wallet_global) - 1] = 0;

    /* Offline runs skip the pool; no fee reconnects. */
    cp_fee_init(wallet_global, offline ? 0 : 1, opt.algo_sel);
    cp_api_set_algo(cp_algo_name(opt.algo_sel), NULL);
    if(opt.api_port > 0) cp_api_start(opt.api_bind, opt.api_port);

    if(opt.algo_sel == CP_ALGO_QUANTUS){
        if(cp_quantus_backend_init(&opt) != 0)
            return 1;
        cp_api_set_algo(NULL, cp_worker_backend_name());
        if(cp_api_device_count() == 0) cp_api_add_device(cp_worker_backend_name(), NULL);
        int qrc;
        if(g_mock){
            qrc = cp_qpow_mine_mock(worker_global);
        } else {
            qrc = cp_run_quantus_pool(opt.pool_host, opt.pool_port);
        }
        cp_quantus_backend_shutdown();
        return qrc;
    }

    if(cp_pearl_backend_setup(&opt) != 0)
        return 1;
    if(opt.simd_test)
        return cp_pearl_simd_test();
    if(opt.prepack_test)
        return cp_prepack_test();

    if(opt.cutlass_fused){
        if(opt.no_period_gemm){
            fprintf(stderr, "CUTLASS fused path requires period GEMM (omit --no-period-gemm)\n");
            return 1;
        }
    }

    cp_init_workdir();

    cp_print_pearl_mode(&opt);

    cp_worker_init(opt.devs, opt.ndev);
    if(!cp_worker_is_ready()){
        fprintf(stderr, "[%s] backend init failed; exiting\n", cp_worker_backend_name());
        return 1;
    }
    cp_api_set_algo(NULL, cp_worker_backend_name());
    if(cp_api_device_count() == 0) cp_api_add_device(cp_worker_backend_name(), NULL);
    {
        const int contiguous = cp_worker_uses_contiguous_tiles();
        const uint64_t t_tiles =
            (uint64_t)cp_pp_num_row_parts(g_m_active, contiguous) *
            (uint64_t)cp_pp_num_col_parts(g_n_active, contiguous);
        cp_fee_set_tiles_per_matrix(t_tiles);
    }
    cp_mine_init_host_buffers();

    if(g_mock)
        return cp_mine_mock();

    const int rc = cp_run_pearl_pool(opt.pool_host, opt.pool_port);

    cp_mine_free_host_buffers();
    cp_worker_shutdown();
    return rc;
}
