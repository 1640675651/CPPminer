#include "cp_options.h"
#include "cp_algo.h"
#include "cp_cli.h"
#include "cp_config.h"
#include "cp_diagnostics.h"
#include "cp_state.h"
#include "cp_worker.h"

#if defined(CP_ENABLE_ONEDNN) && CP_ENABLE_ONEDNN
#include "cp_onednn_worker.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <stdexcept>
#include <string>

static void print_usage(void)
{
    printf("CPPminer - multi-algo LuckyPool miner (pearl / quantus)\n");
    printf("  --algo NAME        pearl (default) or quantus\n");
    {
        char pb[64], qb[64];
        cp_algo_format_backends(CP_ALGO_PEARL, pb, (int)sizeof(pb));
        cp_algo_format_backends(CP_ALGO_QUANTUS, qb, (int)sizeof(qb));
        printf("                     pearl backends: %s\n", pb);
        printf("                     quantus backends: %s\n", qb);
    }
    printf("  --pool URI         stratum+tcp://host:port (required for quantus unless --mock)\n");
    printf("  --wallet ADDR      wallet address\n");
    printf("  --worker NAME      worker name (default: rig01)\n");
    printf("  --agent NAME       agent string (default: cppminer/0.5)\n");
    printf("  --pool-pass STR    mining.authorize password (default: x); Kryptex custom\n");
    printf("                     share difficulty: d=N (default d=2097152)\n");
    printf("  --api-port N       HTTP stats API on this port: /summary (JSON) and /hiveos\n");
    printf("  --api-bind ADDR    API listen address (default: 127.0.0.1; 0.0.0.0 for all)\n");
    printf("  --backend NAME     cpu");
#if defined(CP_ENABLE_CUDA) && CP_ENABLE_CUDA
    printf("|cuda");
#endif
#if defined(CP_ENABLE_OPENCL) && CP_ENABLE_OPENCL
    printf("|opencl");
#endif
#if defined(CP_ENABLE_ONEDNN) && CP_ENABLE_ONEDNN
    printf("|onednn");
#endif
#if defined(CP_ENABLE_WGPU) && CP_ENABLE_WGPU
    printf("|wgpu");
#endif
#if defined(CP_ENABLE_HEXAGON) && CP_ENABLE_HEXAGON
    printf("|hexagon");
#endif
#if defined(CP_ENABLE_INTEL_NPU) && CP_ENABLE_INTEL_NPU
    printf("|intel_npu");
#endif
    printf(" (built: ");
    {
        int first = 1;
        if(cp_worker_has_cpu()){ printf("%scpu", first ? "" : ","); first = 0; }
        if(cp_worker_has_cuda()){ printf("%scuda", first ? "" : ","); first = 0; }
        if(cp_worker_has_opencl()){ printf("%sopencl", first ? "" : ","); first = 0; }
        if(cp_worker_has_onednn()){ printf("%sonednn", first ? "" : ","); first = 0; }
        if(cp_worker_has_wgpu()){ printf("%swgpu", first ? "" : ","); first = 0; }
        if(cp_worker_has_hexagon()){ printf("%shexagon", first ? "" : ","); first = 0; }
        if(cp_worker_has_intel_npu()){ printf("%sintel_npu", first ? "" : ","); first = 0; }
        if(first) printf("none");
    }
    printf(")\n");
    printf("  --devices N[,M]    device index(es): CUDA ids, OpenCL flat index,\n");
    printf("                     or wgpu mining-adapter indices (--list-devices)\n");
    printf("                     (default: 0; OpenCL prefers discrete GPU first)\n");
    printf("  --list-devices     list devices for the selected backend and exit\n");
#if defined(CP_ENABLE_ONEDNN) && CP_ENABLE_ONEDNN
    printf("  --ocl-platform P   OpenCL/OneDNN: only enumerate platform index P\n");
#else
#if defined(CP_ENABLE_OPENCL) && CP_ENABLE_OPENCL
    printf("  --ocl-platform P   OpenCL: only enumerate platform index P\n");
#endif
#endif
#if defined(CP_ENABLE_OPENCL) && CP_ENABLE_OPENCL
    printf("  --ocl-tile MxN[/MmMm]  OpenCL register tile: 4x4, 4x8 (default), 8x8, 8x16 (auto on AMD);\n");
    printf("                         optional /64x64 or /128x128 macro (same as --ocl-macro)\n");
    printf("  --ocl-macro MxN    OpenCL macro block: 64x64 or 128x128 (default 128x128)\n");
    printf("  --ocl-issue MODE   OpenCL GEMM issue: auto (default), broadcast, or packed\n");
    printf("  --ocl-dot MODE     OpenCL dot backend: auto (default), sudot, sdot4, khr,\n");
    printf("                     force-khr, asm, or off\n");
    printf("  --ocl-cpm-type T   OpenCL scalar (no int8 dot) MAC: auto (default), float, int,\n");
    printf("                     or mad24 (auto = mad24 on AMD GCN gfx6-8, else float)\n");
    printf("  --ocl-lds on|off   OpenCL stage A/B in local memory (default off)\n");
    printf("  --ocl-double-buff on|off  OpenCL register double-buffered GEMM loop (default off;\n");
    printf("                     faster on some GPUs, more registers: may spill on others)\n");
#endif
#if defined(CP_ENABLE_WGPU) && CP_ENABLE_WGPU
    printf("  --wgpu-lds on|off  wgpu (pearl) stage A/B in workgroup memory\n");
    printf("                     (default: on for discrete GPUs, off for integrated)\n");
    printf("  --wgpu-tile MxN[/MmMm]  wgpu (pearl) register tile: 4x4, 4x8, 8x8 (default), 8x16;\n");
    printf("                     optional /64x64 or /128x128 macro (same as --wgpu-macro)\n");
    printf("  --wgpu-macro MxN   wgpu (pearl) macro block: 64x64 or 128x128 (default 128x128)\n");
#endif
#if defined(CP_ENABLE_INTEL_NPU) && CP_ENABLE_INTEL_NPU
    printf("  intel_npu (pearl): Intel NPU (Core Ultra) via OpenVINO and the patched NPU compiler,\n");
    printf("                     16x16 hash tiles; launch = --row-period-batch x --batch-size 128x128\n");
    printf("                     macro blocks (default %d x %d, columns <= 4, rows above 4 in\n",
           CP_INTEL_NPU_ROW_MACROS_DEFAULT, CP_INTEL_NPU_COL_MACROS_DEFAULT);
    printf("                     steps of 4; each new shape compiles once); --fused-jackpot runs the\n");
    printf("                     jackpot on the NPU's SHAVEs (almost no CPU); CP_NPU_REQUESTS =\n");
    printf("                     launches queued (default 4), CP_NPU_CACHE_DIR = compiled-graph cache\n");
#endif
#if defined(CP_ENABLE_HEXAGON) && CP_ENABLE_HEXAGON
    printf("  hexagon (pearl): Snapdragon cDSP HVX, 4x64 hash tiles; needs about m*k + n*k\n");
    printf("                     bytes (1 GiB at the default size; smaller --m/--n on phones);\n");
    printf("                     DSP launch = --row-period-batch x --batch-size 128x128 macro\n");
    printf("                     blocks (default %d x %d); --fused-jackpot runs the jackpot on\n",
           CP_HEXAGON_LAUNCH_MACROS_DEFAULT, CP_HEXAGON_LAUNCH_MACROS_DEFAULT);
    printf("                     the DSP (~1%% slower, almost no CPU)\n");
#endif
    printf("  --m N, --n N         matrix rows/cols in units of %d (default %d; each <= %d,\n",
           CP_MATRIX_UNIT, M_DIM / CP_MATRIX_UNIT, CP_MATRIX_UNITS_MAX);
    printf("                       m*n <= %d)\n", CP_MATRIX_AREA_MAX);
#if defined(CP_ENABLE_CUDA) && CP_ENABLE_CUDA
    printf("  --no-period-gemm     per-tile scan instead of period GEMM (CUDA debug)\n");
    printf("  --batch-size N       launch batch: Pearl col/macro panel (default %d);\n",
           CP_PERIOD_BATCH_DEFAULT);
    printf("                       Quantus wgpu/OpenCL nonces per launch (default 1000000)\n");
    printf("  --period-batch N     alias for --batch-size\n");
    printf("  --col-period-batch N alias for --batch-size\n");
    printf("  --row-period-batch N row-period batch size (default %d, max %d)\n",
           CP_ROW_PERIOD_BATCH_DEFAULT, CP_ROW_PERIOD_BATCH_MAX);
    printf("  --row-major-ap       row-major Ap/BpT (lda=%d; CUTLASS default)\n",
           K_DIM);
    printf("  --step-major         step-major Ap/BpT panels (lda=%d; cuBLAS period default)\n",
           R_RANK);
    printf("  --cutlass-fused      fused CUTLASS GEMM + jackpot (CUDA default)\n");
    printf("  --cuda-mma MODE      CUTLASS warp MMA: auto (default: tensorop80 on sm_80+,\n");
    printf("                       tensorop on sm_75, dp4a SIMT below), simt, tensorop\n");
    printf("                       (mma.m8n8k16, 2-stage), tensorop80 (mma.m16n8k32,\n");
    printf("                       multistage cp.async; sm_80+), tensoropms (multistage\n");
    printf("                       + m8n8k16; A/B and sm_75 validation)\n");
#if defined(CP_ENABLE_CUBLAS) && CP_ENABLE_CUBLAS
    printf("  --cublas-period      debug: cuBLAS period GEMM + separate XOR/jackpot\n");
#endif
    printf("  --no-cutlass-fused   debug: non-CUTLASS period path (CUDA GEMM%s)\n",
#if defined(CP_ENABLE_CUBLAS) && CP_ENABLE_CUBLAS
           " or cuBLAS if probed"
#else
           ""
#endif
           );
    printf("  --cpu-gen           host matrix prep (OpenCL ~1 GiB VRAM; CUDA debug)\n");
    printf("  --align-test         run CPU/GPU hash alignment self-test and exit\n");
    printf("  --align-test-prod    include checks at the --m/--n size (~1 GiB RAM at default, slow)\n");
    printf("  --profile-scan [N]   time GEMM vs jackpot per period batch (default N=10)\n");
#endif
#if defined(CP_ENABLE_ONEDNN) && CP_ENABLE_ONEDNN
    printf("  --fused-jackpot       oneDNN in-reg fold + BLAKE3 + in-kernel jackpot (default)\n");
    printf("  --no-fused-jackpot    oneDNN GEMM + separate device jackpot kernel\n");
    printf("  --onednn-layout NAME  device A/B layout: TN (default), TT, NT, NN (C always N)\n");
    printf("                       (CASE5_GEMM_LAYOUT env when flag unset)\n");
    printf("  --row-period-batch N, --batch-size N  oneDNN panel in hash tiles (default: 8192x8192\n");
    printf("                       GEMM per panel on dp4a GPUs, 2048/4096 tiles on XMX GPUs)\n");
#endif
#if (defined(CP_ENABLE_OPENCL) && CP_ENABLE_OPENCL) || (defined(CP_ENABLE_ONEDNN) && CP_ENABLE_ONEDNN)
    printf("  --ocl-throttle L     OpenCL queue throttle hint: low, med, high, or off. low makes\n");
    printf("                       Intel's driver sleep while waiting for the GPU instead of\n");
    printf("                       spinning a core, but can stall ~70 s on Arc (default off)\n");
#endif
#if defined(CP_ENABLE_OPENCL) && CP_ENABLE_OPENCL
    printf("  --profile-prep [N]   time OpenCL matrix prep phases (default N=3)\n");
#endif
    printf("  --max-nonce N        stop after N matrix attempts per job\n");
    printf("  --dry-run            build proof but do not submit; saves pp_header.bin and\n");
    printf("                       pp_proof.b64 next to the executable (nothing else is written)\n");
    printf("  --verify             run in-process zk-pow verify before submit\n");
    printf("  --cert-version N     force certificate version for verify (1/2=legacy, 3=salted;\n");
    printf("                       default 3; without this flag, pool notify cert_version wins)\n");
    printf("  --mock / -mock       offline: fixed job, mine until first share, verify, exit\n");
    printf("                       (pearl: zk-pow; quantus: Poseidon2 hash < target)\n");
    printf("  --mock-diff D        mock difficulty (higher = longer before share).\n");
    printf("                       default pearl=%.0f (jackpot curve) /\n",
           CP_MOCK_DIFF_PEARL_DEFAULT);
    printf("                       quantus=%.0f (U512::MAX / D)\n",
           CP_MOCK_DIFF_QUANTUS_DEFAULT);
    printf("  --prepack MODE       CPU prepack: fused (default), reuse, separate\n");
    printf("  --inplace-prepack    alias for --prepack reuse\n");
    printf("  --simd ISA           CPU SIMD: auto (default), hybrid, avx512vnni, avxvnni,\n");
    printf("                       avx512 (base F+BW, alias avx512bw), avx2, ssse3, i8mm,\n");
    printf("                       dotprod, neon, scalar (also CP_SIMD / CASE33_ISA env)\n");
    printf("                       quantus: hybrid = scalar + avx2 split across SMT siblings,\n");
    printf("                       auto = best available (currently hybrid), avx2 = all\n");
    printf("                       threads AVX2, anything else = scalar; pearl: hybrid = auto\n");
    printf("  --simd-test          compare every available CPU SIMD kernel with scalar and exit\n");
    printf("  --prepack-test       check CPU fused/reuse prepack against separate (dev size) and exit\n");
    printf("  --threads N          OpenMP threads for the CPU backend (pearl and quantus).\n");
    printf("                       Default: all CPUs the process may use; OMP_NUM_THREADS\n");
    printf("                       overrides it. Threads fill physical cores first\n");
}

/* --ocl-tile / --wgpu-tile: MxN register tile, optionally followed by a /MxN macro block. */
[[maybe_unused]] static void parse_tile(const char* v, int* tile_mr, int* tile_nr,
                                        int* macro_m, int* macro_n)
{
    int mr = 0, nr = 0, mm = 0, mn = 0;
    int nfields;
    try {
        nfields = cp_cli_dimensions(v, &mr, &nr, &mm, &mn);
    } catch(const std::invalid_argument&) {
        throw std::invalid_argument(std::string("expected 4x4, 4x8, 8x8, 8x16, or MxN/64x64|128x128, got \"") +
                                    v + "\"");
    }
    if(!((mr == 4 && (nr == 4 || nr == 8)) || (mr == 8 && (nr == 8 || nr == 16))))
        throw std::invalid_argument(std::string("expected a 4x4, 4x8, 8x8, or 8x16 tile, got \"") + v + "\"");
    if(nfields == 4 && !((mm == 64 && mn == 64) || (mm == 128 && mn == 128)))
        throw std::invalid_argument(std::string("expected a 64x64 or 128x128 macro, got \"") + v + "\"");
    *tile_mr = mr;
    *tile_nr = nr;
    if(nfields == 4){
        *macro_m = mm;
        *macro_n = mn;
    }
}

/* --ocl-macro / --wgpu-macro: 64x64 or 128x128. */
[[maybe_unused]] static void parse_macro(const char* v, int* macro_m, int* macro_n)
{
    int mm = 0, mn = 0;
    try {
        cp_cli_dimensions(v, &mm, &mn);
    } catch(const std::invalid_argument&) {
        mm = mn = 0;
    }
    if(!((mm == 64 && mn == 64) || (mm == 128 && mn == 128)))
        throw std::invalid_argument(std::string("expected 64x64 or 128x128, got \"") + v + "\"");
    *macro_m = mm;
    *macro_n = mn;
}

/* The argv loop of cp_options_parse; option is the option being parsed, for error messages.
 * Returns -1 to continue, else the exit status. */
static int parse_argv(int argc, char** argv, CpOptions* o, std::string& option)
{
    for(int i = 1; i < argc; i++){
        option = cp_cli_option_name(argv[i]);
        if(cp_cli_option(argv[i], "--pool")){
            static char hbuf[256];
            o->pool_port = cp_cli_pool_uri(cp_cli_value(argc, argv, i), hbuf, sizeof(hbuf));
            o->pool_host = hbuf;
            o->pool_specified = 1;
        } else if(cp_cli_option(argv[i], "--algo")){
            const char* algo = cp_cli_value(argc, argv, i);
            if(cp_algo_parse(algo, &o->algo_sel) != 0){
                fprintf(stderr, "unknown --algo %s (want pearl|quantus)\n", algo);
                return 1;
            }
        } else if(cp_cli_option(argv[i], "--wallet")){
            o->wallet = cp_cli_value(argc, argv, i, sizeof(wallet_global));
        } else if(cp_cli_option(argv[i], "--backend")){
            const char* b = cp_cli_value(argc, argv, i);
            if(!strcmp(b, "cpu")) o->backend_sel = CP_BACKEND_CPU;
            else if(!strcmp(b, "cuda")) o->backend_sel = CP_BACKEND_CUDA;
            else if(!strcmp(b, "opencl")) o->backend_sel = CP_BACKEND_OPENCL;
            else if(!strcmp(b, "onednn")) o->backend_sel = CP_BACKEND_ONEDNN;
            else if(!strcmp(b, "wgpu")) o->backend_sel = CP_BACKEND_WGPU;
            else if(!strcmp(b, "hexagon")) o->backend_sel = CP_BACKEND_HEXAGON;
            else if(!strcmp(b, "intel_npu") || !strcmp(b, "npu")) o->backend_sel = CP_BACKEND_INTEL_NPU;
            else {
                fprintf(stderr, "unknown --backend %s\n", b);
                return 1;
            }
        } else if(cp_cli_option(argv[i], "--device") || cp_cli_option(argv[i], "--devices")){
            cp_cli_devices(cp_cli_value(argc, argv, i), o->devs, o->ndev, MAX_GPUS);
            o->devices_specified = 1;
        } else if(!strcmp(argv[i], "--list-devices")){
            o->list_devices = 1;
#if (defined(CP_ENABLE_OPENCL) && CP_ENABLE_OPENCL) || (defined(CP_ENABLE_ONEDNN) && CP_ENABLE_ONEDNN)
        } else if(cp_cli_option(argv[i], "--ocl-platform")){
            o->ocl_platform = cp_cli_int(cp_cli_value(argc, argv, i));
        } else if(cp_cli_option(argv[i], "--ocl-throttle")){
            const char* v = cp_cli_value(argc, argv, i);
            if(!strcmp(v, "low")){
                o->ocl_throttle = 3;
            } else if(!strcmp(v, "med") || !strcmp(v, "medium")){
                o->ocl_throttle = 2;
            } else if(!strcmp(v, "high")){
                o->ocl_throttle = 1;
            } else if(!strcmp(v, "off") || !strcmp(v, "none")){
                o->ocl_throttle = 0;
            } else {
                fprintf(stderr, "invalid --ocl-throttle %s (expected low, med, high, or off)\n", v);
                return 1;
            }
#endif
#if defined(CP_ENABLE_OPENCL) && CP_ENABLE_OPENCL
        } else if(cp_cli_option(argv[i], "--ocl-tile")){
            parse_tile(cp_cli_value(argc, argv, i), &o->ocl_tile_mr, &o->ocl_tile_nr,
                       &o->ocl_macro_m, &o->ocl_macro_n);
        } else if(cp_cli_option(argv[i], "--ocl-macro")){
            parse_macro(cp_cli_value(argc, argv, i), &o->ocl_macro_m, &o->ocl_macro_n);
        } else if(cp_cli_option(argv[i], "--ocl-issue")){
            const char* v = cp_cli_value(argc, argv, i);
            if(!strcmp(v, "auto")){
                o->ocl_issue_mode = 0;
            } else if(!strcmp(v, "broadcast")){
                o->ocl_issue_mode = 1;
            } else if(!strcmp(v, "packed") || !strcmp(v, "dot4")){
                o->ocl_issue_mode = 2;
            } else {
                fprintf(stderr, "invalid --ocl-issue %s (expected auto, broadcast, or packed)\n", v);
                return 1;
            }
        } else if(cp_cli_option(argv[i], "--ocl-dot")){
            const char* v = cp_cli_value(argc, argv, i);
            if(!strcmp(v, "auto")){
                o->ocl_dot_policy = 0;
            } else if(!strcmp(v, "force-khr") || !strcmp(v, "force")){
                o->ocl_dot_policy = 1;
            } else if(!strcmp(v, "off") || !strcmp(v, "scalar") || !strcmp(v, "cpm")){
                o->ocl_dot_policy = 2;
            } else if(!strcmp(v, "sudot") || !strcmp(v, "sudot4")){
                o->ocl_dot_policy = 3;
            } else if(!strcmp(v, "sdot4") || !strcmp(v, "sdot") || !strcmp(v, "builtin")){
                o->ocl_dot_policy = 4;
            } else if(!strcmp(v, "asm") || !strcmp(v, "v_dot4c")){
                o->ocl_dot_policy = 5;
            } else if(!strcmp(v, "khr") || !strcmp(v, "dpi")){
                o->ocl_dot_policy = 6;
            } else {
                fprintf(stderr,
                        "invalid --ocl-dot %s (expected auto, sudot, sdot4, khr, force-khr, asm, or off)\n",
                        v);
                return 1;
            }
        } else if(cp_cli_option(argv[i], "--ocl-cpm-type")){
            const char* v = cp_cli_value(argc, argv, i);
            if(!strcmp(v, "auto")){
                o->ocl_cpm_type = 0;
            } else if(!strcmp(v, "float") || !strcmp(v, "fp32")){
                o->ocl_cpm_type = 1;
            } else if(!strcmp(v, "int") || !strcmp(v, "int32")){
                o->ocl_cpm_type = 2;
            } else if(!strcmp(v, "mad24")){
                o->ocl_cpm_type = 3;
            } else {
                fprintf(stderr,
                        "invalid --ocl-cpm-type %s (expected auto, float, int, or mad24)\n", v);
                return 1;
            }
        } else if(cp_cli_option(argv[i], "--ocl-lds")){
            const char* v = cp_cli_value(argc, argv, i);
            if(!strcmp(v, "on") || !strcmp(v, "1") || !strcmp(v, "true")){
                o->ocl_lds = 1;
            } else if(!strcmp(v, "off") || !strcmp(v, "0") || !strcmp(v, "false")){
                o->ocl_lds = 0;
            } else {
                fprintf(stderr, "invalid --ocl-lds %s (expected on or off)\n", v);
                return 1;
            }
        } else if(cp_cli_option(argv[i], "--ocl-double-buff")){
            const char* v = cp_cli_value(argc, argv, i);
            if(!strcmp(v, "on") || !strcmp(v, "1") || !strcmp(v, "true")){
                o->ocl_double_buff = 1;
            } else if(!strcmp(v, "off") || !strcmp(v, "0") || !strcmp(v, "false")){
                o->ocl_double_buff = 0;
            } else {
                fprintf(stderr, "invalid --ocl-double-buff %s (expected on or off)\n", v);
                return 1;
            }
#endif
#if defined(CP_ENABLE_WGPU) && CP_ENABLE_WGPU
        } else if(cp_cli_option(argv[i], "--wgpu-lds")){
            const char* v = cp_cli_value(argc, argv, i);
            if(!strcmp(v, "on") || !strcmp(v, "1") || !strcmp(v, "true")){
                o->wgpu_lds = 1;
            } else if(!strcmp(v, "off") || !strcmp(v, "0") || !strcmp(v, "false")){
                o->wgpu_lds = 0;
            } else {
                fprintf(stderr, "invalid --wgpu-lds %s (expected on or off)\n", v);
                return 1;
            }
        } else if(cp_cli_option(argv[i], "--wgpu-tile")){
            parse_tile(cp_cli_value(argc, argv, i), &o->wgpu_tile_mr, &o->wgpu_tile_nr,
                       &o->wgpu_macro_m, &o->wgpu_macro_n);
        } else if(cp_cli_option(argv[i], "--wgpu-macro")){
            parse_macro(cp_cli_value(argc, argv, i), &o->wgpu_macro_m, &o->wgpu_macro_n);
#endif
        } else if(cp_cli_option(argv[i], "--m") || cp_cli_option(argv[i], "--n")){
            const int units = cp_cli_int(cp_cli_value(argc, argv, i), 1, CP_MATRIX_UNITS_MAX);
            if(option == "--m") o->m_units = units;
            else o->n_units = units;
        } else if(!strcmp(argv[i], "--no-period-gemm")){
            o->no_period_gemm = 1;
        } else if(cp_cli_option(argv[i], "--batch-size") || cp_cli_option(argv[i], "--period-batch") ||
                  cp_cli_option(argv[i], "--col-period-batch")){
            o->batch_size = cp_cli_int(cp_cli_value(argc, argv, i), 1);
            o->batch_size_set = 1;
        } else if(cp_cli_option(argv[i], "--row-period-batch")){
            o->row_period_batch = cp_cli_int(cp_cli_value(argc, argv, i), 1);
            o->row_period_batch_set = 1;
        } else if(!strcmp(argv[i], "--row-major-ap")){
            o->step_major_ap = 0;
        } else if(!strcmp(argv[i], "--step-major")){
            o->step_major_ap = 1;
        } else if(!strcmp(argv[i], "--cutlass-fused")){
            o->cutlass_fused = 1;
        } else if(!strcmp(argv[i], "--cublas-period")){
#if defined(CP_ENABLE_CUBLAS) && CP_ENABLE_CUBLAS
            o->cutlass_fused = 0;
#else
            fprintf(stderr,
                    "--cublas-period requires rebuild with -DCP_ENABLE_CUBLAS=ON "
                    "(or build.ps1 -EnableCublas)\n");
            return 1;
#endif
        } else if(!strcmp(argv[i], "--no-cutlass-fused")){
            o->cutlass_fused = 0;
        } else if(!strcmp(argv[i], "--cuda-mma") && i + 1 < argc){
            const char* v = argv[++i];
            if(!strcmp(v, "auto")) o->cuda_mma = 0;
            else if(!strcmp(v, "simt")) o->cuda_mma = 1;
            else if(!strcmp(v, "tensorop")) o->cuda_mma = 2;
            else if(!strcmp(v, "tensorop80")) o->cuda_mma = 3;
            else if(!strcmp(v, "tensoropms")) o->cuda_mma = 4;
            else {
                fprintf(stderr, "--cuda-mma requires auto, simt, tensorop, tensorop80, "
                                "or tensoropms\n");
                return 1;
            }
        } else if(!strcmp(argv[i], "--fused-jackpot")){
            o->fused_jackpot = 1;
        } else if(!strcmp(argv[i], "--no-fused-jackpot")){
            o->fused_jackpot = 0;
#if defined(CP_ENABLE_ONEDNN) && CP_ENABLE_ONEDNN
        } else if(cp_cli_option(argv[i], "--onednn-layout")){
            o->onednn_layout = cp_cli_value(argc, argv, i);
            if(!cp_onednn_worker_gemm_layout_valid(o->onednn_layout))
                throw std::invalid_argument(std::string("expected TN, TT, NT, or NN, got \"") +
                                            o->onednn_layout + "\"");
#endif
        } else if(!strcmp(argv[i], "--cpu-gen")){
            g_cpu_matrix_gen = 1;
        } else if(!strcmp(argv[i], "--inplace-prepack")){
            o->prepack_mode = CP_PREPACK_REUSE;
        } else if(cp_cli_option(argv[i], "--prepack")){
            const char* mode = cp_cli_value(argc, argv, i);
            if(!strcmp(mode, "separate"))
                o->prepack_mode = CP_PREPACK_SEPARATE;
            else if(!strcmp(mode, "reuse") || !strcmp(mode, "inplace"))
                o->prepack_mode = CP_PREPACK_REUSE;
            else if(!strcmp(mode, "fused"))
                o->prepack_mode = CP_PREPACK_FUSED;
            else {
                fprintf(stderr, "unknown --prepack mode %s (separate|reuse|fused)\n", mode);
                return 1;
            }
        } else if(cp_cli_option(argv[i], "--simd")){
            const char* isa = cp_cli_value(argc, argv, i);
            if(!strcmp(isa, "auto"))
                o->simd_isa = CP_SIMD_AUTO;
            else if(!strcmp(isa, "avx512vnni") || !strcmp(isa, "avx512-vnni"))
                o->simd_isa = CP_SIMD_AVX512VNNI;
            else if(!strcmp(isa, "avx512") || !strcmp(isa, "avx512bw") ||
                    !strcmp(isa, "avx512-bw"))
                o->simd_isa = CP_SIMD_AVX512BW;
            else if(!strcmp(isa, "avxvnni") || !strcmp(isa, "vnni") ||
                    !strcmp(isa, "avx-vnni"))
                o->simd_isa = CP_SIMD_AVXVNNI;
            else if(!strcmp(isa, "avx2"))
                o->simd_isa = CP_SIMD_AVX2;
            else if(!strcmp(isa, "sse") || !strcmp(isa, "ssse3"))
                o->simd_isa = CP_SIMD_SSE;
            else if(!strcmp(isa, "neon"))
                o->simd_isa = CP_SIMD_NEON;
            else if(!strcmp(isa, "dotprod"))
                o->simd_isa = CP_SIMD_DOTPROD;
            else if(!strcmp(isa, "i8mm"))
                o->simd_isa = CP_SIMD_I8MM;
            else if(!strcmp(isa, "scalar"))
                o->simd_isa = CP_SIMD_SCALAR;
            else if(!strcmp(isa, "hybrid"))
                o->simd_isa = CP_SIMD_HYBRID;
            else {
                fprintf(stderr,
                        "unknown --simd %s (auto|hybrid|avx512vnni|avxvnni|avx512|avx2|ssse3|i8mm|dotprod|neon|scalar)\n",
                        isa);
                return 1;
            }
        } else if(!strcmp(argv[i], "--simd-test")){
            o->simd_test = 1;
        } else if(!strcmp(argv[i], "--prepack-test")){
            o->prepack_test = 1;
        } else if(cp_cli_option(argv[i], "--max-nonce")){
            g_max_nonce = cp_cli_int(cp_cli_value(argc, argv, i));
        } else if(cp_cli_option(argv[i], "--worker")){
            cp_cli_string(argc, argv, i, worker_global, sizeof(worker_global));
        } else if(cp_cli_option(argv[i], "--agent")){
            cp_cli_string(argc, argv, i, agent_global, sizeof(agent_global));
        } else if(cp_cli_option(argv[i], "--api-port")){
            o->api_port = cp_cli_int(cp_cli_value(argc, argv, i), 1, 65535);
        } else if(cp_cli_option(argv[i], "--api-bind")){
            o->api_bind = cp_cli_value(argc, argv, i, 64);
        } else if(cp_cli_option(argv[i], "--pool-pass")){
            cp_cli_string(argc, argv, i, pool_pass_global, sizeof(pool_pass_global));
        } else if(!strcmp(argv[i], "--dry-run")){
            g_dry_run = 1;
        } else if(!strcmp(argv[i], "--verify")){
            g_plain_verify = 1;
        } else if(cp_cli_option(argv[i], "--cert-version")){
            g_cert_version = (uint32_t)cp_cli_int(cp_cli_value(argc, argv, i), 1, 3);
            g_cert_version_forced = 1;
        } else if(!strcmp(argv[i], "--mock") || !strcmp(argv[i], "-mock")){
            g_mock = 1;
        } else if(cp_cli_option(argv[i], "--mock-diff")){
            g_mock_diff = cp_cli_real(cp_cli_value(argc, argv, i), 1.0);
            g_mock_diff_forced = 1;
        } else if(!strcmp(argv[i], "--align-test")){
            o->align_test = 1;
        } else if(!strcmp(argv[i], "--align-test-prod")){
            o->align_test = 1;
            o->align_test_prod = 1;
        } else if(cp_cli_option(argv[i], "--profile-scan")){
            /* Optional count: --profile-scan=N, or a next argument that is not an option. */
            o->profile_scan = 1;
            if(strchr(argv[i], '=') || (i + 1 < argc && argv[i + 1][0] != '-'))
                o->profile_runs = cp_cli_int(cp_cli_value(argc, argv, i), 1);
        } else if(cp_cli_option(argv[i], "--profile-prep")){
            o->profile_prep = 1;
            if(strchr(argv[i], '=') || (i + 1 < argc && argv[i + 1][0] != '-'))
                o->profile_prep_runs = cp_cli_int(cp_cli_value(argc, argv, i), 1);
        } else if(!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")){
            print_usage();
            return 0;
        } else if(cp_cli_option(argv[i], "--threads")){
            const int n = cp_cli_int(cp_cli_value(argc, argv, i));
            g_qpow_threads = n;
            g_cpu_threads = n;
        } else if(!strcmp(argv[i], "--qpow-selftest")){
            return cp_qpow_selftest();
        } else {
            fprintf(stderr, "unknown option: %s\n", option.c_str());
            return 1;
        }
    }
    return -1;
}

int cp_options_parse(int argc, char** argv, CpOptions* o)
{
    int simd_env_invalid = 0;
    {
        const char* env = getenv("CP_SIMD");
        if(!env) env = getenv("CASE33_ISA");
        if(env){
            if(!strcmp(env, "avx512vnni") || !strcmp(env, "avx512-vnni"))
                o->simd_isa = CP_SIMD_AVX512VNNI;
            else if(!strcmp(env, "avx512") || !strcmp(env, "avx512bw") ||
                    !strcmp(env, "avx512-bw"))
                o->simd_isa = CP_SIMD_AVX512BW;
            else if(!strcmp(env, "avxvnni") || !strcmp(env, "vnni") || !strcmp(env, "avx-vnni"))
                o->simd_isa = CP_SIMD_AVXVNNI;
            else if(!strcmp(env, "avx2")) o->simd_isa = CP_SIMD_AVX2;
            else if(!strcmp(env, "sse") || !strcmp(env, "ssse3"))
                o->simd_isa = CP_SIMD_SSE;
            else if(!strcmp(env, "i8mm")) o->simd_isa = CP_SIMD_I8MM;
            else if(!strcmp(env, "dotprod")) o->simd_isa = CP_SIMD_DOTPROD;
            else if(!strcmp(env, "neon")) o->simd_isa = CP_SIMD_NEON;
            else if(!strcmp(env, "scalar")) o->simd_isa = CP_SIMD_SCALAR;
            else if(!strcmp(env, "hybrid")) o->simd_isa = CP_SIMD_HYBRID;
            else if(!strcmp(env, "auto")) o->simd_isa = CP_SIMD_AUTO;
            else {
                fprintf(stderr, "unknown CP_SIMD/CASE33_ISA value %s\n", env);
                simd_env_invalid = 1;
            }
        }
    }

    if(simd_env_invalid)
        return 1;

    std::string option;
    try {
        const int rc = parse_argv(argc, argv, o, option);
        if(rc >= 0) return rc;
    } catch(const std::invalid_argument& error){
        fprintf(stderr, "invalid command line %s: %s\n", option.c_str(), error.what());
        return 1;
    }

    {
        if(!o->m_units) o->m_units = M_DIM / CP_MATRIX_UNIT;
        if(!o->n_units) o->n_units = N_DIM / CP_MATRIX_UNIT;
        if(o->m_units * o->n_units > CP_MATRIX_AREA_MAX){
            fprintf(stderr, "--m %d --n %d too large: m*n must be <= %d (in units of %d^2)\n",
                    o->m_units, o->n_units, CP_MATRIX_AREA_MAX, CP_MATRIX_UNIT);
            return 1;
        }
        g_m_active = o->m_units * CP_MATRIX_UNIT;
        g_n_active = o->n_units * CP_MATRIX_UNIT;
    }
    return -1;
}
