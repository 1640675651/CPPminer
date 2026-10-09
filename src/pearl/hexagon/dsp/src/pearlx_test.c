// Android self-test of the pearlx DSP library: packs a random B^T, then for
// each panel of random A rows packs it (set_a) and runs every column block
// (gemm_xor); checks hash-tile XORs against a CPU reference and reports rates.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "AEEStdErr.h"
#include "remote.h"
#include "rpcmem.h"
#include "pearlx.h"

// Weak so the binary still loads on devices whose libcdsprpc lacks it.
#pragma weak remote_session_control

#define MS_K 128   // values of k per milestone

static uint64_t now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000u + ts.tv_nsec / 1000;
}

static uint32_t lcg(uint32_t *s) {
    *s = *s * 1664525u + 1013904223u;
    return *s;
}

// Full int8 range except -128 (Pearl noisy values are within [-127, 127]).
static void fill(int8_t *p, size_t n, uint32_t seed) {
    for (size_t i = 0; i < n; i++) {
        int v = (int8_t)(lcg(&seed) >> 24);
        p[i] = (int8_t)(v == -128 ? 127 : v);
    }
}

// XOR of hash tile (rows r0.., columns c0..c0+63) after each milestone.
static void ref_tile(const int8_t *A, const int8_t *Bt, int k, int r0, int c0, uint32_t *out) {
    const int nms = k / MS_K;
    memset(out, 0, nms * sizeof(uint32_t));
    for (int i = r0; i < r0 + 4; i++)
        for (int j = c0; j < c0 + 64; j++) {
            const int8_t *a = A + (size_t)i * k, *b = Bt + (size_t)j * k;
            int32_t acc = 0;
            for (int ms = 0; ms < nms; ms++) {
                for (int kk = ms * MS_K; kk < (ms + 1) * MS_K; kk++)
                    acc += a[kk] * b[kk];
                out[ms] ^= (uint32_t)acc;
            }
        }
}

int main(int argc, char *argv[]) {
    int n = 4096, k = 4096, rows = 1024, cols = 4096, panels = 4, threads = 0, check = 1;
    int samples = 4096;
    for (int a = 1; a + 1 < argc; a += 2) {
        int v = atoi(argv[a + 1]);
        if (!strcmp(argv[a], "-n")) n = v;
        else if (!strcmp(argv[a], "-k")) k = v;
        else if (!strcmp(argv[a], "-r")) rows = v;
        else if (!strcmp(argv[a], "-w")) cols = v;
        else if (!strcmp(argv[a], "-g")) panels = v;
        else if (!strcmp(argv[a], "-t")) threads = v;
        else if (!strcmp(argv[a], "-c")) check = v;
        else if (!strcmp(argv[a], "-s")) samples = v;
        else { printf("unknown option %s\n", argv[a]); return 1; }
    }
    printf("Usage: pearlx_test [-n N] [-k K] [-r rows/panel] [-w cols/call] [-g panels]"
           " [-t threads] [-c 0|1] [-s samples/call]\n");
    if (cols > n) cols = n;
    if (n <= 0 || n % 128 || k <= 0 || k % 128 || rows <= 0 || rows % 4 || cols <= 0 ||
        cols % 128 || n % cols || panels <= 0) {
        printf("n, k, cols must be multiples of 128 (cols dividing n); rows a multiple of 4\n");
        return 1;
    }
    const int nms = k / MS_K, lines = (nms * 2 + 31) / 32, Ntc = cols / 128, Mt = rows / 4;
    const int calls = n / cols;
    const size_t xr_words = (size_t)Ntc * Mt * lines * 32;

    int err = 0;
    remote_handle64 h = -1;
    struct remote_rpc_control_unsigned_module ud = { .domain = CDSP_DOMAIN_ID, .enable = 1 };
    if (!remote_session_control ||
        (err = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &ud, sizeof(ud)))) {
        printf("ERROR 0x%x: could not enable unsigned PD\n", err);
        return 1;
    }
    int8_t *Bt = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, (size_t)n * k);
    int8_t *A = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, (size_t)rows * k);
    uint32_t *xr = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, xr_words * 4);
    if (!Bt || !A || !xr) {
        printf("ERROR: rpcmem_alloc failed\n");
        return 1;
    }
    if ((err = pearlx_open(pearlx_URI CDSP_DOMAIN, &h))) {
        printf("ERROR 0x%x: pearlx_open\n", err);
        return 1;
    }
    int hvx = 0, vote = 0, mhz = 0, vtcm = 0;
    pearlx_info(h, &hvx, &vote, &mhz, &vtcm);
    printf("pearlx: n=%d k=%d, %d panels of %d rows, %d columns per call; DSP %d HVX, %d MHz, "
           "VTCM %d KB, vote %s\n", n, k, panels, rows, cols, hvx, mhz, vtcm, vote ? "FAILED" : "ok");

    fill(Bt, (size_t)n * k, 2);
    uint64 dsp_us = 0;
    uint64_t t0 = now_us();
    if ((err = pearlx_set_b(h, Bt, n * k, n, k, &dsp_us))) {
        printf("ERROR 0x%x: set_b\n", err);
        goto bail;
    }
    printf("set_b: %.1f ms call, %.1f ms on the DSP\n", (now_us() - t0) / 1e3, dsp_us / 1e3);

    uint64_t best_call = UINT64_MAX, best_dsp = UINT64_MAX, best_panel = UINT64_MAX;
    uint64_t best_seta = UINT64_MAX;
    size_t checked = 0, bad = 0;
    uint32_t sseed = 7;
    uint32_t ref[64];
    for (int g = 0; g < panels; g++) {
        fill(A, (size_t)rows * k, 100 + g);
        const uint64_t tp = now_us();
        if ((err = pearlx_set_a(h, A, rows * k, rows, threads, &dsp_us))) {
            printf("ERROR 0x%x: set_a\n", err);
            goto bail;
        }
        if (now_us() - tp < best_seta) best_seta = now_us() - tp;
        uint64_t check_us = 0;
        for (int c = 0; c < calls; c++) {
            memset(xr, 0x55, xr_words * 4);
            t0 = now_us();
            if ((err = pearlx_gemm_xor(h, c * cols, cols, xr, (int)xr_words, &dsp_us))) {
                printf("ERROR 0x%x: gemm_xor\n", err);
                goto bail;
            }
            uint64_t call = now_us() - t0;
            if (call < best_call) best_call = call;
            if (dsp_us < best_dsp) best_dsp = dsp_us;
            if (!check)
                continue;
            // Every hash tile when small, else a random sample.
            const uint64_t tc = now_us();
            const size_t tiles = (size_t)Mt * Ntc * 2;
            const size_t todo = tiles <= (size_t)samples ? tiles : (size_t)samples;
            for (size_t s = 0; s < todo; s++) {
                size_t id = todo == tiles ? s : lcg(&sseed) % tiles;
                int t = (int)(id / (Ntc * 2)), nt = (int)(id / 2 % Ntc), half = (int)(id % 2);
                const int col = c * cols + 128 * nt + 64 * half;
                ref_tile(A, Bt, k, 4 * t, col, ref);
                const uint32_t *rec = xr + ((size_t)nt * Mt + t) * lines * 32;
                for (int ms = 0; ms < nms; ms++)
                    if (rec[ms * 2 + half] != ref[ms]) {
                        if (bad < 5)
                            printf("  MISMATCH panel %d tile rows %d cols %d ms %d: dsp %08x ref %08x\n",
                                   g, 4 * t, col, ms, rec[ms * 2 + half], ref[ms]);
                        bad++;
                    }
                checked++;
            }
            check_us += now_us() - tc;
        }
        const uint64_t panel = now_us() - tp - check_us;
        if (panel < best_panel) best_panel = panel;
    }
    const double call_ops = 2.0 * rows * cols * k, panel_ops = 2.0 * rows * n * k;
    printf("set_a: best %.2f ms\n", best_seta / 1e3);
    printf("gemm_xor: best %.2f ms call (%.1f GOPS), %.2f ms DSP (%.1f GOPS)\n", best_call / 1e3,
           call_ops / (best_call * 1e3), best_dsp / 1e3, call_ops / (best_dsp * 1e3));
    printf("row panel (set_a + %d calls): best %.2f ms (%.1f GOPS)\n", calls, best_panel / 1e3,
           panel_ops / (best_panel * 1e3));
    if (check)
        printf("check: %s (%zu hash tiles x %d milestones checked, %zu words differ)\n",
               bad ? "FAIL" : "PASS", checked, nms, bad);
    if (bad)
        err = AEE_EFAILED;

bail:
    pearlx_close(h);
    rpcmem_free(A);
    rpcmem_free(Bt);
    rpcmem_free(xr);
    printf("%s\n", err ? "FAILED" : "Success");
    return err ? 1 : 0;
}
