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
#include "cp_noise.h"

/* cp_noise.c polls this between rows; the self-test never cancels. */
int cp_job_should_cancel(void) { return 0; }

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
static uint32_t rotr32(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

static void b3_g(uint32_t *v, int a, int b, int c, int d, uint32_t x, uint32_t y) {
    v[a] += v[b] + x; v[d] = rotr32(v[d] ^ v[a], 16); v[c] += v[d]; v[b] = rotr32(v[b] ^ v[c], 12);
    v[a] += v[b] + y; v[d] = rotr32(v[d] ^ v[a], 8);  v[c] += v[d]; v[b] = rotr32(v[b] ^ v[c], 7);
}

/* Fold 32 milestone XORs, keyed BLAKE3 compress, digest <= bound (cp_jackpot.hpp). */
static int ref_jackpot(const uint32_t *x, const uint32_t key[8], const uint32_t bound[8]) {
    static const uint32_t iv[4] = {0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au};
    static const uint8_t perm[16] = {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8};
    uint32_t m[16], v[16], t[16];
    for (int i = 0; i < 16; i++) m[i] = rotr32(x[i], 19) ^ x[i + 16];
    for (int i = 0; i < 8; i++) v[i] = key[i];
    for (int i = 0; i < 4; i++) v[8 + i] = iv[i];
    v[12] = 0; v[13] = 0; v[14] = 64; v[15] = 0x1B;
    for (int r = 0; r < 7; r++) {
        b3_g(v, 0, 4, 8, 12, m[0], m[1]);   b3_g(v, 1, 5, 9, 13, m[2], m[3]);
        b3_g(v, 2, 6, 10, 14, m[4], m[5]);  b3_g(v, 3, 7, 11, 15, m[6], m[7]);
        b3_g(v, 0, 5, 10, 15, m[8], m[9]);  b3_g(v, 1, 6, 11, 12, m[10], m[11]);
        b3_g(v, 2, 7, 8, 13, m[12], m[13]); b3_g(v, 3, 4, 9, 14, m[14], m[15]);
        for (int i = 0; i < 16; i++) t[i] = m[perm[i]];
        memcpy(m, t, sizeof(m));
    }
    for (int w = 7; w >= 0; w--) {
        const uint32_t d = v[w] ^ v[w + 8];
        if (d < bound[w]) return 1;
        if (d > bound[w]) return 0;
    }
    return 1;
}

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

static uint64_t now_us(void);

/* Noise seed + label as the 16 words set_*_gen take. */
static void seed_label_words(const uint8_t seed[32], const uint8_t label[32], uint32_t out[16]) {
    memcpy(out, seed, 32);
    memcpy(out + 8, label, 32);
}

/* Records of the packed panel against all of B (one gemm_xor over the full width). */
static int panel_records(remote_handle64 h, int n, uint32_t *xr, size_t xr_words) {
    uint64 us = 0;
    return pearlx_gemm_xor(h, 0, n, xr, (int)xr_words, &us);
}

/* -G 1: DSP-generated noisy B and A panel (set_b_gen, set_a_gen) against the host's
 * cp_noise.c (pearl_build_noisy_b, pearl_fuse_noise_row_a_buf, then set_b / set_a), by
 * comparing every XOR record of the panel against all of B. */
static int gen_test(remote_handle64 h, int n, int k, int rows, int row0) {
    const int nms = k / MS_K, lines = (nms * 2 + 31) / 32;
    const size_t xr_words = (size_t)(n / 128) * (rows / 4) * lines * 32;
    int8_t *bt = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, (size_t)n * k);
    int8_t *sig = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, (size_t)rows * k);
    int8_t *noisy = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, (size_t)rows * k);
    uint32_t *ref = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, xr_words * 4);
    uint32_t *got = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, xr_words * 4);
    uint32_t *pairs_a = malloc((size_t)k * 8), *pairs_b = malloc((size_t)k * 8);
    if (!bt || !sig || !noisy || !ref || !got || !pairs_a || !pairs_b) {
        printf("ERROR: allocation failed\n");
        return 1;
    }
    uint8_t a_seed[32], b_seed[32];
    uint32_t sd = 12345;
    for (int i = 0; i < 32; i++) {
        a_seed[i] = (uint8_t)(lcg(&sd) >> 24);
        b_seed[i] = (uint8_t)(lcg(&sd) >> 24);
    }
    /* Signal A in the miner's range [-64, 63]. */
    for (size_t i = 0; i < (size_t)rows * k; i++)
        sig[i] = (int8_t)((int)(lcg(&sd) >> 25) - 64);
    pearl_build_perm_pairs_a(a_seed, k, 128, pairs_a);
    pearl_build_perm_pairs_b(b_seed, k, 128, pairs_b);
    uint32_t a_sl[16], b_sl[16];
    seed_label_words(a_seed, PEARL_SEED_LABEL_A, a_sl);
    seed_label_words(b_seed, PEARL_SEED_LABEL_B, b_sl);
    printf("gen test: n=%d k=%d, panel rows %d..%d\n", n, k, row0, row0 + rows - 1);

    int err = 0;
    uint64 us = 0, b_ref_dsp = 0, b_gen_dsp = 0, a_ref_dsp = 0, a_gen_dsp = 0;
    /* Reference: host noise, DSP packing. */
    uint64_t t0 = now_us();
    pearl_build_noisy_b(n, k, 128, b_seed, NULL, bt);
    const uint64_t b_host = now_us() - t0;
    if ((err = pearlx_set_b(h, bt, n * k, n, k, &b_ref_dsp))) goto fail;
    t0 = now_us();
    {
        int8_t el[128], nr[4096];
        for (int r = 0; r < rows; r++)
            pearl_fuse_noise_row_a_buf(row0 + r, k, 128, a_seed, pairs_a, sig + (size_t)r * k,
                                       noisy + (size_t)r * k, el, nr);
    }
    const uint64_t a_host = now_us() - t0;
    if ((err = pearlx_set_a(h, noisy, rows * k, rows, 0, &a_ref_dsp))) goto fail;
    if ((err = panel_records(h, n, ref, xr_words))) goto fail;

    /* Both generated on the DSP. */
    if ((err = pearlx_set_b_gen(h, n, k, b_sl, 16, pairs_b, 2 * k, &b_gen_dsp))) goto fail;
    if ((err = pearlx_set_a_gen(h, sig, rows * k, row0, rows, 0, a_sl, 16, pairs_a, 2 * k,
                                &a_gen_dsp)))
        goto fail;
    if ((err = panel_records(h, n, got, xr_words))) goto fail;
    size_t bad = 0;
    for (size_t i = 0; i < xr_words; i++)
        bad += got[i] != ref[i];
    printf("  host noise: B %.1f ms, A panel %.1f ms (CPU)\n", b_host / 1e3, a_host / 1e3);
    printf("  set_b (pack) %.1f ms vs set_b_gen %.1f ms; set_a (pack) %.2f ms vs set_a_gen %.2f ms"
           " (DSP)\n", b_ref_dsp / 1e3, b_gen_dsp / 1e3, a_ref_dsp / 1e3, a_gen_dsp / 1e3);
    printf("gen check: %s (%zu of %zu XOR words differ)\n", bad ? "FAIL" : "PASS", bad, xr_words);
    if (bad) {
        /* Which half is wrong: generated A with the reference B, and the other way round. */
        size_t bad_a = 0, bad_b = 0;
        if ((err = pearlx_set_b(h, bt, n * k, n, k, &us))) goto fail;
        if ((err = pearlx_set_a_gen(h, sig, rows * k, row0, rows, 0, a_sl, 16, pairs_a, 2 * k, &us)))
            goto fail;
        if ((err = panel_records(h, n, got, xr_words))) goto fail;
        for (size_t i = 0; i < xr_words; i++) bad_a += got[i] != ref[i];
        if ((err = pearlx_set_b_gen(h, n, k, b_sl, 16, pairs_b, 2 * k, &us))) goto fail;
        if ((err = pearlx_set_a(h, noisy, rows * k, rows, 0, &us))) goto fail;
        if ((err = panel_records(h, n, got, xr_words))) goto fail;
        for (size_t i = 0; i < xr_words; i++) bad_b += got[i] != ref[i];
        printf("  set_a_gen alone: %zu words differ; set_b_gen alone: %zu words differ\n",
               bad_a, bad_b);
    }
    return bad ? 1 : 0;
fail:
    printf("ERROR 0x%x in gen test\n", err);
    return 1;
}

/* -H 1: chunk_cvs on the DSP (random data, and all zeros) against the host's keyed matrix
 * hash: the root built from the DSP's chunk hashes must equal pearl_keyed_digest_int8. */
static int hash_test(remote_handle64 h, size_t bytes) {
    int8_t *data = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, bytes);
    uint32_t *cvs = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, bytes / 1024 * 32);
    PearlMatrixHash *tree = pearl_matrix_hash_create();
    if (!data || !cvs || !tree) {
        printf("ERROR: allocation failed\n");
        return 1;
    }
    uint8_t key[32];
    uint32_t sd = 777, key8[8];
    for (int i = 0; i < 32; i++) key[i] = (uint8_t)(lcg(&sd) >> 24);
    memcpy(key8, key, 32);
    int bad = 0;
    for (int zero = 0; zero < 2; zero++) {
        if (zero) memset(data, 0, bytes);
        else fill(data, bytes, 31);
        uint64 dsp_us = 0;
        int err = pearlx_chunk_cvs(h, zero ? NULL : data, zero ? 0 : (int)bytes, (int)bytes, key8, 8,
                                   cvs, (int)(bytes / 1024 * 8), &dsp_us);
        if (err) {
            printf("ERROR 0x%x: chunk_cvs\n", err);
            return 1;
        }
        uint8_t root[32], ref[32];
        const uint64_t t0 = now_us();
        pearl_matrix_hash_from_cvs(tree, data, bytes, key, (const uint8_t *)cvs, root);
        const uint64_t tree_us = now_us() - t0;
        const uint64_t t1 = now_us();
        pearl_keyed_digest_int8(data, bytes, key, ref);
        const uint64_t host_us = now_us() - t1;
        const int ok = memcmp(root, ref, 32) == 0;
        printf("hash check (%s, %zu MiB): %s; DSP chunk hashes %.1f ms + host tree %.1f ms, "
               "host full hash %.1f ms\n", zero ? "zeros" : "random", bytes >> 20,
               ok ? "PASS" : "FAIL", dsp_us / 1e3, tree_us / 1e3, host_us / 1e3);
        bad += !ok;
    }
    return bad ? 1 : 0;
}

int main(int argc, char *argv[]) {
    int n = 4096, k = 4096, rows = 1024, cols = 4096, panels = 4, threads = 0, check = 1;
    int samples = 4096, jackpot = 1, hit_bits = 14, xr_uncached = 0, gen = 0, hash = 0;
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
        else if (!strcmp(argv[a], "-j")) jackpot = v;
        else if (!strcmp(argv[a], "-b")) hit_bits = v;
        else if (!strcmp(argv[a], "-u")) xr_uncached = v;
        else if (!strcmp(argv[a], "-G")) gen = v;
        else if (!strcmp(argv[a], "-H")) hash = v;
        else { printf("unknown option %s\n", argv[a]); return 1; }
    }
    printf("Usage: pearlx_test [-n N] [-k K] [-r rows/panel] [-w cols/call] [-g panels]"
           " [-t threads] [-c 0|1] [-s samples/call] [-j 0|1] [-b hit_bits] [-u 0|1]"
           " [-G 0|1] [-H 0|1]\n");
    /* scan cross-check: a fixed key and a bound hitting 1 in 2^hit_bits hash tiles. */
    uint32_t key_bound[16];
    for (int i = 0; i < 8; i++) key_bound[i] = 0x9E3779B9u * (uint32_t)(i + 1);
    for (int i = 0; i < 7; i++) key_bound[8 + i] = 0xFFFFFFFFu;
    key_bound[15] = 0xFFFFFFFFu >> (hit_bits & 31);
    uint32_t scan_hit[3 + 32];
    uint64_t best_scan = UINT64_MAX, best_scan_dsp = UINT64_MAX;
    size_t scan_calls = 0, scan_bad = 0, scan_hits = 0;
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
    /* -u 1: XOR records in an uncached buffer (no per-call CPU cache maintenance). */
    uint32_t *xr = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM,
                                xr_uncached ? RPCMEM_FLAG_UNCACHED : RPCMEM_DEFAULT_FLAGS,
                                xr_words * 4);
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

    if (hash) {
        err = hash_test(h, (size_t)n * k);
        pearlx_close(h);
        printf("%s\n", err ? "FAILED" : "Success");
        return err ? 1 : 0;
    }
    if (gen) {
        err = gen_test(h, n, k, rows, 3 * rows);
        pearlx_close(h);
        printf("%s\n", err ? "FAILED" : "Success");
        return err ? 1 : 0;
    }

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
            if (jackpot && k == 4096) {
                /* Host jackpot over this call's XORs: every hit, and the first in
                 * (column tile, row tile, half) order. */
                const uint64_t tc = now_us();
                size_t host_hits = 0;
                int first_row = -1, first_col = -1;
                const uint32_t *first_words = NULL;
                uint32_t x[32];
                for (int nt = 0; nt < Ntc; nt++)
                    for (int t = 0; t < Mt; t++)
                        for (int half = 0; half < 2; half++) {
                            const uint32_t *rec = xr + ((size_t)nt * Mt + t) * lines * 32;
                            for (int ms = 0; ms < 32; ms++) x[ms] = rec[ms * 2 + half];
                            if (!ref_jackpot(x, key_bound, key_bound + 8)) continue;
                            if (host_hits++ == 0) {
                                first_row = 4 * t;
                                first_col = c * cols + 128 * nt + 64 * half;
                                first_words = rec;
                            }
                        }
                check_us += now_us() - tc;
                const uint64_t ts = now_us();
                uint64 sdsp = 0;
                if ((err = pearlx_scan(h, c * cols, cols, key_bound, 16, 0, scan_hit, 3 + 32, &sdsp))) {
                    printf("ERROR 0x%x: scan\n", err);
                    goto bail;
                }
                const uint64_t scall = now_us() - ts;
                check_us += scall;
                if (scall < best_scan) best_scan = scall;
                if (sdsp < best_scan_dsp) best_scan_dsp = sdsp;
                scan_calls++;
                scan_hits += host_hits;
                int ok = scan_hit[0] == host_hits;
                if (ok && host_hits) {
                    ok = (int)scan_hit[1] == first_row && (int)scan_hit[2] == first_col;
                    for (int ms = 0; ok && ms < 32; ms++)
                        ok = scan_hit[3 + ms] == first_words[ms * 2 + (first_col / 64) % 2];
                }
                if (!ok) {
                    if (scan_bad < 5)
                        printf("  SCAN MISMATCH panel %d call %d: dsp %u hits (first %u,%u), host %zu"
                               " (first %d,%d)\n", g, c, scan_hit[0], scan_hit[1], scan_hit[2],
                               host_hits, first_row, first_col);
                    scan_bad++;
                }
            }
        }
        const uint64_t panel = now_us() - tp - check_us;
        if (panel < best_panel) best_panel = panel;
    }
    const double call_ops = 2.0 * rows * cols * k, panel_ops = 2.0 * rows * n * k;
    printf("set_a: best %.2f ms\n", best_seta / 1e3);
    if (scan_calls) {
        printf("scan (DSP jackpot): best %.2f ms call (%.1f GOPS), %.2f ms DSP (%.1f GOPS)\n",
               best_scan / 1e3, 2.0 * rows * cols * k / (best_scan * 1e3), best_scan_dsp / 1e3,
               2.0 * rows * cols * k / (best_scan_dsp * 1e3));
        printf("scan check: %s (%zu calls, %zu hits on the host, %zu calls differ)\n",
               scan_bad ? "FAIL" : "PASS", scan_calls, scan_hits, scan_bad);
        if (scan_bad) err = AEE_EFAILED;
    }
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
