#ifndef CP_PEARLX_CLIENT_H
#define CP_PEARLX_CLIENT_H

/* Host side of the pearlx cDSP library (dsp/inc/pearlx.idl), without the Hexagon SDK:
 * libcdsprpc.so is loaded at run time and the pearlx methods are marshalled by hand
 * (FastRPC remote_arg ABI). The DSP library itself is libpearlx_skel.so, found through
 * ADSP_LIBRARY_PATH. */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CpPearlx CpPearlx;

/* Loads libcdsprpc.so, enables the unsigned protection domain and opens pearlx on the
 * cDSP. Returns NULL (with a message on stderr) on failure. */
CpPearlx* cp_pearlx_open(void);
void cp_pearlx_close(CpPearlx* px);

/* Buffers shared with the DSP without copies (rpcmem). An uncached buffer needs no CPU
 * cache maintenance per call (FastRPC otherwise cleans or invalidates the whole buffer on
 * every call that passes it), at the price of slower CPU access. */
void* cp_pearlx_alloc(CpPearlx* px, size_t bytes);
void* cp_pearlx_alloc_uncached(CpPearlx* px, size_t bytes);
void cp_pearlx_free(CpPearlx* px, void* p);

/* See pearlx.idl. Return 0 on success, else the FastRPC / AEE error. */
int cp_pearlx_set_b(CpPearlx* px, const int8_t* bt, int n, int k, uint64_t* dsp_us);
int cp_pearlx_set_a(CpPearlx* px, const int8_t* a, int rows, int k, int nthreads,
                    uint64_t* dsp_us);
int cp_pearlx_gemm_xor(CpPearlx* px, int col0, int ncols, uint32_t* xr, size_t xr_words,
                       uint64_t* dsp_us);
/* Noise generated and packed on the DSP (seed_label: noise seed then label, 8 words each;
 * pairs: the k permutation pairs). set_a_gen's a_sig is the panel's signal rows. */
int cp_pearlx_set_b_gen(CpPearlx* px, int n, int k, const uint32_t seed_label[16],
                        const uint32_t* pairs, uint64_t* dsp_us);
int cp_pearlx_set_a_gen(CpPearlx* px, const int8_t* a_sig, int row0, int rows, int k,
                        int nthreads, const uint32_t seed_label[16], const uint32_t* pairs,
                        uint64_t* dsp_us);
/* Keyed BLAKE3 chaining values of the 1 KB chunks of raw_len bytes (a multiple of 32 KB;
 * data NULL: all zeros) into cvs (8 words per chunk). data and cvs 128-byte aligned. */
int cp_pearlx_chunk_cvs(CpPearlx* px, const int8_t* data, size_t raw_len, const uint32_t key[8],
                        uint32_t* cvs, uint64_t* dsp_us);
int cp_pearlx_info(CpPearlx* px, int* hvx_contexts, int* clock_vote_err, int* clock_mhz,
                   int* vtcm_kb);

/* scan_run's control block (pearlx.idl): 64 words, 128-byte aligned, uncached. The DSP
 * writes the first half, the host the second; neither writes the other's half. */
#define CP_PEARLX_CTL_WORDS 64
enum {
    CP_PEARLX_CTL_LAUNCHED = 0,    /* DSP: launches finished (launch s done: s + 1) */
    CP_PEARLX_CTL_EPOCH_ACK = 1,   /* DSP: the epoch it read at the start of the call */
    CP_PEARLX_CTL_STATE = 2,       /* DSP: 1 running, 2 done, 3 stopped, 4 failed */
    CP_PEARLX_CTL_CHECKED = 32,    /* host: launches checked (their slots are free) */
    CP_PEARLX_CTL_STOP = 33,       /* host: nonzero stops the DSP at the next column tile */
    CP_PEARLX_CTL_EPOCH = 34,      /* host: changed per attempt, so stale counts are ignored */
    CP_PEARLX_CTL_TEST_IDLE_US = 35 /* host, self-test only: DSP idles this long per launch */
};

/* The attempt's launches over rows [row0, row0 + rows) of A without a call per launch:
 * a_sig is those rows of signal A; panels of panel_rows, launches of launch_cols columns
 * numbered from s0 into nbuf record slots of xr (xr_words in all). The call returns when
 * the rows are done or the host sets the stop word; launched counts this call's launches.
 * Follow it from another thread through ctl. */
int cp_pearlx_scan_run(CpPearlx* px, const int8_t* a_sig, int row0, int rows, int k,
                       int panel_rows, int launch_cols, int s0, int nbuf, int nthreads,
                       const uint32_t seed_label[16], const uint32_t* pairs, uint32_t* ctl,
                       uint32_t* xr, size_t xr_words, uint64_t* dsp_us, uint64_t* wait_us,
                       int* launched);

#ifdef __cplusplus
}
#endif

#endif /* CP_PEARLX_CLIENT_H */
