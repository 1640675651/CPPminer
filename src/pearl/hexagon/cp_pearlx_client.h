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

/* Buffers shared with the DSP without copies (rpcmem). */
void* cp_pearlx_alloc(CpPearlx* px, size_t bytes);
void cp_pearlx_free(CpPearlx* px, void* p);

/* See pearlx.idl. Return 0 on success, else the FastRPC / AEE error. */
int cp_pearlx_set_b(CpPearlx* px, const int8_t* bt, int n, int k, uint64_t* dsp_us);
int cp_pearlx_set_a(CpPearlx* px, const int8_t* a, int rows, int k, int nthreads,
                    uint64_t* dsp_us);
int cp_pearlx_gemm_xor(CpPearlx* px, int col0, int ncols, uint32_t* xr, size_t xr_words,
                       uint64_t* dsp_us);
int cp_pearlx_info(CpPearlx* px, int* hvx_contexts, int* clock_vote_err, int* clock_mhz,
                   int* vtcm_kb);

#ifdef __cplusplus
}
#endif

#endif /* CP_PEARLX_CLIENT_H */
