#ifndef CP_GPU_H
#define CP_GPU_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void cp_gpu_init(int* devs, int ndev);
void cp_gpu_shutdown(void);
void cp_gpu_set_contiguous_tiles(int on);

/* Returns 1 on share, 0 on miss, -1 if job cancelled.
 * out_tiles_scanned receives hash tiles actually evaluated (optional). */
int cp_gpu_mine_plain_proof(const int8_t* h_A, const int8_t* h_B,
                            const uint8_t* a_key, const uint32_t pool_tgt[8],
                            int m, int n,
                            int* out_t_rows, int* out_t_cols,
                            uint64_t* out_tiles_scanned);

#ifdef __cplusplus
}
#endif

#endif /* CP_GPU_H */
