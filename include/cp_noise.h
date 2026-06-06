#ifndef CP_NOISE_H
#define CP_NOISE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

extern const uint8_t PEARL_SCATTERED_CONFIG[52];
extern const uint8_t PEARL_CONTIGUOUS_CONFIG[52];

void pearl_set_contiguous_tiles(int on);

int pearl_generate_ab(const uint8_t* seed, int seed_len, int m, int n, int k,
                      int8_t* A_out, int8_t* Bt_out);

int pearl_effective_seed(const uint8_t* header, int header_len, uint64_t nonce,
                         uint8_t* out, int out_cap);

void pearl_job_key(const uint8_t* header, int header_len, uint8_t out32[32]);

void pearl_commitment_seeds(const uint8_t job_key[32],
                            const int8_t* A, const int8_t* Bt,
                            int m, int n, int k,
                            uint8_t b_noise_seed[32], uint8_t a_noise_seed[32]);

int pearl_precompute_noise(int m, int n, int k, int rank,
                           const uint8_t b_noise_seed[32],
                           const uint8_t a_noise_seed[32],
                           int32_t* noise_a, int32_t* noise_b);

int pearl_build_noisy_matrices(int m, int n, int k, int rank,
                               const uint8_t b_noise_seed[32],
                               const uint8_t a_noise_seed[32],
                               const int8_t* A, const int8_t* Bt,
                               int8_t* A_out, int8_t* B_out);

#ifdef __cplusplus
}
#endif

#endif /* CP_NOISE_H */
