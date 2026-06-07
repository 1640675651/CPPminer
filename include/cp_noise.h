#ifndef CP_NOISE_H
#define CP_NOISE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

extern const uint8_t PEARL_SCATTERED_CONFIG[52];
extern const uint8_t PEARL_CONTIGUOUS_CONFIG[52];

extern const uint8_t PEARL_SEED_LABEL_A[32];
extern const uint8_t PEARL_SEED_LABEL_B[32];

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

void pearl_derive_noise_seeds(const uint8_t job_key[32],
                              const uint8_t hash_a[32], const uint8_t hash_b[32],
                              uint8_t b_noise_seed[32], uint8_t a_noise_seed[32]);

void pearl_keyed_matrix_digest(const uint8_t* data, size_t len,
                               const uint8_t job_key[32], uint8_t out[32]);

int pearl_root_from_chunk_cvs(const uint8_t* chunk_cvs, int num_chunks,
                              const uint8_t job_key[32], uint8_t out[32]);

/* Returns 0 if chunk-CV Merkle root matches blake3_digest on padded data. */
int pearl_test_keyed_matrix_root(int m, int n, int k);

int pearl_run_alignment_tests(void);

/* CPU Merkle/chunk test at full padded matrix size (may allocate ~512 MiB per matrix). */
int pearl_run_alignment_tests_prod(int m, int n, int k);

void pearl_get_random_hash(int index, const uint8_t seed[32], const uint8_t key[32],
                           int prepend_index, uint8_t out[32]);

void pearl_keyed_digest_int8(const int8_t* mat, size_t raw_len,
                             const uint8_t job_key[32], uint8_t out[32]);

void pearl_build_perm_pairs_a(const uint8_t noise_seed[32], int k, int rank,
                              uint32_t* pairs_out);
void pearl_build_perm_pairs_b(const uint8_t noise_seed[32], int k, int rank,
                              uint32_t* pairs_out);

void pearl_fuse_noise_row_a(int row, int k, int rank,
                            const uint8_t a_noise_seed[32],
                            const uint32_t* pairs, const int8_t* signal_row,
                            int8_t* noisy_row);
void pearl_fuse_noise_row_b(int col, int k, int rank,
                            const uint8_t b_noise_seed[32],
                            const uint32_t* pairs, const int8_t* signal_row,
                            int8_t* noisy_row);

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
