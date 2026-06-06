/*
 * pearl_noise.c — matrix generation, commitment seeds, and pearl_noise precompute.
 * Mirrors zk-pow/src/circuit/pearl_noise.rs and plain_proof_mine.py (CPU path).
 */

#include "cp_noise.h"
#include "cp_job_ctrl.h"
#include "blake3.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define B3_CHUNK 1024
#define B3_DIGEST 32
#define RANGE_MASK 63
#define ZERO_PT 16

static const uint8_t SEED_LABEL_A[32] = "A_tensor";
static const uint8_t SEED_LABEL_B[32] = "B_tensor";

/* bzminer_mining_config(4096, 256) — from pearl_mining.MiningConfiguration.to_bytes() */
const uint8_t PEARL_BZMINER_CONFIG[52] = {
    0x00, 0x10, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00,
    0x07, 0x01, 0x01, 0x03, 0x00, 0x00, 0x00, 0x01,
    0x0f, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00
};

/* pearl_mining.MiningConfiguration rows=[0..7] cols=[0..15] k=4096 r=256 */
const uint8_t PEARL_CONTIGUOUS_CONFIG[52] = {
    0x00, 0x10, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00,
    0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0f, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00
};

static int g_pearl_contiguous_tiles = 0;

void pearl_set_contiguous_tiles(int on){
    g_pearl_contiguous_tiles = on ? 1 : 0;
}

static const uint8_t* pearl_active_mining_config(void){
    return g_pearl_contiguous_tiles ? PEARL_CONTIGUOUS_CONFIG : PEARL_BZMINER_CONFIG;
}

static size_t padded_chunk_len(size_t raw_len){
    return (raw_len + B3_CHUNK - 1) / B3_CHUNK * B3_CHUNK;
}

static void blake3_digest(const uint8_t* data, size_t len,
                          const uint8_t* key_or_null, uint8_t out[32]){
    blake3_hasher h;
    if(key_or_null){
        blake3_hasher_init_keyed(&h, key_or_null);
    } else {
        blake3_hasher_init(&h);
    }
    blake3_hasher_update(&h, data, len);
    blake3_hasher_finalize(&h, out, 32);
}

static void get_random_hash(int index, const uint8_t seed[32], const uint8_t key[32],
                            int prepend_index, uint8_t out[32]){
    uint8_t msg[64];
    memset(msg, 0, sizeof(msg));
    int32_t prep = (int32_t)(1 + index);
    memcpy(msg + prepend_index * 4, &prep, 4);
    memcpy(msg + 32, seed, 32);
    blake3_digest(msg, sizeof(msg), key, out);
}

static uint32_t mul_hi_u32(uint32_t a, uint32_t b){
    return (uint32_t)(((uint64_t)a * b) >> 32);
}

static void generate_permutation_matrix(const uint8_t seed[32], const uint8_t key[32],
                                        int k, int rank, uint32_t* pairs_out){
    const int lines_per_hash = B3_DIGEST / 4;
    uint32_t rank_mask = (uint32_t)(rank - 1);
    for(int i = 0; i < k; i += lines_per_hash){
        uint8_t digest[32];
        get_random_hash(i / lines_per_hash, seed, key, 1, digest);
        for(int j = 0; j < lines_per_hash; j++){
            int col = i + j;
            if(col >= k) break;
            uint32_t w = (uint32_t)digest[j*4] | ((uint32_t)digest[j*4+1] << 8)
                       | ((uint32_t)digest[j*4+2] << 16) | ((uint32_t)digest[j*4+3] << 24);
            uint32_t first = w & rank_mask;
            uint32_t second = first ^ (1u + mul_hi_u32((uint32_t)(rank - 1), w));
            pairs_out[col * 2] = first;
            pairs_out[col * 2 + 1] = second;
        }
    }
}

static void generate_uniform_row(int row_idx, int num_cols,
                               const uint8_t seed[32], const uint8_t key[32],
                               int8_t* row_out){
    int start_idx = row_idx * num_cols;
    int block = start_idx / B3_DIGEST;
    int out_i = 0;
    while(block * B3_DIGEST < start_idx + num_cols){
        uint8_t digest[32];
        get_random_hash(block, seed, key, 0, digest);
        for(int k = 0; k < B3_DIGEST; k++){
            int idx = block * B3_DIGEST + k;
            if(idx >= start_idx && idx < start_idx + num_cols){
                row_out[out_i++] = (int8_t)((digest[k] & RANGE_MASK) - ZERO_PT);
            }
        }
        block++;
    }
}

static void matvec_sparse_perm(const uint32_t* pairs, int k,
                               const int8_t* vec, int8_t* out){
    for(int i = 0; i < k; i++){
        int32_t pos = (int32_t)vec[pairs[i * 2]];
        int32_t neg = (int32_t)vec[pairs[i * 2 + 1]];
        out[i] = (int8_t)(pos - neg);
    }
}

int pearl_effective_seed(const uint8_t* header, int header_len, uint64_t nonce,
                         uint8_t* out, int out_cap)
{
    if(!header || !out || header_len <= 0 || out_cap <= 0) return -1;
    if(nonce == 0){
        if(header_len > out_cap) return -1;
        memcpy(out, header, (size_t)header_len);
        return header_len;
    }
    uint8_t msg[128];
    if(header_len + 8 > (int)sizeof(msg)) return -1;
    memcpy(msg, header, (size_t)header_len);
    for(int i = 0; i < 8; i++)
        msg[header_len + i] = (uint8_t)(nonce >> (8 * i));
    if(out_cap < 32) return -1;
    blake3_digest(msg, (size_t)header_len + 8, NULL, out);
    return 32;
}

/* Returns 0 on ok, -1 if cancelled via pearl_job_should_cancel(). */
int pearl_generate_ab(const uint8_t* seed, int seed_len, int m, int n, int k,
                       int8_t* A_out, int8_t* Bt_out)
{
    #define XOF_CHUNK (256*1024)
    uint8_t* chunk = (uint8_t*)malloc(XOF_CHUNK);
    if(!chunk){ fprintf(stderr, "pearl_generate_ab: OOM\n"); exit(1); }

    const int log_step = (m >= 65536) ? 1 : 0;
    double t0 = 0.0;
#ifdef _OPENMP
    if(log_step) t0 = omp_get_wtime();
#endif

    blake3_hasher h;
    size_t total = (size_t)m * (size_t)k, off = 0;
    blake3_hasher_init(&h);
    blake3_hasher_update(&h, "matrix_A", 8);
    blake3_hasher_update(&h, seed, (size_t)seed_len);
    while(off < total){
        if(cp_job_should_cancel()){ free(chunk); return -1; }
        size_t n2 = total - off;
        if(n2 > XOF_CHUNK) n2 = XOF_CHUNK;
        blake3_hasher_finalize_seek(&h, off, chunk, n2);
        for(size_t x = 0; x < n2; x++)
            A_out[off + x] = (int8_t)((chunk[x] % 128) - 64);
        off += n2;
        if(log_step && (off == total || (off % (64u * 1024u * 1024u)) == 0))
            printf("[gen]   A: %.0f%%\n", 100.0 * (double)off / (double)total);
    }

    total = (size_t)n * (size_t)k; off = 0;
    blake3_hasher_init(&h);
    blake3_hasher_update(&h, "matrix_B", 8);
    blake3_hasher_update(&h, seed, (size_t)seed_len);
    while(off < total){
        if(cp_job_should_cancel()){ free(chunk); return -1; }
        size_t n2 = total - off;
        if(n2 > XOF_CHUNK) n2 = XOF_CHUNK;
        blake3_hasher_finalize_seek(&h, off, chunk, n2);
        for(size_t x = 0; x < n2; x++)
            Bt_out[off + x] = (int8_t)((chunk[x] % 128) - 64);
        off += n2;
        if(log_step && (off == total || (off % (64u * 1024u * 1024u)) == 0))
            printf("[gen]   B^T: %.0f%%\n", 100.0 * (double)off / (double)total);
    }
    free(chunk);
    if(log_step){
#ifdef _OPENMP
        printf("[gen]   A,B done in %.1fs\n", omp_get_wtime() - t0);
#else
        printf("[gen]   A,B done\n");
#endif
    }
    #undef XOF_CHUNK
    return 0;
}

void pearl_job_key(const uint8_t* header, int header_len, uint8_t out32[32]){
    uint8_t buf[128];
    memcpy(buf, header, (size_t)header_len);
    memcpy(buf + header_len, pearl_active_mining_config(), 52);
    blake3_digest(buf, (size_t)header_len + 52, NULL, out32);
}

void pearl_commitment_seeds(const uint8_t job_key[32],
                            const int8_t* A, const int8_t* Bt,
                            int m, int n, int k,
                            uint8_t b_noise_seed[32], uint8_t a_noise_seed[32])
{
    size_t raw_a = (size_t)m * (size_t)k;
    size_t raw_b = (size_t)n * (size_t)k;
    size_t pad_a = padded_chunk_len(raw_a);
    size_t pad_b = padded_chunk_len(raw_b);

    uint8_t* pa = (uint8_t*)malloc(pad_a);
    uint8_t* pb = (uint8_t*)malloc(pad_b);
    if(!pa || !pb){ fprintf(stderr, "pearl_commitment_seeds: OOM\n"); exit(1); }

    for(size_t i = 0; i < raw_a; i++) pa[i] = (uint8_t)A[i];
    memset(pa + raw_a, 0, pad_a - raw_a);
    for(size_t i = 0; i < raw_b; i++) pb[i] = (uint8_t)Bt[i];
    memset(pb + raw_b, 0, pad_b - raw_b);

    uint8_t hash_a[32], hash_b[32];
    blake3_digest(pa, pad_a, job_key, hash_a);
    blake3_digest(pb, pad_b, job_key, hash_b);
    free(pa);
    free(pb);

    uint8_t b_in[64];
    memcpy(b_in, job_key, 32);
    memcpy(b_in + 32, hash_b, 32);
    blake3_digest(b_in, 64, NULL, b_noise_seed);

    uint8_t a_in[64];
    memcpy(a_in, b_noise_seed, 32);
    memcpy(a_in + 32, hash_a, 32);
    blake3_digest(a_in, 64, NULL, a_noise_seed);
}

int pearl_precompute_noise(int m, int n, int k, int rank,
                           const uint8_t b_noise_seed[32],
                           const uint8_t a_noise_seed[32],
                           int32_t* noise_a, int32_t* noise_b)
{
    if(rank <= 0 || (rank & (rank - 1)) != 0 || (rank % B3_DIGEST) != 0)
        return -1;

    uint32_t* e_ar = (uint32_t*)malloc((size_t)k * 2 * sizeof(uint32_t));
    uint32_t* e_bl = (uint32_t*)malloc((size_t)k * 2 * sizeof(uint32_t));
    int8_t* el_row = (int8_t*)malloc((size_t)rank);
    int8_t* br_row = (int8_t*)malloc((size_t)rank);
    if(!e_ar || !e_bl || !el_row || !br_row) return -1;

    generate_permutation_matrix(SEED_LABEL_A, a_noise_seed, k, rank, e_ar);
    generate_permutation_matrix(SEED_LABEL_B, b_noise_seed, k, rank, e_bl);

    const int prog_step = (m >= 8192) ? 4096 : 0;
    double t0 = 0.0;
#ifdef _OPENMP
    if(prog_step) t0 = omp_get_wtime();
#endif

#ifdef _OPENMP
    #pragma omp parallel
    {
        int8_t* el_local = (int8_t*)malloc((size_t)rank);
        int8_t* nr_local = (int8_t*)malloc((size_t)k);
        int row;
        if(el_local && nr_local){
            #pragma omp for schedule(dynamic, 32)
            for(row = 0; row < m; row++){
                generate_uniform_row(row, rank, SEED_LABEL_A, a_noise_seed, el_local);
                matvec_sparse_perm(e_ar, k, el_local, nr_local);
                int32_t* dst = noise_a + (size_t)row * (size_t)k;
                for(int l = 0; l < k; l++) dst[l] = (int32_t)nr_local[l];
                if(prog_step && (row % prog_step) == 0 && row > 0){
                    #pragma omp critical
                    printf("[gen]   noise_a row %d / %d\n", row, m);
                }
            }
        } else {
            #pragma omp critical
            fprintf(stderr, "[gen] noise_a: thread OOM\n");
        }
        free(el_local);
        free(nr_local);
    }
    #pragma omp parallel
    {
        int8_t* br_local = (int8_t*)malloc((size_t)rank);
        int8_t* nr_local = (int8_t*)malloc((size_t)k);
        int col;
        if(br_local && nr_local){
            #pragma omp for schedule(dynamic, 32)
            for(col = 0; col < n; col++){
                generate_uniform_row(col, rank, SEED_LABEL_B, b_noise_seed, br_local);
                matvec_sparse_perm(e_bl, k, br_local, nr_local);
                int32_t* dst = noise_b + (size_t)col * (size_t)k;
                for(int l = 0; l < k; l++) dst[l] = (int32_t)nr_local[l];
                if(prog_step && (col % prog_step) == 0 && col > 0){
                    #pragma omp critical
                    printf("[gen]   noise_b col %d / %d\n", col, n);
                }
            }
        } else {
            #pragma omp critical
            fprintf(stderr, "[gen] noise_b: thread OOM\n");
        }
        free(br_local);
        free(nr_local);
    }
#else
    int8_t* nr = (int8_t*)malloc((size_t)k);
    if(!nr){ free(e_ar); free(e_bl); free(el_row); free(br_row); return -1; }
    for(int row = 0; row < m; row++){
        generate_uniform_row(row, rank, SEED_LABEL_A, a_noise_seed, el_row);
        matvec_sparse_perm(e_ar, k, el_row, nr);
        int32_t* dst = noise_a + (size_t)row * (size_t)k;
        for(int l = 0; l < k; l++) dst[l] = (int32_t)nr[l];
        if(prog_step && (row % prog_step) == 0 && row > 0)
            printf("[gen]   noise_a row %d / %d\n", row, m);
    }
    for(int col = 0; col < n; col++){
        generate_uniform_row(col, rank, SEED_LABEL_B, b_noise_seed, br_row);
        matvec_sparse_perm(e_bl, k, br_row, nr);
        int32_t* dst = noise_b + (size_t)col * (size_t)k;
        for(int l = 0; l < k; l++) dst[l] = (int32_t)nr[l];
        if(prog_step && (col % prog_step) == 0 && col > 0)
            printf("[gen]   noise_b col %d / %d\n", col, n);
    }
    free(nr);
#endif

    if(prog_step){
#ifdef _OPENMP
        printf("[gen]   noise done in %.1fs\n", omp_get_wtime() - t0);
#endif
    }

    free(e_ar);
    free(e_bl);
    free(el_row);
    free(br_row);
    return 0;
}

int pearl_build_noisy_matrices(int m, int n, int k, int rank,
                               const uint8_t b_noise_seed[32],
                               const uint8_t a_noise_seed[32],
                               const int8_t* A, const int8_t* Bt,
                               int8_t* A_out, int8_t* B_out)
{
    if(rank <= 0 || (rank & (rank - 1)) != 0 || (rank % B3_DIGEST) != 0)
        return -1;

    uint32_t* e_ar = (uint32_t*)malloc((size_t)k * 2 * sizeof(uint32_t));
    uint32_t* e_bl = (uint32_t*)malloc((size_t)k * 2 * sizeof(uint32_t));
    if(!e_ar || !e_bl) return -1;

    generate_permutation_matrix(SEED_LABEL_A, a_noise_seed, k, rank, e_ar);
    generate_permutation_matrix(SEED_LABEL_B, b_noise_seed, k, rank, e_bl);

    const int prog_step = (m >= 8192) ? 4096 : 0;
    double t0 = 0.0;
    int aborted = 0;
#ifdef _OPENMP
    if(prog_step) t0 = omp_get_wtime();
#endif

#ifdef _OPENMP
    #pragma omp parallel
    {
        int8_t* el_local = (int8_t*)malloc((size_t)rank);
        int8_t* nr_local = (int8_t*)malloc((size_t)k);
        int row;
        if(el_local && nr_local){
            #pragma omp for schedule(dynamic, 32)
            for(row = 0; row < m; row++){
                if((row & 255) == 0 && cp_job_should_cancel()){
                    #pragma omp critical(pearl_abort)
                    { aborted = 1; }
                }
                if(aborted) continue;
                generate_uniform_row(row, rank, SEED_LABEL_A, a_noise_seed, el_local);
                matvec_sparse_perm(e_ar, k, el_local, nr_local);
                const int8_t* ar = A + (size_t)row * (size_t)k;
                int8_t* dst = A_out + (size_t)row * (size_t)k;
                for(int l = 0; l < k; l++)
                    dst[l] = (int8_t)((int32_t)ar[l] + (int32_t)nr_local[l]);
                if(prog_step && (row % prog_step) == 0 && row > 0){
                    #pragma omp critical
                    printf("[gen]   noisy A row %d / %d\n", row, m);
                }
            }
        }
        free(el_local);
        free(nr_local);
    }
    if(!aborted){
    #pragma omp parallel
    {
        int8_t* br_local = (int8_t*)malloc((size_t)rank);
        int8_t* nr_local = (int8_t*)malloc((size_t)k);
        int col;
        if(br_local && nr_local){
            #pragma omp for schedule(dynamic, 32)
            for(col = 0; col < n; col++){
                if((col & 255) == 0 && cp_job_should_cancel()){
                    #pragma omp critical(pearl_abort)
                    { aborted = 1; }
                }
                if(aborted) continue;
                generate_uniform_row(col, rank, SEED_LABEL_B, b_noise_seed, br_local);
                matvec_sparse_perm(e_bl, k, br_local, nr_local);
                const int8_t* br = Bt + (size_t)col * (size_t)k;
                int8_t* dst = B_out + (size_t)col * (size_t)k;
                for(int l = 0; l < k; l++)
                    dst[l] = (int8_t)((int32_t)br[l] + (int32_t)nr_local[l]);
                if(prog_step && (col % prog_step) == 0 && col > 0){
                    #pragma omp critical
                    printf("[gen]   noisy B^T row %d / %d\n", col, n);
                }
            }
        }
        free(br_local);
        free(nr_local);
    }
    }
#else
    int8_t* el_row = (int8_t*)malloc((size_t)rank);
    int8_t* br_row = (int8_t*)malloc((size_t)rank);
    int8_t* nr = (int8_t*)malloc((size_t)k);
    if(!el_row || !br_row || !nr){ free(e_ar); free(e_bl); return -1; }
    for(int row = 0; row < m; row++){
        if((row & 255) == 0 && cp_job_should_cancel()){ aborted = 1; break; }
        generate_uniform_row(row, rank, SEED_LABEL_A, a_noise_seed, el_row);
        matvec_sparse_perm(e_ar, k, el_row, nr);
        const int8_t* ar = A + (size_t)row * (size_t)k;
        int8_t* dst = A_out + (size_t)row * (size_t)k;
        for(int l = 0; l < k; l++)
            dst[l] = (int8_t)((int32_t)ar[l] + (int32_t)nr[l]);
    }
    if(!aborted){
    for(int col = 0; col < n; col++){
        if((col & 255) == 0 && cp_job_should_cancel()){ aborted = 1; break; }
        generate_uniform_row(col, rank, SEED_LABEL_B, b_noise_seed, br_row);
        matvec_sparse_perm(e_bl, k, br_row, nr);
        const int8_t* br = Bt + (size_t)col * (size_t)k;
        int8_t* dst = B_out + (size_t)col * (size_t)k;
        for(int l = 0; l < k; l++)
            dst[l] = (int8_t)((int32_t)br[l] + (int32_t)nr[l]);
    }
    }
    free(el_row); free(br_row); free(nr);
#endif

    if(aborted){
        free(e_ar);
        free(e_bl);
        return -1;
    }

    if(prog_step){
#ifdef _OPENMP
        printf("[gen]   noisy A,B done in %.1fs\n", omp_get_wtime() - t0);
#endif
    }

    free(e_ar);
    free(e_bl);
    return 0;
}
