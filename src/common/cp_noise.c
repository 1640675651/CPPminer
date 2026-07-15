/*
 * pearl_noise.c — matrix generation, commitment seeds, and pearl_noise precompute.
 * Mirrors zk-pow/src/circuit/pearl_noise.rs and plain_proof_mine.py (CPU path).
 */

#include "cp_noise.h"
#include "cp_job_ctrl.h"
#include "blake3.h"
#include "blake3_impl.h"

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

const uint8_t PEARL_SEED_LABEL_A[32] = "A_tensor";
const uint8_t PEARL_SEED_LABEL_B[32] = "B_tensor";

/* scattered mining config (4096, 256) — pearl_mining.MiningConfiguration.to_bytes() */
const uint8_t PEARL_SCATTERED_CONFIG[52] = {
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
    0x00, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0f,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00
};

/* CUTLASS Case 7.1 tile: rows [0..3,8..11,16..19,24..27], cols stride 32 x8. */
const uint8_t PEARL_CUTLASS_CONFIG[52] = {
    0x00, 0x10, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00,
    0x00, 0x03, 0x01, 0x03, 0x00, 0x00, 0x1f, 0x07,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00
};

static int g_pearl_contiguous_tiles = 0;
static int g_pearl_cutlass_fused = 0;

void pearl_set_contiguous_tiles(int on){
    g_pearl_contiguous_tiles = on ? 1 : 0;
}

void pearl_set_cutlass_fused(int on){
    g_pearl_cutlass_fused = on ? 1 : 0;
}

static const uint8_t* pearl_active_mining_config(void){
    if(g_pearl_cutlass_fused)
        return PEARL_CUTLASS_CONFIG;
    return g_pearl_contiguous_tiles ? PEARL_CONTIGUOUS_CONFIG : PEARL_SCATTERED_CONFIG;
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

static uint64_t pearl_splitmix64(uint64_t x)
{
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

static uint64_t pearl_seed_to_u64(const uint8_t* seed, int seed_len)
{
    uint64_t s = 0;
    for(int i = 0; i < seed_len; i++)
        s ^= (uint64_t)seed[i] << ((i & 7) * 8);
    return s;
}

/* Miner-chosen random A; matches CUDA cp_gen_random_matrix_kernel (matrix_tag = 0). */
int pearl_generate_random_a(const uint8_t* seed, int seed_len, int m, int k,
                            int8_t* A_out)
{
    const uint64_t rng_seed = pearl_seed_to_u64(seed, seed_len);
    const size_t total = (size_t)m * (size_t)k;
    const int log_step = (m >= 65536) ? 1 : 0;
    double t0 = 0.0;
#ifdef _OPENMP
    if(log_step) t0 = omp_get_wtime();
#endif

    for(size_t idx = 0; idx < total; idx++){
        if((idx & 0xFFFF) == 0 && cp_job_should_cancel()) return -1;
        uint64_t s = rng_seed
                   ^ ((uint64_t)0 * 0xD1B54A32D192ED03ULL)
                   ^ (uint64_t)idx * 0x9E3779B97F4A7C15ULL;
        s = pearl_splitmix64(s);
        A_out[idx] = (int8_t)((int)((s >> 32) % 128u) - 64);
        if(log_step && idx > 0 && (idx % (64u * 1024u * 1024u)) == 0)
            printf("[gen]   random A: %.0f%%\n", 100.0 * (double)idx / (double)total);
    }

    if(log_step){
#ifdef _OPENMP
        printf("[gen]   random A done in %.1fs\n", omp_get_wtime() - t0);
#else
        printf("[gen]   random A done\n");
#endif
    }
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

void pearl_keyed_matrix_digest(const uint8_t* data, size_t len,
                               const uint8_t job_key[32], uint8_t out[32])
{
    blake3_digest(data, len, job_key, out);
}

static void pearl_keyed_chunk_cv_cpu(const uint8_t key[32], uint64_t chunk_idx,
                                     const uint8_t* chunk, int chunk_len,
                                     uint8_t cv_out[32])
{
    uint32_t kw[8], cv[8];
    load_key_words(key, kw);
    memcpy(cv, kw, 32);
    int pos = 0;
    int blocks_compressed = 0;
    while(chunk_len - pos > (int)BLAKE3_BLOCK_LEN){
        uint8_t fl = (uint8_t)(KEYED_HASH | (blocks_compressed == 0 ? CHUNK_START : 0));
        blake3_compress_in_place(cv, chunk + (size_t)pos, BLAKE3_BLOCK_LEN, chunk_idx, fl);
        blocks_compressed++;
        pos += (int)BLAKE3_BLOCK_LEN;
    }
    uint8_t tail[BLAKE3_BLOCK_LEN];
    memset(tail, 0, sizeof(tail));
    int tail_len = chunk_len - pos;
    if(tail_len > 0) memcpy(tail, chunk + (size_t)pos, (size_t)tail_len);
    blake3_compress_in_place(cv, tail, (uint8_t)tail_len, chunk_idx,
                             (uint8_t)(KEYED_HASH | CHUNK_END |
                                       (blocks_compressed == 0 ? CHUNK_START : 0)));
    store_cv_words(cv_out, cv);
}

static int pearl_keyed_matrix_root_via_chunks(const uint8_t* data, size_t pad_len,
                                              const uint8_t job_key[32], uint8_t out[32])
{
    if(pad_len == 0 || (pad_len % B3_CHUNK) != 0) return -1;
    int num_chunks = (int)(pad_len / B3_CHUNK);
    if(num_chunks == 1){
        blake3_digest(data, pad_len, job_key, out);
        return 0;
    }
    uint8_t* cvs = (uint8_t*)malloc((size_t)num_chunks * BLAKE3_OUT_LEN);
    if(!cvs) return -1;
    for(int i = 0; i < num_chunks; i++){
        uint8_t chunk[B3_CHUNK];
        size_t off = (size_t)i * B3_CHUNK;
        memset(chunk, 0, B3_CHUNK);
        if(off < pad_len){
            size_t n = pad_len - off;
            if(n > B3_CHUNK) n = B3_CHUNK;
            memcpy(chunk, data + off, n);
        }
        pearl_keyed_chunk_cv_cpu(job_key, (uint64_t)i, chunk, B3_CHUNK,
                                 cvs + (size_t)i * BLAKE3_OUT_LEN);
    }
    int rc = pearl_root_from_chunk_cvs(cvs, num_chunks, job_key, out);
    free(cvs);
    return rc;
}

int pearl_test_keyed_matrix_root(int m, int n, int k)
{
    size_t raw_a = (size_t)m * (size_t)k;
    size_t pad_a = padded_chunk_len(raw_a);
    uint8_t job_key[32];
    uint8_t* pa = (uint8_t*)malloc(pad_a);
    if(!pa) return -1;
    for(size_t i = 0; i < raw_a; i++) pa[i] = (uint8_t)((int8_t)((i * 17 + 3) % 127) - 64);
    memset(pa + raw_a, 0, pad_a - raw_a);
    for(int i = 0; i < 32; i++) job_key[i] = (uint8_t)(i * 11 + 7);

    uint8_t ref[32], via[32];
    blake3_digest(pa, pad_a, job_key, ref);
    if(pearl_keyed_matrix_root_via_chunks(pa, pad_a, job_key, via) != 0){
        free(pa);
        return -1;
    }
    int ok = memcmp(ref, via, 32) == 0;
    if(!ok){
        fprintf(stderr, "[align-test] keyed matrix root mismatch (m=%d n=%d k=%d pad=%zu)\n",
                m, n, k, pad_a);
    }
    free(pa);
    return ok ? 0 : -1;
}

int pearl_test_get_random_hash(void)
{
    uint8_t seed[32], key[32], ref[32], msg[64];
    for(int i = 0; i < 32; i++){ seed[i] = (uint8_t)(i + 3); key[i] = (uint8_t)(i + 5); }
    memset(msg, 0, sizeof(msg));
    int32_t prep = (int32_t)(1 + 7);
    memcpy(msg + 4, &prep, 4);
    memcpy(msg + 32, seed, 32);
    blake3_digest(msg, sizeof(msg), key, ref);

    uint8_t via[32];
    get_random_hash(7, seed, key, 1, via);
    if(memcmp(ref, via, 32) != 0){
        fprintf(stderr, "[align-test] get_random_hash mismatch\n");
        return -1;
    }

    /* Perm-pair path uses exactly 64-byte keyed messages; spot-check several indices. */
    static const int perm_indices[] = {0, 1, 7, 511, 512};
    for(size_t t = 0; t < sizeof(perm_indices) / sizeof(perm_indices[0]); t++){
        int idx = perm_indices[t];
        get_random_hash(idx, PEARL_SEED_LABEL_A, key, 1, via);
        memset(msg, 0, sizeof(msg));
        prep = (int32_t)(1 + idx);
        memcpy(msg + 4, &prep, 4);
        memcpy(msg + 32, PEARL_SEED_LABEL_A, 32);
        blake3_digest(msg, sizeof(msg), key, ref);
        if(memcmp(ref, via, 32) != 0){
            fprintf(stderr, "[align-test] perm hash idx=%d mismatch\n", idx);
            return -1;
        }
    }
    return 0;
}

int pearl_run_alignment_tests(void)
{
    static const int cases[][3] = {
        {1, 1, 1024},
        {4, 4, 256},
        {32, 32, 1024},
        {128, 128, 4096},
    };
    if(pearl_test_get_random_hash() != 0) return -1;
    for(size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++){
        if(pearl_test_keyed_matrix_root(cases[i][0], cases[i][1], cases[i][2]) != 0)
            return -1;
    }
    printf("[align-test] CPU keyed hash + chunk root OK (small cases)\n");
    fflush(stdout);
    return 0;
}

void pearl_get_random_hash(int index, const uint8_t seed[32], const uint8_t key[32],
                           int prepend_index, uint8_t out[32])
{
    get_random_hash(index, seed, key, prepend_index, out);
}

void pearl_keyed_digest_int8(const int8_t* mat, size_t raw_len,
                             const uint8_t job_key[32], uint8_t out[32])
{
    size_t pad_len = padded_chunk_len(raw_len);
    if(pad_len == raw_len){
        blake3_digest((const uint8_t*)mat, pad_len, job_key, out);
        return;
    }
    uint8_t* buf = (uint8_t*)malloc(pad_len);
    if(!buf){
        fprintf(stderr, "pearl_keyed_digest_int8: OOM pad=%zu\n", pad_len);
        exit(1);
    }
    for(size_t i = 0; i < raw_len; i++) buf[i] = (uint8_t)mat[i];
    memset(buf + raw_len, 0, pad_len - raw_len);
    blake3_digest(buf, pad_len, job_key, out);
    free(buf);
}

int pearl_run_alignment_tests_prod(int m, int n, int k)
{
    size_t raw_a = (size_t)m * (size_t)k;
    size_t pad_a = padded_chunk_len(raw_a);
    int num_chunks = (int)(pad_a / B3_CHUNK);
    printf("[align-test-prod] CPU keyed Merkle m=%d n=%d k=%d pad=%zu chunks=%d (~%.1f MiB)\n",
           m, n, k, pad_a, num_chunks, (double)pad_a / (1024.0 * 1024.0));
    fflush(stdout);
    if(pearl_test_keyed_matrix_root(m, n, k) != 0) return -1;
    printf("[align-test-prod] CPU keyed Merkle OK\n");
    fflush(stdout);
    return 0;
}

void pearl_build_perm_pairs_a(const uint8_t noise_seed[32], int k, int rank,
                              uint32_t* pairs_out)
{
    generate_permutation_matrix(PEARL_SEED_LABEL_A, noise_seed, k, rank, pairs_out);
}

void pearl_build_perm_pairs_b(const uint8_t noise_seed[32], int k, int rank,
                              uint32_t* pairs_out)
{
    generate_permutation_matrix(PEARL_SEED_LABEL_B, noise_seed, k, rank, pairs_out);
}

void pearl_fuse_noise_row_a(int row, int k, int rank,
                            const uint8_t a_noise_seed[32],
                            const uint32_t* pairs, const int8_t* signal_row,
                            int8_t* noisy_row)
{
    int8_t el[512];
    int8_t* nr = (int8_t*)malloc((size_t)k);
    if(!nr) return;
    if(rank > (int)sizeof(el)){ free(nr); return; }
    generate_uniform_row(row, rank, PEARL_SEED_LABEL_A, a_noise_seed, el);
    matvec_sparse_perm(pairs, k, el, nr);
    for(int l = 0; l < k; l++)
        noisy_row[l] = (int8_t)((int32_t)signal_row[l] + (int32_t)nr[l]);
    free(nr);
}

void pearl_fuse_noise_row_b(int col, int k, int rank,
                            const uint8_t b_noise_seed[32],
                            const uint32_t* pairs, const int8_t* signal_row,
                            int8_t* noisy_row)
{
    int8_t br[512];
    int8_t* nr = (int8_t*)malloc((size_t)k);
    if(!nr) return;
    if(rank > (int)sizeof(br)){ free(nr); return; }
    generate_uniform_row(col, rank, PEARL_SEED_LABEL_B, b_noise_seed, br);
    matvec_sparse_perm(pairs, k, br, nr);
    for(int l = 0; l < k; l++)
        noisy_row[l] = (int8_t)((int32_t)signal_row[l] + (int32_t)nr[l]);
    free(nr);
}

static size_t pearl_popcnt64(uint64_t x){
#if defined(__GNUC__) || defined(__clang__)
    return (size_t)__builtin_popcountll(x);
#else
    size_t c = 0;
    while(x){ c += (size_t)(x & 1); x >>= 1; }
    return c;
#endif
}

typedef struct {
    uint32_t input_cv[8];
    uint64_t counter;
    uint8_t block[BLAKE3_BLOCK_LEN];
    uint8_t block_len;
    uint8_t flags;
} pearl_output_t;

static pearl_output_t pearl_make_output(const uint32_t input_cv[8],
    const uint8_t block[BLAKE3_BLOCK_LEN], uint8_t block_len,
    uint64_t counter, uint8_t flags)
{
    pearl_output_t ret;
    memcpy(ret.input_cv, input_cv, 32);
    memcpy(ret.block, block, BLAKE3_BLOCK_LEN);
    ret.block_len = block_len;
    ret.counter = counter;
    ret.flags = flags;
    return ret;
}

static void pearl_output_chaining_value(const pearl_output_t* self, uint8_t cv[32]){
    uint32_t cv_words[8];
    memcpy(cv_words, self->input_cv, 32);
    blake3_compress_in_place(cv_words, self->block, self->block_len,
                            self->counter, self->flags);
    store_cv_words(cv, cv_words);
}

static void pearl_output_root_bytes(const pearl_output_t* self, uint8_t out[32])
{
    uint32_t cv_words[8];
    memcpy(cv_words, self->input_cv, 32);
    uint8_t wide[64];
    blake3_compress_xof(cv_words, self->block, self->block_len,
                        self->counter, (uint8_t)(self->flags | ROOT), wide);
    memcpy(out, wide, 32);
}

static pearl_output_t pearl_parent_output(const uint8_t block[BLAKE3_BLOCK_LEN],
    const uint32_t key[8], uint8_t flags)
{
    return pearl_make_output(key, block, BLAKE3_BLOCK_LEN, 0, flags | PARENT);
}

static void pearl_merge_cv_stack(uint8_t* stack, size_t* stack_len,
    const uint32_t key[8], uint8_t flags, uint64_t chunk_counter)
{
    size_t post = pearl_popcnt64(chunk_counter);
    while(*stack_len > post){
        uint8_t* parent_node = stack + (*stack_len - 2) * BLAKE3_OUT_LEN;
        pearl_output_t output = pearl_parent_output(parent_node, key, flags);
        pearl_output_chaining_value(&output, parent_node);
        *stack_len -= 1;
    }
}

static void pearl_push_chunk_cv(uint8_t* stack, size_t* stack_len,
    const uint32_t key[8], uint8_t flags,
    const uint8_t new_cv[32], uint64_t chunk_counter)
{
    pearl_merge_cv_stack(stack, stack_len, key, flags, chunk_counter);
    memcpy(stack + (*stack_len) * BLAKE3_OUT_LEN, new_cv, BLAKE3_OUT_LEN);
    *stack_len += 1;
}

int pearl_root_from_chunk_cvs(const uint8_t* chunk_cvs, int num_chunks,
                              const uint8_t job_key[32], uint8_t out[32])
{
    if(num_chunks <= 1) return -1;
    uint32_t key_words[8];
    load_key_words(job_key, key_words);
    uint8_t stack[(BLAKE3_MAX_DEPTH + 1) * BLAKE3_OUT_LEN];
    size_t stack_len = 0;
    const uint8_t flags = KEYED_HASH;

    for(int i = 0; i < num_chunks; i++){
        pearl_push_chunk_cv(stack, &stack_len, key_words, flags,
                            chunk_cvs + (size_t)i * BLAKE3_OUT_LEN,
                            (uint64_t)i);
    }
    if(stack_len < 2) return -1;

    pearl_output_t output;
    size_t cvs_remaining = stack_len - 2;
    output = pearl_parent_output(stack + cvs_remaining * BLAKE3_OUT_LEN, key_words, flags);
    while(cvs_remaining > 0){
        cvs_remaining--;
        uint8_t parent_block[BLAKE3_BLOCK_LEN];
        memcpy(parent_block, stack + cvs_remaining * BLAKE3_OUT_LEN, 32);
        pearl_output_chaining_value(&output, parent_block + 32);
        output = pearl_parent_output(parent_block, key_words, flags);
    }
    pearl_output_root_bytes(&output, out);
    return 0;
}

void pearl_derive_noise_seeds(const uint8_t job_key[32],
                              const uint8_t hash_a[32], const uint8_t hash_b[32],
                              uint8_t b_noise_seed[32], uint8_t a_noise_seed[32])
{
    uint8_t b_in[64];
    memcpy(b_in, job_key, 32);
    memcpy(b_in + 32, hash_b, 32);
    blake3_digest(b_in, 64, NULL, b_noise_seed);

    uint8_t a_in[64];
    memcpy(a_in, b_noise_seed, 32);
    memcpy(a_in + 32, hash_a, 32);
    blake3_digest(a_in, 64, NULL, a_noise_seed);
}

void pearl_b_noise_seed_from_bt(const uint8_t job_key[32],
                                const int8_t* Bt, int n, int k,
                                uint8_t b_noise_seed[32])
{
    size_t raw_b = (size_t)n * (size_t)k;
    size_t pad_b = padded_chunk_len(raw_b);
    uint8_t* pb = (uint8_t*)malloc(pad_b);
    if(!pb){ fprintf(stderr, "pearl_b_noise_seed_from_bt: OOM\n"); exit(1); }
    for(size_t i = 0; i < raw_b; i++) pb[i] = (uint8_t)Bt[i];
    memset(pb + raw_b, 0, pad_b - raw_b);
    uint8_t hash_b[32];
    blake3_digest(pb, pad_b, job_key, hash_b);
    free(pb);

    uint8_t b_in[64];
    memcpy(b_in, job_key, 32);
    memcpy(b_in + 32, hash_b, 32);
    blake3_digest(b_in, 64, NULL, b_noise_seed);
}

int pearl_build_noisy_a(int m, int k, int rank,
                        const uint8_t a_noise_seed[32],
                        const int8_t* A, int8_t* A_out)
{
    if(rank <= 0 || (rank & (rank - 1)) != 0 || (rank % B3_DIGEST) != 0)
        return -1;

    uint32_t* e_ar = (uint32_t*)malloc((size_t)k * 2 * sizeof(uint32_t));
    if(!e_ar) return -1;

    generate_permutation_matrix(PEARL_SEED_LABEL_A, a_noise_seed, k, rank, e_ar);

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
                generate_uniform_row(row, rank, PEARL_SEED_LABEL_A, a_noise_seed, el_local);
                matvec_sparse_perm(e_ar, k, el_local, nr_local);
                const int8_t* ar = A + (size_t)row * (size_t)k;
                int8_t* dst = A_out + (size_t)row * (size_t)k;
                for(int l = 0; l < k; l++)
                    dst[l] = (int8_t)((int32_t)ar[l] + (int32_t)nr_local[l]);
            }
        }
        free(el_local);
        free(nr_local);
    }
#else
    int8_t* el_row = (int8_t*)malloc((size_t)rank);
    int8_t* nr = (int8_t*)malloc((size_t)k);
    if(!el_row || !nr){ free(e_ar); return -1; }
    for(int row = 0; row < m; row++){
        if((row & 255) == 0 && cp_job_should_cancel()){ aborted = 1; break; }
        generate_uniform_row(row, rank, PEARL_SEED_LABEL_A, a_noise_seed, el_row);
        matvec_sparse_perm(e_ar, k, el_row, nr);
        const int8_t* ar = A + (size_t)row * (size_t)k;
        int8_t* dst = A_out + (size_t)row * (size_t)k;
        for(int l = 0; l < k; l++)
            dst[l] = (int8_t)((int32_t)ar[l] + (int32_t)nr[l]);
    }
    free(el_row);
    free(nr);
#endif

    if(aborted){
        free(e_ar);
        return -1;
    }

    if(prog_step){
#ifdef _OPENMP
        printf("[gen]   noisy A done in %.1fs\n", omp_get_wtime() - t0);
#endif
    }

    free(e_ar);
    return 0;
}

int pearl_build_noisy_b(int n, int k, int rank,
                        const uint8_t b_noise_seed[32],
                        const int8_t* Bt, int8_t* B_out)
{
    if(rank <= 0 || (rank & (rank - 1)) != 0 || (rank % B3_DIGEST) != 0)
        return -1;

    uint32_t* e_bl = (uint32_t*)malloc((size_t)k * 2 * sizeof(uint32_t));
    if(!e_bl) return -1;

    generate_permutation_matrix(PEARL_SEED_LABEL_B, b_noise_seed, k, rank, e_bl);

    const int prog_step = (n >= 8192) ? 4096 : 0;
    double t0 = 0.0;
    int aborted = 0;
#ifdef _OPENMP
    if(prog_step) t0 = omp_get_wtime();
#endif

#ifdef _OPENMP
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
                generate_uniform_row(col, rank, PEARL_SEED_LABEL_B, b_noise_seed, br_local);
                matvec_sparse_perm(e_bl, k, br_local, nr_local);
                const int8_t* br = Bt + (size_t)col * (size_t)k;
                int8_t* dst = B_out + (size_t)col * (size_t)k;
                for(int l = 0; l < k; l++)
                    dst[l] = (int8_t)((int32_t)br[l] + (int32_t)nr_local[l]);
            }
        }
        free(br_local);
        free(nr_local);
    }
#else
    int8_t* br_row = (int8_t*)malloc((size_t)rank);
    int8_t* nr = (int8_t*)malloc((size_t)k);
    if(!br_row || !nr){ free(e_bl); return -1; }
    for(int col = 0; col < n; col++){
        if((col & 255) == 0 && cp_job_should_cancel()){ aborted = 1; break; }
        generate_uniform_row(col, rank, PEARL_SEED_LABEL_B, b_noise_seed, br_row);
        matvec_sparse_perm(e_bl, k, br_row, nr);
        const int8_t* br = Bt + (size_t)col * (size_t)k;
        int8_t* dst = B_out + (size_t)col * (size_t)k;
        for(int l = 0; l < k; l++)
            dst[l] = (int8_t)((int32_t)br[l] + (int32_t)nr[l]);
    }
    free(br_row);
    free(nr);
#endif

    if(aborted){
        free(e_bl);
        return -1;
    }

    if(prog_step){
#ifdef _OPENMP
        printf("[gen]   noisy B done in %.1fs\n", omp_get_wtime() - t0);
#endif
    }

    free(e_bl);
    return 0;
}

int pearl_build_noisy_matrices(int m, int n, int k, int rank,
                               const uint8_t b_noise_seed[32],
                               const uint8_t a_noise_seed[32],
                               const int8_t* A, const int8_t* Bt,
                               int8_t* A_out, int8_t* B_out)
{
    if(pearl_build_noisy_a(m, k, rank, a_noise_seed, A, A_out) != 0)
        return -1;
    return pearl_build_noisy_b(n, k, rank, b_noise_seed, Bt, B_out);
}

int pearl_precompute_noise(int m, int n, int k, int rank,
                           const uint8_t b_noise_seed[32],
                           const uint8_t a_noise_seed[32],
                           int32_t* noise_a, int32_t* noise_b)
{
    (void)m; (void)n; (void)k; (void)rank;
    (void)b_noise_seed; (void)a_noise_seed; (void)noise_a; (void)noise_b;
    return -1;
}
