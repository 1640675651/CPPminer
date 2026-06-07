/* GPU random matrix generation and pearl noise injection (BLAKE3 kept for noise/commitment). */
#ifndef CP_GPU_GEN_CUH
#define CP_GPU_GEN_CUH

#include "cp_blake3_device.cuh"
#include "cp_config.h"
#define CP_RANGE_MASK 63
#define CP_ZERO_PT 16
#define CP_B3_LINES 8

__device__ __constant__ uint8_t CP_SEED_LABEL_A[32] = "A_tensor";
__device__ __constant__ uint8_t CP_SEED_LABEL_B[32] = "B_tensor";

__device__ __forceinline__ uint32_t cp_mul_hi_u32(uint32_t a, uint32_t b){
    return (uint32_t)(((uint64_t)a * b) >> 32);
}

__device__ __forceinline__ void d_keyed_digest(
    const uint8_t* data, int len, const uint8_t key[32], uint8_t out[32])
{
    uint32_t kw[8];
    d_b3_key_words(key, kw);
    d_b3_chunk_state st;
    d_b3_chunk_init(&st, kw, D_B3_KEYED);
    d_b3_chunk_update(&st, data, (size_t)len);
    d_b3_chunk_root_out(&st, out);
}

__device__ __forceinline__ void d_get_random_hash(
    int index, const uint8_t seed[32], const uint8_t key[32],
    int prepend_index, uint8_t out[32])
{
    uint8_t msg[64];
    for(int i=0;i<64;i++) msg[i]=0;
    int32_t prep = (int32_t)(1 + index);
    memcpy(msg + prepend_index * 4, &prep, 4);
    memcpy(msg + 32, seed, 32);
    d_keyed_digest(msg, 64, key, out);
}

__device__ __forceinline__ void d_generate_uniform_row(
    int row_idx, int num_cols, const uint8_t seed[32], const uint8_t key[32],
    int8_t* row_out)
{
    int start_idx = row_idx * num_cols;
    int block = start_idx / D_B3_OUT;
    int out_i = 0;
    while(block * D_B3_OUT < start_idx + num_cols){
        uint8_t digest[32];
        d_get_random_hash(block, seed, key, 0, digest);
        for(int k = 0; k < D_B3_OUT; k++){
            int idx = block * D_B3_OUT + k;
            if(idx >= start_idx && idx < start_idx + num_cols){
                row_out[out_i++] = (int8_t)((digest[k] & CP_RANGE_MASK) - CP_ZERO_PT);
            }
        }
        block++;
    }
}

__device__ __forceinline__ void d_matvec_sparse_perm(
    const uint32_t* pairs, int k, const int8_t* vec, int8_t* out)
{
    for(int i = 0; i < k; i++){
        int32_t pos = (int32_t)vec[pairs[i * 2]];
        int32_t neg = (int32_t)vec[pairs[i * 2 + 1]];
        out[i] = (int8_t)(pos - neg);
    }
}

__device__ __forceinline__ uint64_t cp_splitmix64(uint64_t x){
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

/* Fill int8 matrix with arbitrary random values in [-64, 63] (same range as pearl_generate_ab). */
__global__ void cp_gen_random_matrix_kernel(
    unsigned long long rng_seed, int matrix_tag,
    int total_elems, int8_t* out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= total_elems) return;
    uint64_t s = rng_seed
               ^ ((uint64_t)matrix_tag * 0xD1B54A32D192ED03ULL)
               ^ (uint64_t)idx * 0x9E3779B97F4A7C15ULL;
    s = cp_splitmix64(s);
    out[idx] = (int8_t)((int)((s >> 32) % 128u) - 64);
}
__global__ void cp_keyed_chunk_cv_kernel(
    const uint8_t* mat, size_t raw_len, size_t pad_len,
    const uint8_t job_key[32], uint8_t* chunk_cvs)
{
    int chunk = (int)blockIdx.x;
    size_t off = (size_t)chunk * D_B3_CHUNK;
    if(off >= pad_len) return;
    int len = (int)D_B3_CHUNK;
    if(off + (size_t)len > pad_len) len = (int)(pad_len - off);
    uint8_t local[D_B3_CHUNK];
    for(int i = 0; i < len; i++){
        size_t gi = off + (size_t)i;
        local[i] = (gi < raw_len) ? (uint8_t)mat[gi] : 0;
    }
    for(int i = len; i < (int)D_B3_CHUNK; i++) local[i] = 0;
    d_b3_keyed_chunk_cv(job_key, (uint64_t)chunk, local, len,
                        chunk_cvs + (size_t)chunk * D_B3_OUT);
}

__global__ void cp_test_perm_hash_kernel(
    int index, const uint8_t key[32], uint8_t out[32])
{
    if(threadIdx.x != 0 || blockIdx.x != 0) return;
    d_get_random_hash(index, CP_SEED_LABEL_A, key, 1, out);
}

__global__ void cp_build_perm_pairs_kernel(
    int is_b, const uint8_t noise_seed[32],
    int k, int rank, uint32_t* pairs_out)
{
    if(threadIdx.x != 0 || blockIdx.x != 0) return;
    const uint8_t* label = is_b ? CP_SEED_LABEL_B : CP_SEED_LABEL_A;
    uint32_t rank_mask = (uint32_t)(rank - 1);
    for(int i = 0; i < k; i += CP_B3_LINES){
        uint8_t digest[32];
        d_get_random_hash(i / CP_B3_LINES, label, noise_seed, 1, digest);
        for(int j = 0; j < CP_B3_LINES; j++){
            int col = i + j;
            if(col >= k) break;
            uint32_t w = d_b3_load32(digest + j * 4);
            uint32_t first = w & rank_mask;
            uint32_t second = first ^ (1u + cp_mul_hi_u32((uint32_t)(rank - 1), w));
            pairs_out[col * 2] = first;
            pairs_out[col * 2 + 1] = second;
        }
    }
}

__global__ void cp_fuse_noise_a_kernel(
    const int8_t* signal, int8_t* noisy,
    int m, int k, int rank,
    const uint8_t noise_seed[32],
    const uint32_t* pairs)
{
    int row = (int)blockIdx.x;
    if(row >= m) return;
    extern __shared__ int8_t sh_noise_row[];
    if(threadIdx.x == 0){
        int8_t el[512];
        d_generate_uniform_row(row, rank, CP_SEED_LABEL_A, noise_seed, el);
        d_matvec_sparse_perm(pairs, k, el, sh_noise_row);
    }
    __syncthreads();
    for(int l = (int)threadIdx.x; l < k; l += (int)blockDim.x){
        size_t idx = (size_t)row * (size_t)k + (size_t)l;
        noisy[idx] = (int8_t)((int32_t)signal[idx] + (int32_t)sh_noise_row[l]);
    }
}

__global__ void cp_fuse_noise_b_kernel(
    const int8_t* signal, int8_t* noisy,
    int n, int k, int rank,
    const uint8_t noise_seed[32],
    const uint32_t* pairs)
{
    int col = (int)blockIdx.x;
    if(col >= n) return;
    extern __shared__ int8_t sh_noise_row[];
    if(threadIdx.x == 0){
        int8_t br[512];
        d_generate_uniform_row(col, rank, CP_SEED_LABEL_B, noise_seed, br);
        d_matvec_sparse_perm(pairs, k, br, sh_noise_row);
    }
    __syncthreads();
    for(int l = (int)threadIdx.x; l < k; l += (int)blockDim.x){
        size_t idx = (size_t)col * (size_t)k + (size_t)l;
        noisy[idx] = (int8_t)((int32_t)signal[idx] + (int32_t)sh_noise_row[l]);
    }
}

#endif /* CP_GPU_GEN_CUH */
