/* Two-phase pearl-style noise: parallel dense/sparse gen + parallel apply. */
#ifndef CP_NOISE_PHASE_CUH
#define CP_NOISE_PHASE_CUH

#include "cp_gpu_gen.cuh"

/* Parallel matvec: out[i] = vec[pairs[2i]] - vec[pairs[2i+1]]. */
__device__ __forceinline__ void d_matvec_sparse_perm_par(
    const uint32_t* pairs, int k, const int8_t* vec, int8_t* out,
    int tid, int nthreads)
{
    for(int i = tid; i < k; i += nthreads){
        int32_t pos = (int32_t)vec[pairs[(size_t)i * 2]];
        int32_t neg = (int32_t)vec[pairs[(size_t)i * 2 + 1]];
        out[i] = (int8_t)(pos - neg);
    }
}

/* Phase 1a: E_AL / E_BR dense rows (m×rank or n×rank). One BLAKE3 chunk per thread. */
__global__ void cp_gen_dense_noise_kernel(
    int is_b, int num_rows, int rank,
    const uint8_t* noise_seed, int8_t* out)
{
    const uint8_t* label = is_b ? CP_SEED_LABEL_B : CP_SEED_LABEL_A;
    const int tpr = rank / D_B3_OUT;
    const int rows_per_block = (int)blockDim.x / tpr;
    const int row = (int)blockIdx.x * rows_per_block + (int)threadIdx.x / tpr;
    const int chunk_in_row = (int)threadIdx.x % tpr;
    if(row >= num_rows) return;

    const int global_chunk = row * tpr + chunk_in_row;
    uint8_t digest[D_B3_OUT];
    d_get_random_hash(global_chunk, label, noise_seed, 0, digest);

    int8_t* dst = out + (size_t)row * (size_t)rank + (size_t)chunk_in_row * D_B3_OUT;
    for(int i = 0; i < D_B3_OUT; i++)
        dst[i] = (int8_t)((digest[i] & CP_RANGE_MASK) - CP_ZERO_PT);
}

/* Phase 1b: parallel E_AR / E_BL permutation pair table (k×2 uint32). */
__global__ void cp_build_perm_pairs_par_kernel(
    int is_b, const uint8_t* noise_seed, int k, int rank, uint32_t* pairs_out)
{
    const int block_idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int col0 = block_idx * CP_B3_LINES;
    if(col0 >= k) return;

    const uint8_t* label = is_b ? CP_SEED_LABEL_B : CP_SEED_LABEL_A;
    uint8_t digest[D_B3_OUT];
    d_get_random_hash(block_idx, label, noise_seed, 1, digest);

    const uint32_t rank_mask = (uint32_t)(rank - 1);
    for(int j = 0; j < CP_B3_LINES; j++){
        int col = col0 + j;
        if(col >= k) break;
        uint32_t w = d_b3_load32(digest + j * 4);
        uint32_t first = w & rank_mask;
        uint32_t second = first ^ (1u + cp_mul_hi_u32((uint32_t)(rank - 1), w));
        pairs_out[(size_t)col * 2] = first;
        pairs_out[(size_t)col * 2 + 1] = second;
    }
}

/* Phase 2: Ap = A + E_AL @ E_AR using precomputed E_AL and pair table. Step-major out. */
__global__ void cp_apply_noise_a_kernel(
    const int8_t* signal, const int8_t* eal, int8_t* noisy,
    int m, int k, int rank, const uint32_t* pairs)
{
    const int row = (int)blockIdx.x;
    if(row >= m) return;

    extern __shared__ int8_t sh_noise_buf[];
    int8_t* sh_el = sh_noise_buf;
    int8_t* sh_noise = sh_noise_buf + rank;

    for(int i = (int)threadIdx.x; i < rank; i += (int)blockDim.x)
        sh_el[i] = eal[(size_t)row * (size_t)rank + (size_t)i];
    __syncthreads();

    d_matvec_sparse_perm_par(pairs, k, sh_el, sh_noise, (int)threadIdx.x, (int)blockDim.x);
    __syncthreads();

    for(int l = (int)threadIdx.x; l < k; l += (int)blockDim.x){
        size_t src = (size_t)row * (size_t)k + (size_t)l;
        int step = l / rank;
        int ri = l % rank;
        size_t dst = ((size_t)step * (size_t)m + (size_t)row) * (size_t)rank + (size_t)ri;
        noisy[dst] = (int8_t)((int32_t)signal[src] + (int32_t)sh_noise[l]);
    }
}

__global__ void cp_apply_noise_b_kernel(
    const int8_t* signal, const int8_t* ebr, int8_t* noisy,
    int n, int k, int rank, const uint32_t* pairs)
{
    const int col = (int)blockIdx.x;
    if(col >= n) return;

    extern __shared__ int8_t sh_noise_buf[];
    int8_t* sh_el = sh_noise_buf;
    int8_t* sh_noise = sh_noise_buf + rank;

    for(int i = (int)threadIdx.x; i < rank; i += (int)blockDim.x)
        sh_el[i] = ebr[(size_t)col * (size_t)rank + (size_t)i];
    __syncthreads();

    d_matvec_sparse_perm_par(pairs, k, sh_el, sh_noise, (int)threadIdx.x, (int)blockDim.x);
    __syncthreads();

    for(int l = (int)threadIdx.x; l < k; l += (int)blockDim.x){
        size_t src = (size_t)col * (size_t)k + (size_t)l;
        int step = l / rank;
        int ri = l % rank;
        size_t dst = ((size_t)step * (size_t)n + (size_t)col) * (size_t)rank + (size_t)ri;
        const int32_t sig = signal ? (int32_t)signal[src] : 0;
        noisy[dst] = (int8_t)(sig + (int32_t)sh_noise[l]);
    }
}

__global__ void cp_apply_noise_a_rowmajor_kernel(
    const int8_t* signal, const int8_t* eal, int8_t* noisy,
    int m, int k, int rank, const uint32_t* pairs)
{
    const int row = (int)blockIdx.x;
    if(row >= m) return;

    extern __shared__ int8_t sh_noise_buf[];
    int8_t* sh_el = sh_noise_buf;
    int8_t* sh_noise = sh_noise_buf + rank;

    for(int i = (int)threadIdx.x; i < rank; i += (int)blockDim.x)
        sh_el[i] = eal[(size_t)row * (size_t)rank + (size_t)i];
    __syncthreads();

    d_matvec_sparse_perm_par(pairs, k, sh_el, sh_noise, (int)threadIdx.x, (int)blockDim.x);
    __syncthreads();

    for(int l = (int)threadIdx.x; l < k; l += (int)blockDim.x){
        size_t idx = (size_t)row * (size_t)k + (size_t)l;
        noisy[idx] = (int8_t)((int32_t)signal[idx] + (int32_t)sh_noise[l]);
    }
}

__global__ void cp_apply_noise_b_rowmajor_kernel(
    const int8_t* signal, const int8_t* ebr, int8_t* noisy,
    int n, int k, int rank, const uint32_t* pairs)
{
    const int col = (int)blockIdx.x;
    if(col >= n) return;

    extern __shared__ int8_t sh_noise_buf[];
    int8_t* sh_el = sh_noise_buf;
    int8_t* sh_noise = sh_noise_buf + rank;

    for(int i = (int)threadIdx.x; i < rank; i += (int)blockDim.x)
        sh_el[i] = ebr[(size_t)col * (size_t)rank + (size_t)i];
    __syncthreads();

    d_matvec_sparse_perm_par(pairs, k, sh_el, sh_noise, (int)threadIdx.x, (int)blockDim.x);
    __syncthreads();

    for(int l = (int)threadIdx.x; l < k; l += (int)blockDim.x){
        size_t idx = (size_t)col * (size_t)k + (size_t)l;
        const int32_t sig = signal ? (int32_t)signal[idx] : 0;
        noisy[idx] = (int8_t)(sig + (int32_t)sh_noise[l]);
    }
}

#endif /* CP_NOISE_PHASE_CUH */
