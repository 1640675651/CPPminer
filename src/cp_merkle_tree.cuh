/* Portable GPU Merkle folding for keyed matrix hash (BLAKE3 tree mode). */
#ifndef CP_MERKLE_TREE_CUH
#define CP_MERKLE_TREE_CUH

#include "cp_blake3_device.cuh"

#define CP_MT_THREADS 256
#define CP_MT_CV_WORDS 8
#define CP_MT_SMEM_BYTES (CP_MT_THREADS * D_B3_OUT)

__device__ __forceinline__ int d_mt_ilog2_ceil(uint32_t n)
{
    if(n <= 1) return 0;
    int lg = 31 - __clz(n);
    if((1u << lg) < n) lg++;
    return lg;
}

__device__ __forceinline__ uint32_t& d_mt_leaf_word(
    uint32_t* smem, int word, int leaf, int smem_cols)
{
    return smem[(size_t)word * (size_t)smem_cols + (size_t)leaf];
}

__device__ __forceinline__ void d_mt_load_key(
    const uint8_t job_key[32], uint32_t key[8])
{
    d_b3_key_words(job_key, key);
}

__device__ __forceinline__ void d_mt_parent_cv(
    const uint32_t key[8], const uint32_t left[8], const uint32_t right[8],
    bool as_root, uint32_t out[8])
{
    uint32_t cv[8];
    for(int i = 0; i < 8; i++) cv[i] = key[i];
    uint8_t block[D_B3_BLOCK];
    for(int i = 0; i < 8; i++){
        d_b3_store32(block + 4 * i, left[i]);
        d_b3_store32(block + 32 + 4 * i, right[i]);
    }
    uint8_t fl = (uint8_t)(D_B3_KEYED | D_B3_PARENT | (as_root ? D_B3_ROOT : 0));
    d_b3_compress_in_place(cv, block, D_B3_BLOCK, 0, fl);
    for(int i = 0; i < 8; i++) out[i] = cv[i];
}

template<bool ConsiderRoot>
__device__ void d_mt_compute_perfect(
    uint32_t* smem, int num_leaves, int smem_cols, const uint32_t key[8])
{
    const int tid = threadIdx.x;
    for(int level_size = num_leaves; level_size > 1; level_size >>= 1){
        uint32_t left[8], right[8], parent[8];
        const int num_pairs = level_size >> 1;
        if(tid < num_pairs){
            for(int i = 0; i < 8; i++){
                left[i]  = d_mt_leaf_word(smem, i, 2 * tid, smem_cols);
                right[i] = d_mt_leaf_word(smem, i, 2 * tid + 1, smem_cols);
            }
        }
        __syncthreads();
        if(tid < num_pairs){
            bool as_root = ConsiderRoot && (tid == 0) && (num_pairs == 1);
            d_mt_parent_cv(key, left, right, as_root, parent);
            for(int i = 0; i < 8; i++)
                d_mt_leaf_word(smem, i, tid, smem_cols) = parent[i];
        }
        __syncthreads();
    }
}

template<bool ConsiderRoot>
__device__ void d_mt_compute_blake(
    uint32_t* smem, int num_leaves, int smem_cols, const uint32_t key[8])
{
    const int tid = threadIdx.x;
    int offset = 0;
    int our_num_leaves = 0;
    int virtual_tid = 0;
    int largest_subtree = 0;

    for(int i = d_mt_ilog2_ceil((uint32_t)num_leaves); i >= 0; --i){
        const uint32_t bit_value = 1u << i;
        if(num_leaves & (int)bit_value){
            if(largest_subtree == 0) largest_subtree = (int)bit_value;
            if(offset + (int)bit_value > 2 * tid){
                our_num_leaves = (int)bit_value;
                virtual_tid = tid - (offset / 2);
                break;
            }
            offset += (int)bit_value;
        }
    }

    for(int curr_num_leaves = largest_subtree; curr_num_leaves > 1;
        curr_num_leaves >>= 1){
        uint32_t left[8], right[8], parent[8];
        const int num_pairs = curr_num_leaves >> 1;
        if(curr_num_leaves <= our_num_leaves && virtual_tid < num_pairs){
            for(int i = 0; i < 8; i++){
                left[i]  = d_mt_leaf_word(smem, i, offset + 2 * virtual_tid, smem_cols);
                right[i] = d_mt_leaf_word(smem, i, offset + 2 * virtual_tid + 1, smem_cols);
            }
        }
        __syncthreads();
        if(curr_num_leaves <= our_num_leaves && virtual_tid < num_pairs){
            d_mt_parent_cv(key, left, right, false, parent);
            for(int i = 0; i < 8; i++)
                d_mt_leaf_word(smem, i, offset + virtual_tid, smem_cols) = parent[i];
        }
        __syncthreads();
    }

    if(tid == 0){
        uint32_t rChaining[8], rChunk[16];
        for(int i = 0; i < 8; i++) rChaining[i] = key[i];
        int read_offset = num_leaves;
        bool written_to_chunk = false;
        for(int i = 0; i < d_mt_ilog2_ceil((uint32_t)num_leaves); ++i){
            const uint32_t bit_mask = 1u << i;
            if(read_offset & (int)bit_mask){
                if(!written_to_chunk){
                    read_offset -= (int)bit_mask;
                    for(int j = 0; j < 8; j++)
                        rChunk[j + 8] = d_mt_leaf_word(smem, j, read_offset, smem_cols);
                    written_to_chunk = true;
                }else{
                    read_offset -= (int)bit_mask;
                    for(int j = 0; j < 8; j++)
                        rChunk[j] = d_mt_leaf_word(smem, j, read_offset, smem_cols);
                    for(int j = 0; j < 8; j++) rChaining[j] = key[j];
                    bool as_root = ConsiderRoot && (read_offset == 0);
                    d_mt_parent_cv(key, rChunk, rChunk + 8, as_root, rChaining);
                    for(int j = 0; j < 8; j++) rChunk[j + 8] = rChaining[j];
                }
            }
        }
        for(int i = 0; i < 8; i++)
            d_mt_leaf_word(smem, i, 0, smem_cols) = rChaining[i];
    }
    __syncthreads();
}

__device__ __forceinline__ void d_mt_store_root_words(
    uint32_t* smem, int smem_cols, uint8_t* roots_out, int root_index)
{
    const int tid = threadIdx.x;
    if(tid >= CP_MT_CV_WORDS) return;
    uint32_t w = d_mt_leaf_word(smem, tid, 0, smem_cols);
    d_b3_store32(roots_out + (size_t)root_index * D_B3_OUT + (size_t)tid * 4, w);
}

/* Stage 1: keyed chunk CVs + per-block Merkle fold → sub-roots in roots_out. */
__global__ void cp_keyed_chunk_roots_kernel(
    const uint8_t* mat, size_t raw_len, size_t pad_len,
    const uint8_t job_key[32], uint8_t* roots_out, int num_chunks)
{
    extern __shared__ uint32_t smem_leaves[];
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int num_grid_blocks = gridDim.x;
    const bool is_last_block = (bid == num_grid_blocks - 1);
    const int global_chunk = bid * CP_MT_THREADS + tid;

    uint8_t cv[D_B3_OUT];
    if(global_chunk < num_chunks){
        size_t off = (size_t)global_chunk * D_B3_CHUNK;
        int len = D_B3_CHUNK;
        if(off + (size_t)len > pad_len) len = (int)(pad_len - off);
        d_b3_keyed_chunk_cv_glob(job_key, (uint64_t)global_chunk, mat, off,
                                 raw_len, len, cv);
        for(int i = 0; i < 8; i++)
            d_mt_leaf_word(smem_leaves, i, tid, CP_MT_THREADS) =
                d_b3_load32(cv + 4 * i);
    }
    __syncthreads();

    int num_leaves = CP_MT_THREADS;
    if(is_last_block){
        const int chunks_in_block = num_chunks % CP_MT_THREADS;
        num_leaves = (chunks_in_block == 0) ? CP_MT_THREADS : chunks_in_block;
        const int remainder_bytes = (int)(pad_len % D_B3_CHUNK);
        const bool last_chunk_too_small =
            (remainder_bytes > 0) && (remainder_bytes < D_B3_BLOCK);
        if(last_chunk_too_small)
            num_leaves = (num_leaves > 0) ? num_leaves - 1 : 0;
    }

    if(num_leaves <= 0) return;

    uint32_t key[8];
    d_mt_load_key(job_key, key);

    if(!is_last_block || ((num_leaves & (num_leaves - 1)) == 0))
        d_mt_compute_perfect</*ConsiderRoot=*/false>(
            smem_leaves, num_leaves, CP_MT_THREADS, key);
    else
        d_mt_compute_blake</*ConsiderRoot=*/false>(
            smem_leaves, num_leaves, CP_MT_THREADS, key);

    d_mt_store_root_words(smem_leaves, CP_MT_THREADS, roots_out, bid);
}

template<int kLeavesPerBlock, bool IsSingleBlock>
__global__ void cp_compute_blake_mt_kernel(
    const uint8_t job_key[32], uint8_t* roots, int num_leaves)
{
    extern __shared__ uint32_t smem_leaves[];
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n_blocks = gridDim.x;
    const int remainder = num_leaves % kLeavesPerBlock;
    const bool is_remainder_block =
        (bid == n_blocks - 1) && (remainder > 0);
    const int block_leaves =
        is_remainder_block ? remainder : kLeavesPerBlock;
    const int offset = bid * kLeavesPerBlock;
    uint32_t key[8];
    d_mt_load_key(job_key, key);

    if(tid < block_leaves){
        for(int i = 0; i < CP_MT_CV_WORDS; i++){
            d_mt_leaf_word(smem_leaves, i, tid, kLeavesPerBlock) =
                d_b3_load32(roots + (size_t)(offset + tid) * D_B3_OUT + (size_t)i * 4);
        }
    }
    __syncthreads();

    const bool use_blake_mt =
        is_remainder_block && ((block_leaves & (block_leaves - 1)) != 0);

    if(use_blake_mt){
        if(IsSingleBlock)
            d_mt_compute_blake</*ConsiderRoot=*/true>(
                smem_leaves, block_leaves, kLeavesPerBlock, key);
        else
            d_mt_compute_blake</*ConsiderRoot=*/false>(
                smem_leaves, block_leaves, kLeavesPerBlock, key);
    }else{
        if(IsSingleBlock)
            d_mt_compute_perfect</*ConsiderRoot=*/true>(
                smem_leaves, block_leaves, kLeavesPerBlock, key);
        else
            d_mt_compute_perfect</*ConsiderRoot=*/false>(
                smem_leaves, block_leaves, kLeavesPerBlock, key);
    }
    __syncthreads();

    if(tid < CP_MT_CV_WORDS){
        const size_t base = IsSingleBlock ? 0 : (size_t)bid * D_B3_OUT;
        d_b3_store32(roots + base + (size_t)tid * 4,
                     d_mt_leaf_word(smem_leaves, tid, 0, kLeavesPerBlock));
    }
}

template<int kMaxThreads>
__global__ void cp_reduce_roots_kernel(
    const uint8_t job_key[32], uint8_t* roots, int num_leaves)
{
    extern __shared__ uint32_t smem_leaves[];
    const int tid = threadIdx.x;
    uint32_t key[8];
    d_mt_load_key(job_key, key);

    if(tid < num_leaves){
        for(int i = 0; i < CP_MT_CV_WORDS; i++){
            d_mt_leaf_word(smem_leaves, i, tid, kMaxThreads) =
                d_b3_load32(roots + (size_t)tid * D_B3_OUT + (size_t)i * 4);
        }
    }
    __syncthreads();

    if(__popc((unsigned)num_leaves) == 1)
        d_mt_compute_perfect</*ConsiderRoot=*/true>(
            smem_leaves, num_leaves, kMaxThreads, key);
    else
        d_mt_compute_blake</*ConsiderRoot=*/true>(
            smem_leaves, num_leaves, kMaxThreads, key);
    __syncthreads();

    if(tid < CP_MT_CV_WORDS)
        d_b3_store32(roots + (size_t)tid * 4,
                     d_mt_leaf_word(smem_leaves, tid, 0, kMaxThreads));
}

#endif /* CP_MERKLE_TREE_CUH */
