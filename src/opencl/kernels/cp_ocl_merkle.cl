/* GPU Merkle folding for keyed matrix hash (BLAKE3 tree mode). */

#define CP_MT_THREADS 256
#define CP_MT_CV_WORDS 8

inline int d_mt_ilog2_ceil(uint n) {
    if (n <= 1) {
        return 0;
    }
    int lg = 31 - clz(n);
    if ((1u << lg) < n) {
        lg++;
    }
    return lg;
}

inline uint d_mt_leaf_word(__local uint* smem, int word, int leaf, int smem_cols) {
    return smem[(size_t)word * (size_t)smem_cols + (size_t)leaf];
}

inline void d_mt_set_leaf_word(__local uint* smem, int word, int leaf, int smem_cols, uint v) {
    smem[(size_t)word * (size_t)smem_cols + (size_t)leaf] = v;
}

inline void d_mt_load_key_g(__global const uchar *job_key, uint key[8]) {
    d_b3_key_words_g(job_key, key);
}

inline void d_mt_parent_cv(const uint key[8], const uint left[8], const uint right[8],
                           int as_root, uint out[8]) {
    uint cv[8];
    for (int i = 0; i < 8; i++) {
        cv[i] = key[i];
    }
    uchar block[D_B3_BLOCK];
    for (int i = 0; i < 8; i++) {
        d_b3_store32_priv(block + 4 * i, left[i]);
        d_b3_store32_priv(block + 32 + 4 * i, right[i]);
    }
    uchar fl = (uchar)(D_B3_KEYED | D_B3_PARENT | (as_root ? D_B3_ROOT : 0));
    d_b3_compress_in_place(cv, block, D_B3_BLOCK, 0, fl);
    for (int i = 0; i < 8; i++) {
        out[i] = cv[i];
    }
}

inline void d_mt_compute_perfect(__local uint* smem, int num_leaves, int smem_cols,
                                 const uint key[8], int consider_root) {
    const int tid = get_local_id(0);
    for (int level_size = num_leaves; level_size > 1; level_size >>= 1) {
        uint left[8], right[8], parent[8];
        const int num_pairs = level_size >> 1;
        if (tid < num_pairs) {
            for (int i = 0; i < 8; i++) {
                left[i] = d_mt_leaf_word(smem, i, 2 * tid, smem_cols);
                right[i] = d_mt_leaf_word(smem, i, 2 * tid + 1, smem_cols);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < num_pairs) {
            int as_root = consider_root && (tid == 0) && (num_pairs == 1);
            d_mt_parent_cv(key, left, right, as_root, parent);
            for (int i = 0; i < 8; i++) {
                d_mt_set_leaf_word(smem, i, tid, smem_cols, parent[i]);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

inline void d_mt_compute_blake(__local uint* smem, int num_leaves, int smem_cols,
                               const uint key[8], int consider_root) {
    const int tid = get_local_id(0);
    int offset = 0;
    int our_num_leaves = 0;
    int virtual_tid = 0;
    int largest_subtree = 0;

    for (int i = d_mt_ilog2_ceil((uint)num_leaves); i >= 0; --i) {
        const uint bit_value = 1u << i;
        if (num_leaves & (int)bit_value) {
            if (largest_subtree == 0) {
                largest_subtree = (int)bit_value;
            }
            if (offset + (int)bit_value > 2 * tid) {
                our_num_leaves = (int)bit_value;
                virtual_tid = tid - (offset / 2);
                break;
            }
            offset += (int)bit_value;
        }
    }

    for (int curr_num_leaves = largest_subtree; curr_num_leaves > 1; curr_num_leaves >>= 1) {
        uint left[8], right[8], parent[8];
        const int num_pairs = curr_num_leaves >> 1;
        if (curr_num_leaves <= our_num_leaves && virtual_tid < num_pairs) {
            for (int i = 0; i < 8; i++) {
                left[i] = d_mt_leaf_word(smem, i, offset + 2 * virtual_tid, smem_cols);
                right[i] = d_mt_leaf_word(smem, i, offset + 2 * virtual_tid + 1, smem_cols);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (curr_num_leaves <= our_num_leaves && virtual_tid < num_pairs) {
            d_mt_parent_cv(key, left, right, 0, parent);
            for (int i = 0; i < 8; i++) {
                d_mt_set_leaf_word(smem, i, offset + virtual_tid, smem_cols, parent[i]);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (tid == 0) {
        uint rChaining[8], rChunk[16];
        for (int i = 0; i < 8; i++) {
            rChaining[i] = key[i];
        }
        int read_offset = num_leaves;
        int written_to_chunk = 0;
        for (int i = 0; i < d_mt_ilog2_ceil((uint)num_leaves); ++i) {
            const uint bit_mask = 1u << i;
            if (read_offset & (int)bit_mask) {
                if (!written_to_chunk) {
                    read_offset -= (int)bit_mask;
                    for (int j = 0; j < 8; j++) {
                        rChunk[j + 8] = d_mt_leaf_word(smem, j, read_offset, smem_cols);
                    }
                    written_to_chunk = 1;
                } else {
                    read_offset -= (int)bit_mask;
                    for (int j = 0; j < 8; j++) {
                        rChunk[j] = d_mt_leaf_word(smem, j, read_offset, smem_cols);
                    }
                    for (int j = 0; j < 8; j++) {
                        rChaining[j] = key[j];
                    }
                    int as_root = consider_root && (read_offset == 0);
                    d_mt_parent_cv(key, rChunk, rChunk + 8, as_root, rChaining);
                    for (int j = 0; j < 8; j++) {
                        rChunk[j + 8] = rChaining[j];
                    }
                }
            }
        }
        for (int i = 0; i < 8; i++) {
            d_mt_set_leaf_word(smem, i, 0, smem_cols, rChaining[i]);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

inline void d_mt_store_root_words(__local uint* smem, int smem_cols, __global uchar* roots_out,
                                  int root_index) {
    const int tid = get_local_id(0);
    if (tid >= CP_MT_CV_WORDS) {
        return;
    }
    uint w = d_mt_leaf_word(smem, tid, 0, smem_cols);
    d_b3_store32_g(roots_out + (size_t)root_index * D_B3_OUT + (size_t)tid * 4, w);
}

__kernel void ocl_keyed_chunk_roots(__global const uchar* mat, ulong raw_len, ulong pad_len,
                                    __global const uchar* job_key, __global uchar* roots_out,
                                    int num_chunks) {
    __local uint smem_leaves[CP_MT_CV_WORDS * CP_MT_THREADS];
    const int tid = get_local_id(0);
    const int bid = get_group_id(0);
    const int num_grid_blocks = get_num_groups(0);
    const int is_last_block = (bid == num_grid_blocks - 1);
    const int global_chunk = bid * CP_MT_THREADS + tid;

    uchar cv[D_B3_OUT];
    if (global_chunk < num_chunks) {
        ulong off = (ulong)global_chunk * D_B3_CHUNK;
        int len = D_B3_CHUNK;
        if (off + (ulong)len > pad_len) {
            len = (int)(pad_len - off);
        }
        d_b3_keyed_chunk_cv_glob(job_key, (ulong)global_chunk, mat, off, raw_len, len, cv);
        for (int i = 0; i < 8; i++) {
            d_mt_set_leaf_word(smem_leaves, i, tid, CP_MT_THREADS, d_b3_load32_priv(cv + 4 * i));
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int num_leaves = CP_MT_THREADS;
    if (is_last_block) {
        const int chunks_in_block = num_chunks % CP_MT_THREADS;
        num_leaves = (chunks_in_block == 0) ? CP_MT_THREADS : chunks_in_block;
        const int remainder_bytes = (int)(pad_len % D_B3_CHUNK);
        const int last_chunk_too_small =
                (remainder_bytes > 0) && (remainder_bytes < D_B3_BLOCK);
        if (last_chunk_too_small) {
            num_leaves = (num_leaves > 0) ? num_leaves - 1 : 0;
        }
    }

    if (num_leaves <= 0) {
        return;
    }

    uint key[8];
    d_mt_load_key_g(job_key, key);

    if (!is_last_block || ((num_leaves & (num_leaves - 1)) == 0)) {
        d_mt_compute_perfect(smem_leaves, num_leaves, CP_MT_THREADS, key, 0);
    } else {
        d_mt_compute_blake(smem_leaves, num_leaves, CP_MT_THREADS, key, 0);
    }

    d_mt_store_root_words(smem_leaves, CP_MT_THREADS, roots_out, bid);
}

__kernel void ocl_compute_blake_mt(__global const uchar* job_key, __global uchar* roots,
                                 int num_leaves, int is_single_block) {
    __local uint smem_leaves[CP_MT_CV_WORDS * CP_MT_THREADS];
    const int tid = get_local_id(0);
    const int bid = get_group_id(0);
    const int n_blocks = get_num_groups(0);
    const int remainder = num_leaves % CP_MT_THREADS;
    const int is_remainder_block = (bid == n_blocks - 1) && (remainder > 0);
    const int block_leaves = is_remainder_block ? remainder : CP_MT_THREADS;
    const int offset = bid * CP_MT_THREADS;
    uint key[8];
    d_mt_load_key_g(job_key, key);

    if (tid < block_leaves) {
        for (int i = 0; i < CP_MT_CV_WORDS; i++) {
            d_mt_set_leaf_word(smem_leaves, i, tid, CP_MT_THREADS,
                               d_b3_load32_g(roots + (size_t)(offset + tid) * D_B3_OUT +
                                             (size_t)i * 4));
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int use_blake_mt = is_remainder_block && ((block_leaves & (block_leaves - 1)) != 0);
    const int consider_root = is_single_block;

    if (use_blake_mt) {
        d_mt_compute_blake(smem_leaves, block_leaves, CP_MT_THREADS, key, consider_root);
    } else {
        d_mt_compute_perfect(smem_leaves, block_leaves, CP_MT_THREADS, key, consider_root);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < CP_MT_CV_WORDS) {
        const size_t base = is_single_block ? 0 : (size_t)bid * D_B3_OUT;
        d_b3_store32_g(roots + base + (size_t)tid * 4,
                       d_mt_leaf_word(smem_leaves, tid, 0, CP_MT_THREADS));
    }
}

__kernel void ocl_reduce_roots(__global const uchar* job_key, __global uchar* roots,
                               int num_leaves) {
    __local uint smem_leaves[CP_MT_CV_WORDS * CP_MT_THREADS];
    const int tid = get_local_id(0);
    uint key[8];
    d_mt_load_key_g(job_key, key);

    if (tid < num_leaves) {
        for (int i = 0; i < CP_MT_CV_WORDS; i++) {
            d_mt_set_leaf_word(smem_leaves, i, tid, CP_MT_THREADS,
                               d_b3_load32_g(roots + (size_t)tid * D_B3_OUT + (size_t)i * 4));
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (popcount((uint)num_leaves) == 1) {
        d_mt_compute_perfect(smem_leaves, num_leaves, CP_MT_THREADS, key, 1);
    } else {
        d_mt_compute_blake(smem_leaves, num_leaves, CP_MT_THREADS, key, 1);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < CP_MT_CV_WORDS) {
        d_b3_store32_g(roots + (size_t)tid * 4, d_mt_leaf_word(smem_leaves, tid, 0, CP_MT_THREADS));
    }
}
