/* plain_proof jackpot kernel — 8×16 periodic hash tiles. */
#ifndef PLAIN_PROOF_KERNEL_CUH
#define PLAIN_PROOF_KERNEL_CUH

#define PP_HASH_H 8
#define PP_HASH_W 16
#define PP_JACKPOT_WORDS 16
#define PP_LROT 13

__constant__ int PP_ROW_PAT[PP_HASH_H] = {
    0, 8, 32, 40, 64, 72, 96, 104
};
__constant__ int PP_COL_PAT[PP_HASH_W] = {
    0, 1, 32, 33, 64, 65, 96, 97,
    128, 129, 160, 161, 192, 193, 224, 225
};
__constant__ int PP_CONTIGUOUS_MODE = 0;
/* 1 = step-major panels (default); 0 = row-major m×k (cuBLAS lda=K_DIM). */
__constant__ int PP_STEP_MAJOR_AP = 1;

__device__ __forceinline__ uint32_t pp_rotl32(uint32_t x, int s) {
    return (x << s) | (x >> (32 - s));
}

/* Scattered offsets within 128/256 periods; contiguous mode uses block-aligned tiles. */
__device__ __forceinline__ int pp_row_trows(int part_idx) {
    if(PP_CONTIGUOUS_MODE) return part_idx * PP_HASH_H;
    const int base[16] = {
        0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23
    };
    return (part_idx / 16) * 128 + base[part_idx % 16];
}

__device__ __forceinline__ int pp_col_tcols(int part_idx) {
    if(PP_CONTIGUOUS_MODE) return part_idx * PP_HASH_W;
    /* Valid t_cols for scattered cols_pattern (pearl_mining offset_is_valid): even 0..30 */
    const int base[16] = {
        0, 2, 4, 6, 8, 10, 12, 14,
        16, 18, 20, 22, 24, 26, 28, 30
    };
    return (part_idx / 16) * 256 + base[part_idx % 16];
}

/* int8 row dot [l0,l1) within one rank panel (local offsets 0..R). */
__device__ __forceinline__ int32_t pp_dot_i8_panel(
    int32_t acc, const int8_t* a_row, const int8_t* b_row, int l0, int l1)
{
    for(int l = l0; l < l1; l += 4){
        acc = __dp4a(*(const int32_t*)(a_row + l), *(const int32_t*)(b_row + l), acc);
    }
    return acc;
}

/* Ap/BpT step-major: [step][row_or_col][r] with step plane = dim * R. */
__device__ __forceinline__ const int8_t* pp_ap_step_row(
    const int8_t* ap, int step, int row, int m, int r)
{
    return ap + ((size_t)step * (size_t)m + (size_t)row) * (size_t)r;
}

__device__ __forceinline__ const int8_t* pp_bp_step_row(
    const int8_t* bp, int step, int col, int n, int r)
{
    return bp + ((size_t)step * (size_t)n + (size_t)col) * (size_t)r;
}

__device__ __forceinline__ const int8_t* pp_ap_panel_ptr(
    const int8_t* ap, int step, int row, int m, int k_dim, int r)
{
    if(PP_STEP_MAJOR_AP)
        return pp_ap_step_row(ap, step, row, m, r);
    return ap + (size_t)row * (size_t)k_dim + (size_t)step * (size_t)r;
}

__device__ __forceinline__ const int8_t* pp_bp_panel_ptr(
    const int8_t* bp, int step, int col, int n, int k_dim, int r)
{
    if(PP_STEP_MAJOR_AP)
        return pp_bp_step_row(bp, step, col, n, r);
    return bp + (size_t)col * (size_t)k_dim + (size_t)step * (size_t)r;
}

/* int8 row dot [l0,l1); R and K are multiples of 4. Max |acc| < 4096*128^2 fits int32. */
__device__ __forceinline__ int32_t pp_dot_i8_rows(
    int32_t acc, const int8_t* a_row, const int8_t* b_row, int l0, int l1)
{
    for(int l = l0; l < l1; l += 4){
        acc = __dp4a(*(const int32_t*)(a_row + l), *(const int32_t*)(b_row + l), acc);
    }
    return acc;
}

/*
 * One block = one hash tile at (row_part, col_part).
 * A, B: int8 step-major noisy matrices Ap[step][row][r], Bp[step][col][r].
 * bound: jackpot hash target (LE uint256), already rank-penalized
 * (pool_target × h×w×(k/r)×PENALTY_BASE_RANK).
 */
__global__ void plain_proof_jackpot_kernel(
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    int M, int N, int K, int R,
    int row_part_base,
    int col_part_base,
    int row_parts,
    int col_parts,
    uint32_t b0, uint32_t b1, uint32_t b2, uint32_t b3,
    uint32_t b4, uint32_t b5, uint32_t b6, uint32_t b7,
    const uint32_t* __restrict__ a_key8,
    int* __restrict__ out_t_rows,
    int* __restrict__ out_t_cols,
    int* __restrict__ found_flag
) {
    const int rp = row_part_base + (int)blockIdx.y;
    const int cp = col_part_base + (int)blockIdx.x;
    if(rp >= row_parts || cp >= col_parts) return;

    const int t_rows = pp_row_trows(rp);
    const int t_cols = pp_col_tcols(cp);
    const int u = (int)threadIdx.y;
    const int v = (int)threadIdx.x;
    const int ar = t_rows + PP_ROW_PAT[u];
    const int bc = t_cols + PP_COL_PAT[v];
    if(ar >= M || bc >= N) return;

    const int8_t* a_row = pp_ap_panel_ptr(A, 0, ar, M, K, R);
    const int8_t* b_row = pp_bp_panel_ptr(B, 0, bc, N, K, R);

    __shared__ int32_t s_tile[PP_HASH_H * PP_HASH_W];
    __shared__ uint32_t s_jackpot[PP_JACKPOT_WORDS];

    if(u == 0 && v == 0)
        for(int i = 0; i < PP_JACKPOT_WORDS; i++) s_jackpot[i] = 0u;
    s_tile[u * PP_HASH_W + v] = 0;
    __syncthreads();

    int32_t cell = 0;
    for(int ll = R; ll <= K; ll += R) {
        const int step = (ll / R) - 1;
        a_row = pp_ap_panel_ptr(A, step, ar, M, K, R);
        b_row = pp_bp_panel_ptr(B, step, bc, N, K, R);
        cell = pp_dot_i8_panel(cell, a_row, b_row, 0, R);
        s_tile[u * PP_HASH_W + v] = cell;
        __syncthreads();

        if(u == 0 && v == 0) {
            uint32_t xored = 0u;
            for(int i = 0; i < PP_HASH_H * PP_HASH_W; i++)
                xored ^= (uint32_t)s_tile[i];
            int tid = step % PP_JACKPOT_WORDS;
            s_jackpot[tid] = pp_rotl32(s_jackpot[tid], PP_LROT) ^ xored;
        }
        __syncthreads();
    }

    if(u != 0 || v != 0) return;

    uint32_t msg[PP_JACKPOT_WORDS];
    for(int i = 0; i < PP_JACKPOT_WORDS; i++) msg[i] = s_jackpot[i];

    /* Keyed BLAKE3 single block (defined above in pearl_mine.cu). */
    uint32_t digest[8];
    b3_compress64(a_key8, msg, digest);

    bool ok = false;
    uint32_t tgt[8] = {b0, b1, b2, b3, b4, b5, b6, b7};
    for(int w = 7; w >= 0; w--) {
        if(digest[w] < tgt[w]) { ok = true; break; }
        if(digest[w] > tgt[w]) { ok = false; break; }
        if(w == 0) ok = true;
    }

    if(ok && atomicCAS(found_flag, 0, 1) == 0) {
        *out_t_rows = t_rows;
        *out_t_cols = t_cols;
    }
}

#endif /* PLAIN_PROOF_KERNEL_CUH */
