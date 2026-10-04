/* Device jackpot scan for oneDNN gemmstone milestoned tile_xor output.
 * tile_xor layout (raw milestones): dword_index = ms * tile_count + spatial_id.
 * Case 5.5 wrap-GRF flush: 16 folded msg words per tile, same ms-major layout.
 * Fold + BLAKE3 match cp_jackpot.hpp / plain_proof host verify. */

#define PP_JACKPOT_WORDS 16
#define PP_LROT 13
#ifndef PP_MAX_MILESTONES
#define PP_MAX_MILESTONES 64
#endif

inline uint pp_rotl32(uint x, int s) { return (x << s) | (x >> (32 - s)); }

inline uint b3_rotr32(uint x, int n) { return (x >> n) | (x << (32 - n)); }

inline void b3_g(uint *v, int a, int b, int c, int d, uint x, uint y) {
    v[a] += v[b] + x;
    v[d] = b3_rotr32(v[d] ^ v[a], 16);
    v[c] += v[d];
    v[b] = b3_rotr32(v[b] ^ v[c], 12);
    v[a] += v[b] + y;
    v[d] = b3_rotr32(v[d] ^ v[a], 8);
    v[c] += v[d];
    v[b] = b3_rotr32(v[b] ^ v[c], 7);
}

/* BLAKE3 message schedule per round (permutation applied r times), so every
 * index below is a compile-time constant after unrolling and the state stays
 * in registers instead of private memory. */
__constant uchar kB3Sched[7][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8},
    {3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1},
    {10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6},
    {12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4},
    {9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7},
    {11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13},
};

inline void b3_compress64(__global const uint *key8, const uint *msg16, uint *out8) {
    uint v[16] = {key8[0], key8[1], key8[2], key8[3], key8[4], key8[5], key8[6], key8[7],
                  0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au, 0u, 0u, 64u, 0x1Bu};
#pragma unroll
    for (int round = 0; round < 7; ++round) {
#define M(i) msg16[kB3Sched[round][i]]
        b3_g(v, 0, 4, 8, 12, M(0), M(1));
        b3_g(v, 1, 5, 9, 13, M(2), M(3));
        b3_g(v, 2, 6, 10, 14, M(4), M(5));
        b3_g(v, 3, 7, 11, 15, M(6), M(7));
        b3_g(v, 0, 5, 10, 15, M(8), M(9));
        b3_g(v, 1, 6, 11, 12, M(10), M(11));
        b3_g(v, 2, 7, 8, 13, M(12), M(13));
        b3_g(v, 3, 4, 9, 14, M(14), M(15));
#undef M
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        out8[i] = v[i] ^ v[i + 8];
    }
}

/* Fold of milestone ms into word ms % 16: msg[w] = rotl(msg[w], 13) ^ xor[ms],
 * read straight from tile_xor (ms-major) with constant word indices. */
inline void fold_milestones_glob(__global const uint *tile_xor, int num_milestones,
                                 int tile_count, int sid, uint out_msg[PP_JACKPOT_WORDS]) {
#pragma unroll
    for (int w = 0; w < PP_JACKPOT_WORDS; ++w) {
        uint acc = 0u;
        for (int ms = w; ms < num_milestones; ms += PP_JACKPOT_WORDS) {
            acc = pp_rotl32(acc, PP_LROT) ^ tile_xor[(size_t)ms * (size_t)tile_count + (size_t)sid];
        }
        out_msg[w] = acc;
    }
}
inline bool digest_beats_target(const uint digest[8], __global const uint *bound) {
    for (int w = 7; w >= 0; --w) {
        if (digest[w] < bound[w]) {
            return true;
        }
        if (digest[w] > bound[w]) {
            return false;
        }
    }
    return true;
}

__kernel void cp_onednn_jackpot_scan(__global const uint *tile_xor, int num_milestones,
                                     int tile_count, int panel_tile_cols, int tr_base,
                                     int tc_base, int hash_mr, int hash_nr, int use_folded_msg,
                                     __global const uint *a_key8, __global const uint *bound8,
                                     __global volatile int *found_flag, __global int *out_t_rows,
                                     __global int *out_t_cols) {
    const int sid = get_global_id(0);
    if (sid >= tile_count) {
        return;
    }
    if (found_flag != 0 && *found_flag != 0) {
        return;
    }
    if (num_milestones <= 0 || num_milestones > PP_MAX_MILESTONES) {
        return;
    }

    uint msg[PP_JACKPOT_WORDS];
    if (use_folded_msg != 0) {
        if (num_milestones != PP_JACKPOT_WORDS) {
            return;
        }
        for (int w = 0; w < PP_JACKPOT_WORDS; ++w) {
            msg[w] = tile_xor[(size_t)w * (size_t)tile_count + (size_t)sid];
        }
    } else {
        fold_milestones_glob(tile_xor, num_milestones, tile_count, sid, msg);

    }

    uint digest[8];
    b3_compress64(a_key8, msg, digest);
    if (!digest_beats_target(digest, bound8)) {
        return;
    }
    if (found_flag == 0) {
        return;
    }
    if (atomic_cmpxchg(found_flag, 0, 1) != 0) {
        return;
    }

    const int tr = sid / panel_tile_cols;
    const int tc = sid - tr * panel_tile_cols;
    const int t_rows = (tr_base + tr) * hash_mr;
    const int t_cols = (tc_base + tc) * hash_nr;
    if (out_t_rows != 0) {
        *out_t_rows = t_rows;
    }
    if (out_t_cols != 0) {
        *out_t_cols = t_cols;
    }
}

/* GPU BLAKE3 on folded tile_xor panel; digest layout w*tile_count+sid. CPU judges difficulty. */
__kernel void cp_onednn_blake3_panel(__global const uint *tile_xor, int num_words, int tile_count,
                                       __global const uint *a_key8, __global uint *digest_out) {
    const int sid = get_global_id(0);
    if (sid >= tile_count || num_words <= 0) {
        return;
    }
    uint msg[PP_JACKPOT_WORDS];
    for (int w = 0; w < num_words && w < PP_JACKPOT_WORDS; ++w) {
        msg[w] = tile_xor[(size_t)w * (size_t)tile_count + (size_t)sid];
    }
    for (int w = num_words; w < PP_JACKPOT_WORDS; ++w) {
        msg[w] = 0u;
    }
    uint digest[8];
    b3_compress64(a_key8, msg, digest);
    for (int w = 0; w < 8; ++w) {
        digest_out[(size_t)w * (size_t)tile_count + (size_t)sid] = digest[w];
    }
}
