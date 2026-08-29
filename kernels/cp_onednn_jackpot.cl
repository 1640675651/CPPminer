/* Device jackpot scan for oneDNN gemmstone milestoned tile_xor output.
 * tile_xor layout: dword_index = ms * tile_count + spatial_id (row-major tiles).
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

inline void b3_compress64(__global const uint *key8, const uint *msg16, uint *out8) {
    const uint kIV[8] = {0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
                         0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u};
    uint v[16] = {key8[0], key8[1], key8[2], key8[3], key8[4], key8[5], key8[6], key8[7],
                  kIV[0],  kIV[1],  kIV[2],  kIV[3],  0u,      0u,      64u,     0x1Bu};
    uint m[16];
    for (int i = 0; i < 16; ++i) {
        m[i] = msg16[i];
    }
    const uchar kPerm[16] = {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8};
    for (int round = 0; round < 7; ++round) {
        b3_g(v, 0, 4, 8, 12, m[0], m[1]);
        b3_g(v, 1, 5, 9, 13, m[2], m[3]);
        b3_g(v, 2, 6, 10, 14, m[4], m[5]);
        b3_g(v, 3, 7, 11, 15, m[6], m[7]);
        b3_g(v, 0, 5, 10, 15, m[8], m[9]);
        b3_g(v, 1, 6, 11, 12, m[10], m[11]);
        b3_g(v, 2, 7, 8, 13, m[12], m[13]);
        b3_g(v, 3, 4, 9, 14, m[14], m[15]);
        if (round < 6) {
            uint t[16];
            for (int i = 0; i < 16; ++i) {
                t[i] = m[kPerm[i]];
            }
            for (int i = 0; i < 16; ++i) {
                m[i] = t[i];
            }
        }
    }
    for (int i = 0; i < 8; ++i) {
        out8[i] = v[i] ^ v[i + 8];
    }
}

inline void fold_milestones(const uint *milestone_xor, int num_milestones, uint out_msg[PP_JACKPOT_WORDS]) {
    for (int i = 0; i < PP_JACKPOT_WORDS; ++i) {
        out_msg[i] = 0u;
    }
    for (int step = 0; step < num_milestones; ++step) {
        const int tid = step % PP_JACKPOT_WORDS;
        out_msg[tid] = pp_rotl32(out_msg[tid], PP_LROT) ^ milestone_xor[step];
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
                                     int tc_base, int hash_mr, int hash_nr,
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

    uint milestone_xor[PP_MAX_MILESTONES];
    for (int ms = 0; ms < num_milestones; ++ms) {
        milestone_xor[ms] =
                tile_xor[(size_t)ms * (size_t)tile_count + (size_t)sid];
    }

    uint msg[PP_JACKPOT_WORDS];
    fold_milestones(milestone_xor, num_milestones, msg);

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
