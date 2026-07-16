/* Device BLAKE3 for OpenCL prep (keyed matrix hash + noise XOF). */

#define D_B3_BLOCK 64
#define D_B3_CHUNK 1024
#define D_B3_OUT 32

#define D_B3_CHUNK_START (1u)
#define D_B3_CHUNK_END (2u)
#define D_B3_PARENT (4u)
#define D_B3_ROOT (8u)
#define D_B3_KEYED (16u)

constant uint D_B3_IV[8] = {
    0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
    0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u
};

constant uchar D_B3_MSG_SCHEDULE[7][16] = {
    {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15},
    {2,6,3,10,7,0,4,13,1,11,12,5,9,14,15,8},
    {3,4,10,12,13,2,7,14,6,5,9,0,11,15,8,1},
    {10,7,12,9,14,3,13,15,4,0,11,2,5,8,1,6},
    {12,13,9,11,15,10,14,8,7,2,5,3,0,1,6,4},
    {9,14,11,5,8,12,15,1,13,3,0,10,2,6,4,7},
    {11,15,5,0,1,9,8,6,14,10,2,12,3,4,7,13}
};

constant uchar CP_SEED_LABEL_A[32] = "A_tensor";
constant uchar CP_SEED_LABEL_B[32] = "B_tensor";

inline uint d_b3_load32_priv(const uchar *p) {
    return (uint)p[0] | ((uint)p[1] << 8) | ((uint)p[2] << 16) | ((uint)p[3] << 24);
}

inline uint d_b3_load32_g(__global const uchar *p) {
    return (uint)p[0] | ((uint)p[1] << 8) | ((uint)p[2] << 16) | ((uint)p[3] << 24);
}

inline void d_b3_store32_priv(uchar *p, uint w) {
    p[0] = (uchar)w;
    p[1] = (uchar)(w >> 8);
    p[2] = (uchar)(w >> 16);
    p[3] = (uchar)(w >> 24);
}

inline void d_b3_store32_g(__global uchar *p, uint w) {
    p[0] = (uchar)w;
    p[1] = (uchar)(w >> 8);
    p[2] = (uchar)(w >> 16);
    p[3] = (uchar)(w >> 24);
}

inline uint d_b3_rotr32(uint w, int c) {
    return (w >> c) | (w << (32 - c));
}

inline void d_b3_g(uint *s, int a, int b, int c, int d, uint x, uint y) {
    s[a] += s[b] + x;
    s[d] = d_b3_rotr32(s[d] ^ s[a], 16);
    s[c] += s[d];
    s[b] = d_b3_rotr32(s[b] ^ s[c], 12);
    s[a] += s[b] + y;
    s[d] = d_b3_rotr32(s[d] ^ s[a], 8);
    s[c] += s[d];
    s[b] = d_b3_rotr32(s[b] ^ s[c], 7);
}

inline void d_b3_round(uint s[16], const uint m[16], int round) {
    d_b3_g(s, 0, 4, 8, 12, m[D_B3_MSG_SCHEDULE[round][0]], m[D_B3_MSG_SCHEDULE[round][1]]);
    d_b3_g(s, 1, 5, 9, 13, m[D_B3_MSG_SCHEDULE[round][2]], m[D_B3_MSG_SCHEDULE[round][3]]);
    d_b3_g(s, 2, 6, 10, 14, m[D_B3_MSG_SCHEDULE[round][4]], m[D_B3_MSG_SCHEDULE[round][5]]);
    d_b3_g(s, 3, 7, 11, 15, m[D_B3_MSG_SCHEDULE[round][6]], m[D_B3_MSG_SCHEDULE[round][7]]);
    d_b3_g(s, 0, 5, 10, 15, m[D_B3_MSG_SCHEDULE[round][8]], m[D_B3_MSG_SCHEDULE[round][9]]);
    d_b3_g(s, 1, 6, 11, 12, m[D_B3_MSG_SCHEDULE[round][10]], m[D_B3_MSG_SCHEDULE[round][11]]);
    d_b3_g(s, 2, 7, 8, 13, m[D_B3_MSG_SCHEDULE[round][12]], m[D_B3_MSG_SCHEDULE[round][13]]);
    d_b3_g(s, 3, 4, 9, 14, m[D_B3_MSG_SCHEDULE[round][14]], m[D_B3_MSG_SCHEDULE[round][15]]);
}

inline void d_b3_compress_pre(uint s[16], const uint cv[8], const uchar block[D_B3_BLOCK],
                              uchar block_len, ulong counter, uchar flags) {
    uint m[16];
    for (int i = 0; i < 16; i++) {
        m[i] = d_b3_load32_priv(block + 4 * i);
    }
    s[0] = cv[0];
    s[1] = cv[1];
    s[2] = cv[2];
    s[3] = cv[3];
    s[4] = cv[4];
    s[5] = cv[5];
    s[6] = cv[6];
    s[7] = cv[7];
    s[8] = D_B3_IV[0];
    s[9] = D_B3_IV[1];
    s[10] = D_B3_IV[2];
    s[11] = D_B3_IV[3];
    s[12] = (uint)counter;
    s[13] = (uint)(counter >> 32);
    s[14] = (uint)block_len;
    s[15] = (uint)flags;
    for (int r = 0; r < 7; r++) {
        d_b3_round(s, m, r);
    }
}

inline void d_b3_compress_in_place(uint cv[8], const uchar block[D_B3_BLOCK], uchar block_len,
                                   ulong counter, uchar flags) {
    uint s[16];
    d_b3_compress_pre(s, cv, block, block_len, counter, flags);
    cv[0] = s[0] ^ s[8];
    cv[1] = s[1] ^ s[9];
    cv[2] = s[2] ^ s[10];
    cv[3] = s[3] ^ s[11];
    cv[4] = s[4] ^ s[12];
    cv[5] = s[5] ^ s[13];
    cv[6] = s[6] ^ s[14];
    cv[7] = s[7] ^ s[15];
}

inline void d_b3_key_words_priv(const uchar key[32], uint kw[8]) {
    for (int i = 0; i < 8; i++) {
        kw[i] = d_b3_load32_priv(key + 4 * i);
    }
}

inline void d_b3_key_words_g(__global const uchar *key, uint kw[8]) {
    for (int i = 0; i < 8; i++) {
        kw[i] = d_b3_load32_g(key + 4 * i);
    }
}

inline uchar d_b3_mat_padded_byte_g(__global const uchar *mat, ulong mat_off, ulong raw_len,
                                    int pos) {
    ulong gi = mat_off + (ulong)pos;
    return (gi < raw_len) ? mat[gi] : (uchar)0;
}

inline void d_b3_keyed_chunk_cv_glob(__global const uchar *key, ulong chunk_idx,
                                     __global const uchar *mat, ulong mat_off, ulong raw_len,
                                     int chunk_len, uchar cv_out[32]) {
    uint kw[8];
    d_b3_key_words_g(key, kw);
    uint cv[8];
    for (int i = 0; i < 8; i++) {
        cv[i] = kw[i];
    }
    int pos = 0;
    int blocks_compressed = 0;
    while (chunk_len - pos > D_B3_BLOCK) {
        uchar block[D_B3_BLOCK];
        for (int i = 0; i < D_B3_BLOCK; i++) {
            block[i] = d_b3_mat_padded_byte_g(mat, mat_off, raw_len, pos + i);
        }
        uchar fl = (uchar)(D_B3_KEYED | (blocks_compressed == 0 ? D_B3_CHUNK_START : 0));
        d_b3_compress_in_place(cv, block, D_B3_BLOCK, chunk_idx, fl);
        blocks_compressed++;
        pos += D_B3_BLOCK;
    }
    uchar tail[D_B3_BLOCK];
    for (int i = 0; i < D_B3_BLOCK; i++) {
        tail[i] = d_b3_mat_padded_byte_g(mat, mat_off, raw_len, pos + i);
    }
    d_b3_compress_in_place(cv, tail, (uchar)(chunk_len - pos), chunk_idx,
                           (uchar)(D_B3_KEYED | D_B3_CHUNK_END |
                                   (blocks_compressed == 0 ? D_B3_CHUNK_START : 0)));
    for (int i = 0; i < 8; i++) {
        d_b3_store32_priv(cv_out + 4 * i, cv[i]);
    }
}

typedef struct {
    uint cv[8];
    ulong chunk_counter;
    uchar buf[D_B3_BLOCK];
    uchar buf_len;
    uchar blocks_compressed;
    uchar flags;
} d_b3_chunk_state;

inline void d_b3_chunk_init(d_b3_chunk_state *st, const uint key[8], uchar flags) {
    for (int i = 0; i < 8; i++) {
        st->cv[i] = key[i];
    }
    st->chunk_counter = 0;
    st->buf_len = 0;
    st->blocks_compressed = 0;
    st->flags = flags;
    for (int i = 0; i < D_B3_BLOCK; i++) {
        st->buf[i] = 0;
    }
}

inline void d_b3_chunk_update(d_b3_chunk_state *st, const uchar *input, ulong input_len) {
    if (st->buf_len > 0) {
        ulong take = D_B3_BLOCK - st->buf_len;
        if (take > input_len) {
            take = input_len;
        }
        for (ulong i = 0; i < take; i++) {
            st->buf[st->buf_len + i] = input[i];
        }
        st->buf_len = (uchar)(st->buf_len + take);
        input += take;
        input_len -= take;
        if (input_len > 0 && st->buf_len == D_B3_BLOCK) {
            uchar f = (uchar)(st->flags | (st->blocks_compressed == 0 ? D_B3_CHUNK_START : 0));
            d_b3_compress_in_place(st->cv, st->buf, D_B3_BLOCK, st->chunk_counter, f);
            st->blocks_compressed++;
            st->buf_len = 0;
            for (int i = 0; i < D_B3_BLOCK; i++) {
                st->buf[i] = 0;
            }
        }
    }

    while (input_len > D_B3_BLOCK) {
        uchar f = (uchar)(st->flags | (st->blocks_compressed == 0 ? D_B3_CHUNK_START : 0));
        d_b3_compress_in_place(st->cv, input, D_B3_BLOCK, st->chunk_counter, f);
        st->blocks_compressed++;
        input += D_B3_BLOCK;
        input_len -= D_B3_BLOCK;
    }

    for (ulong i = 0; i < input_len; i++) {
        st->buf[st->buf_len + i] = input[i];
    }
    st->buf_len = (uchar)(st->buf_len + input_len);
}

inline void d_b3_compress_xof(const uint cv[8], const uchar block[D_B3_BLOCK], uchar block_len,
                              ulong counter, uchar flags, uchar out[64]) {
    uint s[16];
    d_b3_compress_pre(s, cv, block, block_len, counter, flags);
    d_b3_store32_priv(out + 0, s[0] ^ s[8]);
    d_b3_store32_priv(out + 4, s[1] ^ s[9]);
    d_b3_store32_priv(out + 8, s[2] ^ s[10]);
    d_b3_store32_priv(out + 12, s[3] ^ s[11]);
    d_b3_store32_priv(out + 16, s[4] ^ s[12]);
    d_b3_store32_priv(out + 20, s[5] ^ s[13]);
    d_b3_store32_priv(out + 24, s[6] ^ s[14]);
    d_b3_store32_priv(out + 28, s[7] ^ s[15]);
    d_b3_store32_priv(out + 32, s[8] ^ cv[0]);
    d_b3_store32_priv(out + 36, s[9] ^ cv[1]);
    d_b3_store32_priv(out + 40, s[10] ^ cv[2]);
    d_b3_store32_priv(out + 44, s[11] ^ cv[3]);
    d_b3_store32_priv(out + 48, s[12] ^ cv[4]);
    d_b3_store32_priv(out + 52, s[13] ^ cv[5]);
    d_b3_store32_priv(out + 56, s[14] ^ cv[6]);
    d_b3_store32_priv(out + 60, s[15] ^ cv[7]);
}

inline void d_b3_chunk_root_out(const d_b3_chunk_state *st, uchar out[32]) {
    uchar f = (uchar)(st->flags | D_B3_CHUNK_END | D_B3_ROOT);
    if (st->blocks_compressed == 0) {
        f = (uchar)(f | D_B3_CHUNK_START);
    }
    uchar wide[64];
    d_b3_compress_xof(st->cv, st->buf, st->buf_len, st->chunk_counter, f, wide);
    for (int i = 0; i < 32; i++) {
        out[i] = wide[i];
    }
}

inline void d_keyed_digest_priv(const uchar *data, int len, const uchar key[32], uchar out[32]) {
    uint kw[8];
    d_b3_key_words_priv(key, kw);
    d_b3_chunk_state st;
    d_b3_chunk_init(&st, kw, D_B3_KEYED);
    d_b3_chunk_update(&st, data, (ulong)len);
    d_b3_chunk_root_out(&st, out);
}

inline void d_copy_label_priv(int is_b, uchar key[32]) {
    for (int i = 0; i < 32; i++) {
        key[i] = is_b ? CP_SEED_LABEL_B[i] : CP_SEED_LABEL_A[i];
    }
}

inline void d_get_random_hash_priv(int index, const uchar seed[32], const uchar key[32],
                                   int prepend_index, uchar out[32]) {
    uchar msg[64];
    for (int i = 0; i < 64; i++) {
        msg[i] = 0;
    }
    int prep = 1 + index;
    msg[prepend_index * 4] = (uchar)prep;
    msg[prepend_index * 4 + 1] = (uchar)(prep >> 8);
    msg[prepend_index * 4 + 2] = (uchar)(prep >> 16);
    msg[prepend_index * 4 + 3] = (uchar)(prep >> 24);
    for (int i = 0; i < 32; i++) {
        msg[32 + i] = seed[i];
    }
    d_keyed_digest_priv(msg, 64, key, out);
}

inline void d_get_random_hash_glob(int index, __global const uchar *noise_seed, int is_b,
                                   int prepend_index, uchar out[32]) {
    uchar seed_local[32];
    uchar key_local[32];
    d_copy_label_priv(is_b, seed_local);
    for (int i = 0; i < 32; i++) {
        key_local[i] = noise_seed[i];
    }
    d_get_random_hash_priv(index, seed_local, key_local, prepend_index, out);
}

inline uint cp_mul_hi_u32(uint a, uint b) {
    return (uint)(((ulong)a * b) >> 32);
}

inline ulong cp_splitmix64(ulong x) {
    x += 0x9E3779B97F4A7C15UL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9UL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBUL;
    return x ^ (x >> 31);
}
