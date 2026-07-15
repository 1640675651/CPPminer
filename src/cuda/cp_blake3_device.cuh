/* Device BLAKE3 (portable) for matrix XOF and keyed chunk hashing. */
#ifndef CP_BLAKE3_DEVICE_CUH
#define CP_BLAKE3_DEVICE_CUH

#include <stdint.h>
#include <string.h>

#define D_B3_BLOCK 64
#define D_B3_CHUNK 1024
#define D_B3_OUT 32

#define D_B3_CHUNK_START (1u << 0)
#define D_B3_CHUNK_END   (1u << 1)
#define D_B3_PARENT      (1u << 2)
#define D_B3_ROOT        (1u << 3)
#define D_B3_KEYED       (1u << 4)

__device__ __constant__ uint32_t D_B3_IV[8] = {
    0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
    0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u
};

__device__ __constant__ uint8_t D_B3_MSG_SCHEDULE[7][16] = {
    {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15},
    {2,6,3,10,7,0,4,13,1,11,12,5,9,14,15,8},
    {3,4,10,12,13,2,7,14,6,5,9,0,11,15,8,1},
    {10,7,12,9,14,3,13,15,4,0,11,2,5,8,1,6},
    {12,13,9,11,15,10,14,8,7,2,5,3,0,1,6,4},
    {9,14,11,5,8,12,15,1,13,3,0,10,2,6,4,7},
    {11,15,5,0,1,9,8,6,14,10,2,12,3,4,7,13}
};

__device__ __forceinline__ uint32_t d_b3_load32(const uint8_t* p){
    return (uint32_t)p[0] | ((uint32_t)p[1]<<8) | ((uint32_t)p[2]<<16) | ((uint32_t)p[3]<<24);
}

__device__ __forceinline__ void d_b3_store32(uint8_t* p, uint32_t w){
    p[0]=(uint8_t)w; p[1]=(uint8_t)(w>>8); p[2]=(uint8_t)(w>>16); p[3]=(uint8_t)(w>>24);
}

__device__ __forceinline__ uint32_t d_b3_rotr32(uint32_t w, int c){
    return (w >> c) | (w << (32 - c));
}

__device__ __forceinline__ void d_b3_g(uint32_t* s,int a,int b,int c,int d,uint32_t x,uint32_t y){
    s[a]+=s[b]+x; s[d]=d_b3_rotr32(s[d]^s[a],16);
    s[c]+=s[d];   s[b]=d_b3_rotr32(s[b]^s[c],12);
    s[a]+=s[b]+y; s[d]=d_b3_rotr32(s[d]^s[a], 8);
    s[c]+=s[d];   s[b]=d_b3_rotr32(s[b]^s[c], 7);
}

__device__ __forceinline__ void d_b3_round(uint32_t s[16], const uint32_t m[16], int round){
    const uint8_t* sch = D_B3_MSG_SCHEDULE[round];
    d_b3_g(s,0,4, 8,12,m[sch[0]], m[sch[1]]);
    d_b3_g(s,1,5, 9,13,m[sch[2]], m[sch[3]]);
    d_b3_g(s,2,6,10,14,m[sch[4]], m[sch[5]]);
    d_b3_g(s,3,7,11,15,m[sch[6]], m[sch[7]]);
    d_b3_g(s,0,5,10,15,m[sch[8]], m[sch[9]]);
    d_b3_g(s,1,6,11,12,m[sch[10]],m[sch[11]]);
    d_b3_g(s,2,7, 8,13,m[sch[12]],m[sch[13]]);
    d_b3_g(s,3,4, 9,14,m[sch[14]],m[sch[15]]);
}

__device__ __forceinline__ void d_b3_compress_pre(
    uint32_t s[16], const uint32_t cv[8], const uint8_t block[D_B3_BLOCK],
    uint8_t block_len, uint64_t counter, uint8_t flags)
{
    uint32_t m[16];
    for(int i=0;i<16;i++) m[i]=d_b3_load32(block + 4*i);
    s[0]=cv[0]; s[1]=cv[1]; s[2]=cv[2]; s[3]=cv[3];
    s[4]=cv[4]; s[5]=cv[5]; s[6]=cv[6]; s[7]=cv[7];
    s[8]=D_B3_IV[0]; s[9]=D_B3_IV[1]; s[10]=D_B3_IV[2]; s[11]=D_B3_IV[3];
    s[12]=(uint32_t)counter;
    s[13]=(uint32_t)(counter >> 32);
    s[14]=(uint32_t)block_len;
    s[15]=(uint32_t)flags;
    for(int r=0;r<7;r++) d_b3_round(s,m,r);
}

__device__ __forceinline__ void d_b3_compress_in_place(
    uint32_t cv[8], const uint8_t block[D_B3_BLOCK],
    uint8_t block_len, uint64_t counter, uint8_t flags)
{
    uint32_t s[16];
    d_b3_compress_pre(s, cv, block, block_len, counter, flags);
    cv[0]=s[0]^s[8]; cv[1]=s[1]^s[9]; cv[2]=s[2]^s[10]; cv[3]=s[3]^s[11];
    cv[4]=s[4]^s[12]; cv[5]=s[5]^s[13]; cv[6]=s[6]^s[14]; cv[7]=s[7]^s[15];
}

__device__ __forceinline__ void d_b3_compress_xof(
    const uint32_t cv[8], const uint8_t block[D_B3_BLOCK],
    uint8_t block_len, uint64_t counter, uint8_t flags, uint8_t out[64])
{
    uint32_t s[16];
    d_b3_compress_pre(s, cv, block, block_len, counter, flags);
    d_b3_store32(out+0,  s[0]^s[8]);  d_b3_store32(out+4,  s[1]^s[9]);
    d_b3_store32(out+8,  s[2]^s[10]); d_b3_store32(out+12, s[3]^s[11]);
    d_b3_store32(out+16, s[4]^s[12]); d_b3_store32(out+20, s[5]^s[13]);
    d_b3_store32(out+24, s[6]^s[14]); d_b3_store32(out+28, s[7]^s[15]);
    d_b3_store32(out+32, s[8]^cv[0]); d_b3_store32(out+36, s[9]^cv[1]);
    d_b3_store32(out+40, s[10]^cv[2]); d_b3_store32(out+44, s[11]^cv[3]);
    d_b3_store32(out+48, s[12]^cv[4]); d_b3_store32(out+52, s[13]^cv[5]);
    d_b3_store32(out+56, s[14]^cv[6]); d_b3_store32(out+60, s[15]^cv[7]);
}

struct d_b3_chunk_state {
    uint32_t cv[8];
    uint64_t chunk_counter;
    uint8_t buf[D_B3_BLOCK];
    uint8_t buf_len;
    uint8_t blocks_compressed;
    uint8_t flags;
};

__device__ __forceinline__ void d_b3_chunk_init(d_b3_chunk_state* st, const uint32_t key[8], uint8_t flags){
    for(int i=0;i<8;i++) st->cv[i]=key[i];
    st->chunk_counter=0;
    st->buf_len=0;
    st->blocks_compressed=0;
    st->flags=flags;
    for(int i=0;i<D_B3_BLOCK;i++) st->buf[i]=0;
}

__device__ __forceinline__ void d_b3_chunk_update(d_b3_chunk_state* st, const uint8_t* input, size_t input_len){
    if(st->buf_len > 0){
        size_t take = D_B3_BLOCK - st->buf_len;
        if(take > input_len) take = input_len;
        for(size_t i = 0; i < take; i++) st->buf[st->buf_len + i] = input[i];
        st->buf_len = (uint8_t)(st->buf_len + take);
        input += take;
        input_len -= take;
        if(input_len > 0 && st->buf_len == D_B3_BLOCK){
            uint8_t f = (uint8_t)(st->flags | (st->blocks_compressed == 0 ? D_B3_CHUNK_START : 0));
            d_b3_compress_in_place(st->cv, st->buf, D_B3_BLOCK, st->chunk_counter, f);
            st->blocks_compressed++;
            st->buf_len = 0;
            for(int i = 0; i < D_B3_BLOCK; i++) st->buf[i] = 0;
        }
    }

    while(input_len > D_B3_BLOCK){
        uint8_t f = (uint8_t)(st->flags | (st->blocks_compressed == 0 ? D_B3_CHUNK_START : 0));
        d_b3_compress_in_place(st->cv, input, D_B3_BLOCK, st->chunk_counter, f);
        st->blocks_compressed++;
        input += D_B3_BLOCK;
        input_len -= D_B3_BLOCK;
    }

    for(size_t i = 0; i < input_len; i++)
        st->buf[st->buf_len + i] = input[i];
    st->buf_len = (uint8_t)(st->buf_len + input_len);
}

__device__ __forceinline__ void d_b3_chunk_root_out(const d_b3_chunk_state* st, uint8_t out[32])
{
    uint8_t f = (uint8_t)(st->flags | D_B3_CHUNK_END | D_B3_ROOT);
    if(st->blocks_compressed == 0) f = (uint8_t)(f | D_B3_CHUNK_START);
    uint8_t wide[64];
    d_b3_compress_xof(st->cv, st->buf, st->buf_len, st->chunk_counter, f, wide);
    for(int i = 0; i < 32; i++) out[i] = wide[i];
}

__device__ __forceinline__ void d_b3_output_root_bytes(
    const d_b3_chunk_state* st, uint64_t seek, uint8_t* out, size_t out_len)
{
    uint8_t f = (uint8_t)(st->flags | D_B3_CHUNK_END | D_B3_ROOT);
    if(st->blocks_compressed == 0) f = (uint8_t)(f | D_B3_CHUNK_START);
    uint64_t block_ctr = seek / 64;
    size_t off = (size_t)(seek % 64);
    uint8_t wide[64];
    if(off){
        d_b3_compress_xof(st->cv, st->buf, st->buf_len, block_ctr, f, wide);
        size_t avail = 64 - off;
        size_t n = out_len < avail ? out_len : avail;
        for(size_t i = 0; i < n; i++) out[i] = wide[off + i];
        out += n; out_len -= n;
        block_ctr++;
    }
    while(out_len >= 64){
        d_b3_compress_xof(st->cv, st->buf, st->buf_len, block_ctr, f, out);
        out += 64; out_len -= 64;
        block_ctr++;
    }
    if(out_len){
        d_b3_compress_xof(st->cv, st->buf, st->buf_len, block_ctr, f, wide);
        for(size_t i = 0; i < out_len; i++) out[i] = wide[i];
    }
}

__device__ __forceinline__ void d_b3_xof_seek(
    const uint8_t* prefix, int prefix_len,
    const uint8_t* seed, int seed_len,
    uint64_t seek, uint8_t* out, size_t out_len)
{
    d_b3_chunk_state st;
    d_b3_chunk_init(&st, D_B3_IV, 0);
    d_b3_chunk_update(&st, prefix, (size_t)prefix_len);
    d_b3_chunk_update(&st, seed, (size_t)seed_len);
    d_b3_output_root_bytes(&st, seek, out, out_len);
}

__device__ __forceinline__ void d_b3_key_words(const uint8_t key[32], uint32_t kw[8]){
    for(int i=0;i<8;i++) kw[i]=d_b3_load32(key + 4*i);
}

__device__ __forceinline__ uint8_t d_b3_mat_padded_byte(
    const uint8_t* mat, size_t mat_off, size_t raw_len, int pos)
{
    size_t gi = mat_off + (size_t)pos;
    return (gi < raw_len) ? mat[gi] : (uint8_t)0;
}

/* Keyed chunk CV from matrix bytes; zero-pads past raw_len (no 1 KiB thread-local copy). */
__device__ __forceinline__ void d_b3_keyed_chunk_cv_glob(
    const uint8_t key[32], uint64_t chunk_idx,
    const uint8_t* mat, size_t mat_off, size_t raw_len, int chunk_len,
    uint8_t cv_out[32])
{
    uint32_t kw[8];
    d_b3_key_words(key, kw);
    uint32_t cv[8];
    for(int i = 0; i < 8; i++) cv[i] = kw[i];
    int pos = 0;
    int blocks_compressed = 0;
    while(chunk_len - pos > D_B3_BLOCK){
        uint8_t block[D_B3_BLOCK];
        for(int i = 0; i < D_B3_BLOCK; i++)
            block[i] = d_b3_mat_padded_byte(mat, mat_off, raw_len, pos + i);
        uint8_t fl = (uint8_t)(D_B3_KEYED | (blocks_compressed == 0 ? D_B3_CHUNK_START : 0));
        d_b3_compress_in_place(cv, block, D_B3_BLOCK, chunk_idx, fl);
        blocks_compressed++;
        pos += D_B3_BLOCK;
    }
    uint8_t tail[D_B3_BLOCK];
    for(int i = 0; i < D_B3_BLOCK; i++)
        tail[i] = d_b3_mat_padded_byte(mat, mat_off, raw_len, pos + i);
    d_b3_compress_in_place(cv, tail, (uint8_t)(chunk_len - pos), chunk_idx,
                           (uint8_t)(D_B3_KEYED | D_B3_CHUNK_END |
                                     (blocks_compressed == 0 ? D_B3_CHUNK_START : 0)));
    for(int i = 0; i < 8; i++) d_b3_store32(cv_out + 4 * i, cv[i]);
}

__device__ __forceinline__ void d_b3_keyed_chunk_cv(
    const uint8_t key[32], uint64_t chunk_idx,
    const uint8_t* chunk_data, int chunk_len, uint8_t cv_out[32])
{
    uint32_t kw[8];
    d_b3_key_words(key, kw);
    uint32_t cv[8];
    for(int i=0;i<8;i++) cv[i]=kw[i];
    int pos = 0;
    int blocks_compressed = 0;
    while(chunk_len - pos > D_B3_BLOCK){
        uint8_t fl = (uint8_t)(D_B3_KEYED | (blocks_compressed == 0 ? D_B3_CHUNK_START : 0));
        d_b3_compress_in_place(cv, chunk_data + pos, D_B3_BLOCK, chunk_idx, fl);
        blocks_compressed++;
        pos += D_B3_BLOCK;
    }
    uint8_t tail[D_B3_BLOCK];
    for(int i=0;i<D_B3_BLOCK;i++){
        int gi = pos + i;
        tail[i] = (gi < chunk_len) ? chunk_data[gi] : 0;
    }
    d_b3_compress_in_place(cv, tail, (uint8_t)(chunk_len - pos), chunk_idx,
                           (uint8_t)(D_B3_KEYED | D_B3_CHUNK_END |
                                     (blocks_compressed == 0 ? D_B3_CHUNK_START : 0)));
    for(int i=0;i<8;i++) d_b3_store32(cv_out + 4*i, cv[i]);
}

#endif /* CP_BLAKE3_DEVICE_CUH */
