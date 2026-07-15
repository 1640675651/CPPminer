/* Device BLAKE3 compression for plain_proof jackpot kernel. */
#ifndef CP_GPU_CUH
#define CP_GPU_CUH

#include <stdint.h>

__device__ __constant__ uint32_t B3IV[8] = {
    0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
    0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u
};

__device__ __forceinline__ uint32_t d_rotr32(uint32_t x, int n){
    return (x >> n) | (x << (32 - n));
}

__device__ __forceinline__ void b3G(uint32_t* v,int a,int b,int c,int d,uint32_t x,uint32_t y){
    v[a]+=v[b]+x; v[d]=d_rotr32(v[d]^v[a],16);
    v[c]+=v[d];   v[b]=d_rotr32(v[b]^v[c],12);
    v[a]+=v[b]+y; v[d]=d_rotr32(v[d]^v[a], 8);
    v[c]+=v[d];   v[b]=d_rotr32(v[b]^v[c], 7);
}

__device__ __forceinline__ void b3_compress64(const uint32_t* key8, const uint32_t* msg16,
                               uint32_t* out8)
{
    uint32_t v[16]={
        key8[0],key8[1],key8[2],key8[3],
        key8[4],key8[5],key8[6],key8[7],
        B3IV[0],B3IV[1],B3IV[2],B3IV[3],
        0,0,64u,0x1Bu
    };
    uint32_t m[16];
    for(int i=0;i<16;i++) m[i]=msg16[i];

    const uint8_t PERM[16]={2,6,3,10,7,0,4,13,1,11,12,5,9,14,15,8};
    for(int round=0;round<7;round++){
        b3G(v,0,4, 8,12,m[0], m[1]);
        b3G(v,1,5, 9,13,m[2], m[3]);
        b3G(v,2,6,10,14,m[4], m[5]);
        b3G(v,3,7,11,15,m[6], m[7]);
        b3G(v,0,5,10,15,m[8], m[9]);
        b3G(v,1,6,11,12,m[10],m[11]);
        b3G(v,2,7, 8,13,m[12],m[13]);
        b3G(v,3,4, 9,14,m[14],m[15]);
        if(round<6){
            uint32_t t[16];
            for(int i=0;i<16;i++) t[i]=m[PERM[i]];
            for(int i=0;i<16;i++) m[i]=t[i];
        }
    }
    for(int i=0;i<8;i++) out8[i]=v[i]^v[i+8];
}

#endif /* CP_GPU_CUH */
