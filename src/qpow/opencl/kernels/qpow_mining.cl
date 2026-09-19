/* Quantus Poseidon2 QPoW — OpenCL port of engine-gpu mining_u64.wgsl (native ulong).
 * Bit-exact with pow_core / qpow host midstate path. */

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

#define P64  ((ulong)0xFFFFFFFF00000001UL)
#define EPS64 ((ulong)0xFFFFFFFFUL)

__constant ulong RC_INTERNAL[22] = {
    0x97f7798a784ad863UL, 0xd1d2bf082f60d4f0UL, 0x69a377a79f9ad206UL, 0xa9d06906a3858e24UL,
    0x295275001eede5b5UL, 0x5874e441117bd746UL, 0x8a084bbba8ed86ccUL, 0x3defd7645cde6425UL,
    0x3998cfe6871cc137UL, 0x3e52ef8bca48314aUL, 0x964a209f85dc9eccUL, 0x3fcc9ee82cc4577eUL,
    0x8e79b4a5d0096d6dUL, 0x8492362ad2392556UL, 0xee72f470262574d6UL, 0x1e0e18496da2444aUL,
    0x0f3a74bf215eaac6UL, 0x1b061b76a1c0ded3UL, 0x192c42d86803d7a6UL, 0xf6d49ff997ae0260UL,
    0x3ec372e7a0fa3786UL, 0x5538cdf4f23445d3UL
};

__constant ulong RC_INITIAL[4][12] = {
    {0xc002e770975b1607UL, 0xbca51a8dfe14593aUL, 0x72938dfbe774f7f9UL, 0xe4f2fe29e03234acUL,
     0xd5e0ba2f541b6449UL, 0xec33b868f3cc46c1UL, 0x486dcb55419d475aUL, 0x6c1cb2a358cc24f1UL,
     0xe3f30d509a1436bbUL, 0xd9a64f068dca7c29UL, 0xe59b3f57aabba1aeUL, 0x2a3dd4505b478fdcUL},
    {0xada1f8dc7676ed25UL, 0x2711aa8b5509d516UL, 0x4ae6acd0c9c92897UL, 0x56eb3d6b5256d67aUL,
     0x1f7a9d55923bf51eUL, 0x3600427d397a7f68UL, 0xe5076df75b72c3d0UL, 0xfcd59aa12c6090adUL,
     0xcd895e8c68b57a9eUL, 0x41df7ef9d730ae3eUL, 0xee3e2b889abe977dUL, 0xd29bb7edbeb9c405UL},
    {0x7d5c08eef608e382UL, 0x89ae889caaf0802cUL, 0xb35a8e976d2af617UL, 0xdb14234eafaf5173UL,
     0x78f04462d48b1c98UL, 0x265293b0e47ce88aUL, 0x999a649b69b9d32fUL, 0x64b0a186698e01d3UL,
     0xee0b22d0dfae8bb8UL, 0x4fd53e50ca04a7eeUL, 0x5762bfe181f25047UL, 0xf51593e2beb5e3bdUL},
    {0x1e5e2b5760e32477UL, 0x622462a1f9aaaeedUL, 0xaa284b3ecdb222aeUL, 0x63c8e72f542bf3fcUL,
     0x3ba588cacb43b5e0UL, 0x23eda6f3c99150ddUL, 0xaad3bea4baac9a5aUL, 0xe9da8d699b94184aUL,
     0xcdb13f4cd93e024cUL, 0x902cbd0956f655e3UL, 0x5b4e40ffc759532fUL, 0xde795c20a2357af7UL}
};

__constant ulong RC_TERMINAL[4][12] = {
    {0x7b72c539e0ea4c6eUL, 0x144573dae2ce9976UL, 0x802028b68f35fc88UL, 0x6d36c5022c4fe7c2UL,
     0xa205d0ffa9b9def3UL, 0xf6e7e38b1ea6ba2fUL, 0x34f7909ae5258d64UL, 0xb0464d9d77b97fcaUL,
     0x64ddb9d5de7e00a6UL, 0x0ed0d75c27975d97UL, 0x1cbb36f11127338bUL, 0x6673e505cfd0b6baUL},
    {0x605f902830872e01UL, 0x3fd5eb927e95fe4fUL, 0xe81025b5a24c69cdUL, 0xf7d0ce75de23f74eUL,
     0xf39942b6a8585089UL, 0x6d808a08f7b71df6UL, 0xf8806b6588f49a8bUL, 0x57df2d8c2a32107aUL,
     0x16e7c2074d654a2dUL, 0x213de241fcf33835UL, 0xb0f2b8905a0976f6UL, 0xd8e3cf2bbd355417UL},
    {0xe498691679d9330fUL, 0x763b45d2a3821b28UL, 0x0908bf65eb0a1f0dUL, 0x7691eb2d194b24f4UL,
     0x0e43551233ae13b2UL, 0x93c393dbfc2fe76fUL, 0x98f607485d48cdeaUL, 0xe3d95f30309819c0UL,
     0x1ef581a93eaf6acfUL, 0x0b24c1b7a030fca4UL, 0x624370be5670b327UL, 0x5f1e28615a11e486UL},
    {0xfe04051f909e042bUL, 0x7257e5b147fd3803UL, 0xe6ae134bb82f2e78UL, 0x5711fd5cf4784511UL,
     0xf83a42660c08c0bcUL, 0x2cd8c96d9a3ce855UL, 0x7d2ffb1bb0e17271UL, 0x85ae1528caea3811UL,
     0x52a345d5c7adb0b8UL, 0x504c4c51f3faee94UL, 0xbce34a649cfccaf9UL, 0xe0a3389266fb6dc9UL}
};

__constant ulong MDS_DIAG[12] = {
    0xc3b6c08e23ba9300UL, 0xd84b5de94a324fb6UL, 0x0d0c371c5b35b84fUL, 0x7964f570e7188037UL,
    0x5daf18bbd996604bUL, 0x6743bc47b9595257UL, 0x5528b9362c59bb70UL, 0xac45e25b7127b68bUL,
    0xa2077d7dfbb606b5UL, 0xf3faac6faee378aeUL, 0x0c6388b51545e883UL, 0xd27dbb6944917b60UL
};

ulong gf64_add(ulong a, ulong b)
{
    ulong s0 = a + b;
    int c1 = (s0 < a);
    ulong s1 = s0 + (c1 ? EPS64 : 0UL);
    int c2 = c1 && (s1 < s0);
    return s1 + (c2 ? EPS64 : 0UL);
}

ulong gf64_reduce(ulong lo, ulong hi)
{
    ulong hi_hi = hi >> 32;
    ulong hi_lo = hi & EPS64;
    ulong t0 = lo - hi_hi;
    t0 = t0 - ((lo < hi_hi) ? EPS64 : 0UL);
    ulong t1 = hi_lo * EPS64;
    ulong t2 = t0 + t1;
    return t2 + ((t2 < t0) ? EPS64 : 0UL);
}

ulong gf64_mul(ulong a, ulong b)
{
    ulong a_lo = a & EPS64;
    ulong a_hi = a >> 32;
    ulong b_lo = b & EPS64;
    ulong b_hi = b >> 32;
    ulong ll = a_lo * b_lo;
    ulong lh = a_lo * b_hi;
    ulong hl = a_hi * b_lo;
    ulong hh = a_hi * b_hi;
    ulong mid = lh + hl;
    ulong mid_c = (mid < lh) ? 1UL : 0UL;
    ulong lo = ll + (mid << 32);
    ulong lo_c = (lo < ll) ? 1UL : 0UL;
    ulong hi = hh + (mid >> 32) + (mid_c << 32) + lo_c;
    return gf64_reduce(lo, hi);
}

ulong gf64_sqr(ulong a)
{
    ulong a_lo = a & EPS64;
    ulong a_hi = a >> 32;
    ulong ll = a_lo * a_lo;
    ulong lh = a_lo * a_hi;
    ulong hh = a_hi * a_hi;
    ulong mid = lh << 1;
    ulong mid_c = lh >> 63;
    ulong lo = ll + (mid << 32);
    ulong lo_c = (lo < ll) ? 1UL : 0UL;
    ulong hi = hh + (mid >> 32) + (mid_c << 32) + lo_c;
    return gf64_reduce(lo, hi);
}

ulong gf64_sbox(ulong x)
{
    ulong x2 = gf64_sqr(x);
    ulong x4 = gf64_sqr(x2);
    ulong x6 = gf64_mul(x4, x2);
    return gf64_mul(x6, x);
}

ulong gf64_canon(ulong a)
{
    return a - ((a >= P64) ? P64 : 0UL);
}

void ext_layer64(ulong state[12])
{
    for(uint chunk = 0; chunk < 3u; chunk++){
        uint o = chunk * 4u;
        ulong x0 = state[o];
        ulong x1 = state[o + 1];
        ulong x2 = state[o + 2];
        ulong x3 = state[o + 3];
        ulong t01 = gf64_add(x0, x1);
        ulong t23 = gf64_add(x2, x3);
        ulong t0123 = gf64_add(t01, t23);
        ulong t01123 = gf64_add(t0123, x1);
        ulong t01233 = gf64_add(t0123, x3);
        state[o + 3] = gf64_add(t01233, gf64_add(x0, x0));
        state[o + 1] = gf64_add(t01123, gf64_add(x2, x2));
        state[o] = gf64_add(t01123, t01);
        state[o + 2] = gf64_add(t01233, t23);
    }
    ulong sums[4];
    for(uint k = 0; k < 4u; k++)
        sums[k] = gf64_add(gf64_add(state[k], state[k + 4]), state[k + 8]);
    for(uint i = 0; i < 12u; i++)
        state[i] = gf64_add(state[i], sums[i % 4u]);
}

void int_layer64(ulong state[12])
{
    ulong sum = state[0];
    for(uint i = 1; i < 12u; i++)
        sum = gf64_add(sum, state[i]);
    for(uint i = 0; i < 12u; i++)
        state[i] = gf64_add(gf64_mul(state[i], MDS_DIAG[i]), sum);
}

void permute64(ulong state[12])
{
    ext_layer64(state);
    for(uint r = 0; r < 4u; r++){
        for(uint i = 0; i < 12u; i++)
            state[i] = gf64_add(state[i], RC_INITIAL[r][i]);
        for(uint i = 0; i < 12u; i++)
            state[i] = gf64_sbox(state[i]);
        ext_layer64(state);
    }
    for(uint r = 0; r < 22u; r++){
        state[0] = gf64_sbox(gf64_add(state[0], RC_INTERNAL[r]));
        int_layer64(state);
    }
    for(uint r = 0; r < 4u; r++){
        for(uint i = 0; i < 12u; i++)
            state[i] = gf64_add(state[i], RC_TERMINAL[r][i]);
        for(uint i = 0; i < 12u; i++)
            state[i] = gf64_sbox(state[i]);
        ext_layer64(state);
    }
}

uint bswap32(uint v)
{
    return ((v & 0xFFu) << 24) | ((v & 0xFF00u) << 8) | ((v >> 8) & 0xFF00u) | (v >> 24);
}

/*
 * Buffers (same layout as wgpu):
 *   results[0]     = found flag
 *   results[1..16] = nonce LE u32s
 *   results[17..32]= hash LE u32s (BE-compared layout as wgpu)
 *   midstate       = 24 LE u32s (12 felts)
 *   start_nonce    = 16 LE u32s
 *   target         = 16 LE u32s (U512 LE limbs)
 *   dispatch_config= [total_threads, nonces_per_thread, total_nonces]
 */
__kernel void mining_main(
    __global volatile uint* results,
    __global const uint* midstate,
    __global const uint* start_nonce,
    __global const uint* difficulty_target,
    __global const uint* dispatch_config)
{
    if(results[0] != 0u)
        return;

    uint thread_id = get_global_id(0);
    uint total_threads = dispatch_config[0];
    uint nonces_per_thread = dispatch_config[1];
    uint total_nonces = dispatch_config[2];
    if(thread_id >= total_threads)
        return;

    uint base_index = thread_id * nonces_per_thread;

    ulong mid[12];
    for(uint i = 0; i < 12u; i++)
        mid[i] = (((ulong)midstate[2u * i + 1u]) << 32) | (ulong)midstate[2u * i];

    uint tgt[16];
    for(uint i = 0; i < 16u; i++)
        tgt[i] = difficulty_target[i];

    uint nonce_base[16];
    for(uint i = 0; i < 16u; i++)
        nonce_base[i] = start_nonce[i];

    for(uint j = 0; j < nonces_per_thread; j++){
        uint logical_index = base_index + j;
        if(logical_index >= total_nonces)
            break;
        if(j > 0u && results[0] != 0u)
            return;

        uint current_nonce[16];
        uint val0 = nonce_base[0];
        uint sum0 = val0 + logical_index;
        current_nonce[0] = sum0;
        uint carry = (sum0 < val0) ? 1u : 0u;
        for(uint i = 1; i < 8u; i++){
            uint val = nonce_base[i];
            uint sum = val + carry;
            current_nonce[i] = sum;
            carry = (sum < val) ? 1u : 0u;
        }
        for(uint i = 8; i < 16u; i++)
            current_nonce[i] = nonce_base[i];

        ulong st[12];
        for(uint i = 0; i < 12u; i++)
            st[i] = mid[i];
        for(uint i = 0; i < 8u; i++)
            st[i] = gf64_add(st[i], (ulong)bswap32(current_nonce[7u - i]));
        permute64(st);
        st[0] = gf64_add(st[0], 1UL);
        st[1] = gf64_add(st[1], 1UL);
        permute64(st);

        uint first[8];
        for(uint i = 0; i < 4u; i++){
            ulong c = gf64_canon(st[i]);
            first[2u * i] = (uint)(c & EPS64);
            first[2u * i + 1u] = (uint)(c >> 32);
        }
        uint cmp = 0u;
        for(uint i = 0; i < 8u; i++){
            uint h = bswap32(first[i]);
            uint t = tgt[15u - i];
            if(h != t){
                cmp = (h > t) ? 1u : 2u;
                break;
            }
        }
        if(cmp == 1u)
            continue;

        uint hash_le[16];
        for(uint i = 0; i < 8u; i++)
            hash_le[15u - i] = bswap32(first[i]);
        permute64(st);
        for(uint i = 0; i < 4u; i++){
            ulong c = gf64_canon(st[i]);
            hash_le[7u - 2u * i] = bswap32((uint)(c & EPS64));
            hash_le[6u - 2u * i] = bswap32((uint)(c >> 32));
        }
        int below = (cmp == 2u);
        if(!below){
            for(uint i = 0; i < 8u; i++){
                uint h = hash_le[7u - i];
                uint t = tgt[7u - i];
                if(h != t){
                    below = (h < t);
                    break;
                }
            }
        }

        if(below){
            if(atomic_cmpxchg(results, 0u, 1u) == 0u){
                for(uint i = 0; i < 16u; i++){
                    results[1u + i] = current_nonce[i];
                    results[17u + i] = hash_le[i];
                }
            }
            return;
        }
    }
}
