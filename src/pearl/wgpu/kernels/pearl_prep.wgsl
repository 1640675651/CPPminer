// pearl_prep.wgsl - GPU prep: BLAKE3 helpers + merkle + gen_random + build_pairs + fused_prepack
// Ports essentials from cp_ocl_blake3.cl, cp_ocl_merkle.cl, cp_ocl_prep.cl
//
// Binding layouts -- each entry uses its own @group (WGSL requires unique (group,binding)
// in one module). Host pipelines bind that group; binding indices match below.
//
// Entry: pearl_gen_random_matrix  @workgroup_size(256)  @group(0)
//   @binding(0) params : uniform PearlGenRandomParams  { rng_seed: vec2<u32>, matrix_tag, total_elems }
//   @binding(1) out    : storage rw array<i32>         // one signed byte value per element (-64..63)
//
// Entry: pearl_build_perm_pairs  @workgroup_size(64)  @group(0) bindings 2-4
//   @binding(0) params      : uniform PearlBuildPairsParams { is_b, k, rank, _pad }
//   @binding(1) noise_seed  : storage read  array<u32, 8>   // 32-byte seed as LE words
//   @binding(2) pairs_out   : storage rw    array<u32>      // [first,second] per column
//
// Entry: pearl_fused_prepack_b  @workgroup_size(8)  @group(0) bindings 5-9
//   @binding(0) params            : uniform PearlPrepackBParams
//   @binding(1) b_pre_out         : storage rw   array<u32>  // packed bytes as LE u32
//   @binding(2) b_noise_seed      : storage read array<u32, 8>
//   @binding(3) pairs             : storage read array<u32>
//   @binding(4) b_signal_colmajor : storage read array<i32>  // signed bytes (optional; has_signal)
//
// Entry: pearl_fused_prepack_a  @workgroup_size(8)  @group(0) bindings 10-14
//   @binding(0) params       : uniform PearlPrepackAParams
//   @binding(1) a_pre_out    : storage rw   array<u32>
//   @binding(2) a_noise_seed : storage read array<u32, 8>
//   @binding(3) pairs        : storage read array<u32>
//   @binding(4) a_signal     : storage read array<i32>
//
// Entry: pearl_keyed_chunk_roots  @workgroup_size(256)  @group(0) bindings 15-18
//   @binding(0) params     : uniform PearlMerkleChunkParams { raw_len, pad_len, num_chunks, _pad }
//   @binding(1) mat        : storage read  array<i32>       // matrix bytes as signed i32 elems
//   @binding(2) job_key    : storage read  array<u32, 8>
//   @binding(3) roots_out  : storage rw    array<u32>       // 8 words per block root
//
// Entry: pearl_compute_blake_mt  @workgroup_size(256)  @group(0) bindings 19-21
//   @binding(0) params  : uniform PearlMerkleMtParams { num_leaves, is_single_block, _pad0, _pad1 }
//   @binding(1) job_key : storage read  array<u32, 8>
//   @binding(2) roots   : storage rw    array<u32>
//
// Entry: pearl_reduce_roots  @workgroup_size(256)  @group(0) bindings 22-24
//   @binding(0) params  : uniform PearlReduceRootsParams { num_leaves, _pad0, _pad1, _pad2 }
//   @binding(1) job_key : storage read  array<u32, 8>
//   @binding(2) roots   : storage rw    array<u32>

const MR: i32 = 8;
const NR: i32 = 8;
const KR: i32 = 128;
const R_RANK: i32 = 128;
const MACRO_M: i32 = 128;
const MACRO_N: i32 = 128;
const MICRO_M: i32 = 16;
const MICRO_N: i32 = 16;
const K_GROUPS: i32 = 32;
const KG_BYTES_A: i32 = 32;
const KG_SLICE_B: i32 = 32;
const MACRO_KG_STRIP_A: i32 = 512;
const MACRO_KG_STRIP_B: i32 = 512;
const MACRO_KB_BLOCK_A: i32 = 16384;
const MACRO_KB_BLOCK_B: i32 = 16384;

const D_B3_BLOCK: i32 = 64;
const D_B3_CHUNK: i32 = 1024;
const D_B3_OUT: i32 = 32;
const D_B3_CHUNK_START: u32 = 1u;
const D_B3_CHUNK_END: u32 = 2u;
const D_B3_PARENT: u32 = 4u;
const D_B3_ROOT: u32 = 8u;
const D_B3_KEYED: u32 = 16u;

const CP_RANGE_MASK: u32 = 63u;
const CP_ZERO_PT: i32 = 16;
const CP_B3_LINES: i32 = 8;
const CP_MT_THREADS: i32 = 256;
const CP_MT_CV_WORDS: i32 = 8;

// u64 as vec2<u32>: .x = lo, .y = hi
alias U64 = vec2<u32>;

fn u64_from_u32(x: u32) -> U64 {
    return vec2(x, 0u);
}

fn u64_from_i32(x: i32) -> U64 {
    return vec2(u32(x), 0u);
}

fn u64_add(a: U64, b: U64) -> U64 {
    let sum_lo = a.x + b.x;
    let carry = select(0u, 1u, sum_lo < a.x);
    return vec2(sum_lo, a.y + b.y + carry);
}

fn u64_xor(a: U64, b: U64) -> U64 {
    return vec2(a.x ^ b.x, a.y ^ b.y);
}

fn u64_shr(a: U64, n: u32) -> U64 {
    if (n == 0u) {
        return a;
    }
    if (n >= 64u) {
        return vec2(0u, 0u);
    }
    if (n >= 32u) {
        return vec2(a.y >> (n - 32u), 0u);
    }
    return vec2((a.x >> n) | (a.y << (32u - n)), a.y >> n);
}

fn mul_wide_u32(a: u32, b: u32) -> U64 {
    let a_lo = a & 0xffffu;
    let a_hi = a >> 16u;
    let b_lo = b & 0xffffu;
    let b_hi = b >> 16u;
    let p0 = a_lo * b_lo;
    let p1 = a_lo * b_hi;
    let p2 = a_hi * b_lo;
    let p3 = a_hi * b_hi;
    let mid = (p0 >> 16u) + (p1 & 0xffffu) + (p2 & 0xffffu);
    let lo = (p0 & 0xffffu) | (mid << 16u);
    let hi = p3 + (p1 >> 16u) + (p2 >> 16u) + (mid >> 16u);
    return vec2(lo, hi);
}

fn u64_mul(a: U64, b: U64) -> U64 {
    let p0 = mul_wide_u32(a.x, b.x);
    let cross = a.x * b.y + a.y * b.x;
    return vec2(p0.x, p0.y + cross);
}

fn cp_mul_hi_u32(a: u32, b: u32) -> u32 {
    return mul_wide_u32(a, b).y;
}

fn cp_splitmix64(x_in: U64) -> U64 {
    var x = u64_add(x_in, vec2(0x7F4A7C15u, 0x9E3779B9u)); // + 0x9E3779B97F4A7C15
    x = u64_mul(u64_xor(x, u64_shr(x, 30u)), vec2(0x1CE4E5B9u, 0xBF58476Du));
    x = u64_mul(u64_xor(x, u64_shr(x, 27u)), vec2(0x133111EBu, 0x94D049BBu));
    return u64_xor(x, u64_shr(x, 31u));
}

fn d_b3_rotr32(w: u32, c: u32) -> u32 {
    return (w >> c) | (w << (32u - c));
}

fn d_b3_g(s: ptr<function, array<u32, 16>>, a: i32, b: i32, c: i32, d: i32, x: u32, y: u32) {
    (*s)[a] = (*s)[a] + (*s)[b] + x;
    (*s)[d] = d_b3_rotr32((*s)[d] ^ (*s)[a], 16u);
    (*s)[c] = (*s)[c] + (*s)[d];
    (*s)[b] = d_b3_rotr32((*s)[b] ^ (*s)[c], 12u);
    (*s)[a] = (*s)[a] + (*s)[b] + y;
    (*s)[d] = d_b3_rotr32((*s)[d] ^ (*s)[a], 8u);
    (*s)[c] = (*s)[c] + (*s)[d];
    (*s)[b] = d_b3_rotr32((*s)[b] ^ (*s)[c], 7u);
}

fn d_b3_msg_schedule(round: i32, i: i32) -> i32 {
    // D_B3_MSG_SCHEDULE[7][16] from OpenCL
    let r0 = array<i32, 16>(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    let r1 = array<i32, 16>(2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8);
    let r2 = array<i32, 16>(3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1);
    let r3 = array<i32, 16>(10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6);
    let r4 = array<i32, 16>(12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4);
    let r5 = array<i32, 16>(9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7);
    let r6 = array<i32, 16>(11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13);
    if (round == 0) { return r0[i]; }
    if (round == 1) { return r1[i]; }
    if (round == 2) { return r2[i]; }
    if (round == 3) { return r3[i]; }
    if (round == 4) { return r4[i]; }
    if (round == 5) { return r5[i]; }
    return r6[i];
}

fn d_b3_round(s: ptr<function, array<u32, 16>>, m: ptr<function, array<u32, 16>>, round: i32) {
    d_b3_g(s, 0, 4, 8, 12, (*m)[d_b3_msg_schedule(round, 0)], (*m)[d_b3_msg_schedule(round, 1)]);
    d_b3_g(s, 1, 5, 9, 13, (*m)[d_b3_msg_schedule(round, 2)], (*m)[d_b3_msg_schedule(round, 3)]);
    d_b3_g(s, 2, 6, 10, 14, (*m)[d_b3_msg_schedule(round, 4)], (*m)[d_b3_msg_schedule(round, 5)]);
    d_b3_g(s, 3, 7, 11, 15, (*m)[d_b3_msg_schedule(round, 6)], (*m)[d_b3_msg_schedule(round, 7)]);
    d_b3_g(s, 0, 5, 10, 15, (*m)[d_b3_msg_schedule(round, 8)], (*m)[d_b3_msg_schedule(round, 9)]);
    d_b3_g(s, 1, 6, 11, 12, (*m)[d_b3_msg_schedule(round, 10)], (*m)[d_b3_msg_schedule(round, 11)]);
    d_b3_g(s, 2, 7, 8, 13, (*m)[d_b3_msg_schedule(round, 12)], (*m)[d_b3_msg_schedule(round, 13)]);
    d_b3_g(s, 3, 4, 9, 14, (*m)[d_b3_msg_schedule(round, 14)], (*m)[d_b3_msg_schedule(round, 15)]);
}

fn d_b3_iv(i: i32) -> u32 {
    let iv = array<u32, 8>(
        0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
        0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u
    );
    return iv[i];
}

fn d_b3_load32_bytes(bytes: ptr<function, array<u32, 64>>, off: i32) -> u32 {
    // bytes[] stores one byte in low 8 bits of each u32
    return ((*bytes)[off])
        | ((*bytes)[off + 1] << 8u)
        | ((*bytes)[off + 2] << 16u)
        | ((*bytes)[off + 3] << 24u);
}

fn d_b3_store32_bytes(bytes: ptr<function, array<u32, 64>>, off: i32, w: u32) {
    (*bytes)[off] = w & 0xffu;
    (*bytes)[off + 1] = (w >> 8u) & 0xffu;
    (*bytes)[off + 2] = (w >> 16u) & 0xffu;
    (*bytes)[off + 3] = (w >> 24u) & 0xffu;
}

fn d_b3_store32_bytes32(bytes: ptr<function, array<u32, 32>>, off: i32, w: u32) {
    (*bytes)[off] = w & 0xffu;
    (*bytes)[off + 1] = (w >> 8u) & 0xffu;
    (*bytes)[off + 2] = (w >> 16u) & 0xffu;
    (*bytes)[off + 3] = (w >> 24u) & 0xffu;
}

fn d_b3_load32_bytes32(bytes: ptr<function, array<u32, 32>>, off: i32) -> u32 {
    return ((*bytes)[off])
        | ((*bytes)[off + 1] << 8u)
        | ((*bytes)[off + 2] << 16u)
        | ((*bytes)[off + 3] << 24u);
}

fn d_b3_compress_pre(
    s: ptr<function, array<u32, 16>>,
    cv: ptr<function, array<u32, 8>>,
    block: ptr<function, array<u32, 64>>,
    block_len: u32,
    counter_lo: u32,
    counter_hi: u32,
    flags: u32,
) {
    var m: array<u32, 16>;
    for (var i = 0; i < 16; i = i + 1) {
        m[i] = d_b3_load32_bytes(block, 4 * i);
    }
    (*s)[0] = (*cv)[0];
    (*s)[1] = (*cv)[1];
    (*s)[2] = (*cv)[2];
    (*s)[3] = (*cv)[3];
    (*s)[4] = (*cv)[4];
    (*s)[5] = (*cv)[5];
    (*s)[6] = (*cv)[6];
    (*s)[7] = (*cv)[7];
    (*s)[8] = d_b3_iv(0);
    (*s)[9] = d_b3_iv(1);
    (*s)[10] = d_b3_iv(2);
    (*s)[11] = d_b3_iv(3);
    (*s)[12] = counter_lo;
    (*s)[13] = counter_hi;
    (*s)[14] = block_len;
    (*s)[15] = flags;
    for (var r = 0; r < 7; r = r + 1) {
        d_b3_round(s, &m, r);
    }
}

fn d_b3_compress_in_place(
    cv: ptr<function, array<u32, 8>>,
    block: ptr<function, array<u32, 64>>,
    block_len: u32,
    counter_lo: u32,
    counter_hi: u32,
    flags: u32,
) {
    var s: array<u32, 16>;
    d_b3_compress_pre(&s, cv, block, block_len, counter_lo, counter_hi, flags);
    (*cv)[0] = s[0] ^ s[8];
    (*cv)[1] = s[1] ^ s[9];
    (*cv)[2] = s[2] ^ s[10];
    (*cv)[3] = s[3] ^ s[11];
    (*cv)[4] = s[4] ^ s[12];
    (*cv)[5] = s[5] ^ s[13];
    (*cv)[6] = s[6] ^ s[14];
    (*cv)[7] = s[7] ^ s[15];
}

fn d_b3_compress_xof(
    cv: ptr<function, array<u32, 8>>,
    block: ptr<function, array<u32, 64>>,
    block_len: u32,
    counter_lo: u32,
    counter_hi: u32,
    flags: u32,
    out: ptr<function, array<u32, 64>>,
) {
    var s: array<u32, 16>;
    d_b3_compress_pre(&s, cv, block, block_len, counter_lo, counter_hi, flags);
    d_b3_store32_bytes(out, 0, s[0] ^ s[8]);
    d_b3_store32_bytes(out, 4, s[1] ^ s[9]);
    d_b3_store32_bytes(out, 8, s[2] ^ s[10]);
    d_b3_store32_bytes(out, 12, s[3] ^ s[11]);
    d_b3_store32_bytes(out, 16, s[4] ^ s[12]);
    d_b3_store32_bytes(out, 20, s[5] ^ s[13]);
    d_b3_store32_bytes(out, 24, s[6] ^ s[14]);
    d_b3_store32_bytes(out, 28, s[7] ^ s[15]);
    d_b3_store32_bytes(out, 32, s[8] ^ (*cv)[0]);
    d_b3_store32_bytes(out, 36, s[9] ^ (*cv)[1]);
    d_b3_store32_bytes(out, 40, s[10] ^ (*cv)[2]);
    d_b3_store32_bytes(out, 44, s[11] ^ (*cv)[3]);
    d_b3_store32_bytes(out, 48, s[12] ^ (*cv)[4]);
    d_b3_store32_bytes(out, 52, s[13] ^ (*cv)[5]);
    d_b3_store32_bytes(out, 56, s[14] ^ (*cv)[6]);
    d_b3_store32_bytes(out, 60, s[15] ^ (*cv)[7]);
}

fn sign_extend_i8(b: u32) -> i32 {
    return (i32(b << 24u)) >> 24;
}

fn i32_as_u8(v: i32) -> u32 {
    return u32(bitcast<u32>(v)) & 0xffu;
}

fn load_s8_from_u32_word(word: u32, byte_idx: u32) -> i32 {
    let shift = (byte_idx & 3u) * 8u;
    return sign_extend_i8((word >> shift) & 0xffu);
}

fn seed_label_byte(is_b: i32, i: i32) -> u32 {
    // "A_tensor" / "B_tensor" padded to 32 zeros
    // A_tensor: 41 5F 74 65 6E 73 6F 72
    // B_tensor: 42 5F 74 65 6E 73 6F 72
    if (i >= 8) {
        return 0u;
    }
    let rest = array<u32, 7>(0x5Fu, 0x74u, 0x65u, 0x6Eu, 0x73u, 0x6Fu, 0x72u);
    if (i == 0) {
        return select(0x41u, 0x42u, is_b != 0);
    }
    return rest[i - 1];
}

fn d_keyed_digest_64(msg: ptr<function, array<u32, 64>>, key: ptr<function, array<u32, 32>>, out: ptr<function, array<u32, 32>>) {
    var kw: array<u32, 8>;
    for (var i = 0; i < 8; i = i + 1) {
        kw[i] = d_b3_load32_bytes32(key, 4 * i);
    }
    var cv: array<u32, 8>;
    for (var i = 0; i < 8; i = i + 1) {
        cv[i] = kw[i];
    }
    var buf: array<u32, 64>;
    for (var i = 0; i < 64; i = i + 1) {
        buf[i] = (*msg)[i];
    }
    // Single 64-byte input: one block then root (chunk_state path for len==64)
    // OpenCL: chunk_update 64 bytes into empty buf -> buf_len=64, then on root:
    // Actually chunk_update with empty buf and 64 bytes: buf_len starts 0, goes to while(input_len > BLOCK)?
    // while (input_len > D_B3_BLOCK) — 64 > 64 is false, so copies all 64 into buf, buf_len=64.
    // root: compress_xof with CHUNK_END|ROOT|CHUNK_START (blocks_compressed==0), block_len=64.
    var flags = D_B3_KEYED | D_B3_CHUNK_END | D_B3_ROOT | D_B3_CHUNK_START;
    var wide: array<u32, 64>;
    d_b3_compress_xof(&cv, &buf, 64u, 0u, 0u, flags, &wide);
    for (var i = 0; i < 32; i = i + 1) {
        (*out)[i] = wide[i];
    }
}

fn d_get_random_hash(
    index: i32,
    is_b: i32,
    noise_seed: array<u32, 8>,
    prepend_index: i32,
    out: ptr<function, array<u32, 32>>,
) {
    var seed_local: array<u32, 32>;
    var key_local: array<u32, 32>;
    for (var i = 0; i < 32; i = i + 1) {
        seed_local[i] = seed_label_byte(is_b, i);
    }
    for (var i = 0; i < 8; i = i + 1) {
        let w = noise_seed[i];
        key_local[4 * i] = w & 0xffu;
        key_local[4 * i + 1] = (w >> 8u) & 0xffu;
        key_local[4 * i + 2] = (w >> 16u) & 0xffu;
        key_local[4 * i + 3] = (w >> 24u) & 0xffu;
    }

    var msg: array<u32, 64>;
    for (var i = 0; i < 64; i = i + 1) {
        msg[i] = 0u;
    }
    let prep = 1 + index;
    let pi = prepend_index * 4;
    msg[pi] = u32(prep) & 0xffu;
    msg[pi + 1] = u32(prep >> 8) & 0xffu;
    msg[pi + 2] = u32(prep >> 16) & 0xffu;
    msg[pi + 3] = u32(prep >> 24) & 0xffu;
    for (var i = 0; i < 32; i = i + 1) {
        msg[32 + i] = seed_local[i];
    }
    d_keyed_digest_64(&msg, &key_local, out);
}

fn ocl_generate_uniform_row(
    row_idx: i32,
    num_cols: i32,
    noise_seed: array<u32, 8>,
    is_b: i32,
    row_out: ptr<function, array<u32, 128>>,
) {
    let start_idx = row_idx * num_cols;
    var block = start_idx / D_B3_OUT;
    var out_i = 0;
    while (block * D_B3_OUT < start_idx + num_cols) {
        var digest: array<u32, 32>;
        d_get_random_hash(block, is_b, noise_seed, 0, &digest);
        for (var k = 0; k < D_B3_OUT; k = k + 1) {
            let idx = block * D_B3_OUT + k;
            if (idx >= start_idx && idx < start_idx + num_cols) {
                let v = i32(digest[k] & CP_RANGE_MASK) - CP_ZERO_PT;
                (*row_out)[out_i] = i32_as_u8(v);
                out_i = out_i + 1;
            }
        }
        block = block + 1;
    }
}

// Writes into global b_pre_out / a_pre_out (naga rejects ptr<storage> function args).
fn store_u8_packed_b(byte_off: u32, val: u32) {
    let wi = byte_off / 4u;
    let shift = (byte_off % 4u) * 8u;
    let mask = ~(0xffu << shift);
    b_pre_out[wi] = (b_pre_out[wi] & mask) | ((val & 0xffu) << shift);
}

fn store_u8_packed_a(byte_off: u32, val: u32) {
    let wi = byte_off / 4u;
    let shift = (byte_off % 4u) * 8u;
    let mask = ~(0xffu << shift);
    a_pre_out[wi] = (a_pre_out[wi] & mask) | ((val & 0xffu) << shift);
}

// ---------------------------------------------------------------------------
// Merkle helpers
// ---------------------------------------------------------------------------

fn d_mt_ilog2_ceil(n: u32) -> i32 {
    if (n <= 1u) {
        return 0;
    }
    var lg = 31 - i32(countLeadingZeros(n));
    if ((1u << u32(lg)) < n) {
        lg = lg + 1;
    }
    return lg;
}

fn d_mt_leaf_word(word: i32, leaf: i32, smem_cols: i32) -> u32 {
    return smem_leaves[word * smem_cols + leaf];
}

fn d_mt_set_leaf_word(word: i32, leaf: i32, smem_cols: i32, v: u32) {
    smem_leaves[word * smem_cols + leaf] = v;
}

fn d_mt_parent_cv(
    key: ptr<function, array<u32, 8>>,
    left: ptr<function, array<u32, 8>>,
    right: ptr<function, array<u32, 8>>,
    as_root: bool,
    out: ptr<function, array<u32, 8>>,
) {
    var cv: array<u32, 8>;
    for (var i = 0; i < 8; i = i + 1) {
        cv[i] = (*key)[i];
    }
    var block: array<u32, 64>;
    for (var i = 0; i < 64; i = i + 1) {
        block[i] = 0u;
    }
    for (var i = 0; i < 8; i = i + 1) {
        d_b3_store32_bytes(&block, 4 * i, (*left)[i]);
        d_b3_store32_bytes(&block, 32 + 4 * i, (*right)[i]);
    }
    var fl = D_B3_KEYED | D_B3_PARENT;
    if (as_root) {
        fl = fl | D_B3_ROOT;
    }
    d_b3_compress_in_place(&cv, &block, u32(D_B3_BLOCK), 0u, 0u, fl);
    for (var i = 0; i < 8; i = i + 1) {
        (*out)[i] = cv[i];
    }
}

fn d_mt_compute_perfect(
    num_leaves: i32,
    smem_cols: i32,
    key: ptr<function, array<u32, 8>>,
    consider_root: bool,
    tid: i32,
) {
    var level_size = num_leaves;
    while (level_size > 1) {
        var left: array<u32, 8>;
        var right: array<u32, 8>;
        var parent: array<u32, 8>;
        let num_pairs = level_size >> 1;
        if (tid < num_pairs) {
            for (var i = 0; i < 8; i = i + 1) {
                left[i] = d_mt_leaf_word(i, 2 * tid, smem_cols);
                right[i] = d_mt_leaf_word(i, 2 * tid + 1, smem_cols);
            }
        }
        workgroupBarrier();
        if (tid < num_pairs) {
            let as_root = consider_root && (tid == 0) && (num_pairs == 1);
            d_mt_parent_cv(key, &left, &right, as_root, &parent);
            for (var i = 0; i < 8; i = i + 1) {
                d_mt_set_leaf_word(i, tid, smem_cols, parent[i]);
            }
        }
        workgroupBarrier();
        level_size = level_size >> 1;
    }
}

fn d_mt_compute_blake(
    num_leaves: i32,
    smem_cols: i32,
    key: ptr<function, array<u32, 8>>,
    consider_root: bool,
    tid: i32,
) {
    var offset = 0;
    var our_num_leaves = 0;
    var virtual_tid = 0;
    var largest_subtree = 0;

    for (var i = d_mt_ilog2_ceil(u32(num_leaves)); i >= 0; i = i - 1) {
        let bit_value = 1u << u32(i);
        if ((u32(num_leaves) & bit_value) != 0u) {
            if (largest_subtree == 0) {
                largest_subtree = i32(bit_value);
            }
            if (offset + i32(bit_value) > 2 * tid) {
                our_num_leaves = i32(bit_value);
                virtual_tid = tid - (offset / 2);
                break;
            }
            offset = offset + i32(bit_value);
        }
    }

    var curr_num_leaves = largest_subtree;
    while (curr_num_leaves > 1) {
        var left: array<u32, 8>;
        var right: array<u32, 8>;
        var parent: array<u32, 8>;
        let num_pairs = curr_num_leaves >> 1;
        if (curr_num_leaves <= our_num_leaves && virtual_tid < num_pairs) {
            for (var i = 0; i < 8; i = i + 1) {
                left[i] = d_mt_leaf_word(i, offset + 2 * virtual_tid, smem_cols);
                right[i] = d_mt_leaf_word(i, offset + 2 * virtual_tid + 1, smem_cols);
            }
        }
        workgroupBarrier();
        if (curr_num_leaves <= our_num_leaves && virtual_tid < num_pairs) {
            d_mt_parent_cv(key, &left, &right, false, &parent);
            for (var i = 0; i < 8; i = i + 1) {
                d_mt_set_leaf_word(i, offset + virtual_tid, smem_cols, parent[i]);
            }
        }
        workgroupBarrier();
        curr_num_leaves = curr_num_leaves >> 1;
    }

    if (tid == 0) {
        var rChaining: array<u32, 8>;
        var rChunk: array<u32, 16>;
        for (var i = 0; i < 8; i = i + 1) {
            rChaining[i] = (*key)[i];
            rChunk[i] = 0u;
            rChunk[i + 8] = 0u;
        }
        var read_offset = num_leaves;
        var written_to_chunk = 0;
        let levels = d_mt_ilog2_ceil(u32(num_leaves));
        for (var i = 0; i < levels; i = i + 1) {
            let bit_mask = 1u << u32(i);
            if ((u32(read_offset) & bit_mask) != 0u) {
                if (written_to_chunk == 0) {
                    read_offset = read_offset - i32(bit_mask);
                    for (var j = 0; j < 8; j = j + 1) {
                        rChunk[j + 8] = d_mt_leaf_word(j, read_offset, smem_cols);
                    }
                    written_to_chunk = 1;
                } else {
                    read_offset = read_offset - i32(bit_mask);
                    for (var j = 0; j < 8; j = j + 1) {
                        rChunk[j] = d_mt_leaf_word(j, read_offset, smem_cols);
                    }
                    for (var j = 0; j < 8; j = j + 1) {
                        rChaining[j] = (*key)[j];
                    }
                    let as_root = consider_root && (read_offset == 0);
                    var left: array<u32, 8>;
                    var right: array<u32, 8>;
                    for (var j = 0; j < 8; j = j + 1) {
                        left[j] = rChunk[j];
                        right[j] = rChunk[j + 8];
                    }
                    d_mt_parent_cv(key, &left, &right, as_root, &rChaining);
                    for (var j = 0; j < 8; j = j + 1) {
                        rChunk[j + 8] = rChaining[j];
                    }
                }
            }
        }
        for (var i = 0; i < 8; i = i + 1) {
            d_mt_set_leaf_word(i, 0, smem_cols, rChaining[i]);
        }
    }
    workgroupBarrier();
}

fn d_b3_mat_padded_byte(mat_off: u32, raw_len: u32, pos: i32) -> u32 {
    let gi = mat_off + u32(pos);
    if (gi < raw_len) {
        return i32_as_u8(load_s8_from_u32_word(chunk_mat[gi >> 2u], gi));
    }
    return 0u;
}

fn d_b3_keyed_chunk_cv(
    job_key: array<u32, 8>,
    chunk_idx: u32,
    mat_off: u32,
    raw_len: u32,
    chunk_len: i32,
    cv_out: ptr<function, array<u32, 8>>,
) {
    var kw: array<u32, 8>;
    for (var i = 0; i < 8; i = i + 1) {
        kw[i] = job_key[i];
    }
    var cv: array<u32, 8>;
    for (var i = 0; i < 8; i = i + 1) {
        cv[i] = kw[i];
    }
    var pos = 0;
    var blocks_compressed = 0;
    while (chunk_len - pos > D_B3_BLOCK) {
        var block: array<u32, 64>;
        for (var i = 0; i < D_B3_BLOCK; i = i + 1) {
            block[i] = d_b3_mat_padded_byte(mat_off, raw_len, pos + i);
        }
        var fl = D_B3_KEYED;
        if (blocks_compressed == 0) {
            fl = fl | D_B3_CHUNK_START;
        }
        d_b3_compress_in_place(&cv, &block, u32(D_B3_BLOCK), chunk_idx, 0u, fl);
        blocks_compressed = blocks_compressed + 1;
        pos = pos + D_B3_BLOCK;
    }
    var tail: array<u32, 64>;
    for (var i = 0; i < D_B3_BLOCK; i = i + 1) {
        tail[i] = d_b3_mat_padded_byte(mat_off, raw_len, pos + i);
    }
    var fl2 = D_B3_KEYED | D_B3_CHUNK_END;
    if (blocks_compressed == 0) {
        fl2 = fl2 | D_B3_CHUNK_START;
    }
    d_b3_compress_in_place(&cv, &tail, u32(chunk_len - pos), chunk_idx, 0u, fl2);
    for (var i = 0; i < 8; i = i + 1) {
        (*cv_out)[i] = cv[i];
    }
}

// ---------------------------------------------------------------------------
// Uniform param structs
// ---------------------------------------------------------------------------

struct PearlGenRandomParams {
    rng_seed: vec2<u32>, // u64 lo/hi
    matrix_tag: i32,
    total_elems: i32,
    wg_x: i32,
    word_begin: i32, // base packed-u32 index for chunked dispatch (TDR-safe)
    _pad1: i32,
    _pad2: i32,
}

struct PearlBuildPairsParams {
    is_b: i32,
    k: i32,
    rank: i32,
    _pad: i32,
}

struct PearlPrepackBParams {
    N: i32,
    K: i32,
    rank: i32,
    blocks_k: i32,
    macro_cols: i32,
    has_signal: i32,
    wg_x: i32,
    g_begin: i32,
}

struct PearlPrepackAParams {
    M: i32,
    K: i32,
    rank: i32,
    blocks_k: i32,
    macro_rows: i32,
    wg_x: i32,
    g_begin: i32,
    _pad1: i32,
}

struct PearlMerkleChunkParams {
    raw_len: u32,
    pad_len: u32,
    num_chunks: i32,
    _pad: i32,
}

struct PearlMerkleMtParams {
    num_leaves: i32,
    is_single_block: i32,
    _pad0: i32,
    _pad1: i32,
}

struct PearlReduceRootsParams {
    num_leaves: i32,
    _pad0: i32,
    _pad1: i32,
    _pad2: i32,
}

// ---------------------------------------------------------------------------
// Workgroup shared for merkle / prepack (declared per-entry where needed)
// WGSL requires workgroup vars at module scope — use max needed size.
// ---------------------------------------------------------------------------

var<workgroup> smem_leaves: array<u32, 2048>; // CP_MT_CV_WORDS * CP_MT_THREADS
var<workgroup> stripe: array<array<u32, 128>, 8>; // MR/NR rows × KR bytes (low 8 bits)

// ---------------------------------------------------------------------------
// Entry: pearl_gen_random_matrix
// ---------------------------------------------------------------------------

@group(0) @binding(0) var<uniform> gen_params: PearlGenRandomParams;
@group(0) @binding(1) var<storage, read_write> gen_out: array<u32>;

@compute @workgroup_size(256)
fn pearl_gen_random_matrix(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) local_invocation_index: u32,
) {
    // One WI packs 4 s8 values into one u32 (avoid RMW races on shared words).
    let wg = i32(workgroup_id.y) * gen_params.wg_x + i32(workgroup_id.x);
    let word_idx = gen_params.word_begin + wg * 256 + i32(local_invocation_index);
    let base = word_idx * 4;
    if (base >= gen_params.total_elems) {
        return;
    }
    var packed = 0u;
    for (var i = 0; i < 4; i = i + 1) {
        let idx = base + i;
        if (idx >= gen_params.total_elems) {
            break;
        }
        let tag_mul = u64_mul(u64_from_i32(gen_params.matrix_tag), vec2(0xD192ED03u, 0xD1B54A32u));
        let idx_mul = u64_mul(u64_from_i32(idx), vec2(0x7F4A7C15u, 0x9E3779B9u));
        var s = u64_xor(u64_xor(gen_params.rng_seed, tag_mul), idx_mul);
        s = cp_splitmix64(s);
        let byte = i32_as_u8(i32(s.y % 128u) - 64);
        packed = packed | (byte << u32(i * 8));
    }
    gen_out[u32(word_idx)] = packed;
}

// ---------------------------------------------------------------------------
// Entry: pearl_build_perm_pairs
// ---------------------------------------------------------------------------

@group(0) @binding(2) var<uniform> pairs_params: PearlBuildPairsParams;
@group(0) @binding(3) var<storage, read> pairs_noise_seed: array<u32, 8>;
@group(0) @binding(4) var<storage, read_write> pairs_out: array<u32>;

@compute @workgroup_size(64)
fn pearl_build_perm_pairs(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) local_invocation_index: u32,
) {
    let block_idx = i32(workgroup_id.x * 64u + local_invocation_index);
    let col0 = block_idx * CP_B3_LINES;
    if (col0 >= pairs_params.k) {
        return;
    }
    var digest: array<u32, 32>;
    d_get_random_hash(block_idx, pairs_params.is_b, pairs_noise_seed, 1, &digest);

    let rank_mask = u32(pairs_params.rank - 1);
    for (var j = 0; j < CP_B3_LINES; j = j + 1) {
        let col = col0 + j;
        if (col >= pairs_params.k) {
            break;
        }
        let w = d_b3_load32_bytes32(&digest, j * 4);
        let first = w & rank_mask;
        let second = first ^ (1u + cp_mul_hi_u32(u32(pairs_params.rank - 1), w));
        pairs_out[u32(col) * 2u] = first;
        pairs_out[u32(col) * 2u + 1u] = second;
    }
}

// ---------------------------------------------------------------------------
// Entry: pearl_fused_prepack_b
// ---------------------------------------------------------------------------

@group(0) @binding(5) var<uniform> pre_b_params: PearlPrepackBParams;
@group(0) @binding(6) var<storage, read_write> b_pre_out: array<u32>;
@group(0) @binding(7) var<storage, read> b_noise_seed: array<u32, 8>;
@group(0) @binding(8) var<storage, read> b_pairs: array<u32>;
@group(0) @binding(9) var<storage, read> b_signal_colmajor: array<u32>;

@compute @workgroup_size(8)
fn pearl_fused_prepack_b(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) local_invocation_index: u32,
) {
    // 2D grid + g_begin: chunked host dispatch stays under Windows TDR on iGPUs.
    let g = pre_b_params.g_begin + i32(workgroup_id.y) * pre_b_params.wg_x + i32(workgroup_id.x);
    let tc = g % MICRO_N;
    let kb = (g / MICRO_N) % pre_b_params.blocks_k;
    let jm = g / (MICRO_N * pre_b_params.blocks_k);
    let col = i32(local_invocation_index);
    if (jm >= pre_b_params.macro_cols || col >= NR) {
        return;
    }

    let k0 = kb * KR;
    let ncol = (jm * MICRO_N + tc) * NR + col;
    var el: array<u32, 128>;
    ocl_generate_uniform_row(ncol, pre_b_params.rank, b_noise_seed, 1, &el);

    for (var t = 0; t < KR; t = t + 1) {
        let l = k0 + t;
        let pos = sign_extend_i8(el[b_pairs[u32(l) * 2u]]);
        let neg = sign_extend_i8(el[b_pairs[u32(l) * 2u + 1u]]);
        var sig = 0;
        if (pre_b_params.has_signal != 0) {
            let sidx = u32(ncol) * u32(pre_b_params.K) + u32(l);
            sig = load_s8_from_u32_word(b_signal_colmajor[sidx >> 2u], sidx);
        }
        stripe[col][t] = i32_as_u8(sig + (pos - neg));
    }
    workgroupBarrier();

    if (col == 0) {
        let block_base = (u32(jm) * u32(pre_b_params.blocks_k) + u32(kb)) * u32(MACRO_KB_BLOCK_B);
        for (var kg = 0; kg < K_GROUPS; kg = kg + 1) {
            let dst = block_base + u32(kg) * u32(MACRO_KG_STRIP_B) + u32(tc) * u32(KG_SLICE_B);
            for (var j = 0; j < NR; j = j + 1) {
                for (var ko = 0; ko < 4; ko = ko + 1) {
                    store_u8_packed_b(dst + u32(j) * 4u + u32(ko), stripe[j][kg * 4 + ko]);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Entry: pearl_fused_prepack_a
// ---------------------------------------------------------------------------

@group(0) @binding(10) var<uniform> pre_a_params: PearlPrepackAParams;
@group(0) @binding(11) var<storage, read_write> a_pre_out: array<u32>;
@group(0) @binding(12) var<storage, read> a_noise_seed: array<u32, 8>;
@group(0) @binding(13) var<storage, read> a_pairs: array<u32>;
@group(0) @binding(14) var<storage, read> a_signal: array<u32>;

@compute @workgroup_size(8)
fn pearl_fused_prepack_a(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) local_invocation_index: u32,
) {
    let g = pre_a_params.g_begin + i32(workgroup_id.y) * pre_a_params.wg_x + i32(workgroup_id.x);
    let tr = g % MICRO_M;
    let kb = (g / MICRO_M) % pre_a_params.blocks_k;
    let im = g / (MICRO_M * pre_a_params.blocks_k);
    let row = i32(local_invocation_index);
    if (im >= pre_a_params.macro_rows || row >= MR) {
        return;
    }

    let k0 = kb * KR;
    let nrow = (im * MICRO_M + tr) * MR + row;
    var el: array<u32, 128>;
    ocl_generate_uniform_row(nrow, pre_a_params.rank, a_noise_seed, 0, &el);

    for (var t = 0; t < KR; t = t + 1) {
        let l = k0 + t;
        let pos = sign_extend_i8(el[a_pairs[u32(l) * 2u]]);
        let neg = sign_extend_i8(el[a_pairs[u32(l) * 2u + 1u]]);
        let sidx = u32(nrow) * u32(pre_a_params.K) + u32(l);
        let sig = load_s8_from_u32_word(a_signal[sidx >> 2u], sidx);
        stripe[row][t] = i32_as_u8(sig + (pos - neg));
    }
    workgroupBarrier();

    if (row == 0) {
        let block_base = (u32(im) * u32(pre_a_params.blocks_k) + u32(kb)) * u32(MACRO_KB_BLOCK_A);
        for (var kg = 0; kg < K_GROUPS; kg = kg + 1) {
            let dst = block_base + u32(kg) * u32(MACRO_KG_STRIP_A) + u32(tr) * u32(KG_BYTES_A);
            for (var r = 0; r < MR; r = r + 1) {
                for (var ko = 0; ko < 4; ko = ko + 1) {
                    store_u8_packed_a(dst + u32(r) * 4u + u32(ko), stripe[r][kg * 4 + ko]);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Entry: pearl_keyed_chunk_roots
// ---------------------------------------------------------------------------

@group(0) @binding(15) var<uniform> chunk_params: PearlMerkleChunkParams;
@group(0) @binding(16) var<storage, read> chunk_mat: array<u32>;
@group(0) @binding(17) var<storage, read> chunk_job_key: array<u32, 8>;
@group(0) @binding(18) var<storage, read_write> chunk_roots_out: array<u32>;

@compute @workgroup_size(256)
fn pearl_keyed_chunk_roots(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) local_invocation_index: u32,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let tid = i32(local_invocation_index);
    let bid = i32(workgroup_id.x);
    let num_grid_blocks = i32(num_workgroups.x);
    let is_last_block = (bid == num_grid_blocks - 1);
    let global_chunk = bid * CP_MT_THREADS + tid;

    if (global_chunk < chunk_params.num_chunks) {
        let off = u32(global_chunk) * u32(D_B3_CHUNK);
        var len = D_B3_CHUNK;
        if (off + u32(len) > chunk_params.pad_len) {
            len = i32(chunk_params.pad_len - off);
        }
        var cv: array<u32, 8>;
        d_b3_keyed_chunk_cv(chunk_job_key, u32(global_chunk), off, chunk_params.raw_len, len, &cv);
        for (var i = 0; i < 8; i = i + 1) {
            d_mt_set_leaf_word(i, tid, CP_MT_THREADS, cv[i]);
        }
    }
    workgroupBarrier();

    var num_leaves = CP_MT_THREADS;
    if (is_last_block) {
        let chunks_in_block = chunk_params.num_chunks % CP_MT_THREADS;
        num_leaves = select(chunks_in_block, CP_MT_THREADS, chunks_in_block == 0);
        let remainder_bytes = i32(chunk_params.pad_len % u32(D_B3_CHUNK));
        let last_chunk_too_small = (remainder_bytes > 0) && (remainder_bytes < D_B3_BLOCK);
        if (last_chunk_too_small) {
            num_leaves = select(0, num_leaves - 1, num_leaves > 0);
        }
    }

    if (num_leaves <= 0) {
        return;
    }

    var key: array<u32, 8>;
    for (var i = 0; i < 8; i = i + 1) {
        key[i] = chunk_job_key[i];
    }

    let power_of_two = (u32(num_leaves) & u32(num_leaves - 1)) == 0u;
    if (!is_last_block || power_of_two) {
        d_mt_compute_perfect(num_leaves, CP_MT_THREADS, &key, false, tid);
    } else {
        d_mt_compute_blake(num_leaves, CP_MT_THREADS, &key, false, tid);
    }

    if (tid < CP_MT_CV_WORDS) {
        let w = d_mt_leaf_word(tid, 0, CP_MT_THREADS);
        chunk_roots_out[u32(bid) * u32(D_B3_OUT / 4) + u32(tid)] = w;
    }
}

// ---------------------------------------------------------------------------
// Entry: pearl_compute_blake_mt
// ---------------------------------------------------------------------------

@group(0) @binding(19) var<uniform> mt_params: PearlMerkleMtParams;
@group(0) @binding(20) var<storage, read> mt_job_key: array<u32, 8>;
@group(0) @binding(21) var<storage, read_write> mt_roots: array<u32>;

@compute @workgroup_size(256)
fn pearl_compute_blake_mt(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) local_invocation_index: u32,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let tid = i32(local_invocation_index);
    let bid = i32(workgroup_id.x);
    let n_blocks = i32(num_workgroups.x);
    let remainder = mt_params.num_leaves % CP_MT_THREADS;
    let is_remainder_block = (bid == n_blocks - 1) && (remainder > 0);
    let block_leaves = select(CP_MT_THREADS, remainder, is_remainder_block);
    let offset = bid * CP_MT_THREADS;

    var key: array<u32, 8>;
    for (var i = 0; i < 8; i = i + 1) {
        key[i] = mt_job_key[i];
    }

    if (tid < block_leaves) {
        for (var i = 0; i < CP_MT_CV_WORDS; i = i + 1) {
            let word = mt_roots[u32(offset + tid) * u32(D_B3_OUT / 4) + u32(i)];
            d_mt_set_leaf_word(i, tid, CP_MT_THREADS, word);
        }
    }
    workgroupBarrier();

    let use_blake_mt = is_remainder_block && ((u32(block_leaves) & u32(block_leaves - 1)) != 0u);
    let consider_root = mt_params.is_single_block != 0;

    if (use_blake_mt) {
        d_mt_compute_blake(block_leaves, CP_MT_THREADS, &key, consider_root, tid);
    } else {
        d_mt_compute_perfect(block_leaves, CP_MT_THREADS, &key, consider_root, tid);
    }
    workgroupBarrier();

    if (tid < CP_MT_CV_WORDS) {
        let base = select(u32(bid) * u32(D_B3_OUT / 4), 0u, mt_params.is_single_block != 0);
        mt_roots[base + u32(tid)] = d_mt_leaf_word(tid, 0, CP_MT_THREADS);
    }
}

// ---------------------------------------------------------------------------
// Entry: pearl_reduce_roots
// ---------------------------------------------------------------------------

@group(0) @binding(22) var<uniform> reduce_params: PearlReduceRootsParams;
@group(0) @binding(23) var<storage, read> reduce_job_key: array<u32, 8>;
@group(0) @binding(24) var<storage, read_write> reduce_roots: array<u32>;

@compute @workgroup_size(256)
fn pearl_reduce_roots(
    @builtin(local_invocation_index) local_invocation_index: u32,
) {
    let tid = i32(local_invocation_index);
    var key: array<u32, 8>;
    for (var i = 0; i < 8; i = i + 1) {
        key[i] = reduce_job_key[i];
    }

    if (tid < reduce_params.num_leaves) {
        for (var i = 0; i < CP_MT_CV_WORDS; i = i + 1) {
            let word = reduce_roots[u32(tid) * u32(D_B3_OUT / 4) + u32(i)];
            d_mt_set_leaf_word(i, tid, CP_MT_THREADS, word);
        }
    }
    workgroupBarrier();

    if (countOneBits(u32(reduce_params.num_leaves)) == 1u) {
        d_mt_compute_perfect(reduce_params.num_leaves, CP_MT_THREADS, &key, true, tid);
    } else {
        d_mt_compute_blake(reduce_params.num_leaves, CP_MT_THREADS, &key, true, tid);
    }
    workgroupBarrier();

    if (tid < CP_MT_CV_WORDS) {
        reduce_roots[u32(tid)] = d_mt_leaf_word(tid, 0, CP_MT_THREADS);
    }
}
