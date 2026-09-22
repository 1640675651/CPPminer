// pearl_gemm_xor.wgsl - fused GEMM+XOR+BLAKE3+jackpot mining kernel
// Port of fuse_jackpot=1 / CASE32_COALESCE / packed-dot path from case33_gemm_xor.cl
//
// Binding layout (group 0):
//   @binding(0) a_pre        : storage read  array<u32>   // packed int8 panels (LE u32 words)
//   @binding(1) b_pre        : storage read  array<u32>
//   @binding(2) params       : uniform       PearlScanParams
//   @binding(3) a_key8       : storage read  array<u32, 8>
//   @binding(4) bound        : storage read  array<u32, 8>
//   @binding(5) found_flag   : storage rw    atomic<i32>
//   @binding(6) out_t_rows   : storage rw    array<i32, 1>
//   @binding(7) out_t_cols   : storage rw    array<i32, 1>
//
// Dispatch: workgroups = macro tiles in batch; workgroup_size = 256
//   (one WI per hash tile in macro; micro_m_count=16, HASH_MICRO_N=16)

requires packed_4x8_integer_dot_product;

const MR: i32 = 8;
const NR: i32 = 8;
const HASH_NR: i32 = 8;
const MACRO_M: i32 = 128;
const MACRO_N: i32 = 128;
const KR: i32 = 128;
const RANK: i32 = 4;
const R_RANK: i32 = 128;
const MICRO_M: i32 = 16;
const MICRO_N: i32 = 16;
const HASH_MICRO_N: i32 = 16;
const KGROUPS: i32 = 32;
const KG_BYTES_A: i32 = 32;
const KG_SLICE_B: i32 = 32;
const MACRO_KG_STRIP_A: i32 = 512;
const MACRO_KG_STRIP_B: i32 = 512;
const MACRO_KB_BLOCK_A: i32 = 16384;
const MACRO_KB_BLOCK_B: i32 = 16384;
const PP_JACKPOT_WORDS: i32 = 16;
const PP_LROT: i32 = 13;
const PP_MAX_MILESTONES: i32 = 32;
const HASH_REG_TILES_N: i32 = 1;

struct PearlScanParams {
    N: i32,
    blocks_k: i32,
    num_milestones: i32,
    tile_count: i32,
    macro_rows: i32,
    macro_cols: i32,
    mb_begin: i32,
    micro_m_begin: i32,
    micro_m_count: i32,
    wg_x: i32,
    batch_count: i32,
    _pad2: i32,
}

@group(0) @binding(0) var<storage, read> a_pre: array<u32>;
@group(0) @binding(1) var<storage, read> b_pre: array<u32>;
@group(0) @binding(2) var<uniform> params: PearlScanParams;
@group(0) @binding(3) var<storage, read> a_key8: array<u32, 8>;
@group(0) @binding(4) var<storage, read> bound: array<u32, 8>;
@group(0) @binding(5) var<storage, read_write> found_flag: atomic<i32>;
@group(0) @binding(6) var<storage, read_write> out_t_rows: array<i32, 1>;
@group(0) @binding(7) var<storage, read_write> out_t_cols: array<i32, 1>;

fn load_u32_a(byte_off: u32) -> u32 {
    return a_pre[byte_off / 4u];
}

fn load_u32_b(byte_off: u32) -> u32 {
    return b_pre[byte_off / 4u];
}

fn pp_rotl32(x: u32, s: u32) -> u32 {
    return (x << s) | (x >> (32u - s));
}

fn b3_rotr32(x: u32, n: u32) -> u32 {
    return (x >> n) | (x << (32u - n));
}

fn b3_g(v: ptr<function, array<u32, 16>>, a: i32, b: i32, c: i32, d: i32, x: u32, y: u32) {
    (*v)[a] = (*v)[a] + (*v)[b] + x;
    (*v)[d] = b3_rotr32((*v)[d] ^ (*v)[a], 16u);
    (*v)[c] = (*v)[c] + (*v)[d];
    (*v)[b] = b3_rotr32((*v)[b] ^ (*v)[c], 12u);
    (*v)[a] = (*v)[a] + (*v)[b] + y;
    (*v)[d] = b3_rotr32((*v)[d] ^ (*v)[a], 8u);
    (*v)[c] = (*v)[c] + (*v)[d];
    (*v)[b] = b3_rotr32((*v)[b] ^ (*v)[c], 7u);
}

// Matches OpenCL b3_compress64 (keyed root compress of 64-byte msg).
fn b3_compress64(msg16: ptr<function, array<u32, 16>>, out8: ptr<function, array<u32, 8>>) {
    let kIV0 = 0x6A09E667u;
    let kIV1 = 0xBB67AE85u;
    let kIV2 = 0x3C6EF372u;
    let kIV3 = 0xA54FF53Au;
    let kIV4 = 0x510E527Fu;
    let kIV5 = 0x9B05688Cu;
    let kIV6 = 0x1F83D9ABu;
    let kIV7 = 0x5BE0CD19u;

    var v: array<u32, 16>;
    v[0] = a_key8[0];
    v[1] = a_key8[1];
    v[2] = a_key8[2];
    v[3] = a_key8[3];
    v[4] = a_key8[4];
    v[5] = a_key8[5];
    v[6] = a_key8[6];
    v[7] = a_key8[7];
    v[8] = kIV0;
    v[9] = kIV1;
    v[10] = kIV2;
    v[11] = kIV3;
    v[12] = 0u;
    v[13] = 0u;
    v[14] = 64u;
    v[15] = 0x1Bu;

    var m: array<u32, 16>;
    for (var i = 0; i < 16; i = i + 1) {
        m[i] = (*msg16)[i];
    }

    // BLAKE3 message permutation (same as OpenCL kPerm)
    let kPerm = array<u32, 16>(
        2u, 6u, 3u, 10u, 7u, 0u, 4u, 13u, 1u, 11u, 12u, 5u, 9u, 14u, 15u, 8u
    );

    for (var round = 0; round < 7; round = round + 1) {
        b3_g(&v, 0, 4, 8, 12, m[0], m[1]);
        b3_g(&v, 1, 5, 9, 13, m[2], m[3]);
        b3_g(&v, 2, 6, 10, 14, m[4], m[5]);
        b3_g(&v, 3, 7, 11, 15, m[6], m[7]);
        b3_g(&v, 0, 5, 10, 15, m[8], m[9]);
        b3_g(&v, 1, 6, 11, 12, m[10], m[11]);
        b3_g(&v, 2, 7, 8, 13, m[12], m[13]);
        b3_g(&v, 3, 4, 9, 14, m[14], m[15]);
        if (round < 6) {
            var t: array<u32, 16>;
            for (var i = 0; i < 16; i = i + 1) {
                t[i] = m[kPerm[i]];
            }
            for (var i = 0; i < 16; i = i + 1) {
                m[i] = t[i];
            }
        }
    }
    for (var i = 0; i < 8; i = i + 1) {
        (*out8)[i] = v[i] ^ v[i + 8];
    }
}

fn digest_beats_target(digest: ptr<function, array<u32, 8>>) -> bool {
    for (var w = 7; w >= 0; w = w - 1) {
        if ((*digest)[w] < bound[w]) {
            return true;
        }
        if ((*digest)[w] > bound[w]) {
            return false;
        }
    }
    return true;
}

@compute @workgroup_size(256)
fn pearl_macro_gemm_xor(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) local_invocation_index: u32,
) {
    if (atomicLoad(&found_flag) != 0) {
        return;
    }

    let lid = i32(local_invocation_index);
    let local_wg = i32(workgroup_id.y) * params.wg_x + i32(workgroup_id.x);
    if (local_wg >= params.batch_count) {
        return;
    }
    let mb = params.mb_begin + local_wg;
    let jm = mb / params.macro_rows;
    let im = mb % params.macro_rows;

    // Column-major WI layout (!CASE32_WI_ROWMAJOR)
    let tr_in_slice = lid % params.micro_m_count;
    let hash_tc = lid / params.micro_m_count;
    if (tr_in_slice >= params.micro_m_count) {
        return;
    }

    let tr = params.micro_m_begin + tr_in_slice;

    var msg: array<u32, 16>;
    for (var i = 0; i < PP_JACKPOT_WORDS; i = i + 1) {
        msg[i] = 0u;
    }

    // HASH_REG_TILES_N == 1
    let tc = hash_tc;

    var acc: array<i32, 64>; // NR * MR = 64
    for (var i = 0; i < NR * MR; i = i + 1) {
        acc[i] = 0;
    }

    var ms = 0;
    for (var kb = 0; kb < params.blocks_k; kb = kb + 1) {
        let a_kb_base = u32(im) * u32(params.blocks_k) * u32(MACRO_KB_BLOCK_A)
            + u32(kb) * u32(MACRO_KB_BLOCK_A);
        let b_kb_base = u32(jm) * u32(params.blocks_k) * u32(MACRO_KB_BLOCK_B)
            + u32(kb) * u32(MACRO_KB_BLOCK_B);

        for (var kg = 0; kg < KGROUPS; kg = kg + 1) {
            let a_kg = a_kb_base + u32(kg) * u32(MACRO_KG_STRIP_A) + u32(tr) * u32(KG_BYTES_A);
            let b_kg = b_kb_base + u32(kg) * u32(MACRO_KG_STRIP_B) + u32(tc) * u32(KG_SLICE_B);

            var a_pack: array<u32, 8>;
            var b_pack: array<u32, 8>;
            for (var i = 0; i < MR; i = i + 1) {
                a_pack[i] = load_u32_a(a_kg + u32(i) * u32(RANK));
            }
            for (var j = 0; j < NR; j = j + 1) {
                b_pack[j] = load_u32_b(b_kg + u32(j) * u32(RANK));
            }

            // Packed int8 dots: acc[j*MR+i] += dot4I8Packed(a_pack[i], b_pack[j])
            for (var j = 0; j < NR; j = j + 1) {
                let base = j * MR;
                let b0 = b_pack[j];
                for (var i = 0; i < MR; i = i + 1) {
                    acc[base + i] = acc[base + i] + dot4I8Packed(a_pack[i], b0);
                }
            }
        }

        // Milestone XOR fold into msg[16] (fuse_jackpot online path)
        var x = 0u;
        for (var i = 0; i < NR * MR; i = i + 1) {
            x = x ^ bitcast<u32>(acc[i]);
        }
        if (ms < PP_MAX_MILESTONES) {
            let tid = ms % PP_JACKPOT_WORDS;
            var contribution = x;
            if (ms + PP_JACKPOT_WORDS < params.num_milestones) {
                contribution = pp_rotl32(x, u32(PP_LROT));
            }
            msg[tid] = msg[tid] ^ contribution;
        }
        ms = ms + 1;
    }

    var digest: array<u32, 8>;
    b3_compress64(&msg, &digest);
    if (!digest_beats_target(&digest)) {
        return;
    }

    let exchanged = atomicCompareExchangeWeak(&found_flag, 0, 1);
    if (exchanged.old_value != 0) {
        return;
    }

    let t_rows = im * MACRO_M + tr * MR;
    let t_cols = jm * MACRO_N + hash_tc * HASH_NR;
    out_t_rows[0] = t_rows;
    out_t_cols[0] = t_cols;
}
