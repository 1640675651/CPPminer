// pearl_gemm_xor.wgsl - fused GEMM+XOR+BLAKE3+jackpot mining kernel
// Port of fuse_jackpot=1 / CASE32_COALESCE / packed-dot path from case33_gemm_xor.cl
//
// Binding layout (group 0):
//   @binding(0) a_pre        : storage read  array<vec4<u32>>   // packed int8 panels (LE u32 words)
//   @binding(1) b_pre        : storage read  array<vec4<u32>>
//   @binding(2) params       : uniform       PearlScanParams
//   @binding(3) a_key8       : storage read  array<u32, 8>
//   @binding(4) bound        : storage read  array<u32, 8>
//   @binding(5) found_flag   : storage rw    atomic<i32>
//   @binding(6) out_t_rows   : storage rw    array<i32, 1>
//   @binding(7) out_t_cols   : storage rw    array<i32, 1>
//
// Dispatch: workgroups = macro tiles in batch; one WI per hash tile in the macro
//   (WG_SIZE = MICRO_M * HASH_MICRO_N, column-major like !CASE32_WI_ROWMAJOR).
//
// engine.rs replaces the marker below with MR, NR, HASH_NR, MACRO_M, MACRO_N and the generated
// tile code (struct Acc, acc_kgroup, acc_xor, kgroup_global, kgroup_lds) for the register tile.
// Register tiles 4x4 (hash 4x8: two 4x4 halves per WI), 4x8, 8x8, 8x16; macro 64 or 128.
// Packed layout (both tiles): byte ((im*blocks_k + kb)*KGROUPS + kg)*MACRO_M*4 + row*4 + k%4.

requires packed_4x8_integer_dot_product;

// @TILE_CONFIG@

const KR: i32 = 128;
const RANK: i32 = 4;
const R_RANK: i32 = 128;
const KGROUPS: i32 = KR / RANK;
const MICRO_M: i32 = MACRO_M / MR;
const HASH_MICRO_N: i32 = MACRO_N / HASH_NR;
const HASH_REG_TILES_N: i32 = HASH_NR / NR;
const WG_SIZE: u32 = u32(MICRO_M * HASH_MICRO_N);
const KG_VEC4_A: u32 = u32(MR / 4);  // one k-group of one register tile row block
const KG_VEC4_B: u32 = u32(NR / 4);
const STRIP_VEC4_A: u32 = u32(MACRO_M / 4); // one k-group of the whole macro (MACRO_M*RANK B)
const STRIP_VEC4_B: u32 = u32(MACRO_N / 4);
const KB_VEC4_A: u32 = u32(KGROUPS) * STRIP_VEC4_A; // one k-block of the macro
const KB_VEC4_B: u32 = u32(KGROUPS) * STRIP_VEC4_B;
const PP_JACKPOT_WORDS: i32 = 16;
const PP_LROT: i32 = 13;
const PP_MAX_MILESTONES: i32 = 32;

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

@group(0) @binding(0) var<storage, read> a_pre: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> b_pre: array<vec4<u32>>;
@group(0) @binding(2) var<uniform> params: PearlScanParams;
@group(0) @binding(3) var<storage, read> a_key8: array<u32, 8>;
@group(0) @binding(4) var<storage, read> bound: array<u32, 8>;
@group(0) @binding(5) var<storage, read_write> found_flag: atomic<i32>;
@group(0) @binding(6) var<storage, read_write> out_t_rows: array<i32, 1>;
@group(0) @binding(7) var<storage, read_write> out_t_cols: array<i32, 1>;

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

// Milestone XOR fold of one register tile (x = acc_xor) into msg[16] (fuse_jackpot online path).
// XOR is linear, so the two 4x4 halves of a 4x8 hash tile can fold in separately.
fn milestone_fold(x: u32, msg: ptr<function, array<u32, 16>>, ms: i32) {
    if (ms < PP_MAX_MILESTONES) {
        let tid = ms % PP_JACKPOT_WORDS;
        var contribution = x;
        if (ms + PP_JACKPOT_WORDS < params.num_milestones) {
            contribution = pp_rotl32(x, u32(PP_LROT));
        }
        (*msg)[tid] = (*msg)[tid] ^ contribution;
    }
}

fn finish_tile(msg: ptr<function, array<u32, 16>>, im: i32, jm: i32, tr: i32, hash_tc: i32) {
    var digest: array<u32, 8>;
    b3_compress64(msg, &digest);
    if (!digest_beats_target(&digest)) {
        return;
    }
    let exchanged = atomicCompareExchangeWeak(&found_flag, 0, 1);
    if (exchanged.old_value != 0) {
        return;
    }
    out_t_rows[0] = im * MACRO_M + tr * MR;
    out_t_cols[0] = jm * MACRO_N + hash_tc * HASH_NR;
}

// Direct global loads: each WI reads its own A/B slices per k-group.
@compute @workgroup_size(WG_SIZE)
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

    let tr = params.micro_m_begin + lid % MICRO_M;
    let hash_tc = lid / MICRO_M;
    let a_off = u32(tr) * KG_VEC4_A;

    var msg: array<u32, 16>;
    var acc: Acc;
    for (var half = 0; half < HASH_REG_TILES_N; half = half + 1) {
        let b_off = u32(hash_tc * HASH_REG_TILES_N + half) * KG_VEC4_B;
        acc = Acc(); // naga hoists loop-local vars to function entry: re-zero explicitly
        for (var kb = 0; kb < params.blocks_k; kb = kb + 1) {
            let a_kb = (u32(im) * u32(params.blocks_k) + u32(kb)) * KB_VEC4_A + a_off;
            let b_kb = (u32(jm) * u32(params.blocks_k) + u32(kb)) * KB_VEC4_B + b_off;
            for (var kg = 0u; kg < u32(KGROUPS); kg = kg + 1u) {
                kgroup_global(&acc, a_kb + kg * STRIP_VEC4_A, b_kb + kg * STRIP_VEC4_B);
            }
            milestone_fold(acc_xor(&acc), &msg, kb);
        }
    }
    finish_tile(&msg, im, jm, tr, hash_tc);
}

// LDS staging (--wgpu-lds, like the OpenCL CASE32_USE_LDS path): per k-block, all WIs copy the
// whole macro A and B k-block (MACRO*128 B each, contiguous in a_pre/b_pre) into workgroup
// memory, barrier, compute all KGROUPS k-groups from it, barrier. Needs MACRO*256 B workgroup
// storage (32 KiB at macro 128). 8x8/128: GTX 1070 ~4.2 TMAC/s vs ~3.4 direct (8/16 k-group
// panels and double buffering were slower); UHD 770 ~380 GMAC/s vs ~455 direct.
var<workgroup> lds_a: array<vec4<u32>, KB_VEC4_A>;
var<workgroup> lds_b: array<vec4<u32>, KB_VEC4_B>;

@compute @workgroup_size(WG_SIZE)
fn pearl_macro_gemm_xor_lds(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) local_invocation_index: u32,
) {
    let local_wg = i32(workgroup_id.y) * params.wg_x + i32(workgroup_id.x);
    if (local_wg >= params.batch_count) {
        return;
    }
    let lid = local_invocation_index;
    if (lid == 0u) {
        lds_a[0] = vec4<u32>(bitcast<u32>(atomicLoad(&found_flag)));
    }
    if (workgroupUniformLoad(&lds_a[0]).x != 0u) {
        return;
    }
    workgroupBarrier(); // all WIs have read lds_a[0] before the first copy overwrites it

    let mb = params.mb_begin + local_wg;
    let jm = mb / params.macro_rows;
    let im = mb % params.macro_rows;
    let tr = params.micro_m_begin + i32(lid) % MICRO_M;
    let hash_tc = i32(lid) / MICRO_M;
    let a_off = u32(tr) * KG_VEC4_A;

    var msg: array<u32, 16>;
    var acc: Acc;
    for (var half = 0; half < HASH_REG_TILES_N; half = half + 1) {
        let b_off = u32(hash_tc * HASH_REG_TILES_N + half) * KG_VEC4_B;
        acc = Acc(); // naga hoists loop-local vars to function entry: re-zero explicitly
        for (var kb = 0; kb < params.blocks_k; kb = kb + 1) {
            let a_src = (u32(im) * u32(params.blocks_k) + u32(kb)) * KB_VEC4_A;
            let b_src = (u32(jm) * u32(params.blocks_k) + u32(kb)) * KB_VEC4_B;
            for (var v = lid; v < KB_VEC4_A; v = v + WG_SIZE) {
                lds_a[v] = a_pre[a_src + v];
            }
            for (var v = lid; v < KB_VEC4_B; v = v + WG_SIZE) {
                lds_b[v] = b_pre[b_src + v];
            }
            workgroupBarrier();

            for (var kg = 0u; kg < u32(KGROUPS); kg = kg + 1u) {
                kgroup_lds(&acc, kg * STRIP_VEC4_A + a_off, kg * STRIP_VEC4_B + b_off);
            }
            milestone_fold(acc_xor(&acc), &msg, kb);
            workgroupBarrier();
        }
    }
    finish_tile(&msg, im, jm, tr, hash_tc);
}
