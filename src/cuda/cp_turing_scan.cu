/* Turing (sm_75) fused scan kernel, see include/cp_turing_scan.h.
 *
 * CTA 256x128 (two virtual 128x128 CTAs of the CUTLASS kernel), eight warps of
 * 64x64, k-tile 32 double-buffered in shared memory (16-byte halves XOR-swizzled
 * so ldmatrix is conflict-free), operands fetched one k-tile ahead through
 * registers. A k-tile of a CTA is one contiguous block of the packed layout,
 * which is what keeps the L2->SM feed efficient on Turing.
 *
 * Milestone fold of milestone s is done at the start of the first k-tile of
 * milestone s + 1, after the barrier, so its XOR/shuffle work shares a
 * scheduling region with that tile's IMMAs. The transcript (2 tiles x 16 words
 * per thread) lives in shared memory. */
#include <cstdio>

#include <cuda_runtime.h>

#include "cp_config.h"
#include "cp_cutlass_jackpot.cuh"
#include "cp_gpu.cuh"
#include "cp_turing_layout.cuh"
#include "cp_turing_scan.h"
#include "hash_tile_policy.h"

namespace {

constexpr int kBM = 256, kBN = 128, kBK = CP_TURING_BK, kThreads = 256;
constexpr int kTiles = K_DIM / kBK;     /* 128 k-tiles */
constexpr int kMilestones = K_DIM / R_RANK;
constexpr int kTilesPerMilestone = R_RANK / kBK;
constexpr int kStageA = kBM * kBK;      /* 8 KB */
constexpr int kStageB = kBN * kBK;      /* 4 KB */
constexpr int kWords = CP_CUTLASS_JACKPOT_WORDS;
constexpr int kSmem = 2 * (kStageA + kStageB) + 2 * kWords * kThreads * 4;  /* 56 KB */

static_assert(kTilesPerMilestone == 4, "loop is unrolled for 4 k-tiles per milestone");
static_assert(kMilestones % kWords == 0, "transcript folds whole 16-word cycles");

struct TuringParams {
    const int8_t* A;
    const int8_t* B;
    int row_block0;      /* first 256-row block of the panel */
    int col_block0;      /* first 128-col block of the panel */
    int row_blocks;
    int col_blocks;
    int group;
    uint32_t bound[8];
    const uint32_t* a_key8;
    int* found;
    int* out_t_rows;
    int* out_t_cols;
    uint32_t* dump_words;
};

__device__ __forceinline__ uint32_t smem_u32(const void* p)
{
    return (uint32_t)__cvta_generic_to_shared(p);
}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 750
__device__ __forceinline__ void ldsm_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2,
                                        uint32_t& r3)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}

__device__ __forceinline__ void imma(int& c0, int& c1, uint32_t a, uint32_t b)
{
    asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};\n"
        : "+r"(c0), "+r"(c1)
        : "r"(a), "r"(b));
}
#endif

__device__ __forceinline__ int4 ldg_cg(const int8_t* p)
{
    int4 v;
    asm volatile("ld.global.cg.L2::128B.v4.s32 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

/* Milestone fold shared by both kernels: reduce-scatter the lane's 16 partial
 * hash-tile XORs (HashTileTensorOp's three steps) so the lane owns the full
 * word of its two hash tiles, then fold them into transcript word idx (rotl
 * 13, as cp_cutlass_jackpot_fold_step). tr = [2 tiles x 16 words][kThreads]
 * in shared memory. */
__device__ __forceinline__ void fold_milestone(uint32_t* tr, int tid, int lane, int idx,
                                               uint32_t (&part)[16])
{
    cp_cutlass::reduce_scatter_step<16, 8>(part, lane);
    cp_cutlass::reduce_scatter_step<8, 4>(part, lane);
    cp_cutlass::reduce_scatter_step<4, 1>(part, lane);
    uint32_t* t0 = tr + idx * kThreads + tid;
    uint32_t* t1 = tr + (kWords + idx) * kThreads + tid;
    *t0 = cp_cutlass_rotl32(*t0, CP_CUTLASS_JACKPOT_LROT) ^ part[0];
    *t1 = cp_cutlass_rotl32(*t1, CP_CUTLASS_JACKPOT_LROT) ^ part[1];
}

/* Tail shared by both kernels: the two hash tiles this lane owns after the
 * reduce-scatter (HashTileTensorOp::virtual_thread) as their virtual 128x128
 * CTA and virtual SIMT thread, so the dump index, the BLAKE3/target check and
 * the hit coordinates are exactly the CUTLASS kernel's (cp_cutlass_jackpot_try
 * -> MmaLaneTile128x128). */
__device__ __forceinline__ void emit_hash_tiles(const TuringParams& p, const uint32_t* tr,
                                                int tid, int lane, int wm, int wn, int rb,
                                                int cb)
{
    const int b0 = lane & 1, b1 = (lane >> 1) & 1, b2 = (lane >> 2) & 1;
    const int b3 = (lane >> 3) & 1, b4 = (lane >> 4) & 1;
    const int vr = rb * 2 + (wm >> 1), vc = cb; /* global virtual 128x128 CTA */
#pragma unroll
    for (int t = 0; t < 2; ++t) {
        const int j = (b3 * 4 + b2 * 2 + b0) * 2 + t;
        const int h = j >> 3, aa = (j >> 2) & 1, bb = j & 3;
        const int gm = aa * 2 + b4, gn = bb * 2 + b1;
        const int simt_warp = ((wm * 2 + h) & 3) + 4 * wn;
        const int simt_lane = (gn >> 1) * 8 + gm * 2 + (gn & 1);
        const int vt = simt_warp * 32 + simt_lane;
        uint32_t msg[kWords];
#pragma unroll
        for (int w = 0; w < kWords; ++w) msg[w] = tr[(t * kWords + w) * kThreads + tid];
        if (p.dump_words) {
            const int cta = (vr - 2 * p.row_block0) * p.col_blocks + (vc - p.col_block0);
            const size_t tile = (size_t)cta * CutlassJackpotTile::kThreadsPerCta + vt;
#pragma unroll
            for (int w = 0; w < kWords; ++w) p.dump_words[tile * kWords + w] = msg[w];
        }
        if (p.found && p.a_key8)
            cp_cutlass_jackpot_try(msg, p.a_key8, p.bound, vr, vc, vt, p.found, p.out_t_rows,
                                   p.out_t_cols);
    }
}

/* XOR of this lane's 8 cells of each of its 16 partial hash tiles,
 * p = h*8 + a*4 + b: accumulator rows mi = 4h + a + 2dm, cols ni = b + 4dn
 * (HashTileTensorOp mapping of the 64x64 warp tile). */
__device__ __forceinline__ void local_xor(const int (&acc)[8][8][2], uint32_t (&part)[16])
{
#pragma unroll
    for (int p = 0; p < 16; ++p) {
        const int h = p >> 3, a = (p >> 2) & 1, b = p & 3;
        uint32_t x = 0;
#pragma unroll
        for (int dm = 0; dm < 2; ++dm)
#pragma unroll
            for (int dn = 0; dn < 2; ++dn) {
                const int mi = 4 * h + a + 2 * dm, ni = b + 4 * dn;
                x ^= (uint32_t)acc[mi][ni][0] ^ (uint32_t)acc[mi][ni][1];
            }
        part[p] = x;
    }
}

__global__ void __launch_bounds__(kThreads, 1) cp_turing_scan_kernel(TuringParams p)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    if (p.found && *p.found != 0) return;

    extern __shared__ __align__(128) uint8_t smem[];
    uint8_t* sA = smem;
    uint8_t* sB = smem + 2 * kStageA;
    uint32_t* tr = reinterpret_cast<uint32_t*>(smem + 2 * (kStageA + kStageB));

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp & 3, wn = warp >> 2;  /* warp tile 64x64 at (wm*64, wn*64) */
    /* Groups of `group` column blocks are walked row block first, so the CTAs in
     * flight share their B tiles in L2. */
    const int g = p.group;
    const int grp = blockIdx.x / (p.row_blocks * g), rr = blockIdx.x % (p.row_blocks * g);
    const int rb = p.row_block0 + rr % p.row_blocks;
    const int cb = p.col_block0 + grp * g + rr / p.row_blocks;

    const int8_t* gA0 = p.A + (size_t)rb * kTiles * kStageA + tid * 16;
    const int8_t* gA1 = gA0 + kStageA / 2;
    const int8_t* gB = p.B + (size_t)cb * kTiles * kStageB + tid * 16;
    /* Shared layout: 32-byte rows, 16-byte halves swizzled by row bit 2. */
    const int lr = tid >> 1, hh = tid & 1;
    const int sto = lr * 32 + ((hh ^ ((lr >> 2) & 1)) << 4);
    const int stoA1 = sto + (kBM / 2) * 32;
    const int swz = (lane >> 2) & 1;
    const uint32_t ldA = smem_u32(sA) + (wm * 64 + lane) * 32;
    const uint32_t ldB = smem_u32(sB) + (wn * 64 + lane) * 32;

#pragma unroll
    for (int i = 0; i < 2 * kWords; ++i) tr[i * kThreads + tid] = 0u;

    int acc[8][8][2];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[i][j][0] = acc[i][j][1] = 0;

    int4 ra0 = ldg_cg(gA0), ra1 = ldg_cg(gA1), rbv = ldg_cg(gB);
    *reinterpret_cast<int4*>(sA + sto) = ra0;
    *reinterpret_cast<int4*>(sA + stoA1) = ra1;
    *reinterpret_cast<int4*>(sB + sto) = rbv;
    ra0 = ldg_cg(gA0 + kStageA);
    ra1 = ldg_cg(gA1 + kStageA);
    rbv = ldg_cg(gB + kStageB);
    __syncthreads();

    /* Fragment double buffering across the barrier: half 0 of the next tile is
     * loaded right after the barrier while half 1 of the current tile is
     * multiplied, half 1 while half 0 is multiplied. */
    uint32_t a[2][8], b[2][8];
    auto load_frags = [&](int buf, int kk, uint32_t(&fa)[8], uint32_t(&fb)[8]) {
        const uint32_t base_a = ldA + buf * kStageA + ((kk ^ swz) << 4);
        const uint32_t base_b = ldB + buf * kStageB + ((kk ^ swz) << 4);
        ldsm_x4(base_a, fa[0], fa[1], fa[2], fa[3]);
        ldsm_x4(base_a + 1024, fa[4], fa[5], fa[6], fa[7]);
        ldsm_x4(base_b, fb[0], fb[1], fb[2], fb[3]);
        ldsm_x4(base_b + 1024, fb[4], fb[5], fb[6], fb[7]);
    };
    auto mma_half = [&](const uint32_t(&fa)[8], const uint32_t(&fb)[8]) {
#pragma unroll
        for (int mi = 0; mi < 8; ++mi)
#pragma unroll
            for (int nj = 0; nj < 8; ++nj) {
                const int ni = (mi & 1) ? 7 - nj : nj;  /* serpentine keeps a B fragment hot */
                imma(acc[mi][ni][0], acc[mi][ni][1], fa[mi], fb[ni]);
            }
    };
    auto fold = [&](int idx, uint32_t(&part)[16]) { fold_milestone(tr, tid, lane, idx, part); };
    load_frags(0, 0, a[0], b[0]);

    uint32_t part[16];
    for (int ms = 0; ms < kMilestones; ++ms) {
#pragma unroll
        for (int kq = 0; kq < kTilesPerMilestone; ++kq) {
            const int kt = ms * kTilesPerMilestone + kq;
            const int buf = kq & 1, nb = buf ^ 1;
            load_frags(buf, 1, a[1], b[1]);
            /* Fold of the previous milestone (complete after the last tile's
             * half 1). For ms == 0 the accumulators are zero and the fold is a
             * no-op on word 15. */
            if (kq == 0) local_xor(acc, part);
            mma_half(a[0], b[0]);
            if (kq == 0) fold((ms + kWords - 1) % kWords, part);
            *reinterpret_cast<int4*>(sA + nb * kStageA + sto) = ra0;
            *reinterpret_cast<int4*>(sA + nb * kStageA + stoA1) = ra1;
            *reinterpret_cast<int4*>(sB + nb * kStageB + sto) = rbv;
            const int nk = min(kt + 2, kTiles - 1);
            ra0 = ldg_cg(gA0 + (size_t)nk * kStageA);
            ra1 = ldg_cg(gA1 + (size_t)nk * kStageA);
            rbv = ldg_cg(gB + (size_t)nk * kStageB);
            __syncthreads();
            load_frags(nb, 0, a[0], b[0]);
            mma_half(a[1], b[1]);
        }
    }
    local_xor(acc, part);
    fold(kWords - 1, part);

    if (p.found && *p.found != 0 && !p.dump_words) return;
    emit_hash_tiles(p, tr, tid, lane, wm, wn, rb, cb);
#endif
}

/* ---- sm_80+ (Ampere/Ada) variant: mma.sync.m16n8k32 and a 4-stage cp.async ring. ---- */

constexpr int kStages80 = 4;
constexpr int kStage80 = kStageA + kStageB;
constexpr int kSmem80 = kStages80 * kStage80 + 2 * kWords * kThreads * 4;  /* 80 KB */

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
__device__ __forceinline__ void mma16832(int (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1)
{
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void cp_async16(uint32_t saddr, const void* gptr)
{
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" ::"r"(saddr), "l"(gptr));
}

__device__ __forceinline__ void cp_commit()
{
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int N>
__device__ __forceinline__ void cp_wait()
{
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
#endif

/* m16n8k32 warp tile 64x64: acc[mi][ni][e] holds row mi*16 + (e>>1)*8 + lane/4, col
 * ni*8 + (lane%4)*2 + (e&1). Partial p = h*8 + a*4 + b of the SIMT tiles this lane
 * touches: rows mi = 2h + dm, cols ni = b + 4dn, elements e = 2a + j. */
__device__ __forceinline__ void local_xor_m16(const int (&acc)[4][8][4], uint32_t (&part)[16])
{
#pragma unroll
    for (int p = 0; p < 16; ++p) {
        const int h = p >> 3, a = (p >> 2) & 1, b = p & 3;
        uint32_t x = 0;
#pragma unroll
        for (int dm = 0; dm < 2; ++dm)
#pragma unroll
            for (int dn = 0; dn < 2; ++dn)
                x ^= (uint32_t)acc[2 * h + dm][b + 4 * dn][2 * a] ^
                     (uint32_t)acc[2 * h + dm][b + 4 * dn][2 * a + 1];
        part[p] = x;
    }
}

__global__ void __launch_bounds__(kThreads, 1) cp_ampere_scan_kernel(TuringParams p)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if (p.found && *p.found != 0) return;

    extern __shared__ __align__(128) uint8_t smem[];
    uint32_t* tr = reinterpret_cast<uint32_t*>(smem + kStages80 * kStage80);

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp & 3, wn = warp >> 2;
    const int g = p.group;
    const int grp = blockIdx.x / (p.row_blocks * g), rr = blockIdx.x % (p.row_blocks * g);
    const int rb = p.row_block0 + rr % p.row_blocks;
    const int cb = p.col_block0 + grp * g + rr / p.row_blocks;

    /* Copy plan: A k-tile = 512 16-byte chunks (2 per thread), B = 256 (1 per thread); both
     * contiguous in the packed layout, stored as 32-byte rows with swizzled halves. */
    const int8_t* ga = p.A + (size_t)rb * kTiles * kStageA + tid * 16;
    const int8_t* gb = p.B + (size_t)cb * kTiles * kStageB + tid * 16;
    const int lr = tid >> 1, hh = tid & 1;
    const uint32_t sto = lr * kBK + ((hh ^ ((lr >> 2) & 1)) << 4);
    const uint32_t sbase = smem_u32(smem);
    auto issue = [&](int kt) {  /* clamped: the tail re-reads the last tile into a free stage */
        const size_t k = (size_t)min(kt, kTiles - 1);
        const uint32_t so = sbase + (kt % kStages80) * kStage80;
        cp_async16(so + sto, ga + k * kStageA);
        cp_async16(so + sto + (kBM / 2) * kBK, ga + k * kStageA + kStageA / 2);
        cp_async16(so + kStageA + sto, gb + k * kStageB);
        cp_commit();
    };

#pragma unroll
    for (int i = 0; i < 2 * kWords; ++i) tr[i * kThreads + tid] = 0u;

    int acc[4][8][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0;

    /* ldmatrix: A x4 per m16 tile = (rows 0-7,k0-15),(8-15,k0-15),(0-7,k16-31),(8-15,k16-31);
     * B x4 per pair of n8 tiles = (n0-7,k0-15),(n0-7,k16-31),(n8-15,k0-15),(n8-15,k16-31). */
    const int q = lane >> 3, r8 = lane & 7;
    const int a_row = wm * 64 + (q & 1) * 8 + r8, a_chunk = q >> 1;
    const uint32_t a_addr = sbase + a_row * kBK + ((a_chunk ^ ((a_row >> 2) & 1)) << 4);
    const int b_row = wn * 64 + (q >> 1) * 8 + r8, b_chunk = q & 1;
    const uint32_t b_addr = sbase + kStageA + b_row * kBK + ((b_chunk ^ ((b_row >> 2) & 1)) << 4);

    uint32_t fa[2][4][4], fb[2][8][2];
    auto load_frags = [&](int stage, uint32_t(&A)[4][4], uint32_t(&B)[8][2]) {
        const uint32_t so = stage * kStage80;
#pragma unroll
        for (int mi = 0; mi < 4; ++mi)
            ldsm_x4(a_addr + so + mi * 16 * kBK, A[mi][0], A[mi][1], A[mi][2], A[mi][3]);
#pragma unroll
        for (int nj = 0; nj < 4; ++nj)
            ldsm_x4(b_addr + so + nj * 16 * kBK, B[2 * nj][0], B[2 * nj][1], B[2 * nj + 1][0],
                    B[2 * nj + 1][1]);
    };
    auto mma_tile = [&](const uint32_t(&A)[4][4], const uint32_t(&B)[8][2]) {
#pragma unroll
        for (int mi = 0; mi < 4; ++mi)
#pragma unroll
            for (int nj = 0; nj < 8; ++nj) {
                const int ni = (mi & 1) ? 7 - nj : nj;
                mma16832(acc[mi][ni], A[mi], B[ni][0], B[ni][1]);
            }
    };
    auto fold = [&](int idx, uint32_t(&part)[16]) { fold_milestone(tr, tid, lane, idx, part); };

#pragma unroll
    for (int s = 0; s < kStages80 - 1; ++s) issue(s);
    cp_wait<kStages80 - 2>();
    __syncthreads();
    load_frags(0, fa[0], fb[0]);

    /* One k-tile = one k32 step; fragments of tile kt live in buffer kt & 1. */
    uint32_t part[16];
    for (int ms = 0; ms < kMilestones; ++ms) {
#pragma unroll
        for (int kq = 0; kq < kTilesPerMilestone; ++kq) {
            const int kt = ms * kTilesPerMilestone + kq;
            const int cur = kq & 1;
            if (kq == 0) local_xor_m16(acc, part);  /* previous milestone; ms == 0 is a no-op */
            cp_wait<kStages80 - 3>();               /* tile kt + 1 resident */
            __syncthreads();
            issue(kt + kStages80 - 1);
            load_frags((kt + 1) % kStages80, fa[cur ^ 1], fb[cur ^ 1]);
            mma_tile(fa[cur], fb[cur]);
            if (kq == 0) fold((ms + kWords - 1) % kWords, part);
        }
    }
    cp_wait<0>();
    local_xor_m16(acc, part);
    fold(kWords - 1, part);

    if (p.found && *p.found != 0 && !p.dump_words) return;
    emit_hash_tiles(p, tr, tid, lane, wm, wn, rb, cb);
#endif
}

__global__ void cp_turing_pack_kernel(const int8_t* src, int8_t* dst, int rows, int blk)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)rows * K_DIM) return;
    const int row = (int)(i / K_DIM), l = (int)(i % K_DIM);
    dst[cp_turing_packed_offset(row, l, K_DIM, blk)] = src[i];
}

/* Which packed kernel a device runs. */
enum PackedKernel { kPackedNone = 0, kPackedM8 = 1, kPackedM16 = 2 };

/* PTX version of the image of `fn` loaded on dev (0 if none). Both kernels
 * have their bodies compiled out below their arch (750 / 800), so e.g. a
 * --cuda-arch 75 binary on an Ampere card would JIT an empty m16n8k32 kernel
 * from the compute_75 PTX; this is how that is detected. */
static int kernel_ptx_version(int dev, const void* fn)
{
    int prev = -1;
    cudaGetDevice(&prev);
    if (prev != dev && cudaSetDevice(dev) != cudaSuccess) return 0;
    cudaFuncAttributes attr;
    const cudaError_t e = cudaFuncGetAttributes(&attr, fn);
    if (prev >= 0 && prev != dev) cudaSetDevice(prev);
    if (e != cudaSuccess) {
        cudaGetLastError();
        return 0;
    }
    return attr.ptxVersion;
}

struct PackedDeviceInfo {
    int kernel;  /* PackedKernel */
    int support; /* cp_turing_scan_supported() */
};

static PackedDeviceInfo packed_device_info_uncached(int dev)
{
    PackedDeviceInfo info{kPackedNone, 0};
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) return info;
    const int sm = prop.major * 10 + prop.minor;
    /* sm_75: m8n8k16 kernel. sm_8x (Ampere/Ada) and sm_12x (Blackwell GeForce): m16n8k32
     * kernel, the same mma.sync/cp.async/ldmatrix path. Datacenter Hopper/Blackwell
     * (sm_90/sm_10x) can run it too but keep CUTLASS unless asked (2 = opt-in).
     * Without a kernel image for the arch the packed path is off (the CUTLASS
     * kinds do the same check and fall back on their own). */
    if (sm >= 80) {
        if (kernel_ptx_version(dev, (const void*)cp_ampere_scan_kernel) >= 80)
            info.kernel = kPackedM16;
    } else if (sm == 75) {
        if (kernel_ptx_version(dev, (const void*)cp_turing_scan_kernel) >= 75)
            info.kernel = kPackedM8;
    }
    if (info.kernel != kPackedNone)
        info.support = (sm == 75 || prop.major == 8 || prop.major == 12) ? 1 : 2;
    return info;
}

constexpr int kMaxCachedDevices = 64;

static PackedDeviceInfo packed_device_info(int dev)
{
    static PackedDeviceInfo cache[kMaxCachedDevices];
    static bool cached[kMaxCachedDevices];
    if (dev < 0 || dev >= kMaxCachedDevices) return packed_device_info_uncached(dev);
    if (!cached[dev]) {
        cache[dev] = packed_device_info_uncached(dev);
        cached[dev] = true;
    }
    return cache[dev];
}

/* Dynamic shared memory above 48 KiB needs the opt-in attribute, set per
 * device; devices past the cache are set on every launch. */
static cudaError_t ensure_kernel_attributes(int dev, int kernel)
{
    static bool done[kMaxCachedDevices];
    const bool cacheable = dev >= 0 && dev < kMaxCachedDevices;
    if (cacheable && done[dev]) return cudaSuccess;
    const cudaError_t e =
        kernel == kPackedM16
            ? cudaFuncSetAttribute(cp_ampere_scan_kernel,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem80)
            : cudaFuncSetAttribute(cp_turing_scan_kernel,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem);
    if (e == cudaSuccess && cacheable) done[dev] = true;
    return e;
}

}  // namespace

extern "C" int cp_turing_scan_supported(int dev)
{
    return packed_device_info(dev).support;
}

extern "C" const char* cp_turing_scan_name(int dev)
{
    switch (packed_device_info(dev).kernel) {
    case kPackedM16:
        return "packed 256x128x32 / 64x64x32 / mma.m16n8k32.s8 / 4-stage cp.async "
               "(256 thr, 80 KiB smem)";
    case kPackedM8:
        return "packed 256x128x32 / 64x64x32 / mma.m8n8k16.s8 / 2-stage "
               "(256 thr, 56 KiB smem)";
    default:
        return "packed (not available)";
    }
}

extern "C" int cp_turing_period_batch(int dev, const int8_t* d_Ap, const int8_t* d_BpT, int m,
                                      int n, int row_period0, int col_period0, int row_batch,
                                      int col_batch, const CpCutlassJackpotLaunch* jackpot,
                                      uint32_t* d_dump_words)
{
    (void)n;
    if ((row_period0 | row_batch) & 1 || m % kBM != 0) {
        fprintf(stderr, "[turing] row periods must come in pairs (got %d+%d, m=%d)\n",
                row_period0, row_batch, m);
        return -1;
    }
    const int kernel = packed_device_info(dev).kernel;
    if (kernel == kPackedNone) {
        fprintf(stderr, "[turing] GPU%d: no packed scan kernel image for this arch\n", dev);
        return -1;
    }
    if (ensure_kernel_attributes(dev, kernel) != cudaSuccess) return -1;
    const bool use80 = kernel == kPackedM16;
    TuringParams p{};
    p.A = d_Ap;
    p.B = d_BpT;
    p.row_block0 = row_period0 / 2;
    p.col_block0 = col_period0;
    p.row_blocks = row_batch / 2;
    p.col_blocks = col_batch;
    p.group = col_batch % 8 == 0 ? 8 : 1;
    if (jackpot) {
        for (int i = 0; i < 8; ++i) p.bound[i] = jackpot->bound[i];
        p.a_key8 = jackpot->d_a_key8;
        p.found = jackpot->d_found;
        p.out_t_rows = jackpot->d_out_t_rows;
        p.out_t_cols = jackpot->d_out_t_cols;
    }
    p.dump_words = d_dump_words;
    if (use80)
        cp_ampere_scan_kernel<<<p.row_blocks * p.col_blocks, kThreads, kSmem80>>>(p);
    else
        cp_turing_scan_kernel<<<p.row_blocks * p.col_blocks, kThreads, kSmem>>>(p);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[turing] launch failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

extern "C" int cp_turing_pack(const int8_t* d_src, int8_t* d_dst, int rows, int blk)
{
    const size_t total = (size_t)rows * K_DIM;
    const int tpb = 256;
    cp_turing_pack_kernel<<<(unsigned)((total + tpb - 1) / tpb), tpb>>>(d_src, d_dst, rows, blk);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}
