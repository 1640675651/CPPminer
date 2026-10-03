// Hash-tile policies for the fused Case-10 kernel (gemm_inline_xor_kernel.h).
//
// The proof format (CP_TILE_LAYOUT_CUTLASS, PEARL_CUTLASS_CONFIG) is defined by
// the SIMT accumulator lane map of MmaLaneTile128x128: each of the 256 SIMT
// threads of a 128x128 CTA owns one 8x8 "hash tile" made of four 4x4 blocks of
// the int32 C tile (rows {g*4..g*4+3, +16}, cols {h*4..h*4+3, +32} inside its
// 32x64 warp tile). At every milestone (R_RANK = 128 K columns) the 64 cells of
// a hash tile are XORed into one 32-bit word which is folded into 16 jackpot
// words (rotl 13), and BLAKE3 of those words is the share candidate. The Rust
// verifier recomputes exactly that, so every kernel must reproduce the same
// per-hash-tile XOR words, in the same milestone order.
//
// HashTileSimt     : accumulator fragment == hash tile, XOR all 64 elements.
// HashTileTensorOp : mma.sync m8n8k16 fragments own different C cells, so the
//                    per-lane partial XORs are regrouped per "virtual SIMT
//                    thread" with a warp-shuffle reduce-scatter (XOR is
//                    associative/commutative, so grouping order is irrelevant).
//                    Also exact for mma.sync m16n8k32 (Sm80+): its warp
//                    accumulator fragment has the identical register layout,
//                    see HashTileTensorOpFor below.
//
// Virtual CTAs: a tensor-op threadblock may be larger than the proof's 128x128
// CTA (256x128 or 128x256). It then behaves as kVirtM x kVirtN "virtual"
// 128x128 CTAs: every warp tile (32 or 64 rows x 64 cols) lies inside exactly
// one of them, so each lane's hash tiles, their virtual SIMT thread index and
// the virtual CTA coordinate are functions of (warp, lane) alone.
// virtual_cta() returns that coordinate; the kernel adds it to its threadblock
// tile offset scaled by (kVirtM, kVirtN) and from there on uses exactly the
// 128x128 indexing (tile-xor offsets, jackpot row/col period, t_rows/t_cols).
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"

namespace cp_cutlass {

/* SIMT (dp4a) kernel: one hash tile per thread, FragmentC is the hash tile. */
struct HashTileSimt {
  static constexpr int kTilesPerThread = 1;
  static constexpr int kSimtWarps = 8;
  static constexpr int kVirtM = 1;
  static constexpr int kVirtN = 1;

  CUTLASS_DEVICE static void virtual_cta(int /*warp_idx*/, int &vm, int &vn) {
    vm = 0;
    vn = 0;
  }

  template <typename FragmentC, typename Emit>
  CUTLASS_DEVICE static void milestone_xor(FragmentC const &accum, int /*lane*/,
                                           Emit &&emit) {
    uint32_t xv = 0u;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < FragmentC::kElements; ++i)
      xv ^= static_cast<uint32_t>(accum[i]);
    emit(0, xv);
  }

  /* Virtual SIMT thread index (0..255) that owns hash tile j of this thread. */
  CUTLASS_DEVICE static int virtual_thread(int warp_idx, int lane,
                                           int /*j*/) {
    return warp_idx * 32 + lane;
  }
};

/* Tensor-op (mma.sync.m8n8k16 s8, Sm75+) kernel.
 *
 * Requirements: threadblock a multiple of 128x128, warp tile (kWarpM x 64)
 * with kWarpM in {32, 64}, instruction 8x8xK. Warp (wm, wn) covers rows
 * wm*kWarpM.., cols wn*64.. of the threadblock, i.e. global SIMT warp rows
 * wm*kHalves+h (h < kHalves = kWarpM/32) and column wn; within its virtual
 * 128x128 CTA (wm*kHalves/4, wn/2) that is SIMT warp
 * (wm*kHalves+h)%4 + 4*(wn%2).
 *
 * mma.sync m8n8k16 accumulator (one 8x8 instruction tile, lane bits b4..b0):
 *
 *            col:  0 1 | 2 3 | 4 5 | 6 7         row  = lane>>2 = b4 b3 b2
 *      row 0 lane  0 0 | 1 1 | 2 2 | 3 3         col  = (lane&3)*2 + i
 *      row 1 lane  4 4 | 5 5 | 6 6 | 7 7         i    = element 0/1 of the pair
 *      ...                                         ^ 4-col block = b1
 *      row 3 lane 12 .. 13 .. 14 .. 15 ..
 *      --- 4-row block boundary (b4) ---
 *      row 4 lane 16 16| 17 17| 18 18| 19 19
 *      ...
 *      row 7 lane 28 .. 29 .. 30 .. 31 ..
 *
 * FragmentC element (mi, ni, i) = accum[2*(ni*kRowIters + mi) + i], where
 * instruction tile (mi, ni) sits at rows mi*8.., cols ni*8.. of the warp tile.
 *
 * SIMT hash tile of a cell (r, c) inside a 32x64 SIMT warp tile:
 *   group_m = (r % 16) / 4 = (mi % 2) * 2 + b4      (partner rows: mi ^ 2)
 *   group_n = (c % 32) / 4 = (ni % 4) * 2 + b1      (partner cols: ni ^ 4)
 *   simt_lane = (group_n / 2) * 8 + group_m * 2 + (group_n % 2)
 *
 * So for a lane, instruction tiles {4h+a, 4h+a+2} x {b, b+4} (8 int32 each)
 * all belong to hash tile (h, a, b); j = h*8 + a*4 + b indexes the 8*kHalves
 * per-lane partials w[j]. The 8 lanes sharing (b4, b1) — i.e. differing in
 * b3, b2, b0 — hold partials of the same hash tiles, so a 3-step XOR
 * reduce-scatter over masks 8, 4, 1 leaves each lane with kHalves fully
 * reduced words, for hash tiles j = (b3*4 + b2*2 + b0) * kHalves + t.
 * The owner lane keeps the jackpot fold state for those tiles.
 */
template <typename WarpShape, typename WarpCount, int kRowIters, int kColIters>
struct HashTileTensorOp {
  static_assert(WarpShape::kN == 64, "warp tile N must be 64 (SIMT warp N)");
  static_assert(WarpShape::kM % 32 == 0 && WarpShape::kM <= 64,
                "warp tile M must be 32 or 64");
  static_assert(kRowIters == WarpShape::kM / 8 && kColIters == 8,
                "instruction tile must be 8x8");
  static constexpr int kHalves = WarpShape::kM / 32;
  static constexpr int kTilesPerThread = kHalves;
  static constexpr int kPartials = 8 * kHalves;
  static constexpr int kSimtWarps = 8;
  /* Threadblock = kVirtM x kVirtN virtual 128x128 CTAs (4 SIMT warps of 32
   * rows in M, 2 of 64 cols in N each). */
  static_assert((WarpCount::kM * kHalves) % 4 == 0 && WarpCount::kN % 2 == 0,
                "threadblock must be a multiple of 128x128");
  static constexpr int kVirtM = WarpCount::kM * kHalves / 4;
  static constexpr int kVirtN = WarpCount::kN / 2;

  /* Virtual 128x128 CTA (vm, vn) of the threadblock that holds this warp's
   * tile: SIMT warp row warp_m*kHalves + h with h < kHalves never crosses a
   * multiple of 4 (kHalves divides 4), so all hash tiles of a warp share it. */
  CUTLASS_DEVICE static void virtual_cta(int warp_idx, int &vm, int &vn) {
    int const warp_idx_mn = warp_idx % (WarpCount::kM * WarpCount::kN);
    int const warp_m = warp_idx_mn % WarpCount::kM;
    int const warp_n = warp_idx_mn / WarpCount::kM;
    vm = (warp_m * kHalves) / 4;
    vn = warp_n / 2;
  }

  template <typename FragmentC, typename Emit>
  CUTLASS_DEVICE static void milestone_xor(FragmentC const &accum, int lane,
                                           Emit &&emit) {
    static_assert(FragmentC::kElements == 2 * kRowIters * kColIters,
                  "unexpected accumulator fragment size");
    uint32_t w[kPartials];
    CUTLASS_PRAGMA_UNROLL
    for (int h = 0; h < kHalves; ++h) {
      CUTLASS_PRAGMA_UNROLL
      for (int a = 0; a < 2; ++a) {
        CUTLASS_PRAGMA_UNROLL
        for (int b = 0; b < 4; ++b) {
          uint32_t x = 0u;
          CUTLASS_PRAGMA_UNROLL
          for (int dm = 0; dm < 2; ++dm) {
            CUTLASS_PRAGMA_UNROLL
            for (int dn = 0; dn < 2; ++dn) {
              int const mi = 4 * h + a + 2 * dm;
              int const ni = b + 4 * dn;
              int const e = 2 * (ni * kRowIters + mi);
              x ^= static_cast<uint32_t>(accum[e]) ^
                   static_cast<uint32_t>(accum[e + 1]);
            }
          }
          w[h * 8 + a * 4 + b] = x;
        }
      }
    }

    /* Reduce-scatter across the 8 lanes sharing (b4, b1). */
    reduce_scatter_step<kPartials, 8>(w, lane);
    reduce_scatter_step<kPartials / 2, 4>(w, lane);
    reduce_scatter_step<kPartials / 4, 1>(w, lane);

    CUTLASS_PRAGMA_UNROLL
    for (int t = 0; t < kHalves; ++t)
      emit(t, w[t]);
  }

  CUTLASS_DEVICE static int virtual_thread(int warp_idx, int lane, int t) {
    int const warp_idx_mn = warp_idx % (WarpCount::kM * WarpCount::kN);
    int const warp_m = warp_idx_mn % WarpCount::kM;
    int const warp_n = warp_idx_mn / WarpCount::kM;
    int const b0 = lane & 1;
    int const b1 = (lane >> 1) & 1;
    int const b2 = (lane >> 2) & 1;
    int const b3 = (lane >> 3) & 1;
    int const b4 = (lane >> 4) & 1;
    int const j = (b3 * 4 + b2 * 2 + b0) * kHalves + t;
    int const h = j / 8;
    int const a = (j / 4) % 2;
    int const b = j % 4;
    int const group_m = a * 2 + b4;
    int const group_n = b * 2 + b1;
    int const simt_lane = (group_n / 2) * 8 + group_m * 2 + (group_n % 2);
    /* SIMT warp inside the warp's virtual 128x128 CTA (see virtual_cta). */
    int const simt_warp = (warp_m * kHalves + h) % 4 + 4 * (warp_n % 2);
    return simt_warp * 32 + simt_lane;
  }

private:
  /* One halving step: lanes with (lane & kMask) keep the upper half. */
  template <int kCount, int kMask>
  CUTLASS_DEVICE static void reduce_scatter_step(uint32_t *w, int lane) {
    constexpr int kHalf = kCount / 2;
    bool const hi = (lane & kMask) != 0;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kHalf; ++i) {
      uint32_t const send = hi ? w[i] : w[i + kHalf];
      uint32_t const keep = hi ? w[i + kHalf] : w[i];
      uint32_t const recv = __shfl_xor_sync(0xffffffffu, send, kMask);
      w[i] = keep ^ recv;
    }
  }
};

/* Policy for a 128x128 threadblock, warp tile WarpShape and mma.sync
 * instruction InstructionShape (8x8x16 on Sm75, 16x8x32 on Sm80+).
 *
 * m16n8k32 s8 -> s32 accumulator (PTX ISA "mma.m16n8k32 C/D fragment";
 * CUTLASS MmaTensorOpAccumulatorTileIterator<RowMajor>: kElementsPerAccess =
 * N/4 = 2, kRowsPerTile = 8, kAccumulatorRows = M/8 = 2). Lane l holds 4
 * int32 c0..c3 of the 16x8 instruction tile:
 *
 *            col:  0 1 | 2 3 | 4 5 | 6 7      c0,c1: row = l>>2,     col = 2*(l&3)+{0,1}
 *      row  0 lane  0 0 | 1 1 | 2 2 | 3 3      c2,c3: row = (l>>2)+8, col = 2*(l&3)+{0,1}
 *      ...                                     i.e. c_e: row = (l>>2) + 8*(e>>1),
 *      row  7 lane 28 28| 29 29| 30 30| 31 31              col = 2*(l&3) + (e&1)
 *      ---- upper 8x8 half (c0,c1) / lower 8x8 half (c2,c3) ----
 *      row  8 lane  0 0 | 1 1 | 2 2 | 3 3
 *      ...
 *      row 15 lane 28 28| 29 29| 30 30| 31 31
 *
 * Each 8-row half is exactly an m8n8k16 accumulator tile (same lane -> (row,
 * col) map, two consecutive columns per lane).
 *
 * Warp fragment order: MmaTensorOp::operator() writes instruction tile (m, n)
 * to ptr_D[m + n * MmaIterations::kRow] (both the sm_75 and the sm_80
 * serpentine loops only change the visiting order, not the slot), and the
 * epilogue's IteratorC reads frag[kAccumulatorRows*kElementsPerAccess*(n*kRow
 * + m) + row*kElementsPerAccess + col]. With R8 = WarpShape::kM / 8 8-row
 * groups, the m16n8k32 element (m16, n, e) is
 *     accum[4*(n*R8/2 + m16) + e]
 *   = accum[2*(n*R8 + 2*m16 + (e>>1)) + (e&1)]
 *   = accum[2*(n*R8 + m8) + i]          with m8 = 2*m16 + (e>>1), i = e&1,
 * which is the m8n8k16 index of the cell in 8x8 tile (m8, n), element i --
 * the formula HashTileTensorOp uses (e = 2*(ni*kRowIters + mi)). So for the
 * same warp tile, cell (r, c) lives in the same lane and the same register
 * slot under both instructions, and HashTileTensorOp (partials over the
 * 8x8 tiles {4h+a, 4h+a+2} x {b, b+4}, reduce-scatter over lane masks 8,4,1)
 * emits the identical virtual SIMT hash-tile words. cp_cutlass_hash_policy_
 * selftest() checks this against CUTLASS's own IteratorC for every tensor-op
 * instantiation (it uses no mma instruction, so it also runs on sm_75). */
template <typename WarpShape, typename InstructionShape,
          typename ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 64>>
struct HashTileTensorOpSelect {
  static_assert(InstructionShape::kN == 8 &&
                    (InstructionShape::kM == 8 || InstructionShape::kM == 16),
                "hash-tile policy supports mma.sync m8n8k* and m16n8k*");
  static_assert(WarpShape::kM % InstructionShape::kM == 0,
                "warp tile M must be a multiple of the instruction M");
  static_assert(ThreadblockShape::kM % 128 == 0 &&
                    ThreadblockShape::kN % 128 == 0,
                "threadblock must tile into virtual 128x128 CTAs");
  using type = HashTileTensorOp<
      WarpShape,
      cutlass::gemm::GemmShape<ThreadblockShape::kM / WarpShape::kM,
                               ThreadblockShape::kN / WarpShape::kN, 1>,
      WarpShape::kM / 8, WarpShape::kN / 8>;
};

/* ThreadblockShape larger than 128x128 (e.g. 256x128, 128x256) runs as
 * several virtual 128x128 CTAs, see virtual_cta(). */
template <typename WarpShape, typename InstructionShape,
          typename ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 64>>
using HashTileTensorOpFor =
    typename HashTileTensorOpSelect<WarpShape, InstructionShape,
                                    ThreadblockShape>::type;

} // namespace cp_cutlass
