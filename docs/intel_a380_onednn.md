# Intel Arc (Xe-HPG): oneDNN systolic (XMX) backend

On an Arc A380 the oneDNN/gemmstone backend with XeHPG systolic kernels mines
at **16.66 TMAC/s** per complete 131072² attempt, up from 3.2 TMAC/s before
these changes (the backend never selected a systolic kernel on Xe-HPG).
Use it on Arc:

```sh
cppminer --backend onednn --devices 0 --verify \
  --pool stratum+tcp://POOL:PORT --wallet WALLET --worker arc-a380
```

Results apply to this A380 and driver; other Arc models have not been tested.

## What changed

The `onednn` backend existed for Xe-LP/Xe-LPG iGPUs, which have no XMX. On the
A380 it had never selected a systolic kernel:

- **Milestone check.** oneDNN's catalog ranks ten systolic int8 kernels for
  DG2. Each unrolls K by 256–512 (`ks64`/`ks128` × `cab3`/`cab4`), while the
  Pearl milestone XOR fired only at unrolled-panel boundaries and therefore
  required `unrollK | 128`. All ten were rejected. When `unrollK` exceeds the
  128-term milestone, the XOR is now scheduled every 128 k inside the panel
  (`case5XorEveryK`, `k_loop.cxx`).
- **Fallback crash.** The non-catalog fallback candidates were flagged as
  catalog entries with a null entry and segfaulted in `evaluate()`.
- **Kernel choice.** oneDNN's model ranks plain GEMM. With the milestone fold,
  its first choice (`catalog-334`, 16x16) scanned at 9.7 TMAC/s. The 8x4
  work-group 16x32 entry (`catalog-333`/`352`) is now moved to the front for
  XeHPG; it is matched by strategy string and unroll, because the catalog
  repeats the string with a 16x16 unroll (`catalog-343`). Dropping `af`
  (atomic FMA) from its strategy adds 1.06% (ABAB at 131072²).
- **Panels.** Systolic kernels default to 1024x1024 hash tiles per panel
  (16384² GEMM per launch) unless `--batch-size`/`--row-period-batch` are
  given. `--batch-size 1024` was previously mistaken for the default and
  replaced by 256.
- **Jackpot pass and pipelining.** The separate jackpot kernel kept its
  milestone array, fold slots and BLAKE3 message permutation in private
  memory (dynamic indexing). Constant indexing and a constant message schedule
  give identical digests 20% faster. The scan now runs GEMM panel i and the
  jackpot of panel i-1 on an out-of-order queue with two `tile_xor` buffers,
  waiting only for the previous panel's found flag (`CASE5_PIPELINE=0` restores
  the synchronous loop).
- **Build.** `src/onednn/prepare_onednn_deps.sh` fetches oneDNN v3.13.2 and
  vendors nGEN/gemmstone on Linux, like `prepare_onednn_deps.bat` on Windows;
  the oneDNN jackpot kernel is copied next to the binary when OpenCL is also
  enabled.

## Measurements

Arc A380 (8086:56a5), Ubuntu kernel 6.8 with i915, PCIe 2.0 x1,
`intel-opencl-icd` 23.43.27642.40, IGC 1.0.15468.25. Complete zero-target
scans on a loopback pool (no early hit), first attempt discarded; each case
first built and verified a mock share.

**131072², default flags, 5 attempts:** 16.66 TMAC/s per full attempt
(scan 17.02). The first build of this backend, before the jackpot/pipeline/`af`
changes, measured 15.47 (15.78).

**Systolic catalog kernels, 32768², 256x256 panels:**

| Kernel | Unroll / work group | Full | Scan |
|---|---|---:|---:|
| catalog-333 / 352 | 16x32 / 8x4 | **11.387** | 13.178 |
| catalog-343 | 16x16 / 8x4 | 8.985 | 10.064 |
| catalog-353 / 334 | 16x16 / 8x8 | 8.705 | 9.714 |
| catalog-335 | 8x16 / 8x4 | 5.984 | 6.446 |
| catalog-337 | 8x8 / 8x8 | 3.651 | 3.818 |
| catalog-339 | 8x8 / 8x4 | 3.602 | 3.764 |
| catalog-341 | 16x4 / 2x8 | — | 16x4 hash tile; proof build fails |

**Panels for catalog-333 at 131072²:** 256x1024 hash tiles 14.30, 512x512
14.39, 512x1024 14.81, 1024x1024 **15.15** TMAC/s per full attempt.

**Where the time goes:**

- Milestone XOR: skipping the fold (`CASE5_XOR_NOP=1`, diagnostic only) gives
  11.86 vs 11.39 TMAC/s, about 4%.
- Compute-bound: the XMX scan runs at 2000 MHz. Capping the clock at
  1600 MHz lowers the scan rate from 15.79 to 12.55 TMAC/s (65536²), the
  same 1.25x ratio as the clocks, so DRAM bandwidth is not the limit. At
  2000 MHz the XMX peak is 8 Xe-cores × 2048 int8 MAC/clock = 32.8 TMAC/s.
- `grf128` variants (16x16, 8x32 per thread) double the thread slots but scan
  at 11.4–12.6 TMAC/s; dropping `dw` (dpasw) costs 20%; `cs`, `sm`/`sn` and
  `sB32 sB32` were neutral or slower; deeper SLM pipelines (`cab8`, `ks128`)
  do not fit or are unsupported.
- Larger custom tiles (32x32, 16x64 per thread; 8x8 work groups) either do
  not fit DG2's registers/SLM or are slower. Hilbert walk order, other
  B access widths and dropping `sr`/`pab` changed results within ±3%;
  removing boustrophedon walk order cost 27%. The fused in-kernel jackpot
  (`--fused-jackpot`) is 7% slower than the separate jackpot pass.
- Level Zero `ComputeBasic` counters while mining at 131072²: XMX pipeline
  active 36.6% before and 43.4% after the jackpot, pipeline and `af` changes;
  memory traffic is about a fifth of the 186 GB/s bandwidth. The kernel is
  stall-bound: `grf256` caps occupancy at half the hardware threads, so SLM
  loads and the per-stage barriers are poorly hidden.

## Validation

- 20 mock shares with the final kernel built and verified through the proof
  verifier, with winning tiles in both 16x16 halves of the 16x32 unroll
  (plus 13 with other catalog kernels).
- One live pool share accepted.

Not established: long-term acceptance rate, other Arc models, and layouts
other than the default TN (an `NT` mock proof failed and was not
investigated).

## Tuning hooks

| Variable | Purpose |
|---|---|
| `CASE5_KERNEL=catalog-N` | Keep one catalog candidate |
| `CASE5_STRATEGY`, `CASE5_UNROLL=MxN` | Try a custom gemmstone strategy first (`_` may replace spaces) |
| `CASE5_XOR_NOP=1` | Diagnostic ceiling without the fold; shares will not verify |
| `CASE5_DEBUG_SELECT=1` | Log every candidate and why it was rejected |
| `CASE5_PIPELINE=0` | Synchronous GEMM -> jackpot -> flag loop |

## Build on Linux

```sh
cd src/onednn && ./prepare_onednn_deps.sh && cd ../..
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCP_ENABLE_OPENCL=ON -DCP_ENABLE_ONEDNN=ON
cmake --build build -j4
```
