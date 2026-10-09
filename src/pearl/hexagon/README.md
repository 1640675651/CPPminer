# Pearl Hexagon backend (`--backend hexagon`)

Pearl mining on the compute DSP (cDSP) of Snapdragon phones. The DSP's HVX
vector units generate the noise, pack the matrices, and run the GEMM and the
milestone XOR. The host CPU keeps signal A and its hash, derives the seeds, and
runs the BLAKE3 jackpot. Tested on a Snapdragon 480 (SM4350, Hexagon v66, 2 HVX
contexts).

| Device | Backend | Hashrate (m = n = 32768) |
|---|---|---|
| Snapdragon 480 cDSP | hexagon | ~398 GMAC/s while scanning, ~397 GMAC/s overall |
| Snapdragon 480 CPU (2x A76 + 6x A55) | cpu (NEON DotProd) | ~130 GMAC/s |

The overall rate includes about 0.03 s of host work per 11.07 s attempt, while
the DSP is idle: the random signal A, its hash for the noise seed, and the
permutation pairs. Each new job also idles the DSP for ~0.38 s:
- ~0.07 s for the zero B's hash, which gives the B noise seed;
- ~0.10 s for the DSP to generate B;
- ~0.19 s to rebuild signal A's hash tree under the new job key, so the job's
  first attempt is incremental too.

Both hashes compute their 1 KB chunk hashes on the DSP (`pearlx chunk_cvs`, HVX
BLAKE3, 32 chunks per vector) and build the tree above them on the host
(`pearl_matrix_hash_from_cvs`). The host alone takes ~0.36 s per 128 MiB.

Measured from `adb shell`, which limits the process to the six A55 cores.

The hash of signal A is incremental (`pearl_matrix_hash_digest` in
`cp_noise.c`). An attempt changes one value per column, so only about 4,000 of
signal A's 131,072 1 KB chunks change. The worker keeps every chunk's chaining
value and every complete parent of the BLAKE3 tree, and hashes again only the
changed chunks and their ancestors. That takes 0.03 s instead of 0.35 s; the
first attempt of each job hashes everything. Set `CP_HEXAGON_HASH_CHECK=1` to
compare it with a full hash on every attempt.

Where the scan rate goes, against the kernel's ~413 GMAC/s peak:

| Cost | Share |
|---|---|
| Copying a 128 KB slice of B into VTCM per 128-column tile and K block, amortized over the launch's 1024 row tiles | ~1.9% |
| Generating and packing each row panel's noisy A (`set_a_gen`, ~9.7 ms per 8 launches) | ~0.7% |
| FastRPC call per launch (~1.1 ms of 170 ms) and the gap between launches | ~1.4% |

The per-launch call overhead was ~3 ms until the tile-XOR records moved to an
**uncached** shared buffer. FastRPC invalidates a cached output buffer's 8 MiB in
the CPU cache after every call. The host jackpot copies each 256-byte record in
with wide loads, which keeps uncached reads cheap: about 0.9 s of CPU per
attempt, overlapped with the DSP.

The DSP library also has a `scan` method that runs the jackpot on the DSP (HVX
BLAKE3, 32 hash tiles per vector) and returns only the first hit. Its results
match the host jackpot exactly (checked by `pearlx_test`), but it adds ~2.9 ms of
DSP time per launch. That's more than the ~1 ms of FastRPC overhead it would
save, so the miner keeps the jackpot on the host.

## Hash tile: 4 x 64

The DSP kernel keeps a 4 x 128 register tile of C in 16 HVX accumulators.
zk-pow accepts hash tiles of at most h*w = 256 elements, so 4 x 128 cannot be a
hash tile. The kernel instead XORs each 4 x 64 half separately (gemm_xor
Case 1.1):

- `mining_config`: rows 0..3, columns 0..63 (`PEARL_CONTIGUOUS_4x64_CONFIG`).
- Proof tile layout: `CP_TILE_LAYOUT_CONTIGUOUS_4x64` = 6, in `cp_proof.h` and
  in `rust/cp-proof-ffi`.
- Jackpot scale factor: 4 * 64 * (k/r) * 128.

The Rust tests build and zk-pow-verify a 4 x 64 proof. A share mined on the
phone was also proven and verified with `cp_proof_verify`. Whether a pool
accepts this new configuration has not been tested.

## How it works

The matrix is scanned in **launches** of 128 x 128 macro blocks, like the GPU
backends' macro panels: by default 32 x 32 macros, so 4096 rows x 4096 columns
(65,536 hash tiles) per DSP launch. A row of launches is a **row panel**.

| Step | Where | When |
|---|---|---|
| B noise seed (hash of the zero B^T), B permutation pairs | host | once per job |
| Noisy B (signal B = 0) generated in packed layout (`pearlx set_b_gen`) | cDSP | once per job |
| Sparse random signal A, its incremental hash, `a_noise_seed`, A permutation pairs | host | every attempt |
| Noisy A of a row panel generated from its signal rows, in packed layout (`pearlx set_a_gen`) | cDSP | per panel, at its first launch |
| GEMM + milestone XOR of the panel against one block of columns (`pearlx gemm_xor`) | cDSP | per launch |
| BLAKE3 jackpot over the launch's tile XORs | host, OpenMP | per launch |

A dedicated thread issues the launches back to back. The mining thread follows,
checking each finished launch. The DSP only waits when the record buffer it is
about to reuse hasn't been checked yet. In practice it is busy about 98.8% of
the scan.

Every hit is recomputed on the host before it becomes a share. The host builds
the tile's A rows from signal A and its B columns from the job seed, using its
own noise code (`cp_noise.c`). A wrong result from the DSP, in the kernel or in
the noise it generated, can therefore never be submitted.

Proofs use the host signal A with an all-zero B^T, like the CPU backend.
Signal A lives in memory shared with the DSP (`cp_worker_alloc_host_signal_a`),
which reads each panel's signal rows in place.

**Noise generation on the DSP** reproduces `cp_noise.c` bit for bit; `pearlx_test
-G 1` checks every XOR word against host-generated matrices. For a noise row (a
row of A, or a column of B):
1. **Table:** 4 keyed-BLAKE3 digests (key: the noise seed; message: the block
   index and the label) give a 128-entry table of `(byte & 63) - 16`.
2. **Noise:** `noise[l] = table[first[l]] - table[second[l]]` over the k
   permutation pairs.

The DSP handles 128 rows at a time:
- An HVX BLAKE3 computes the tables, 32 digests per vector.
- A byte transpose makes them index-major, so one noise column for 128 rows is
  one vector subtract.
- For A, a second transpose turns 128 columns back into rows, which get the
  signal added and are packed into the kernel's 4-row chunks. The gather, both
  transpose passes, the add and the packing run fused in registers.
- For B, four noise columns interleave directly into packed B's 4-byte words.

B takes ~0.1 s per job instead of ~0.75 s of host noise plus packing, and there
is no 128 MiB staging copy. A takes ~9.7 ms per 4096-row panel versus 6.8 ms
for packing host-built rows.

In each launch, for each 128-column tile of B and each K block of 1024, the DSP
copies a 128 KB slice of B into VTCM and streams the panel's row tiles through
the kernel. The kernel is the HVX `vrmpyz`
(Z-buffer) GEMM from `gemm_xor/hexagon`. After every 128 values of k it XORs
each half's 8 accumulators lane-wise, packs both halves into one vector with a
`vdeal`, and reduces the 8 milestones to words between tiles. The XOR costs
about 4.5% over the plain GEMM.

## Files

| Path | Contents |
|---|---|
| `cp_pearl_hexagon_worker.cpp` | Worker: job and attempt prep (seeds, signal A, its incremental hash), the launch pipeline, jackpot, hit recompute |
| `cp_pearlx_client.c` | FastRPC client: loads `libcdsprpc.so` at run time and marshals the `pearlx` calls by hand. No Hexagon SDK headers are needed |
| `dsp/inc/pearlx.idl` | DSP interface: `set_b_gen` and `set_a_gen` (noise generated on the DSP), `gemm_xor`, `chunk_cvs` (matrix hash leaves), `info`; `set_b`, `set_a` (pack host-built matrices) and `scan` (jackpot on the DSP) are used only by the self-test |
| `dsp/src/pearlx_imp.c` | DSP side: noise generation, packing, VTCM, threads over both HVX contexts, the DSP jackpot |
| `dsp/src/pearlx.S`, `dsp/src/pearlx_xor.inc` | HVX kernel (Case 1.1 fold and reduction) |
| `dsp/src/pearlx_test.c` | Android self-test against CPU references, plus timing: GEMM + XOR, the DSP jackpot (`-b` hit rate), the noise generators (`-G 1`) and the chunk hashes (`-H 1`), with the miner's `cp_noise.c` linked in as the reference |
| `dsp/CMakeLists.txt`, `dsp/build.ps1` | Hexagon SDK build of the DSP library and the self-test |

## Build

There are two parts:

1. **The DSP library** `libpearlx_skel.so`. It needs the Hexagon SDK on an x86
   host (tested with SDK 5.5.7.0 on Windows):

   ```powershell
   cd src\pearl\hexagon\dsp
   .\build.ps1                                   # hexagon_Release_toolv87_v66\ship\libpearlx_skel.so
   .\build.ps1 -NoBuild -Test "-n 32768 -r 4096 -w 4096 -g 2"   # push + self-test on the phone
   .\build.ps1 -NoBuild -Test "-n 32768 -r 4096 -G 1"           # DSP noise vs cp_noise.c
   ```

   Ship `libpearlx_skel.so` and the SDK's `libworker_pool.so` from
   `hexagon_Release_toolv87_v66\ship`.

2. **The miner.** It needs no SDK.
   - On the phone in Termux: `./build.sh --backend cpu,hexagon`.
   - Cross-compiled with the Android NDK, for example the one in the Hexagon
     SDK. Push `libomp.so` from the NDK alongside the miner.

     ```powershell
     cmake -S . -B build/android-hexagon -G Ninja `
       -DCMAKE_TOOLCHAIN_FILE=<ndk>/build/cmake/android.toolchain.cmake `
       -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 -DANDROID_STL=c++_static `
       -DCMAKE_BUILD_TYPE=Release -DCP_ENABLE_CPU=ON -DCP_ENABLE_HEXAGON=ON
     cmake --build build/android-hexagon
     ```

     A cross-build without an Android Rust target falls back to the proof stub,
     which can't submit shares. Build in Termux, which has Rust, to get proofs.

## Run

The DSP library loads in an unsigned protection domain, so root is not needed.
FastRPC finds the skel through `ADSP_LIBRARY_PATH`:

```sh
export ADSP_LIBRARY_PATH=/path/to/dir/with/libpearlx_skel.so   # and libworker_pool.so
# the default 128x128 size needs ~1 GiB; --m 32 --n 32 needs ~300 MiB
./cppminer --backend hexagon --m 32 --n 32 --pool stratum+tcp://HOST:PORT --wallet prl1... --worker phone
./cppminer --backend hexagon --m 32 --n 32 --mock --mock-diff 40   # offline: first share + verify
```

| Option | Hexagon meaning |
|---|---|
| `--m N --n N` | Matrix size in units of 1024 (default 128, as for every backend). Memory is about m·k + n·k bytes: signal A (shared with the DSP) plus packed B in DSP memory, plus ~50 MiB of buffers. The default 131072 needs ~1 GiB; the measurements here use `--m 32 --n 32` (~300 MiB) |
| `--row-period-batch N` | 128-row macros per DSP launch (default 32 = 4096 rows). Rows set how far each VTCM slice of B is amortized. Per row panel: 2048 rows ~770 GOPS, 4096 ~792 |
| `--batch-size N` / `--col-period-batch N` | 128-column macros per DSP launch (default 32 = 4096 columns). Columns barely matter: a full-width launch is only 0.3% faster than 4096 columns |

Testing was done from `adb shell` (`/data/local/tmp`). Running from Termux has
not been tested yet. If `cp_pearlx_open` fails, check that the app may open
`/dev/fastrpc-cdsp` and that `ADSP_LIBRARY_PATH` is set.

Only Hexagon v66 has been tested. `vrmpyz` (the Z buffer) is a v66 HVX
instruction, so other DSP versions need checking before use.
