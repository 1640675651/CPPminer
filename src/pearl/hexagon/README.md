# Pearl Hexagon backend (`--backend hexagon`)

Pearl mining on the compute DSP (cDSP) of Snapdragon phones. The DSP's HVX
vector units generate the noise, pack the matrices, and run the GEMM and the
milestone XOR. The host CPU keeps signal A and its hash, derives the seeds, and
runs the BLAKE3 jackpot. Tested on a Snapdragon 480 (SM4350, Hexagon v66, 2 HVX
contexts).

| Device | Backend | Hashrate (m = n = 32768) |
|---|---|---|
| Snapdragon 480 cDSP | hexagon | ~402 GMAC/s while scanning, ~401 GMAC/s overall |
| Snapdragon 480 CPU (2x A76 + 6x A55) | cpu (NEON DotProd) | ~130 GMAC/s |

The overall rate includes about 0.03 s of host work per 10.98 s attempt, while
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
| Copying a 128 KB slice of B into VTCM per 128-column tile and K block, amortized over the launch's 1024 row tiles. An `l2fetch` of the slice before the copy cut this from ~1.6%; v66 has no user DMA (`dmstart` is v68+) to overlap it with compute | ~1.1% |
| Other fixed costs per column tile and K block (kernel entry, the first row tile's A) | ~0.3% |
| Generating and packing each row panel's noisy A (~9.7 ms per 8 launches) | ~0.7% |
| Writing each launch's 8 MiB of records out of the DSP's cache before announcing it (from the rise in DSP busy time) | ~0.2% |
| The DSP idle: starting the `scan_run` call, and the host checking the last launch after the DSP has finished | ~0.25% |

With one FastRPC call per launch, the call and the gap between launches cost
~1.1% (DSP busy 98.9% of the scan); the DSP now runs the launches itself (see
[How it works](#how-it-works)). The tile-XOR records live in an **uncached**
shared buffer, which the host reads while the DSP is still running. The host
jackpot copies each 256-byte record in with wide loads, which keeps uncached
reads cheap: about 0.9 s of CPU per attempt, overlapped with the DSP.

The DSP library also has a `scan` method that runs the jackpot on the DSP (HVX
BLAKE3, 32 hash tiles per vector) and returns only the first hit. Its results
match the host jackpot exactly (checked by `pearlx_test`), but it adds ~2.9 ms of
DSP time per launch, so the miner keeps the jackpot on the host.

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
| Noisy A of a row panel generated from its signal rows, in packed layout | cDSP | per panel, before its first launch |
| GEMM + milestone XOR of the panel against one block of columns | cDSP | per launch |
| BLAKE3 jackpot over the launch's tile XORs | host, OpenMP | per launch |

The DSP runs the launches itself (`pearlx scan_run`): one call covers whole row
panels, with at most 128 MiB of signal A (the call maps those rows into the
DSP), so an attempt at `--m 32` is one call and at the default size four. A
dedicated host thread makes the calls back to back. Host and DSP follow each
other through a 256-byte uncached control block:

- **DSP → host:** after a launch's records are written out to memory, the DSP
  publishes the number of launches finished.
- **Host → DSP:** the mining thread polls that count, runs the jackpot over the
  new launch, and publishes the number of launches checked. Launch s writes to
  record slot s % 2 and starts once launch s - 2 is checked, so the host has a
  whole launch of slack; the DSP sleeps 100 µs at a time if it gets ahead.
- **Stop:** on a hit or a new job the host sets a stop word. The DSP checks it
  between launches and at every 128-column tile inside one, so it returns within
  ~12 ms (`pearlx_test -L 1` measures this).
- **Epoch:** the host changes an epoch word every attempt and ignores counts the
  DSP has not tagged with it, so nothing carries over from the previous attempt.

Each side writes only its own 128-byte half of the block, so neither side's
cache write-back can overwrite the other's words: the DSP flushes its half after
writing it and invalidates the host's half before reading it. The host's start
of the next launch no longer depends on its thread scheduling, and the DSP is
busy ~99.75% of the scan, against 98.9% with one call per launch.

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

## Memory

All of it is phone RAM: the cDSP has no memory of its own, and its heap comes
from system memory. Signal A and noisy B dominate; the rest depends only on the
launch size. With r rows and c columns per launch (`--row-period-batch` x 128,
`--batch-size` x 128) and k = 4096:

| Buffer | Size | Default (m = n = 131072) | `--m 32 --n 32` | Held in | Kept |
|---|---|---|---|---|---|
| Signal A | m·k | 512 MiB | 128 MiB | rpcmem, shared by host and DSP | whole run |
| Signal A's hash tree (every chunk value and complete parent) | ~m·k / 16 | 32 MiB | 8 MiB | host heap | whole run |
| Noisy packed B | n·k | 512 MiB | 128 MiB | DSP heap | one job; freed before the next job's B |
| Panel buffer: packed A of one row panel plus partial sums | (r/4 + 2) x 28 KiB | 28 MiB | 28 MiB | DSP heap | reused by every panel |
| Tile-XOR records, 2 buffers | r·c in total | 16 MiB | 16 MiB | rpcmem, uncached | whole run |
| **Total** | | **~1.07 GiB** | **~310 MiB** | | |

The panel buffer holds, for each 4-row tile, 16 KiB of packed A and 12 KiB of
partial sums. The kernel walks k in 4 blocks of 1024, the size of the B slice
that fits in VTCM, so after each of the first 3 blocks a tile's int32
accumulators (4 x 128, 2 KiB) are stored next to its A, one slot per DSP
thread. Each record buffer has 256 bytes per 4 x 128 register tile.

Job setup briefly needs another ~48 MiB at the default size, freed before the
scan: the chunk hashes from the DSP (max(m, n)·k / 32, rpcmem) and the zero B's
temporary hash tree (n·k / 16, host). VTCM (128 KB per thread) is on-chip and
not counted.

## Files

| Path | Contents |
|---|---|
| `cp_pearl_hexagon_worker.cpp` | Worker: job and attempt prep (seeds, signal A, its incremental hash), the launch pipeline, jackpot, hit recompute |
| `cp_pearlx_client.c` | FastRPC client: loads `libcdsprpc.so` at run time and marshals the `pearlx` calls by hand. No Hexagon SDK headers are needed |
| `dsp/inc/pearlx.idl` | DSP interface: `scan_run` (an attempt's launches, run by the DSP, with the control block), `set_b_gen` (noisy B generated on the DSP), `chunk_cvs` (matrix hash leaves), `info`; `set_a_gen` and `gemm_xor` (one panel / one launch per call), `set_b`, `set_a` (pack host-built matrices) and `scan` (jackpot on the DSP) are used only by the self-test |
| `dsp/src/pearlx_imp.c` | DSP side: noise generation, packing, VTCM, threads over both HVX contexts, the DSP jackpot |
| `dsp/src/pearlx.S`, `dsp/src/pearlx_xor.inc` | HVX kernel (Case 1.1 fold and reduction) |
| `dsp/src/pearlx_test.c` | Android self-test against CPU references, plus timing: GEMM + XOR, the DSP jackpot (`-b` hit rate), the noise generators (`-G 1`), the chunk hashes (`-H 1`), and `scan_run` against one call per launch with records read during the run and a mid-run stop (`-L 1`; `-L 2` with a host slower than the DSP), with the miner's `cp_noise.c` linked in as the reference |
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
     SDK. It links the NDK's static OpenMP, so the miner needs no `libomp.so`
     or `LD_LIBRARY_PATH` on the phone.

     ```powershell
     cmake -S . -B build/android-hexagon -G Ninja `
       -DCMAKE_TOOLCHAIN_FILE=<ndk>/build/cmake/android.toolchain.cmake `
       -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 -DANDROID_STL=c++_static `
       -DCMAKE_BUILD_TYPE=Release -DCP_ENABLE_CPU=ON -DCP_ENABLE_HEXAGON=ON
     cmake --build build/android-hexagon
     ```

     The proof library (`rust/cp-proof-ffi`) is cross-built by CMake for
     `aarch64-linux-android` when cargo is found and has that target's standard
     library: `rustup target add aarch64-linux-android`, or with conda's Rust,
     `conda install -c conda-forge rust-std-aarch64-linux-android=<rustc version>`.
     Pass `-DCARGO_EXECUTABLE=<path to cargo>` if cargo is not on `PATH`. Without
     it the build falls back to the proof stub, which can't submit shares. On the
     phone, a `--m 32 --n 32` share's proof builds in ~0.14 s and verifies in
     ~0.03 s.

## Run

The DSP library loads in an unsigned protection domain, so root is not needed.
FastRPC looks for `libpearlx_skel.so` and `libworker_pool.so` in the current
directory, then in the system DSP directories. Run the miner from the directory
that holds them, or list that directory in `ADSP_LIBRARY_PATH`. No other
environment is needed.

```sh
cd /path/to/dir/with/libpearlx_skel.so                    # or: export ADSP_LIBRARY_PATH=<that dir>
# the default 128x128 size needs ~1.07 GiB; --m 32 --n 32 needs ~310 MiB
./cppminer --backend hexagon --m 32 --n 32 --pool stratum+tcp://HOST:PORT --wallet prl1... --worker phone
./cppminer --backend hexagon --m 32 --n 32 --mock --mock-diff 40   # offline: first share + verify
```

| Option | Hexagon meaning |
|---|---|
| `--m N --n N` | Matrix size in units of 1024 (default 128, as for every backend). Memory is about m·k + n·k bytes plus 50–80 MiB (see [Memory](#memory)): ~1.07 GiB at the default 131072; the measurements here use `--m 32 --n 32` (~310 MiB) |
| `--row-period-batch N` | 128-row macros per DSP launch (default 32 = 4096 rows). Rows set how far each VTCM slice of B and each FastRPC call are amortized. `pearlx_test` per call: 4096 rows 802 GOPS, 8192 rows 813, 16384 rows 818; each doubling also doubles the panel buffer and the record buffers (see [Memory](#memory)) |
| `--batch-size N` / `--col-period-batch N` | 128-column macros per DSP launch (default 32 = 4096 columns). Columns barely matter: a full-width launch is only 0.3% faster than 4096 columns |

Testing was done from `adb shell` (`/data/local/tmp`). Running from Termux has
not been tested yet. If `cp_pearlx_open` fails, check that the app may open
`/dev/fastrpc-cdsp`, and that the two DSP libraries are in the current directory
or in `ADSP_LIBRARY_PATH` (error `0x80000406` when they are not found).

Only Hexagon v66 has been tested. `vrmpyz` (the Z buffer) is a v66 HVX
instruction, so other DSP versions need checking before use.
