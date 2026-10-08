# Independent CUDA attempts on multiple GPUs

With `--devices 0,1,...`, GPU0 used to prepare signal A and then copy its noisy A
and jackpot key to every other device. Each device scanned the same tile
coordinates of the same matrix, so additional GPUs repeated existing work, and
proof readback always used GPU0.

Each CUDA device now prepares a fresh signal A, derives its own noise/jackpot key,
and keeps its own Merkle commitment and sub-roots. Noisy B stays shared because
the zero-B construction makes it constant for a job; its one-time transfer uses
`cudaMemcpyPeer` with explicit source and destination devices. Scanning the same
tile coordinates on different A matrices is independent work. The winning device
supplies both the signal and the witness readback. Aggregate scanned tiles count
every independent attempt and use 64-bit totals.

Multi-GPU preparation runs serially before the concurrent device scans.
CPU-prepared/explicit matrix inputs keep their single identity and distribute
disjoint batches over the devices, counted once; that path synchronizes each
batch. Duplicate CUDA device IDs are rejected. Job cancellation is checked
between both row and column batches.

## Validation

The change was validated on a host with one CUDA GPU (CMP 50HX), so the harness
uses two independent logical contexts with separate allocations on that GPU; it
does **not** validate inter-device transfer or simultaneous execution on two
physical NVIDIA GPUs. CUTLASS fused, period GEMM and the direct kernel passed,
with these checks across two jobs:

- Distinct A commitments and jackpot keys on every attempt.
- Aggregate work accounting for independent and shared-input scans.
- An actual GPU1 jackpot with no GPU0 hit, followed by winner signal/witness fetch.
- Winner root against CPU BLAKE3, byte-identical witness and full-host proofs,
  and proof verification against the actual jackpot target.
- Rejection of a witness paired with another device's commitment.
- Both CPU-upload APIs, cancellation and resource cleanup.

A 40-attempt ABBA comparison at 131072² on the single GPU measured
63.171 TMAC/s before and 63.132 TMAC/s after (-0.06%): no material single-GPU
change. This is not a multi-GPU scaling measurement.

## Reproducing the ownership test

Build a CUDA miner with the real Rust proof FFI using CMake's Unix Makefiles
generator, then link the harness against its objects:

```sh
cmake --build build -j4
python3 scripts/build_cuda_multigpu_test.py --build build
build/cp_cuda_multigpu_test             # CUTLASS fused
build/cp_cuda_multigpu_test --period    # period GEMM
build/cp_cuda_multigpu_test --scalar    # direct kernel
# On a host with two CUDA GPUs, use distinct physical devices 0 and 1:
build/cp_cuda_multigpu_test --physical
```

The single-device logical setup exists only inside the test translation unit;
the miner has no duplicate-device or injected-hit testing switch.
