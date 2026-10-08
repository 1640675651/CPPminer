# Intel Arc: opt-in XMX/DPAS OpenCL backend

`--ocl-dot dpas` runs the OpenCL GEMM on Intel XMX units through
`cl_intel_subgroup_matrix_multiply_accumulate`:

```sh
./cppminer --backend opencl --devices 0 --ocl-dot dpas --verify \
  --pool stratum+tcp://POOL:PORT --wallet WALLET --worker arc-a380
```

The Intel `auto` policy is unchanged (KHR integer dot). DPAS requires the Intel
subgroup matrix multiply extension and an 8x16 hash tile: an automatic tile
becomes 8x16, an incompatible explicit tile or LDS staging is rejected, and there
is no silent scalar fallback. The kernel uses
`intel_sub_group_i8_i8_matrix_mad_k32` for exact integer matrix multiplication,
keeps the cumulative rank-128 milestones and feeds the existing jackpot/proof
pipeline. `--list-devices` shows the DPAS capability of Intel devices.

Every DPAS kernel initialization runs a lane-layout self-test before the kernel
is adopted. The device's minimum sub-group size is required: 8 on Xe-HPG (tested
on an A380). SG16 code (Xe2, Xe-HPC) is included but not hardware-validated; its
selected layout must also pass the self-test. Environment integers are parsed
strictly and tile dimensions are bounded before multiplication.

| Optional variable | Purpose |
|---|---|
| `CP_OCL_DPAS_TM`, `CP_OCL_DPAS_TN` | Hash tiles per sub-group; positive powers of two, product at most the sub-group size |
| `CP_OCL_DPAS_SG` | Explicit 8 or 16; must agree with the device minimum |
| `CP_OCL_DPAS_AK` | SG16 A packing variant; 0 is the default |
| `CP_OCL_DPAS_FORCE=1` | Test override for a device reporting no hardware DPAS units |
| `CP_OCL_DPAS_EMULATE=8\|16` | Functional model of the DPAS path on AMD; does not measure Intel hardware |

## Validation on Arc A380

Arc A380 (8086:56a5), Ubuntu kernel 6.8 with i915, PCIe 2.0 x1,
`intel-opencl-icd` 23.43.27642.40, IGC 1.0.15468.25.

KHR and DPAS `--align-test` passed. The SG8 builtin/layout self-test passed,
and all 131072 cumulative GEMM milestone words agreed with the CPU reference.
Every tile variant below built and verified a full mock share through the Rust
proof implementation at difficulty 40. A three-attempt 131072² scan ran at
3.195 TMAC/s per complete attempt with no allocation failure on the 6 GiB card.

Full zero-target scans at 32768², five attempts per configuration with the first
discarded:

| Backend | Full attempt, TMAC/s | Scan, TMAC/s | Register spill, bytes/work-item |
|---|---:|---:|---:|
| KHR integer dot, 4x8 | 0.886 | 0.931 | not queried |
| DPAS 4x1 hash tiles/sub-group (default) | 2.773 | 3.371 | 512 |
| DPAS 2x1 | 2.556 | 3.056 | 0 |
| DPAS 1x1 | 2.152 | 2.480 | 0 |

The default 4x1 setting was fastest on this driver despite the register spills,
about 3.1x KHR's full-attempt rate. These are measured complete scans, not
extrapolated early-hit rates, and do not establish performance on other Intel
models or drivers. On Arc, the `onednn` backend with XeHPG systolic kernels is
considerably faster still.
