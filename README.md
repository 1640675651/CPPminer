# CPPminer

**C**ross-**P**latform **P**earl **miner** written in C++. It started as a Pearl miner and is now multi-algo, mining **Pearl** and **Quantus**. It runs on almost any hardware: x86 and ARM CPUs, NVIDIA, AMD and Intel GPUs, Intel Core Ultra NPUs and Snapdragon DSPs. Tested on LuckyPool, HeroMiners and Kryptex.

## Supported hardware

Pick the backend for your hardware with `--backend`:

| Hardware | `--backend` | Pearl | Quantus |
|---------------|-------------|:-----:|:-------:|
| Any x86 or ARM CPU | `cpu` (default) | ✓ | ✓ |
| NVIDIA GPU (now optimized for GTX 10-series) | `cuda` | ✓ | ✓ |
| Intel Arc, Iris Xe, UHD 7xx and Core Ultra graphics | `onednn` | ✓ | |
| AMD and other GPUs, older Intel graphics (UHD 6xx and earlier) | `opencl` | ✓ | ✓ |
| Any GPU with Vulkan, DirectX 12 or Metal | `wgpu` | ✓ | ✓ |
| Intel Core Ultra NPU | `intel_npu` | ✓ | |
| Snapdragon phone or board (Hexagon DSP) | `hexagon` | ✓ | |

`--algo pearl` is the default; add `--algo quantus` for Quantus. A build only contains the backends it was built with; asking for a missing one stops the miner with an error.

For measured hashrates on each kind of hardware, see [Performance](#performance).

## Quick start

### Windows

1. Download the latest release zip and extract it.
2. Open `start_pearl.bat` in a text editor and set `WALLET` to your Pearl wallet address (and `WORKER` to a name for this machine, if you like).
3. Double-click `start_pearl.bat`.

That mines Pearl on the CPU at LuckyPool. For other hardware, run `cppminer.exe` from a terminal in the same folder, or change the `--backend` line in the bat file:

```bat
:: NVIDIA GPU
cppminer.exe --backend cuda --devices 0 --pool stratum+tcp://pearl-eu.lproute.com:3360 --wallet YOUR_WALLET --worker rig01

:: Intel Arc GPU
cppminer.exe --backend onednn --devices 0 --pool stratum+tcp://pearl-eu.lproute.com:3360 --wallet YOUR_WALLET --worker rig01

:: AMD GPU
cppminer.exe --backend opencl --devices 0 --pool stratum+tcp://pearl-eu.lproute.com:3360 --wallet YOUR_WALLET --worker rig01

:: Quantus on the CPU
cppminer.exe --algo quantus --backend cpu --pool stratum+tcp://eu.lproute.com:5660 --wallet YOUR_QUANTUS_WALLET --worker rig01
```

The Intel NPU backend needs an extra download; see [Intel NPU](#intel-npu).

### Linux, macOS, Android

Build from source (see [docs/building.md](docs/building.md)), e.g. for the CPU and OpenCL:

```sh
./build.sh --backend cpu,opencl
./cppminer --backend cpu --wallet YOUR_WALLET --worker rig01
```

On Android, build in Termux ([notes](docs/termux_note.md)). The Snapdragon DSP backend has its own guide: [src/pearl/hexagon/README.md](src/pearl/hexagon/README.md).

### Find your device and test it

```sh
cppminer --backend opencl --list-devices      # device numbers for --devices
cppminer --backend opencl --devices 0 --mock  # mine offline until one share, check it, exit
```

`--mock` needs no pool or wallet. It confirms the miner works on your hardware before you point it at a pool; on slow devices it can take a few minutes.

## Common options

| Option | What it does |
|--------|--------------|
| `--backend NAME` | Hardware to mine on: `cpu`, `cuda`, `opencl`, `onednn`, `wgpu`, `intel_npu`, `hexagon` |
| `--algo NAME` | `pearl` (default) or `quantus` |
| `--devices N` | Which GPU to use (from `--list-devices`) |
| `--list-devices` | Show the devices for the chosen backend and exit |
| `--pool URL` | Pool address, `stratum+tcp://host:port` |
| `--wallet ADDRESS` | Your wallet address |
| `--worker NAME` | Name for this machine on the pool (default `rig01`) |
| `--pool-pass TEXT` | Pool password (default `x`); Kryptex reads `d=NUMBER` from it |
| `--threads N` | CPU: number of threads (default: all). Use fewer to keep the computer responsive |
| `--simd NAME` | CPU: force a kernel (`avx2`, `avxvnni`, ...). The default picks the best one for your CPU |
| `--api-port N` | Serve hashrate and share stats over HTTP on port N ([details](docs/api.md)) |
| `--api-bind ADDR` | Address for the stats API (default `127.0.0.1`; `0.0.0.0` to reach it from other machines) |
| `--mock` | Offline test: mine until one share, verify it and exit |

Options take `--option VALUE` or `--option=VALUE`. A mistyped option stops the miner before it starts. All options, including GPU tuning: [docs/options.md](docs/options.md).

## Multiple GPUs

Currently one process mines on one GPU. For several GPUs, start one miner per GPU with its own `--devices`, `--worker` (and `--api-port`, if you use the API):

```bat
start cppminer.exe --backend cuda --devices 0 --pool stratum+tcp://pearl-eu.lproute.com:3360 --wallet YOUR_WALLET --worker rig01-gpu0
start cppminer.exe --backend cuda --devices 1 --pool stratum+tcp://pearl-eu.lproute.com:3360 --wallet YOUR_WALLET --worker rig01-gpu1
```

## Intel NPU

Releases with the Intel NPU backend come as two zips: the main one, and `..._intel_npu_runtime.zip` with the NPU compiler and kernels (about 30 MB). Extract both, then copy everything inside the runtime zip's top folder next to `cppminer.exe`, keeping the `kernels\intel_npu` subfolder:

```text
cppminer_<version>_<suffix>\
  cppminer.exe
  openvino.dll
  openvino_intel_npu_plugin.dll            <- from the runtime zip
  openvino_intel_npu_compiler.dll          <- from the runtime zip
  openvino_intel_npu_compiler_loader.dll   <- from the runtime zip
  ...
  kernels\intel_npu\*.elf                  <- from the runtime zip
```

The NPU also needs Intel's NPU driver. The first start compiles the NPU program, which takes a while; it is cached in `npu_cache\` for later starts. Details: [src/pearl/intel_npu/README.md](src/pearl/intel_npu/README.md).

## Performance

Pearl hashrate at the default matrix size. Following pools' convention, one multiply-add (MAC) is counted as one hash (H), TH/s here means 10¹² multiply-adds per second ([how it is counted](docs/hashrate_calculation.md)). Your results depend on clocks, cooling, power limits and drivers. Numbers below are tested on stock settings unless noted; power from HWiNFO64.

### NVIDIA GPU (`cuda`)

| Device | ISA | Hashrate | Power (W) |
|--------|-----|----------|-----------|
| GTX 1070 | DP4A | 9.1 TH/s | 150 |

### AMD GPU (`opencl`)

| Device | ISA | Hashrate | Power (W) | Note |
|--------|-----|----------|-----------|------|
| Radeon Pro 5500M | DP4A | 5.5 TH/s | 50 | --ocl-double-buff on |

### Intel GPU (`onednn`)

| Device | ISA | Hashrate | Power (W) |
|--------|-----|----------|-----------|
| UHD 770 | DP4A | 1.4 TH/s | 13 |
| Xe-LPG 64EU (Core Ultra 9 275HX) | DP4A | 3.5 TH/s | 14 |
| ARC B580 | DPAS | 104 TH/s | 200 |
| ARC A770 | DPAS | 124 TH/s | 205 |

### Other GPU (`opencl`)

| Device | ISA | Tile size | Hashrate |
|--------|-----|-----------|----------|
| Intel HD graphics 10EU (Haswell) | Scalar | 4x8 | 25 GH/s |
| Intel UHD 630 | Scalar | 4x8 | 140 GH/s |
| Mali-G57 MC2 | DP4A | 4x8 | 115 GH/s |

### Hexagon DSP (`hexagon`)

| Device | Hashrate |
|--------|----------|
| Hexagon 686 (Snapdragon 480, v66, 2 HVX, 940MHz) | 400 GH/s |

### Intel NPU (`intel_npu`)

| Device | Hashrate |
|--------|----------|
| NPU3720 (Core Ultra 9 275HX, 2 compute tiles) | 800 GH/s |

### CPU (`cpu`)

| Device | Clock Speed (GHz) | ISA | Hashrate | Power (W) | Note |
|--------|-------------|-----|----------|-----------|------|
| Celeron G1840 | 2.8 | SSSE3 | 35 GH/s | | |
| Core i9 9980HK | 2.4 | AVX2 | 490 GH/s | 37 | no turbo |
| Core i5 12490F | 4.0 | AVX-VNNI | 1.1 TH/s | | |
| Core i9 12900K | 8P @ 3.2 + 8E @ 2.4 | AVX-VNNI | 1.7 TH/s | 82 | no turbo |
| Core i9 12900K | 8P @ 4.9 + 8E @ 3.7 | AVX-VNNI | 2.5 TH/s | 250 | |
| Core Ultra 9 275HX | 8P @ 2.7 + 16E @ 2.1 | AVX-VNNI-INT8 | 2.3 TH/s | 52 | no turbo |
| Core Ultra 9 275HX | 8P @ 4.3 + 16E @ 3.8 | AVX-VNNI-INT8 | 3.5 TH/s | 125 | |
| BCM2837 | 1.2 | NEON | 9 GH/s | | rpi3b, m=n=32 |
| Dimensity 6300 | 2x A76 @ 2.6 + 6x A55 @ 2.0 | NEON DotProd | 160 GH/s | |

## Troubleshooting

- **"unknown option" or the backend is rejected:** your build does not include that backend. Use a release or build that has it (`--backend` list above).
- **`--list-devices` shows nothing:** install or update the GPU driver. OpenCL and the Intel GPU backend come with the vendor driver.
- **No hashrate for the first seconds:** each new pool job prepares its matrices first; the hashrate appears once the first scan starts.
- **Hashrate lower than the table:** close other heavy programs, check that the device is not hitting a power or temperature limit (laptops: plug in and use a performance power plan), and try `--mock` to compare.
- **The antivirus removes `cppminer.exe`:** antivirus software often flags miners. Add an exception for the folder, or build the miner yourself from this source.
- **Something else:** open an issue with the miner's console output.

## Dev fee

CPPminer keeps **1%** of mining for its development. About one matrix scan in every 100 (for Quantus, 1% of the mining time) is mined to the developer's wallet, on LuckyPool or Kryptex, falling back to your own pool if those cannot be reached. Your pool password is never sent to the fee pools. Details: [docs/fee_pools.txt](docs/fee_pools.txt) and [docs/structure.md](docs/structure.md#dev-fee-schedule).

## Building from source

Windows: `powershell -ExecutionPolicy Bypass -File build.ps1 -Backend Cpu,OpenCl` (pick your backends). Linux and macOS: `./build.sh --backend cpu,opencl`. Requirements, all backends and build options: [docs/building.md](docs/building.md). How the code is organized: [docs/structure.md](docs/structure.md).

## Limitations
Close sourced miners already extensively optimized Nvidia RTX20-50 and AMD RDNA3/4 GPUs. Competing with them is not the first priority of CPPminer today. The current CUDA backend is optimized for GTX10 (Pascal) GPUs only, which is unsupported in most miners. Support for those GPUs will be introduced in future versions, likely after the FP8 fork.

## License

MIT, see [LICENSE](LICENSE). Changes per release: [CHANGELOG.md](CHANGELOG.md).
