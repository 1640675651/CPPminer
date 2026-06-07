# Build cpminer.exe (CUDA) on Windows.
# Requires: NVIDIA CUDA Toolkit 12.x, MSVC (VS Build Tools), nvidia-smi, Rust (cargo on PATH).
#
# Usage:
#   conda activate pearl
#   powershell -ExecutionPolicy Bypass -File build.ps1
#   powershell -ExecutionPolicy Bypass -File build.ps1 -CudaArch 61

param(
    [string]$CudaArch = "",
    [string]$CudaRoot = ""
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$BuildDir = Join-Path $Root "build\win"
$B3Dir = Join-Path $BuildDir "b3"
$OutExe = Join-Path $Root "cpminer.exe"
$script:VcvarsBat = $null
$script:ClExe = $null

function Get-SanitizedPathForNvcc {
    param([string]$PathValue)
    # Drop conda envs and mingw so nvcc/host compile use MSVC, not pearl's toolchain.
    ($PathValue -split ';' | Where-Object {
        $_ -and $_ -notmatch '\\miniconda3\\|\\anaconda3\\|\\envs\\|mingw-w64|\\.cargo\\bin'
    }) -join ';'
}

function Clear-CondaToolchainOverrides {
    # Conda envs (e.g. pearl) often set CC/CXX to non-MSVC tools, which breaks nvcc's
    # Visual Studio setup even when vcvars64.bat is present.
    foreach ($var in @('CC', 'CXX', 'CFLAGS', 'CXXFLAGS', 'LDFLAGS', 'CPPFLAGS')) {
        if (Test-Path "env:$var") {
            Remove-Item "env:$var" -ErrorAction SilentlyContinue
        }
    }
    if ($env:CONDA_PREFIX) {
        Write-Host "=== Conda active ($env:CONDA_PREFIX); cleared CC/CXX for nvcc ==="
    }
}

function Initialize-MSVC {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) {
        throw "Visual Studio Build Tools not found (vswhere missing)."
    }
    $vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $vsPath) { throw "MSVC toolchain not found." }
    $vcvars = Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat"
    if (-not (Test-Path $vcvars)) { throw "vcvars64.bat not found under $vsPath" }
    $script:VcvarsBat = $vcvars
    Write-Host "=== MSVC: $vsPath ==="
    cmd /c "`"$vcvars`" >nul 2>&1 && set" | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path "env:$($Matches[1])" -Value $Matches[2] }
    }
    if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
        throw "cl.exe not on PATH after vcvars64"
    }
    $script:ClExe = (Get-Command cl.exe).Source
    $env:CUDAHOSTCXX = $script:ClExe
    $env:CUDAHOSTC = $script:ClExe
    Write-Host "=== cl.exe: $($script:ClExe) ==="
}

function Find-CudaRoot {
    if ($CudaRoot -and (Test-Path (Join-Path $CudaRoot "bin\nvcc.exe"))) { return $CudaRoot }
    $base = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    if (Test-Path $base) {
        $latest = Get-ChildItem $base -Directory | Sort-Object Name -Descending | Select-Object -First 1
        if ($latest) { return $latest.FullName }
    }
    throw "CUDA Toolkit not found."
}

function Get-GpuArch {
    if ($CudaArch) { return $CudaArch.Trim() }
    try {
        $cap = & nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>$null | Select-Object -First 1
        if ($cap -match '(\d+)\.(\d+)') {
            return "$($Matches[1])$($Matches[2])"
        }
    } catch {}
    return "75"
}

function Ensure-Blake3 {
    $b3Src = Join-Path $Root "third_party\blake3"
    if (-not (Test-Path (Join-Path $b3Src "blake3.c"))) {
        Write-Host "=== Fetching BLAKE3 1.5.4 ==="
        New-Item -ItemType Directory -Force -Path $b3Src | Out-Null
        $base = "https://raw.githubusercontent.com/BLAKE3-team/BLAKE3/1.5.4/c"
        foreach ($f in @(
            "blake3.c", "blake3.h", "blake3_dispatch.c", "blake3_portable.c", "blake3_impl.h",
            "blake3_sse2.c", "blake3_sse41.c", "blake3_avx2.c", "blake3_avx512.c"
        )) {
            Invoke-WebRequest -Uri "$base/$f" -OutFile (Join-Path $b3Src $f) -UseBasicParsing
        }
    }
    New-Item -ItemType Directory -Force -Path $B3Dir | Out-Null
    Copy-Item (Join-Path $b3Src "*") $B3Dir -Force
}

function Invoke-Nvcc {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$NvccArgs)
    if (-not $script:ClExe) { throw "MSVC cl.exe not configured (Initialize-MSVC missing?)" }
    $ccbin = @("-ccbin", $script:ClExe)
    $allArgs = $ccbin + $NvccArgs
    $quoted = ($allArgs | ForEach-Object {
        if ($_ -match '[\s"]') { '"' + ($_ -replace '"', '""') + '"' } else { $_ }
    }) -join ' '
    $savedPath = $env:PATH
    $env:PATH = Get-SanitizedPathForNvcc -PathValue $env:PATH
    try {
        if ($script:VcvarsBat) {
            cmd /c "`"$script:VcvarsBat`" >nul 2>&1 && set CC=&& set CXX=&& `"$Nvcc`" $quoted"
        } else {
            cmd /c "set CC=&& set CXX=&& `"$Nvcc`" $quoted"
        }
    } finally {
        $env:PATH = $savedPath
    }
    if ($LASTEXITCODE -ne 0) { throw "nvcc failed: $($NvccArgs -join ' ')" }
}

Clear-CondaToolchainOverrides
Initialize-MSVC
$CudaRoot = Find-CudaRoot
$Nvcc = Join-Path $CudaRoot "bin\nvcc.exe"
$Arch = Get-GpuArch
$NvccArch = "sm_$Arch"
$Inc = @("-I$(Join-Path $Root 'include')", "-I$(Join-Path $Root 'src')", "-I$B3Dir")
$HostOpenMP = @("-Xcompiler", "/openmp")
$Nosimd = @("-DBLAKE3_NO_AVX512", "-DBLAKE3_NO_AVX2", "-DBLAKE3_NO_SSE41", "-DBLAKE3_NO_SSE2")

Write-Host "=== CUDA: $CudaRoot  Arch: $NvccArch ==="
New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
Ensure-Blake3

Write-Host "=== Building cp-proof-ffi (Rust) ==="
$RustDir = Join-Path $Root "rust\cp-proof-ffi"
$PearlBlake3 = Join-Path $Root "third_party\pearl-blake3\Cargo.toml"
if (-not (Test-Path $PearlBlake3)) {
    throw "Vendored pearl-blake3 missing at third_party/pearl-blake3"
}
Push-Location $RustDir
try {
    if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
        throw "Rust cargo not found. conda activate pearl (or install from https://rustup.rs)"
    }
    cargo build --release
    if ($LASTEXITCODE -ne 0) { throw "cargo build failed" }
} finally {
    Pop-Location
}
$RustLib = Join-Path $RustDir "target\release\cp_proof_ffi.lib"
if (-not (Test-Path $RustLib)) { throw "Missing $RustLib" }
Write-Host "=== Rust lib: $RustLib ==="

foreach ($unit in @(
    @{ src = "blake3.c"; extra = $Nosimd },
    @{ src = "blake3_dispatch.c"; extra = $Nosimd },
    @{ src = "blake3_portable.c"; extra = @() },
    @{ src = "cp_noise.c"; extra = @(); path = (Join-Path $Root "src\cp_noise.c") }
)) {
    $srcPath = if ($unit.path) { $unit.path } else { Join-Path $B3Dir $unit.src }
    $obj = Join-Path $BuildDir ($unit.src -replace '\.c$', '.obj')
    $flags = @("-c", $srcPath, "-o", $obj, "-O2") + $Inc + $unit.extra + $HostOpenMP
    Invoke-Nvcc @flags
}

foreach ($unit in @("cp_util.cpp", "cp_pool.cpp", "cp_job_ctrl.cpp", "cp_mine.cpp", "cp_state.cpp", "main.cpp")) {
    $srcPath = Join-Path $Root "src\$unit"
    $obj = Join-Path $BuildDir ($unit -replace '\.cpp$', '.obj')
    $flags = @("-c", $srcPath, "-o", $obj, "-O2") + $Inc + $HostOpenMP
    Invoke-Nvcc @flags
}

$gpuFlags = @("-arch=$NvccArch", "-O3") + $HostOpenMP + $Inc + @(
    "-c", (Join-Path $Root "src\cp_gpu.cu"),
    "-o", (Join-Path $BuildDir "cp_gpu.obj")
)
Invoke-Nvcc @gpuFlags

Write-Host "=== Linking $OutExe ==="
$RustExtra = @(
    $RustLib,
    "cublas.lib",
    "userenv.lib", "ws2_32.lib", "bcrypt.lib", "ntdll.lib", "advapi32.lib"
)
$linkArgs = @(
    "-arch=$NvccArch", "-O3",
    (Join-Path $BuildDir "main.obj"),
    (Join-Path $BuildDir "cp_util.obj"),
    (Join-Path $BuildDir "cp_pool.obj"),
    (Join-Path $BuildDir "cp_job_ctrl.obj"),
    (Join-Path $BuildDir "cp_mine.obj"),
    (Join-Path $BuildDir "cp_state.obj"),
    (Join-Path $BuildDir "cp_gpu.obj"),
    (Join-Path $BuildDir "cp_noise.obj"),
    (Join-Path $BuildDir "blake3.obj"),
    (Join-Path $BuildDir "blake3_dispatch.obj"),
    (Join-Path $BuildDir "blake3_portable.obj")
) + $RustExtra + @("-o", $OutExe)
Invoke-Nvcc @linkArgs

Write-Host "=== Done: $OutExe ==="
& $OutExe --help 2>&1 | Select-Object -First 15
