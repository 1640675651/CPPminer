# Build cppminer on Windows via CMake (same pipeline as build.sh on *nix).
# Default: CPU backend only (no CUDA Toolkit required).
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File build.ps1
#   powershell -ExecutionPolicy Bypass -File build.ps1 -Backend Cpu
#   powershell -ExecutionPolicy Bypass -File build.ps1 -Backend Cuda -CudaArch 61
#   powershell -ExecutionPolicy Bypass -File build.ps1 -Backend Cpu,OpenCl
#   powershell -ExecutionPolicy Bypass -File build.ps1 -Backend Cpu,Cuda,OpenCl -CudaArch 75
#   powershell -ExecutionPolicy Bypass -File build.ps1 -Backend Cuda -EnableCublas
#
# Requires: MSVC, CMake, and (for proofs) cargo. OpenCL headers/CUTLASS are
# fetched as needed. The script snapshots and restores your shell environment.

param(
    [ValidateSet("Cpu", "Cuda", "OpenCl")]
    [string[]]$Backend = @("Cpu"),
    [string]$CudaArch = "",
    [string]$CudaRoot = "",
    [switch]$EnableCublas
)

$ErrorActionPreference = "Stop"
if ($Error.Count -gt 0) { $Error.Clear() }
$Root = $PSScriptRoot
$BuildDir = Join-Path $Root "build\win"
$B3Dir = Join-Path $BuildDir "b3"
$OutExe = Join-Path $Root "cppminer.exe"

$BackendList = @($Backend | ForEach-Object { "$_".Trim() } | Where-Object { $_ } | Select-Object -Unique)
if ($BackendList.Count -eq 0) {
    $BackendList = @("Cpu")
}
$EnableCpu = $BackendList -contains "Cpu"
$EnableCuda = $BackendList -contains "Cuda"
$EnableOpenCl = $BackendList -contains "OpenCl"
if (-not ($EnableCpu -or $EnableCuda -or $EnableOpenCl)) {
    throw "Select at least one backend: -Backend Cpu,Cuda,OpenCl"
}
if ($EnableCublas -and -not $EnableCuda) {
    throw "-EnableCublas requires -Backend Cuda (or Cpu,Cuda / ...)"
}
$script:VcvarsBat = $null
$script:ClExe = $null
$script:CmakeExe = $null
$script:OrigEnv = $null

function Save-ShellEnvironment {
    $script:OrigEnv = @{}
    Get-ChildItem Env: | ForEach-Object { $script:OrigEnv[$_.Name] = $_.Value }
}

function Restore-ShellEnvironment {
    if (-not $script:OrigEnv) { return }
    Get-ChildItem Env: | ForEach-Object {
        if (-not $script:OrigEnv.ContainsKey($_.Name)) {
            Remove-Item "env:$($_.Name)" -ErrorAction SilentlyContinue
        }
    }
    foreach ($kv in $script:OrigEnv.GetEnumerator()) {
        Set-Item -Path "env:$($kv.Key)" -Value $kv.Value
    }
    $script:OrigEnv = $null
}

function Clear-CondaToolchainOverrides {
    foreach ($var in @('CC', 'CXX', 'CFLAGS', 'CXXFLAGS', 'LDFLAGS', 'CPPFLAGS')) {
        if (Test-Path "env:$var") {
            Remove-Item "env:$var" -ErrorAction SilentlyContinue
        }
    }
    if ($env:CONDA_PREFIX) {
        Write-Host "=== Conda active ($env:CONDA_PREFIX); cleared CC/CXX for toolchain ==="
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
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    $clOut = $null
    try {
        $clOut = cmd /c "`"$vcvars`" >nul 2>&1 && where cl.exe" 2>&1
    } finally {
        $ErrorActionPreference = $prevEap
    }
    $script:ClExe = ($clOut | ForEach-Object {
        if ($_ -is [System.Management.Automation.ErrorRecord]) { $_.ToString() } else { $_ }
    } | Where-Object { $_ -and $_ -match 'cl\.exe' } | Select-Object -First 1)
    if ($script:ClExe) { $script:ClExe = $script:ClExe.Trim() }
    if (-not $script:ClExe) { throw "cl.exe not found after vcvars64" }
    Write-Host "=== cl.exe: $($script:ClExe) ==="
}

function Find-CMake {
    $cmd = Get-Command cmake -ErrorAction SilentlyContinue
    if ($cmd -and $cmd.Source -and (Test-Path $cmd.Source)) {
        return $cmd.Source
    }

    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $hits = @(& $vswhere -latest -products * -find "**/CMake/CMake/bin/cmake.exe" 2>$null)
        foreach ($hit in $hits) {
            if ($hit -and (Test-Path $hit)) { return $hit }
        }
        $hits = @(& $vswhere -latest -products * -find "**/cmake.exe" 2>$null)
        foreach ($hit in $hits) {
            if ($hit -and $hit -match '[\\/]CMake[\\/]bin[\\/]cmake\.exe$' -and (Test-Path $hit)) {
                return $hit
            }
        }
    }

    foreach ($p in @(
        "${env:ProgramFiles}\CMake\bin\cmake.exe",
        "${env:ProgramFiles(x86)}\CMake\bin\cmake.exe",
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe",
        "${env:ProgramFiles}\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe",
        "${env:ProgramFiles}\Microsoft Visual Studio\2022\Professional\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe",
        "${env:ProgramFiles}\Microsoft Visual Studio\2022\Enterprise\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
    )) {
        if (Test-Path $p) { return $p }
    }
    return $null
}

function Find-CudaRoot {
    if ($CudaRoot -and (Test-Path (Join-Path $CudaRoot "bin\nvcc.exe"))) { return $CudaRoot }
    $base = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    if (Test-Path $base) {
        $latest = Get-ChildItem $base -Directory | Sort-Object Name -Descending | Select-Object -First 1
        if ($latest) { return $latest.FullName }
    }
    throw "CUDA Toolkit not found (needed when -Backend includes Cuda)."
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

function Ensure-Cutlass {
    $cutlassRoot = Join-Path $Root "third_party\cutlass"
    $cutlassHdr = Join-Path $cutlassRoot "include\cutlass\cutlass.h"
    if (Test-Path $cutlassHdr) { return $cutlassRoot }
    Write-Host "=== Fetching CUTLASS v2.11.0 ==="
    $zipPath = Join-Path $BuildDir "cutlass-v2.11.0.zip"
    $extractParent = Join-Path $Root "third_party"
    New-Item -ItemType Directory -Force -Path $extractParent | Out-Null
    Invoke-WebRequest -Uri "https://github.com/NVIDIA/cutlass/archive/refs/tags/v2.11.0.zip" `
        -OutFile $zipPath -UseBasicParsing
    Expand-Archive -Path $zipPath -DestinationPath $extractParent -Force
    $extracted = Join-Path $extractParent "cutlass-2.11.0"
    if (Test-Path $cutlassRoot) { Remove-Item $cutlassRoot -Recurse -Force }
    Rename-Item $extracted $cutlassRoot
    Remove-Item $zipPath -Force -ErrorAction SilentlyContinue
    if (-not (Test-Path $cutlassHdr)) {
        throw "CUTLASS fetch failed: missing $cutlassHdr"
    }
    return $cutlassRoot
}

function Ensure-OpenClHeaders {
    $hdrRoot = Join-Path $Root "third_party\opencl-headers"
    if (-not (Test-Path (Join-Path $hdrRoot "CL\cl.h"))) {
        Write-Host "=== Fetching Khronos OpenCL-Headers ==="
        if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
            throw "git required to fetch OpenCL headers into third_party/opencl-headers"
        }
        if (Test-Path $hdrRoot) { Remove-Item $hdrRoot -Recurse -Force }
        Invoke-External -Command {
            git clone --depth 1 --branch v2024.10.24 `
                https://github.com/KhronosGroup/OpenCL-Headers.git $hdrRoot
        } -FailureMessage "OpenCL-Headers clone failed"
    }
    if (-not (Test-Path (Join-Path $hdrRoot "CL\cl.h"))) {
        throw "OpenCL headers missing at $hdrRoot"
    }
    return $hdrRoot
}

function Find-OpenClLib {
    $vendored = Join-Path $Root "third_party\opencl\lib\x64\OpenCL.lib"
    if (Test-Path $vendored) { return $vendored }
    throw "Vendored OpenCL.lib missing at $vendored (see third_party/opencl/README.md)"
}

function Ensure-CargoOnPath {
    $cmd = Get-Command cargo -ErrorAction SilentlyContinue
    if ($cmd -and $cmd.Source) {
        Write-Host "=== cargo: $($cmd.Source) ==="
        return $true
    }
    $dirs = @(
        $(if ($env:CARGO_HOME) { Join-Path $env:CARGO_HOME "bin" } else { $null }),
        (Join-Path $env:USERPROFILE ".cargo\bin")
    )
    if ($env:CONDA_PREFIX) {
        $dirs += @(
            (Join-Path $env:CONDA_PREFIX "Library\bin"),
            (Join-Path $env:CONDA_PREFIX "bin"),
            (Join-Path $env:CONDA_PREFIX "Scripts")
        )
    }
    foreach ($dir in $dirs) {
        if (-not $dir) { continue }
        $exe = Join-Path $dir "cargo.exe"
        if (Test-Path $exe) {
            $env:PATH = "$dir;$env:PATH"
            Write-Host "=== cargo: $exe (prepended to PATH for CMake) ==="
            return $true
        }
    }
    Write-Host "=== WARNING: cargo not found; CMake will link the proof stub ==="
    return $false
}

function Copy-OpenClKernels {
    $kernelSrcDir = Join-Path $Root "src\opencl\kernels"
    $kernelDstDir = Join-Path $Root "kernels"
    New-Item -ItemType Directory -Force -Path $kernelDstDir | Out-Null
    foreach ($name in @(
        "case33_gemm_xor.cl",
        "cp_ocl_blake3.cl",
        "cp_ocl_merkle.cl",
        "cp_ocl_prep.cl"
    )) {
        Copy-Item (Join-Path $kernelSrcDir $name) (Join-Path $kernelDstDir $name) -Force
    }
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

function Invoke-External {
    param(
        [Parameter(Mandatory = $true)][scriptblock]$Command,
        [Parameter(Mandatory = $true)][string]$FailureMessage
    )
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    $exitCode = 0
    try {
        & $Command 2>&1 | ForEach-Object {
            if ($_ -is [System.Management.Automation.ErrorRecord]) {
                Write-Host $_.ToString()
            } else {
                Write-Host $_
            }
        }
        if ($null -ne $LASTEXITCODE) { $exitCode = $LASTEXITCODE }
    } finally {
        $ErrorActionPreference = $prevEap
    }
    if ($exitCode -ne 0) { throw $FailureMessage }
}

Save-ShellEnvironment
$buildExitCode = 0
try {
    Clear-CondaToolchainOverrides
    Initialize-MSVC
    $script:CmakeExe = Find-CMake
    if (-not $script:CmakeExe) {
        throw "cmake not found. Install VS 'CMake tools for Windows' or add cmake to PATH (same as build.sh)."
    }
    $cmakeBin = Split-Path -Parent $script:CmakeExe
    if ($env:PATH -notlike "*$cmakeBin*") {
        $env:PATH = "$cmakeBin;$env:PATH"
    }
    Write-Host "=== cmake: $($script:CmakeExe) ==="
    $null = Ensure-CargoOnPath

    New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
    Ensure-Blake3

    Write-Host "=== Backend: $($BackendList -join ',') (CPU=$EnableCpu CUDA=$EnableCuda OpenCL=$EnableOpenCl CUBLAS=$EnableCublas) ==="

    if ($EnableOpenCl) {
        $null = Ensure-OpenClHeaders
        $null = Find-OpenClLib
    }
    if ($EnableCuda) {
        $null = Ensure-Cutlass
        $CudaRoot = Find-CudaRoot
        Write-Host "=== CUDA: $CudaRoot ==="
        if (-not $CudaArch) { $CudaArch = Get-GpuArch }
        Write-Host "=== CUDA arch: $CudaArch ==="
    }

    $CmakeBuild = Join-Path $BuildDir "cmake"
    New-Item -ItemType Directory -Force -Path $CmakeBuild | Out-Null
    $cmakeArgs = @(
        "-S", $Root,
        "-B", $CmakeBuild,
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCP_ENABLE_CPU=$(if ($EnableCpu) { 'ON' } else { 'OFF' })",
        "-DCP_ENABLE_CUDA=$(if ($EnableCuda) { 'ON' } else { 'OFF' })",
        "-DCP_ENABLE_OPENCL=$(if ($EnableOpenCl) { 'ON' } else { 'OFF' })",
        "-DCP_ENABLE_CUBLAS=$(if ($EnableCublas) { 'ON' } else { 'OFF' })"
    )
    if ($EnableCuda -and $CudaArch) {
        $cmakeArgs += "-DCP_CUDA_ARCH=$CudaArch"
    }

    Write-Host "=== CMake configure ==="
    Invoke-External -Command { & $script:CmakeExe @cmakeArgs } -FailureMessage "cmake configure failed"
    Write-Host "=== CMake build ==="
    Invoke-External -Command { & $script:CmakeExe --build $CmakeBuild --config Release } -FailureMessage "cmake build failed"

    $built = @(
        (Join-Path $CmakeBuild "Release\cppminer.exe"),
        (Join-Path $CmakeBuild "cppminer.exe")
    ) | Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not $built) { throw "cppminer.exe not found under $CmakeBuild" }
    Copy-Item $built $OutExe -Force
    if ($EnableOpenCl) {
        Copy-OpenClKernels
    }

    # Keep legacy cpminer.exe name in sync (docs / habit).
    if (Test-Path $OutExe) {
        Copy-Item $OutExe (Join-Path $Root "cpminer.exe") -Force
    }

    Write-Host "=== Done: $OutExe ==="
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $OutExe --help 2>&1 | Select-Object -First 20 | ForEach-Object { Write-Host $_ }
    } finally {
        $ErrorActionPreference = $prevEap
    }
} catch {
    Write-Host $_.Exception.Message
    $buildExitCode = 1
} finally {
    Restore-ShellEnvironment
    if ($Error.Count -gt 0) { $Error.Clear() }
}
exit $buildExitCode
