# Build cppminer on Windows.
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
# The script snapshots and restores your shell environment on exit.

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

function Get-SanitizedPathForNvcc {
    param([string]$PathValue)
    ($PathValue -split ';' | Where-Object {
        $_ -and $_ -notmatch '\\miniconda3\\|\\anaconda3\\|\\envs\\|mingw-w64|\\.cargo\\bin'
    }) -join ';'
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

function Invoke-Cl {
    param([Parameter(Mandatory = $true)][string[]]$ClArgs)
    $quoted = ($ClArgs | ForEach-Object {
        if ($_ -match '[\s"]') { '"' + ($_ -replace '"', '""') + '"' } else { $_ }
    }) -join ' '
    $cmdLine = "`"$script:VcvarsBat`" >nul 2>&1 && cl.exe $quoted"
    Invoke-External -Command { cmd /c $cmdLine } -FailureMessage "cl failed: $($ClArgs -join ' ')"
}

function Invoke-Nvcc {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$NvccArgs)
    if (-not $script:ClExe) { throw "MSVC cl.exe not configured" }
    $ccbin = @("-ccbin", $script:ClExe)
    $allArgs = $ccbin + $NvccArgs
    $quoted = ($allArgs | ForEach-Object {
        if ($_ -match '[\s"]') { '"' + ($_ -replace '"', '""') + '"' } else { $_ }
    }) -join ' '
    $savedPath = $env:PATH
    $env:PATH = Get-SanitizedPathForNvcc -PathValue $env:PATH
    try {
        $envPreamble = "set INCLUDE=&& set LIB=&& set LIBPATH=&& set CC=&& set CXX=&& set CUDAHOSTC=&& set CUDAHOSTCXX=&& "
        $cmdLine = "${envPreamble}`"$script:VcvarsBat`" >nul 2>&1 && `"$Nvcc`" $quoted"
        Invoke-External -Command { cmd /c $cmdLine } -FailureMessage "nvcc failed: $($NvccArgs -join ' ')"
    } finally {
        $env:PATH = $savedPath
    }
}

Save-ShellEnvironment
$buildExitCode = 0
try {
    Clear-CondaToolchainOverrides
    Initialize-MSVC
    $script:CmakeExe = Find-CMake
    if ($script:CmakeExe) {
        $cmakeBin = Split-Path -Parent $script:CmakeExe
        if ($env:PATH -notlike "*$cmakeBin*") {
            $env:PATH = "$cmakeBin;$env:PATH"
        }
        Write-Host "=== cmake: $($script:CmakeExe) ==="
    }
    New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
    Ensure-Blake3

    Write-Host "=== Backend: $($BackendList -join ',') (CPU=$EnableCpu CUDA=$EnableCuda OpenCL=$EnableOpenCl CUBLAS=$EnableCublas) ==="

    function Ensure-CpProofFfi {
        Write-Host "=== Building cp-proof-ffi (Rust) ==="
        $RustDir = Join-Path $Root "rust\cp-proof-ffi"
        $PearlBlake3 = Join-Path $Root "third_party\pearl-blake3\Cargo.toml"
        $ZkPow = Join-Path $Root "third_party\zk-pow\Cargo.toml"
        $Plonky2 = Join-Path $Root "third_party\plonky2\plonky2\Cargo.toml"
        $script:RustLib = Join-Path $RustDir "target\release\cp_proof_ffi.lib"
        if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
            Write-Host "=== WARNING: cargo not found; linking proof stub (no share submit proofs) ==="
            return $false
        }
        if (-not (Test-Path $PearlBlake3)) {
            Write-Host "=== WARNING: third_party/pearl-blake3 missing; linking proof stub ==="
            return $false
        }
        if (-not (Test-Path $ZkPow)) {
            Write-Host "=== WARNING: third_party/zk-pow missing; linking proof stub ==="
            return $false
        }
        if (-not (Test-Path $Plonky2)) {
            Write-Host "=== WARNING: third_party/plonky2 missing (zk-pow dep); linking proof stub ==="
            return $false
        }
        Push-Location $RustDir
        try {
            Invoke-External -Command { cargo build --release } -FailureMessage "cargo build failed"
        } finally {
            Pop-Location
        }
        if (-not (Test-Path $script:RustLib)) { throw "Missing $($script:RustLib)" }
        return $true
    }

    function Build-CpuWithCl {
        Write-Host "=== Direct MSVC build (CPU, no cmake) ==="
        $null = Ensure-CpProofFfi
        $Inc = @(
            "/I$(Join-Path $Root 'include')",
            "/I$(Join-Path $Root 'src\common')",
            "/I$(Join-Path $Root 'src\cpu')",
            "/I$B3Dir"
        )
        $Defs = @("/DCP_ENABLE_CPU=1", "/DCP_ENABLE_CUDA=0", "/DCP_ENABLE_OPENCL=0",
                  "/DBLAKE3_NO_AVX512", "/DBLAKE3_NO_AVX2", "/DBLAKE3_NO_SSE41", "/DBLAKE3_NO_SSE2")
        # Baseline without /arch:AVX2 so SSE/scalar paths stay safe on non-AVX2 CPUs.
        $Flags = @("/nologo", "/O2", "/EHsc", "/std:c++17", "/openmp") + $Inc + $Defs
        $objs = @()

        $cUnits = @(
            @{ src = (Join-Path $B3Dir "blake3.c"); obj = "blake3.obj" },
            @{ src = (Join-Path $B3Dir "blake3_dispatch.c"); obj = "blake3_dispatch.obj" },
            @{ src = (Join-Path $B3Dir "blake3_portable.c"); obj = "blake3_portable.obj" },
            @{ src = (Join-Path $Root "src\common\cp_noise.c"); obj = "cp_noise.obj" }
        )
        foreach ($u in $cUnits) {
            $objPath = Join-Path $BuildDir $u.obj
            $objs += $objPath
            Invoke-Cl -ClArgs ($Flags + @("/TC", "/c", $u.src, "/Fo$objPath"))
        }

        $cppUnits = @(
            "src\common\main.cpp",
            "src\common\cp_util.cpp",
            "src\common\cp_pool.cpp",
            "src\common\cp_job_ctrl.cpp",
            "src\common\cp_mine.cpp",
            "src\common\cp_fee.cpp",
            "src\common\cp_share_queue.cpp",
            "src\common\cp_state.cpp",
            "src\common\cp_worker.cpp",
            "src\cpu\cp_cpu_worker.cpp",
            "src\cpu\gemm\case33_gemm_xor.cpp",
            "src\cpu\gemm\case33_gemm_xor_avx2.cpp"
        )
        if (-not (Test-Path $script:RustLib)) {
            $cppUnits += "src\common\cp_proof_stub.cpp"
        }
        foreach ($rel in $cppUnits) {
            $srcPath = Join-Path $Root $rel
            $objName = [IO.Path]::GetFileNameWithoutExtension($rel) + ".obj"
            $objPath = Join-Path $BuildDir $objName
            $objs += $objPath
            $unitFlags = $Flags
            if ($rel -like "*case33_gemm_xor_avx2.cpp") {
                $unitFlags = $Flags + @("/arch:AVX2")
            }
            Invoke-Cl -ClArgs ($unitFlags + @("/c", $srcPath, "/Fo$objPath"))
        }

        $linkLibs = @("ws2_32.lib")
        if ((Test-Path variable:script:RustLib) -and (Test-Path $script:RustLib)) {
            $linkLibs += @($script:RustLib, "userenv.lib", "bcrypt.lib", "ntdll.lib", "advapi32.lib")
        }
        $linkArgs = @("/nologo", "/Fe$OutExe") + $objs + $linkLibs
        Invoke-Cl -ClArgs $linkArgs
    }

    function Build-OpenClWithCl {
        $oclHdr = Ensure-OpenClHeaders
        $oclLib = Find-OpenClLib
        Write-Host "=== Direct MSVC build (OpenCL, no cmake) ==="
        Write-Host "=== OpenCL.lib: $oclLib ==="
        if ($EnableCpu) { $null = Ensure-CpProofFfi }

        $Inc = @(
            "/I$(Join-Path $Root 'include')",
            "/I$(Join-Path $Root 'src\common')",
            "/I$(Join-Path $Root 'src\cpu')",
            "/I$(Join-Path $Root 'src\opencl')",
            "/I$oclHdr",
            "/I$B3Dir"
        )
        $Defs = @(
            "/DCP_ENABLE_CPU=$(if ($EnableCpu) { '1' } else { '0' })",
            "/DCP_ENABLE_CUDA=0",
            "/DCP_ENABLE_OPENCL=1",
            "/DBLAKE3_NO_AVX512", "/DBLAKE3_NO_AVX2", "/DBLAKE3_NO_SSE41", "/DBLAKE3_NO_SSE2"
        )
        $Flags = @("/nologo", "/O2", "/EHsc", "/std:c++17", "/openmp") + $Inc + $Defs
        $objs = @()

        $cUnits = @(
            @{ src = (Join-Path $B3Dir "blake3.c"); obj = "blake3.obj" },
            @{ src = (Join-Path $B3Dir "blake3_dispatch.c"); obj = "blake3_dispatch.obj" },
            @{ src = (Join-Path $B3Dir "blake3_portable.c"); obj = "blake3_portable.obj" },
            @{ src = (Join-Path $Root "src\common\cp_noise.c"); obj = "cp_noise.obj" }
        )
        foreach ($u in $cUnits) {
            $objPath = Join-Path $BuildDir $u.obj
            $objs += $objPath
            Invoke-Cl -ClArgs ($Flags + @("/TC", "/c", $u.src, "/Fo$objPath"))
        }

        $cppUnits = @(
            "src\common\main.cpp",
            "src\common\cp_util.cpp",
            "src\common\cp_pool.cpp",
            "src\common\cp_job_ctrl.cpp",
            "src\common\cp_mine.cpp",
            "src\common\cp_fee.cpp",
            "src\common\cp_share_queue.cpp",
            "src\common\cp_state.cpp",
            "src\common\cp_worker.cpp",
            "src\opencl\cp_opencl_worker.cpp",
            "src\opencl\case33_gemm_ocl.cpp",
            "src\opencl\case33_ocl_prep.cpp",
            "src\opencl\cp_ocl_align_test.cpp",
            "src\opencl\cp_ocl_prep_profile.cpp",
            "src\opencl\case32_prepack.cpp",
            "src\opencl\opencl_context.cpp"
        )
        if ($EnableCpu) {
            $cppUnits += @(
                "src\cpu\cp_cpu_worker.cpp",
                "src\cpu\gemm\case33_gemm_xor.cpp",
                "src\cpu\gemm\case33_gemm_xor_avx2.cpp"
            )
        }
        if (-not ((Test-Path variable:script:RustLib) -and (Test-Path $script:RustLib))) {
            $cppUnits += "src\common\cp_proof_stub.cpp"
        }
        foreach ($rel in $cppUnits) {
            $srcPath = Join-Path $Root $rel
            $objName = [IO.Path]::GetFileNameWithoutExtension($rel) + ".obj"
            $objPath = Join-Path $BuildDir $objName
            $objs += $objPath
            $unitFlags = $Flags
            if ($rel -like "*case33_gemm_xor_avx2.cpp") {
                $unitFlags = $Flags + @("/arch:AVX2")
            }
            Invoke-Cl -ClArgs ($unitFlags + @("/c", $srcPath, "/Fo$objPath"))
        }

        $linkLibs = @("ws2_32.lib", $oclLib)
        if ((Test-Path variable:script:RustLib) -and (Test-Path $script:RustLib)) {
            $linkLibs += @($script:RustLib, "userenv.lib", "bcrypt.lib", "ntdll.lib", "advapi32.lib")
        }
        $linkArgs = @("/nologo", "/Fe$OutExe") + $objs + $linkLibs
        Invoke-Cl -ClArgs $linkArgs
        Copy-OpenClKernels
    }

    $hasCmake = [bool]$script:CmakeExe
    if ($EnableCuda -and -not $hasCmake) {
        throw "CUDA build requires cmake (install VS 'CMake tools for Windows' component, or add cmake to PATH)."
    }

    if ($hasCmake -and ($EnableCuda -or $EnableOpenCl -or -not $EnableCpu)) {
        $CmakeBuild = Join-Path $BuildDir "cmake"
        New-Item -ItemType Directory -Force -Path $CmakeBuild | Out-Null
        $cmakeArgs = @(
            "-S", $Root,
            "-B", $CmakeBuild,
            "-DCP_ENABLE_CPU=$(if ($EnableCpu) { 'ON' } else { 'OFF' })",
            "-DCP_ENABLE_CUDA=$(if ($EnableCuda) { 'ON' } else { 'OFF' })",
            "-DCP_ENABLE_OPENCL=$(if ($EnableOpenCl) { 'ON' } else { 'OFF' })",
            "-DCP_ENABLE_CUBLAS=$(if ($EnableCublas) { 'ON' } else { 'OFF' })"
        )
        if ($EnableCuda -and $CudaArch) {
            $cmakeArgs += "-DCP_CUDA_ARCH=$CudaArch"
        }
        if ($EnableCuda) {
            $null = Ensure-Cutlass
            $CudaRoot = Find-CudaRoot
            Write-Host "=== CUDA: $CudaRoot ==="
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
    } elseif ($EnableOpenCl -and -not $EnableCuda) {
        Build-OpenClWithCl
    } else {
        if (-not $EnableCpu -or $EnableCuda) {
            throw "Without cmake, use -Backend Cpu and/or OpenCl (Cuda requires cmake)."
        }
        Build-CpuWithCl
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
