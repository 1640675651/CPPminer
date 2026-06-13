# Build cpminer.exe (CUDA) on Windows.
# Requires: NVIDIA CUDA Toolkit 12.x, MSVC (VS Build Tools), nvidia-smi, Rust (cargo on PATH).
#
# Usage:
#   conda activate pearl
#   powershell -ExecutionPolicy Bypass -File build.ps1
#   powershell -ExecutionPolicy Bypass -File build.ps1 -CudaArch 61
#
# The script snapshots and restores your shell environment on exit (success or failure),
# so repeated builds in the same terminal do not require restarting it.

param(
    [string]$CudaArch = "",
    [string]$CudaRoot = ""
)

$ErrorActionPreference = "Stop"
# Drop stale records from an earlier failed build in the same shell session.
if ($Error.Count -gt 0) { $Error.Clear() }
$Root = $PSScriptRoot
$BuildDir = Join-Path $Root "build\win"
$B3Dir = Join-Path $BuildDir "b3"
$OutExe = Join-Path $Root "cpminer.exe"
$script:VcvarsBat = $null
$script:ClExe = $null
$script:OrigEnv = $null

function Save-ShellEnvironment {
    # Snapshot the caller's shell so vcvars/nvcc work cannot poison later builds.
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
    # Resolve cl.exe in an isolated cmd session — do not import vcvars into this shell.
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
    if (-not $script:ClExe) {
        throw "cl.exe not found after vcvars64"
    }
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
    <#
    Run a native command without treating stderr as a terminating PowerShell error.
    MSVC/nvcc/cargo routinely print warnings to stderr even on success; with
    $ErrorActionPreference = Stop that aborts the script and poisons the shell.
    #>
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
        # Clear stale MSVC/conda toolchain vars inherited by cmd, then apply vcvars once.
        $envPreamble = "set INCLUDE=&& set LIB=&& set LIBPATH=&& set CC=&& set CXX=&& set CUDAHOSTC=&& set CUDAHOSTCXX=&& "
        if ($script:VcvarsBat) {
            $cmdLine = "${envPreamble}`"$script:VcvarsBat`" >nul 2>&1 && `"$Nvcc`" $quoted"
        } else {
            $cmdLine = "${envPreamble}`"$Nvcc`" $quoted"
        }
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
    $CutlassRoot = Ensure-Cutlass
    $Inc += "-I$(Join-Path $CutlassRoot 'include')"
    $Inc += "-I$(Join-Path $CutlassRoot 'examples/35_gemm_softmax')"

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
        Invoke-External -Command { cargo build --release } -FailureMessage "cargo build failed"
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
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $OutExe --help 2>&1 | Select-Object -First 15 | ForEach-Object { Write-Host $_ }
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
