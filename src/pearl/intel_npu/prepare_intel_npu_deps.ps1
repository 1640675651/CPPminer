# Fetches OpenVINO and the NPU compiler (openvinotoolkit/npu_compiler) at the pinned commits, applies the
# pearl patch to the compiler (int32 DPU output, the TileXorAcc op, external SHAVE kernels, cycle costs),
# builds both, and writes third_party/intel_npu/paths.cmake for the CMake build (-DCP_ENABLE_INTEL_NPU=ON).
#
#   powershell -ExecutionPolicy Bypass -File src\pearl\intel_npu\prepare_intel_npu_deps.ps1 [-Refresh] [-NoBuild]
#
# Pins: patches\OPENVINO_COMMIT and patches\NPU_COMPILER_COMMIT (the commit npu_compiler_all.diff applies to).
# Environment overrides:
#   OPENVINO_COMMIT, NPU_COMPILER_COMMIT   other revisions (the patch may then not apply; the two must match)
#   OPENVINO_SRC                           an existing OpenVINO checkout (default third_party\openvino-src)
#   OPENVINO_BUILD_DIR, OPENVINO_BIN_DIR   its build tree and binaries (default <src>\build-x86_64\Release,
#                                          <src>\bin\intel64\Release)
#   NPU_COMPILER_SRC                       an existing npu_compiler checkout (default third_party\npu_compiler-src);
#                                          its HEAD is never moved, the patch is applied if it is not there yet
# Requires Git, MSVC 2022 (C++ tools), CMake 3.24+, Ninja and Python 3 on PATH. A first build from scratch
# downloads several GB (OpenVINO and LLVM submodules) and takes hours; later runs only rebuild what changed.
param(
    [switch]$Refresh,   # re-run CMake configure for both projects
    [switch]$NoBuild    # fetch and patch only
)

$ErrorActionPreference = "Stop"
$NpuDir = $PSScriptRoot
$Root = (Resolve-Path (Join-Path $NpuDir "..\..\..")).Path
$Patches = Join-Path $NpuDir "patches"
$PatchFile = Join-Path $Patches "npu_compiler_all.diff"
$ThirdParty = Join-Path $Root "third_party"
$OutDir = Join-Path $ThirdParty "intel_npu"

function Read-Pin([string]$name) {
    $value = [Environment]::GetEnvironmentVariable($name)
    if ($value) { return $value.Trim() }
    return (Get-Content (Join-Path $Patches $name) -TotalCount 1).Trim()
}

$OvCommit = Read-Pin "OPENVINO_COMMIT"
$CompilerCommit = Read-Pin "NPU_COMPILER_COMMIT"
$OvOwned = -not $env:OPENVINO_SRC
$OvSrc = if ($env:OPENVINO_SRC) { $env:OPENVINO_SRC } else { Join-Path $ThirdParty "openvino-src" }
$OvBuild = if ($env:OPENVINO_BUILD_DIR) { $env:OPENVINO_BUILD_DIR } else { Join-Path $OvSrc "build-x86_64\Release" }
$OvBin = if ($env:OPENVINO_BIN_DIR) { $env:OPENVINO_BIN_DIR } else { Join-Path $OvSrc "bin\intel64\Release" }
$CompilerOwned = -not $env:NPU_COMPILER_SRC
$CompilerSrc = if ($env:NPU_COMPILER_SRC) { $env:NPU_COMPILER_SRC } else { Join-Path $ThirdParty "npu_compiler-src" }
$CompilerBuild = Join-Path $CompilerSrc "build-x86_64\Release"

function Invoke-Checked([string]$what, [scriptblock]$cmd) {
    & $cmd
    if ($LASTEXITCODE -ne 0) { throw "$what failed (exit $LASTEXITCODE)" }
}

function Import-MsvcEnvironment {
    if ($env:VSCMD_VER -and (Get-Command cl.exe -ErrorAction SilentlyContinue)) { return }
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) { throw "Visual Studio Build Tools not found (vswhere missing)" }
    $vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $vs) { throw "MSVC C++ tools not found" }
    $vcvars = Join-Path $vs "VC\Auxiliary\Build\vcvars64.bat"
    $lines = cmd /c "`"$vcvars`" >nul 2>&1 && set"
    foreach ($line in $lines) {
        if ($line -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], "Process") }
    }
    if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) { throw "cl.exe not found after vcvars64.bat" }
    Write-Host "=== MSVC: $vs ==="
}

function Find-Tool([string]$exe, [string]$vsPattern) {
    $cmd = Get-Command $exe -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        foreach ($hit in @(& $vswhere -latest -products * -find $vsPattern 2>$null)) {
            if ($hit -and (Test-Path $hit)) { return $hit }
        }
    }
    throw "$exe not found on PATH or in Visual Studio"
}

function Get-Head([string]$repo) {
    return (git -C $repo rev-parse HEAD 2>$null)
}

function Ensure-Clone([string]$url, [string]$dir, [string]$commit) {
    if (-not (Test-Path (Join-Path $dir ".git"))) {
        Write-Host "=== Cloning $url at $commit into $dir ==="
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dir) | Out-Null
        Invoke-Checked "git clone $url" { git -c core.longpaths=true clone --no-checkout $url $dir }
    }
    if ((Get-Head $dir) -ne $commit) {
        Invoke-Checked "git fetch $commit" { git -C $dir -c core.longpaths=true fetch --quiet origin $commit }
        Invoke-Checked "git checkout $commit" { git -C $dir -c core.longpaths=true -c advice.detachedHead=false checkout --force $commit }
    }
    Write-Host "=== Updating submodules of $dir ==="
    Invoke-Checked "git submodule update ($dir)" { git -C $dir -c core.longpaths=true submodule update --init --recursive --jobs 8 }
}

function Ensure-OpenVino {
    $devPackage = Join-Path $OvBuild "OpenVINODeveloperPackageConfig.cmake"
    if ((Test-Path $devPackage) -and (Test-Path (Join-Path $OvBin "openvino.dll")) -and -not $Refresh) {
        $head = Get-Head $OvSrc
        Write-Host "=== OpenVINO: $OvBuild ($(if ($head) { $head.Substring(0, 10) } else { 'no git' })) ==="
        if ($head -and $head -ne $OvCommit) {
            Write-Host "    note: pinned $($OvCommit.Substring(0, 10)); the NPU compiler must match this revision"
        }
        return
    }
    if ($OvOwned) {
        Ensure-Clone "https://github.com/openvinotoolkit/openvino.git" $OvSrc $OvCommit
    } elseif (-not (Test-Path (Join-Path $OvSrc "CMakeLists.txt"))) {
        throw "OPENVINO_SRC=$OvSrc is not an OpenVINO checkout"
    }
    if ($NoBuild) { return }
    Write-Host "=== Configuring OpenVINO (NPU plugin only) ==="
    Invoke-Checked "OpenVINO configure" {
        & $script:Cmake -S $OvSrc -B $OvBuild -G Ninja "-DCMAKE_MAKE_PROGRAM=$script:Ninja" -DCMAKE_BUILD_TYPE=Release `
            -DENABLE_INTEL_NPU=ON -DENABLE_NPU_PLUGIN_ENGINE=ON -DENABLE_INTEL_NPU_COMPILER=OFF `
            -DENABLE_INTEL_CPU=OFF -DENABLE_INTEL_GPU=OFF -DENABLE_TEMPLATE=OFF -DENABLE_AUTO=OFF `
            -DENABLE_AUTO_BATCH=OFF -DENABLE_HETERO=OFF -DENABLE_PYTHON=OFF -DENABLE_WHEEL=OFF `
            -DENABLE_TESTS=OFF -DENABLE_SAMPLES=OFF -DENABLE_DOCS=OFF -DENABLE_JS=OFF `
            -DENABLE_OV_ONNX_FRONTEND=OFF -DENABLE_OV_TF_FRONTEND=OFF -DENABLE_OV_TF_LITE_FRONTEND=OFF `
            -DENABLE_OV_PYTORCH_FRONTEND=OFF -DENABLE_OV_PADDLE_FRONTEND=OFF -DENABLE_OV_JAX_FRONTEND=OFF
    }
    Write-Host "=== Building OpenVINO (long on the first run) ==="
    Invoke-Checked "OpenVINO build" { & $script:Cmake --build $OvBuild --parallel }
    if (-not (Test-Path $devPackage)) { throw "OpenVINO developer package missing after the build: $devPackage" }
}

function Ensure-Compiler {
    if ($CompilerOwned) {
        Ensure-Clone "https://github.com/openvinotoolkit/npu_compiler.git" $CompilerSrc $CompilerCommit
    } elseif (-not (Test-Path (Join-Path $CompilerSrc "src\vpux_compiler"))) {
        throw "NPU_COMPILER_SRC=$CompilerSrc is not an npu_compiler checkout"
    }
    # Apply the patch unless the tree already has it (it is one diff against the pinned commit).
    git -C $CompilerSrc -c core.longpaths=true apply --reverse --check --whitespace=nowarn $PatchFile 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "=== NPU compiler patch already applied ($CompilerSrc) ==="
    } else {
        git -C $CompilerSrc -c core.longpaths=true apply --check --whitespace=nowarn $PatchFile
        if ($LASTEXITCODE -ne 0) {
            throw "npu_compiler_all.diff does not apply to $CompilerSrc (HEAD $(Get-Head $CompilerSrc)); it is made " +
                  "for $CompilerCommit"
        }
        Invoke-Checked "git apply" { git -C $CompilerSrc -c core.longpaths=true apply --whitespace=nowarn $PatchFile }
        Write-Host "=== Applied the NPU compiler patch to $CompilerSrc ==="
    }
    if ($NoBuild) { return }
    if ($Refresh -or -not (Test-Path (Join-Path $CompilerBuild "build.ninja"))) {
        Write-Host "=== Configuring the NPU compiler against $OvBuild ==="
        Invoke-Checked "NPU compiler configure" {
            & $script:Cmake -S $CompilerSrc -B $CompilerBuild -G Ninja "-DCMAKE_MAKE_PROGRAM=$script:Ninja" `
                -DCMAKE_BUILD_TYPE=Release "-DOpenVINODeveloperPackage_DIR=$OvBuild" `
                -DENABLE_TESTS=OFF -DENABLE_FUNCTIONAL_TESTS=OFF -DENABLE_PRIVATE_TESTS=OFF
        }
    }
    Write-Host "=== Building openvino_intel_npu_compiler (long on the first run) ==="
    Invoke-Checked "NPU compiler build" {
        & $script:Cmake --build $CompilerBuild --parallel --target openvino_intel_npu_compiler
    }
    $dll = Join-Path $OvBin "openvino_intel_npu_compiler.dll"
    if (-not (Test-Path $dll)) { throw "patched compiler missing after the build: $dll" }
}

$prevPath = $env:Path
try {
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw "git not found" }
    if (-not (Test-Path $PatchFile)) { throw "missing $PatchFile" }
    if (-not $NoBuild) {
        Import-MsvcEnvironment
        $script:Cmake = Find-Tool "cmake.exe" "**\CMake\bin\cmake.exe"
        $script:Ninja = Find-Tool "ninja.exe" "**\CMake\Ninja\ninja.exe"
        Write-Host "=== cmake: $script:Cmake, ninja: $script:Ninja ==="
    }
    Write-Host "=== Pins: OpenVINO $($OvCommit.Substring(0, 10)), NPU compiler $($CompilerCommit.Substring(0, 10)) ==="
    Ensure-OpenVino
    Ensure-Compiler

    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
    $fwd = { param($p) ((Resolve-Path $p).Path -replace '\\', '/') }
    if (-not $NoBuild) {
        @(
            "# Written by src/pearl/intel_npu/prepare_intel_npu_deps.ps1",
            "set(CP_INTEL_NPU_OPENVINO_DIR `"$(& $fwd $OvBuild)`")",
            "set(CP_INTEL_NPU_OPENVINO_BIN_DIR `"$(& $fwd $OvBin)`")"
        ) | Set-Content -Encoding ascii (Join-Path $OutDir "paths.cmake")
        "$OvCommit $CompilerCommit $((Get-FileHash $PatchFile -Algorithm SHA256).Hash)" |
            Set-Content -Encoding ascii (Join-Path $OutDir "deps.stamp")
        Write-Host "=== Intel NPU deps ready: OpenVINO $OvBuild, runtime and patched compiler in $OvBin ==="
    } else {
        Write-Host "=== Fetched and patched (no build): $OvSrc, $CompilerSrc ==="
    }
    exit 0
} catch {
    Write-Host "prepare_intel_npu_deps: $($_.Exception.Message)"
    exit 1
} finally {
    $env:Path = $prevPath
}
