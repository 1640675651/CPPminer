# Rebuilds the Intel NPU SHAVE kernels in this directory from src\ (see ..\README.md, "SHAVE kernels"):
#   tile_xor_acc.3720xx*.elf   src\tile_xor_acc.c + the hand-scheduled routines src\gen_tile_xor_asm.py generates
#   tile_jackpot.3720xx*.elf   src\tile_jackpot.c (plain C)
# The build is deterministic: unchanged sources give byte-identical ELFs.
#
# Environment:
#   MOVITOOLS_DIR     the MoviTools MVC_DEPEND directory (bin\moviCompile64.dll, moviAsm64.dll, moviLLD64.dll, lib\)
#   NPU_COMPILER_SRC  the npu_compiler checkout, for its kernel linker script (default third_party\npu_compiler-src)
#   PYTHON            the Python 3 to use (default: python on PATH; the scripts need only the standard library)
#
#   powershell -ExecutionPolicy Bypass -File src\pearl\intel_npu\kernels\build_kernels.ps1 [-OutDir DIR] [-KeepAsm]
param(
    [string]$OutDir = $PSScriptRoot,  # where the ELFs go (default: here, replacing the checked-in ones)
    [switch]$KeepAsm                  # also keep the generated assembly (.s) next to them
)
$ErrorActionPreference = "Stop"
$src = Join-Path $PSScriptRoot "src"
$py = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$tmp = Join-Path ([IO.Path]::GetTempPath()) ("cp_npu_kernels_" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $tmp | Out-Null

function Invoke-Step([string]$what, [scriptblock]$cmd) {
    & $cmd
    if ($LASTEXITCODE -ne 0) { throw "$what failed (exit $LASTEXITCODE)" }
}

try {
    Invoke-Step "gen_tile_xor_asm.py" { & $py "$src\gen_tile_xor_asm.py" "$tmp\tx_strip.s" 4 }
    Invoke-Step "tile_xor_acc" {
        & $py "$src\movi.py" "$src\tile_xor_acc.c" tile_xor_acc -D TX_ASM_G4=1 --extra-asm "$tmp\tx_strip.s" --out $tmp
    }
    Invoke-Step "tile_jackpot" { & $py "$src\movi.py" "$src\tile_jackpot.c" tile_jackpot --out $tmp }
    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
    Copy-Item "$tmp\*.elf" $OutDir -Force
    if ($KeepAsm) { Copy-Item "$tmp\*.s" $OutDir -Force }
    Get-ChildItem "$OutDir\*.elf" | ForEach-Object { "{0,-34} {1,6} bytes" -f $_.Name, $_.Length }
} finally {
    Remove-Item -Recurse -Force $tmp
}
