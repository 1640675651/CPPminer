# Build the pearlx DSP library (libpearlx_skel.so) and its Android self-test with the
# Hexagon SDK, and optionally push and run the self-test. The miner itself does not need
# the SDK: it calls pearlx through ../cp_pearlx_client.c.
# Usage: .\build.ps1 [-Sdk C:\Qualcomm\Hexagon_SDK\5.5.7.0] [-NoBuild] [-Push] [-Test "-n 4096 -g 4"]
param(
    [string]$Sdk = 'C:\Qualcomm\Hexagon_SDK\5.5.7.0',
    [string]$Dev = '/data/local/tmp/cppminer',
    [switch]$NoBuild,
    [switch]$Push,
    [string]$Test = $null
)

$root = $PSScriptRoot
$hexShip = "$root\hexagon_Release_toolv87_v66\ship"
$androidShip = "$root\android_Release_aarch64\ship"

if (-not $NoBuild) {
    # build_cmake can exit 0 even when ninja fails, so check the output too.
    $out = cmd /c "call `"$Sdk\setup_sdk_env.cmd`" >nul && cd /d `"$root`" && build_cmake hexagon BUILD=Release DSP_ARCH=v66 && build_cmake android BUILD=Release" 2>&1
    $out
    if ($LASTEXITCODE -ne 0 -or ($out | Select-String -Pattern '^FAILED:|build stopped' -Quiet)) {
        Write-Error "Build failed"; exit 1
    }
}

if ($Push -or $Test) {
    adb shell mkdir -p $Dev
    adb push "$hexShip\libpearlx_skel.so" "$hexShip\libworker_pool.so" "$androidShip\pearlx_test" "$androidShip\libpearlx.so" "$Dev/"
    if ($LASTEXITCODE -ne 0) { Write-Error "adb push failed"; exit 1 }
}

if ($Test) {
    adb shell "cd $Dev && chmod +x pearlx_test && export LD_LIBRARY_PATH=$Dev ADSP_LIBRARY_PATH=$Dev DSP_LIBRARY_PATH=$Dev && ./pearlx_test $Test"
}
