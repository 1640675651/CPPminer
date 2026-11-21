@echo off
REM Example: mine Pearl on the CPU.
REM Edit WALLET (and WORKER if you like), then double-click this file.
REM Other backends: replace "--backend cpu" with cuda, opencl, onednn, wgpu or intel_npu
REM and pick a device with --devices N (list them with: cppminer.exe --backend opencl --list-devices).

set WALLET=YOUR_WALLET_ADDRESS
set WORKER=rig01
set POOL=stratum+tcp://pearl-cpu-eu1.luckypool.io:3370

REM Optional extra flags, e.g.:
REM   --threads 8        use 8 threads (default: every logical CPU)
REM   --simd avx2        force a SIMD kernel (default: best for this CPU)
REM   --api-port 4068    read-only HTTP stats on 127.0.0.1:4068
set EXTRA=

cd /d "%~dp0"

if "%WALLET%"=="YOUR_WALLET_ADDRESS" (
    echo Edit %~nx0 and set WALLET to your Pearl wallet address first.
    pause
    exit /b 1
)

cppminer.exe --algo pearl --backend cpu --pool %POOL% --wallet %WALLET% --worker %WORKER% %EXTRA%

pause
