@echo off
setlocal EnableDelayedExpansion

cd /d "%~dp0"

REM Vendors nGEN and gemmstone (with the Case 5 modifications) from the case5 fork of oneDNN:
REM   third_party\ngen       <- third_party\ngen
REM   third_party\gemmstone  <- src\gpu\intel\gemm\jit\{include,generator,selector,gemmstone_config.hpp}
REM (both under src\onednn). Source: ONEDNN_SRC (a local checkout, e.g. a clone of the fork you are
REM editing), else a shallow clone of ONEDNN_REPO at ONEDNN_REF in the repo root's third_party\onednn-src. The copy is a mirror and runs on
REM every call; robocopy only rewrites files that changed, so unchanged sources do not rebuild.
REM   prepare_onednn_deps.bat          use third_party\onednn-src as it is (clone it if missing or if
REM                                    it holds another repository / ref)
REM   prepare_onednn_deps.bat refresh  re-fetch ONEDNN_REF (picks up new commits on the branch)
REM The same script and defaults are in gemm_xor (onednn\prepare_case5_deps.bat): both build from the
REM same branch. CMake runs this script at configure time.
if not defined ONEDNN_REPO set "ONEDNN_REPO=https://github.com/1640675651/oneDNN.git"
if not defined ONEDNN_REF set "ONEDNN_REF=v3.13.2-prl-patch"
set "ONEDNN_FETCH_DIR=..\..\third_party\onednn-src"
set "FETCH_STAMP=..\..\third_party\.case5_onednn_ref"
set "NGEN_DEST=third_party\ngen"
set "GEMMSTONE_DEST=third_party\gemmstone"
set "REFRESH=0"
if /I "%~1"=="refresh" set "REFRESH=1"

if defined ONEDNN_SRC (
    set "ONEDNN_ROOT=!ONEDNN_SRC!"
) else (
    call :fetch_onednn
    if errorlevel 1 exit /b 1
    set "ONEDNN_ROOT=!ONEDNN_FETCH_DIR!"
)

call :validate_onednn_root "!ONEDNN_ROOT!"
if errorlevel 1 (
    echo oneDNN tree at !ONEDNN_ROOT! is not the case5 fork ^(missing nGEN, gemmstone or the Case 5 sources^).
    echo   expected: third_party\ngen\ngen.hpp
    echo             src\gpu\intel\gemm\jit\gemmstone_config.hpp
    echo             src\gpu\intel\gemm\jit\generator\pieces\case5_tile_xor.cxx
    echo   ONEDNN_REPO=!ONEDNN_REPO! ONEDNN_REF=!ONEDNN_REF!
    exit /b 1
)

set "JIT=!ONEDNN_ROOT!\src\gpu\intel\gemm\jit"
if not exist "third_party" mkdir "third_party"
call :mirror "!ONEDNN_ROOT!\third_party\ngen" "!NGEN_DEST!"
if errorlevel 1 exit /b 1
call :mirror "!JIT!\include" "!GEMMSTONE_DEST!\include"
if errorlevel 1 exit /b 1
call :mirror "!JIT!\generator" "!GEMMSTONE_DEST!\generator"
if errorlevel 1 exit /b 1
call :mirror "!JIT!\selector" "!GEMMSTONE_DEST!\selector"
if errorlevel 1 exit /b 1
robocopy "!JIT!" "!GEMMSTONE_DEST!" gemmstone_config.hpp /NFL /NDL /NJH /NJS /nc /ns /np >nul
if !ERRORLEVEL! GEQ 8 exit /b 1

set "DESC=!ONEDNN_ROOT!"
for /f %%h in ('git -C "!ONEDNN_ROOT!" rev-parse --short HEAD 2^>nul') do set "DESC=!ONEDNN_ROOT! @ %%h"
echo Case5 deps ready: !NGEN_DEST! and !GEMMSTONE_DEST! from !DESC!
exit /b 0

:validate_onednn_root
if not exist "%~1\third_party\ngen\ngen.hpp" exit /b 1
if not exist "%~1\src\gpu\intel\gemm\jit\include\gemmstone\generator.hpp" exit /b 1
if not exist "%~1\src\gpu\intel\gemm\jit\gemmstone_config.hpp" exit /b 1
if not exist "%~1\src\gpu\intel\gemm\jit\generator\pieces\case5_tile_xor.cxx" exit /b 1
exit /b 0

:fetch_onednn
set "WANT=!ONEDNN_REPO! !ONEDNN_REF!"
set "HAVE="
if exist "!FETCH_STAMP!" set /p HAVE=<"!FETCH_STAMP!"
if "!REFRESH!"=="0" if "!HAVE!"=="!WANT!" (
    call :validate_onednn_root "!ONEDNN_FETCH_DIR!"
    if not errorlevel 1 exit /b 0
)
where git >nul 2>&1
if errorlevel 1 (
    echo git not found. Install Git or set ONEDNN_SRC to a local checkout of the case5 fork.
    exit /b 1
)
if not exist "..\..\third_party" mkdir "..\..\third_party"
if exist "!ONEDNN_FETCH_DIR!" rmdir /S /Q "!ONEDNN_FETCH_DIR!"
if exist "!FETCH_STAMP!" del "!FETCH_STAMP!"
echo Cloning !ONEDNN_REPO! ^(!ONEDNN_REF!^) into !ONEDNN_FETCH_DIR! ...
git clone --depth 1 --branch !ONEDNN_REF! !ONEDNN_REPO! "!ONEDNN_FETCH_DIR!"
if errorlevel 1 (
    echo Failed to clone !ONEDNN_REPO! at !ONEDNN_REF!
    exit /b 1
)
call :validate_onednn_root "!ONEDNN_FETCH_DIR!"
if errorlevel 1 exit /b 1
echo !WANT!> "!FETCH_STAMP!"
exit /b 0

:mirror
robocopy "%~1" "%~2" /MIR /NFL /NDL /NJH /NJS /nc /ns /np >nul
set "RC=!ERRORLEVEL!"
if !RC! GEQ 8 exit /b 1
exit /b 0
