@echo off
setlocal EnableDelayedExpansion

cd /d "%~dp0"

REM oneDNN source: ONEDNN_SRC if set, else clone into project third_party\onednn-src.
if not defined ONEDNN_TAG set "ONEDNN_TAG=v3.13.2"
set "ONEDNN_FETCH_DIR=..\..\third_party\onednn-src"
set "NGEN_DEST=third_party\ngen"
set "GEMMSTONE_DEST=third_party\gemmstone"
set "PATCHES=case5_patches"
set "TAG_STAMP=..\..\third_party\.case5_onednn_tag"
set "REFRESH=0"

if /I "%~1"=="refresh" set "REFRESH=1"

if "!REFRESH!"=="0" (
    if exist "!NGEN_DEST!\ngen.hpp" if exist "!GEMMSTONE_DEST!\include\gemmstone\generator.hpp" (
        if exist "!TAG_STAMP!" (
            set /p VENDORED_TAG=<"!TAG_STAMP!"
            if /I "!VENDORED_TAG!"=="!ONEDNN_TAG!" (
                echo Case5 deps already vendored ^(!ONEDNN_TAG!^); re-applying patches...
                call :apply_patches
                if errorlevel 1 exit /b 1
                exit /b 0
            )
            echo Case5 deps vendored for !VENDORED_TAG! but ONEDNN_TAG=!ONEDNN_TAG!; re-vendoring...
        ) else (
            echo Case5 deps present but tag stamp missing; re-vendoring for !ONEDNN_TAG!...
        )
        set "REFRESH=1"
    )
)

set "ONEDNN_ROOT="
if defined ONEDNN_SRC (
    set "ONEDNN_ROOT=!ONEDNN_SRC!"
    call :validate_onednn_root "!ONEDNN_ROOT!"
    if errorlevel 1 (
        echo ONEDNN_SRC=!ONEDNN_SRC! is missing gemmstone JIT or third_party\ngen.
        echo Use oneDNN v3.10 or newer ^(default fetch tag: !ONEDNN_TAG!^).
        exit /b 1
    )
) else (
    call :fetch_onednn
    if errorlevel 1 exit /b 1
    set "ONEDNN_ROOT=!ONEDNN_FETCH_DIR!"
)

call :validate_onednn_root "!ONEDNN_ROOT!"
if errorlevel 1 (
    echo oneDNN checkout at !ONEDNN_ROOT! is missing required Case5 sources.
    echo   expected: third_party\ngen\ngen.hpp
    echo             src\gpu\intel\gemm\jit\include\gemmstone\generator.hpp
    echo If !ONEDNN_FETCH_DIR! was cloned with an old tag ^(e.g. v3.7.2^), delete it and re-run:
    echo   rmdir /S /Q !ONEDNN_FETCH_DIR!
    echo   prepare_onednn_deps.bat refresh
    exit /b 1
)

echo Vendoring Case5 deps from oneDNN: !ONEDNN_ROOT!

if not exist "third_party" mkdir "third_party"

call :robocopy_tree "!ONEDNN_ROOT!\third_party\ngen" "!NGEN_DEST!"
if errorlevel 1 exit /b 1

if exist "!GEMMSTONE_DEST!" rmdir /S /Q "!GEMMSTONE_DEST!"
mkdir "!GEMMSTONE_DEST!"

call :robocopy_tree "!ONEDNN_ROOT!\src\gpu\intel\gemm\jit\include" "!GEMMSTONE_DEST!\include"
if errorlevel 1 exit /b 1
call :robocopy_tree "!ONEDNN_ROOT!\src\gpu\intel\gemm\jit\generator" "!GEMMSTONE_DEST!\generator"
if errorlevel 1 exit /b 1

call :apply_patches
if errorlevel 1 exit /b 1

echo !ONEDNN_TAG!> "!TAG_STAMP!"

echo Case5 deps ready: !NGEN_DEST! and !GEMMSTONE_DEST! ^(oneDNN !ONEDNN_TAG!^)
exit /b 0

:apply_patches
if not exist "!PATCHES!\gemmstone\gemmstone_config.hpp" (
    echo Missing Case5 patch overlay: !PATCHES!\gemmstone\gemmstone_config.hpp
    exit /b 1
)
echo Applying Case5 patches from !PATCHES! ...
call :robocopy_tree "!PATCHES!\ngen" "!NGEN_DEST!"
if errorlevel 1 exit /b 1
call :robocopy_tree "!PATCHES!\gemmstone" "!GEMMSTONE_DEST!"
if errorlevel 1 exit /b 1
exit /b 0

:validate_onednn_root
if not exist "%~1\third_party\ngen\ngen.hpp" exit /b 1
if not exist "%~1\src\gpu\intel\gemm\jit\include\gemmstone\generator.hpp" exit /b 1
exit /b 0

:fetch_onednn
call :validate_onednn_root "!ONEDNN_FETCH_DIR!"
if not errorlevel 1 (
    echo Using fetched oneDNN at !ONEDNN_FETCH_DIR!
    exit /b 0
)

where git >nul 2>&1
if errorlevel 1 (
    echo git not found. Install Git or set ONEDNN_SRC to a local oneDNN checkout.
    exit /b 1
)

for %%D in ("!ONEDNN_FETCH_DIR!") do if not exist "%%~dpD" mkdir "%%~dpD"

if exist "!ONEDNN_FETCH_DIR!" (
    echo Removing incomplete or outdated oneDNN tree at !ONEDNN_FETCH_DIR! ...
    rmdir /S /Q "!ONEDNN_FETCH_DIR!"
)

echo Cloning oneDNN !ONEDNN_TAG! into !ONEDNN_FETCH_DIR! ...
git clone --depth 1 --branch !ONEDNN_TAG! https://github.com/uxlfoundation/oneDNN.git "!ONEDNN_FETCH_DIR!"
if errorlevel 1 (
    echo Failed to clone oneDNN !ONEDNN_TAG!
    exit /b 1
)

call :validate_onednn_root "!ONEDNN_FETCH_DIR!"
if errorlevel 1 (
    echo Cloned oneDNN !ONEDNN_TAG! is missing gemmstone JIT or third_party\ngen.
    exit /b 1
)
exit /b 0

:robocopy_tree
robocopy "%~1" "%~2" /E /NFL /NDL /NJH /NJS /nc /ns /np >nul
set "RC=!ERRORLEVEL!"
if !RC! GEQ 8 exit /b 1
exit /b 0
