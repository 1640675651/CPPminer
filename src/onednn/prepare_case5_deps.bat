@echo off
REM Old name of prepare_onednn_deps.bat (vendors nGEN + gemmstone from the case5 oneDNN fork).
call "%~dp0prepare_onednn_deps.bat" %*
exit /b %ERRORLEVEL%
