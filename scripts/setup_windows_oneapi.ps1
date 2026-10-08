# Install only the Windows compiler and runtime needed for cp_esimd.dll.
param([Parameter(Mandatory=$true)][string]$Prefix)
$ErrorActionPreference='Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12   # Windows PowerShell 5.1 default is too old for github.com
$Prefix=[IO.Path]::GetFullPath($Prefix)
$required=@('Library\bin\icx.exe','Library\lib\libircmt.lib','Library\bin\ur_loader.dll','Library\bin\ur_adapter_opencl.dll')
if(@($required | Where-Object { -not(Test-Path (Join-Path $Prefix $_)) }).Count -eq 0) { return }
$tools=Join-Path (Split-Path $Prefix -Parent) 'oneapi-bootstrap'
New-Item -ItemType Directory -Force $tools | Out-Null
# micromamba as a plain executable: the .tar.bz2 from micro.mamba.pm needs an external bzip2 that
# Windows' tar.exe (bsdtar) does not have, so `tar -xf` hangs forever (seen on the CI runner).
$mamba=Join-Path $tools 'micromamba.exe'
if(-not(Test-Path $mamba)) {
    Invoke-WebRequest -UseBasicParsing -TimeoutSec 300 `
        -Uri 'https://github.com/mamba-org/micromamba-releases/releases/download/2.9.0-0/micromamba-win-64.exe' -OutFile $mamba
}
& $mamba --version
if($LASTEXITCODE -ne 0) { throw 'micromamba does not run' }
& $mamba create -y --no-rc -r (Join-Path $tools 'cache') -p $Prefix --override-channels `
    -c https://software.repos.intel.com/python/conda/ -c conda-forge 'dpcpp_win-64=2026.1.1'
if($LASTEXITCODE -ne 0) { throw 'oneAPI installation failed' }
foreach($file in $required) {
    if(-not(Test-Path (Join-Path $Prefix $file))) { throw "oneAPI component missing after installation: $file" }
}
