# Install only the Windows compiler and runtime needed for cp_esimd.dll.
param([Parameter(Mandatory=$true)][string]$Prefix)
$ErrorActionPreference='Stop'
$Prefix=[IO.Path]::GetFullPath($Prefix)
$required=@('Library\bin\icx.exe','Library\lib\libircmt.lib','Library\bin\ur_loader.dll','Library\bin\ur_adapter_opencl.dll')
if(@($required | Where-Object { -not(Test-Path (Join-Path $Prefix $_)) }).Count -eq 0) { return }
$tools=Join-Path (Split-Path $Prefix -Parent) 'oneapi-bootstrap'
New-Item -ItemType Directory -Force $tools | Out-Null
$archive=Join-Path $tools 'micromamba.tar.bz2'
Invoke-WebRequest -UseBasicParsing -Uri 'https://micro.mamba.pm/api/micromamba/win-64/latest' -OutFile $archive
tar -xf $archive -C $tools
if($LASTEXITCODE -ne 0) { throw 'micromamba extraction failed' }
$mamba=Join-Path $tools 'Library\bin\micromamba.exe'
& $mamba create -y -r (Join-Path $tools 'cache') -p $Prefix --override-channels `
    -c https://software.repos.intel.com/python/conda/ -c conda-forge 'dpcpp_win-64=2026.1.1'
if($LASTEXITCODE -ne 0) { throw 'oneAPI installation failed' }
foreach($file in $required) {
    if(-not(Test-Path (Join-Path $Prefix $file))) { throw "oneAPI component missing after installation: $file" }
}
