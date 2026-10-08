# Bundle ESIMD's Intel runtime, including the dynamically loaded OpenCL adapter.
# Supports the Intel conda layout used by setup_windows_oneapi.ps1.
param(
    [Parameter(Mandatory=$true)][string]$KernelDir,
    [Parameter(Mandatory=$true)][string]$RuntimeRoot,
    [Parameter(Mandatory=$true)][string]$Dumpbin,
    [string[]]$AdditionalSearchDirs=@()
)
$ErrorActionPreference='Stop'
$KernelDir=(Resolve-Path -LiteralPath $KernelDir).Path
$RuntimeRoot=(Resolve-Path -LiteralPath $RuntimeRoot).Path
$lib=Join-Path $KernelDir 'cp_esimd.dll'
if(-not(Test-Path $lib)) { throw "Missing $lib" }
$searchDirs=@((Join-Path $RuntimeRoot 'Library\bin'),$KernelDir)+$AdditionalSearchDirs
$queue=[Collections.Generic.Queue[string]]::new()
$seen=@{}
$bundled=[Collections.Generic.List[string]]::new()
$queue.Enqueue($lib)
foreach($name in 'ur_loader.dll','ur_adapter_opencl.dll') {
    # Windows' UR proxy loads the actual loader by name; neither it nor the
    # runtime-selected adapter is guaranteed to appear in the PE import table.
    $source=Join-Path $RuntimeRoot "Library\bin\$name"
    if(-not(Test-Path $source)) { throw "Dynamic UR runtime missing: $name" }
    Copy-Item -LiteralPath $source -Destination $KernelDir -Force
    $queue.Enqueue((Join-Path $KernelDir $name))
    $bundled.Add($name)
}
while($queue.Count) {
    $obj=$queue.Dequeue()
    if($seen.ContainsKey($obj)) { continue }
    $seen[$obj]=$true
    $lines=& $Dumpbin /dependents $obj
    if($LASTEXITCODE -ne 0) { throw "dumpbin failed for $obj" }
    foreach($line in $lines) {
        if($line -notmatch '^\s+([A-Za-z0-9_.-]+\.dll)\s*$') { continue }
        $name=$Matches[1]
        if($name -match '^(api-ms-|ext-ms-)') { continue }
        $source=$null
        foreach($dir in $searchDirs) {
            $candidate=Join-Path $dir $name
            if(Test-Path -LiteralPath $candidate) { $source=$candidate; break }
        }
        if(-not $source) {
            # OS libraries stay with Windows. Vendor runtimes must be found above.
            if(Test-Path (Join-Path $env:SystemRoot "System32\$name")) { continue }
            throw "Unresolved ESIMD dependency: $name (required by $obj)"
        }
        $dest=Join-Path $KernelDir $name
        if([IO.Path]::GetFullPath($source) -ne [IO.Path]::GetFullPath($dest)) {
            Copy-Item -LiteralPath $source -Destination $dest -Force
        }
        $bundled.Add($name)
        $queue.Enqueue($dest)
    }
}
# Include licenses for the packages owning bundled DLLs, plus Intel's separate
# license package. Conda also keeps upstream license files in its package cache.
$licenseDir=Join-Path $KernelDir 'oneapi-licensing'
New-Item -ItemType Directory -Force $licenseDir | Out-Null
$licenseCount=0
$dllNames=@($bundled | Sort-Object -Unique)
foreach($meta in Get-ChildItem (Join-Path $RuntimeRoot 'conda-meta') -Filter '*.json') {
    $package=Get-Content -LiteralPath $meta.FullName -Raw | ConvertFrom-Json
    $ownsDll=@($package.files | Where-Object { $dllNames -contains [IO.Path]::GetFileName($_) }).Count -gt 0
    if(-not $ownsDll -and $package.name -ne 'intel-cmplr-lic-rt') { continue }
    foreach($file in $package.files) {
        if($file -notmatch '(?i)(license|licensing|third.party)') { continue }
        $source=Join-Path $RuntimeRoot $file
        if(-not(Test-Path -LiteralPath $source -PathType Leaf)) { continue }
        $dest=Join-Path $licenseDir ($package.name+'-'+($file -replace '[\\/:]','_'))
        Copy-Item -LiteralPath $source -Destination $dest -Force
        $licenseCount++
    }
    if($package.link.source) {
        $cachedLicenses=Join-Path $package.link.source 'info\licenses'
        if(Test-Path -LiteralPath $cachedLicenses) {
            foreach($file in Get-ChildItem -LiteralPath $cachedLicenses -File -Recurse) {
                $relative=$file.FullName.Substring($cachedLicenses.Length).TrimStart('\','/')
                $dest=Join-Path $licenseDir ($package.name+'-'+($relative -replace '[\\/:]','_'))
                Copy-Item -LiteralPath $file.FullName -Destination $dest -Force
                $licenseCount++
            }
        }
    }
}
if(-not $licenseCount) { throw 'Intel redistributable license files were not found' }
$bundled | Sort-Object -Unique | Set-Content (Join-Path $KernelDir '.esimd_runtime_files')
Write-Host "Bundled $(@($bundled | Sort-Object -Unique).Count) ESIMD runtime DLLs and $licenseCount license files"
