# Vendored OpenCL import library (Windows x64)

`lib/x64/OpenCL.lib` is a **vendor-neutral import library** for `OpenCL.dll` (the Khronos ICD loader).

- **Link-time:** this `.lib`
- **Headers:** `../opencl-headers` (Khronos)
- **Run-time:** `OpenCL.dll` from the GPU driver (`System32`), which loads vendor ICDs

It does **not** require NVIDIA CUDA Toolkit, Intel oneAPI, or AMD APP SDK to build.

## Regenerate (optional)

On a Windows machine with MSVC and an installed OpenCL ICD loader (`System32\OpenCL.dll`):

```powershell
# After vcvars64:
dumpbin /exports C:\Windows\System32\OpenCL.dll > exports.txt
# Build OpenCL.def EXPORTS from the name column, then:
lib /machine:x64 /def:OpenCL.def /out:lib\x64\OpenCL.lib
```