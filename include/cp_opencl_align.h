#ifndef CP_OPENCL_ALIGN_H
#define CP_OPENCL_ALIGN_H

#ifdef __cplusplus
extern "C" {
#endif

/* GPU vs CPU alignment tests for OpenCL prep (keyed hash, noise, prepack). */
int cp_opencl_run_alignment_tests(int device_index, int m, int n);

/* Fused GEMM kernel vs CPU reference: every milestone word (prefix-GEMM XOR of each
   hash tile) of a small random M x N problem, with the configured --ocl-dot backend. */
int cp_opencl_run_gemm_align_test(int device_index);

#ifdef __cplusplus
}
#endif

#endif /* CP_OPENCL_ALIGN_H */
