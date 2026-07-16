#ifndef CP_OPENCL_ALIGN_H
#define CP_OPENCL_ALIGN_H

#ifdef __cplusplus
extern "C" {
#endif

/* GPU vs CPU alignment tests for OpenCL prep (keyed hash, noise, prepack). */
int cp_opencl_run_alignment_tests(int device_index, int m, int n);

#ifdef __cplusplus
}
#endif

#endif /* CP_OPENCL_ALIGN_H */
