#ifndef CP_OPENCL_PREP_PROFILE_H
#define CP_OPENCL_PREP_PROFILE_H

#ifdef __cplusplus
extern "C" {
#endif

/* Time OpenCL matrix prep phases (random A/B, keyed hashes, noise fusion). */
int cp_opencl_run_prep_profile(int device_index, int m, int n, int warmup, int runs);

#ifdef __cplusplus
}
#endif

#endif /* CP_OPENCL_PREP_PROFILE_H */
