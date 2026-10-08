#ifndef CP_ESIMD_SCAN_H
#define CP_ESIMD_SCAN_H

/* C ABI of libcp_esimd: the ESIMD XMX scan kernel (GEMM + milestone tile XOR +
 * fold + keyed BLAKE3 + target compare in one pass), launched on the miner's
 * own OpenCL context and queue through SYCL interop. Loaded with dlopen, so
 * the miner itself needs no oneAPI toolchain. */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CP_ESIMD_ABI_VERSION 2
#define CP_ESIMD_K 4096
#define CP_ESIMD_HASH_TILE 16

typedef struct CpEsimdScan CpEsimdScan;

typedef struct CpEsimdInfo {
    int exec_size; /* DPAS execution size: B blocks are 32 x exec_size VNNI */
    int tile_m;    /* rows per work group; panels must be multiples */
    int tile_n;    /* cols per work group */
    int dpasw;
} CpEsimdInfo;

/* OpenCL handles are passed as void* (cl_context, cl_device_id,
 * cl_command_queue, cl_mem) to keep this header free of CL headers. */
typedef int (*cp_esimd_abi_version_fn)(void);
typedef CpEsimdScan *(*cp_esimd_create_fn)(void *cl_ctx, void *cl_dev, void *cl_queue,
                                           CpEsimdInfo *info, char *err, int err_len);
/* a_blocked: noisy A in 8x32 DPAS blocks, [m/8][k/32] (M_total rows);
 * bt_vnni: noisy B^T in 32 x exec_size VNNI blocks, [n/es][k/32];
 * found: int[4] {flag, t_rows, t_cols, 0}; the first hit sets it.
 * Enqueues the panel [m0, m0+m) x [n0, n0+n) asynchronously; *done (if not NULL)
 * receives a retained cl_event of the kernel for the caller's wait lists. */
typedef int (*cp_esimd_scan_panel_fn)(CpEsimdScan *s, void *a_blocked, void *bt_vnni,
                                      void *found, int m0, int n0, int m, int n,
                                      const uint32_t key[8], const uint32_t bound[8],
                                      void **done);
/* Waits for every panel submitted so far. */
typedef void (*cp_esimd_wait_fn)(CpEsimdScan *s);
typedef void (*cp_esimd_destroy_fn)(CpEsimdScan *s);

#ifdef __cplusplus
}
#endif

#endif /* CP_ESIMD_SCAN_H */
