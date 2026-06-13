#ifndef CP_CONFIG_H
#define CP_CONFIG_H

#define M_DIM  131072
#define N_DIM  131072
#define K_DIM  4096
#define R_RANK 256

#define PP_HASH_H 8
#define PP_HASH_W 16

#define PP_ROW_PERIOD 128
#define PP_COL_PERIOD 256

#define INCOMPLETE_HEADER_BYTES 76
#define HEADER_HEX_LEN (INCOMPLETE_HEADER_BYTES * 2)
#define TARGET_HEX_LEN 64

#define PLAIN_PROOF_B64_MAX (512 * 1024)

#define DEV_M_DIM 8192
#define DEV_N_DIM 8192

#define MAX_GPUS 16

/* Job return codes (mine loop). */
#define CP_PERIOD_BATCH_DEFAULT 32
#define CP_PERIOD_BATCH_MAX     512 /* 131072 cols / 256 cols per period */
#define CP_ROW_PERIOD_BATCH_DEFAULT 1
#define CP_ROW_PERIOD_BATCH_MAX   1024 /* 131072 rows / 128 rows per period */

#define CP_JOB_NONE       0
#define CP_JOB_CANCELLED (-1)

#endif /* CP_CONFIG_H */
