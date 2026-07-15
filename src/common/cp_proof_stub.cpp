/* Fallback when rust/cp-proof-ffi was not built. */
#include "cp_proof.h"

#include <stdio.h>
#include <string.h>

int cp_proof_build(
    const uint8_t* header,
    size_t header_len,
    const uint8_t* mining_config,
    size_t config_len,
    const int8_t* a,
    const int8_t* bt,
    int m,
    int n,
    int k,
    int rank,
    int t_rows,
    int t_cols,
    int tile_layout,
    char* out_b64,
    size_t out_cap,
    char* err,
    size_t err_cap)
{
    (void)header; (void)header_len; (void)mining_config; (void)config_len;
    (void)a; (void)bt; (void)m; (void)n; (void)k; (void)rank;
    (void)t_rows; (void)t_cols; (void)tile_layout; (void)out_b64; (void)out_cap;
    if(err && err_cap)
        snprintf(err, err_cap, "cp_proof_ffi not linked (build rust/cp-proof-ffi)");
    return -1;
}

int cp_proof_verify(
    const uint8_t* header,
    size_t header_len,
    const uint8_t* proof_b64,
    size_t proof_b64_len,
    const uint8_t* pool_target_be,
    char* err,
    size_t err_cap)
{
    (void)header; (void)header_len; (void)proof_b64; (void)proof_b64_len;
    (void)pool_target_be;
    if(err && err_cap)
        snprintf(err, err_cap, "cp_proof_ffi not linked (build rust/cp-proof-ffi)");
    return -1;
}
