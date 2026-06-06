#ifndef CP_PROOF_H
#define CP_PROOF_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Build plain_proof base64 in-process (Rust/pearl-blake3). Returns 0 on ok, -1 on error. */
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
    int contiguous_tiles,
    char* out_b64,
    size_t out_cap,
    char* err,
    size_t err_cap);

#ifdef __cplusplus
}
#endif

#endif /* CP_PROOF_H */
