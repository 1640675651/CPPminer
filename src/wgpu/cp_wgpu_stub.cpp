/* Stub when CP_ENABLE_WGPU is on but the Rust lib was not linked. */
#include "cp_wgpu.h"

#include <stdio.h>

extern "C" int cp_wgpu_list_devices(void)
{
    fprintf(stderr, "[wgpu] stub: cp-wgpu-ffi not linked (build with cargo)\n");
    return 0;
}

extern "C" int cp_wgpu_init(uint32_t batch_size, int allow_integrated,
                            const int* devices, int ndev)
{
    (void)batch_size;
    (void)allow_integrated;
    (void)devices;
    (void)ndev;
    fprintf(stderr, "[wgpu] stub: cp-wgpu-ffi not linked (build with cargo)\n");
    return -1;
}

extern "C" void cp_wgpu_shutdown(void) {}

extern "C" int cp_wgpu_is_ready(void) { return 0; }

extern "C" int cp_wgpu_search_range(
    const uint8_t* header,
    uint64_t difficulty_u64,
    const uint8_t* target_be,
    const uint8_t* start_nonce_be,
    uint64_t count,
    uint8_t* out_nonce_be,
    uint8_t* out_hash_be,
    uint64_t* out_hashes,
    int (*cancel_check)(void))
{
    (void)header;
    (void)difficulty_u64;
    (void)target_be;
    (void)start_nonce_be;
    (void)count;
    (void)out_nonce_be;
    (void)out_hash_be;
    (void)cancel_check;
    if(out_hashes) *out_hashes = 0;
    return CP_WGPU_ERROR;
}
