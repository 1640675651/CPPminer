#include "cp_wgpu_worker.h"

#include "cp_job_ctrl.h"
#include "cp_pool.h"
#include "cp_wgpu.h"

#include <stdio.h>

static int g_wgpu_ready = 0;
static uint32_t g_batch_size = 1000000u;

extern "C" void cp_wgpu_worker_set_batch_size(uint32_t batch)
{
    g_batch_size = batch == 0 ? 1000000u : batch;
}

extern "C" uint32_t cp_wgpu_worker_batch_size(void)
{
    return g_batch_size;
}

static int wgpu_cancel_check(void)
{
    return cp_job_should_cancel() || cp_pool_conn_lost();
}

extern "C" int cp_wgpu_worker_init(int* devices, int ndev)
{
    /* Prefer discrete (allow_integrated=0) unless --devices picks adapters. */
    if(cp_wgpu_init(g_batch_size, 0, devices, ndev) != 0){
        fprintf(stderr, "[wgpu] worker init failed\n");
        g_wgpu_ready = 0;
        return -1;
    }
    g_wgpu_ready = 1;
    if(devices && ndev > 0){
        printf("[wgpu] GpuEngine ready (batch=%u, devices=", g_batch_size);
        for(int i = 0; i < ndev; i++)
            printf("%s%d", i ? "," : "", devices[i]);
        printf(")\n");
    } else {
        printf("[wgpu] GpuEngine ready (batch=%u, devices=auto)\n", g_batch_size);
    }
    fflush(stdout);
    return 0;
}

extern "C" void cp_wgpu_worker_shutdown(void)
{
    if(g_wgpu_ready){
        cp_wgpu_shutdown();
        g_wgpu_ready = 0;
    }
}

extern "C" int cp_wgpu_worker_is_ready(void)
{
    return g_wgpu_ready && cp_wgpu_is_ready();
}

extern "C" int cp_wgpu_worker_list_devices(void)
{
    return cp_wgpu_list_devices();
}

extern "C" int cp_wgpu_worker_search(
    const uint8_t header[32],
    uint64_t difficulty_u64,
    const uint8_t target[64],
    const uint8_t start[64],
    uint64_t count,
    uint8_t out_nonce[64],
    uint8_t out_hash[64],
    uint64_t* out_hashes)
{
    if(!g_wgpu_ready) return CP_WGPU_ERROR;
    return cp_wgpu_search_range(header, difficulty_u64, target, start, count,
                                out_nonce, out_hash, out_hashes, wgpu_cancel_check);
}
