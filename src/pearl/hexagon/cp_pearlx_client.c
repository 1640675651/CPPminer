#include "cp_pearlx_client.h"

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* FastRPC ABI (as in the open-source fastrpc remote.h). */
typedef uint64_t remote_handle64;

typedef struct {
    void* pv;
    size_t nLen;
} remote_buf;

typedef union {
    remote_buf buf;
    uint32_t h;
    remote_handle64 h64;
    struct {
        int fd;
        uint32_t offset;
    } dma;
} remote_arg;

#define REMOTE_SCALARS_MAKEX(nAttr, nMethod, nIn, nOut, noIn, noOut)                     \
    ((((uint32_t)(nAttr) & 0x7) << 29) | (((uint32_t)(nMethod) & 0x1f) << 24) |         \
     (((uint32_t)(nIn) & 0xff) << 16) | (((uint32_t)(nOut) & 0xff) << 8) |              \
     (((uint32_t)(noIn) & 0x0f) << 4) | ((uint32_t)(noOut) & 0x0f))

#define CDSP_DOMAIN_ID 3
#define DSPRPC_CONTROL_UNSIGNED_MODULE 2
#define RPCMEM_HEAP_ID_SYSTEM 25
#define RPCMEM_DEFAULT_FLAGS 1 /* cached */
#define RPCMEM_FLAG_UNCACHED 0

struct remote_rpc_control_unsigned_module {
    int domain;
    int enable;
};

/* Method ids follow the declaration order in pearlx.idl after open (0) and close (1). */
#define PEARLX_URI "file:///libpearlx_skel.so?pearlx_skel_handle_invoke&_modver=1.0&_dom=cdsp"
enum { PX_SET_B = 2, PX_SET_A = 3, PX_GEMM_XOR = 4, PX_INFO = 5, PX_SCAN = 6,
       PX_SET_B_GEN = 7, PX_SET_A_GEN = 8, PX_CHUNK_CVS = 9, PX_SCAN_RUN = 10 };

struct CpPearlx {
    void* lib;
    remote_handle64 h;
    int (*open)(const char*, remote_handle64*);
    int (*close)(remote_handle64);
    int (*invoke)(remote_handle64, uint32_t, remote_arg*);
    int (*session_control)(uint32_t, void*, uint32_t);
    void* (*rpcmem_alloc)(int, uint32_t, int);
    void (*rpcmem_free)(void*);
};

CpPearlx* cp_pearlx_open(void)
{
    CpPearlx* px = (CpPearlx*)calloc(1, sizeof(*px));
    if(!px) return NULL;
    px->lib = dlopen("libcdsprpc.so", RTLD_NOW | RTLD_LOCAL);
    if(!px->lib){
        fprintf(stderr, "[hexagon] cannot load libcdsprpc.so: %s\n", dlerror());
        free(px);
        return NULL;
    }
    px->open = (int (*)(const char*, remote_handle64*))dlsym(px->lib, "remote_handle64_open");
    px->close = (int (*)(remote_handle64))dlsym(px->lib, "remote_handle64_close");
    px->invoke = (int (*)(remote_handle64, uint32_t, remote_arg*))dlsym(px->lib, "remote_handle64_invoke");
    px->session_control =
        (int (*)(uint32_t, void*, uint32_t))dlsym(px->lib, "remote_session_control");
    px->rpcmem_alloc = (void* (*)(int, uint32_t, int))dlsym(px->lib, "rpcmem_alloc");
    px->rpcmem_free = (void (*)(void*))dlsym(px->lib, "rpcmem_free");
    if(!px->open || !px->close || !px->invoke || !px->rpcmem_alloc || !px->rpcmem_free){
        fprintf(stderr, "[hexagon] libcdsprpc.so lacks remote_handle64_* / rpcmem_*\n");
        dlclose(px->lib);
        free(px);
        return NULL;
    }
    /* Unsigned PD: no signed skel or root needed. Older libcdsprpc may lack the call. */
    if(px->session_control){
        struct remote_rpc_control_unsigned_module data = {CDSP_DOMAIN_ID, 1};
        int err = px->session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &data, sizeof(data));
        if(err)
            fprintf(stderr, "[hexagon] unsigned PD request failed (0x%x); trying signed PD\n", err);
    }
    int err = px->open(PEARLX_URI, &px->h);
    if(err){
        fprintf(stderr,
                "[hexagon] cannot open pearlx on the cDSP (0x%x): are libpearlx_skel.so and "
                "libworker_pool.so in the current directory or in ADSP_LIBRARY_PATH?\n", err);
        dlclose(px->lib);
        free(px);
        return NULL;
    }
    return px;
}

void cp_pearlx_close(CpPearlx* px)
{
    if(!px) return;
    px->close(px->h);
    dlclose(px->lib);
    free(px);
}

static void* alloc_flags(CpPearlx* px, size_t bytes, uint32_t flags)
{
    if(!px || bytes == 0 || bytes > 0x7fffffffu) return NULL;
    return px->rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, flags, (int)bytes);
}

void* cp_pearlx_alloc(CpPearlx* px, size_t bytes)
{
    return alloc_flags(px, bytes, RPCMEM_DEFAULT_FLAGS);
}

void* cp_pearlx_alloc_uncached(CpPearlx* px, size_t bytes)
{
    return alloc_flags(px, bytes, RPCMEM_FLAG_UNCACHED);
}

void cp_pearlx_free(CpPearlx* px, void* p)
{
    if(px && p) px->rpcmem_free(p);
}

/* Each method passes its scalar inputs in one leading buffer (32-bit words, in
 * declaration order, with each sequence's length in elements in its place), then the
 * input sequences, then one buffer of scalar outputs, then the output sequences. */

/* set_b and set_a: (in sequence<int8>, in long, in long, rout uint64). */
static int seq_two_longs(CpPearlx* px, int method, const int8_t* seq, size_t len, int x, int y,
                         uint64_t* dsp_us)
{
    if(len > 0x7fffffffu) return -1;
    uint32_t in[3] = {(uint32_t)len, (uint32_t)x, (uint32_t)y};
    uint64_t out[1] = {0};
    remote_arg ra[3];
    memset(ra, 0, sizeof(ra));
    ra[0].buf.pv = in;
    ra[0].buf.nLen = sizeof(in);
    ra[1].buf.pv = (void*)seq;
    ra[1].buf.nLen = len;
    ra[2].buf.pv = out;
    ra[2].buf.nLen = sizeof(out);
    int err = px->invoke(px->h, REMOTE_SCALARS_MAKEX(0, method, 2, 1, 0, 0), ra);
    if(dsp_us) *dsp_us = out[0];
    return err;
}

int cp_pearlx_set_b(CpPearlx* px, const int8_t* bt, int n, int k, uint64_t* dsp_us)
{
    return seq_two_longs(px, PX_SET_B, bt, (size_t)n * (size_t)k, n, k, dsp_us);
}

int cp_pearlx_set_a(CpPearlx* px, const int8_t* a, int rows, int k, int nthreads,
                    uint64_t* dsp_us)
{
    return seq_two_longs(px, PX_SET_A, a, (size_t)rows * (size_t)k, rows, nthreads, dsp_us);
}

int cp_pearlx_gemm_xor(CpPearlx* px, int col0, int ncols, uint32_t* xr, size_t xr_words,
                       uint64_t* dsp_us)
{
    if(xr_words > 0x7fffffffu / 4) return -1;
    uint32_t in[3] = {(uint32_t)col0, (uint32_t)ncols, (uint32_t)xr_words};
    uint64_t out[1] = {0};
    remote_arg ra[3];
    memset(ra, 0, sizeof(ra));
    ra[0].buf.pv = in;
    ra[0].buf.nLen = sizeof(in);
    ra[1].buf.pv = out;
    ra[1].buf.nLen = sizeof(out);
    ra[2].buf.pv = xr;
    ra[2].buf.nLen = xr_words * 4;
    int err = px->invoke(px->h, REMOTE_SCALARS_MAKEX(0, PX_GEMM_XOR, 1, 2, 0, 0), ra);
    if(dsp_us) *dsp_us = out[0];
    return err;
}

int cp_pearlx_set_b_gen(CpPearlx* px, int n, int k, const uint32_t seed_label[16],
                        const uint32_t* pairs, uint64_t* dsp_us)
{
    uint32_t in[4] = {(uint32_t)n, (uint32_t)k, 16, (uint32_t)(2 * k)};
    uint64_t out[1] = {0};
    remote_arg ra[4];
    memset(ra, 0, sizeof(ra));
    ra[0].buf.pv = in;
    ra[0].buf.nLen = sizeof(in);
    ra[1].buf.pv = (void*)seed_label;
    ra[1].buf.nLen = 16 * 4;
    ra[2].buf.pv = (void*)pairs;
    ra[2].buf.nLen = (size_t)2 * k * 4;
    ra[3].buf.pv = out;
    ra[3].buf.nLen = sizeof(out);
    int err = px->invoke(px->h, REMOTE_SCALARS_MAKEX(0, PX_SET_B_GEN, 3, 1, 0, 0), ra);
    if(dsp_us) *dsp_us = out[0];
    return err;
}

int cp_pearlx_set_a_gen(CpPearlx* px, const int8_t* a_sig, int row0, int rows, int k,
                        int nthreads, const uint32_t seed_label[16], const uint32_t* pairs,
                        uint64_t* dsp_us)
{
    const size_t len = (size_t)rows * (size_t)k;
    if(len > 0x7fffffffu) return -1;
    uint32_t in[6] = {(uint32_t)len, (uint32_t)row0, (uint32_t)rows, (uint32_t)nthreads, 16,
                      (uint32_t)(2 * k)};
    uint64_t out[1] = {0};
    remote_arg ra[5];
    memset(ra, 0, sizeof(ra));
    ra[0].buf.pv = in;
    ra[0].buf.nLen = sizeof(in);
    ra[1].buf.pv = (void*)a_sig;
    ra[1].buf.nLen = len;
    ra[2].buf.pv = (void*)seed_label;
    ra[2].buf.nLen = 16 * 4;
    ra[3].buf.pv = (void*)pairs;
    ra[3].buf.nLen = (size_t)2 * k * 4;
    ra[4].buf.pv = out;
    ra[4].buf.nLen = sizeof(out);
    int err = px->invoke(px->h, REMOTE_SCALARS_MAKEX(0, PX_SET_A_GEN, 4, 1, 0, 0), ra);
    if(dsp_us) *dsp_us = out[0];
    return err;
}

int cp_pearlx_chunk_cvs(CpPearlx* px, const int8_t* data, size_t raw_len, const uint32_t key[8],
                        uint32_t* cvs, uint64_t* dsp_us)
{
    if(raw_len > 0x7fffffffu) return -1;
    const size_t cvs_words = raw_len / 1024 * 8;
    uint32_t in[4] = {data ? (uint32_t)raw_len : 0u, (uint32_t)raw_len, 8, (uint32_t)cvs_words};
    uint64_t out[1] = {0};
    remote_arg ra[5];
    memset(ra, 0, sizeof(ra));
    ra[0].buf.pv = in;
    ra[0].buf.nLen = sizeof(in);
    ra[1].buf.pv = (void*)data;
    ra[1].buf.nLen = data ? raw_len : 0;
    ra[2].buf.pv = (void*)key;
    ra[2].buf.nLen = 8 * 4;
    ra[3].buf.pv = out;
    ra[3].buf.nLen = sizeof(out);
    ra[4].buf.pv = cvs;
    ra[4].buf.nLen = cvs_words * 4;
    int err = px->invoke(px->h, REMOTE_SCALARS_MAKEX(0, PX_CHUNK_CVS, 3, 2, 0, 0), ra);
    if(dsp_us) *dsp_us = out[0];
    return err;
}

int cp_pearlx_scan_run(CpPearlx* px, const int8_t* a_sig, int row0, int rows, int k,
                       int panel_rows, int launch_cols, int s0, int nbuf, int nthreads,
                       const uint32_t seed_label[16], const uint32_t* pairs, uint32_t* ctl,
                       uint32_t* xr, size_t xr_words, uint64_t* dsp_us, uint64_t* wait_us,
                       int* launched)
{
    const size_t len = (size_t)rows * (size_t)k;
    if(len > 0x7fffffffu || xr_words > 0x7fffffffu / 4) return -1;
    uint32_t in[12] = {(uint32_t)len, (uint32_t)row0, (uint32_t)rows, (uint32_t)panel_rows,
                       (uint32_t)launch_cols, (uint32_t)s0, (uint32_t)nbuf, (uint32_t)nthreads,
                       16, (uint32_t)(2 * k), CP_PEARLX_CTL_WORDS, (uint32_t)xr_words};
    uint64_t out[3] = {0, 0, 0};   /* dsp_us, wait_us, launched */
    remote_arg ra[7];
    memset(ra, 0, sizeof(ra));
    ra[0].buf.pv = in;
    ra[0].buf.nLen = sizeof(in);
    ra[1].buf.pv = (void*)a_sig;
    ra[1].buf.nLen = len;
    ra[2].buf.pv = (void*)seed_label;
    ra[2].buf.nLen = 16 * 4;
    ra[3].buf.pv = (void*)pairs;
    ra[3].buf.nLen = (size_t)2 * k * 4;
    ra[4].buf.pv = out;
    ra[4].buf.nLen = sizeof(out);
    ra[5].buf.pv = ctl;
    ra[5].buf.nLen = CP_PEARLX_CTL_WORDS * 4;
    ra[6].buf.pv = xr;
    ra[6].buf.nLen = xr_words * 4;
    int err = px->invoke(px->h, REMOTE_SCALARS_MAKEX(0, PX_SCAN_RUN, 4, 3, 0, 0), ra);
    if(dsp_us) *dsp_us = out[0];
    if(wait_us) *wait_us = out[1];
    if(launched) memcpy(launched, &out[2], sizeof(int));
    return err;
}

int cp_pearlx_info(CpPearlx* px, int* hvx_contexts, int* clock_vote_err, int* clock_mhz,
                   int* vtcm_kb)
{
    uint32_t out[4] = {0, 0, 0, 0};
    remote_arg ra[1];
    memset(ra, 0, sizeof(ra));
    ra[0].buf.pv = out;
    ra[0].buf.nLen = sizeof(out);
    int err = px->invoke(px->h, REMOTE_SCALARS_MAKEX(0, PX_INFO, 0, 1, 0, 0), ra);
    *hvx_contexts = (int)out[0];
    *clock_vote_err = (int)out[1];
    *clock_mhz = (int)out[2];
    *vtcm_kb = (int)out[3];
    return err;
}
