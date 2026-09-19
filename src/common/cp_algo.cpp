#include "cp_algo.h"

#include <stdio.h>
#include <string.h>

int cp_algo_parse(const char* name, CpAlgoId* out)
{
    if(!name || !out) return -1;
    if(!strcmp(name, "pearl")){
        *out = CP_ALGO_PEARL;
        return 0;
    }
    if(!strcmp(name, "quantus") || !strcmp(name, "qpow") ||
       !strcmp(name, "qpow-poseidon2")){
        *out = CP_ALGO_QUANTUS;
        return 0;
    }
    return -1;
}

const char* cp_algo_name(CpAlgoId algo)
{
    switch(algo){
    case CP_ALGO_QUANTUS: return "quantus";
    case CP_ALGO_PEARL:
    default: return "pearl";
    }
}

int cp_algo_supports(CpAlgoId algo, CpBackendId backend)
{
    if(backend == CP_BACKEND_NONE) return 0;
    if(algo == CP_ALGO_QUANTUS)
        return backend == CP_BACKEND_CPU && cp_worker_has_cpu();
    /* Pearl: any built backend. */
    switch(backend){
    case CP_BACKEND_CPU:    return cp_worker_has_cpu();
    case CP_BACKEND_CUDA:   return cp_worker_has_cuda();
    case CP_BACKEND_OPENCL: return cp_worker_has_opencl();
    case CP_BACKEND_ONEDNN: return cp_worker_has_onednn();
    default: return 0;
    }
}

void cp_algo_format_backends(CpAlgoId algo, char* buf, int buf_len)
{
    if(!buf || buf_len <= 0) return;
    buf[0] = 0;
    int first = 1;
    const CpBackendId ids[] = {
        CP_BACKEND_CPU, CP_BACKEND_CUDA, CP_BACKEND_OPENCL, CP_BACKEND_ONEDNN
    };
    const char* names[] = { "cpu", "cuda", "opencl", "onednn" };
    for(int i = 0; i < 4; i++){
        if(!cp_algo_supports(algo, ids[i])) continue;
        int n = (int)strlen(buf);
        snprintf(buf + n, (size_t)(buf_len - n), "%s%s", first ? "" : "|", names[i]);
        first = 0;
    }
    if(first)
        snprintf(buf, (size_t)buf_len, "none");
}
