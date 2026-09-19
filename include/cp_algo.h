#ifndef CP_ALGO_H
#define CP_ALGO_H

#include "cp_worker.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    CP_ALGO_PEARL   = 0,
    CP_ALGO_QUANTUS = 1
} CpAlgoId;

/* Parse "pearl" / "quantus". Returns 0 on ok, -1 on unknown. */
int cp_algo_parse(const char* name, CpAlgoId* out);

const char* cp_algo_name(CpAlgoId algo);

/* 1 if this algo can run on the given backend in this build. */
int cp_algo_supports(CpAlgoId algo, CpBackendId backend);

/* Print supported backends for an algo into buf (NUL-terminated). */
void cp_algo_format_backends(CpAlgoId algo, char* buf, int buf_len);

#ifdef __cplusplus
}
#endif

#endif /* CP_ALGO_H */
