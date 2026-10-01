#ifndef CP_CPU_AFFINITY_H
#define CP_CPU_AFFINITY_H

#ifdef __cplusplus
extern "C" {
#endif

/* Discover topology and build physical-first CPU order. Returns 0 on success. */
int cp_cpu_affinity_init(void);

/* Pin the OpenMP worker pool (physical cores, then SMT siblings). */
void cp_cpu_affinity_bind_openmp_pool(void);

/* Short summary for logs, e.g. "8 physical + 8 SMT, 16 OpenMP threads". */
const char *cp_cpu_affinity_summary(void);

/* Physical core count from the discovered topology; 0 if unknown or disabled.
 * OpenMP thread ids below this value sit on distinct physical cores, ids at or
 * above it are their SMT siblings (see cp_cpu_affinity_bind_openmp_pool). */
int cp_cpu_affinity_physical_cores(void);

/* Pin the calling thread to the slot for OpenMP thread id `tid` (same order as
 * cp_cpu_affinity_bind_openmp_pool). For parallel regions whose team size
 * differs from the pre-bound pool, where the runtime may supply other OS
 * threads. Returns 0 on success, -1 if affinity is disabled or unknown. */
int cp_cpu_affinity_bind_thread(int tid);

#ifdef __cplusplus
}
#endif

#endif /* CP_CPU_AFFINITY_H */
