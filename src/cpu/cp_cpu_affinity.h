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

#ifdef __cplusplus
}
#endif

#endif /* CP_CPU_AFFINITY_H */
