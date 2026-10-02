#ifndef CP_CPU_AFFINITY_H
#define CP_CPU_AFFINITY_H

#ifdef __cplusplus
extern "C" {
#endif

/* Discover topology and build the pin order. smt != 0: one slot per logical
 * CPU, physical cores first then their SMT siblings. smt == 0: one slot per
 * physical core, each covering all of that core's siblings. Returns 0 on
 * success (also when disabled via CP_CPU_AFFINITY=0), -1 if topology is unknown. */
int cp_cpu_affinity_init(int smt);

/* Pin the OpenMP worker pool to the slots built by cp_cpu_affinity_init (thread
 * id i -> slot i mod slots). Skipped when OMP_PLACES or OMP_PROC_BIND is set,
 * leaving placement to the runtime. On Linux the main thread's original mask is
 * restored afterwards so later std::threads do not inherit a one-CPU mask. */
void cp_cpu_affinity_bind_openmp_pool(void);

/* Short summary for logs, e.g.
 * "8 physical + 8 SMT, 16 logical CPUs, 8 OpenMP threads, pinned 1/core". */
const char *cp_cpu_affinity_summary(void);

/* Physical core count from the discovered topology; 0 if unknown or disabled.
 * With smt slots (cp_cpu_affinity_init(1)) OpenMP thread ids below this value
 * sit on distinct physical cores, ids at or above it are their SMT siblings. */
int cp_cpu_affinity_physical_cores(void);

/* Logical CPU count from the discovered topology; 0 if unknown or disabled. */
int cp_cpu_affinity_logical_cpus(void);

/* Pin the calling thread to the slot for OpenMP thread id `tid` (same order as
 * cp_cpu_affinity_bind_openmp_pool). For parallel regions whose team size
 * differs from the pre-bound pool, where the runtime may supply other OS
 * threads. Returns 0 on success, -1 if affinity is disabled or unknown. */
int cp_cpu_affinity_bind_thread(int tid);

#ifdef __cplusplus
}
#endif

#endif /* CP_CPU_AFFINITY_H */
