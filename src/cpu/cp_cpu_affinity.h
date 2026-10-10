#ifndef CP_CPU_AFFINITY_H
#define CP_CPU_AFFINITY_H

#ifdef __cplusplus
extern "C" {
#endif

/* Discover topology, restricted to the CPUs the process may use (taskset, cgroup
 * cpusets, Windows process affinity), and build the pin order: one slot per
 * logical CPU, physical cores first, then their SMT siblings. Returns 0 on
 * success (also when disabled via CP_CPU_AFFINITY=0), -1 if topology is unknown. */
int cp_cpu_affinity_init(void);

/* Pin the OpenMP worker pool to the slots built by cp_cpu_affinity_init (thread
 * id i -> slot i mod slots), including thread 0, the main thread. Skipped when
 * OMP_PLACES or OMP_PROC_BIND is set, leaving placement to the runtime. */
void cp_cpu_affinity_bind_openmp_pool(void);

/* Short summary for logs, e.g.
 * "8 physical + 8 SMT, 16 logical CPUs, 8 OpenMP threads, pinned (cores first, then SMT siblings)". */
const char *cp_cpu_affinity_summary(void);

/* Physical core count from the discovered topology; 0 if unknown or disabled.
 * OpenMP thread ids below this value sit on distinct physical cores, ids at or
 * above it are their SMT siblings. */
int cp_cpu_affinity_physical_cores(void);

/* Logical CPU count from the discovered topology (allowed CPUs only); 0 if
 * unknown or disabled. */
int cp_cpu_affinity_logical_cpus(void);

/* Pin the calling thread to the slot for OpenMP thread id `tid` (same order as
 * cp_cpu_affinity_bind_openmp_pool). For parallel regions whose team size
 * differs from the pre-bound pool, where the runtime may supply other OS
 * threads. Returns 0 on success, -1 if affinity is disabled or unknown. */
int cp_cpu_affinity_bind_thread(int tid);

/* Give the calling thread the whole process CPU set back. Helper threads (pool
 * reader, share queue, progress) call it first: on Linux they inherit the mask of
 * the pinned main thread that created them. No-op when nothing is pinned. */
void cp_cpu_affinity_release_thread(void);

#ifdef __cplusplus
}
#endif

#endif /* CP_CPU_AFFINITY_H */
