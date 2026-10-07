#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif

#include "cp_cpu_affinity.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#elif defined(__linux__)
#include <dirent.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <pthread.h>
#include <sys/sysctl.h>
#include <unistd.h>
#endif

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace {

/* One pin target: a single logical CPU. */
struct CpuSlot {
#if defined(_WIN32)
    WORD group = 0;
    KAFFINITY mask = 0;
#else
    std::vector<int> cpus;
#endif
};

struct CoreGroup {
    std::vector<int> logical;
};

enum class BindMode { None, Pinned, Runtime };

static std::vector<CpuSlot> g_cpu_order;
static int g_physical_cores = 0;
static int g_logical_cpus = 0;
static BindMode g_bind_mode = BindMode::None;
static char g_summary[200] = "disabled";

#if defined(__linux__)
/* The CPUs the process may run on (taskset, cgroup cpusets), read once before
 * anything is pinned. Topology is restricted to it, and helper threads return to
 * it (cp_cpu_affinity_release_thread). */
static cpu_set_t g_process_mask;
static bool g_have_process_mask = false;
#endif

static bool affinity_disabled(void) {
    const char *env = std::getenv("CP_CPU_AFFINITY");
    return env && (env[0] == '0' || env[0] == 'n' || env[0] == 'N');
}

/* OMP_PLACES / OMP_PROC_BIND mean the user told the OpenMP runtime where to put
 * its threads; our pins would fight that, so we stand down. */
static bool runtime_binds_threads(void) {
    const char *places = std::getenv("OMP_PLACES");
    const char *bind = std::getenv("OMP_PROC_BIND");
    return (places && *places) || (bind && *bind);
}

static CpuSlot make_slot(int cpu) {
    CpuSlot slot{};
#if defined(_WIN32)
    slot.group = static_cast<WORD>(cpu / 64);
    slot.mask = KAFFINITY(1) << (cpu % 64);
#else
    slot.cpus = {cpu};
#endif
    return slot;
}

/* One slot per logical CPU, physical cores first, then their SMT siblings: OpenMP
 * thread ids below the core count sit on distinct cores (Quantus hybrid relies
 * on that), so --threads N up to the core count never doubles up on a core. */
static void build_cpu_order(const std::vector<CoreGroup> &cores) {
    g_cpu_order.clear();
    g_physical_cores = static_cast<int>(cores.size());

    size_t max_smt = 1;
    size_t logical = 0;
    for (const CoreGroup &core : cores) {
        max_smt = std::max(max_smt, core.logical.size());
        logical += core.logical.size();
    }
    g_logical_cpus = static_cast<int>(logical);

    for (size_t s = 0; s < max_smt; ++s) {
        for (const CoreGroup &core : cores) {
            if (s < core.logical.size()) {
                g_cpu_order.push_back(make_slot(core.logical[s]));
            }
        }
    }
}

template <typename Allowed>
static void keep_allowed(std::vector<CoreGroup> *cores, Allowed allowed) {
    std::vector<CoreGroup> kept;
    for (const CoreGroup &core : *cores) {
        CoreGroup c;
        for (int cpu : core.logical) {
            if (allowed(cpu)) {
                c.logical.push_back(cpu);
            }
        }
        if (!c.logical.empty()) {
            kept.push_back(std::move(c));
        }
    }
    if (!kept.empty()) {
        *cores = std::move(kept);
    }
}

/* Keep only the CPUs this process may run on (taskset, cgroup cpusets, Windows
 * process affinity), dropping cores left with none, so the default thread count
 * and the pins stay inside the allowed set. */
static void restrict_to_process_mask(std::vector<CoreGroup> *cores) {
#if defined(_WIN32)
    DWORD_PTR process = 0, system = 0;
    if (!GetProcessAffinityMask(GetCurrentProcess(), &process, &system) || process == 0) {
        return;
    }
    /* The mask covers the process's own processor group only; leave machines
     * with more than one group (>64 logical CPUs) unfiltered. */
    for (const CoreGroup &core : *cores) {
        for (int cpu : core.logical) {
            if (cpu >= 64) {
                return;
            }
        }
    }
    keep_allowed(cores, [&](int cpu) { return ((process >> cpu) & 1) != 0; });
#elif defined(__linux__)
    if (!g_have_process_mask) {
        CPU_ZERO(&g_process_mask);
        g_have_process_mask = sched_getaffinity(0, sizeof(g_process_mask), &g_process_mask) == 0;
    }
    if (g_have_process_mask) {
        keep_allowed(cores, [](int cpu) {
            return cpu >= 0 && cpu < CPU_SETSIZE && CPU_ISSET(cpu, &g_process_mask);
        });
    }
#else
    (void)cores; /* no per-process CPU masks */
#endif
}

#if defined(_WIN32)

static int flat_processor_index(WORD group, int bit) {
    return static_cast<int>(group) * 64 + bit;
}

static bool win32_collect_cores(std::vector<CoreGroup> *out) {
    DWORD bytes = 0;
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &bytes) &&
        GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
        return false;
    }

    std::vector<unsigned char> buffer(bytes);
    if (!GetLogicalProcessorInformationEx(
                RelationProcessorCore,
                reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data()),
                &bytes)) {
        return false;
    }

    out->clear();
    unsigned char *cursor = buffer.data();
    unsigned char *end = buffer.data() + bytes;
    while (cursor < end) {
        auto *info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(cursor);
        if (info->Relationship == RelationProcessorCore) {
            CoreGroup core;
            for (WORD g = 0; g < info->Processor.GroupCount; ++g) {
                const GROUP_AFFINITY &ga = info->Processor.GroupMask[g];
                KAFFINITY mask = ga.Mask;
                for (int bit = 0; bit < 64; ++bit) {
                    if (mask & (KAFFINITY(1) << bit)) {
                        core.logical.push_back(flat_processor_index(ga.Group, bit));
                    }
                }
            }
            std::sort(core.logical.begin(), core.logical.end());
            if (!core.logical.empty()) {
                out->push_back(std::move(core));
            }
        }
        cursor += info->Size;
    }

    std::sort(out->begin(), out->end(), [](const CoreGroup &a, const CoreGroup &b) {
        return a.logical[0] < b.logical[0];
    });
    return !out->empty();
}

static bool bind_current_thread(const CpuSlot &slot) {
    GROUP_AFFINITY ga{};
    ga.Group = slot.group;
    ga.Mask = slot.mask;
    return SetThreadGroupAffinity(GetCurrentThread(), &ga, nullptr) != 0;
}

#elif defined(__linux__)

static bool read_int_file(const char *path, int *out) {
    FILE *f = std::fopen(path, "r");
    if (!f) {
        return false;
    }
    const int ok = (std::fscanf(f, "%d", out) == 1);
    std::fclose(f);
    return ok;
}

static bool parse_cpu_list(const char *text, std::vector<int> *cpus) {
    cpus->clear();
    if (!text || !*text) {
        return false;
    }
    const char *p = text;
    while (*p) {
        while (*p == ' ' || *p == '\t' || *p == '\n' || *p == ',') {
            ++p;
        }
        if (!*p) {
            break;
        }
        char *end = nullptr;
        const long a = std::strtol(p, &end, 10);
        if (end == p) {
            return false;
        }
        p = end;
        long b = a;
        if (*p == '-') {
            ++p;
            b = std::strtol(p, &end, 10);
            if (end == p) {
                return false;
            }
            p = end;
        }
        for (long cpu = a; cpu <= b; ++cpu) {
            cpus->push_back(static_cast<int>(cpu));
        }
    }
    std::sort(cpus->begin(), cpus->end());
    cpus->erase(std::unique(cpus->begin(), cpus->end()), cpus->end());
    return !cpus->empty();
}

static bool linux_collect_cores(std::vector<CoreGroup> *out) {
    DIR *dir = opendir("/sys/devices/system/cpu");
    if (!dir) {
        return false;
    }

    struct CpuRec {
        int cpu = -1;
        int core_id = -1;
        std::vector<int> siblings;
    };
    std::vector<CpuRec> records;

    for (dirent *ent = readdir(dir); ent; ent = readdir(dir)) {
        if (std::strncmp(ent->d_name, "cpu", 3) != 0) {
            continue;
        }
        const char *num = ent->d_name + 3;
        if (*num < '0' || *num > '9') {
            continue;
        }
        char *end = nullptr;
        const long cpu = std::strtol(num, &end, 10);
        if (!end || *end != '\0' || cpu < 0) {
            continue;
        }

        char path[256];
        std::snprintf(path, sizeof(path),
                      "/sys/devices/system/cpu/cpu%ld/topology/core_id", cpu);
        int core_id = -1;
        if (!read_int_file(path, &core_id)) {
            continue;
        }

        std::snprintf(path, sizeof(path),
                      "/sys/devices/system/cpu/cpu%ld/topology/thread_siblings_list",
                      cpu);
        FILE *sf = std::fopen(path, "r");
        if (!sf) {
            continue;
        }
        char siblings_buf[256] = {};
        if (!std::fgets(siblings_buf, sizeof(siblings_buf), sf)) {
            std::fclose(sf);
            continue;
        }
        std::fclose(sf);

        CpuRec rec;
        rec.cpu = static_cast<int>(cpu);
        rec.core_id = core_id;
        if (!parse_cpu_list(siblings_buf, &rec.siblings)) {
            rec.siblings = {rec.cpu};
        }
        records.push_back(std::move(rec));
    }
    closedir(dir);

    if (records.empty()) {
        return false;
    }

    std::vector<std::vector<int>> seen;
    out->clear();
    for (const CpuRec &rec : records) {
        bool dup = false;
        for (const std::vector<int> &s : seen) {
            if (s == rec.siblings) {
                dup = true;
                break;
            }
        }
        if (dup) {
            continue;
        }
        seen.push_back(rec.siblings);
        CoreGroup core;
        core.logical = rec.siblings;
        std::sort(core.logical.begin(), core.logical.end());
        out->push_back(std::move(core));
    }

    std::sort(out->begin(), out->end(), [](const CoreGroup &a, const CoreGroup &b) {
        return a.logical[0] < b.logical[0];
    });
    return !out->empty();
}

static bool bind_current_thread(const CpuSlot &slot) {
    if (slot.cpus.empty()) {
        return false;
    }
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int cpu : slot.cpus) {
        if (cpu >= 0 && cpu < CPU_SETSIZE) {
            CPU_SET(cpu, &set);
        }
    }
    /* pid 0 targets the calling thread on Linux, including Android/Bionic. */
    return sched_setaffinity(0, sizeof(set), &set) == 0;
}

#elif defined(__APPLE__)

static bool apple_collect_cores(std::vector<CoreGroup> *out) {
    int physical = 0;
    int logical = 0;
    size_t sz = sizeof(physical);
    if (sysctlbyname("hw.physicalcpu", &physical, &sz, nullptr, 0) != 0 || physical <= 0) {
        return false;
    }
    sz = sizeof(logical);
    if (sysctlbyname("hw.logicalcpu", &logical, &sz, nullptr, 0) != 0 || logical <= 0) {
        return false;
    }

    out->clear();
    if (logical == physical * 2) {
        for (int core = 0; core < physical; ++core) {
            CoreGroup group;
            group.logical = {core, core + physical};
            out->push_back(std::move(group));
        }
    } else {
        for (int cpu = 0; cpu < logical; ++cpu) {
            CoreGroup group;
            group.logical = {cpu};
            out->push_back(std::move(group));
        }
    }
    return !out->empty();
}

static bool bind_current_thread(const CpuSlot &slot) {
    if (slot.cpus.empty()) {
        return false;
    }
    /* affinity_tag: unique tags prefer distinct cores (scheduler hint). The
     * first CPU of a slot is unique per slot in both SMT and 1/core mode. */
    thread_affinity_policy_data_t policy = {slot.cpus[0] + 1};
    return thread_policy_set(pthread_mach_thread_np(pthread_self()), THREAD_AFFINITY_POLICY,
                             reinterpret_cast<thread_policy_t>(&policy),
                             THREAD_AFFINITY_POLICY_COUNT) == KERN_SUCCESS;
}

#else

static bool bind_current_thread(const CpuSlot &) {
    return false;
}

#endif

static bool collect_cores(std::vector<CoreGroup> *out) {
#if defined(_WIN32)
    return win32_collect_cores(out);
#elif defined(__linux__)
    return linux_collect_cores(out);
#elif defined(__APPLE__)
    return apple_collect_cores(out);
#else
    (void)out;
    return false;
#endif
}

static int current_omp_threads(void) {
#if defined(_OPENMP)
    return omp_get_max_threads();
#else
    return 1;
#endif
}

static void update_summary(void) {
    if (g_physical_cores <= 0 || g_logical_cpus <= 0) {
        std::snprintf(g_summary, sizeof(g_summary), "unavailable");
        return;
    }
    const char *placement = "";
    switch (g_bind_mode) {
    case BindMode::None:
        placement = ", unpinned";
        break;
    case BindMode::Pinned:
        placement = ", pinned (cores first, then SMT siblings)";
        break;
    case BindMode::Runtime:
        placement = ", placement left to OpenMP runtime (OMP_PLACES/OMP_PROC_BIND)";
        break;
    }
    const int smt = g_logical_cpus - g_physical_cores;
    if (smt > 0) {
        std::snprintf(g_summary, sizeof(g_summary),
                      "%d physical + %d SMT, %d logical CPUs, %d OpenMP threads%s",
                      g_physical_cores, smt, g_logical_cpus, current_omp_threads(),
                      placement);
    } else {
        std::snprintf(g_summary, sizeof(g_summary), "%d cores, %d OpenMP threads%s",
                      g_physical_cores, current_omp_threads(), placement);
    }
}

} /* namespace */

extern "C" int cp_cpu_affinity_init(void) {
    g_cpu_order.clear();
    g_physical_cores = 0;
    g_logical_cpus = 0;
    g_bind_mode = BindMode::None;
    std::snprintf(g_summary, sizeof(g_summary), "disabled");

    if (affinity_disabled()) {
        std::snprintf(g_summary, sizeof(g_summary), "disabled (CP_CPU_AFFINITY=0)");
        return 0;
    }

    std::vector<CoreGroup> cores;
    if (!collect_cores(&cores)) {
        std::snprintf(g_summary, sizeof(g_summary), "unavailable");
        return -1;
    }

    restrict_to_process_mask(&cores);
    build_cpu_order(cores);
    update_summary();
    return 0;
}

extern "C" void cp_cpu_affinity_bind_openmp_pool(void) {
    if (affinity_disabled() || g_cpu_order.empty()) {
        return;
    }
    if (runtime_binds_threads()) {
        g_bind_mode = BindMode::Runtime;
        update_summary();
        return;
    }

#if defined(_OPENMP)
    /* OpenMP thread 0 is the main thread and stays pinned like the others; helper
     * threads it creates later call cp_cpu_affinity_release_thread(). */
#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        const CpuSlot &slot = g_cpu_order[static_cast<size_t>(tid) % g_cpu_order.size()];
        bind_current_thread(slot);
    }
#else
    bind_current_thread(g_cpu_order[0]);
#endif
    g_bind_mode = BindMode::Pinned;
    update_summary();
}

extern "C" const char *cp_cpu_affinity_summary(void) {
    return g_summary;
}

extern "C" int cp_cpu_affinity_physical_cores(void) {
    if (affinity_disabled() || g_cpu_order.empty()) {
        return 0;
    }
    return g_physical_cores;
}

extern "C" int cp_cpu_affinity_logical_cpus(void) {
    if (affinity_disabled() || g_cpu_order.empty()) {
        return 0;
    }
    return g_logical_cpus;
}

extern "C" void cp_cpu_affinity_release_thread(void) {
#if defined(__linux__)
    /* Linux threads inherit their creator's mask; a helper created by the pinned
     * main thread would otherwise share its single CPU with OpenMP thread 0.
     * Windows and macOS threads start from the process mask, nothing to undo. */
    if (g_bind_mode == BindMode::Pinned && g_have_process_mask) {
        sched_setaffinity(0, sizeof(g_process_mask), &g_process_mask);
    }
#endif
}

extern "C" int cp_cpu_affinity_bind_thread(int tid) {
    if (affinity_disabled() || g_cpu_order.empty() || tid < 0) {
        return -1;
    }
    const CpuSlot &slot = g_cpu_order[static_cast<size_t>(tid) % g_cpu_order.size()];
    return bind_current_thread(slot) ? 0 : -1;
}
