#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif

#include "cp_cpu_affinity.h"

#include <algorithm>
#include <cstdio>
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

struct CpuSlot {
#if defined(_WIN32)
    WORD group = 0;
    KAFFINITY mask = 0;
#else
    int cpu = -1;
#endif
};

struct CoreGroup {
    std::vector<int> logical;
};

static std::vector<CpuSlot> g_cpu_order;
static int g_physical_cores = 0;
static int g_logical_cpus = 0;
static char g_summary[160] = "disabled";

static bool affinity_disabled(void) {
    const char *env = std::getenv("CP_CPU_AFFINITY");
    return env && (env[0] == '0' || env[0] == 'n' || env[0] == 'N');
}

static void build_cpu_order(const std::vector<CoreGroup> &cores) {
    g_cpu_order.clear();
    g_physical_cores = static_cast<int>(cores.size());

    auto append_slot = [](int cpu) {
        CpuSlot slot{};
#if defined(_WIN32)
        slot.group = static_cast<WORD>(cpu / 64);
        slot.mask = KAFFINITY(1) << (cpu % 64);
#else
        slot.cpu = cpu;
#endif
        g_cpu_order.push_back(slot);
    };

    size_t max_smt = 1;
    for (const CoreGroup &core : cores) {
        max_smt = std::max(max_smt, core.logical.size());
    }

    for (size_t smt = 0; smt < max_smt; ++smt) {
        for (const CoreGroup &core : cores) {
            if (smt < core.logical.size()) {
                append_slot(core.logical[smt]);
            }
        }
    }

    g_logical_cpus = static_cast<int>(g_cpu_order.size());
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
    if (slot.cpu < 0) {
        return false;
    }
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(slot.cpu, &set);
    return pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0;
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
    if (slot.cpu < 0) {
        return false;
    }
    /* affinity_tag: unique tags prefer distinct cores (scheduler hint). */
    thread_affinity_policy_data_t policy = {slot.cpu + 1};
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

static void update_summary(int omp_threads) {
    if (g_physical_cores <= 0 || g_logical_cpus <= 0) {
        std::snprintf(g_summary, sizeof(g_summary), "unavailable");
        return;
    }
    const int smt = g_logical_cpus - g_physical_cores;
    if (smt > 0) {
        std::snprintf(g_summary, sizeof(g_summary),
                      "%d physical + %d SMT, %d logical CPUs, %d OpenMP threads",
                      g_physical_cores, smt, g_logical_cpus, omp_threads);
    } else {
        std::snprintf(g_summary, sizeof(g_summary),
                      "%d cores, %d OpenMP threads", g_physical_cores, omp_threads);
    }
}

} /* namespace */

extern "C" int cp_cpu_affinity_init(void) {
    g_cpu_order.clear();
    g_physical_cores = 0;
    g_logical_cpus = 0;
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

    build_cpu_order(cores);
#if defined(_OPENMP)
    update_summary(omp_get_max_threads());
#else
    update_summary(1);
#endif
    return 0;
}

extern "C" void cp_cpu_affinity_bind_openmp_pool(void) {
    if (affinity_disabled() || g_cpu_order.empty()) {
        return;
    }

#if defined(_OPENMP)
#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        const CpuSlot &slot = g_cpu_order[static_cast<size_t>(tid) % g_cpu_order.size()];
        bind_current_thread(slot);
    }
#else
    bind_current_thread(g_cpu_order[0]);
#endif
}

extern "C" const char *cp_cpu_affinity_summary(void) {
    return g_summary;
}
