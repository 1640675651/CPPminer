#include "cp_job_ctrl.h"
#include "cp_pool.h"

#include <atomic>
#include <cstring>
#include <mutex>

static std::atomic<uint64_t> g_mine_epoch{0};
static std::atomic<uint64_t> g_cancel_epoch{0};
static std::atomic<int> g_mining_active{0};
static char g_mining_job_key[320] = {0};
static std::mutex g_mine_mx;

extern "C" void cp_job_mine_begin(const char* job_key)
{
    std::lock_guard<std::mutex> lk(g_mine_mx);
    g_mine_epoch.fetch_add(1);
    g_cancel_epoch.store(0);
    strncpy(g_mining_job_key, job_key, sizeof(g_mining_job_key) - 1);
    g_mining_job_key[sizeof(g_mining_job_key) - 1] = 0;
    g_mining_active.store(1);
}

extern "C" void cp_job_mine_end(void)
{
    g_mining_active.store(0);
}

extern "C" int cp_job_should_cancel(void)
{
    if(!g_mining_active.load()) return 0;
    if(cp_pool_conn_lost()) return 1;
    uint64_t ep = g_mine_epoch.load();
    return g_cancel_epoch.load() == ep && ep != 0;
}

/* Called from pool module when a new job arrives during mining. */
extern "C" void cp_job_request_cancel(void)
{
    g_cancel_epoch.store(g_mine_epoch.load());
}

extern "C" int cp_job_mining_active(void)
{
    return g_mining_active.load();
}

extern "C" const char* cp_job_mining_key(void)
{
    return g_mining_job_key;
}

extern "C" int cp_job_key_matches(const char* job_key)
{
    return !strcmp(g_mining_job_key, job_key);
}
