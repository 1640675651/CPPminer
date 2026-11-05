#include "cp_fee.h"

#include <stdio.h>
#include <string.h>

/*
 * Developer fee wallets — light XOR obfuscation (defeats casual `strings`).
 * Not real secrecy: recoverable at runtime from authorize traffic or reversing.
 *
 * To change an address:
 *   python internal/encode_fee_wallet.py 'prl1...'   # or qzpp...
 * and paste the printed bytes into the matching k_*_dev_wallet_enc below.
 */

static const unsigned char k_dev_wallet_key[] = {
    'c', 'p', 0x9e, 'm', 'i', 'n', 'A', 0x11, 'z'
};

static const unsigned char k_pearl_dev_wallet_enc[] = {
    0x13, 0x02, 0xf2, 0x5c, 0x19, 0x56, 0x73, 0x7a, 0x17, 0x12, 0x05, 0xad,
    0x09, 0x13, 0x0a, 0x73, 0x76, 0x4d, 0x5a, 0x15, 0xec, 0x15, 0x1e, 0x5c,
    0x3b, 0x24, 0x12, 0x1b, 0x1c, 0xe4, 0x06, 0x5a, 0x57, 0x38, 0x7f, 0x1e,
    0x10, 0x13, 0xf8, 0x0c, 0x04, 0x04, 0x20, 0x76, 0x0a, 0x16, 0x1e, 0xf2,
    0x55, 0x0e, 0x09, 0x79, 0x64, 0x1c, 0x12, 0x1b, 0xef, 0x0b, 0x5c, 0x57,
    0x38, 0x27, 0x4f,
};

static const unsigned char k_quantus_dev_wallet_enc[] = {
    0x12, 0x0a, 0xf0, 0x1d, 0x21, 0x3d, 0x33, 0x43, 0x19, 0x54, 0x00, 0xfa,
    0x37, 0x21, 0x07, 0x34, 0x7e, 0x43, 0x50, 0x32, 0xfc, 0x00, 0x0e, 0x5c,
    0x03, 0x42, 0x09, 0x15, 0x01, 0xcd, 0x18, 0x38, 0x0a, 0x38, 0x45, 0x19,
    0x3a, 0x32, 0xf1, 0x18, 0x5d, 0x58, 0x3b, 0x66, 0x32, 0x09, 0x18, 0xf3,
    0x0a,
};

/* Fixed fee pools, tried in order before the user's pool. A fee window is about one
 * matrix scan, so the first entry is the CPU port: its starting difficulty (26000)
 * is far below the GPU ports' (3360: 0.9M-1.5M), which makes the
 * fee's share count less noisy. Only pearl-eu2 (the same server) also has 3370 open. */
typedef struct { const char* host; int port; } CpFeePool;

static const CpFeePool k_pearl_fee_pools[] = {
    {"pearl-cpu-eu1.luckypool.io", 3370},
    {"pearl-eu.lproute.com", 3360},
    {"pearl-us-east.lproute.com", 3360},
    {"pearl-sg.lproute.com", 3360},
    {"pearl-ru.lproute.com", 3360},
    /* Kryptex last: its default difficulty (~4.2M) is the highest. The fee wallet is
     * sent as WALLET.worker on kryptex hosts (authorize_wallet_string). */
    {"prl.kryptex.network", 7048},
};

/* Quantus: the same LuckyPool servers (port 5660), then Kryptex. The fee is a 10 s
 * hashing turn, so starting difficulty matters less than for Pearl. */
static const CpFeePool k_quantus_fee_pools[] = {
    {"eu.lproute.com", 5660},
    {"us-east.lproute.com", 5660},
    {"sg.lproute.com", 5660},
    {"ru.lproute.com", 5660},
    {"qtc.kryptex.network", 7049},
};

#define CP_FEE_BACKOFF_MAX_CYCLES 64

static char g_user_wallet[256];
static char g_dev_wallet[256];
static int g_enabled = 0;
static int g_auth_is_dev = 0;
static int g_fee_active = 0;
static uint64_t g_debt = 0;
static uint64_t g_tiles_per_matrix = 0; /* T: tiles (Pearl) or turn length in us (Quantus) */
static CpAlgoId g_algo = CP_ALGO_PEARL;

/* Fee pool failover. g_pool_idx counts through the fixed pools; fee_pool_count()
 * means the user's pool. Each fee cycle starts at the fixed pool that last worked. */
static int g_pool_idx = 0;
static int g_pool_good = 0;
static int g_session_dev = 0;       /* the session being opened or run mines for the fee */
static int g_session_got_job = 0;
/* After every fixed pool failed: fee cycles that go straight to the user's pool. */
static int g_fixed_rest = 0, g_fixed_backoff = 0;
/* After the user's pool failed too: fee cycles skipped without trying. */
static int g_skip_rest = 0, g_skip_backoff = 0;

static int time_based(void){ return g_algo == CP_ALGO_QUANTUS; }

static const CpFeePool* fee_pools(void)
{
    return g_algo == CP_ALGO_QUANTUS ? k_quantus_fee_pools : k_pearl_fee_pools;
}

static int fee_pool_count(void)
{
    return g_algo == CP_ALGO_QUANTUS
               ? (int)(sizeof(k_quantus_fee_pools) / sizeof(k_quantus_fee_pools[0]))
               : (int)(sizeof(k_pearl_fee_pools) / sizeof(k_pearl_fee_pools[0]));
}

static const CpFeePool* fee_pool(int idx)
{
    return idx < fee_pool_count() ? &fee_pools()[idx] : NULL;
}

static int next_backoff(int cycles)
{
    if(cycles <= 0) return 1;
    return cycles * 2 < CP_FEE_BACKOFF_MAX_CYCLES ? cycles * 2 : CP_FEE_BACKOFF_MAX_CYCLES;
}

static uint64_t threshold_tiles(void);

/* Settles the current fee cycle as if it had been mined. */
static void pay_fee_cycle(void)
{
    const uint64_t thr = threshold_tiles();
    if(time_based() || g_debt < thr) g_debt = 0;
    else g_debt -= thr;
    g_fee_active = 0;
}

const char* cp_fee_format(uint64_t units, char* buf, int cap)
{
    if(time_based())
        snprintf(buf, (size_t)cap, "%.1f s", (double)units / 1e6);
    else
        snprintf(buf, (size_t)cap, "%llu tiles", (unsigned long long)units);
    return buf;
}

static void load_dev_wallet(CpAlgoId algo)
{
    const unsigned char* enc = k_pearl_dev_wallet_enc;
    size_t n = sizeof(k_pearl_dev_wallet_enc);
    if(algo == CP_ALGO_QUANTUS){
        enc = k_quantus_dev_wallet_enc;
        n = sizeof(k_quantus_dev_wallet_enc);
    }
    g_dev_wallet[0] = 0;
    if(n == 0) return;
    const size_t klen = sizeof(k_dev_wallet_key);
    if(n >= sizeof(g_dev_wallet)) return;
    for(size_t i = 0; i < n; i++)
        g_dev_wallet[i] = (char)(enc[i] ^ k_dev_wallet_key[i % klen]);
    g_dev_wallet[n] = 0;
}

static uint64_t threshold_tiles(void)
{
    if(g_tiles_per_matrix == 0) return 0;
    return (uint64_t)CP_FEE_PERIOD * g_tiles_per_matrix;
}

void cp_fee_init(const char* user_wallet, int enable, CpAlgoId algo)
{
    g_algo = algo;
    g_user_wallet[0] = 0;
    if(user_wallet){
        strncpy(g_user_wallet, user_wallet, sizeof(g_user_wallet) - 1);
        g_user_wallet[sizeof(g_user_wallet) - 1] = 0;
    }
    load_dev_wallet(algo);
    fflush(stdout);

    g_enabled = enable && g_user_wallet[0] && g_dev_wallet[0] &&
                strcmp(g_user_wallet, g_dev_wallet) != 0;
    g_auth_is_dev = 0;
    g_fee_active = 0;
    g_debt = 0;
    g_tiles_per_matrix = 0;
    g_pool_idx = g_pool_good = 0;
    g_session_dev = g_session_got_job = 0;
    g_fixed_rest = g_fixed_backoff = g_skip_rest = g_skip_backoff = 0;

    if(enable && !g_enabled){
        if(algo == CP_ALGO_QUANTUS && !g_dev_wallet[0]){
            fprintf(stderr,
                    "[fee] disabled for quantus (no fee wallet encoded yet)\n");
        } else {
            fprintf(stderr,
                    "[fee] disabled (missing wallet, or user wallet equals fee wallet)\n");
        }
    }
}

void cp_fee_set_tiles_per_matrix(uint64_t tiles_per_matrix)
{
    if(tiles_per_matrix == 0) return;
    const int first = (g_tiles_per_matrix == 0);
    char a[48], b[48];
    if(!first && g_tiles_per_matrix != tiles_per_matrix){
        printf("[fee] T changed %s -> %s (threshold 100*T)\n",
               cp_fee_format(g_tiles_per_matrix, a, sizeof(a)),
               cp_fee_format(tiles_per_matrix, b, sizeof(b)));
        fflush(stdout);
    }
    g_tiles_per_matrix = tiles_per_matrix;
    if(first && g_enabled){
        g_debt = ((uint64_t)CP_FEE_PERIOD / 2) * tiles_per_matrix;
        printf("[fee] debt seeded at 50*T = %s (T=%s)\n",
               cp_fee_format(g_debt, a, sizeof(a)),
               cp_fee_format(tiles_per_matrix, b, sizeof(b)));
        fflush(stdout);
    }
}

void cp_fee_on_authorized(void)
{
    g_auth_is_dev = g_enabled && cp_fee_next_is_dev();
}

const char* cp_fee_wallet(void)
{
    if(g_enabled && cp_fee_next_is_dev()) return g_dev_wallet;
    return g_user_wallet;
}

void cp_fee_prepare_matrix(void)
{
    if(!g_enabled || g_tiles_per_matrix == 0) return;
    if(!g_fee_active && g_debt >= threshold_tiles()){
        char a[48], b[48];
        cp_fee_format(g_debt, a, sizeof(a));
        cp_fee_format(threshold_tiles(), b, sizeof(b));
        if(g_skip_rest > 0){
            g_skip_rest--;
            pay_fee_cycle();
            printf("[fee] debt %s >= 100*T (%s): fee cycle skipped (no pool took the fee "
                   "wallet; %d more to skip)\n", a, b, g_skip_rest);
            fflush(stdout);
            return;
        }
        g_fee_active = 1;
        if(g_fixed_rest > 0){
            g_fixed_rest--;
            g_pool_idx = fee_pool_count();
        } else {
            g_pool_idx = g_pool_good;
        }
        printf("[fee] debt %s >= 100*T (%s): starting fee cycle\n", a, b);
        fflush(stdout);
    }
}

int cp_fee_next_is_dev(void)
{
    return g_enabled && g_fee_active;
}

int cp_fee_needs_switch(void)
{
    if(!g_enabled) return 0;
    return cp_fee_next_is_dev() != g_auth_is_dev;
}

void cp_fee_note_tiles(uint64_t tiles)
{
    if(!g_enabled || tiles == 0) return;
    /* Charge by the wallet the work was done under, not the one wanted next: Quantus CPU
     * threads finish their chunk on the user's session after a fee turn starts, and on
     * the fee session after it ends. */
    if(g_auth_is_dev){
        if(tiles > UINT64_MAX / (uint64_t)CP_FEE_PERIOD){
            g_debt = 0;
        } else {
            const uint64_t pay = tiles * (uint64_t)CP_FEE_PERIOD;
            if(pay >= g_debt) g_debt = 0;
            else g_debt -= pay;
        }
        if(g_fee_active && (time_based() ? g_debt == 0 : g_debt < threshold_tiles())){
            char a[48];
            printf("[fee] debt %s: fee cycle complete\n", cp_fee_format(g_debt, a, sizeof(a)));
            fflush(stdout);
            g_fee_active = 0;
        }
    } else {
        const uint64_t room = UINT64_MAX - g_debt;
        g_debt += (tiles > room) ? room : tiles;
    }
}

uint64_t cp_fee_debt(void){ return g_debt; }
uint64_t cp_fee_tiles_per_matrix(void){ return g_tiles_per_matrix; }
uint64_t cp_fee_threshold(void){ return threshold_tiles(); }
int cp_fee_enabled(void){ return g_enabled; }
int cp_fee_session_is_dev(void){ return g_enabled && g_auth_is_dev; }

void cp_fee_print_mode(void)
{
    if(!g_enabled) return;
    const CpFeePool* p = fee_pool(0);
    if(p)
        printf("[mode] dev fee: 1%% on %s:%d (%d backup fee pools, then your pool)\n",
               p->host, p->port, fee_pool_count() - 1);
    else
        printf("[mode] dev fee: 1%% on your pool\n");
    fflush(stdout);
}

void cp_fee_session_pool(const char* user_host, int user_port, const char** host, int* port)
{
    g_session_dev = cp_fee_next_is_dev();
    g_session_got_job = 0;
    const CpFeePool* p = g_session_dev ? fee_pool(g_pool_idx) : NULL;
    *host = p ? p->host : user_host;
    *port = p ? p->port : user_port;
}

int cp_fee_session_on_fee_pool(void)
{
    return g_session_dev && fee_pool(g_pool_idx) != NULL;
}

void cp_fee_session_ok(void)
{
    if(!g_session_dev || g_session_got_job) return;
    g_session_got_job = 1;
    g_skip_backoff = 0;
    if(fee_pool(g_pool_idx)){
        g_pool_good = g_pool_idx;
        g_fixed_backoff = 0;
    }
}

int cp_fee_session_end(void)
{
    const int dev = g_session_dev, got_job = g_session_got_job;
    g_session_dev = g_session_got_job = 0;
    /* Only a fee session that never got a job counts as failed: one that dropped
     * mid-cycle reconnects to the same pool. */
    if(!dev || got_job || !g_fee_active) return 0;

    const CpFeePool* p = fee_pool(g_pool_idx);
    if(p){
        g_pool_idx++;
        const CpFeePool* next = fee_pool(g_pool_idx);
        if(next){
            printf("[fee] fee pool %s:%d gave no job; trying %s:%d\n",
                   p->host, p->port, next->host, next->port);
        } else {
            /* Every fixed pool failed: use the user's pool for the next fee cycles too. */
            g_pool_good = 0;
            g_fixed_backoff = next_backoff(g_fixed_backoff);
            g_fixed_rest = g_fixed_backoff;
            printf("[fee] fee pool %s:%d gave no job; no fee pool reachable, trying your "
                   "pool (the next %d fee cycles go there too)\n",
                   p->host, p->port, g_fixed_rest);
        }
    } else {
        /* The user's pool failed under the fee wallet too (e.g. it rejects unknown
         * wallets): give up this cycle instead of retrying it forever. */
        pay_fee_cycle();
        g_skip_backoff = next_backoff(g_skip_backoff);
        g_skip_rest = g_skip_backoff;
        printf("[fee] your pool gave no job for the fee wallet; fee cycle skipped "
               "(next %d fee cycles skipped too)\n", g_skip_rest);
    }
    fflush(stdout);
    return 1;
}
