#include "cp_fee.h"

#include <stdio.h>
#include <string.h>

/*
 * Developer fee wallet — light XOR obfuscation (defeats casual `strings`).
 * Not real secrecy: recoverable at runtime from authorize traffic or reversing.
 *
 * To change the address:
 *   python internal/encode_fee_wallet.py 'prl1...'
 * and paste the printed bytes into k_dev_wallet_enc below.
 */

static const unsigned char k_dev_wallet_key[] = {
    'c', 'p', 0x9e, 'm', 'i', 'n', 'A', 0x11, 'z'
};

static const unsigned char k_dev_wallet_enc[] = {
    0x13, 0x02, 0xf2, 0x5c, 0x19, 0x56, 0x73, 0x7a, 0x17, 0x12, 0x05, 0xad,
    0x09, 0x13, 0x0a, 0x73, 0x76, 0x4d, 0x5a, 0x15, 0xec, 0x15, 0x1e, 0x5c,
    0x3b, 0x24, 0x12, 0x1b, 0x1c, 0xe4, 0x06, 0x5a, 0x57, 0x38, 0x7f, 0x1e,
    0x10, 0x13, 0xf8, 0x0c, 0x04, 0x04, 0x20, 0x76, 0x0a, 0x16, 0x1e, 0xf2,
    0x55, 0x0e, 0x09, 0x79, 0x64, 0x1c, 0x12, 0x1b, 0xef, 0x0b, 0x5c, 0x57,
    0x38, 0x27, 0x4f,
};

static char g_user_wallet[256];
static char g_dev_wallet[256];
static int g_enabled = 0;
static int g_auth_is_dev = 0;
static int g_fee_active = 0;
static uint64_t g_debt = 0;
static uint64_t g_tiles_per_matrix = 0; /* T: hash tiles per full matrix scan */

static void load_dev_wallet(void)
{
    const size_t n = sizeof(k_dev_wallet_enc);
    const size_t klen = sizeof(k_dev_wallet_key);
    if (n >= sizeof(g_dev_wallet)) {
        g_dev_wallet[0] = 0;
        return;
    }
    for (size_t i = 0; i < n; i++) {
        g_dev_wallet[i] = (char)(k_dev_wallet_enc[i] ^ k_dev_wallet_key[i % klen]);
    }
    g_dev_wallet[n] = 0;
}

static uint64_t threshold_tiles(void)
{
    if (g_tiles_per_matrix == 0) {
        return 0;
    }
    /* 100 * T — overflow-safe: T fits production hash-tile counts far below 2^64/100. */
    return (uint64_t)CP_FEE_PERIOD * g_tiles_per_matrix;
}

void cp_fee_init(const char* user_wallet, int enable)
{
    g_user_wallet[0] = 0;
    if (user_wallet) {
        strncpy(g_user_wallet, user_wallet, sizeof(g_user_wallet) - 1);
        g_user_wallet[sizeof(g_user_wallet) - 1] = 0;
    }
    load_dev_wallet();
    fflush(stdout);

    g_enabled = enable && g_user_wallet[0] && g_dev_wallet[0] &&
                strcmp(g_user_wallet, g_dev_wallet) != 0;
    g_auth_is_dev = 0;
    g_fee_active = 0;
    g_debt = 0;
    g_tiles_per_matrix = 0;

    if (enable && !g_enabled) {
        fprintf(stderr,
                "[fee] disabled (missing wallet, or user wallet equals fee wallet)\n");
    }
}

void cp_fee_set_tiles_per_matrix(uint64_t tiles_per_matrix)
{
    if (tiles_per_matrix == 0) {
        return;
    }
    const int first = (g_tiles_per_matrix == 0);
    if (!first && g_tiles_per_matrix != tiles_per_matrix) {
        printf("[fee] tiles/matrix T changed %llu -> %llu (threshold 100*T)\n",
               (unsigned long long)g_tiles_per_matrix,
               (unsigned long long)tiles_per_matrix);
        fflush(stdout);
    }
    g_tiles_per_matrix = tiles_per_matrix;
    /* Center first fee: seed debt at 50*T so the next 50*T user tiles trigger fee. */
    if (first && g_enabled) {
        g_debt = ((uint64_t)CP_FEE_PERIOD / 2) * tiles_per_matrix;
        printf("[fee] tile-debt seeded at 50*T = %llu (T=%llu)\n",
               (unsigned long long)g_debt, (unsigned long long)tiles_per_matrix);
        fflush(stdout);
    }
}

void cp_fee_on_authorized(void)
{
    g_auth_is_dev = g_enabled && cp_fee_next_is_dev();
}

const char* cp_fee_wallet(void)
{
    if (g_enabled && cp_fee_next_is_dev()) {
        return g_dev_wallet;
    }
    return g_user_wallet;
}

void cp_fee_prepare_matrix(void)
{
    if (!g_enabled || g_tiles_per_matrix == 0) {
        return;
    }
    if (!g_fee_active && g_debt >= threshold_tiles()) {
        g_fee_active = 1;
        printf("[fee] debt %llu >= 100*T (%llu): starting fee cycle\n",
               (unsigned long long)g_debt, (unsigned long long)threshold_tiles());
        fflush(stdout);
    }
}

int cp_fee_next_is_dev(void)
{
    return g_enabled && g_fee_active;
}

int cp_fee_needs_switch(void)
{
    if (!g_enabled) {
        return 0;
    }
    return cp_fee_next_is_dev() != g_auth_is_dev;
}

void cp_fee_note_tiles(uint64_t tiles)
{
    if (!g_enabled || tiles == 0) {
        return;
    }
    if (g_fee_active) {
        if (tiles > UINT64_MAX / (uint64_t)CP_FEE_PERIOD) {
            g_debt = 0;
        } else {
            const uint64_t pay = tiles * (uint64_t)CP_FEE_PERIOD;
            if (pay >= g_debt) {
                g_debt = 0;
            } else {
                g_debt -= pay;
            }
        }
        if (g_debt < threshold_tiles()) {
            if (g_fee_active) {
                printf("[fee] debt %llu < 100*T (%llu): fee cycle complete\n",
                       (unsigned long long)g_debt, (unsigned long long)threshold_tiles());
                fflush(stdout);
            }
            g_fee_active = 0;
        }
    } else {
        const uint64_t room = UINT64_MAX - g_debt;
        g_debt += (tiles > room) ? room : tiles;
    }
}

uint64_t cp_fee_debt(void)
{
    return g_debt;
}

uint64_t cp_fee_tiles_per_matrix(void)
{
    return g_tiles_per_matrix;
}

uint64_t cp_fee_threshold(void)
{
    return threshold_tiles();
}

int cp_fee_enabled(void)
{
    return g_enabled;
}
