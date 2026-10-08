#include "cp_util.h"
#include "cp_config.h"
#include "cp_state.h"
#include "cp_platform.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef _WIN32
#include <sys/time.h>
#include <unistd.h>
#else
#include <bcrypt.h>
#pragma comment(lib, "bcrypt.lib")
#endif

double cp_now_sec(void)
{
#ifdef _WIN32
    static LARGE_INTEGER freq = {0};
    LARGE_INTEGER t;
    if(!freq.QuadPart) QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / (double)freq.QuadPart;
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec * 1e-6;
#endif
}

int cp_random_bytes(void* buf, size_t n)
{
    if(!buf || n == 0) return -1;
#ifdef _WIN32
    NTSTATUS st = BCryptGenRandom(NULL, (PUCHAR)buf, (ULONG)n, BCRYPT_USE_SYSTEM_PREFERRED_RNG);
    return BCRYPT_SUCCESS(st) ? 0 : -1;
#else
    FILE* f = fopen("/dev/urandom", "rb");
    if(!f) return -1;
    size_t got = fread(buf, 1, n, f);
    fclose(f);
    return (got == n) ? 0 : -1;
#endif
}

int cp_random_u64(uint64_t* out)
{
    if(!out) return -1;
    uint64_t v = 0;
    if(cp_random_bytes(&v, sizeof(v)) != 0) return -1;
    if(v == 0) v = 1;
    *out = v;
    return 0;
}

int cp_file_exists(const char* path)
{
    FILE* f = fopen(path, "rb");
    if(f){ fclose(f); return 1; }
    return 0;
}

void cp_path_to_posix(char* path)
{
    for(char* p = path; *p; ++p){
        if(*p == '\\') *p = '/';
    }
}

void cp_path_abs(char* path, size_t cap)
{
    if(!path || !path[0] || cap == 0) return;
#ifdef _WIN32
    char full[MAX_PATH];
    if(GetFullPathNameA(path, MAX_PATH, full, NULL) && full[0]){
        strncpy(path, full, cap - 1);
        path[cap - 1] = 0;
    }
#else
    char* rp = realpath(path, NULL);
    if(rp){
        strncpy(path, rp, cap - 1);
        path[cap - 1] = 0;
        free(rp);
    }
#endif
}

void cp_init_workdir(void)
{
#ifdef _WIN32
    if(GetModuleFileNameA(NULL, g_workdir, MAX_PATH)){
        char* slash = strrchr(g_workdir, '\\');
        if(slash) *slash = 0;
    } else {
        GetCurrentDirectoryA(MAX_PATH, g_workdir);
    }
#else
    ssize_t n = readlink("/proc/self/exe", g_workdir, sizeof(g_workdir) - 1);
    if(n > 0){
        g_workdir[n] = 0;
        char* slash = strrchr(g_workdir, '/');
        if(slash) *slash = 0;
    } else {
        if(!getcwd(g_workdir, sizeof(g_workdir)))
            strncpy(g_workdir, ".", sizeof(g_workdir) - 1);
    }
#endif
}

int cp_read_file_bin(const char* path, void* buf, size_t nbytes)
{
    FILE* f = fopen(path, "rb");
    if(!f) return 0;
    size_t n = fread(buf, 1, nbytes, f);
    fclose(f);
    return n == nbytes;
}

int cp_read_file_text(const char* path, char* out, int cap)
{
    FILE* f = fopen(path, "rb");
    if(!f) return 0;
    int n = (int)fread(out, 1, (size_t)cap - 1, f);
    out[n] = 0;
    int truncated = 0;
    if(n == cap - 1){
        int extra = fgetc(f);
        if(extra != EOF) truncated = 1;
    }
    fclose(f);
    if(truncated) return 0;
    while(n > 0 && (out[n-1]=='\n' || out[n-1]=='\r' || out[n-1]==' ')) out[--n] = 0;
    return n > 0;
}

int cp_write_file_bin(const char* path, const void* buf, size_t nbytes)
{
    FILE* f = fopen(path, "wb");
    if(!f) return 0;
    size_t n = fwrite(buf, 1, nbytes, f);
    fclose(f);
    return n == nbytes;
}

void cp_bin_to_hex(const uint8_t* in, size_t n, char* out)
{
    for(size_t i = 0; i < n; i++) sprintf(out + i*2, "%02x", in[i]);
    out[n*2] = 0;
}

int cp_hex_to_bytes(const char* hex, uint8_t* out, int out_cap)
{
    if(!hex || !out || out_cap < 0) return 0;
    size_t n = strlen(hex);
    if((n & 1) != 0 || n / 2 > (size_t)out_cap) return 0;
    int nb = (int)(n / 2);
    for(int i = 0; i < nb; i++){
        unsigned value = 0;
        for(int j = 0; j < 2; ++j){
            const unsigned char c = (unsigned char)hex[i * 2 + j];
            unsigned digit;
            if(c >= '0' && c <= '9') digit = c - '0';
            else if(c >= 'a' && c <= 'f') digit = c - 'a' + 10;
            else if(c >= 'A' && c <= 'F') digit = c - 'A' + 10;
            else return 0;
            value = (value << 4) | digit;
        }
        out[i] = (uint8_t)value;
    }
    return nb;
}

/* Find a JSON member name, skipping string contents (including escapes).
 * The pool protocol uses unique member names for the fields read here. */
static const char* cp_json_value(const char* json, const char* key)
{
    if(!json || !key || !*key) return NULL;
    const size_t key_len = strlen(key);
    for(const char* p = json; *p; ++p){
        if(*p != '"') continue;
        const char* begin = ++p;
        while(*p && *p != '"'){
            if(*p == '\\' && p[1]) ++p;
            ++p;
        }
        if(!*p) return NULL;
        const char* after = p + 1;
        while(*after == ' ' || *after == '\t' || *after == '\r' || *after == '\n') ++after;
        if((size_t)(p - begin) == key_len && !memcmp(begin, key, key_len) && *after == ':'){
            ++after;
            while(*after == ' ' || *after == '\t' || *after == '\r' || *after == '\n') ++after;
            return after;
        }
    }
    return NULL;
}

int cp_json_str(const char* json, const char* key, char* out, int outlen)
{
    if(!out || outlen <= 0) return 0;
    out[0] = 0;
    const char* p = cp_json_value(json, key);
    if(!p || *p++ != '"') return 0;
    int i = 0;
    while(*p && *p != '"'){
        unsigned char c = (unsigned char)*p++;
        if(c == '\\'){
            c = (unsigned char)*p++;
            switch(c){
                case '"': case '\\': case '/': break;
                case 'b': c = '\b'; break;
                case 'f': c = '\f'; break;
                case 'n': c = '\n'; break;
                case 'r': c = '\r'; break;
                case 't': c = '\t'; break;
                default: out[0] = 0; return 0;
            }
        }
        if(c < 0x20 || i >= outlen - 1){ out[0] = 0; return 0; }
        out[i++] = (char)c;
    }
    if(*p != '"'){ out[0] = 0; return 0; }
    out[i] = 0;
    return 1;
}

double cp_json_num(const char* json, const char* key)
{
    const char* p = cp_json_value(json, key);
    if(!p) return 0;
    char* end = NULL;
    double value = strtod(p, &end);
    return end != p && isfinite(value) ? value : 0;
}

static int g_pp_hash_h_override = 0;
static int g_pp_hash_w_override = 0;

void cp_pp_set_hash_tile(int h, int w)
{
    if(h > 0 && w > 0){
        g_pp_hash_h_override = h;
        g_pp_hash_w_override = w;
    } else {
        g_pp_hash_h_override = 0;
        g_pp_hash_w_override = 0;
    }
}

static int cp_active_hash_h(void)
{
    if(g_cutlass_fused) return CP_CUTLASS_HASH_H;
    if(g_pp_hash_h_override > 0) return g_pp_hash_h_override;
    return PP_HASH_H;
}

static int cp_active_hash_w(void)
{
    if(g_cutlass_fused) return CP_CUTLASS_HASH_W;
    if(g_pp_hash_w_override > 0) return g_pp_hash_w_override;
    return PP_HASH_W;
}

int cp_pp_hash_tile_h(void)
{
    return cp_active_hash_h();
}

int cp_pp_hash_tile_w(void)
{
    return cp_active_hash_w();
}

/* 256-bit LE target from a long double value (saturates at 2^256 - 1). */
static void target_from_ld(long double v, uint32_t tgt[8])
{
    memset(tgt, 0, 8 * sizeof(uint32_t));
    if(!(v > 0.0L)) return;
    if(!isfinite((double)v) || v >= 115792089237316195423570985008687907853269984665640564039457584007913129639936.0L){
        for(int i=0;i<8;i++) tgt[i]=0xFFFFFFFFu;
        return;
    }
    const long double base = 4294967296.0L;
    for(int i=0;i<8;i++){
        long double rem = fmodl(v, base);
        if(rem < 0.0L) rem = 0.0L;
        if(rem > 4294967295.0L) rem = 4294967295.0L;
        tgt[i] = (uint32_t)rem;
        v = floorl(v / base);
        if(v <= 0.0L) break;
    }
}

void cp_mock_target_from_difficulty(double difficulty, uint32_t tgt[8])
{
    /* Offline mock: the unscaled share target is 2^(256 - D). The hash tile is
     * applied once, later, by cp_scale_jackpot_target, so the mock share rate
     * per attempt does not depend on the tile shape. */
    memset(tgt, 0, 8 * sizeof(uint32_t));
    const long double exp_val = 256.0L - (long double)difficulty;
    if(exp_val >= 256.0L){
        for(int i=0;i<8;i++) tgt[i]=0xFFFFFFFFu;
    } else if(exp_val > 0.0L){
        target_from_ld(powl(2.0L, exp_val), tgt);
    }
}

void cp_pool_target_from_difficulty(double difficulty, uint32_t tgt[8])
{
    /* Stratum convention: target = (0xFFFF << 208) / diff, e.g. diff 26000 ->
     * 0000000000028544877baaede211544877baaede211544877baaede211544877. Exact
     * 256-bit division for integral difficulties; fractional ones go through
     * long double. Only used when mining.notify carries no target. */
    memset(tgt, 0, 8 * sizeof(uint32_t));
    if(!(difficulty > 0.0)) return;
    const long double d = (long double)difficulty;
    /* 2^64 is exact in long double; with MSVC's 53-bit long double the
     * literal 2^64 - 1 would round up to it and overflow the cast. */
    if(d == floorl(d) && d < 18446744073709551616.0L){
        const uint64_t div = (uint64_t)d;
        /* numerator 0xFFFF << 208 as LE 32-bit words: word 6 = 0xFFFF0000 */
        const uint32_t num[8] = {0, 0, 0, 0, 0, 0, 0xFFFF0000u, 0};
        /* Bitwise long division: rem < div, so 2*rem + 1 fits in 65 bits; the
         * bit shifted out of rem marks the cases where it must exceed div. */
        uint64_t rem = 0;
        for(int bit = 255; bit >= 0; bit--){
            const uint64_t carry = rem >> 63;
            rem = (rem << 1) | ((num[bit >> 5] >> (bit & 31)) & 1u);
            if(carry || rem >= div){
                rem -= div;
                tgt[bit >> 5] |= 1u << (bit & 31);
            }
        }
    } else {
        target_from_ld(65535.0L * powl(2.0L, 208.0L) / d, tgt);
    }
}

int cp_be_target_hex_to_le_words(const char* hex, uint32_t tgt[8])
{
    uint8_t b[32];
    if(cp_hex_to_bytes(hex, b, 32) != 32) return 0;
    for(int i= 0; i < 8; i++){
        int o = i * 4;
        tgt[7 - i] = ((uint32_t)b[o] << 24) | ((uint32_t)b[o+1] << 16)
                   | ((uint32_t)b[o+2] << 8) | (uint32_t)b[o+3];
    }
    return 1;
}

void cp_le_words_to_be_target_hex(const uint32_t tgt[8], char hex[65])
{
    uint8_t b[32];
    for(int i = 0; i < 8; i++){
        const uint32_t w = tgt[7 - i];
        const int o = i * 4;
        b[o] = (uint8_t)(w >> 24);
        b[o + 1] = (uint8_t)(w >> 16);
        b[o + 2] = (uint8_t)(w >> 8);
        b[o + 3] = (uint8_t)w;
    }
    cp_bin_to_hex(b, 32, hex);
}

void cp_scale_target_le(uint32_t tgt[8], uint64_t factor)
{
    if(factor <= 1) return;
    uint64_t carry = 0;
    for(int i = 0; i < 8; i++){
        uint64_t p = (uint64_t)tgt[i] * factor + carry;
        tgt[i] = (uint32_t)p;
        carry = p >> 32;
    }
    if(carry){
        for(int i = 0; i < 8; i++) tgt[i] = 0xFFFFFFFFu;
    }
}

uint64_t cp_jackpot_scale_factor(void)
{
    return (uint64_t)cp_active_hash_h() * (uint64_t)cp_active_hash_w()
         * (uint64_t)(K_DIM / R_RANK) * (uint64_t)PENALTY_BASE_RANK;
}

void cp_scale_jackpot_target(const uint32_t pool_tgt[8], uint32_t bound[8])
{
    /* Rank-penalized bound: target * h * w * (k/r) * PENALTY_BASE_RANK
     * (pearl penalized_target_bound / check_rank_penalty). At r == 128 this
     * equals the legacy unpenalized h*w*k scale. */
    memcpy(bound, pool_tgt, 8 * sizeof(uint32_t));
    cp_scale_target_le(bound, cp_jackpot_scale_factor());
}

int cp_send_all(int sock, const void* data, size_t len)
{
    const char* p = (const char*)data;
    while(len > 0){
#ifdef _WIN32
        int chunk = (len > 65536) ? 65536 : (int)len;
        int n = send(sock, p, chunk, 0);
#else
        ssize_t n = send(sock, p, len, 0);
#endif
        if(n <= 0){
            perror("send");
            return 0;
        }
        p += n;
        len -= (size_t)n;
    }
    return 1;
}

int cp_send_json(int sock, const char* json)
{
    if(!json || !json[0]) return 0;
    if(!cp_send_all(sock, json, strlen(json))) return 0;
    if(!cp_send_all(sock, "\n", 1)) return 0;
    return 1;
}

int cp_pp_num_row_parts(int m, int contiguous)
{
    if(g_cutlass_fused) return m / CP_CUTLASS_HASH_H;
    if(contiguous) return m / cp_active_hash_h();
    return (m / 128) * 16;
}

int cp_pp_num_col_parts(int n, int contiguous)
{
    if(g_cutlass_fused) return n / CP_CUTLASS_HASH_W;
    if(contiguous) return n / cp_active_hash_w();
    return (n / 256) * 16;
}

int cp_pp_num_row_periods(int m, int contiguous)
{
    if(g_cutlass_fused) return m / CP_CUTLASS_CTA_M;
    if(contiguous) return m / cp_active_hash_h();
    return m / 128;
}

int cp_pp_num_col_periods(int n, int contiguous)
{
    if(g_cutlass_fused) return n / CP_CUTLASS_CTA_N;
    if(contiguous) return n / cp_active_hash_w();
    return n / 256;
}

double cp_pp_macs_per_hash_tile(void)
{
    return (double)cp_active_hash_h() * (double)cp_active_hash_w() * (double)K_DIM;
}

double cp_pp_mac_rate_from_tiles(uint64_t tiles_scanned, double elapsed_sec)
{
    if(elapsed_sec < 1e-9) elapsed_sec = 1e-9;
    return (double)tiles_scanned * cp_pp_macs_per_hash_tile() / elapsed_sec;
}

void cp_pp_fmt_mac_rate(double mac_s, char* out, size_t out_sz)
{
    if(mac_s >= 1e15)
        snprintf(out, out_sz, "%.2f PMAC/s", mac_s / 1e15);
    else if(mac_s >= 1e12)
        snprintf(out, out_sz, "%.2f TMAC/s", mac_s / 1e12);
    else if(mac_s >= 1e9)
        snprintf(out, out_sz, "%.2f GMAC/s", mac_s / 1e9);
    else if(mac_s >= 1e6)
        snprintf(out, out_sz, "%.2f MMAC/s", mac_s / 1e6);
    else if(mac_s >= 1e3)
        snprintf(out, out_sz, "%.2f KMAC/s", mac_s / 1e3);
    else
        snprintf(out, out_sz, "%.0f MAC/s", mac_s);
}

void cp_log_attempt_timing(const char* tag, double prep_sec, double scan_sec, uint64_t tiles,
                           double post_sec)
{
    char mac_buf[32];
    double scan_d = scan_sec < 1e-9 ? 1e-9 : scan_sec;
    cp_pp_fmt_mac_rate(cp_pp_mac_rate_from_tiles(tiles, scan_d), mac_buf, sizeof(mac_buf));
    if(post_sec >= 1e-3)
        printf("[%s] attempt timing: prep=%.3fs scan=%.3fs post=%.3fs %s\n",
               tag, prep_sec, scan_sec, post_sec, mac_buf);
    else
        printf("[%s] attempt timing: prep=%.3fs scan=%.3fs %s\n",
               tag, prep_sec, scan_sec, mac_buf);
    fflush(stdout);
}
