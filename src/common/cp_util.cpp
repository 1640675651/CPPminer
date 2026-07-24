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

static void cp_resolve_python(void)
{
    const char* env_py = getenv("CP_PYTHON");
    if(!env_py || !env_py[0]) env_py = getenv("PEARL_PYTHON");
    if(env_py && env_py[0]){
        strncpy(g_python_exe, env_py, sizeof(g_python_exe) - 1);
        g_python_exe[sizeof(g_python_exe) - 1] = 0;
    }

#ifdef _WIN32
    if(!cp_file_exists(g_python_exe)){
        char found[MAX_PATH];
        if(SearchPathA(NULL, "python.exe", NULL, MAX_PATH, found, NULL)){
            strncpy(g_python_exe, found, sizeof(g_python_exe) - 1);
            g_python_exe[sizeof(g_python_exe) - 1] = 0;
        }
    }
#endif

    cp_path_abs(g_python_exe, sizeof(g_python_exe));
    cp_path_to_posix(g_python_exe);

    if(!cp_file_exists(g_python_exe)){
        fprintf(stderr,
            "[plain] python not found: %s\n"
            "  Set CP_PYTHON (or PEARL_PYTHON) or pass --python\n",
            g_python_exe);
    }
}

static int cp_run_cmd(char* cmdline)
{
    printf("[host] %s\n", cmdline); fflush(stdout);
#ifdef _WIN32
    STARTUPINFOA si;
    PROCESS_INFORMATION pi;
    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);
    ZeroMemory(&pi, sizeof(pi));
    if(!CreateProcessA(NULL, cmdline, NULL, NULL, FALSE, 0, NULL, g_workdir, &si, &pi)){
        fprintf(stderr, "[host] CreateProcess failed (err=%lu)\n", (unsigned long)GetLastError());
        return -1;
    }
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD ec = 1;
    GetExitCodeProcess(pi.hProcess, &ec);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    return (int)ec;
#else
    return system(cmdline);
#endif
}

int cp_run_python(const char* subcmd)
{
    char cmd[8192];
    snprintf(cmd, sizeof(cmd), "\"%s\" \"%s\" %s", g_python_exe, g_host_bridge, subcmd);
    return cp_run_cmd(cmd);
}

void cp_resolve_paths(int argc, char** argv)
{
    if(!cp_file_exists(g_host_bridge)){
        char alt[768];
        snprintf(alt, sizeof(alt), "%s/scripts/plain_proof_host.py", g_workdir);
        if(cp_file_exists(alt)){
            strncpy(g_host_bridge, alt, sizeof(g_host_bridge) - 1);
            g_host_bridge[sizeof(g_host_bridge) - 1] = 0;
        } else {
            snprintf(alt, sizeof(alt), "scripts/plain_proof_host.py");
            if(cp_file_exists(alt)){
                strncpy(g_host_bridge, alt, sizeof(g_host_bridge) - 1);
                g_host_bridge[sizeof(g_host_bridge) - 1] = 0;
            }
        }
    }

    cp_path_abs(g_host_bridge, sizeof(g_host_bridge));
    cp_path_to_posix(g_host_bridge);

    if(!cp_file_exists(g_host_bridge)){
        fprintf(stderr, "[plain] host bridge not found: %s\n", g_host_bridge);
    }

    cp_resolve_python();
    printf("[plain] workdir=%s\n[plain] python=%s\n[plain] bridge=%s\n",
           g_workdir, g_python_exe, g_host_bridge);
    fflush(stdout);
    (void)argc; (void)argv;
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
    int n = (int)strlen(hex);
    if((n & 1) != 0) return 0;
    int nb = n / 2;
    if(nb > out_cap) return 0;
    for(int i = 0; i < nb; i++){
        unsigned v = 0;
        if(sscanf(hex + i*2, "%02x", &v) != 1) return 0;
        out[i] = (uint8_t)v;
    }
    return nb;
}

int cp_json_str(const char* json, const char* key, char* out, int outlen)
{
    char pat[128]; snprintf(pat,sizeof(pat),"\"%s\"",key);
    const char* p = strstr(json,pat); if(!p) return 0;
    p += strlen(pat);
    while(*p==' '||*p==':') p++;
    if(*p!='"') return 0; p++;
    int i=0;
    while(*p && *p!='"' && i<outlen-1) out[i++]=*p++;
    out[i]=0; return 1;
}

double cp_json_num(const char* json, const char* key)
{
    char pat[128]; snprintf(pat,sizeof(pat),"\"%s\"",key);
    const char* p = strstr(json,pat); if(!p) return 0;
    p += strlen(pat);
    while(*p==' '||*p==':') p++;
    return atof(p);
}

static int cp_active_hash_h(void)
{
    return g_cutlass_fused ? CP_CUTLASS_HASH_H : PP_HASH_H;
}

static int cp_active_hash_w(void)
{
    return g_cutlass_fused ? CP_CUTLASS_HASH_W : PP_HASH_W;
}

void cp_target_from_difficulty(double difficulty, uint32_t tgt[8])
{
    memset(tgt, 0, 8 * sizeof(uint32_t));
    long double exp_val = 256.0L - (long double)difficulty
        + log2l((long double)(R_RANK * cp_active_hash_h() * cp_active_hash_w()));
    if(exp_val >= 256.0L){
        for(int i=0;i<8;i++) tgt[i]=0xFFFFFFFFu;
    } else if(exp_val > 0.0L){
        long double v = powl(2.0L, exp_val);
        if(!isfinite((double)v)){
            for(int i=0;i<8;i++) tgt[i]=0xFFFFFFFFu;
        } else {
            long double base = 4294967296.0L;
            for(int i=0;i<8;i++){
                long double rem = fmodl(v, base);
                if(rem < 0.0L) rem = 0.0L;
                if(rem > 4294967295.0L) rem = 4294967295.0L;
                tgt[i] = (uint32_t)rem;
                v = floorl(v / base);
                if(v <= 0.0L) break;
            }
        }
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

void cp_scale_jackpot_target(const uint32_t pool_tgt[8], uint32_t bound[8])
{
    uint64_t factor = (uint64_t)cp_active_hash_h() * (uint64_t)cp_active_hash_w()
                    * (uint64_t)K_DIM;
    memcpy(bound, pool_tgt, 8 * sizeof(uint32_t));
    cp_scale_target_le(bound, factor);
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
    if(contiguous) return m / PP_HASH_H;
    return (m / 128) * 16;
}

int cp_pp_num_col_parts(int n, int contiguous)
{
    if(g_cutlass_fused) return n / CP_CUTLASS_HASH_W;
    if(contiguous) return n / PP_HASH_W;
    return (n / 256) * 16;
}

int cp_pp_num_row_periods(int m, int contiguous)
{
    if(g_cutlass_fused) return m / CP_CUTLASS_CTA_M;
    if(contiguous) return m / PP_HASH_H;
    return m / 128;
}

int cp_pp_num_col_periods(int n, int contiguous)
{
    if(g_cutlass_fused) return n / CP_CUTLASS_CTA_N;
    if(contiguous) return n / PP_HASH_W;
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
