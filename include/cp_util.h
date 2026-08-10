#ifndef CP_UTIL_H
#define CP_UTIL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

double cp_now_sec(void);
int cp_file_exists(const char* path);
void cp_path_to_posix(char* path);
void cp_path_abs(char* path, size_t cap);

void cp_init_workdir(void);
void cp_resolve_paths(int argc, char** argv);
int cp_run_python(const char* subcmd);

int cp_read_file_bin(const char* path, void* buf, size_t nbytes);
int cp_read_file_text(const char* path, char* out, int cap);
int cp_write_file_bin(const char* path, const void* buf, size_t nbytes);

void cp_bin_to_hex(const uint8_t* in, size_t n, char* out);
int cp_hex_to_bytes(const char* hex, uint8_t* out, int out_cap);

int cp_json_str(const char* json, const char* key, char* out, int outlen);
double cp_json_num(const char* json, const char* key);

void cp_target_from_difficulty(double difficulty, uint32_t tgt[8]);
int cp_be_target_hex_to_le_words(const char* hex, uint32_t tgt[8]);
/* Inverse of cp_be_target_hex_to_le_words. Writes 64 hex chars + NUL into hex[65]. */
void cp_le_words_to_be_target_hex(const uint32_t tgt[8], char hex[65]);
void cp_scale_target_le(uint32_t tgt[8], uint64_t factor);
void cp_scale_jackpot_target(const uint32_t pool_tgt[8], uint32_t bound[8]);

int cp_send_all(int sock, const void* data, size_t len);
int cp_send_json(int sock, const char* json);

int cp_pp_num_row_parts(int m, int contiguous);
int cp_pp_num_col_parts(int n, int contiguous);
int cp_pp_num_row_periods(int m, int contiguous);
int cp_pp_num_col_periods(int n, int contiguous);
double cp_pp_macs_per_hash_tile(void);
void cp_pp_set_hash_tile(int h, int w);
double cp_pp_mac_rate_from_tiles(uint64_t tiles_scanned, double elapsed_sec);
void cp_pp_fmt_mac_rate(double mac_s, char* out, size_t out_sz);
void cp_log_attempt_timing(const char* tag, double prep_sec, double scan_sec, uint64_t tiles,
                           double post_sec);

/* Fill buf with cryptographically secure random bytes. Returns 0 on success, -1 on failure. */
int cp_random_bytes(void* buf, size_t n);
/* Convenience: 64-bit CSPRNG value (never zero). Returns 0 on success, -1 on failure. */
int cp_random_u64(uint64_t* out);

#ifdef __cplusplus
}
#endif

#endif /* CP_UTIL_H */
