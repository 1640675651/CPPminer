/*
 * Read-only HTTP stats API (--api-port): JSON summary for dashboards, HiveOS and containers.
 *
 *   GET /  or /summary   full summary (algo, backend, devices, hashrate windows, shares)
 *   GET /hiveos          {"khs": total, "stats": {...}} in HiveOS h-stats.sh form
 *
 * Hashrate is in H/s. For Pearl one hash is one int8 multiply-accumulate (TMAC/s = TH/s), the
 * unit pools and other Pearl miners report. The hashrate is what the hardware does, dev-fee time
 * included; shares count only the user's own pool session.
 */
#ifndef CP_API_H
#define CP_API_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Start the server thread. bind_addr NULL = 127.0.0.1. Returns 0 on success; on failure mining
 * goes on without the API. */
int cp_api_start(const char* bind_addr, int port);
int cp_api_enabled(void);

void cp_api_set_algo(const char* algo, const char* backend);
void cp_api_set_pool(const char* host, int port, int connected);
void cp_api_set_pool_connected(int connected);

/* One mining device of this process. pci: "0000:03:00.0" or NULL when unknown. */
void cp_api_add_device(const char* name, const char* pci);
int cp_api_device_count(void);

/* Work done since the last call: hashes (Quantus) or multiply-accumulates (Pearl). */
void cp_api_add_work(double units);

/* A pool reply to one of our submits. */
void cp_api_on_share(int accepted);

#ifdef __cplusplus
}
#endif

#endif /* CP_API_H */
