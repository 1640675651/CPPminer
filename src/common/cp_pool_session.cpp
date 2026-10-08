#include "cp_api.h"
#include "cp_pool_session.h"
#include "cp_config.h"
#include "cp_fee.h"
#include "cp_mine.h"
#include "cp_platform.h"
#include "cp_pool.h"
#include "cp_qpow_mine.h"
#include "cp_qpow_pool.h"
#include "cp_share_queue.h"
#include "cp_state.h"
#include "cp_util.h"
#include "cp_worker.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Remembers the job being mined; key_size is the size of the caller's cur_job_key buffer. */
static void set_cur_job_key(char* cur_job_key, size_t key_size, const char* job_key)
{
    strncpy(cur_job_key, job_key, key_size - 1);
    cur_job_key[key_size - 1] = 0;
}

static int handle_notify_line(const char* line, int* msg_id, char* cur_job_key, size_t key_size)
{
    char job_id[128] = {0};
    char header_hex[320] = {0};
    char target_hex[80] = {0};
    uint32_t cert_version = 0;
    if(!cp_pool_parse_notify(line, job_id, sizeof(job_id),
                            header_hex, sizeof(header_hex),
                            target_hex, sizeof(target_hex),
                            &cert_version)){
        printf("[pool] mining.notify parse failed\n"); fflush(stdout);
        return CP_JOB_NONE;
    }
    cert_version = cp_resolve_cert_version(cert_version);

    char job_key[320];
    snprintf(job_key, sizeof(job_key), "%s:%.16s", job_id, header_hex);
    if(!strcmp(job_key, cur_job_key)){
        printf("[pool] duplicate notify ignored job=%s\n", job_id); fflush(stdout);
        return CP_JOB_NONE;
    }
    set_cur_job_key(cur_job_key, key_size, job_key);

    uint8_t header[INCOMPLETE_HEADER_BYTES];
    int hlen = cp_hex_to_bytes(header_hex, header, INCOMPLETE_HEADER_BYTES);
    if(hlen != INCOMPLETE_HEADER_BYTES){
        printf("[pool] bad header length %d (need %d)\n", hlen, INCOMPLETE_HEADER_BYTES);
        fflush(stdout);
        return CP_JOB_NONE;
    }

    uint32_t tgt[8];
    memset(tgt, 0, sizeof(tgt));
    if(target_hex[0] && cp_be_target_hex_to_le_words(target_hex, tgt)){
        printf("[job] notify id=%s header=%.16s... pool_target (unscaled) cert_version=%u\n",
               job_id, header_hex, (unsigned)cert_version);
    } else {
        /* An unusable target is mined like a missing one; kept, it would fail proof
         * verification and drop every share. */
        const int bad_target = target_hex[0] != 0;
        target_hex[0] = 0;
        cp_pool_target_from_difficulty(cp_pool_difficulty(), tgt);
        printf("[job] notify id=%s header=%.16s... diff=%.1f (%s) cert_version=%u\n",
               job_id, header_hex, cp_pool_difficulty(),
               bad_target ? "notify target unusable" : "no target in notify",
               (unsigned)cert_version);
    }
    fflush(stdout);

    printf("[plain] mining job=%s%s...\n", job_id,
           cp_fee_next_is_dev() ? " [DEV FEE]" : "");
    fflush(stdout);
    int rc = cp_mine_job(header, hlen, job_id, target_hex, tgt, cert_version,
                         cp_pool_socket(), msg_id);
    if(rc == CP_JOB_FEE_SWITCH){
        printf("[fee] pausing job for wallet switch\n"); fflush(stdout);
        return rc;
    }
    if(rc == CP_JOB_ERROR) return rc;
    if(rc == CP_JOB_CANCELLED){
        printf("[plain] job ended (new notify or disconnect)\n"); fflush(stdout);
    } else if(rc == CP_JOB_NONE){
        printf("[plain] job stopped (max_nonce or error)\n"); fflush(stdout);
    }

    CpPendingJob pj;
    while(rc == CP_JOB_CANCELLED && cp_pool_take_pending_job(&pj)){
        set_cur_job_key(cur_job_key, key_size, pj.job_key);
        printf("[plain] mining queued job=%s%s...\n", pj.job_id,
               cp_fee_next_is_dev() ? " [DEV FEE]" : "");
        fflush(stdout);
        rc = cp_mine_job(pj.header, INCOMPLETE_HEADER_BYTES, pj.job_id,
                         pj.target_hex, pj.tgt, pj.cert_version, cp_pool_socket(), msg_id);
        if(rc == CP_JOB_FEE_SWITCH){
            printf("[fee] pausing job for wallet switch\n"); fflush(stdout);
            return rc;
        }
        if(rc == CP_JOB_ERROR) return rc;
        if(rc == CP_JOB_CANCELLED){
            printf("[plain] job ended (new notify or disconnect)\n"); fflush(stdout);
        } else if(rc == CP_JOB_NONE){
            printf("[plain] job stopped (max_nonce or error)\n"); fflush(stdout);
        }
    }
    return rc;
}

static int handle_qpow_job(const CpQpowJob* job, int* msg_id, char* cur_job_key, size_t key_size)
{
    if(!job || !job->job_id[0]) return CP_JOB_NONE;
    if(!strcmp(job->job_key, cur_job_key)){
        printf("[pool] duplicate quantus job ignored id=%s\n", job->job_id);
        fflush(stdout);
        return CP_JOB_NONE;
    }
    set_cur_job_key(cur_job_key, key_size, job->job_key);

    printf("[qpow] job id=%s mining_hash=%.16s... diff=%.0f\n", job->job_id,
           job->job_key + (int)strlen(job->job_id) + 1, job->difficulty);
    fflush(stdout);

    int rc = cp_qpow_mine_job(job, cp_pool_socket(), msg_id, worker_global);
    if(rc == CP_JOB_FEE_SWITCH){
        printf("[fee] pausing quantus job for wallet switch\n");
        fflush(stdout);
        return rc;
    }
    if(rc == CP_JOB_ERROR) return rc;
    if(rc == CP_JOB_CANCELLED){
        printf("[qpow] job ended (new job or disconnect)\n");
        fflush(stdout);
    }

    CpQpowJob pj;
    while(rc == CP_JOB_CANCELLED && cp_qpow_pool_take_pending(&pj)){
        set_cur_job_key(cur_job_key, key_size, pj.job_key);
        printf("[qpow] mining queued job=%s%s...\n", pj.job_id,
               cp_fee_next_is_dev() ? " [DEV FEE]" : "");
        fflush(stdout);
        rc = cp_qpow_mine_job(&pj, cp_pool_socket(), msg_id, worker_global);
        if(rc == CP_JOB_FEE_SWITCH){
            printf("[fee] pausing quantus job for wallet switch\n");
            fflush(stdout);
            return rc;
        }
        if(rc == CP_JOB_ERROR) return rc;
        if(rc == CP_JOB_CANCELLED){
            printf("[qpow] job ended (new job or disconnect)\n");
            fflush(stdout);
        }
    }
    return rc;
}

/* Pool reconnect backoff. Reconnecting after a session that got a job (fee switch, dropped
 * connection) is immediate; each further attempt that gets no job (connect refused, login
 * rejected, pool hung up) waits 5 s, doubling up to 60 s, so a failing pool is not hammered. */
#define CP_RECONNECT_DELAY_MIN_SEC 5
#define CP_RECONNECT_DELAY_MAX_SEC 60
static int g_reconnect_delay_sec = 0;

static void pool_reconnect_wait(void)
{
    const int d = g_reconnect_delay_sec;
    if(d > 0){
        printf("[main] Reconnecting in %d sec...\n", d);
        fflush(stdout);
        cp_sleep(d);
    }
    g_reconnect_delay_sec = d == 0 ? CP_RECONNECT_DELAY_MIN_SEC
                          : (d * 2 < CP_RECONNECT_DELAY_MAX_SEC ? d * 2 : CP_RECONNECT_DELAY_MAX_SEC);
}

static void pool_session_ok(void)
{
    g_reconnect_delay_sec = 0;
}

/* CP_JOB_ERROR: mining cannot continue on this backend; stop instead of reconnecting. */
static int pool_session_fatal(void)
{
    printf("[main] stopping: backend or resource failure (see the error above)\n");
    fflush(stdout);
    cp_pool_reader_stop();
    cp_pool_disconnect();
    return 1;
}

int cp_run_quantus_pool(const char* pool_host, int pool_port)
{
    char cur_job_key[160] = {0};
    int msg_id = 1;

    cp_qpow_pool_set_active(1);
    printf("[mode] algo=quantus backend=%s\n", cp_worker_backend_name());
    if(cp_fee_enabled())
        printf("[mode] dev fee: 1%%\n");
    fflush(stdout);

reconnect:
    cp_pool_reader_stop();
    cp_pool_disconnect();
    cp_pool_inbox_clear();
    cp_api_set_pool_connected(0);
    cp_qpow_pool_clear();
    cur_job_key[0] = 0;

    while(1){
        pool_reconnect_wait();
        printf("[main] Connecting to %s:%d (quantus)...\n", pool_host, pool_port);
        fflush(stdout);
        if(cp_pool_connect(pool_host, pool_port)) break;
    }

    const int login_id = msg_id++;
    if(!cp_qpow_pool_send_login(login_id, cp_fee_wallet(), worker_global, agent_global))
        goto reconnect;

    char session[80] = {0};
    CpQpowJob first_job;
    if(!cp_qpow_pool_wait_login(login_id, session, (int)sizeof(session), &first_job))
        goto reconnect;
    /* Reset the reconnect backoff only once a job arrives (as for Pearl). */
    if(first_job.job_id[0]){
        pool_session_ok();
        cp_pool_note_job();
    } else {
        cp_pool_expect_job();
    }
    cp_qpow_pool_set_session_id(session);
    cp_fee_on_authorized();
    cp_api_set_pool(pool_host, pool_port, 1);
    if(cp_fee_enabled()){
        char debt_s[48], thr_s[48];
        printf("[fee] logged in as %s (debt=%s / 100*T=%s of hashing time)\n",
               cp_fee_next_is_dev() ? "DEV FEE wallet" : "your wallet",
               cp_fee_format(cp_fee_debt(), debt_s, sizeof(debt_s)),
               cp_fee_format(cp_fee_threshold(), thr_s, sizeof(thr_s)));
        fflush(stdout);
    }
    printf("[net] session=%s first_job=%s\n", session, first_job.job_id);
    fflush(stdout);

    cp_pool_reader_start();

    {
        int rc = handle_qpow_job(&first_job, &msg_id, cur_job_key, sizeof(cur_job_key));
        if(rc == CP_JOB_ERROR) return pool_session_fatal();
        if(rc == CP_JOB_FEE_SWITCH || cp_pool_conn_lost()) goto reconnect;
    }

    while(1){
        char line_buf[65536];
        int wr = cp_pool_wait_line(line_buf, sizeof(line_buf), -1);
        if(wr < 0){
            printf("[net] Connection lost, reconnecting...\n");
            fflush(stdout);
            goto reconnect;
        }
        if(wr == 0) continue;

        if(cp_qpow_pool_is_job(line_buf)){
            CpQpowJob job;
            if(!cp_qpow_pool_parse_job(line_buf, &job)){
                printf("[pool] quantus job parse failed\n");
                fflush(stdout);
                continue;
            }
            pool_session_ok();
            int rc = handle_qpow_job(&job, &msg_id, cur_job_key, sizeof(cur_job_key));
            if(rc == CP_JOB_ERROR) return pool_session_fatal();
            if(rc == CP_JOB_FEE_SWITCH || cp_pool_conn_lost()) goto reconnect;
            continue;
        }

        if(strstr(line_buf, "result") || strstr(line_buf, "error")){
            printf("[pool] jsonrpc: %s\n", line_buf);
            fflush(stdout);
            continue;
        }

        printf("[pool] (unhandled) %s\n", line_buf);
        fflush(stdout);
    }
}

int cp_run_pearl_pool(const char* pool_host, int pool_port)
{
    char cur_job_key[320] = {0};
    int msg_id = 1;

reconnect:
    cp_pool_reader_stop();
    cp_pool_disconnect();
    cp_pool_inbox_clear();
    cp_api_set_pool_connected(0);
    cur_job_key[0] = 0;

    while(1){
        pool_reconnect_wait();
        printf("[main] Connecting to %s:%d...\n", pool_host, pool_port); fflush(stdout);
        if(cp_pool_connect(pool_host, pool_port)) break;
    }

    if(!cp_pool_send_authorize(msg_id++, cp_fee_wallet(), worker_global, agent_global,
                               pool_pass_global))
        goto reconnect;
    cp_fee_on_authorized();
    cp_api_set_pool(pool_host, pool_port, 1);
    if(cp_fee_enabled()){
        printf("[fee] authorized as %s (debt=%llu / 100*T=%llu)\n",
               cp_fee_next_is_dev() ? "DEV FEE wallet" : "your wallet",
               (unsigned long long)cp_fee_debt(),
               (unsigned long long)cp_fee_threshold());
        fflush(stdout);
    }

    cp_pool_reader_start();

    while(1){
        char line_buf[65536];
        int got = cp_pool_wait_line(line_buf, sizeof(line_buf), -1);
        if(got < 0){
            printf("[net] Connection lost, reconnecting...\n"); fflush(stdout);
            goto reconnect;
        }
        if(got == 0) continue;

        if(strstr(line_buf, "mining.notify")){
            pool_session_ok();
            int rc = handle_notify_line(line_buf, &msg_id, cur_job_key, sizeof(cur_job_key));
            if(rc == CP_JOB_ERROR) return pool_session_fatal();
            if(rc == CP_JOB_FEE_SWITCH || cp_pool_conn_lost()) goto reconnect;
            continue;
        }

        if(strstr(line_buf, "mining.set_difficulty")){
            const double d = cp_pool_parse_difficulty(line_buf);
            if(d > 0.0){
                cp_pool_set_difficulty(d);
                printf("[pool] mining.set_difficulty %.0f\n", d); fflush(stdout);
            }
            continue;
        }

        if(strstr(line_buf, "result") || strstr(line_buf, "error")){
            printf("[pool] jsonrpc: %s\n", line_buf); fflush(stdout);
            continue;
        }

        printf("[pool] (unhandled) %s\n", line_buf); fflush(stdout);
    }
}
