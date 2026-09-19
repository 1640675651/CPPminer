#include "cp_qpow_pool.h"

#include "cp_job_ctrl.h"
#include "cp_pool.h"
#include "cp_util.h"

#include <cstring>
#include <mutex>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_qpow_active = 0;
static char g_session_id[80] = {0};
static std::mutex g_pending_mx;
static CpQpowJob g_pending;
static int g_pending_valid = 0;

static int json_str_from(const char* json, const char* key, char* out, int outlen)
{
    if(!json) return 0;
    return cp_json_str(json, key, out, outlen);
}

static void queue_pending(const CpQpowJob* job)
{
    std::lock_guard<std::mutex> lk(g_pending_mx);
    g_pending = *job;
    g_pending_valid = 1;
}

void cp_qpow_pool_set_active(int on)
{
    g_qpow_active = on ? 1 : 0;
}

void cp_qpow_pool_clear(void)
{
    std::lock_guard<std::mutex> lk(g_pending_mx);
    g_pending_valid = 0;
    memset(&g_pending, 0, sizeof(g_pending));
    g_session_id[0] = 0;
}

void cp_qpow_pool_set_session_id(const char* id)
{
    if(!id){
        g_session_id[0] = 0;
        return;
    }
    strncpy(g_session_id, id, sizeof(g_session_id) - 1);
    g_session_id[sizeof(g_session_id) - 1] = 0;
}

const char* cp_qpow_pool_session_id(void)
{
    return g_session_id;
}

int cp_qpow_pool_send_login(int msg_id, const char* login, const char* worker,
                            const char* agent)
{
    char msg[768];
    snprintf(msg, sizeof(msg),
             "{\"id\":%d,\"method\":\"login\",\"params\":{"
             "\"login\":\"%s\",\"pass\":\"x\",\"worker\":\"%s\",\"agent\":\"%s\"}}",
             msg_id, login ? login : "",
             worker ? worker : "rig01",
             agent ? agent : "cppminer/1.0");
    printf("[net] Quantus login (login/worker/agent, pass=x)\n");
    fflush(stdout);
    return cp_send_json(cp_pool_socket(), msg);
}

int cp_qpow_pool_send_submit(int sock, int msg_id, const char* job_id,
                             const uint8_t nonce[CP_QPOW_NONCE_BYTES])
{
    char nonce_hex[CP_QPOW_NONCE_BYTES * 2 + 1];
    cp_bin_to_hex(nonce, CP_QPOW_NONCE_BYTES, nonce_hex);
    char msg[512];
    int nw = snprintf(msg, sizeof(msg),
                      "{\"id\":%d,\"method\":\"submit\",\"params\":{"
                      "\"id\":\"%s\",\"job_id\":\"%s\",\"nonce\":\"%s\"}}",
                      msg_id, g_session_id, job_id ? job_id : "", nonce_hex);
    if(nw < 0 || (size_t)nw >= sizeof(msg)){
        fprintf(stderr, "[net] quantus submit JSON too large\n");
        return 0;
    }
    printf("[net] quantus submit job=%s nonce=%.16s...\n", job_id, nonce_hex);
    fflush(stdout);
    int ok = cp_send_json(sock, msg);
    if(ok) cp_pool_set_submit_inflight(1);
    return ok;
}

int cp_qpow_pool_parse_job(const char* json, CpQpowJob* out)
{
    if(!json || !out) return 0;
    memset(out, 0, sizeof(*out));

    if(!json_str_from(json, "job_id", out->job_id, (int)sizeof(out->job_id)))
        return 0;

    char mh[CP_QPOW_HEADER_BYTES * 2 + 4] = {0};
    char th[CP_QPOW_TARGET_BYTES * 2 + 4] = {0};
    char en[CP_QPOW_EXTRANONCE_MAX * 2 + 4] = {0};
    if(!json_str_from(json, "mining_hash", mh, (int)sizeof(mh))) return 0;
    if(!json_str_from(json, "target", th, (int)sizeof(th))) return 0;
    json_str_from(json, "extranonce", en, (int)sizeof(en));

    int mhlen = cp_hex_to_bytes(mh, out->mining_hash, CP_QPOW_HEADER_BYTES);
    int thlen = cp_hex_to_bytes(th, out->target, CP_QPOW_TARGET_BYTES);
    if(mhlen != CP_QPOW_HEADER_BYTES || thlen != CP_QPOW_TARGET_BYTES) return 0;

    if(en[0]){
        int elen = (int)strlen(en);
        if(elen & 1) return 0;
        out->extranonce_len = elen / 2;
        if(out->extranonce_len > CP_QPOW_EXTRANONCE_MAX) return 0;
        if(cp_hex_to_bytes(en, out->extranonce, out->extranonce_len) !=
           out->extranonce_len)
            return 0;
    }

    out->difficulty = cp_json_num(json, "difficulty");
    out->seq = (uint64_t)cp_json_num(json, "seq");
    out->clean_jobs = strstr(json, "\"clean_jobs\":true") != NULL ||
                      strstr(json, "\"clean_jobs\": true") != NULL;

    snprintf(out->job_key, sizeof(out->job_key), "%s:%.16s", out->job_id, mh);
    return 1;
}

int cp_qpow_pool_parse_login_result(const char* json, char* session_out, int session_len,
                                    CpQpowJob* job_out)
{
    if(!json) return 0;
    const char* result = strstr(json, "\"result\"");
    if(!result) return 0;
    if(strstr(result, "\"status\":\"OK\"") == NULL &&
       strstr(result, "\"status\": \"OK\"") == NULL){
        /* Some pools omit status; require session id instead. */
    }
    if(session_out && session_len > 0){
        session_out[0] = 0;
        if(!json_str_from(result, "id", session_out, session_len) || !session_out[0])
            return 0;
    }
    if(job_out){
        if(!cp_qpow_pool_parse_job(result, job_out)) return 0;
    }
    return 1;
}

int cp_qpow_pool_on_line(const char* line)
{
    if(!g_qpow_active || !line) return 0;

    /* Job notification (and login-embedded jobs are handled by main). */
    const int is_job_method =
        (strstr(line, "\"method\":\"job\"") != NULL) ||
        (strstr(line, "\"method\": \"job\"") != NULL);
    if(!is_job_method) return 0;

    CpQpowJob job;
    if(!cp_qpow_pool_parse_job(line, &job)){
        printf("[pool] quantus job parse failed\n");
        fflush(stdout);
        return 1;
    }

    if(cp_job_mining_active()){
        if(cp_job_key_matches(job.job_key)) return 1;
        cp_job_request_cancel();
        queue_pending(&job);
        printf("[net] new quantus job %s while mining %s - cancelling\n",
               job.job_id, cp_job_mining_key());
        fflush(stdout);
        return 1;
    }

    /* Idle: push raw line to inbox for main wait loop. */
    return 0; /* let default inbox push happen */
}

int cp_qpow_pool_take_pending(CpQpowJob* out)
{
    std::lock_guard<std::mutex> lk(g_pending_mx);
    if(!g_pending_valid) return 0;
    if(out) *out = g_pending;
    g_pending_valid = 0;
    return 1;
}
