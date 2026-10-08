#include "cp_qpow_pool.h"

#include "cp_job_ctrl.h"
#include "cp_json_frame.h"
#include "cp_json_text.hpp"
#include "cp_pool.h"
#include "cp_state.h"
#include "cp_util.h"

#include <atomic>
#include <cstring>
#include <mutex>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static std::atomic<int> g_qpow_active{0};
static char g_session_id[80] = {0};

static int json_str_from(const char* json, const char* key, char* out, int outlen)
{
    if(!json) return 0;
    return cp_json_str_value(cp_json_member(json, key), out, outlen);
}

static int json_object(const char* value, std::string& out)
{
    size_t len = 0;
    if(!value || *value != '{' ||
       cp_json_object_length(value, strlen(value), &len) != 1) return 0;
    out.assign(value, len);
    return 1;
}

void cp_qpow_pool_set_active(int on)
{
    g_qpow_active.store(on ? 1 : 0);
}

void cp_qpow_pool_clear(void)
{
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
    const std::string w = worker ? worker : "";
    const std::string ident = std::string(login ? login : "") +
                              (w.empty() ? "" : "." + w);
    const std::string msg =
        "{\"id\":" + std::to_string(msg_id) +
        ",\"method\":\"login\",\"params\":{\"login\":\"" +
        cp_json_escape(ident.c_str()) + "\",\"pass\":\"" +
        cp_json_escape(w.empty() ? "x" : w.c_str()) + "\",\"agent\":\"" +
        cp_json_escape(agent ? agent : "cppminer/1.0") + "\"}}";
    printf("[net] Quantus login (login=wallet.worker, pass=worker)\n");
    fflush(stdout);
    return cp_send_json(cp_pool_socket(), msg.c_str());
}

int cp_qpow_pool_send_submit(int sock, int msg_id, const char* job_id,
                             const uint8_t nonce[CP_QPOW_NONCE_BYTES])
{
    char nonce_hex[CP_QPOW_NONCE_BYTES * 2 + 1];
    cp_bin_to_hex(nonce, CP_QPOW_NONCE_BYTES, nonce_hex);
    const std::string msg =
        "{\"id\":" + std::to_string(msg_id) +
        ",\"method\":\"submit\",\"params\":{\"id\":\"" +
        cp_json_escape(g_session_id) + "\",\"job_id\":\"" +
        cp_json_escape(job_id) + "\",\"nonce\":\"" + nonce_hex + "\"}}";
    printf("[net] quantus submit job=%s nonce=%.16s...\n", job_id ? job_id : "", nonce_hex);
    fflush(stdout);
    return cp_pool_send_tracked_submit(sock, msg_id, msg.c_str());
}

int cp_qpow_pool_submit_share(const CpQpowJob* job, int sock, int* msg_id,
                               const uint8_t nonce[CP_QPOW_NONCE_BYTES], int tid)
{
    if(!job || cp_job_should_cancel() || !cp_job_key_matches(job->job_key) || cp_pool_conn_lost()){
        printf("[qpow] stale share dropped before submit (tid=%d)\n", tid);
        fflush(stdout);
        return 0;
    }
    if(g_dry_run){
        char nh[CP_QPOW_NONCE_BYTES * 2 + 1];
        cp_bin_to_hex(nonce, CP_QPOW_NONCE_BYTES, nh);
        printf("[qpow] dry-run share nonce=%s (tid=%d)\n", nh, tid);
        fflush(stdout);
        return 1;
    }
    if(sock < 0 || !msg_id) return 0;
    const int sid = (*msg_id)++;
    if(!cp_qpow_pool_send_submit(sock, sid, job->job_id, nonce)){
        printf("[qpow] submit send failed\n");
        fflush(stdout);
        return 0;
    }
    cp_pool_log_share_submit_outcome();
    return 1;
}

int cp_qpow_pool_parse_job(const char* json, CpQpowJob* out)
{
    if(!cp_json_valid(json) || !out) return 0;
    memset(out, 0, sizeof(*out));

    std::string params, nested_job;
    if(cp_json_member(json, "method")){
        char method[16];
        if(!json_str_from(json, "method", method, sizeof(method)) || strcmp(method, "job") ||
           !json_object(cp_json_member(json, "params"), params)) return 0;
        json = params.c_str();
    }
    /* clean_jobs can be alongside params.job, as well as in a direct job. */
    const char* clean = cp_json_member(json, "clean_jobs");
    if(cp_json_member(json, "job")){
        if(!json_object(cp_json_member(json, "job"), nested_job)) return 0;
        json = nested_job.c_str();
    }
    if(!clean) clean = cp_json_member(json, "clean_jobs");
    if(clean && strncmp(clean, "true", 4) && strncmp(clean, "false", 5)) return 0;
    out->clean_jobs = clean && !strncmp(clean, "true", 4);

    if(!json_str_from(json, "job_id", out->job_id, (int)sizeof(out->job_id)) || !out->job_id[0])
        return 0;

    char mh[CP_QPOW_HEADER_BYTES * 2 + 4] = {0};
    char th[CP_QPOW_TARGET_BYTES * 2 + 4] = {0};
    char en[CP_QPOW_EXTRANONCE_MAX * 2 + 4] = {0};
    if(!json_str_from(json, "mining_hash", mh, (int)sizeof(mh))) return 0;
    if(!json_str_from(json, "target", th, (int)sizeof(th))) return 0;
    if(cp_json_member(json, "extranonce") &&
       !json_str_from(json, "extranonce", en, (int)sizeof(en))) return 0;

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

    const char* difficulty = cp_json_member(json, "difficulty");
    if(difficulty && (!cp_json_number_value(difficulty, &out->difficulty) || out->difficulty <= 0))
        return 0;
    const char* seq = cp_json_member(json, "seq");
    if(seq && !cp_json_uint64_value(seq, &out->seq)) return 0;

    /* Canonical bytes: case-only hex changes are duplicates, but every work field matters. */
    cp_bin_to_hex(out->mining_hash, sizeof(out->mining_hash), mh);
    cp_bin_to_hex(out->target, sizeof(out->target), th);
    cp_bin_to_hex(out->extranonce, (size_t)out->extranonce_len, en);
    const int size = snprintf(out->job_key, sizeof(out->job_key), "%s:%s:%s:%s",
                              out->job_id, mh, th, en);
    return size >= 0 && (size_t)size < sizeof(out->job_key);
}

int cp_qpow_pool_parse_login_result(const char* json, int expected_id, char* session_out, int session_len,
                                    CpQpowJob* job_out)
{
    if(!session_out || session_len <= 0) return 0;
    session_out[0] = 0;
    if(job_out) memset(job_out, 0, sizeof(*job_out));
    int id = -1, accepted = 0;
    if(!cp_json_rpc_response(json, &id, &accepted) || id != expected_id || !accepted)
        return 0;
    std::string result;
    if(!json_object(cp_json_member(json, "result"), result)) return 0;
    char status[16], session[80];
    if(cp_json_member(result.c_str(), "status") &&
       (!json_str_from(result.c_str(), "status", status, sizeof(status)) || strcmp(status, "OK")))
        return 0;
    if(!json_str_from(result.c_str(), "id", session, sizeof(session)) || !session[0] ||
       strlen(session) >= (size_t)session_len) return 0;
    if(job_out && cp_json_member(result.c_str(), "job") &&
       !cp_qpow_pool_parse_job(result.c_str(), job_out)) return 0;
    strcpy(session_out, session);
    return 1;
}

int cp_qpow_pool_wait_login(int expected_id, char* session_out, int session_len,
                             CpQpowJob* job_out)
{
    if(!session_out || session_len <= 0 || !job_out) return 0;
    session_out[0] = 0;
    memset(job_out, 0, sizeof(*job_out));
    const double deadline = cp_now_sec() + 30.0;
    int authorized = 0;
    while(cp_now_sec() < deadline){
        char line[65536];
        const int remaining = (int)((deadline - cp_now_sec()) * 1000);
        if(remaining <= 0 || cp_pool_recv_one(line, sizeof(line), remaining) <= 0) break;
        printf("[pool-raw] %s\n", line);
        fflush(stdout);
        int id = -1;
        if(!authorized && cp_json_rpc_response(line, &id, nullptr) && id == expected_id){
            CpQpowJob embedded;
            if(!cp_qpow_pool_parse_login_result(line, expected_id, session_out, session_len, &embedded)){
                printf("[net] Quantus login rejected or malformed\n");
                fflush(stdout);
                return 0;
            }
            authorized = 1;
            if(!job_out->job_id[0]) *job_out = embedded;
        }else{
            char method[16];
            CpQpowJob early;
            if(json_str_from(line, "method", method, sizeof(method)) && !strcmp(method, "job") &&
               cp_qpow_pool_parse_job(line, &early)) *job_out = early;
        }
        if(authorized && job_out->job_id[0]) return 1;
    }
    printf("[net] Quantus login/first job missing or timed out (30 s budget)\n");
    fflush(stdout);
    return 0;
}

int cp_qpow_pool_on_line(const char* line)
{
    if(!g_qpow_active.load() || !line) return 0;

    /* Job notification (and login-embedded jobs are handled by main). */
    char method[16];
    const int is_job_method = json_str_from(line, "method", method, sizeof(method)) &&
                              !strcmp(method, "job");
    if(!is_job_method) return 0;

    CpQpowJob job;
    if(!cp_qpow_pool_parse_job(line, &job)){
        printf("[pool] quantus job parse failed\n");
        fflush(stdout);
        return 1;
    }

    if(cp_job_mining_active() && !cp_job_key_matches(job.job_key)){
        printf("[net] new quantus job %s while mining %s - cancelling\n",
               job.job_id, cp_job_mining_key());
        fflush(stdout);
    }

    cp_pool_publish_quantus(&job);
    return 1;
}
