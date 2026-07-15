#include "cp_pool.h"
#include "cp_config.h"
#include "cp_job_ctrl.h"
#include "cp_platform.h"
#include "cp_util.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <thread>

static int tcp_sock = -1;
static double g_diff = 32.0;
static int g_submit_inflight = 0;
static std::atomic<int> g_net_reader_run{0};
static std::atomic<int> g_net_conn_lost{0};
static std::mutex g_net_mx;
static std::mutex g_pending_mx;
static CpPendingJob g_pending_job;
static int g_pending_valid = 0;
static std::thread g_net_reader;
static std::deque<std::string> g_pool_inbox;
static std::mutex g_inbox_mx;
static std::condition_variable g_inbox_cv;
static char net_buf[65536];
static int net_pos = 0;
static char json_msg[65536];

static int tcp_connect(const char* host, int port)
{
    if(cp_net_init() != 0){
        fprintf(stderr, "WSAStartup failed\n");
        return (int)CP_INVALID_SOCK;
    }
    struct hostent* he = gethostbyname(host);
    if(!he){ perror("gethostbyname"); return (int)CP_INVALID_SOCK; }
    cp_sock_t s = socket(AF_INET, SOCK_STREAM, 0);
    if(s == CP_INVALID_SOCK){ perror("socket"); return (int)CP_INVALID_SOCK; }
    struct sockaddr_in sa;
    memset(&sa, 0, sizeof(sa));
    sa.sin_family = AF_INET;
    sa.sin_port = htons((u_short)port);
    memcpy(&sa.sin_addr, he->h_addr_list[0], (size_t)he->h_length);
    if(connect(s, (struct sockaddr*)&sa, sizeof(sa)) < 0){
        perror("connect");
        CP_SOCK_CLOSE(s);
        return (int)CP_INVALID_SOCK;
    }
    return (int)s;
}

static void queue_pending_job(
    const char* job_id, const char* job_key,
    const uint8_t* header, const char* target_hex, const uint32_t tgt[8])
{
    std::lock_guard<std::mutex> lk(g_pending_mx);
    strncpy(g_pending_job.job_id, job_id, sizeof(g_pending_job.job_id) - 1);
    g_pending_job.job_id[sizeof(g_pending_job.job_id) - 1] = 0;
    strncpy(g_pending_job.job_key, job_key, sizeof(g_pending_job.job_key) - 1);
    g_pending_job.job_key[sizeof(g_pending_job.job_key) - 1] = 0;
    strncpy(g_pending_job.target_hex, target_hex, sizeof(g_pending_job.target_hex) - 1);
    g_pending_job.target_hex[sizeof(g_pending_job.target_hex) - 1] = 0;
    memcpy(g_pending_job.header, header, INCOMPLETE_HEADER_BYTES);
    memcpy(g_pending_job.tgt, tgt, 8 * sizeof(uint32_t));
    g_pending_valid = 1;
}

static void pool_inbox_push(const char* line)
{
    std::lock_guard<std::mutex> lk(g_inbox_mx);
    g_pool_inbox.emplace_back(line);
    g_inbox_cv.notify_one();
}

static int net_wait_readable(int sock, int timeout_ms)
{
    fd_set fds;
    FD_ZERO(&fds);
    FD_SET((cp_sock_t)sock, &fds);
    struct timeval tv;
    tv.tv_sec = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;
#ifdef _WIN32
    return select(0, &fds, NULL, NULL, &tv) > 0;
#else
    return select(sock + 1, &fds, NULL, NULL, &tv) > 0;
#endif
}

static int net_buf_has_complete_json(void)
{
    char* start = (char*)memchr(net_buf, '{', net_pos);
    if(!start) return 0;
    int depth = 0;
    for(char* p = start; p < net_buf + net_pos; p++){
        if(*p == '{') depth++;
        else if(*p == '}'){
            depth--;
            if(depth == 0) return 1;
        }
    }
    return 0;
}

static char* pop_json_message(int sock)
{
    while(1){
        char* start = (char*)memchr(net_buf, '{', net_pos);
        if(!start){
            if(net_pos >= (int)sizeof(net_buf) - 1) net_pos = 0;
            int n = recv(sock, net_buf + net_pos, (int)sizeof(net_buf) - net_pos - 1, 0);
            if(n <= 0) return NULL;
            net_pos += n;
            continue;
        }
        int depth = 0;
        char* end = start;
        for(; end < net_buf + net_pos; end++){
            if(*end == '{') depth++;
            else if(*end == '}'){
                depth--;
                if(depth == 0) break;
            }
        }
        if(depth != 0 || end >= net_buf + net_pos){
            if(net_pos >= (int)sizeof(net_buf) - 1) net_pos = 0;
            int n = recv(sock, net_buf + net_pos, (int)sizeof(net_buf) - net_pos - 1, 0);
            if(n <= 0) return NULL;
            net_pos += n;
            continue;
        }
        int len = (int)(end - start) + 1;
        if(len >= (int)sizeof(json_msg)) len = (int)sizeof(json_msg) - 1;
        memcpy(json_msg, start, (size_t)len);
        json_msg[len] = 0;
        int tail = (int)(net_buf + net_pos - (end + 1));
        memmove(net_buf, end + 1, (size_t)tail);
        net_pos = tail;
        return json_msg;
    }
}

static void pool_dispatch_line(const char* line)
{
    if(strstr(line, "mining.set_difficulty")){
        double d = cp_json_num(line, "params");
        if(!d){
            const char* p = strstr(line, "\"params\":[");
            if(p){
                p = strchr(p, '[');
                if(p) d = atof(p + 1);
            }
        }
        if(d > 0.0){
            g_diff = d;
            printf("[pool] mining.set_difficulty %.0f%s\n", g_diff,
                   cp_job_mining_active() ? " (during mine)" : "");
            fflush(stdout);
        }
        return;
    }

    if(strstr(line, "result") || strstr(line, "error")){
        if(g_submit_inflight)
            printf("[pool] submit response: %s\n", line);
        else
            printf("[pool] jsonrpc: %s\n", line);
        fflush(stdout);
        g_submit_inflight = 0;
        return;
    }

    if(strstr(line, "mining.notify")){
        char job_id[128] = {0};
        char header_hex[320] = {0};
        char target_hex[80] = {0};
        if(!cp_pool_parse_notify(line, job_id, sizeof(job_id),
                                header_hex, sizeof(header_hex),
                                target_hex, sizeof(target_hex))){
            return;
        }

        char job_key[320];
        snprintf(job_key, sizeof(job_key), "%s:%.16s", job_id, header_hex);

        uint8_t header[INCOMPLETE_HEADER_BYTES];
        int hlen = cp_hex_to_bytes(header_hex, header, INCOMPLETE_HEADER_BYTES);
        if(hlen != INCOMPLETE_HEADER_BYTES) return;

        uint32_t tgt[8];
        memset(tgt, 0, sizeof(tgt));
        if(!target_hex[0] || !cp_be_target_hex_to_le_words(target_hex, tgt))
            cp_target_from_difficulty(g_diff, tgt);

        if(cp_job_mining_active()){
            if(cp_job_key_matches(job_key)) return;
            cp_job_request_cancel();
            queue_pending_job(job_id, job_key, header, target_hex, tgt);
            printf("[net] new job %s while mining %s - cancelling stale work\n",
                   job_id, cp_job_mining_key());
            fflush(stdout);
            return;
        }

        pool_inbox_push(line);
        return;
    }

    if(!cp_job_mining_active())
        pool_inbox_push(line);
    else
        printf("[pool] (during mine) %s\n", line);
    fflush(stdout);
}

static void pool_net_reader_thread(void)
{
    while(g_net_reader_run.load()){
        /* Drain buffered messages before waiting — pool often sends authorize
         * ack + mining.notify back-to-back in one TCP segment. */
        if(!net_buf_has_complete_json()){
            if(!net_wait_readable(tcp_sock, 100)) continue;
        }
        std::lock_guard<std::mutex> lk(g_net_mx);
        char* line = pop_json_message(tcp_sock);
        if(!line){
            g_net_conn_lost.store(1);
            g_inbox_cv.notify_all();
            printf("[net] connection lost (reader)\n"); fflush(stdout);
            return;
        }
        printf("[pool-raw] %s\n", line); fflush(stdout);
        pool_dispatch_line(line);
    }
}

int cp_pool_connect(const char* host, int port)
{
    tcp_sock = tcp_connect(host, port);
    return tcp_sock >= 0;
}

void cp_pool_disconnect(void)
{
    if(tcp_sock >= 0){
        CP_SOCK_CLOSE(tcp_sock);
        tcp_sock = -1;
    }
    net_pos = 0;
}

int cp_pool_socket(void)
{
    return tcp_sock;
}

int cp_pool_send_authorize(int msg_id, const char* wallet,
                           const char* worker, const char* agent)
{
    char msg[512];
    snprintf(msg, sizeof(msg),
        "{\"jsonrpc\":\"2.0\",\"id\":%d,\"method\":\"mining.authorize\","
        "\"params\":{\"wallet\":\"%s\",\"worker\":\"%s\",\"agent\":\"%s\"}}",
        msg_id, wallet, worker, agent);
    printf("[net] LuckyPool authorize (wallet/worker/agent)\n"); fflush(stdout);
    return cp_send_json(tcp_sock, msg);
}

int cp_pool_send_plain_proof_submit(int sock, int msg_id, const char* job_id,
                                    const char* plain_b64, double hs)
{
    size_t blen = plain_b64 ? strlen(plain_b64) : 0;
    size_t need = blen + 256;
    char* sub = (char*)malloc(need);
    if(!sub){
        fprintf(stderr, "[net] plain_proof submit OOM (%zu b64 bytes)\n", blen);
        return 0;
    }
    int nw = snprintf(sub, need,
        "{\"jsonrpc\":\"2.0\",\"id\":%d,\"method\":\"mining.submit\","
        "\"params\":{\"job_id\":\"%s\",\"plain_proof\":\"%s\",\"hs\":%.0f}}",
        msg_id, job_id, plain_b64, hs);
    if(nw < 0 || (size_t)nw >= need){
        fprintf(stderr, "[net] plain_proof submit JSON too large (b64=%zu need>=%zu)\n",
                blen, need);
        free(sub);
        return 0;
    }
    printf("[net] plain_proof submit job=%s b64_len=%zu json_len=%d hs=%.0f\n",
           job_id, blen, nw, hs);
    fflush(stdout);
    int ok = cp_send_json(sock, sub);
    free(sub);
    return ok;
}

void cp_pool_reader_start(void)
{
    if(g_net_reader_run.load()) return;
    g_net_conn_lost.store(0);
    g_net_reader_run.store(1);
    g_net_reader = std::thread(pool_net_reader_thread);
    printf("[net] pool reader started (always on)\n");
    fflush(stdout);
}

void cp_pool_reader_stop(void)
{
    if(!g_net_reader_run.load()) return;
    g_net_reader_run.store(0);
    g_inbox_cv.notify_all();
    if(g_net_reader.joinable()) g_net_reader.join();
}

void cp_pool_inbox_clear(void)
{
    std::lock_guard<std::mutex> lk(g_inbox_mx);
    g_pool_inbox.clear();
}

int cp_pool_wait_line(char* out, size_t out_cap, int timeout_ms)
{
    std::unique_lock<std::mutex> lk(g_inbox_mx);
    for(;;){
        if(!g_pool_inbox.empty()){
            strncpy(out, g_pool_inbox.front().c_str(), out_cap - 1);
            out[out_cap - 1] = 0;
            g_pool_inbox.pop_front();
            return 1;
        }
        if(g_net_conn_lost.load()) return -1;
        if(timeout_ms < 0){
            g_inbox_cv.wait(lk);
            continue;
        }
        if(g_inbox_cv.wait_for(lk, std::chrono::milliseconds(timeout_ms))
           == std::cv_status::timeout){
            return 0;
        }
    }
}

int cp_pool_conn_lost(void)
{
    return g_net_conn_lost.load();
}

void cp_pool_set_submit_inflight(int on)
{
    g_submit_inflight = on;
}

void cp_pool_log_share_submit_outcome(void)
{
    if(g_submit_inflight)
        printf("[plain] share submitted; pool ack pending (reader will log [pool] submit response)\n");
    else
        printf("[plain] share submitted; pool response already received\n");
    fflush(stdout);
}

int cp_pool_parse_notify(const char* json,
                         char* job_id, int job_len,
                         char* header_hex, int header_len,
                         char* target_hex, int target_len)
{
    job_id[0] = header_hex[0] = target_hex[0] = 0;

    if(strstr(json, "\"header\"")){
        cp_json_str(json, "job_id", job_id, job_len);
        cp_json_str(json, "header", header_hex, header_len);
        cp_json_str(json, "target", target_hex, target_len);
        return header_hex[0] != 0;
    }

    const char* p = strstr(json, "\"params\":[");
    if(!p) return 0;
    p = strchr(p, '[');
    if(!p) return 0;
    p++;
    while(*p && *p != ']'){
        while(*p==' ' || *p=='\t' || *p==',') p++;
        if(*p == '"'){
            p++;
            char tmp[320];
            int i = 0;
            while(*p && *p != '"' && i < (int)sizeof(tmp) - 1) tmp[i++] = *p++;
            tmp[i] = 0;
            if(i == 0){ if(*p=='"') p++; continue; }
            if(!job_id[0]){
                strncpy(job_id, tmp, job_len - 1);
                job_id[job_len - 1] = 0;
            } else if((int)strlen(tmp) == TARGET_HEX_LEN && !target_hex[0]){
                strncpy(target_hex, tmp, target_len - 1);
                target_hex[target_len - 1] = 0;
            } else if((int)strlen(tmp) == HEADER_HEX_LEN && !header_hex[0]){
                strncpy(header_hex, tmp, header_len - 1);
                header_hex[header_len - 1] = 0;
            }
            if(*p=='"') p++;
        } else if(*p=='{' ){
            break;
        } else {
            while(*p && *p != ',' && *p != ']') p++;
        }
    }
    return header_hex[0] != 0;
}

int cp_pool_take_pending_job(CpPendingJob* out)
{
    std::lock_guard<std::mutex> lk(g_pending_mx);
    if(!g_pending_valid) return 0;
    *out = g_pending_job;
    g_pending_valid = 0;
    return 1;
}

double cp_pool_difficulty(void)
{
    return g_diff;
}

void cp_pool_set_difficulty(double d)
{
    g_diff = d;
}
