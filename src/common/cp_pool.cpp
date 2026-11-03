#include "cp_pool.h"
#include "cp_config.h"
#include "cp_job_ctrl.h"
#include "cp_json_frame.h"
#include "cp_json_text.hpp"
#include "cp_platform.h"
#include "cp_qpow_pool.h"
#include "cp_state.h"
#include "cp_submit_table.hpp"
#include "cp_tcp.h"
#include "cp_util.h"
#if defined(CP_ENABLE_CPU) && CP_ENABLE_CPU
#include "cp_cpu_affinity.h"
#endif

#ifdef _WIN32
#include <mstcpip.h> /* SIO_KEEPALIVE_VALS, struct tcp_keepalive */
#else
#include <netinet/tcp.h> /* TCP_KEEPIDLE/TCP_KEEPINTVL/TCP_KEEPCNT, TCP_NODELAY */
#endif

#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <thread>

static int tcp_sock = -1;
static std::atomic<double> g_diff{32.0};
/* Every submit waits here until the pool replies to its id; the reader gives up on the
 * connection when the oldest one has no reply within CP_SUBMIT_ACK_TIMEOUT_SEC. */
static CpSubmitTable g_submits;
#define CP_SUBMIT_ACK_TIMEOUT_SEC 60.0

/* Session deadlines. A pool that keeps the connection open but never answers the
 * authorize, or never sends a job, would otherwise leave the miner idle for good
 * (keepalive only notices a dead peer). The reader drops the connection when the
 * authorize reply is missing for CP_AUTHORIZE_TIMEOUT_SEC, or no job follows the
 * login for CP_FIRST_JOB_TIMEOUT_SEC. */
#define CP_AUTHORIZE_TIMEOUT_SEC 30.0
#define CP_FIRST_JOB_TIMEOUT_SEC 30.0
enum { CP_SESSION_IDLE, CP_SESSION_AUTHORIZE, CP_SESSION_FIRST_JOB, CP_SESSION_READY };
static std::atomic<int> g_session_stage{CP_SESSION_IDLE};
static std::atomic<double> g_session_deadline{0.0};

/* Pools and middleboxes drop idle stratum connections without a RST. Without
 * probes the miner keeps scanning a stale job and queues every submit into a
 * dead socket (seen on Kryptex: 127 KB unacknowledged, 9 retransmits, no
 * notify for 10 minutes, nothing in the log). Probe after 30 s idle, every
 * 10 s, give up after 3, so a dead peer is noticed within about a minute and
 * NAT/load-balancer entries stay fresh. TCP_NODELAY: every message is one
 * small JSON line that should go out at once. */
static void tcp_tune(cp_sock_t s)
{
    int one = 1;
#ifdef _WIN32
    setsockopt(s, SOL_SOCKET, SO_KEEPALIVE, (const char*)&one, sizeof(one));
    struct tcp_keepalive ka;
    ka.onoff = 1;
    ka.keepalivetime = 30000;
    ka.keepaliveinterval = 10000;
    DWORD ret = 0;
    WSAIoctl(s, SIO_KEEPALIVE_VALS, &ka, sizeof(ka), NULL, 0, &ret, NULL, NULL);
    setsockopt(s, IPPROTO_TCP, TCP_NODELAY, (const char*)&one, sizeof(one));
#else
    setsockopt(s, SOL_SOCKET, SO_KEEPALIVE, &one, sizeof(one));
    int idle = 30, intvl = 10, cnt = 3;
#if defined(TCP_KEEPIDLE)
    setsockopt(s, IPPROTO_TCP, TCP_KEEPIDLE, &idle, sizeof(idle));
#elif defined(TCP_KEEPALIVE) /* macOS */
    setsockopt(s, IPPROTO_TCP, TCP_KEEPALIVE, &idle, sizeof(idle));
#endif
#if defined(TCP_KEEPINTVL)
    setsockopt(s, IPPROTO_TCP, TCP_KEEPINTVL, &intvl, sizeof(intvl));
#endif
#if defined(TCP_KEEPCNT)
    setsockopt(s, IPPROTO_TCP, TCP_KEEPCNT, &cnt, sizeof(cnt));
#endif
    setsockopt(s, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
#endif
}
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
/* id of the mining.authorize awaiting its response (-1 = none). */
static std::atomic<int> g_auth_msg_id{-1};
/* Written by the reader thread, read by the share-queue thread. */
static std::atomic<int> g_pool_proof_gzip{0};
static char g_pool_host[256] = {0};

/* Budget for resolving and connecting to every address of the pool host. */
#define CP_CONNECT_TIMEOUT_MS 10000

static int tcp_connect(const char* host, int port)
{
    cp_sock_t s = cp_tcp_connect(host, port, CP_CONNECT_TIMEOUT_MS);
    if(s == CP_INVALID_SOCK) return (int)CP_INVALID_SOCK;
    tcp_tune(s);
    return (int)s;
}

static void session_set(int stage, double timeout_sec)
{
    g_session_deadline.store(cp_now_sec() + timeout_sec);
    g_session_stage.store(stage);
}

/* Reason the session is over, or NULL. */
static const char* session_expired(void)
{
    const double now = cp_now_sec();
    const int stage = g_session_stage.load();
    if(stage == CP_SESSION_AUTHORIZE && now > g_session_deadline.load())
        return "no authorize response for 30 s";
    if(stage == CP_SESSION_FIRST_JOB && now > g_session_deadline.load())
        return "no job from the pool for 30 s after login";
    /* A submit that gets no reply is the first visible symptom of a dead connection
     * (keepalive catches the rest); reconnect rather than mine on. Timed from the oldest
     * unanswered submit, so newer submits cannot push the deadline back. */
    double sent_at = 0.0;
    if(g_submits.oldest(&sent_at) && now - sent_at > CP_SUBMIT_ACK_TIMEOUT_SEC)
        return "no pool reply to a submit for 60 s";
    return NULL;
}

static void pool_connection_lost(const char* reason)
{
    g_submits.clear();
    g_net_conn_lost.store(1);
    g_inbox_cv.notify_all();
    printf("[net] %s; treating the connection as lost\n", reason);
    fflush(stdout);
}

void cp_pool_note_job(void)
{
    g_session_stage.store(CP_SESSION_READY);
}

void cp_pool_expect_job(void)
{
    if(g_session_stage.load() != CP_SESSION_READY)
        session_set(CP_SESSION_FIRST_JOB, CP_FIRST_JOB_TIMEOUT_SEC);
}

double cp_pool_parse_difficulty(const char* line)
{
    /* "params":[D] (stratum) or "params":D, with any whitespace around the number. */
    const char* p = cp_json_member(line, "params");
    if(!p) return 0.0;
    if(*p == '['){
        ++p;
        while(*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') ++p;
    }
    char* end = NULL;
    const double d = strtod(p, &end);
    return end != p && d > 0.0 && d < 1e300 ? d : 0.0;
}

static void queue_pending_job(
    const char* job_id, const char* job_key,
    const uint8_t* header, const char* target_hex, const uint32_t tgt[8],
    uint32_t cert_version)
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
    g_pending_job.cert_version = cp_resolve_cert_version(cert_version);
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

static int net_buf_message_state(void)
{
    char* start = (char*)memchr(net_buf, '{', net_pos);
    if(!start) return 0;
    size_t len = 0;
    return cp_json_object_length(start, (size_t)(net_buf + net_pos - start), &len);
}

/* Extract only buffered data; the caller handles readable waits and recv. */
static char* pop_json_message(void)
{
    char* start = (char*)memchr(net_buf, '{', net_pos);
    if(!start) return NULL;
    size_t frame_len = 0;
    if(cp_json_object_length(start, (size_t)(net_buf + net_pos - start), &frame_len) != 1)
        return NULL;
    const int len = (int)frame_len;
    memcpy(json_msg, start, (size_t)len);
    json_msg[len] = 0;
    const int tail = (int)(net_buf + net_pos - (start + frame_len));
    memmove(net_buf, start + frame_len, (size_t)tail);
    net_pos = tail;
    return json_msg;
}

static void pool_dispatch_line(const char* line)
{
    if(cp_qpow_pool_on_line(line))
        return;

    if(strstr(line, "mining.set_difficulty")){
        const double d = cp_pool_parse_difficulty(line);
        if(d > 0.0){
            g_diff.store(d);
            printf("[pool] mining.set_difficulty %.0f%s\n", d,
                   cp_job_mining_active() ? " (during mine)" : "");
            fflush(stdout);
        }
        return;
    }

    if(cp_pool_on_authorize_response(line))
        return;

    if(strstr(line, "result") || strstr(line, "error")){
        /* Only the reply carrying a submit's own id answers that submit. */
        int response_id = 0;
        if(cp_json_rpc_response_id(line, &response_id) && g_submits.finish(response_id))
            printf("[pool] submit response: %s\n", line);
        else
            printf("[pool] jsonrpc: %s\n", line);
        fflush(stdout);
        return;
    }

    if(strstr(line, "mining.notify")){
        char job_id[128] = {0};
        char header_hex[320] = {0};
        char target_hex[80] = {0};
        uint32_t cert_version = 0;
        if(!cp_pool_parse_notify(line, job_id, sizeof(job_id),
                                header_hex, sizeof(header_hex),
                                target_hex, sizeof(target_hex),
                                &cert_version)){
            return;
        }
        cert_version = cp_resolve_cert_version(cert_version);

        char job_key[320];
        snprintf(job_key, sizeof(job_key), "%s:%.16s", job_id, header_hex);

        uint8_t header[INCOMPLETE_HEADER_BYTES];
        int hlen = cp_hex_to_bytes(header_hex, header, INCOMPLETE_HEADER_BYTES);
        if(hlen != INCOMPLETE_HEADER_BYTES) return;
        cp_pool_note_job();

        uint32_t tgt[8];
        memset(tgt, 0, sizeof(tgt));
        if(!target_hex[0] || !cp_be_target_hex_to_le_words(target_hex, tgt))
            cp_pool_target_from_difficulty(g_diff.load(), tgt);

        if(cp_job_mining_active()){
            if(cp_job_key_matches(job_key)) return;
            cp_job_request_cancel();
            queue_pending_job(job_id, job_key, header, target_hex, tgt, cert_version);
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
#if defined(CP_ENABLE_CPU) && CP_ENABLE_CPU
    cp_cpu_affinity_release_thread();
#endif
    while(g_net_reader_run.load()){
        /* Checked on every pass, so steady pool traffic cannot postpone a deadline. */
        if(const char* reason = session_expired()){
            pool_connection_lost(reason);
            return;
        }
        /* Drain buffered messages before waiting — pool often sends authorize
         * ack + mining.notify back-to-back in one TCP segment. */
        int state;
        {
            std::lock_guard<std::mutex> lk(g_net_mx);
            state = net_buf_message_state();
            if(state == 1){
                char* line = pop_json_message();
                printf("[pool-raw] %s\n", line); fflush(stdout);
                pool_dispatch_line(line);
                continue;
            }
        }
        if(state < 0 || net_pos >= (int)sizeof(net_buf) - 1){
            g_net_conn_lost.store(1);
            g_inbox_cv.notify_all();
            printf("[net] invalid or oversized pool message\n"); fflush(stdout);
            return;
        }
        if(!net_wait_readable(tcp_sock, 100) || !g_net_reader_run.load())
            continue;
        std::lock_guard<std::mutex> lk(g_net_mx);
        int n = recv(tcp_sock, net_buf + net_pos,
                     (int)sizeof(net_buf) - net_pos - 1, 0);
        if(n <= 0){
            g_net_conn_lost.store(1);
            g_inbox_cv.notify_all();
            printf("[net] connection lost (reader)\n"); fflush(stdout);
            return;
        }
        net_pos += n;
    }
}

int cp_pool_connect(const char* host, int port)
{
    strncpy(g_pool_host, host ? host : "", sizeof(g_pool_host) - 1);
    g_pool_host[sizeof(g_pool_host) - 1] = 0;
    g_session_stage.store(CP_SESSION_IDLE);
    g_submits.clear();
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

/* Kryptex identifies the worker as "WALLET.worker" in the wallet string
 * (gist maxmalysh/eaaf4332…); LuckyPool uses the separate "worker" field.
 * Both are sent: on a kryptex host the wallet gets ".worker" appended unless
 * it already carries a dot. */
static int pool_host_contains_ci(const char* needle)
{
    std::string h = g_pool_host, n = needle;
    for(char& c : h) c = (char)tolower((unsigned char)c);
    for(char& c : n) c = (char)tolower((unsigned char)c);
    return h.find(n) != std::string::npos;
}

static std::string authorize_wallet_string(const char* wallet, const char* worker)
{
    std::string w = wallet ? wallet : "";
    if(worker && *worker && w.find('.') == std::string::npos &&
       pool_host_contains_ci("kryptex"))
        w += std::string(".") + worker;
    return w;
}

int cp_pool_send_authorize(int msg_id, const char* wallet,
                           const char* worker, const char* agent,
                           const char* password)
{
    /* Strictly response-driven: assume plain proofs until this authorize's
     * response says "type":"v2" (re-evaluated on every authorize, including
     * the dev-fee wallet switch). */
    g_pool_proof_gzip.store(0);
    g_auth_msg_id.store(msg_id);
    session_set(CP_SESSION_AUTHORIZE, CP_AUTHORIZE_TIMEOUT_SEC);
    const std::string msg =
        "{\"jsonrpc\":\"2.0\",\"id\":" + std::to_string(msg_id) +
        ",\"method\":\"mining.authorize\",\"params\":{\"wallet\":\"" +
        cp_json_escape(authorize_wallet_string(wallet, worker).c_str()) +
        "\",\"worker\":\"" + cp_json_escape(worker) +
        "\",\"agent\":\"" + cp_json_escape(agent) +
        "\",\"password\":\"" + cp_json_escape(password ? password : "x") +
        "\",\"type\":\"v2\"}}";
    /* Never log the request itself: it carries the password and, during the dev-fee
     * window, the fee wallet. */
    printf("[net] authorize (wallet/worker/agent/password, offering type v2 gzip proofs)\n");
    fflush(stdout);
    return cp_send_json(tcp_sock, msg.c_str());
}

int cp_pool_on_authorize_response(const char* line)
{
    const int want = g_auth_msg_id.load();
    int id = 0;
    if(want < 0 || !line || !cp_json_rpc_response_id(line, &id) || id != want) return 0;
    g_auth_msg_id.store(-1);
    if(cp_json_rpc_rejected(line)){
        printf("[pool] authorize response: %s\n", line);
        pool_connection_lost("pool rejected the authorize");
        return 1;
    }
    if(g_session_stage.load() == CP_SESSION_AUTHORIZE)
        session_set(CP_SESSION_FIRST_JOB, CP_FIRST_JOB_TIMEOUT_SEC);

    char type[16];
    const int v2 = cp_json_str(line, "type", type, (int)sizeof(type)) &&
                   strcmp(type, "v2") == 0;
    g_pool_proof_gzip.store(v2);
    printf("[pool] authorize response: %s\n", line);
    printf("[pool] proof encoding: %s\n",
           v2 ? "gzip (pool answered type v2)" : "plain base64 (no type v2 in response)");
    fflush(stdout);
    return 1;
}

int cp_pool_proof_gzip(void)
{
    return g_pool_proof_gzip.load();
}

int cp_pool_send_plain_proof_submit(int sock, int msg_id, const char* job_id,
                                    const char* plain_b64, double hs)
{
    size_t blen = plain_b64 ? strlen(plain_b64) : 0;
    const std::string escaped_job_id = cp_json_escape(job_id);
    size_t need = blen + escaped_job_id.size() + 256;
    char* sub = (char*)malloc(need);
    if(!sub){
        fprintf(stderr, "[net] plain_proof submit OOM (%zu b64 bytes)\n", blen);
        return 0;
    }
    int nw = snprintf(sub, need,
        "{\"jsonrpc\":\"2.0\",\"id\":%d,\"method\":\"mining.submit\","
        "\"params\":{\"job_id\":\"%s\",\"plain_proof\":\"%s\",\"hs\":%.0f}}",
        msg_id, escaped_job_id.c_str(), plain_b64 ? plain_b64 : "", hs);
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

int cp_pool_recv_one(char* out, size_t out_cap, int timeout_ms)
{
    if(!out || out_cap == 0 || tcp_sock < 0) return -1;
    auto deadline = std::chrono::steady_clock::now() +
                    std::chrono::milliseconds(timeout_ms < 0 ? 60000 : timeout_ms);
    std::lock_guard<std::mutex> lk(g_net_mx);
    while(1){
        int state = net_buf_message_state();
        if(state < 0) return -1;
        if(state == 1){
            char* line = pop_json_message();
            if(!line) return -1;
            const size_t line_len = strlen(line);
            if(line_len >= out_cap) return -1;
            memcpy(out, line, line_len + 1);
            return 1;
        }
        int remain_ms = (int)std::chrono::duration_cast<std::chrono::milliseconds>(
                            deadline - std::chrono::steady_clock::now())
                            .count();
        if(remain_ms <= 0) return 0;
        if(remain_ms > 200) remain_ms = 200;
        if(net_pos >= (int)sizeof(net_buf) - 1) return -1;
        if(!net_wait_readable(tcp_sock, remain_ms)) continue;
        int n = recv(tcp_sock, net_buf + net_pos,
                     (int)sizeof(net_buf) - net_pos - 1, 0);
        if(n <= 0) return -1;
        net_pos += n;
    }
}

int cp_pool_wait_line(char* out, size_t out_cap, int timeout_ms)
{
    if(!out || out_cap == 0) return -1;
    std::unique_lock<std::mutex> lk(g_inbox_mx);
    for(;;){
        if(!g_pool_inbox.empty()){
            const std::string& line = g_pool_inbox.front();
            if(line.size() >= out_cap){
                g_pool_inbox.pop_front();
                return -1;
            }
            memcpy(out, line.c_str(), line.size() + 1);
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

void cp_pool_track_submit(int msg_id)
{
    g_submits.track(msg_id, cp_now_sec());
}

void cp_pool_untrack_submit(int msg_id)
{
    g_submits.finish(msg_id);
}

void cp_pool_log_share_submit_outcome(int msg_id)
{
    if(g_submits.pending(msg_id))
        printf("[plain] share submitted; pool ack pending (reader will log [pool] submit response)\n");
    else
        printf("[plain] share submitted; pool response already received\n");
    fflush(stdout);
}

int cp_pool_parse_notify(const char* json,
                         char* job_id, int job_len,
                         char* header_hex, int header_len,
                         char* target_hex, int target_len,
                         uint32_t* cert_version_out)
{
    job_id[0] = header_hex[0] = target_hex[0] = 0;
    uint32_t cert_version = 0;
    double cv = cp_json_num(json, "cert_version");
    if(cv >= 1.0 && cv <= 3.0)
        cert_version = (uint32_t)cv;
    if(cert_version_out)
        *cert_version_out = cert_version;

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
    return g_diff.load();
}

void cp_pool_set_difficulty(double d)
{
    g_diff.store(d);
}
