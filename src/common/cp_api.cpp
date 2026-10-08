/*
 * Read-only HTTP stats API, see include/cp_api.h.
 *
 * One listener thread answers each request with a small JSON document and closes the connection.
 * Mining threads only touch a mutex-protected counter block, a few times per second at most.
 */
#include "cp_api.h"

#include "cp_json_text.hpp"
#include "cp_platform.h"
#include "cp_state.h"
#include "cp_util.h"

#include <cstdio>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifndef _WIN32
#include <sys/select.h>
#endif

#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0
#endif

namespace {

struct Device {
    std::string name;
    std::string pci; /* "0000:03:00.0" or empty */
    int bus = -1;    /* decimal PCI bus, -1 when unknown */
};

struct Sample {
    double t;
    double cum;
};

std::mutex g_mx;
bool g_enabled = false;
const double g_t_start = cp_now_sec();
std::string g_algo = "pearl";
std::string g_backend;
std::string g_pool;
bool g_pool_connected = false;
std::vector<Device> g_devices;
double g_cum = 0;            /* work since start */
std::deque<Sample> g_samples; /* at most one per second, last ~16 minutes */
double g_t_last = 0;          /* time of the last work report */
double g_dt_last = 0;         /* gap between the last two reports */
unsigned long long g_accepted = 0, g_rejected = 0;
double g_last_share = 0;

/* "cppminer/0.5-fork.8" at start-up (before --agent can change it) -> "0.5-fork.8" */
const std::string g_version = [] {
    const char* a = agent_global;
    const char* s = strchr(a, '/');
    return std::string(s ? s + 1 : a);
}();

const double k_windows[3] = {10.0, 60.0, 900.0};

int parse_pci_bus(const char* pci)
{
    /* "dddd:bb:dd.f" or "bb:dd.f" */
    if(!pci || !*pci) return -1;
    const char* c1 = strchr(pci, ':');
    if(!c1) return -1;
    const char* c2 = strchr(c1 + 1, ':');
    const char* bus = c2 ? c1 + 1 : pci;
    unsigned v = 0;
    if(sscanf(bus, "%x", &v) != 1) return -1;
    return (int)v;
}

/* Rate over about the last w seconds, caller holds g_mx. Work arrives in chunks (a Pearl attempt
 * can take tens of seconds on a slow GPU), so the window ends at the last report rather than at
 * "now"; once reports stop for much longer than their usual gap it ends at now and decays. */
double rate_locked(double now, double w)
{
    if(g_samples.empty()) return 0;
    double end = g_t_last > 0 ? g_t_last : now;
    if(now - end > 3.0 * g_dt_last + 10.0) end = now;
    const Sample* s = &g_samples.front();
    for(const Sample& x : g_samples){
        if(x.t <= end - w) s = &x;
        else break;
    }
    const double dt = end - s->t;
    if(dt < 0.5) return 0;
    return (g_cum - s->cum) / dt;
}

std::string fmt_num(double v)
{
    char b[64];
    snprintf(b, sizeof(b), "%.3f", v);
    return b;
}

std::string summary_json()
{
    std::lock_guard<std::mutex> lk(g_mx);
    const double now = cp_now_sec();
    const int ndev = g_devices.empty() ? 1 : (int)g_devices.size();
    double r[3];
    for(int i = 0; i < 3; i++) r[i] = rate_locked(now, k_windows[i]);

    std::string j = "{";
    j += "\"miner\":\"cppminer\",\"version\":\"" + cp_json_escape(g_version.c_str()) + "\",";
    j += "\"algo\":\"" + cp_json_escape(g_algo.c_str()) + "\",";
    j += "\"backend\":\"" + cp_json_escape(g_backend.c_str()) + "\",";
    j += "\"worker\":\"" + cp_json_escape(worker_global) + "\",";
    j += "\"pool\":\"" + cp_json_escape(g_pool.c_str()) + "\",";
    j += std::string("\"pool_connected\":") + (g_pool_connected ? "true" : "false") + ",";
    j += "\"uptime\":" + std::to_string((long long)(now - g_t_start)) + ",";
    j += "\"hashrate\":{\"unit\":\"H/s\",\"windows\":[10,60,900],\"total\":[" + fmt_num(r[0]) +
         "," + fmt_num(r[1]) + "," + fmt_num(r[2]) + "]},";
    j += "\"devices\":[";
    for(int d = 0; d < (int)g_devices.size(); d++){
        const Device& dev = g_devices[d];
        if(d) j += ",";
        j += "{\"id\":" + std::to_string(d) + ",\"name\":\"" + cp_json_escape(dev.name.c_str()) +
             "\",\"pci\":\"" + cp_json_escape(dev.pci.c_str()) + "\",\"bus\":" +
             std::to_string(dev.bus) + ",\"hashrate\":[" + fmt_num(r[0] / ndev) + "," +
             fmt_num(r[1] / ndev) + "," + fmt_num(r[2] / ndev) + "]}";
    }
    j += "],";
    j += "\"shares\":{\"accepted\":" + std::to_string(g_accepted) +
         ",\"rejected\":" + std::to_string(g_rejected) + ",\"last_share_ago\":" +
         (g_last_share > 0 ? std::to_string((long long)(now - g_last_share)) : std::string("-1")) +
         "}";
    j += "}";
    return j;
}

/* HiveOS h-stats.sh form: khs = total kH/s, stats.hs per device in kH/s (60 s window). */
std::string hiveos_json()
{
    std::lock_guard<std::mutex> lk(g_mx);
    const double now = cp_now_sec();
    const int ndev = g_devices.empty() ? 1 : (int)g_devices.size();
    const double khs = rate_locked(now, 60.0) / 1000.0;
    std::string hs, bus;
    bool all_bus = !g_devices.empty();
    for(int d = 0; d < ndev; d++){
        if(d){
            hs += ",";
            bus += ",";
        }
        hs += fmt_num(khs / ndev);
        const int b = d < (int)g_devices.size() ? g_devices[d].bus : -1;
        if(b < 0) all_bus = false;
        bus += std::to_string(b);
    }
    std::string j = "{\"khs\":" + fmt_num(khs) + ",\"stats\":{";
    j += "\"hs\":[" + hs + "],\"hs_units\":\"khs\",\"temp\":[],\"fan\":[],";
    j += "\"uptime\":" + std::to_string((long long)(now - g_t_start)) + ",";
    j += "\"ver\":\"" + cp_json_escape(g_version.c_str()) + "\",";
    j += "\"ar\":[" + std::to_string(g_accepted) + "," + std::to_string(g_rejected) + "],";
    j += "\"algo\":\"" + cp_json_escape(g_algo.c_str()) + "\"";
    if(all_bus) j += ",\"bus_numbers\":[" + bus + "]";
    j += "}}";
    return j;
}

void send_all(cp_sock_t s, const std::string& data)
{
    size_t off = 0;
    while(off < data.size()){
        const int n = (int)send(s, data.data() + off, (int)(data.size() - off), MSG_NOSIGNAL);
        if(n <= 0) return;
        off += (size_t)n;
    }
}

void handle_client(cp_sock_t c)
{
    char req[2048];
    int len = 0;
    /* read the request line (and headers) for up to 2 s */
    while(len < (int)sizeof(req) - 1){
        fd_set rf;
        FD_ZERO(&rf);
        FD_SET(c, &rf);
        timeval tv{2, 0};
        if(select((int)c + 1, &rf, nullptr, nullptr, &tv) <= 0) break;
        const int n = (int)recv(c, req + len, (int)(sizeof(req) - 1 - len), 0);
        if(n <= 0) break;
        len += n;
        req[len] = 0;
        if(strstr(req, "\r\n\r\n") || strstr(req, "\n\n")) break;
    }
    req[len] = 0;

    char path[256] = "/";
    {
        const char* sp = strchr(req, ' ');
        if(sp){
            const char* e = sp + 1;
            size_t k = 0;
            while(*e && *e != ' ' && *e != '?' && *e != '\r' && *e != '\n' && k < sizeof(path) - 1)
                path[k++] = *e++;
            path[k] = 0;
        }
    }

    int status = 200;
    std::string body;
    if(!strcmp(path, "/") || !strcmp(path, "/summary") || !strcmp(path, "/api") ||
       !strcmp(path, "/1/summary")){
        body = summary_json();
    } else if(!strcmp(path, "/hiveos")){
        body = hiveos_json();
    } else {
        status = 404;
        body = "{\"error\":\"not found\",\"paths\":[\"/summary\",\"/hiveos\"]}";
    }
    std::string head = std::string("HTTP/1.1 ") + (status == 200 ? "200 OK" : "404 Not Found") +
                       "\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\n"
                       "Cache-Control: no-store\r\nConnection: close\r\nContent-Length: " +
                       std::to_string(body.size()) + "\r\n\r\n";
    send_all(c, head + body);
    CP_SOCK_CLOSE(c);
}

void serve(cp_sock_t ls)
{
    for(;;){
        sockaddr_in peer;
        socklen_t plen = sizeof(peer);
        const cp_sock_t c = accept(ls, (sockaddr*)&peer, &plen);
        if(c == CP_INVALID_SOCK){
            cp_sleep(1);
            continue;
        }
        handle_client(c);
    }
}

} // namespace

extern "C" {

int cp_api_start(const char* bind_addr, int port)
{
    if(port <= 0 || port > 65535) return -1;
    cp_net_init();
    const char* addr = (bind_addr && *bind_addr) ? bind_addr : "127.0.0.1";
    const cp_sock_t ls = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if(ls == CP_INVALID_SOCK){
        fprintf(stderr, "[api] socket() failed; running without the API\n");
        return -1;
    }
    int one = 1;
    setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, (const char*)&one, sizeof(one));
    sockaddr_in sa;
    memset(&sa, 0, sizeof(sa));
    sa.sin_family = AF_INET;
    sa.sin_port = htons((unsigned short)port);
    if(inet_pton(AF_INET, addr, &sa.sin_addr) != 1){
        fprintf(stderr, "[api] bad --api-bind address %s; running without the API\n", addr);
        CP_SOCK_CLOSE(ls);
        return -1;
    }
    if(bind(ls, (sockaddr*)&sa, sizeof(sa)) != 0 || listen(ls, 16) != 0){
        fprintf(stderr, "[api] cannot listen on %s:%d (port in use?); running without the API\n",
                addr, port);
        CP_SOCK_CLOSE(ls);
        return -1;
    }
    {
        std::lock_guard<std::mutex> lk(g_mx);
        g_enabled = true;
    }
    std::thread(serve, ls).detach();
    printf("[api] http://%s:%d/summary (also /hiveos)\n", addr, port);
    fflush(stdout);
    return 0;
}

int cp_api_enabled(void)
{
    std::lock_guard<std::mutex> lk(g_mx);
    return g_enabled ? 1 : 0;
}

void cp_api_set_algo(const char* algo, const char* backend)
{
    std::lock_guard<std::mutex> lk(g_mx);
    if(algo) g_algo = algo;
    if(backend){
        g_backend = backend;
        /* the backend is set right before mining starts: the rate counts from here */
        if(g_samples.empty()) g_samples.push_back({cp_now_sec(), g_cum});
    }
}

void cp_api_set_pool(const char* host, int port, int connected)
{
    std::lock_guard<std::mutex> lk(g_mx);
    g_pool = std::string(host ? host : "") + ":" + std::to_string(port);
    g_pool_connected = connected != 0;
}

void cp_api_set_pool_connected(int connected)
{
    std::lock_guard<std::mutex> lk(g_mx);
    g_pool_connected = connected != 0;
}

void cp_api_add_device(const char* name, const char* pci)
{
    std::lock_guard<std::mutex> lk(g_mx);
    Device d;
    d.name = name ? name : "";
    d.pci = pci ? pci : "";
    d.bus = parse_pci_bus(pci);
    g_devices.push_back(d);
}

int cp_api_device_count(void)
{
    std::lock_guard<std::mutex> lk(g_mx);
    return (int)g_devices.size();
}

void cp_api_add_work(double units)
{
    if(units <= 0) return;
    const double now = cp_now_sec();
    std::lock_guard<std::mutex> lk(g_mx);
    if(g_samples.empty()) g_samples.push_back({now, g_cum}); /* no start mark: count from here */
    g_cum += units;
    g_dt_last = now - (g_t_last > 0 ? g_t_last : g_samples.front().t);
    g_t_last = now;
    if(now - g_samples.back().t >= 1.0) g_samples.push_back({now, g_cum});
    /* keep one sample at or before the longest window */
    while(g_samples.size() > 2 && g_samples[1].t <= now - (k_windows[2] + 30.0))
        g_samples.pop_front();
}

void cp_api_on_share(int accepted)
{
    std::lock_guard<std::mutex> lk(g_mx);
    if(accepted) g_accepted++;
    else g_rejected++;
    g_last_share = cp_now_sec();
}

} // extern "C"
