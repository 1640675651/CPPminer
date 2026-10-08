#include "cp_tcp.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#ifndef _WIN32
#include <cerrno>
#include <fcntl.h>
#include <poll.h>
#endif

using Clock = std::chrono::steady_clock;

static bool set_nonblocking(cp_sock_t sock, bool on)
{
#ifdef _WIN32
    u_long mode = on ? 1 : 0;
    return ioctlsocket(sock, FIONBIO, &mode) == 0;
#else
    const int flags = fcntl(sock, F_GETFL, 0);
    return flags >= 0 && fcntl(sock, F_SETFL, on ? flags | O_NONBLOCK : flags & ~O_NONBLOCK) == 0;
#endif
}

static bool finish_connect(cp_sock_t sock, Clock::time_point deadline)
{
    for(;;){
        const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(
                             deadline - Clock::now()).count();
        if(left <= 0) return false;
#ifdef _WIN32
        fd_set writable, errors;
        FD_ZERO(&writable); FD_ZERO(&errors);
        FD_SET(sock, &writable); FD_SET(sock, &errors);
        timeval timeout = {(long)(left / 1000), (long)((left % 1000) * 1000)};
        const int ready = select(0, nullptr, &writable, &errors, &timeout);
        if(ready < 0 && WSAGetLastError() == WSAEINTR) continue;
#else
        pollfd fd = {sock, POLLOUT, 0};
        const int ready = poll(&fd, 1, (int)left);
        if(ready < 0 && errno == EINTR) continue;
#endif
        if(ready <= 0) return false;
        int error = 0;
#ifdef _WIN32
        int size = sizeof(error);
        return getsockopt(sock, SOL_SOCKET, SO_ERROR, (char*)&error, &size) == 0 && error == 0;
#else
        socklen_t size = sizeof(error);
        return getsockopt(sock, SOL_SOCKET, SO_ERROR, &error, &size) == 0 && error == 0;
#endif
    }
}

cp_sock_t cp_tcp_connect_addresses(const addrinfo* addresses, int timeout_ms)
{
    if(timeout_ms <= 0 || cp_net_init() != 0) return CP_INVALID_SOCK;
    int remaining = 0;
    for(auto address = addresses; address; address = address->ai_next) ++remaining;
    const auto deadline = Clock::now() + std::chrono::milliseconds(timeout_ms);
    for(auto address = addresses; address; address = address->ai_next, --remaining){
        const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(
                             deadline - Clock::now()).count();
        if(left <= 0) break;
        // Reserve part of the budget for later addresses even if the first one is silent.
        const auto attempt_deadline = (std::min)(deadline, Clock::now() +
            std::chrono::milliseconds(std::max<int64_t>(1, left / remaining)));
        cp_sock_t sock = socket(address->ai_family, address->ai_socktype, address->ai_protocol);
        if(sock == CP_INVALID_SOCK) continue;
#ifdef SO_NOSIGPIPE
        int one = 1;
        setsockopt(sock, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
#endif
        bool ok = false;
        if(set_nonblocking(sock, true)){
            const int result = connect(sock, address->ai_addr, (int)address->ai_addrlen);
            if(result == 0) ok = true;
            else {
#ifdef _WIN32
                const int error = WSAGetLastError();
                const bool pending = error == WSAEWOULDBLOCK || error == WSAEINPROGRESS;
#else
                const bool pending = errno == EINPROGRESS || errno == EWOULDBLOCK || errno == EINTR;
#endif
                if(pending) ok = finish_connect(sock, attempt_deadline);
            }
        }
        if(ok && set_nonblocking(sock, false)) return sock;
        CP_SOCK_CLOSE(sock);
    }
    return CP_INVALID_SOCK;
}

cp_sock_t cp_tcp_connect(const char* host, int port, int timeout_ms)
{
    if(!host || !*host || port < 1 || port > 65535 || timeout_ms <= 0 || cp_net_init() != 0)
        return CP_INVALID_SOCK;
    addrinfo hints = {}, *addresses = nullptr;
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;
    char service[6];
    snprintf(service, sizeof(service), "%d", port);
    const int result = getaddrinfo(host, service, &hints, &addresses);
    if(result != 0){
        fprintf(stderr, "[net] cannot resolve %s (resolver error %d)\n", host, result);
        return CP_INVALID_SOCK;
    }
    cp_sock_t sock = cp_tcp_connect_addresses(addresses, timeout_ms);
    freeaddrinfo(addresses);
    if(sock == CP_INVALID_SOCK)
        fprintf(stderr, "[net] TCP connect to %s:%d failed (budget %d ms)\n", host, port, timeout_ms);
    return sock;
}

int cp_pool_parse_uri(const char* uri, char* host, size_t cap, int* port)
{
    if(!uri || !host || !cap || !port) return 0;
    host[0] = 0;
    const char* start = strstr(uri, "://");
    if(!start || start == uri) return 0;
    start += 3;
    const char* end;
    if(*start == '['){
        ++start;
        end = strchr(start, ']');
        if(!end || end[1] != ':') return 0;
    } else {
        end = strchr(start, ':');
        if(!end) return 0;
    }
    const size_t size = (size_t)(end - start);
    if(!size || size >= cap) return 0;
    const char* number = end + (*end == ']' ? 2 : 1);
    if(!*number) return 0;
    int value = 0;
    for(const char* p = number; *p; ++p){
        if(*p < '0' || *p > '9') return 0;
        value = value * 10 + (*p - '0');
        if(value > 65535) return 0;
    }
    if(!value) return 0;
    memcpy(host, start, size);
    host[size] = 0;
    *port = value;
    return 1;
}
