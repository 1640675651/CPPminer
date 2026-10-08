#include "cp_tcp.h"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#ifndef _WIN32
#include <fcntl.h>
#endif

static cp_sock_t listener(sockaddr_in& address, int backlog = 2)
{
    cp_sock_t sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    assert(sock != CP_INVALID_SOCK);
    address = {};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    assert(bind(sock, (sockaddr*)&address, sizeof(address)) == 0);
#ifdef _WIN32
    int size = sizeof(address);
#else
    socklen_t size = sizeof(address);
#endif
    assert(getsockname(sock, (sockaddr*)&address, &size) == 0);
    assert(listen(sock, backlog) == 0);
    return sock;
}

static addrinfo endpoint(sockaddr_in& address, addrinfo* next = nullptr)
{
    addrinfo info = {};
    info.ai_family = AF_INET;
    info.ai_socktype = SOCK_STREAM;
    info.ai_protocol = IPPROTO_TCP;
    info.ai_addr = (sockaddr*)&address;
    info.ai_addrlen = sizeof(address);
    info.ai_next = next;
    return info;
}

static void transfer(cp_sock_t server, cp_sock_t client)
{
    assert(client != CP_INVALID_SOCK);
    std::thread peer([server] {
        cp_sock_t accepted = accept(server, nullptr, nullptr);
        assert(accepted != CP_INVALID_SOCK);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        assert(send(accepted, "x", 1, 0) == 1);
        CP_SOCK_CLOSE(accepted);
    });
    char data = 0;
    assert(recv(client, &data, 1, 0) == 1 && data == 'x'); // Returned socket is blocking.
    CP_SOCK_CLOSE(client);
    peer.join();
}

static void test_address_fallback()
{
    sockaddr_in good, refused;
    cp_sock_t server = listener(good);
    cp_sock_t reserved = listener(refused);
    CP_SOCK_CLOSE(reserved);
    // Keep the refused endpoint bound, without listen, so its port cannot be reused.
    reserved = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    assert(bind(reserved, (sockaddr*)&refused, sizeof(refused)) == 0);
    addrinfo second = endpoint(good), first = endpoint(refused, &second);
    transfer(server, cp_tcp_connect_addresses(&first, 1000));
    transfer(server, cp_tcp_connect("localhost", ntohs(good.sin_port), 1000));
    assert(cp_tcp_connect_addresses(&first, 0) == CP_INVALID_SOCK);
    assert(cp_tcp_connect_addresses(nullptr, 1000) == CP_INVALID_SOCK);
    first.ai_next = nullptr;
    assert(cp_tcp_connect_addresses(&first, 1000) == CP_INVALID_SOCK);
    CP_SOCK_CLOSE(reserved);
    CP_SOCK_CLOSE(server);
}

static void test_ipv6()
{
    cp_sock_t server = socket(AF_INET6, SOCK_STREAM, IPPROTO_TCP);
    if(server == CP_INVALID_SOCK){ puts("SKIP IPv6: unavailable"); return; }
    sockaddr_in6 address = {};
    address.sin6_family = AF_INET6;
    address.sin6_addr = in6addr_loopback;
    if(bind(server, (sockaddr*)&address, sizeof(address)) != 0){
        CP_SOCK_CLOSE(server);
        puts("SKIP IPv6: loopback unavailable");
        return;
    }
#ifdef _WIN32
    int size = sizeof(address);
#else
    socklen_t size = sizeof(address);
#endif
    assert(getsockname(server, (sockaddr*)&address, &size) == 0);
    assert(listen(server, 2) == 0);
    transfer(server, cp_tcp_connect("::1", ntohs(address.sin6_port), 1000));
    CP_SOCK_CLOSE(server);
}

#ifdef __linux__
static void test_silent_address_timeout()
{
    // A full loopback accept backlog makes Linux silently drop new SYNs.
    // This produces a real pending connect without external network or firewall changes.
    sockaddr_in blocked, good;
    cp_sock_t saturated = listener(blocked, 0), server = listener(good);
    addrinfo second = endpoint(good), first = endpoint(blocked);
    cp_sock_t filling = cp_tcp_connect_addresses(&first, 1000);
    assert(filling != CP_INVALID_SOCK);
    const auto start = std::chrono::steady_clock::now();
    assert(cp_tcp_connect_addresses(&first, 100) == CP_INVALID_SOCK);
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    assert(elapsed >= 0.06 && elapsed < 1.0);
    first.ai_next = &second;
    transfer(server, cp_tcp_connect_addresses(&first, 200));
    CP_SOCK_CLOSE(filling);
    CP_SOCK_CLOSE(saturated);
    CP_SOCK_CLOSE(server);
}
#endif

int main()
{
    assert(cp_net_init() == 0);
    char host[256]; int port = 0;
    assert(cp_pool_parse_uri("stratum+tcp://pool.example:1200", host, sizeof(host), &port));
    assert(!strcmp(host, "pool.example") && port == 1200);
    assert(cp_pool_parse_uri("stratum+tcp://[::1]:1200", host, sizeof(host), &port));
    assert(!strcmp(host, "::1"));
    for(const char* uri : {"stratum+tcp://:1", "stratum+tcp://a:0", "stratum+tcp://a:65536",
                          "stratum+tcp://a:999999999999", "stratum+tcp://a:1x",
                          "stratum+tcp://[::1:1200", "stratum+tcp://a:"})
        assert(!cp_pool_parse_uri(uri, host, sizeof(host), &port));
    const std::string long_uri = "stratum+tcp://" + std::string(256, 'a') + ":1200";
    assert(!cp_pool_parse_uri(long_uri.c_str(), host, sizeof(host), &port));
    test_address_fallback();
    test_ipv6();
#ifdef __linux__
    test_silent_address_timeout();
#endif
}
