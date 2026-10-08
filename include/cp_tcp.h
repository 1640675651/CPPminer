#ifndef CP_TCP_H
#define CP_TCP_H

#include "cp_platform.h"
#include <stddef.h>

/* DNS resolution uses the system resolver. The budget covers all TCP attempts. */
cp_sock_t cp_tcp_connect(const char* host, int port, int timeout_ms);
cp_sock_t cp_tcp_connect_addresses(const struct addrinfo* addresses, int timeout_ms);
/* URI host may be a DNS name, IPv4 address, or bracketed IPv6 address. */
int cp_pool_parse_uri(const char* uri, char* host, size_t cap, int* port);

#endif
