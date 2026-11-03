#ifndef CP_TCP_H
#define CP_TCP_H

#include "cp_platform.h"

/* Resolve host with getaddrinfo (IPv4 and IPv6) and try each address with a
 * non-blocking connect; timeout_ms is the budget for all attempts together, and each
 * attempt gets a share of what is left, so a silent first address cannot use it all.
 * Returns a blocking socket, or CP_INVALID_SOCK. */
cp_sock_t cp_tcp_connect(const char* host, int port, int timeout_ms);
cp_sock_t cp_tcp_connect_addresses(const struct addrinfo* addresses, int timeout_ms);

#endif /* CP_TCP_H */
