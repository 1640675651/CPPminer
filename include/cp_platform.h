/*
 * Cross-platform shims for CPminer (Windows MSVC + POSIX).
 */
#ifndef CP_PLATFORM_H
#define CP_PLATFORM_H

#ifdef _WIN32

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>

#pragma comment(lib, "ws2_32.lib")

typedef int ssize_t;

#define CP_SOCK_CLOSE(s) closesocket((SOCKET)(s))
#define CP_INVALID_SOCK INVALID_SOCKET
typedef SOCKET cp_sock_t;

static int cp_net_init(void) {
    static int started = 0;
    if (started) return 0;
    WSADATA wsa;
    int rc = WSAStartup(MAKEWORD(2, 2), &wsa);
    started = (rc == 0);
    return rc;
}

static inline void cp_sleep(unsigned seconds) {
    Sleep((DWORD)(seconds * 1000));
}

#else

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#ifndef MAX_PATH
#define MAX_PATH 4096
#endif

#define CP_SOCK_CLOSE(s) close((s))
#define CP_INVALID_SOCK (-1)
typedef int cp_sock_t;

static int cp_net_init(void) { return 0; }

static inline void cp_sleep(unsigned seconds) {
    sleep((unsigned int)seconds);
}

#endif

#endif /* CP_PLATFORM_H */
