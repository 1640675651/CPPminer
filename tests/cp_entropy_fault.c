/* Linux integration fixture: make the miner's entropy source unavailable. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>

FILE* fopen(const char* path, const char* mode)
{
    if(!strcmp(path, "/dev/urandom")){
        errno = EIO;
        return NULL;
    }
    FILE* (*real_fopen)(const char*, const char*) = dlsym(RTLD_NEXT, "fopen");
    return real_fopen(path, mode);
}
