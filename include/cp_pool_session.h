#ifndef CP_POOL_SESSION_H
#define CP_POOL_SESSION_H

/* Pool sessions: connect, log in, mine each job the pool sends, and reconnect with backoff.
 * Neither returns while the pool keeps answering. */
int cp_run_quantus_pool(const char* pool_host, int pool_port);
int cp_run_pearl_pool(const char* pool_host, int pool_port);

#endif /* CP_POOL_SESSION_H */
