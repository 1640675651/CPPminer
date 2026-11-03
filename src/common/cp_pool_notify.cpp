/* mining.notify parsing, apart from the pool connection so the protocol test can link it. */
#include "cp_pool.h"
#include "cp_state.h"
#include "cp_util.h"

#include <cstdio>
#include <cstring>

static int reject_notify(const char* why)
{
    printf("[pool] mining.notify rejected: %s\n", why);
    fflush(stdout);
    return 0;
}

int cp_pool_parse_notify(const char* json,
                         char* job_id, int job_len,
                         char* header_hex, int header_len,
                         char* target_hex, int target_len,
                         uint32_t* cert_version_out)
{
    job_id[0] = header_hex[0] = target_hex[0] = 0;
    if(cert_version_out)
        *cert_version_out = 0;

    /* cert_version may sit beside params, inside it, or in an object in a params list. A
     * version this miner does not know would only produce proofs the pool rejects. */
    uint32_t cert_version = 0;
    if(strstr(json, "\"cert_version\"")){
        const double cv = cp_json_num(json, "cert_version");
        if(cv == 1.0 || cv == 2.0 || cv == 3.0)
            cert_version = (uint32_t)cv;
        else if(!g_cert_version_forced)
            return reject_notify("unsupported cert_version");
    }

    const char* params = cp_json_member(json, "params");
    if(!params || *params == '{'){
        /* Members of the params object, or of the message itself when it has no params;
         * never of objects nested inside them. An over-long job id leaves job_id empty. */
        const char* fields = params ? params : json;
        cp_json_member_str(fields, "job_id", job_id, job_len);
        cp_json_member_str(fields, "header", header_hex, header_len);
        cp_json_member_str(fields, "target", target_hex, target_len);
    } else if(*params == '['){
        /* Positional: the first string is the job id, then the target and header by length. */
        const char* p = params + 1;
        int seen_id = 0;
        while(*p && *p != ']'){
            while(*p==' ' || *p=='\t' || *p==',') p++;
            if(*p == '"'){
                p++;
                char tmp[320];
                int i = 0, len = 0;
                for(; *p && *p != '"'; ++p, ++len)
                    if(i < (int)sizeof(tmp) - 1) tmp[i++] = *p;
                tmp[i] = 0;
                if(*p=='"') p++;
                if(len == 0) continue;
                if(!seen_id){
                    seen_id = 1;
                    if(len < job_len) memcpy(job_id, tmp, (size_t)len + 1);
                } else if(len == TARGET_HEX_LEN && !target_hex[0] && len < target_len){
                    memcpy(target_hex, tmp, (size_t)len + 1);
                } else if(len == HEADER_HEX_LEN && !header_hex[0] && len < header_len){
                    memcpy(header_hex, tmp, (size_t)len + 1);
                }
            } else if(*p=='{' ){
                break;
            } else {
                while(*p && *p != ',' && *p != ']') p++;
            }
        }
    } else {
        return 0;
    }

    if(!header_hex[0]) return 0;
    if(!job_id[0]) return reject_notify("missing, empty or over-long job_id");
    if(cert_version_out)
        *cert_version_out = cert_version;
    return 1;
}
