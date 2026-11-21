# Pool protocol notes

Stratum details that matter for pool compatibility.

## Kryptex gzip stratum (v2)

Kryptex's "Pearl stratum gzip protocol" ([spec](https://gist.github.com/maxmalysh/eaaf4332dbc5ca99d0a78f24a733fffe)) is response-driven:

- `mining.authorize` always carries `"password"` (`--pool-pass`, default `x`) and `"type":"v2"`. On a `kryptex` host the `wallet` is sent as `WALLET.worker`; the separate `worker`/`agent` fields (LuckyPool) are sent too.
- If the pool answers `{"error":null,"id":1,"result":true,"type":"v2"}` the miner logs `[pool] proof encoding: gzip` and every `mining.submit` carries `plain_proof` = base64 of the **gzip** stream (zlib wbits 31) of the bincode proof (`cp_proof_gzip_b64` in `rust/cp-proof-ffi`, pure-Rust miniz_oxide). `[pearl] proof gzip: N -> M chars` shows the saving per share.
- A response without `"type":"v2"` (LuckyPool) keeps plain base64 proofs. The flag is re-evaluated on every authorize (dev-fee wallet switch included).
- Custom share difficulty: `--pool-pass d=2097152` (Kryptex default; `d = hashrate_H/s * target_share_seconds / 4294967296`).

gzip helps because the committed matrices are repetitive: the light random A (one random row) and the all-zero B^T compress well, e.g. an OpenCL proof goes from 68160 to 1948 base64 chars (35x).
