# Stats API

`--api-port N` starts a read-only HTTP API on `127.0.0.1:N` (`--api-bind 0.0.0.0` to listen on
all interfaces, e.g. in a container). Mining continues if the port cannot be opened.

| Path | Content |
|---|---|
| `/summary` (also `/`) | full JSON summary |
| `/hiveos` | `{"khs": ..., "stats": {...}}` as a HiveOS `h-stats.sh` expects |

Example `/summary`:

```json
{
  "miner": "cppminer", "version": "0.5-fork.8", "algo": "quantus", "backend": "cuda",
  "worker": "rig01", "pool": "qtc-ru.kryptex.network:7049", "pool_connected": true,
  "uptime": 200,
  "hashrate": {"unit": "H/s", "windows": [10, 60, 900], "total": [133824333.9, 134612016.4, 134497562.2]},
  "devices": [{"id": 0, "name": "NVIDIA CMP 50HX", "pci": "0000:01:00.0", "bus": 1,
               "hashrate": [133824333.9, 134612016.4, 134497562.2]}],
  "shares": {"accepted": 5, "rejected": 0, "last_share_ago": 36}
}
```

- Hashrate is in H/s over the last 10 s, 60 s and 15 min. For Pearl one hash is one int8
  multiply-accumulate, so the value in TH/s equals the TMAC/s the miner logs.
- Per-device rates are the process total divided by its devices (exact with one GPU per process).
- Shares count the pool's replies to this miner's submits; dev-fee sessions are not counted.
- `pci` is empty and `bus` is -1 when the driver does not report a PCI address (CPU backend).
