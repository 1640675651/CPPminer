# Hashrate calculation

All displayed and submitted hashrates are derived from **hash tiles scanned** and a wall-clock interval:

```text
MAC/s = tiles × (PP_HASH_H × PP_HASH_W × K_DIM) / elapsed_sec
```

With current constants (`include/cp_config.h`):

| Symbol | Value | Role |
|--------|------:|------|
| `PP_HASH_H` | 8 | Hash-tile height |
| `PP_HASH_W` | 16 | Hash-tile width |
| `K_DIM` | 4096 | Inner dimension |
| MACs / hash tile | **524 288** | `8 × 16 × 4096` |

Implementation: `cp_pp_macs_per_hash_tile()`, `cp_pp_mac_rate_from_tiles()` in `src/common/cp_util.cpp`.

Rates are printed as TMAC/s when ≥ 10¹² MAC/s (`cp_pp_fmt_mac_rate`).

---

## Where each number comes from

| Log line | Formula | Time base | Includes |
|----------|---------|-----------|----------|
| `[ocl]` / `[cpu]` / `[gpu] plain_proof progress` | tiles_so_far / scan_elapsed | From scan start of **this** matrix attempt | Scan only |
| `[…] attempt timing: … X TMAC/s` | tiles_this_attempt / **scan_sec** | Scan start → scan end | Scan only (`prep=` is printed separately and **not** in the rate) |
| `[plain] … no share yet` | `tiles_scanned_total` / `(now − job_t0)` | Job start → now | Prep, scans, gaps, hit handling since job start |
| `[plain] proof ready … X TMAC/s (hs=…)` | `tiles_since_prev` / `interval_sec` | Previous share → this share (or **job start** → first share) | Prep, misses, scans, and gaps in that interval — **not** proof build |
| Pool submit `hs` | Same as proof-ready | Same interval | Same as proof-ready |

### Progress (in-scan)

Printed about every 2s during a matrix scan (interruptible wait; does not stall the mine loop after scan ends).

- **Numerator:** tiles completed so far in this attempt  
- **Denominator:** `now − scan_t0`  
- Matches GEMM throughput (~steady ~4.7 TMAC/s on typical OpenCL runs)

### Attempt timing

Logged once per matrix attempt when the scan returns:

```text
[ocl] attempt timing: prep=0.34s scan=10.15s 4.66 TMAC/s
```

- **TMAC/s** = tiles / `scan_sec` only  
- Early exit shortens both tiles and `scan_sec` together → rate stays close to progress  
- Prep (~0.3s GPU matrix prep) does **not** dilute this number  

### Job-level `[plain] … no share yet`

```text
[plain] nonce=… attempts=… (…/s) 4.16 TMAC/s no share yet
```

- **Numerator:** cumulative tiles for the whole job  
- **Denominator:** wall time since `job_t0`  
- Diluted by every prep, early-exit overhead fraction, and gaps  

Used for operator visibility only; **not** sent to the pool.

### Share / pool hashrate

On each jackpot hit, the mine loop records:

- `tiles_since_prev` = tiles scanned since the previous accepted share enqueue (0 baseline at job start)  
- `interval_sec` = wall time since that previous share timestamp (job start for the first share)

Then:

```text
hs = cp_pp_mac_rate_from_tiles(tiles_since_prev, interval_sec)
```

Submitted as JSON field `hs` in `mining.submit` (`cp_pool_send_plain_proof_submit`). The same value is printed on `[plain] proof ready`.

**Proof construction is async and is not included in `hs`.** The interval is snapped when the hit is enqueued, before proof build.

---

## Why the numbers disagree

Typical ordering (OpenCL production):

```text
progress / attempt timing   ≈ 4.6–4.8 TMAC/s   (scan-only)
proof ready / pool hs       ≈ slightly lower   (interval includes prep + prior misses)
no share yet                ≈ lower still      (whole job wall clock)
```

### Early exit does not lower attempt TMAC

Stopping mid-matrix reduces work and scan time together. Attempt timing stays near in-scan rate.

### Early exit *does* lower interval / job rates

Each attempt still pays fixed cost (prep ≈ 0.3s; on hit, device→host A readback). Short hits make that overhead a larger share of wall time → fewer tiles per second of calendar time.

### First share of a job

Baseline is **job start**, not “start of this attempt.” If nonce 0 was a full miss and nonce 1 hits, pool `hs` includes the miss’s prep+scan as well as the hitting attempt. That is why `proof ready` can read ~4.42 while that attempt’s `attempt timing` shows ~4.66.

### Later shares

Interval resets at each successfully enqueued share, so an early exit on an old share does **not** keep diluting later `hs` values.

---

## Timeline sketch (one job)

```text
job_t0
  ├─ attempt 0: prep ── scan (miss, full matrix) ──►
  ├─ attempt 1: prep ── scan … HIT
  │                 ▲
  │                 └─ attempt timing: tiles₁ / scan₁
  │                 └─ pool hs (1st share): (tiles₀+tiles₁) / (now − job_t0)
  ├─ … reclaim / D2H / enqueue (not in hs) … proof async (not in hs)
  ├─ attempt 2: …
  └─ next HIT
                    └─ pool hs: Δtiles / (now − t_prev_share)
```

---

## Code map

| Piece | Location |
|-------|----------|
| MAC/tile and rate helpers | `src/common/cp_util.cpp` |
| Job totals / share interval | `src/common/cp_mine.cpp` |
| Pool `hs` | `src/common/cp_share_queue.cpp` → `cp_pool_send_plain_proof_submit` |
| OpenCL progress + attempt timing | `src/opencl/cp_opencl_worker.cpp` |
| CPU progress + attempt timing | `src/cpu/cp_cpu_worker.cpp` |
| CUDA progress + attempt timing | `src/cuda/cp_gpu.cu` |
| Tile size constants | `include/cp_config.h` |
