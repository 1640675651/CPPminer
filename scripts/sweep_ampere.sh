#!/usr/bin/env bash
# Sweep the CUDA fused-GEMM variants on a rented RTX 3090 / 4090 (or the
# sm_75 tensorop kernel on Turing) and print one summary table.
#
# Per variant  = kind x threadblock tile (CP_CUDA_TB) x CP_CUDA_OVERLAP:
#   1. --align-test-prod --m 8 --n 8   (once per kind x tile; the variant is
#      skipped on any difference: non-zero exit or no "tile-xor OK" line)
#   2. --mock --m 8 --n 8 --mock-diff 40   until "verify OK" (once)
#   3. --mock --m 128 --n 128 --max-nonce $NONCES --mock-diff 1e15, logging
#      power.draw / clocks.sm with nvidia-smi -lms 500
# then the best variant (highest effective TMAC/s) again with the SM clock
# locked (nvidia-smi -lgc F,F for F in $LOCK_CLOCKS; restored with -rgc) to
# find the best TMAC/s and TMAC/s per W.
#
# Usage: scripts/sweep_ampere.sh [path/to/cppminer]
# Environment:
#   DEV=0                  CUDA device index (also the nvidia-smi -i index)
#   KINDS="tensorop80"     --cuda-mma kinds; default by compute capability:
#                          >= 8.0 -> tensorop80, 7.5 -> tensorop
#   TBS="128x128 256x128 128x256"
#   OVERLAPS="0 1"
#   NONCES=4               attempts in the throughput run
#   LOCK_CLOCKS="1200 1400 1600 1800"   ("" skips the locked-clock part)
#   SKIP_ALIGN=1 / SKIP_VERIFY=1        skip steps 1 / 2
#   OUT=sweep-<host>-<time>             log directory
#
# Throughput: "scan" is the mean scan rate of attempts 2..N (the miner's
# "[gpu] attempt timing" lines), "eff" weights it by scan/(prep+scan), i.e.
# what the attempt loop delivers. Power and SM clock are averaged over the
# nvidia-smi samples between the first and last attempt line.
# Locking clocks needs root (or nvidia-smi permission); without it that part
# is reported as failed and the table is still printed.
set -u
export LC_ALL=C   # '.' decimal point in $EPOCHREALTIME and awk

BIN=${1:-}
if [[ -z "$BIN" ]]; then
    for c in ./build/cmake/cppminer ./cppminer; do
        [[ -x "$c" ]] && { BIN=$c; break; }
    done
fi
[[ -n "$BIN" && -x "$BIN" ]] || { echo "cppminer binary not found (pass its path)"; exit 1; }

DEV=${DEV:-0}
TBS=${TBS:-"128x128 256x128 128x256"}
OVERLAPS=${OVERLAPS:-"0 1"}
NONCES=${NONCES:-4}
LOCK_CLOCKS=${LOCK_CLOCKS-"1200 1400 1600 1800"}
OUT=${OUT:-sweep-$(hostname)-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$OUT"

CC=$(nvidia-smi -i "$DEV" --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr -d ' ')
GPU_NAME=$(nvidia-smi -i "$DEV" --query-gpu=name --format=csv,noheader 2>/dev/null)
PLIMIT=$(nvidia-smi -i "$DEV" --query-gpu=power.limit --format=csv,noheader,nounits 2>/dev/null)
if [[ -z "${KINDS:-}" ]]; then
    case "$CC" in
        7.5) KINDS="tensorop" ;;
        *)   KINDS="tensorop80" ;;
    esac
fi
echo "GPU $DEV: $GPU_NAME (cc $CC, power limit ${PLIMIT} W), kinds: $KINDS, logs: $OUT"

SMI_PID=""
# Samples "epoch, power.draw W, clocks.sm MHz" every 500 ms (host clock, so
# they can be matched to the miner log lines stamped the same way).
smi_start() {  # $1 = csv file
    {
        nvidia-smi -i "$DEV" --query-gpu=power.draw,clocks.sm \
            --format=csv,noheader,nounits -lms 500 2>/dev/null |
        while IFS= read -r l; do printf '%s, %s\n' "$EPOCHREALTIME" "$l"; done
    } > "$1" 2>/dev/null &
    SMI_PID=$!
}
smi_stop() {
    if [[ -n "$SMI_PID" ]]; then
        { pkill -P "$SMI_PID"; kill "$SMI_PID"; wait "$SMI_PID"; } 2>/dev/null
    fi
    SMI_PID=""
}
cleanup() {
    smi_stop
    [[ -n "${LOCKED:-}" ]] && nvidia-smi -i "$DEV" -rgc >/dev/null 2>&1
}
trap cleanup EXIT INT TERM

# Average power / SM clock over samples whose time lies in [t0, t1] (epoch s).
smi_avg() {  # csv t0 t1 -> "W MHz n"
    awk -F', *' -v t0="$2" -v t1="$3" '
        { if ($1 + 0 >= t0 + 0 && $1 + 0 <= t1 + 0 && $2 + 0 > 0) { w += $2; c += $3; n++ } }
        END { if (n) printf "%.1f %.0f %d\n", w / n, c / n, n; else print "nan nan 0" }' "$1"
}

# Throughput run; prints "scan_tmac eff_tmac W MHz samples".
perf_run() {  # kind tb overlap log [extra env]
    local kind=$1 tb=$2 ov=$3 log=$4
    local csv=${log%.log}.smi.csv
    smi_start "$csv"
    sleep 1
    CP_CUDA_TB=$tb CP_CUDA_OVERLAP=$ov timeout 3600 "$BIN" --backend cuda --devices "$DEV" \
        --cuda-mma "$kind" --mock --m 128 --n 128 --max-nonce "$NONCES" \
        --mock-diff 1e15 2>&1 | while IFS= read -r line; do
            printf '%s %s\n' "$EPOCHREALTIME" "$line"
        done > "$log"
    smi_stop
    awk '/\[gpu\] attempt timing:/ {
            n++; t[n] = $1
            for (i = 1; i <= NF; i++) {
                if ($i ~ /^prep=/) { p = substr($i, 6) + 0 }
                if ($i ~ /^scan=/) { s = substr($i, 6) + 0 }
                if ($i == "TMAC/s" || $i == "PMAC/s" || $i == "GMAC/s") {
                    r = $(i - 1) + 0
                    if ($i == "PMAC/s") r *= 1000; else if ($i == "GMAC/s") r /= 1000
                }
            }
            if (n >= 2) { rs += r; k++; ts += s; tp += p; eff += r * s }
         }
         END {
            if (k == 0) { print "nan nan 0 0"; exit }
            printf "%.2f %.2f %s %s\n", rs / k, eff / (ts + tp), t[1], t[n]
         }' "$log" > "$log.rate"
    read -r scan eff t0 t1 < "$log.rate"
    if [[ "$t0" == "0" ]]; then
        echo "$scan $eff nan nan 0"
        return
    fi
    read -r w mhz ns < <(smi_avg "$csv" "$t0" "$t1")
    echo "$scan $eff $w $mhz $ns"
}

declare -A ALIGN_OK
ROWS=()
BEST_EFF=0
BEST=""
for kind in $KINDS; do
  for tb in $TBS; do
    akey="$kind/$tb"
    if [[ -z "${SKIP_ALIGN:-}" && -z "${ALIGN_OK[$akey]:-}" ]]; then
        alog="$OUT/align-$kind-$tb.log"
        echo "== align-test-prod $kind $tb"
        CP_CUDA_TB=$tb timeout 1800 "$BIN" --backend cuda --devices "$DEV" --cuda-mma "$kind" \
            --align-test-prod --m 8 --n 8 > "$alog" 2>&1
        arc=$?
        if [[ $arc -eq 0 ]] && grep -q "GPU pipeline OK" "$alog" &&
           grep -q "CUTLASS simt/128x128 vs $kind/$tb tile-xor OK" "$alog" &&
           ! grep -q "mismatch\|bad tiles" "$alog"; then
            ALIGN_OK[$akey]=OK
        else
            ALIGN_OK[$akey]=FAIL
            echo "   align-test-prod FAILED (rc=$arc), see $alog"
        fi
    fi
    for ov in $OVERLAPS; do
        name="$kind tb=$tb overlap=$ov"
        align=${ALIGN_OK[$akey]:-skipped}
        if [[ "$align" == "FAIL" ]]; then
            ROWS+=("$(printf '%-34s %-6s %-7s %8s %8s %7s %6s %8s' "$name" FAIL - - - - - -)")
            continue
        fi
        verify="skipped"
        if [[ -z "${SKIP_VERIFY:-}" ]]; then
            vlog="$OUT/verify-$kind-$tb-ov$ov.log"
            echo "== mock verify $name"
            CP_CUDA_TB=$tb CP_CUDA_OVERLAP=$ov timeout 1800 "$BIN" --backend cuda --devices "$DEV" \
                --cuda-mma "$kind" --mock --m 8 --n 8 --mock-diff 40 > "$vlog" 2>&1
            if grep -q "verify OK" "$vlog"; then verify=OK; else
                verify=FAIL; echo "   mock verify FAILED, see $vlog"; fi
        fi
        if [[ "$verify" == "FAIL" ]]; then
            ROWS+=("$(printf '%-34s %-6s %-7s %8s %8s %7s %6s %8s' "$name" "$align" FAIL - - - - -)")
            continue
        fi
        echo "== throughput $name"
        read -r scan eff w mhz ns < <(perf_run "$kind" "$tb" "$ov" "$OUT/perf-$kind-$tb-ov$ov.log")
        perw=$(awk -v e="$eff" -v w="$w" 'BEGIN { if (w + 0 > 0) printf "%.4f", e / w; else print "nan" }')
        ROWS+=("$(printf '%-34s %-6s %-7s %8s %8s %7s %6s %8s' "$name" "$align" "$verify" "$scan" "$eff" "$w" "$mhz" "$perw")")
        if awk -v a="$eff" -v b="$BEST_EFF" 'BEGIN { exit !(a + 0 > b + 0) }'; then
            BEST_EFF=$eff
            BEST="$kind $tb $ov"
        fi
    done
  done
done

LROWS=()
if [[ -n "$BEST" && -n "$LOCK_CLOCKS" ]]; then
    read -r bkind btb bov <<< "$BEST"
    for f in $LOCK_CLOCKS; do
        echo "== locked SM clock $f MHz: $bkind tb=$btb overlap=$bov"
        if ! nvidia-smi -i "$DEV" -lgc "$f,$f" >/dev/null 2>&1; then
            LROWS+=("$(printf '%-34s %8s' "lgc $f" "lock failed (root?)")")
            continue
        fi
        LOCKED=1
        read -r scan eff w mhz ns < <(perf_run "$bkind" "$btb" "$bov" "$OUT/lock$f-$bkind-$btb-ov$bov.log")
        nvidia-smi -i "$DEV" -rgc >/dev/null 2>&1
        LOCKED=""
        perw=$(awk -v e="$eff" -v w="$w" 'BEGIN { if (w + 0 > 0) printf "%.4f", e / w; else print "nan" }')
        LROWS+=("$(printf '%-34s %8s %8s %7s %6s %8s' "lgc $f: $bkind $btb ov=$bov" "$scan" "$eff" "$w" "$mhz" "$perw")")
    done
fi

{
    echo
    echo "Summary: $GPU_NAME (cc $CC, limit ${PLIMIT} W), m=n=128, $NONCES attempts, $(date)"
    printf '%-34s %-6s %-7s %8s %8s %7s %6s %8s\n' variant align verify scanTMAC effTMAC avgW smMHz TMAC/W
    for r in "${ROWS[@]}"; do echo "$r"; done
    if [[ ${#LROWS[@]} -gt 0 ]]; then
        echo
        printf '%-34s %8s %8s %7s %6s %8s\n' "locked clock (best variant)" scanTMAC effTMAC avgW smMHz TMAC/W
        for r in "${LROWS[@]}"; do echo "$r"; done
    fi
    echo "best (effective TMAC/s): $BEST ($BEST_EFF)"
} | tee "$OUT/summary.txt"
