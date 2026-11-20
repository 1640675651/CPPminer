// Boustrophedon walk arithmetic for the launch code (case5_gemm_launch.cpp); the same file as
// hexagon_test/gemm_xor/onednn/case5_walk.hpp (which also has the host check tool). No GPU dependency.
//
// gemmstone's boustrophedon walk covers the group grid in slices of `slice` m-groups (negative: n-groups),
// sweeping each slice back and forth; `thresh` changes the slice size by one from that slice on (negative:
// smaller). The default slice (gemmstone / oneDNN) makes the concurrent wave of work groups about square in
// C elements: sqrt(concurrent groups / bias) m-groups, 6 on the A770 (64 groups of 256 x 128 elements) and
// 5 on the B580 (20 groups of 256 x 256).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace case5_walk {

struct BoustrophedonPair {
    int32_t slice = 0;
    int32_t thresh = 0;
};

// gemmstone's default slice / threshold (oneDNN's derivation), with the m-slice target scaled by mscale.
inline BoustrophedonPair derive_boustrophedon(uint32_t groups_m, uint32_t groups_n, uint32_t concurrent_tg,
        double bias, double mscale) {
    auto div_up = [](int a, int b) { return (a + b - 1) / b; };
    const double sm = std::sqrt(double(concurrent_tg) / bias) * mscale;
    const double sn = std::sqrt(double(concurrent_tg) * bias);

    int32_t slice = 0;
    int32_t thresh = 0;
    bool ok = false;

    for (bool nslice : {groups_m > groups_n, groups_m <= groups_n}) {
        double s = nslice ? sn : sm;
        auto sf = int(std::floor(s));
        auto sc = int(std::ceil(s));
        if (concurrent_tg % sc == 0) {
            s = sf = sc;
        }
        if (concurrent_tg % (sc + 1) == 0) {
            s = sf = sc = sc + 1;
        }

        const int gc = nslice ? int(groups_n) : int(groups_m);
        const int gco = nslice ? int(groups_m) : int(groups_n);

        for (int srange = 0; srange <= 2 && !ok; ++srange) {
            int s0 = (srange < 2) ? sc : sf;
            const bool up = (srange == 1);
            const int s1 = s0 + (up ? 1 : -1);
            if (s1 <= 0) {
                continue;
            }

            const auto rem = gc % s0;
            if (!rem || up) {
                thresh = gc / s0 - rem;
            } else {
                thresh = div_up(gc, s0) - (s0 - rem);
            }

            ok = (thresh >= 0) && (gco >= 2 * std::max(s0, s1));
            slice = s0;
            if (!up) {
                if (thresh > 0) {
                    thresh = -thresh;
                } else {
                    slice--;
                    thresh = gc;
                }
            }
            if (nslice) {
                slice *= -1;
            }
        }

        if (ok) {
            break;
        }
    }

    if (!ok) {
        const bool nslice = (groups_m > groups_n);
        const double s = nslice ? sn : sm;
        const int gc = nslice ? int(groups_n) : int(groups_m);

        if (gc < s * 1.5) {
            slice = gc;
        } else {
            slice = gc / div_up(gc, int(std::round(s)));
        }

        thresh = std::max(0, (gc / slice) - (gc % slice));
        if (nslice) {
            slice *= -1;
        }
    }

    if (slice == 0) {
        slice = 1;
        thresh = int(groups_m);
    }
    return {slice, thresh};
}

// One group id through gemmstone's boustrophedon decode (walk_orders.cxx, gemmBoustrophedonOrder; its
// XeHPC+ and older instruction sequences compute the same; same arithmetic as tools/boustrophedon_check.py).
// The float-to-int conversion after the square root rounds to nearest (the kernel's mov); 2q + 1.25 is never
// the square of a half-integer, so there are no ties. False: the kernel would divide by zero.
inline bool boustrophedon_decode(int64_t g, int64_t s0, int64_t thresh, int64_t gcm, int64_t gcn, int64_t &m,
        int64_t &n) {
    const int64_t as0 = s0 < 0 ? -s0 : s0;
    const int64_t athresh = thresh < 0 ? -thresh : thresh;
    const int64_t ithresh = athresh * as0;
    const int64_t v = (s0 < 0) ? gcm : gcn;
    const int64_t u = (s0 >= 0) ? gcm : gcn;
    int64_t t0 = g - v * ithresh;
    const bool past = t0 >= 0;
    int64_t q = past ? t0 : g;
    int64_t s1 = as0 + (past ? -1 : 0);
    if (past && thresh >= 0) s1 = as0 + 1;
    int64_t t1 = s1 * v;
    if (t1 <= 0) return false;
    int64_t i0 = (q / t1) * s1;
    int64_t islice = q / t1;
    if (past) i0 += ithresh;
    q = q % t1;
    t0 = std::max<int64_t>(0, u - i0);
    const int64_t s = std::min(s1, t0);
    if (past) islice += athresh;
    const int64_t t2 = s * s;
    t1 -= s * s;
    const bool notfirst = i0 > 0, notlast = s1 < t0;
    int mode = 0;
    bool flip_i = false;
    if (notfirst && q < t2) {
        t0 = (s * s - s) / 2;
        flip_i = true;
        if (q < t0) { q = t0 - q - 1; mode = 1; } else { q -= t0; mode = 2; }
    } else if (notlast && q >= t1) {
        q -= t1;
        t0 = (s * s + s) / 2;
        if (q < t0) { q = t0 - q - 1; mode = 2; } else { q -= t0; mode = 1; }
    }
    int64_t i = 0, j = 0;
    if (mode == 0) {
        if (s <= 0) return false;
        j = q / s;
        i = i0 + q % s;
    } else {
        const int64_t x = std::llround(std::sqrt(2.0 * double(q) + 1.25));
        if (mode == 2) {
            j = x;
            const int64_t tri = (j * j - j + 1) >> 1;
            j -= 1;
            i = q - tri;
        } else {
            i = x;
            const int64_t tri = (i * i - i + 1) >> 1;
            j = q - tri;
        }
        if (flip_i) i = s - i - 1; else j = v - j - 1;
        i += i0;
    }
    if (islice & 1) j = v - j - 1;
    if (s0 >= 0) { m = i; n = j; } else { m = j; n = i; }
    return true;
}

// True if the pair maps the groups of the grid one to one onto its tiles (and fits the kernel's 16-bit
// multiplies: group counts and |slice| * |thresh| below 65536).
inline bool boustrophedon_map_ok(uint32_t groups_m, uint32_t groups_n, const BoustrophedonPair &p) {
    if (p.slice == 0 || groups_m == 0 || groups_n == 0 || groups_m >= 0x10000 || groups_n >= 0x10000)
        return false;
    const int64_t as0 = p.slice < 0 ? -int64_t(p.slice) : p.slice;
    const int64_t ath = p.thresh < 0 ? -int64_t(p.thresh) : p.thresh;
    if (as0 * ath >= 0x10000) return false;
    const int64_t count = int64_t(groups_m) * groups_n;
    std::vector<uint8_t> seen(size_t(count), 0);
    for (int64_t g = 0; g < count; ++g) {
        int64_t m = 0, n = 0;
        if (!boustrophedon_decode(g, p.slice, p.thresh, groups_m, groups_n, m, n)) return false;
        if (m < 0 || m >= int64_t(groups_m) || n < 0 || n >= int64_t(groups_n)) return false;
        auto &cell = seen[size_t(m * groups_n + n)];
        if (cell) return false;
        cell = 1;
    }
    return true;
}

}  // namespace case5_walk
