/*
 * tile_jackpot: ACT-SHAVE kernel behind the compiler's VPU.TileJackpot op (NPU3720), built by
 * ../build_kernels.ps1 (movi.py, plain C).
 *
 * The keyed BLAKE3 jackpot of the CPU miner (cp_jackpot.hpp) over the tile words that TileXorAcc ops leave in CMX,
 * so only one word per tile row goes back to DDR:
 *   per tile, the 32 milestone words (K = 4096) are folded into 16 (word t = rotl(word t, 13) ^ milestone,
 *   t = milestone % 16), hashed by one BLAKE3 compression keyed with the attempt key (counter 0, block length 64,
 *   flags 0x1B) and the digest compared with the bound from word 7 down (equal beats it).
 *
 * Parameter block (compiler ABI: one 0x28-byte MemRefData record per tensor, then int64 kernel args):
 *   params             int32 [1, 1, T, 16]        every row: key[8], bound[8]
 *   words_0 .. n-1     int32 [1, 1, T, G * 4]     TileXorAcc tile words, n = steps * chains, step-major: words
 *                                                 s * chains + c are milestones s*G .. s*G+G-1 of chain c's 4 tile
 *                                                 columns (partial-major within a tile row); steps * G = 32
 *   hits               int32 [1, 1, T, 1]         per tile row 1 + the lowest tile column c * 4 + j that beats the
 *                                                 bound, or 0
 *   int64 chains
 *   [int64 0x5A5A0001, int64 shave_index, int64 shave_count]   (added when the compiler runs the op on all ACT
 *                                                               SHAVEs of a tile; each takes a share of rows)
 *
 * The 4 tile columns of a chain are the 4 lanes of a vector: one 128-bit load per milestone, one vector compression
 * per 4 tiles. Almost every digest loses on word 7, so the rest is compared only for lanes that do not.
 */

typedef unsigned uint4v __attribute__((ext_vector_type(4)));

typedef struct __attribute__((packed)) MemRef {
    unsigned data_lo, data_hi, is_static, num_dims, dims_addr, strides_addr, data_type;
    unsigned order_lo, order_hi, location;
} MemRef;

#define MILESTONES 32
#define MSG_WORDS 16
#define LROT 13
#define MAX_WORD_INPUTS 64
#define SHAVE_SPLIT_MAGIC 0x5A5A0001u

/* The kernel image is linked without a C runtime; aggregate initialisation may still call these. */
void *memset(void *dst, int value, unsigned size) {
    volatile unsigned char *d = (volatile unsigned char *)dst;
    for (unsigned i = 0; i < size; ++i) {
        d[i] = (unsigned char)value;
    }
    return dst;
}

void *memcpy(void *dst, const void *src, unsigned size) {
    volatile unsigned char *d = (volatile unsigned char *)dst;
    const volatile unsigned char *s = (const volatile unsigned char *)src;
    for (unsigned i = 0; i < size; ++i) {
        d[i] = s[i];
    }
    return dst;
}

/* Stride of dimension i in bytes (strides are int64 bits; only the low word matters here). */
static unsigned stride_bytes(const MemRef *r, unsigned i) {
    return ((const unsigned *)r->strides_addr)[2 * i] / 8;
}

/* Number of MemRef records: the word right after them is the chain count (small), while a record starts with a
 * tensor address (one uncached load of ~70 cycles per record). Returns 0 when there are too many records. */
static unsigned count_records(const MemRef *r) {
    for (unsigned k = 3; k <= MAX_WORD_INPUTS + 2; ++k) {
        if (r[k].data_lo <= MAX_WORD_INPUTS) {
            return k;
        }
    }
    return 0;
}

#define ROTR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#define G(a, b, c, d, x, y)        \
    do {                           \
        a = a + b + (x);           \
        d = ROTR(d ^ a, 16);       \
        c = c + d;                 \
        b = ROTR(b ^ c, 12);       \
        a = a + b + (y);           \
        d = ROTR(d ^ a, 8);        \
        c = c + d;                 \
        b = ROTR(b ^ c, 7);        \
    } while (0)

/* Message word order of each round (BLAKE3's permutation applied r times). */
static const unsigned char SCHEDULE[7][16] = {
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},  {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8},
        {3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1},  {10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6},
        {12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4},  {9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7},
        {11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13}};

/* Keyed BLAKE3 compression of 4 messages (one per lane); digest words 0..7 into out. About 1130 cycles
 * (750 VAU ops): moviCompile's schedule, limited by the VAU's dependent-op latencies and write ports. */
static void compress4(const unsigned key[8], const uint4v *m, uint4v *out) {
    uint4v v0 = (uint4v)key[0], v1 = (uint4v)key[1], v2 = (uint4v)key[2], v3 = (uint4v)key[3];
    uint4v v4 = (uint4v)key[4], v5 = (uint4v)key[5], v6 = (uint4v)key[6], v7 = (uint4v)key[7];
    uint4v v8 = (uint4v)0x6A09E667u, v9 = (uint4v)0xBB67AE85u, v10 = (uint4v)0x3C6EF372u, v11 = (uint4v)0xA54FF53Au;
    uint4v v12 = (uint4v)0u, v13 = (uint4v)0u, v14 = (uint4v)64u, v15 = (uint4v)0x1Bu;
    /* Unrolled, so the schedule indices are constants and the four G of each half-round interleave. */
#pragma clang loop unroll(full)
    for (unsigned r = 0; r < 7; ++r) {
        const unsigned char *s = SCHEDULE[r];
        G(v0, v4, v8, v12, m[s[0]], m[s[1]]);
        G(v1, v5, v9, v13, m[s[2]], m[s[3]]);
        G(v2, v6, v10, v14, m[s[4]], m[s[5]]);
        G(v3, v7, v11, v15, m[s[6]], m[s[7]]);
        G(v0, v5, v10, v15, m[s[8]], m[s[9]]);
        G(v1, v6, v11, v12, m[s[10]], m[s[11]]);
        G(v2, v7, v8, v13, m[s[12]], m[s[13]]);
        G(v3, v4, v9, v14, m[s[14]], m[s[15]]);
    }
    out[0] = v0 ^ v8;
    out[1] = v1 ^ v9;
    out[2] = v2 ^ v10;
    out[3] = v3 ^ v11;
    out[4] = v4 ^ v12;
    out[5] = v5 ^ v13;
    out[6] = v6 ^ v14;
    out[7] = v7 ^ v15;
}

/* Lowest lane (0..3) whose digest is at most the bound, or -1. */
static int first_beating_lane(const uint4v *digest, const unsigned bound[8]) {
    /* Through memory: moviCompile would otherwise use SAU lane ops in bundles moviAsm rejects. */
    volatile uint4v d7 = digest[7];
    const volatile unsigned *w7 = (const volatile unsigned *)&d7;
    if (w7[0] > bound[7] && w7[1] > bound[7] && w7[2] > bound[7] && w7[3] > bound[7]) {
        return -1;
    }
    volatile uint4v all[8];
    for (unsigned w = 0; w < 8; ++w) {
        all[w] = digest[w];
    }
    const volatile unsigned *words = (const volatile unsigned *)all;
    for (int lane = 0; lane < 4; ++lane) {
        int beats = 1;
        for (int w = 7; w >= 0; --w) {
            const unsigned d = words[4 * w + lane];
            if (d < bound[w]) {
                break;
            }
            if (d > bound[w]) {
                beats = 0;
                break;
            }
        }
        if (beats) {
            return lane;
        }
    }
    return -1;
}

void controlled_act(unsigned params) {
    const MemRef *r = (const MemRef *)params;

    const unsigned records = count_records(r);
    if (records < 3) {
        return;
    }
    const unsigned *args = (const unsigned *)&r[records]; /* int64 values, little endian */
    const unsigned chains = args[0];
    const unsigned num_words = records - 2;
    if (chains == 0 || num_words % chains != 0 || num_words > MAX_WORD_INPUTS) {
        return;
    }
    const unsigned steps = num_words / chains;
    if (MILESTONES % steps != 0) {
        return;
    }
    const unsigned per_step = MILESTONES / steps; /* G */

    const unsigned long long dims01 = *(const unsigned long long *)r[1].dims_addr; /* W, H in one load */
    const unsigned width = (unsigned)dims01;
    const unsigned tile_rows = (unsigned)(dims01 >> 32);
    if (width != per_step * 4) {
        return; /* 4 tile columns per chain: one vector lane each */
    }

    const unsigned char *word_base[MAX_WORD_INPUTS];
    for (unsigned i = 0; i < num_words; ++i) {
        word_base[i] = (const unsigned char *)r[1 + i].data_lo;
    }
    const unsigned word_row = stride_bytes(&r[1], 1); /* the compiler allocates the inputs identically */
    const MemRef *out = &r[records - 1];
    unsigned char *hits = (unsigned char *)out->data_lo;
    const unsigned hit_row = stride_bytes(out, 1);

    unsigned key[8], bound[8];
    const volatile unsigned *p = (const volatile unsigned *)r[0].data_lo;
    for (unsigned i = 0; i < 8; ++i) {
        key[i] = p[i];
        bound[i] = p[8 + i];
    }

    unsigned shave = 0, shaves = 1;
    if (args[2] == SHAVE_SPLIT_MAGIC && args[6] > 0 && args[4] < args[6]) {
        shave = args[4];
        shaves = args[6];
    }
    const unsigned row_begin = tile_rows * shave / shaves, row_end = tile_rows * (shave + 1) / shaves;

    uint4v m[MSG_WORDS];
    uint4v digest[8];
    for (unsigned tr = row_begin; tr < row_end; ++tr) {
        unsigned hit = 0;
        for (unsigned c = 0; c < chains && !hit; ++c) {
            /* Milestones t and t + 16 (32 in all): m[t] = rotl(w_t, 13) ^ w_{t+16}. */
            unsigned t = 0;
            for (unsigned s = 0; s < steps; ++s) {
                const uint4v *w = (const uint4v *)(word_base[s * chains + c] + tr * word_row);
                for (unsigned g = 0; g < per_step; ++g, ++t) {
                    if (t < MSG_WORDS) {
                        m[t] = w[g];
                    } else {
                        m[t - MSG_WORDS] = ((m[t - MSG_WORDS] << LROT) | (m[t - MSG_WORDS] >> (32 - LROT))) ^ w[g];
                    }
                }
            }
            compress4(key, m, digest);
            const int lane = first_beating_lane(digest, bound);
            if (lane >= 0) {
                hit = c * 4 + (unsigned)lane + 1;
            }
        }
        *(unsigned *)(hits + tr * hit_row) = hit;
    }
}
