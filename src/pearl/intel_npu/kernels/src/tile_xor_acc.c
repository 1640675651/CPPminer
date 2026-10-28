/*
 * tile_xor_acc: ACT-SHAVE kernel behind the compiler's VPU.TileXorAcc op (NPU3720), built by
 * ../build_kernels.ps1 together with the hand-scheduled routines from gen_tile_xor_asm.py.
 *
 * Parameter block (compiler ABI: one 0x28-byte MemRefData record per tensor, then int64 kernel args):
 *   in_0 .. in_{n-1}   int32 [1, 1, H, W]      (in_0 is the running accumulator when has_acc)
 *   acc_out            int32 [1, 1, H, W]
 *   xor_out            int32 [1, 1, H/R, G * W/16]   (tile words, partial-major within a tile row; R = 8 or 16, the
 *                                                     tile height, is H over xor_out's rows)
 *   int64 acc_mode      0: no accumulator, 1: in_0 is the accumulator, 2: in_0 is only a scheduling dependency,
 *                       3: every input is already a cumulative sum: XOR-reduce each, write nothing to acc_out
 *   [int64 0x5A5A0001, int64 shave_index, int64 shave_count]   (added when the compiler runs the op on all ACT
 *                                                               SHAVEs of a tile; each takes a share of rows)
 *
 * For each partial g: acc += partial_g, then xor_out[i][g * W/16 + j] = XOR of acc over the R x 16 tile (i, j).
 * The records are counted by looking for acc_mode right after them: it is 0..2, never a tensor address. The
 * parameter block lives in uncached memory (~70 cycles per 32-bit load, not pipelined), so the kernel reads as few
 * words of it as it can: the record count is found by probing the two common layouts (4 or 8 partials with an
 * accumulator) before falling back to a scan, all partials are assumed to share the first one's row stride (the
 * compiler allocates them identically), and dims come in one 64-bit load.
 * dims[] and strides[] are listed innermost first; strides are in bits.
 *
 * With TX_ASM_G4, the hand-scheduled routines of gen_tile_xor_asm.py do the work: eight partials in one pass
 * (tx_strips_g8, or tx_strips_g8_r16 for 16-row tiles; the accumulator stays in registers across all of them), other
 * multiples of four as G = 4 passes (tx_strips_g4: one call per SHAVE with a strip record per 16-column strip and per
 * group of four partials, the accumulator going through acc_out in place between passes), cumulative inputs through
 * tx_xor_strips. The G = 4 and XOR-only routines are 8-row only; other partial counts and tile heights use the plain
 * C loops.
 */

typedef int int4v __attribute__((ext_vector_type(4)));

typedef struct __attribute__((packed)) MemRef {
    unsigned data_lo, data_hi, is_static, num_dims, dims_addr, strides_addr, data_type;
    unsigned order_lo, order_hi, location;
} MemRef;

#define TILE_COLS 16
#define MAX_PARTIALS 8
#define MAX_RECORDS (MAX_PARTIALS + 3) /* accumulator, partials, acc_out, xor_out */
#define SHAVE_SPLIT_MAGIC 0x5A5A0001u
#ifndef TX_G8_OFF
#define TX_G8_OFF 0 /* 1: run eight partials as two G = 4 passes instead of the single-pass routine */
#endif

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

/* Number of MemRef records: the word right after them is acc_mode (0..2), while a record starts with a tensor
 * address (acc_mode is 0..3). The scan starts at 6 (4 partials without an accumulator), so the op needs at least 4 partials, or 3
 * with an accumulator; 4 partials then cost 1-2 loads and 8 partials 6, instead of 5 and 9 from 3. Returns 0 when
 * there are too many records. */
static unsigned count_records(const MemRef *r) {
    for (unsigned k = 6; k <= MAX_RECORDS; ++k) {
        if (r[k].data_lo <= 3) {
            return k;
        }
    }
    return 0;
}

typedef struct Job {
    unsigned g;                  /* partial sums */
    unsigned th;                 /* tile rows: 8 or 16 */
    const unsigned char *acc_in; /* a zeroed 64-byte buffer with zero strides when there is no accumulator */
    unsigned acc_row, acc_col;   /* byte strides of acc_in rows and 16-column tiles */
    const unsigned char *part[MAX_PARTIALS];
    unsigned char *acc_out;
    unsigned char *xor_out;
    unsigned tile_row_begin, tile_row_end, tile_col_begin, tile_col_end; /* this SHAVE's share of the tiles */
    unsigned in_row, out_row, x_row, x_part;
} Job;

/* Plain C path: per tile, 2-row quarters of 8 vectors; X accumulators per partial. */
static void run_c(const Job *job) {
    const unsigned G = job->g;
    for (unsigned tr = job->tile_row_begin; tr < job->tile_row_end; ++tr) {
        for (unsigned tc = job->tile_col_begin; tc < job->tile_col_end; ++tc) {
            int4v xv[MAX_PARTIALS];
            for (unsigned g = 0; g < G; ++g) {
                xv[g] = (int4v)0;
            }
            for (unsigned q = 0; q < job->th / 2; ++q) {
                const unsigned row = tr * job->th + 2 * q;
                const unsigned i0 = row * job->in_row + tc * TILE_COLS * 4;
                const int4v *s0 = (const int4v *)(job->acc_in + row * job->acc_row + tc * job->acc_col);
                const int4v *s1 = (const int4v *)((const unsigned char *)s0 + job->acc_row);
                int4v a0 = s0[0], a1 = s0[1], a2 = s0[2], a3 = s0[3];
                int4v a4 = s1[0], a5 = s1[1], a6 = s1[2], a7 = s1[3];
                for (unsigned g = 0; g < G; ++g) {
                    const int4v *p0 = (const int4v *)(job->part[g] + i0);
                    const int4v *p1 = (const int4v *)(job->part[g] + i0 + job->in_row);
                    a0 += p0[0]; a1 += p0[1]; a2 += p0[2]; a3 += p0[3];
                    a4 += p1[0]; a5 += p1[1]; a6 += p1[2]; a7 += p1[3];
                    xv[g] ^= ((a0 ^ a1) ^ (a2 ^ a3)) ^ ((a4 ^ a5) ^ (a6 ^ a7));
                }
                int4v *d0 = (int4v *)(job->acc_out + row * job->out_row + tc * TILE_COLS * 4);
                int4v *d1 = (int4v *)((unsigned char *)d0 + job->out_row);
                d0[0] = a0; d0[1] = a1; d0[2] = a2; d0[3] = a3;
                d1[0] = a4; d1[1] = a5; d1[2] = a6; d1[3] = a7;
            }
            /* Fold the 4 lanes through memory: moviCompile otherwise emits SAU.XORX in a bundle that moviAsm
             * rejects (shared resource with the VAU). */
            volatile int4v lanes[MAX_PARTIALS];
            for (unsigned g = 0; g < G; ++g) {
                lanes[g] = xv[g];
            }
            for (unsigned g = 0; g < G; ++g) {
                const volatile int *w = (const volatile int *)&lanes[g];
                *(int *)(job->xor_out + g * job->x_part + tr * job->x_row + tc * 4) = (w[0] ^ w[1]) ^ (w[2] ^ w[3]);
            }
        }
    }
}

#if defined(TX_ASM_G4) && !defined(TX_XOR_C)
/* Hand-scheduled XOR-only strips; gen_tile_xor_asm.py (generate_xor) documents the records. */
extern void tx_xor_strips(const unsigned *records);

#define XOR_RECORD_WORDS 10
#define MAX_XOR_RECORDS 16

/* Cumulative inputs (acc_mode 3), 8-row tiles: xor_out[g] = XOR over every 8x16 tile of part[g]; no accumulation.
 * One strip record per input and 16-column strip of this SHAVE's share, in batches. */
static void run_xor_only_asm(const Job *job) {
    unsigned records[MAX_XOR_RECORDS * XOR_RECORD_WORDS];
    unsigned dummy[4];
    const unsigned row0 = job->tile_row_begin * 8;
    const unsigned ntiles = job->tile_row_end - job->tile_row_begin;
    unsigned *a = records;
    unsigned count = 0;
    for (unsigned g = 0; g < job->g; ++g) {
        for (unsigned tc = job->tile_col_begin; tc < job->tile_col_end; ++tc) {
            if (count == MAX_XOR_RECORDS) {
                a[-1] = 0;
                tx_xor_strips(records);
                a = records;
                count = 0;
            }
            const unsigned p = (unsigned)job->part[g] + row0 * job->in_row + tc * TILE_COLS * 4;
            *a++ = p;
            *a++ = p + job->in_row;
            *a++ = p + 2 * job->in_row;
            *a++ = p + 3 * job->in_row;
            *a++ = 4 * job->in_row - 64; /* the routine post-increments by 4 x 16 bytes per quarter */
            *a++ = (unsigned)&dummy[0];
            *a++ = job->x_row;
            *a++ = (unsigned)(job->xor_out + g * job->x_part + job->tile_row_begin * job->x_row + tc * 4);
            *a++ = ntiles;
            *a++ = 1; /* another record follows */
            ++count;
        }
    }
    a[-1] = 0;
    tx_xor_strips(records);
}
#endif

static void run_xor_only_c(const Job *job) {
    for (unsigned g = 0; g < job->g; ++g) {
        const unsigned char *part = job->part[g];
        unsigned char *xo = job->xor_out + g * job->x_part;
        for (unsigned tr = job->tile_row_begin; tr < job->tile_row_end; ++tr) {
            for (unsigned tc = job->tile_col_begin; tc < job->tile_col_end; ++tc) {
                const unsigned base = tr * job->th * job->in_row + tc * TILE_COLS * 4;
                const int4v *p0 = (const int4v *)(part + base);
                int4v x = p0[0] ^ p0[1] ^ p0[2] ^ p0[3];
                for (unsigned row = 1; row < job->th; ++row) {
                    const int4v *p = (const int4v *)(part + base + row * job->in_row);
                    x ^= p[0] ^ p[1] ^ p[2] ^ p[3];
                }
                volatile int4v lanes = x;
                const volatile int *w = (const volatile int *)&lanes;
                *(int *)(xo + tr * job->x_row + tc * 4) = (w[0] ^ w[1]) ^ (w[2] ^ w[3]);
            }
        }
    }
}

#if defined(TX_ASM_G4) && !defined(TX_XOR_C)
/* Hand-scheduled routine (8-row tiles); gen_tile_xor_asm.py documents the strip records. */
extern void tx_strips_g4(const unsigned *records);

#define RECORD_WORDS 24
#define MAX_STRIPS 16 /* strip records per call (bounds the records on the stack) */

/* Partials in groups of four: one strip record per tile column of this SHAVE's share and per group, all in one
 * call. Group k > 0 reads the accumulator back from acc_out, in place: the routine loads every word of a quarter
 * before it stores it. */
static void run_asm(const Job *job) {
    unsigned records[MAX_STRIPS * RECORD_WORDS];
    int4v dummy[8]; /* target of the first (garbage) accumulator store of each strip */
    const unsigned row0 = job->tile_row_begin * 8;
    const unsigned ntiles = job->tile_row_end - job->tile_row_begin;
    unsigned *a = records;
    unsigned count = 0;
    for (unsigned g0 = 0; g0 < job->g; g0 += 4) {
        const unsigned char *acc_in = g0 == 0 ? job->acc_in : job->acc_out;
        const unsigned acc_row = g0 == 0 ? job->acc_row : job->out_row;
        const unsigned acc_col = g0 == 0 ? job->acc_col : TILE_COLS * 4;
        for (unsigned tc = job->tile_col_begin; tc < job->tile_col_end; ++tc) {
            if (count == MAX_STRIPS) {
                a[-1] = 0;
                tx_strips_g4(records);
                a = records;
                count = 0;
            }
            const unsigned ai = (unsigned)acc_in + row0 * acc_row + tc * acc_col;
            *a++ = ai;
            *a++ = ai + acc_row;
            *a++ = 2 * acc_row - 64; /* the routine post-increments by 4 x 16 bytes per quarter */
            for (unsigned g = 0; g < 4; ++g) {
                const unsigned p = (unsigned)job->part[g0 + g] + row0 * job->in_row + tc * TILE_COLS * 4;
                *a++ = p;
                *a++ = p + job->in_row;
            }
            *a++ = 2 * job->in_row - 64;
            *a++ = (unsigned)&dummy[0];
            *a++ = (unsigned)&dummy[4];
            const unsigned ao = (unsigned)job->acc_out + row0 * job->out_row + tc * TILE_COLS * 4;
            *a++ = ao;
            *a++ = ao + job->out_row;
            *a++ = 2 * job->out_row;
            for (unsigned g = 0; g < 4; ++g) {
                *a++ = (unsigned)(job->xor_out + (g0 + g) * job->x_part + job->tile_row_begin * job->x_row + tc * 4);
            }
            *a++ = job->x_row;
            *a++ = ntiles;
            *a++ = 1; /* another record follows */
            ++count;
        }
    }
    a[-1] = 0;
    tx_strips_g4(records);
}

/* Single-pass routines for eight partials (8- and 16-row tiles); gen_tile_xor_asm.py (LayoutG8) documents the
 * record. */
extern void tx_strips_g8(const unsigned *records);
extern void tx_strips_g8_r16(const unsigned *records);

#define RECORD_WORDS_G8 24

static void run_asm_g8(const Job *job) {
    void (*const strips)(const unsigned *) = job->th == 16 ? tx_strips_g8_r16 : tx_strips_g8;
    unsigned records[MAX_STRIPS * RECORD_WORDS_G8];
    const unsigned row0 = job->tile_row_begin * job->th;
    const unsigned ntiles = job->tile_row_end - job->tile_row_begin;
    unsigned *a = records;
    unsigned count = 0;
    for (unsigned tc = job->tile_col_begin; tc < job->tile_col_end; ++tc) {
        if (count == MAX_STRIPS) {
            a[-1] = 0;
            strips(records);
            a = records;
            count = 0;
        }
        for (unsigned g = 0; g < 8; ++g) {
            *a++ = (unsigned)job->part[g] + row0 * job->in_row + tc * TILE_COLS * 4;
        }
        *a++ = 0;            /* off0: row 0 of the quarter being loaded */
        *a++ = job->in_row;  /* off1: row 1 */
        const unsigned ai = (unsigned)job->acc_in + row0 * job->acc_row + tc * job->acc_col;
        *a++ = ai;
        *a++ = ai + job->acc_row;
        *a++ = 2 * job->acc_row - 64; /* the routine post-increments by 4 x 16 bytes per quarter */
        const unsigned ao = (unsigned)job->acc_out + row0 * job->out_row + tc * TILE_COLS * 4;
        *a++ = ao;
        *a++ = ao + job->out_row;
        *a++ = 2 * job->out_row - 64;
        const unsigned xw = (unsigned)(job->xor_out + job->tile_row_begin * job->x_row + tc * 4);
        for (unsigned g = 0; g < 4; ++g) {
            *a++ = xw + g * job->x_part;
        }
        *a++ = 4 * job->x_part;
        *a++ = job->x_row - 4 * job->x_part;
        *a++ = job->th / 4 * ntiles; /* half tiles (4 rows each) */
        *a++ = 1;                    /* another record follows */
        ++count;
    }
    a[-1] = 0;
    strips(records);
}
#endif

void controlled_act(unsigned params) {
    const MemRef *r = (const MemRef *)params;

    const unsigned records = count_records(r);
    if (records == 0) {
        return;
    }
    const unsigned *args = (const unsigned *)&r[records]; /* int64 values, little endian */
    const unsigned acc_mode = args[0];
    const unsigned has_acc = acc_mode == 1 || acc_mode == 2;
    const unsigned num_partials = records - 2 - has_acc;
    if (num_partials == 0 || num_partials > MAX_PARTIALS) {
        return;
    }
    const MemRef *acc_out = &r[records - 2];
    const MemRef *xor_out = &r[records - 1];

    Job job;
    int4v zeros[4]; /* stored one by one: an initializer becomes a (slow, byte-wise) memset call */
    for (unsigned i = 0; i < 4; ++i) {
        ((volatile int4v *)zeros)[i] = (int4v)0;
    }
    job.g = num_partials;
    const int use_acc = has_acc && acc_mode != 2;
    job.acc_in = use_acc ? (const unsigned char *)r[0].data_lo : (const unsigned char *)zeros;
    job.acc_row = use_acc ? stride_bytes(&r[0], 1) : 0;
    job.acc_col = use_acc ? TILE_COLS * 4 : 0;
    for (unsigned g = 0; g < num_partials; ++g) {
        job.part[g] = (const unsigned char *)r[has_acc + g].data_lo;
    }
    job.acc_out = (unsigned char *)acc_out->data_lo;
    job.xor_out = (unsigned char *)xor_out->data_lo;
    job.in_row = stride_bytes(&r[has_acc], 1);
    job.out_row = stride_bytes(acc_out, 1);
    job.x_row = stride_bytes(xor_out, 1);

    const unsigned long long dims01 = *(const unsigned long long *)r[0].dims_addr; /* W, H in one load */
    const unsigned tile_rows = (unsigned)(*(const unsigned long long *)xor_out->dims_addr >> 32);
    job.th = tile_rows ? (unsigned)(dims01 >> 32) / tile_rows : 0;
    if (job.th != 8 && job.th != 16) {
        return;
    }
    job.x_part = (unsigned)dims01 / TILE_COLS * 4; /* the words of partial g follow those of g - 1 in a row */
    unsigned shave = 0, shaves = 1;
    if (args[2] == SHAVE_SPLIT_MAGIC && args[6] > 0 && args[4] < args[6]) {
        shave = args[4];
        shaves = args[6];
    }
    /* Split columns when they divide evenly: fewer, longer strips (each strip costs a few hundred cycles). */
    const unsigned tile_cols = (unsigned)dims01 / TILE_COLS;
    const int by_cols = tile_cols % shaves == 0;
    job.tile_row_begin = by_cols ? 0 : tile_rows * shave / shaves;
    job.tile_row_end = by_cols ? tile_rows : tile_rows * (shave + 1) / shaves;
    job.tile_col_begin = by_cols ? tile_cols * shave / shaves : 0;
    job.tile_col_end = by_cols ? tile_cols * (shave + 1) / shaves : tile_cols;

#if defined(TX_ASM_G4) && !defined(TX_XOR_C)
    if (acc_mode == 3) {
        if (job.th == 8) {
            run_xor_only_asm(&job);
        } else {
            run_xor_only_c(&job);
        }
        return;
    }
    /* Eight partials run in one pass (the accumulator stays in registers across all of them); other multiples of
     * four as G = 4 passes (8-row tiles only). Both need the partials to share one row stride, which the compiler's
     * identical per-cluster buffers do. */
    if (num_partials == 8 && !(TX_G8_OFF)) {
        run_asm_g8(&job);
        return;
    }
    if (num_partials % 4 == 0 && job.th == 8) {
        run_asm(&job);
        return;
    }
#endif
    if (acc_mode == 3) {
        run_xor_only_c(&job);
        return;
    }
    run_c(&job);
}
