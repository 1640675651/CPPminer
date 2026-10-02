//! Build plain_proof base64 for CPminer (stock pearl-blake3 + zk-pow-compatible bincode).

mod mining_config;
mod verify;
mod witness;

use std::collections::BTreeMap;
use std::sync::Mutex;

use base64::{engine::general_purpose::STANDARD, Engine as _};
use mining_config::{mining_config_bytes, validate_tile_anchor};
use pearl_blake3::{blake3_digest, MerkleProof};
use serde::{Deserialize, Serialize};
use verify::{jackpot_verify_detail, verify_plain_proof_with_pool_target};
use witness::{
    build_matrix_proof_in_place, build_matrix_proof_witness, needed_blocks, zero_subroots,
    MatrixWitness, BLOCK_BYTES,
};

/// BzMiner production hash tile (8x16 scattered cells within 128x256 period).
const SCATTERED_ROWS: [usize; 8] = [0, 8, 32, 40, 64, 72, 96, 104];
const SCATTERED_COLS: [usize; 16] = [
    0, 1, 32, 33, 64, 65, 96, 97, 128, 129, 160, 161, 192, 193, 224, 225,
];

/// CUTLASS Case 9 MMA lane tile (128x128 CTA). FragmentC maps to four 4x4 blocks
/// (row stride 16, col stride 32). Must match `MmaLaneTile128x128`.
const CUTLASS_ROWS: [usize; 8] = [0, 1, 2, 3, 16, 17, 18, 19];
const CUTLASS_COLS: [usize; 8] = [0, 1, 2, 3, 32, 33, 34, 35];

const CONTIGUOUS_16X16_ROWS: [usize; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

#[derive(Clone, Copy, PartialEq, Eq)]
enum TileLayout {
    Scattered = 0,
    Contiguous = 1,
    Cutlass = 2,
    Contiguous8x8 = 3,
    Contiguous4x8 = 4,
    Contiguous16x16 = 5,
}

impl TileLayout {
    fn from_i32(v: i32) -> Result<Self, String> {
        match v {
            0 => Ok(Self::Scattered),
            1 => Ok(Self::Contiguous),
            2 => Ok(Self::Cutlass),
            3 => Ok(Self::Contiguous8x8),
            4 => Ok(Self::Contiguous4x8),
            5 => Ok(Self::Contiguous16x16),
            _ => Err(format!("invalid tile_layout {v} (expected 0, 1, 2, 3, 4, or 5)")),
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
struct MatrixMerkleProof {
    proof: MerkleProof,
    row_indices: Vec<usize>,
}

#[derive(Clone, Serialize, Deserialize)]
struct PlainProof {
    m: usize,
    n: usize,
    k: usize,
    noise_rank: usize,
    a: MatrixMerkleProof,
    bt: MatrixMerkleProof,
}

fn job_key(header: &[u8], mining_config: &[u8]) -> [u8; 32] {
    let mut buf = Vec::with_capacity(header.len() + mining_config.len());
    buf.extend_from_slice(header);
    buf.extend_from_slice(mining_config);
    blake3_digest(&buf, None)
}

fn row_patterns(layout: TileLayout) -> (&'static [usize], &'static [usize]) {
    match layout {
        TileLayout::Scattered => (&SCATTERED_ROWS, &SCATTERED_COLS),
        TileLayout::Contiguous => (
            &[0, 1, 2, 3, 4, 5, 6, 7],
            &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        ),
        TileLayout::Contiguous8x8 => (
            &[0, 1, 2, 3, 4, 5, 6, 7],
            &[0, 1, 2, 3, 4, 5, 6, 7],
        ),
        TileLayout::Contiguous4x8 => (
            &[0, 1, 2, 3],
            &[0, 1, 2, 3, 4, 5, 6, 7],
        ),
        TileLayout::Contiguous16x16 => (
            &CONTIGUOUS_16X16_ROWS,
            &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        ),
        TileLayout::Cutlass => (&CUTLASS_ROWS, &CUTLASS_COLS),
    }
}

fn i8_as_bytes(data: &[i8]) -> &[u8] {
    // SAFETY: i8 and u8 have the same size and alignment; the pool commits the raw bytes.
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len()) }
}

fn build_matrix_proof(
    matrix: &[i8],
    rows: usize,
    cols: usize,
    job_key: [u8; 32],
    row_indices: &[usize],
) -> Result<MatrixMerkleProof, String> {
    Ok(MatrixMerkleProof {
        proof: build_matrix_proof_in_place(i8_as_bytes(matrix), rows, cols, job_key, row_indices)?,
        row_indices: row_indices.to_vec(),
    })
}

/// Validates the tile anchor and mining_config; returns (job_key, A rows, B^T rows) to prove.
fn proof_rows(
    header: &[u8],
    mining_config: &[u8],
    k: usize,
    rank: usize,
    t_rows: usize,
    t_cols: usize,
    layout: TileLayout,
) -> Result<([u8; 32], Vec<usize>, Vec<usize>), String> {
    if mining_config.len() != 52 {
        return Err(format!(
            "mining_config must be 52 bytes, got {}",
            mining_config.len()
        ));
    }

    let (rows_pat, cols_pat) = row_patterns(layout);
    let row_offsets: Vec<u32> = rows_pat.iter().map(|&o| o as u32).collect();
    let col_offsets: Vec<u32> = cols_pat.iter().map(|&o| o as u32).collect();
    validate_tile_anchor(
        &row_offsets,
        &col_offsets,
        t_rows as u32,
        t_cols as u32,
    )?;

    let expected_cfg =
        mining_config_bytes(k as u32, rank as u16, &row_offsets, &col_offsets)?;
    if expected_cfg != mining_config {
        return Err(
            "mining_config bytes do not match tile_layout row/col patterns".into(),
        );
    }

    let key = job_key(header, mining_config);
    let a_rows: Vec<usize> = rows_pat.iter().map(|o| t_rows + o).collect();
    let bt_rows: Vec<usize> = cols_pat.iter().map(|o| t_cols + o).collect();
    Ok((key, a_rows, bt_rows))
}

fn encode_plain_proof(pp: &PlainProof) -> Result<String, String> {
    let bytes = bincode::serialize(pp).map_err(|e| format!("bincode serialize: {e}"))?;
    Ok(STANDARD.encode(bytes))
}

struct ZeroSubroots {
    key: [u8; 32],
    rows: usize,
    cols: usize,
    subroots: Vec<[u8; 32]>,
}

/// One entry per job: every share of a job proves the same all-zero B^T.
static ZERO_SUBROOTS: Mutex<Option<ZeroSubroots>> = Mutex::new(None);

fn build_zero_matrix_proof(
    rows: usize,
    cols: usize,
    key: [u8; 32],
    row_indices: &[usize],
) -> Result<MatrixMerkleProof, String> {
    let mut cache = ZERO_SUBROOTS.lock().unwrap_or_else(|e| e.into_inner());
    let hit = matches!(&*cache, Some(c) if c.key == key && c.rows == rows && c.cols == cols);
    if !hit {
        *cache = Some(ZeroSubroots {
            key,
            rows,
            cols,
            subroots: zero_subroots(rows, cols, key),
        });
    }
    let witness = MatrixWitness {
        subroots: &cache.as_ref().unwrap().subroots,
        blocks: None,
        expected_root: None,
    };
    Ok(MatrixMerkleProof {
        proof: build_matrix_proof_witness(rows, cols, key, row_indices, &witness)?,
        row_indices: row_indices.to_vec(),
    })
}

/// `bt = None` proves an all-zero B^T without a host copy of it.
fn build_plain_proof_b64(
    header: &[u8],
    mining_config: &[u8],
    a: &[i8],
    bt: Option<&[i8]>,
    m: usize,
    n: usize,
    k: usize,
    rank: usize,
    t_rows: usize,
    t_cols: usize,
    layout: TileLayout,
) -> Result<String, String> {
    if a.len() != m * k {
        return Err(format!("A size mismatch: need {} got {}", m * k, a.len()));
    }
    if let Some(bt) = bt {
        if bt.len() != n * k {
            return Err(format!("B^T size mismatch: need {} got {}", n * k, bt.len()));
        }
    }
    let (key, a_rows, bt_rows) =
        proof_rows(header, mining_config, k, rank, t_rows, t_cols, layout)?;

    let a_proof = build_matrix_proof(a, m, k, key, &a_rows).map_err(|e| format!("A: {e}"))?;
    let bt_proof = match bt {
        Some(bt) => build_matrix_proof(bt, n, k, key, &bt_rows),
        None => build_zero_matrix_proof(n, k, key, &bt_rows),
    }
    .map_err(|e| format!("B^T: {e}"))?;
    encode_plain_proof(&PlainProof {
        m,
        n,
        k,
        noise_rank: rank,
        a: a_proof,
        bt: bt_proof,
    })
}

fn build_plain_proof_witness_b64(
    header: &[u8],
    mining_config: &[u8],
    a: &MatrixWitness,
    bt: &MatrixWitness,
    m: usize,
    n: usize,
    k: usize,
    rank: usize,
    t_rows: usize,
    t_cols: usize,
    layout: TileLayout,
) -> Result<String, String> {
    let (key, a_rows, bt_rows) =
        proof_rows(header, mining_config, k, rank, t_rows, t_cols, layout)?;
    let a_proof = build_matrix_proof_witness(m, k, key, &a_rows, a).map_err(|e| format!("A: {e}"))?;
    let bt_proof = if bt.blocks.is_none() && bt.subroots.is_empty() {
        let p = build_zero_matrix_proof(n, k, key, &bt_rows).map_err(|e| format!("B^T: {e}"))?;
        if matches!(bt.expected_root, Some(r) if r != p.proof.root) {
            return Err("B^T: zero-matrix root does not match device commitment".into());
        }
        p.proof
    } else {
        build_matrix_proof_witness(n, k, key, &bt_rows, bt).map_err(|e| format!("B^T: {e}"))?
    };

    encode_plain_proof(&PlainProof {
        m,
        n,
        k,
        noise_rank: rank,
        a: MatrixMerkleProof {
            proof: a_proof,
            row_indices: a_rows,
        },
        bt: MatrixMerkleProof {
            proof: bt_proof,
            row_indices: bt_rows,
        },
    })
}

fn write_err(out: Option<&mut [u8]>, msg: &str) {
    if let Some(buf) = out {
        let n = msg.len().min(buf.len().saturating_sub(1));
        buf[..n].copy_from_slice(&msg.as_bytes()[..n]);
        if !buf.is_empty() {
            buf[n.min(buf.len() - 1)] = 0;
        }
    }
}

/// Build plain_proof base64. Returns 0 on success, -1 on error.
///
/// `tile_layout`: 0 = BzMiner scattered 8x16, 1 = contiguous 8x16, 2 = CUTLASS Case 9 MMA 8x8,
/// 3 = contiguous 8x8, 4 = contiguous 4x8.
/// `mining_config` must be the 52-byte config used for GPU job_key (must match tile_layout).
/// `bt` may be null: B^T is then all zeros (zero-B mining), proven without a host copy.
#[no_mangle]
pub unsafe extern "C" fn cp_proof_build(
    header: *const u8,
    header_len: usize,
    mining_config: *const u8,
    config_len: usize,
    a: *const i8,
    bt: *const i8,
    m: i32,
    n: i32,
    k: i32,
    rank: i32,
    t_rows: i32,
    t_cols: i32,
    tile_layout: i32,
    out_b64: *mut u8,
    out_cap: usize,
    err: *mut u8,
    err_cap: usize,
) -> i32 {
    let err_slice = if err.is_null() || err_cap == 0 {
        None
    } else {
        Some(std::slice::from_raw_parts_mut(err, err_cap))
    };

    let fail = |msg: String| {
        write_err(err_slice, &msg);
        -1
    };

    if header.is_null() || mining_config.is_null() || a.is_null() || out_b64.is_null() {
        return fail("null pointer".into());
    }
    if m <= 0 || n <= 0 || k <= 0 || rank <= 0 {
        return fail("invalid dimensions".into());
    }
    let layout = match TileLayout::from_i32(tile_layout) {
        Ok(l) => l,
        Err(e) => return fail(e),
    };

    let m = m as usize;
    let n = n as usize;
    let k = k as usize;
    let rank = rank as usize;

    let header_slice = std::slice::from_raw_parts(header, header_len);
    let config_slice = std::slice::from_raw_parts(mining_config, config_len);
    let a_slice = std::slice::from_raw_parts(a, m * k);
    let bt_slice = if bt.is_null() {
        None
    } else {
        Some(std::slice::from_raw_parts(bt, n * k))
    };

    let b64 = match build_plain_proof_b64(
        header_slice,
        config_slice,
        a_slice,
        bt_slice,
        m,
        n,
        k,
        rank,
        t_rows as usize,
        t_cols as usize,
        layout,
    ) {
        Ok(s) => s,
        Err(e) => return fail(e),
    };

    if b64.len() >= out_cap {
        return fail(format!(
            "out_b64 too small: need {} bytes, cap {}",
            b64.len() + 1,
            out_cap
        ));
    }

    let out = std::slice::from_raw_parts_mut(out_b64, out_cap);
    out[..b64.len()].copy_from_slice(b64.as_bytes());
    out[b64.len()] = 0;
    0
}

/// C mirror: `CpMatrixWitness` in cp_proof.h.
#[repr(C)]
pub struct CpMatrixWitness {
    subroots: *const u8,
    num_subroots: usize,
    blocks: *const u8,
    block_idx: *const u32,
    num_blocks: usize,
    root: *const u8,
}

unsafe fn matrix_witness_from_c(w: &CpMatrixWitness) -> Result<MatrixWitness<'_>, String> {
    let subroots: &[[u8; 32]] = if w.subroots.is_null() || w.num_subroots == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(w.subroots as *const [u8; 32], w.num_subroots)
    };
    let blocks = if w.blocks.is_null() {
        None
    } else {
        if w.block_idx.is_null() || w.num_blocks == 0 {
            return Err("witness blocks without block indices".into());
        }
        let idx = std::slice::from_raw_parts(w.block_idx, w.num_blocks);
        let data = std::slice::from_raw_parts(w.blocks, w.num_blocks * BLOCK_BYTES);
        let map: BTreeMap<usize, &[u8]> = idx
            .iter()
            .enumerate()
            .map(|(i, &b)| (b as usize, &data[i * BLOCK_BYTES..(i + 1) * BLOCK_BYTES]))
            .collect();
        Some(map)
    };
    let expected_root = if w.root.is_null() {
        None
    } else {
        Some(*(w.root as *const [u8; 32]))
    };
    Ok(MatrixWitness {
        subroots,
        blocks,
        expected_root,
    })
}

/// Which `CP_WITNESS_BLOCK_BYTES` blocks of a matrix a proof reads.
/// `is_bt` = 0: A rows anchored at `anchor` (t_rows); 1: B^T rows anchored at t_cols.
/// Writes sorted block indices to `out_idx`; returns the count, or -1 on error / too small `cap`.
#[no_mangle]
pub unsafe extern "C" fn cp_proof_witness_blocks(
    tile_layout: i32,
    is_bt: i32,
    anchor: i32,
    rows: i32,
    k: i32,
    out_idx: *mut u32,
    cap: usize,
) -> i32 {
    let layout = match TileLayout::from_i32(tile_layout) {
        Ok(l) => l,
        Err(_) => return -1,
    };
    if anchor < 0 || rows <= 0 || k <= 0 || out_idx.is_null() {
        return -1;
    }
    let (rows_pat, cols_pat) = row_patterns(layout);
    let pat = if is_bt != 0 { cols_pat } else { rows_pat };
    let row_indices: Vec<usize> = pat.iter().map(|&o| anchor as usize + o).collect();
    if row_indices.iter().any(|&r| r >= rows as usize) {
        return -1;
    }
    let blocks = needed_blocks(rows as usize, k as usize, &row_indices);
    if blocks.len() > cap {
        return -1;
    }
    let out = std::slice::from_raw_parts_mut(out_idx, cap);
    for (dst, &b) in out.iter_mut().zip(&blocks) {
        *dst = b as u32;
    }
    blocks.len() as i32
}

/// Same output as `cp_proof_build`, but from device Merkle sub-roots plus only the blocks
/// listed by `cp_proof_witness_blocks` (or none, for an all-zero matrix).
#[no_mangle]
pub unsafe extern "C" fn cp_proof_build_witness(
    header: *const u8,
    header_len: usize,
    mining_config: *const u8,
    config_len: usize,
    a: *const CpMatrixWitness,
    bt: *const CpMatrixWitness,
    m: i32,
    n: i32,
    k: i32,
    rank: i32,
    t_rows: i32,
    t_cols: i32,
    tile_layout: i32,
    out_b64: *mut u8,
    out_cap: usize,
    err: *mut u8,
    err_cap: usize,
) -> i32 {
    let err_slice = if err.is_null() || err_cap == 0 {
        None
    } else {
        Some(std::slice::from_raw_parts_mut(err, err_cap))
    };

    let fail = |msg: String| {
        write_err(err_slice, &msg);
        -1
    };

    if header.is_null() || mining_config.is_null() || a.is_null() || bt.is_null() || out_b64.is_null() {
        return fail("null pointer".into());
    }
    if m <= 0 || n <= 0 || k <= 0 || rank <= 0 || t_rows < 0 || t_cols < 0 {
        return fail("invalid dimensions".into());
    }
    let layout = match TileLayout::from_i32(tile_layout) {
        Ok(l) => l,
        Err(e) => return fail(e),
    };
    let a_w = match matrix_witness_from_c(&*a) {
        Ok(w) => w,
        Err(e) => return fail(format!("A: {e}")),
    };
    let bt_w = match matrix_witness_from_c(&*bt) {
        Ok(w) => w,
        Err(e) => return fail(format!("B^T: {e}")),
    };

    let header_slice = std::slice::from_raw_parts(header, header_len);
    let config_slice = std::slice::from_raw_parts(mining_config, config_len);
    let b64 = match build_plain_proof_witness_b64(
        header_slice,
        config_slice,
        &a_w,
        &bt_w,
        m as usize,
        n as usize,
        k as usize,
        rank as usize,
        t_rows as usize,
        t_cols as usize,
        layout,
    ) {
        Ok(s) => s,
        Err(e) => return fail(e),
    };

    if b64.len() >= out_cap {
        return fail(format!(
            "out_b64 too small: need {} bytes, cap {}",
            b64.len() + 1,
            out_cap
        ));
    }

    let out = std::slice::from_raw_parts_mut(out_b64, out_cap);
    out[..b64.len()].copy_from_slice(b64.as_bytes());
    out[b64.len()] = 0;
    0
}

/// Verify plain_proof base64 against pool target (32-byte BE U256).
/// `cert_version`: 1/2 = legacy seeds, 3 = salted (V3). Returns 0 on success, -1 on error.
#[no_mangle]
pub unsafe extern "C" fn cp_proof_verify(
    header: *const u8,
    header_len: usize,
    proof_b64: *const u8,
    proof_b64_len: usize,
    pool_target_be: *const u8,
    cert_version: u32,
    err: *mut u8,
    err_cap: usize,
) -> i32 {
    use std::str;
    use zk_pow::api::proof::IncompleteBlockHeader;
    use zk_pow::ffi::plain_proof::PlainProof;

    let err_slice = if err.is_null() || err_cap == 0 {
        None
    } else {
        Some(std::slice::from_raw_parts_mut(err, err_cap))
    };

    let fail = |msg: String| {
        write_err(err_slice, &msg);
        -1
    };

    if header.is_null() || proof_b64.is_null() || pool_target_be.is_null() {
        return fail("null pointer".into());
    }
    if header_len != IncompleteBlockHeader::SERIALIZED_SIZE {
        return fail(format!(
            "header must be {} bytes, got {}",
            IncompleteBlockHeader::SERIALIZED_SIZE,
            header_len
        ));
    }

    let header_slice = std::slice::from_raw_parts(header, header_len);
    let block_header = match IncompleteBlockHeader::from_bytes(header_slice) {
        Ok(h) => h,
        Err(e) => return fail(format!("invalid header: {e}")),
    };

    let b64_slice = std::slice::from_raw_parts(proof_b64, proof_b64_len);
    let b64 = match str::from_utf8(b64_slice) {
        Ok(s) => s.trim(),
        Err(e) => return fail(format!("proof_b64 is not UTF-8: {e}")),
    };

    let raw = match STANDARD.decode(b64) {
        Ok(b) => b,
        Err(e) => return fail(format!("base64 decode: {e}")),
    };

    let plain_proof: PlainProof = match PlainProof::deserialize_compat(&raw) {
        Ok(p) => p,
        Err(e) => return fail(format!("bincode deserialize: {e}")),
    };

    let mut target_arr = [0u8; 32];
    std::ptr::copy_nonoverlapping(pool_target_be, target_arr.as_mut_ptr(), 32);
    match verify_plain_proof_with_pool_target(
        &block_header,
        &plain_proof,
        &target_arr,
        cert_version,
    ) {
        Ok(()) => 0,
        Err(e) => {
            let msg = e.to_string();
            if msg.contains("Jackpot condition not satisfied") {
                if let Ok(detail) =
                    jackpot_verify_detail(&block_header, &plain_proof, &target_arr, cert_version)
                {
                    return fail(detail);
                }
            }
            fail(msg)
        }
    }
}

/// gzip a base64 plain_proof for the Kryptex stratum v2 ("type":"v2") submit path:
/// base64-decode `in_b64`, gzip the raw bincode bytes (standard gzip stream, zlib wbits 31),
/// base64-encode the gzip stream into `out_b64`. Returns 0 on success, -1 on error.
fn gzip_b64(in_b64: &str, level: u32) -> Result<String, String> {
    use std::io::Write;
    let raw = STANDARD
        .decode(in_b64.trim())
        .map_err(|e| format!("base64 decode: {e}"))?;
    let mut enc = flate2::write::GzEncoder::new(
        Vec::with_capacity(raw.len() / 2 + 64),
        flate2::Compression::new(level),
    );
    enc.write_all(&raw).map_err(|e| format!("gzip: {e}"))?;
    let gz = enc.finish().map_err(|e| format!("gzip finish: {e}"))?;
    Ok(STANDARD.encode(gz))
}

#[no_mangle]
pub unsafe extern "C" fn cp_proof_gzip_b64(
    in_b64: *const u8,
    out_b64: *mut u8,
    out_cap: usize,
    err: *mut u8,
    err_cap: usize,
) -> i32 {
    let err_slice = if err.is_null() || err_cap == 0 {
        None
    } else {
        Some(std::slice::from_raw_parts_mut(err, err_cap))
    };
    let fail = |msg: String| {
        write_err(err_slice, &msg);
        -1
    };
    if in_b64.is_null() || out_b64.is_null() || out_cap == 0 {
        return fail("null pointer".into());
    }
    let input = match std::ffi::CStr::from_ptr(in_b64 as *const std::os::raw::c_char).to_str() {
        Ok(s) => s,
        Err(e) => return fail(format!("in_b64 is not UTF-8: {e}")),
    };
    let gz_b64 = match gzip_b64(input, flate2::Compression::default().level()) {
        Ok(s) => s,
        Err(e) => return fail(e),
    };
    if gz_b64.len() >= out_cap {
        return fail(format!(
            "out_b64 too small: need {} bytes, cap {}",
            gz_b64.len() + 1,
            out_cap
        ));
    }
    let out = std::slice::from_raw_parts_mut(out_b64, out_cap);
    out[..gz_b64.len()].copy_from_slice(gz_b64.as_bytes());
    out[gz_b64.len()] = 0;
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use pearl_blake3::{pad_to_chunk_boundary, MerkleTree};

    #[test]
    fn gzip_b64_round_trip_is_a_gzip_stream() {
        use std::io::Read;
        // Repetitive payload like a sparse A row + zero B^T strip.
        let mut raw = vec![0u8; 8192];
        raw[17] = 5;
        raw[4000] = 0xF3;
        let in_b64 = STANDARD.encode(&raw);
        let gz_b64 = gzip_b64(&in_b64, 6).unwrap();
        let gz = STANDARD.decode(&gz_b64).unwrap();
        assert_eq!(&gz[..2], &[0x1f, 0x8b], "gzip magic");
        assert_eq!(gz[2], 8, "deflate method");
        assert!(gz.len() < raw.len() / 20, "gzip did not compress: {} -> {}", raw.len(), gz.len());
        let mut back = Vec::new();
        flate2::read::GzDecoder::new(&gz[..]).read_to_end(&mut back).unwrap();
        assert_eq!(back, raw);

        // C entry point: NUL-terminated in, bounded out, error on short buffer.
        let cin = std::ffi::CString::new(in_b64.clone()).unwrap();
        let mut out = vec![0u8; gz_b64.len() + 1];
        let mut err = vec![0u8; 128];
        let rc = unsafe {
            cp_proof_gzip_b64(cin.as_ptr() as *const u8, out.as_mut_ptr(), out.len(),
                              err.as_mut_ptr(), err.len())
        };
        assert_eq!(rc, 0);
        assert_eq!(&out[..gz_b64.len()], gz_b64.as_bytes());
        assert_eq!(out[gz_b64.len()], 0);
        let rc = unsafe {
            cp_proof_gzip_b64(cin.as_ptr() as *const u8, out.as_mut_ptr(), 8,
                              err.as_mut_ptr(), err.len())
        };
        assert_eq!(rc, -1);
        assert!(err.starts_with(b"out_b64 too small"));
        let rc = unsafe {
            cp_proof_gzip_b64(b"not*base64\0".as_ptr(), out.as_mut_ptr(), out.len(),
                              err.as_mut_ptr(), err.len())
        };
        assert_eq!(rc, -1);
        assert!(err.starts_with(b"base64 decode"));
    }

    /// Stock pearl-blake3 full-tree proof: the reference every in-place / witness proof must match.
    fn reference_matrix_proof(
        matrix: &[i8],
        rows: usize,
        cols: usize,
        key: [u8; 32],
        row_indices: &[usize],
    ) -> MatrixMerkleProof {
        let flat: Vec<u8> = matrix.iter().map(|&x| x as u8).collect();
        let tree = MerkleTree::new(&pad_to_chunk_boundary(&flat), key);
        let leaf_indices = MerkleTree::compute_leaf_indices_from_rows(row_indices, (rows, cols));
        MatrixMerkleProof {
            proof: tree.get_multileaf_proof(&leaf_indices),
            row_indices: row_indices.to_vec(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn reference_proof_b64(
        header: &[u8],
        config: &[u8],
        a: &[i8],
        bt: &[i8],
        m: usize,
        n: usize,
        k: usize,
        rank: usize,
        t_rows: usize,
        t_cols: usize,
        layout: TileLayout,
    ) -> String {
        let (key, a_rows, bt_rows) =
            proof_rows(header, config, k, rank, t_rows, t_cols, layout).unwrap();
        encode_plain_proof(&PlainProof {
            m,
            n,
            k,
            noise_rank: rank,
            a: reference_matrix_proof(a, m, k, key, &a_rows),
            bt: reference_matrix_proof(bt, n, k, key, &bt_rows),
        })
        .unwrap()
    }

    #[test]
    fn in_place_matches_reference_unaligned_sizes() {
        let key = [5u8; 32];
        // rows * cols not a multiple of the chunk or block size: padded last chunk and block.
        for &(rows, cols, ref rows_to_prove) in &[
            (300usize, 1000usize, vec![0usize, 1, 150, 299]),
            (7, 100, vec![2, 6]),
            (1100, 257, vec![1000, 1099]),
        ] {
            let m = test_matrix(rows, cols, 11);
            let reference = reference_matrix_proof(&m, rows, cols, key, rows_to_prove);
            let got = build_matrix_proof(&m, rows, cols, key, rows_to_prove).unwrap();
            assert_eq!(
                bincode::serialize(&got).unwrap(),
                bincode::serialize(&reference).unwrap(),
                "rows={rows} cols={cols}"
            );
        }
    }

    #[test]
    fn fast_subroots_match_reference() {
        let key = [3u8; 32];
        let data: Vec<u8> = test_matrix(300, 1000, 17).iter().map(|&x| x as u8).collect();
        assert_eq!(
            witness::matrix_subroots(&data, key),
            witness::subroots_from_matrix(&data, key)
        );
        let zeros = vec![0u8; 300 * 1000];
        assert_eq!(
            zero_subroots(300, 1000, key),
            witness::subroots_from_matrix(&zeros, key)
        );
    }

    #[test]
    fn round_trip_bincode_header() {
        // Contiguous tile proves 8 A rows and 16 B^T rows.
        let m = 8;
        let n = 16;
        let k = 256;
        let a: Vec<i8> = (0..(m * k)).map(|i| (i % 127) as i8 - 64).collect();
        let bt: Vec<i8> = (0..(n * k)).map(|i| ((i * 3) % 127) as i8 - 64).collect();
        let header = [0u8; 76];
        let row_offsets: Vec<u32> = (0..8).map(|i| i as u32).collect();
        let col_offsets: Vec<u32> = (0..16).map(|i| i as u32).collect();
        let config = mining_config_bytes(256, 256, &row_offsets, &col_offsets).unwrap();
        let b64 = build_plain_proof_b64(
            &header,
            &config,
            &a,
            Some(&bt),
            m,
            n,
            k,
            256,
            0,
            0,
            TileLayout::Contiguous,
        )
        .expect("build");
        assert!(b64.len() > 64);
        let raw = STANDARD.decode(&b64).unwrap();
        let pp: PlainProof = bincode::deserialize(&raw).unwrap();
        assert_eq!(pp.m, m);
        assert_eq!(pp.k, k);
        assert_eq!(pp.a.row_indices.len(), 8);
        assert_eq!(pp.bt.row_indices.len(), 16);
    }

    #[test]
    fn contiguous_4x8_proof_row_counts() {
        let m = 128;
        let n = 128;
        let k = 256;
        let a: Vec<i8> = vec![0; m * k];
        let bt: Vec<i8> = vec![0; n * k];
        let header = [0u8; 76];
        let row_offsets: Vec<u32> = (0..4).map(|i| i as u32).collect();
        let col_offsets: Vec<u32> = (0..8).map(|i| i as u32).collect();
        let config = mining_config_bytes(256, 256, &row_offsets, &col_offsets).unwrap();
        let b64 = build_plain_proof_b64(
            &header,
            &config,
            &a,
            Some(&bt),
            m,
            n,
            k,
            256,
            0,
            0,
            TileLayout::Contiguous4x8,
        )
        .expect("4x8 build");
        let raw = STANDARD.decode(&b64).unwrap();
        let pp: PlainProof = bincode::deserialize(&raw).unwrap();
        assert_eq!(pp.a.row_indices.len(), 4);
        assert_eq!(pp.bt.row_indices.len(), 8);
        assert_eq!(pp.a.row_indices, vec![0, 1, 2, 3]);
        assert_eq!(pp.bt.row_indices, vec![0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn cutlass_proof_row_counts() {
        let m = 128;
        let n = 128;
        let k = 256;
        let a: Vec<i8> = vec![0; m * k];
        let bt: Vec<i8> = vec![0; n * k];
        let header = [0u8; 76];
        let row_offsets: Vec<u32> = CUTLASS_ROWS.iter().map(|&o| o as u32).collect();
        let col_offsets: Vec<u32> = CUTLASS_COLS.iter().map(|&o| o as u32).collect();
        let config = mining_config_bytes(256, 256, &row_offsets, &col_offsets).unwrap();
        let b64 = build_plain_proof_b64(
            &header,
            &config,
            &a,
            Some(&bt),
            m,
            n,
            k,
            256,
            8,
            16,
            TileLayout::Cutlass,
        )
        .expect("cutlass build");
        let raw = STANDARD.decode(&b64).unwrap();
        let pp: PlainProof = bincode::deserialize(&raw).unwrap();
        assert_eq!(pp.a.row_indices.len(), 8);
        assert_eq!(pp.bt.row_indices.len(), 8);
        assert_eq!(pp.a.row_indices[0], 8);
        assert_eq!(pp.bt.row_indices[0], 16);
    }

    #[test]
    fn build_then_verify_round_trip() {
        use zk_pow::api::proof::IncompleteBlockHeader;
        use zk_pow::ffi::plain_proof::PlainProof as ZkPlainProof;

        // zk-pow requires 16r <= k <= 4r^2 and r >= 128; use the production k/r.
        let m = 8;
        let n = 16;
        let k = 4096;
        let rank = 128;
        let a: Vec<i8> = (0..(m * k)).map(|i| (i % 127) as i8 - 64).collect();
        let bt: Vec<i8> = (0..(n * k)).map(|i| ((i * 3) % 127) as i8 - 64).collect();
        let header = [0u8; 76];
        let row_offsets: Vec<u32> = (0..8).map(|i| i as u32).collect();
        let col_offsets: Vec<u32> = (0..16).map(|i| i as u32).collect();
        let config =
            mining_config_bytes(k as u32, rank as u16, &row_offsets, &col_offsets).unwrap();
        let b64 = build_plain_proof_b64(
            &header,
            &config,
            &a,
            Some(&bt),
            m,
            n,
            k,
            rank,
            0,
            0,
            TileLayout::Contiguous,
        )
        .expect("build");

        let block_header = IncompleteBlockHeader::from_bytes(&header).unwrap();
        let raw = STANDARD.decode(&b64).unwrap();
        let pp: ZkPlainProof = ZkPlainProof::deserialize_compat(&raw).unwrap();
        // Near-max share target that still scales under the rank-penalized factor
        // h*w*(k/r)*128 = 8*16*32*128.
        let factor = primitive_types::U256::from(8u64 * 16 * (4096 / 128) * 128);
        let mut pool_target = [0u8; 32];
        (primitive_types::U256::MAX / factor).to_big_endian(&mut pool_target);
        verify_plain_proof_with_pool_target(&block_header, &pp, &pool_target, 2).expect("verify");
    }

    fn test_matrix(rows: usize, k: usize, mul: usize) -> Vec<i8> {
        (0..rows * k).map(|i| ((i * mul + i / 977) % 127) as i8 - 64).collect()
    }

    fn layout_config(layout: TileLayout, k: usize) -> [u8; 52] {
        let (rows_pat, cols_pat) = row_patterns(layout);
        let rows: Vec<u32> = rows_pat.iter().map(|&o| o as u32).collect();
        let cols: Vec<u32> = cols_pat.iter().map(|&o| o as u32).collect();
        mining_config_bytes(k as u32, 128, &rows, &cols).unwrap()
    }

    /// Blocks a device would download for these rows, zero-padded to BLOCK_BYTES.
    fn device_blocks(data: &[i8], rows: usize, k: usize, row_indices: &[usize]) -> Vec<(usize, Vec<u8>)> {
        needed_blocks(rows, k, row_indices)
            .into_iter()
            .map(|b| {
                let mut block = vec![0u8; BLOCK_BYTES];
                let start = (b * BLOCK_BYTES).min(data.len());
                let end = ((b + 1) * BLOCK_BYTES).min(data.len());
                for (dst, &src) in block.iter_mut().zip(&data[start..end]) {
                    *dst = src as u8;
                }
                (b, block)
            })
            .collect()
    }

    fn assert_witness_matches(
        m: usize,
        n: usize,
        k: usize,
        t_rows: usize,
        t_cols: usize,
        layout: TileLayout,
        zero_bt: bool,
    ) {
        let header: Vec<u8> = (0..76u8).collect();
        let config = layout_config(layout, k);
        let a = test_matrix(m, k, 7);
        let bt = if zero_bt { vec![0i8; n * k] } else { test_matrix(n, k, 13) };

        let reference =
            reference_proof_b64(&header, &config, &a, &bt, m, n, k, 128, t_rows, t_cols, layout);
        let in_place =
            build_plain_proof_b64(&header, &config, &a, Some(&bt), m, n, k, 128, t_rows, t_cols, layout)
                .expect("in-place proof");
        assert_eq!(in_place, reference);

        let key = job_key(&header, &config);
        let (rows_pat, cols_pat) = row_patterns(layout);
        let a_rows: Vec<usize> = rows_pat.iter().map(|o| t_rows + o).collect();
        let bt_rows: Vec<usize> = cols_pat.iter().map(|o| t_cols + o).collect();
        let a_bytes: Vec<u8> = a.iter().map(|&x| x as u8).collect();
        let bt_bytes: Vec<u8> = bt.iter().map(|&x| x as u8).collect();

        let a_sub = witness::subroots_from_matrix(&a_bytes, key);
        let bt_sub = witness::subroots_from_matrix(&bt_bytes, key);
        let a_blocks = device_blocks(&a, m, k, &a_rows);
        let bt_blocks = device_blocks(&bt, n, k, &bt_rows);
        let a_root = MerkleTree::new(&pad_to_chunk_boundary(&a_bytes), key).root();
        let bt_root = MerkleTree::new(&pad_to_chunk_boundary(&bt_bytes), key).root();

        let a_w = MatrixWitness {
            subroots: &a_sub,
            blocks: Some(a_blocks.iter().map(|(b, d)| (*b, d.as_slice())).collect()),
            expected_root: Some(a_root),
        };
        let bt_w = MatrixWitness {
            subroots: &bt_sub,
            blocks: if zero_bt {
                None
            } else {
                Some(bt_blocks.iter().map(|(b, d)| (*b, d.as_slice())).collect())
            },
            expected_root: Some(bt_root),
        };
        let got = build_plain_proof_witness_b64(
            &header, &config, &a_w, &bt_w, m, n, k, 128, t_rows, t_cols, layout,
        )
        .expect("witness proof");
        assert_eq!(got, reference);

        if zero_bt {
            let bt_empty = MatrixWitness {
                subroots: &[],
                blocks: None,
                expected_root: Some(bt_root),
            };
            let got = build_plain_proof_witness_b64(
                &header, &config, &a_w, &bt_empty, m, n, k, 128, t_rows, t_cols, layout,
            )
            .expect("witness proof, host zero B^T");
            assert_eq!(got, reference);
        }
    }

    #[test]
    fn witness_matches_full_proof_partial_last_block() {
        // 300 rows x 4 KiB = 1200 chunks: four full 256-chunk blocks plus a 176-chunk tail.
        assert_witness_matches(300, 64, 4096, 288, 16, TileLayout::Contiguous, false);
        assert_witness_matches(300, 64, 4096, 0, 0, TileLayout::Contiguous8x8, false);
    }

    #[test]
    fn witness_matches_full_proof_scattered_multi_block() {
        assert_witness_matches(300, 600, 4096, 128, 256, TileLayout::Scattered, false);
    }

    #[test]
    fn witness_matches_full_proof_zero_bt() {
        assert_witness_matches(300, 600, 4096, 128, 256, TileLayout::Scattered, true);
        assert_witness_matches(256, 128, 4096, 64, 64, TileLayout::Contiguous4x8, true);
    }

    #[test]
    fn witness_matches_full_proof_single_block() {
        assert_witness_matches(16, 32, 4096, 8, 16, TileLayout::Contiguous, false);
    }

    fn assert_null_bt_matches(
        m: usize,
        n: usize,
        t_rows: usize,
        t_cols: usize,
        layout: TileLayout,
        header: &[u8],
    ) {
        let k = 4096;
        let config = layout_config(layout, k);
        let a = test_matrix(m, k, 7);
        let zeros = vec![0i8; n * k];
        let reference =
            reference_proof_b64(header, &config, &a, &zeros, m, n, k, 128, t_rows, t_cols, layout);
        let got =
            build_plain_proof_b64(header, &config, &a, None, m, n, k, 128, t_rows, t_cols, layout)
                .expect("null-B^T proof");
        assert_eq!(got, reference);
    }

    #[test]
    fn null_bt_matches_zero_matrix() {
        let h1: Vec<u8> = (0..76u8).collect();
        let h2: Vec<u8> = (0..76u8).rev().collect();
        // Multi-block with a partial tail, then a cache hit, then a new job key.
        assert_null_bt_matches(128, 600, 0, 256, TileLayout::Scattered, &h1);
        assert_null_bt_matches(128, 600, 8, 64, TileLayout::Contiguous, &h1);
        assert_null_bt_matches(128, 600, 0, 256, TileLayout::Scattered, &h2);
        // Single block.
        assert_null_bt_matches(16, 32, 8, 16, TileLayout::Contiguous, &h1);
    }

    #[test]
    fn witness_rejects_corrupt_subroot() {
        let (m, k) = (300usize, 4096usize);
        let key = [9u8; 32];
        let a = test_matrix(m, k, 7);
        let a_bytes: Vec<u8> = a.iter().map(|&x| x as u8).collect();
        let rows: Vec<usize> = (0..8).collect();
        let mut sub = witness::subroots_from_matrix(&a_bytes, key);
        sub[0][0] ^= 1;
        let blocks = device_blocks(&a, m, k, &rows);
        let w = MatrixWitness {
            subroots: &sub,
            blocks: Some(blocks.iter().map(|(b, d)| (*b, d.as_slice())).collect()),
            expected_root: None,
        };
        assert!(build_matrix_proof_witness(m, k, key, &rows, &w).is_err());
    }
}
