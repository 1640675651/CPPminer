//! Build plain_proof base64 for CPminer (stock pearl-blake3 + zk-pow-compatible bincode).

use base64::{engine::general_purpose::STANDARD, Engine as _};
use pearl_blake3::{blake3_digest, pad_to_chunk_boundary, MerkleProof, MerkleTree};
use serde::{Deserialize, Serialize};

const SCATTERED_ROWS: [usize; 8] = [0, 8, 32, 40, 64, 72, 96, 104];
const SCATTERED_COLS: [usize; 16] = [
    0, 1, 32, 33, 64, 65, 96, 97, 128, 129, 160, 161, 192, 193, 224, 225,
];

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

fn row_patterns(contiguous: bool) -> (&'static [usize], &'static [usize]) {
    if contiguous {
        (&[0, 1, 2, 3, 4, 5, 6, 7], &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    } else {
        (&SCATTERED_ROWS, &SCATTERED_COLS)
    }
}

fn flatten_i8_row_major(data: &[i8], rows: usize, cols: usize) -> Vec<u8> {
    debug_assert_eq!(data.len(), rows * cols);
    data.iter().map(|&x| x as u8).collect()
}

fn build_matrix_proof(
    matrix: &[i8],
    rows: usize,
    cols: usize,
    job_key: [u8; 32],
    row_indices: &[usize],
) -> MatrixMerkleProof {
    let flat = flatten_i8_row_major(matrix, rows, cols);
    let padded = pad_to_chunk_boundary(&flat);
    let tree = MerkleTree::new(&padded, job_key);
    let leaf_indices = MerkleTree::compute_leaf_indices_from_rows(row_indices, (rows, cols));
    MatrixMerkleProof {
        proof: tree.get_multileaf_proof(&leaf_indices),
        row_indices: row_indices.to_vec(),
    }
}

fn build_plain_proof_b64(
    header: &[u8],
    mining_config: &[u8],
    a: &[i8],
    bt: &[i8],
    m: usize,
    n: usize,
    k: usize,
    rank: usize,
    t_rows: usize,
    t_cols: usize,
    contiguous: bool,
) -> Result<String, String> {
    if mining_config.len() != 52 {
        return Err(format!(
            "mining_config must be 52 bytes, got {}",
            mining_config.len()
        ));
    }
    if a.len() != m * k {
        return Err(format!("A size mismatch: need {} got {}", m * k, a.len()));
    }
    if bt.len() != n * k {
        return Err(format!("B^T size mismatch: need {} got {}", n * k, bt.len()));
    }

    let key = job_key(header, mining_config);
    let (rows_pat, cols_pat) = row_patterns(contiguous);
    let a_rows: Vec<usize> = rows_pat.iter().map(|o| t_rows + o).collect();
    let bt_rows: Vec<usize> = cols_pat.iter().map(|o| t_cols + o).collect();

    let pp = PlainProof {
        m,
        n,
        k,
        noise_rank: rank,
        a: build_matrix_proof(a, m, k, key, &a_rows),
        bt: build_matrix_proof(bt, n, k, key, &bt_rows),
    };

    let bytes = bincode::serialize(&pp).map_err(|e| format!("bincode serialize: {e}"))?;
    Ok(STANDARD.encode(bytes))
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
/// `mining_config` must be 52 bytes (scattered or contiguous MiningConfiguration.to_bytes()).
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
    contiguous_tiles: i32,
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
    if m <= 0 || n <= 0 || k <= 0 || rank <= 0 {
        return fail("invalid dimensions".into());
    }
    let m = m as usize;
    let n = n as usize;
    let k = k as usize;
    let rank = rank as usize;

    let header_slice = std::slice::from_raw_parts(header, header_len);
    let config_slice = std::slice::from_raw_parts(mining_config, config_len);
    let a_slice = std::slice::from_raw_parts(a, m * k);
    let bt_slice = std::slice::from_raw_parts(bt, n * k);

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
        contiguous_tiles != 0,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_bincode_header() {
        let m = 4;
        let n = 4;
        let k = 256;
        let a: Vec<i8> = (0..(m * k)).map(|i| (i % 127) as i8 - 64).collect();
        let bt: Vec<i8> = (0..(n * k)).map(|i| ((i * 3) % 127) as i8 - 64).collect();
        let header = [0u8; 76];
        let config = [0u8; 52];
        let b64 = build_plain_proof_b64(&header, &config, &a, &bt, m, n, k, 256, 0, 0, true)
            .expect("build");
        assert!(b64.len() > 64);
        let raw = STANDARD.decode(&b64).unwrap();
        let pp: PlainProof = bincode::deserialize(&raw).unwrap();
        assert_eq!(pp.m, m);
        assert_eq!(pp.k, k);
        assert_eq!(pp.a.row_indices.len(), 8);
        assert_eq!(pp.bt.row_indices.len(), 16);
    }
}
