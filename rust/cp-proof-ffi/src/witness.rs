//! Matrix Merkle proofs from GPU sub-roots plus only the blocks covering the proven rows.
//!
//! GPU hashing folds each run of `SUBTREE_CHUNKS` chunk CVs into one sub-root. Those runs are
//! aligned, power-of-two subtrees of the pearl-blake3 tree, so the full sub-root list plus the
//! raw bytes of the blocks holding the proven rows are enough to rebuild the exact
//! `MerkleProof` that `MerkleTree::get_multileaf_proof` would produce from the whole matrix.

use std::collections::{BTreeMap, BTreeSet};

use blake3::{CHUNK_LEN, OUT_LEN};
use pearl_blake3::{padded_chunk_len, Blake3Hasher, MerkleProof, MerkleTree};

type Digest = [u8; OUT_LEN];

pub const SUBTREE_LEVELS: usize = 8;
pub const SUBTREE_CHUNKS: usize = 1 << SUBTREE_LEVELS;
pub const BLOCK_BYTES: usize = SUBTREE_CHUNKS * CHUNK_LEN;

pub struct MatrixWitness<'a> {
    /// One CV per `SUBTREE_CHUNKS` chunks (last one may cover fewer). Unused for single-block matrices.
    pub subroots: &'a [Digest],
    /// Block index -> `BLOCK_BYTES` of row-major data (zero-padded). `None` = all-zero matrix.
    pub blocks: Option<BTreeMap<usize, &'a [u8]>>,
    /// Root the device used for its commitment; checked against the rebuilt proof when set.
    pub expected_root: Option<Digest>,
}

/// Sorted block indices whose bytes are needed to prove `row_indices` of a `rows x cols` matrix.
pub fn needed_blocks(rows: usize, cols: usize, row_indices: &[usize]) -> Vec<usize> {
    MerkleTree::compute_leaf_indices_from_rows(row_indices, (rows, cols))
        .iter()
        .map(|&leaf| leaf / SUBTREE_CHUNKS)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn combine_layer(hasher: &Blake3Hasher, prev: &[Digest]) -> Vec<Digest> {
    prev.chunks(2)
        .map(|pair| {
            if pair.len() == 2 {
                hasher.parent_cv(&pair[0], &pair[1])
            } else {
                pair[0]
            }
        })
        .collect()
}

/// Layers 0..=SUBTREE_LEVELS of one block's subtree; the last layer holds its sub-root.
fn block_layers(
    hasher: &Blake3Hasher,
    data: &[u8],
    block: usize,
    leaves_in_block: usize,
) -> Vec<Vec<Digest>> {
    let base = block * SUBTREE_CHUNKS;
    let leaves: Vec<Digest> = (0..leaves_in_block)
        .map(|i| hasher.chunk_cv(&data[i * CHUNK_LEN..(i + 1) * CHUNK_LEN], (base + i) as u64))
        .collect();
    let mut layers = vec![leaves];
    for _ in 0..SUBTREE_LEVELS {
        let next = combine_layer(hasher, layers.last().unwrap());
        layers.push(next);
    }
    layers
}

fn block_slice<'w>(
    witness: &'w MatrixWitness<'_>,
    zero_block: &'w [u8],
    b: usize,
) -> Result<&'w [u8], String> {
    let data = match &witness.blocks {
        None => zero_block,
        Some(map) => map
            .get(&b)
            .copied()
            .ok_or_else(|| format!("witness is missing block {b}"))?,
    };
    if data.len() < BLOCK_BYTES {
        return Err(format!("witness block {b} is {} bytes, need {BLOCK_BYTES}", data.len()));
    }
    Ok(data)
}

fn check_root(proof: &MerkleProof, expected: Option<Digest>) -> Result<(), String> {
    match expected {
        Some(r) if r != proof.root => {
            Err("rebuilt Merkle root does not match device commitment".into())
        }
        _ => Ok(()),
    }
}

pub fn build_matrix_proof_witness(
    rows: usize,
    cols: usize,
    key: Digest,
    row_indices: &[usize],
    witness: &MatrixWitness,
) -> Result<MerkleProof, String> {
    let pad_len = padded_chunk_len(rows * cols);
    let total_leaves = pad_len / CHUNK_LEN;
    let num_blocks = total_leaves.div_ceil(SUBTREE_CHUNKS);
    if row_indices.iter().any(|&r| r >= rows) {
        return Err(format!("proof row index out of range (rows={rows})"));
    }
    let leaf_indices = MerkleTree::compute_leaf_indices_from_rows(row_indices, (rows, cols));
    if leaf_indices.is_empty() {
        return Err("no leaves to prove".into());
    }

    let zero_block = if witness.blocks.is_none() {
        vec![0u8; BLOCK_BYTES]
    } else {
        Vec::new()
    };
    let block_data = |b: usize| block_slice(witness, &zero_block, b);

    if num_blocks == 1 {
        let tree = MerkleTree::new(&block_data(0)?[..pad_len], key);
        let proof = tree.get_multileaf_proof(&leaf_indices);
        check_root(&proof, witness.expected_root)?;
        return Ok(proof);
    }

    if witness.subroots.len() != num_blocks {
        return Err(format!(
            "witness has {} sub-roots, matrix needs {num_blocks}",
            witness.subroots.len()
        ));
    }

    let hasher = Blake3Hasher::with_key(key);
    let blocks_needed: BTreeSet<usize> =
        leaf_indices.iter().map(|&l| l / SUBTREE_CHUNKS).collect();
    let mut low: BTreeMap<usize, Vec<Vec<Digest>>> = BTreeMap::new();
    for &b in &blocks_needed {
        let leaves_in_block = (total_leaves - b * SUBTREE_CHUNKS).min(SUBTREE_CHUNKS);
        let layers = block_layers(&hasher, block_data(b)?, b, leaves_in_block);
        if layers[SUBTREE_LEVELS][0] != witness.subroots[b] {
            return Err(format!("block {b} does not hash to its device sub-root"));
        }
        low.insert(b, layers);
    }

    // num_blocks >= 2, so every level below the sub-roots has > 2 nodes and the tree
    // above them ends in exactly two nodes, matching MerkleTree::new's layer loop.
    let mut high = vec![witness.subroots.to_vec()];
    while high.last().unwrap().len() > 2 {
        let next = combine_layer(&hasher, high.last().unwrap());
        high.push(next);
    }
    let top = high.last().unwrap();
    let root = hasher.root_cv(&top[0], &top[1]);

    let node = |level: usize, idx: usize| -> Result<Digest, String> {
        if level < SUBTREE_LEVELS {
            let shift = SUBTREE_LEVELS - level;
            let b = idx >> shift;
            low.get(&b)
                .and_then(|layers| layers[level].get(idx - (b << shift)))
                .copied()
                .ok_or_else(|| format!("sibling at level {level} needs block {b}"))
        } else {
            high.get(level - SUBTREE_LEVELS)
                .and_then(|layer| layer.get(idx))
                .copied()
                .ok_or_else(|| format!("sibling {idx} missing at level {level}"))
        }
    };

    let unique: BTreeSet<usize> = leaf_indices.iter().copied().collect();
    let mut siblings = Vec::new();
    let mut current = unique.clone();
    let mut level_len = total_leaves;
    let mut level = 0;
    while level_len > 1 && !current.is_empty() {
        for &i in &current {
            if i % 2 == 1 {
                if !current.contains(&(i - 1)) {
                    siblings.push(node(level, i - 1)?);
                }
            } else if !current.contains(&(i + 1)) && (i + 1) < level_len {
                siblings.push(node(level, i + 1)?);
            }
        }
        current = current.iter().map(|&i| i / 2).collect();
        level_len = level_len.div_ceil(2);
        level += 1;
    }

    let mut leaf_data = Vec::with_capacity(unique.len());
    for &leaf in &unique {
        let data = block_data(leaf / SUBTREE_CHUNKS)?;
        let off = (leaf % SUBTREE_CHUNKS) * CHUNK_LEN;
        let mut chunk = [0u8; CHUNK_LEN];
        chunk.copy_from_slice(&data[off..off + CHUNK_LEN]);
        leaf_data.push(chunk);
    }

    let proof = MerkleProof {
        leaf_data,
        leaf_indices: unique.into_iter().collect(),
        total_leaves,
        root,
        siblings,
    };
    check_root(&proof, witness.expected_root)?;
    Ok(proof)
}

/// Sub-roots of a full matrix, as the GPU chunk-roots kernel emits them.
#[cfg(test)]
pub fn subroots_from_matrix(data: &[u8], key: Digest) -> Vec<Digest> {
    let hasher = Blake3Hasher::with_key(key);
    let pad_len = padded_chunk_len(data.len());
    let total_leaves = pad_len / CHUNK_LEN;
    let mut padded = data.to_vec();
    padded.resize(total_leaves.div_ceil(SUBTREE_CHUNKS) * BLOCK_BYTES, 0);
    (0..total_leaves.div_ceil(SUBTREE_CHUNKS))
        .map(|b| {
            let leaves_in_block = (total_leaves - b * SUBTREE_CHUNKS).min(SUBTREE_CHUNKS);
            let block = &padded[b * BLOCK_BYTES..(b + 1) * BLOCK_BYTES];
            block_layers(&hasher, block, b, leaves_in_block)[SUBTREE_LEVELS][0]
        })
        .collect()
}
