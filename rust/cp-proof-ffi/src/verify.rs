//! Pool-target plain_proof verify (zk-pow jackpot check without plonky2).

use anyhow::{ensure, Result};
use primitive_types::U256;
use zk_pow::api::proof::{IncompleteBlockHeader, MiningConfiguration};
use zk_pow::api::proof_utils::{compute_jackpot_hash, CompiledPublicParams};
use zk_pow::circuit::chip::compute_jackpot;
use zk_pow::circuit::pearl_noise::compute_noise;
use zk_pow::ffi::plain_proof::PlainProof;

/// Scale an unscaled pool target (BE U256) by h×w×k_eff, matching miner `cp_scale_jackpot_target`.
pub fn extract_difficulty_bound_from_pool_target(
    pool_target_be: &[u8; 32],
    config: &MiningConfiguration,
) -> U256 {
    let base = U256::from_big_endian(pool_target_be);
    let factor = (config.rows_pattern.size() as usize)
        * (config.cols_pattern.size() as usize)
        * config.dot_product_length();
    let factor_u = U256::from(factor as u64);
    if factor == 0 || base > U256::MAX / factor_u {
        U256::MAX
    } else {
        base * factor_u
    }
}

/// Verify plain_proof against pool share target (32-byte BE U256, unscaled).
pub fn verify_plain_proof_with_pool_target(
    block_header: &IncompleteBlockHeader,
    plain_proof: &PlainProof,
    pool_target_be: &[u8; 32],
) -> Result<()> {
    let (private_params, mut public_params) = plain_proof.parse_proof(*block_header)?;
    public_params.sanity_check()?;

    for strip in private_params.s_a.iter().chain(private_params.s_b.iter()) {
        for &val in strip {
            ensure!(
                (-64..=64).contains(&val),
                "Matrix value {} out of range [-64, 64]",
                val
            );
        }
    }

    let compiled = CompiledPublicParams::from(&public_params);
    let noise = compute_noise(&compiled);
    let jackpot = compute_jackpot(&compiled, &private_params.s_a, &private_params.s_b, &noise);
    public_params.hash_jackpot = compute_jackpot_hash(&jackpot, compiled.a_noise_seed());

    let bound = extract_difficulty_bound_from_pool_target(
        pool_target_be,
        &public_params.mining_config,
    );
    let hash_u = U256::from_little_endian(&public_params.hash_jackpot());
    ensure!(
        hash_u <= bound,
        "Jackpot condition not satisfied: hash does not meet pool target"
    );
    Ok(())
}

/// Detailed jackpot failure message (for cp_proof_verify error buffer).
pub fn jackpot_verify_detail(
    block_header: &IncompleteBlockHeader,
    plain_proof: &PlainProof,
    pool_target_be: &[u8; 32],
) -> Result<String, String> {
    let (private, public) = plain_proof
        .parse_proof(*block_header)
        .map_err(|e| e.to_string())?;
    let compiled = CompiledPublicParams::from(&public);
    let noise = compute_noise(&compiled);
    let jackpot = compute_jackpot(&compiled, &private.s_a, &private.s_b, &noise);
    let hash = compute_jackpot_hash(&jackpot, compiled.a_noise_seed());
    let bound = extract_difficulty_bound_from_pool_target(pool_target_be, &public.mining_config);
    let hash_u = U256::from_little_endian(&hash);
    let msg_hex: String = jackpot
        .iter()
        .map(|w| format!("{:08x}", w))
        .collect::<Vec<_>>()
        .join("");
    Ok(format!(
        "Jackpot hash 0x{:064x} exceeds bound 0x{:064x} (scaled work factor h*w*k_eff={}; recomputed msg={msg_hex})",
        hash_u,
        bound,
        (public.mining_config.rows_pattern.size() as usize)
            * (public.mining_config.cols_pattern.size() as usize)
            * public.dot_product_length(),
    ))
}
