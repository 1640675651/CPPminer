//! Pool-target plain_proof verify (zk-pow jackpot check without plonky2).

use anyhow::{ensure, Result};
use primitive_types::U256;
use zk_pow::api::proof::{IncompleteBlockHeader, MiningConfiguration, SeedDerivation};
use zk_pow::api::proof_utils::{compute_jackpot_hash, CompiledPublicParams};
use zk_pow::api::sanity_checks::{penalized_target_bound, PENALTY_BASE_RANK};
use zk_pow::circuit::chip::compute_jackpot;
use zk_pow::circuit::pearl_noise::compute_noise;
use zk_pow::ffi::plain_proof::{CertificateVersion, PlainProof};

/// Rank-penalized jackpot bound from an unscaled pool share target (BE U256).
///
/// Matches pearl `penalized_target_bound` / gateway `adjust_target`, and miner
/// `cp_scale_jackpot_target`: `target × h × w × (k/r) × 128`.
pub fn extract_difficulty_bound_from_pool_target(
    pool_target_be: &[u8; 32],
    config: &MiningConfiguration,
) -> Result<U256> {
    let rank = config.rank as usize;
    ensure!(
        rank >= PENALTY_BASE_RANK,
        "Rank must be >= {PENALTY_BASE_RANK} || r={rank}"
    );
    let base = U256::from_big_endian(pool_target_be);
    penalized_target_bound(base, config).ok_or_else(|| {
        anyhow::anyhow!(
            "no penalized target for pool share (degenerate config or target too easy); \
             rank={} h*w={} k_eff={}",
            rank,
            config.rows_pattern.size() as usize * config.cols_pattern.size() as usize,
            config.dot_product_length()
        )
    })
}

fn penalized_work_factor(config: &MiningConfiguration) -> usize {
    let tile = config.rows_pattern.size() as usize * config.cols_pattern.size() as usize;
    let rank = config.rank as usize;
    if rank == 0 {
        return 0;
    }
    tile * (config.dot_product_length() / rank) * PENALTY_BASE_RANK
}

/// Map stratum / template `cert_version` to seed derivation (1/2=legacy, 3=salted).
pub fn seed_derivation_from_cert_version(cert_version: u32) -> Result<SeedDerivation> {
    let version = CertificateVersion::try_from(cert_version)?;
    Ok(version.seed_derivation())
}

/// Verify plain_proof against pool share target (32-byte BE U256, unscaled).
///
/// `cert_version` selects noise-seed derivation (V3 = salted). Applies the
/// rank-penalty jackpot bound against the share target.
pub fn verify_plain_proof_with_pool_target(
    block_header: &IncompleteBlockHeader,
    plain_proof: &PlainProof,
    pool_target_be: &[u8; 32],
    cert_version: u32,
) -> Result<()> {
    let seed_derivation = seed_derivation_from_cert_version(cert_version)?;
    let (private_params, mut public_params) =
        plain_proof.parse_proof(*block_header, seed_derivation)?;
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

    let bound =
        extract_difficulty_bound_from_pool_target(pool_target_be, &public_params.mining_config)?;
    let hash_u = U256::from_little_endian(&public_params.hash_jackpot());
    ensure!(
        hash_u <= bound,
        "Jackpot condition not satisfied: hash does not meet the rank-penalized difficulty target"
    );
    Ok(())
}

/// Detailed jackpot failure message (for cp_proof_verify error buffer).
pub fn jackpot_verify_detail(
    block_header: &IncompleteBlockHeader,
    plain_proof: &PlainProof,
    pool_target_be: &[u8; 32],
    cert_version: u32,
) -> Result<String, String> {
    let seed_derivation = seed_derivation_from_cert_version(cert_version).map_err(|e| e.to_string())?;
    let (private, public) = plain_proof
        .parse_proof(*block_header, seed_derivation)
        .map_err(|e| e.to_string())?;
    let compiled = CompiledPublicParams::from(&public);
    let noise = compute_noise(&compiled);
    let jackpot = compute_jackpot(&compiled, &private.s_a, &private.s_b, &noise);
    let hash = compute_jackpot_hash(&jackpot, compiled.a_noise_seed());
    let bound = extract_difficulty_bound_from_pool_target(pool_target_be, &public.mining_config)
        .map_err(|e| e.to_string())?;
    let hash_u = U256::from_little_endian(&hash);
    let msg_hex: String = jackpot
        .iter()
        .map(|w| format!("{:08x}", w))
        .collect::<Vec<_>>()
        .join("");
    let factor = penalized_work_factor(&public.mining_config);
    let seed_label = match seed_derivation {
        SeedDerivation::Legacy => "legacy",
        SeedDerivation::Salted => "salted",
    };
    Ok(format!(
        "Jackpot hash 0x{:064x} exceeds rank-penalized bound 0x{:064x} \
         (cert_version={cert_version} seed={seed_label}; factor h*w*(k/r)*{}={factor}; \
         rank={}; recomputed msg={msg_hex})",
        hash_u,
        bound,
        PENALTY_BASE_RANK,
        public.mining_config.rank,
    ))
}
