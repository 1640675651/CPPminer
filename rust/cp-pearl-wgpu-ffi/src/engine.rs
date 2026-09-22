//! Pearl wgpu mining engine: fused prep + GEMM/XOR/jackpot scan.

use bytemuck::{Pod, Zeroable};
use std::sync::atomic::{AtomicU8, Ordering};

// Fixed tiling (MR=NR=8, MACRO=128, KR=128, K=4096).
const MR: i32 = 8;
const NR: i32 = 8;
const MACRO: i32 = 128;
const KR: i32 = 128;
const R_RANK: i32 = 128;
const MICRO: i32 = 16;
const BLOCKS_K: i32 = 32;
const NUM_MILESTONES: i32 = 32;
const MACRO_KB_BLOCK: u64 = 16384; // bytes per macro x kb panel
const HASH_TILES_PER_MACRO: u64 = (MICRO as u64) * (MICRO as u64); // 256
const B3_CHUNK: u64 = 1024;

const PREP_WGSL_RAW: &str =
    include_str!("../../../src/pearl/wgpu/kernels/pearl_prep.wgsl");
const GEMM_WGSL: &str =
    include_str!("../../../src/pearl/wgpu/kernels/pearl_gemm_xor.wgsl");

fn prep_wgsl() -> &'static str {
    PREP_WGSL_RAW
}

// Domain salt: blake3("pearl/cert-v3/noise-seed/A") - pinned in pearl seed.rs / cp_noise.c
const PEARL_SEED_SALT_A: [u8; 32] = [
    0x82, 0x49, 0x40, 0x6c, 0xa0, 0xed, 0x15, 0x16, 0x96, 0x16, 0xf6, 0x92, 0xfc, 0xf0, 0x76, 0xf8,
    0x92, 0xdb, 0xdb, 0x2a, 0x70, 0x23, 0xb8, 0x52, 0xf0, 0xd4, 0x77, 0x19, 0xc3, 0x90, 0x01, 0x7b,
];

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PearlGenRandomParams {
    rng_seed: [u32; 2],
    matrix_tag: i32,
    total_elems: i32,
    wg_x: i32,
    word_begin: i32,
    _pad1: i32,
    _pad2: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PearlBuildPairsParams {
    is_b: i32,
    k: i32,
    rank: i32,
    _pad: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PearlPrepackBParams {
    n: i32,
    k: i32,
    rank: i32,
    blocks_k: i32,
    macro_cols: i32,
    has_signal: i32,
    wg_x: i32,
    g_begin: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PearlPrepackAParams {
    m: i32,
    k: i32,
    rank: i32,
    blocks_k: i32,
    macro_rows: i32,
    wg_x: i32,
    g_begin: i32,
    _pad1: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PearlMerkleChunkParams {
    raw_len: u32,
    pad_len: u32,
    num_chunks: i32,
    _pad: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PearlMerkleMtParams {
    num_leaves: i32,
    is_single_block: i32,
    _pad0: i32,
    _pad1: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PearlReduceRootsParams {
    num_leaves: i32,
    _pad0: i32,
    _pad1: i32,
    _pad2: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PearlScanParams {
    n: i32,
    blocks_k: i32,
    num_milestones: i32,
    tile_count: i32,
    macro_rows: i32,
    macro_cols: i32,
    mb_begin: i32,
    micro_m_begin: i32,
    micro_m_count: i32,
    wg_x: i32,
    batch_count: i32,
    _pad2: i32,
}

/// Split a 1D workgroup count into a 2D grid within the wgpu/Vulkan limit (65535 per dim).
fn dispatch_2d(num_wg: u32) -> (u32, u32, i32) {
    const MAX_WG: u32 = 65535;
    let n = num_wg.max(1);
    let wg_x = n.min(MAX_WG);
    let wg_y = ((n + wg_x - 1) / wg_x).max(1);
    (wg_x, wg_y, wg_x as i32)
}

/// Max workgroups per submit 鈥?Windows TDR is ~2s; iGPU prepack must stay under that.
fn max_wg_per_submit(device_type: wgpu::DeviceType) -> u32 {
    match device_type {
        wgpu::DeviceType::IntegratedGpu | wgpu::DeviceType::Cpu => 2048,
        _ => 16384,
    }
}

fn seed_to_u64(seed: &[u8]) -> u64 {
    let mut s = 0u64;
    for (i, &b) in seed.iter().enumerate() {
        s ^= (b as u64) << ((i & 7) * 8);
    }
    s
}

fn blake3_digest(data: &[u8], key: Option<&[u8; 32]>) -> [u8; 32] {
    let mut out = [0u8; 32];
    match key {
        Some(k) => {
            let mut h = blake3::Hasher::new_keyed(k);
            h.update(data);
            out.copy_from_slice(h.finalize().as_bytes());
        }
        None => {
            out.copy_from_slice(blake3::hash(data).as_bytes());
        }
    }
    out
}

fn pearl_bind_message(root: &[u8; 32], dim: u32) -> [u8; 64] {
    let mut msg = [0u8; 64];
    msg[..32].copy_from_slice(root);
    msg[32] = (dim & 0xff) as u8;
    msg[33] = ((dim >> 8) & 0xff) as u8;
    msg[34] = ((dim >> 16) & 0xff) as u8;
    msg[35] = ((dim >> 24) & 0xff) as u8;
    msg
}

pub fn pearl_bind_root_a(hash_a: &[u8; 32], m: u32) -> [u8; 32] {
    let msg = pearl_bind_message(hash_a, m);
    blake3_digest(&msg, Some(&PEARL_SEED_SALT_A))
}

pub fn pearl_a_noise_seed_from_hash(
    b_noise_seed: &[u8; 32],
    hash_a: &[u8; 32],
    m: u32,
    salted: bool,
) -> [u8; 32] {
    let bound_a;
    let root_a = if salted {
        bound_a = pearl_bind_root_a(hash_a, m);
        &bound_a
    } else {
        hash_a
    };
    let mut a_in = [0u8; 64];
    a_in[..32].copy_from_slice(b_noise_seed);
    a_in[32..].copy_from_slice(root_a);
    blake3_digest(&a_in, None)
}

fn backend_rank(backend: wgpu::Backend) -> u8 {
    match backend {
        wgpu::Backend::Vulkan | wgpu::Backend::Metal => 0,
        wgpu::Backend::Dx12 => 1,
        _ => 2,
    }
}

fn device_type_label(dt: wgpu::DeviceType) -> &'static str {
    match dt {
        wgpu::DeviceType::DiscreteGpu => "DiscreteGpu",
        wgpu::DeviceType::IntegratedGpu => "IntegratedGpu",
        wgpu::DeviceType::VirtualGpu => "VirtualGpu",
        wgpu::DeviceType::Cpu => "Cpu",
        _ => "Other",
    }
}

fn select_mining_adapters(infos: &[wgpu::AdapterInfo]) -> Vec<usize> {
    let usable: Vec<usize> = (0..infos.len())
        .filter(|&i| infos[i].device_type != wgpu::DeviceType::Cpu)
        .collect();
    let Some(best) = usable.iter().map(|&i| backend_rank(infos[i].backend)).min() else {
        return Vec::new();
    };
    let mut selected: Vec<usize> = usable
        .into_iter()
        .filter(|&i| backend_rank(infos[i].backend) == best)
        .collect();
    selected.sort_by_key(|&i| match infos[i].device_type {
        wgpu::DeviceType::DiscreteGpu => 0,
        wgpu::DeviceType::IntegratedGpu => 1,
        _ => 2,
    });
    selected
}

pub fn list_devices() -> i32 {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        ..Default::default()
    });
    let adapters = instance.enumerate_adapters(wgpu::Backends::PRIMARY);
    let infos: Vec<wgpu::AdapterInfo> = adapters.iter().map(|a| a.get_info()).collect();
    let selected = select_mining_adapters(&infos);
    if selected.is_empty() {
        eprintln!("[pearl-wgpu] no usable GPU adapters (PRIMARY backends)");
        return 0;
    }
    for (i, &raw) in selected.iter().enumerate() {
        let info = &infos[raw];
        println!(
            "[pearl-wgpu] {}: {} ({}, {:?})",
            i,
            info.name,
            device_type_label(info.device_type),
            info.backend
        );
    }
    selected.len() as i32
}

struct Pipelines {
    gen_random: wgpu::ComputePipeline,
    build_pairs: wgpu::ComputePipeline,
    prepack_b: wgpu::ComputePipeline,
    prepack_a: wgpu::ComputePipeline,
    keyed_chunk: wgpu::ComputePipeline,
    blake_mt: wgpu::ComputePipeline,
    reduce_roots: wgpu::ComputePipeline,
    gemm_xor: wgpu::ComputePipeline,
}

struct JobBuffers {
    a_pre: wgpu::Buffer,
    b_pre: wgpu::Buffer,
    a_sig: wgpu::Buffer,
    dummy_sig: wgpu::Buffer,
    pairs: wgpu::Buffer,
    noise_seed: wgpu::Buffer,
    job_key: wgpu::Buffer,
    merkle_roots: wgpu::Buffer,
    a_key8: wgpu::Buffer,
    bound: wgpu::Buffer,
    found_flag: wgpu::Buffer,
    out_t_rows: wgpu::Buffer,
    out_t_cols: wgpu::Buffer,
    // uniforms
    u_gen: wgpu::Buffer,
    u_pairs: wgpu::Buffer,
    u_pre_b: wgpu::Buffer,
    u_pre_a: wgpu::Buffer,
    u_chunk: wgpu::Buffer,
    u_mt: wgpu::Buffer,
    u_reduce: wgpu::Buffer,
    u_scan: wgpu::Buffer,
    staging: wgpu::Buffer,
    staging_size: u64,
    m: i32,
    n: i32,
    k: i32,
    macro_rows: i32,
    macro_cols: i32,
    blocks_k: i32,
    macro_blocks: i32,
    tile_count: i32,
}

pub struct PearlEngine {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipelines: Pipelines,
    job: Option<JobBuffers>,
    device_type: wgpu::DeviceType,
    max_wg_submit: u32,
}

pub enum ScanOutcome {
    Found {
        t_rows: i32,
        t_cols: i32,
        tiles_scanned: u64,
    },
    Exhausted {
        tiles_scanned: u64,
    },
    Cancelled,
}

impl PearlEngine {
    pub fn try_new(device_index: Option<usize>) -> Result<Self, String> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });
        let adapters = instance.enumerate_adapters(wgpu::Backends::PRIMARY);
        let infos: Vec<wgpu::AdapterInfo> = adapters.iter().map(|a| a.get_info()).collect();
        let selected = select_mining_adapters(&infos);
        if selected.is_empty() {
            return Err("no usable GPU adapters".into());
        }
        let list_i = device_index.unwrap_or(0);
        if list_i >= selected.len() {
            return Err(format!(
                "device index {list_i} out of range (0..{})",
                selected.len()
            ));
        }
        let raw = selected[list_i];
        let adapter = adapters
            .into_iter()
            .nth(raw)
            .ok_or_else(|| "adapter missing".to_string())?;
        let info = adapter.get_info();
        eprintln!(
            "[pearl-wgpu] using {}: {} ({}, {:?})",
            list_i,
            info.name,
            device_type_label(info.device_type),
            info.backend
        );

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("Pearl wgpu"),
            required_features: wgpu::Features::empty(),
            required_limits: adapter.limits(),
            memory_hints: Default::default(),
            ..Default::default()
        }))
        .map_err(|e| format!("request_device: {e}"))?;

        // Default wgpu handler panics; log so begin_job/scan can return an error after TDR.
        device.on_uncaptured_error(std::sync::Arc::new(|err| {
            eprintln!("[pearl-wgpu] uncaptured: {err}");
        }));
        device.set_device_lost_callback(|reason, msg| {
            eprintln!("[pearl-wgpu] device lost ({reason:?}): {msg}");
        });

        let max_wg_submit = max_wg_per_submit(info.device_type);
        if matches!(
            info.device_type,
            wgpu::DeviceType::IntegratedGpu | wgpu::DeviceType::Cpu
        ) {
            eprintln!(
                "[pearl-wgpu] iGPU: capping submits at {max_wg_submit} WGs (Windows TDR); production prep is slow - prefer a discrete GPU or --dev"
            );
        }

        let prep_src = prep_wgsl();
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let prep_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pearl_prep"),
            source: wgpu::ShaderSource::Wgsl(prep_src.into()),
        });
        if let Some(err) = pollster::block_on(device.pop_error_scope()) {
            eprintln!("[pearl-wgpu] prep shader create failed: {err}");
            return Err(format!("prep shader: {err}"));
        }

        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let gemm_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pearl_gemm_xor"),
            source: wgpu::ShaderSource::Wgsl(GEMM_WGSL.into()),
        });
        if let Some(err) = pollster::block_on(device.pop_error_scope()) {
            eprintln!("[pearl-wgpu] gemm shader create failed: {err}");
            return Err(format!("gemm shader: {err}"));
        }

        let mk_pipe = |module: &wgpu::ShaderModule, entry: &str, label: &str| {
            device.push_error_scope(wgpu::ErrorFilter::Validation);
            let p = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            });
            let err = pollster::block_on(device.pop_error_scope());
            (p, err)
        };

        let (gen_random, e) = mk_pipe(&prep_module, "pearl_gen_random_matrix", "gen_random");
        if let Some(err) = e {
            eprintln!("[pearl-wgpu] pipeline gen_random failed: {err}");
            return Err(format!("pipeline: {err}"));
        }
        let (build_pairs, e) = mk_pipe(&prep_module, "pearl_build_perm_pairs", "build_pairs");
        if let Some(err) = e {
            eprintln!("[pearl-wgpu] pipeline build_pairs failed: {err}");
            return Err(format!("pipeline: {err}"));
        }
        let (prepack_b, e) = mk_pipe(&prep_module, "pearl_fused_prepack_b", "prepack_b");
        if let Some(err) = e {
            eprintln!("[pearl-wgpu] pipeline prepack_b failed: {err}");
            return Err(format!("pipeline: {err}"));
        }
        let (prepack_a, e) = mk_pipe(&prep_module, "pearl_fused_prepack_a", "prepack_a");
        if let Some(err) = e {
            eprintln!("[pearl-wgpu] pipeline prepack_a failed: {err}");
            return Err(format!("pipeline: {err}"));
        }
        let (keyed_chunk, e) = mk_pipe(&prep_module, "pearl_keyed_chunk_roots", "keyed_chunk");
        if let Some(err) = e {
            eprintln!("[pearl-wgpu] pipeline keyed_chunk failed: {err}");
            return Err(format!("pipeline: {err}"));
        }
        let (blake_mt, e) = mk_pipe(&prep_module, "pearl_compute_blake_mt", "blake_mt");
        if let Some(err) = e {
            eprintln!("[pearl-wgpu] pipeline blake_mt failed: {err}");
            return Err(format!("pipeline: {err}"));
        }
        let (reduce_roots, e) = mk_pipe(&prep_module, "pearl_reduce_roots", "reduce_roots");
        if let Some(err) = e {
            eprintln!("[pearl-wgpu] pipeline reduce_roots failed: {err}");
            return Err(format!("pipeline: {err}"));
        }
        let (gemm_xor, e) = mk_pipe(&gemm_module, "pearl_macro_gemm_xor", "gemm_xor");
        if let Some(err) = e {
            eprintln!("[pearl-wgpu] pipeline gemm_xor failed: {err}");
            return Err(format!("pipeline: {err}"));
        }

        Ok(Self {
            device,
            queue,
            pipelines: Pipelines {
                gen_random,
                build_pairs,
                prepack_b,
                prepack_a,
                keyed_chunk,
                blake_mt,
                reduce_roots,
                gemm_xor,
            },
            job: None,
            device_type: info.device_type,
            max_wg_submit,
        })
    }

    fn mk_storage(device: &wgpu::Device, size: u64, label: &str, extra: wgpu::BufferUsages) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size.max(4),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC
                | extra,
            mapped_at_creation: false,
        })
    }

    fn mk_uniform(device: &wgpu::Device, size: u64, label: &str) -> wgpu::Buffer {
        // Uniform binding size must be multiple of 16; pad to 256 for safety.
        let size = size.max(16).div_ceil(16) * 16;
        let size = size.max(256);
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn ensure_dims(&mut self, m: i32, n: i32, k: i32) -> Result<(), String> {
        if m <= 0 || n <= 0 || k <= 0 {
            return Err("invalid dims".into());
        }
        if m % MACRO != 0 || n % MACRO != 0 || k % KR != 0 {
            return Err(format!(
                "dims must be multiples of MACRO={MACRO}/KR={KR} (got m={m} n={n} k={k})"
            ));
        }
        let blocks_k = k / KR;
        if blocks_k != BLOCKS_K {
            eprintln!(
                "[pearl-wgpu] warning: blocks_k={blocks_k} (expected {BLOCKS_K} for K=4096)"
            );
        }
        let macro_rows = m / MACRO;
        let macro_cols = n / MACRO;
        let macro_blocks = macro_rows * macro_cols;
        let tile_count = (m / MR) * (n / NR);

        if let Some(ref j) = self.job {
            if j.m == m && j.n == n && j.k == k {
                return Ok(());
            }
        }

        let a_pre_bytes = (macro_rows as u64) * (blocks_k as u64) * MACRO_KB_BLOCK;
        let b_pre_bytes = (macro_cols as u64) * (blocks_k as u64) * MACRO_KB_BLOCK;
        let a_sig_bytes = ((m as u64) * (k as u64)).next_multiple_of(4);
        let pairs_bytes = (k as u64) * 2 * 4;

        let raw_len = (m as u64) * (k as u64);
        let pad_len = (raw_len + B3_CHUNK - 1) / B3_CHUNK * B3_CHUNK;
        let num_chunks = (pad_len / B3_CHUNK) as i32;
        let num_subroots = (num_chunks + 255) / 256;
        let merkle_bytes = (num_subroots.max(1) as u64) * 32;

        let staging_size = a_sig_bytes
            .max(32)
            .max(4)
            .max(merkle_bytes)
            .next_multiple_of(256);

        let device = &self.device;
        let job = JobBuffers {
            a_pre: Self::mk_storage(device, a_pre_bytes, "a_pre", wgpu::BufferUsages::empty()),
            b_pre: Self::mk_storage(device, b_pre_bytes, "b_pre", wgpu::BufferUsages::empty()),
            a_sig: Self::mk_storage(device, a_sig_bytes, "a_sig", wgpu::BufferUsages::empty()),
            dummy_sig: Self::mk_storage(device, 4, "dummy_sig", wgpu::BufferUsages::empty()),
            pairs: Self::mk_storage(device, pairs_bytes, "pairs", wgpu::BufferUsages::empty()),
            noise_seed: Self::mk_storage(device, 32, "noise_seed", wgpu::BufferUsages::empty()),
            job_key: Self::mk_storage(device, 32, "job_key", wgpu::BufferUsages::empty()),
            merkle_roots: Self::mk_storage(device, merkle_bytes.max(32), "merkle_roots", wgpu::BufferUsages::empty()),
            a_key8: Self::mk_storage(device, 32, "a_key8", wgpu::BufferUsages::empty()),
            bound: Self::mk_storage(device, 32, "bound", wgpu::BufferUsages::empty()),
            found_flag: Self::mk_storage(device, 4, "found_flag", wgpu::BufferUsages::empty()),
            out_t_rows: Self::mk_storage(device, 4, "out_t_rows", wgpu::BufferUsages::empty()),
            out_t_cols: Self::mk_storage(device, 4, "out_t_cols", wgpu::BufferUsages::empty()),
            u_gen: Self::mk_uniform(device, std::mem::size_of::<PearlGenRandomParams>() as u64, "u_gen"),
            u_pairs: Self::mk_uniform(device, std::mem::size_of::<PearlBuildPairsParams>() as u64, "u_pairs"),
            u_pre_b: Self::mk_uniform(device, std::mem::size_of::<PearlPrepackBParams>() as u64, "u_pre_b"),
            u_pre_a: Self::mk_uniform(device, std::mem::size_of::<PearlPrepackAParams>() as u64, "u_pre_a"),
            u_chunk: Self::mk_uniform(device, std::mem::size_of::<PearlMerkleChunkParams>() as u64, "u_chunk"),
            u_mt: Self::mk_uniform(device, std::mem::size_of::<PearlMerkleMtParams>() as u64, "u_mt"),
            u_reduce: Self::mk_uniform(device, std::mem::size_of::<PearlReduceRootsParams>() as u64, "u_reduce"),
            u_scan: Self::mk_uniform(device, std::mem::size_of::<PearlScanParams>() as u64, "u_scan"),
            staging: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("staging"),
                size: staging_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            staging_size,
            m,
            n,
            k,
            macro_rows,
            macro_cols,
            blocks_k,
            macro_blocks,
            tile_count,
        };
        self.job = Some(job);
        let _ = num_chunks;
        Ok(())
    }

    fn write_uniform<T: Pod>(&self, buf: &wgpu::Buffer, val: &T) {
        self.queue.write_buffer(buf, 0, bytemuck::bytes_of(val));
    }

    fn submit_and_wait(&self, encoder: wgpu::CommandEncoder) -> Result<(), String> {
        self.queue.submit(Some(encoder.finish()));
        match self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(120)),
        }) {
            Ok(_) => Ok(()),
            Err(e) => Err(format!(
                "gpu poll failed (often Windows TDR / device lost on iGPU): {e}"
            )),
        }
    }

    /// Dispatch `total` logical workgroups in chunks of `max_wg_submit` (2D grid each chunk).
    fn dispatch_chunked(
        &self,
        total: u32,
        mut write_and_encode: impl FnMut(u32 /*g_begin*/, u32 /*wg_x*/, u32 /*wg_y*/, i32 /*wg_x_i*/) -> Result<(), String>,
    ) -> Result<(), String> {
        let mut g0 = 0u32;
        let cap = self.max_wg_submit.max(1);
        while g0 < total {
            let chunk = (total - g0).min(cap);
            let (wg_x, wg_y, wg_x_i) = dispatch_2d(chunk);
            write_and_encode(g0, wg_x, wg_y, wg_x_i)?;
            g0 += chunk;
        }
        Ok(())
    }

    fn read_bytes(&self, src: &wgpu::Buffer, size: u64) -> Result<Vec<u8>, String> {
        let job = self.job.as_ref().ok_or("no job")?;
        if size > job.staging_size {
            return Err(format!(
                "read size {size} exceeds staging {}",
                job.staging_size
            ));
        }
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("pearl-read"),
            });
        encoder.copy_buffer_to_buffer(src, 0, &job.staging, 0, size.max(4));
        self.queue.submit(Some(encoder.finish()));

        let slice = job.staging.slice(..size.max(4));
        let status = std::sync::Arc::new(AtomicU8::new(0));
        let status2 = status.clone();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            status2.store(if r.is_ok() { 1 } else { 2 }, Ordering::Release);
        });
        let start = std::time::Instant::now();
        loop {
            let _ = self.device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(std::time::Duration::from_millis(10)),
            });
            match status.load(Ordering::Acquire) {
                1 => break,
                2 => return Err("map_async failed".into()),
                _ if start.elapsed() > std::time::Duration::from_secs(60) => {
                    return Err("map_async timeout".into());
                }
                _ => {}
            }
        }
        let data = slice.get_mapped_range();
        let out = data[..size as usize].to_vec();
        drop(data);
        job.staging.unmap();
        Ok(out)
    }

    pub fn begin_job(&mut self, m: i32, n: i32, k: i32, b_noise_seed: &[u8; 32]) -> Result<(), String> {
        self.ensure_dims(m, n, k)?;
        let job = self.job.as_ref().unwrap();
        self.queue
            .write_buffer(&job.noise_seed, 0, b_noise_seed);

        // build_perm_pairs is_b=1
        self.write_uniform(
            &job.u_pairs,
            &PearlBuildPairsParams {
                is_b: 1,
                k,
                rank: R_RANK,
                _pad: 0,
            },
        );
        {
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bg_pairs_b"),
                layout: &self.pipelines.build_pairs.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: job.u_pairs.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: job.noise_seed.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: job.pairs.as_entire_binding(),
                    },
                ],
            });
            let num_blocks = (k + 7) / 8;
            let num_wg = ((num_blocks + 63) / 64) as u32;
            let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("build_pairs_b"),
            });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("build_pairs_b"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipelines.build_pairs);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(num_wg.max(1), 1, 1);
            }
            self.submit_and_wait(enc)?;
        }

        let total_wg = {
            let job = self.job.as_ref().unwrap();
            (job.macro_cols * MICRO * job.blocks_k) as u32
        };
        let blocks_k = self.job.as_ref().unwrap().blocks_k;
        let macro_cols = self.job.as_ref().unwrap().macro_cols;
        self.dispatch_chunked(total_wg, |g_begin, wg_x, wg_y, wg_x_i| {
            let job = self.job.as_ref().unwrap();
            self.write_uniform(
                &job.u_pre_b,
                &PearlPrepackBParams {
                    n,
                    k,
                    rank: R_RANK,
                    blocks_k,
                    macro_cols,
                    has_signal: 0,
                    wg_x: wg_x_i,
                    g_begin: g_begin as i32,
                },
            );
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bg_pre_b"),
                layout: &self.pipelines.prepack_b.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: job.u_pre_b.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: job.b_pre.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: job.noise_seed.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: job.pairs.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: job.dummy_sig.as_entire_binding(),
                    },
                ],
            });
            let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("prepack_b"),
            });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("prepack_b"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipelines.prepack_b);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(wg_x, wg_y, 1);
            }
            self.submit_and_wait(enc)
        })?;
        Ok(())
    }

    pub fn prep_a_signal(
        &mut self,
        ab_seed: &[u8],
        job_key: &[u8; 32],
        hash_a_out: &mut [u8; 32],
    ) -> Result<(), String> {
        let (m, k, blocks_k) = {
            let job = self.job.as_ref().ok_or("begin_job not called")?;
            (job.m, job.k, job.blocks_k)
        };
        let _ = blocks_k;
        let total = m * k;
        let rng = seed_to_u64(ab_seed);

        // Pack 4 s8 / WI 鈫?workgroups cover ceil(total/4) threads; chunk for TDR.
        let num_words = ((total + 3) / 4) as u32;
        let num_wg_total = ((num_words + 255) / 256).max(1);
        self.dispatch_chunked(num_wg_total, |wg_begin, wg_x, wg_y, wg_x_i| {
            let job = self.job.as_ref().unwrap();
            let word_begin = (wg_begin * 256) as i32;
            self.write_uniform(
                &job.u_gen,
                &PearlGenRandomParams {
                    rng_seed: [rng as u32, (rng >> 32) as u32],
                    matrix_tag: 0,
                    total_elems: total,
                    wg_x: wg_x_i,
                    word_begin,
                    _pad1: 0,
                    _pad2: 0,
                },
            );
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bg_gen"),
                layout: &self.pipelines.gen_random.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: job.u_gen.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: job.a_sig.as_entire_binding(),
                    },
                ],
            });
            let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gen_random"),
            });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("gen_random"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipelines.gen_random);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(wg_x, wg_y, 1);
            }
            self.submit_and_wait(enc)
        })?;

        {
            let job = self.job.as_ref().unwrap();
            self.queue.write_buffer(&job.job_key, 0, job_key);
        }
        self.matrix_keyed_hash(job_key, hash_a_out)?;
        Ok(())
    }

    fn matrix_keyed_hash(&self, job_key: &[u8; 32], out: &mut [u8; 32]) -> Result<(), String> {
        let job = self.job.as_ref().ok_or("no job")?;
        let raw_len = (job.m as u64) * (job.k as u64);
        let pad_len = (raw_len + B3_CHUNK - 1) / B3_CHUNK * B3_CHUNK;
        let num_chunks = (pad_len / B3_CHUNK) as i32;
        if num_chunks <= 0 {
            return Err("empty matrix".into());
        }

        if num_chunks == 1 {
            // Host keyed blake3 over packed s8 bytes.
            let bytes = self.read_bytes(&job.a_sig, raw_len)?;
            let mut tmp = vec![0u8; pad_len as usize];
            tmp[..raw_len as usize].copy_from_slice(&bytes[..raw_len as usize]);
            *out = blake3_digest(&tmp, Some(job_key));
            return Ok(());
        }

        self.queue.write_buffer(&job.job_key, 0, job_key);
        let num_subroots = (num_chunks + 255) / 256;

        self.write_uniform(
            &job.u_chunk,
            &PearlMerkleChunkParams {
                raw_len: raw_len as u32,
                pad_len: pad_len as u32,
                num_chunks,
                _pad: 0,
            },
        );
        {
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bg_chunk"),
                layout: &self.pipelines.keyed_chunk.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 15,
                        resource: job.u_chunk.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 16,
                        resource: job.a_sig.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 17,
                        resource: job.job_key.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 18,
                        resource: job.merkle_roots.as_entire_binding(),
                    },
                ],
            });
            let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("keyed_chunk"),
            });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("keyed_chunk"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipelines.keyed_chunk);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(num_subroots as u32, 1, 1);
            }
            self.submit_and_wait(enc)?;
        }

        self.merkle_finish_root(num_subroots)?;
        let root_bytes = self.read_bytes(&job.merkle_roots, 32)?;
        out.copy_from_slice(&root_bytes);
        Ok(())
    }

    fn merkle_finish_root(&self, num_subroots: i32) -> Result<(), String> {
        let job = self.job.as_ref().ok_or("no job")?;
        let num_mt_blocks = (num_subroots + 255) / 256;
        let is_single = if num_mt_blocks == 1 { 1 } else { 0 };

        self.write_uniform(
            &job.u_mt,
            &PearlMerkleMtParams {
                num_leaves: num_subroots,
                is_single_block: is_single,
                _pad0: 0,
                _pad1: 0,
            },
        );
        {
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bg_mt"),
                layout: &self.pipelines.blake_mt.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 19,
                        resource: job.u_mt.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 20,
                        resource: job.job_key.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 21,
                        resource: job.merkle_roots.as_entire_binding(),
                    },
                ],
            });
            let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("blake_mt"),
            });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("blake_mt"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipelines.blake_mt);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(num_mt_blocks as u32, 1, 1);
            }
            self.submit_and_wait(enc)?;
        }

        if num_mt_blocks > 1 {
            self.write_uniform(
                &job.u_reduce,
                &PearlReduceRootsParams {
                    num_leaves: num_mt_blocks,
                    _pad0: 0,
                    _pad1: 0,
                    _pad2: 0,
                },
            );
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bg_reduce"),
                layout: &self.pipelines.reduce_roots.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 22,
                        resource: job.u_reduce.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 23,
                        resource: job.job_key.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 24,
                        resource: job.merkle_roots.as_entire_binding(),
                    },
                ],
            });
            let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("reduce_roots"),
            });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("reduce_roots"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipelines.reduce_roots);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            self.submit_and_wait(enc)?;
        }
        Ok(())
    }

    pub fn prepack_a(&mut self, a_noise_seed: &[u8; 32]) -> Result<(), String> {
        let job = self.job.as_ref().ok_or("begin_job not called")?;
        let m = job.m;
        let k = job.k;
        self.queue
            .write_buffer(&job.noise_seed, 0, a_noise_seed);

        self.write_uniform(
            &job.u_pairs,
            &PearlBuildPairsParams {
                is_b: 0,
                k,
                rank: R_RANK,
                _pad: 0,
            },
        );
        {
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bg_pairs_a"),
                layout: &self.pipelines.build_pairs.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: job.u_pairs.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: job.noise_seed.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: job.pairs.as_entire_binding(),
                    },
                ],
            });
            let num_blocks = (k + 7) / 8;
            let num_wg = ((num_blocks + 63) / 64) as u32;
            let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("build_pairs_a"),
            });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("build_pairs_a"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipelines.build_pairs);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(num_wg.max(1), 1, 1);
            }
            self.submit_and_wait(enc)?;
        }

        let (total_wg, blocks_k, macro_rows) = {
            let job = self.job.as_ref().unwrap();
            (
                (job.macro_rows * MICRO * job.blocks_k) as u32,
                job.blocks_k,
                job.macro_rows,
            )
        };
        self.dispatch_chunked(total_wg, |g_begin, wg_x, wg_y, wg_x_i| {
            let job = self.job.as_ref().unwrap();
            self.write_uniform(
                &job.u_pre_a,
                &PearlPrepackAParams {
                    m,
                    k,
                    rank: R_RANK,
                    blocks_k,
                    macro_rows,
                    wg_x: wg_x_i,
                    g_begin: g_begin as i32,
                    _pad1: 0,
                },
            );
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bg_pre_a"),
                layout: &self.pipelines.prepack_a.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 10,
                        resource: job.u_pre_a.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 11,
                        resource: job.a_pre.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 12,
                        resource: job.noise_seed.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 13,
                        resource: job.pairs.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 14,
                        resource: job.a_sig.as_entire_binding(),
                    },
                ],
            });
            let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("prepack_a"),
            });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("prepack_a"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipelines.prepack_a);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(wg_x, wg_y, 1);
            }
            self.submit_and_wait(enc)
        })?;
        Ok(())
    }

    pub fn scan(
        &mut self,
        a_key8: &[u32; 8],
        bound: &[u32; 8],
        macro_batch: i32,
        cancel: &dyn Fn() -> bool,
        on_progress: &dyn Fn(u64),
    ) -> Result<ScanOutcome, String> {
        let job = self.job.as_ref().ok_or("begin_job not called")?;
        let mut macro_batch = macro_batch;
        if macro_batch < 1 {
            return Err("macro_batch < 1".into());
        }
        // Keep each scan submit under TDR budget (esp. iGPU).
        let cap = self.max_wg_submit as i32;
        if macro_batch > cap {
            eprintln!(
                "[pearl-wgpu] clamping macro_batch {macro_batch} -> {cap} (device TDR cap)"
            );
            macro_batch = cap;
        }

        self.queue
            .write_buffer(&job.a_key8, 0, bytemuck::cast_slice(a_key8));
        self.queue
            .write_buffer(&job.bound, 0, bytemuck::cast_slice(bound));
        let zero = 0i32;
        self.queue
            .write_buffer(&job.found_flag, 0, bytemuck::bytes_of(&zero));

        let mut tiles_scanned = 0u64;
        let macro_blocks = job.macro_blocks;

        for mb0 in (0..macro_blocks).step_by(macro_batch as usize) {
            if cancel() {
                return Ok(ScanOutcome::Cancelled);
            }
            let batch_count = (macro_batch).min(macro_blocks - mb0);
            let (wg_x, wg_y, wg_x_i) = dispatch_2d(batch_count as u32);

            self.write_uniform(
                &job.u_scan,
                &PearlScanParams {
                    n: job.n,
                    blocks_k: job.blocks_k,
                    num_milestones: NUM_MILESTONES.min(job.blocks_k),
                    tile_count: job.tile_count,
                    macro_rows: job.macro_rows,
                    macro_cols: job.macro_cols,
                    mb_begin: mb0,
                    micro_m_begin: 0,
                    micro_m_count: MICRO,
                    wg_x: wg_x_i,
                    batch_count,
                    _pad2: 0,
                },
            );

            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bg_scan"),
                layout: &self.pipelines.gemm_xor.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: job.a_pre.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: job.b_pre.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: job.u_scan.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: job.a_key8.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: job.bound.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: job.found_flag.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: job.out_t_rows.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: job.out_t_cols.as_entire_binding(),
                    },
                ],
            });

            let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("scan"),
            });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("scan"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipelines.gemm_xor);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(wg_x, wg_y, 1);
            }
            self.submit_and_wait(enc)?;

            let found_bytes = self.read_bytes(&job.found_flag, 4)?;
            let found = i32::from_le_bytes([
                found_bytes[0],
                found_bytes[1],
                found_bytes[2],
                found_bytes[3],
            ]);
            tiles_scanned += (batch_count as u64) * HASH_TILES_PER_MACRO;
            on_progress(tiles_scanned);

            if found != 0 {
                let rows_b = self.read_bytes(&job.out_t_rows, 4)?;
                let cols_b = self.read_bytes(&job.out_t_cols, 4)?;
                let t_rows = i32::from_le_bytes([rows_b[0], rows_b[1], rows_b[2], rows_b[3]]);
                let t_cols = i32::from_le_bytes([cols_b[0], cols_b[1], cols_b[2], cols_b[3]]);
                return Ok(ScanOutcome::Found {
                    t_rows,
                    t_cols,
                    tiles_scanned,
                });
            }
        }

        Ok(ScanOutcome::Exhausted { tiles_scanned })
    }

    pub fn download_a_sig(&self, out: &mut [i8]) -> Result<(), String> {
        let job = self.job.as_ref().ok_or("begin_job not called")?;
        let need = (job.m as usize) * (job.k as usize);
        if out.len() < need {
            return Err(format!("out len {} < {}", out.len(), need));
        }
        let bytes = self.read_bytes(&job.a_sig, need as u64)?;
        for i in 0..need {
            out[i] = bytes[i] as i8;
        }
        Ok(())
    }
}
