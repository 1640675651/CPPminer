//! Pearl wgpu mining engine: fused prep + GEMM/XOR/jackpot scan.

use bytemuck::{Pod, Zeroable};
use std::sync::atomic::{AtomicU8, Ordering};

const KR: i32 = 128;
const R_RANK: i32 = 128;
const BLOCKS_K: i32 = 32;
const NUM_MILESTONES: i32 = 32;
const B3_CHUNK: u64 = 1024;
/// Prepack row/column group (pearl_prep.wgsl MR/NR), independent of the GEMM register tile.
const PREP_GROUP: i32 = 8;

/// --wgpu-lds: -1 auto (discrete GPUs only), 0 off, 1 on. Read at engine creation.
static LDS_MODE: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(-1);

pub fn set_lds_mode(mode: i32) {
    LDS_MODE.store(mode.clamp(-1, 1), Ordering::Relaxed);
}

/// GEMM register tile + macro block (same choices as --ocl-tile / --ocl-macro).
#[derive(Clone, Copy)]
struct TileConfig {
    mr: i32,
    nr: i32,
    macro_mn: i32,
}

impl TileConfig {
    /// Jackpot hash tile width: 4x4 register tiles hash as 4x8 (two halves per WI).
    fn hash_nr(&self) -> i32 {
        if self.mr == 4 && self.nr == 4 { 8 } else { self.nr }
    }
    fn wg_size(&self) -> u32 {
        ((self.macro_mn / self.mr) * (self.macro_mn / self.hash_nr())) as u32
    }
    fn hash_tiles_per_macro(&self) -> u64 {
        self.wg_size() as u64
    }
    /// Bytes of one packed macro x k-block panel (A or B).
    fn kb_block_bytes(&self) -> u64 {
        (self.macro_mn as u64) * (KR as u64)
    }
    /// Workgroup storage of pearl_macro_gemm_xor_lds: A + B macro k-block.
    fn lds_bytes(&self) -> u32 {
        (2 * self.kb_block_bytes()) as u32
    }
}

static TILE: std::sync::Mutex<TileConfig> =
    std::sync::Mutex::new(TileConfig { mr: 8, nr: 8, macro_mn: 128 });

/// Register tile 4x4, 4x8, 8x8 or 8x16; macro 64x64 or 128x128. Read at engine creation.
pub fn set_tile(mr: i32, nr: i32, macro_m: i32, macro_n: i32) -> Result<(), String> {
    let tile_ok = matches!((mr, nr), (4, 4) | (4, 8) | (8, 8) | (8, 16));
    if !tile_ok {
        return Err(format!("register tile must be 4x4, 4x8, 8x8 or 8x16 (got {mr}x{nr})"));
    }
    if macro_m != macro_n || !(macro_m == 64 || macro_m == 128) {
        return Err(format!("macro must be 64x64 or 128x128 (got {macro_m}x{macro_n})"));
    }
    *TILE.lock().unwrap() = TileConfig { mr, nr, macro_mn: macro_m };
    Ok(())
}

const PREP_WGSL_RAW: &str =
    include_str!("../../../src/pearl/wgpu/kernels/pearl_prep.wgsl");
const GEMM_WGSL_RAW: &str =
    include_str!("../../../src/pearl/wgpu/kernels/pearl_gemm_xor.wgsl");

fn buf_range(buffer: &wgpu::Buffer, offset: u64, size: u64) -> wgpu::BindingResource<'_> {
    wgpu::BindingResource::Buffer(wgpu::BufferBinding {
        buffer,
        offset,
        size: wgpu::BufferSize::new(size),
    })
}

fn inject(src: &str, marker: &str, code: &str) -> String {
    assert!(src.contains(marker), "shader marker {marker} missing");
    src.replacen(marker, code, 1)
}

fn prep_wgsl(tile: &TileConfig) -> String {
    let m = tile.macro_mn;
    inject(
        PREP_WGSL_RAW,
        "// @MACRO_CONFIG@",
        &format!("const MACRO_M: i32 = {m};\nconst MACRO_N: i32 = {m};"),
    )
}

/// Tile constants plus per-tile accumulator code. The accumulator is named vec4<i32> fields
/// (cJ_I = A rows 4I..4I+3 x B col J), not array<i32, MR*NR>: Intel IGC spills the array.
fn gemm_wgsl(tile: &TileConfig) -> String {
    use std::fmt::Write;
    let (mr, nr, hash_nr, m) = (tile.mr, tile.nr, tile.hash_nr(), tile.macro_mn);
    let av = (mr / 4) as usize;
    let bv = (nr / 4) as usize;
    let lanes = ["x", "y", "z", "w"];
    let fields: Vec<String> =
        (0..nr as usize).flat_map(|j| (0..av).map(move |i| format!("c{j}_{i}"))).collect();

    let mut s = String::new();
    let _ = writeln!(s, "const MR: i32 = {mr};");
    let _ = writeln!(s, "const NR: i32 = {nr};");
    let _ = writeln!(s, "const HASH_NR: i32 = {hash_nr};");
    let _ = writeln!(s, "const MACRO_M: i32 = {m};");
    let _ = writeln!(s, "const MACRO_N: i32 = {m};");
    let _ = writeln!(s);
    let _ = writeln!(s, "struct Acc {{");
    for f in &fields {
        let _ = writeln!(s, "    {f}: vec4<i32>,");
    }
    let _ = writeln!(s, "}}");
    let _ = writeln!(s);

    let a_params: Vec<String> = (0..av).map(|i| format!("a{i}: vec4<u32>")).collect();
    let b_params: Vec<String> = (0..bv).map(|i| format!("b{i}: vec4<u32>")).collect();
    let _ = writeln!(
        s,
        "fn acc_kgroup(acc: ptr<function, Acc>, {}, {}) {{",
        a_params.join(", "),
        b_params.join(", ")
    );
    // Every dot4I8Packed operand is written out fresh: naga's MSL backend names the packed_char4
    // temp after the argument expression, so reusing one (e.g. a helper's `b` param) redefines it.
    for j in 0..nr as usize {
        let b = format!("b{}.{}", j / 4, lanes[j % 4]);
        for i in 0..av {
            let dots: Vec<String> =
                lanes.iter().map(|l| format!("dot4I8Packed(a{i}.{l}, {b})")).collect();
            let _ = writeln!(s, "    (*acc).c{j}_{i} += vec4<i32>({});", dots.join(", "));
        }
    }
    let _ = writeln!(s, "}}");
    let _ = writeln!(s);

    let terms: Vec<String> = fields.iter().map(|f| format!("a.{f}")).collect();
    let _ = writeln!(s, "fn acc_xor(acc: ptr<function, Acc>) -> u32 {{");
    let _ = writeln!(s, "    let a = *acc;");
    let _ = writeln!(s, "    let xv = bitcast<vec4<u32>>({});", terms.join(" ^ "));
    let _ = writeln!(s, "    return xv.x ^ xv.y ^ xv.z ^ xv.w;");
    let _ = writeln!(s, "}}");

    for (name, a_buf, b_buf) in [("kgroup_global", "a_pre", "b_pre"), ("kgroup_lds", "lds_a", "lds_b")] {
        let a_args: Vec<String> = (0..av).map(|i| format!("{a_buf}[ai + {i}u]")).collect();
        let b_args: Vec<String> = (0..bv).map(|i| format!("{b_buf}[bi + {i}u]")).collect();
        let _ = writeln!(s);
        let _ = writeln!(s, "fn {name}(acc: ptr<function, Acc>, ai: u32, bi: u32) {{");
        let _ = writeln!(s, "    acc_kgroup(acc, {}, {});", a_args.join(", "), b_args.join(", "));
        let _ = writeln!(s, "}}");
    }
    inject(GEMM_WGSL_RAW, "// @TILE_CONFIG@", &s)
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
    jm_base: i32,
    _pad0: i32,
    _pad1: i32,
    _pad2: i32,
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
    im_base: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PearlMerkleChunkParams {
    raw_len: u32,
    pad_len: u32,
    num_chunks: i32,
    bid_begin: i32,
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
    a_im_base: i32,
    b_jm_base: i32,
    _pad2: i32,
    _pad3: i32,
    _pad4: i32,
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
    /// max_storage_buffer_binding_size: a_pre / b_pre / a_sig are bound in windows below this.
    max_binding: u64,
    tile: TileConfig,
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
                "[pearl-wgpu] iGPU: capping submits at {max_wg_submit} WGs (Windows TDR)"
            );
        }

        let tile = *TILE.lock().unwrap();
        let limits = device.limits();
        let wg_size = tile.wg_size();
        if wg_size > limits.max_compute_invocations_per_workgroup
            || wg_size > limits.max_compute_workgroup_size_x
        {
            return Err(format!(
                "tile {}x{} / macro {m}x{m} needs {wg_size} invocations per workgroup (device max {})",
                tile.mr,
                tile.nr,
                limits
                    .max_compute_invocations_per_workgroup
                    .min(limits.max_compute_workgroup_size_x),
                m = tile.macro_mn
            ));
        }
        eprintln!(
            "[pearl-wgpu] register tile {}x{}, hash tile {}x{}, macro {m}x{m}, {wg_size} WI/WG",
            tile.mr,
            tile.nr,
            tile.mr,
            tile.hash_nr(),
            m = tile.macro_mn
        );
        let max_binding = (limits.max_storage_buffer_binding_size as u64) & !255;
        if max_binding < (1 << 30) {
            eprintln!(
                "[pearl-wgpu] storage binding limit {} MiB: large buffers are bound in windows",
                max_binding >> 20
            );
        }

        let prep_src = prep_wgsl(&tile);
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
        // Loop bounding blocks driver unrolling, spilling acc[]/a_pack/b_pack to local memory
        // (~36x slower on GTX 1070). All loops in this shader have fixed or uniform trip counts.
        let mut gemm_checks = wgpu::ShaderRuntimeChecks::checked();
        gemm_checks.force_loop_bounding = false;
        let gemm_module = unsafe {
            device.create_shader_module_trusted(
                wgpu::ShaderModuleDescriptor {
                    label: Some("pearl_gemm_xor"),
                    source: wgpu::ShaderSource::Wgsl(gemm_wgsl(&tile).into()),
                },
                gemm_checks,
            )
        };
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
        // --wgpu-lds: auto = on for discrete GPUs (GTX 1070 ~3.4 -> ~4.2 TMAC/s), off otherwise
        // (UHD 770 ~455 -> ~380 GMAC/s).
        let storage_limit = limits.max_compute_workgroup_storage_size;
        let lds_bytes = tile.lds_bytes();
        let lds_fits = storage_limit >= lds_bytes;
        let lds_mode = LDS_MODE.load(Ordering::Relaxed);
        let want_lds = match lds_mode {
            0 => false,
            1 => true,
            _ => info.device_type == wgpu::DeviceType::DiscreteGpu,
        };
        if want_lds && !lds_fits {
            eprintln!(
                "[pearl-wgpu] LDS staging needs {lds_bytes} B workgroup storage (device limit {storage_limit} B); using direct loads"
            );
        }
        let use_lds = want_lds && lds_fits;
        let gemm_entry = if use_lds { "pearl_macro_gemm_xor_lds" } else { "pearl_macro_gemm_xor" };
        eprintln!(
            "[pearl-wgpu] LDS staging: {} (--wgpu-lds {})",
            if use_lds { "on" } else { "off" },
            match lds_mode {
                0 => "off",
                1 => "on",
                _ => "auto",
            }
        );
        let (gemm_xor, e) = mk_pipe(&gemm_module, gemm_entry, "gemm_xor");
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
            max_binding,
            tile,
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
        let macro_mn = self.tile.macro_mn;
        if m % macro_mn != 0 || n % macro_mn != 0 || k % KR != 0 {
            return Err(format!(
                "dims must be multiples of MACRO={macro_mn}/KR={KR} (got m={m} n={n} k={k})"
            ));
        }
        let blocks_k = k / KR;
        if blocks_k != BLOCKS_K {
            eprintln!(
                "[pearl-wgpu] warning: blocks_k={blocks_k} (expected {BLOCKS_K} for K=4096)"
            );
        }
        let macro_rows = m / macro_mn;
        let macro_cols = n / macro_mn;
        let macro_blocks = macro_rows * macro_cols;
        let tile_count = (m / self.tile.mr) * (n / self.tile.hash_nr());

        if let Some(ref j) = self.job {
            if j.m == m && j.n == n && j.k == k {
                return Ok(());
            }
        }

        let kb_block = self.tile.kb_block_bytes();
        let a_pre_bytes = (macro_rows as u64) * (blocks_k as u64) * kb_block;
        let b_pre_bytes = (macro_cols as u64) * (blocks_k as u64) * kb_block;
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

    /// Bytes of one macro row of a_pre (or column of b_pre) over all k-blocks; also the a_sig
    /// bytes of one macro row (MACRO rows x K).
    fn panel_bytes(&self) -> u64 {
        let job = self.job.as_ref().unwrap();
        job.blocks_k as u64 * self.tile.kb_block_bytes()
    }

    /// Whole panels that fit in one binding.
    fn panels_per_binding(&self) -> i32 {
        (self.max_binding / self.panel_bytes()).clamp(1, i32::MAX as u64) as i32
    }

    /// Workgroup cap so a chunk of consecutive workgroups (`per_panel` per macro row/column)
    /// touches at most panels_per_binding() panels.
    fn panel_chunk_cap(&self, per_panel: u32) -> u32 {
        let w = self.panels_per_binding() as u32;
        let cap = if w >= 2 { (w - 1) * per_panel } else { per_panel };
        cap.min(self.max_wg_submit)
    }

    /// Dispatch `total` logical workgroups in chunks of `max_wg_submit` (2D grid each chunk).
    fn dispatch_chunked(
        &self,
        total: u32,
        write_and_encode: impl FnMut(u32 /*g_begin*/, u32 /*wg_x*/, u32 /*wg_y*/, i32 /*wg_x_i*/) -> Result<(), String>,
    ) -> Result<(), String> {
        self.dispatch_chunked_cap(total, self.max_wg_submit, write_and_encode)
    }

    /// Like `dispatch_chunked`, with an explicit per-submit workgroup cap.
    fn dispatch_chunked_cap(
        &self,
        total: u32,
        cap: u32,
        mut write_and_encode: impl FnMut(u32 /*g_begin*/, u32 /*wg_x*/, u32 /*wg_y*/, i32 /*wg_x_i*/) -> Result<(), String>,
    ) -> Result<(), String> {
        let mut g0 = 0u32;
        let cap = cap.max(1);
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
            (job.n / PREP_GROUP * job.blocks_k) as u32
        };
        let blocks_k = self.job.as_ref().unwrap().blocks_k;
        let macro_cols = self.job.as_ref().unwrap().macro_cols;
        let per_col = (self.tile.macro_mn / PREP_GROUP * blocks_k) as u32;
        let panel = self.panel_bytes();
        let cap = self.panel_chunk_cap(per_col);
        self.dispatch_chunked_cap(total_wg, cap, |g_begin, wg_x, wg_y, wg_x_i| {
            let job = self.job.as_ref().unwrap();
            let jm_lo = g_begin / per_col;
            let jm_hi = ((g_begin + wg_x * wg_y - 1) / per_col).min(macro_cols as u32 - 1);
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
                    jm_base: jm_lo as i32,
                    _pad0: 0,
                    _pad1: 0,
                    _pad2: 0,
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
                        resource: buf_range(
                            &job.b_pre,
                            jm_lo as u64 * panel,
                            (jm_hi - jm_lo + 1) as u64 * panel,
                        ),
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
        let cap = self.max_wg_submit.min((self.max_binding / 1024).max(1) as u32);
        self.dispatch_chunked_cap(num_wg_total, cap, |wg_begin, wg_x, wg_y, wg_x_i| {
            let job = self.job.as_ref().unwrap();
            let word_begin = (wg_begin * 256) as i32;
            let words = (wg_x * wg_y * 256).min(num_words - wg_begin * 256);
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
                        resource: buf_range(&job.a_sig, word_begin as u64 * 4, words as u64 * 4),
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

        // One workgroup hashes 256 chunks (256 KiB of a_sig); bind a_sig per dispatch window.
        const BLOCK_BYTES: u64 = 256 * B3_CHUNK;
        let a_sig_bytes = raw_len.next_multiple_of(4);
        let blocks_per_dispatch = (self.max_binding / BLOCK_BYTES).clamp(1, i32::MAX as u64) as i32;
        for bid_begin in (0..num_subroots).step_by(blocks_per_dispatch as usize) {
            let blocks = blocks_per_dispatch.min(num_subroots - bid_begin);
            let mat_off = bid_begin as u64 * BLOCK_BYTES;
            let mat_len = (blocks as u64 * BLOCK_BYTES).min(a_sig_bytes - mat_off);
            self.write_uniform(
                &job.u_chunk,
                &PearlMerkleChunkParams {
                    raw_len: raw_len as u32,
                    pad_len: pad_len as u32,
                    num_chunks,
                    bid_begin,
                },
            );
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
                        resource: buf_range(&job.a_sig, mat_off, mat_len),
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
                pass.dispatch_workgroups(blocks as u32, 1, 1);
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
            ((job.m / PREP_GROUP) as u32, job.blocks_k, job.macro_rows)
        };
        // Each WG covers all blocks_k k-blocks, so scale the TDR cap down accordingly.
        let per_row = (self.tile.macro_mn / PREP_GROUP) as u32;
        let panel = self.panel_bytes();
        let cap = (self.max_wg_submit / (blocks_k.max(1) as u32)).min(self.panel_chunk_cap(per_row));
        self.dispatch_chunked_cap(total_wg, cap, |g_begin, wg_x, wg_y, wg_x_i| {
            let job = self.job.as_ref().unwrap();
            let im_lo = g_begin / per_row;
            let im_hi = ((g_begin + wg_x * wg_y - 1) / per_row).min(macro_rows as u32 - 1);
            let (win_off, win_len) = (im_lo as u64 * panel, (im_hi - im_lo + 1) as u64 * panel);
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
                    im_base: im_lo as i32,
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
                        resource: buf_range(&job.a_pre, win_off, win_len),
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
                        // MACRO rows x K bytes of a_sig per macro row == panel bytes of a_pre.
                        resource: buf_range(&job.a_sig, win_off, win_len),
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
        let rows = job.macro_rows;
        let panel = self.panel_bytes();
        let window = self.panels_per_binding();
        // Macro blocks run column-major (im = mb % macro_rows): (im_lo, im_count, jm_lo, jm_count).
        let span = |mb0: i32, cnt: i32| {
            let (j0, j1) = (mb0 / rows, (mb0 + cnt - 1) / rows);
            if j0 == j1 {
                (mb0 % rows, cnt, j0, 1)
            } else {
                (0, rows, j0, j1 - j0 + 1)
            }
        };

        let mut mb0 = 0;
        while mb0 < macro_blocks {
            if cancel() {
                return Ok(ScanOutcome::Cancelled);
            }
            let mut batch_count = (macro_batch).min(macro_blocks - mb0);
            let (_, im_n, _, jm_n) = span(mb0, batch_count);
            if im_n > window || jm_n > window {
                // Stay inside one macro column and at most `window` rows of a_pre.
                batch_count = batch_count.min(rows - mb0 % rows).min(window);
            }
            let (im_lo, im_n, jm_lo, jm_n) = span(mb0, batch_count);
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
                    micro_m_count: self.tile.macro_mn / self.tile.mr,
                    wg_x: wg_x_i,
                    batch_count,
                    a_im_base: im_lo,
                    b_jm_base: jm_lo,
                    _pad2: 0,
                    _pad3: 0,
                    _pad4: 0,
                },
            );

            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bg_scan"),
                layout: &self.pipelines.gemm_xor.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buf_range(&job.a_pre, im_lo as u64 * panel, im_n as u64 * panel),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buf_range(&job.b_pre, jm_lo as u64 * panel, jm_n as u64 * panel),
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
            tiles_scanned += (batch_count as u64) * self.tile.hash_tiles_per_macro();
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
            mb0 += batch_count;
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