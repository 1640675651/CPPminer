//! C ABI for Quantus `engine-gpu` (wgpu) used by CPPminer.

use engine_cpu::{CancelCheck, EngineStatus, MinerEngine, Range};
use engine_gpu::{list_mining_adapters, GpuEngine};
use parking_lot::Mutex;
use pow_core::JobContext;
use primitive_types::U512;
use std::os::raw::c_int;
use std::slice;
use std::sync::OnceLock;

static ENGINE: OnceLock<Mutex<Option<GpuEngine>>> = OnceLock::new();

fn engine_slot() -> &'static Mutex<Option<GpuEngine>> {
    ENGINE.get_or_init(|| Mutex::new(None))
}

struct CCancelCheck {
    check: Option<unsafe extern "C" fn() -> c_int>,
}

impl CancelCheck for CCancelCheck {
    fn is_cancelled(&self) -> bool {
        match self.check {
            Some(f) => unsafe { f() != 0 },
            None => false,
        }
    }
}

/// Return codes for `cp_wgpu_search_range`.
pub const CP_WGPU_OK_FOUND: c_int = 1;
pub const CP_WGPU_OK_EXHAUSTED: c_int = 0;
pub const CP_WGPU_CANCELLED: c_int = -1;
pub const CP_WGPU_ERROR: c_int = -2;
pub const CP_WGPU_DEVICE_LOST: c_int = -3;

/// Print mining adapters (same indices as `--devices`) and return count.
#[no_mangle]
pub unsafe extern "C" fn cp_wgpu_list_devices() -> c_int {
    let list = list_mining_adapters();
    if list.is_empty() {
        eprintln!("[wgpu] no usable GPU adapters (PRIMARY backends)");
        return 0;
    }
    for a in &list {
        println!(
            "[wgpu] {}: {} ({}, {})",
            a.index, a.name, a.device_type, a.backend
        );
    }
    list.len() as c_int
}

/// Initialize the global GpuEngine.
///
/// `batch_size` 0 → default 1_000_000.
/// `allow_integrated` non-zero includes iGPUs when discrete GPUs exist (ignored
/// when `devices`/`ndev` select adapters explicitly).
/// `devices`/`ndev`: optional indices from `cp_wgpu_list_devices`; NULL/0 → all.
#[no_mangle]
pub unsafe extern "C" fn cp_wgpu_init(
    batch_size: u32,
    allow_integrated: c_int,
    devices: *const c_int,
    ndev: c_int,
) -> c_int {
    let batch = if batch_size == 0 {
        1_000_000
    } else {
        batch_size
    };

    let indices: Option<Vec<usize>> = if !devices.is_null() && ndev > 0 {
        let slice = slice::from_raw_parts(devices, ndev as usize);
        let mut v = Vec::with_capacity(slice.len());
        for &d in slice {
            if d < 0 {
                eprintln!("[wgpu] invalid device index {d}");
                return -1;
            }
            v.push(d as usize);
        }
        Some(v)
    } else {
        None
    };

    match GpuEngine::try_new_with_devices(
        batch,
        0,
        allow_integrated != 0,
        indices.as_deref(),
    ) {
        Ok(eng) => {
            let mut slot = engine_slot().lock();
            if let Some(old) = slot.take() {
                drop(old);
                GpuEngine::clear_worker_resources();
            }
            *slot = Some(eng);
            0
        }
        Err(e) => {
            eprintln!("[wgpu] init failed: {e}");
            -1
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn cp_wgpu_shutdown() {
    let mut slot = engine_slot().lock();
    if slot.take().is_some() {
        GpuEngine::clear_worker_resources();
    }
}

#[no_mangle]
pub unsafe extern "C" fn cp_wgpu_is_ready() -> c_int {
    if engine_slot().lock().is_some() {
        1
    } else {
        0
    }
}

/// Search `[start, start+count)` for a hash below `target_be`.
///
/// `difficulty_u64` must be non-zero (pool difficulty). `target_be` is the
/// authoritative 64-byte big-endian target from the pool.
///
/// `cancel_check`: optional `() -> int` returning non-zero to cancel.
///
/// On found: writes BE nonce/hash and returns `CP_WGPU_OK_FOUND`.
#[no_mangle]
pub unsafe extern "C" fn cp_wgpu_search_range(
    header: *const u8,
    difficulty_u64: u64,
    target_be: *const u8,
    start_nonce_be: *const u8,
    count: u64,
    out_nonce_be: *mut u8,
    out_hash_be: *mut u8,
    out_hashes: *mut u64,
    cancel_check: Option<unsafe extern "C" fn() -> c_int>,
) -> c_int {
    if header.is_null()
        || target_be.is_null()
        || start_nonce_be.is_null()
        || out_hashes.is_null()
        || difficulty_u64 == 0
        || count == 0
    {
        return CP_WGPU_ERROR;
    }

    let mut hdr = [0u8; 32];
    std::ptr::copy_nonoverlapping(header, hdr.as_mut_ptr(), 32);
    let mut tgt_bytes = [0u8; 64];
    std::ptr::copy_nonoverlapping(target_be, tgt_bytes.as_mut_ptr(), 64);
    let mut start_bytes = [0u8; 64];
    std::ptr::copy_nonoverlapping(start_nonce_be, start_bytes.as_mut_ptr(), 64);

    let mut ctx = JobContext::new(hdr, U512::from(difficulty_u64));
    ctx.target = U512::from_big_endian(&tgt_bytes);

    let start = U512::from_big_endian(&start_bytes);
    let end = start.saturating_add(U512::from(count.saturating_sub(1)));
    let range = Range { start, end };
    let cancel = CCancelCheck {
        check: cancel_check,
    };

    let status = {
        let slot = engine_slot().lock();
        let Some(eng) = slot.as_ref() else {
            return CP_WGPU_ERROR;
        };
        eng.search_range(&ctx, range, &cancel)
    };

    match status {
        EngineStatus::Found {
            candidate,
            hash_count,
            ..
        } => {
            *out_hashes = hash_count;
            if !out_nonce_be.is_null() {
                std::ptr::copy_nonoverlapping(candidate.work.as_ptr(), out_nonce_be, 64);
            }
            if !out_hash_be.is_null() {
                let hb = candidate.hash.to_big_endian();
                std::ptr::copy_nonoverlapping(hb.as_ptr(), out_hash_be, 64);
            }
            CP_WGPU_OK_FOUND
        }
        EngineStatus::Exhausted { hash_count } => {
            *out_hashes = hash_count;
            CP_WGPU_OK_EXHAUSTED
        }
        EngineStatus::Cancelled { hash_count } | EngineStatus::Running { hash_count } => {
            *out_hashes = hash_count;
            CP_WGPU_CANCELLED
        }
        EngineStatus::DeviceLost { hash_count } => {
            *out_hashes = hash_count;
            CP_WGPU_DEVICE_LOST
        }
    }
}
