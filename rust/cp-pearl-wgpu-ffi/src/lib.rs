//! C ABI for Pearl wgpu fused GEMM+jackpot mining.

mod engine;

use engine::{pearl_a_noise_seed_from_hash, list_devices, PearlEngine, ScanOutcome};
use parking_lot::Mutex;
use std::os::raw::c_int;
use std::slice;
use std::sync::OnceLock;

static ENGINE: OnceLock<Mutex<Option<PearlEngine>>> = OnceLock::new();

fn engine_slot() -> &'static Mutex<Option<PearlEngine>> {
    ENGINE.get_or_init(|| Mutex::new(None))
}

#[no_mangle]
pub unsafe extern "C" fn cp_pearl_wgpu_list_devices() -> c_int {
    list_devices()
}

#[no_mangle]
pub unsafe extern "C" fn cp_pearl_wgpu_init(devices: *const c_int, ndev: c_int) -> c_int {
    let device_index = if !devices.is_null() && ndev > 0 {
        let slice = slice::from_raw_parts(devices, ndev as usize);
        let d = slice[0];
        if d < 0 {
            eprintln!("[pearl-wgpu] invalid device index {d}");
            return -1;
        }
        Some(d as usize)
    } else {
        None
    };

    match PearlEngine::try_new(device_index) {
        Ok(eng) => {
            let mut slot = engine_slot().lock();
            *slot = Some(eng);
            0
        }
        Err(e) => {
            eprintln!("[pearl-wgpu] init failed: {e}");
            -1
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn cp_pearl_wgpu_shutdown() {
    let mut slot = engine_slot().lock();
    *slot = None;
}

#[no_mangle]
pub unsafe extern "C" fn cp_pearl_wgpu_is_ready() -> c_int {
    if engine_slot().lock().is_some() {
        1
    } else {
        0
    }
}

#[no_mangle]
pub unsafe extern "C" fn cp_pearl_wgpu_begin_job(
    m: c_int,
    n: c_int,
    k: c_int,
    b_noise_seed: *const u8,
) -> c_int {
    if b_noise_seed.is_null() {
        eprintln!("[pearl-wgpu] begin_job: null b_noise_seed");
        return -1;
    }
    let mut seed = [0u8; 32];
    std::ptr::copy_nonoverlapping(b_noise_seed, seed.as_mut_ptr(), 32);

    let mut slot = engine_slot().lock();
    let Some(eng) = slot.as_mut() else {
        eprintln!("[pearl-wgpu] begin_job: not initialized");
        return -1;
    };
    match eng.begin_job(m, n, k, &seed) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("[pearl-wgpu] begin_job failed: {e}");
            -1
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn cp_pearl_wgpu_prep_a_signal(
    ab_seed: *const u8,
    ab_seed_len: c_int,
    job_key: *const u8,
    hash_a_out: *mut u8,
) -> c_int {
    if ab_seed.is_null() || ab_seed_len < 0 || job_key.is_null() || hash_a_out.is_null() {
        eprintln!("[pearl-wgpu] prep_a_signal: bad args");
        return -1;
    }
    let seed = slice::from_raw_parts(ab_seed, ab_seed_len as usize);
    let mut key = [0u8; 32];
    std::ptr::copy_nonoverlapping(job_key, key.as_mut_ptr(), 32);
    let mut hash = [0u8; 32];

    let mut slot = engine_slot().lock();
    let Some(eng) = slot.as_mut() else {
        eprintln!("[pearl-wgpu] prep_a_signal: not initialized");
        return -1;
    };
    match eng.prep_a_signal(seed, &key, &mut hash) {
        Ok(()) => {
            std::ptr::copy_nonoverlapping(hash.as_ptr(), hash_a_out, 32);
            0
        }
        Err(e) => {
            eprintln!("[pearl-wgpu] prep_a_signal failed: {e}");
            -1
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn cp_pearl_wgpu_a_noise_seed_from_hash(
    b_noise_seed: *const u8,
    hash_a: *const u8,
    m: u32,
    salted: c_int,
    a_noise_seed: *mut u8,
) {
    if b_noise_seed.is_null() || hash_a.is_null() || a_noise_seed.is_null() {
        eprintln!("[pearl-wgpu] a_noise_seed_from_hash: null ptr");
        return;
    }
    let mut b = [0u8; 32];
    let mut h = [0u8; 32];
    std::ptr::copy_nonoverlapping(b_noise_seed, b.as_mut_ptr(), 32);
    std::ptr::copy_nonoverlapping(hash_a, h.as_mut_ptr(), 32);
    let out = pearl_a_noise_seed_from_hash(&b, &h, m, salted != 0);
    std::ptr::copy_nonoverlapping(out.as_ptr(), a_noise_seed, 32);
}

#[no_mangle]
pub unsafe extern "C" fn cp_pearl_wgpu_prepack_a(a_noise_seed: *const u8) -> c_int {
    if a_noise_seed.is_null() {
        eprintln!("[pearl-wgpu] prepack_a: null seed");
        return -1;
    }
    let mut seed = [0u8; 32];
    std::ptr::copy_nonoverlapping(a_noise_seed, seed.as_mut_ptr(), 32);

    let mut slot = engine_slot().lock();
    let Some(eng) = slot.as_mut() else {
        eprintln!("[pearl-wgpu] prepack_a: not initialized");
        return -1;
    };
    match eng.prepack_a(&seed) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("[pearl-wgpu] prepack_a failed: {e}");
            -1
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn cp_pearl_wgpu_scan(
    a_key8: *const u32,
    bound: *const u32,
    macro_batch: c_int,
    out_found: *mut c_int,
    out_t_rows: *mut c_int,
    out_t_cols: *mut c_int,
    out_tiles_scanned: *mut u64,
    cancel_check: Option<unsafe extern "C" fn() -> c_int>,
    on_progress: Option<unsafe extern "C" fn(u64)>,
) -> c_int {
    if a_key8.is_null() || bound.is_null() {
        eprintln!("[pearl-wgpu] scan: null key/bound");
        return -1;
    }
    let mut key = [0u32; 8];
    let mut bnd = [0u32; 8];
    std::ptr::copy_nonoverlapping(a_key8, key.as_mut_ptr(), 8);
    std::ptr::copy_nonoverlapping(bound, bnd.as_mut_ptr(), 8);

    if !out_found.is_null() {
        *out_found = 0;
    }
    if !out_t_rows.is_null() {
        *out_t_rows = -1;
    }
    if !out_t_cols.is_null() {
        *out_t_cols = -1;
    }
    if !out_tiles_scanned.is_null() {
        *out_tiles_scanned = 0;
    }

    let cancel = || match cancel_check {
        Some(f) => unsafe { f() != 0 },
        None => false,
    };
    let progress = |tiles: u64| {
        if let Some(f) = on_progress {
            unsafe { f(tiles) }
        }
        if !out_tiles_scanned.is_null() {
            unsafe {
                *out_tiles_scanned = tiles;
            }
        }
    };

    let mut slot = engine_slot().lock();
    let Some(eng) = slot.as_mut() else {
        eprintln!("[pearl-wgpu] scan: not initialized");
        return -1;
    };

    match eng.scan(&key, &bnd, macro_batch, &cancel, &progress) {
        Ok(ScanOutcome::Found {
            t_rows,
            t_cols,
            tiles_scanned,
        }) => {
            if !out_found.is_null() {
                *out_found = 1;
            }
            if !out_t_rows.is_null() {
                *out_t_rows = t_rows;
            }
            if !out_t_cols.is_null() {
                *out_t_cols = t_cols;
            }
            if !out_tiles_scanned.is_null() {
                *out_tiles_scanned = tiles_scanned;
            }
            1
        }
        Ok(ScanOutcome::Exhausted { tiles_scanned }) => {
            if !out_tiles_scanned.is_null() {
                *out_tiles_scanned = tiles_scanned;
            }
            0
        }
        Ok(ScanOutcome::Cancelled) => -1,
        Err(e) => {
            eprintln!("[pearl-wgpu] scan failed: {e}");
            -1
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn cp_pearl_wgpu_download_a_sig(out: *mut i8, elems: usize) -> c_int {
    if out.is_null() {
        eprintln!("[pearl-wgpu] download_a_sig: null out");
        return -1;
    }
    let slice = slice::from_raw_parts_mut(out, elems);
    let slot = engine_slot().lock();
    let Some(eng) = slot.as_ref() else {
        eprintln!("[pearl-wgpu] download_a_sig: not initialized");
        return -1;
    };
    match eng.download_a_sig(slice) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("[pearl-wgpu] download_a_sig failed: {e}");
            -1
        }
    }
}
