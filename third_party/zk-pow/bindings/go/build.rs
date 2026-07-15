use std::env;
use std::path::PathBuf;

fn main() {
    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let out_dir = PathBuf::from(&crate_dir);

    // Generate C header file using cbindgen.toml configuration
    cbindgen::generate(&crate_dir)
        .expect("Unable to generate bindings")
        .write_to_file(out_dir.join("zk_pow_ffi.h"));
}
