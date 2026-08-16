//! Embeds `catalog/**/*.toml` into the crate as `catalog::ENTRIES` so the
//! hardware and model presets ship with the library (native and wasm) and
//! resolve without filesystem access.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out = Path::new(&env::var("OUT_DIR").unwrap()).join("catalog_entries.rs");
    let mut code = String::from(
        "/// (kind, name, TOML source) for every shipped catalog entry.\n\
         pub(crate) static ENTRIES: &[(&str, &str, &str)] = &[\n",
    );
    for kind in ["hardware", "models"] {
        let dir = manifest.join("catalog").join(kind);
        println!("cargo:rerun-if-changed={}", dir.display());
        let mut files: Vec<PathBuf> = fs::read_dir(&dir)
            .unwrap_or_else(|e| panic!("catalog dir {}: {e}", dir.display()))
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "toml"))
            .collect();
        files.sort();
        for p in files {
            println!("cargo:rerun-if-changed={}", p.display());
            let name = p.file_stem().unwrap().to_str().unwrap();
            code.push_str(&format!(
                "    ({kind:?}, {name:?}, include_str!({:?})),\n",
                p.display()
            ));
        }
    }
    code.push_str("];\n");
    fs::write(&out, code).unwrap();
}
