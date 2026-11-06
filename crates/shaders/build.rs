//! Compiles every `shaders/*.slang` to SPIR-V and MSL (plus DXIL on Windows hosts, where Slang
//! can load the DXC compiler) and generates `shaders.rs` with the bytes embedded. On macOS hosts
//! the MSL is also compiled with Apple's Metal compiler to a `.metallib`, so MSL that Slang emits
//! but Metal rejects fails the build instead of surfacing at runtime.
//! Shared Slang modules live in `shaders/include/` and are imported by name.
//! Slang is found through `SLANGC`, else `slangc` on PATH. A missing compiler or a failed
//! compile fails the build.

use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

struct Target {
    slang: &'static str,
    ext: &'static str,
    suffix: &'static str,
    extra: &'static [&'static str],
}

const SPIRV: Target = Target {
    slang: "spirv",
    ext: "spv",
    suffix: "SPIRV",
    // Slang renames SPIR-V entry points to `main` unless told otherwise; the RHI looks them up
    // by their source name, as it does on Metal.
    extra: &["-fvk-use-entrypoint-name"],
};
const MSL: Target = Target {
    slang: "metal",
    ext: "metal",
    suffix: "MSL",
    extra: &[],
};
const DXIL: Target = Target {
    slang: "dxil",
    ext: "dxil",
    suffix: "DXIL",
    extra: &["-profile", "sm_6_6"],
};

fn main() {
    println!("cargo:rerun-if-env-changed=SLANGC");
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let src_dir = manifest.join("../../shaders");
    let include_dir = src_dir.join("include");
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let slangc = env::var_os("SLANGC")
        .map(PathBuf::from)
        .unwrap_or_else(|| "slangc".into());
    println!("cargo:rerun-if-changed={}", src_dir.display());
    println!("cargo:rerun-if-changed={}", include_dir.display());
    for module in list(&include_dir, "slang") {
        println!("cargo:rerun-if-changed={}", module.display());
    }

    let mut targets = vec![&SPIRV, &MSL];
    if cfg!(windows) {
        targets.push(&DXIL);
    }

    let mut generated = String::new();
    for src in list(&src_dir, "slang") {
        println!("cargo:rerun-if-changed={}", src.display());
        let stem = src.file_stem().unwrap().to_str().unwrap();
        let name = stem.to_uppercase();
        for t in &targets {
            let out = out_dir.join(format!("{stem}.{}", t.ext));
            compile(&slangc, &include_dir, &src, t, &out);
            embed(&mut generated, &name, t.suffix, &out);
        }
        if cfg!(target_os = "macos") {
            let msl = out_dir.join(format!("{stem}.metal"));
            let lib = out_dir.join(format!("{stem}.metallib"));
            metal_compile(&msl, &lib);
            embed(&mut generated, &name, "METALLIB", &lib);
        }
    }
    fs::write(out_dir.join("shaders.rs"), generated).unwrap();
}

fn list(dir: &Path, ext: &str) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()))
        .map(|e| e.unwrap().path())
        .filter(|p| p.is_file() && p.extension().is_some_and(|e| e == ext))
        .collect();
    files.sort();
    files
}

fn embed(generated: &mut String, name: &str, suffix: &str, file: &Path) {
    writeln!(
        generated,
        "pub static {name}_{suffix}: &[u8] = include_bytes!({:?});",
        file.to_str().unwrap()
    )
    .unwrap();
}

fn run(mut cmd: Command, what: &str, hint: &str) {
    let output = cmd
        .output()
        .unwrap_or_else(|e| panic!("cannot run {what} ({hint}): {e}"));
    if !output.status.success() {
        panic!(
            "{what} failed:\n{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

fn compile(slangc: &Path, include_dir: &Path, src: &Path, t: &Target, out: &Path) {
    let mut cmd = Command::new(slangc);
    cmd.arg(src)
        .arg("-I")
        .arg(include_dir)
        .args(["-target", t.slang])
        .args(t.extra)
        .arg("-o")
        .arg(out);
    run(
        cmd,
        &format!("slangc {} -> {}", src.display(), t.slang),
        "set SLANGC or put slangc on PATH",
    );
}

fn metal_compile(msl: &Path, metallib: &Path) {
    let air = metallib.with_extension("air");
    let hint = "needs Xcode's Metal toolchain: `xcodebuild -downloadComponent MetalToolchain`";
    let mut cmd = Command::new("xcrun");
    cmd.args(["-sdk", "macosx", "metal", "-c"])
        .arg(msl)
        .arg("-o")
        .arg(&air);
    run(cmd, &format!("metal {}", msl.display()), hint);
    let mut cmd = Command::new("xcrun");
    cmd.args(["-sdk", "macosx", "metallib"])
        .arg(&air)
        .arg("-o")
        .arg(metallib);
    run(cmd, &format!("metallib {}", air.display()), hint);
}
