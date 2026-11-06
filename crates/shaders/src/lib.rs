//! Engine shaders, compiled from `shaders/*.slang` at build time (ADR 0008).
//! Statics are named `<FILE>_<TARGET>`: `SPIRV` (Vulkan), `MSL` (Metal source), `METALLIB`
//! (compiled Metal library, only on macOS hosts), and `DXIL` (D3D12, only on Windows hosts,
//! where Slang can load DXC).

include!(concat!(env!("OUT_DIR"), "/shaders.rs"));

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atomic64_max_compiles_on_every_target() {
        assert_eq!(&ATOMIC64_SPIRV[..4], &0x0723_0203u32.to_le_bytes());
        // OpAtomicUMax (opcode 239) must survive in the SPIR-V: the Metal workaround in
        // shaders/include/atomics.slang must not leak into other targets.
        let words: Vec<u32> = ATOMIC64_SPIRV
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let mut i = 5; // skip the 5-word header
        let mut found = false;
        while i < words.len() {
            let (op, len) = (words[i] & 0xffff, (words[i] >> 16) as usize);
            found |= op == 239;
            i += len.max(1);
        }
        assert!(found, "no OpAtomicUMax in SPIR-V");

        let msl = std::str::from_utf8(ATOMIC64_MSL).unwrap();
        assert!(msl.contains("atomic_max_explicit"));
        assert!(!msl.contains("atomic_fetch_max_explicit"));
        #[cfg(target_os = "macos")]
        assert_eq!(&ATOMIC64_METALLIB[..4], b"MTLB");
        #[cfg(windows)]
        assert!(ATOMIC64_DXIL.starts_with(b"DXBC"));
    }
}
