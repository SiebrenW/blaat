//! Behaviour every backend must share. Each test is generic over `Device` and instantiated once
//! per backend below.

use engine_rhi::*;
use engine_shaders as sh;

macro_rules! shader {
    ($spirv:ident, $metallib:ident) => {
        ShaderCode {
            spirv: sh::$spirv,
            #[cfg(target_os = "macos")]
            metallib: Some(sh::$metallib),
            #[cfg(not(target_os = "macos"))]
            metallib: None,
        }
    };
}

const GROUP: u32 = 64;

fn iota_pipeline<D: Device>(d: &D) -> D::ComputePipeline {
    d.create_compute_pipeline(&ComputePipelineDesc {
        code: shader!(IOTA_SPIRV, IOTA_METALLIB),
        entry: "iotaMain",
        workgroup_size: [GROUP, 1, 1],
        push_constant_bytes: 16,
    })
    .unwrap()
}

fn args_pipeline<D: Device>(d: &D) -> D::ComputePipeline {
    d.create_compute_pipeline(&ComputePipelineDesc {
        code: shader!(DISPATCH_ARGS_SPIRV, DISPATCH_ARGS_METALLIB),
        entry: "argsMain",
        workgroup_size: [1, 1, 1],
        push_constant_bytes: 24,
    })
    .unwrap()
}

fn atomic_pipeline<D: Device>(d: &D) -> D::ComputePipeline {
    d.create_compute_pipeline(&ComputePipelineDesc {
        code: shader!(ATOMIC64_SPIRV, ATOMIC64_METALLIB),
        entry: "visibilityMain",
        workgroup_size: [GROUP, 1, 1],
        push_constant_bytes: 16,
    })
    .unwrap()
}

enum Arg {
    U32(u32),
    U64(u64),
}

/// Lays arguments out like a C struct: 8-byte values 8-aligned, total rounded up to 8.
fn pack(args: &[Arg]) -> Vec<u8> {
    let mut out = Vec::new();
    for a in args {
        match a {
            Arg::U32(v) => out.extend_from_slice(&v.to_le_bytes()),
            Arg::U64(v) => {
                while out.len() % 8 != 0 {
                    out.push(0);
                }
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
    }
    while out.len() % 8 != 0 {
        out.push(0);
    }
    out
}

fn buffer<D: Device>(d: &D, size: u64, location: MemoryLocation) -> D::Buffer {
    d.create_buffer(&BufferDesc { size, location }).unwrap()
}

fn read_u32s<B: Buffer>(b: &B, count: usize) -> Vec<u32> {
    let mut bytes = vec![0u8; count * 4];
    b.read(0, &mut bytes).unwrap();
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn device<D: Device>() -> D {
    // RHI_VALIDATION=1 turns on the backend's validation layer; a validation error then fails
    // the test that triggered it.
    let validation = std::env::var_os("RHI_VALIDATION").is_some();
    D::new(&DeviceDesc {
        display: None,
        validation,
    })
    .unwrap()
}

fn copy_roundtrip<D: Device>() {
    let d = device::<D>();
    let data: Vec<u8> = (0..=255u8).cycle().take(4096).collect();
    let up = buffer(&d, 4096, MemoryLocation::Upload);
    let gpu = buffer(&d, 4096, MemoryLocation::GpuOnly);
    let down = buffer(&d, 4096, MemoryLocation::Readback);
    up.write(0, &data).unwrap();
    d.submit_and_wait(|c| {
        c.copy_buffer(&up, 0, &gpu, 0, 4096);
        c.barrier();
        c.copy_buffer(&gpu, 1024, &down, 0, 2048);
    })
    .unwrap();
    let mut out = vec![0u8; 2048];
    down.read(0, &mut out).unwrap();
    assert_eq!(out, data[1024..3072]);
}

fn fill_ranges<D: Device>() {
    let d = device::<D>();
    let gpu = buffer(&d, 64, MemoryLocation::GpuOnly);
    let down = buffer(&d, 64, MemoryLocation::Readback);
    d.submit_and_wait(|c| {
        c.fill_buffer(&gpu, 0, 64, 0x11);
        c.barrier();
        c.fill_buffer(&gpu, 16, 32, 0xAB);
        c.barrier();
        c.copy_buffer(&gpu, 0, &down, 0, 64);
    })
    .unwrap();
    let mut out = [0u8; 64];
    down.read(0, &mut out).unwrap();
    assert!(out[..16].iter().all(|&b| b == 0x11));
    assert!(out[16..48].iter().all(|&b| b == 0xAB));
    assert!(out[48..].iter().all(|&b| b == 0x11));
}

fn host_access_errors<D: Device>() {
    let d = device::<D>();
    let up = buffer(&d, 16, MemoryLocation::Upload);
    let gpu = buffer(&d, 16, MemoryLocation::GpuOnly);
    let down = buffer(&d, 16, MemoryLocation::Readback);
    assert!(up.write(8, &[0; 16]).is_err(), "write past the end");
    assert!(up.write(u64::MAX, &[0; 1]).is_err(), "offset overflow");
    assert!(gpu.write(0, &[0; 4]).is_err(), "write to GpuOnly");
    assert!(up.read(0, &mut [0; 4]).is_err(), "read from Upload");
    assert!(down.read(8, &mut [0; 16]).is_err(), "read past the end");
    assert!(
        d.create_buffer(&BufferDesc {
            size: 0,
            location: MemoryLocation::GpuOnly
        })
        .is_err()
    );
}

/// A dispatch whose group count is computed by an earlier dispatch on the GPU, the pattern
/// culling passes use. Launches exactly `count` useful threads rounded up to whole groups.
fn indirect_dispatch<D: Device>() {
    let d = device::<D>();
    const COUNT: u32 = 1000;
    const SLOTS: usize = 1024; // 16 groups of 64
    let count = buffer(&d, 4, MemoryLocation::Upload);
    count.write(0, &COUNT.to_le_bytes()).unwrap();
    let args = buffer(&d, 12, MemoryLocation::GpuOnly);
    let out = buffer(&d, (SLOTS * 4) as u64, MemoryLocation::GpuOnly);
    let down = buffer(&d, (SLOTS * 4) as u64, MemoryLocation::Readback);
    let (args_p, iota_p) = (args_pipeline(&d), iota_pipeline(&d));

    d.submit_and_wait(|c| {
        c.fill_buffer(&out, 0, (SLOTS * 4) as u64, 0);
        c.bind_pipeline(&args_p);
        c.use_buffer(&args, Access::ReadWrite);
        c.use_buffer(&count, Access::Read);
        c.push_constants(&pack(&[
            Arg::U64(args.gpu_address()),
            Arg::U64(count.gpu_address()),
            Arg::U32(GROUP),
        ]));
        c.dispatch([1, 1, 1]);
        c.barrier();
        c.bind_pipeline(&iota_p);
        c.use_buffer(&out, Access::ReadWrite);
        c.push_constants(&pack(&[Arg::U64(out.gpu_address()), Arg::U32(COUNT)]));
        c.dispatch_indirect(&args, 0);
        c.barrier();
        c.copy_buffer(&out, 0, &down, 0, (SLOTS * 4) as u64);
    })
    .unwrap();

    let got = read_u32s(&down, SLOTS);
    for (i, v) in got.iter().enumerate() {
        let want = if (i as u32) < COUNT { i as u32 + 1 } else { 0 };
        assert_eq!(*v, want, "slot {i}");
    }
}

/// Pipeline, push constants and buffer residency must survive a copy or fill between two
/// dispatches, which on Metal forces a new encoder.
fn state_survives_copy_between_dispatches<D: Device>() {
    let d = device::<D>();
    const N: usize = 128;
    let out = buffer(&d, (N * 4) as u64, MemoryLocation::GpuOnly);
    let down = buffer(&d, (N * 4) as u64, MemoryLocation::Readback);
    let iota_p = iota_pipeline(&d);
    d.submit_and_wait(|c| {
        c.fill_buffer(&out, 0, (N * 4) as u64, 0);
        c.barrier();
        c.bind_pipeline(&iota_p);
        c.use_buffer(&out, Access::ReadWrite);
        c.push_constants(&pack(&[Arg::U64(out.gpu_address()), Arg::U32(N as u32)]));
        c.dispatch([2, 1, 1]);
        c.barrier();
        c.fill_buffer(&out, 0, (N * 4) as u64, 0xFF); // forces an encoder switch
        c.barrier();
        c.dispatch([2, 1, 1]); // no rebinding
        c.barrier();
        c.copy_buffer(&out, 0, &down, 0, (N * 4) as u64);
    })
    .unwrap();
    let got = read_u32s(&down, N);
    assert!(
        got.iter().enumerate().all(|(i, v)| *v == i as u32 + 1),
        "{got:?}"
    );
}

/// The visibility-buffer primitive (ADR 0005): many threads race on 16 slots with a 64-bit max.
fn atomic_max_64<D: Device>() {
    let d = device::<D>();
    if !d.info().int64_atomics {
        eprintln!(
            "SKIP atomic_max_64: {} reports no 64-bit atomics",
            d.info().name
        );
        return;
    }
    const THREADS: u32 = 1024;
    let vis = buffer(&d, 16 * 8, MemoryLocation::GpuOnly);
    let down = buffer(&d, 16 * 8, MemoryLocation::Readback);
    let p = atomic_pipeline(&d);
    d.submit_and_wait(|c| {
        c.fill_buffer(&vis, 0, 16 * 8, 0);
        c.barrier();
        c.bind_pipeline(&p);
        c.use_buffer(&vis, Access::ReadWrite);
        c.push_constants(&pack(&[Arg::U64(vis.gpu_address()), Arg::U32(THREADS)]));
        c.dispatch([THREADS / GROUP, 1, 1]);
        c.barrier();
        c.copy_buffer(&vis, 0, &down, 0, 16 * 8);
    })
    .unwrap();
    let mut bytes = [0u8; 128];
    down.read(0, &mut bytes).unwrap();
    for k in 0..16u64 {
        let got = u64::from_le_bytes(bytes[k as usize * 8..][..8].try_into().unwrap());
        let t = u64::from(THREADS) - 16 + k; // the highest thread id that maps to slot k
        assert_eq!(got, (t << 32) | t, "slot {k}");
    }
}

fn pipeline_errors<D: Device>() {
    let d = device::<D>();
    let bad_entry = d.create_compute_pipeline(&ComputePipelineDesc {
        code: shader!(IOTA_SPIRV, IOTA_METALLIB),
        entry: "doesNotExist",
        workgroup_size: [GROUP, 1, 1],
        push_constant_bytes: 16,
    });
    assert!(bad_entry.is_err());
    let too_big = d.create_compute_pipeline(&ComputePipelineDesc {
        code: shader!(IOTA_SPIRV, IOTA_METALLIB),
        entry: "iotaMain",
        workgroup_size: [GROUP, 1, 1],
        push_constant_bytes: (MAX_PUSH_CONSTANT_BYTES + 4) as u32,
    });
    assert!(too_big.is_err());
}

macro_rules! backend_tests {
    ($module:ident, $device:ty) => {
        mod $module {
            #[test]
            fn copy_roundtrip() {
                super::copy_roundtrip::<$device>()
            }
            #[test]
            fn fill_ranges() {
                super::fill_ranges::<$device>()
            }
            #[test]
            fn host_access_errors() {
                super::host_access_errors::<$device>()
            }
            #[test]
            fn indirect_dispatch() {
                super::indirect_dispatch::<$device>()
            }
            #[test]
            fn state_survives_copy_between_dispatches() {
                super::state_survives_copy_between_dispatches::<$device>()
            }
            #[test]
            fn atomic_max_64() {
                super::atomic_max_64::<$device>()
            }
            #[test]
            fn pipeline_errors() {
                super::pipeline_errors::<$device>()
            }
        }
    };
}

#[cfg(target_vendor = "apple")]
backend_tests!(metal, engine_rhi::metal::MetalDevice);

#[cfg(feature = "vulkan")]
backend_tests!(vulkan, engine_rhi::vulkan::VulkanDevice);
