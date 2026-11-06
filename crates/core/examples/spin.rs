//! Smoke run of the measurement harness: a CPU workload with some allocation, 300 frames.
//! `cargo run --release -p engine-core --example spin`

use engine_core::perf::{self, CountingAlloc, FrameRecorder, Report};

#[global_allocator]
static ALLOC: CountingAlloc = CountingAlloc;

fn main() {
    const FRAMES: usize = 300;
    let mut rec = FrameRecorder::with_capacity(FRAMES);
    let mut keep: Vec<Vec<u8>> = Vec::new();
    let mut acc = 0u64;
    for frame in 0..FRAMES {
        rec.begin_frame();
        for i in 0..2_000_000u64 {
            acc = acc.wrapping_mul(6364136223846793005).wrapping_add(i);
        }
        if frame % 50 == 0 {
            keep.push(vec![0u8; 1 << 20]); // 1 MiB retained
        }
        rec.end_frame();
    }
    std::hint::black_box(acc);
    let report = Report {
        name: "spin",
        frames: rec.stats(),
        heap: perf::heap_stats(),
        resident_bytes: perf::resident_bytes(),
    };
    println!("{}", report.to_json());
}
