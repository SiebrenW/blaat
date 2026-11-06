//! Frame-time and memory measurement, so performance regressions are visible from the first
//! milestone (ADR 0006). Frame times are recorded into a pre-sized buffer, so recording does not
//! allocate.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::time::{Duration, Instant};

/// Global allocator wrapper that counts live heap bytes, the peak, and allocation calls.
///
/// Opt-in: a binary installs it with
/// `#[global_allocator] static A: engine_core::perf::CountingAlloc = engine_core::perf::CountingAlloc;`
/// Cost is three relaxed atomic operations per allocation, so use it in measurement builds.
pub struct CountingAlloc;

static CURRENT: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

fn grow(bytes: usize) {
    let now = CURRENT.fetch_add(bytes, Relaxed) + bytes;
    PEAK.fetch_max(now, Relaxed);
}

// SAFETY: forwards every call to `System` unchanged and only adds bookkeeping.
unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            ALLOCATIONS.fetch_add(1, Relaxed);
            grow(layout.size());
        }
        p
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc_zeroed(layout) };
        if !p.is_null() {
            ALLOCATIONS.fetch_add(1, Relaxed);
            grow(layout.size());
        }
        p
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        CURRENT.fetch_sub(layout.size(), Relaxed);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let p = unsafe { System.realloc(ptr, layout, new_size) };
        if !p.is_null() {
            ALLOCATIONS.fetch_add(1, Relaxed);
            if new_size >= layout.size() {
                grow(new_size - layout.size());
            } else {
                CURRENT.fetch_sub(layout.size() - new_size, Relaxed);
            }
        }
        p
    }
}

/// Heap usage as seen by [`CountingAlloc`]. All zero when it is not installed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeapStats {
    pub current_bytes: usize,
    pub peak_bytes: usize,
    pub allocations: usize,
}

pub fn heap_stats() -> HeapStats {
    HeapStats {
        current_bytes: CURRENT.load(Relaxed),
        peak_bytes: PEAK.load(Relaxed),
        allocations: ALLOCATIONS.load(Relaxed),
    }
}

/// Resident physical memory of the process in bytes, or `None` if the OS query fails.
pub fn resident_bytes() -> Option<usize> {
    memory_stats::memory_stats().map(|m| m.physical_mem)
}

/// Frame-time distribution in milliseconds. Percentiles use the nearest-rank method.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameStats {
    pub frames: usize,
    pub min_ms: f64,
    pub mean_ms: f64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub max_ms: f64,
}

impl FrameStats {
    /// `None` for an empty slice. Non-finite samples are not supported.
    pub fn from_ms(samples: &[f64]) -> Option<Self> {
        if samples.is_empty() {
            return None;
        }
        let mut sorted = samples.to_vec();
        sorted.sort_by(f64::total_cmp);
        let n = sorted.len();
        let rank = |p: f64| sorted[((p / 100.0 * n as f64).ceil() as usize).clamp(1, n) - 1];
        Some(Self {
            frames: n,
            min_ms: sorted[0],
            mean_ms: sorted.iter().sum::<f64>() / n as f64,
            p50_ms: rank(50.0),
            p95_ms: rank(95.0),
            p99_ms: rank(99.0),
            max_ms: sorted[n - 1],
        })
    }
}

/// Records frame durations into a buffer allocated up front.
pub struct FrameRecorder {
    samples_ms: Vec<f64>,
    started: Option<Instant>,
}

impl FrameRecorder {
    pub fn with_capacity(frames: usize) -> Self {
        Self {
            samples_ms: Vec::with_capacity(frames),
            started: None,
        }
    }

    pub fn begin_frame(&mut self) {
        self.started = Some(Instant::now());
    }

    /// Ends the frame started by [`begin_frame`](Self::begin_frame). No-op without a begin.
    pub fn end_frame(&mut self) {
        if let Some(t) = self.started.take() {
            self.record(t.elapsed());
        }
    }

    pub fn record(&mut self, frame: Duration) {
        self.samples_ms.push(frame.as_secs_f64() * 1000.0);
    }

    pub fn stats(&self) -> Option<FrameStats> {
        FrameStats::from_ms(&self.samples_ms)
    }
}

/// One measurement run, serializable as a single JSON object.
pub struct Report<'a> {
    pub name: &'a str,
    pub frames: Option<FrameStats>,
    pub heap: HeapStats,
    pub resident_bytes: Option<usize>,
}

impl Report<'_> {
    pub fn to_json(&self) -> String {
        let frames = match &self.frames {
            Some(f) => format!(
                "{{\"frames\":{},\"min_ms\":{:.4},\"mean_ms\":{:.4},\"p50_ms\":{:.4},\"p95_ms\":{:.4},\"p99_ms\":{:.4},\"max_ms\":{:.4}}}",
                f.frames, f.min_ms, f.mean_ms, f.p50_ms, f.p95_ms, f.p99_ms, f.max_ms
            ),
            None => "null".into(),
        };
        let resident = self.resident_bytes.map_or("null".into(), |b| b.to_string());
        format!(
            "{{\"name\":\"{}\",\"frame_time\":{},\"heap\":{{\"current_bytes\":{},\"peak_bytes\":{},\"allocations\":{}}},\"resident_bytes\":{}}}",
            self.name.escape_default(),
            frames,
            self.heap.current_bytes,
            self.heap.peak_bytes,
            self.heap.allocations,
            resident
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_rank_percentiles() {
        let samples: Vec<f64> = (1..=100).rev().map(f64::from).collect();
        let s = FrameStats::from_ms(&samples).unwrap();
        assert_eq!(
            (s.min_ms, s.p50_ms, s.p95_ms, s.p99_ms, s.max_ms),
            (1.0, 50.0, 95.0, 99.0, 100.0)
        );
        assert_eq!(s.mean_ms, 50.5);
        assert_eq!(s.frames, 100);
    }

    #[test]
    fn single_sample_and_empty() {
        let s = FrameStats::from_ms(&[7.0]).unwrap();
        assert_eq!(
            (s.min_ms, s.p50_ms, s.p99_ms, s.max_ms),
            (7.0, 7.0, 7.0, 7.0)
        );
        assert!(FrameStats::from_ms(&[]).is_none());
    }

    #[test]
    fn one_slow_frame_shows_in_p99_not_p50() {
        let mut samples = vec![1.0; 99];
        samples.push(50.0);
        let s = FrameStats::from_ms(&samples).unwrap();
        assert_eq!(s.p50_ms, 1.0);
        assert_eq!(s.p99_ms, 1.0);
        assert_eq!(s.max_ms, 50.0);
        samples.push(50.0);
        samples.push(50.0);
        assert_eq!(FrameStats::from_ms(&samples).unwrap().p99_ms, 50.0);
    }

    #[test]
    fn recorder_does_not_reallocate_within_capacity() {
        let mut r = FrameRecorder::with_capacity(8);
        let ptr = r.samples_ms.as_ptr();
        for _ in 0..8 {
            r.begin_frame();
            r.end_frame();
        }
        assert_eq!(r.samples_ms.as_ptr(), ptr);
        assert_eq!(r.stats().unwrap().frames, 8);
    }

    #[test]
    fn end_without_begin_records_nothing() {
        let mut r = FrameRecorder::with_capacity(1);
        r.end_frame();
        assert!(r.stats().is_none());
    }

    #[test]
    fn json_escapes_name_and_nulls_missing() {
        let r = Report {
            name: "a\"b",
            frames: None,
            heap: HeapStats {
                current_bytes: 1,
                peak_bytes: 2,
                allocations: 3,
            },
            resident_bytes: None,
        };
        let j = r.to_json();
        assert!(j.contains(r#""name":"a\"b""#));
        assert!(j.contains(r#""frame_time":null"#));
        assert!(j.contains(r#""resident_bytes":null"#));
    }
}
