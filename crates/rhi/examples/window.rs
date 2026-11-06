//! Opens a window and, every frame, runs compute work through the RHI, checks the result on the
//! CPU, and presents a cleared image: green while every check passes, red otherwise.
//!
//! `cargo run -p engine-rhi --example window -- [frames] ` (0 = until closed; default 300)
//! `RHI_BACKEND=vulkan|metal` picks the backend, `RHI_VALIDATION=1` enables validation.
//! Prints a frame-time and memory report as JSON when it exits.

use std::sync::Arc;

use engine_core::perf::{self, CountingAlloc, FrameRecorder, Report};
use engine_rhi::*;
use engine_shaders as sh;
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::raw_window_handle::HasDisplayHandle;
use winit::window::{Window, WindowId};

#[global_allocator]
static ALLOC: CountingAlloc = CountingAlloc;

const COUNT: u32 = 1000;
const SLOTS: usize = 1024;

struct Gpu<D: Device> {
    device: D,
    surface: D::Surface,
    window: Arc<Window>,
    iota: D::ComputePipeline,
    atomic: Option<D::ComputePipeline>,
    out: D::Buffer,
    vis: D::Buffer,
    out_down: D::Buffer,
    vis_down: D::Buffer,
}

struct App<D: Device> {
    gpu: Option<Gpu<D>>,
    recorder: FrameRecorder,
    frames_left: Option<u64>,
    frame: u64,
    ok: bool,
}

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

fn pack(addr: u64, count: u32) -> [u8; 16] {
    let mut p = [0u8; 16];
    p[..8].copy_from_slice(&addr.to_le_bytes());
    p[8..12].copy_from_slice(&count.to_le_bytes());
    p
}

impl<D: Device> Gpu<D> {
    fn new(event_loop: &ActiveEventLoop) -> Self {
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("engine rhi smoke")
                        .with_inner_size(LogicalSize::new(960, 540)),
                )
                .unwrap(),
        );
        let display = event_loop.display_handle().unwrap().as_raw();
        let validation = std::env::var_os("RHI_VALIDATION").is_some();
        let device = D::new(&DeviceDesc {
            display: Some(display),
            validation,
        })
        .unwrap();
        eprintln!(
            "device: {} ({:?}), int64 atomics: {}",
            device.info().name,
            device.info().backend,
            device.info().int64_atomics
        );
        let size = window.inner_size();
        let surface = device
            .create_surface(&*window, (size.width, size.height))
            .unwrap();
        let iota = device
            .create_compute_pipeline(&ComputePipelineDesc {
                code: shader!(IOTA_SPIRV, IOTA_METALLIB),
                entry: "iotaMain",
                workgroup_size: [64, 1, 1],
                push_constant_bytes: 16,
            })
            .unwrap();
        let atomic = device.info().int64_atomics.then(|| {
            device
                .create_compute_pipeline(&ComputePipelineDesc {
                    code: shader!(ATOMIC64_SPIRV, ATOMIC64_METALLIB),
                    entry: "visibilityMain",
                    workgroup_size: [64, 1, 1],
                    push_constant_bytes: 16,
                })
                .unwrap()
        });
        let buf = |size, location| {
            device
                .create_buffer(&BufferDesc { size, location })
                .unwrap()
        };
        Self {
            out: buf((SLOTS * 4) as u64, MemoryLocation::GpuOnly),
            vis: buf(16 * 8, MemoryLocation::GpuOnly),
            out_down: buf((SLOTS * 4) as u64, MemoryLocation::Readback),
            vis_down: buf(16 * 8, MemoryLocation::Readback),
            device,
            surface,
            window,
            iota,
            atomic,
        }
    }

    /// Runs both kernels, reads the results back, and reports whether they are correct.
    fn compute_ok(&self) -> bool {
        self.device
            .submit_and_wait(|c| {
                c.fill_buffer(&self.out, 0, (SLOTS * 4) as u64, 0);
                c.fill_buffer(&self.vis, 0, 16 * 8, 0);
                c.barrier();
                c.bind_pipeline(&self.iota);
                c.use_buffer(&self.out, Access::ReadWrite);
                c.push_constants(&pack(self.out.gpu_address(), COUNT));
                c.dispatch([COUNT.div_ceil(64), 1, 1]);
                if let Some(atomic) = &self.atomic {
                    c.bind_pipeline(atomic);
                    c.use_buffer(&self.vis, Access::ReadWrite);
                    c.push_constants(&pack(self.vis.gpu_address(), 1024));
                    c.dispatch([16, 1, 1]);
                }
                c.barrier();
                c.copy_buffer(&self.out, 0, &self.out_down, 0, (SLOTS * 4) as u64);
                c.copy_buffer(&self.vis, 0, &self.vis_down, 0, 16 * 8);
            })
            .is_ok_and(|()| {
                let mut bytes = vec![0u8; SLOTS * 4];
                self.out_down.read(0, &mut bytes).unwrap();
                let iota_ok = bytes.chunks_exact(4).enumerate().all(|(i, c)| {
                    let v = u32::from_le_bytes(c.try_into().unwrap());
                    v == if (i as u32) < COUNT { i as u32 + 1 } else { 0 }
                });
                let mut vis = [0u8; 128];
                self.vis_down.read(0, &mut vis).unwrap();
                let vis_ok = self.atomic.is_none()
                    || (0..16u64).all(|k| {
                        let got =
                            u64::from_le_bytes(vis[k as usize * 8..][..8].try_into().unwrap());
                        let t = 1024 - 16 + k;
                        got == (t << 32) | t
                    });
                iota_ok && vis_ok
            })
    }
}

impl<D: Device> ApplicationHandler for App<D> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gpu.is_none() {
            let gpu = Gpu::<D>::new(event_loop);
            gpu.window.request_redraw();
            self.gpu = Some(gpu);
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(gpu) = self.gpu.as_mut() else { return };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) if size.width > 0 && size.height > 0 => {
                gpu.surface.resize(size.width, size.height).unwrap();
            }
            WindowEvent::RedrawRequested => {
                self.recorder.begin_frame();
                let ok = gpu.compute_ok();
                self.ok &= ok;
                let pulse = 0.5 + 0.5 * (self.frame as f32 * 0.05).sin();
                let color = if ok {
                    [0.0, 0.15 + 0.25 * pulse, 0.05, 1.0]
                } else {
                    [0.6, 0.0, 0.0, 1.0]
                };
                gpu.surface.clear_and_present(color).unwrap();
                self.recorder.end_frame();
                self.frame += 1;
                if let Some(left) = &mut self.frames_left {
                    *left -= 1;
                    if *left == 0 {
                        event_loop.exit();
                        return;
                    }
                }
                gpu.window.request_redraw();
            }
            _ => {}
        }
    }
}

fn run<D: Device>(frames: u64) -> bool {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::<D> {
        gpu: None,
        recorder: FrameRecorder::with_capacity(if frames == 0 {
            1 << 16
        } else {
            frames as usize
        }),
        frames_left: (frames > 0).then_some(frames),
        frame: 0,
        ok: true,
    };
    event_loop.run_app(&mut app).unwrap();
    let report = Report {
        name: "rhi window",
        frames: app.recorder.stats(),
        heap: perf::heap_stats(),
        resident_bytes: perf::resident_bytes(),
    };
    println!("{}", report.to_json());
    println!(
        "compute checks: {}",
        if app.ok { "all passed" } else { "FAILED" }
    );
    app.ok
}

fn main() {
    let frames = std::env::args()
        .nth(1)
        .map_or(300, |s| s.parse().expect("frames must be a number"));
    let ok = match std::env::var("RHI_BACKEND").as_deref() {
        #[cfg(target_vendor = "apple")]
        Ok("metal") => run::<engine_rhi::metal::MetalDevice>(frames),
        #[cfg(feature = "vulkan")]
        Ok("vulkan") => run::<engine_rhi::vulkan::VulkanDevice>(frames),
        Ok(other) => panic!("unknown or unavailable RHI_BACKEND '{other}'"),
        Err(_) => run::<DefaultDevice>(frames),
    };
    std::process::exit(if ok { 0 } else { 1 });
}
