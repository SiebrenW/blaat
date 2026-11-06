//! Metal backend (Apple GPUs; 64-bit atomic max needs M2 or newer, ADR 0005).

use std::ffi::c_void;
use std::ptr::NonNull;

use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::ProtocolObject;
use objc2_app_kit::NSView;
use objc2_foundation::{NSRange, NSString};
use objc2_metal::{
    MTLBarrierScope, MTLBlitCommandEncoder, MTLBuffer, MTLClearColor, MTLCommandBuffer,
    MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLDispatchType, MTLDrawable,
    MTLGPUFamily, MTLLibrary, MTLLoadAction, MTLPixelFormat, MTLRenderPassDescriptor, MTLResource,
    MTLResourceOptions, MTLResourceUsage, MTLSize, MTLStoreAction,
};
use objc2_quartz_core::{CAMetalDrawable, CAMetalLayer};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawWindowHandle};

use crate::{
    Access, Backend, Buffer, BufferDesc, Commands, ComputePipelineDesc, Device, DeviceDesc,
    DeviceInfo, MAX_PUSH_CONSTANT_BYTES, MemoryLocation, Result, Surface, err,
};

type Dyn<T> = Retained<ProtocolObject<T>>;

pub struct MetalDevice {
    device: Dyn<dyn MTLDevice>,
    queue: Dyn<dyn MTLCommandQueue>,
    info: DeviceInfo,
}

pub struct MetalBuffer {
    raw: Dyn<dyn MTLBuffer>,
    size: u64,
    location: MemoryLocation,
}

pub struct MetalPipeline {
    state: Dyn<dyn MTLComputePipelineState>,
    workgroup: MTLSize,
    push_constant_bytes: usize,
}

impl Buffer for MetalBuffer {
    fn size(&self) -> u64 {
        self.size
    }

    fn gpu_address(&self) -> u64 {
        self.raw.gpuAddress()
    }

    fn write(&self, offset: u64, data: &[u8]) -> Result<()> {
        if self.location != MemoryLocation::Upload {
            return err("write: buffer is not an Upload buffer");
        }
        check_range(self.size, offset, data.len() as u64, "write")?;
        // SAFETY: Shared storage is CPU-visible for the buffer's lifetime, the range was checked
        // against its length, and `data` cannot overlap GPU-owned memory.
        unsafe {
            let dst = self
                .raw
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(offset as usize);
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
        }
        Ok(())
    }

    fn read(&self, offset: u64, out: &mut [u8]) -> Result<()> {
        if self.location != MemoryLocation::Readback {
            return err("read: buffer is not a Readback buffer");
        }
        check_range(self.size, offset, out.len() as u64, "read")?;
        // SAFETY: as in `write`; the caller guarantees the GPU work has completed.
        unsafe {
            let src = self
                .raw
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(offset as usize);
            std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), out.len());
        }
        Ok(())
    }
}

fn check_range(size: u64, offset: u64, len: u64, what: &str) -> Result<()> {
    match offset.checked_add(len) {
        Some(end) if end <= size => Ok(()),
        _ => err(format!(
            "{what}: range {offset}+{len} exceeds buffer size {size}"
        )),
    }
}

impl Device for MetalDevice {
    type Buffer = MetalBuffer;
    type ComputePipeline = MetalPipeline;
    type Commands = MetalCommands;
    type Surface = MetalSurface;

    fn new(_desc: &DeviceDesc) -> Result<Self> {
        // Metal's API validation is controlled by the MTL_DEBUG_LAYER environment variable, so
        // `DeviceDesc::validation` has nothing to enable here.
        let Some(device) = MTLCreateSystemDefaultDevice() else {
            return err("no Metal device");
        };
        let Some(queue) = device.newCommandQueue() else {
            return err("cannot create Metal command queue");
        };
        let info = DeviceInfo {
            name: device.name().to_string(),
            backend: Backend::Metal,
            int64_atomics: device.supportsFamily(MTLGPUFamily::Apple8),
        };
        Ok(Self {
            device,
            queue,
            info,
        })
    }

    fn info(&self) -> &DeviceInfo {
        &self.info
    }

    fn create_buffer(&self, desc: &BufferDesc) -> Result<MetalBuffer> {
        if desc.size == 0 {
            return err("create_buffer: size must be non-zero");
        }
        let options = match desc.location {
            MemoryLocation::GpuOnly => MTLResourceOptions::StorageModePrivate,
            MemoryLocation::Upload => {
                MTLResourceOptions::StorageModeShared
                    | MTLResourceOptions::CPUCacheModeWriteCombined
            }
            MemoryLocation::Readback => MTLResourceOptions::StorageModeShared,
        };
        let Some(raw) = self
            .device
            .newBufferWithLength_options(desc.size as usize, options)
        else {
            return err(format!("cannot allocate {} byte Metal buffer", desc.size));
        };
        Ok(MetalBuffer {
            raw,
            size: desc.size,
            location: desc.location,
        })
    }

    fn create_compute_pipeline(&self, desc: &ComputePipelineDesc) -> Result<MetalPipeline> {
        if desc.push_constant_bytes as usize > MAX_PUSH_CONSTANT_BYTES {
            return err("push_constant_bytes exceeds MAX_PUSH_CONSTANT_BYTES");
        }
        let Some(metallib) = desc.code.metallib else {
            return err("shader has no metallib");
        };
        let data = dispatch2::DispatchData::from_bytes(metallib);
        let library = self.device.newLibraryWithData_error(&data).map_err(|e| {
            crate::Error(format!(
                "cannot load metallib: {}",
                e.localizedDescription()
            ))
        })?;
        let Some(function) = library.newFunctionWithName(&NSString::from_str(desc.entry)) else {
            return err(format!("metallib has no function '{}'", desc.entry));
        };
        let state = self
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| {
                crate::Error(format!(
                    "cannot create pipeline: {}",
                    e.localizedDescription()
                ))
            })?;
        let [x, y, z] = desc.workgroup_size.map(|v| v as usize);
        Ok(MetalPipeline {
            state,
            workgroup: MTLSize {
                width: x,
                height: y,
                depth: z,
            },
            push_constant_bytes: desc.push_constant_bytes as usize,
        })
    }

    fn create_surface<W: HasWindowHandle + HasDisplayHandle>(
        &self,
        window: &W,
        size: (u32, u32),
    ) -> Result<MetalSurface> {
        let handle = window
            .window_handle()
            .map_err(|e| crate::Error(e.to_string()))?;
        let RawWindowHandle::AppKit(appkit) = handle.as_raw() else {
            return err("create_surface: not an AppKit window");
        };
        // SAFETY: the window handle guarantees `ns_view` is a live NSView for as long as `window`
        // is borrowed, and surfaces are created on the main thread by the caller.
        let view: &NSView = unsafe { appkit.ns_view.cast::<NSView>().as_ref() };
        let layer = CAMetalLayer::new();
        layer.setDevice(Some(&self.device));
        layer.setPixelFormat(MTLPixelFormat::BGRA8Unorm_sRGB);
        layer.setFramebufferOnly(true);
        view.setWantsLayer(true);
        view.setLayer(Some(&layer));
        let mut surface = MetalSurface {
            layer,
            queue: self.queue.clone(),
        };
        surface.resize(size.0, size.1)?;
        Ok(surface)
    }

    fn submit_and_wait(&self, record: impl FnOnce(&mut MetalCommands)) -> Result<()> {
        autoreleasepool(|_| {
            let Some(cb) = self.queue.commandBuffer() else {
                return err("cannot create Metal command buffer");
            };
            let mut commands = MetalCommands::new(cb.clone());
            record(&mut commands);
            commands.end_encoder();
            cb.commit();
            cb.waitUntilCompleted();
            if cb.status() == MTLCommandBufferStatus::Error {
                let msg = cb
                    .error()
                    .map(|e| e.localizedDescription().to_string())
                    .unwrap_or_else(|| "unknown error".into());
                return err(format!("Metal command buffer failed: {msg}"));
            }
            Ok(())
        })
    }
}

enum Encoder {
    None,
    Compute(Dyn<dyn MTLComputeCommandEncoder>),
    Blit(Dyn<dyn MTLBlitCommandEncoder>),
}

pub struct MetalCommands {
    cb: Dyn<dyn MTLCommandBuffer>,
    encoder: Encoder,
    // Compute state is replayed when an intervening blit forced a new compute encoder.
    pipeline: Option<(Dyn<dyn MTLComputePipelineState>, MTLSize, usize)>,
    push: [u8; MAX_PUSH_CONSTANT_BYTES],
    push_len: usize,
    used: Vec<(Dyn<dyn MTLResource>, MTLResourceUsage)>,
}

impl MetalCommands {
    fn new(cb: Dyn<dyn MTLCommandBuffer>) -> Self {
        Self {
            cb,
            encoder: Encoder::None,
            pipeline: None,
            push: [0; MAX_PUSH_CONSTANT_BYTES],
            push_len: 0,
            used: Vec::new(),
        }
    }

    fn end_encoder(&mut self) {
        match std::mem::replace(&mut self.encoder, Encoder::None) {
            Encoder::None => {}
            Encoder::Compute(e) => e.endEncoding(),
            Encoder::Blit(e) => e.endEncoding(),
        }
    }

    fn compute(&mut self) -> &Dyn<dyn MTLComputeCommandEncoder> {
        if !matches!(self.encoder, Encoder::Compute(_)) {
            self.end_encoder();
            let e = self
                .cb
                .computeCommandEncoderWithDispatchType(MTLDispatchType::Concurrent)
                .expect("Metal refused to create a compute encoder");
            if let Some((state, _, _)) = &self.pipeline {
                e.setComputePipelineState(state);
            }
            if self.push_len > 0 {
                set_push(&e, &self.push[..self.push_len]);
            }
            for (resource, usage) in &self.used {
                e.useResource_usage(resource, *usage);
            }
            self.encoder = Encoder::Compute(e);
        }
        match &self.encoder {
            Encoder::Compute(e) => e,
            _ => unreachable!(),
        }
    }

    fn blit(&mut self) -> &Dyn<dyn MTLBlitCommandEncoder> {
        if !matches!(self.encoder, Encoder::Blit(_)) {
            self.end_encoder();
            let e = self
                .cb
                .blitCommandEncoder()
                .expect("Metal refused to create a blit encoder");
            self.encoder = Encoder::Blit(e);
        }
        match &self.encoder {
            Encoder::Blit(e) => e,
            _ => unreachable!(),
        }
    }
}

fn set_push(e: &Dyn<dyn MTLComputeCommandEncoder>, data: &[u8]) {
    // SAFETY: `data` is a live slice; Metal copies it before the call returns. Slang places the
    // push constant block at buffer index 0.
    unsafe {
        e.setBytes_length_atIndex(NonNull::from(data).cast::<c_void>(), data.len(), 0);
    }
}

impl Commands for MetalCommands {
    type Buffer = MetalBuffer;
    type ComputePipeline = MetalPipeline;

    fn bind_pipeline(&mut self, pipeline: &MetalPipeline) {
        self.pipeline = Some((
            pipeline.state.clone(),
            pipeline.workgroup,
            pipeline.push_constant_bytes,
        ));
        let state = pipeline.state.clone();
        self.compute().setComputePipelineState(&state);
    }

    fn push_constants(&mut self, data: &[u8]) {
        assert!(
            data.len() <= MAX_PUSH_CONSTANT_BYTES,
            "push constants exceed the maximum"
        );
        assert!(
            data.len() % 4 == 0,
            "push constants must be a multiple of 4 bytes"
        );
        if let Some((_, _, expected)) = &self.pipeline {
            assert!(
                data.len() >= *expected,
                "push constants shorter than the pipeline reads"
            );
        }
        self.push[..data.len()].copy_from_slice(data);
        self.push_len = data.len();
        let data = self.push;
        let len = self.push_len;
        set_push(self.compute(), &data[..len]);
    }

    fn use_buffer(&mut self, buffer: &MetalBuffer, access: Access) {
        let usage = match access {
            Access::Read => MTLResourceUsage::Read,
            Access::ReadWrite => MTLResourceUsage::Read | MTLResourceUsage::Write,
        };
        let resource: Dyn<dyn MTLResource> = ProtocolObject::from_retained(buffer.raw.clone());
        self.compute().useResource_usage(&resource, usage);
        self.used.push((resource, usage));
    }

    fn dispatch(&mut self, groups: [u32; 3]) {
        let workgroup = self
            .pipeline
            .as_ref()
            .expect("dispatch without a bound pipeline")
            .1;
        let [x, y, z] = groups.map(|v| v as usize);
        self.compute().dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: x,
                height: y,
                depth: z,
            },
            workgroup,
        );
    }

    fn dispatch_indirect(&mut self, args: &MetalBuffer, offset: u64) {
        assert!(offset % 4 == 0, "indirect offset must be a multiple of 4");
        assert!(
            offset + 12 <= args.size,
            "indirect arguments exceed the buffer"
        );
        let workgroup = self
            .pipeline
            .as_ref()
            .expect("dispatch without a bound pipeline")
            .1;
        let raw = args.raw.clone();
        // SAFETY: the range was checked above; the contents are three u32 written by the caller.
        unsafe {
            self.compute()
                .dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup(
                    &raw,
                    offset as usize,
                    workgroup,
                );
        }
    }

    fn barrier(&mut self) {
        self.compute()
            .memoryBarrierWithScope(MTLBarrierScope::Buffers);
    }

    fn copy_buffer(
        &mut self,
        src: &MetalBuffer,
        src_offset: u64,
        dst: &MetalBuffer,
        dst_offset: u64,
        size: u64,
    ) {
        assert!(
            src_offset % 4 == 0 && dst_offset % 4 == 0 && size % 4 == 0,
            "copy must be 4-byte aligned"
        );
        assert!(
            src_offset + size <= src.size && dst_offset + size <= dst.size,
            "copy out of range"
        );
        // SAFETY: ranges were checked above.
        unsafe {
            self.blit()
                .copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    &src.raw,
                    src_offset as usize,
                    &dst.raw,
                    dst_offset as usize,
                    size as usize,
                );
        }
    }

    fn fill_buffer(&mut self, buffer: &MetalBuffer, offset: u64, size: u64, value: u8) {
        assert!(
            offset % 4 == 0 && size % 4 == 0,
            "fill must be 4-byte aligned"
        );
        assert!(offset + size <= buffer.size, "fill out of range");
        self.blit().fillBuffer_range_value(
            &buffer.raw,
            NSRange {
                location: offset as usize,
                length: size as usize,
            },
            value,
        );
    }
}

pub struct MetalSurface {
    layer: Retained<CAMetalLayer>,
    queue: Dyn<dyn MTLCommandQueue>,
}

impl Surface for MetalSurface {
    fn resize(&mut self, width: u32, height: u32) -> Result<()> {
        if width == 0 || height == 0 {
            return err("resize: zero-sized surface");
        }
        self.layer
            .setDrawableSize(objc2_core_foundation::CGSize::new(
                width as f64,
                height as f64,
            ));
        Ok(())
    }

    fn clear_and_present(&mut self, color: [f32; 4]) -> Result<()> {
        autoreleasepool(|_| {
            let Some(drawable) = self.layer.nextDrawable() else {
                return err("no drawable available");
            };
            let pass = MTLRenderPassDescriptor::renderPassDescriptor();
            // SAFETY: index 0 is always a valid color attachment slot.
            let attachment = unsafe { pass.colorAttachments().objectAtIndexedSubscript(0) };
            attachment.setTexture(Some(&drawable.texture()));
            attachment.setLoadAction(MTLLoadAction::Clear);
            attachment.setStoreAction(MTLStoreAction::Store);
            attachment.setClearColor(MTLClearColor {
                red: color[0] as f64,
                green: color[1] as f64,
                blue: color[2] as f64,
                alpha: color[3] as f64,
            });
            let Some(cb) = self.queue.commandBuffer() else {
                return err("cannot create Metal command buffer");
            };
            let Some(encoder) = cb.renderCommandEncoderWithDescriptor(&pass) else {
                return err("cannot create render encoder");
            };
            encoder.endEncoding();
            let drawable: Dyn<dyn MTLDrawable> = ProtocolObject::from_retained(drawable);
            cb.presentDrawable(&drawable);
            cb.commit();
            Ok(())
        })
    }
}
