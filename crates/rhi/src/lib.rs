//! Thin graphics layer over Metal and Vulkan (ADR 0015); D3D12 follows.
//!
//! Design, in short:
//! - Backends are static types implementing [`Device`]; there is no dynamic dispatch. Code that
//!   must work on every backend is generic over `D: Device`. [`DefaultDevice`] is the backend
//!   used on the current platform.
//! - Buffers are reached from shaders by address, not through descriptors (ADR 0019). A shader
//!   receives addresses through push constants. Every buffer a dispatch may touch, directly or by
//!   following a pointer, must be declared with [`Commands::use_buffer`]; Metal needs it for
//!   residency and hazard tracking, Vulkan ignores it.
//! - Synchronization is explicit: nothing in one [`Commands`] is ordered after anything else
//!   until [`Commands::barrier`], including copies and fills. Backends may order more strictly.
//! - [`Device::submit_and_wait`] is synchronous, so resources can be dropped as soon as it
//!   returns. Frames in flight and deferred destruction arrive with the renderer.

use std::fmt;

use raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle};

#[cfg(target_vendor = "apple")]
pub mod metal;
#[cfg(feature = "vulkan")]
pub mod vulkan;

/// The backend used on this platform unless the caller picks another.
#[cfg(target_vendor = "apple")]
pub type DefaultDevice = metal::MetalDevice;
/// The backend used on this platform unless the caller picks another.
#[cfg(not(target_vendor = "apple"))]
pub type DefaultDevice = vulkan::VulkanDevice;

/// Largest push constant block, in bytes. Vulkan guarantees 128.
pub const MAX_PUSH_CONSTANT_BYTES: usize = 128;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error(pub String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

pub type Result<T, E = Error> = std::result::Result<T, E>;

pub(crate) fn err<T>(msg: impl Into<String>) -> Result<T> {
    Err(Error(msg.into()))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Metal,
    Vulkan,
}

#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub name: String,
    pub backend: Backend,
    /// 64-bit integer atomic max/min on buffers, required by the visibility buffer (ADR 0005).
    /// MoltenVK reports false even on GPUs where Metal supports it.
    pub int64_atomics: bool,
}

#[derive(Default)]
pub struct DeviceDesc {
    /// Display the device will present to. Vulkan needs it to pick instance extensions.
    pub display: Option<RawDisplayHandle>,
    /// Enable the backend's validation layer. Fails if it is not installed.
    pub validation: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryLocation {
    /// Device-local memory, not host accessible.
    GpuOnly,
    /// Host-writable, GPU-readable. Fill with [`Buffer::write`].
    Upload,
    /// Host-readable. Copy results into it, then [`Buffer::read`].
    Readback,
}

/// Every buffer can be a storage buffer, an indirect argument buffer, and a copy source or
/// destination, and is addressable from shaders.
#[derive(Debug, Clone, Copy)]
pub struct BufferDesc {
    pub size: u64,
    pub location: MemoryLocation,
}

pub trait Buffer {
    fn size(&self) -> u64;
    /// Address to hand to shaders. Stable for the buffer's lifetime.
    fn gpu_address(&self) -> u64;
    /// Copies `data` into an [`Upload`](MemoryLocation::Upload) buffer.
    fn write(&self, offset: u64, data: &[u8]) -> Result<()>;
    /// Copies from a [`Readback`](MemoryLocation::Readback) buffer. The GPU work that produced
    /// the data must have completed ([`Device::submit_and_wait`]).
    fn read(&self, offset: u64, out: &mut [u8]) -> Result<()>;
}

/// Compiled shader in every format available. A backend uses the one it needs and fails if it
/// is missing.
#[derive(Clone, Copy)]
pub struct ShaderCode<'a> {
    pub spirv: &'a [u8],
    pub metallib: Option<&'a [u8]>,
}

pub struct ComputePipelineDesc<'a> {
    pub code: ShaderCode<'a>,
    /// Entry point name, identical in every format.
    pub entry: &'a str,
    /// Threads per threadgroup; must equal the shader's `numthreads`. Metal needs it at dispatch.
    /// A mismatch is not detected on Vulkan. Slang reflection should replace this field once the
    /// shader build exposes it.
    pub workgroup_size: [u32; 3],
    /// Bytes of push constants the shader reads, at most [`MAX_PUSH_CONSTANT_BYTES`].
    pub push_constant_bytes: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Access {
    Read,
    ReadWrite,
}

/// Recording interface, valid inside [`Device::submit_and_wait`].
pub trait Commands {
    type Buffer: Buffer;
    type ComputePipeline;

    fn bind_pipeline(&mut self, pipeline: &Self::ComputePipeline);
    /// Sets the push constants for following dispatches. `data` must be at least as long as the
    /// bound pipeline's `push_constant_bytes`, at most [`MAX_PUSH_CONSTANT_BYTES`], and a
    /// multiple of 4 bytes. Call after [`bind_pipeline`](Self::bind_pipeline).
    fn push_constants(&mut self, data: &[u8]);
    /// Declares that following dispatches may access `buffer` through an address.
    fn use_buffer(&mut self, buffer: &Self::Buffer, access: Access);
    fn dispatch(&mut self, groups: [u32; 3]);
    /// Reads three `u32` threadgroup counts at `offset` in `args`, which must be a multiple
    /// of 4. The writer must be separated from this call by a [`barrier`](Self::barrier).
    fn dispatch_indirect(&mut self, args: &Self::Buffer, offset: u64);
    /// Makes earlier writes (dispatches, copies, fills) visible to later reads and writes.
    fn barrier(&mut self);
    /// Offsets and sizes must be multiples of 4.
    fn copy_buffer(
        &mut self,
        src: &Self::Buffer,
        src_offset: u64,
        dst: &Self::Buffer,
        dst_offset: u64,
        size: u64,
    );
    /// Fills with a repeated byte. Offset and size must be multiples of 4.
    fn fill_buffer(&mut self, buffer: &Self::Buffer, offset: u64, size: u64, value: u8);
}

pub trait Surface {
    /// New size in physical pixels.
    fn resize(&mut self, width: u32, height: u32) -> Result<()>;
    /// Clears the next image to `color` (linear RGBA) and presents it.
    fn clear_and_present(&mut self, color: [f32; 4]) -> Result<()>;
}

pub trait Device: Sized {
    type Buffer: Buffer;
    type ComputePipeline;
    type Commands: Commands<Buffer = Self::Buffer, ComputePipeline = Self::ComputePipeline>;
    type Surface: Surface;

    fn new(desc: &DeviceDesc) -> Result<Self>;
    fn info(&self) -> &DeviceInfo;
    fn create_buffer(&self, desc: &BufferDesc) -> Result<Self::Buffer>;
    fn create_compute_pipeline(&self, desc: &ComputePipelineDesc) -> Result<Self::ComputePipeline>;
    /// `size` is the window's size in physical pixels.
    fn create_surface<W: HasWindowHandle + HasDisplayHandle>(
        &self,
        window: &W,
        size: (u32, u32),
    ) -> Result<Self::Surface>;
    /// Records commands with `record`, submits them, and blocks until the GPU finishes. Returns
    /// the backend's error if execution failed.
    fn submit_and_wait(&self, record: impl FnOnce(&mut Self::Commands)) -> Result<()>;
}
