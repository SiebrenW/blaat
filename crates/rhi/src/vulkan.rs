//! Vulkan backend. Requires Vulkan 1.3 with buffer device address, synchronization2 and 64-bit
//! integers. 64-bit buffer atomics are optional and reported in [`DeviceInfo::int64_atomics`].
//! On macOS it runs through MoltenVK, which currently reports no 64-bit atomics.

use std::ffi::{CStr, c_char, c_void};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use ash::{khr, vk};
use gpu_allocator::MemoryLocation as GpuLocation;
use gpu_allocator::vulkan::{
    Allocation, AllocationCreateDesc, AllocationScheme, Allocator, AllocatorCreateDesc,
};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

use crate::{
    Access, Backend, Buffer, BufferDesc, Commands, ComputePipelineDesc, Device, DeviceDesc,
    DeviceInfo, Error, MAX_PUSH_CONSTANT_BYTES, MemoryLocation, Result, Surface, err,
};

const VALIDATION_LAYER: &CStr = c"VK_LAYER_KHRONOS_validation";

fn check<T>(r: ash::prelude::VkResult<T>, what: &str) -> Result<T> {
    r.map_err(|e| Error(format!("{what}: {e:?}")))
}

struct Shared {
    entry: ash::Entry,
    instance: ash::Instance,
    physical: vk::PhysicalDevice,
    device: ash::Device,
    queue: vk::Queue,
    queue_family: u32,
    // Vulkan needs external synchronization of the queue and of command pools.
    submit_lock: Mutex<vk::CommandPool>,
    allocator: Mutex<Option<Allocator>>,
    display_enabled: bool,
    debug: Option<(ash::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
    validation_errors: Arc<AtomicUsize>,
}

impl Drop for Shared {
    fn drop(&mut self) {
        // SAFETY: every child object holds an `Arc<Shared>`, so none outlive this drop; the
        // allocator is released before the device it allocates from.
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device
                .destroy_command_pool(*self.submit_lock.get_mut().unwrap(), None);
            drop(self.allocator.get_mut().unwrap().take());
            self.device.destroy_device(None);
            if let Some((loader, messenger)) = self.debug.take() {
                loader.destroy_debug_utils_messenger(messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

pub struct VulkanDevice {
    shared: Arc<Shared>,
    info: DeviceInfo,
}

pub struct VulkanBuffer {
    shared: Arc<Shared>,
    raw: vk::Buffer,
    allocation: Option<Allocation>,
    size: u64,
    address: u64,
    location: MemoryLocation,
}

impl Drop for VulkanBuffer {
    fn drop(&mut self) {
        // SAFETY: submit_and_wait is synchronous, so no GPU work uses the buffer any more.
        unsafe { self.shared.device.destroy_buffer(self.raw, None) };
        if let Some(a) = self.allocation.take() {
            let _ = self
                .shared
                .allocator
                .lock()
                .unwrap()
                .as_mut()
                .unwrap()
                .free(a);
        }
    }
}

impl VulkanBuffer {
    fn mapped(&self) -> Result<*mut u8> {
        match self.allocation.as_ref().and_then(Allocation::mapped_ptr) {
            Some(p) => Ok(p.as_ptr().cast()),
            None => err("buffer is not host mapped"),
        }
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

impl Buffer for VulkanBuffer {
    fn size(&self) -> u64 {
        self.size
    }

    fn gpu_address(&self) -> u64 {
        self.address
    }

    fn write(&self, offset: u64, data: &[u8]) -> Result<()> {
        if self.location != MemoryLocation::Upload {
            return err("write: buffer is not an Upload buffer");
        }
        check_range(self.size, offset, data.len() as u64, "write")?;
        // SAFETY: host-visible memory mapped for the buffer's lifetime; range checked above.
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                self.mapped()?.add(offset as usize),
                data.len(),
            );
        }
        Ok(())
    }

    fn read(&self, offset: u64, out: &mut [u8]) -> Result<()> {
        if self.location != MemoryLocation::Readback {
            return err("read: buffer is not a Readback buffer");
        }
        check_range(self.size, offset, out.len() as u64, "read")?;
        // SAFETY: as in `write`. Readback memory is host-cached and coherent, or flushed by the
        // fence wait in `submit_and_wait`.
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.mapped()?.add(offset as usize),
                out.as_mut_ptr(),
                out.len(),
            );
        }
        Ok(())
    }
}

pub struct VulkanPipeline {
    shared: Arc<Shared>,
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    push_constant_bytes: usize,
}

impl Drop for VulkanPipeline {
    fn drop(&mut self) {
        // SAFETY: see VulkanBuffer::drop.
        unsafe {
            self.shared.device.destroy_pipeline(self.pipeline, None);
            self.shared
                .device
                .destroy_pipeline_layout(self.layout, None);
        }
    }
}

unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _kind: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    user: *mut c_void,
) -> vk::Bool32 {
    // SAFETY: the loader passes a valid callback data pointer; `user` is the counter registered
    // in `VulkanDevice::new`, kept alive by `Shared`.
    unsafe {
        let msg = CStr::from_ptr((*data).p_message).to_string_lossy();
        eprintln!("[vulkan {severity:?}] {msg}");
        if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
            (*user.cast::<AtomicUsize>()).fetch_add(1, Ordering::Relaxed);
        }
    }
    vk::FALSE
}

fn load_entry() -> Result<ash::Entry> {
    // SAFETY: loading the system Vulkan loader runs its initializers; nothing else is required.
    let first = unsafe { ash::Entry::load() };
    if let Ok(entry) = first {
        return Ok(entry);
    }
    // Homebrew installs the loader outside the default dlopen search path.
    for path in [
        "/opt/homebrew/lib/libvulkan.dylib",
        "/usr/local/lib/libvulkan.dylib",
    ] {
        // SAFETY: as above.
        if let Ok(entry) = unsafe { ash::Entry::load_from(path) } {
            return Ok(entry);
        }
    }
    err("cannot load the Vulkan loader")
}

impl Device for VulkanDevice {
    type Buffer = VulkanBuffer;
    type ComputePipeline = VulkanPipeline;
    type Commands = VulkanCommands;
    type Surface = VulkanSurface;

    fn new(desc: &DeviceDesc) -> Result<Self> {
        let entry = load_entry()?;
        let validation_errors = Arc::new(AtomicUsize::new(0));

        // Instance.
        let mut instance_exts: Vec<*const c_char> = Vec::new();
        if let Some(display) = desc.display {
            instance_exts.extend(check(
                ash_window::enumerate_required_extensions(display),
                "required surface extensions",
            )?);
        }
        let mut flags = vk::InstanceCreateFlags::empty();
        // SAFETY: plain enumeration queries on a valid entry.
        let available_exts = check(
            unsafe { entry.enumerate_instance_extension_properties(None) },
            "enumerate instance extensions",
        )?;
        let has = |name: &CStr| {
            available_exts
                .iter()
                .any(|e| e.extension_name_as_c_str().is_ok_and(|n| n == name))
        };
        if has(khr::portability_enumeration::NAME) {
            instance_exts.push(khr::portability_enumeration::NAME.as_ptr());
            flags |= vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR;
        }
        if desc.validation {
            if !has(ash::ext::debug_utils::NAME) {
                return err("validation requested but VK_EXT_debug_utils is missing");
            }
            instance_exts.push(ash::ext::debug_utils::NAME.as_ptr());
        }
        let mut layers: Vec<*const c_char> = Vec::new();
        if desc.validation {
            // SAFETY: plain enumeration query.
            let available = check(
                unsafe { entry.enumerate_instance_layer_properties() },
                "enumerate layers",
            )?;
            if !available
                .iter()
                .any(|l| l.layer_name_as_c_str().is_ok_and(|n| n == VALIDATION_LAYER))
            {
                return err(
                    "validation requested but VK_LAYER_KHRONOS_validation is not installed",
                );
            }
            layers.push(VALIDATION_LAYER.as_ptr());
        }
        let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);
        let instance_info = vk::InstanceCreateInfo::default()
            .application_info(&app)
            .enabled_extension_names(&instance_exts)
            .enabled_layer_names(&layers)
            .flags(flags);
        // SAFETY: all pointers in the create info outlive the call.
        let instance = check(
            unsafe { entry.create_instance(&instance_info, None) },
            "vkCreateInstance",
        )?;

        let debug = if desc.validation {
            let loader = ash::ext::debug_utils::Instance::new(&entry, &instance);
            let info = vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                        | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                )
                .pfn_user_callback(Some(debug_callback))
                .user_data(Arc::as_ptr(&validation_errors) as *mut c_void);
            // SAFETY: the counter outlives the messenger because both are owned by `Shared`.
            let messenger = check(
                unsafe { loader.create_debug_utils_messenger(&info, None) },
                "create debug messenger",
            )?;
            Some((loader, messenger))
        } else {
            None
        };

        // Physical device: first by type preference that has a graphics+compute queue family and
        // the features the Engine requires.
        // SAFETY: plain enumeration queries on a live instance.
        let physicals = check(
            unsafe { instance.enumerate_physical_devices() },
            "enumerate devices",
        )?;
        let mut best: Option<(u32, vk::PhysicalDevice, u32)> = None;
        for &p in &physicals {
            // SAFETY: `p` came from this instance.
            let props = unsafe { instance.get_physical_device_properties(p) };
            if props.api_version < vk::API_VERSION_1_3 {
                continue;
            }
            let mut f12 = vk::PhysicalDeviceVulkan12Features::default();
            let mut f13 = vk::PhysicalDeviceVulkan13Features::default();
            let mut f2 = vk::PhysicalDeviceFeatures2::default()
                .push_next(&mut f12)
                .push_next(&mut f13);
            // SAFETY: as above; the chain structs outlive the call.
            unsafe { instance.get_physical_device_features2(p, &mut f2) };
            if f2.features.shader_int64 == vk::FALSE
                || f12.buffer_device_address == vk::FALSE
                || f13.synchronization2 == vk::FALSE
            {
                continue;
            }
            // SAFETY: as above.
            let families = unsafe { instance.get_physical_device_queue_family_properties(p) };
            let Some(family) = families.iter().position(|f| {
                f.queue_flags
                    .contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
            }) else {
                continue;
            };
            let rank = match props.device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => 3,
                vk::PhysicalDeviceType::INTEGRATED_GPU => 2,
                vk::PhysicalDeviceType::VIRTUAL_GPU => 1,
                _ => 0,
            };
            if best.is_none_or(|(r, ..)| rank > r) {
                best = Some((rank, p, family as u32));
            }
        }
        let Some((_, physical, queue_family)) = best else {
            return err(
                "no Vulkan 1.3 device with buffer device address, synchronization2 and shaderInt64",
            );
        };

        // SAFETY: `physical` is valid.
        let props = unsafe { instance.get_physical_device_properties(physical) };
        let name = props
            .device_name_as_c_str()
            .map(|c| c.to_string_lossy().into_owned())
            .unwrap_or_default();
        let mut f12 = vk::PhysicalDeviceVulkan12Features::default();
        let mut f2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut f12);
        // SAFETY: as above.
        unsafe { instance.get_physical_device_features2(physical, &mut f2) };
        let int64_atomics = f12.shader_buffer_int64_atomics == vk::TRUE;

        // Device.
        // SAFETY: as above.
        let device_exts = check(
            unsafe { instance.enumerate_device_extension_properties(physical) },
            "enumerate device extensions",
        )?;
        let dev_has = |name: &CStr| {
            device_exts
                .iter()
                .any(|e| e.extension_name_as_c_str().is_ok_and(|n| n == name))
        };
        let mut exts: Vec<*const c_char> = Vec::new();
        if desc.display.is_some() {
            exts.push(khr::swapchain::NAME.as_ptr());
        }
        if dev_has(c"VK_KHR_portability_subset") {
            exts.push(c"VK_KHR_portability_subset".as_ptr());
        }
        let mut enable12 = vk::PhysicalDeviceVulkan12Features::default()
            .buffer_device_address(true)
            .shader_buffer_int64_atomics(int64_atomics)
            .vulkan_memory_model(f12.vulkan_memory_model == vk::TRUE)
            .vulkan_memory_model_device_scope(f12.vulkan_memory_model_device_scope == vk::TRUE);
        let mut enable13 = vk::PhysicalDeviceVulkan13Features::default().synchronization2(true);
        let mut enable2 = vk::PhysicalDeviceFeatures2::default()
            .features(vk::PhysicalDeviceFeatures::default().shader_int64(true))
            .push_next(&mut enable12)
            .push_next(&mut enable13);
        let priorities = [1.0f32];
        let queue_infos = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&priorities)];
        let device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&exts)
            .push_next(&mut enable2);
        // SAFETY: all pointers in the create info outlive the call.
        let device = check(
            unsafe { instance.create_device(physical, &device_info, None) },
            "vkCreateDevice",
        )?;
        // SAFETY: the family index was requested at device creation.
        let queue = unsafe { device.get_device_queue(queue_family, 0) };

        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family)
            .flags(vk::CommandPoolCreateFlags::TRANSIENT);
        // SAFETY: valid device and create info.
        let pool = check(
            unsafe { device.create_command_pool(&pool_info, None) },
            "create command pool",
        )?;

        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device: physical,
            debug_settings: Default::default(),
            buffer_device_address: true,
            allocation_sizes: Default::default(),
        })
        .map_err(|e| Error(format!("create allocator: {e}")))?;

        let shared = Arc::new(Shared {
            entry,
            instance,
            physical,
            device,
            queue,
            queue_family,
            submit_lock: Mutex::new(pool),
            allocator: Mutex::new(Some(allocator)),
            display_enabled: desc.display.is_some(),
            debug,
            validation_errors,
        });
        Ok(Self {
            shared,
            info: DeviceInfo {
                name,
                backend: Backend::Vulkan,
                int64_atomics,
            },
        })
    }

    fn info(&self) -> &DeviceInfo {
        &self.info
    }

    fn create_buffer(&self, desc: &BufferDesc) -> Result<VulkanBuffer> {
        if desc.size == 0 {
            return err("create_buffer: size must be non-zero");
        }
        let s = &self.shared;
        let info = vk::BufferCreateInfo::default()
            .size(desc.size)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .usage(
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::INDIRECT_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            );
        // SAFETY: valid device and create info.
        let raw = check(
            unsafe { s.device.create_buffer(&info, None) },
            "create buffer",
        )?;
        // SAFETY: `raw` was just created.
        let requirements = unsafe { s.device.get_buffer_memory_requirements(raw) };
        let location = match desc.location {
            MemoryLocation::GpuOnly => GpuLocation::GpuOnly,
            MemoryLocation::Upload => GpuLocation::CpuToGpu,
            MemoryLocation::Readback => GpuLocation::GpuToCpu,
        };
        let allocation =
            s.allocator
                .lock()
                .unwrap()
                .as_mut()
                .unwrap()
                .allocate(&AllocationCreateDesc {
                    name: "rhi buffer",
                    requirements,
                    location,
                    linear: true,
                    allocation_scheme: AllocationScheme::GpuAllocatorManaged,
                });
        let allocation = match allocation {
            Ok(a) => a,
            Err(e) => {
                // SAFETY: `raw` is unused.
                unsafe { s.device.destroy_buffer(raw, None) };
                return err(format!("cannot allocate {} byte buffer: {e}", desc.size));
            }
        };
        // SAFETY: memory and offset come from an allocation made for these requirements.
        let bound = unsafe {
            s.device
                .bind_buffer_memory(raw, allocation.memory(), allocation.offset())
        };
        // SAFETY: `raw` has memory bound; it was created with SHADER_DEVICE_ADDRESS.
        let address = bound.map(|()| unsafe {
            s.device
                .get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(raw))
        });
        match address {
            Ok(address) => Ok(VulkanBuffer {
                shared: s.clone(),
                raw,
                allocation: Some(allocation),
                size: desc.size,
                address,
                location: desc.location,
            }),
            Err(e) => {
                // SAFETY: `raw` is unused.
                unsafe { s.device.destroy_buffer(raw, None) };
                let _ = s
                    .allocator
                    .lock()
                    .unwrap()
                    .as_mut()
                    .unwrap()
                    .free(allocation);
                err(format!("bind buffer memory: {e:?}"))
            }
        }
    }

    fn create_compute_pipeline(&self, desc: &ComputePipelineDesc) -> Result<VulkanPipeline> {
        if desc.push_constant_bytes as usize > MAX_PUSH_CONSTANT_BYTES
            || desc.push_constant_bytes % 4 != 0
        {
            return err(
                "push_constant_bytes must be a multiple of 4 and at most MAX_PUSH_CONSTANT_BYTES",
            );
        }
        let s = &self.shared;
        let words = ash::util::read_spv(&mut std::io::Cursor::new(desc.code.spirv))
            .map_err(|e| Error(format!("invalid SPIR-V: {e}")))?;
        let entry =
            std::ffi::CString::new(desc.entry).map_err(|_| Error("entry contains NUL".into()))?;
        // MoltenVK aborts the process on an unknown entry point, so check it here.
        if !spirv_entry_points(&words).iter().any(|n| n == desc.entry) {
            return err(format!("SPIR-V has no entry point '{}'", desc.entry));
        }

        let module_info = vk::ShaderModuleCreateInfo::default().code(&words);
        // SAFETY: valid device; `words` outlives the call.
        let module = check(
            unsafe { s.device.create_shader_module(&module_info, None) },
            "create shader module",
        )?;
        let ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(desc.push_constant_bytes)];
        let mut layout_info = vk::PipelineLayoutCreateInfo::default();
        if desc.push_constant_bytes > 0 {
            layout_info = layout_info.push_constant_ranges(&ranges);
        }
        // SAFETY: as above.
        let layout = match unsafe { s.device.create_pipeline_layout(&layout_info, None) } {
            Ok(l) => l,
            Err(e) => {
                // SAFETY: the module is unused.
                unsafe { s.device.destroy_shader_module(module, None) };
                return err(format!("create pipeline layout: {e:?}"));
            }
        };
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(&entry);
        let info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(layout);
        // SAFETY: as above.
        let result = unsafe {
            s.device
                .create_compute_pipelines(vk::PipelineCache::null(), &[info], None)
        };
        // SAFETY: the module is no longer needed once the pipeline exists or creation failed.
        unsafe { s.device.destroy_shader_module(module, None) };
        match result {
            Ok(pipelines) => Ok(VulkanPipeline {
                shared: s.clone(),
                pipeline: pipelines[0],
                layout,
                push_constant_bytes: desc.push_constant_bytes as usize,
            }),
            Err((_, e)) => {
                // SAFETY: the layout is unused.
                unsafe { s.device.destroy_pipeline_layout(layout, None) };
                err(format!("create compute pipeline '{}': {e:?}", desc.entry))
            }
        }
    }

    fn create_surface<W: HasWindowHandle + HasDisplayHandle>(
        &self,
        window: &W,
        size: (u32, u32),
    ) -> Result<VulkanSurface> {
        VulkanSurface::new(self.shared.clone(), window, size)
    }

    fn submit_and_wait(&self, record: impl FnOnce(&mut VulkanCommands)) -> Result<()> {
        let s = &self.shared;
        let errors_before = s.validation_errors.load(Ordering::Relaxed);
        let pool = s.submit_lock.lock().unwrap();
        // SAFETY: the pool is externally synchronized by the lock held for the whole submission.
        unsafe {
            check(
                s.device
                    .reset_command_pool(*pool, vk::CommandPoolResetFlags::empty()),
                "reset pool",
            )?
        };
        let alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(*pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // SAFETY: as above.
        let cb = check(
            unsafe { s.device.allocate_command_buffers(&alloc) },
            "allocate command buffer",
        )?[0];
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: as above.
        unsafe {
            check(
                s.device.begin_command_buffer(cb, &begin),
                "begin command buffer",
            )?
        };

        let mut commands = VulkanCommands {
            device: s.device.clone(),
            cb,
            bound: None,
        };
        record(&mut commands);

        // SAFETY: as above; the fence is destroyed on every path below.
        unsafe {
            check(s.device.end_command_buffer(cb), "end command buffer")?;
            let fence = check(
                s.device.create_fence(&vk::FenceCreateInfo::default(), None),
                "create fence",
            )?;
            let cbs = [cb];
            let submit = vk::SubmitInfo::default().command_buffers(&cbs);
            let result = check(
                s.device.queue_submit(s.queue, &[submit], fence),
                "queue submit",
            )
            .and_then(|()| {
                check(
                    s.device.wait_for_fences(&[fence], true, u64::MAX),
                    "wait for fence",
                )
            });
            s.device.destroy_fence(fence, None);
            result?;
        }
        if s.validation_errors.load(Ordering::Relaxed) != errors_before {
            return err("Vulkan validation reported errors (see stderr)");
        }
        Ok(())
    }
}

pub struct VulkanCommands {
    device: ash::Device,
    cb: vk::CommandBuffer,
    bound: Option<(vk::PipelineLayout, usize)>,
}

impl Commands for VulkanCommands {
    type Buffer = VulkanBuffer;
    type ComputePipeline = VulkanPipeline;

    fn bind_pipeline(&mut self, pipeline: &VulkanPipeline) {
        // SAFETY: the command buffer is in the recording state for the closure's duration.
        unsafe {
            self.device.cmd_bind_pipeline(
                self.cb,
                vk::PipelineBindPoint::COMPUTE,
                pipeline.pipeline,
            )
        };
        self.bound = Some((pipeline.layout, pipeline.push_constant_bytes));
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
        let (layout, bytes) = self.bound.expect("push_constants before bind_pipeline");
        assert!(
            data.len() >= bytes,
            "push constants shorter than the pipeline reads"
        );
        // SAFETY: as above; the range is within the layout's push constant range.
        unsafe {
            self.device.cmd_push_constants(
                self.cb,
                layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &data[..bytes],
            );
        }
    }

    fn use_buffer(&mut self, _buffer: &VulkanBuffer, _access: Access) {}

    fn dispatch(&mut self, groups: [u32; 3]) {
        // SAFETY: as above.
        unsafe {
            self.device
                .cmd_dispatch(self.cb, groups[0], groups[1], groups[2])
        };
    }

    fn dispatch_indirect(&mut self, args: &VulkanBuffer, offset: u64) {
        assert!(offset % 4 == 0, "indirect offset must be a multiple of 4");
        assert!(
            offset + 12 <= args.size,
            "indirect arguments exceed the buffer"
        );
        // SAFETY: as above; the range was checked.
        unsafe { self.device.cmd_dispatch_indirect(self.cb, args.raw, offset) };
    }

    fn barrier(&mut self) {
        use vk::{AccessFlags2 as A, PipelineStageFlags2 as S};
        let barrier = vk::MemoryBarrier2::default()
            .src_stage_mask(S::COMPUTE_SHADER | S::ALL_TRANSFER)
            .src_access_mask(A::SHADER_STORAGE_WRITE | A::TRANSFER_WRITE)
            .dst_stage_mask(S::COMPUTE_SHADER | S::ALL_TRANSFER | S::DRAW_INDIRECT)
            .dst_access_mask(
                A::SHADER_STORAGE_READ
                    | A::SHADER_STORAGE_WRITE
                    | A::TRANSFER_READ
                    | A::TRANSFER_WRITE
                    | A::INDIRECT_COMMAND_READ,
            );
        let barriers = [barrier];
        let dependency = vk::DependencyInfo::default().memory_barriers(&barriers);
        // SAFETY: as above.
        unsafe { self.device.cmd_pipeline_barrier2(self.cb, &dependency) };
    }

    fn copy_buffer(
        &mut self,
        src: &VulkanBuffer,
        src_offset: u64,
        dst: &VulkanBuffer,
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
        let region = vk::BufferCopy::default()
            .src_offset(src_offset)
            .dst_offset(dst_offset)
            .size(size);
        // SAFETY: as above; ranges were checked.
        unsafe {
            self.device
                .cmd_copy_buffer(self.cb, src.raw, dst.raw, &[region])
        };
    }

    fn fill_buffer(&mut self, buffer: &VulkanBuffer, offset: u64, size: u64, value: u8) {
        assert!(
            offset % 4 == 0 && size % 4 == 0,
            "fill must be 4-byte aligned"
        );
        assert!(offset + size <= buffer.size, "fill out of range");
        // SAFETY: as above; the range was checked.
        unsafe {
            self.device.cmd_fill_buffer(
                self.cb,
                buffer.raw,
                offset,
                size,
                u32::from_ne_bytes([value; 4]),
            )
        };
    }
}

const FRAMES_IN_FLIGHT: usize = 2;

struct Frame {
    cb: vk::CommandBuffer,
    fence: vk::Fence,
    image_available: vk::Semaphore,
}

pub struct VulkanSurface {
    shared: Arc<Shared>,
    surface_loader: khr::surface::Instance,
    swapchain_loader: khr::swapchain::Device,
    surface: vk::SurfaceKHR,
    swapchain: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    render_done: Vec<vk::Semaphore>,
    format: vk::Format,
    extent: (u32, u32),
    dirty: bool,
    pool: vk::CommandPool,
    frames: [Frame; FRAMES_IN_FLIGHT],
    frame: usize,
}

impl VulkanSurface {
    fn new<W: HasWindowHandle + HasDisplayHandle>(
        shared: Arc<Shared>,
        window: &W,
        size: (u32, u32),
    ) -> Result<Self> {
        if !shared.display_enabled {
            return err("create_surface: the device was created without DeviceDesc::display");
        }
        let display = window
            .display_handle()
            .map_err(|e| Error(e.to_string()))?
            .as_raw();
        let handle = window
            .window_handle()
            .map_err(|e| Error(e.to_string()))?
            .as_raw();
        // SAFETY: the handles are valid for the borrow of `window`; the surface is destroyed in
        // `Drop` before the instance (it holds an Arc<Shared>).
        let surface = check(
            unsafe {
                ash_window::create_surface(&shared.entry, &shared.instance, display, handle, None)
            },
            "create surface",
        )?;
        let surface_loader = khr::surface::Instance::new(&shared.entry, &shared.instance);
        // SAFETY: valid handles from the same instance.
        let supported = check(
            unsafe {
                surface_loader.get_physical_device_surface_support(
                    shared.physical,
                    shared.queue_family,
                    surface,
                )
            },
            "surface support",
        )?;
        if !supported {
            // SAFETY: `surface` is unused.
            unsafe { surface_loader.destroy_surface(surface, None) };
            return err("the graphics queue family cannot present to this surface");
        }
        let swapchain_loader = khr::swapchain::Device::new(&shared.instance, &shared.device);

        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(shared.queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        // SAFETY: valid device.
        let pool = check(
            unsafe { shared.device.create_command_pool(&pool_info, None) },
            "create command pool",
        )?;
        let alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(FRAMES_IN_FLIGHT as u32);
        // SAFETY: as above.
        let cbs = check(
            unsafe { shared.device.allocate_command_buffers(&alloc) },
            "allocate command buffers",
        )?;
        let mut frames = Vec::new();
        for cb in cbs {
            // SAFETY: as above.
            let fence = check(
                unsafe {
                    shared.device.create_fence(
                        &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                        None,
                    )
                },
                "create fence",
            )?;
            // SAFETY: as above.
            let image_available = check(
                unsafe {
                    shared
                        .device
                        .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                },
                "create semaphore",
            )?;
            frames.push(Frame {
                cb,
                fence,
                image_available,
            });
        }
        let frames: [Frame; FRAMES_IN_FLIGHT] = frames.try_into().ok().expect("frame count");

        let mut surface = Self {
            shared,
            surface_loader,
            swapchain_loader,
            surface,
            swapchain: vk::SwapchainKHR::null(),
            images: Vec::new(),
            render_done: Vec::new(),
            format: vk::Format::UNDEFINED,
            extent: size,
            dirty: true,
            pool,
            frames,
            frame: 0,
        };
        surface.recreate_swapchain()?;
        Ok(surface)
    }

    fn recreate_swapchain(&mut self) -> Result<()> {
        let s = &self.shared;
        // SAFETY: valid handles; waiting for idle makes destroying the old swapchain objects safe.
        unsafe {
            check(s.device.device_wait_idle(), "wait idle")?;
            let caps = check(
                self.surface_loader
                    .get_physical_device_surface_capabilities(s.physical, self.surface),
                "surface capabilities",
            )?;
            let formats = check(
                self.surface_loader
                    .get_physical_device_surface_formats(s.physical, self.surface),
                "surface formats",
            )?;
            let chosen = formats
                .iter()
                .find(|f| f.format == vk::Format::B8G8R8A8_SRGB)
                .or_else(|| {
                    formats
                        .iter()
                        .find(|f| f.format == vk::Format::R8G8B8A8_SRGB)
                })
                .or(formats.first())
                .copied()
                .ok_or_else(|| Error("surface reports no formats".into()))?;
            let extent = if caps.current_extent.width != u32::MAX {
                caps.current_extent
            } else {
                vk::Extent2D {
                    width: self
                        .extent
                        .0
                        .clamp(caps.min_image_extent.width, caps.max_image_extent.width),
                    height: self
                        .extent
                        .1
                        .clamp(caps.min_image_extent.height, caps.max_image_extent.height),
                }
            };
            if extent.width == 0 || extent.height == 0 {
                return err("surface has zero size");
            }
            let mut count = caps.min_image_count + 1;
            if caps.max_image_count != 0 {
                count = count.min(caps.max_image_count);
            }
            let info = vk::SwapchainCreateInfoKHR::default()
                .surface(self.surface)
                .min_image_count(count)
                .image_format(chosen.format)
                .image_color_space(chosen.color_space)
                .image_extent(extent)
                .image_array_layers(1)
                .image_usage(
                    vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::COLOR_ATTACHMENT,
                )
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                .pre_transform(caps.current_transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(vk::PresentModeKHR::FIFO)
                .clipped(true)
                .old_swapchain(self.swapchain);
            let new = check(
                self.swapchain_loader.create_swapchain(&info, None),
                "create swapchain",
            )?;
            for sem in self.render_done.drain(..) {
                s.device.destroy_semaphore(sem, None);
            }
            if self.swapchain != vk::SwapchainKHR::null() {
                self.swapchain_loader
                    .destroy_swapchain(self.swapchain, None);
            }
            self.swapchain = new;
            self.images = check(
                self.swapchain_loader.get_swapchain_images(new),
                "swapchain images",
            )?;
            for _ in &self.images {
                self.render_done.push(check(
                    s.device
                        .create_semaphore(&vk::SemaphoreCreateInfo::default(), None),
                    "create semaphore",
                )?);
            }
            self.format = chosen.format;
            self.extent = (extent.width, extent.height);
        }
        self.dirty = false;
        Ok(())
    }
}

impl Drop for VulkanSurface {
    fn drop(&mut self) {
        let d = &self.shared.device;
        // SAFETY: idle device; objects are destroyed before the Arc<Shared> releases the device.
        unsafe {
            let _ = d.device_wait_idle();
            for f in &self.frames {
                d.destroy_fence(f.fence, None);
                d.destroy_semaphore(f.image_available, None);
            }
            for s in self.render_done.drain(..) {
                d.destroy_semaphore(s, None);
            }
            d.destroy_command_pool(self.pool, None);
            if self.swapchain != vk::SwapchainKHR::null() {
                self.swapchain_loader
                    .destroy_swapchain(self.swapchain, None);
            }
            self.surface_loader.destroy_surface(self.surface, None);
        }
    }
}

impl Surface for VulkanSurface {
    fn resize(&mut self, width: u32, height: u32) -> Result<()> {
        if width == 0 || height == 0 {
            return err("resize: zero-sized surface");
        }
        self.extent = (width, height);
        self.dirty = true;
        Ok(())
    }

    fn clear_and_present(&mut self, color: [f32; 4]) -> Result<()> {
        use vk::{AccessFlags2 as A, ImageLayout as L, PipelineStageFlags2 as S};
        for _ in 0..2 {
            if self.dirty {
                self.recreate_swapchain()?;
            }
            let device = self.shared.device.clone();
            let frame = &self.frames[self.frame];
            // SAFETY: handles are valid; per-frame objects are only reused after their fence.
            unsafe {
                check(
                    device.wait_for_fences(&[frame.fence], true, u64::MAX),
                    "wait frame fence",
                )?;
                let acquired = self.swapchain_loader.acquire_next_image(
                    self.swapchain,
                    u64::MAX,
                    frame.image_available,
                    vk::Fence::null(),
                );
                let index = match acquired {
                    Ok((index, suboptimal)) => {
                        self.dirty |= suboptimal;
                        index as usize
                    }
                    Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                        self.dirty = true;
                        continue;
                    }
                    Err(e) => return err(format!("acquire image: {e:?}")),
                };
                check(device.reset_fences(&[frame.fence]), "reset fence")?;
                let cb = frame.cb;
                check(
                    device.reset_command_buffer(cb, vk::CommandBufferResetFlags::empty()),
                    "reset cb",
                )?;
                check(
                    device.begin_command_buffer(
                        cb,
                        &vk::CommandBufferBeginInfo::default()
                            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                    ),
                    "begin cb",
                )?;
                let image = self.images[index];
                let range = vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .level_count(1)
                    .layer_count(1);
                let to_clear = [vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(S::ALL_TRANSFER)
                    .src_access_mask(A::NONE)
                    .dst_stage_mask(S::ALL_TRANSFER)
                    .dst_access_mask(A::TRANSFER_WRITE)
                    .old_layout(L::UNDEFINED)
                    .new_layout(L::TRANSFER_DST_OPTIMAL)
                    .image(image)
                    .subresource_range(range)];
                device.cmd_pipeline_barrier2(
                    cb,
                    &vk::DependencyInfo::default().image_memory_barriers(&to_clear),
                );
                let clear = vk::ClearColorValue { float32: color };
                device.cmd_clear_color_image(cb, image, L::TRANSFER_DST_OPTIMAL, &clear, &[range]);
                let to_present = [vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(S::ALL_TRANSFER)
                    .src_access_mask(A::TRANSFER_WRITE)
                    .dst_stage_mask(S::NONE)
                    .dst_access_mask(A::NONE)
                    .old_layout(L::TRANSFER_DST_OPTIMAL)
                    .new_layout(L::PRESENT_SRC_KHR)
                    .image(image)
                    .subresource_range(range)];
                device.cmd_pipeline_barrier2(
                    cb,
                    &vk::DependencyInfo::default().image_memory_barriers(&to_present),
                );
                check(device.end_command_buffer(cb), "end cb")?;

                let waits = [frame.image_available];
                let stages = [vk::PipelineStageFlags::TRANSFER];
                let cbs = [cb];
                let signals = [self.render_done[index]];
                let submit = vk::SubmitInfo::default()
                    .wait_semaphores(&waits)
                    .wait_dst_stage_mask(&stages)
                    .command_buffers(&cbs)
                    .signal_semaphores(&signals);
                check(
                    device.queue_submit(self.shared.queue, &[submit], frame.fence),
                    "queue submit",
                )?;

                let swapchains = [self.swapchain];
                let indices = [index as u32];
                let present = vk::PresentInfoKHR::default()
                    .wait_semaphores(&signals)
                    .swapchains(&swapchains)
                    .image_indices(&indices);
                match self
                    .swapchain_loader
                    .queue_present(self.shared.queue, &present)
                {
                    Ok(suboptimal) => self.dirty |= suboptimal,
                    Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => self.dirty = true,
                    Err(e) => return err(format!("present: {e:?}")),
                }
            }
            self.frame = (self.frame + 1) % FRAMES_IN_FLIGHT;
            return Ok(());
        }
        err("swapchain stayed out of date")
    }
}

/// Names of the `OpEntryPoint` instructions in a SPIR-V module.
fn spirv_entry_points(words: &[u32]) -> Vec<String> {
    const OP_ENTRY_POINT: u32 = 15;
    let mut names = Vec::new();
    let mut i = 5; // skip the header
    while i < words.len() {
        let (op, len) = (words[i] & 0xffff, (words[i] >> 16) as usize);
        if len == 0 || i + len > words.len() {
            break;
        }
        if op == OP_ENTRY_POINT && len >= 4 {
            // words[i+1] = execution model, words[i+2] = id, then the NUL-terminated name.
            let bytes: Vec<u8> = words[i + 3..i + len]
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect();
            let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
            names.push(String::from_utf8_lossy(&bytes[..end]).into_owned());
        }
        i += len;
    }
    names
}
