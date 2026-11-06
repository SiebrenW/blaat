# Shaders reach buffers by address, passed in push constants, not through descriptor tables

A shader receives the GPU addresses of the buffers it uses in a small push constant block (at most 128 bytes) and reads them through pointers. Vulkan uses buffer device addresses and Metal uses `MTLBuffer.gpuAddress`, and Slang lowers its pointer type to both. We chose this over descriptor tables and argument buffers because virtual geometry chases pointers between cluster, instance, and page data (ADR 0005), pointers avoid a descriptor-update path and slot management for buffers, and the same shader source works on both backends without a binding model per backend. The cost is that the RHI cannot see which buffers a dispatch touches, so callers must declare them (`use_buffer`), which Metal needs for residency, and that Vulkan needs `bufferDeviceAddress`, which the floor GPU has. Pointers also carry no bounds, so an out-of-range read is a GPU fault or silent garbage and not a clamped access.

This covers buffers only. Textures and samplers will need an indexed descriptor table when the first texture is sampled, in the virtual texturing milestone, and that decision is not made here.

Metal's 64-bit atomic max takes a pointer to `atomic_ulong` and returns nothing, and Slang's default lowering of it is rejected by the Metal compiler, so the Engine supplies its own lowering (see ADR 0008).
