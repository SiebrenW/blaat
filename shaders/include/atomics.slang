// Atomics whose default Slang lowering is wrong on some backend.

// 64-bit atomic max, as used to resolve the visibility buffer (ADR 0005).
//
// Metal only has a non-returning 64-bit max (`atomic_max_explicit`, Apple GPUs from M2). Slang's
// default Metal lowering emits `atomic_fetch_max_explicit`, which the Metal compiler rejects for
// `ulong`, so Metal gets a hand-written call. Vulkan and D3D12 use the normal lowering.
// The original value is not returned because Metal cannot provide it.
public void atomicMax64(uint64_t* target, uint64_t value)
{
    __target_switch
    {
    case metal:
        __intrinsic_asm "atomic_max_explicit((device atomic_ulong*)$0, $1, memory_order_relaxed)";
    default:
        InterlockedMax(*target, value);
    }
}
