// Packs depth and cluster ID into 64 bits and resolves visibility with a 64-bit atomic max.
// Exercises the capability virtual geometry depends on (ADR 0005) on every backend.
// Buffers are reached by address, passed in push constants (ADR 0019).
import atomics;

struct Params
{
    uint64_t* visibility; // 16 slots
    uint count;           // threads doing work; the rest return
};

[[vk::push_constant]] ConstantBuffer<Params> params;

[shader("compute")]
[numthreads(64, 1, 1)]
void visibilityMain(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= params.count)
        return;
    uint64_t packed = ((uint64_t)tid.x << 32) | (uint64_t)tid.x;
    atomicMax64(params.visibility + (tid.x & 15), packed);
}
