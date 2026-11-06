// Writes index + 1 to out[0..count). Plain stores only, so it runs on every GPU; used to check
// how many threads an indirect dispatch actually launched.
struct Params
{
    uint* out;
    uint count;
};

[[vk::push_constant]] ConstantBuffer<Params> params;

[shader("compute")]
[numthreads(64, 1, 1)]
void iotaMain(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x < params.count)
        params.out[tid.x] = tid.x + 1;
}
