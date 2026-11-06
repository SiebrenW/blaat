// Writes the arguments of an indirect dispatch from a count held in GPU memory, as a culling
// pass would.
struct Params
{
    uint3* args;        // output: threadgroup counts
    uint* count;        // input: number of work items
    uint groupSize;     // work items per threadgroup
};

[[vk::push_constant]] ConstantBuffer<Params> params;

[shader("compute")]
[numthreads(1, 1, 1)]
void argsMain()
{
    uint n = *params.count;
    *params.args = uint3((n + params.groupSize - 1) / params.groupSize, 1, 1);
}
