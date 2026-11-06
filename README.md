# Engine

See `CONTEXT.md` for the vocabulary, `docs/adr/` for decisions, and `PLAN.md` for the build order.

## Building

Requires Rust (stable) and the [Slang](https://github.com/shader-slang/slang/releases) compiler, release 2026.21 or compatible. The shaders crate finds it through the `SLANGC` environment variable, else `slangc` on `PATH`.

```sh
export SLANGC=/path/to/slang/bin/slangc
cargo test --workspace
cargo run --release -p engine-core --example spin   # frame-time and memory harness, prints JSON
```

DXIL for D3D12 is only built on Windows hosts, because Slang needs `dxcompiler.dll` and `dxil.dll` from the DirectX Shader Compiler next to `slangc`. On macOS the build also compiles the MSL to a `.metallib` and requires Xcode's Metal toolchain (`xcodebuild -downloadComponent MetalToolchain`).

## Graphics (`engine-rhi`)

`cargo test -p engine-rhi` runs the same conformance tests on every backend available on the machine: Metal on macOS, and Vulkan everywhere (on macOS through MoltenVK). The window smoke run opens a window, runs compute work every frame, checks the results, and prints a frame-time report:

```sh
cargo run --release -p engine-rhi --example window -- 300     # frames; 0 = until closed
RHI_BACKEND=vulkan cargo run --release -p engine-rhi --example window   # or metal
```

`RHI_VALIDATION=1` enables the Vulkan validation layer, and a validation error then fails the test that caused it. On macOS install the Vulkan side with `brew install molten-vk vulkan-loader vulkan-validationlayers` and run with `VK_LAYER_PATH=/opt/homebrew/opt/vulkan-validationlayers/share/vulkan/explicit_layer.d` and `DYLD_LIBRARY_PATH=/opt/homebrew/opt/vulkan-validationlayers/lib`. MoltenVK reports no 64-bit atomics, so the Vulkan atomic-max test prints `SKIP` there.
