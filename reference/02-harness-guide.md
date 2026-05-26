# Harness Guide

A **profiling harness** is a small standalone HIP executable whose sole purpose is to launch the kernel you want to profile, with realistic inputs, using compile flags that rocprofv3 / rocprof-compute can consume (specifically `-gline-tables-only`).

You should almost always build a harness when profiling a kernel that lives inside:

- **PyTorch / `torch.utils.cpp_extension.load` on ROCm** — usually compiled without `-gline-tables-only`, hard to inject.
- **Triton (Triton-MLIR for AMD)** — the MLIR pipeline owns the final HSACO; user nvcc/hipcc flags are ignored.
- **hipBLASLt / rocBLAS / Composable Kernel JIT** — JIT cache hides the actual hipcc invocation.
- **A larger binary** where the kernel of interest is buried under initialization, data loading, or NCCL/RCCL networking code that makes each profile run take minutes.

You can skip the harness if the kernel already builds with `-gline-tables-only` (or `-g`) *and* iterating on the full binary is fast enough.

---

## What a good harness contains

1. The kernel (verbatim copy of the device code + any `__device__` helpers it calls).
2. Explicit template instantiations for every template-parameter combination you plan to profile (e.g. `<TILE_M, TILE_N>`, `<VEC_WIDTH, BLOCK_SIZE>`, whatever the kernel is parameterized on).
3. Optional input loading — from a binary file, safetensors, or synthetic.
4. A minimal `main()` that parses CLI args, allocates GPU memory, launches the kernel, synchronizes, and exits.

Things that should NOT be in the harness:

- Framework dependencies (torch, pybind11, RCCL) — they slow the build and create noise in the profile.
- Multi-kernel pipelines — profile each kernel separately unless measuring kernel-to-kernel interactions.
- Repeated warmup / timing loops — rocprof's PMC replays automatically (one replay per counter group), so run the kernel exactly once per profiled launch.
- Correctness checks — verify correctness separately, don't couple it to profiling.

---

## Template

A complete reusable template lives at [`../helpers/harness_template.hip`](../helpers/harness_template.hip). Customize these sections:

1. **Replace `KERNEL_INCLUDE_GOES_HERE`** with `#include` or paste the kernel source.
2. **Add explicit instantiations** for every template parameter combination you want to profile.
3. **Define the input shape parameters** (grid/block sizes, tensor shapes, any knobs).
4. **Fill in `alloc_and_fill()`** to allocate/initialize inputs correctly for your kernel.
5. **Fill in `launch_kernel()`** to do the actual kernel launch with the right arguments.

Compile with:
```bash
# MI300X only (CDNA3, gfx942)
# -DHARNESS_FILLED_IN=1 disables the helpers/harness_template.hip guard that
# stops you from accidentally compiling the unedited template; drop the flag
# when compiling your own from-scratch harness.
hipcc -O3 -std=c++17 -gline-tables-only \
      --offload-arch=gfx942 \
      -munsafe-fp-atomics \
      -DHARNESS_FILLED_IN=1 \
      harness.hip -o harness

# MI300X + MI355X fat binary (ROCm 7+ required for gfx950)
hipcc -O3 -std=c++17 -gline-tables-only \
      --offload-arch=gfx942 --offload-arch=gfx950 \
      -munsafe-fp-atomics \
      -DHARNESS_FILLED_IN=1 \
      harness.hip -o harness
```

Check your GPU's arch with `rocminfo | grep gfx` or `rocm-smi --showproductname`. If `hipErrorInvalidDeviceFunction` fires at launch, the binary doesn't contain a code object matching the runtime's `gfx` ID — add the matching `--offload-arch=...`.

> **Note on `-O3` vs `-O2`:** clang/AMDGPU defaults to `-O3` and most ROCm internal kernels are tuned for that. Unlike older nvcc (`-O2` was the safe default), prefer `-O3` here.

> **Note on `__launch_bounds__`:** HIP supports `__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)` exactly like CUDA. `nvcc --maxregcount` has no HIP equivalent — use `__launch_bounds__` to budget VGPR.

---

## Real data vs synthetic data

There are three levels of fidelity for harness inputs:

### Level 1: Arbitrary synthetic

`hipMalloc(&x, ...)` without initialization.

**Use when:** you only care about shape-dependent perf and the kernel has no data-dependent branches (no `if (x > threshold)` paths, no early-exit, no branch-on-NaN).

**Avoid when:** you're not sure, or the user asks for "real" profiling. Uninitialized HBM can contain garbage that triggers NaN paths.

### Level 2: Random-but-reasonable synthetic (shape-matched)

`std::uniform_real_distribution` with sensible ranges (e.g., weights in `[-0.5, 0.5]`, probabilities in `[0, 1]`). Set the *exact* shape (all variable axes of the workload) to match a specific real instance from the dataset.

**Use when:** the kernel has no data-dependent branches that materially affect perf, but you want stable inputs. This is the default for most perf profiling.

Example pattern:
```cpp
fill_bf16_random(h_input_main, 0xA0A0ULL, 0.5f);    // main activations in [-0.5, 0.5]
fill_bf16_random(h_input_small, 0xD0D0ULL, 0.25f);  // smaller-magnitude side input
fill_f32_random(h_params,       0x22222ULL, 1.0f);  // parameters — any range that avoids NaN/Inf
for (auto& x : h_params) x = -1.0f - std::fabs(x);  // squash into a specific sign/range if the kernel requires it
```

### Level 3: Actual dataset tensors (real safetensors)

Load the exact BF16/F32 bytes from a `.safetensors` file shipped with the workload.

**Use when:**
- The kernel has branches that might depend on input values (e.g., an early-exit on a magnitude threshold, or a special-case path for denormals / large values).
- The user explicitly asks to profile with real data ("必须 load real workload").
- You're comparing against a reference implementation's output for correctness.

A header-only safetensors reader (no external deps) lives at [`../helpers/safetensors_loader.h`](../helpers/safetensors_loader.h). It parses the 8-byte header length + JSON header + raw tensor bytes — everything a safetensors file ships. Same reader works on AMD/NVIDIA/CPU because it never touches a GPU API.

Example:
```cpp
#include "safetensors_loader.h"

SafetensorsFile st = SafetensorsFile::load("/path/to/workload.safetensors");
const uint8_t* input_bytes = st.tensor_bytes("<input_tensor_name>");
std::memcpy(h_input.data(), input_bytes, n_elems * sizeof(<dtype>));

// Shapes are parsed from the header — read what the definition says is variable:
int axis_0 = (int)st.entry("<input_tensor_name>").shape[0];
int axis_1 = (int)st.entry("<other_tensor_name>").shape[0] - 1;  // etc.
```

This is free relative to compilation time and removes all doubt about data-dependent effects.

---

## Choosing representative workloads

If the user's dataset has many workloads, you cannot profile them all. Pick 2-3 workloads that together cover:

1. **Each active dispatch path.** If the kernel's host-side dispatcher picks different template instantiations / grid configs based on input shape, profile one workload per path. Identify the dispatch rules by reading the launcher code, not by guessing.
2. **The largest realistic workload** in the hot-path dispatch — usually the most performance-sensitive.
3. **A worst-case-imbalance workload** if the kernel has a variable-length inner loop (one where different workgroups perform different amounts of work based on the input). Pick an input whose per-workgroup work distribution has a high max/min ratio — that's your tail-effect probe.

Example selection approach (for a kernel whose dispatcher picks between two template instantiations by batch size):
- A canonical large-batch workload — exercises the primary dispatch path.
- A small-batch workload — small grid, often reveals CU idleness or under-fill (especially across MI300X's 8 XCDs).
- If the large-batch workload has highly uneven per-element work, that's your tail-effect probe; if every element has the same work, hunt for a separately-imbalanced workload.

### Discovering workload shapes in a flashinfer-trace dataset

When the project uses `flashinfer-bench`, the workloads live in a "flashinfer-trace" dataset with this layout:

```
<dataset_root>/                                 # $FIB_DATASET_PATH
├── definitions/<category>/<definition>.json    # axes (const vs var), input/output shapes, dtypes, reference impl
├── workloads/<category>/<definition>.jsonl     # one line per workload: uuid + concrete axis values + input paths
└── blob/workloads/<category>/<definition>/
    └── <definition>_<uuid>.safetensors         # raw tensors for that workload
```

Each `.jsonl` line looks like:

```json
{
  "definition": "<name>",
  "workload": {
    "uuid": "...",
    "axes": { "<var_axis_1>": <value>, "<var_axis_2>": <value>, ... },
    "inputs": {
      "<tensor_input>": { "type": "safetensors", "path": "./blob/.../<...>.safetensors", "tensor_key": "<tensor_input>" },
      ...
      "<scalar_input>":  { "type": "scalar", "value": <number> }
    }
  }
}
```

Scalars are inline; tensors live in the safetensors blob at the given relative path (relative to the dataset root).

**Helper: [`../helpers/list_flashinfer_workloads.py`](../helpers/list_flashinfer_workloads.py).** (Vendor-neutral — same script that shipped with the NVIDIA version.)

```bash
export FIB_DATASET_PATH=/abs/path/to/flashinfer-trace

# (1) Inspect the definition: axes (which are const vs var), input/output shapes, dtypes
python3 list_flashinfer_workloads.py --definition <name> --show-definition

# (2) See the shape distribution across all workloads (default mode)
python3 list_flashinfer_workloads.py --definition <name>

# (3) List all workloads matching a filter — gives UUIDs + absolute safetensors paths
python3 list_flashinfer_workloads.py --definition <name> --list --filter <axis>=<value>

# (4) One representative per unique (axis1, axis2, ...) combination —
#     useful for dispatch coverage
python3 list_flashinfer_workloads.py --definition <name> --unique-axes <axis1>,<axis2>

# (5) Look up a specific UUID — prints axes, scalar inputs, absolute safetensors path
python3 list_flashinfer_workloads.py --definition <name> --uuid <uuid>
```

Use steps (1-2) to understand the shape space, then (3) or (4) to pick a few representative UUIDs. Copy the absolute safetensors path straight into your harness's `--workload` CLI argument, or hardcode it in a small launcher script.

If `FIB_DATASET_PATH` is not set, pass `--dataset /path/to/root` explicitly.

### If the user's dataset is NOT flashinfer-trace

The principles still apply — you need to learn the dataset's layout and locate:

1. A schema / definition (what axes are variable, which are const, what the tensor shapes/dtypes are).
2. A list of concrete workload instances (what values the var axes take).
3. The raw tensor bytes for each instance.

Write a short inspector script equivalent to `list_flashinfer_workloads.py` for that format, drop it under `$PROFILE_RUN_DIR/harness/` if it's one-shot, or generalize it under `helpers/` if you'll reuse it.

---

## Explicit template instantiation

If the kernel is a template, you must force the compiler to emit each variant you'll profile. Without this, instantiations that aren't used by `main()` will be stripped, and rocprofv3's `--kernel-include-regex "..."` won't find them.

```cpp
template __global__ void my_kernel<8, 256>(
    const __hip_bfloat16*, const __hip_bfloat16*, /* ... other args ... */,
    float*, float*);

template __global__ void my_kernel<4, 256>(
    const __hip_bfloat16*, const __hip_bfloat16*, /* ... other args ... */,
    float*, float*);
```

The launch site in `main()` picks the right instantiation based on a CLI flag.

> **HIP type names:** `__hip_bfloat16` and `__hip_bfloat162` (from `<hip/hip_bf16.h>`) replace `__nv_bfloat16` / `__nv_bfloat162`. `__half` / `__half2` (from `<hip/hip_fp16.h>`) keep the same names. The CUDA→HIP cheat sheet is in [`../cdna3-cdna4-hip-programming.md`](../cdna3-cdna4-hip-programming.md) (search "Common pitfalls porting from CUDA") and the [HIPIFY device-API support table](https://rocm.docs.amd.com/projects/HIPIFY/en/latest/tables/CUDA_Device_API_supported_by_HIP.html).

---

## Discovering the demangled kernel name

`rocprofv3 --kernel-include-regex` matches against the demangled symbol exactly as `llvm-objdump` shows it. To list every kernel in a HIP binary:

```bash
# Modern (ROCm 6.4+)
llvm-objdump --offloading --arch-name=amdgcn-amd-amdhsa--gfx942 \
             ./harness -o harness_gfx942.co
llvm-objdump --syms --demangle harness_gfx942.co | grep -i FUNC

# Legacy (ROCm 6.x, still works)
roc-obj-ls ./harness
roc-obj-extract ./harness
roc-obj -t gfx942 -d -o out/ ./harness
```

The symbol you'll see for `template __global__ void my_kernel<8, 256>(...)` is something like `_Z9my_kernelILi8ELi256EEvPK...`. Use that or its demangled form (`my_kernel<8, 256>(...)`) directly in the regex.

---

## Sanity check before profiling

Always run the harness once without rocprof to confirm it launches correctly:

```bash
./harness --workload /path/to/workload.safetensors
# expected stderr (exact text depends on the harness you wrote):
# [harness] loaded workload: <axis1>=... <axis2>=...
# [harness] grid=(...) block=(...) launching <variant>...
# [harness] done.
```

If it crashes or hangs, fix that *before* adding rocprof to the mix — rocprof errors are far less descriptive than plain HIP runtime errors.

Common HIP runtime triage envs (use them on the first run only, never under rocprof):
```bash
AMD_LOG_LEVEL=4 ./harness ...              # verbose HIP runtime log
AMD_SERIALIZE_KERNEL=3 ./harness ...        # synchronous kernel launch — backtrace points to the actual offender
HIP_LAUNCH_BLOCKING=1 ./harness ...         # CUDA-style blocking launch
```

If feasible, also spot-check correctness with `compute-sanitizer` (NVIDIA-only — no equivalent on ROCm) — on AMD use `rocgdb`, an in-harness CPU reference, or `AMD_SERIALIZE_KERNEL=3` plus a host-side checksum. Do not embed the correctness check inside the profiling harness itself.

---

## When NOT to harness

Sometimes the kernel's perf genuinely depends on surrounding code — e.g., the kernel reuses a specific L2 state set up by a prior kernel, or the kernel's launch configuration depends on a runtime dispatch step. In that case profile through the original binary (even if it's slower to iterate) and make sure the build system has `-gline-tables-only`. Check the hipcc invocation with `ninja -v` or `make V=1`.

Alternatively, you can build a harness that runs the *prior* kernel too, to reproduce the right L2 state. But this is rare — most kernels are essentially independent of prior state once a warmup pass has happened.
