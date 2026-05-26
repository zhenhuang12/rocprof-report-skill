# CDNA3 (MI300X) & CDNA4 (MI355X) HIP Programming Reference

Background reference for kernel-design agents writing or optimizing HIP kernels for AMD Instinct MI300X (gfx942, CDNA3) and MI355X (gfx950, CDNA4). Companion to the profiling reference under [`reference/`](reference/) — the profiling docs tell you *what* to look at; this doc tells you *why* AMD hardware behaves the way it does and what programming patterns target it well.

If you are coming from NVIDIA, the most important mental adjustments are:

1. **Wave size is 64**, not 32. Affects `__syncwarp`-style code, ballot ops, and per-thread arithmetic for tile sizes.
2. **No tensor memory accelerator (TMA), no TMEM**: data movement is per-lane LDS / global loads. CDNA3+ has `global_load_lds` (direct-to-LDS) which is the closest analog and the right tool for double-buffered prefetch.
3. **Matrix cores use MFMA intrinsics + AGPR**: a separate per-SIMD accumulator register file (256 AGPR + 256 VGPR per SIMD on CDNA3; the same physical pool partitions differently on CDNA4). MFMA writes to AGPR; you copy in/out via `v_accvgpr_*`.
4. **No L1 data cache** in the NVIDIA sense — there's a per-CU vector L1 (TCP / vL1, 16 KB read-only-ish on CDNA3), but its hit rate is usually much lower than NVIDIA's L1. Use LDS for reuse.
5. **Scratch memory lives in HBM**, not L1. Register spill is much more expensive than on NVIDIA; budget VGPR with `__launch_bounds__`.
6. **Front-end issue is per-wave64, not per-warp32**: a divergent branch on 64 lanes is twice as costly. Avoid divergent control flow more aggressively than on NVIDIA.
7. **Atomics**: `-munsafe-fp-atomics` is the analog of NVIDIA's hardware FP atomic ops; without it, hipcc emits CAS loops.

---

## Architecture summary

### CDNA3 / gfx942 / MI300X (discrete) and MI300A (APU variant)

| Property | MI300X | MI300A |
|---|---|---|
| Compute Units | 304 (8 XCDs × 38 CUs) | 228 (6 XCDs × 38 CUs) + 24 Zen4 cores |
| SIMDs per CU | 4 | 4 |
| Wave size | 64 | 64 |
| VGPR per SIMD | 256 × 32-bit | 256 |
| AGPR per SIMD | 256 × 32-bit (separately addressable, used by MFMA) | 256 |
| LDS per CU | 64 KB (32 banks × 4 B) | 64 KB |
| Vector L1 (TCP) | per-CU, 16 KB | per-CU, 16 KB |
| L2 (TCC) | per-XCD, 4 MB | per-XCD, 4 MB |
| **Infinity Cache (MALL)** | **256 MB shared across XCDs** | shared with CPU |
| HBM | 192 GB HBM3 @ 5.3 TB/s | 128 GB unified |
| Memory channels | 8 HBM3 stacks; 2 channels visible per XCD (`TCC_EA0`, `TCC_EA1`) | unified |
| Peak FP64 (MFMA) | ~163 TFLOPS | ~122 TFLOPS |
| Peak BF16 (MFMA) | ~1307 TFLOPS dense | ~980 TFLOPS |
| Peak FP8 (MFMA) | ~2614 TFLOPS dense (OCP-FNUZ) | ~1960 TFLOPS |
| Peak FP16 (MFMA) | ~1307 TFLOPS dense | — |
| Wave dispatch | 8 waves/SIMD theoretical | same |
| Compute partitions | SPX (1) / DPX (2) / QPX (4) / CPX (8) | SPX / DPX / CPX (fewer) |
| Memory partitions (NPS) | NPS1 (interleaved) / NPS2 / NPS4 | NPS1 |

### CDNA4 / gfx950 / MI355X

| Property | MI355X |
|---|---|
| Compute Units | 256 (2 IODs × 8 XCDs × 16 CUs per XCD, or similar — see ROCm 7 release notes) |
| SIMDs per CU | 4 |
| Wave size | 64 |
| VGPR / AGPR per SIMD | 256 each, but pool repartitionable (more flexible than CDNA3) |
| LDS per CU | **160 KB** (up from 64 KB on CDNA3) |
| Vector L1 (TCP) | per-CU, 32 KB (doubled) |
| L2 (TCC) | per-XCD, 4 MB |
| **Infinity Cache** | **removed on CDNA4** — relies on HBM3E BW directly |
| HBM | 288 GB HBM3E @ 8.0 TB/s |
| Peak FP64 (MFMA) | ~halved vs CDNA3 (~80 TFLOPS) |
| Peak BF16/FP16 (MFMA) | ~higher than CDNA3 |
| Peak FP8 (MFMA, OCP standard) | ~5 PFLOPS dense |
| Peak FP6 / FP4 / MXFP | new — see MFMA shape table below |
| TF32 | **removed** |
| 2:4 sparsity | **added** — SQ_INSTS_MFMA_SPARSE_* |
| `global_load_lds` width | **widened to 128 bits/lane** (4× CDNA3) — major double-buffering improvement |
| Compute partitions | SPX / CPX (others depend on ROCm 7 final) |
| Memory partitions | NPS1 / NPS2 (typically; check rocm-smi) |

### Per-XCD layout (MI300X NPS1 SPX, the default)

```
       MI300X package
  ┌──────────────────────────────────────┐
  │  XCD0  XCD1  XCD2  XCD3              │     8 XCDs, each 38 CUs
  │  XCD4  XCD5  XCD6  XCD7              │     Total 304 CUs
  │                                      │
  │       Infinity Cache (256 MB)         │     Shared across XCDs
  │                                      │
  │  HBM3 stack0 ... HBM3 stack7         │     8 channels × 24 GB
  └──────────────────────────────────────┘
```

Each XCD has its own L2 (TCC) and the L2-to-HBM EA path has 2 channels per XCD (`TCC_EA0_*`, `TCC_EA1_*`). When you read about "the per-channel HBM BW", that's per-EA.

In **SPX (Single Partition)** mode, all 304 CUs see one logical GPU and one HBM address space. In **CPX (8 partitions)** mode, each XCD is its own logical GPU with 38 CUs and ~24 GB HBM. **NPS** controls memory interleaving: NPS1 = all stacks interleaved; NPS4 = each pair of stacks bound to a quadrant.

```bash
rocm-smi --showcomputepartition --showmemorypartition
# To change (requires reboot or driver reset):
sudo rocm-smi --setcomputepartition CPX --setmemorypartition NPS4
```

A kernel that performs differently on CPX vs SPX usually has a small-grid problem (under-fills 304 CUs but happily fills 38). Always profile in the production partition.

---

## Wave64 and the implicit `exec` mask

Every wavefront has 64 lanes. The hardware tracks an `exec` mask — a 64-bit register specifying which lanes are currently active. A divergent branch executes both sides with subsetted `exec`, then merges. The cost model:

- A two-way branch where 32 lanes go each way: both sides execute serially with `exec` of 0x00000000FFFFFFFF and 0xFFFFFFFF00000000 — **no SIMT magic that hides this cost**.
- A `__syncthreads()` cost is paid per workgroup, not per wave. Use `s_barrier` ISA-level — `__syncthreads()` lowers to one `s_barrier`.
- `__shfl_*_sync(__activemask(), ...)` works on AMD with `warpSize == 64`. Don't hardcode 32 in your offsets.

Common bugs in CUDA-ported code:

```cpp
// CUDA (warpSize=32) — wrong on AMD
unsigned mask = __activemask();
for (int offset = 16; offset > 0; offset >>= 1)
    sum += __shfl_down_sync(mask, sum, offset);

// HIP / wave64 — correct
constexpr int W = warpSize;            // 64 on CDNA
unsigned long long mask = __activemask();   // 64-bit on AMD
for (int offset = W/2; offset > 0; offset >>= 1)
    sum += __shfl_down(sum, offset);   // _sync variants supported; mask 64-bit
```

`warpSize` is a compile-time constant in HIP, **but the value depends on compilation target**:
- gfx9 / CDNA: 64
- gfx10 / RDNA: 32

So use `warpSize` rather than a literal.

---

## VGPR + AGPR — the two register files

On CDNA, each SIMD has two architecturally separate 256-entry × 32-bit register files:

- **VGPR (v0..v255)** — general vector regs; readable by VALU, VMEM, LDS, MFMA *source* operands.
- **AGPR (a0..a255)** — *accumulator* regs; readable/writable only by MFMA (as destination) and by `v_accvgpr_read_b32 / v_accvgpr_write_b32` (move-between-files).

MFMA reads sources from VGPR (or AGPR via `accumulate=true`) and writes results to AGPR. To use MFMA productively:

```cpp
// 16x16x16 BF16 -> FP32 example (CDNA3+)
using fragment = __attribute__((__vector_size__(4 * sizeof(float)))) float;
fragment acc = {0.f, 0.f, 0.f, 0.f};         // 4 FP32 accumulators per lane → AGPR
__attribute__((__vector_size__(4 * sizeof(short)))) short a = ...;
__attribute__((__vector_size__(4 * sizeof(short)))) short b = ...;
acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, acc, /*cbsz*/0, /*abid*/0, /*blgp*/0);
```

The full MFMA intrinsic list lives in `clang/include/clang/Basic/BuiltinsAMDGPU.def`. Composable Kernel (CK) and rocWMMA wrap these.

**Register pressure rule of thumb:**
- **Total budget per SIMD**: 256 VGPR + 256 AGPR = 512 × 32-bit on CDNA3. CDNA4 lets the compiler repartition more freely between V and A.
- **Achieved occupancy** (waves per SIMD, max 8) drops as soon as either pool exceeds `256 / waves` per wave. Example: a wave using 80 VGPR + 80 AGPR caps occupancy at `256/80 ≈ 3 waves`.
- **`Scratch_Per_Workitem > 0` in `pmc_perf.csv`**: the compiler couldn't fit you in 256+256 and spilled. Scratch on AMD is HBM-backed, so this is *much* more expensive than spilling to L1 on NVIDIA. Treat any non-zero scratch as a bug.

Fixes for high VGPR pressure:

```cpp
// 1. Use __launch_bounds__ to tell the compiler your target occupancy.
//    The 2nd arg is min blocks/CU you want resident.
__global__ __launch_bounds__(256, 4)
void my_kernel(...) { ... }

// 2. Use AGPR explicitly for MFMA accumulators (the compiler already does
//    this for MFMA intrinsics; verify with `llvm-objdump -d`).

// 3. Tile smaller — bigger tiles use more registers for accumulators.

// 4. Re-order operations to shorten live ranges (often the LLVM scheduler
//    handles this, but big templated kernels can confuse it).
```

---

## MFMA — matrix-core instructions

CDNA matrix cores are called **MFMA** (Matrix Fused Multiply-Add). Unlike NVIDIA's WMMA / WGMMA which uses warp-cooperative tile loads, MFMA is per-wave64 and you handle loads yourself.

### CDNA3 (gfx942) MFMA shape catalog

| Shape | A dtype | B dtype | C/D dtype | Notes |
|---|---|---|---|---|
| `v_mfma_f32_16x16x4f32` | FP32 | FP32 | FP32 | "TF32-like", legacy CDNA1 |
| `v_mfma_f32_32x32x2f32` | FP32 | FP32 | FP32 | |
| `v_mfma_f32_16x16x16f16` | FP16 | FP16 | FP32 | dense |
| `v_mfma_f32_32x32x8f16` | FP16 | FP16 | FP32 | dense |
| `v_mfma_f32_16x16x16bf16_1k` | BF16 | BF16 | FP32 | "1k" = 1 k-block per instruction |
| `v_mfma_f32_32x32x8bf16_1k` | BF16 | BF16 | FP32 | |
| `v_mfma_f64_16x16x4f64` | FP64 | FP64 | FP64 | dense |
| `v_mfma_i32_16x16x32i8` | INT8 | INT8 | INT32 | |
| `v_mfma_i32_32x32x16i8` | INT8 | INT8 | INT32 | |
| `v_mfma_f32_16x16x32_fp8_fp8` | FP8 | FP8 | FP32 | OCP-FNUZ (CDNA3 native FP8) |
| `v_mfma_f32_16x16x32_bf8_fp8` | BF8 | FP8 | FP32 | mixed |
| `v_mfma_f32_32x32x16_fp8_fp8` | FP8 | FP8 | FP32 | bigger tile |

Peak throughput (per CU, per cycle, dense): the bigger tile (32x32x8 vs 16x16x16) is the same FLOPS but lower issue overhead. Use the biggest tile your data layout supports.

### CDNA4 (gfx950) additions

| New shape | A/B dtype | C dtype | Notes |
|---|---|---|---|
| `v_mfma_f32_16x16x32_f8f6f4` | FP8/FP6/FP4 mix | FP32 | OCP standard FP8 (NOT FNUZ) |
| `v_mfma_f32_32x32x16_f8f6f4` | mixed | FP32 | bigger tile |
| `v_mfma_scale_f32_*` | scaled FP8/6/4 with E8M0 exponent (MXFP) | FP32 | new scale-block format |
| `v_mfma_*_sparse` | A 2:4 sparse | FP32 | new on gfx950 |
| `v_mfma_f64_*` | FP64 | FP64 | **halved throughput** vs CDNA3 |
| TF32-like `*_16x16x4f32` | FP32 | FP32 | **removed**; use BF16 instead |

**Implications for kernel design:**

- **Use FP8 / FP6 / FP4 / MXFP on CDNA4** for max throughput; CDNA3 only supports FP8 OCP-FNUZ.
- **Avoid FP64 paths on CDNA4** if you have a choice — it's half the rate of CDNA3.
- **OCP standard FP8 (E4M3 / E5M2) ≠ FNUZ FP8** (no negative zero, different denormal handling). A model trained with CDNA3 FP8 needs careful conversion to run on CDNA4 standard FP8.
- **Use rocWMMA / Composable Kernel for portable matrix code** — they pick the right MFMA shape per gfx target.

---

## LDS (Local Data Share) — the AMD shared memory

LDS is per-CU, 32 banks of 4 bytes each (so 128 bytes per cycle peak). On CDNA3 it's 64 KB per CU; on CDNA4 it's **160 KB per CU**, which substantially reduces tiling pressure.

### Bank conflicts

The bank function is `bank = (addr / 4) mod 32`. Two waves accessing the same bank from different lanes in the same cycle serialize (`SQ_LDS_BANK_CONFLICT` counter increments). To avoid:

```cpp
// Common pattern: pad LDS row to 33 dwords instead of 32, so consecutive
// rows hit different banks.
__shared__ float tile[BM][BK + 1];   // +1 padding
```

With CDNA4's 160 KB LDS, you can afford bigger tiles before padding becomes a constraint.

### `global_load_lds` — direct-to-LDS prefetch

CDNA3+ has `s_buffer_load_dwordx4` / `global_load_lds_dwordx4` (depending on addressing) that load directly from HBM into LDS, bypassing VGPR. This is the **double-buffer trick** for matrix kernels and the AMD analog of NVIDIA's `cp.async`:

- CDNA3: 32 bits per lane per instruction.
- **CDNA4: widened to 128 bits per lane** — 4× the LDS-fill rate.

```cpp
// Pseudo: prefetch one tile of A while MFMA on the previous tile
// (use rocWMMA or write inline asm if you need precise control)
asm volatile("global_load_lds_dwordx4 v[0:1] off offset:0\n" : : : "memory");
__builtin_amdgcn_s_waitcnt(0);     // s_waitcnt vmcnt(0) lgkmcnt(0)
// then MFMA on the previous LDS-buffered tile
```

In practice you express this via:
- **Composable Kernel (CK)** templates — they handle double/triple buffering automatically.
- **rocWMMA** for portable C++ matrix tile ops.
- **hipBLASLt** for GEMM — same idea as cuBLASLt; tuned heuristics per problem.

### LDS atomics

`s_lds_atomic_add` etc. work on LDS without going to HBM. Useful for histogramming / reduction within a workgroup. Bank conflicts apply.

---

## Memory hierarchy & access patterns

```
  VGPR / AGPR  (1 cycle)
       │
       ▼
  LDS (per-CU, 1-2 cycles, 128 B/cycle peak)
       │
       ▼
  vL1 / TCP (per-CU, ~16-32 KB, 4-8 cycles)
       │
       ▼
  L2 / TCC   (per-XCD, ~4 MB, 30-50 cycles)
       │
       ▼
  Infinity Cache / MALL  (CDNA3 only: 256 MB, ~80 cycles)
       │
       ▼
  HBM3 / HBM3E (192-288 GB, 200-300 cycles, 5.3-8.0 TB/s)
```

### Coalescing

A wave64 issuing a `global_load_dwordx4` (16 B/lane × 64 lanes = 1024 B) ideally generates **4 TCC requests** (one per 256-byte cacheline-equivalent). Check via rocprof-compute §2.1.11 "Bytes per wavefront" — the peak is 256 B for a fully coalesced wave64 doing dword loads.

| Pattern | Bytes/wavefront | Sectors/req |
|---|---|---|
| `data[tid]` (perfectly coalesced) | 256 B | 4 |
| `data[tid * 2]` | 512 B (2× HBM traffic) | 8 |
| `data[hash(tid)]` (random) | up to 4 KB (16× waste) | 64 |
| `data[bid * 64 + tid]` with bid uniform per wave | 256 B | 4 |

Fix uncoalesced reads by **transposing to LDS first**: each lane reads its own coalesced slice into LDS, then the workgroup reads the transposed layout from LDS for the actual compute.

### vL1 vs L2 vs Infinity Cache

- **vL1 (TCP)**: per-CU. Tracks read-only data and is often *cold* — kernels rarely re-hit. Don't rely on it.
- **L2 (TCC)**: per-XCD, ~4 MB. Acts as the main reuse buffer between waves on the same XCD.
- **Infinity Cache (MALL, CDNA3 only)**: 256 MB shared. Catches cross-XCD reuse. **Removed on CDNA4** — kernels that relied on it should be retuned to stay in L2 or fit entirely in HBM bandwidth.

### Per-channel HBM balance

On MI300X, `TCC_EA0_*` and `TCC_EA1_*` should each see roughly half the traffic. A 90/10 split usually means an address-stride bug (your kernel hammers one DRAM channel). Fix by changing the address-to-channel mapping (typically: increment outer loop index by `gridDim.x * blockDim.x` instead of `1`).

---

## Stalls — what the SQ_WAIT_* counters mean

When a wave can't issue an instruction, the SQ counter for the reason it's waiting increments. These map to source-line via PC sampling / ATT:

| Counter | Wait reason | Most common cause |
|---|---|---|
| `SQ_WAIT_INST_VMEM` | vmem op in flight | Outstanding global load — increase ILP, prefetch with `global_load_lds`, fuse loads |
| `SQ_WAIT_INST_LDS` | LDS op in flight | LDS read latency — overlap with compute |
| `SQ_WAIT_ANY_LDS` | any LDS-related stall | Bank conflicts, or LDS pressure |
| `SQ_WAIT_BARRIER` | at `s_barrier` | Workgroup-wide sync — usually fundamental, but check if you can split barrier into halves |
| `SQ_WAIT_INST_VSCRATCH` | scratch op in flight | **Register spill** — fix VGPR/AGPR pressure |
| `SQ_WAIT_INST_SCA` | scalar ALU op | Rare; usually a `s_*` op blocking |
| `SQ_WAIT_VMCNT` | vmcnt > 0 | Drain outstanding vmem before continuing — usually from explicit `s_waitcnt vmcnt(0)` or mem ordering |
| `SQ_WAIT_LGKMCNT` | lgkmcnt > 0 | Drain LDS/GDS/scalar/const before continuing |

`SQ_WAIT_INST_VMEM > 30%` is usually the #1 bottleneck in a non-trivial kernel. Treatments, in order:

1. **Reduce traffic** — recompute, fuse, exploit symmetry, or use lower precision (FP16/BF16/FP8/FP4 on supported gens).
2. **Hide latency with ILP** — load 4 or 8 cachelines ahead of compute (double/quad-buffer).
3. **Move to LDS** — `global_load_lds` + LDS-resident reuse beats re-issuing the same global load.
4. **Reorder to improve coalescing** — if `Bytes per wavefront > 256`, you have a coalescing problem; fix it first.

---

## Atomics

```bash
hipcc -munsafe-fp-atomics ...
```

This emits hardware FP atomics (`global_atomic_add_f32` etc.) that don't return the old value and skip denormals. Without the flag, FP atomics are CAS loops, which serialize and are much slower under contention.

Use cases:
- **Reduction across workgroups** → use FP atomic add into a scratch buffer.
- **Histogram into HBM** → atomic; or build per-workgroup partial histogram in LDS first (much faster).
- **Counters** → integer atomics are always hardware.

Avoid atomics on a single contended address: serialize on the L2 channel. Spread the destination across multiple addresses (per-CU partials) and reduce in a separate kernel.

---

## Composable Kernel (CK), hipBLASLt, rocWMMA, MIOpen

When in doubt, use a high-level library instead of writing MFMA by hand:

- **hipBLASLt**: GEMM (matmul). Drop-in replacement for rocBLAS GEMM with much better tuning. The AMD analog of cuBLASLt. For FP8 / FP4 / mixed-precision, this is your first stop.
- **Composable Kernel (CK)**: templated C++ kernels for GEMM, attention, conv, fused ops. Performance-critical templates are tuned per gfx target. Used as the backend for hipBLASLt and many internal kernels.
- **rocWMMA**: C++ template tile-ops, similar to NVIDIA's `nvcuda::wmma`. Wraps MFMA intrinsics with shape selection.
- **MIOpen**: convolutions. Backend for PyTorch's `torch.nn.conv2d` on ROCm.

If you're writing a custom kernel and observe Compute SoL < 30% with MFMA dominant in the instruction mix, **switch to a CK/hipBLASLt template** unless you have a specific reason not to (e.g., a fused epilogue not yet in CK).

---

## Common pitfalls porting from CUDA

| CUDA | HIP equivalent | Notes |
|---|---|---|
| `__nv_bfloat16` | `__hip_bfloat16` | `<hip/hip_bf16.h>` |
| `__nv_bfloat162` | `__hip_bfloat162` | |
| `__half` / `__half2` | same | `<hip/hip_fp16.h>` |
| `cudaMalloc` | `hipMalloc` | |
| `cudaMemcpy` | `hipMemcpy` + matching kind enums | |
| `cudaStream_t` | `hipStream_t` | |
| `cudaEvent_t` | `hipEvent_t` | |
| `__syncthreads()` | same | Lowers to `s_barrier` |
| `__syncwarp()` | none / no-op | Wave64 is intrinsically lockstep; you can omit. |
| `__shfl_*_sync(mask, ...)` | `__shfl_*` (mask is 64-bit) | warpSize=64 |
| `__ballot_sync(mask, pred)` | `__ballot(pred)` (returns 64-bit) | |
| `__activemask()` | `__activemask()` (64-bit) | |
| `cudaDeviceSynchronize` | `hipDeviceSynchronize` | |
| `nvcc --maxregcount` | none — use `__launch_bounds__` | |
| `cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, ...)` | `hipFuncSetAttribute(..., hipFuncAttributeMaxDynamicSharedMemorySize, ...)` | |
| `cudaGraphLaunch` | `hipGraphLaunch` | |
| `cudaMallocAsync` | `hipMallocAsync` | |
| `cp.async` (PTX) | `global_load_lds_*` (AMDGPU asm) | CDNA3+ direct LDS load |
| TMA / TMEM | no equivalent | Use `global_load_lds` + LDS for prefetch |
| `tcgen05.mma` (Blackwell) | `mfma_*_f8f6f4` (CDNA4) | new tile formats |
| `wmma::*` | `rocwmma::*` | Portable matrix tile ops |
| `cuBLASLt` | `hipBLASLt` | |
| `cuDNN` | `MIOpen` | |
| `compute-sanitizer` | none direct — use `rocgdb`, `AMD_SERIALIZE_KERNEL=3`, in-harness CPU reference | |
| `cudaGetDevice` etc. | `hipGetDevice` | hipify-perl / hipify-clang do this mechanically |

Auto-translation: `hipify-perl input.cu > input.hip` (or `hipify-clang` for AST-aware). Handles 90% of mechanical changes. Manually fix wave64 logic, warp shuffle masks, atomic flags, TMA usage.

---

## Programming principles for AMD CDNA

These are the patterns that consistently produce well-performing CDNA kernels — distilled from CK, hipBLASLt, and the AMD perf-tuning guides.

### 1. Fill the GPU
- MI300X SPX = 304 CUs × 4 SIMDs × 8 waves = **9728 waves max** in-flight.
- CPX (1 partition) = 38 CUs × 4 × 8 = 1216 waves.
- A grid with fewer than `8 × #CUs` workgroups can't fill the GPU; tail latency dominates. Either increase grid size (smaller tiles) or accept that small problems will be CPU-bound.

### 2. Coalesce global loads
- Each wave64 should fetch contiguous bytes. `data[tid]` is correct; `data[tid*stride]` is wrong unless `stride==1`.
- Target 256 B per wavefront in rocprof-compute §2.1.11.

### 3. Use LDS for reuse
- The vL1 hit rate is unreliable; the L2 is shared across an XCD's CUs but limited.
- For tiled algorithms, stage tiles in LDS. Pad to avoid bank conflicts.

### 4. Use `global_load_lds` for prefetch
- Direct HBM→LDS bypasses VGPR pressure.
- CDNA4 widened to 128 b/lane — double-buffering throughput is ~4× CDNA3 for matrix kernels.

### 5. Use MFMA via CK / hipBLASLt / rocWMMA
- Hand-written MFMA is hard. The wrappers pick the right tile shape and emit correct AGPR moves.

### 6. Budget registers with `__launch_bounds__`
- High occupancy needs `VGPR + AGPR ≤ 256 / waves` per wave. `__launch_bounds__(WG, MIN_BLOCKS)` tells the compiler.

### 7. Eliminate scratch (= register spill)
- `Scratch_Per_Workitem > 0` → kernel is going to HBM for spills. Refactor.

### 8. Avoid divergent control flow
- 64-lane divergence is twice as costly as 32-lane.
- Prefer predicated execution (`select`) over branches for short conditional updates.

### 9. Pad LDS to avoid bank conflicts
- `__shared__ T tile[N][K + 1]` is the canonical fix.
- Check `SQ_LDS_BANK_CONFLICT` after.

### 10. Spread atomics
- Per-CU partial reductions in LDS → per-XCD aggregation in L2 → final aggregation. Avoid single contended HBM atomic.

### 11. Use `-munsafe-fp-atomics`
- Unless you specifically need ordered FP atomics, the safe (CAS) form is much slower.

### 12. Match the right partition (SPX vs CPX, NPS1 vs NPS4) to the workload
- Multi-tenant inference often runs CPX/NPS4 (8 logical GPUs).
- Training and large single-tenant inference want SPX/NPS1.
- Profile in production partition.

### 13. Watch per-channel HBM balance
- `TCC_EA0_*` and `TCC_EA1_*` should be balanced. A skew means a stride bug.

### 14. Use FP8 / FP4 / MXFP where supported
- CDNA3: FP8 (OCP-FNUZ) — 2× peak vs BF16.
- CDNA4: FP8 (OCP standard), FP6, FP4, MXFP — up to 4× peak vs BF16.
- Calibration / scale management matters; use AMD's quantizer or Marlin-style scales.

### 15. Don't expect TF32 / FP64 magic on CDNA4
- TF32 removed; FP64 halved. If your algorithm needs FP64, MI300X is the better target.

### 16. Use rocprof-compute SoL as the first signal
- §2.1.1 names the bottleneck subsystem and gives a gap-to-peak number. Use that to decide whether to chase compute, memory BW, or latency.

---

## ROCm-specific intrinsics & inline asm cheat sheet

```cpp
// Wait counters — explicit synchronization
__builtin_amdgcn_s_waitcnt(0);            // s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
// Or via inline asm with finer control:
asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");

// Direct HBM→LDS prefetch (CDNA3+) — usually emitted by CK / compiler
// asm volatile("global_load_lds_dword v[%0:%1] off offset:0" : "=v"(addr_lo), "=v"(addr_hi));

// MFMA — see clang/include/clang/Basic/BuiltinsAMDGPU.def for full list
// 16x16x16 BF16:
acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, acc, 0, 0, 0);
// 32x32x8 BF16:
acc = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(a, b, acc, 0, 0, 0);
// 16x16x32 FP8 (CDNA3 FNUZ):
acc = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(a, b, acc, 0, 0, 0);
// 16x16x32 f8/f6/f4 (CDNA4 standard):
acc = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a, b, acc, ...);

// Atomic add (hardware, with -munsafe-fp-atomics):
atomicAdd(&counter, 1.f);
// Or explicitly:
__atomic_fetch_add(&counter, 1.f, __ATOMIC_RELAXED);   // emits global_atomic_add_f32

// Permute / shuffle (wave64):
int v = __shfl_down(my_val, 32);       // lane 0 gets lane 32's value
unsigned long long mask = __ballot(pred);  // 64-bit
```

---

## Build / debug knobs (host side)

```bash
# Verbose HIP runtime log — use on first run, not under rocprof
AMD_LOG_LEVEL=4 ./harness ...

# Synchronous kernel launches — backtrace points to actual offender
AMD_SERIALIZE_KERNEL=3 ./harness ...

# CUDA-style blocking launches
HIP_LAUNCH_BLOCKING=1 ./harness ...

# Force a specific GPU (multi-GPU node)
HIP_VISIBLE_DEVICES=0 ./harness ...

# Force a specific gfx (for fat binaries)
HIP_PLATFORM=amd HSA_OVERRIDE_GFX_VERSION=9.4.2 ./harness ...   # gfx942

# Dump the LLVM IR / assembly during build
hipcc -save-temps=obj ... 2>&1 | grep -E '\.s$|\.ll$'
llvm-objdump --disassemble --arch=gfx942 harness | grep -i mfma
```

---

## Where to dig deeper

These are the authoritative AMD references — keep them open while tuning:

1. **AMD CDNA3 white paper** ("AMD Instinct MI300X Architecture") — chip layout, peak rates, MFMA tables.
2. **AMD CDNA4 white paper** ("AMD Instinct MI355X Architecture") — CDNA4 deltas including FP4/FP6/MXFP, 2:4 sparsity, LDS expansion, Infinity Cache removal.
3. **ROCm Documentation Portal** (rocm.docs.amd.com) — current ROCm release notes, rocprofv3 / rocprof-compute / hipBLASLt user guides.
4. **HIP API Reference** + **HIPIFY** translation guide.
5. **AMD GPU ISA Reference** ("AMD Instinct MI300 Instruction Set Architecture") — full VALU/SALU/MFMA/VMEM ISA. Required reading if you write inline asm.
6. **Composable Kernel** GitHub — pattern catalog for tiled algorithms.
7. **AMD GPU Open** blog — perf-tuning case studies, often with specific counter walkthroughs.
8. **rocm/llvm-project** — `clang/include/clang/Basic/BuiltinsAMDGPU.def` lists every MFMA / wave / LDS intrinsic exposed in HIP.
9. **AMD Lab Notes / ROCm performance blog** — short, focused articles on specific bottlenecks (occupancy, LDS conflicts, MFMA shapes).

When ROCm release notes mention a counter rename or section ID shift, **re-enumerate counters before trusting old scripts** — see [`reference/08-mi300x-mi355x-counter-names.md`](reference/08-mi300x-mi355x-counter-names.md).
