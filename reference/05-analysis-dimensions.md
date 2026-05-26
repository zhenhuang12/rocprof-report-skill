# Six Analysis Dimensions

Every kernel profile report is ambiguous until you look at it through specific lenses. These six dimensions are the ones that consistently matter. Walk through all six; don't stop at the first finding.

For each dimension this doc describes:

- **What you're answering**
- **Which counters to read** (gfx942 / gfx950 names — see [`08-mi300x-mi355x-counter-names.md`](08-mi300x-mi355x-counter-names.md) for the full inventory)
- **How to read them** (what's "normal", what's "bad")
- **Which `helpers/` to run**

---

## Dimension 1 — CU occupancy & launch geometry

**What:** is the grid large enough to fill the GPU? Is occupancy being limited by VGPR/AGPR, LDS, or workgroup-size constraints?

**Counters (rocprof-compute section 2.1.0, 2.1.2):**
```
Grid_Size, Workgroup_Size (each dim)
Wave_Size                                       # 64 on CDNA gfx9
VGPRs (per work-item)
SGPRs (per wavefront)
AGPRs (per work-item) — used as MFMA accumulators on CDNA3 (gfx942) when present
LDS_Per_Workgroup (bytes, statically + dynamically allocated)
Scratch_Per_Workitem                            # = register spill bytes (DRAM-backed)
Wavefronts_Per_Workgroup
Workgroups_Launched
Achieved_Occupancy_pct                          # rocprof-compute derives this
Theoretical_Occupancy_pct                       # from VGPR/LDS/wkg limits
Compute_Unit_Count                              # 304 on MI300X, 256 on MI355X
XCD_Count                                       # MI300X: 8 XCDs over 4 IODs (1-8 visible depending on CPX/SPX partition mode). MI355X: 8 XCDs over 2 IODs.
```

> **CDNA3 register model:** each SIMD has 256 VGPRs **plus** 256 AGPRs (Accumulator GPRs). MFMA reads source operands from VGPR/AGPR and writes accumulator results to AGPR. CDNA3 introduced free movement between VGPR and AGPR, but the budget is *two separate pools* — a register-pressure report listing 256 "VGPRs" is hiding the AGPR half. rocprof-compute reports both.

**Reading:**

- **Waves per CU < 1**: grid is too small to fill the chip. On MI300X with 304 CUs (SPX/NPS1), if `Workgroups_Launched < 304 × workgroups_per_CU`, some CUs sit idle the entire time. Even more brutal in NPS2/NPS4 or CPX modes where the kernel sees fewer CUs per visible GPU.
- **Waves per CU in [1, 2)**: you have a tail wave (partial last wave). Tail effect magnitude is roughly `(last_wave_blocks / wave_size) × (block_exec_time / total_kernel_time)`.
- **Waves per CU > 4**: grid is plenty big, scheduling averages out.
- **Theoretical occupancy 100% but achieved << 100%**: stalls are the bottleneck, not launch config. Move to Dimension 3.
- **Theoretical occupancy < 100% and VGPR (or AGPR) is the tightest limiter**: reduce register usage or add `__launch_bounds__`.
- **LDS** the tightest: workgroup LDS budget too large; reduce tile size or split. MI355X keeps the same **64 KB LDS/CU** as MI300X, so don't expect extra headroom on CDNA4 — shrinking the tile or splitting the workgroup is still the fix.

**Derived: wave math**

```python
waves_per_wkg = ceil(workgroup_size / 64)              # CDNA gfx9 wave = 64
wkgs_per_cu_vgpr = min_limit_from_vgpr_budget
wkgs_per_cu_lds  = min_limit_from_lds_budget
wkgs_per_cu_max  = min_workgroups_per_cu_HW            # 32 on gfx942
wkgs_per_cu = min(wkgs_per_cu_vgpr, wkgs_per_cu_lds, wkgs_per_cu_max)
wave_size = wkgs_per_cu * num_cus                       # global concurrent wkgs
num_waves = ceil(total_workgroups / wave_size)
last_wave_blocks = total_workgroups - (num_waves - 1) * wave_size
last_wave_utilization_pct = last_wave_blocks / wave_size * 100
```

**MI300X partition note:** in SPX/NPS1 mode the kernel sees 1 GPU × 304 CU. In CPX mode it sees 8 GPUs × 38 CU each, with separate HBM partitions. Recompute the wave math against the partition the run actually used. `rocm-smi --showcomputepartition --showmemorypartition` + `rocminfo` confirm the active mode.

**Helper:** `analyze_reports.py` prints all the key launch counters under "Launch geometry" in the output txt.

---

## Dimension 2 — Workgroup balance (tail effect)

**What:** are workgroups finishing at roughly the same time, or do a few outliers drag out the kernel?

**Counters / signals:**

```
# Per-CU active-cycle distribution (from rocprof-compute section 2.1.23 if present;
# otherwise compute from per-CU SQ_BUSY_CYCLES if you collect that PMC)
SQ_BUSY_CYCLES (per CU)
GRBM_GUI_ACTIVE                       # global active cycles
GRBM_CP_BUSY                          # cmd-processor busy
GRBM_SPI_BUSY                         # shader-pipe input busy (workgroup scheduling)

# Timeseries (rocprof-compute --timeseries-sampling-rate)
SQ_WAVES                              # rises with workgroup dispatch, falls with completion
SQ_INSTS_VALU
SQ_WAIT_INST_VMEM
SQ_WAIT_INST_LDS

# ATT timeline (gold standard but only covers the captured CUs)
```

**Reading:**

- rocprof-compute's workgroup-balance section already says it: `"max workgroup duration: X µs, min: Y µs, std: Z"`. A max/min ratio > 5× is severe imbalance.
- Render the timeseries PMC (use `plot_timeline.py`). Possible shapes:
  - **Flat high → clean drop**: ideal. Well-balanced, good CU fill.
  - **Flat high → gradual tail**: tail effect. The tail's length is how much time a few slow workgroups waste. Usually caused by variable-length inputs (e.g. seq_len varies per batch element).
  - **Flat low**: grid is too small (Dimension 1).
  - **Periodic waves / sawtooth**: pipeline bubbles — compute and memory alternate, nothing overlaps.
- **Per-XCD imbalance on MI300X**: in SPX/NPS1, the host-side dispatcher round-robins workgroups across 8 XCDs. If your work has a few very heavy workgroups, they will cluster on the XCD that drew them. Look at SQ_WAVES per XCD over time — divergence > 30% between fastest and slowest XCD wastes one full XCD's compute.

**Where imbalance typically comes from:**

1. **Variable-length per-workgroup work**: when each workgroup's iteration count depends on an input axis (e.g., per-element lengths driven by a prefix-sum / cumulative-length array), workgroups take very different times.
2. **Branch-and-early-exit inside the kernel**: some workgroups bail early via `return`, others don't.
3. **XCD locality on MI300X**: when one XCD's L2 / Infinity Cache has the hot data and others miss to HBM, the missing XCDs run slower.
4. **NPS4 / CPX mode**: each visible "GPU" has its own HBM stack — a workgroup that needs cross-stack data pays XGMI hop cost (and the dispatcher cannot rebalance across these visible GPUs).

**Fix direction:** chunk the variable-length work (e.g., time-chunking for sequence-style workloads), or oversubscribe with persistent-kernel work-stealing.

**Helper:** `plot_timeline.py` produces ASCII timeline plots. If you see a gradual slope on the right side, that's your tail effect.

Additionally, **always inspect the input distribution**. If per-workgroup work is driven by an array like `per_element_lengths`:
```python
work_per_wkg = [...]  # derive this from whatever drives the inner loop count
avg = sum(work_per_wkg) / len(work_per_wkg)
print(f"max/avg = {max(work_per_wkg)/avg:.2f}x, max/min = {max(work_per_wkg)/min(work_per_wkg):.2f}x")
```

Ratios > 5x indicate significant potential for tail effect.

---

## Dimension 3 — Stall reason breakdown + per-line hotspots

**What:** when waves aren't issuing, what are they waiting for? Which source lines generate the most stalls?

**Aggregate wait counters (rocprof-compute section 2.1.13):**

> **Verify counter names on your ROCm install** with `rocprofv3 -L | grep SQ_WAIT`. The list below
> reflects gfx942 / gfx950 names that have been observed; non-listed `SQ_WAIT_*` variants
> (e.g. INST_VALU, INST_SCA, ANY_LDS, ANY, VSCRATCH) are *not* guaranteed to exist and may
> have been derived metrics in older ROCm releases. Use the rocprof-compute section 2.1.13
> derived totals instead of inventing counter names.

```
# "stall cycles" — number of cycles a wave was waiting on the named resource
SQ_WAIT_INST_VMEM                     # waiting on vector memory (global load/store, L1)
SQ_WAIT_INST_LDS                      # waiting on LDS instruction issue
SQ_WAIT_INST_SMEM                     # waiting on scalar memory
SQ_WAIT_INST_FLAT                     # waiting on FLAT (generic) memory
SQ_WAIT_INST_EXP                      # waiting on export (mostly pixel/vertex, rare in compute)
SQ_WAIT_INST_MISC                     # other instruction-side waits
SQ_WAIT_BARRIER                       # waiting at s_barrier
SQ_WAIT_VMCNT, SQ_WAIT_LGKMCNT        # waiting on the s_waitcnt counters

# Issue / activity
SQ_INSTS_VALU
SQ_INSTS_MFMA
SQ_INSTS_VMEM_RD, SQ_INSTS_VMEM_WR
SQ_INSTS_LDS
SQ_BUSY_CYCLES
```

**Per-PC wait metrics (from `rocprofv3 --pc-sampling-beta-enabled --pc-sampling-method host_trap` or ATT):**

PC sampling output is a CSV with one row per sample. Aggregate by `Wait_Reason` (an enum string) and by `Source` (file:line) — see [`04-python-api.md`](04-python-api.md) example.

```
Wait_Reason possible values (rocprofv3 PC sampling — verify with `rocprofv3 -L`):
WAIT_INST_VMEM        # waiting on vector memory access
WAIT_INST_LDS         # waiting on LDS instruction
WAIT_INST_SMEM        # waiting on scalar memory
WAIT_INST_FLAT        # waiting on FLAT memory
WAIT_BARRIER          # waiting at barrier
WAIT_VMCNT, WAIT_LGKMCNT
ISSUED                # productively issuing — the "good" reason
NOT_SELECTED          # eligible but scheduler picked another wave — good sign (plenty of parallelism)
NO_INST               # wave prologue / drain — usually minor
OTHER
```

Wait-reason enum names vary between ROCm versions; the canonical list for your install is in
`rocprofv3 -L` output and in the rocprof-compute section-2.1.13 column headers. Treat any
name above as a label to match against your actual data, not as a hard guarantee.

**Stall reasons you need to know (approximate NVIDIA analogs in parens):**

| Reason | Meaning | Typical cause | Fix direction | NVIDIA analog |
|---|---|---|---|---|
| `WAIT_INST_VMEM` / `SQ_WAIT_INST_VMEM` | waiting on global / vL1 memory | uncoalesced load, latency-bound | coalesce, reuse, add ILP, use `global_load_lds` | `long_scoreboard` |
| `WAIT_INST_LDS` / `SQ_WAIT_INST_LDS` | LDS instruction issue stall (covers bank-conflict serialization on most ROCm versions) | LDS bank conflict, too many LDS ops in flight, long LDS dep chain | pad LDS, swizzle, vectorize (`ds_read_b128`) | `short_scoreboard` / `mio_throttle` |
| `WAIT_BARRIER` / `SQ_WAIT_BARRIER` | waiting at `s_barrier` | other waves haven't arrived | reduce barriers, fix divergence | `barrier` |
| `WAIT_INST_SMEM` / `SQ_WAIT_INST_SMEM` | waiting on scalar memory | scalar load latency (uniform args, constant cache) | bake to constants, prefetch | (no direct analog) |
| `WAIT_INST_FLAT` / `SQ_WAIT_INST_FLAT` | waiting on FLAT (generic) memory | generic-addressing global/LDS access | use typed global/LDS when possible | `long_scoreboard` |
| `WAIT_VMCNT` | explicit `s_waitcnt vmcnt(N)` | inserted by compiler to enforce vmem ordering | unroll, more ILP between waitcnts | partly `long_scoreboard` |
| `WAIT_LGKMCNT` | explicit `s_waitcnt lgkmcnt(N)` | LDS/GDS/Kernarg ordering | reduce LDS dep chains | (no direct analog) |
| `NOT_SELECTED` | eligible but scheduler picked another | **good sign** — plenty of parallelism | ignore | `not_selected` |
| `ISSUED` | actually issuing this cycle | **productive** | ignore | `selected` |
| `NO_INST` | wave has nothing to issue | kernel prologue/epilogue | usually minor | `no_instruction` |

**Reading the aggregate counters:** normalize by `SQ_BUSY_CYCLES` (or `SQ_WAVES × SQ_ACTIVE_INST_VALU` for issue-relative). A value of e.g. `SQ_WAIT_INST_VMEM / SQ_BUSY_CYCLES = 0.45` means waves spent 45% of busy cycles waiting on vector memory.

**Reading the PC-sampling percentages:** sum `Sample_Count` over all rows = total samples. Per-Wait_Reason fraction = "% of samples stalled on X". Rules of thumb:

- **`WAIT_INST_VMEM` > 40% of samples**: kernel is memory-latency-bound. Check Dimension 6 (access patterns) next.
- **`WAIT_INST_LDS` > 30%**: LDS bank conflicts or long dep chains; check `SQ_LDS_BANK_CONFLICT`.
- **`WAIT_BARRIER` > 20%**: too much synchronization, or wave divergence before a barrier.
- **`ISSUED` < 10%**: very little actual issue — the whole kernel is stall-bound.

**Helper:** `extract_stall_hotspots.py` produces `stall_hotspots_<tag>.txt` which ranks source lines by total stall samples. This directly points at the offending `global_load_dwordx4`, `s_barrier`, `ds_read_b128`, or compute op in source.

---

## Dimension 4 — Matrix-Core (MFMA) utilization

**What:** is the kernel using Matrix Cores at all? If yes, how well?

**Counters (rocprof-compute section 2.1.10):**

> MFMA per-dtype counters are named `SQ_INSTS_VALU_MFMA_MOPS_<DTYPE>` on gfx942 / gfx950
> ("MOPS" = matrix-ops). The exact set of `<DTYPE>` suffixes (F16, BF16, F32, F64, I8, F8,
> MXFP4/MXFP6 etc.) varies by ROCm version — list what your install exposes with
> `rocprofv3 -L | grep MFMA`. The aggregate `SQ_INSTS_MFMA` is always available.

```
SQ_INSTS_MFMA                          # total MFMA instruction count
SQ_INSTS_VALU                          # total VALU instruction count
SQ_INSTS_VALU_MFMA_MOPS_F16            # per-dtype MFMA MOPs (when exposed by ROCm)
SQ_INSTS_VALU_MFMA_MOPS_BF16
SQ_INSTS_VALU_MFMA_MOPS_F32
SQ_INSTS_VALU_MFMA_MOPS_F64            # CDNA3 (gfx942)
SQ_INSTS_VALU_MFMA_MOPS_I8
SQ_INSTS_VALU_MFMA_MOPS_F8             # CDNA3 (OCP-FNUZ) / CDNA4 (OCP standard FP8)
# Block-scaled MX formats (FP4/FP6) are CDNA4 (gfx950) only — name suffix is install-specific
SQ_BUSY_CYCLES
GRBM_GUI_ACTIVE                        # for normalizing to total time
```

**Reading:**

- **`SQ_INSTS_MFMA == 0`**: no Matrix Core usage at all. For matmul-ish kernels (attention, GEMM, conv), this is almost always a missed optimization.
- **`SQ_INSTS_MFMA / SQ_INSTS_VALU > 0.1`**: MFMA is in the mix; check how dense it is over time.
- **rocprof-compute SoL "matrix engine busy %" > 50%**: kernel is doing well on the Matrix-Core front. Focus elsewhere.
- **MFMA busy %** much lower than expected on a matmul-ish kernel: data isn't arriving fast enough (Dimension 6) or tile sizes don't fit the MFMA shape.

**MI300X (CDNA3 / gfx942) MFMA notes:**

- Supported shapes: `mfma_*_{4x4x4, 16x16x4, 16x16x16, 32x32x4, 32x32x8}` for F32/F16/BF16/I8; FP8 variants via OCP-FNUZ; FP64 added.
- Accumulators live in **AGPR** (not VGPR). When `Scratch_Per_Workitem > 0` *and* MFMA-heavy, suspect over-allocation of AGPR forcing spill.
- Use `v_mfma_f32_32x32x8_bf16` (or `_16x16x16_bf16`) over the older 16x16x4 — same throughput per cycle but better register reuse.

**MI355X (CDNA4 / gfx950) MFMA notes:**

- FP4 / FP6 / MXFP added (`mfma_scale_*` with E8M0 block-exponent operand) — 8× throughput vs FP16 in best case.
- TF32 *removed* (does not exist on CDNA4).
- FP64 throughput **halved** vs CDNA3 — gfx950 is not the GPU for FP64-dense workloads.
- 2:4 sparse MFMA variants added.
- FP8 switched from OCP-FNUZ (CDNA3) to OCP standard (CDNA4); numerics differ slightly.

**Fix direction:** if you see 0% MFMA and the workload is matrix-multiplication-shaped, redesign around MFMA. This is usually a major refactor but gives 2-10× on compute-bound paths. Most projects should use **Composable Kernel** (CK) or **hipBLASLt** instead of hand-rolling, the same way most NVIDIA projects use CUTLASS / cuBLAS.

---

## Dimension 5 — CU utilization timeline

**What:** how does CU utilization vary over the kernel's lifetime?

**Counters (rocprof-compute timeseries mode):**
```
GRBM_GUI_ACTIVE                             # global active cycles (denominator for utilization)
GRBM_COUNT                                  # total elapsed cycles since last reset
SQ_BUSY_CYCLES                              # per-CU
SQ_WAVES                                    # in-flight waves
SQ_WAIT_INST_VMEM, SQ_WAIT_INST_LDS         # over time
TCC_EA0_RDREQ_sum, TCC_EA1_RDREQ_sum        # HBM read pressure over time
```

**Reading (timeline shapes):**

- **Flat high, clean drop**: ideal.
- **Flat high, long tail**: tail effect (Dimension 2).
- **Flat low**: grid too small (Dimension 1) or severely stall-bound (Dimension 3).
- **Periodic sawtooth (compute ↕ memory)**: no compute-memory overlap — missing double-buffering / prefetch.
- **Slow ramp up, flat middle, clean drop**: kernel has warmup work (prologue), then steady state. Usually fine.
- **Per-XCD divergence on MI300X**: plot SQ_BUSY per XCD; > 30% gap between fastest and slowest XCD signals scheduling imbalance.

**Helper:** `plot_timeline.py` — renders ASCII plots. Look at multiple series side-by-side (SQ_WAVES + HBM RDREQ + WAIT_INST_VMEM) to distinguish the shapes.

**Note:** rocprof-compute's timeseries minimum interval is ~1 ms (much coarser than NVIDIA PM sampling's ~2 µs). For very short kernels (< 100 µs) prefer ATT for time-resolved per-CU activity; rocprof-compute timeseries is for longer kernels and full-app traces.

---

## Dimension 6 — Memory access pattern & cache efficiency

**What:** are global loads coalesced? Are caches hit? Is HBM actually busy?

**Counters (rocprof-compute sections 2.1.15, 2.1.16, 2.1.17):**
```
# HBM (TCC_EA = "Effective Address" memory subsystem; channels EA0/EA1 per XCD)
TCC_EA0_RDREQ_sum                              # HBM read requests (count)
TCC_EA0_RDREQ_32B_sum                          # 32B-sized read requests
TCC_EA0_WRREQ_sum                              # HBM write requests
TCC_EA0_WRREQ_64B_sum
TCC_EA1_RDREQ_sum                              # second channel
TCC_EA0_REQ                                    # all requests
TCC_EA0_MEM_REQ_LATENCY                        # latency histogram (if collected)
# Achieved HBM BW = (RDREQ_32B + WRREQ_64B*2) * 32 / kernel_time

# L2 (TCC = Texture Cache (controlled by L2 on CDNA))
TCC_HIT_sum, TCC_MISS_sum                      # L2 hit/miss counts (aggregated across channels)
TCC_ATOMIC_sum                                 # L2 atomic activity (any return / no-return)
# L2 hit rate = TCC_HIT_sum / (TCC_HIT_sum + TCC_MISS_sum)
# More granular atomic / writeback counter names vary by ROCm release — check
# `rocprofv3 -L | grep -i tcc` rather than assuming a specific spelling.

# vL1 cache (TCP)
TCP_TOTAL_CACHE_ACCESSES_sum                   # total vL1 accesses
TCP_TCC_READ_REQ_sum                           # vL1 → L2 read requests (= vL1 read misses)
TCP_TCC_WRITE_REQ_sum                          # vL1 → L2 writes
# vL1 hit rate = 1 - (TCP_TCC_READ_REQ_sum + TCP_TCC_WRITE_REQ_sum) / TCP_TOTAL_CACHE_ACCESSES_sum

# LDS
SQ_LDS_BANK_CONFLICT                            # LDS bank conflict cycles
SQ_INSTS_LDS                                    # LDS instruction count
SQ_LDS_IDX_ACTIVE                               # LDS bank index active

# Scratch (= register spill on AMD; backed by global HBM via the scratch buffer)
# The exact "scratch read/write" counter names vary by ROCm version — check
# `rocprofv3 -L | grep -i scratch`. Reliable indicators that work everywhere:
#   - `Scratch_Per_Workitem` > 0 from launch info means *some* spilling
#   - the rocprof-compute SoL panel calls it out as "Scratch" / "Spill"

# Global LD/ST
SQ_INSTS_VMEM_RD, SQ_INSTS_VMEM_WR              # global instr count
SQ_INSTS_FLAT                                    # FLAT addressing path
```

**Reading:**

- **`(TCC_EA0_RDREQ_32B_sum × 32 bytes) / kernel_time / peak_HBM_BW ≈ 80-100%**: genuinely HBM-BW-bound. Reduce bytes / amortize reads with LDS or L2. Peak HBM3 BW on MI300X ≈ 5.3 TB/s (per 8-stack); peak HBM3E on MI355X ≈ 8.0 TB/s.
- **HBM achieved BW << 10% of peak but kernel is slow**: *not* bandwidth-bound. It's latency-bound (Dimension 3) or compute-bound (check `SQ_INSTS_MFMA` + busy %).
- **vL1 hit rate > 90%**: good data locality, vL1 is absorbing the reuse.
- **L2 hit rate < 50%**: L2 is being blown through (or the kernel is reading something it never reuses), reads fall to HBM.
- **Bytes per wavefront (rocprof-compute reports this in section 2.1.11)**: ideal is 256 B (= one `global_load_dwordx4` per lane × 64 lanes × 4 B). Substantially less means under-vectorization; gather/scatter; or stride > 1 access.
- **`SQ_LDS_BANK_CONFLICT > 0`**: bank conflicts. The 32-bank LDS serializes same-bank accesses. Pad tile dims or swizzle indices. LDS budget is 64 KB/CU on both MI300X and MI355X, so the padding-cost / occupancy trade is the same on both gens.
- **`Scratch_Per_Workitem > 0` (from launch info) or rocprof-compute SoL "Scratch" non-zero**: **register spill** — very bad, scratch is HBM-backed. Reduce VGPR/AGPR pressure with `__launch_bounds__` or kernel splitting.
- **`SQ_INSTS_LDS == 0`**: kernel uses no LDS. Fine for element-wise kernels; often a missed optimization for data-reuse-heavy kernels.

rocprof-compute's section 2.1.17 reports the HBM utilization in the same "Speed-of-Light" form NCU uses:
```
HBM        Avg     Min     Max     Pct Peak
Bandwidth  X TB/s  ...     ...     Y%
```

**MI300X-specific reading:**

- **Per-XCD HBM** (NPS2/NPS4 partitions): each XCD has its own HBM stack. A workgroup running on XCD-N pays cross-XCD XGMI hops if it touches data placed on XCD-M's stack. Check this with `TCC_EAi_RDREQ_sum` per channel — wildly different values across i = 0..7 means cross-XCD traffic.
- **Infinity Cache (256 MB shared L3-ish)**: on a kernel that re-reads a working-set < 256 MB, an Infinity Cache hit (visible as TCC_HIT without going to HBM) is much cheaper than HBM. If the working set is just over 256 MB, even a tiny tiling change can swing perf hugely.

**MI355X-specific reading:**

- **`global_load_lds` widened on CDNA4** — direct HBM/L2→LDS path bypassing VGPR has wider per-lane variants on gfx950 than on gfx942. Check ISA / LLVM intrinsics for the exact widths your toolchain emits. If you see `SQ_INSTS_VMEM_RD` high *and* `SQ_INSTS_VALU` low on the data-loading lines, you're probably already using it.
- **Infinity Cache (256 MB)** is retained on MI355X but coupled to HBM3E (8.0 TB/s) instead of HBM3 — the working-set sweet spot is similar in size to MI300X but the HBM penalty for missing it is smaller.
- **FP4 / FP6 / MXFP** reduce HBM BW pressure on the same workload by 2-4× relative to FP8/FP16. The per-dtype `SQ_INSTS_VALU_MFMA_MOPS_*` counter for block-scaled MX formats is install-specific — `rocprofv3 -L | grep -i mfma` shows what your build exposes.

NCU's rule engine often reports the coalescing issue directly; rocprof-compute analogs are in section 2.1.11 ("Memory Pipe — bytes per wavefront", "Access pattern utilization"):
```
Memory Pipe — Bytes per wavefront                   Avg  Min  Max  Peak (256)
  global_load_dwordx4 (vector load 16B/lane × 64)   62   16   256  256
```

A value of 62 (out of 256) means roughly 4 of 16 bytes per lane are actually used — significant coalescing problem.

**Fix directions (by pattern):**

- Strided access (e.g. `x[lane * stride + i]`): change the per-thread layout so lanes access contiguous elements.
- AoS → SoA: restructure the data.
- Sparse writes: pack writes into a single coalesced `global_store_dwordx4` at the end of the wave.
- Register spill: add `__launch_bounds__`, reduce intermediate variables, or split kernel.
- LDS bank conflict: pad to 33 (or 65) instead of 32 (or 64); or apply swizzle.

---

## Cross-dimension synthesis

After walking through all six, write a one-line diagnosis combining them. Structure: name the top 3–4 signals, each tied to a specific dimension. For example:

> "The kernel runs at X% of peak HBM and Y% of peak MFMA throughput (Dim 1, 6). Wait time is dominated by `WAIT_INST_VMEM` (Z% of PC samples, Dim 3), concentrated on <N> source lines whose access pattern is <coalesced/uncoalesced/...> (Dim 6). The PMC timeline shows <flat / tail / sawtooth> shape (Dim 2/5), with <even / N% imbalance across XCDs>. Matrix Cores <used / unused at W%> (Dim 4)."

Fill in the X/Y/Z/W/N values and <classifications> from your own report. That sentence is the deliverable. Everything else in the report is evidence backing it.
