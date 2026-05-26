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

**Counters (rocprof-compute blocks 7 (wavefront launch) and 5 (CS/wavefront)):**

From `pmc_perf.csv` per-dispatch columns (verified):
```
Grid_Size, Workgroup_Size (each dim)
Wave_Size                                       # 64 on CDNA gfx9
Arch_VGPR (per work-item)                       # the verified column name (NOT "VGPRs")
SGPR (per wavefront)                            # singular, NOT "SGPRs"
Accum_VGPR (per work-item)                      # = AGPR pool on CDNA3+ (NOT "AGPRs")
LDS_Per_Workgroup (bytes, statically + dynamically allocated)
Scratch_Per_Workitem                            # = register spill bytes (DRAM-backed)
```

Derived / sysinfo (NOT separate `pmc_perf.csv` columns — compute or look up):
```
Wavefronts_Per_Workgroup  = ceil(prod(Workgroup_Size) / 64)
Total_Workgroups          = prod(Grid_Size) / prod(Workgroup_Size)
Compute_Unit_Count        # sysinfo.csv: cu_per_gpu (304 on MI300X SPX/NPS1; 256 on MI355X)
XCD_Count                 # sysinfo.csv: num_xcd (1 in SPX/NPS1 visible context, 8 physical
                          # on both gens; MI300X = 4 IODs × 2 XCDs/IOD = 8 XCDs;
                          # MI355X = 2 IODs × 4 XCDs/IOD = 8 XCDs)
Achieved_Occupancy        # rocprof-compute SoL block (`-b 2`) / wavefront block (`-b 5`),
                          # NOT in pmc_perf.csv as a single column
Theoretical_Occupancy     # rocprof-compute wavefront block (`-b 5`)
```

> **CDNA3 register model:** each SIMD has 256 VGPRs **plus** 256 AGPRs (Accumulator GPRs). MFMA reads source operands from VGPR/AGPR and writes accumulator results to AGPR. CDNA3 introduced free movement between VGPR and AGPR, but the budget is *two separate pools* — a register-pressure report listing 256 "VGPRs" is hiding the AGPR half. rocprof-compute reports both.

**Reading:**

- **Waves per CU < 1**: grid is too small to fill the chip. On MI300X with 304 CUs (SPX/NPS1), if `Total_Workgroups < 304 × workgroups_per_CU` (compute `Total_Workgroups` as `prod(Grid_Size) / prod(Workgroup_Size)`), some CUs sit idle the entire time. Even more brutal in NPS2/NPS4 or CPX modes where the kernel sees fewer CUs per visible GPU.
- **Waves per CU in [1, 2)**: you have a tail wave (partial last wave). Tail effect magnitude is roughly `(last_wave_blocks / wave_size) × (block_exec_time / total_kernel_time)`.
- **Waves per CU > 4**: grid is plenty big, scheduling averages out.
- **Theoretical occupancy 100% but achieved << 100%**: stalls are the bottleneck, not launch config. Move to Dimension 3.
- **Theoretical occupancy < 100% and VGPR (or AGPR) is the tightest limiter**: reduce register usage or add `__launch_bounds__`.
- **LDS** the tightest: workgroup LDS budget too large; reduce tile size or split. LDS budget is **64 KB/CU on MI300X** and **160 KB/CU on MI355X** (CDNA4 enlarged LDS 2.5× with 2× read BW) — so on MI355X you genuinely have room for bigger tiles before LDS becomes the occupancy limiter; on MI300X, shrinking/splitting is still the fix.

**Derived: wave math**

```python
waves_per_wkg = ceil(workgroup_size / 64)              # CDNA gfx9 wave = 64
wkgs_per_cu_vgpr = min_limit_from_vgpr_budget
wkgs_per_cu_lds  = min_limit_from_lds_budget
# HW wave limit: max 32 waves/CU = 8 waves/SIMD × 4 SIMDs on gfx942 / gfx950.
# Convert to a workgroup ceiling by dividing by waves_per_wkg; gfx9 also has a
# hard 16 workgroups/CU cap independent of register / LDS pressure.
wkgs_per_cu_waves = 32 // waves_per_wkg
wkgs_per_cu_max   = min(wkgs_per_cu_waves, 16)
wkgs_per_cu = min(wkgs_per_cu_vgpr, wkgs_per_cu_lds, wkgs_per_cu_max)
# Distinct from the per-wave WAVE_SIZE=64 hardware constant — this is the
# number of workgroups that can be co-resident on the whole chip in one
# scheduling pass.
wkgs_in_flight = wkgs_per_cu * num_cus
num_waves = ceil(total_workgroups / wkgs_in_flight)
last_wave_blocks = total_workgroups - (num_waves - 1) * wkgs_in_flight
last_wave_utilization_pct = last_wave_blocks / wkgs_in_flight * 100
```

**MI300X partition note:** in SPX/NPS1 mode the kernel sees 1 GPU × 304 CU. In CPX mode it sees 8 GPUs × 38 CU each, with separate HBM partitions. Recompute the wave math against the partition the run actually used. `rocm-smi --showcomputepartition --showmemorypartition` + `rocminfo` confirm the active mode.

**Helper:** `analyze_reports.py` prints all the key launch counters under "Launch geometry" in the output txt.

---

## Dimension 2 — Workgroup balance (tail effect)

**What:** are workgroups finishing at roughly the same time, or do a few outliers drag out the kernel?

> Requires the optional `--timeseries-sampling-rate` collection pass (Recipe 2b in [`03-collection.md`](03-collection.md)) for the timeline signal. The static `pmc_perf.csv` from Recipe 2 only gives the per-kernel average — it cannot reveal a tail.

**Counters / signals:**

```
# Per-CU active-cycle distribution — not exposed as a dedicated rocprof-compute
# section in current releases; the workgroup-balance breakdown shows up in the
# CS / wavefront block (-b 5). For per-CU resolution, collect SQ_BUSY_CYCLES /
# GRBM_GUI_ACTIVE manually and aggregate by CU.
SQ_BUSY_CYCLES (per CU)
GRBM_GUI_ACTIVE                       # global active cycles
GRBM_CP_BUSY                          # cmd-processor busy
# (SPI scheduling busy is exposed via the SPI block, not GRBM, on gfx942/gfx950 —
# `rocprofv3 -L | grep SPI_` to enumerate per-install)

# Timeseries (rocprof-compute --timeseries-sampling-rate)
SQ_WAVES                              # rises with workgroup dispatch, falls with completion
SQ_INSTS_VALU
SQ_WAIT_INST_ANY                      # broad instruction-issue stall (gfx942/gfx950 PMC)
SQ_WAIT_INST_LDS                      # LDS-issue stall (gfx942/gfx950 PMC)
# Granular VMEM/SMEM/FLAT/BARRIER/VMCNT/LGKMCNT classification comes from PC sampling only

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

Ratios > 3x indicate potential tail effect — investigate (see Pattern B in [`06-diagnosis-playbook.md`](06-diagnosis-playbook.md), which uses the same `max/avg > 3` action threshold).

---

## Dimension 3 — Stall reason breakdown + per-line hotspots

**What:** when waves aren't issuing, what are they waiting for? Which source lines generate the most stalls?

**Aggregate hardware wait counters (verified via `rocprofv3 -L` on gfx950):**

> Only three `SQ_WAIT_*` PMC counters exist on gfx942 / gfx950:
> `SQ_WAIT_ANY`, `SQ_WAIT_INST_ANY`, `SQ_WAIT_INST_LDS`.
> The granular wait-reason classification is **NOT exposed as PMC counters** on gfx942 /
> gfx950 — it comes only from PC sampling's **stochastic** CSV (`Stall_Reason` column).
> The `host_trap` mode does NOT populate `Stall_Reason`; it only gives sampled PCs (per-line
> hotspots). See AMD's PC-sampling docs:
> https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-pc-sampling.html

```
# Coarse "any wait" PMC counters (gfx942 / gfx950 verified set)
SQ_WAIT_ANY                            # any wait state (broadest signal)
SQ_WAIT_INST_ANY                       # waiting on any instruction-side resource
SQ_WAIT_INST_LDS                       # LDS instruction issue stall (covers bank-conflict serialization)

# Issue / activity (verified)
SQ_INSTS_VALU
SQ_INSTS_MFMA                          # aggregate MFMA — per-dtype MOPs use `SQ_INSTS_VALU_MFMA_MOPS_<DTYPE>` (see §08)
SQ_INSTS_VMEM_RD, SQ_INSTS_VMEM_WR
SQ_INSTS_LDS
SQ_INSTS_FLAT
SQ_BUSY_CYCLES
SQ_WAVES
SQ_LDS_BANK_CONFLICT
```

**Per-PC stall metrics — the ONLY source of the granular wait-reason classification on gfx942/gfx950** (from `rocprofv3 --pc-sampling-beta-enabled --pc-sampling-method stochastic`; the `host_trap` mode does NOT populate `Stall_Reason`):

PC sampling's stochastic CSV (`<pid>_pc_sampling_stochastic.csv`) has one row per sample with these columns: `Sample_Timestamp`, `Exec_Mask`, `Dispatch_Id`, `Instruction`, `Instruction_Comment`, `Correlation_Id`, `Wave_Issued_Instruction`, `Instruction_Type`, `Stall_Reason`, `Wave_Count`. The `Stall_Reason` column is populated only when `Wave_Issued_Instruction == 0`. Aggregate by `Stall_Reason` and by source line (via the `Instruction` PC + `addr2line` against the binary built with `-gline-tables-only`) — see [`04-python-api.md`](04-python-api.md) example. The `host_trap` CSV (`<pid>_pc_sampling_host_trap.csv`) lacks `Stall_Reason`; use it only for per-line hotspots, not a breakdown.

```
Stall_Reason enum values (from ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_*
in /opt/rocm/include/rocprofiler-sdk/pc_sampling.h — verify exact values per install):

NONE                         # sentinel; should not appear on stalled rows
NO_INSTRUCTION_AVAILABLE     # front-end empty (I-cache miss / fetch stall)
ALU_DEPENDENCY               # VALU / MFMA result not yet ready
WAITCNT                      # explicit s_waitcnt drain (vmcnt / lgkmcnt / expcnt)
INTERNAL_INSTRUCTION         # internal microcode / fixed-latency op in flight
BARRIER_WAIT                 # waiting at s_barrier (workgroup sync)
ARBITER_NOT_WIN              # lost issue-arbitration round
ARBITER_WIN_EX_STALL         # won arbitration but execution unit busy
OTHER_WAIT                   # catch-all
SLEEP_WAIT                   # wave in s_sleep
```

> **Note — JSON-only per-pipe snapshot.** The finer per-execution-pipe state
> (`arb_state_stall_valu`, `arb_state_stall_matrix`, `arb_state_stall_lds`,
> `arb_state_stall_lds_direct`, `arb_state_stall_scalar`, `arb_state_stall_vmem_tex`,
> `arb_state_stall_flat`, `arb_state_stall_exp`, `arb_state_stall_misc`,
> `arb_state_stall_brmsg`, with matching `arb_state_issue_*`) is **NOT** in the CSV. These
> are 1-bit fields of the `rocprofiler_pc_sampling_snapshot_v0_t` C struct in
> `/opt/rocm/include/rocprofiler-sdk/pc_sampling.h` and surface only in the JSON output —
> collect with `rocprofv3 ... -f json` instead of `-f csv` and read them from the
> `snapshot` object of each PC-sample record.

Stall-reason enum values vary between ROCm versions; the canonical list for your install
is the enum in `/opt/rocm/include/rocprofiler-sdk/pc_sampling.h` and the actual
`Stall_Reason` values present in your stochastic CSV. Treat the list above as a label set
to match against your actual data, not as a hard guarantee.

**Stall reasons you need to know (approximate NVIDIA analogs in parens):**

| `Stall_Reason` value (stochastic CSV) | Meaning | Typical cause | Fix direction | NVIDIA analog |
|---|---|---|---|---|
| `WAITCNT` | explicit `s_waitcnt` drain (`vmcnt` / `lgkmcnt` / `expcnt`) | outstanding global / vL1 / LDS / scalar memory op — uncoalesced load, latency-bound, or compiler-inserted ordering | coalesce, reuse, add ILP, use `global_load_lds`, unroll between waitcnts | `long_scoreboard` (vmem) / `short_scoreboard` (lds) / `membar` |
| `ALU_DEPENDENCY` | VALU / MFMA result not yet ready | long VALU dep chain, VALU port pressure, or matrix-core throughput limit / AGPR dep | break dep chain, more ILP, reorder MFMA tiles, reduce AGPR pressure | `math_pipe_throttle` / (partial) `wait` |
| `BARRIER_WAIT` | waiting at `s_barrier` (workgroup sync) | other waves haven't arrived; too many barriers, divergent work per wave | reduce barriers, fix divergence, more work per workgroup | `barrier` |
| `ARBITER_NOT_WIN` | lost issue-arbitration round | issue-slot contention with other waves | n/a — typically benign in isolation | `dispatch_stall` (proxy) |
| `ARBITER_WIN_EX_STALL` | won arbitration but execution unit busy | pipe contention downstream of arbitration | similar to `ALU_DEPENDENCY` — break dep chain, more ILP | `math_pipe_throttle` / `mio_throttle` |
| `INTERNAL_INSTRUCTION` | internal microcode / fixed-latency op in flight | misc fixed-latency wait | usually not actionable | `wait` |
| `NO_INSTRUCTION_AVAILABLE` | front-end empty (fetch stall) | I-cache miss; corroborate with `SQC_*` PMCs | shrink hot code, improve I-cache locality | `no_instruction` |
| `OTHER_WAIT` | catch-all | anything not in the above buckets | inspect ISA at the hotspot | (no direct analog) |
| `SLEEP_WAIT` | wave in `s_sleep` | explicit sleep (rare on compute) | remove the sleep | (no direct analog) |
| productive issue: `Wave_Issued_Instruction == 1` | actually issuing this cycle | **productive** | ignore | `selected` |

LDS bank-conflict stalls surface in two places: aggregate cycles via the PMC `SQ_WAIT_INST_LDS`, and per-sample via `Stall_Reason == WAITCNT` on LDS-instruction (`ds_read_*` / `ds_write_*`) source lines. The CSV `Stall_Reason` alone does not split LDS waits from global-memory waits — distinguish via the ISA mnemonic at the sampled PC (`Instruction_Comment`).

**Reading the aggregate counters:** normalize by `SQ_BUSY_CYCLES`. For example `SQ_WAIT_INST_LDS / SQ_BUSY_CYCLES = 0.45` means waves spent 45% of busy cycles waiting on LDS. For the granular wait-reason breakdown, count stochastic PC-sampling samples per `Stall_Reason`.

**Reading the PC-sampling percentages:** sum `Sample_Count` (or count rows) over all stochastic samples = total samples. Per-`Stall_Reason` fraction = "% of samples stalled on X". Rules of thumb:

- **`WAITCNT` > 40% of samples concentrated on `global_load_*` / `flat_load_*` lines**: kernel is memory-latency-bound. Check Dimension 6 (access patterns) next.
- **`WAITCNT` on `ds_*` (LDS) lines + non-zero `SQ_LDS_BANK_CONFLICT`**: LDS bank conflicts or long dep chains.
- **`BARRIER_WAIT` > 20% of samples (or `s_barrier` source-line hotspot)**: too much synchronization, or wave divergence before a barrier.
- **`ALU_DEPENDENCY` dominates on MFMA-heavy kernels**: matrix-pipe throughput limit / AGPR dep — reorder MFMA tiles.
- **`Wave_Issued_Instruction == 1` fraction < 10%**: very little actual issue — the whole kernel is stall-bound.

**Helper:** `extract_stall_hotspots.py` produces `stall_hotspots_<tag>.txt` which ranks source lines by total stall samples. This directly points at the offending `global_load_dwordx4`, `s_barrier`, `ds_read_b128`, or compute op in source.

---

## Dimension 4 — Matrix-Core (MFMA) utilization

**What:** is the kernel using Matrix Cores at all? If yes, how well?

**Counters (rocprof-compute compute-pipe block, `-b 10`, formerly §2.1.10):**

> MFMA per-dtype counters are named `SQ_INSTS_VALU_MFMA_MOPS_<DTYPE>` on gfx942 / gfx950
> ("MOPS" = matrix-ops). The canonical `<DTYPE>` set is `F16, BF16, F32, F64, I8, F8, BF8, XF32`
> on **both** gfx942 and gfx950; gfx950 adds `F6F4` (covers MXFP4/MXFP6 traffic — there is no
> separate MXFP4 or MXFP6 PMC suffix). See [`reference/08-mi300x-mi355x-counter-names.md`](08-mi300x-mi355x-counter-names.md)
> for the authoritative list. Always verify against `rocprofv3 -L | grep MFMA` on your install.
> The aggregate `SQ_INSTS_MFMA` is always available.

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
- **Matrix-Core busy % > 50%** (rocprof-compute SoL "matrix engine busy %" line, computed as `SQ_VALU_MFMA_BUSY_CYCLES / GRBM_GUI_ACTIVE * 100`): kernel is doing well on the Matrix-Core front. Focus elsewhere.
- **Matrix-Core busy %** much lower than expected on a matmul-ish kernel: data isn't arriving fast enough (Dimension 6) or tile sizes don't fit the MFMA shape.

**MI300X (CDNA3 / gfx942) MFMA notes:**

- Supported shapes for F32/F16/BF16: `mfma_*_{4x4x4, 16x16x4, 16x16x16, 32x32x4, 32x32x8}`. **I8 uses larger K**: `v_mfma_i32_{16x16x32, 32x32x16}i8` (K doubled, since INT8 packs 4 bytes per 32-bit register; note legacy I8/F16/BF16/F64 dtype suffixes glue to the tile shape with no underscore, while FP8/BF8/F8F6F4/XF32 forms use an underscore separator — e.g. `v_mfma_f32_16x16x32_fp8_fp8`). FP8 variants via OCP-FNUZ; FP64 added.
- Accumulators live in **AGPR** (not VGPR). When `Scratch_Per_Workitem > 0` *and* MFMA-heavy, suspect over-allocation of AGPR forcing spill.
- Use `v_mfma_f32_32x32x8bf16_1k` (or `v_mfma_f32_16x16x16bf16_1k`) over the older 16x16x4 — same throughput per cycle but better register reuse. The `_1k` suffix is the AMD-canonical name in the LLVM/AMDGPU back-end for the 1-block form.

**MI355X (CDNA4 / gfx950) MFMA notes:**

- New block-scaled `F6F4` family added (`v_mfma_*_f6f4` / `v_mfma_scale_*` with E8M0 block-exponent operand), covering FP4 and FP6 storage formats. Canonical tile shapes are `16x16x128` and `32x32x64` for both the unscaled `v_mfma_f32_*_f8f6f4` and the scaled `v_mfma_scale_f32_*_f8f6f4` forms (see the table in [`../cdna3-cdna4-hip-programming.md`](../cdna3-cdna4-hip-programming.md)). Verified per-dtype PMC counters on gfx950 include `SQ_INSTS_VALU_MFMA_MOPS_F6F4` and `SQ_INSTS_VALU_MFMA_MOPS_XF32`; check `rocprofv3 -L | grep MFMA` for the exact suffix list your install exposes (raw `_F4` / `_F6` / `_MXFP4` / `_MXFP6` / `_MXFP8` are **not** distinct PMC counters on gfx950).
- `XF32` (extended-FP32) MFMA exposed (`SQ_INSTS_VALU_MFMA_MOPS_XF32`); TF32 from prior gens does not exist on CDNA4.
- FP64 throughput **halved** vs CDNA3 — gfx950 is not the GPU for FP64-dense workloads.
- 2:4 sparse MFMA variants added (check ISA documentation for exact opcodes).
- FP8 switched from OCP-FNUZ (CDNA3) to OCP standard (CDNA4); numerics differ slightly.

**Fix direction:** if you see 0% MFMA and the workload is matrix-multiplication-shaped, redesign around MFMA. This is usually a major refactor but gives 2-10× on compute-bound paths. Most projects should use **Composable Kernel** (CK) or **hipBLASLt** instead of hand-rolling, the same way most NVIDIA projects use CUTLASS / cuBLAS.

---

## Dimension 5 — CU utilization timeline

**What:** how does CU utilization vary over the kernel's lifetime?

> Requires the optional `--timeseries-sampling-rate` collection pass (Recipe 2b in [`03-collection.md`](03-collection.md)). The default Recipe 2 produces only per-kernel averages.

**Counters (rocprof-compute timeseries mode):**
```
GRBM_GUI_ACTIVE                             # global active cycles (denominator for utilization)
GRBM_COUNT                                  # total elapsed cycles since last reset
SQ_BUSY_CYCLES                              # per-CU
SQ_WAVES                                    # in-flight waves
SQ_WAIT_INST_LDS, SQ_WAIT_INST_ANY          # over time (the only granular SQ_WAIT_* PMCs on gfx942/gfx950)
TCC_EA0_RDREQ_sum                           # HBM read pressure over time (TCC_EA1_* does NOT exist on gfx942/gfx950)
```

**Reading (timeline shapes):**

- **Flat high, clean drop**: ideal.
- **Flat high, long tail**: tail effect (Dimension 2).
- **Flat low**: grid too small (Dimension 1) or severely stall-bound (Dimension 3).
- **Periodic sawtooth (compute ↕ memory)**: no compute-memory overlap — missing double-buffering / prefetch.
- **Slow ramp up, flat middle, clean drop**: kernel has warmup work (prologue), then steady state. Usually fine.
- **Per-XCD divergence on MI300X**: plot SQ_BUSY per XCD; > 30% gap between fastest and slowest XCD signals scheduling imbalance.

**Helper:** `plot_timeline.py` — renders ASCII plots. Look at multiple series side-by-side (SQ_WAVES + HBM RDREQ + the count of `Stall_Reason == WAITCNT` samples on global-load lines per time bucket) to distinguish the shapes.

**Note:** rocprof-compute's timeseries minimum interval is ~1 ms (much coarser than NVIDIA PM sampling's ~2 µs). For very short kernels (< 100 µs) prefer ATT for time-resolved per-CU activity; rocprof-compute timeseries is for longer kernels and full-app traces.

---

## Dimension 6 — Memory access pattern & cache efficiency

**What:** are global loads coalesced? Are caches hit? Is HBM actually busy?

**Counters (rocprof-compute L1D / L2 / HBM blocks, `-b 15` / `-b 16` / `-b 17`, formerly §2.1.15 / §2.1.16 / §2.1.17):**
```
# HBM (TCC_EA = "Effective Address" memory subsystem)
# On gfx942 / gfx950, only `TCC_EA0_*` exists — there is NO `TCC_EA1_*` (a
# second EA channel was a gfx906/gfx908 thing). Verify with `rocprofv3 -L | grep TCC_EA`.
TCC_EA0_RDREQ_sum                              # HBM read requests (count)
TCC_EA0_RDREQ_32B_sum                          # 32B-sized read requests
TCC_EA0_WRREQ_sum                              # HBM write requests
TCC_EA0_WRREQ_64B_sum
# NOTE: there is NO `TCC_EA0_REQ` aggregate on gfx942/gfx950. For total L2
# requests use `TCC_REQ_sum` (L2-side); for EA-side totals, sum the
# `TCC_EA0_RDREQ_sum` + `TCC_EA0_WRREQ_sum` rows above.
TCC_EA0_MEM_REQ_LATENCY                        # latency histogram (if collected)
# Achieved HBM BW ≈ (TCC_EA0_RDREQ_32B_sum * 32 + TCC_EA0_WRREQ_64B_sum * 64) / kernel_time

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
- **Bytes per wavefront (rocprof-compute reports this in `-b 11` "Instruction Mix" / `-b 15` "L1D Cache")**: peak is opcode-dependent — 256 B for `global_load_dword` (4 B/lane × 64 lanes), 512 B for `global_load_dwordx2`, 1024 B for `global_load_dwordx4` (16 B/lane × 64 lanes). Substantially less than the opcode peak means under-vectorization, gather/scatter, or stride > 1 access; values above the peak indicate uncoalesced gathers.
- **`SQ_LDS_BANK_CONFLICT > 0`**: bank conflicts. The 32-bank LDS serializes same-bank accesses. Pad tile dims or swizzle indices. LDS budget is **64 KB/CU on MI300X** and **160 KB/CU on MI355X** (CDNA4 enlarged LDS 2.5× with 2× read BW) — on MI355X the padding-cost / occupancy trade is much cheaper since you have headroom for the +1-dword padding plus a bigger tile.
- **`Scratch_Per_Workitem > 0` (from launch info) or rocprof-compute SoL "Scratch" non-zero**: **register spill** — very bad, scratch is HBM-backed. Reduce VGPR/AGPR pressure with `__launch_bounds__` or kernel splitting.
- **`SQ_INSTS_LDS == 0`**: kernel uses no LDS. Fine for element-wise kernels; often a missed optimization for data-reuse-heavy kernels.

rocprof-compute's L2-fabric / HBM block (`-b 17`, formerly §2.1.17) reports the HBM utilization in the same "Speed-of-Light" form NCU uses:
```
HBM        Avg     Min     Max     Pct Peak
Bandwidth  X TB/s  ...     ...     Y%
```

**MI300X-specific reading:**

- **Per-XCD HBM** (NPS2/NPS4 partitions): each XCD has its own HBM stack. A workgroup running on XCD-N pays cross-XCD XGMI hops if it touches data placed on XCD-M's stack. On gfx942/gfx950 the per-XCD HBM-channel counter exposed in PMC is `TCC_EA0_*` only (a single channel per XCD). The older two-channel `TCC_EA0_* + TCC_EA1_*` formula seen in gfx906/gfx908 docs does **not** apply on MI300X/MI355X — `TCC_EA1_*` does not exist on these gens.
- **Infinity Cache (256 MB shared L3-ish)**: on a kernel that re-reads a working-set < 256 MB, an Infinity Cache hit (visible as TCC_HIT without going to HBM) is much cheaper than HBM. If the working set is just over 256 MB, even a tiny tiling change can swing perf hugely.

**MI355X-specific reading:**

- **`global_load_lds` widened on CDNA4** — direct HBM/L2→LDS path bypassing VGPR has wider per-lane variants on gfx950 than on gfx942. Check ISA / LLVM intrinsics for the exact widths your toolchain emits. If you see `SQ_INSTS_VMEM_RD` high *and* `SQ_INSTS_VALU` low on the data-loading lines, you're probably already using it.
- **Infinity Cache (256 MB)** is retained on MI355X but coupled to HBM3E (8.0 TB/s) instead of HBM3 — the working-set sweet spot is similar in size to MI300X but the HBM penalty for missing it is smaller.
- **FP4 / FP6 / MXFP** reduce HBM BW pressure on the same workload by 2-4× relative to FP8/FP16. The per-dtype `SQ_INSTS_VALU_MFMA_MOPS_*` counter for block-scaled MX formats is install-specific — `rocprofv3 -L | grep -i mfma` shows what your build exposes.

NCU's rule engine often reports the coalescing issue directly; rocprof-compute analogs are in the Instruction Mix block (`-b 11`, formerly §2.1.11 — "Bytes per wavefront", "Access pattern utilization") and the L1D block (`-b 15`):
```
Instruction Mix — Bytes per wavefront               Avg  Min  Max  Peak
  global_load_dwordx4 (vector load 16B/lane × 64)   62   16   1024 1024
```

A value of 62 (out of 1024 peak for `global_load_dwordx4`) means roughly 1 of 16 bytes per lane are actually used — significant coalescing problem. (For a plain `global_load_dword` the peak is 256 B/wave; for `dwordx2` it's 512 B/wave.)

**Fix directions (by pattern):**

- Strided access (e.g. `x[lane * stride + i]`): change the per-thread layout so lanes access contiguous elements.
- AoS → SoA: restructure the data.
- Sparse writes: pack writes into a single coalesced `global_store_dwordx4` at the end of the wave.
- Register spill: add `__launch_bounds__`, reduce intermediate variables, or split kernel.
- LDS bank conflict: pad to 33 (or 65) instead of 32 (or 64); or apply swizzle.

---

## Cross-dimension synthesis

After walking through all six, write a one-line diagnosis combining them. Structure: name the top 3–4 signals, each tied to a specific dimension. For example:

> "The kernel runs at X% of peak HBM and Y% of peak MFMA throughput (Dim 1, 6). Wait time is dominated by `Stall_Reason == WAITCNT` on global-load lines (Z% of stochastic PC samples, Dim 3), concentrated on <N> source lines whose access pattern is <coalesced/uncoalesced/...> (Dim 6). The PMC timeline shows <flat / tail / sawtooth> shape (Dim 2/5), with <even / N% imbalance across XCDs>. Matrix Cores <used / unused at W%> (Dim 4)."

Fill in the X/Y/Z/W/N values and <classifications> from your own report. That sentence is the deliverable. Everything else in the report is evidence backing it.
