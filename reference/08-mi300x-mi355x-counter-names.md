# MI300X (gfx942) & MI355X (gfx950) PMC Counter Reference

AMD's PMC namespace is partitioned by IP block: `SQ_*` (Shader Engine/Wave scheduler), `SQC_*` (scalar/instr cache), `TA_*` / `TD_*` (texture address/data, used by vector mem), `TCP_*` (vL1 — per-CU vector L1), `TCC_*` (L2 cache, per-channel), `TCC_EA0_*` (HBM "Effective Address" channel — **only `EA0` exists on gfx942/gfx950**; `TCC_EA1_*` does NOT exist on these gens), `GRBM_*` (graphics register bus master, the GPU-wide cycle/active counters), `GL2C_*` (legacy alias for L2 on some builds), and `CPC_*` / `CPF_*` (command processor).

When in doubt, enumerate:

```bash
rocprofv3 -L > /tmp/all_counters.txt                    # the canonical authority for your ROCm install
# Equivalent long form (rocprofv3 on ROCm 6.2+):
rocprofv3 --list-avail > /tmp/all_counters.txt
# NOTE: the rocprof v1/v2 flags `--list-counters`, `--list-metrics`,
# `--list-basic`, `--list-derived` are all gone — use `-L` / `--list-avail`.
# On newer ROCm builds, the companion CLI is `rocprofv3-avail list --pmc`.
# Or, programmatically from a collected report:
python3 -c "
import pandas as pd
print('\n'.join(sorted(pd.read_csv('rpc_<tag>/pmc_perf.csv', nrows=1).columns)))
"
```

> **Always verify** any counter name in this doc against `rocprofv3 -L` output on your
> ROCm install before relying on it. The names below are observed on gfx942 / gfx950
> with ROCm 6.4 / 7.x, but the PMC namespace shifts between releases and a missing
> counter will silently produce an empty column (not an error).

---

## Counter names that changed across gfx generations

| gfx906 / gfx908 / gfx90a (MI50 / MI100 / MI200) | gfx942 (MI300X) / gfx950 (MI355X) |
|---|---|
| `TCC_HIT_sum` (single L2 channel) | **`TCC_HIT_sum`** — still the aggregate name; per-channel hit breakdown is via rocprof-compute, not a stable `TCC_HIT<i>_sum` PMC |
| `TCC_EA_RDREQ_sum` | **`TCC_EA0_RDREQ_sum`** — single EA channel per XCD on gfx942/gfx950 (NO `TCC_EA1_*`) |
| `TCC_EA_RDREQ_32B_sum` | **`TCC_EA0_RDREQ_32B_sum`** |
| `TCC_EA_WRREQ_sum` | **`TCC_EA0_WRREQ_sum`** |
| `TA_BUSY_avr` | **`TA_BUSY_avr`** — same name; per-SIMD breakdown comes via rocprof-compute aggregates, not a `_per_simd` PMC |
| `SQ_INSTS_MFMA` exists on gfx908+ | Same aggregate name; per-dtype detail moves to `SQ_INSTS_VALU_MFMA_MOPS_<DTYPE>` on gfx942+ (per-shape PMCs not stable across ROCm versions) |
| `SQ_BUSY_CYCLES` (gfx9 family) | Same, but on MI300X read **`GRBM_GUI_ACTIVE`** for true GPU-wide active cycles (denominator) |
| `SQ_INSTS_VALU_MFMA_MOPS_*` (no F6F4) | **gfx950 adds** `SQ_INSTS_VALU_MFMA_MOPS_F6F4` (block-scaled FP4/FP6 family). Note: `SQ_INSTS_VALU_MFMA_MOPS_XF32` exists on **both** gfx942 and gfx950 (AMD's XF32 = NVIDIA TF32 equivalent). Verify the exact suffix list with `rocprofv3 -L \| grep MFMA` |
| `SQ_WAVES` (total wavefronts) | Same name everywhere |

---

## Canonical MI300X / MI355X counter set (curated)

These PMC names have been confirmed to exist and return meaningful values on gfx942 and gfx950 with ROCm 6.4 / 7.x. Always verify for your specific build with `rocprofv3 -L` (long form `--list-avail`). The legacy `--list-metrics` / `--list-counters` / `--list-basic` / `--list-derived` flags are rocprof v1/v2 only and not part of rocprofv3.

### Launch / wave geometry (from `pmc_perf.csv` per-dispatch columns, not PMCs)
```
Dispatch_ID
Kernel_Name
GPU_ID, Queue_ID, PID, TID
Grid_Size, Workgroup_Size
LDS_Per_Workgroup            # bytes
Scratch_Per_Workitem         # bytes = register spill volume per work-item
Arch_VGPR                    # per-work-item architectural VGPR count (NOT "VGPRs")
Accum_VGPR                   # per-work-item AGPR pool count on CDNA3+ (NOT "AGPRs")
SGPR                         # per-wavefront SGPR (singular, NOT "SGPRs")
# NOTE: pmc_perf.csv does NOT include a Wave_Size column on gfx942/gfx950 —
# wave size is fixed at 64 on CDNA and reported in sysinfo.csv.
# pmc_perf.csv also does NOT include Start_Timestamp / End_Timestamp on
# rocprof-compute in ROCm 7.x — use the kernel_trace.csv produced by
# `rocprofv3 --kernel-trace` for timing.
```

### Shader (SQ) — wave activity & instruction mix
```
SQ_WAVES                        # total wavefronts dispatched
SQ_INSTS                        # total instructions issued
SQ_INSTS_VALU                   # vector ALU (V_*)
SQ_INSTS_SALU                   # scalar ALU (S_*)
SQ_INSTS_VMEM                   # vector memory (buffer/global)
SQ_INSTS_VMEM_RD                # vmem reads
SQ_INSTS_VMEM_WR                # vmem writes
SQ_INSTS_SMEM                   # scalar memory
SQ_INSTS_FLAT                   # flat / generic addressing
SQ_INSTS_LDS                    # LDS instructions
SQ_INSTS_GDS                    # GDS (rarely used)
SQ_INSTS_BRANCH                 # branches
SQ_INSTS_MFMA                   # matrix-core total
# Per-dtype MFMA op counts use the SQ_INSTS_VALU_MFMA_MOPS_<DTYPE> family —
# see the dedicated MFMA section below. The legacy SQ_INSTS_MFMA_F32_<TILE>
# per-shape names are NOT a stable PMC set across ROCm releases; prefer the
# MOPS family and verify with `rocprofv3 -L | grep -i mfma`.
SQ_INSTS_VALU_TRANS_F16         # transcendentals
SQ_INSTS_VALU_TRANS_F32
SQ_BUSY_CYCLES                  # cycles SQ was issuing on any SE
SQ_ACTIVE_INST_VALU             # VALU active cycles (if exposed by this ROCm)
SQ_ACTIVE_INST_VMEM             # VMEM active cycles (if exposed by this ROCm)
```

### Wait / stall reasons (SQ) — analog of NVIDIA `smsp__average_warps_issue_stalled_*`

**Only three `SQ_WAIT_*` PMCs exist on gfx942 / gfx950** (verified via `rocprofv3 -L`):

```
SQ_WAIT_ANY                    # any wait state (broadest signal)
SQ_WAIT_INST_ANY               # waiting on any instruction-side resource
SQ_WAIT_INST_LDS               # LDS instruction issue stall (covers bank-conflict serialization)
SQ_INST_LEVEL_LDS              # outstanding LDS level (peak concurrency)
```

The granular wait-reason classification seen in older gfx9 docs is **NOT exposed as PMC
counters** on gfx942 / gfx950 — that classification comes ONLY from PC sampling's
**stochastic** mode (the `Stall_Reason` column in `<pid>_pc_sampling_stochastic.csv`).
The `host_trap` mode does NOT populate `Stall_Reason` — it only emits sampled PCs
(per-line hotspots). The authoritative enum is
`ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_*` in
`/opt/rocm/include/rocprofiler-sdk/pc_sampling.h`, with values `NONE`,
`NO_INSTRUCTION_AVAILABLE`, `ALU_DEPENDENCY`, `WAITCNT`, `INTERNAL_INSTRUCTION`,
`BARRIER_WAIT`, `ARBITER_NOT_WIN`, `ARBITER_WIN_EX_STALL`, `OTHER_WAIT`, `SLEEP_WAIT`.
See https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-pc-sampling.html

(The finer per-execution-pipe `arb_state_stall_*` / `arb_state_issue_*` bit-fields are
JSON-only — collect with `rocprofv3 ... -f json` and read the `snapshot` object on each
PC-sample record. They are not CSV columns.)

Use rocprof-compute's derived stall totals (see the wavefront-stall breakdown in the
per-block dump) and the stochastic-PC-sampling `Stall_Reason` aggregation for the
categorical split.

### IPC / occupancy
```
SQ_BUSY_CYCLES / GRBM_GUI_ACTIVE     # GPU-wide SQ busy ratio
SQ_WAVES / GRBM_GUI_ACTIVE           # wave throughput
SQ_INSTS / SQ_BUSY_CYCLES             # IPC across all SIMDs (per-SE — divide by # SEs for per-SIMD)
SQ_ACCUM_PREV_HIRES                  # achieved occupancy (waves/SIMD/cycle) — populated by rocprof-compute wavefront block (-b 5)
```

### vL1 cache (TCP — per-CU vector L1)
```
TCP_TOTAL_CACHE_ACCESSES_sum
TCP_TCC_READ_REQ_sum                  # vL1 miss requests sent to L2 (reads)
TCP_TCC_WRITE_REQ_sum                 # writes
TCP_TCC_ATOMIC_WITH_RET_REQ_sum       # atomics with return
TCP_TCC_ATOMIC_WITHOUT_RET_REQ_sum    # atomics fire-and-forget
TCP_TCP_LATENCY / TCP_TCP_LATENCY_sum # average vL1 latency (cycles)
TCP_GATE_EN1                          # gate-enable cycles
TCP_GATE_EN2
TCP_TA_TCP_STATE_READ_sum             # vL1 reads from TA
TCP_PENDING_STALL_CYCLES_sum
TCP_VOLATILE_sum                      # volatile (uncached) accesses
```

vL1 hit rate (computed): `1 - (TCP_TCC_READ_REQ_sum + TCP_TCC_WRITE_REQ_sum) / TCP_TOTAL_CACHE_ACCESSES_sum`

### L2 cache (TCC — aggregate on MI300X / MI355X)

Verified PMC names on gfx942 / gfx950 (`rocprofv3 -L | grep '^TCC_'`):
```
TCC_HIT_sum                           # aggregate L2 hits
TCC_MISS_sum                          # aggregate L2 misses
TCC_REQ_sum                           # total L2 requests
TCC_ATOMIC_sum                        # atomic ops at L2
TCC_BUBBLE_sum                        # cycles the L2 pipe stalled (no work issued)
TCC_NORMAL_WRITEBACK_sum              # standard L2 → HBM writebacks
TCC_NC_REQ_sum                        # non-coherent requests
TCC_UC_REQ_sum                        # uncached requests
TCC_CC_REQ_sum                        # coherent requests
TCC_RW_REQ_sum                        # read/write requests breakdown
```

`TCC_REQ_READ_sum` / `TCC_REQ_WRITE_sum` and per-channel `TCC_HIT<i>_sum` are **NOT in
the verified gfx942/gfx950 set**; they were valid on older gfx releases. Always confirm
with `rocprofv3 -L | grep '^TCC_'` on your install.

L2 hit rate: `TCC_HIT_sum / (TCC_HIT_sum + TCC_MISS_sum)`

### HBM / memory channel (TCC_EA0 — single EA channel per XCD on gfx942 / gfx950)

**`TCC_EA1_*` does NOT exist on gfx942 or gfx950.** Older gfx906/gfx908 docs showing a
two-channel `EA0 + EA1` formula do not apply. Enumerate with
`rocprofv3 -L | grep TCC_EA` to confirm what your install exposes.

```
TCC_EA0_RDREQ_sum                     # read requests issued to HBM
TCC_EA0_RDREQ_32B_sum                 # 32B-granular read requests
TCC_EA0_WRREQ_sum                     # writes
TCC_EA0_WRREQ_64B_sum                 # 64B-granular writes
TCC_EA0_RDREQ_DRAM_sum                # filtered to DRAM (excludes other agents)
TCC_EA0_WRREQ_DRAM_sum
TCC_EA0_ATOMIC_sum
TCC_EA0_RDREQ_CREDIT_STALL_sum        # cycles stalled waiting for read credits to EA
TCC_EA0_WRREQ_CREDIT_STALL_sum        # cycles stalled waiting for write credits to EA
# NOTE: bare `TCC_EA0_{RD,WR}REQ_IO_sum` (xGMI / PCIe split) is NOT a stable
# PMC on gfx942/gfx950. Verify with `rocprofv3 -L | grep TCC_EA0`.
```

Computed achieved HBM read BW (GB/s):
```
TCC_EA0_RDREQ_32B_sum * 32 / kernel_duration_seconds / 1e9
```

Peak HBM BW: 5.3 TB/s on MI300X (HBM3), 8.0 TB/s on MI355X (HBM3E).

### LDS (per-CU local data share)
```
SQ_LDS_BANK_CONFLICT                  # bank conflicts (counts cycles lost; ideally 0)
SQ_LDS_IDX_ACTIVE                     # indexed LDS accesses
SQ_LDS_ATOMIC_RETURN                  # LDS atomics with return
SQ_LDS_UNALIGNED_STALL                # unaligned LDS access stalls
SQ_INSTS_LDS                          # LDS instruction count
```

### Matrix-core / MFMA

The MFMA per-dtype counters on gfx942 / gfx950 use the prefix
**`SQ_INSTS_VALU_MFMA_MOPS_<DTYPE>`** ("MOPS" = matrix-ops). The exact set of `<DTYPE>`
suffixes exposed depends on the ROCm version — enumerate with
`rocprofv3 -L | grep -i MFMA`. The aggregate `SQ_INSTS_MFMA` is always available.

Verified `SQ_INSTS_VALU_MFMA_MOPS_<DTYPE>` suffixes (`rocprofv3 -L | grep MFMA`):
**F16, BF16, F32, F64, I8, F8, XF32** on gfx942; the same set **plus F6F4** on gfx950.
There is no dedicated `_BF8` counter — `E5M2` (BF8) inputs are bucketed under
`SQ_INSTS_VALU_MFMA_MOPS_F8` on both generations (including mixed `f8_bf8` MFMA shapes).

```
SQ_INSTS_MFMA                         # total MFMA issued (aggregate)
SQ_INSTS_VALU_MFMA_MOPS_F16           # by source dtype
SQ_INSTS_VALU_MFMA_MOPS_BF16
SQ_INSTS_VALU_MFMA_MOPS_F32
SQ_INSTS_VALU_MFMA_MOPS_F64           # CDNA3 full throughput; CDNA4 halved
SQ_INSTS_VALU_MFMA_MOPS_I8
SQ_INSTS_VALU_MFMA_MOPS_F8            # all FP8 MFMA (E4M3 + E5M2). FNUZ on gfx942, OCP-standard on gfx950.
                                      # E5M2 (BF8) issues are bucketed here too — no separate _BF8 counter.
SQ_INSTS_VALU_MFMA_MOPS_XF32          # XF32 (19-bit, TF32-equivalent) MFMA — present on BOTH gfx942 and gfx950
SQ_INSTS_VALU_MFMA_MOPS_F6F4          # gfx950 only — block-scaled FP6/FP4 family
# There are NO distinct `_F4` / `_F6` / `_MXFP4` / `_MXFP6` / `_MXFP8` PMC counters
# on gfx950 — the block-scaled MX-style formats roll up into `_F6F4`.
SQ_VALU_MFMA_BUSY_CYCLES              # cycles MFMA pipe was busy (proxy for "matrix-core busy")
```

Matrix-core busy %: `SQ_VALU_MFMA_BUSY_CYCLES / GRBM_GUI_ACTIVE * 100`.

Per-MFMA-shape counters (e.g. `_16X16X16`, `_32X32X8`) are not consistently exposed across
ROCm versions; if you need the shape distribution, inspect the ISA emitted by the kernel
(`llvm-objdump -d`) rather than relying on a per-shape PMC.

### Scratch / register spill
```
Scratch_Per_Workitem                  # from pmc_perf.csv columns — bytes per work-item allocated to scratch
SQ_INSTS_VMEM_WR (with scratch-targeted addresses)   # writes to scratch buffer go through vmem
```

Counter-side scratch read/write counters exist on most ROCm versions but the exact names
vary — `rocprofv3 -L | grep -i scratch` shows what your install exposes.

On AMD, **scratch lives in HBM** (backed by an HBM-resident buffer, with read-through L2). A non-zero `Scratch_Per_Workitem` means the compiler had to spill VGPR/AGPR; this turns register reads into HBM round trips. Treat it like NVIDIA's local memory traffic (and worse, since AMD has no register-cache equivalent).

### GPU-wide cycle counters

Verified GRBM PMCs on gfx942 / gfx950 (`rocprofv3 -L | grep '^GRBM_'`) — 10 total:
```
GRBM_GUI_ACTIVE                       # cycles the GPU was active — use as denominator for "% busy"
GRBM_COUNT                            # total cycles elapsed
GRBM_CP_BUSY                          # command processor busy
GRBM_CPC_BUSY / GRBM_CPF_BUSY         # CP-compute / CP-frontend sub-blocks
GRBM_EA_BUSY                          # any memory-controller (EA) busy
GRBM_SPI_BUSY                         # shader-pipe-interface busy (wave dispatch)
GRBM_TA_BUSY / GRBM_TC_BUSY           # texture-address / texture-cache busy
GRBM_UTCL2_BUSY                       # unified TLB / L2 busy
```

`GRBM_GDS_BUSY` and `GRBM_SDMA_BUSY` are **NOT exposed** on gfx942 / gfx950 in current ROCm
(they appeared on older gfx). For SDMA copy activity, use the HIP / HSA trace timeline
instead.

### Per-CU activity (workgroup imbalance)

There is **no** raw `SQ_WAVES_PER_CU` PMC. rocprof-compute's workgroup-balance breakdown
(`-b 7`) gives you the per-CU / per-XCD view, synthesized from `SQ_WAVES` plus the
topology in `sysinfo.csv` (`cu_per_gpu`, XCD layout).

For a manual per-CU view, the verified raw PMCs are:
```
SQ_WAVES                              # wave-launch counter (aggregated; rocprof-compute splits by CU)
SQ_WAVES_EQ_64                        # waves that issued a full-occupancy block
SQ_WAVES_LT_{16,32,48,64}             # under-utilized wave buckets
SQ_WAVES_RESTORED / SQ_WAVES_SAVED    # context-switch wave events (rare under normal profiling)
```

---

## Rocprof-compute block IDs

The current rocprof-compute uses **top-level integer block IDs** with `-b <N>`. Use:

```bash
rocprof-compute analyze --list-stats                       # lists profiled KERNELS / dispatches
rocprof-compute analyze --list-metrics gfx942              # lists metric / block IDs for MI300X
rocprof-compute analyze --list-metrics gfx950              # lists metric / block IDs for MI355X
```

Verified top-level block IDs (rocprof-compute ROCm 7.x; ALWAYS confirm on your install with
`--list-metrics <gfx_arch>`):

| ID | Block | What it tells you |
|---:|---|---|
|  2 | Speed-of-Light (SoL)          | % of peak for compute, HBM, vL1, L2, scratch |
|  5 | CS / Wavefront                | Waves/wkg, achieved occupancy, wavefront launch |
|  7 | Wavefront Launch Stats        | Grid/workgroup, registers, LDS, occupancy limiter |
| 10 | Compute Pipe                  | VALU / SALU / matrix-core busy %, IPC, FMA |
| 11 | Instruction Mix               | VALU / SALU / MFMA / VMEM / LDS / FLAT instruction counts |
| 12 | Pipe SoL                      | Per-pipe Speed-of-Light |
| 13 | Pipe Stats                    | Per-pipe stall / activity stats |
| 14 | Cache (overview)              | Cache summary across vL1 / L2 |
| 15 | L1D Cache (TCP)               | vL1 hit rate, sectors per request, bytes per wave |
| 16 | L2 Cache (TCC)                | L2 hit rate, atomics, bytes |
| 17 | L2 - Fabric / HBM (TCC_EA)    | HBM read/write bytes, achieved BW |
| 18 | Scratch / Spill               | Scratch reads/writes (= register spill on AMD) |
| 21 | Misc                          | Other derived metrics |

The dotted-section IDs (`2.1.0`, `2.1.13`, `2.1.23` etc.) from older Omniperf docs are
not how current rocprof-compute is invoked — use the integer `-b` flag.

---

## NVIDIA NCU → AMD rocprofv3 stall-reason cheat sheet

For users coming from NCU, the wait/stall taxonomy maps roughly as:

> **AMD-side note:** the `Stall_Reason` values below come from the **stochastic PC sampling's
> `Stall_Reason` column** on gfx942 / gfx950 — these are NOT PMC counters (only
> `SQ_WAIT_ANY`, `SQ_WAIT_INST_ANY`, `SQ_WAIT_INST_LDS` exist as PMCs). The `host_trap`
> PC-sampling mode does NOT populate `Stall_Reason`; use stochastic for the breakdown.
> Memory-type subcategories (global vs LDS vs scalar) are distinguished by reading the
> ISA mnemonic in `Instruction_Comment` at the sampled PC, not by separate `Stall_Reason`
> values. Authoritative enum:
> `ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_*` in
> `/opt/rocm/include/rocprofiler-sdk/pc_sampling.h`. See
> https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-pc-sampling.html
>
> Per-execution-pipe `arb_state_stall_*` / `arb_state_issue_*` bit-fields are JSON-only
> (collect with `-f json` and read the `snapshot` object on each PC-sample record), not
> CSV columns.

| NVIDIA (NCU) `smsp__average_warps_issue_stalled_<X>` | AMD (stochastic CSV `Stall_Reason` value + ISA mnemonic filter, or PMC where noted) | Notes |
|---|---|---|
| `long_scoreboard` | `WAITCNT` on `global_load_*` / `flat_load_*` ISA mnemonics — scoreboard drains via PMC `SQ_WAIT_ANY` / `SQ_WAIT_INST_ANY` | Global / texture memory load waits |
| `short_scoreboard` | `WAITCNT` on `ds_read_*` / `ds_write_*` (LDS) or `s_load_*` (scalar/constant) mnemonics; PMC `SQ_WAIT_INST_LDS` for LDS | Shared-mem / constant / scalar memory waits |
| `barrier` | `BARRIER_WAIT` (with `s_barrier` source-line hotspot) — corroborate via PMC `SQ_WAIT_ANY` | `__syncthreads` / `s_barrier` |
| `math_pipe_throttle` | `ALU_DEPENDENCY` (or `ARBITER_WIN_EX_STALL`) on VALU lines | VALU pipe contention |
| `mio_throttle` | `WAITCNT` on `ds_*` lines (LDS pressure); LSU back-pressure visible in `TCP_PENDING_STALL_CYCLES_sum` | MIO unit is NV-specific |
| `lg_throttle` | TCP saturation (back-pressure visible in `TCP_PENDING_STALL_CYCLES_sum`) | "load/global throttle" |
| `tex_throttle` | TA/TD pressure — `TA_BUSY_*` saturated | Texture pipe in NV terms; image pipe on AMD |
| `wait` | `INTERNAL_INSTRUCTION` or `OTHER_WAIT` | Misc fixed-latency waits |
| `membar` | `WAITCNT` after an explicit fence — corroborate via PMC `SQ_WAIT_ANY` / `SQ_WAIT_INST_ANY` | AMD uses explicit vmcnt/lgkmcnt drains |
| `dispatch_stall` | `ARBITER_NOT_WIN` | Issue-slot blocked |
| `drain` | `WAITCNT` at kernel end; PMC `SQ_WAIT_ANY` / `SQ_WAIT_INST_ANY` / `SQ_WAIT_INST_LDS` | Outstanding ops drain |
| `no_instruction` | `NO_INSTRUCTION_AVAILABLE` — corroborate with I-cache miss PMCs (`SQC_*`) | Scalar/instr cache miss |
| `branch_resolving` | `WAITCNT` on `s_cbranch_*` / branch-message ISA + `SQ_INSTS_BRANCH` | Conditional branch / message resolution |
| `selected` (productive) | rows with `Wave_Issued_Instruction == 1`; `SQ_INSTS_VALU` / `SQ_INSTS_VMEM_*` | Actually issuing |
| (matrix-pipe dep on MFMA path) | `ALU_DEPENDENCY` on `v_mfma_*` mnemonics | MFMA pipe busy / AGPR dep |
| (flat-addressing waits) | `WAITCNT` on `flat_*` mnemonics | FLAT-addressing op in flight |
| (export-count drain) | `WAITCNT` on `exp` / export drains — uncommon on compute | Export-count drain |

This is approximate — AMD and NVIDIA model the front-end stall categories differently. Use it as a sanity check, not a 1:1 translation. Verify exact `Stall_Reason` values against the stochastic CSV produced by your install.

---

## Discovering counters for new GPUs (gfx906, gfx908, gfx90a, future gfx12X)

```bash
# Per-build authoritative list
rocprofv3 -L > /tmp/counters_$(rocminfo | awk '/gfx/{print $2; exit}').txt

# Filter by IP block
rocprofv3 -L | grep -i '^TCC'
rocprofv3 -L | grep -i WAIT
```

From Python (after running a profile pass):
```python
import pandas as pd
cols = pd.read_csv('rpc_<tag>/pmc_perf.csv', nrows=1).columns
print('\n'.join(sorted(c for c in cols if c.startswith('SQ_'))))
```

---

## Gotchas

1. **Counter exists in `rocprofv3 -L` but column missing in CSV**: the counter wasn't *collected* in this PMC group. Rerun with a different `--section` or `--pmc` list.
2. **Counter value is `0`**: either the hardware feature reports zero (e.g., no MFMA activity), or the counter is conditional on a feature flag (e.g., FP4 counters require gfx950).
3. **`_sum` vs `_avr` vs `_max`**: counter suffix indicates aggregation. `_sum` = total across all CUs / channels / SEs; `_avr` = per-instance average; `_max` = max instance value. Don't re-sum a `_sum`.
4. **`TCC_EA_*` (no number) on MI300X**: that's the gfx906/908 spelling — on gfx942/gfx950 use `TCC_EA0_*`. There is **no** `TCC_EA1_*` on these gens, so don't sum two channels.
5. **MFMA per-dtype counters**: `F6F4` is gfx950-only (requires ROCm 7+ on MI355X). `XF32` exists on BOTH gfx942 and gfx950 (AMD's XF32 ≡ NVIDIA TF32). There is **no** `_BF8` counter — `E5M2` (BF8) inputs are bucketed under `SQ_INSTS_VALU_MFMA_MOPS_F8` on both gens (including mixed `f8_bf8` shapes). Always check `rocprofv3 -L | grep -i MFMA` for the exact suffixes your build exposes (no `_F4` / `_F6` / `_BF8` / `_MXFP4` / `_MXFP6` / `_MXFP8` counters exist — block-scaled formats roll up into `_F6F4`).
6. **`Accum_VGPR` column reports 0** on a kernel you expect to use MFMA: either the compiler chose to spill MFMA accumulators into VGPRs (lower performance), or MFMA isn't being emitted — check ISA with `llvm-objdump -d`. The column is named `Accum_VGPR`, not `AGPRs`.
7. **MI300A (APU variant of gfx942)** shares the gfx942 counter set with MI300X discrete, but `TCC_EA*_IO_*` traffic includes CPU↔GPU coherent memory.
8. **Per-XCD attribution**: rocprof-compute's workgroup-balance breakdown aggregates per CU, but MI300X has 8 XCDs each with 38 CUs (MI355X: 8 × 32). To see per-XCD load you have to group CUs by index range (CU 0-37 = XCD0, 38-75 = XCD1, …) or rely on the per-CU active-cycle output and grouping by `cu // (cus_per_xcd)`.
9. **CPX/NPS4 partitioning** changes the visible CU count and channel layout — `sysinfo.csv` (wide single-row format on current rocprof-compute) records the partition state at profile time. Profile with the same partition setup as production.
