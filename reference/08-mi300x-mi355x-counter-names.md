# MI300X (gfx942) & MI355X (gfx950) PMC Counter Reference

AMD's PMC namespace is partitioned by IP block: `SQ_*` (Shader Engine/Wave scheduler), `SQC_*` (scalar/instr cache), `TA_*` / `TD_*` (texture address/data, used by vector mem), `TCP_*` (vL1 — per-CU vector L1), `TCC_*` (L2 cache, per-channel), `TCC_EA_*` (memory channels — note: per-channel on gfx942/950 → `TCC_EA0_*` / `TCC_EA1_*`), `GRBM_*` (graphics register bus master, the GPU-wide cycle/active counters), `GL2C_*` (legacy alias for L2 on some builds), and `CPC_*` / `CPF_*` (command processor).

When in doubt, enumerate:

```bash
rocprofv3 -L > /tmp/all_counters.txt                    # the canonical authority for your ROCm install
# Equivalent long form (rocprofv3 ≥ ROCm 6.2):
rocprofv3 --list-avail > /tmp/all_counters.txt
# On newer ROCm builds, the companion CLI is `rocprofv3-avail list --pmc`.
# Or, programmatically from a collected report:
python3 -c "
import pandas as pd, glob
seen = set()
for p in glob.glob('rpc_<tag>/SoC/*.csv') + ['rpc_<tag>/pmc_perf.csv']:
    seen.update(pd.read_csv(p, nrows=1).columns)
print('\n'.join(sorted(seen)))
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
| `TCC_HIT_sum` (single L2 channel) | **`TCC_HIT[0..n]_sum`** — per-channel (multiple TCC instances) |
| `TCC_EA_RDREQ_sum` | **`TCC_EA0_RDREQ_sum` + `TCC_EA1_RDREQ_sum`** (two memory channels per XCD on MI300X) |
| `TCC_EA_RDREQ_32B_sum` | **`TCC_EA0_RDREQ_32B_sum` + `TCC_EA1_RDREQ_32B_sum`** |
| `TCC_EA_WRREQ_sum` | **`TCC_EA0_WRREQ_sum` + `TCC_EA1_WRREQ_sum`** |
| `TA_BUSY_avr` | **`TA_BUSY_avr_per_simd`** (or sum across SIMDs) |
| `SQ_INSTS_MFMA` exists on gfx908+ | Same aggregate name; per-dtype detail moves to `SQ_INSTS_VALU_MFMA_MOPS_<DTYPE>` on gfx942+ (per-shape PMCs not stable across ROCm versions) |
| `SQ_BUSY_CYCLES` (gfx9 family) | Same, but on MI300X read **`GRBM_GUI_ACTIVE`** for true GPU-wide active cycles (denominator) |
| `SQ_INSTS_VALU_MFMA_MOPS_*` (no FP4/FP6/MXFP) | **gfx950 adds** FP4/FP6/MXFP-scaled MFMA dtype suffixes (exact names install-specific) |
| `SQ_WAVES` (total wavefronts) | Same name everywhere |

---

## Canonical MI300X / MI355X counter set (curated)

These PMC names have been confirmed to exist and return meaningful values on gfx942 and gfx950 with ROCm 6.4 / 7.x. Always verify for your specific build with `rocprofv3 --list-metrics`.

### Launch / wave geometry (from `pmc_perf.csv` per-dispatch columns, not PMCs)
```
Dispatch_ID
Kernel_Name
GPU_ID, Queue_ID, PID, TID
Grid_Size, Workgroup_Size
LDS_Per_Workgroup            # bytes
Scratch_Per_Workitem         # bytes = register spill volume per work-item
VGPRs                        # per-wave allocation
SGPRs
AGPRs                        # CDNA3+ accumulation registers (MFMA destination pool)
Wave_Size                    # 64 on CDNA
Start_Timestamp, End_Timestamp   # ns
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
SQ_INSTS_FLAT_LDS_ONLY          # flat that ended up as LDS
SQ_INSTS_LDS                    # LDS instructions
SQ_INSTS_GDS                    # GDS (rarely used)
SQ_INSTS_BRANCH                 # branches
SQ_INSTS_MFMA                   # matrix-core total
SQ_INSTS_MFMA_F16
SQ_INSTS_MFMA_BF16
SQ_INSTS_MFMA_F32_16X16X16BF16  # per-shape breakdown (gfx942+)
SQ_INSTS_MFMA_F32_16X16X32_F8F6F4    # CDNA4 only (gfx950)
SQ_INSTS_MFMA_*_F4              # CDNA4 only
SQ_INSTS_MFMA_*_F6              # CDNA4 only
SQ_INSTS_MFMA_*_MXF*            # CDNA4 only (MXFP scaling)
SQ_INSTS_VALU_TRANS_F16         # transcendentals
SQ_INSTS_VALU_TRANS_F32
SQ_BUSY_CYCLES                  # cycles SQ was issuing on any SE
SQ_ACTIVE_INST_VALU             # VALU active cycles (if exposed by this ROCm)
SQ_ACTIVE_INST_VMEM             # VMEM active cycles (if exposed by this ROCm)
```

### Wait / stall reasons (SQ) — analog of NVIDIA `smsp__average_warps_issue_stalled_*`

The set of `SQ_WAIT_*` counters has churned across ROCm releases. The names below have
been observed on gfx942 / gfx950 with recent ROCm; treat anything not in this list as
unverified and check `rocprofv3 -L | grep SQ_WAIT` on your install.

```
SQ_WAIT_INST_VMEM               # waiting on vmem completion (load/store to HBM/L2/vL1)
SQ_WAIT_INST_LDS                # waiting on LDS read/write completion (covers bank-conflict serialization)
SQ_WAIT_INST_SMEM               # waiting on scalar memory
SQ_WAIT_INST_FLAT               # waiting on flat addressing
SQ_WAIT_INST_EXP                # export waits (mostly graphics, rare in compute)
SQ_WAIT_INST_MISC               # misc waits
SQ_WAIT_BARRIER                 # waiting at s_barrier (workgroup-wide sync)
SQ_WAIT_VMCNT                   # waiting for vmcnt (outstanding vmem) to drain
SQ_WAIT_LGKMCNT                 # waiting for lgkmcnt (LDS/GDS/scalar/const) to drain
SQ_WAIT_EXPCNT                  # waiting for expcnt (export, mostly graphics)
SQ_INST_LEVEL_VMEM              # outstanding vmem level (peak concurrency)
SQ_INST_LEVEL_LDS               # outstanding LDS level
```

Wait categories that *do not* have a stable dedicated `SQ_WAIT_*` counter on gfx942/gfx950
include scalar-ALU pipe occupancy, VALU pipe occupancy, scratch (= spill) traffic, and an
LDS-specific "ANY" aggregate. Use rocprof-compute's section 2.1.13 derived totals instead
of inventing counter names for those.

### IPC / occupancy
```
SQ_BUSY_CYCLES / GRBM_GUI_ACTIVE     # GPU-wide SQ busy ratio
SQ_WAVES / GRBM_GUI_ACTIVE           # wave throughput
SQ_INSTS / SQ_BUSY_CYCLES             # IPC across all SIMDs (per-SE — divide by # SEs for per-SIMD)
SQ_ACCUM_PREV_HIRES                  # achieved occupancy (waves/SIMD/cycle) — populated by rocprof-compute §2.1.2
```

### vL1 cache (TCP — per-CU vector L1)
```
TCP_TOTAL_CACHE_ACCESSES_sum
TCP_TCC_READ_REQ_sum                  # vL1 miss requests sent to L2 (reads)
TCP_TCC_WRITE_REQ_sum                 # writes
TCP_TCC_ATOMIC_WITH_RET_REQ_sum       # atomics with return
TCP_TCC_ATOMIC_WITHOUT_RET_REQ_sum    # atomics fire-and-forget
TCP_PERF_SEL_TCP_LATENCY_*            # latency sampling buckets
TCP_GATE_EN1                          # gate-enable cycles
TCP_GATE_EN2
TCP_TA_TCP_STATE_READ_sum             # vL1 reads from TA
TCP_PENDING_STALL_CYCLES_sum
TCP_VOLATILE_sum                      # volatile (uncached) accesses
```

vL1 hit rate (computed): `1 - (TCP_TCC_READ_REQ_sum + TCP_TCC_WRITE_REQ_sum) / TCP_TOTAL_CACHE_ACCESSES_sum`

### L2 cache (TCC — per-channel on MI300X/MI355X)
```
TCC_HIT_sum                           # aggregate
TCC_MISS_sum
TCC_HIT[0..n]_sum                     # per-channel hits (n = # TCC channels on this XCD)
TCC_REQ_sum
TCC_REQ_READ_sum
TCC_REQ_WRITE_sum
TCC_ATOMIC_sum                        # atomic ops at L2
TCC_BUBBLE_sum
TCC_NORMAL_WRITEBACK_sum
TCC_ALL_TC_OP_WB_sum
TCC_NC_REQ_sum                        # non-coherent
TCC_UC_REQ_sum                        # uncached
TCC_CC_REQ_sum                        # coherent
TCC_RW_REQ_sum                        # read/write mix
```

L2 hit rate: `TCC_HIT_sum / (TCC_HIT_sum + TCC_MISS_sum)`

### HBM / memory channel (TCC_EA — per channel, MI300X has 2: EA0, EA1; MI355X has more channels per IOD)
```
TCC_EA0_RDREQ_sum                     # read requests issued to HBM channel 0
TCC_EA0_RDREQ_32B_sum                 # 32B-granular read requests
TCC_EA0_WRREQ_sum                     # writes
TCC_EA0_WRREQ_64B_sum                 # 64B-granular writes
TCC_EA0_RDREQ_DRAM_sum                # filtered to DRAM (excludes other agents)
TCC_EA0_WRREQ_DRAM_sum
TCC_EA1_RDREQ_sum                     # second HBM channel
TCC_EA1_RDREQ_32B_sum
TCC_EA1_WRREQ_sum
TCC_EA1_WRREQ_64B_sum
TCC_EA1_RDREQ_DRAM_sum
TCC_EA1_WRREQ_DRAM_sum
TCC_EA0_ATOMIC_sum
TCC_EA1_ATOMIC_sum
TCC_EA0_RDREQ_IO_sum                  # I/O (xGMI / PCIe) reads
TCC_EA1_RDREQ_IO_sum
TCC_EA0_WRREQ_IO_sum
TCC_EA1_WRREQ_IO_sum
```

Computed achieved HBM read BW (GB/s):
```
(TCC_EA0_RDREQ_32B_sum + TCC_EA1_RDREQ_32B_sum) * 32 / kernel_duration_seconds / 1e9
```

Peak HBM BW: 5.3 TB/s on MI300X, 8.0 TB/s on MI355X (per-package; per-channel = peak / # channels).

### LDS (per-CU local data share)
```
SQ_LDS_BANK_CONFLICT                  # bank conflicts (counts cycles lost; ideally 0)
SQ_LDS_IDX_ACTIVE                     # indexed LDS accesses
SQ_LDS_ATOMIC_RETURN                  # LDS atomics with return
SQ_LDS_UNALIGNED_STALL                # unaligned LDS access stalls
SQ_INSTS_LDS                          # LDS instruction count
SQ_INSTS_FLAT_LDS_ONLY                # flat that became LDS
```

### Matrix-core / MFMA

The MFMA per-dtype counters on gfx942 / gfx950 use the prefix
**`SQ_INSTS_VALU_MFMA_MOPS_<DTYPE>`** ("MOPS" = matrix-ops). The exact set of `<DTYPE>`
suffixes exposed depends on the ROCm version — enumerate with
`rocprofv3 -L | grep -i MFMA`. The aggregate `SQ_INSTS_MFMA` is always available.

```
SQ_INSTS_MFMA                         # total MFMA issued (aggregate)
SQ_INSTS_VALU_MFMA_MOPS_F16           # by source dtype (when exposed)
SQ_INSTS_VALU_MFMA_MOPS_BF16
SQ_INSTS_VALU_MFMA_MOPS_F32
SQ_INSTS_VALU_MFMA_MOPS_F64           # CDNA3 only at full throughput; CDNA4 halved
SQ_INSTS_VALU_MFMA_MOPS_I8
SQ_INSTS_VALU_MFMA_MOPS_F8            # CDNA3 (OCP-FNUZ) / CDNA4 (OCP standard)
# CDNA4 (gfx950) — block-scaled MX formats, exact suffixes are install-specific
# `rocprofv3 -L | grep -i MXF` is the authoritative list on your build.
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
```
GRBM_GUI_ACTIVE                       # cycles the GPU was active — use as denominator for "% busy"
GRBM_COUNT                            # total cycles elapsed
GRBM_CP_BUSY                          # command processor busy
GRBM_GDS_BUSY                         # GDS busy
GRBM_SDMA_BUSY                        # SDMA engine busy (mem copies)
```

### Per-CU activity (workgroup imbalance)
```
SQ_WAVES_PER_CU                       # waves per CU (rocprof-compute synthesizes this from SQ_WAVES + topology)
SQ_BUSY_CYCLES_per_cu                 # — same
```

rocprof-compute §2.1.23 ("Workgroup imbalance") gives you the per-CU breakdown without needing to do this manually.

---

## Rocprof-compute section IDs

```bash
rocprof-compute analyze --list-stats
```

Section ID map (MI300X / rocprof-compute ROCm 7.x; verify on your install):

| ID | Section | What it tells you |
|---|---|---|
| 2.1.0  | Launch Statistics             | Grid/workgroup, waves/CU, registers/wave, scratch, LDS |
| 2.1.1  | Speed-of-Light (SoL)          | % of peak for compute, HBM, vL1, L2, scratch |
| 2.1.2  | Wavefront Launch Stats        | Waves/wkg, achieved occupancy, register/LDS pressure |
| 2.1.5  | Pipeline / instruction mix    | VALU / SALU / MFMA / VMEM / LDS instruction counts |
| 2.1.10 | Compute Units — Compute Pipe  | VALU / SALU / matrix-core busy %, IPC, FMA |
| 2.1.11 | Compute Units — Memory Pipe   | Bytes per wavefront (peak 256B coalesced), LDS bank conflicts |
| 2.1.13 | Wavefront Stall Reasons       | WAIT_INST_VMEM, WAIT_INST_LDS, WAIT_INST_SMEM, WAIT_INST_FLAT, WAIT_BARRIER, WAIT_VMCNT, WAIT_LGKMCNT |
| 2.1.15 | Memory — vL1 cache (TCP)      | Hit rate, sectors per request, coalescing |
| 2.1.16 | Memory — L2 cache (TCC)       | Per-channel hit rate, atomics, bytes |
| 2.1.17 | Memory — HBM (TCC_EA)         | Per-channel HBM read/write bytes, achieved BW |
| 2.1.20 | Roofline                       | Compute vs memory roofline, kernel position |
| 2.1.22 | Scratch / Spill                | Scratch reads/writes (= register spill on AMD) |
| 2.1.23 | Workgroup imbalance           | Per-CU active-cycle distribution |

---

## NVIDIA NCU → AMD rocprofv3 stall-reason cheat sheet

For users coming from NCU, the wait/stall taxonomy maps roughly as:

| NVIDIA (NCU) `smsp__average_warps_issue_stalled_<X>` | AMD (SQ) | Notes |
|---|---|---|
| `long_scoreboard` | `SQ_WAIT_INST_VMEM` / `SQ_WAIT_VMCNT` | Global / texture memory load waits |
| `short_scoreboard` | `SQ_WAIT_INST_LDS` / `SQ_WAIT_LGKMCNT` | Shared-mem / constant / scalar memory waits |
| `barrier` | `SQ_WAIT_BARRIER` | `__syncthreads` / `s_barrier` |
| `math_pipe_throttle` | (no direct counter — derive from `SQ_INSTS_VALU / SQ_BUSY_CYCLES` ratio) | VALU pipe contention |
| `mio_throttle` | LSU pressure (TCP saturated; back-pressure visible in `TCP_PENDING_STALL_CYCLES_sum`) | MIO unit is NV-specific |
| `lg_throttle` | TCP saturation (back-pressure visible in `TCP_PENDING_STALL_CYCLES_sum`) | "load/global throttle" |
| `tex_throttle` | TA/TD pressure — `TA_BUSY_*` saturated | Texture pipe in NV terms; image pipe on AMD |
| `wait` | `SQ_WAIT_INST_MISC` | Misc fixed-latency waits |
| `membar` | `SQ_WAIT_VMCNT` after a fence | AMD uses explicit vmcnt/lgkmcnt drains |
| `dispatch_stall` | `SQ_WAIT_INST_MISC` (proxy) | Issue-slot blocked |
| `drain` | `SQ_WAIT_VMCNT` / `SQ_WAIT_LGKMCNT` at kernel end | Outstanding ops drain |
| `no_instruction` | I-cache miss — `SQC_*` counters | Scalar/instr cache miss |
| `branch_resolving` | `SQ_INSTS_BRANCH` + scalar branch wait | Conditional branch resolution |
| `selected` (productive) | `SQ_ACTIVE_INST_VALU` / `SQ_ACTIVE_INST_VMEM` | Actually issuing |

This is approximate — AMD and NVIDIA model the front-end stall categories differently. Use it as a sanity check, not a 1:1 translation.

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
import pandas as pd, glob
seen = set()
for p in glob.glob('rpc_<tag>/SoC/*.csv') + ['rpc_<tag>/pmc_perf.csv']:
    seen.update(pd.read_csv(p, nrows=1).columns)
print('\n'.join(sorted(c for c in seen if c.startswith('SQ_'))))
```

---

## Gotchas

1. **Counter exists in `rocprofv3 -L` but column missing in CSV**: the counter wasn't *collected* in this PMC group. Rerun with a different `--section` or `--pmc` list.
2. **Counter value is `0`**: either the hardware feature reports zero (e.g., no MFMA activity), or the counter is conditional on a feature flag (e.g., FP4 counters require gfx950).
3. **`_sum` vs `_avr` vs `_max`**: counter suffix indicates aggregation. `_sum` = total across all CUs / channels / SEs; `_avr` = per-instance average; `_max` = max instance value. Don't re-sum a `_sum`.
4. **`TCC_EA_*` (no number) on MI300X**: that's the gfx906/908 spelling — on gfx942 you want `TCC_EA0_*` and `TCC_EA1_*` (per channel) and you'll have to sum them yourself.
5. **MFMA FP4 / FP6 / MXFP per-dtype counters missing**: gfx950 only. ROCm 6.x will not enumerate them; ROCm 7+ required for MI355X. Always check `rocprofv3 -L | grep -i MFMA` for the exact suffixes your build exposes.
6. **`AGPRs` column reports 0** on a kernel you expect to use MFMA: either the compiler chose to spill MFMA accumulators into VGPRs (lower performance), or MFMA isn't being emitted — check ISA with `llvm-objdump -d`.
7. **MI300A (APU variant of gfx942)** shares the gfx942 counter set with MI300X discrete, but `TCC_EA*_IO_*` traffic includes CPU↔GPU coherent memory.
8. **Per-XCD attribution**: rocprof-compute's "Workgroup imbalance" section aggregates per CU, but MI300X has 8 XCDs each with 38 CUs. To see per-XCD load you have to group CUs by index range (CU 0-37 = XCD0, 38-75 = XCD1, …) or rely on §2.1.23's automatic grouping.
9. **CPX/NPS4 partitioning** changes the visible CU count and channel layout — `sysinfo.csv` records the partition state at profile time. Profile with the same partition setup as production.
