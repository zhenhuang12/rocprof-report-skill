# Final Report Template

The report is the deliverable. Everything else (rocprof-compute output dirs, ATT JSONs, CSVs) is evidence. Structure matters: a busy reader should see the top findings in 30 seconds and be able to drill into details if they want.

Save as `$PROFILE_RUN_DIR/REPORT.md`.

---

## Template

```markdown
# `<kernel_name>` Profiling Report

**Kernel:** `<exact demangled kernel name or template instantiation>`
**Target GPU:** AMD Instinct MI300X (gfx942, CDNA3, 304 CU)   (or MI355X / gfx950 / CDNA4 / 256 CU)
**ROCm version:** 7.x (rocprofv3 X.Y, rocprof-compute X.Y)
**Compile flags:** `hipcc -O3 -std=c++17 -gline-tables-only --offload-arch=gfx942 -munsafe-fp-atomics`
**Profile date:** YYYY-MM-DD
**Run directory:** `profile/<run_name>/`

---

## 0. Profiling setup

> How exactly did we get these numbers? Required for reproducibility.

- Harness: `profile/<run_name>/harness/*.hip` — what it is (standalone driver / the original binary / something else) and why.
- Workloads: which real tensors / shapes were used. Cite the workload UUID or shape tuple.
- Dispatch paths covered: list each `(template params, grid, workgroup)` combination profiled.
- Counter-name caveats: any PMC names that differ from rocprof-compute docs (e.g., `TCC_EA0_*` is the only EA channel on gfx942/gfx950; the `TCC_EA_*` / `TCC_EA1_*` forms are gfx906/gfx908 history). See [`08-mi300x-mi355x-counter-names.md`](08-mi300x-mi355x-counter-names.md).
- GPU partitioning: `SPX/NPS1` (default) or `CPX/NPS4` etc. Check with `rocm-smi --showcomputepartition --showmemorypartition`.

Minimal runnable command listing:

    # Compile (MI300X)
    hipcc -O3 -std=c++17 -gline-tables-only \
          --offload-arch=gfx942 -munsafe-fp-atomics \
          harness.hip -o harness

    # 1. Kernel-trace overview (cheap, no PMC overhead)
    rocprofv3 --kernel-trace --hip-trace --hsa-trace \
        --kernel-include-regex "<kernel_regex>" \
        -d profile/<run_name>/reports/trace_<tag> \
        -- ./harness [args]

    # 2. Section-based perf metrics (analog of `ncu --set full`)
    # rocprof-compute's kernel filter is `-k` / `--kernel` (substring, not regex).
    # Roofline is ON by default; pass `--no-roof` to skip. There is NO `--roofline` flag.
    rocprof-compute profile -n <run_name>_<tag> \
        -k "<kernel_substring>" \
        -p profile/<run_name>/reports/rpc_<tag> \
        -- ./harness [args]
    rocprof-compute analyze -p profile/<run_name>/reports/rpc_<tag> \
        > profile/<run_name>/analysis/details_<tag>.txt

    # 3. Per-line stall sampling (analog of `ncu --set source`)
    # Note the underscore in `host_trap` (not `host-trap`).
    # `host_trap` only supports `--pc-sampling-unit time` (`cycles`/`instructions`
    # are stochastic-only and the runtime rejects the wrong combo).
    # For host_trap + time, --pc-sampling-interval is in MICROSECONDS (1000 = 1 ms).
    rocprofv3 --pc-sampling-beta-enabled \
        --pc-sampling-method host_trap \
        --pc-sampling-interval 1000 --pc-sampling-unit time \
        --kernel-include-regex "<kernel_regex>" \
        -f csv \
        -d profile/<run_name>/reports/pcsamp_<tag> \
        -- ./harness [args]

### Artifacts

    profile/<run_name>/
    ├── REPORT.md                           ← this file
    ├── harness/...                         ← standalone harness
    ├── reports/
    │   ├── trace_<tag>/                    ← rocprofv3 kernel-trace (+ .db on ROCm 7+)
    │   ├── rpc_<tag>/                      ← rocprof-compute profile dir
    │   │   ├── pmc_perf.csv                ← merged PMCs, one row per (dispatch × PMC-group)
    │   │   ├── pmc_kernel_top.csv          ← top-K kernels by dispatch count
    │   │   ├── sysinfo.csv                 ← wide single-row sysinfo (NOT param/value)
    │   │   ├── roofline.pdf                ← PDF when roofline ran (default-on; --no-roof to skip)
    │   │   ├── profiling_config.yaml
    │   │   └── out/pmc_<N>/<host>/<pid>_*.csv   ← raw per-PMC-group passes
    │   ├── pcsamp_<tag>/                   ← PC sampling CSV (e.g. pc_sampling_host_trap_v0.csv)
    │   └── att_<tag>/                      ← optional ATT (one JSON per SE/CU)
    └── analysis/                           ← scripts + extracted metrics
        ├── details_<tag>.txt               ← `rocprof-compute analyze` dump
        ├── metrics_key_<tag>.json
        └── compare_<tag1>_vs_<tag2>.txt

---

## 1. Headline numbers

> A single table that tells the whole story at a glance.

| Metric | `<tag1>` | `<tag2>` | Source |
|---|---:|---:|---|
| **Duration** | X µs | Y µs | `kernel_trace.csv: End_Timestamp - Start_Timestamp` |
| SoL — Compute (% peak) | X% | Y% | rocprof-compute SoL block (`-b 2`) |
| SoL — HBM (% peak) | X% | Y% | rocprof-compute SoL block (`-b 2`) |
| SoL — vL1 (TCP) | X% | Y% | rocprof-compute L1D block (`-b 15`) |
| SoL — L2 (TCC) | X% | Y% | rocprof-compute L2 block (`-b 16`) |
| HBM read BW (achieved / peak) | X / 5300 GB/s | … | `TCC_EA0_RDREQ_32B_sum × 32 / duration` (TCC_EA1_* does NOT exist on gfx942/gfx950) |
| vL1 hit rate | X% | Y% | rocprof-compute L1D block (`-b 15`) |
| L2 hit rate | X% | Y% | `TCC_HIT_sum / (TCC_HIT_sum + TCC_MISS_sum)` |
| MFMA busy (% peak) | X% | Y% | rocprof-compute compute-pipe block (`-b 10`) |
| Arch_VGPR / work-item | X | Y | `pmc_perf.csv: Arch_VGPR` |
| Accum_VGPR / work-item (= AGPR on CDNA3+) | X | Y | `pmc_perf.csv: Accum_VGPR` |
| SGPR / wavefront | X | Y | `pmc_perf.csv: SGPR` (singular) |
| LDS / workgroup | X B | Y B | `pmc_perf.csv: LDS_Per_Workgroup` |
| Achieved occupancy (waves/SIMD) | X / 8 | Y / 8 | rocprof-compute wavefront block (`-b 5`) |
| Scratch (= register spill, bytes/wi) | X | Y | `pmc_perf.csv: Scratch_Per_Workitem` |
| Stall: WAIT_INST_VMEM (% PC samples) | X% | Y% | PC-sampling CSV `Wait_Reason` aggregation (NOT a PMC on gfx942/gfx950) |

**One-line read:** <"The kernel runs at X% of compute SoL — it's VMEM-wait-bound on Y, not HBM-BW-bound."> — this is the punchline.

---

## 2. Per-dimension analysis

> Walk through the six analysis dimensions (see [`05-analysis-dimensions.md`](05-analysis-dimensions.md)), cite counters, state findings.

### 2.1 CU occupancy & launch geometry
<grid size, workgroup size, waves/CU, theoretical vs achieved occupancy, VGPR/AGPR/LDS limits, wave64 math; XCD layout (MI300X: 8 XCDs × 38 CUs over 4 IODs; MI355X: 8 XCDs × 32 CUs over 2 IODs) and whether grid fills them>

### 2.2 Workgroup balance (tail effect)
<per-XCD active cycles, rocprof-compute workgroup-balance breakdown, timeseries shape, input distribution imbalance ratios>

### 2.3 Instruction-level stall analysis
<stall breakdown % from PC-sampling `Wait_Reason` aggregation (the ONLY granular source on gfx942/gfx950 — only `SQ_WAIT_ANY`, `SQ_WAIT_INST_ANY`, `SQ_WAIT_INST_LDS` exist as PMCs). Top source-line hotspots from PC sampling: `(file:line, Wait_Reason, sample %)`. Wait reasons to call out: WAIT_INST_VMEM, WAIT_INST_LDS, WAIT_INST_SMEM, WAIT_INST_FLAT, WAIT_BARRIER, WAIT_VMCNT, WAIT_LGKMCNT, WAIT_EXPCNT, WAIT_MISC (the last two may not be present on every install — verify against your PC-sampling CSV's `Wait_Reason` column and `rocprofv3 -L`).>

### 2.4 MFMA / matrix-core utilization
<MFMA busy % from rocprof-compute compute-pipe block (`-b 10`); MFMA instruction counts from instruction-mix block (`-b 11`); instruction shape (16×16×16 BF16 / 32×32×8 / FP8 on CDNA3; F6F4 / XF32 on CDNA4), Accum_VGPR (AGPR) usage; or "0%, n/a — kernel is non-MFMA". Cite the actual per-dtype `SQ_INSTS_VALU_MFMA_MOPS_<DTYPE>` counters your install exposes (`rocprofv3 -L | grep MFMA`).>

### 2.5 CU timeline
<shape: flat-high / flat-low / tail / sawtooth — reference the ASCII plot in `analysis/timeline_plots.txt` (single file per run, regardless of tag count). Note rocprof-compute timeseries minimum interval is ~1 ms vs NVIDIA PM ~2 µs, so very-short kernels need ATT instead>

### 2.6 Memory access pattern
<Bytes per wavefront from rocprof-compute instruction-mix / L1D block (`-b 11` / `-b 15`) — peak 256 B for a coalesced wave64 dword load, 1024 B for `global_load_dwordx4`; vL1 / L2 / HBM hit rates; HBM read pressure from `TCC_EA0_RDREQ_*` (single EA channel per XCD on gfx942/gfx950 — `TCC_EA1_*` does NOT exist); scratch traffic (= register spill, on AMD scratch lives in HBM); LDS bank conflicts (`SQ_LDS_BANK_CONFLICT`)>

### 2.7 Additional findings
<items from rocprof-compute SoL gaps not otherwise mentioned — each with the gap-to-peak %>

---

## 3. Summary diagnosis

| Factor | `<tag1>` | `<tag2>` | Impact |
|---|---|---|---|
| <factor 1> | <status> | <status> | <ranked impact> |
| <factor 2> | ... | ... | ... |

---

## 4. Optimization directions (ranked by impact)

> Each priority: name the change, cite evidence, estimate magnitude, flag effort.

### Priority 1 — <one-line name>

<what to do, concretely, with line numbers / function names from the existing kernel>

**Evidence:**
- <counter + value, e.g., "PC-sampling `Wait_Reason == WAIT_INST_VMEM` accounts for 62% of samples">
- <rocprof-compute SoL gap, e.g., "Compute SoL = 18%, HBM SoL = 22% — neither resource saturated, bottleneck is stall">

**Expected impact:** <X% end-to-end, Y% on the hot path>, <which workloads benefit>

**Effort:** <low/medium/high + rough description of the code change>

### Priority 2 — ...

<same structure>

### Priority 3 — ...

(Stop at 3-5. More dilutes the signal.)

---

## 5. Confidence & caveats

- What I'm sure about: <list>
- What I'm uncertain about: <list + what would resolve the uncertainty>
- Anything the profile couldn't answer that the user should know: <list>
- GPU partitioning at profile time (SPX/CPX, NPS1/2/4) and whether the production setup matches.

---

## 6. Reproduction

    cd /abs/path/to/repo
    export PROFILE_RUN_DIR=profile/<run_name>
    <one-block runnable script that builds the harness, runs rocprofv3 + rocprof-compute, and parses>
```

---

## Style rules

- **Cite specific counter values for every claim.** "HBM BW = 3.8 / 5.3 TB/s = 72% of peak" (with the actual number from your report) > "HBM is well-utilized".
- **Name files and line numbers.** "Line L of `harness.hip`" (pasting the actual file/line) > a high-level description like "the main memory load".
- **Use rocprof-compute's SoL gap as the ranking signal.** Each section has a "Speed-of-Light" line that names the bottleneck subsystem and a numeric gap to peak — this is the AMD analog of NCU's `Est. Speedup` rule. Use it instead of guessing.
- **Rank by magnitude.** Fix the 50% problem before the 5% problem.
- **Keep the top-line summary dense.** A reader should be able to get the #1 finding in 10 seconds of reading.
- **Link to artifacts.** Don't paste huge tables into the prose — link to `analysis/compare_<tag1>_vs_<tag2>.txt`, `analysis/details_<tag>.txt`, etc.
- **Always include GPU + partitioning context.** A "300% better than baseline" claim on CPX/NPS4 means nothing if the baseline was SPX/NPS1.

## Anti-patterns

- ❌ Generic advice without evidence ("you might consider using LDS").
- ❌ More than 5 "priorities" — you're probably padding.
- ❌ Re-running the same profile with different tags and copy-pasting the same analysis — consolidate.
- ❌ Reporting from the rocprof-compute terminal table directly. Extract, interpret, write — don't dump.
- ❌ Omitting the setup section. Without it, nobody can reproduce or trust the numbers.
- ❌ Confusing CDNA3 (MI300X) and CDNA4 (MI355X) counters — they share most names but differ in MFMA shapes (block-scaled `F6F4` + `XF32` added on CDNA4, TF32 removed), FP64 throughput (halved on CDNA4), and HBM (HBM3 → HBM3E). The 256 MB Infinity Cache is retained on both. Neither has `TCC_EA1_*` (single EA channel per XCD).
