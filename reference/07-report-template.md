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
- Counter-name caveats: any PMC names that differ from rocprof-compute docs (e.g., `TCC_EA0_*` / `TCC_EA1_*` vs `TCC_EA_*` on gfx906/908). See [`08-mi300x-mi355x-counter-names.md`](08-mi300x-mi355x-counter-names.md).
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
    rocprof-compute profile -n <run_name>_<tag> --roofline \
        -k "<kernel_substring>" \
        -p profile/<run_name>/reports/rpc_<tag> \
        -- ./harness [args]
    rocprof-compute analyze -p profile/<run_name>/reports/rpc_<tag> \
        > profile/<run_name>/analysis/details_<tag>.txt

    # 3. Per-line stall sampling (analog of `ncu --set source`)
    # Note the underscore in `host_trap` (not `host-trap`).
    # `host_trap` only supports `--pc-sampling-unit time` (`cycles`/`instructions`
    # are stochastic-only and the runtime rejects the wrong combo).
    rocprofv3 --pc-sampling-beta-enabled \
        --pc-sampling-method host_trap \
        --pc-sampling-interval 1000000 --pc-sampling-unit time \
        --kernel-include-regex "<kernel_regex>" \
        -d profile/<run_name>/reports/pcsamp_<tag> \
        -- ./harness [args]

### Artifacts

    profile/<run_name>/
    ├── REPORT.md                           ← this file
    ├── harness/...                         ← standalone harness
    ├── reports/
    │   ├── trace_<tag>/                    ← rocprofv3 kernel-trace (+ .db on ROCm 7+)
    │   ├── rpc_<tag>/                      ← rocprof-compute profile dir (flat)
    │   │   ├── pmc_perf.csv                ← all PMCs land here, one row per dispatch
    │   │   ├── timestamps.csv
    │   │   ├── sysinfo.csv
    │   │   ├── roofline.csv                ← when --roofline was passed
    │   │   ├── profiling_config.yaml
    │   │   └── perfmon/                    ← per-PMC-group .txt/.yaml inputs
    │   ├── pcsamp_<tag>/                   ← PC sampling CSV
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
| **Duration** | X µs | Y µs | `timestamps.csv: End - Start` |
| SoL — Compute (% peak) | X% | Y% | rocprof-compute §2.1.1 |
| SoL — HBM (% peak) | X% | Y% | rocprof-compute §2.1.1 |
| SoL — vL1 (TCP) | X% | Y% | rocprof-compute §2.1.1 |
| SoL — L2 (TCC) | X% | Y% | rocprof-compute §2.1.1 |
| HBM read BW (achieved / peak) | X / 5300 GB/s | … | `TCC_EA0_RDREQ_32B_sum + TCC_EA1_RDREQ_32B_sum` × 32 |
| vL1 hit rate | X% | Y% | rocprof-compute §2.1.15 |
| L2 hit rate | X% | Y% | rocprof-compute §2.1.16 |
| MFMA busy (% peak) | X% | Y% | rocprof-compute §2.1.10 |
| VGPR / wave | X | Y | `pmc_perf.csv: VGPRs` |
| AGPR / wave | X | Y | `pmc_perf.csv: AGPRs` (CDNA3+) |
| LDS / workgroup | X B | Y B | `pmc_perf.csv: LDS_Per_Workgroup` |
| Achieved occupancy (waves/SIMD) | X / 8 | Y / 8 | rocprof-compute §2.1.2 |
| Scratch (= register spill, bytes/wi) | X | Y | `pmc_perf.csv: Scratch_Per_Workitem` |
| Stall: WAIT_INST_VMEM (% issue slots) | X% | Y% | rocprof-compute §2.1.13 |

**One-line read:** <"The kernel runs at X% of compute SoL — it's VMEM-wait-bound on Y, not HBM-BW-bound."> — this is the punchline.

---

## 2. Per-dimension analysis

> Walk through the six analysis dimensions (see [`05-analysis-dimensions.md`](05-analysis-dimensions.md)), cite counters, state findings.

### 2.1 CU occupancy & launch geometry
<grid size, workgroup size, waves/CU, theoretical vs achieved occupancy, VGPR/AGPR/LDS limits, wave64 math; XCD layout (MI300X: 8 XCDs × 38 CUs over 4 IODs; MI355X: 8 XCDs × 32 CUs over 2 IODs) and whether grid fills them>

### 2.2 Workgroup balance (tail effect)
<per-XCD active cycles, rocprof-compute §2.1.23 imbalance, timeseries shape, input distribution imbalance ratios>

### 2.3 Instruction-level stall analysis
<stall breakdown % from §2.1.13, top source-line hotspots from PC sampling: `(file:line, Wait_Reason, sample %)`. Wait reasons to call out: WAIT_INST_VMEM, WAIT_INST_LDS, WAIT_INST_SMEM, WAIT_INST_FLAT, WAIT_BARRIER, WAIT_VMCNT, WAIT_LGKMCNT. Verify the exact label spelling with `rocprofv3 -L | grep SQ_WAIT` on the install you collected on.>

### 2.4 MFMA / matrix-core utilization
<MFMA busy % from §2.1.10, instruction shape (16×16×16 BF16 / 32×32×8 / FP8 / CDNA4 FP4/FP6/MXFP), AGPR usage; or "0%, n/a — kernel is non-MFMA">

### 2.5 CU timeline
<shape: flat-high / flat-low / tail / sawtooth — reference the ASCII plot in `analysis/timeline_plots.txt` (single file per run, regardless of tag count). Note rocprof-compute timeseries minimum interval is ~1 ms vs NVIDIA PM ~2 µs, so very-short kernels need ATT instead>

### 2.6 Memory access pattern
<Bytes per wavefront from §2.1.11 (peak 256B for fully coalesced wave64 × dword), vL1 / L2 / HBM hit rates, per-channel HBM balance (TCC_EA0 vs TCC_EA1 — should be ~50/50), scratch traffic (= register spill, on AMD scratch lives in HBM), LDS bank conflicts (`SQ_LDS_BANK_CONFLICT`)>

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
- <counter + value, e.g., `SQ_WAIT_INST_VMEM = 62%` of issue slots>
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
- ❌ Confusing CDNA3 (MI300X) and CDNA4 (MI355X) counters — they share most names but differ in MFMA shapes (FP4/FP6/MXFP + 2:4 sparsity added, TF32 removed), FP64 throughput (halved on CDNA4), and HBM (HBM3 → HBM3E). The 256 MB Infinity Cache is retained on both.
