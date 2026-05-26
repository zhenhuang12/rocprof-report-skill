# Final Report Template

The report is the deliverable. Everything else (rocprof-compute output dirs, ATT JSONs, CSVs) is evidence. Structure matters: a busy reader should see the top findings in 30 seconds and be able to drill into details if they want.

Save as `$PROFILE_RUN_DIR/REPORT.md`. Copy everything inside the fenced block below (drop the outer ```` ```markdown ```` / ```` ``` ```` fence) and fill in the placeholders.

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

    # Compile (MI300X) — drop -DHARNESS_FILLED_IN=1 if you wrote the harness from
    # scratch (the flag only clears the template's #error guard in helpers/harness_template.hip).
    hipcc -O3 -std=c++17 -gline-tables-only \
          --offload-arch=gfx942 -munsafe-fp-atomics \
          -DHARNESS_FILLED_IN=1 \
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
    # Use STOCHASTIC mode to get the wait-reason breakdown — its CSV has the
    # `Stall_Reason` column. The `host_trap` mode does NOT populate `Stall_Reason`;
    # it only gives sampled PCs (good for per-line hotspots, but not for a
    # wait-reason breakdown).
    # See https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-pc-sampling.html
    # Stochastic supports `--pc-sampling-unit cycles` or `instructions` (NOT `time`).
    # Output default: rocprofv3 nests under <hostname>/ with a PID prefix —
    #   pcsamp_<tag>/<hostname>/<pid>_pc_sampling_stochastic.csv
    # Pass `--output-file <prefix>` to collapse to a flat
    #   pcsamp_<tag>/<prefix>_pc_sampling_stochastic.csv
    rocprofv3 --pc-sampling-beta-enabled \
        --pc-sampling-method stochastic \
        --pc-sampling-interval 1048576 --pc-sampling-unit cycles \
        --kernel-include-regex "<kernel_regex>" \
        -f csv \
        -d profile/<run_name>/reports/pcsamp_<tag> \
        -- ./harness [args]

    # (Cheaper alternative — host_trap, hotspots only, no Stall_Reason)
    #   rocprofv3 --pc-sampling-beta-enabled \
    #       --pc-sampling-method host_trap \
    #       --pc-sampling-interval 1000 --pc-sampling-unit time \
    #       --kernel-include-regex "<kernel_regex>" \
    #       -f csv \
    #       -d profile/<run_name>/reports/pcsamp_<tag> \
    #       -- ./harness [args]

### Artifacts

    profile/<run_name>/
    ├── REPORT.md                           ← this file
    ├── harness/...                         ← standalone harness
    ├── reports/
    │   ├── trace_<tag>/                    ← rocprofv3 kernel-trace (+ .db on ROCm 7+)
    │   ├── rpc_<tag>/                      ← rocprof-compute profile output root (the `-p` value)
    │   │   ├── pmc_perf.csv                ← merged PMCs, one row per (dispatch × PMC-group)
    │   │   ├── timestamps.csv              ← per-dispatch Start/End_Timestamp
    │   │   ├── sysinfo.csv                 ← wide single-row sysinfo (NOT param/value)
    │   │   ├── roofline.csv                ← roofline benchmark results (when collected)
    │   │   ├── empirRoof_gpu-0_*.pdf       ← roofline PDF plots (only with --roof-only / --kernel-names)
    │   │   ├── log.txt
    │   │   ├── profiling_config.yaml
    │   │   └── out/pmc_<N>/<host>/<pid>_*.csv   ← raw per-PMC-group passes
    │   │   (Opt-in nested layouts — `--subpath gpu_model`, `--subpath node_name`,
    │   │    or omitted `-p` — move the above under a `<gpu_model>/` or `<hostname>/`
    │   │    child; helpers handle either form.)
    │   ├── rpc_ts_<tag>/                   ← optional `rocprofv3 -P` windowed PMC pass
    │   │   └── <pid>_counter_collection.csv ← one CSV per window; see Recipe 2b
    │   ├── pcsamp_<tag>/<hostname>/        ← rocprofv3 PC sampling output (default nests under <hostname>/)
    │   │   ├── <pid>_pc_sampling_stochastic.csv   ← stochastic mode: has the `Stall_Reason` column (use for breakdown)
    │   │   └── <pid>_pc_sampling_host_trap.csv    ← host_trap mode: per-line hotspots only (no `Stall_Reason`)
    │   │   (Pass `--output-file <prefix>` to collapse to a flat layout.)
    │   └── att_<tag>/                      ← optional ATT (one JSON per SE/CU)
    └── analysis/                           ← extracted metrics (helpers run from $SKILL/helpers/)
        ├── details_<tag>.txt               ← `rocprof-compute analyze` dump
        ├── metrics_all_<tag>.json          ← every parsed counter, full archive
        ├── metrics_key_<tag>.{json,txt}    ← curated key metrics (helper writes both)
        ├── stall_hotspots_<tag>.txt        ← per-line stall aggregation (PC sampling / ATT)
        ├── timeline_plots_<tag_suffix>.txt ← ASCII per-CU / timeseries plots (one file per `plot_timeline.py` invocation; suffix = joined --tag values)
        └── compare_<tag1>_vs_<tag2>.txt

---

## 1. Headline numbers

> A single table that tells the whole story at a glance.

| Metric | `<tag1>` | `<tag2>` | Source |
|---|---:|---:|---|
| **Duration** | X µs | Y µs | `timestamps.csv: End_Timestamp - Start_Timestamp` (or rocprofv3 `kernel_trace.csv`) |
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
| Stall: `WAITCNT` on `global_load_*` lines (% PC samples) | X% | Y% | Stochastic PC-sampling CSV `Stall_Reason` aggregation, filtered by `Instruction_Comment` (NOT a PMC on gfx942/gfx950; host_trap mode does NOT populate `Stall_Reason`) |

**One-line read:** <"The kernel runs at X% of compute SoL — it's VMEM-wait-bound on Y, not HBM-BW-bound."> — this is the punchline.

---

## 2. Per-dimension analysis

> Walk through the six analysis dimensions (see [`05-analysis-dimensions.md`](05-analysis-dimensions.md)), cite counters, state findings.

### 2.1 CU occupancy & launch geometry
<grid size, workgroup size, waves/CU, theoretical vs achieved occupancy, VGPR/AGPR/LDS limits, wave64 math; XCD layout (MI300X: 8 XCDs × 38 CUs over 4 IODs; MI355X: 8 XCDs × 32 CUs over 2 IODs) and whether grid fills them>

### 2.2 Workgroup balance (tail effect)
<per-XCD active cycles, rocprof-compute workgroup-balance breakdown, timeseries shape, input distribution imbalance ratios>

### 2.3 Instruction-level stall analysis
<stall breakdown % from **stochastic** PC-sampling `Stall_Reason` aggregation (the ONLY granular source on gfx942/gfx950 — only `SQ_WAIT_ANY`, `SQ_WAIT_INST_ANY`, `SQ_WAIT_INST_LDS` exist as PMCs; the `host_trap` PC-sampling mode does NOT populate `Stall_Reason`, so use it only for per-line hotspots). Top source-line hotspots from PC sampling: `(file:line, Stall_Reason, sample %)`. The authoritative enum is `ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_*` in `/opt/rocm/include/rocprofiler-sdk/pc_sampling.h`. `Stall_Reason` values in the stochastic CSV: `NONE`, `NO_INSTRUCTION_AVAILABLE`, `ALU_DEPENDENCY`, `WAITCNT`, `INTERNAL_INSTRUCTION`, `BARRIER_WAIT`, `ARBITER_NOT_WIN`, `ARBITER_WIN_EX_STALL`, `OTHER_WAIT`, `SLEEP_WAIT`. Distinguish memory-type subcategories (global vs LDS vs scalar) by reading the ISA mnemonic in `Instruction_Comment` at the sampled PC — e.g. `WAITCNT` on a `global_load_*` line is a VMEM wait; on a `ds_read_*` line it is an LDS wait. (Per-execution-pipe `arb_state_stall_*` / `arb_state_issue_*` bit-fields are JSON-only — use `rocprofv3 ... -f json` and read the `snapshot` object — not CSV columns.) Always verify against your stochastic CSV's actual `Stall_Reason` values rather than assuming the list above is complete — enum values vary across ROCm releases. See [AMD's PC-sampling docs](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-pc-sampling.html).>

### 2.4 MFMA / matrix-core utilization
<MFMA busy % from rocprof-compute compute-pipe block (`-b 10`); MFMA instruction counts from instruction-mix block (`-b 11`); instruction shape (16×16×16 BF16 / 32×32×8 / FP8 on CDNA3; F6F4 / XF32 on CDNA4), Accum_VGPR (AGPR) usage; or "0%, n/a — kernel is non-MFMA". Cite the actual per-dtype `SQ_INSTS_VALU_MFMA_MOPS_<DTYPE>` counters your install exposes (`rocprofv3 -L | grep MFMA`).>

### 2.5 CU timeline
<shape: flat-high / flat-low / tail / sawtooth — reference the ASCII plot in `analysis/timeline_plots_<tag_suffix>.txt` (one file per `plot_timeline.py` invocation; the suffix is the joined `--tag` values, so a single 2-tag invocation produces `timeline_plots_<tag1>_<tag2>.txt`). Note `rocprofv3 -P` windowed-PMC granularity is ~1 ms — very-short kernels need ATT instead>

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
- <counter + value, e.g., "Stochastic PC-sampling `Stall_Reason == WAITCNT` on `global_load_*` lines accounts for 62% of stalled samples">
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
    export PROFILE_RUN_DIR="$PWD/profile/<run_name>"
    export SKILL=~/.claude/skills/rocprof-report-skill
    mkdir -p "$PROFILE_RUN_DIR"/{harness,reports,analysis}

    # 1) Build the standalone harness (same recipe used in helpers/README.md
    #    and helpers/harness_template.hip; keep them in sync if you change it)
    hipcc -O3 -std=c++17 -gline-tables-only \
        --offload-arch=gfx942 -munsafe-fp-atomics \
        -DHARNESS_FILLED_IN=1 \
        "$PROFILE_RUN_DIR/harness/harness.hip" \
        -o "$PROFILE_RUN_DIR/harness/harness"

    # 2) Kernel trace + section perf-metrics (the two canonical passes)
    rocprofv3 --kernel-trace --hip-trace --hsa-trace \
        --kernel-include-regex "<kernel_regex>" \
        -f csv \
        -d "$PROFILE_RUN_DIR/reports/trace_<tag>" \
        -- "$PROFILE_RUN_DIR/harness/harness" <args>

    rocprof-compute profile -n <tag> -k <kernel_substring> --no-roof \
        -p "$PROFILE_RUN_DIR/reports/rpc_<tag>" \
        -- "$PROFILE_RUN_DIR/harness/harness" <args>

    # 3) Parse + write analysis artifacts
    python3 "$SKILL/helpers/analyze_reports.py" \
        --run-dir "$PROFILE_RUN_DIR" \
        --rpc "$PROFILE_RUN_DIR/reports/rpc_<tag>" --tag <tag> \
        --kernel "<kernel_substring>" --arch gfx942
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
