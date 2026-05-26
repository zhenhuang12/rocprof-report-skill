# Profiling Workflow — End-to-End

This is the complete checklist from "user asks to profile" to "final report". Every step has a short rationale and a pointer to the detailed doc.

---

## Phase 0 — Create a new run directory

**Always start here.** See [`00-directory-layout.md`](00-directory-layout.md) for the full convention.

```bash
# At the repo root
PROFILE_RUN_DIR=profile/<descriptive_run_name>        # e.g. <kernel>_v1_baseline
mkdir -p "$PROFILE_RUN_DIR"/{harness,reports,analysis}
```

- Pick a new, descriptive name for this run. Never reuse an existing directory.
- If you're profiling a new version of a kernel you've profiled before, that's a **new** run (e.g. `<kernel>_v2_optimized/`, not overwriting `<kernel>_v1_baseline/`).
- If you're profiling the same version against a different workload, that's also a new run — or, at minimum, each workload's report gets a distinct tag and the analysis scripts are kept separate.

Every artifact produced in subsequent phases is written **only** under `$PROFILE_RUN_DIR`. Never into a sibling run's directory.

---

## Phase 0.5 — Frame the problem (before any tools)

Before typing any commands, answer these in your head (or in a short note to the user):

1. **What kernel(s) am I profiling?** Get the exact kernel name or regex. Kernels are often templated (`foo_kernel<8, 256>`) and rocprofv3's `--kernel-include-regex "..."` needs to match the *demangled* name (the same Itanium-mangled C++ symbol you would see in `llvm-objdump --syms --demangle`).
2. **Which workload / input shape?** If the kernel takes variable-sized inputs, pick a **specific** real workload — don't invent shapes. If the user has multiple representative shapes, profile the hottest one first; profile others only if the first reveals nothing.
3. **Which dispatch path?** Many production kernels branch on input shape or other runtime values to pick different grid configs or template instantiations. Profile each *active* dispatch path separately — treating them as one kernel will average out the real patterns.
4. **What question am I answering?** "Why is this slow?" is too vague. Better: "At shape X, is the kernel latency-bound or HBM-BW-bound?" or "We spent 2 weeks on LDS double-buffering — did it actually help?"
5. **What is the baseline?** If there's a reference implementation (PyTorch, hipBLASLt, rocBLAS, Composable Kernel, a previous version), profile it too for comparison.

If any of 1-4 are unclear, **ask the user** before profiling. Profiling the wrong thing wastes an hour.

---

## Phase 1 — Environment check

```bash
# 1. rocprofv3 CLI is available (ROCm 6.2+)
rocprofv3 --version

# 2. rocprof-compute CLI is available (ROCm 6.3+; was "omniperf" before)
rocprof-compute --version
# fallback name on slightly older systems:
# omniperf --version

# 3. GPU is visible and at the expected arch
rocminfo | grep -E "Name:|gfx"        # expect gfx942 for MI300X, gfx950 for MI355X
rocm-smi --showproductname
rocm-smi --showmeminfo vram

# 4. HIP compiler is available
hipcc --version                       # bundles amdclang++

# 5. Permissions. On most servers the user just needs to be in the `render`
#    (and often `video`) group; no root needed for rocprofv3 or rocprof-compute.
#    ATT and PC sampling sometimes need CAP_PERFMON or `kfd_admin_group`.
#    See 09-common-issues.md if rocprofv3 reports missing counters.
id | tr ',' '\n' | grep -E "render|video|kfd"
```

For Python scripts that parse `.rpd` / `.db` SQLite files, no PYTHONPATH gymnastics are needed — Python's built-in `sqlite3` is sufficient. For `pandas` CSV parsing:

```bash
python3 -c "import pandas, sqlite3; print('OK')"
```

If you want the `rocpd` query helper or the rocprof-analyze utility:

```bash
python3 -c "import rocpd; print(rocpd.__file__)"   # ships with ROCprofiler-SDK in ROCm 7+
```

---

## Phase 2 — Build a profile target

**Option A (preferred): standalone harness.** Build a small C++/HIP driver that launches your kernel directly. See [`02-harness-guide.md`](02-harness-guide.md). This is the right choice when:

- The kernel lives inside a JIT/template build system (PyTorch's `torch.utils.cpp_extension`, Triton-MLIR, hipBLASLt JIT, Composable Kernel JIT) where you can't easily add `-gline-tables-only`.
- You want fast iteration — the harness compiles in < 10 seconds, vs minutes for rebuilding the whole framework.
- You want precise control over inputs (e.g., load specific workload tensors from the dataset).

**Option B: profile through existing binary.** Skip the harness if:

- The build system already compiles with `-gline-tables-only` (or `-g`). Check the `hipcc` / `amdclang++` command line in your build log.
- You *need* to profile in-context (e.g., kernel interacts with other kernels, host-side CPU work matters).

Either way, **make sure `-gline-tables-only` (or `-g`) is in the hipcc command**. Without it, the `Source` column in ATT and the `Instruction_Comment` column in PC-sampling will be blank, and per-line stall analysis is impossible.

---

## Phase 3 — Collect profiles

Run three rocprof invocations — **all outputs go under `$PROFILE_RUN_DIR/reports/`**. Details in [`03-collection.md`](03-collection.md).

```bash
# (1) Kernel-trace overview — runtime/host events, kernel durations
rocprofv3 --kernel-trace --hip-trace --hsa-trace \
    --kernel-include-regex "YOUR_KERNEL_NAME" \
    -d "$PROFILE_RUN_DIR/reports/trace_<tag>" \
    -- "$PROFILE_RUN_DIR/harness/your_harness" [args]

# (2) Section-based perf metrics (analog of ncu --set full)
rocprof-compute profile -n <run_name>_<tag> \
    --roofline \
    --kernel-name "YOUR_KERNEL_NAME" \
    -p "$PROFILE_RUN_DIR/reports/rpc_<tag>" \
    -- "$PROFILE_RUN_DIR/harness/your_harness" [args]

# (3) Per-source-line stall sampling (analog of ncu --set source)
# Prefer PC sampling on MI300X+ when supported (lower overhead than ATT).
# `--pc-sampling-beta-enabled` is REQUIRED in ROCm 6.4+ — the feature is still beta.
# Note the underscore in `host_trap` (not `host-trap`).
rocprofv3 --pc-sampling-beta-enabled \
    --pc-sampling-method host_trap \
    --pc-sampling-interval 1000 --pc-sampling-unit cycles \
    --kernel-include-regex "YOUR_KERNEL_NAME" \
    -d "$PROFILE_RUN_DIR/reports/pcsamp_<tag>" \
    -- "$PROFILE_RUN_DIR/harness/your_harness" [args]
# Fall back to ATT (Advanced Thread Trace / SQTT) when PC sampling is unavailable:
rocprofv3 --att --att-target-cu 0 --att-buffer-size 0x10000000 \
    --kernel-include-regex "YOUR_KERNEL_NAME" \
    -d "$PROFILE_RUN_DIR/reports/att_<tag>" \
    -- "$PROFILE_RUN_DIR/harness/your_harness" [args]
```

Run the triple once per (kernel, dispatch path, representative workload) combination.

Timing budget:
- `rocprofv3 --kernel-trace`: ~1 pass, near-zero overhead.
- `rocprof-compute profile --roofline`: 15-30 replay passes (each PMC group needs its own pass). For a 3 ms kernel ≈ 10-20 s wall time.
- `rocprofv3 --att`: one extra pass; output can be very large (hundreds of MB per CU).
- `rocprofv3 --pc-sampling-beta-enabled --pc-sampling-method host_trap`: low overhead; one pass.

---

## Phase 4 — Extract structured data

Do not eyeball the CLI output. Parse reports in Python so you can compare, aggregate, and archive. See [`04-python-api.md`](04-python-api.md) and use the helpers in [`../helpers/`](../helpers/).

Minimum analysis artifacts to produce:

| Artifact | Tool | What it tells you |
|---|---|---|
| `metrics_key_<tag>.txt` | `analyze_reports.py` | ~80 curated counters (launch geom, SOL, occupancy, stalls, sectors) |
| `metrics_all_<tag>.json` | `analyze_reports.py` | Full PMC dump, archive for later |
| `compare_<a>_vs_<b>.txt` | `analyze_reports.py` | Side-by-side metric comparison between workloads / versions |
| `stall_hotspots_<tag>.txt` | `extract_stall_hotspots.py` | Top source lines ranked by stall samples (from PC-sampling or ATT) |
| `timeline_plots.txt` | `plot_timeline.py` | ASCII time-series plots — reveals tail effect visually |
| `details_<tag>.txt` | `rocprof-compute analyze -p ...` | rocprof-compute's built-in section reports (each with peak-comparison + bottleneck hints). `--list-stats` only *lists* section IDs; omit it to get the full dump. |

Save everything under `$PROFILE_RUN_DIR/analysis/`. The user will want to re-inspect these; if two runs mix artifacts, you've already failed.

---

## Phase 5 — Diagnose

Work through the six analysis dimensions — see [`05-analysis-dimensions.md`](05-analysis-dimensions.md):

1. **CU occupancy & wave structure** — are enough workgroups launched to fill the chip (304 CUs across 8 XCDs on MI300X; 256 CUs across 2 IODs on MI355X)? Is occupancy register- / LDS- / workgroup-limited?
2. **Workgroup balance (tail effect)** — do per-CU / per-XCD active cycles match? Does the PMC timeline show a clean drop or a gradual tail?
3. **Instruction-level stall analysis** — what wait reason dominates (`SQ_WAIT_INST_VMEM`, `SQ_WAIT_INST_LDS`, `SQ_WAIT_BARRIER`, plus the PC-sampling `Wait_Reason` enums)? Which source line generates it?
4. **Matrix-Core utilization** — if this is a GEMM-ish kernel, are MFMA instructions actually being issued (`SQ_INSTS_VALU_MFMA_MOPS_<dtype>` / `SQ_VALU_MFMA_BUSY_CYCLES`)?
5. **CU utilization timeline** — flat high, flat low, periodic waves, gradual tail?
6. **Memory access pattern** — bytes/wavefront, vL1/L2 hit rates, HBM throughput, LDS bank conflicts, register/scratch spill.

For each dimension, write down the observed signal *and the specific counter value* that produced it. "Kernel is memory bound" is useless; something like "`TCC_EA0_RDREQ_sum / GRBM_GUI_ACTIVE` works out to X% of peak HBM3 BW (well below peak) shows the kernel is *not* HBM-BW-bound — the `SQ_WAIT_INST_LDS / SQ_INSTS_VALU` of Y% says it's latency-bound on LDS bank conflicts" is diagnosis. Fill in X and Y from your own report.

Then consult [`06-diagnosis-playbook.md`](06-diagnosis-playbook.md) which maps observed patterns to likely causes and concrete fixes.

---

## Phase 6 — Write the report

Structure described in [`07-report-template.md`](07-report-template.md). Key elements:

1. **Setup section**: exactly how you profiled (harness path, workloads, rocprof commands, counter-name caveats, ROCm version, gfx arch). Required for reproducibility.
2. **Headline numbers**: duration, CU active %, vL1/L2 hit, HBM throughput, MFMA active %, achieved occupancy. A table on the first page.
3. **Per-dimension analysis** with evidence (counter values + rocprof-compute section text).
4. **Optimization directions** ranked by expected impact (use the section's "Speed-of-Light gap" numbers and the roofline distance when available).
5. **Confidence & caveats**.

Keep the report short enough that a busy reader can see the top 3 findings in 30 seconds. Put deep detail in the artifacts, not the prose.

---

## Anti-patterns to avoid

- ❌ **"I ran rocprof and it says memory throughput is 14%"** — without naming the counter, workload, and kernel, this is un-actionable. Always give counter + value + what it means.
- ❌ **Profiling with synthetic shapes that don't match real workloads.** A uniform-element batch is a very different problem than a batch with highly skewed per-element work (the latter exposes tail effects the former hides). If the production workload has imbalance, you must profile on an imbalanced workload.
- ❌ **Dumping the full rocprof-compute CLI output into the report.** It's noisy, narrow-formatted, and has no interpretation. Extract the numbers, cite the source, add your reading.
- ❌ **Proposing optimizations without evidence.** "Maybe we should use LDS" is not a profiling result. A real proposal cites a specific source line, its stall-sample count, the relevant section's peak-gap, and the mechanism of the fix — e.g. "line L's global-load instruction accounts for N% of `SQ_WAIT_INST_VMEM` samples; rocprof-compute reports the per-wave global-load is M bytes (only K% of the peak 256-bit `global_load_dwordx4`); rewriting the per-thread index from stride-K to contiguous + using `global_load_lds_dwordx4` should eliminate most of those stalls."
- ❌ **Missing the #1 finding because you got distracted by a smaller one.** Rank findings by impact. Tail effects (especially across XCDs on MI300X SPX/NPS1) and CU idle time often dwarf coalescing issues; fix the big one first.
