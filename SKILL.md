---
name: rocprof-report-skill
description: Use when the user asks to profile a HIP / ROCm kernel on AMD Instinct MI300X (gfx942 / CDNA3) or MI355X (gfx950 / CDNA4), analyze its performance, diagnose a bottleneck, read a rocprofv3 / rocprof-compute / Omniperf report, or write an optimization plan from rocprof evidence — including Chinese phrasings ("profile 一下", "为什么慢", "rocprof 报告说...", "Omniperf 看一下").
---

# Skill: HIP Kernel Profiling (MI300X / MI355X, rocprofv3 + rocprof-compute)

**When to use:** user asks to profile a HIP kernel, analyze its performance, find its bottlenecks, or write an optimization plan based on rocprofv3 / rocprof-compute (formerly Omniperf) data. Triggers include: "profile X", "为什么这个 kernel 慢", "rocprof / Omniperf 报告说...", "下一步怎么优化", "帮我看一下这份 rocprof 报告".

**Target hardware (this repo):** AMD Instinct **MI300X** (gfx942, CDNA3, 304 CUs across 8 XCDs over 4 IODs, 192 GB HBM3 @ 5.3 TB/s, 256 MB Infinity Cache, 32 KB vL1/CU) and **MI355X** (gfx950, CDNA4, 256 CUs = 8 XCDs × 32 CUs over 2 IODs, 288 GB HBM3E @ 8.0 TB/s, 256 MB Infinity Cache retained, FP4/FP6/MXFP support). Most advice below is generic; gfx942- and gfx950-specific notes are explicitly marked.

---

## Golden rule

**Profile → Diagnose → Plan, in that order. Never guess.**

Most under-performing HIP kernels are under-performing for exactly one reason that rocprofv3 + rocprof-compute can tell you in 10 seconds. Don't invent hypotheses before you have the report. Don't start coding a fix before you've matched the observed pattern to a known diagnosis. Don't write a wall of suggestions — rank them by evidence and expected impact.

---

## Quickstart (what to do when someone says "profile this kernel")

0. **Create a new run directory first** under `profile/<run_name>/` at the repo root — **one directory per run**, never reuse an existing one. Each run contains its own `harness/`, `reports/`, `analysis/`, and `REPORT.md`. This rule is mandatory in this repo. See [`reference/00-directory-layout.md`](reference/00-directory-layout.md). Pin both env vars up front; every later step uses them:

   ```bash
   export PROFILE_RUN_DIR="$PWD/profile/<run_name>"
   export SKILL=~/.claude/skills/rocprof-report-skill   # or <repo>/.claude/skills/rocprof-report-skill
   mkdir -p "$PROFILE_RUN_DIR"/{harness,reports,analysis}
   ```

   Throughout this skill `<tag>` is a short, descriptive label per profile invocation (e.g. `baseline`, `v2_lds`, `shape_seq2048`) that gets appended to report directory names — `reports/rpc_<tag>/`, `reports/pcsamp_<tag>/`, `analysis/metrics_key_<tag>.txt`. Use the **same** tag across all three (a/b/c) profiles in a single run so the helpers can correlate them. If you're profiling multiple shapes / variants in one run directory, give each its own tag.

1. **Decide what you're profiling.** What inputs? Which dispatch path? What question do you want answered? If the kernel takes variable-sized inputs (variable seq lengths, variable batch sizes), you must pick specific representative shapes from the user's workload — don't profile with arbitrary inputs.

2. **Build a standalone harness** unless the user is profiling through their existing binary. Harnesses compile in seconds, run the kernel in isolation, and let you use `-gline-tables-only` cleanly so ATT / PC-sampling can map ISA back to source. Compile into `profile/<run_name>/harness/`. See [`reference/02-harness-guide.md`](reference/02-harness-guide.md) and the template in [`helpers/harness_template.hip`](helpers/harness_template.hip). **The template ships with a `#error` guard**; pass `-DHARNESS_FILLED_IN=1` once you've replaced its TODOs.

3. **Run three (sometimes four) profiles** — write all outputs to `profile/<run_name>/reports/`. See [`reference/03-collection.md`](reference/03-collection.md) for the full recipes. Begin every shell that runs a profile recipe with these guards so a missing var fails loudly instead of writing into `/reports/...`:

   ```bash
   : "${PROFILE_RUN_DIR:?run step 0 first — PROFILE_RUN_DIR is unset}"
   : "${SKILL:?export SKILL=... to your skill install path}"
   ```

   - **(a) Overview timeline:** `rocprofv3 --kernel-trace --hip-trace --hsa-trace ...`.
   - **(b) Section-based perf metrics:** `rocprof-compute profile -k <kernel-substring> ...` — the AMD analog of `ncu --set full`. Roofline is on by default; pass `--no-roof` to skip.
   - **(c) Per-line stall attribution** (AMD analog of `ncu --set source --section SourceCounters`) — **prefer PC sampling**:
     - **Stochastic** mode is the only one that populates the `Stall_Reason` CSV column: `rocprofv3 --pc-sampling-beta-enabled --pc-sampling-method stochastic --pc-sampling-unit cycles --pc-sampling-interval 1048576`. **Note:** stochastic mode requires `--pc-sampling-unit cycles` (canonical; `instructions` exists in the SDK enum but may not be CLI-wired in your build — see [`reference/03-collection.md`](reference/03-collection.md) for the full compatibility table) — `time` is runtime-rejected on stochastic. See [AMD docs](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-pc-sampling.html).
     - **Host-trap** mode is cheaper but emits sampled PCs only — use for per-line hotspots when you don't need the stall-reason classification: `rocprofv3 --pc-sampling-beta-enabled --pc-sampling-method host_trap --pc-sampling-unit time --pc-sampling-interval 1000`.
     - **ATT/SQTT** (`rocprofv3 --att ...`) is the fallback only when PC sampling isn't available.
     - rocprofv3's default `-d` nests outputs under `<hostname>/` with a `<pid>_` filename prefix — `pcsamp_<tag>/<hostname>/<pid>_pc_sampling_stochastic.csv`. Pass `--output-file <prefix>` to collapse to a flat `pcsamp_<tag>/<prefix>_pc_sampling_*.csv`. This is distinct from rocprof-compute's raw per-pass PMC layout, which lands under `<-p>/out/pmc_<N>/<hostname>/<pid>_*.csv` (one `<N>` per PMC group); the merged `pmc_perf.csv` / `timestamps.csv` / `sysinfo.csv` from rocprof-compute live flat directly under `-p`.
   - **(d) — only if you need CU timeline / tail-effect shape:** windowed PMC capture via `rocprofv3 -P 0:1:50 --collection-period-unit msec --pmc SQ_BUSY_CYCLES ... -d profile/<run_name>/reports/rpc_ts_<tag>`. `rocprof-compute` does **not** expose a `--timeseries-sampling-rate` flag — use `rocprofv3 -P` instead. See Recipe 2b in [`reference/03-collection.md`](reference/03-collection.md).

4. **Parse with Python** — `pandas` for the CSVs, `sqlite3` for `.rpd` / `.db` files, and `rocpd` / `rocprof-analyze` for cross-section queries — not by eye-balling the CLI. Write analysis outputs to `profile/<run_name>/analysis/`. Use the helpers under `$SKILL/helpers/` — invoke them with their full path (`python3 "$SKILL/helpers/analyze_reports.py" --run-dir "$PROFILE_RUN_DIR" ...`), passing `$PROFILE_RUN_DIR` rather than `cd`-ing into it. **Set `$SKILL` once, explicitly, to your install path** — e.g. `export SKILL=~/.claude/skills/rocprof-report-skill` (or `<repo>/.claude/skills/rocprof-report-skill` for a project-level install); do not rely on cwd-based auto-derivation. See [`reference/04-python-api.md`](reference/04-python-api.md).

5. **Work through the six analysis dimensions.** See [`reference/05-analysis-dimensions.md`](reference/05-analysis-dimensions.md). Every one matters, but on any given kernel only 1–2 will dominate.

6. **Match patterns to the diagnosis playbook.** See [`reference/06-diagnosis-playbook.md`](reference/06-diagnosis-playbook.md). It maps rocprof signal → likely cause → concrete fix, with example counts for "how big is this".

7. **Write the report** at `profile/<run_name>/REPORT.md` with evidence-backed recommendations, ranked by expected impact. See [`reference/07-report-template.md`](reference/07-report-template.md).

---

## File index

### Reference docs (read these when you need details)

| File | Purpose |
|---|---|
| [`reference/00-directory-layout.md`](reference/00-directory-layout.md) | **Read first.** Directory / naming conventions — one run = one subdirectory, no cross-contamination |
| [`reference/01-workflow.md`](reference/01-workflow.md) | End-to-end checklist from "user request" to "final report" |
| [`reference/02-harness-guide.md`](reference/02-harness-guide.md) | When and how to build a standalone HIP harness (mandatory for PyTorch + ROCm, Triton-MLIR, hipBLASLt JIT) |
| [`reference/03-collection.md`](reference/03-collection.md) | rocprofv3 / rocprof-compute command recipes: kernel trace, section perf-metrics, ATT, PC sampling, custom PMC |
| [`reference/04-python-api.md`](reference/04-python-api.md) | Pandas + sqlite3 (rocpd) patterns with copy-pasteable code |
| [`reference/05-analysis-dimensions.md`](reference/05-analysis-dimensions.md) | Six analysis dimensions: occupancy, balance, stalls, Matrix-Core, timeline, memory |
| [`reference/06-diagnosis-playbook.md`](reference/06-diagnosis-playbook.md) | Pattern → diagnosis → fix. Merges CDNA3 / CDNA4 programming principles with rocprof signals |
| [`reference/07-report-template.md`](reference/07-report-template.md) | How to structure the final report |
| [`reference/08-mi300x-mi355x-counter-names.md`](reference/08-mi300x-mi355x-counter-names.md) | gfx942 / gfx950 PMC counter names and rocprof-compute section IDs |
| [`reference/09-common-issues.md`](reference/09-common-issues.md) | Permissions, ATT capture gaps, ROCm version, PyTorch / Triton gotchas |

### Helpers (reusable code)

| File | Purpose |
|---|---|
| [`helpers/harness_template.hip`](helpers/harness_template.hip) | Standalone HIP harness template — paste your kernel, fill in input allocation, done |
| [`helpers/safetensors_loader.h`](helpers/safetensors_loader.h) | Header-only safetensors reader (no external deps) for loading real workload tensors |
| [`helpers/analyze_reports.py`](helpers/analyze_reports.py) | Extract key metrics from rocprof-compute CSVs, produce side-by-side comparisons |
| [`helpers/extract_stall_hotspots.py`](helpers/extract_stall_hotspots.py) | Per-line stall aggregation from ATT / PC-sampling output |
| [`helpers/plot_timeline.py`](helpers/plot_timeline.py) | ASCII PMC-timestamp / rocprof-compute timeseries plotter — makes tail effect visible |
| [`helpers/list_flashinfer_workloads.py`](helpers/list_flashinfer_workloads.py) | Browse a flashinfer-trace dataset — shape histograms, filter by axis, resolve safetensors paths for specific UUIDs |
| [`helpers/rocprof_utils.py`](helpers/rocprof_utils.py) | Shared Python helpers: safe CSV-column access, rocpd query helpers, MI300X / MI355X key counter list |

---

## Critical lessons (don't skip)

1. **Don't transplant NVIDIA-style metric names.** AMD has its own counter taxonomy: `SQ_*` (shader / wave), `TCP_*` (vL1 cache), `TCC_EA0_*` (L2 on MI300+ — note the `_EA0` / `_EA1` channel suffix, NOT `TCC_EA_*`), `GRBM_*` (graphics / global). Many third-party blog posts cite gfx906 (MI50) or gfx908 (MI100) names that no longer exist on gfx942 / gfx950. Use the lists in [`reference/08-mi300x-mi355x-counter-names.md`](reference/08-mi300x-mi355x-counter-names.md) or enumerate via `rocprofv3 -L` (long form `--list-avail`; the older rocprof v1 `--list-counters` / `--list-metrics` flags are gone, but `-L` / `--list-avail` is the canonical rocprofv3 spelling — verified against `rocprofv3 --help` on ROCm 7.x).

2. **Always compile with `-gline-tables-only` (or `-g`).** Without it, the `Source` column in ATT output is blank, and PC-sampling per-line attribution (`file:line` reconstructed via `addr2line` from the `Instruction` PC) cannot be done — `Instruction_Comment` itself (the ISA mnemonic) is always populated, but you'll have no way to map PCs back to source lines. If you can't add `-gline-tables-only` to the build system (PyTorch's `torch.utils.cpp_extension`, Triton, hipBLASLt JIT), **build a standalone harness** — that's the whole point.

3. **rocprof-compute timeseries (or sampled PMC) is the only way to see tail effects.** Static averaged counters average over the whole kernel; only timeseries (the ASCII plotter in `helpers/` or `rocprof-compute analyze -p <dir> --gui`) shows the shape of utilization over time.

4. **Load-imbalance on variable-length inputs is often the #1 bottleneck.** If the user's workload has sequences of varying length, per-CU active-cycle variance (and per-XCD on MI300X) will often dwarf every other effect. Always check the input distribution.

5. **rocprof-compute's "Speed-of-Light" panel already does half the work.** Each section (2.1.x ID) summarizes the gap between achieved and peak and ranks which subsystem is the bottleneck. Read it first — it often points straight at the answer.

6. **Don't delegate understanding.** Run the profiles yourself, open the reports, cite specific counter values. Never write "the profile shows it's memory-bound" — instead, name the two or three counter values that back your conclusion (e.g., "`TCC_EA0_RDREQ_sum` per kernel sits at `<NUMBER>` (only `<NUMBER>`% of peak HBM BW), and `SQ_WAIT_INST_LDS / SQ_BUSY_CYCLES` exceeds 30%, so the kernel is **LDS-bank-conflict-bound on the shared-memory phase**, not HBM-BW-bound"). Replace every `<NUMBER>` with the actual value from your report — do **not** copy literal `X` / `Y` / `<NUMBER>` into the final report. Verify each counter name with `rocprofv3 -L | grep` if you're unsure. Specificity is the deliverable.

---

## Related skills

- [`cdna3-cdna4-hip-programming.md`](cdna3-cdna4-hip-programming.md) — CDNA3 (MI300X) and CDNA4 (MI355X) specific programming principles and checklists, preserved as a companion reference. Use it when proposing *new* kernel designs; use this skill when diagnosing *existing* kernels.
