# Profile Directory Layout & Naming

**Read this first, before any collection.** Bad directory layout is the single most common cause of mixing results from different runs, overwriting prior profiles, or losing track of which `.rpd` / rocprof-compute output belongs to which kernel version. The rules below are non-negotiable for work in this repo.

---

## Top-level rule

**All profiling artifacts live under a single `profile/` directory at the repo root.** Never scatter `.rpd` / `.db` / `.csv` files across random locations. Never put profile artifacts under `solution/`, `src/`, `scripts/`, or other source directories.

```
<repo_root>/
├── profile/                        ← everything profiling-related lives here
│   ├── <run_1>/
│   ├── <run_2>/
│   └── ...
├── solution/                       ← untouched by profiling
├── src/
└── ...
```

---

## One run = one subdirectory

Every time you profile a kernel — whether it's a new kernel, a new version of the same kernel, or the same kernel on a different workload — **create a new subdirectory under `profile/`**. Never write into an existing run's directory.

Rationale:

- Profiles of different implementations of the same kernel must not overwrite each other. If you profile `<kernel>_v1` today and `<kernel>_v2` tomorrow, both reports need to coexist for A/B comparison.
- The harness itself is part of the profile: it encodes which kernel code was compiled, with which flags, against which workload. Keeping the harness source in the run dir pins the provenance.
- Analysis artifacts (`metrics_*.json`, `compare_*.txt`, ASCII plots) are tied to a specific set of underlying rocprof outputs; they must not be mixed.

---

## Run directory naming

Use descriptive, short, kebab-case names. Include **what** was profiled and **when/why**, not how.

Good:
```
profile/<kernel>_v1_baseline/
profile/<kernel>_v2_optimized/
profile/<kernel>_v2_optimized_vs_v1/         # for comparison run
profile/moe_fp8_v4_lds_prefetch/
profile/flash_attn_mi300x_h128_baseline/
profile/flash_attn_mi355x_h128_mxfp4/
```

Bad:
```
profile/test/                   # too vague
profile/run1/                   # meaningless
profile/20260413/               # dates with no context
profile/final/                  # there's never a "final"
```

If you genuinely have multiple runs on the same day for the same kernel/version combo, append a short distinguisher or a date suffix: `<kernel>_v1_baseline_20260526_am` / `<kernel>_v1_baseline_20260526_pm`.

---

## Standard run layout

Inside each run subdirectory, use this structure:

```
profile/<run_name>/
├── REPORT.md                       ← human-readable final report (Markdown)
├── harness/
│   ├── <kernel>_harness.hip        ← the exact source that was compiled
│   ├── <kernel>_harness            ← compiled binary (with -gline-tables-only)
│   └── build_command.sh            ← optional: shell script that compiled it
├── reports/
│   ├── trace_<tag1>/               ← rocprofv3 kernel-trace output dir (.csv / .json / .pftrace / .db)
│   ├── trace_<tag2>/
│   ├── rpc_<tag1>/                 ← rocprof-compute "profile" output dir
│   │   ├── pmc_perf.csv            ← merged PMCs (one row per dispatch × PMC group)
│   │   ├── pmc_kernel_top.csv      ← top-K kernels by dispatch count
│   │   ├── sysinfo.csv             ← wide single-row sysinfo (NOT param/value)
│   │   ├── roofline.pdf            ← PDF when roofline ran (default-on; --no-roof to skip)
│   │   ├── profiling_config.yaml
│   │   └── out/pmc_<N>/<host>/<pid>_*.csv   ← raw per-PMC-group passes
│   ├── rpc_<tag2>/
│   ├── rpc_ts_<tag1>/              ← (optional) rocprof-compute --timeseries-sampling-rate output
│   │   └── pmc_perf_timeseries.csv ← consumed by plot_timeline.py / Dimension 5 (CU timeline)
│   ├── att_<tag1>/                 ← rocprofv3 --att output dir (JSON traces per CU)
│   ├── att_<tag2>/
│   ├── pcsamp_<tag1>/              ← rocprofv3 --pc-sampling output dir (CSV per kernel)
│   └── pcsamp_<tag2>/
└── analysis/                       ← only the OUTPUT artifacts live here;
    │                                  invoke the helpers from $SKILL/helpers/
    │                                  with --run-dir, do NOT copy them in.
    ├── metrics_all_<tag>.json      ← every parsed counter, full archive
    ├── metrics_key_<tag>.{txt,json}← curated key metrics
    ├── compare_<a>_vs_<b>.txt      ← side-by-side
    ├── details_<tag>.txt           ← rocprof-compute analyze section dump
    ├── stall_hotspots_<tag>.txt    ← per-line stall aggregation (from ATT / PC-sampling)
    ├── timeline_plots.txt          ← ASCII time-series (from plot_timeline.py); one file per run, all tags concatenated
    └── raw_<tag>.csv               ← optional: cleaned PMC csv export
```

Notes:

- `<tag>` is the per-workload / per-dispatch-path label, e.g. `path_a_shapeA`, `path_b_shapeB`. Pick tags that are short and name the representative workload, not the file UUID.
- If you profile only one tag, you can omit the tag suffix from filenames. But as soon as you profile a second, backfill the tag to avoid ambiguity.
- **Do not copy the helper scripts per-run.** Invoke them from `$SKILL/helpers/` (where `$SKILL` points at the installed skill root) and pass `--run-dir $PROFILE_RUN_DIR`. The scripts are stateless w.r.t. the script path; only the data under `analysis/` is per-run. If you genuinely need an archival snapshot of the helper code, copy it once at promotion time, not on every run.
- **rocprof-compute writes a partly flat directory.** The merged `pmc_perf.csv`, `pmc_kernel_top.csv`, `sysinfo.csv`, `roofline.pdf`, and `profiling_config.yaml` land directly under `<path>/`; the raw per-PMC-group CSVs land under `<path>/out/pmc_<N>/<hostname>/<pid>_*.csv`. When `-p <path>` is passed, this is rooted at `<path>/`; when omitted, output defaults to `./workloads/<name>/` (no `<gpu_model>` subdir on current rocprof-compute). Keep the whole tree — the helpers and the GUI walk it together.
- **rocprofv3** in ROCm 7+ defaults to a SQLite `.db` (the `rocpd` schema) plus CSVs. In ROCm 6.x it defaulted to CSVs only. Keep whatever rocprofv3 produced — pandas + sqlite3 handle both.

---

## Comparing two runs

For A/B comparisons (optimization-before vs after, or two dispatch variants on the same build), create a comparison run that *references* both underlying runs:

```
profile/<kernel>_v2_vs_v1/
├── REPORT.md                       ← describes both runs + the comparison
└── analysis/
    ├── compare.py                  ← loads reports from the two runs below
    ├── compare_key_metrics.txt     ← side-by-side on key metrics
    └── compare_stalls.txt          ← side-by-side on stall breakdown
    (No rocprof outputs — they live in the referenced runs)
```

In `compare.py`, hardcode the paths to both referenced runs:
```python
V1_DIR = Path("/abs/path/to/profile/<kernel>_v1_baseline")
V2_DIR = Path("/abs/path/to/profile/<kernel>_v2_optimized")
```

The comparison run does not re-profile; it only produces comparison artifacts and prose.

---

## What does NOT go in a run directory

- `.rpd.old` / `.db.old` backup files — if you need a prior version, you should have made it a separate run.
- Temporary scratch files — `/tmp` is for those.
- The dataset / workload files themselves — these belong in a shared dataset dir (e.g. `/home/<user>/dataset/flashinfer-trace/`). Reference them by absolute path in scripts.
- Compiler intermediates (`*.o`, `*.d`, `.hip.bc`). Put them under `harness/build/` or just rely on rebuilding from source.
- rocprof-compute caches — delete these after profiling; regenerable from raw CSVs.

Add a simple `.gitignore` inside `profile/` if you want to keep the run dirs out of git:
```
profile/*/
!profile/README.md
```

Or, if you want a few canonical runs tracked in git, `.gitignore` only the data-heavy subdirs:
```
profile/*/reports/
profile/*/analysis/metrics_all_*.json
profile/*/analysis/raw_*.csv
profile/*/harness/*_harness          # binary only, keep the .hip
```

---

## Environment variable convention (optional but recommended)

Scripts and rocprof invocations should pick up the run directory from a single env var, so they're easy to redirect to different runs:

```bash
export PROFILE_RUN_DIR=/abs/path/to/profile/<kernel>_v1_baseline
export SKILL=~/.claude/skills/rocprof-report-skill   # or wherever the skill is installed
mkdir -p "$PROFILE_RUN_DIR"/{harness,reports,analysis}

# build harness (MI300X gfx942; add --offload-arch=gfx950 for MI355X).
# `-DHARNESS_FILLED_IN=1` clears the template's #error guard.
hipcc -O3 -std=c++17 -gline-tables-only \
      --offload-arch=gfx942 \
      -munsafe-fp-atomics \
      -DHARNESS_FILLED_IN=1 \
      harness.hip -o "$PROFILE_RUN_DIR/harness/kernel_harness"

# kernel-trace overview
rocprofv3 --kernel-trace --hip-trace --hsa-trace \
    --kernel-include-regex "my_kernel" \
    -d "$PROFILE_RUN_DIR/reports/trace_<tag>" \
    -- "$PROFILE_RUN_DIR/harness/kernel_harness" [args]

# rocprof-compute section perf (analog of ncu --set full)
# Flag is -k / --kernel (substring match), not --kernel-name.
# Roofline is ON by default; pass `--no-roof` to skip. There is NO `--roofline` flag.
rocprof-compute profile -n <run_name>_<tag> \
    -k "my_kernel" \
    -p "$PROFILE_RUN_DIR/reports/rpc_<tag>" \
    -- "$PROFILE_RUN_DIR/harness/kernel_harness" [args]

# (Optional) timeseries pass — required for Dimension 5 (CU timeline) /
# Pattern M (tail effect). Lands a separate `pmc_perf_timeseries.csv` under
# the rpc_ts_<tag> directory; plot_timeline.py / Dimension 5 consume it.
rocprof-compute profile -n <run_name>_<tag>_ts \
    -k "my_kernel" --timeseries-sampling-rate 1ms \
    -p "$PROFILE_RUN_DIR/reports/rpc_ts_<tag>" \
    -- "$PROFILE_RUN_DIR/harness/kernel_harness" [args]

# ATT / per-line source attribution (analog of ncu --set source)
rocprofv3 --att --att-target-cu 0 \
    --kernel-include-regex "my_kernel" \
    -d "$PROFILE_RUN_DIR/reports/att_<tag>" \
    -- "$PROFILE_RUN_DIR/harness/kernel_harness" [args]

# parse — --rpc and --tag are required, once per report dir.
# --kernel-trace is optional; analyze_reports.py auto-resolves the rocprofv3
# nested path (trace_<tag>/**/*_kernel_trace.csv) when omitted.
# Pass `--arch` explicitly (gfx942 / gfx950); the script defaults to gfx942.
python3 "$SKILL/helpers/analyze_reports.py" --run-dir "$PROFILE_RUN_DIR" \
    --rpc "$PROFILE_RUN_DIR/reports/rpc_<tag>" --tag <tag> \
    --kernel "my_kernel" --arch gfx942
```

All of the helper scripts in `../helpers/` accept an explicit `--run-dir` (or equivalent path) argument. Pass it explicitly rather than relying on cwd — the scripts assume the standard run layout above and need to find both `reports/` and `analysis/` relative to that root.

---

## Checklist before starting a profile run

0. **Export both env vars first** — every later step (and every helper script under `$SKILL/helpers/`) depends on them:
   ```bash
   export PROFILE_RUN_DIR="$PWD/profile/<new_run_name>"
   export SKILL=~/.claude/skills/rocprof-report-skill   # or <repo>/.claude/skills/rocprof-report-skill
   ```
1. `mkdir -p "$PROFILE_RUN_DIR"/{harness,reports,analysis}` — make the three subdirs up front.
2. Copy or write the harness source into `$PROFILE_RUN_DIR/harness/`.
3. Compile into the same dir with `-gline-tables-only` (or `-g`) and the correct `--offload-arch`.
4. Run rocprofv3 / rocprof-compute with `-d` / `-p` pointing under `$PROFILE_RUN_DIR/reports/`.
5. Run the parser helpers from `$SKILL/helpers/` (e.g. `python3 "$SKILL/helpers/analyze_reports.py" --run-dir "$PROFILE_RUN_DIR" ...`); their output lands under `$PROFILE_RUN_DIR/analysis/`.
6. Write `REPORT.md` at `$PROFILE_RUN_DIR/REPORT.md`.
7. Before starting a *new* run, re-export `$PROFILE_RUN_DIR` with a new name and go back to step 1 — never write into an existing run dir.
