# Helpers

Reusable code for AMD profiling harnesses and rocprof / rocprof-compute report analysis. See `../SKILL.md` for context.

## C++ / HIP

| File | Purpose |
|---|---|
| `harness_template.hip` | Starting point for an AMD profiling harness. Copy into your run dir, fill in the `TODO(you)` sections. |
| `safetensors_loader.h` | Header-only safetensors reader (no external deps). Vendor-neutral — works on NVIDIA / AMD / CPU. |

### Typical harness setup

```bash
export PROFILE_RUN_DIR=/abs/path/to/profile/<run_name>
export SKILL=~/.claude/skills/rocprof-report-skill   # or wherever installed
cp "$SKILL/helpers/harness_template.hip" "$PROFILE_RUN_DIR/harness/my_kernel_harness.hip"
cp "$SKILL/helpers/safetensors_loader.h" "$PROFILE_RUN_DIR/harness/"
# edit my_kernel_harness.hip to include your kernel + fill in main();
# the template ships with a `#error HARNESS_FILLED_IN` guard — pass
# -DHARNESS_FILLED_IN=1 once you've replaced its TODOs.
hipcc -O3 -std=c++17 -gline-tables-only \
      --offload-arch=gfx942 -munsafe-fp-atomics \
      -DHARNESS_FILLED_IN=1 \
      "$PROFILE_RUN_DIR/harness/my_kernel_harness.hip" \
      -o "$PROFILE_RUN_DIR/harness/my_kernel_harness"
```

For MI355X (gfx950), use `--offload-arch=gfx950` and ROCm 7+.

## Python

| File | Purpose |
|---|---|
| `rocprof_utils.py` | Shared helpers: `load_rpc_dir`, `safe_col`, `key_counters_for_arch`, `dump_all_counters`, `MI300X_KEY_COUNTERS`, `MI355X_KEY_COUNTERS`, `per_kernel_durations_from_db`, ... |
| `analyze_reports.py` | Extract key counters + side-by-side comparison from one or more rocprof-compute output dirs |
| `extract_stall_hotspots.py` | Aggregate PC-sampling CSV (stochastic preferred — has the `Stall_Reason` column; host_trap = hotspots only) or ATT JSON → per-source-line rankings (requires `-gline-tables-only` for `file:line` attribution) |
| `plot_timeline.py` | ASCII plot rocprof-compute timeseries CSV / per-CU distribution (reveals tail effect, pipeline bubbles, workgroup imbalance) |
| `list_flashinfer_workloads.py` | Browse a flashinfer-trace dataset: show axes, histogram workload shapes, print safetensors paths for specific UUIDs |

### Typical Python workflow

```bash
export PROFILE_RUN_DIR="$PWD/profile/<run_name>"   # absolute path
export SKILL=~/.claude/skills/rocprof-report-skill   # or wherever the skill is installed
export FIB_DATASET_PATH=/path/to/flashinfer-trace    # if using FIB workloads

# (Optional) Browse workload shapes for a flashinfer-trace dataset
python3 $SKILL/helpers/list_flashinfer_workloads.py --definition <def_name>
python3 $SKILL/helpers/list_flashinfer_workloads.py --definition <def_name> --unique-axes <axis1>,<axis2> --no-paths

# Extract key counters for each rocprof-compute report
python3 $SKILL/helpers/analyze_reports.py --run-dir $PROFILE_RUN_DIR \
    --rpc $PROFILE_RUN_DIR/reports/rpc_<tag1> --tag <tag1> \
    --rpc $PROFILE_RUN_DIR/reports/rpc_<tag2> --tag <tag2> \
    --kernel "my_kernel_regex" --arch gfx942

# Per-line stall hotspots from PC sampling — prefer --pcsamp-dir; it rglobs the
# rocprofv3 default layout (pcsamp_<tag>/<hostname>/<pid>_pc_sampling_{stochastic,host_trap}.csv),
# the explicit flat form (pcsamp_<tag>/<prefix>_pc_sampling_*.csv when you passed
# `--output-file <prefix>` to rocprofv3), and the older rocprof-compute-wrapped
# layout (out/pmc_<N>/<hostname>/...) as a defensive fallback, so you don't have
# to know the hostname or PID. Use
# --pcsamp <file> only when you need to pin a specific CSV. The helper prefers
# the stochastic CSV when both modes are present — it's the only mode that
# populates `Stall_Reason`.
python3 $SKILL/helpers/extract_stall_hotspots.py --run-dir $PROFILE_RUN_DIR \
    --pcsamp-dir $PROFILE_RUN_DIR/reports/pcsamp_<tag1> --tag <tag1> \
    --pcsamp-dir $PROFILE_RUN_DIR/reports/pcsamp_<tag2> --tag <tag2>

# Or ATT-based hotspots
python3 $SKILL/helpers/extract_stall_hotspots.py --run-dir $PROFILE_RUN_DIR \
    --att-dir $PROFILE_RUN_DIR/reports/att_<tag> --tag <tag>

# ASCII timeline plots — the --timeseries path expects a single CSV that current
# rocprof-compute does NOT emit (--timeseries-sampling-rate is not a real flag).
# Until the helper is updated to read the rocprofv3 -P window layout, prefer the
# --per-cu path on the Recipe-2 pmc_perf.csv.

# Per-CU distribution (no timeseries needed)
python3 $SKILL/helpers/plot_timeline.py --run-dir $PROFILE_RUN_DIR \
    --rpc $PROFILE_RUN_DIR/reports/rpc_<tag> --tag <tag> --per-cu
```

The `analyze_reports.py`, `extract_stall_hotspots.py`, and `plot_timeline.py` helpers
all take `--run-dir` and write under `<run-dir>/analysis/`. `list_flashinfer_workloads.py`
is a dataset browser and only prints to stdout; it does not write to `<run-dir>/`.

### Dependencies

```bash
python3 -m pip install --user pandas
# Optional (ROCm 7+ rocpd Python helper):
#   /opt/rocm-7.0.0/share/rocprofiler-sdk/python/rocpd
#   (rocprof_utils.py falls back to plain sqlite3 if rocpd isn't importable)
```
