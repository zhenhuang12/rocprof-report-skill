# Helpers

Reusable code for AMD profiling harnesses and rocprof / rocprof-compute report analysis. See `../SKILL.md` for context.

## C++ / HIP

| File | Purpose |
|---|---|
| `harness_template.hip` | Starting point for an AMD profiling harness. Copy into your run dir, fill in the `TODO(you)` sections. |
| `safetensors_loader.h` | Header-only safetensors reader (no external deps). Vendor-neutral — works on NVIDIA / AMD / CPU. |

### Typical harness setup

```bash
cd profile/<run_name>/harness/
cp /path/to/skills/rocprof-report-skill/helpers/harness_template.hip my_kernel_harness.hip
cp /path/to/skills/rocprof-report-skill/helpers/safetensors_loader.h .
# edit my_kernel_harness.hip to include your kernel + fill in main()
hipcc -O3 -std=c++17 -gline-tables-only \
      --offload-arch=gfx942 -munsafe-fp-atomics \
      my_kernel_harness.hip -o my_kernel_harness
```

For MI355X (gfx950), use `--offload-arch=gfx950` and ROCm 7+.

## Python

| File | Purpose |
|---|---|
| `rocprof_utils.py` | Shared helpers: `load_rpc_dir`, `safe_col`, `key_counters_for_arch`, `dump_all_counters`, `MI300X_KEY_COUNTERS`, `MI355X_KEY_COUNTERS`, `per_kernel_durations_from_db`, ... |
| `analyze_reports.py` | Extract key counters + side-by-side comparison from one or more rocprof-compute output dirs |
| `extract_stall_hotspots.py` | Aggregate PC-sampling CSV (or ATT JSON) → per-source-line rankings by Wait_Reason (requires `-gline-tables-only`) |
| `plot_timeline.py` | ASCII plot rocprof-compute timeseries CSV / per-CU distribution (reveals tail effect, pipeline bubbles, workgroup imbalance) |
| `list_flashinfer_workloads.py` | Browse a flashinfer-trace dataset: show axes, histogram workload shapes, print safetensors paths for specific UUIDs |

### Typical Python workflow

```bash
export PROFILE_RUN_DIR=profile/<run_name>
HELPERS=/path/to/skills/rocprof-report-skill/helpers
export FIB_DATASET_PATH=/path/to/flashinfer-trace  # if using FIB workloads

# (Optional) Browse workload shapes for a flashinfer-trace dataset
python3 $HELPERS/list_flashinfer_workloads.py --definition <def_name>
python3 $HELPERS/list_flashinfer_workloads.py --definition <def_name> --unique-axes <axis1>,<axis2> --no-paths

# Extract key counters for each rocprof-compute report
python3 $HELPERS/analyze_reports.py --run-dir $PROFILE_RUN_DIR \
    --rpc $PROFILE_RUN_DIR/reports/rpc_<tag1> --tag <tag1> \
    --rpc $PROFILE_RUN_DIR/reports/rpc_<tag2> --tag <tag2> \
    --kernel "my_kernel_regex" --arch gfx942

# Per-line stall hotspots from PC sampling — prefer --pcsamp-dir; it globs the
# rocprofv3 nested layout (pcsamp_<tag>/pmc_1/<host>/<pid>_pc_sampling_*.csv)
# so you don't have to know the host or PID. Use --pcsamp <file> only when you
# need to pin a specific CSV.
python3 $HELPERS/extract_stall_hotspots.py --run-dir $PROFILE_RUN_DIR \
    --pcsamp-dir $PROFILE_RUN_DIR/reports/pcsamp_<tag1> --tag <tag1> \
    --pcsamp-dir $PROFILE_RUN_DIR/reports/pcsamp_<tag2> --tag <tag2>

# Or ATT-based hotspots
python3 $HELPERS/extract_stall_hotspots.py --run-dir $PROFILE_RUN_DIR \
    --att-dir $PROFILE_RUN_DIR/reports/att_<tag> --tag <tag>

# ASCII timeline plots — needs --timeseries-sampling-rate at collection time
python3 $HELPERS/plot_timeline.py --run-dir $PROFILE_RUN_DIR \
    --timeseries $PROFILE_RUN_DIR/reports/rpc_ts_<tag>/pmc_perf_timeseries.csv --tag <tag>

# Per-CU distribution (no timeseries needed)
python3 $HELPERS/plot_timeline.py --run-dir $PROFILE_RUN_DIR \
    --rpc $PROFILE_RUN_DIR/reports/rpc_<tag> --tag <tag> --per-cu
```

All scripts take `--run-dir` and write under `<run-dir>/analysis/`.

### Dependencies

```bash
python3 -m pip install --user pandas
# Optional (ROCm 7+ rocpd Python helper):
#   /opt/rocm-7.0.0/share/rocprofiler-sdk/python/rocpd
#   (rocprof_utils.py falls back to plain sqlite3 if rocpd isn't importable)
```
