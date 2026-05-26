# Profile Collection Commands

This document lists the exact `rocprofv3` and `rocprof-compute` commands you should run, in what order, and what each flag does.

---

## Prerequisites recap

- `-gline-tables-only` (or `-g`) in the compile flags (see `02-harness-guide.md`).
- `rocprofv3` (ROCm 6.2+) and `rocprof-compute` (ROCm 6.3+, formerly Omniperf) on PATH.
- User in `render` group (and `video` on some distros); ATT and PC sampling may additionally need `CAP_PERFMON` or `kfd_admin_group` — check `getfacl /dev/kfd`.
- Kernel name known (check with `llvm-objdump --syms --demangle <code-object>` if unsure — see `02-harness-guide.md`).

Quick smoke test:
```bash
rocprofv3 --kernel-trace \
    --kernel-include-regex "YOUR_KERNEL_NAME" \
    -d /tmp/rocprof_smoke \
    -- ./harness [args]
ls /tmp/rocprof_smoke   # should contain at least *_kernel_trace.csv (and *.db on ROCm 7+)
```

If `kernel_trace.csv` is empty, the regex didn't match — verify with `llvm-objdump --syms --demangle`. If you get a permission error, see `09-common-issues.md`.

---

## Recipe 1: Kernel-trace overview (first pass)

Cheapest pass — runtime APIs, HSA dispatch records, kernel begin/end timestamps. No PMC overhead. Use this to confirm the regex matches and to get a wall-time map.

```bash
rocprofv3 --kernel-trace --hip-trace --hsa-trace \
    --kernel-include-regex "KERNEL_REGEX" \
    -d $PROFILE_RUN_DIR/reports/trace_<tag> \
    -- ./harness [args]
```

| Flag | Meaning |
|---|---|
| `--kernel-trace` | Record kernel dispatch records (name, grid, block, agent, queue, duration). |
| `--hip-trace` | Record HIP runtime API calls. |
| `--hsa-trace` | Record HSA runtime API calls (lower-level, useful when you want to see queue / signal activity). |
| `--kernel-include-regex` | Only trace kernels whose demangled name matches. Reduces output volume. Itanium-ABI demangled symbol — same as `llvm-objdump --syms --demangle`. |
| `-d` | Output directory (rocprofv3 writes multiple files inside). |
| `--` | Separates rocprofv3 options from the command to launch. |

Output: `*_kernel_trace.csv`, `*_hip_api_trace.csv`, `*_hsa_api_trace.csv` (ROCm 6.x) or a single `*.db` SQLite (ROCm 7+ default, the `rocpd` schema). Add `--output-format pftrace` for a Perfetto trace, or `--output-format json` for the rocprofv3 JSON.

Replay count: 1 pass. Wall time: kernel duration + ~tens of ms init.

---

## Recipe 2: Section-based perf metrics (second pass) — analog of `ncu --set full`

This is the bread-and-butter run. `rocprof-compute profile` collects the PMC groups that back rocprof-compute's ~24 perf sections (2.1.0 launch, 2.1.1 SoL, 2.1.2 Wavefronts, 2.1.10 Compute, 2.1.15 Memory, …) and the roofline model.

```bash
rocprof-compute profile \
    -n <run_name>_<tag> \
    --roofline \
    --kernel-name "KERNEL_REGEX" \
    -p $PROFILE_RUN_DIR/reports/rpc_<tag> \
    -- ./harness [args]
```

| Flag | Meaning |
|---|---|
| `-n` / `--name` | Workload name used in the output directory + report titles. |
| `--roofline` | Also run the empirical roofline benchmarks (peak HBM BW, peak FLOPS, …) so the section reports can place this kernel on the roofline plot. Skips if you already have a cached roofline for this GPU. |
| `--kernel-name` | Filter (substring / regex depending on rocprof-compute version) on demangled kernel names. Limits the kernels measured per PMC group. |
| `-p` / `--path` | Output directory; rocprof-compute creates a `SoC/` subdir, `pmc_perf.csv`, `timestamps.csv`, `sysinfo.csv`, etc. |

Replay count: ~15-30 passes (one per PMC group; rocprofv3 replays the whole binary, not just the kernel). Wall time: kernel time × number of groups + init.

After collection, render the section reports:

```bash
# Print all sections to stdout (use for diff & archive)
rocprof-compute analyze -p $PROFILE_RUN_DIR/reports/rpc_<tag> > \
    $PROFILE_RUN_DIR/analysis/details_<tag>.txt

# A single section (e.g., 2.1.15 Memory Workload)
rocprof-compute analyze -p $PROFILE_RUN_DIR/reports/rpc_<tag> -b 15

# List all sections
rocprof-compute analyze -p $PROFILE_RUN_DIR/reports/rpc_<tag> --list-stats
```

**Always read `details_<tag>.txt` first.** Each section has a "Speed-of-Light" line that names the bottleneck subsystem and a numeric gap to peak — this is the AMD analog of NCU's `Est. Speedup` rule.

For a browsable GUI, use the **ROCprof Compute Viewer** (the AMD analog of `ncu-ui`). RGP does *not* support CDNA / Instinct GPUs; do not try to open these reports with it.

---

## Recipe 3: Per-line stall sampling (third pass) — analog of `ncu --set source`

Two options. Prefer PC sampling (lower overhead) when available; fall back to ATT.

### 3a) PC sampling

```bash
rocprofv3 --pc-sampling-method host-trap \
    --pc-sampling-interval 1000 \
    --pc-sampling-unit cycles \
    --kernel-include-regex "KERNEL_REGEX" \
    -d $PROFILE_RUN_DIR/reports/pcsamp_<tag> \
    -- ./harness [args]
```

| Flag | Meaning |
|---|---|
| `--pc-sampling-method` | `host-trap` (works on MI200+) is the most portable; `stochastic` is lower-overhead on MI300+ if your ROCm build enables it. |
| `--pc-sampling-interval` | Sample every N cycles (or instructions, depending on `--pc-sampling-unit`). |
| `--pc-sampling-unit` | `cycles` or `instructions`. |
| `--kernel-include-regex` | Limits sampling to matching kernels. |

Output: per-kernel CSV with `Instruction_Address`, `Source` (file:line, populated only when compiled with `-gline-tables-only`/`-g`), `Instruction_Comment` (the SASS-equivalent text on AMD: the ISA mnemonic), `Wait_Reason`, `Sample_Count`.

Use `extract_stall_hotspots.py` to aggregate these by `(file, line)` and by wait reason.

### 3b) ATT (Advanced Thread Trace / SQTT)

Heavier — captures every wave's instruction stream on the targeted CU(s). Default capture is **1 kernel × 1 CU per SE**, so plan accordingly.

```bash
rocprofv3 --att \
    --att-target-cu 0 \
    --att-buffer-size 0x10000000 \
    --att-shader-engine-mask 0xF \
    --kernel-include-regex "KERNEL_REGEX" \
    -d $PROFILE_RUN_DIR/reports/att_<tag> \
    -- ./harness [args]
```

| Flag | Meaning |
|---|---|
| `--att` | Enable Advanced Thread Trace. |
| `--att-target-cu N` | Capture this CU index (within each enabled SE). |
| `--att-shader-engine-mask` | Bitmask of SEs to enable. `0xF` = first 4 SEs. |
| `--att-buffer-size` | Per-SE trace buffer in bytes. Bump if traces are getting truncated. |

Output: per-SE JSON / binary traces; open with ROCprof Compute Viewer or process programmatically via the `att_tool` JSON. Source attribution requires `-gline-tables-only`/`-g`.

---

## Recipe 4: Targeted PMC only (fast)

If you already know which counters you want (e.g., re-running after a code change to check the fix), collect just those:

```bash
# Inline list
rocprofv3 --pmc SQ_WAVES,SQ_INSTS_VALU,SQ_INSTS_MFMA,SQ_WAIT_INST_VMEM,TCP_TCC_READ_REQ_sum,TCC_EA0_RDREQ_sum,GRBM_GUI_ACTIVE \
    --kernel-include-regex "KERNEL_REGEX" \
    -d $PROFILE_RUN_DIR/reports/pmc_<tag> \
    -- ./harness [args]

# Or a YAML/JSON job file (preferred for reproducibility)
cat > /tmp/pmc.yaml <<'EOF'
jobs:
  - name: my_targeted
    pmc:
      - SQ_WAVES
      - SQ_INSTS_VALU
      - SQ_INSTS_MFMA
      - SQ_WAIT_INST_VMEM
      - SQ_WAIT_INST_LDS
      - SQ_LDS_BANK_CONFLICT
      - TCC_EA0_RDREQ_sum
      - TCC_EA0_RDREQ_32B_sum
      - TCC_EA1_RDREQ_sum
      - GRBM_GUI_ACTIVE
    kernel_include_regex: "KERNEL_REGEX"
EOF
rocprofv3 -i /tmp/pmc.yaml -d $PROFILE_RUN_DIR/reports/pmc_<tag> -- ./harness [args]
```

rocprofv3 automatically splits the PMC list into groups that fit the hardware counter budget and replays the binary once per group. Counters that share a unit (SQ_*, TCP_*, TCC_*) often share a group; rocprofv3 reports the grouping in its log.

The full list of available counters: `rocprofv3 --list-metrics` (or `--list-counters` on some builds).

---

## Recipe 5: A/B comparison (before vs after optimization)

```bash
# Before
rocprof-compute profile -n v1 --kernel-name "my_kernel" \
    -p $PROFILE_RUN_DIR/reports/rpc_v1 -- ./harness_v1 [args]

# After
rocprof-compute profile -n v2 --kernel-name "my_kernel" \
    -p $PROFILE_RUN_DIR/reports/rpc_v2 -- ./harness_v2 [args]

# Side-by-side from the CLI
rocprof-compute analyze \
    -p $PROFILE_RUN_DIR/reports/rpc_v1 \
    -p $PROFILE_RUN_DIR/reports/rpc_v2 \
    > $PROFILE_RUN_DIR/analysis/compare_v1_vs_v2.txt
```

Or in Python (see `04-python-api.md`):

```python
import pandas as pd
d1 = pd.read_csv("$PROFILE_RUN_DIR/reports/rpc_v1/timestamps.csv")
d2 = pd.read_csv("$PROFILE_RUN_DIR/reports/rpc_v2/timestamps.csv")
t1 = (d1["End_Timestamp"] - d1["Start_Timestamp"]).sum()
t2 = (d2["End_Timestamp"] - d2["Start_Timestamp"]).sum()
print(f"Speedup: {t1/t2:.2f}x")
```

---

## What rocprof-compute sections cover

```bash
rocprof-compute analyze --list-stats
```

Section ID map (MI300X, rocprof-compute ROCm 7.x — may shift slightly between releases):

| ID | Section | What it tells you |
|---|---|---|
| 2.1.0  | Launch Statistics             | Grid/block, waves/CU, registers/wave, scratch, LDS |
| 2.1.1  | Speed-of-Light (SoL)          | % of peak for compute, HBM, vL1, L2, scratch — the headline |
| 2.1.2  | Wavefront Launch Stats        | Waves/wkg, achieved occupancy, register/LDS pressure |
| 2.1.5  | Pipeline / instruction mix    | VALU vs MFMA vs VMEM vs LDS instruction counts |
| 2.1.10 | Compute Units — Compute Pipe  | VALU / SALU / Matrix-core busy %, IPC, FMA |
| 2.1.11 | Compute Units — Memory Pipe   | Bytes per wavefront, LDS bank conflicts |
| 2.1.13 | Wavefront Stall Reasons       | `WAIT_INST_VMEM`, `WAIT_INST_LDS`, `WAIT_ANY_LDS`, `WAIT_BARRIER`, `WAIT_INST_SCA` |
| 2.1.15 | Memory — vL1 cache (TCP)      | Hit rate, sectors per request, coalescing |
| 2.1.16 | Memory — L2 cache (TCC)       | Per-channel hit rate, atomics, bytes |
| 2.1.17 | Memory — HBM (TCC_EA)         | Per-channel HBM read/write bytes, achieved BW |
| 2.1.20 | Roofline                       | Compute vs memory roofline, kernel position |
| 2.1.22 | Scratch / Spill                | Scratch reads/writes (= register spill on AMD) |
| 2.1.23 | Workgroup imbalance           | Per-CU active-cycle distribution |

(Run `--list-stats` on your actual install to confirm the IDs.)

---

## Profiling multiple kernel launches

If the kernel is called multiple times and you want different iterations:

```bash
# Limit to the first N matches via the include-regex (rocprofv3 doesn't have a direct
# "skip" flag like ncu -s); instead, use --kernel-iteration-range:
rocprofv3 --kernel-trace \
    --kernel-include-regex "my_kernel" \
    --kernel-iteration-range "[6:9]" \
    -d $PROFILE_RUN_DIR/reports/trace_iter6_8 -- ./harness
```

`--kernel-iteration-range "[N:M]"` selects the Nth through Mth match (zero-indexed, half-open on some builds — check `rocprofv3 --help`). Useful for ignoring warmup or focusing on a steady-state iteration.

---

## GPU clock / power state (for reproducibility)

```bash
# Show current clocks and SCLK/MCLK states
rocm-smi --showclocks
rocm-smi --showperflevel

# Lock to a deterministic perf level (sudo required)
sudo rocm-smi --setperflevel high      # or "manual" + setsclk/setmclk

# Restore
sudo rocm-smi --resetclocks
```

For MI300X / MI355X the GPU normally reaches steady-state during rocprof-compute replays because the binary is rerun once per PMC group. If results jitter between runs, lock to `high` perf level. Note that on MI300A APU SKUs (gfx942 APU variant) thermal headroom is shared with CPU cores; rocm-smi may report throttling even at "high".

---

## Gotchas

- **`rocprof-compute profile` wall time blows up**: that's normal — each PMC group replays the whole binary. Profile a smaller representative workload if the kernel is expensive.
- **`--kernel-include-regex` matches nothing**: check with `llvm-objdump --syms --demangle <code-object>` and make sure you're matching the demangled name. Templates produce names like `my_kernel<8, 256>(...)`.
- **Output dir is empty / 0 KB**: profile terminated before the kernel launched. Usually means the regex didn't match, or the harness crashed.
- **ATT JSON is empty or "Source" column blank**: rebuild with `-gline-tables-only` (or `-g`), and confirm symbols weren't stripped at link time.
- **PC sampling silently no-ops**: not all GPU/driver/ROCm combinations expose PC sampling. Try `--pc-sampling-method host-trap` first (most portable) before `stochastic`.
- **`rocprofv3` replays the *application*, not just the kernel**: any expensive host-side init (data loading, NCCL init, hipBLASLt warmup) is paid on every replay. Move it outside the profile window — or just shrink the harness, which is the whole point.
- **MI355X support requires ROCm 7+**. ROCm 6.x will refuse `--offload-arch=gfx950` and may also refuse to enumerate gfx950 counters in `--list-metrics`.
