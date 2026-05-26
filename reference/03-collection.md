# Profile Collection Commands

This document lists the exact `rocprofv3` and `rocprof-compute` commands you should run, in what order, and what each flag does.

---

## Prerequisites recap

- `-gline-tables-only` (or `-g`) in the compile flags (see `02-harness-guide.md`).
- `rocprofv3` (ROCm 6.2+) and `rocprof-compute` (ROCm 6.3+, formerly Omniperf) on PATH.
- User in `render` group (and `video` on some distros); ATT and PC sampling may additionally need `CAP_PERFMON` or `kfd_admin_group` — check `getfacl /dev/kfd`.
- Kernel name known (check with `llvm-objdump --syms --demangle <code-object>` if unsure — see `02-harness-guide.md`).
- **`$PROFILE_RUN_DIR` exported** (Phase 0 in [`01-workflow.md`](01-workflow.md) and the Quickstart in [`../SKILL.md`](../SKILL.md)). Every recipe below writes to `$PROFILE_RUN_DIR/reports/...`. Prefix any shell you run a recipe in with the fail-loud guards so a missing var stops you instead of writing into `/reports/...`:

```bash
: "${PROFILE_RUN_DIR:?run Phase 0 first — PROFILE_RUN_DIR is unset}"
: "${SKILL:?export SKILL=... to your skill install path}"
```

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
    -f csv \
    -d $PROFILE_RUN_DIR/reports/trace_<tag> \
    -- ./harness [args]
```

| Flag | Meaning |
|---|---|
| `--kernel-trace` | Record kernel dispatch records (name, grid, block, agent, queue, duration). |
| `--hip-trace` | Record HIP runtime API calls. |
| `--hsa-trace` | Record HSA runtime API calls (lower-level, useful when you want to see queue / signal activity). |
| `--kernel-include-regex` | Only trace kernels whose demangled name matches. Reduces output volume. Itanium-ABI demangled symbol — same as `llvm-objdump --syms --demangle`. |
| `-d` | Output directory. rocprofv3's default file template is `<hostname>/<pid>_<file>.csv` — `<hostname>` is a subdirectory under `-d`, `<pid>_` is a filename prefix (not a directory). Pass `--output-file <prefix>` to collapse to a flat `<prefix>_<file>.csv` directly under `-d`. |
| `-f` / `--output-format` | `{csv, json, pftrace, otf2, rocpd}`. Pass `csv` to get the human-/pandas-friendly CSVs; the default depends on the build. |
| `--` | Separates rocprofv3 options from the command to launch. |

Output: `*_kernel_trace.csv`, `*_hip_api_trace.csv`, `*_hsa_api_trace.csv` when `-f csv`; with `-f rocpd` you get a single `.db` SQLite (the `rocpd` schema, default on ROCm 7+). Add `-f pftrace` for a Perfetto trace, or `-f json` for the rocprofv3 JSON.

Replay count: 1 pass. Wall time: kernel duration + ~tens of ms init.

---

## Recipe 2: Section-based perf metrics (second pass) — analog of `ncu --set full`

This is the bread-and-butter run. `rocprof-compute profile` collects the PMC groups that back rocprof-compute's ~24 perf sections (block `2` SoL, `5` CS / wavefront, `7` wavefront launch, `10` compute pipe, `11` instruction mix, `15` L1D, `16` L2 cache, `17` L2-fabric / HBM, `18` scratch / spill — see `--list-metrics gfx942` for the full list; older docs use dotted IDs like `2.1.10` for the same blocks) and the roofline model.

```bash
rocprof-compute profile \
    -n <run_name>_<tag> \
    -k "KERNEL_SUBSTRING" \
    -p $PROFILE_RUN_DIR/reports/rpc_<tag> \
    -- ./harness [args]
```

> **Roofline is ON by default** in current rocprof-compute (7.x). Pass `--no-roof` to skip the empirical roofline benchmarks; pass `--roof-only` to run them without the regular PMC pass. **There is no `--roofline` flag** — invoking it crashes. Don't combine `--no-roof` with `--roof-only` (one suppresses what the other runs); `--no-roof` *is* compatible with `-b` and `-k`.

| Flag | Meaning |
|---|---|
| `-n` / `--name` | Workload name used in report titles and folded into the output path (see `-p` below). |
| `--no-roof` | Skip the empirical roofline benchmarks (otherwise they run by default and add ~30 s). |
| `-k` / `--kernel` | Filter on demangled kernel names (substring, accepts multiple values). Limits the kernels measured per PMC group. (Note: the flag is `--kernel`, not `--kernel-name`.) |
| `-p` / `--path` | Output root. With an explicit `-p` and the default `--subpath gpu`, rocprof-compute writes **flat directly under `-p`**: `pmc_perf.csv`, `timestamps.csv`, `sysinfo.csv`, `log.txt`, `roofline.csv` (when roofline ran), PDF plots named `empirRoof_gpu-0_<datatypes>.pdf` (only with `--roof-only` or `--kernel-names`), `profiling_config.yaml`, plus `perfmon/<group>.{txt,yaml}` and `out/pmc_<N>/<hostname>/<pid>_{kernel_trace,counter_collection,agent_info}.csv` raw-per-pass subdirs all land directly there. The default `--subpath` value `"gpu"` matches neither nesting branch in `rocprof_compute_base.py`; only `--subpath gpu_model` adds a `<gpu_model>/` child, and `--subpath node_name` adds a `<hostname>/` child. When `-p` is **omitted** (resolved value equals the argparser default `<cwd>/workloads`), rocprof-compute auto-appends `<name>/<gpu_model>/`, giving `<cwd>/workloads/<name>/<gpu_model>/`. The helpers in `$SKILL/helpers/` accept either form — they first check `<-p>/pmc_perf.csv` and fall back to globbing `<-p>/*/pmc_perf.csv` for the opt-in nested layout. |
| `-b` / `--block` | In **analyze** mode (`rocprof-compute analyze -b`): pass a metric ID (e.g. `12.1.1`) or top-level section ID (e.g. `12`). No alias map — `-b lds` / `-b l1i` are not recognized; use `rocprof-compute analyze --list-metrics <gfx>` to find the right ID. In **profile** mode (`rocprof-compute profile -b`): the validator is `validate_block`, which accepts only the uppercase hardware-block names `{SQ, SQC, TA, TD, TCP, TCC, SPI, CPC, CPF}`. |

Replay count: ~15-30 passes (one per PMC group; rocprofv3 replays the whole binary, not just the kernel). Wall time: kernel time × number of groups + init.

After collection, render the section reports:

```bash
# Print all sections to stdout (use for diff & archive)
rocprof-compute analyze -p $PROFILE_RUN_DIR/reports/rpc_<tag> > \
    $PROFILE_RUN_DIR/analysis/details_<tag>.txt

# A single section (e.g., `-b 15` = L1D cache; older dotted form `2.1.15`)
rocprof-compute analyze -p $PROFILE_RUN_DIR/reports/rpc_<tag> -b 15

# List all kernels & dispatches (NOT section IDs)
rocprof-compute analyze -p $PROFILE_RUN_DIR/reports/rpc_<tag> --list-stats

# List all section / metric IDs for this arch (gfx942 for MI300X, gfx950 for MI355X)
rocprof-compute analyze --list-metrics gfx942
```

**Always read `details_<tag>.txt` first.** Each section has a "Speed-of-Light" line that names the bottleneck subsystem and a numeric gap to peak — this is the AMD analog of NCU's `Est. Speedup` rule.

For a browsable GUI, use the **ROCprof Compute Viewer** (the AMD analog of `ncu-ui`). RGP does *not* support CDNA / Instinct GPUs; do not try to open these reports with it.

---

## Recipe 2b: Windowed PMC collection (optional, for CU timeline / tail-effect analysis)

Recipe 2 averages PMCs over the whole kernel. To see the *shape* of utilization over time — pipeline bubbles, tail effect, ramp-up / ramp-down — repeatedly sample PMCs over a series of short windows and post-process the result. Dimension 5 (CU timeline), Pattern B (tail effect from variable-length inputs), and Pattern M (pipeline bubbles) in the diagnosis playbook all benefit from a windowed view. Skip this recipe if a per-kernel average is enough.

There is **no** `rocprof-compute profile --timeseries-sampling-rate` flag in current ROCm — verify with `rocprof-compute profile --help`. The supported primitive is `rocprofv3 -P/--collection-period <delay>:<dur>:<repeat>`, which collects N discrete profile windows that you stitch together.

```bash
# Collect 50 windows of 1 ms each (50 ms total wall coverage). Tune the second
# triplet field to your kernel's duration; the unit is set by --collection-period-unit.
rocprofv3 --pmc SQ_BUSY_CYCLES SQ_INSTS_VALU TCC_EA0_RDREQ_sum GRBM_GUI_ACTIVE \
    -P 0:1:50 --collection-period-unit msec \
    --kernel-include-regex "KERNEL_REGEX" -f csv \
    -d $PROFILE_RUN_DIR/reports/rpc_ts_<tag> \
    -- ./harness [args]
```

Each window writes its own `*_counter_collection.csv` under the output directory. Concatenate them (preserving window index as a synthetic time axis) before feeding to `plot_timeline.py`. For very short kernels (sub-ms) `-P` granularity is too coarse — fall back to ATT for per-instruction temporal detail (see Recipe 3b) or to the per-CU spatial view (`plot_timeline.py --per-cu`) on the static Recipe-2 `pmc_perf.csv`.

> **Heads-up:** at the time of writing, `plot_timeline.py --timeseries` still expects a single `pmc_perf_timeseries.csv`. Until the helper is updated to consume the `-P` window layout, the most reliable timeline signal comes from `--per-cu` on the Recipe-2 output or from ATT.

---

## Recipe 3: Per-line stall sampling (third pass) — analog of `ncu --set source`

Two options. Prefer PC sampling (lower overhead) when available; fall back to ATT.

### 3a) PC sampling

**Prefer `stochastic` mode** — it's the only PC-sampling mode that populates the `Stall_Reason` CSV column needed for a true wait-reason breakdown. The `host_trap` mode emits sampled PCs only (good for per-line hotspots, but no wait-reason classification). See AMD's docs: https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-pc-sampling.html

```bash
# Primary: stochastic mode (MI300+; required for the granular stall breakdown)
rocprofv3 --pc-sampling-beta-enabled \
    --pc-sampling-method stochastic \
    --pc-sampling-unit cycles \
    --pc-sampling-interval 1048576 \
    --kernel-include-regex "KERNEL_REGEX" \
    -f csv \
    -d $PROFILE_RUN_DIR/reports/pcsamp_<tag> \
    -- ./harness [args]
```

| Flag | Meaning |
|---|---|
| `--pc-sampling-beta-enabled` | **Required in ROCm 6.4+** — PC sampling is still a beta feature; sets `ROCPROFILER_PC_SAMPLING_BETA_ENABLED=1` internally. |
| `--pc-sampling-method` | `stochastic` (MI300+) is the only mode that populates `Stall_Reason`. `host_trap` (MI200+) is portable but gives PC hotspots only. **Note the underscore in `host_trap` — not `host-trap`.** |
| `--pc-sampling-interval` | Sample every N units (per `--pc-sampling-unit`). For `stochastic` + `cycles`, `1048576` (= 2^20) is a sensible default; for `host_trap` + `time`, units are **microseconds** (`1000` = 1 ms). |
| `--pc-sampling-unit` | **`stochastic`**: use `cycles` (canonical; the only unit shown in upstream examples). The SDK enum also has `ROCPROFILER_PC_SAMPLING_UNIT_INSTRUCTIONS`, but it may not be wired through the CLI in your build — verify first. `time` is runtime-rejected on `stochastic`. **`host_trap`** requires `time` (`cycles` / `instructions` are rejected at runtime as "PC sampling configuration is not supported"). |
| `--kernel-include-regex` | Limits sampling to matching kernels. |
| `-f` / `--output-format` | Format of output: `{csv, json, pftrace, otf2, rocpd}`. Use `csv` for downstream pandas parsing. |

rocprofv3 nests output under `<hostname>/` by default. With only `-d <dir>` set (no `--output-file`), the SDK's documented default file template is `<hostname>/<pid>_<file>` — `<hostname>` is a directory under `-d`, `<pid>_` is a filename prefix (NOT a `<pid>/` directory). So the CSVs land at:

```
pcsamp_<tag>/<hostname>/<pid>_pc_sampling_stochastic.csv   # stochastic: has Stall_Reason
pcsamp_<tag>/<hostname>/<pid>_pc_sampling_host_trap.csv    # host_trap: sampled PCs only, NO Stall_Reason
```

Pass `--output-file <prefix>` to override that default and collapse to a flat `<dir>/<prefix>_pc_sampling_*.csv`. The helper `_resolve_pcsamp_dir` `rglob`s for `*_pc_sampling_*.csv`, so either layout works.

Stochastic CSV columns (per AMD's PC-sampling docs): `Sample_Timestamp`, `Exec_Mask`, `Dispatch_Id`, `Instruction` (PC), `Instruction_Comment` (the ISA mnemonic — the SASS-equivalent text on AMD), `Correlation_Id`, `Wave_Issued_Instruction` (0 = stalled / 1 = productively issued), `Instruction_Type`, `Stall_Reason` (populated only when `Wave_Issued_Instruction == 0`), and `Wave_Count`. The `Stall_Reason` value is one of the `ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_*` enum values: `NONE`, `NO_INSTRUCTION_AVAILABLE`, `ALU_DEPENDENCY`, `WAITCNT`, `INTERNAL_INSTRUCTION`, `BARRIER_WAIT`, `ARBITER_NOT_WIN`, `ARBITER_WIN_EX_STALL`, `OTHER_WAIT`, `SLEEP_WAIT`. Source attribution (`file:line`) requires `-gline-tables-only`/`-g` on the build and is reconstructed from `Instruction` via `addr2line` against the binary.

> **Note — JSON-only per-pipe snapshot.** The per-execution-pipe `arb_state_stall_{valu, matrix, lds, lds_direct, scalar, vmem_tex, flat, exp, misc, brmsg}` and matching `arb_state_issue_*` bit-fields are **NOT** CSV columns. They live in the `snapshot` object of the JSON output only — use `-f json` instead of `-f csv` if you need them, and read them from each PC-sample record. They are 1-bit fields of `rocprofiler_pc_sampling_snapshot_v0_t` in `/opt/rocm/include/rocprofiler-sdk/pc_sampling.h`.

Host_trap CSV columns are a strict subset: `Sample_Timestamp`, `Exec_Mask`, `Dispatch_Id`, `Instruction`, `Instruction_Comment`, `Correlation_Id`. Use it only if you need cheap per-line hotspots and don't care about the wait-reason breakdown.

**PC-sampling method × unit compatibility:**

| `--pc-sampling-method` ↓ \ `--pc-sampling-unit` → | `cycles` | `instructions` | `time` (µs) |
|---|---|---|---|
| `stochastic` (MI300+) | ✅ canonical, populates `Stall_Reason` | ⚠️ SDK supports `ROCPROFILER_PC_SAMPLING_UNIT_INSTRUCTIONS` but the CLI / current builds may reject it — upstream PC-sampling examples use `cycles` only; verify before relying on it | ❌ runtime-rejected |
| `host_trap`           | ❌ runtime-rejected | ❌ runtime-rejected | ✅ supported, hotspots only (no `Stall_Reason`) |

Pick the row by data need (wait-reason vs hotspots-only), then the column by what's accepted on the row. **When in doubt for stochastic, use `cycles`** — that's the only unit demonstrated in the upstream `using-pc-sampling.html` examples.

**Host_trap alternative (cheaper, hotspots only):**

```bash
rocprofv3 --pc-sampling-beta-enabled \
    --pc-sampling-method host_trap \
    --pc-sampling-interval 1000 \
    --pc-sampling-unit time \
    --kernel-include-regex "KERNEL_REGEX" \
    -f csv \
    -d $PROFILE_RUN_DIR/reports/pcsamp_<tag> \
    -- ./harness [args]
```

If `stochastic` is rejected on your build with "PC sampling configuration is not supported" (older silicon, partition mode mismatch, or unsupported runtime), use `host_trap` instead — you'll get per-line hotspots but no `Stall_Reason` breakdown.

Use `extract_stall_hotspots.py` to aggregate stochastic samples by `(file, line)` and by stall reason; on a host_trap CSV the helper degrades gracefully to per-line hotspot ranking only.

### 3b) ATT (Advanced Thread Trace / SQTT)

Heavier — captures every wave's instruction stream on the targeted CU(s). Default capture is **1 kernel × 1 CU per SE**, so plan accordingly.

```bash
# --att-target-cu 0 picks the CU at index 0 within each enabled SE. There is
# no special meaning to either 0 or the default of 1 — both are just CU
# indices. The examples in this skill use 0 throughout for reproducibility;
# omit the flag to fall back to the rocprofv3 default (1).
rocprofv3 --att \
    --att-target-cu 0 \
    --att-buffer-size 0x6000000 \
    --att-shader-engine-mask 0x1 \
    --kernel-include-regex "KERNEL_REGEX" \
    -d $PROFILE_RUN_DIR/reports/att_<tag> \
    -- ./harness [args]
```

| Flag | Meaning |
|---|---|
| `--att` | Enable Advanced Thread Trace. |
| `--att-target-cu N` | Capture this CU index (within each enabled SE). Default is `1`; `0` is equally valid. Both are plain indices — there's no special-case meaning. To cover more CUs, run multiple invocations or script around it. |
| `--att-shader-engine-mask` | 32-bit bitmask of SEs to enable. **On MI3xx (gfx942/gfx950) each hex nibble selects SEs within one XCD** (MI300X has 8 XCDs × 4 SEs = 32 bits): `0x1` = one SE on XCD0 (conservative default — recommended starting point), `0x11111111` = one SE per XCD across all 8 XCDs (good coverage without overflow), `0xFFFFFFFF` = all 4 SEs on all XCDs (max coverage; upstream warns this risks dropped packets / buffer overflow). Bump cautiously and watch for truncation. |
| `--att-buffer-size` | Per-SE trace buffer in bytes. The upstream thread-trace docs cite a typical value of `0x6000000` (96 MB), with a supported range of 1 MB – 2 GB; the value above matches that. Bump it (e.g. `0x40000000` = 1 GB) only if traces report truncation. |

Output (after `rocprof-trace-decoder` runs automatically inside rocprofv3 — bundled in ROCm ≥ 7.13; install the `rocprof-trace-decoder` package or pass `--att-library-path <dir>` on older ROCm): `att_<tag>/stats_*.csv` (per-instruction latency / stall / idle summary), `att_<tag>/ui_output_agent_<id>_dispatch_<id>/*.json` (UI tree), raw `.att` SQTT binaries, and `.out` code-object copies. Open the `ui_output_*/` tree with **`rocprof-compute-viewer`**, or aggregate the JSONs programmatically via `rglob("att_<tag>/**/*.json")`. Source attribution requires `-gline-tables-only`/`-g`.

---

## Recipe 4: Targeted PMC only (fast)

If you already know which counters you want (e.g., re-running after a code change to check the fix), collect just those:

```bash
# Inline list
rocprofv3 --pmc SQ_WAVES,SQ_INSTS_VALU,SQ_INSTS_MFMA,SQ_WAIT_INST_ANY,TCP_TCC_READ_REQ_sum,TCC_EA0_RDREQ_sum,GRBM_GUI_ACTIVE \
    --kernel-include-regex "KERNEL_REGEX" \
    -f csv \
    -d $PROFILE_RUN_DIR/reports/pmc_<tag> \
    -- ./harness [args]

# Or a YAML/JSON job file (preferred for reproducibility). Each `jobs` entry
# mirrors a single rocprofv3 CLI invocation; there is no `name:` field at the
# job level (use the file name or comments to label).
# Note: gfx942 / gfx950 only expose `TCC_EA0_*` (no `TCC_EA1_*`) and only the
# three SQ_WAIT_* PMCs `SQ_WAIT_ANY`, `SQ_WAIT_INST_ANY`, `SQ_WAIT_INST_LDS`.
cat > /tmp/pmc.yaml <<'EOF'
jobs:
  - pmc:
      - SQ_WAVES
      - SQ_INSTS_VALU
      - SQ_WAIT_INST_ANY
      - SQ_WAIT_INST_LDS
      - SQ_LDS_BANK_CONFLICT
      - TCC_EA0_RDREQ_sum
      - TCC_EA0_RDREQ_32B_sum
      - GRBM_GUI_ACTIVE
    kernel_include_regex: "KERNEL_REGEX"
EOF
rocprofv3 -i /tmp/pmc.yaml -f csv -d $PROFILE_RUN_DIR/reports/pmc_<tag> -- ./harness [args]
```

A single `pmc:` list must fit in one hardware pass — rocprofv3 will **fail** the job if the counters don't fit, it does not auto-split. To collect more than one pass' worth, add multiple `- pmc: ...` entries under `jobs:` (one entry = one extra pass), or use `pmc_groups:` for explicit grouping. Counters that share a unit (SQ_*, TCP_*, TCC_*) often fit in one group; check `rocprofv3 -L` output for the conflict list.

The full list of available counters: `rocprofv3 -L` (long form `--list-avail`; verified against `rocprofv3 --help` on ROCm 7.x). On newer ROCm builds, `rocprofv3-avail list --pmc` is the dedicated companion. The legacy `--list-counters` / `--list-metrics` / `--list-basic` / `--list-derived` flags are from rocprof v1/v2 and are not part of rocprofv3.

---

## Recipe 5: A/B comparison (before vs after optimization)

```bash
# Before
rocprof-compute profile -n v1 -k my_kernel \
    -p $PROFILE_RUN_DIR/reports/rpc_v1 -- ./harness_v1 [args]

# After
rocprof-compute profile -n v2 -k my_kernel \
    -p $PROFILE_RUN_DIR/reports/rpc_v2 -- ./harness_v2 [args]

# Side-by-side from the CLI
rocprof-compute analyze \
    -p $PROFILE_RUN_DIR/reports/rpc_v1 \
    -p $PROFILE_RUN_DIR/reports/rpc_v2 \
    > $PROFILE_RUN_DIR/analysis/compare_v1_vs_v2.txt
```

Or in Python (see `04-python-api.md`):

```python
import os
from pathlib import Path
import pandas as pd
RUN = os.environ["PROFILE_RUN_DIR"]
# Per-kernel wall-clock duration: prefer rocprof-compute's `timestamps.csv`
# (sibling to `pmc_perf.csv` under `-p`), or use rocprofv3's `kernel_trace.csv`
# from a separate `rocprofv3 --kernel-trace -f csv -d <path>` run. rocprofv3
# nests the trace as `<host>/<pid>_*_kernel_trace.csv` by default (`<pid>` is
# a filename prefix, not a directory), so rglob the trace dir rather than
# guessing the hostname.
def _load_trace(tag):
    # rocprofv3 always emits a name prefix: default `<hostname>/<pid>_kernel_trace.csv`
    # or flat `<prefix>_kernel_trace.csv` with `--output-file <prefix>`. The
    # single rglob below covers both — no bare-filename form exists.
    paths = sorted(Path(f"{RUN}/reports/trace_{tag}").rglob("*_kernel_trace.csv"))
    if not paths:
        raise FileNotFoundError(f"no *_kernel_trace.csv under {RUN}/reports/trace_{tag}")
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)

d1, d2 = _load_trace("v1"), _load_trace("v2")
t1 = (d1["End_Timestamp"] - d1["Start_Timestamp"]).sum()
t2 = (d2["End_Timestamp"] - d2["Start_Timestamp"]).sum()
print(f"Speedup: {t1/t2:.2f}x")
```

---

## What rocprof-compute sections cover

```bash
# List all kernels & dispatches captured in this run (NOT section IDs).
rocprof-compute analyze -p <dir> --list-stats

# List all section / metric IDs available for this arch (source of truth).
rocprof-compute analyze --list-metrics gfx942        # MI300X
rocprof-compute analyze --list-metrics gfx950        # MI355X
```

Top-level block / section IDs (verified from `--list-metrics gfx942` on rocprof-compute 7.x). In **analyze** mode, `-b` accepts EITHER a block ID like `12` OR a metric ID like `12.1.1`. There is no alias map — `-b lds` / `-b l1i` / `-b sl1d` are not recognized; use `--list-metrics <gfx>` to find the right ID.

| ID | Block | What it tells you |
|---|---|---|
| 2  | Speed-of-Light (SoL)          | % of peak for compute, HBM, vL1, L2, scratch — the headline |
| 5  | CS / Wavefront                | Waves/wkg, achieved occupancy, wavefront launch |
| 7  | Wavefront Launch Stats        | Per-dispatch grid, workgroup, registers, LDS, occupancy limiter |
| 10 | Compute Pipe                  | VALU / SALU / Matrix-core busy %, IPC, FMA |
| 11 | Instruction Mix               | VALU / SALU / MFMA / VMEM / LDS / FLAT instruction counts |
| 12 | Pipe SoL                      | Per-pipe Speed-of-Light |
| 13 | Pipe Stats                    | Per-pipe stall / activity stats |
| 14 | Cache (overview)              | Cache summary across vL1 / L2 |
| 15 | L1D Cache (TCP)               | vL1 hit rate, sectors per request, bytes per wave |
| 16 | L2 Cache (TCC)                | L2 hit rate, atomics, bytes |
| 17 | L2 – Fabric / HBM (TCC_EA)    | HBM read/write bytes, achieved BW |
| 18 | Scratch / Spill               | Scratch reads/writes (= register spill on AMD) |
| 21 | Misc / PC sampling derived    | Other derived metrics (incl. PC-sample-driven aggregates) |

(Run `--list-metrics gfx942` on your actual install to confirm; the IDs above are top-level *block* IDs. Sub-metric IDs use decimal nesting like `12.1.1`.)

---

## Profiling multiple kernel launches

If the kernel is called multiple times and you want different iterations:

```bash
# Limit to the first N matches via the include-regex (rocprofv3 doesn't have a direct
# "skip" flag like ncu -s); instead, use --kernel-iteration-range:
rocprofv3 --kernel-trace \
    --kernel-include-regex "my_kernel" \
    --kernel-iteration-range "[6-9]" \
    -d $PROFILE_RUN_DIR/reports/trace_iter6_9 -- ./harness
```

`--kernel-iteration-range "[N-M]"` selects iterations N through M (1-indexed, inclusive on both ends, per the upstream `using-rocprofv3.rst` examples). Multiple ranges can be combined with commas, e.g. `"[1,2,[5-8]]"`. Useful for ignoring warmup or focusing on a steady-state iteration. (Confirm with `rocprofv3 --help` on your install; do **not** use the Python-slice form `[6:9]` — rocprofv3 uses a hyphen.)

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
- **PC sampling silently no-ops**: not all GPU/driver/ROCm combinations expose PC sampling. Make sure you passed `--pc-sampling-beta-enabled`, and try `--pc-sampling-method host_trap` first (most portable) before `stochastic`.
- **`rocprofv3` replays the *application*, not just the kernel**: any expensive host-side init (data loading, NCCL init, hipBLASLt warmup) is paid on every replay. Move it outside the profile window — or just shrink the harness, which is the whole point.
- **MI355X support requires ROCm 7+**. ROCm 6.x will refuse `--offload-arch=gfx950` and may also refuse to enumerate gfx950 counters in `rocprofv3 -L` output.
