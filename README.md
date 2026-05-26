# rocprof-report-skill

> [!IMPORTANT]
> This skill is maintained as a standalone submodule of
> [Kernel Design Agents (KDA)](https://github.com/mit-han-lab/kernel-design-agents)
> for easy installation.
>
> For bug reports, feature requests, and discussions, please use the main KDA repository:
> https://github.com/mit-han-lab/kernel-design-agents

A Claude Code skill for profiling HIP / ROCm kernels with `rocprofv3` and `rocprof-compute` (formerly Omniperf) on AMD Instinct **MI300X** (gfx942 / CDNA3) and **MI355X** (gfx950 / CDNA4). Covers the full workflow: build a standalone HIP harness, run `rocprofv3` + `rocprof-compute`, parse outputs with pandas / sqlite3 / `rocpd`, walk through six analysis dimensions, match patterns to a diagnosis playbook, and write an evidence-backed optimization report.

The skill is self-contained: reference docs, reusable helper scripts (HIP harness template, safetensors loader, report-analysis Python), and a companion CDNA3 / CDNA4 programming reference all ship in this repo.

> **Heritage:** the directory layout, six-dimension structure, and report template were proven on an earlier NVIDIA Nsight Compute version of this skill and have been re-grounded against the AMD ROCm 7.x profiling stack here.

---

## What's in this repo

```
.
├── SKILL.md                              ← skill entry point (with YAML frontmatter)
├── helpers/                              ← reusable code
│   ├── harness_template.hip              ← standalone HIP profiling harness template
│   ├── safetensors_loader.h              ← header-only safetensors reader (no deps, vendor-neutral)
│   ├── list_flashinfer_workloads.py      ← browse flashinfer-trace datasets (vendor-neutral)
│   ├── analyze_reports.py                ← extract + compare key metrics from rocprof-compute CSVs
│   ├── extract_stall_hotspots.py         ← per-line stall aggregation from ATT / PC-sampling output
│   ├── plot_timeline.py                  ← ASCII PMC-timeseries plots (reveals tail effects)
│   ├── rocprof_utils.py                  ← shared Python helpers, MI300X / MI355X key counter list
│   └── README.md
├── reference/                            ← detailed reference docs
│   ├── 00-directory-layout.md            ← profile/ directory conventions (read first)
│   ├── 01-workflow.md                    ← end-to-end profiling checklist
│   ├── 02-harness-guide.md               ← how to build a standalone HIP profiling harness
│   ├── 03-collection.md                  ← rocprofv3 / rocprof-compute command recipes
│   ├── 04-python-api.md                  ← pandas / sqlite3 / rocpd patterns
│   ├── 05-analysis-dimensions.md         ← six analysis dimensions
│   ├── 06-diagnosis-playbook.md          ← pattern → cause → fix
│   ├── 07-report-template.md             ← final report structure
│   ├── 08-mi300x-mi355x-counter-names.md ← gfx942 / gfx950 counter reference
│   └── 09-common-issues.md               ← permissions, ATT capture, ROCm version, etc.
└── cdna3-cdna4-hip-programming.md        ← companion: CDNA3 / CDNA4 programming principles
```

---

## Installation

Claude Code discovers skills in two locations:

- `~/.claude/skills/<skill_name>/SKILL.md` — **user-level**, available in every project
- `<repo>/.claude/skills/<skill_name>/SKILL.md` — **project-level**, scoped to one repo

Three ways to install this skill:

### Option 1 — Symlink from a clone (recommended)

Keeps the skill version-controlled and easy to update; edits in the clone are picked up instantly.

```bash
git clone git@github.com:zhenhuang12/rocprof-report-skill.git ~/workspace/rocprof-report-skill

mkdir -p ~/.claude/skills
ln -s ~/workspace/rocprof-report-skill ~/.claude/skills/rocprof-report-skill

# Or project-level install: scope to one repo
cd /path/to/other-repo
mkdir -p .claude/skills
ln -s ~/workspace/rocprof-report-skill .claude/skills/rocprof-report-skill
```

Pull updates with `cd ~/workspace/rocprof-report-skill && git pull`. The symlinks pick up the new content automatically.

### Option 2 — Copy into place

```bash
git clone git@github.com:zhenhuang12/rocprof-report-skill.git /tmp/rps
mkdir -p ~/.claude/skills
cp -r /tmp/rps ~/.claude/skills/rocprof-report-skill
```

### Option 3 — Git submodule (project-level only)

Scopes the skill to a single repo (the submodule path lands under `.claude/skills/`, the project-level discovery slot).

```bash
cd /path/to/other-repo
git submodule add git@github.com:zhenhuang12/rocprof-report-skill.git .claude/skills/rocprof-report-skill
git commit -m "Add rocprof-report-skill as a submodule"
```

---

## How Claude uses this skill

Once installed at `~/.claude/skills/rocprof-report-skill/` (or project-level), Claude Code will:

1. Advertise the skill's name + description in the system reminder of new conversations.
2. Let the user invoke it manually via `/rocprof-report-skill` or let the model invoke it with the Skill tool when the conversation matches the `description` triggers.

When invoked, Claude reads `SKILL.md`, follows its workflow (phases 0 → 6 in `reference/01-workflow.md`), and uses the helper scripts in `helpers/` as needed.

---

## Running the helpers directly (no Claude needed)

The Python helpers work standalone on any rocprof / rocprof-compute output you have:

```bash
# Create a run directory (the reference/ docs and Python helpers expect this exact var name)
export PROFILE_RUN_DIR=/path/to/your/profile/myrun
export SKILL=~/.claude/skills/rocprof-report-skill
mkdir -p "$PROFILE_RUN_DIR"/{harness,reports,analysis}

# Extract key metrics from a rocprof-compute "profile" output directory.
# Pass --arch explicitly (gfx942 for MI300X, gfx950 for MI355X) and
# --kernel to filter the dispatches; the script defaults to gfx942 otherwise.
python3 "$SKILL/helpers/analyze_reports.py" \
    --run-dir "$PROFILE_RUN_DIR" \
    --rpc "$PROFILE_RUN_DIR/reports/rpc_<tag>" --tag <tag> \
    --kernel "<your_kernel_substring>" --arch gfx942

# Per-line stall hotspots: --pcsamp-dir globs the rocprofv3 nested layout
# (e.g. pmc_1/<host>/<pid>_pc_sampling_host_trap_v0.csv), so you don't have to
# hardcode the exact filename. Use --pcsamp <file> only if you need a specific CSV.
python3 "$SKILL/helpers/extract_stall_hotspots.py" \
    --run-dir "$PROFILE_RUN_DIR" \
    --pcsamp-dir "$PROFILE_RUN_DIR/reports/pcsamp_<tag>" --tag <tag>

# … or from ATT JSON traces
python3 "$SKILL/helpers/extract_stall_hotspots.py" \
    --run-dir "$PROFILE_RUN_DIR" \
    --att-dir "$PROFILE_RUN_DIR/reports/att_<tag>" --tag <tag>

# ASCII timelines: per-CU distribution from a rocprof-compute output dir
python3 "$SKILL/helpers/plot_timeline.py" \
    --run-dir "$PROFILE_RUN_DIR" \
    --rpc "$PROFILE_RUN_DIR/reports/rpc_<tag>" --tag <tag> --per-cu

# … or from a rocprof-compute timeseries CSV (collect with
# `rocprof-compute profile --timeseries-sampling-rate 1ms ...`)
python3 "$SKILL/helpers/plot_timeline.py" \
    --run-dir "$PROFILE_RUN_DIR" \
    --timeseries "$PROFILE_RUN_DIR/reports/rpc_ts_<tag>/pmc_perf_timeseries.csv" --tag <tag>

# Browse a flashinfer-trace dataset to pick workload shapes
export FIB_DATASET_PATH=/path/to/flashinfer-trace
python3 "$SKILL/helpers/list_flashinfer_workloads.py" \
    --definition <your_definition_name>
```

The HIP harness template + safetensors loader live under `helpers/`; copy them into your profile run's `harness/` directory and fill in the kernel body. See `reference/02-harness-guide.md` for details.

---

## Requirements

- **ROCm 6.2+** for `rocprofv3`; **ROCm 6.3+** for `rocprof-compute` (renamed from Omniperf). **ROCm 7.0+** is required if you want to target MI355X (gfx950).
- HIP compiler `hipcc` (or `amdclang++` directly).
- Python 3.9+ with `pandas`. For ATT GUI: install ROCprof Compute Viewer (the AMD analog of `ncu-ui`); RGP does **not** support CDNA / Instinct.
- An AMD Instinct GPU with permission to access performance counters. On most systems, membership in the `render` group is sufficient — no root needed. See `reference/09-common-issues.md` if rocprofv3 reports missing counters or ATT capture fails.

The skill is optimized for MI300X (gfx942) and MI355X (gfx950) counter names, but the workflow and helpers work on any AMD GPU rocprofv3 supports. Some counter names may differ on older GPUs (gfx906 MI50, gfx908 MI100, gfx90a MI200/MI250X) — see `reference/08-mi300x-mi355x-counter-names.md` for guidance.

---

## License

MIT
