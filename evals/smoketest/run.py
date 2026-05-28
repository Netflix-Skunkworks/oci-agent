"""Orchestrator: for each ACIC 2016 treatment subdirectory, sample K response
files and run the OCI pipeline against each (treatment, response) pair via
`oci_agent.agent run`.

Each run writes to <output-dir>/t<TT>_r<RRR>/. The companion script
`eval_smoketest.py` reads these directories and computes bias / RMSE /
95% CI coverage against ACIC ground truth.

Parallelism: --parallel P launches up to P subprocesses concurrently. The
per-run N_JOBS (XGBoost / sklearn thread count) is automatically capped at
cpu_count() // P so total threads ≈ cpu_count. Override with --n-jobs J.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import os
import random
import subprocess
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_ACIC_DIR = REPO_ROOT / "evals" / "acic2016"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "smoketest"
DEFAULT_SPECS_DIR = REPO_ROOT / "specs" / "smoketest" / "batch"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "smoketest.yaml"


def load_config(path: Path) -> dict:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def build_spec(acic_treatment: int, acic_response: int, n_jobs: int) -> dict:
    """First-iteration eval spec per plans/smoketest.md."""
    return {
        "acic_treatment": acic_treatment,
        "acic_response": acic_response,
        "seed": 1024,
        "covariates": [f"x_{i}" for i in range(1, 59)],
        "augment_continuous_covariates": False,
        "ato_threshold": 0.1,
        "estimate_ate": True,
        "estimate_att": True,
        "estimate_ato": True,
        "n_jobs": n_jobs,
        "analysis_notebook": "notebooks/econml.ipynb",
    }


def sample_responses(treatment_dir: Path, k: int, rng: random.Random) -> list[int]:
    files = sorted(treatment_dir.glob("zymu_*.csv"))
    chosen = rng.sample(files, min(k, len(files)))
    return [int(p.stem.split("_", 1)[1]) for p in chosen]


def discover_treatments(acic_dir: Path) -> list[int]:
    return sorted(int(p.name) for p in acic_dir.iterdir() if p.is_dir() and p.name.isdigit())


def run_one(spec_path: Path, output_dir: Path, run_name: str) -> tuple[str, bool, str]:
    """Execute one (spec, run_name) via subprocess. Returns (name, ok, last_stderr)."""
    result = subprocess.run(
        [
            sys.executable, "-m", "oci_agent.agent", "run", str(spec_path),
            "--output-dir", str(output_dir),
            "--name", run_name,
            "--skip-actor",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    return (run_name, result.returncode == 0, result.stderr[-500:] if result.stderr else "")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help=f"YAML config providing defaults (default: {DEFAULT_CONFIG}).")
    parser.add_argument("--k", type=int, default=None, help="Responses sampled per treatment.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--acic-dir", type=Path, default=DEFAULT_ACIC_DIR)
    parser.add_argument("--specs-dir", type=Path, default=DEFAULT_SPECS_DIR)
    parser.add_argument("--seed", type=int, default=None, help="Seed for response sampling.")
    parser.add_argument(
        "--treatments",
        type=str,
        default=None,
        help="Comma-separated treatment indices to restrict to (default: all 77).",
    )
    parser.add_argument(
        "--parallel", type=int, default=None,
        help="Number of concurrent subprocesses (default from config; 1 = sequential).",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=None,
        help="Per-run XGBoost/sklearn thread count. Default: cpu_count() // parallel.",
    )
    args = parser.parse_args()

    # Apply config-file defaults for any flag the user didn't pass on the CLI.
    cfg = load_config(args.config)
    if args.k is None:
        args.k = cfg.get("k", 1)
    if args.seed is None:
        args.seed = cfg.get("seed", 42)
    if args.parallel is None:
        args.parallel = cfg.get("parallel", 1)
    if args.treatments is None:
        args.treatments = cfg.get("treatments")

    cores = os.cpu_count() or 1
    if args.n_jobs is None:
        args.n_jobs = max(1, cores // args.parallel)
    if args.n_jobs * args.parallel > cores:
        print(
            f"WARNING: parallel * n_jobs ({args.parallel} * {args.n_jobs} = "
            f"{args.parallel * args.n_jobs}) exceeds cpu_count ({cores}); expect thrashing.",
            file=sys.stderr,
        )

    rng = random.Random(args.seed)
    args.specs_dir.mkdir(parents=True, exist_ok=True)

    treatment_ixs = (
        [int(x) for x in args.treatments.split(",")]
        if args.treatments else discover_treatments(args.acic_dir)
    )

    runs = []
    for t in treatment_ixs:
        tdir = args.acic_dir / str(t)
        if not tdir.is_dir():
            print(f"Skipping treatment {t}: {tdir} not found.", file=sys.stderr)
            continue
        for r in sample_responses(tdir, args.k, rng):
            runs.append((t, r))

    print(
        f"Planning {len(runs)} runs across {len(treatment_ixs)} treatments "
        f"(K={args.k}, parallel={args.parallel}, n_jobs={args.n_jobs}).",
        flush=True,
    )

    # Write all specs up front so the dispatch loop is purely IO-bounded by subprocess.
    spec_paths = []
    for t, r in runs:
        run_name = f"t{t:02d}_r{r}"
        spec_path = args.specs_dir / f"{run_name}.yaml"
        with open(spec_path, "w") as f:
            yaml.dump(build_spec(t, r, args.n_jobs), f, default_flow_style=False, sort_keys=False)
        spec_paths.append((spec_path, run_name))

    failures = []
    if args.parallel <= 1:
        for i, (spec_path, run_name) in enumerate(spec_paths, 1):
            print(f"[{i}/{len(runs)}] {run_name} ... ", end="", flush=True)
            name, ok, err = run_one(spec_path, args.output_dir, run_name)
            print("ok" if ok else "FAIL")
            if not ok:
                failures.append((name, err))
    else:
        completed = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as ex:
            futures = {
                ex.submit(run_one, spec_path, args.output_dir, run_name): run_name
                for spec_path, run_name in spec_paths
            }
            for fut in concurrent.futures.as_completed(futures):
                name, ok, err = fut.result()
                completed += 1
                print(f"[{completed}/{len(runs)}] {name} ... {'ok' if ok else 'FAIL'}", flush=True)
                if not ok:
                    failures.append((name, err))

    print(f"\nDone. {len(runs) - len(failures)}/{len(runs)} runs succeeded.")
    if failures:
        print("\nFailures:")
        for name, err in failures:
            last = err.strip().splitlines()[-1] if err.strip() else "(no stderr)"
            print(f"  {name}: {last}")


if __name__ == "__main__":
    main()
