"""Aggregator: reads every output/<smoketest>/t*_r*/results.json, computes
bias / RMSE / 95% CI coverage against ACIC ground truth, and prints a per-
estimand pass/fail summary.

Ground truth is computed independently from the raw ACIC zymu CSVs
(`mu1 - mu0`) — it is never read from results.json. The ATO ground truth
uses `keep_indices` reported by the notebook in the ATO result object.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_ACIC_DIR = REPO_ROOT / "evals" / "acic2016"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "smoketest"
DEFAULT_OUT = REPO_ROOT / "evals" / "smoketest" / "results.json"


def compute_ground_truth(spec: dict, results: dict, acic_dir: Path, run_dir: Path) -> dict:
    """Compute true ATE/ATT/ATO from the raw zymu CSV.

    The ATO trimmed-row indices live in a sibling `keep_indices.json` written
    by the runner — they are not in `results.json` to keep that file small.
    """
    t = spec["acic_treatment"]
    r = spec["acic_response"]
    zymu = pd.read_csv(acic_dir / str(t) / f"zymu_{r}.csv")
    ite = zymu["mu1"] - zymu["mu0"]
    z = zymu["z"]
    gt = {
        "true_ate": float(ite.mean()),
        "true_att": float(ite[z == 1].mean()),
        "true_ato": None,
    }
    keep_path = run_dir / "keep_indices.json"
    if results["estimands"].get("ato") and keep_path.exists():
        keep = json.loads(keep_path.read_text()).get("ato")
        if keep:
            gt["true_ato"] = float(ite.iloc[keep].mean())
    return gt


def build_run_records(run_dirs: list[Path], acic_dir: Path) -> list[dict]:
    rows = []
    for d in run_dirs:
        rp = d / "results.json"
        if not rp.exists():
            continue
        results = json.loads(rp.read_text())
        spec = results["spec"]
        gt = compute_ground_truth(spec, results, acic_dir, d)
        for key in ("ate", "att", "ato"):
            est_obj = results["estimands"].get(key)
            if est_obj is None:
                continue
            true_val = gt.get(f"true_{key}")
            est = est_obj["estimate"]
            ci_lo = est_obj["ci_lower"]
            ci_hi = est_obj["ci_upper"]
            cb = est_obj["diagnostics"]["covariate_balance"]
            ov = est_obj["diagnostics"]["overlap"]
            pl = est_obj["diagnostics"]["placebo"]
            rows.append({
                "run": d.name,
                "acic_treatment": spec["acic_treatment"],
                "acic_response": spec["acic_response"],
                "estimand": key,
                "estimate": est,
                "true_value": true_val,
                "bias": est - true_val if true_val is not None else None,
                "stderr": est_obj["stderr"],
                "ci_lower": ci_lo,
                "ci_upper": ci_hi,
                "covered": (true_val is not None and ci_lo <= true_val <= ci_hi),
                "max_abs_weighted_smd": cb.get("max_abs_weighted_smd"),
                "fraction_outside_0_1_0_9": ov.get("fraction_outside_0_1_0_9"),
                "placebo_p_value": pl.get("p_value"),
            })
    return rows


def aggregate(rows: list[dict], key: str) -> dict:
    rows = [r for r in rows if r["estimand"] == key and r["bias"] is not None]
    n = len(rows)
    if n == 0:
        return {"n": 0, "bias": None, "rmse": None, "coverage_95": None, "interval_width": None}
    bias = sum(r["bias"] for r in rows) / n
    rmse = math.sqrt(sum(r["bias"] ** 2 for r in rows) / n)
    coverage = sum(r["covered"] for r in rows) / n
    interval_width = sum(r["ci_upper"] - r["ci_lower"] for r in rows) / n
    return {"n": n, "bias": bias, "rmse": rmse, "coverage_95": coverage, "interval_width": interval_width}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--acic-dir", type=Path, default=DEFAULT_ACIC_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    run_dirs = sorted(p for p in args.output_dir.glob("t*_r*") if p.is_dir())
    print(f"Found {len(run_dirs)} run directories under {args.output_dir}")

    rows = build_run_records(run_dirs, args.acic_dir)
    print(f"Collected {len(rows)} (run x estimand) records")

    summary = {k: aggregate(rows, k) for k in ("ate", "att", "ato")}

    args.out.write_text(json.dumps({"summary": summary, "runs": rows}, indent=2))
    print(f"Serialized -> {args.out}\n")

    header = f"{'Estimand':<8}  {'N':>4}  {'Bias':>10}  {'RMSE':>10}  {'Cov95':>8}  {'IntWidth':>10}"
    print(header)
    print("-" * len(header))
    for k in ("ate", "att", "ato"):
        s = summary[k]
        if s["n"] == 0:
            print(f"{k.upper():<8}  {0:>4}  {'-':>10}  {'-':>10}  {'-':>8}  {'-':>10}")
        else:
            print(
                f"{k.upper():<8}  {s['n']:>4}  {s['bias']:>10.4f}"
                f"  {s['rmse']:>10.4f}  {s['coverage_95']:>7.1%}"
                f"  {s['interval_width']:>10.4f}"
            )

    print("\nPass criteria:")
    for k in ("ate", "att", "ato"):
        s = summary[k]
        if s["n"] == 0:
            print(f"  {k.upper()}: NO DATA")
            continue
        bias_ok = abs(s["bias"]) < 0.5
        cov_ok = s["coverage_95"] >= 0.6
        status = "PASS" if (bias_ok and cov_ok) else "FAIL"
        print(
            f"  {k.upper()}: {status}"
            f"  (|bias|<0.5: {'ok' if bias_ok else 'FAIL'},"
            f" coverage>=60%: {'ok' if cov_ok else 'FAIL'})"
        )


if __name__ == "__main__":
    main()
