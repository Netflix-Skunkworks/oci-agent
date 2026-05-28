"""Generate two figures:

  evals/smoketest/benchmark_plot.png    Coverage vs RMSE — our ATT estimate vs the full
                              ACIC 2016 competition (DIY + black-box combined).
                              RMSE axis is reversed so "further right = better".

  evals/smoketest/judge_ate_plot.png    Coverage vs RMSE — our ATE estimate, three
                              subsets (all / judge-satisfactory / judge-not-
                              satisfactory). Illustrates the judge's ability
                              to pick out the unsatisfactory ATE runs. ATT is
                              omitted (only ~2 of 231 runs flagged).

Inputs:
  evals/smoketest/benchmarks.csv          competition methods (SATT/ATT)
  evals/smoketest/judge_results.json      our smoketest rows with det+LLM judge tags

Requires `evals/smoketest/judge_results.json` to be present. Generate it first if
the smoketest hasn't been judged:

    python -m evals.smoketest.run --k 3
    python -m evals.smoketest.eval
    python -m evals.smoketest.judge --judge-mode both
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_BENCHMARKS = REPO_ROOT / "evals" / "smoketest" / "benchmarks.csv"
DEFAULT_RESULTS = REPO_ROOT / "evals" / "smoketest" / "judge_results.json"
DEFAULT_BENCHMARK_OUT = REPO_ROOT / "evals" / "smoketest" / "benchmark_plot.png"
DEFAULT_JUDGE_OUT = REPO_ROOT / "evals" / "smoketest" / "judge_ate_plot.png"

Z = 1.96
BOOTSTRAP_N = 2000


def compute_stats(rows: list[dict], rng: np.random.Generator) -> dict:
    """Point estimates + 95% CIs for bias / RMSE / coverage / width."""
    rows = [r for r in rows if r.get("bias") is not None]
    n = len(rows)
    if n == 0:
        return {"n": 0}
    bias = np.array([r["bias"] for r in rows])
    cov = np.array([1 if r["covered"] else 0 for r in rows])
    width = np.array([r["ci_upper"] - r["ci_lower"] for r in rows])

    mean_bias = float(bias.mean())
    se_bias = float(bias.std(ddof=1) / math.sqrt(n))

    rmse = float(math.sqrt(np.mean(bias ** 2)))
    boot_rmses = np.empty(BOOTSTRAP_N)
    for b in range(BOOTSTRAP_N):
        sample = rng.choice(bias, size=n, replace=True)
        boot_rmses[b] = math.sqrt(np.mean(sample ** 2))
    rmse_lo, rmse_hi = np.percentile(boot_rmses, [2.5, 97.5])

    cov_point = float(cov.mean())
    cov_se = math.sqrt(cov_point * (1 - cov_point) / n)
    cov_lo = max(0.0, cov_point - Z * cov_se)
    cov_hi = min(1.0, cov_point + Z * cov_se)

    return {
        "n": n,
        "bias": mean_bias,
        "bias_lo": mean_bias - Z * se_bias,
        "bias_hi": mean_bias + Z * se_bias,
        "rmse": rmse,
        "rmse_lo": float(rmse_lo),
        "rmse_hi": float(rmse_hi),
        "coverage": cov_point,
        "coverage_lo": cov_lo,
        "coverage_hi": cov_hi,
        "width": float(width.mean()),
    }


def setup_cov_rmse_axes(ax, xlim_right: float):
    """Coverage vs RMSE with reversed x-axis: higher x = lower RMSE = better.

    Caller passes a hint; we enforce a floor of 0.5 so cross-figure
    comparisons stay on a common axis but wider data still extends the
    range when needed."""
    xlim_right = max(0.5, xlim_right)
    ax.set_xlim(xlim_right, 0)  # invert: 0 on the right (best), max on the left
    ax.set_ylim(-0.02, 1.05)
    ax.axhline(0.95, color="grey", linestyle="--", linewidth=0.8, alpha=0.6,
               label="Nominal 95% coverage")
    ax.set_xlabel("RMSE  (→ better)")
    ax.set_ylabel("95% CI Coverage")
    ax.grid(True, alpha=0.3)


def make_benchmark_plot(benchmarks: pd.DataFrame, att_stats: dict, out: Path):
    """Coverage vs RMSE — our ATT vs full ACIC 2016 competition (combined)."""
    fig, ax = plt.subplots(figsize=(8.5, 6.5))

    setup_cov_rmse_axes(ax, max(float(benchmarks["rmse"].max()), att_stats["rmse_hi"]) * 1.08)

    ax.scatter(
        benchmarks["rmse"], benchmarks["coverage"],
        c="tab:blue", marker="o", s=55, alpha=0.55,
        edgecolors="white", linewidths=0.7,
        label=f"ACIC 2016 competition (n={len(benchmarks)})",
    )

    ax.errorbar(
        att_stats["rmse"], att_stats["coverage"],
        xerr=[[att_stats["rmse"] - att_stats["rmse_lo"]],
              [att_stats["rmse_hi"] - att_stats["rmse"]]],
        yerr=[[att_stats["coverage"] - att_stats["coverage_lo"]],
              [att_stats["coverage_hi"] - att_stats["coverage"]]],
        ecolor="grey", elinewidth=1.2, capsize=5,
    )

    ax.scatter(
        att_stats["rmse"], att_stats["coverage"],
        marker="*", s=200, c="black", edgecolors="white",
        linewidths=0.6, zorder=5,
        label="DRLearner ATT (ours)",
    )

    ax.set_title(
        "OCI Agent vs ACIC 2016 Competition — Coverage vs RMSE",
        fontsize=12,
    )
    ax.legend(loc="lower left", fontsize=9, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Wrote {out}")


def make_judge_plot(ate_subsets: dict, out: Path, judge_label: str):
    """Coverage vs RMSE — ATE only, three judge tiers."""
    fig, ax = plt.subplots(figsize=(8.5, 6.5))

    setup_cov_rmse_axes(ax, max(s["rmse_hi"] for s in ate_subsets.values() if s["n"] > 0) * 1.08)

    style = {
        "all":             ("tab:purple", "o", 0, "ATE — all"),
        "satisfactory":    ("tab:blue",   "*", 0, f"ATE — {judge_label} satisfactory"),
        "not_satisfactory":("tab:red",    "X", 0, f"ATE — {judge_label} not satisfactory"),
    }
    marker_size = {"all": 80, "satisfactory": 200, "not_satisfactory": 120}
    marker_alpha = {"all": 1.0, "satisfactory": 1.0, "not_satisfactory": 1.0}
    for key, s in ate_subsets.items():
        if s["n"] == 0:
            continue
        color, marker, ms, label = style[key]
        a = marker_alpha[key]
        ax.errorbar(
            s["rmse"], s["coverage"],
            xerr=[[s["rmse"] - s["rmse_lo"]], [s["rmse_hi"] - s["rmse"]]],
            yerr=[[s["coverage"] - s["coverage_lo"]], [s["coverage_hi"] - s["coverage"]]],
            fmt=marker, ms=ms, mfc=color, mec="black", mew=1.0,
            ecolor="gray", elinewidth=1.1, capsize=4, alpha=a,
        )
        ax.scatter(
            s["rmse"], s["coverage"],
            marker=marker, s=marker_size[key], c=color, zorder=5, alpha=a,
            label=f"{label} (n={s['n']})",
        )

    ax.set_title(
        f"{judge_label} Judge— ATE Coverage vs RMSE",
        fontsize=12,
    )
    ax.legend(loc="lower left", fontsize=9, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks", type=Path, default=DEFAULT_BENCHMARKS)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--judge-key", choices=["llm_satisfaction", "det_satisfaction"],
                        default="llm_satisfaction")
    parser.add_argument("--benchmark-out", type=Path, default=DEFAULT_BENCHMARK_OUT)
    parser.add_argument("--judge-out", type=Path, default=DEFAULT_JUDGE_OUT)
    args = parser.parse_args()

    benchmarks = pd.read_csv(args.benchmarks).dropna(subset=["rmse", "coverage"])
    payload = json.loads(args.results.read_text())
    rows = payload["runs"]
    rng = np.random.default_rng(0)

    judge_label = "LLM" if args.judge_key == "llm_satisfaction" else "Deterministic"

    att_all = compute_stats([r for r in rows if r["estimand"] == "att"], rng)
    make_benchmark_plot(benchmarks, att_all, args.benchmark_out)

    ate_rows = [r for r in rows if r["estimand"] == "ate"]
    ate_subsets = {
        "all":             compute_stats(ate_rows, rng),
        "satisfactory":    compute_stats(
            [r for r in ate_rows if r.get(args.judge_key) in
             ("fully_satisfactory", "satisfactory_with_caveats")], rng),
        "not_satisfactory": compute_stats(
            [r for r in ate_rows if r.get(args.judge_key) == "not_satisfactory"], rng),
    }
    make_judge_plot(ate_subsets, args.judge_out, judge_label)

    print("\nATT (overlay on benchmark plot):")
    print(f"  n={att_all['n']}  rmse={att_all['rmse']:.4f} "
          f"[{att_all['rmse_lo']:.4f}, {att_all['rmse_hi']:.4f}]  "
          f"cov={att_all['coverage']:.1%} [{att_all['coverage_lo']:.1%}, {att_all['coverage_hi']:.1%}]")
    print(f"\nATE subsets by {args.judge_key}:")
    for key, s in ate_subsets.items():
        if s["n"] == 0:
            print(f"  {key:<20} n=0 (empty)")
            continue
        print(f"  {key:<20} n={s['n']:>3}  rmse={s['rmse']:.4f} "
              f"[{s['rmse_lo']:.4f}, {s['rmse_hi']:.4f}]  "
              f"cov={s['coverage']:.1%} [{s['coverage_lo']:.1%}, {s['coverage_hi']:.1%}]")


if __name__ == "__main__":
    main()
