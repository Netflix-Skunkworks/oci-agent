"""Forest plot of scaffolded vs unscaffolded ATT estimates against truth.

Reads every `contrast.json` under `output/baseline_vs_scaffolded/`
(one per invocation of `evals/baseline_vs_scaffolded/run.py`) and produces a
single figure with:

  - One row per sampled (treatment, response) dataset, sorted by true ATT.
  - For each row: a vertical tick at the true ATT, the scaffolded estimate
    + 95% CI (green, slightly above the row), and the baseline estimate +
    95% CI (red, slightly below). Rows where a path's CI covers the truth
    get a filled marker; misses get an open marker.
  - A suptitle aggregating per-path mean |error|, RMSE, coverage rate, and
    mean interval width.

Requires `output/baseline_vs_scaffolded/` to be populated. Generate
samples first, e.g.:

    python -m evals.baseline_vs_scaffolded.run --seed 42 --n-studies 10

Then call this script to refresh the figure.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_GLOB_ROOT = REPO_ROOT / "output" / "baseline_vs_scaffolded"
DEFAULT_OUT = REPO_ROOT / "evals" / "baseline_vs_scaffolded" / "plot.png"


def load_records(root: Path) -> list[dict]:
    paths = sorted(root.glob("*/contrast.json"))
    return [json.loads(p.read_text()) for p in paths]


def path_stats(records: list[dict], key: str) -> dict:
    """Mean |error|, RMSE, coverage, mean interval width for one path."""
    errs = []
    widths = []
    covered = []
    for r in records:
        rec = r.get(key) or {}
        est = rec.get("att_estimate")
        true_att = r["truth"]["true_att"]
        if est is None or not math.isfinite(est):
            continue
        errs.append(est - true_att)
        lo, hi = rec.get("ci_lower"), rec.get("ci_upper")
        if lo is not None and hi is not None and math.isfinite(lo) and math.isfinite(hi):
            widths.append(hi - lo)
            covered.append(lo <= true_att <= hi)
    n = len(errs)
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "mean_abs_err": float(np.mean(np.abs(errs))),
        "rmse": float(math.sqrt(np.mean(np.square(errs)))),
        "coverage": float(np.mean(covered)) if covered else None,
        "mean_width": float(np.mean(widths)) if widths else None,
    }


def fmt_stats(label: str, s: dict) -> str:
    if s["n"] == 0:
        return f"{label}: no parseable runs"
    cov = f"{s['coverage']:.0%}" if s["coverage"] is not None else "—"
    width = f"{s['mean_width']:.3f}" if s["mean_width"] is not None else "—"
    return (
        f"{label} (n={s['n']}): "
        f"|err|={s['mean_abs_err']:.3f}, "
        f"RMSE={s['rmse']:.3f}, "
        f"cov={cov}, "
        f"width={width}"
    )


def draw_path(ax, y: float, est: float | None, lo: float | None, hi: float | None,
              covers: bool | None, color: str, marker: str) -> None:
    if est is None or not math.isfinite(est):
        ax.scatter(0, y, color=color, marker="x", s=40,
                   linewidths=1.5, alpha=0.6)
        return
    if lo is not None and hi is not None and math.isfinite(lo) and math.isfinite(hi):
        ax.plot([lo, hi], [y, y], color=color, linewidth=1.6, alpha=0.55,
                solid_capstyle="butt")
    face = color if covers else "white"
    ax.scatter(est, y, facecolor=face, edgecolor=color, marker=marker,
               s=50, linewidths=1.5, zorder=5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_GLOB_ROOT,
                        help=f"Directory containing <ts>/contrast.json files "
                             f"(default: {DEFAULT_GLOB_ROOT}).")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    records = load_records(args.root)
    if not records:
        raise SystemExit(f"No contrast.json files under {args.root}.")
    records.sort(key=lambda r: r["truth"]["true_att"])

    n = len(records)
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    y_pos = np.arange(n)

    # Truth markers — short black vertical ticks at the row.
    for y, r in zip(y_pos, records):
        ax.scatter(r["truth"]["true_att"], y, marker="|", s=275,
                   color="black", linewidths=1.5, zorder=6)

    # Two parallel rows per dataset: scaffolded above, baseline below.
    offset = 0.18
    for y, r in zip(y_pos, records):
        true_att = r["truth"]["true_att"]
        s = r.get("scaffolded") or {}
        b = r.get("baseline") or {}
        s_lo, s_hi = s.get("ci_lower"), s.get("ci_upper")
        b_lo, b_hi = b.get("ci_lower"), b.get("ci_upper")
        s_covers = (
            s_lo is not None and s_hi is not None
            and math.isfinite(s_lo) and math.isfinite(s_hi)
            and s_lo <= true_att <= s_hi
        )
        b_covers = (
            b_lo is not None and b_hi is not None
            and math.isfinite(b_lo) and math.isfinite(b_hi)
            and b_lo <= true_att <= b_hi
        )
        draw_path(ax, y + offset, s.get("att_estimate"), s_lo, s_hi,
                  s_covers, "tab:green", "o")
        draw_path(ax, y - offset, b.get("att_estimate"), b_lo, b_hi,
                  b_covers, "tab:red", "s")

    # Legend (proxies — separate from the per-row scatters).
    ax.scatter([], [], facecolor="tab:green", edgecolor="tab:green",
               marker="o", s=50, label="Scaffolded (CI covers)")
    ax.scatter([], [], facecolor="white", edgecolor="tab:green",
               marker="o", s=50, linewidths=1.5, label="Scaffolded (CI misses)")
    ax.scatter([], [], facecolor="tab:red", edgecolor="tab:red",
               marker="s", s=50, label="Prompt-only (CI covers)")
    ax.scatter([], [], facecolor="white", edgecolor="tab:red",
               marker="s", s=50, linewidths=1.5, label="Prompt-only (CI misses)")
    ax.scatter([], [], marker="|", s=55, color="black", linewidths=1.5,
               label="Truth")

    labels = [f"{r['treatment']:02d}:{r['response']:09d}" for r in records]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7, fontfamily="monospace")
    ax.set_ylabel("Dataset")
    ax.set_xlabel("ATT Estimate (95% CI)")
    ax.set_ylim(-0.6, n - 0.4)
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.95)

    scaff = path_stats(records, "scaffolded")
    base = path_stats(records, "baseline")
    ax.set_title("Scaffolded vs Prompt-only ATT Estimates on ACIC 2016 Datasets", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out, dpi=300)
    print(f"Wrote {args.out}  (n_samples={n})")
    print(f"  Scaffolded: {fmt_stats('', scaff)}")
    print(f"  Prompt-only:   {fmt_stats('', base)}")


if __name__ == "__main__":
    main()
