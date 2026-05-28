"""Agentic smoketest judge.

Two judgment modes (selectable via configs/smoketest.yaml `judge.mode` or
`--judge-mode`):

  - "deterministic" — applies the writing-reports decision rules in Python:
      Covariate balance: max P1 SMD < 0.2 (else ⛔ fail).
      Overlap: < 10% of observations with propensity outside (0.1, 0.9)
        (else ⚠️ warning).
      Placebo: p > 0.05 (else ⚠️ warning).
    Reports two cuts:
      strict — all three checks pass
      loose  — no hard fail on balance

  - "llm" — calls the Critic with a cheap model (default
    `claude-haiku-4-5-20251001`) per run, persists the critique to
    <run>/critique.json, and partitions the contrast on the critic's
    per-estimand `satisfaction` dict. This deliberately uses the same
    Critic class the agent uses end-to-end so we measure how reliably an
    LLM follows the writing-reports / suggesting-remedies skills in
    practice — even though the rules themselves are deterministic.

  - "both" — runs deterministic and llm and prints two contrast tables.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import sys
import threading
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
# Make the `oci_agent` package importable when this script is run as
# `python -m evals.smoketest.judge` (which only adds evals/ to sys.path).
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_RESULTS = REPO_ROOT / "evals" / "smoketest" / "results.json"
DEFAULT_RUN_DIR = REPO_ROOT / "output" / "smoketest"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "smoketest.yaml"
DEFAULT_PLAN = REPO_ROOT / "plans" / "smoketest.md"


def load_config(path: Path) -> dict:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


# ── Deterministic judgment ──────────────────────────────────────────────────

def judge_deterministic(row: dict) -> tuple[str, list[str]]:
    """Return (satisfaction_tier, diagnostic_flags) per writing-reports rules."""
    flags = []
    smd = row.get("max_abs_weighted_smd")
    ov = row.get("fraction_outside_0_1_0_9")
    pp = row.get("placebo_p_value")
    balance_fail = smd is not None and smd >= 0.2
    overlap_warn = ov is not None and ov > 0.10
    placebo_warn = pp is not None and pp < 0.05
    if balance_fail: flags.append("⛔ balance")
    if overlap_warn: flags.append("⚠️ overlap")
    if placebo_warn: flags.append("⚠️ placebo")
    if balance_fail:
        satisfaction = "not_satisfactory"
    elif overlap_warn or placebo_warn:
        satisfaction = "satisfactory_with_caveats"
    else:
        satisfaction = "fully_satisfactory"
    return satisfaction, flags


# ── LLM judgment (cached per-run on disk) ───────────────────────────────────

def get_or_compute_critique(run_dir: Path, plan_path: Path, model: str, refresh: bool) -> dict:
    """Read cached critique.json from a run dir, or call the Critic to produce one."""
    crit_path = run_dir / "critique.json"
    if crit_path.exists() and not refresh:
        return json.loads(crit_path.read_text())

    # Lazy import so deterministic-mode users don't need the anthropic SDK.
    from oci_agent.critic import Critic

    results_path = run_dir / "results.json"
    spec_path = run_dir / "spec.yaml"
    if not (results_path.exists() and spec_path.exists()):
        raise FileNotFoundError(f"Missing results.json or spec.yaml in {run_dir}")

    critic = Critic(model=model)
    plan_text = plan_path.read_text() if plan_path.exists() else None
    critique = critic.evaluate(
        results_path.read_text(),
        plan_text=plan_text,
        spec_text=spec_path.read_text(),
        iteration=1,  # smoketest specs are always iter_01 by construction
    )
    out = {
        "satisfaction": critique.satisfaction,
        "is_satisfactory": critique.is_satisfactory,  # derived, back-compat
        "summary": critique.summary,
        "issues": critique.issues,
        "suggestions": critique.suggestions,
        "model": model,
    }
    crit_path.write_text(json.dumps(out, indent=2))
    return out


def judge_llm(rows: list[dict], run_dir_root: Path, plan_path: Path, model: str,
              refresh: bool, parallel: int = 1) -> None:
    """Annotate each row with `llm_satisfaction` (same value for all 3 estimands of a run).
    Calls the Critic concurrently when parallel > 1; cached critiques skip the API call."""
    by_run: dict[str, list[dict]] = {}
    for r in rows:
        by_run.setdefault(r["run"], []).append(r)

    items = sorted(by_run.items())
    total = len(items)
    print_lock = threading.Lock()
    counter = {"n": 0}

    def judge_one(run_name: str, group: list[dict]) -> None:
        try:
            critique = get_or_compute_critique(run_dir_root / run_name, plan_path, model, refresh)
            raw_sat = critique.get("satisfaction")
            # Per-estimand schema is a dict; legacy str gets broadcast.
            if isinstance(raw_sat, dict):
                per_est = raw_sat
            elif isinstance(raw_sat, str) and raw_sat:
                per_est = {est: raw_sat for est in ("ate", "att", "ato")}
            else:
                legacy_tier = "fully_satisfactory" if critique.get("is_satisfactory") else "not_satisfactory"
                per_est = {est: legacy_tier for est in ("ate", "att", "ato")}
            summary = critique.get("summary", "")
            err = None
        except Exception as e:
            per_est, summary, err = {}, str(e), e

        with print_lock:
            counter["n"] += 1
            if err:
                tag = f"FAIL ({err})"
            else:
                tag = ", ".join(f"{est}={tier}" for est, tier in per_est.items())
            print(f"[{counter['n']}/{total}] LLM-judging {run_name} ... {tag}", flush=True)

        for r in group:
            r["llm_satisfaction"] = per_est.get(r["estimand"])
            r["llm_summary"] = summary

    if parallel <= 1:
        for run_name, group in items:
            judge_one(run_name, group)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as ex:
            futures = [ex.submit(judge_one, name, group) for name, group in items]
            for f in concurrent.futures.as_completed(futures):
                f.result()


# ── Aggregation and printing ────────────────────────────────────────────────

def aggregate(rows: list[dict]) -> dict:
    rows = [r for r in rows if r.get("bias") is not None]
    n = len(rows)
    if n == 0:
        return {"n": 0, "bias": None, "rmse": None, "coverage_95": None, "interval_width": None}
    bias = sum(r["bias"] for r in rows) / n
    rmse = math.sqrt(sum(r["bias"] ** 2 for r in rows) / n)
    cov = sum(r["covered"] for r in rows) / n
    iw = sum(r["ci_upper"] - r["ci_lower"] for r in rows) / n
    return {"n": n, "bias": bias, "rmse": rmse, "coverage_95": cov, "interval_width": iw}


def fmt_row(label: str, s: dict) -> str:
    if s["n"] == 0:
        return f"{label:<26}  {0:>4}  {'-':>10}  {'-':>10}  {'-':>8}  {'-':>10}"
    return (
        f"{label:<26}  {s['n']:>4}  {s['bias']:>10.4f}"
        f"  {s['rmse']:>10.4f}  {s['coverage_95']:>7.1%}"
        f"  {s['interval_width']:>10.4f}"
    )


def print_contrast(rows: list[dict], cut_label: str, cut_key: str) -> None:
    """Partition by a boolean cut and print two-row tables per estimand."""
    print(f"Contrast — {cut_label}")
    header = f"{'Group':<32}  {'N':>4}  {'Bias':>10}  {'RMSE':>10}  {'Cov95':>8}  {'IntWidth':>10}"
    print(header)
    print("-" * len(header))
    for est in ("ate", "att", "ato"):
        sub = [r for r in rows if r["estimand"] == est]
        ready = aggregate([r for r in sub if r.get(cut_key) is True])
        not_ready = aggregate([r for r in sub if r.get(cut_key) is False])
        print(fmt_row(f"{est.upper()} ready", ready))
        print(fmt_row(f"{est.upper()} not ready", not_ready))
    print()


TIER_ORDER = ("fully_satisfactory", "satisfactory_with_caveats", "not_satisfactory")
TIER_SHORT = {"fully_satisfactory": "fully", "satisfactory_with_caveats": "caveats", "not_satisfactory": "not"}


def print_confusion_matrix(rows: list[dict]) -> None:
    """Confusion matrix at the (run x estimand) level: rows = deterministic
    (ground truth), cols = LLM. Both judges now emit a per-estimand verdict,
    so no collapse is needed."""
    pairs = [
        (r["det_satisfaction"], r["llm_satisfaction"])
        for r in rows
        if r.get("det_satisfaction") in TIER_ORDER and r.get("llm_satisfaction") in TIER_ORDER
    ]
    if not pairs:
        print("Confusion matrix: no LLM judgments available.\n")
        return

    print(f"Confusion matrix ({len(pairs)} run x estimand records)")
    print("Rows = deterministic ground truth, columns = LLM judgment.")
    col_labels = [TIER_SHORT[t] for t in TIER_ORDER]
    print(f"{'det / llm':<26}  " + "  ".join(f"{c:>8}" for c in col_labels) + f"  {'sum':>5}")
    print("-" * 64)
    for det in TIER_ORDER:
        row = [sum(1 for d, l in pairs if d == det and l == llm) for llm in TIER_ORDER]
        print(f"{det.replace('_', ' '):<26}  " + "  ".join(f"{v:>8}" for v in row) + f"  {sum(row):>5}")
    totals = [sum(1 for _, l in pairs if l == llm) for llm in TIER_ORDER]
    print("-" * 64)
    print(f"{'sum':<26}  " + "  ".join(f"{v:>8}" for v in totals) + f"  {len(pairs):>5}")

    agree = sum(1 for d, l in pairs if d == l)
    print(f"\nAgreement: {agree}/{len(pairs)} ({agree/len(pairs):.0%}).")
    print()


def print_three_tier_contrast(rows: list[dict], cut_label: str, cut_key: str) -> None:
    """Partition by satisfaction (3 tiers) and print per-estimand rows."""
    print(f"Contrast — {cut_label}")
    header = f"{'Group':<32}  {'N':>4}  {'Bias':>10}  {'RMSE':>10}  {'Cov95':>8}  {'IntWidth':>10}"
    print(header)
    print("-" * len(header))
    tiers = (
        ("fully satisfactory",       "fully_satisfactory"),
        ("satisfactory w/ caveats",  "satisfactory_with_caveats"),
        ("not satisfactory",         "not_satisfactory"),
    )
    for est in ("ate", "att", "ato"):
        sub = [r for r in rows if r["estimand"] == est]
        for label, tier in tiers:
            group = aggregate([r for r in sub if r.get(cut_key) == tier])
            print(fmt_row(f"{est.upper()} {label}", group))
    print()


# ── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR,
                        help="Root containing t<TT>_r<RRR>/ subdirs with results.json + spec.yaml.")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--judge-mode", choices=["deterministic", "llm", "both"], default=None)
    parser.add_argument("--judge-model", type=str, default=None,
                        help="Anthropic model name when --judge-mode is llm/both.")
    parser.add_argument("--judge-parallel", type=int, default=None,
                        help="Concurrent LLM judge calls (default from config).")
    parser.add_argument("--refresh-judgments", action="store_true",
                        help="Recompute cached <run>/critique.json instead of reusing.")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "evals" / "smoketest" / "judge_results.json")
    args = parser.parse_args()

    cfg = load_config(args.config).get("judge", {})
    mode = args.judge_mode or cfg.get("mode", "deterministic")
    model = args.judge_model or cfg.get("llm_model", "claude-haiku-4-5-20251001")
    parallel = args.judge_parallel if args.judge_parallel is not None else cfg.get("llm_parallel", 1)
    print(f"judge mode = {mode}" + (
        f", model = {model}, parallel = {parallel}" if mode in ("llm", "both") else ""
    ))
    print()

    payload = json.loads(args.results.read_text())
    rows = payload["runs"]

    estimands = ("ate", "att", "ato")
    tiers = ("fully_satisfactory", "satisfactory_with_caveats", "not_satisfactory")

    if mode in ("deterministic", "both"):
        for r in rows:
            sat, flags = judge_deterministic(r)
            r["det_satisfaction"] = sat
            r["diagnostic_flags"] = flags

        print("Deterministic tier counts (per estimand):")
        print(f"{'Estimand':<8}  {'fully':>6}  {'caveats':>8}  {'not':>4}  {'N':>4}")
        print("-" * 38)
        for est in estimands:
            sub = [r for r in rows if r["estimand"] == est]
            counts = {t: sum(1 for r in sub if r["det_satisfaction"] == t) for t in tiers}
            print(f"{est.upper():<8}  {counts['fully_satisfactory']:>6}"
                  f"  {counts['satisfactory_with_caveats']:>8}"
                  f"  {counts['not_satisfactory']:>4}  {len(sub):>4}")
        print()
        print_three_tier_contrast(rows, "DETERMINISTIC three-tier", "det_satisfaction")

    if mode in ("llm", "both"):
        judge_llm(rows, args.run_dir, args.plan, model, args.refresh_judgments, parallel=parallel)
        print()
        print("LLM tier counts (per estimand — the LLM now emits an independent verdict per estimand):")
        print(f"{'Estimand':<8}  {'fully':>6}  {'caveats':>8}  {'not':>4}  {'failed':>6}  {'N':>4}")
        print("-" * 50)
        for est in estimands:
            sub = [r for r in rows if r["estimand"] == est]
            counts = {t: sum(1 for r in sub if r.get("llm_satisfaction") == t) for t in tiers}
            failed = sum(1 for r in sub if r.get("llm_satisfaction") is None)
            print(f"{est.upper():<8}  {counts['fully_satisfactory']:>6}"
                  f"  {counts['satisfactory_with_caveats']:>8}"
                  f"  {counts['not_satisfactory']:>4}"
                  f"  {failed:>6}  {len(sub):>4}")
        print()
        print_three_tier_contrast(rows, f"LLM ({model}) three-tier", "llm_satisfaction")

        if mode == "both":
            print_confusion_matrix(rows)

    args.out.write_text(json.dumps({"runs": rows}, indent=2))
    print(f"Annotated rows -> {args.out}")


if __name__ == "__main__":
    main()
