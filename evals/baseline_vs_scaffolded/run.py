"""Fair comparison of two LLM approaches to producing an ATT estimate.

Both paths receive the exact same rendered plan text (written to
`plans/baseline_vs_scaffolded.md` with placeholders filled per invocation):

  Path A — scaffolded:    one iteration of `oci-agent loop` (Actor + notebook
                          + Critic), invoked as a subprocess. The notebook
                          (DRLearner / AIPW) produces the ATT estimate.
  Path B — unscaffolded:  one Anthropic messages.create call against the same
                          Sonnet model with the rendered plan as the only
                          user-message content. No tools, no system prompt,
                          no skills.

The script samples one random (acic_treatment, acic_response), runs both
paths, computes the ground truth from mu1 - mu0, and emits a contrast table.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.smoketest.run import (  # noqa: E402  (sys.path tweak above)
    build_spec,
    discover_treatments,
    sample_responses,
)

DEFAULT_ACIC_DIR = REPO_ROOT / "evals" / "acic2016"
DEFAULT_OUT_DIR = REPO_ROOT / "output" / "baseline_vs_scaffolded"
DEFAULT_TEMPLATE = REPO_ROOT / "plans" / "baseline_vs_scaffolded.md"


# ── Sampling ────────────────────────────────────────────────────────────────

def sample_pair(args: argparse.Namespace, rng: random.Random) -> tuple[int, int]:
    if args.treatment is not None and args.response is not None:
        return args.treatment, args.response
    if args.treatment is None:
        treatments = discover_treatments(args.acic_dir)
        treatment = rng.choice(treatments)
    else:
        treatment = args.treatment
    if args.response is None:
        response = sample_responses(args.acic_dir / str(treatment), 1, rng)[0]
    else:
        response = args.response
    return treatment, response


# ── Plan rendering ──────────────────────────────────────────────────────────

def load_realized(acic_dir: Path, treatment: int, response: int) -> pd.DataFrame:
    """Return the (x_1..x_58, z, y) view the analyst would see — no
    counterfactuals (y0/y1) and no ground-truth means (mu0/mu1)."""
    x = pd.read_csv(acic_dir / "x.csv")
    zymu = pd.read_csv(acic_dir / str(treatment) / f"zymu_{response}.csv")
    assert len(x) == len(zymu), "x.csv and zymu_*.csv row counts disagree"
    y = zymu["y1"].where(zymu["z"] == 1, zymu["y0"])
    return pd.concat([x, zymu["z"], y.rename("y")], axis=1)


def render_plan(template_path: Path, treatment: int, response: int,
                df: pd.DataFrame) -> str:
    template = template_path.read_text()
    n_total = len(df)
    n_treated = int((df["z"] == 1).sum())
    n_control = n_total - n_treated
    head_csv = df.head(5).to_csv(index=False).rstrip()
    return (
        template
        .replace("{{TREATMENT}}", str(treatment))
        .replace("{{RESPONSE}}", str(response))
        .replace("{{N_TOTAL}}", str(n_total))
        .replace("{{N_TREATED}}", str(n_treated))
        .replace("{{N_CONTROL}}", str(n_control))
        .replace("{{HEAD_CSV}}", head_csv)
    )


# ── Ground truth ────────────────────────────────────────────────────────────

def compute_truth(acic_dir: Path, treatment: int, response: int) -> dict:
    zymu = pd.read_csv(acic_dir / str(treatment) / f"zymu_{response}.csv")
    ite = zymu["mu1"] - zymu["mu0"]
    true_att = float(ite[zymu["z"] == 1].mean())
    return {
        "true_att": true_att,
        "n_treated": int((zymu["z"] == 1).sum()),
        "n_total": int(len(zymu)),
    }


# ── Path A — Scaffolded ─────────────────────────────────────────────────────

def run_scaffolded(treatment: int, response: int, plan_path: Path,
                   out_dir: Path, args: argparse.Namespace) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    specs_dir = out_dir / "specs"
    specs_dir.mkdir(exist_ok=True)
    spec = build_spec(treatment, response, args.n_jobs)
    spec["seed"] = args.seed
    spec_path = specs_dir / "iter_01.yaml"
    spec_path.write_text(yaml.dump(spec, default_flow_style=False, sort_keys=False))

    t0 = time.time()
    proc = subprocess.run(
        [
            sys.executable, "-m", "oci_agent.agent", "loop",
            str(spec_path),
            "--output-dir", str(out_dir),
            "--specs-dir", str(specs_dir),
            "--plan", str(plan_path),
            "--iterations", "1",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - t0
    (out_dir / "stdout.log").write_text(proc.stdout)
    (out_dir / "stderr.log").write_text(proc.stderr)
    if proc.returncode != 0:
        (out_dir / "error.txt").write_text(
            f"Subprocess exited {proc.returncode}\n--- stderr ---\n{proc.stderr}"
        )
        return {"runtime_s": elapsed, "error": f"exit {proc.returncode}"}

    results_path = out_dir / "iter_01" / "results.json"
    if not results_path.exists():
        return {"runtime_s": elapsed, "error": "no results.json"}
    results = json.loads(results_path.read_text())
    att = results.get("estimands", {}).get("att")
    if not att:
        return {"runtime_s": elapsed, "error": "no ATT in results.json"}
    return {
        "att_estimate": att["estimate"],
        "stderr": att["stderr"],
        "ci_lower": att["ci_lower"],
        "ci_upper": att["ci_upper"],
        "runtime_s": elapsed,
    }


# ── Path B — Unscaffolded baseline ──────────────────────────────────────────

JSON_FENCE_RE = re.compile(r"```json\s*")


def parse_json_fence(text: str) -> dict | None:
    """Mirror oci_agent.critic._parse_critique's fence extractor."""
    start = JSON_FENCE_RE.search(text)
    end = text.rfind("```")
    if not start or end <= start.end():
        return None
    try:
        return json.loads(text[start.end():end])
    except Exception:
        return None


def validate_baseline_payload(data: dict) -> tuple[bool, str | None]:
    required = ("att_estimate", "ci_lower", "ci_upper")
    for k in required:
        if k not in data:
            return False, f"missing key: {k}"
        v = data[k]
        if not isinstance(v, (int, float)) or not math.isfinite(v):
            return False, f"non-finite {k}: {v!r}"
    if not (data["ci_lower"] <= data["att_estimate"] <= data["ci_upper"]):
        return False, "ci_lower <= att_estimate <= ci_upper violated"
    return True, None


def run_baseline(plan_text: str, out_dir: Path, model: str) -> dict:
    """One messages.create call. One retry on parse failure. No tools."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "prompt.txt").write_text(plan_text)

    import anthropic
    # Anthropic() reads ANTHROPIC_API_KEY and ANTHROPIC_BASE_URL from the env.
    client = anthropic.Anthropic()

    messages = [{"role": "user", "content": plan_text}]

    def one_call() -> str:
        resp = client.messages.create(
            model=model,
            max_tokens=2048,
            temperature=0,  # reproducibility — Sonnet is near-deterministic at T=0
            messages=messages,
        )
        return resp.content[0].text

    t0 = time.time()
    raw = one_call()
    (out_dir / "response.txt").write_text(raw)
    payload = parse_json_fence(raw)
    ok, err = (False, "no fenced json") if payload is None else validate_baseline_payload(payload)

    if not ok:
        messages.append({"role": "assistant", "content": raw})
        messages.append({
            "role": "user",
            "content": "Your last reply could not be parsed. Reply with ONLY "
                       "the fenced JSON block, no prose before or after.",
        })
        retry_raw = one_call()
        (out_dir / "retry_response.txt").write_text(retry_raw)
        payload = parse_json_fence(retry_raw)
        ok, err = (False, "no fenced json") if payload is None else validate_baseline_payload(payload)
        if ok:
            raw = retry_raw  # promote retry to the canonical response
    elapsed = time.time() - t0

    out = {"runtime_s": elapsed, "raw_response": raw, "parse_ok": ok}
    if not ok:
        out["parse_error"] = err
        out["att_estimate"] = None
        return out

    out["att_estimate"] = float(payload["att_estimate"])
    out["ci_lower"] = float(payload["ci_lower"])
    out["ci_upper"] = float(payload["ci_upper"])
    out["reasoning"] = payload.get("reasoning", "")
    width = out["ci_upper"] - out["ci_lower"]
    out["degenerate_ci"] = not math.isfinite(width) or width < 1e-9
    return out


# ── Contrast & reporting ────────────────────────────────────────────────────

def annotate(record: dict, true_att: float) -> dict:
    """Add covered / abs_error fields when the path produced a usable answer."""
    if record.get("att_estimate") is None or not math.isfinite(record["att_estimate"]):
        record["abs_error"] = None
        record["covered"] = None
        return record
    record["abs_error"] = abs(record["att_estimate"] - true_att)
    lo, hi = record.get("ci_lower"), record.get("ci_upper")
    if lo is None or hi is None or not (math.isfinite(lo) and math.isfinite(hi)):
        record["covered"] = None
    else:
        record["covered"] = bool(lo <= true_att <= hi)
    return record


def print_table(truth: dict, scaffolded: dict, baseline: dict) -> None:
    def fmt(v, w=10, d=4):
        if v is None:
            return f"{'-':>{w}}"
        if isinstance(v, bool):
            return f"{'yes' if v else 'no':>{w}}"
        if isinstance(v, (int, float)):
            return f"{v:>{w}.{d}f}"
        return f"{str(v):>{w}}"

    header = (
        f"{'':<14}{'estimate':>12}{'ci_lower':>12}{'ci_upper':>12}"
        f"{'abs_err':>10}{'covered':>10}"
    )
    print(header)
    print("-" * len(header))
    print(
        f"{'truth':<14}{fmt(truth['true_att'], 12)}{fmt(None, 12)}{fmt(None, 12)}"
        f"{fmt(None, 10)}{fmt(None, 10)}"
    )
    for label, rec in (("scaffolded", scaffolded), ("baseline", baseline)):
        if rec.get("error"):
            print(f"{label:<14}  ERROR: {rec['error']}")
            continue
        print(
            f"{label:<14}{fmt(rec.get('att_estimate'), 12)}"
            f"{fmt(rec.get('ci_lower'), 12)}{fmt(rec.get('ci_upper'), 12)}"
            f"{fmt(rec.get('abs_error'), 10)}{fmt(rec.get('covered'), 10)}"
        )


# ── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42,
                   help="Seeds sampling AND the scaffolded spec's seed field.")
    p.add_argument("--treatment", type=int, default=None)
    p.add_argument("--response", type=int, default=None)
    p.add_argument("--model", type=str, default="claude-sonnet-4-6",
                   help="Baseline model. The scaffolded Actor/Critic models "
                        "are configured inside oci_agent and not overridden here.")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--acic-dir", type=Path, default=DEFAULT_ACIC_DIR)
    p.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    p.add_argument("--n-jobs", type=int,
                   default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument("--scaffolded-only", action="store_true")
    p.add_argument("--baseline-only", action="store_true")
    p.add_argument("--n-studies", type=int, default=1,
                   help="Number of (treatment, response) pairs to sample and "
                        "run sequentially from a single Random(seed). At N>1, "
                        "the --treatment/--response overrides are disallowed.")
    return p.parse_args()


def run_one_study(treatment: int, response: int, args: argparse.Namespace,
                  label: str = "") -> dict:
    """Run a single (treatment, response) pair end-to-end. Writes
    `<out_dir>/t{treatment}_r{response}/{contrast.json, plan.md, scaffolded/,
    baseline/}` and returns the contrast dict."""
    study_dir = args.out_dir / f"t{treatment}_r{response}"
    study_dir.mkdir(parents=True, exist_ok=True)

    df = load_realized(args.acic_dir, treatment, response)
    plan_text = render_plan(args.template, treatment, response, df)
    plan_path = study_dir / "plan.md"
    plan_path.write_text(plan_text)

    truth = compute_truth(args.acic_dir, treatment, response)
    print(f"{label}(treatment={treatment}, response={response}); "
          f"true_att={truth['true_att']:.4f}", flush=True)

    scaffolded: dict = {}
    baseline: dict = {}

    if not args.baseline_only:
        print(f"  Path A (scaffolded): launching `oci-agent loop --iterations 1` ...",
              flush=True)
        scaffolded = run_scaffolded(treatment, response, plan_path,
                                    study_dir / "scaffolded", args)
        annotate(scaffolded, truth["true_att"])
        print(f"    ... done in {scaffolded['runtime_s']:.1f}s", flush=True)

    if not args.scaffolded_only:
        print(f"  Path B (baseline): one {args.model} call ...", flush=True)
        baseline = run_baseline(plan_text, study_dir / "baseline", args.model)
        annotate(baseline, truth["true_att"])
        print(f"    ... done in {baseline['runtime_s']:.1f}s "
              f"(parse_ok={baseline.get('parse_ok')})", flush=True)

    contrast = {
        "seed": args.seed,
        "treatment": treatment,
        "response": response,
        "model": args.model,
        "truth": truth,
        "scaffolded": scaffolded,
        "baseline": baseline,
    }
    (study_dir / "contrast.json").write_text(json.dumps(contrast, indent=2, default=str))
    return contrast


def print_aggregate(contrasts: list[dict]) -> None:
    """Aggregate summary for a multi-study run."""
    import statistics
    n = len(contrasts)
    print(f"\n=== Aggregate over {n} studies ===")
    print(f"{'tx':>3} {'response':>10} {'truth':>7} | "
          f"{'s_est':>8} {'s_err':>9} {'s_cov':>5} | "
          f"{'b_est':>8} {'b_err':>9} {'b_cov':>5}")
    print("-" * 80)

    s_errs, s_covs, s_widths = [], [], []
    b_errs, b_covs, b_widths = [], [], []
    for c in contrasts:
        truth = c["truth"]["true_att"]
        s = c.get("scaffolded") or {}
        b = c.get("baseline") or {}
        s_est = s.get("att_estimate")
        b_est = b.get("att_estimate")
        s_cell = f"{s_est:>8.3f}" if s_est is not None else f"{'ERROR':>8}"
        s_err_cell = f"{s_est - truth:>+9.4f}" if s_est is not None else f"{'-':>9}"
        s_cov_cell = ("Y" if s.get("covered") else "N") if s_est is not None else "-"
        b_cell = f"{b_est:>8.3f}" if b_est is not None else f"{'ERROR':>8}"
        b_err_cell = f"{b_est - truth:>+9.4f}" if b_est is not None else f"{'-':>9}"
        b_cov_cell = ("Y" if b.get("covered") else "N") if b_est is not None else "-"
        print(f"{c['treatment']:>3} {c['response']:>10} {truth:>7.3f} | "
              f"{s_cell} {s_err_cell}   {s_cov_cell:>3}  | "
              f"{b_cell} {b_err_cell}   {b_cov_cell:>3}")

        if s_est is not None:
            s_errs.append(abs(s_est - truth))
            s_covs.append(bool(s.get("covered")))
            s_widths.append(s["ci_upper"] - s["ci_lower"])
        if b_est is not None:
            b_errs.append(abs(b_est - truth))
            b_covs.append(bool(b.get("covered")))
            b_widths.append(b["ci_upper"] - b["ci_lower"])

    print()
    if s_errs:
        rmse = (sum(e * e for e in s_errs) / len(s_errs)) ** 0.5
        print(f"Scaffolded:  N={len(s_errs)}  mean |err|={statistics.mean(s_errs):.3f}  "
              f"RMSE={rmse:.3f}  cov={sum(s_covs)}/{len(s_covs)}  "
              f"mean width={statistics.mean(s_widths):.3f}")
    if b_errs:
        rmse = (sum(e * e for e in b_errs) / len(b_errs)) ** 0.5
        print(f"Baseline:    N={len(b_errs)}  mean |err|={statistics.mean(b_errs):.3f}  "
              f"RMSE={rmse:.3f}  cov={sum(b_covs)}/{len(b_covs)}  "
              f"mean width={statistics.mean(b_widths):.3f}")


def main() -> None:
    args = parse_args()
    if args.n_studies < 1:
        sys.exit("--n-studies must be >= 1")
    if args.n_studies > 1 and (args.treatment is not None or args.response is not None):
        sys.exit("--treatment/--response are not allowed with --n-studies > 1")

    rng = random.Random(args.seed)
    if args.n_studies == 1:
        pairs = [sample_pair(args, rng)]
    else:
        treatments = discover_treatments(args.acic_dir)
        pairs = []
        for _ in range(args.n_studies):
            t = rng.choice(treatments)
            r = sample_responses(args.acic_dir / str(t), 1, rng)[0]
            pairs.append((t, r))

    contrasts = []
    for i, (t, r) in enumerate(pairs, 1):
        label = f"[{i}/{len(pairs)}] " if len(pairs) > 1 else "Sampled "
        contrasts.append(run_one_study(t, r, args, label=label))

    if len(contrasts) == 1:
        c = contrasts[0]
        study_dir = args.out_dir / f"t{c['treatment']}_r{c['response']}"
        print()
        print_table(c["truth"], c["scaffolded"], c["baseline"])
        print(f"\nWrote -> {study_dir}")
    else:
        print_aggregate(contrasts)
        print(f"\nWrote {len(contrasts)} studies -> {args.out_dir}")


if __name__ == "__main__":
    main()
