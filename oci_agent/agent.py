# Orchestrates the OCI analysis lifecycle: draft -> run -> evaluate -> revise.
"""OCI Agent CLI."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import yaml

from .actor import Actor
from .critic import Critic
from .nb_runner import run as run_notebook


def _load_spec(path: Path) -> dict:
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        return yaml.safe_load(text)
    if path.suffix == ".json":
        return json.loads(text)
    raise ValueError(f"Unsupported spec format: {path.suffix}")


def _next_iter_path(parent: Path, suffix: str) -> Path:
    """Return parent/iter_NN<suffix> with NN one greater than the highest existing."""
    parent.mkdir(parents=True, exist_ok=True)
    existing = sorted(parent.glob(f"iter_*{suffix}"))
    n = 1
    if existing:
        last = existing[-1].name
        try:
            n = int(last.removesuffix(suffix).split("_")[-1]) + 1
        except ValueError:
            n = len(existing) + 1
    return parent / f"iter_{n:02d}{suffix}"


def _draft_mode(args: argparse.Namespace) -> None:
    plan_path = Path(args.plan)
    if not plan_path.exists():
        sys.exit(f"Error: plan file not found: {plan_path}")

    actor = Actor()
    print("Actor: drafting spec from plan...", flush=True)
    result = actor.draft(plan_path.read_text())
    print(f"Actor: {result.notes[:300]}", flush=True)

    specs_dir = Path(args.specs_dir) / plan_path.stem
    spec_path = _next_iter_path(specs_dir, ".yaml")
    with open(spec_path, "w") as f:
        yaml.dump(result.spec, f, default_flow_style=False, sort_keys=False)
    print(f"Spec: {spec_path}", flush=True)


def _execute_mode(args: argparse.Namespace) -> None:
    spec = _load_spec(Path(args.spec))

    if not args.skip_actor:
        print("Actor: validating spec...", flush=True)
        result = Actor().propose(spec)
        spec = result.spec
        print(f"Actor: {result.notes}", flush=True)

    output_dir = Path(args.output_dir)
    if args.name:
        iter_dir = output_dir / args.name
    else:
        iter_dir = _next_iter_path(output_dir, "")
    iter_dir.mkdir(parents=True, exist_ok=True)

    with open(iter_dir / "spec.yaml", "w") as f:
        yaml.dump(spec, f, default_flow_style=False, sort_keys=False)

    print(f"Runner: executing {spec['analysis_notebook']} -> {iter_dir}", flush=True)
    run_notebook(spec, iter_dir)
    print(f"Results: {iter_dir / 'results.json'}", flush=True)


def _evaluate_mode(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    iter_dirs = sorted(p for p in output_dir.glob("iter_*") if p.is_dir())
    if not iter_dirs:
        sys.exit(f"Error: no iter_NN/ directories in {output_dir}")

    iter_dir = (
        output_dir / f"iter_{args.iteration:02d}"
        if args.iteration is not None
        else iter_dirs[-1]
    )

    results_path = iter_dir / "results.json"
    if not results_path.exists():
        sys.exit(f"Error: no results.json at {results_path}")

    plan_text = Path(args.plan).read_text() if args.plan else None
    spec_path = iter_dir / "spec.yaml"
    spec_text = spec_path.read_text() if spec_path.exists() else None

    # Parse iter_NN suffix from the directory name so the critic knows which iteration this is.
    try:
        iteration = int(iter_dir.name.split("_")[-1])
    except ValueError:
        iteration = None

    critic = Critic()
    print(f"Critic: evaluating {results_path}...", flush=True)
    critique = critic.evaluate(
        results_path.read_text(),
        plan_text=plan_text,
        spec_text=spec_text,
        iteration=iteration,
    )

    (iter_dir / "oci_report.md").write_text(critique.report)
    (iter_dir / "critique.json").write_text(json.dumps({
        "satisfaction": critique.satisfaction,
        "is_satisfactory": critique.is_satisfactory,  # derived, kept for back-compat
        "summary": critique.summary,
        "issues": critique.issues,
        "suggestions": critique.suggestions,
    }, indent=2))

    per_est = ", ".join(f"{est.upper()}={tier}" for est, tier in critique.satisfaction.items())
    print(f"Critic: [{per_est}] -- {critique.summary[:200]}", flush=True)
    for issue in critique.issues:
        print(f"  issue: {issue}", flush=True)
    for sug in critique.suggestions:
        print(f"  suggest: {sug}", flush=True)
    print(f"Report: {iter_dir / 'oci_report.md'}", flush=True)


def _confirm_continue(next_iter: int) -> bool:
    """Ask the user whether to revise and run the next iteration.

    Returns False on EOF / non-tty / negative answer so the loop stops cleanly
    when stdin is not interactive (e.g. piped or detached).
    """
    if not sys.stdin.isatty():
        print(f"(stdin not a tty — stopping before iter {next_iter:02d})", flush=True)
        return False
    try:
        answer = input(f"Revise and run iter_{next_iter:02d}? [y/N] ").strip().lower()
    except EOFError:
        return False
    return answer in ("y", "yes")


def _loop_mode(args: argparse.Namespace) -> None:
    spec_path = Path(args.spec)
    output_dir = Path(args.output_dir)
    specs_dir = Path(args.specs_dir)
    plan_path = Path(args.plan) if args.plan else None

    current_spec_path = spec_path
    for i in range(1, args.iterations + 1):
        run_args = argparse.Namespace(
            spec=str(current_spec_path),
            output_dir=str(output_dir),
            skip_actor=args.skip_actor,
            name=None,
        )
        _execute_mode(run_args)

        eval_args = argparse.Namespace(
            output_dir=str(output_dir),
            iteration=None,
            plan=str(plan_path) if plan_path else None,
        )
        _evaluate_mode(eval_args)

        if i == args.iterations:
            break
        if not _confirm_continue(i + 1):
            break

        revise_args = argparse.Namespace(
            output_dir=str(output_dir),
            specs_dir=str(specs_dir),
        )
        _revise_mode(revise_args)
        current_spec_path = sorted(specs_dir.glob("iter_*.yaml"))[-1]


def _revise_mode(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    iter_dirs = sorted(p for p in output_dir.glob("iter_*") if p.is_dir())
    if not iter_dirs:
        sys.exit(f"Error: no iter_NN/ directories in {output_dir}")
    iter_dir = iter_dirs[-1]

    critique_path = iter_dir / "critique.json"
    if not critique_path.exists():
        sys.exit(f"Error: no critique.json at {critique_path}; run `evaluate` first.")

    prev_spec_text = (iter_dir / "spec.yaml").read_text()
    report_text = (iter_dir / "oci_report.md").read_text()
    critique = json.loads(critique_path.read_text())
    suggestions = critique.get("suggestions", [])

    print(f"Actor: revising spec from {iter_dir}...", flush=True)
    result = Actor().revise(prev_spec_text, report_text, suggestions)
    print(f"Actor: {result.notes[:300]}", flush=True)

    specs_dir = Path(args.specs_dir)
    new_spec_path = _next_iter_path(specs_dir, ".yaml")
    with open(new_spec_path, "w") as f:
        yaml.dump(result.spec, f, default_flow_style=False, sort_keys=False)
    print(f"Revised spec: {new_spec_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="OCI Agent: notebook-driven observational causal inference pipeline.")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("draft", help="Draft a spec from a plan.")
    p.add_argument("--plan", required=True)
    p.add_argument("--specs-dir", default="specs")

    p = sub.add_parser("run", help="Execute a spec via nb_runner.")
    p.add_argument("spec")
    p.add_argument("--output-dir", default="output")
    p.add_argument("--skip-actor", action="store_true")
    p.add_argument("--name", default=None,
                   help="Use this name for the run directory instead of iter_NN.")

    p = sub.add_parser("evaluate", help="Evaluate the latest iteration's results.json with the critic.")
    p.add_argument("output_dir")
    p.add_argument("--iteration", type=int, default=None)
    p.add_argument("--plan", default=None)

    p = sub.add_parser("revise", help="Apply the latest critic suggestions to produce a new spec.")
    p.add_argument("output_dir")
    p.add_argument("--specs-dir", required=True, help="Directory to write the revised spec into.")

    p = sub.add_parser(
        "loop",
        help="Run the full actor-critic loop: run -> evaluate -> (prompt) -> revise, "
             "up to --iterations times. Prompts the user before each revise+run.",
    )
    p.add_argument("spec", help="Initial spec path; iter_01 runs against this.")
    p.add_argument("--output-dir", default="output")
    p.add_argument("--specs-dir", required=True, help="Directory to write revised specs into.")
    p.add_argument("--plan", default=None, help="Plan markdown passed to the critic.")
    p.add_argument("--iterations", type=int, default=1,
                   help="Maximum number of iterations to run (default: 1).")
    p.add_argument("--skip-actor", action="store_true")

    args = parser.parse_args()
    {
        "draft": _draft_mode,
        "run": _execute_mode,
        "evaluate": _evaluate_mode,
        "revise": _revise_mode,
        "loop": _loop_mode,
    }.get(args.command, lambda _: parser.print_help())(args)


if __name__ == "__main__":
    main()
