# Parameterizes an analysis notebook with a spec, executes it, and returns results.
from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Any

import nbformat
import yaml
from nbclient import NotebookClient

INJECTED_HEADER = "# === INJECTED BY OCI AGENT === #"


def check_python3_kernel_matches_current_interpreter() -> None:
    """
    Check whether the 'python3' Jupyter kernelspec points to the current interpreter.

    This function does not modify the user's Jupyter environment.
    """
    import shutil
    from jupyter_client.kernelspec import KernelSpecManager, NoSuchKernel

    ksm = KernelSpecManager()

    try:
        spec = ksm.get_kernel_spec("python3")
    except NoSuchKernel:
        raise RuntimeError(
            "No Jupyter kernel named 'python3' is registered. "
            "To register the current environment, run:\n\n"
            f"  {sys.executable} -m ipykernel install --user --name python3"
        )

    argv0 = spec.argv[0]
    # argv[0] is often a bare name like "python" rather than an absolute path;
    # resolve it via PATH before comparing.
    resolved_argv0 = argv0 if Path(argv0).is_absolute() else shutil.which(argv0)
    kernel_exe = Path(resolved_argv0).resolve() if resolved_argv0 else None
    current_exe = Path(sys.executable).resolve()

    if kernel_exe != current_exe:
        raise RuntimeError(
            "The registered 'python3' Jupyter kernel does not point to the current interpreter.\n\n"
            f"python3 kernel: {kernel_exe or argv0!r} (unresolvable)\n"
            f"current interpreter: {current_exe}\n\n"
            "Recommended fix:\n\n"
            f"  {sys.executable} -m ipykernel install --user --name python3\n\n"
            "Then re-run."
        )


def run(spec: dict[str, Any] | str | Path, output_dir: str | Path) -> dict:
    """Inject `spec` parameters into the notebook's top configuration cell,
    append a results serialization cell that writes results.json, execute
    the notebook, and return the parsed results.

    `spec` may be a dict or a path to a YAML/JSON file. It must contain
    `analysis_notebook` pointing at the .ipynb to execute.

    Notebook contract — the .ipynb at `spec['analysis_notebook']` must:
    - have a top code cell containing uppercase parameter assignments
      (e.g. `SEED = ...`, `COVARIATES = ...`); an injected cell with the
      spec's values is placed immediately after it.
    - by the end of execution, expose ate_result, att_result, ato_result.
      Each is None if its estimand toggle is False, or a dict with keys:
        estimate, stderr, ci_lower, ci_upper,
        diagnostics: {covariate_balance, overlap, placebo}
      Diagnostics are per-estimand (balance, overlap, and placebo depend
      on the weighting and trimming used for that estimand).
    """
    spec = _load_spec(spec)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nb_path = Path(spec["analysis_notebook"])
    if not nb_path.exists():
        raise FileNotFoundError(f"Notebook not found: {nb_path}")

    nb = nbformat.read(nb_path, as_version=4)
    _inject_parameters_cell(nb, spec)
    results_path = output_dir / "results.json"
    _append_serialization_cell(nb, results_path, spec)

    executed_path = output_dir / nb_path.name
    check_python3_kernel_matches_current_interpreter()
    NotebookClient(nb, timeout=-1, kernel_name="python3").execute(cwd=str(nb_path.parent))
    nbformat.write(nb, executed_path)

    with open(results_path) as f:
        return json.load(f)


def _load_spec(spec: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(spec, dict):
        return spec
    path = Path(spec)
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        return yaml.safe_load(text)
    if path.suffix == ".json":
        return json.loads(text)
    raise ValueError(f"Unsupported spec format: {path.suffix}")


def _inject_parameters_cell(nb, spec: dict[str, Any]) -> None:
    config_idx = _find_config_cell(nb)
    assignments = [INJECTED_HEADER]
    for key, value in spec.items():
        assignments.append(f"{key.upper()} = {value!r}")
    nb.cells.insert(config_idx + 1, nbformat.v4.new_code_cell(source="\n".join(assignments)))


def _find_config_cell(nb) -> int:
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code" and ("SEED" in cell.source or "COVARIATES" in cell.source):
            return i
    raise ValueError("No configuration cell found (expected uppercase parameter assignments).")


def _append_serialization_cell(nb, results_path: Path, spec: dict[str, Any]) -> None:
    keep_indices_path = results_path.parent / "keep_indices.json"
    src = f"""{INJECTED_HEADER}
import json
from pathlib import Path

_estimands = {{}}
if ESTIMATE_ATE: _estimands['ate'] = ate_result
if ESTIMATE_ATT: _estimands['att'] = att_result
if ESTIMATE_ATO: _estimands['ato'] = ato_result

_results = {{
    'spec': {spec!r},
    'estimands': _estimands,
}}
Path({str(results_path.resolve())!r}).write_text(json.dumps(_results, indent=2, default=str))

# Write ATO trimmed-sample row indices to a sibling file so they do not bloat
# results.json (a few thousand integers would crowd out diagnostics when the
# critic reads a truncated view).
if ESTIMATE_ATO and ato_keep_indices is not None:
    Path({str(keep_indices_path.resolve())!r}).write_text(
        json.dumps({{'ato': ato_keep_indices}})
    )
"""
    nb.cells.append(nbformat.v4.new_code_cell(source=src))
