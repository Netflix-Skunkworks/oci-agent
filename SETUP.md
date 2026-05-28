# Setup

`oci-agent` is an installable Python package (declared in `pyproject.toml`).
A single venv that has the package installed and the scientific stack pinned
to `numpy<2` is the simplest path.

## 1. Create a venv and install

```bash
python -m venv ~/.venvs/oci-py
source ~/.venvs/oci-py/bin/activate

# numpy<2 is required because econml's transitive `shap` references the
# removed np.bool8 in numpy >= 2.0. The pin propagates from pyproject.toml.
pip install -e .
```

`pip install -e .` also installs an `oci-agent` console-script entry point that
mirrors `python -m oci_agent.agent`.

## 2. ACIC 2016 datasets

The 77-treatment ACIC 2016 benchmark (~2.5 GB) is **not tracked in this
repo**. Download the competition data and unzip the contents under
`eval_datasets/acic2016/`:

```
eval_datasets/acic2016/
    x.csv                # the 58 pre-treatment covariates (~5,000 rows)
    1/zymu_<id>.csv      # treatment setting 1, response replications
    2/zymu_<id>.csv      # ...
    ...
    77/zymu_<id>.csv     # treatment setting 77
```

Source: <https://jenniferhill7.wixsite.com/acic-2016/competition>. The symlink
`evals/acic2016 -> ../eval_datasets/acic2016` is what the notebook and the
smoketest reference; don't rename either side.

## 3. ANTHROPIC_API_KEY

The actor (spec drafting and revision) and the critic (report writing) call
the Anthropic API. Set the key before invoking any subcommand that uses them:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

`oci_agent.agent run --skip-actor` and a manually-written spec do not
require the key. `draft`, `evaluate`, `revise`, and the LLM judge path of
`evals/smoketest/judge.py` do.

## Verifying

```bash
# Package importable from anywhere.
python -c "from oci_agent.backends.econml_helpers import aipw_pseudo_outcome; print('OK')"

# Console script wired up.
oci-agent --help

# End-to-end on one ACIC dataset.
python -m oci_agent.agent run specs/smoketest/iter_01.yaml \\
    --output-dir output/smoketest --name t10_r7692299 --skip-actor
# Expect: output/smoketest/t10_r7692299/results.json
```
