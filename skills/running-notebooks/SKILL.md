---
name: running-notebooks
description: |
  Invoke this skill before executing any OCI analysis (notebook run).
  Keywords:
  - nb_runner.py, notebook execution
  - analysis spec, spec parameter injection
  - results.json, results serialization cell
  - .ipynb, notebooks/ directory
  - papermill-style parameterization
---

- Specs must point to an .ipynb file in `notebooks/`.
- Ask before analyzing any data outside of a notebook.
- nb_runner.py injects the spec parameters into the configuration cell at the top of the notebook.
- nb_runner.py appends a results serialization cell at the bottom that writes results.json.
