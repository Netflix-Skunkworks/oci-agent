---
name: changing-notebooks
description: |
  Invoke this skill when making any modification to a notebook (.ipynb) in the OCI agent.
  Keywords:
  - .ipynb, Jupyter notebook
  - cell injection, comment-out cell
  - "# === INJECTED BY OCI AGENT === #"
  - "# === COMMENTED OUT BY OCI AGENT === #"
  - econml.ipynb, notebooks/ directory
  - parameterizing notebooks
---

- Never delete cells. Comment out their contents and add the header `# === COMMENTED OUT BY OCI AGENT === #`
- Mark injected cells with the header `# === INJECTED BY OCI AGENT === #`
- Never modify cell contents. Inject a new cell that implements the modification and comment out the old one.
- **Always leave the audit trail. No exceptions.** Apply comment-out + inject for every notebook edit — including one-line fixes, edits the user requested directly, and edits in areas the user has previously collapsed to clean cells. The audit trail is the contract; pruning it is the user's call, not the agent's. The only exception: when in the same session you are rewriting your *own* prior injected cell — that case has no user-authored work to preserve.

Cell placement
- Place each injected cell immediately after the cell it replaces. Keep the commented-out cell and its replacement adjacent so the diff is local.
- Preserve cell type: replace a code cell with a code cell, a markdown cell with a markdown cell.
- Do not inject above the top-of-notebook configuration cell. Do not inject below the results serialization cell at the bottom — nb_runner.py appends that at execution time.

Scope
- In-scope modifications: estimator config, diagnostic / plotting cells, intermediate computation cells, code-cell logic changes.
- Out-of-scope: editing the configuration cell to change spec parameters — change the spec instead, nb_runner.py injects parameters at run time.
- Out-of-scope: adding a results serialization cell — nb_runner.py owns that and will append it.
