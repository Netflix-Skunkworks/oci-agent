---
name: writing-specs
description: |
  Invoke this skill when initiating an OCI analysis from a plan or user prompt — i.e. before any notebook is parameterized or executed.
  Keywords:
  - analysis spec, eval spec, spec.yaml
  - treatment, outcome, covariates, confounders
  - ATE, ATT, ATO, ato_threshold
  - econml.ipynb, analysis_notebook
  - ACIC, acic_treatment, acic_response
---

Process
- Confirm the plan or prompt names a tabular data source (table or CSV) with a treatment column and an outcome column. If either is missing, ask the user before drafting.
- specs are YAML files. Write them under `specs/{plan}/`. Create the folder if it does not exist.
- Point `analysis_notebook` at an .ipynb file in `notebooks/`.

Validation (before saving)
- (skip for ACIC evals) Treatment and outcome columns exist in the data source
- `covariates` is non-empty and excludes the treatment and outcome columns.
- At least one estimand toggle (`estimate_ate` / `estimate_att` / `estimate_ato`) is true.
- `ato_threshold` is in (0, 0.5).
- `analysis_notebook` exists in `notebooks/`.

Fields — general analyses
- data_path, treatment, outcome
- covariates
- estimate_ate, estimate_att, estimate_ato
- ato_threshold
- seed
- analysis_notebook

Fields — ACIC evals only
- acic_treatment, acic_response   # replace data_path / treatment / outcome
- covariates
- seed
- estimate_ate, estimate_att, estimate_ato
- ato_threshold
- analysis_notebook: econml.ipynb
