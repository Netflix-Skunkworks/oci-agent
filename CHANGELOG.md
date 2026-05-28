# Changelog

All notable changes to the OCI Agent are noted here.

## [0.1.0] — 2026-05-28 (Initial public release)

First public release of the actor–critic pipeline for observational
causal inference. Highlights:

### Pipeline
- Plan → spec → notebook → critique → revised spec loop, driven by the
  `oci-agent` CLI with auto-incrementing iteration directories.
- Single analysis notebook (`notebooks/econml.ipynb`): EconML
  `DRLearner` with cross-fitted XGBoost nuisances. Three estimands per
  run: ATE (AIPW), ATT (centered-EIF AIPW), ATO (overlap-trimmed AIPW
  with `ATO_THRESHOLD=0.1`).
- Notebook parameter injection via `oci_agent.nb_runner`, with an
  audit-trail convention (`# === INJECTED BY OCI AGENT === #` /
  `COMMENTED OUT BY OCI AGENT`) so every notebook edit is reviewable.

### Evaluation
- `evals/smoketest/run.py` + `evals/smoketest/eval.py`: 77 × K ACIC
  2016 battery (default K=3) producing bias / RMSE / coverage /
  interval-width summaries.
- `evals/smoketest/judge.py`: deterministic and / or LLM (Claude Haiku)
  satisfaction tiers, with a deterministic × LLM confusion matrix in
  `both` mode.
- `evals/baseline_vs_scaffolded/run.py`: head-to-head against an
  unscaffolded single Sonnet call on the same plan + rendered data;
  `--n-studies N` samples N pairs deterministically from one master
  seed.
- `evals/smoketest/plot.py` and `evals/baseline_vs_scaffolded/plot.py`
  produce the figures committed under `evals/`.

### Benchmark numbers (231 ACIC 2016 datasets, seed 42, K=3)

| Estimand | \|Bias\| | RMSE  | Cov95 |
|----------|----------|-------|-------|
| ATE      | 0.015    | 0.173 | 84.8% |
| ATT      | 0.017    | 0.083 | 96.1% |
| ATO      | 0.014    | 0.066 | 97.0% |

- ATT ranks 5th of 16 by |bias| and 9th of 16 by RMSE against the ACIC
  2016 black-box benchmark (see `evals/smoketest/benchmark_plot.png`).
- Judge agreement between the deterministic rules and Claude Haiku is
  666/693 = 96% of run × estimand records.
- Filtering ATE runs by the LLM-judge tier (`not_satisfactory`, n=39)
  drops ATE RMSE from 0.173 to 0.105 and lifts coverage from 84.8% to
  92.2%; the judge does its real triage work on ATE.

### Safety and Credentials
- `evals/generate_synthetic_acic.py` refuses to overwrite an existing
  `eval_datasets/acic2016/x.csv` unless `--force` is passed, so the
  real ACIC bundle can't be silently replaced by synthetic data.
- The Anthropic SDK reads `ANTHROPIC_API_KEY` and `ANTHROPIC_BASE_URL`
  from the environment.

### Known limitations
- Python 3.10+ only; `numpy<2` pinned (econml transitive constraint).
- CPU-only XGBoost.
- Single estimator notebook; ACIC-2016-shaped data only.
- No maintenance commitment; PRs not accepted (feedback via Issues
  welcome).
