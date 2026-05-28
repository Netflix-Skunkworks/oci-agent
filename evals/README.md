# Evals

Two evaluation suites live here, each in its own subdirectory:

- **`smoketest/`** — 77-DGP × K-response ACIC 2016 battery. Runs the
  notebook against every (treatment, response) pair, aggregates bias /
  RMSE / coverage / interval-width, judges each (run × estimand) against
  the writing-reports rules (deterministic and / or LLM), and produces
  the headline benchmark plots. See `smoketest/README.md` for the full
  write-up, the deterministic × LLM confusion matrix, the per-tier
  contrast tables, and the rank against the ACIC 2016 competition.

- **`baseline_vs_scaffolded/`** — head-to-head between the full
  actor-critic loop and an unscaffolded single Sonnet 4.6 call given
  only the rendered plan plus a 5-row data head. Same plan, same data,
  same model on Path B; the only thing changing is the scaffolding.
  Output is a forest plot of per-study ATT estimates and CIs vs truth.

Shared at the top level:

- **`acic2016`** — symlink to `../eval_datasets/acic2016`. Both evals
  read from this. The directory itself is gitignored; populate it with
  either the official ACIC 2016 release (see `SETUP.md`) or the
  in-repo synthetic generator.
- **`generate_synthetic_acic.py`** — generates ACIC-shaped synthetic
  data when the official bundle isn't on hand. Refuses to overwrite an
  existing populated directory unless `--force` is passed.

Re-run any of them with `python -m evals.<subdir>.<module>`; pass
`--help` to each for the full flag set.
