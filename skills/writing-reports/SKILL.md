---
name: writing-reports
description: |
  Invoke this skill when writing oci-report.md from an executed OCI analysis.
  Keywords:
  - oci-report.md, actor-critic report
  - results.json, analysis spec, analysis plan
  - diagnostics: covariate balance, overlap, placebo outcome
  - SMD, P1 confounders, propensity scores
  - ATE, ATT, ATO estimands
  - decision-grade results, diagnostic table
---

- Review only the analysis plan, spec, and results.json. Do not use ground truth when reviewing evals.
- Keep reports short.

Interpreting the plan
- Plan rules scoped to a specific iteration (e.g. "for the first iteration", "for iter_01") bind only that iteration. They are starting conditions for the actor-critic loop, not invariants.
- When reviewing iter_NN with NN > 1, do not flag deviations from first-iteration defaults as compliance issues. Subsequent iterations are graded against (a) results-vs-spec consistency, (b) the diagnostic rules below, and (c) iteration-agnostic plan rules only.
- Iteration-agnostic plan rules (no iteration-scoping phrase) bind every iteration.
- Plan instructions about *how to revise* on later iterations (e.g. "if diagnostics fail for the ATO, rerun with a higher ATO_THRESHOLD") describe legitimate revisions — do not flag them as deviations.

Fail conditions
- The parameterization in results.json does not match the spec.
- results.json includes no estimands.
- An estimand is missing its estimate, standard error, or 95% confidence interval.

Diagnostics
- Covariate balance: max standardized mean difference (SMD) of P1 confounders must be < 0.2. Otherwise fail.
- Overlap: warn if more than 10% of observations have propensity scores outside (0.1, 0.9).
- Placebo outcome: warn if the treatment effect on the placebo is significant (p < 0.05).

Report structure
- Executive summary: treatment, outcome, dataset, P1 confounders, estimand(s), estimate(s), confidence interval(s).
- Diagnostic table and whether the results are satisfactory.
- Next steps: remedies for failed diagnostics and warnings, unobserved confounders, confirmatory experiments.

Diagnostic table columns
- Estimand: ATE, ATT, or ATO (diagnostics are per-estimand — one row per estimand × diagnostic).
- Diagnostic: Covariate Balance, Overlap, Placebo Outcome.
- Target Result: Max P1 SMD < 0.2, < 10% extreme propensities, no significant effect.
- Actual Result.
- Grade: ✅ Pass, ⛔ Fail, ⚠️ Warning. ⛔ is reserved exclusively for Max P1 SMD > 0.2 (covariate balance) — it is the only blocking failure. Overlap and placebo can only be ✅ or ⚠️ — never ⛔.

Satisfaction (three tiers)
- Decide one of three labels for the overall analysis and report it in the executive summary:
  - **fully_satisfactory** — every diagnostic is ✅. Results are decision-grade as-is.
  - **satisfactory_with_caveats** — no ⛔, but at least one ⚠️ (overlap and/or placebo). Results are usable but the caveats must be called out, and remedies should still be suggested.
  - **not_satisfactory** — at least one ⛔ (covariate balance failure). Results are blocked from decision use until the balance failure is remedied.
- The covariate-balance check is the only blocking failure. Overlap and placebo warnings degrade results to "with caveats" but never block.

- If any diagnostic fails or warns, invoke the suggesting-remedies skill.
