---
name: suggesting-remedies
description: |
  Invoke this skill when an OCI diagnostic fails or warns and a spec change is needed.
  Keywords:
  - covariate balance failure, overlap warning, placebo warning
  - ATO_THRESHOLD, trimming extreme propensities
  - changing estimand: ATE → ATT, ATE → ATO
  - adding covariates, dataset insufficiency
  - actor-critic feedback loop, suggested spec changes
---

- Trimming, raising ATO_THRESHOLD, or any sample restriction by propensity changes the estimand from the ATE to the ATO (at the chosen threshold). Never suggest "trim the ATE" or an "ATE trimming threshold" — those are contradictions. Suggest switching the estimand to ATO instead, and call out the estimand change explicitly.

Covariate Balance
- A failure means treatment and control units differ systematically in at least one confounder.
- Most likely affects the ATE — not every population covariate profile is well-represented in all treatment levels.
- Remedies: switch the estimand to ATT or ATO; raise ATO_THRESHOLD above 0.1 to trim extreme propensities; manually trim units with extreme values of the problematic covariates.
- Note in the report that all three remedies change the estimand away from the ATE.

Overlap
- A warning means many units have propensity scores near 0 or 1 (near-deterministic assignment).
- Not a hard fail — the estimator stays consistent — but extreme inverse weights cause instability and make counterfactuals depend on a few observations.
- Remedies: switch the estimand to ATT or ATO; trim by raising ATO_THRESHOLD. Note that trimming changes the estimand to the ATO at the chosen threshold.

Placebo
- A warning means a placebo outcome — known to be unaffected by treatment — has a statistically significant effect after adjustment.
- Not disqualifying: the placebo outcome is already controlled for in the actual analysis.
- The report MUST state that a placebo failure signals (a) high sensitivity of the results to a single confounder, and (b) likely residual confounding from variables the model has not captured.
- Primary remedy (MUST appear in suggestions whenever placebo warns): find and add covariates that explain the placebo outcome. These are the variables most likely driving the residual confounding. Identify candidates by training a model with the placebo as the target and inspecting which existing covariates have the highest feature importance — then add covariates that capture the same signal as those drivers.
- If the spec's covariate set already exhausts what's available in the dataset (e.g., the eval uses every x_* column), the report MUST explicitly state that the dataset may be insufficient for a credible analysis. Do not silently fall through to model-flexibility remedies.
- Propensity / outcome model flexibility (see Propensity model below) is a secondary remedy for placebo failures — list it only after the covariate-set remedy or the dataset-insufficiency note, never as the headline placebo response.

Propensity model
- Covariate balance failures and placebo warnings often indicate the propensity model is misspecified — it isn't capturing the true treatment-assignment process.
- Remedies: enable AUGMENT_CONTINUOUS_COVARIATES to add polynomial and bin-encoded versions of continuous features; switch to a more flexible propensity learner (e.g., gradient-boosted or deeper trees).
- These remedies do not change the estimand.
- If the propensity learner is already a flexible nonlinear model (e.g., XGBoost / LightGBM with sufficient depth and trees), augmentation is less likely to help — prefer adding more covariates, or accept that some confounding may be unobserved.
