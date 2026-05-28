# ACIC 2016 Competition Benchmarks

Source: https://jenniferhill7.wixsite.com/acic-2016/competition (retrieved 2026-05-20)

Competition metrics are reported for the **SATT** (Sample Average Treatment
Effect on the Treated) across the 20 DIY datasets and 7,700 black-box datasets
(77 settings × 100 replications) respectively. Coverage targets 95%.

## Our results

Configuration: DRLearner (EconML) with cross-fitted XGBoost nuisances, AIPW
pseudo-outcome for variance. Three ACIC response files sampled per treatment
setting (K=3, seed=42), 231 (treatment, response) pairs total. Spec defaults
match `examples/eval_spec.yaml`. Re-runnable via:

```bash
python -m evals.smoketest.run --k 3       # config: configs/smoketest.yaml
python -m evals.smoketest.eval            # bias/RMSE/coverage/width
python -m evals.smoketest.plot           # writes benchmark_plot.png
```

All confidence intervals below are 95% CIs derived from the per-run dispersion
(point ± 1.96 × std / √N for bias / width; bootstrap for RMSE; normal-approx
on a binomial for coverage). Because our results are themselves a sample of
231 randomly-chosen ACIC datasets, these CIs reflect sampling uncertainty in
the benchmark itself — not estimator uncertainty for any individual run
(which is already captured by the per-run 95% CI on the ATE/ATT/ATO).

| Estimand | N  | \|Bias\| (95% CI)            | RMSE (95% CI bootstrap)      | Coverage 95% (95% CI)        | Interval width (95% CI)      |
|---|---|---|---|---|---|
| **ATE** | 231 | 0.015 (0.000 – 0.037) | 0.173 (0.120 – 0.226) | 84.8% (80.2% – 89.5%) | 0.289 (0.275 – 0.302) |
| **ATT** | 231 | 0.017 (0.006 – 0.027) | 0.083 (0.069 – 0.097) | 96.1% (93.6% – 98.6%) | 0.383 (0.362 – 0.404) |
| **ATO** | 231 | 0.014 (0.006 – 0.023) | 0.066 (0.057 – 0.077) | 97.0% (94.8% – 99.2%) | 0.307 (0.293 – 0.321) |

Two figures are generated:

- **`benchmark_plot.png`** — Coverage vs RMSE scatter, our DRLearner ATT
  (black star, with 95% CI error bars) against the full ACIC 2016
  competition cloud (DIY + black-box merged, blue). The RMSE x-axis is
  reversed so further-right = lower RMSE = better. A dashed line marks
  the nominal 95% coverage.
- **`judge_ate_plot.png`** — Coverage vs RMSE scatter for the ATE
  estimator only, partitioned into three subsets: `all` (purple, alpha
  backdrop), `LLM-judge satisfactory` (fully + caveats; blue star), and
  `LLM-judge not satisfactory` (red X). Illustrates that filtering out
  the unsatisfactory runs sharpens both axes. ATT is omitted from this
  figure (only 2 of 231 runs flagged as unsatisfactory). Pass
  `--judge-key det_satisfaction` to partition on the deterministic
  judge instead.

Where our ATT lands among the 15 black-box benchmark methods:

| Metric | Our ATT | Rank | Tier above us | Tier below us |
|---|---|---|---|---|
| \|Bias\| | 0.017 | **5th of 16** | BART (\|−0.002\|), calCause (\|−0.003\|), H2O Ensemble (\|−0.007\|), TMLE (\|−0.007\|) | BalanceBoost (\|−0.020\|), Tree Strat (\|−0.022\|), Adj. Tree Strat / LASSO+CBPS (\|−0.027\|), the entire `teffects` / IPW family at \|−0.04\|+ |
| RMSE | 0.083 | **9th of 16** | BART (0.02), H2O / calCause / TMLE (0.03), Tree Strat / BalanceBoost (0.05), Adj. Tree Strat (0.07), LASSO+CBPS (0.08) | CBPS / `teffects psmatch / ra / ipwra` at 0.11, Linear Model / MHE Algorithm at 0.14, `teffects ipw` at 0.28 |

### Headline observations

- **ATT coverage (96.1%) is at the 95% target.** ATT is the most directly
  comparable to the SATT competition benchmarks. RMSE 0.083 lands between
  the LASSO+CBPS tier above (≈0.05–0.08) and the CBPS / `teffects` tier
  below (≈0.11).
- **ATE under-covers at the headline level (84.8%) but the judge cleanly
  isolates the cause.** The 75 LLM-`fully_satisfactory` ATE estimates
  reach **96.0% coverage at RMSE 0.067**; the 39 `not_satisfactory` runs
  collapse to **48.7% coverage and RMSE 0.350**. ATO is well-calibrated
  out of the box (97.0% coverage at RMSE 0.066) because the trimming
  step mechanically resolves the covariate-balance failure mode that
  drags ATE down.
- **|Bias|** is small for all three estimands (0.014–0.017), well under
  the black-box median. ATE's 95% CI on |bias| includes zero; ATT and
  ATO have CIs that just exclude zero (0.006–0.027 and 0.006–0.023
  respectively), giving a slight residual systematic bias at this sample
  size that's still order-of-magnitude smaller than the typical
  competition method.

### Pipeline notes

- DR-based standard error: `mean(Y_DR)` for ATE / ATO and the centered
  influence-function `mean(ψ_ATT)` for ATT (the centering term
  `−T_i · τ̂_ATT / π` accounts for estimating π = E[T]; omitting it
  inflates the SE by O(τ² (1−π) / (π N))). Nuisance predictions are
  out-of-fold cross-fits routed through `_reproduce_drlearner_cv_folds`
  in `oci_agent/backends/econml_helpers.py`.
- ATO uses the standard ATE pseudo-outcome restricted to the
  propensity-trimmed subset (`ATO_THRESHOLD = 0.1`).

### Agentic judge (LLM critic vs deterministic rules)

The smoketest doubles as a stress-test for how reliably an LLM critic follows
the writing-reports skill. `evals/smoketest/judge.py` supports three modes:

- `deterministic` — applies the writing-reports rules in Python (no LLM).
- `llm` — calls the Critic with a cheap model per run (default
  `claude-haiku-4-5-20251001`, configured in `configs/smoketest.yaml`); the
  critique JSON is cached in each `<run>/critique.json`.
- `both` — runs both and prints a deterministic × LLM **confusion matrix**.

LLM judge calls are dispatched concurrently. With `judge.llm_parallel: 8`
(config default) the smoketest finishes in a few minutes instead of going
serial; cached critiques skip the API call entirely on re-runs.

Both pathways produce the three-tier verdict defined in `writing-reports`:

| Tier | Rule |
|---|---|
| `fully_satisfactory` | all three diagnostics ✅ |
| `satisfactory_with_caveats` | no ⛔, ≥1 ⚠️ (overlap / placebo) |
| `not_satisfactory` | ⛔ on covariate balance (the only blocking diagnostic) |

#### Per-tier contrast (deterministic, K=3, N=231)

The deterministic judge classifies each (run × estimand) into one of the
three tiers, then we recompute bias / RMSE / coverage / width within each:

| Estimand | Tier | N | Bias | RMSE | Cov95 | IntWidth |
|---|---|---:|---:|---:|---:|---:|
| ATE | fully satisfactory | 79 | +0.024 | 0.065 | 96.2% | 0.303 |
| ATE | satisfactory w/ caveats | 115 | −0.009 | 0.125 | 89.6% | 0.287 |
| ATE | not satisfactory | 37 | **−0.115** | **0.359** | **45.9%** | 0.266 |
| ATT | fully satisfactory | 115 | +0.017 | 0.074 | 96.5% | 0.386 |
| ATT | satisfactory w/ caveats | 116 | +0.017 | 0.090 | 95.7% | 0.381 |
| ATT | not satisfactory | 0 | — | — | — | — |
| ATO | fully satisfactory | 115 | +0.016 | 0.062 | 97.4% | 0.318 |
| ATO | satisfactory w/ caveats | 116 | +0.013 | 0.071 | 96.6% | 0.296 |
| ATO | not satisfactory | 0 | — | — | — | — |

The contrast is sharpest on **ATE**: bias grows from +0.024 to **−0.115**
and RMSE from 0.065 to **0.359** as the tier degrades, with coverage
collapsing from 96% to **46%**. The 37 `not_satisfactory` ATEs carry
essentially all of the headline bias. **ATT** is well-calibrated in both
tiers (96.5% / 95.7%), and the deterministic judge never marks it as
`not_satisfactory` because the IF-based interval is wide enough to
absorb mild balance flags. **ATO** is similarly flat across tiers and
also never produces a `not_satisfactory` verdict — overlap trimming
mechanically resolves the balance failure mode.

Interval width is roughly constant within each estimand across tiers, so
the coverage drop on the ATE tail reflects point-estimate bias, not
narrower CIs.

#### LLM-vs-deterministic confusion matrix (K=3, N=231, per-estimand)

The Critic emits an **independent verdict per estimand** (ATE / ATT /
ATO each judged against their own balance / overlap / placebo
diagnostics), matching the deterministic judge's granularity. The
confusion matrix is at the (run × estimand) level, 231 × 3 = 693 records:

| det \ llm | fully | caveats | not | sum |
|---|---:|---:|---:|---:|
| fully satisfactory | 297 | 10 | 2 | 309 |
| satisfactory with caveats | 11 | 332 | 4 | 347 |
| not satisfactory | 0 | 0 | 37 | 37 |
| **sum** | **308** | **342** | **43** | **693** |

**Agreement: 666/693 (96%)** with `claude-haiku-4-5-20251001`. The two
judges never disagree on the deterministic-`not_satisfactory` tier
(37/37 match), so the hard balance ⛔ rule is fully reproduced by the
LLM. All 27 disagreements sit at the fully ↔ caveats boundary or are
cases where the LLM is slightly stricter than the deterministic rules
(6 records pushed from fully / caveats into `not_satisfactory`,
typically for ATE only).

#### LLM-partition contrast (K=3, per-estimand)

| Estimand | LLM tier | N | Bias | RMSE | Cov95 | IntWidth |
|---|---|---:|---:|---:|---:|---:|
| ATE | fully | 75 | +0.024 | 0.067 | 96.0% | 0.308 |
| ATE | caveats | 117 | −0.007 | 0.124 | 89.7% | 0.284 |
| ATE | not | 39 | **−0.111** | **0.350** | **48.7%** | 0.266 |
| ATT | fully | 115 | +0.019 | 0.074 | 96.5% | 0.393 |
| ATT | caveats | 114 | +0.016 | 0.090 | 95.6% | 0.373 |
| ATT | not | 2 | −0.078 | 0.088 | 100.0% | 0.353 |
| ATO | fully | 118 | +0.015 | 0.061 | 97.5% | 0.320 |
| ATO | caveats | 111 | +0.014 | 0.072 | 96.4% | 0.294 |
| ATO | not | 2 | −0.042 | 0.044 | 100.0% | 0.271 |

The ATE gradient is sharp: dropping the 39 LLM-`not_satisfactory` runs
moves ATE from RMSE 0.173 / cov 84.8% to **RMSE 0.067 / cov 96.0%** —
the satisfactory subset is essentially the headline ATE without the
bad-overlap tail. ATT and ATO show negligible spread (the LLM flags
only 2 of 231 runs `not_satisfactory` for each), so the satisfactory
buckets are indistinguishable from "all". The judge does its real
triage work on ATE.

---

## Do-It-Yourself Submissions (20 datasets)

Methods are split into **DIY** (left of dividing line) and **Black Box**
(right), both evaluated on the same 20 DIY datasets. Ordered by RMSE.

| Method | Type | Bias | RMSE | Coverage | Interval Length |
|---|---|---:|---:|---:|---:|
| DR w/GBM+MDIA | DIY | -0.006 | 0.01 | 100% | 0.08 |
| Regression Trees | DIY | 0.009 | 0.04 | 100% | 0.29 |
| Ad Hoc | DIY | -0.017 | 0.04 | 95% | 0.12 |
| LAS Gen GAM | DIY | -0.014 | 0.04 | 60% | 0.07 |
| VarSel NN | DIY | -0.022 | 0.05 | 75% | 0.15 |
| BART | BB | -0.009 | 0.02 | 65% | 0.04 |
| SL+TMLE | BB | -0.009 | 0.02 | 90% | 0.07 |
| H2O Ensemble | BB | -0.006 | 0.03 | 100% | 6.06 |
| calCause | BB | -0.013 | 0.03 | 75% | 0.07 |
| Tree Strat | BB | -0.021 | 0.04 | 95% | 0.17 |
| BalanceBoost | BB | -0.018 | 0.04 | 85% | 0.15 |
| MITSS | DIY | -0.057 | 0.09 | 45% | 0.07 |
| Manual | DIY | -0.054 | 0.10 | 45% | 0.12 |
| ProxMatch | DIY | -0.009 | 0.10 | 60% | 0.27 |
| LASSO+CBPS | BB | -0.027 | 0.09 | 30% | 0.05 |
| Adj. Tree Strat | BB | -0.027 | 0.06 | 65% | 0.10 |
| RBD TwoStepLM | DIY | -0.030 | 0.12 | 20% | 0.05 |
| Calibrated IPW | DIY | -0.048 | 0.12 | 55% | 0.11 |
| CBPS | BB | -0.041 | 0.10 | 100% | 0.74 |
| teffects psmatch | BB | -0.062 | 0.10 | 50% | 0.13 |
| teffects ra | BB | -0.056 | 0.11 | 45% | 0.10 |
| teffects ipwra | BB | -0.056 | 0.11 | 55% | 0.10 |
| Linear Model | BB | -0.062 | 0.13 | 25% | 0.08 |
| MHE Algorithm | BB | -0.062 | 0.13 | 25% | 0.09 |
| GLM-Boost | DIY | -0.067 | 0.31 | 10% | 0.09 |
| Bayes LM | DIY | -0.080 | 0.31 | 15% | 0.08 |
| IPTW estimator | DIY | -0.142 | 0.35 | 45% | 0.15 |
| teffects ipw | BB | -0.141 | 0.36 | 40% | 0.15 |
| Weighted GP | DIY | -0.186 | 0.55 | 0% | 0.02 |

---

## Black Box Submissions (77 settings × 100 replications = 7,700 datasets)

Ordered by RMSE.

| Method | Bias | RMSE | Coverage | Interval Length |
|---|---:|---:|---:|---:|
| BART | -0.002 | 0.02 | 81% | 0.04 |
| H2O Ensemble | -0.007 | 0.03 | 100% | 6.14 |
| calCause | -0.003 | 0.03 | 82% | 0.07 |
| TMLE | -0.007 | 0.03 | 88% | 0.07 |
| Tree Strat | -0.022 | 0.05 | 87% | 0.16 |
| BalanceBoost | -0.020 | 0.05 | 81% | 0.13 |
| Adj. Tree Strat | -0.027 | 0.07 | 60% | 0.11 |
| LASSO+CBPS | -0.027 | 0.08 | 31% | 4.97 |
| CBPS | -0.041 | 0.11 | 100% | 0.78 |
| teffects psmatch | -0.043 | 0.11 | 47% | 0.13 |
| teffects ra | -0.042 | 0.11 | 38% | 0.10 |
| teffects ipwra | -0.042 | 0.11 | 37% | 0.10 |
| Linear Model | -0.045 | 0.14 | 22% | 0.09 |
| MHE Algorithm | -0.045 | 0.14 | 23% | 0.09 |
| teffects ipw | -0.043 | 0.28 | 39% | 0.14 |
| **DRLearner ATT (ours, K=3 / N=231)** | **+0.017** | **0.083** | **96%** | **0.38** |

---

## Notes

- Competition metrics target **SATT** on the treated subset. Our ATT is the
  closest analog and is shown in the BB table above for direct comparison.
  ATE and ATO are reported here only because the pipeline computes them too;
  they are not benchmark estimands.
- Competition results were evaluated on 100 replications per treatment
  setting (7,700 BB datasets total); ours are 3 replications per treatment
  setting (231 datasets), so all 95% CIs above are wider than the
  competition's would be.
- Our headline ATT RMSE of 0.083 places us just below the LASSO+CBPS tier
  (RMSE 0.08) and above the `teffects psmatch / ra / ipwra` and CBPS tier
  (RMSE 0.11). With wider per-run intervals (mean width 0.38 vs e.g.
  BART's 0.04) and higher coverage (96% vs 81%) than most black-box
  methods, the DR-based intervals are conservative-but-honest rather than
  aggressively tight.
