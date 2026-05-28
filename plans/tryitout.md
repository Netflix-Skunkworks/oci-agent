Goal: Verify that the OCI pipeline (actor → notebook → critic → actor) runs
end-to-end on a single ACIC-2016-shaped dataset.

The dataset is produced by `evals/generate_synthetic_acic.py` and lives at
`evals/acic2016/`. There is one covariate matrix `x.csv` with 58 columns
named `x_1` through `x_58` (pre-treatment covariates) and one treatment
scenario in subdirectory `1/`, holding one response replication
`zymu_1.csv` (columns: `z`, `y0`, `y1`, `mu0`, `mu1`).

Rules
1. Specs must conform to `examples/eval_spec.yaml`.
2. Set `analysis_notebook: notebooks/econml.ipynb` — that is the only
   analysis notebook this repo ships.
3. Target the single (`acic_treatment: 1`, `acic_response: 1`) dataset that
   the generator produces by default.
4. For the first iteration, include all 58 `x_*` columns as `covariates`
   (no semantic prioritisation — the data are fully synthetic), set
   `augment_continuous_covariates: false`, enable all three of
   `estimate_ate`, `estimate_att`, `estimate_ato`, and use
   `ato_threshold: 0.1`.
