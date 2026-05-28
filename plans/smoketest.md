Goal: Evaluate whether econml.ipynb recovers causal effects in synthetic data.

The datasets are in `evals/acic2016/`. The `x.csv` in this directory has 58 columns titled `x_1` through `x_58`. These represent pre-treatment covariates. It also contains subdirectories numbered 1 to 77. Each number represents a synthetic treatment. Each subdirectory contains files titled `zymu_*.csv` that contain treatment and response data.

Rules
1. Specs must conform to `examples/eval_spec.yaml`
2. For the first iteration, specs should set:
    - COVARIATES to all x_* columns
    - AUGMENT_CONTINUOUS_COVARIATES to False.
    - ESTIMATE_ATE to True
    - ESTIMATE_ATT to True
    - ESTIMATE_ATO to True
    - ATO_THRESHOLD = 0.1
3. Do not give covariates semantic meaning or priority. The data are completely synthetic.
