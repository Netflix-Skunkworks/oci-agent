# ATT estimation — ACIC 2016 dataset {{TREATMENT}} / response {{RESPONSE}}

Estimate the Average Treatment effect on the Treated (ATT),
E[Y(1) - Y(0) | Z=1], for the ACIC 2016 synthetic dataset with
acic_treatment={{TREATMENT}} and acic_response={{RESPONSE}}. The
dataset lives at evals/acic2016/.

File schema:
  x_1..x_58  pre-treatment covariates (mixed numeric / categorical)
  z          binary treatment indicator (0/1)
  y          observed outcome

Counts:
  N_total   = {{N_TOTAL}}
  N_treated = {{N_TREATED}}
  N_control = {{N_CONTROL}}

First 5 rows of the realized data:
{{HEAD_CSV}}

Reply with exactly one fenced JSON block:

```json
{
  "att_estimate": <float>,
  "ci_lower":     <float>,
  "ci_upper":     <float>,
  "reasoning":    "<one short paragraph>"
}
```

No other text.
