# Baseline vs Scaffolded

Pits the full actor-critic loop against an unscaffolded single
Claude Sonnet call. Both paths receive the *identical* rendered plan
(`plans/baseline_vs_scaffolded.md`, populated per study with treatment
/ response IDs, sample-size summary, and a 5-row data head); the only
thing changing is whether the LLM has the OCI pipeline, the writing-
reports skill, the notebook executor, and the critic to lean on.

```bash
# One sampled study, fast taste.
python -m evals.baseline_vs_scaffolded.run --seed 42

# Reproduces the headline 10-study aggregate (~30 min sequential).
python -m evals.baseline_vs_scaffolded.run --seed 42 --n-studies 10

# Regenerates the per-study forest plot from output/baseline_vs_scaffolded/.
python -m evals.baseline_vs_scaffolded.plot
```

## Headline (10 studies, master seed 42)

|                | Scaffolded | Prompt-only |
|----------------|-----------:|------------:|
| coverage @ 95% | **9/10**   | 3/10        |
| mean \|error\| | **0.054**  | 2.572       |
| RMSE           | **0.064**  | 3.260       |
| mean CI width  | 0.352      | 1.475       |

The per-study breakdown lives in `plot.png` (regenerable from
`output/baseline_vs_scaffolded/t*_r*/contrast.json`). The same Sonnet
4.6 model produces estimates that miss truth by ~48× more on average
when stripped of the actor-critic scaffolding.
