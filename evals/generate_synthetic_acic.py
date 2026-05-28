"""Generate synthetic ACIC-2016-shaped datasets for end-to-end testing.

The real ACIC 2016 release (Dorie et al. 2019) bundles 77 data-generating
processes (DGPs) layered on top of a single 4802-row covariate matrix derived
from the IHDP / NHANES IHDP follow-up. The covariates are NOT regenerated;
only the (z, y0, y1, mu0, mu1) columns vary per (treatment, response).

This script does NOT reproduce the official DGPs. It synthesizes one or more
ACIC-shaped instances (one shared covariate matrix + one
(z, y0, y1, mu0, mu1) file per (treatment, response) pair) with the same
on-disk schema, so the notebook pipeline and the smoketest battery can be
exercised end-to-end without the ~2.5 GB official release. Each DGP
intentionally mirrors the "DGP 1" flavor from the paper: low-complexity,
linear response surfaces, moderate confounding, a constant treatment effect
with a small heterogeneous component, propensities clipped away from 0/1 to
preserve overlap. The treatment-effect magnitude varies smoothly across
treatment indices so the smoketest sees a non-trivial range of estimands.

Outputs (paths the notebook reads via the `evals/acic2016` symlink):
    eval_datasets/acic2016/x.csv                          # 58 covariates, N rows
    eval_datasets/acic2016/<T>/zymu_<R>.csv               # z, y0, y1, mu0, mu1
        for each treatment T in --treatments and response R in 1..K

Usage:
    # Single dataset for the "try it out" walk-through (treatment=1, response=1):
    python evals/generate_synthetic_acic.py

    # Full synthetic battery for evals/smoketest/run.py (~80 MB, ~10 s):
    python evals/generate_synthetic_acic.py \\
        --treatments 1-77 --responses-per-treatment 5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _project_root() -> Path:
    """Walk up from this file until we find the directory with pyproject.toml."""
    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError("Could not locate pyproject.toml from generator script.")


def generate_covariates(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """58 ACIC-shaped covariates: a mix of continuous, binary, and count types.

    Real ACIC `x.csv` has a few object-dtype columns; we keep this all-numeric
    to maximise compatibility with the notebook's xgboost backends without
    depending on the preprocessor's one-hot path.
    """
    cols: dict[str, np.ndarray] = {}

    for i in range(1, 31):
        cols[f"x_{i}"] = rng.standard_normal(n).astype(np.float64)

    for i in range(31, 49):
        cols[f"x_{i}"] = rng.binomial(1, 0.35, size=n).astype(np.int64)

    for i in range(49, 59):
        cols[f"x_{i}"] = rng.poisson(lam=2.0, size=n).astype(np.int64)

    return pd.DataFrame(cols)


def generate_zymu(
    rng: np.random.Generator,
    X: pd.DataFrame,
    true_ate: float = 3.0,
) -> pd.DataFrame:
    """Generate treatment + potential outcomes for one ACIC response replication.

    Confounding: propensity and outcome both depend on x_1..x_8 (continuous)
    plus x_31..x_34 (binary). True ATE is `true_ate` plus a small heterogeneous
    component driven by x_5; mean of the heterogeneity is ~0 so the realised
    ATE is close to `true_ate`.
    """
    n = len(X)

    confounders = X[[f"x_{i}" for i in range(1, 9)]].to_numpy()
    bin_confounders = X[[f"x_{i}" for i in range(31, 35)]].to_numpy()

    propensity_weights = np.array([0.6, -0.4, 0.3, -0.25, 0.2, -0.15, 0.1, 0.1])
    bin_propensity_weights = np.array([0.3, -0.2, 0.25, -0.15])
    logit_e = (
        confounders @ propensity_weights
        + bin_confounders @ bin_propensity_weights
        - 0.2
    )
    e = 1.0 / (1.0 + np.exp(-logit_e))
    e = np.clip(e, 0.05, 0.95)
    z = rng.binomial(1, e).astype(np.int64)

    outcome_weights = np.array([0.5, 0.4, -0.3, 0.25, 0.6, -0.2, 0.15, -0.1])
    bin_outcome_weights = np.array([0.4, 0.3, -0.25, 0.2])
    base = (
        confounders @ outcome_weights
        + bin_confounders @ bin_outcome_weights
        + 1.5
    )

    tau = true_ate + 0.4 * X["x_5"].to_numpy()

    sigma = 1.0
    eps0 = rng.normal(0.0, sigma, size=n)
    eps1 = rng.normal(0.0, sigma, size=n)

    mu0 = base
    mu1 = base + tau
    y0 = mu0 + eps0
    y1 = mu1 + eps1

    return pd.DataFrame(
        {
            "z": z,
            "y0": y0,
            "y1": y1,
            "mu0": mu0,
            "mu1": mu1,
        }
    )


def parse_treatments(spec: str) -> list[int]:
    """Parse a `--treatments` argument like "1", "1,3,5", or "1-77"."""
    spec = spec.strip()
    if not spec:
        raise ValueError("--treatments cannot be empty")
    out: list[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if "-" in chunk:
            lo_s, hi_s = chunk.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if lo > hi:
                raise ValueError(f"--treatments range {chunk!r} has lo>hi")
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(chunk))
    if not all(t >= 1 for t in out):
        raise ValueError("--treatments indices must be >= 1")
    return sorted(set(out))


def true_ate_for_treatment(treatment: int) -> float:
    """Vary the constant treatment-effect component smoothly across treatments
    so the smoketest battery sees a non-trivial range of true ATEs (~1.0..5.0)
    instead of every dataset hitting the same point."""
    return round(1.0 + 4.0 * ((treatment - 1) % 77) / 76.0, 4)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=42, help="Master RNG seed (default: 42).")
    parser.add_argument("--n", type=int, default=4802, help="Sample size (default: 4802, matches ACIC 2016).")
    parser.add_argument(
        "--treatments",
        type=str,
        default="1",
        help='Treatment indices: a single int ("1"), a comma list ("1,3,5"), '
             'or an inclusive range ("1-77"). Default: "1".',
    )
    parser.add_argument(
        "--responses-per-treatment",
        type=int,
        default=1,
        help="Number of zymu_<R>.csv files to write per treatment, with R = 1..K. Default: 1.",
    )
    parser.add_argument(
        "--true-ate",
        type=float,
        default=None,
        help="Override the constant treatment-effect component. Default: a smooth "
             "function of the treatment index so the battery covers a range of true ATEs.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files in eval_datasets/acic2016/. Without this "
             "flag the script refuses to clobber a populated directory — this "
             "protects the real ACIC-2016 bundle from being silently replaced "
             "with synthetic data, which would leave x.csv inconsistent with "
             "the original zymu_*.csv files and produce nonsense estimates.",
    )
    args = parser.parse_args()

    treatments = parse_treatments(args.treatments)
    k = args.responses_per_treatment
    if k < 1:
        parser.error("--responses-per-treatment must be >= 1")

    root = _project_root() / "eval_datasets" / "acic2016"
    root.mkdir(parents=True, exist_ok=True)

    x_path = root / "x.csv"
    if x_path.exists() and not args.force:
        parser.error(
            f"{x_path.relative_to(_project_root())} already exists. Refusing to "
            f"overwrite — the real ACIC-2016 bundle and any previously-generated "
            f"synthetic data live here. Pass --force to overwrite, or clear the "
            f"directory first if you really want a fresh synthetic run."
        )

    # Covariates are shared across the whole bundle, generated once from the
    # master seed so re-runs are deterministic.
    rng_cov = np.random.default_rng(args.seed)
    X = generate_covariates(rng_cov, args.n)
    X.to_csv(x_path, index=False)
    print(f"wrote {x_path.relative_to(_project_root())}  ({X.shape[0]} rows x {X.shape[1]} cols)")

    total_files = 0
    for t in treatments:
        treatment_dir = root / str(t)
        treatment_dir.mkdir(parents=True, exist_ok=True)
        true_ate = args.true_ate if args.true_ate is not None else true_ate_for_treatment(t)
        for r in range(1, k + 1):
            # Each (treatment, response) gets a derived seed so adding more
            # responses later doesn't perturb the earlier files.
            zymu_seed = args.seed + 1000 * t + r
            rng_zr = np.random.default_rng(zymu_seed)
            zymu = generate_zymu(rng_zr, X, true_ate=true_ate)
            zymu_path = treatment_dir / f"zymu_{r}.csv"
            zymu.to_csv(zymu_path, index=False)
            total_files += 1

            if len(treatments) == 1 and k == 1:
                realised_ate = float(np.mean(zymu["mu1"] - zymu["mu0"]))
                treated = zymu["z"] == 1
                realised_att = float(np.mean(zymu.loc[treated, "mu1"] - zymu.loc[treated, "mu0"]))
                print(f"wrote {zymu_path.relative_to(_project_root())}  ({zymu.shape[0]} rows x {zymu.shape[1]} cols)")
                print(f"realised true ATE = {realised_ate:+.4f}  (parameter true_ate={true_ate})")
                print(f"realised true ATT = {realised_att:+.4f}")
                print(f"share treated (z=1) = {treated.mean():.3f}")

    if len(treatments) > 1 or k > 1:
        print(
            f"wrote {total_files} zymu files across {len(treatments)} treatment(s) "
            f"x {k} response(s) each."
        )


if __name__ == "__main__":
    main()
