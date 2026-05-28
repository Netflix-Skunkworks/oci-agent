# This module contains helper functions for the EconML notebook.

import numpy as np
import pandas as pd


def make_nuisance_design_matrix(X=None, W=None):
    """
    Construct the covariate matrix used by EconML nuisance models.

    If both X and W are supplied, EconML nuisance models generally use
    the concatenation of X and W.
    """
    if X is None and W is None:
        raise ValueError("At least one of X or W must be provided.")

    parts = []
    colnames = []

    if X is not None:
        X_df = pd.DataFrame(X).copy()
        X_df.columns = [f"X__{c}" for c in X_df.columns]
        parts.append(X_df.reset_index(drop=True))
        colnames.extend(X_df.columns)

    if W is not None:
        W_df = pd.DataFrame(W).copy()
        W_df.columns = [f"W__{c}" for c in W_df.columns]
        parts.append(W_df.reset_index(drop=True))
        colnames.extend(W_df.columns)

    Z = pd.concat(parts, axis=1)
    return Z


def extract_feature_importances_from_econml_models(
    models_nested,
    feature_names,
    *,
    importance_attr="feature_importances_"
):
    """
    Average feature importances across EconML cross-fitted nuisance models.

    Parameters
    ----------
    models_nested : nested list
        Example: est.models_propensity or est.models_regression.

    feature_names : list-like
        Names of features used by the nuisance model.

    importance_attr : str
        Usually "feature_importances_" for tree models or "coef_" for linear models.

    Returns
    -------
    pd.DataFrame
        Feature importance table sorted descending.
    """

    rows = []

    for mc_idx, fold_models in enumerate(models_nested):
        for fold_idx, model in enumerate(fold_models):
            if not hasattr(model, importance_attr):
                raise AttributeError(
                    f"Model of type {type(model)} does not have `{importance_attr}`. "
                    "Use permutation importance instead."
                )

            vals = getattr(model, importance_attr)

            # Handle linear/logistic coefficients
            vals = np.asarray(vals)

            if vals.ndim == 2:
                # For multiclass classifiers or multi-output regressors, aggregate across outputs.
                vals = np.mean(np.abs(vals), axis=0)
            else:
                vals = np.abs(vals)

            rows.append(
                pd.DataFrame({
                    "feature": feature_names,
                    "importance": vals,
                    "mc_iteration": mc_idx,
                    "fold": fold_idx
                })
            )

    long = pd.concat(rows, ignore_index=True)

    out = (
        long
        .groupby("feature", as_index=False)
        .agg(
            mean_importance=("importance", "mean"),
            sd_importance=("importance", "std"),
            min_importance=("importance", "min"),
            max_importance=("importance", "max")
        )
        .sort_values("mean_importance", ascending=False)
        .reset_index(drop=True)
    )

    denom = out["mean_importance"].sum()
    if denom > 0:
        out["relative_importance"] = out["mean_importance"] / denom
    else:
        out["relative_importance"] = np.nan

    return out


def _reproduce_drlearner_cv_folds(est, T):
    """Reproduce the (train_idx, test_idx) splits a discrete-treatment DRLearner
    used at fit time, so that each row's nuisance prediction can be routed
    through the fold model that did NOT see it during training.

    EconML's `_OrthoLearner._fit_nuisances` builds the splitter as
    `check_cv(self.cv, ..., classifier=stratify)` and then sets
    `splitter.random_state = self._random_state` where
    `self._random_state = check_random_state(self.random_state)`. For
    discrete-treatment DRLearner with an integer `cv`, this yields a
    `StratifiedKFold(n_splits=cv, shuffle=True, random_state=...)` stratified
    on T. Between `check_random_state(...)` and the splitter consuming the rng,
    no other call advances the rng — so we get the exact same folds by
    re-running that construction here.
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.utils import check_random_state

    if est.mc_iters not in (None, 1):
        raise NotImplementedError(
            "OOF nuisance extraction currently supports only mc_iters in "
            f"{{None, 1}}; got mc_iters={est.mc_iters!r}."
        )
    if not isinstance(est.cv, int):
        raise NotImplementedError(
            "OOF nuisance extraction currently supports only integer `cv`; "
            f"got cv={est.cv!r}."
        )

    rng = check_random_state(est.random_state)
    splitter = StratifiedKFold(n_splits=est.cv, shuffle=True, random_state=rng)
    T_arr = np.asarray(T).ravel()
    return list(splitter.split(np.zeros((T_arr.shape[0], 1)), T_arr))


def _oof_predict(fold_models, folds, Z, predict_fn):
    if len(fold_models) != len(folds):
        raise RuntimeError(
            f"Got {len(fold_models)} fold models but reproduced {len(folds)} folds."
        )

    out = np.full(Z.shape[0], np.nan, dtype=float)

    for (_, test_idx), model in zip(folds, fold_models):
        pred = np.asarray(predict_fn(model, Z[test_idx])).reshape(-1)
        if pred.shape[0] != len(test_idx):
            raise RuntimeError(
                f"Prediction length mismatch: got {pred.shape[0]}, "
                f"expected {len(test_idx)}."
            )
        out[test_idx] = pred

    if np.isnan(out).any():
        raise RuntimeError("OOF prediction left some rows unset.")

    return out


def extract_avg_propensity_scores(est, T, X=None, W=None, treatment_value=1):
    """
    Cross-fitted (out-of-fold) propensity scores from an EconML DR-style
    estimator. Each row's prediction comes from the fold model whose training
    set excluded it — required for valid double-ML inference downstream.

    Parameters
    ----------
    est : fitted EconML DRLearner or LinearDRLearner
        Must have `models_propensity`, `cv`, `random_state`, `mc_iters` attrs.
    T : array-like, shape (n,)
        The treatment vector passed to `est.fit`, used to reproduce the
        stratified-KFold splits.
    X, W : array-like or None
        Same X, W passed to `est.fit`.
    treatment_value : int, default=1
        Class whose propensity to extract.

    Returns
    -------
    np.ndarray, shape (n,)
        OOF predicted P(T = treatment_value | X, W).
    """
    if X is None and W is None:
        raise ValueError("At least one of X or W must be provided.")
    if X is None:
        Z = np.asarray(W)
    elif W is None:
        Z = np.asarray(X)
    else:
        Z = np.column_stack([np.asarray(X), np.asarray(W)])

    folds = _reproduce_drlearner_cv_folds(est, T)
    if len(est.models_propensity) != 1:
        raise NotImplementedError(
            f"Expected one monte-carlo iteration of nuisance models; got "
            f"{len(est.models_propensity)}."
        )
    fold_models = est.models_propensity[0]

    def _predict(model, Z_test):
        proba = model.predict_proba(Z_test)
        if hasattr(model, "classes_"):
            class_idx = list(model.classes_).index(treatment_value)
        else:
            class_idx = treatment_value
        return proba[:, class_idx]

    return _oof_predict(fold_models, folds, Z, _predict)


def _make_econml_treatment_design(est, n, treatment_value):
    if not hasattr(est, "transformer") or est.transformer is None:
        return np.full((n, 1), treatment_value)

    raw_t = np.full((n, 1), treatment_value)
    return est.transformer.transform(raw_t)


def extract_avg_outcome_predictions(est, T, X=None, W=None, treatment_value=1):
    if X is None and W is None:
        raise ValueError("At least one of X or W must be provided.")

    if X is None:
        Z = np.asarray(W)
    elif W is None:
        Z = np.asarray(X)
    else:
        Z = np.column_stack([np.asarray(X), np.asarray(W)])

    n = Z.shape[0]
    T_design = _make_econml_treatment_design(est, n, treatment_value)
    Z_with_T = np.hstack([Z, T_design])

    folds = _reproduce_drlearner_cv_folds(est, T)

    if len(est.models_regression) != 1:
        raise NotImplementedError(
            f"Expected one monte-carlo iteration of nuisance models; got "
            f"{len(est.models_regression)}."
        )

    fold_models = est.models_regression[0]

    return _oof_predict(
        fold_models,
        folds,
        Z_with_T,
        lambda model, Z_test: np.asarray(model.predict(Z_test)).reshape(-1),
    )


def aipw_pseudo_outcome(Y, T, m0_hat, m1_hat, e_hat):
    """Doubly-robust (AIPW) pseudo-outcome for the ATE per unit.

        Y_DR_i = m1(X_i) - m0(X_i)
                 + (T_i / e_i) (Y_i - m1(X_i))
                 - ((1 - T_i) / (1 - e_i)) (Y_i - m0(X_i))

    mean(Y_DR) is an asymptotically normal estimator of ATE; its standard
    error is std(Y_DR) / sqrt(N).
    """
    Y = np.asarray(Y, dtype=float)
    T = np.asarray(T, dtype=float)
    return (
        m1_hat - m0_hat
        + (T / e_hat) * (Y - m1_hat)
        - ((1 - T) / (1 - e_hat)) * (Y - m0_hat)
    )


def att_dr_pseudo_outcome(Y, T, m0_hat, e_hat):
    """ATT influence-function-based pseudo-outcome.

        psi_ATT_i = (1 / pi) * [
            T_i (Y_i - m0(X_i))
            - (1 - T_i) (e_i / (1 - e_i)) (Y_i - m0(X_i))
        ]

    where pi = mean(T). mean(psi_ATT) is the AIPW estimator of the ATT;
    its standard error is std(psi_ATT) / sqrt(N).
    """
    Y = np.asarray(Y, dtype=float)
    T = np.asarray(T, dtype=float)
    pi = float(np.mean(T))
    return (1.0 / pi) * (
        T * (Y - m0_hat)
        - (1 - T) * (e_hat / (1 - e_hat)) * (Y - m0_hat)
    )


def aipw_summary(pseudo, alpha=0.05, centering=None):
    """Estimate, standard error, and (1 - alpha) CI from DR pseudo-outcomes.

    When `centering` is supplied (array, same length as `pseudo`), the SE is
    computed from the centered influence function `psi_i = pseudo_i - est *
    centering_i` instead of from `pseudo` directly. The AIPW ATT estimator
    requires this: its efficient influence function carries an extra
    `-T_i * tau_ATT / pi` term that accounts for estimating `pi = E[T]`.
    """
    from scipy import stats
    pseudo = np.asarray(pseudo, dtype=float)
    n = len(pseudo)
    est = float(np.mean(pseudo))
    if centering is None:
        psi = pseudo
    else:
        psi = pseudo - est * np.asarray(centering, dtype=float)
    se = float(np.std(psi, ddof=1) / np.sqrt(n))
    z = stats.norm.ppf(1 - alpha / 2)
    return est, se, (est - z * se, est + z * se)
