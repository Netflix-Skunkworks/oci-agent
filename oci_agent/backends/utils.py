# This module contains utility functions, such as for data preprocessing.

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_bool_dtype, is_datetime64_any_dtype


def make_numeric_ml_dataframe(
    df: pd.DataFrame,
    *,
    drop_first: bool = False,
    datetime_features: bool = True,
    preserve_index: bool = True
) -> pd.DataFrame:
    """
    Convert a mixed-type pandas DataFrame into a fully numeric DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with no missing values.
    
    drop_first : bool, default=False
        Whether to drop the first level of each categorical variable.
        Use True for unregularized linear/logistic regression to reduce collinearity.
        Use False for tree models, regularized models, and most ML workflows.
    
    datetime_features : bool, default=True
        Whether to expand datetime columns into numeric components.
    
    preserve_index : bool, default=True
        Whether to preserve the original dataframe index.

    Returns
    -------
    pd.DataFrame
        Fully numeric dataframe.
    """

    if df.isna().any().any():
        raise ValueError("Input dataframe contains missing values.")

    df_work = df.copy()

    numeric_parts = []

    # Numeric columns
    numeric_cols = df_work.select_dtypes(include=["number"]).columns
    if len(numeric_cols) > 0:
        numeric_parts.append(df_work[numeric_cols])

    # Boolean columns
    bool_cols = df_work.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        bool_df = df_work[bool_cols].astype(int)
        numeric_parts.append(bool_df)

    # Datetime columns
    datetime_cols = df_work.select_dtypes(
        include=["datetime64[ns]", "datetimetz"]
    ).columns

    if datetime_features and len(datetime_cols) > 0:
        dt_parts = []

        for col in datetime_cols:
            s = df_work[col]

            dt_part = pd.DataFrame(
                {
                    f"{col}_year": s.dt.year,
                    f"{col}_month": s.dt.month,
                    f"{col}_day": s.dt.day,
                    f"{col}_dayofweek": s.dt.dayofweek,
                    f"{col}_dayofyear": s.dt.dayofyear,
                    f"{col}_quarter": s.dt.quarter,
                    f"{col}_is_month_start": s.dt.is_month_start.astype(int),
                    f"{col}_is_month_end": s.dt.is_month_end.astype(int),
                },
                index=df_work.index
            )

            # Add time-of-day features only if they vary
            if not (s.dt.hour.eq(0).all() and s.dt.minute.eq(0).all() and s.dt.second.eq(0).all()):
                dt_part[f"{col}_hour"] = s.dt.hour
                dt_part[f"{col}_minute"] = s.dt.minute
                dt_part[f"{col}_second"] = s.dt.second

            dt_parts.append(dt_part)

        numeric_parts.append(pd.concat(dt_parts, axis=1))

    # Object and category columns
    categorical_cols = df_work.select_dtypes(
        include=["object", "category", "string"]
    ).columns

    if len(categorical_cols) > 0:
        categorical_df = pd.get_dummies(
            df_work[categorical_cols],
            drop_first=drop_first,
            dtype=int
        )
        numeric_parts.append(categorical_df)

    # Any unsupported columns
    handled_cols = set(numeric_cols) | set(bool_cols) | set(datetime_cols) | set(categorical_cols)
    unsupported_cols = [col for col in df_work.columns if col not in handled_cols]

    if unsupported_cols:
        raise TypeError(
            f"Unsupported column types found: {unsupported_cols}. "
            "Convert these manually before calling this function."
        )

    if not numeric_parts:
        raise ValueError("No usable columns found.")

    out = pd.concat(numeric_parts, axis=1)

    if not preserve_index:
        out = out.reset_index(drop=True)

    return out


def summarize(cate_hat, alpha=0.05):
    from scipy import stats

    est = np.mean(cate_hat)
    stderr = np.std(cate_hat) / np.sqrt(len(cate_hat))
    z = stats.norm.ppf(1 - alpha / 2)
    return est, stderr, (est - z * stderr, est + z * stderr)


def standardized_mean_differences(df, treatment, covariates):
    """
    Compute standardized mean differences for binary treatment.

    SMD = (mean_treated - mean_control) / pooled_sd
    """

    if not covariates:
        return pd.DataFrame(columns=["feature", "mean_treated", "mean_control", "smd", "abs_smd"])

    t = pd.Series(treatment).reset_index(drop=True)
    d = pd.DataFrame(df[covariates]).reset_index(drop=True)

    rows = []

    for col in covariates:
        x1 = d.loc[t == 1, col]
        x0 = d.loc[t == 0, col]

        mean1 = x1.mean()
        mean0 = x0.mean()

        var1 = x1.var(ddof=1)
        var0 = x0.var(ddof=1)

        pooled_sd = np.sqrt((var1 + var0) / 2)

        smd = np.nan if pooled_sd == 0 else (mean1 - mean0) / pooled_sd

        rows.append({
            "feature": col,
            "mean_treated": mean1,
            "mean_control": mean0,
            "smd": smd,
            "abs_smd": abs(smd)
        })

    return (
        pd.DataFrame(rows)
        .sort_values("abs_smd", ascending=False)
        .reset_index(drop=True)
    )


def weighted_mean(x, w):
    return np.sum(w * x) / np.sum(w)


def weighted_var(x, w):
    mu = weighted_mean(x, w)
    return np.sum(w * (x - mu) ** 2) / np.sum(w)


def weighted_standardized_mean_differences(df, treatment, weights, covariates):
    """
    Compute weighted standardized mean differences for binary treatment.
    """

    if not covariates:
        return pd.DataFrame(columns=["feature", "weighted_mean_treated", "weighted_mean_control", "weighted_smd", "abs_weighted_smd"])

    t = pd.Series(treatment).reset_index(drop=True)
    w = pd.Series(weights).reset_index(drop=True)
    d = pd.DataFrame(df[covariates]).reset_index(drop=True)

    rows = []

    for col in covariates:
        x = d[col].to_numpy()
        tt = t.to_numpy()
        ww = w.to_numpy()

        x1 = x[tt == 1]
        x0 = x[tt == 0]

        w1 = ww[tt == 1]
        w0 = ww[tt == 0]

        mean1 = weighted_mean(x1, w1)
        mean0 = weighted_mean(x0, w0)

        var1 = weighted_var(x1, w1)
        var0 = weighted_var(x0, w0)

        pooled_sd = np.sqrt((var1 + var0) / 2)

        smd = np.nan if pooled_sd == 0 else (mean1 - mean0) / pooled_sd

        rows.append({
            "feature": col,
            "weighted_mean_treated": mean1,
            "weighted_mean_control": mean0,
            "weighted_smd": smd,
            "abs_weighted_smd": abs(smd)
        })

    return (
        pd.DataFrame(rows)
        .sort_values("abs_weighted_smd", ascending=False)
        .reset_index(drop=True)
    )


def is_probably_continuous(
    s: pd.Series,
    *,
    sample_size: int = 10_000,
    min_unique: int = 20,
    min_unique_ratio: float = 0.05,
    random_state: int = 123,
) -> bool:
    """
    Heuristically determine whether a pandas Series is probably continuous.

    This avoids scanning the full column by using at most `sample_size` rows.

    Parameters
    ----------
    s : pd.Series
        Column to classify.

    sample_size : int
        Maximum number of rows to inspect.

    min_unique : int
        Minimum number of distinct sampled values required to call numeric data continuous.

    min_unique_ratio : float
        Minimum sampled unique-value share required to call numeric data continuous.

    random_state : int
        Random seed for sampling.

    Returns
    -------
    bool
        True if the column appears continuous; False otherwise.
    """

    # Clearly non-continuous types
    if is_bool_dtype(s) or is_datetime64_any_dtype(s):
        return False

    # Object/category/string columns are usually categorical
    if not is_numeric_dtype(s):
        return False

    n = len(s)

    if n == 0:
        return False

    # Sample without scanning the whole dataset
    if n > sample_size:
        x = s.sample(sample_size, random_state=random_state)
    else:
        x = s

    x = x.dropna()

    if len(x) == 0:
        return False

    unique_count = x.nunique(dropna=True)
    unique_ratio = unique_count / len(x)

    return (unique_count >= min_unique) and (unique_ratio >= min_unique_ratio)


def att_ipw_weights(propensity_scores, treatment, eps=1e-6, normalize=False):
    """
    Compute ATT inverse-propensity weights.

    Parameters
    ----------
    propensity_scores : array-like
        Estimated Pr(T=1 | X).

    treatment : array-like
        Binary treatment indicator, coded 0/1.

    eps : float
        Clipping threshold to avoid division by zero.

    normalize : bool
        If True, normalize weights so treated and control groups each sum to 1
        within group. This is often useful for balance diagnostics.

    Returns
    -------
    np.ndarray
        ATT IPW weights.
    """

    e = np.asarray(propensity_scores, dtype=float)
    t = np.asarray(treatment, dtype=int)

    if not np.all(np.isin(t, [0, 1])):
        raise ValueError("treatment must be binary and coded as 0/1.")

    e = np.clip(e, eps, 1 - eps)

    weights = np.where(
        t == 1,
        1.0,
        e / (1.0 - e)
    )

    if normalize:
        treated = t == 1
        control = t == 0

        weights[treated] = weights[treated] / weights[treated].sum()
        weights[control] = weights[control] / weights[control].sum()

    return weights


def trim_by_propensity(ps, T, lower=0.01, upper=0.99):
    ps = np.asarray(ps)
    T = np.asarray(T)

    keep = (ps >= lower) & (ps <= upper)
    return keep
