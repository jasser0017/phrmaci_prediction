# src/preprocessing/outlier_detection.py

import numpy as np
import pandas as pd
from scipy.stats import shapiro, skew
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import PowerTransformer


def outliers_iqr(series, k=2.0):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - k * IQR
    upper_bound = Q3 + k * IQR
    filtered = series[(series >= lower_bound) & (series <= upper_bound)]
    return filtered, lower_bound, upper_bound


def outliers_winsorize(series, limits=(0.05, 0.05)):
    wins_series = winsorize(series, limits=limits)
    wins_series = pd.Series(wins_series, index=series.index)
    return wins_series


def outliers_3sigma(series):
    stat, p_value = shapiro(series)

    if p_value < 0.05:
        transformer = PowerTransformer(method='yeo-johnson', standardize=False)
        series_values = series.values.reshape(-1, 1)
        transformed = transformer.fit_transform(series_values).flatten()

        mu = np.mean(transformed)
        sigma = np.std(transformed)
        lower_bound = mu - 3 * sigma
        upper_bound = mu + 3 * sigma

        mask = (transformed >= lower_bound) & (transformed <= upper_bound)
        filtered_transformed = transformed[mask]
        indices = series.index[mask]

        filtered_original = transformer.inverse_transform(filtered_transformed.reshape(-1, 1)).flatten()
        filtered_series = pd.Series(filtered_original, index=indices)

        return filtered_series, lower_bound, upper_bound, transformer, True
    else:
        mu = series.mean()
        sigma = series.std()
        lower_bound = mu - 3 * sigma
        upper_bound = mu + 3 * sigma
        filtered = series[(series >= lower_bound) & (series <= upper_bound)]
        return filtered, lower_bound, upper_bound, None, False


def evaluate_method(series, method_func, method_name, **kwargs):
    original_len = len(series)

    if method_name == "winsorize":
        filtered = method_func(series, **kwargs)
        changed = np.sum(series != filtered)
        percent_removed = 0
    else:
        filtered, lb, ub, *rest = method_func(series, **kwargs)
        filtered = filtered.dropna()
        percent_removed = 100 * (original_len - len(filtered)) / original_len

    try:
        stat, p_val = shapiro(filtered)
    except Exception as e:
        p_val = np.nan

    skw = skew(filtered)
    return filtered, percent_removed, p_val, skw


def compute_score(p_val, percent_removed):
    percent_kept = 100 - percent_removed
    score = p_val * (percent_kept / 100)
    return score


def find_best_methods_for_df(df, columns_of_interest, group_col="Code prdt"):
    results_summary = {}

    grouped = df.groupby(group_col)

    for prod, group in grouped:
        group = group.sort_values("Date")
        results_summary[prod] = {}

        for col in columns_of_interest:
            series = group[col].dropna()

            if len(series) < 10:
                results_summary[prod][col] = {
                    "IQR": None,
                    "Winsorize": None,
                    "3sigma": None,
                    "Best_method": None
                }
                continue

            filtered_iqr, perc_iqr, p_iqr, skew_iqr = evaluate_method(series, outliers_iqr, "iqr", k=2.0)
            score_iqr = compute_score(p_iqr, perc_iqr)

            filtered_wins, perc_wins, p_wins, skew_wins = evaluate_method(series, outliers_winsorize, "winsorize", limits=(0.05, 0.05))
            score_wins = compute_score(p_wins, perc_wins)

            filtered_3sigma, perc_3sigma, p_3sigma, skew_3sigma = evaluate_method(series, outliers_3sigma, "3sigma")
            score_3sigma = compute_score(p_3sigma, perc_3sigma)

            scores = {
                "IQR": score_iqr,
                "Winsorize": score_wins,
                "3sigma": score_3sigma
            }
            best_method = max(scores, key=scores.get)

            results_summary[prod][col] = {
                "IQR": {
                    "percent_removed": perc_iqr,
                    "p_value": p_iqr,
                    "skew": skew_iqr,
                    "score": score_iqr
                },
                "Winsorize": {
                    "percent_removed": perc_wins,
                    "p_value": p_wins,
                    "skew": skew_wins,
                    "score": score_wins
                },
                "3sigma": {
                    "percent_removed": perc_3sigma,
                    "p_value": p_3sigma,
                    "skew": skew_3sigma,
                    "score": score_3sigma
                },
                "Best_method": best_method
            }

    return results_summary
