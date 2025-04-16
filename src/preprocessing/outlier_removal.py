# src/preprocessing/outlier_removal.py

import pandas as pd
from preprocessing.outlier_detection import (
    outliers_iqr as apply_iqr,
    outliers_winsorize as apply_winsorize,
    outliers_3sigma as apply_3sigma
)


def apply_best_methods(df, results_summary, columns, group_col='Code prdt'):
    cleaned_dfs = []

    for prdt, group in df.groupby(group_col):
        if prdt not in results_summary:
            cleaned_dfs.append(group)
            continue

        group = group.copy()

        for col in columns:
            if col not in results_summary[prdt] or results_summary[prdt][col]["Best_method"] is None:
                continue

            method = results_summary[prdt][col]["Best_method"]
            original = group[col].dropna()

            if len(original) < 3:
                continue

            if method == "IQR":
                cleaned_col, *_ = apply_iqr(original, k=2.0)
            elif method == "Winsorize":
                cleaned_col = apply_winsorize(original, limits=(0.05, 0.05))
                group.loc[cleaned_col.index, col] = cleaned_col
                continue  
            elif method == "3sigma":
                cleaned_col, *_ = apply_3sigma(original)
            else:
                continue

            group = group.loc[group.index.isin(cleaned_col.index)]

        cleaned_dfs.append(group)

    df_cleaned = pd.concat(cleaned_dfs).sort_index()
    return df_cleaned
