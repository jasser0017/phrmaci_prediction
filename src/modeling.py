
import pandas as pd
import numpy as np



def select_predictive_features(df, target_col='Total', excluded_cols=None):
    if excluded_cols is None:
        excluded_cols = {'Date', target_col}

    candidate_cols = [c for c in df.columns if c not in excluded_cols]
    corrs = []

    for c in candidate_cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            corr = df[[c, target_col]].corr().iloc[0, 1]
            corrs.append((c, abs(corr)))
        else:
            corrs.append((c, 0.0))

    corrs_sorted = sorted(corrs, key=lambda x: x[1], reverse=True)
    top_corr = [x[0] for x in corrs_sorted if x[1] > 0]
    return top_corr


def split_train_test(df, target_col='Total', keep_cols=None, train_ratio=0.8):
    if keep_cols is None:
        raise ValueError("keep_cols must be defined")

    df1 = df[keep_cols].copy()
    train_size = int(len(df1) * train_ratio)

    df_train = df1.iloc[:train_size].copy()
    df_test = df1.iloc[train_size:].copy()

    X_train = df_train.drop(['Date', target_col], axis=1)
    y_train = df_train[target_col].copy()
    X_test = df_test.drop(['Date', target_col], axis=1)
    y_test = df_test[target_col].copy()

    return X_train, y_train, X_test, y_test


def adf_test(series, title=''):
    from statsmodels.tsa.stattools import adfuller
    print(f"\n=== ADF Test: {title} ===")
    result = adfuller(series, autolag='AIC')
    labels = ['ADF Statistic', 'p-value', '# Lags Used', 'Observations']
    out = dict(zip(labels, result[0:4]))
    for k, v in out.items():
        print(f"   {k} = {v}")
    for key, val in result[4].items():
        print(f"   Critical Value {key} = {val}")
    print("=> Stationnaire\n" if result[1] < 0.05 else "=> Non Stationnaire\n")


def extract_december_2024_evaluation_set(df, date_col="Date"):
    df[date_col] = pd.to_datetime(df[date_col])
    df_eval = df[(df[date_col].dt.year == 2024) & (df[date_col].dt.month == 12)].copy()
    df_remaining = df[~((df[date_col].dt.year == 2024) & (df[date_col].dt.month == 12))].copy()
    return df_eval, df_remaining