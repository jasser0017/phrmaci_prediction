

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

def generate_lags(df, group_col="Code prdt"):
    df["Date"] = pd.to_datetime(df["Date"])
    lagged = df.copy()
    lagged['vente_lag1'] = lagged.groupby(group_col)['vente'].shift(1)
    lagged['vente_lag2'] = lagged.groupby(group_col)['vente'].shift(2)
    lagged['stock_lag1'] = lagged.groupby(group_col)['stock'].shift(1)
    lagged['stock_lag2'] = lagged.groupby(group_col)['stock'].shift(2)
    lagged['total_lag1'] = lagged.groupby(group_col)['Total'].shift(1)
    lagged['total_lag2'] = lagged.groupby(group_col)['Total'].shift(2)
    lagged['feature_lag1'] = lagged.groupby(group_col)['feature'].shift(1)
    lagged['feature_lag2'] = lagged.groupby(group_col)['feature'].shift(2)
    return lagged

def generate_rolling(df, group_col="Code prdt"):
    rolled = df.copy()
    rolled['vente_roll3'] = rolled.groupby(group_col)['vente'].transform(lambda s: s.shift(1).rolling(window=3, min_periods=1).mean())
    rolled['stock_roll3'] = rolled.groupby(group_col)['stock'].transform(lambda s: s.shift(1).rolling(window=3, min_periods=1).mean())
    rolled['total_roll3'] = rolled.groupby(group_col)['Total'].transform(lambda s: s.shift(1).rolling(window=3, min_periods=1).mean())
    rolled['feature_roll3'] = rolled.groupby(group_col)['feature'].transform(lambda s: s.shift(1).rolling(window=3, min_periods=1).mean())
    return rolled

def encode_month_cyclic(df):
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    return df

def apply_stl_decomposition(df, group_col="Code prdt"):
    df = df.copy()
    for col in ['vente', 'stock', 'Total', 'feature']:
        df[f'{col}_trend'] = np.nan
        df[f'{col}_season'] = np.nan

    for prod, group in df.groupby(group_col):
        groupe_ordre = group.sort_values('Date')
        for col in ['vente', 'stock', 'Total', 'feature']:
            try:
                stl = STL(groupe_ordre[col], period=12, robust=True).fit()
                df.loc[groupe_ordre.index, f'{col}_trend'] = stl.trend
                df.loc[groupe_ordre.index, f'{col}_season'] = stl.seasonal
            except Exception as e:
                print(f"⚠️ STL decomposition failed for {col} in product {prod}: {e}")
    return df

def run_feature_engineering(df):
    df = generate_lags(df)
    df = generate_rolling(df)
    df = encode_month_cyclic(df)
    df = apply_stl_decomposition(df)
    return df
