from feature_engineering import (run_feature_engineering, generate_lags,generate_rolling,encode_month_cyclic,apply_stl_decomposition)

import pandas as pd

df_cleaned=pd.read_csv("outputs/df_cleaned.csv")
print(df_cleaned.head())
df_final = run_feature_engineering(df_cleaned)
print(df_final.isnull().sum())




 







