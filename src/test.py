from feature_engineering import (run_feature_engineering, generate_lags,generate_rolling,encode_month_cyclic,apply_stl_decomposition)

import pandas as pd

from preprocessing.imputation import apply_best_imputations_by_group, find_best_imputation_methods
from clustring import run_kshape_clustering

df_cleaned=pd.read_csv("outputs/df_cleaned.csv")
product="PF009"
columns_to_impute = ['vente_lag1','vente_lag2','stock_lag1','stock_lag2','total_lag1','total_lag2','vente_roll3','stock_roll3','total_roll3','feature_lag1','feature_lag2','feature_roll3']
print(df_cleaned.head())
df_final = run_feature_engineering(df_cleaned)
jk=run_kshape_clustering(df_final)
print(jk[0]['cluster_label'].value_counts())

#print(df_final.isnull().sum())
#best_methodes=find_best_imputation_methods(df_final,product,columns_to_impute)
#print(best_methodes)
#df=apply_best_imputations_by_group(df_final,best_methodes)
#print(df.isnull().sum())




 







