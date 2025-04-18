from feature_engineering import (run_feature_engineering, generate_lags,generate_rolling,encode_month_cyclic,apply_stl_decomposition)
from modeling import select_predictive_features,split_train_test,adf_test,extract_december_2024_evaluation_set

import pandas as pd
import joblib

from preprocessing.imputation import apply_best_imputations_by_group, find_best_imputation_methods
from clustring import run_kshape_clustering
from training import tune_sarimax_parameters,train_all_models
from evaluation import evaluate_model

df_cleaned=pd.read_csv("outputs/df_cleaned.csv")
product="PF009"
columns_to_impute = ['vente_lag1','vente_lag2','stock_lag1','stock_lag2','total_lag1','total_lag2','vente_roll3','stock_roll3','total_roll3','feature_lag1','feature_lag2','feature_roll3']
print(df_cleaned.head())
df_final = run_feature_engineering(df_cleaned)
jk=run_kshape_clustering(df_final)
print(jk[0]['cluster_label'].value_counts())


target_col="Total"
print(df_final.isnull().sum())
best_methodes=find_best_imputation_methods(df_final,product,columns_to_impute)
print(best_methodes)
df=apply_best_imputations_by_group(df_final,best_methodes)
print(df.isnull().sum())

top_corr=select_predictive_features(df)
df_eval,df_remaining=extract_december_2024_evaluation_set(df)
X_train, y_train, X_test, y_test=split_train_test(df_remaining,keep_cols=['Date', target_col] + top_corr)
adf_test(y_train)
tune_sarimax_parameters(y_train,y_test)
df55=train_all_models(X_train,y_train,X_test,y_test)
best_model = joblib.load("models/best_model.joblib")
X_cols = [col for col in top_corr]
X = df_eval[X_cols]
y_pred = best_model.predict(X)

df_eval["Prediction"] = y_pred
print(evaluate_model(df_eval['Total'],df_eval['Prediction'],model_name=best_model))









 







