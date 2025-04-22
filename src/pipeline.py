
import sys
import os
import pandas as pd

# Assure l'import des modules internes (src/ et src/preprocessing/)
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

# === IMPORTS PROJET ===
from preprocessing import prepro
from preprocessing.data_loader import load_data
from preprocessing.negative_cleaning import handle_negative_values
from preprocessing.filtering import filter_products_advanced
from preprocessing.imputation import evaluate_imputation_methods_for_product, apply_global_imputations,find_best_imputation_methods,apply_best_imputations_by_group
from preprocessing.outlier_detection import find_best_methods_for_df
from preprocessing.outlier_removal import apply_best_methods

import pandas as pd
from joblib import load
from preprocessing import prepro

from feature_engineering import run_feature_engineering
from clustring import run_kshape_clustering

from modeling import (
    select_predictive_features,
    split_train_test,
    adf_test,
    extract_december_2024_evaluation_set
)
from training import train_all_models
from forecasting import forecast_and_evaluate


# 1. Prétraitement complet (déjà encapsulé)
print("\n🧹 Lancement du prétraitement...")
prepro.main()
df_cleaned = pd.read_csv("outputs/df_cleaned.csv", parse_dates=["Date"])

# 2. Feature Engineering
print("\n⚙️ Feature engineering...")
df_final = run_feature_engineering(df_cleaned)

# 3. Clustering
print("\n🔗 Clustering K-Shape...")
df_final, cluster_map = run_kshape_clustering(df_final)

# 4. Imputation des features dérivés
print("\n🧪 Imputation des lags et rollings...")
cols_derived = [
    'vente_lag1','vente_lag2','stock_lag1','stock_lag2','total_lag1','total_lag2',
    'vente_roll3','stock_roll3','total_roll3','feature_lag1','feature_lag2','feature_roll3'
]
best_methods_derived = find_best_imputation_methods(df_final, "PF009", cols_derived)
df_final = apply_best_imputations_by_group(df_final, best_methods_derived)
df_final.to_csv("outputs/df_featured.csv", index=False)

# 5. Sélection des meilleures features
print("\n📊 Sélection des variables prédictives...")
top_corr = select_predictive_features(df_final, target_col="Total")
pd.DataFrame({"feature": top_corr}).to_csv("outputs/top_features.csv", index=False)

df_eval,df_remaining=extract_december_2024_evaluation_set(df_final)

# 6. Split Train/Test
X_train, y_train, X_test, y_test = split_train_test(df_remaining, target_col="Total", keep_cols=["Date", "Total"] + top_corr)

# 7. ADF Test
print("\n🔬 Test de stationnarité sur y_train...")
adf_test(y_train, title="y_train")

# 8. Entraînement des modèles & tuning SARIMAX
print("\n🏋️ Entraînement des modèles...")
best_model, best_score, all_results = train_all_models(X_train, y_train, X_test, y_test)


print("\n📈 Forecasting sur décembre 2024...")

df_forecast, forecast_metrics = forecast_and_evaluate(df_eval, top_corr)

output_path = "outputs/df_forecast_december2024.csv"
df_forecast.to_csv(output_path, index=False)
print(f"\n✅ Prédictions sauvegardées dans : {output_path}")
