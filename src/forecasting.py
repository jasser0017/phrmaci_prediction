import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def forecast_and_evaluate(df_eval, top_corr, model_path="models/best_model.joblib", target_col="Total"):
    
    model = joblib.load(model_path)

   
    features = [col for col in top_corr if col in df_eval.columns]
    X_eval = df_eval[features]
    y_true = df_eval[target_col]

   
    y_pred = model.predict(X_eval)

    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"📊 Résultats sur les données d'évaluation (Décembre 2024)")
    print(f"✅ RMSE = {rmse:.2f}, MAE = {mae:.2f}, R² = {r2:.2f}")

    # Ajout des prédictions à la DataFrame
    df_eval = df_eval.copy()
    df_eval["Prediction"] = y_pred

    return df_eval, {"rmse": rmse, "mae": mae, "r2": r2}
