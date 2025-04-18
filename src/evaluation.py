import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt


def evaluate_model(y_true, y_pred, model_name=""):
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"[{model_name}] 📊 RMSE = {rmse:.2f}, MAE = {mae:.2f}, R² = {r2:.2f}")

    return {"model": model_name, "rmse": rmse, "mae": mae, "r2": r2}