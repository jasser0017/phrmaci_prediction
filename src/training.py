# src/preprocessing/training.py

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import lightgbm as lgb
from statsmodels.tsa.statespace.sarimax import SARIMAX
from evaluation import evaluate_model
import warnings
warnings.filterwarnings("ignore")


def tune_sarimax_parameters(y_train, y_test, p_range=(0, 2), d_range=(0, 1), q_range=(0, 2)):
    """
    Recherche les meilleurs paramètres SARIMAX (p, d, q) en testant toutes les combinaisons.
    Retourne le modèle ayant le meilleur RMSE sur y_test.
    """
    best_rmse = float("inf")
    best_order = None
    best_model = None

    for p in range(*p_range):
        for d in range(*d_range):
            for q in range(*q_range):
                try:
                    model = SARIMAX(y_train, order=(p, d, q), seasonal_order=(0, 0, 0, 0),
                                    enforce_stationarity=False, enforce_invertibility=False)
                    result = model.fit(disp=False)
                    forecast = result.forecast(steps=len(y_test))
                    rmse = np.sqrt(mean_squared_error(y_test, forecast))

                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_order = (p, d, q)
                        best_model = result

                except Exception as e:
                    continue

    print(f"\n🧪 Meilleur SARIMAX trouvé: order={best_order}, RMSE={best_rmse:.2f}")
    return best_model, best_order, best_rmse


def train_all_models(X_train, y_train, X_test, y_test, model_save_path='models/best_model.joblib', n_iter=10, n_splits=3, random_state=42):
    models = {
        "LinearRegression": (LinearRegression(), {}),
        "Ridge": (Ridge(), {"alpha": np.logspace(-3, 3, 10)}),
        "XGBoost": (XGBRegressor(objective='reg:squarederror', random_state=random_state), {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1]
        }),
        "LightGBM": (lgb.LGBMRegressor(random_state=random_state), {
            "n_estimators": [50, 100, 200],
            "max_depth": [-1, 5, 10],
            "learning_rate": [0.01, 0.05, 0.1]
        }),
        "HistGradientBoosting": (HistGradientBoostingRegressor(random_state=random_state), {
            "learning_rate": [0.01, 0.05, 0.1],
            "max_iter": [100, 200],
            "max_depth": [None, 5, 10]
        })
    }

    results = []
    best_model = None
    best_score = float("inf")
    tscv = TimeSeriesSplit(n_splits=n_splits)

    for name, (model, param_grid) in models.items():
        print(f"\n🔍 Entraînement du modèle: {name}")

        if param_grid:
            search = RandomizedSearchCV(model, param_distributions=param_grid,
                                        n_iter=n_iter, cv=tscv, scoring="neg_root_mean_squared_error",
                                        random_state=random_state, n_jobs=-1, verbose=0)
            search.fit(X_train, y_train)
            best_estimator = search.best_estimator_
        else:
            model.fit(X_train, y_train)
            best_estimator = model

        y_pred = best_estimator.predict(X_test)
        metrics = evaluate_model(y_test, y_pred, model_name=name)

        results.append((name, best_estimator, metrics))

        if metrics['rmse'] < best_score:
            best_score = metrics['rmse']
            best_model = best_estimator

    try:
        print("\n🔍 Tuning des paramètres SARIMAX...")
        best_sarimax, best_order, best_sarimax_rmse = tune_sarimax_parameters(y_train, y_test)
        forecast = best_sarimax.forecast(steps=len(y_test))
        metrics = evaluate_model(y_test, forecast, model_name=f"SARIMAX{best_order}")

        results.append((f"SARIMAX{best_order}", best_sarimax, metrics))

        if metrics['rmse'] < best_score:
            best_score = metrics['rmse']
            best_model = best_sarimax

    except Exception as e:
        print(f"❌ Erreur SARIMAX : {e}")

    print(f"\n💾 Meilleur modèle sauvegardé sous : {model_save_path}")
    joblib.dump(best_model, model_save_path)

    return best_model, best_score, results
