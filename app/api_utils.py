import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

def get_model_and_features(
    model_path: str = "models/best_model.joblib",
    features_path: str = "outputs/top_features.csv",
    feature_col: str = "feature",
):
    """Charge le meilleur modèle + la liste ordonnée des colonnes utilisées au fit."""
    model = joblib.load(model_path)
    top_features = pd.read_csv(features_path)[feature_col].tolist()
    return model, top_features


def load_featured_data(data_path: str = "outputs/df_featured.csv") -> pd.DataFrame:
    """Charge le dataset déjà enrichi (Date parsée)."""
    return pd.read_csv(data_path, parse_dates=["Date"])

def _safe(val):
    """Retourne NaN si val est NaN; sinon la valeur elle‑même (pour rolling np.mean)."""
    return np.nan if pd.isna(val) else val



def prepare_input_for_forecast(
    df: pd.DataFrame,
    code_prdt: str,
    target_date: datetime,
    top_features: list[str],
) -> pd.DataFrame:
    """Construit une ligne d'entrée prête pour le `.predict()` du modèle.

    Recalcule lags 1/2, rolling‑3, et encode le mois. Propage trend/season.
    """
    df_prod = df[df["Code prdt"] == code_prdt].copy()
    if df_prod.empty:
        raise ValueError(f"Produit {code_prdt} introuvable dans le dataset.")

    df_prod = df_prod.sort_values("Date")
    
    
     
    base_df = df_prod


    last_row = base_df[base_df["Date"] < target_date].iloc[-1:].copy()
    if last_row.empty:
        raise ValueError("Pas de données historiques suffisantes pour générer la ligne future.")

    new = last_row.copy()
    new["Date"] = target_date
    new["month"] = target_date.month
    new["year"] = target_date.year
    new["month_sin"] = np.sin(2 * np.pi * target_date.month / 12)
    new["month_cos"] = np.cos(2 * np.pi * target_date.month / 12)

 
    lag1_row = base_df.iloc[-1]
    lag2_row = base_df.iloc[-2] if len(base_df) >= 2 else lag1_row
    for base in ["vente", "stock", "Total", "feature"]:
        new[f"{base}_lag1"] = _safe(lag1_row[base])
        new[f"{base}_lag2"] = _safe(lag2_row[base])

    window = base_df.tail(3)[["vente", "stock", "Total", "feature"]].mean()
    for base in ["vente", "stock", "Total", "feature"]:
        new[f"{base}_roll3"] = _safe(window[base])

    for base in ["vente", "stock", "Total", "feature"]:
        for comp in ["trend", "season"]:
            col_name = f"{base}_{comp}"
            if col_name in last_row:
                new[col_name] = last_row[col_name].values[0]

    if "cluster_label" in last_row:
        new["cluster_label"] = int(last_row["cluster_label"].values[0])

    new.reset_index(drop=True, inplace=True)

    missing = [c for c in top_features if c not in new.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")

    return new[top_features]


def recursive_forecast(
    model,
    df_prod: pd.DataFrame,
    start_date: datetime,
    target_date: datetime,
    top_features: list[str],
):
    """Prédit de manière incrémentale mois par mois jusqu'à *target_date*.

    Parameters
    ----------
    model : sklearn‑like (méthode predict)
    df_prod : dataframe du produit déjà feature‑eng (jusqu'à start_date inclus)
    start_date : dernière date observée dans df_prod
    target_date : date finale à prédire
    top_features : ordre des colonnes attendues par le modèle

    Returns
    -------
    float  -> prédiction pour target_date
    """
    current_date = (start_date + pd.offsets.MonthBegin(1)).to_pydatetime()
    last_pred = None

    while current_date <= target_date:
        X_in = prepare_input_for_forecast(df_prod, df_prod["Code prdt"].iloc[0], current_date, top_features)
        last_pred = float(model.predict(X_in)[0])

        # On ajoute la prédiction comme si c'était une vraie observation pour l'itération suivante
        new_line = X_in.copy()
        new_line["Total"] = last_pred
        new_line["Code prdt"] = df_prod["Code prdt"].iloc[0]
        new_line["Date"] = current_date
        df_prod = pd.concat([df_prod, new_line], ignore_index=True)

        current_date = (pd.Timestamp(current_date) + pd.offsets.MonthBegin(1)).to_pydatetime()

    return last_pred