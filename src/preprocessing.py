
"""
Module : preprocessing

Objet  : Prétraitement des données : nettoyage, typage, gestion des outliers et des valeurs manquantes
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import os
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

def load_data(path):
    df = pd.read_csv(path, parse_dates=["Date"])
    df_main = df[~((df['Date'].dt.year == 2025) & (df['Date'].dt.month == 1))]

    return df_main

def quick_explore(df):
    print("\n--- Aperçu du dataset ---")
    print(df.head())
    print("\n--- Info ---")
    print(df.info())
    print("\n--- Statistiques ---")
    print(df.describe())
    print("\n--- Valeurs manquantes ---")
    print(df.isna().sum())



def handle_negative_values(df, cols_to_check):
    print("\n🔍 Vérification des valeurs négatives :")
    for col in cols_to_check:
        count_neg = (df[col] < 0).sum()
        if count_neg > 0:
            print(f"⚠️ {count_neg} valeur(s) négative(s) détectée(s) dans '{col}' — remplacement par NaN.")
            df.loc[df[col] < 0, col] = pd.NA
        else:
            print(f"✅ Pas de valeurs négatives dans '{col}'")
    return df


def show_missing_values(df, output_path="outputs/visualisations/missing_values.png"):
    msno.matrix(df)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"✅ Visualisation des valeurs manquantes sauvegardée sous : {output_path}")




def simulate_missing_values(col, missing_rate=0.1, random_state=42):
    np.random.seed(random_state)
    col = col.copy()
    non_nan_idx = col.dropna().index
    mask_size = int(len(non_nan_idx) * missing_rate)
    masked_idx = np.random.choice(non_nan_idx, size=mask_size, replace=False)
    col_masked = col.copy()
    col_masked[masked_idx] = np.nan
    return col_masked, masked_idx


def evaluate_imputation(original, imputed):
    mask = ~original.isna()
    return np.sqrt(mean_squared_error(original[mask], imputed[mask]))

def apply_imputation_strategy(df, col_name, strategy):
    if strategy == 'mean':
        imputer = SimpleImputer(strategy='mean')
        return pd.Series(imputer.fit_transform(df[[col_name]]).ravel(), index=df.index)
    elif strategy == 'median':
        imputer = SimpleImputer(strategy='median')
        return pd.Series(imputer.fit_transform(df[[col_name]]).ravel(), index=df.index)
    elif strategy == 'knn':
        imputer = KNNImputer(n_neighbors=5)
        return pd.Series(imputer.fit_transform(df[[col_name]])[:, 0], index=df.index)
    elif strategy == 'forward_fill':
        return df[col_name].fillna(method='ffill')
    elif strategy == 'linear':
        return df[col_name].interpolate(method='linear')
    else:
        raise ValueError(f"Stratégie d'imputation inconnue : {strategy}")


def select_best_imputation_method_per_column(df, columns, verbose=True):
    strategies = ['mean', 'median', 'knn', 'forward_fill', 'linear']
    best_methods = {}

    for col in columns:
        if df[col].isna().sum() == 0:
            continue

        errors = {}
        base_col = df[col]
        simulated_col, masked_idx = simulate_missing_values(base_col)

        for strategy in strategies:
            try:
                temp_df = df.copy()
                temp_df[col] = simulated_col
                imputed_col = apply_imputation_strategy(temp_df, col, strategy)
                error = evaluate_imputation(base_col, imputed_col)
                errors[strategy] = error
            except Exception as e:
                errors[strategy] = float('inf')
                if verbose:
                    print(f"❌ Erreur avec {strategy} sur {col}: {e}")

        best_method = min(errors, key=errors.get)
        best_methods[col] = best_method

        if verbose:
            print(f"✅ {col} → meilleure méthode : {best_method} (RMSE={errors[best_method]:.4f})")

    return best_methods
def impute_columns_with_best_methods(df, best_methods, verbose=True):
    print("\n🚀 Imputation finale avec les meilleures méthodes sélectionnées :")

    for col, method in best_methods.items():
        if verbose:
            print(f"🔧 Colonne '{col}' imputation par '{method}'...")

        df[col] = apply_imputation_strategy(df, col, method)

    print("✅ Imputation finale terminée.")
    return df


def compute_cv_threshold(df, target_col="Total", verbose=True):
    """
    Calcule le coefficient de variation (std / mean) pour chaque produit
    puis déduit un seuil de filtrage à l'aide de l'IQR.
    """
    grouped = df.groupby("Code prdt")[target_col]
    std_sales = grouped.std(ddof=0)
    avg_sales = grouped.mean()

    cv = std_sales / avg_sales.replace(0, np.nan)
    cv_values = cv.dropna()

    Q1_cv, Q3_cv = np.percentile(cv_values, [25, 75])
    IQR_cv = Q3_cv - Q1_cv
    threshold = Q3_cv + 1.5 * IQR_cv

    if verbose:
        print(f"📐 Seuil IQR du coefficient de variation (CV) : {threshold:.4f}")

    return cv, threshold

def max_gap_months(date_series):
    """
    Calcule le plus grand écart (en mois) entre deux dates successives.
    Conversion explicite de la série en datetime.
    """
    dates = pd.to_datetime(date_series).sort_values()
    max_gap = 0
    for i in range(1, len(dates)):
        prev, curr = dates.iloc[i - 1], dates.iloc[i]
        gap = (curr.year - prev.year) * 12 + (curr.month - prev.month) - 1
        if gap > max_gap:
            max_gap = gap
    return max_gap


def filter_products_advanced(df, max_nan_ratio=0.1, min_months=20, min_total_mean=50, max_gap_allowed=2, verbose=True):
    """
    Filtre les produits non pertinents selon plusieurs critères :
    - Proportion maximale de NaN
    - Minimum d'historique en mois
    - Moyenne minimale du Total
    - Coefficient de variation (CV) par IQR
    - Nombre maximal de mois consécutifs sans donnée
    """
    produits_valides = []
    grouped = df.groupby("Code prdt")
    cv_series, cv_threshold = compute_cv_threshold(df, target_col="Total", verbose=verbose)

    for prdt, group in grouped:
        total_series = group["Total"]
        date_series = group["Date"]

      
        nan_ratio = total_series.isna().mean()
        if nan_ratio > max_nan_ratio:
            if verbose:
                print(f"❌ {prdt} supprimé (trop de NaN : {nan_ratio:.2%})")
            continue

        
        if len(group) < min_months:
            if verbose:
                print(f"❌ {prdt} supprimé (série trop courte : {len(group)} mois)")
            continue

       
        if total_series.mean() < min_total_mean:
            if verbose:
                print(f"❌ {prdt} supprimé (moyenne Total trop faible : {total_series.mean():.2f})")
            continue

        
        if prdt in cv_series and cv_series[prdt] > cv_threshold:
            if verbose:
                print(f"❌ {prdt} supprimé (CV = {cv_series[prdt]:.2f} > seuil {cv_threshold:.2f})")
            continue

        gap_months = max_gap_months(date_series)
        if gap_months > max_gap_allowed:
            if verbose:
                print(f"❌ {prdt} supprimé (gap = {gap_months} mois > {max_gap_allowed})")
            continue

        produits_valides.append(prdt)

    df_filtré = df[df["Code prdt"].isin(produits_valides)].copy()

    if verbose:
        print(f"\n✅ Produits conservés : {len(produits_valides)} / {df['Code prdt'].nunique()}")

    return df_filtré











def main():
    path = "data/clean_data_total.csv"
    df_main = load_data(path)
    handle_negative_values(df_main, ["vente","stock","Total","feature"])
    quick_explore(df_main)



    df = filter_products_advanced(df_main,
    max_nan_ratio=0.1,
    min_months=20,
    min_total_mean=50,
    max_gap_allowed=2)
    quick_explore(df)
    



    




    
 



   
    #df.to_csv("data/df_cleaned.csv", index=False)
    print("✅ Données chargées et sauvegardées temporairement dans df_cleaned.csv")

if __name__ == "__main__":
    main()
