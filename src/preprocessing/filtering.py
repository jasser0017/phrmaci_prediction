import numpy as np
import pandas as pd

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