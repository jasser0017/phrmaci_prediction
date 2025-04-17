# src/preprocessing/preprocessing.py

from data_loader import load_data
from negative_cleaning import handle_negative_values
from filtering import filter_products_advanced

from imputation import (
    evaluate_imputation_methods_for_product,
    apply_global_imputations
)
from outlier_detection import find_best_methods_for_df
from outlier_removal import apply_best_methods

import os
import pandas as pd

def main():
    # Étape 1 : Chargement des données
    df = load_data("data/clean_data_total.csv")

    # Étape 2 : Nettoyage des valeurs négatives
    cols = ["vente", "stock", "Total", "feature"]
    df = handle_negative_values(df, cols)

    # Étape 3 : Filtrage des produits valides
    df = filter_products_advanced(df)

    # Étape 4 : Imputation (benchmark sur un produit de référence)
    ref_product = "PF009"  # peut être changé
    print(f"\n Benchmark des méthodes d'imputation sur le produit '{ref_product}'")
    imputation_results = evaluate_imputation_methods_for_product(df, ref_product, cols)
    best_methods = {col: imputation_results[col]['best'] for col in imputation_results}
    df_imputed = apply_global_imputations(df, best_methods, cols)

    # Étape 5 : Détection des meilleures méthodes d’outliers
    print("\n Détection des méthodes d'outliers pour chaque produit/colonne...")
    results_summary = find_best_methods_for_df(df_imputed, columns_of_interest=cols)

    # Étape 6 : Application des nettoyages d’outliers
    print("\n Application du nettoyage des outliers...")
    df_cleaned = apply_best_methods(df_imputed, results_summary, columns=cols)
    

    # Étape 7 : Export des données nettoyées
    os.makedirs("outputs", exist_ok=True)
    output_path = "outputs/df_cleaned.csv"
    df_cleaned.to_csv(output_path, index=False)
    print(f"\n✅ Données finales sauvegardées dans : {output_path}")


if __name__ == "__main__":
    main()
