

from src.preprocessing.data_loader import load_data
from src.preprocessing.negative_cleaning import handle_negative_values
from  src.preprocessing.filtering import filter_products_advanced

from src.preprocessing.imputation import (
    evaluate_imputation_methods_for_product,
    apply_global_imputations
)
from src.preprocessing.outlier_detection import find_best_methods_for_df
from src.preprocessing. outlier_removal import apply_best_methods

import os
import pandas as pd

def main():

    df = load_data("data/clean_data_total.csv")


    cols = ["vente", "stock", "Total", "feature"]
    df = handle_negative_values(df, cols)

    
    df = filter_products_advanced(df)

    
    ref_product = "PF009" 
    print(f"\n Benchmark des méthodes d'imputation sur le produit '{ref_product}'")
    imputation_results = evaluate_imputation_methods_for_product(df, ref_product, cols)
    best_methods = {col: imputation_results[col]['best'] for col in imputation_results}
    df_imputed = apply_global_imputations(df, best_methods, cols)

    
    print("\n Détection des méthodes d'outliers pour chaque produit/colonne...")
    results_summary = find_best_methods_for_df(df_imputed, columns_of_interest=cols)

    
    print("\n Application du nettoyage des outliers...")
    df_cleaned = apply_best_methods(df_imputed, results_summary, columns=cols)
    

    
    os.makedirs("outputs", exist_ok=True)
    output_path = "outputs/df_cleaned.csv"
    df_cleaned.to_csv(output_path, index=False)
    print(f"\n✅ Données finales sauvegardées dans : {output_path}")


if __name__ == "__main__":
    main()
