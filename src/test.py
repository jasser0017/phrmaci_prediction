from preprocessing.data_loader import load_data
from preprocessing.negative_cleaning import handle_negative_values
from preprocessing.filtering import filter_products_advanced

from preprocessing.imputation import evaluate_imputation_methods_for_product, apply_global_imputations
from preprocessing.outlier_detection import find_best_methods_for_df
from preprocessing.outlier_removal import apply_best_methods

# 1. Charger les données
df = load_data("data/clean_data_total.csv")

# 2. Nettoyer les valeurs négatives
cols = ["vente", "stock", "Total", "feature"]
df = handle_negative_values(df, cols)
print(df.isnull().sum())



# 3. Filtrage des produits
df = filter_products_advanced(df)
print("\n✅ Données finales nettoyées :")
print(df.head())
print(f"📊 Dimensions finales : {df.shape}")
print(f"🧪 Produits restants : {df['Code prdt'].nunique()}")
print(df.isnull().sum())
best_methods = evaluate_imputation_methods_for_product(df, "PF009", ["feature", "vente", "stock", "Total"])

# Étape 2 – Application à tout le dataset
df_imputé = apply_global_imputations(df, 
    best_methods={col: val["best"] for col, val in best_methods.items()}, 
    columns=["feature", "vente", "stock", "Total"]
)


print(df_imputé.shape)
print(df_imputé.isnull().sum())
results_summary = find_best_methods_for_df(df_imputé, columns_of_interest=cols)
print("Résumé des meilleures méthodes par produit et par colonne :\n")
for prod, col_dict in results_summary.items():
    print(f"Produit {prod} :")
    for col, methods_info in col_dict.items():
        best = methods_info["Best_method"]
        print(f"  - Colonne {col} -> Meilleure méthode : {best}")
    print("-"*50)

# Étape 6 : Application des nettoyages outliers
df_cleaned = apply_best_methods(df_imputé, results_summary, columns=cols)

 







