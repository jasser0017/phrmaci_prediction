# src/preprocessing/imputation.py

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import mean_squared_error


def evaluate_imputation_methods_for_product(df, product_code, columns, min_size=20):
    results = {}
    filtered_df = df[df['Code prdt'] == product_code].copy()

    for col in columns:
        print(f"\nImputation pour la colonne: {col}")
        df_ex_notnull = filtered_df[filtered_df[col].notnull()].copy()
        print(f"Avant masquage {col}: {df_ex_notnull.shape}")

        if len(df_ex_notnull) <= min_size:
            print(f"\nPas assez de données pour faire une comparaison d'imputation sur {col}.")
            results[col] = {"best": None, "rmses": {}}
            continue

        np.random.seed(42)
        mask_indices = np.random.choice(df_ex_notnull.index, size=int(0.1 * len(df_ex_notnull)), replace=False)
        true_values = df_ex_notnull.loc[mask_indices, col].copy()
        df_ex_notnull.loc[mask_indices, col] = np.nan

        methods = {}

        # Mean
        imp = SimpleImputer(strategy='mean')
        df_temp = df_ex_notnull.copy()
        df_temp[col] = imp.fit_transform(df_temp[[col]])
        methods['mean'] = np.sqrt(mean_squared_error(df_temp.loc[mask_indices, col], true_values))

        # Median
        imp = SimpleImputer(strategy='median')
        df_temp = df_ex_notnull.copy()
        df_temp[col] = imp.fit_transform(df_temp[[col]])
        methods['median'] = np.sqrt(mean_squared_error(df_temp.loc[mask_indices, col], true_values))

        # KNN
        imp = KNNImputer(n_neighbors=4)
        df_temp = df_ex_notnull.copy()
        df_temp[col] = imp.fit_transform(df_temp[[col]])
        methods['KNN'] = np.sqrt(mean_squared_error(df_temp.loc[mask_indices, col], true_values))

        # Forward Fill
        df_temp = df_ex_notnull.copy().sort_values(by="Date")
        df_temp[col] = df_temp[col].ffill()
        methods['ffill'] = np.sqrt(mean_squared_error(df_temp.loc[mask_indices, col], true_values))

        # Linear Interpolation
        df_temp = df_ex_notnull.copy().sort_values(by="Date")
        df_temp[col] = df_temp[col].interpolate(method="linear")
        methods['lin'] = np.sqrt(mean_squared_error(df_temp.loc[mask_indices, col], true_values))

        print("\nComparaison d'imputation sur sample (", col, ") : RMSE\n", methods)

        best_method = min(methods, key=methods.get)
        print(f"Meilleure méthode selon test pour la colonne {col} =", best_method)

        results[col] = {"best": best_method, "rmses": methods}

    return results


def apply_global_imputations(df, best_methods, columns):
    df = df.sort_values(by="Date").copy()

    for col in columns:
        method = best_methods.get(col)
        if method is None:
            continue

        print(f"Application de l'imputation '{method}' pour la colonne '{col}'")

        if method == 'mean':
            imp = SimpleImputer(strategy='mean')
            df[col] = imp.fit_transform(df[[col]])
        elif method == 'median':
            imp = SimpleImputer(strategy='median')
            df[col] = imp.fit_transform(df[[col]])
        elif method == 'KNN':
            imp = KNNImputer(n_neighbors=4)
            df[col] = imp.fit_transform(df[[col]])
        elif method == 'ffill':
            df[col] = df[col].ffill()
        elif method == 'lin':
            df[col] = df[col].interpolate(method='linear')

    return df
