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
    
    
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    df=df.sort_values(by=['Date','Code prdt']).reset_index(drop=True)

    return df




def apply_best_methods_to_derived_features(df, best_methods):
    df = df.sort_values(by="Date").copy()

    for col, method in best_methods.items():
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
        
        
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    df=df.sort_values(by=['Date','Code prdt']).reset_index(drop=True)


    return df


def find_best_imputation_methods(df: pd.DataFrame, product_code: str, columns_to_impute: list, mask_fraction: float = 0.1, random_state: int = 41):
    best_methods = {}
    df_product = df[df["Code prdt"] == product_code].copy()

    for col in columns_to_impute:
        df_col = df_product[df_product[col].notnull()].copy()
        df_col.sort_values(by="Date", inplace=True)

        print(f"\nImputation pour la colonne: {col}")
        print(f"Avant masquage {col}: {df_col.shape}")

        if len(df_col) >= 20:
            np.random.seed(random_state)
            mask_indices = np.random.choice(df_col.index, size=int(len(df_col) * mask_fraction), replace=False)
            true_values = df_col.loc[mask_indices, col].copy()
            df_col.loc[mask_indices, col] = np.nan
            methods = {}

            # Imputations
            imp = SimpleImputer(strategy='mean')
            df_tmp = df_col.copy()
            df_tmp[col] = imp.fit_transform(df_tmp[[col]])
            methods['mean'] = np.sqrt(mean_squared_error(df_tmp.loc[mask_indices, col], true_values))

            imp = SimpleImputer(strategy='median')
            df_tmp = df_col.copy()
            df_tmp[col] = imp.fit_transform(df_tmp[[col]])
            methods['median'] = np.sqrt(mean_squared_error(df_tmp.loc[mask_indices, col], true_values))

            imp = KNNImputer(n_neighbors=4)
            df_tmp = df_col.copy()
            df_tmp[col] = imp.fit_transform(df_tmp[[col]])
            methods['KNN'] = np.sqrt(mean_squared_error(df_tmp.loc[mask_indices, col], true_values))

            df_tmp = df_col.copy().sort_values(by="Date")
            df_tmp[col] = df_tmp[col].ffill()
            methods['ffill'] = np.sqrt(mean_squared_error(df_tmp.loc[mask_indices, col], true_values))

            df_tmp = df_col.copy().sort_values(by="Date")
            df_tmp[col] = df_tmp[col].interpolate(method="linear")
            methods['lin'] = np.sqrt(mean_squared_error(df_tmp.loc[mask_indices, col], true_values))

            best_method = min(methods, key=methods.get)
            best_methods[col] = best_method

            print(f"✔️ Meilleure méthode pour {col} : {best_method} (MAE: {methods[best_method]:.2f})")
        else:
            print(f"⚠️ Pas assez de données valides pour {col}.")
    return best_methods

def apply_best_imputations_by_group(df, best_methods: dict, group_col: str = "Code prdt", date_col: str = "Date") -> pd.DataFrame:
    """
    Applique les meilleures méthodes d'imputation spécifiées dans best_methods 
    à chaque colonne cible, en respectant les groupes définis par group_col.

    :param df: DataFrame contenant les données
    :param best_methods: Dictionnaire {colonne: méthode} avec les méthodes d'imputation ('mean', 'median', 'KNN', 'ffill', 'lin')
    :param group_col: Colonne représentant le groupe (par défaut "Code prdt")
    :param date_col: Colonne représentant la date pour le tri (par défaut "Date")
    :return: DataFrame avec les valeurs imputées
    """
    df = df.copy()

    for col, method in best_methods.items():
        print(f"Application de l'imputation '{method}' pour la colonne '{col}'")

        if method == 'mean':
            imputer = SimpleImputer(strategy='mean')
            df[col] = imputer.fit_transform(df[[col]])

        elif method == 'median':
            imputer = SimpleImputer(strategy='median')
            df[col] = imputer.fit_transform(df[[col]])

        elif method == 'KNN':
            imputer = KNNImputer(n_neighbors=5)
            df[col] = imputer.fit_transform(df[[col]])

        elif method == 'lin':
            df[col] = (
                df.sort_values(by=[group_col, date_col])
                  .groupby(group_col)[col]
                  .transform(lambda grp: grp.interpolate(method="linear").ffill().bfill())
            )

        elif method == 'ffill':
            df[col] = (
                df.sort_values(by=[group_col, date_col])
                  .groupby(group_col)[col]
                  .transform(lambda grp: grp.ffill().bfill())
            )

        else:
            raise ValueError(f"❌ Méthode d'imputation non reconnue : {method}")

    return df

    

    
    

