import pandas as pd

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
