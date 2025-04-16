
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







def show_missing_values(df, output_path="outputs/visualisations/missing_values.png"):
    msno.matrix(df)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"✅ Visualisation des valeurs manquantes sauvegardée sous : {output_path}")






















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
