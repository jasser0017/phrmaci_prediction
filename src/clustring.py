import numpy as np
from tslearn.clustering import KShape
from tslearn.utils import to_time_series_dataset


def run_kshape_clustering(df, value_col="Total", group_col="Code prdt", date_col="Date", n_clusters=3, random_state=0):
    """
    Applique le clustering K-Shape sur les séries temporelles d'un DataFrame groupées par produit.

    :param df: DataFrame d'entrée avec colonnes produit, date, valeur (Total)
    :param value_col: Colonne sur laquelle effectuer le clustering (ex: "Total")
    :param group_col: Colonne identifiant les produits (ex: "Code prdt")
    :param date_col: Colonne de date pour trier les séries
    :param n_clusters: Nombre de clusters à créer
    :param random_state: Graine aléatoire pour reproductibilité
    :return: df avec colonne 'cluster_label' ajoutée
    """
    series_list = []
    product_indices = []

    for prod, group in df.groupby(group_col):
        group_sorted = group.sort_values(date_col)
        series = group_sorted[value_col].values
        series_list.append(series)
        product_indices.append(prod)

    X = to_time_series_dataset(series_list)
    X_filled = np.nan_to_num(X, nan=0.0)

    kshape = KShape(n_clusters=n_clusters, n_init=10, verbose=False, random_state=random_state)
    labels = kshape.fit_predict(X_filled)

    cluster_map = {prod: label for prod, label in zip(product_indices, labels)}
    df = df.copy()
    df['cluster_label'] = df[group_col].map(cluster_map)

    return df, cluster_map