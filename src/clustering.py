Módulo responsável por agrupar os pedidos usando K-Means.

import pandas as pd
from sklearn.cluster import KMeans

def agrupar_pedidos(pedidos_com_coords, num_clusters):
    """
    Agrupa pedidos geograficamente usando o algoritmo K-Means.

    Args:
        pedidos_com_coords (pd.DataFrame): DataFrame contendo 'x' e 'y'.
        num_clusters (int): O número de clusters (K), idealmente o número
                            de entregadores disponíveis.

    Returns:
        tuple: (DataFrame com coluna 'cluster', objeto KMeans model)
    """
    if pedidos_com_coords.empty:
        return pedidos_com_coords, None

    # Garante que K (clusters) não seja maior que N (pedidos)
    k = min(num_clusters, len(pedidos_com_coords))
    if k == 0:
         return pedidos_com_coords, None
    
    # Extrai as coordenadas para o clustering
    coords = pedidos_com_coords[['x', 'y']]

    # Cria e treina o modelo K-Means
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    
    # Adiciona a coluna 'cluster' ao DataFrame original
    pedidos_com_coords['cluster'] = kmeans.fit_predict(coords)
    
    print(f"[+] Pedidos agrupados em {k} clusters.")
    
    return pedidos_com_coords, kmeans
