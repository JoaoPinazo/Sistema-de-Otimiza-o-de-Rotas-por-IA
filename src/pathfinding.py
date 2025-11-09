Módulo para criar o grafo e encontrar os caminhos (A* e TSP).

import networkx as nx
import itertools
import math

def criar_grafo(mapa_df):
    """
    Cria um grafo ponderado do NetworkX a partir de um DataFrame.
    
    Args:
        mapa_df (pd.DataFrame): DataFrame com colunas 'origem', 'destino', 'peso'.
        
    Returns:
        nx.Graph: O grafo da cidade.
    """
    G = nx.Graph()
    for _, row in mapa_df.iterrows():
        G.add_edge(row['origem'], row['destino'], weight=row['peso'])
    print(f"[+] Grafo da cidade criado com {G.number_of_nodes()} nós e {G.number_of_edges()} arestas.")
    return G

def heuristica_distancia(u, v, locais_df):
    """
    Heurística de distância Euclidiana (linha reta) para o A*.
    É essencial que locais_df esteja indexado por 'local_id'.
    """
    try:
        pos_u = locais_df.loc[u]
        pos_v = locais_df.loc[v]
        return math.sqrt((pos_u['x'] - pos_v['x'])**2 + (pos_u['y'] - pos_v['y'])**2)
    except KeyError as e:
        print(f"Erro na heurística: Nó {e} não encontrado em locais_coordenadas.csv")
        return 0

def calcular_rota_a_star(grafo, locais_df, origem, destino):
    """
    Calcula o menor caminho entre dois pontos usando o algoritmo A*.
    
    Args:
        grafo (nx.Graph): O grafo da cidade.
        locais_df (pd.DataFrame): DataFrame de locais (indexado) para a heurística.
        origem (str): Nó de partida.
        destino (str): Nó de chegada.
        
    Returns:
        tuple: (lista do caminho, custo total do caminho)
    """
    try:
        # Define a função heurística que o A* usará
        def h(u, v):
            return heuristica_distancia(u, v, locais_df)

        caminho = nx.astar_path(grafo, origem, destino, heuristic=h, weight='weight')
        custo = nx.astar_path_length(grafo, origem, destino, heuristic=h, weight='weight')
        return caminho, custo
    except nx.NetworkXNoPath:
        return None, float('inf')

def otimizar_sequencia_tsp(grafo, locais_visita):
    """
    Resolve o Problema do Caixeiro Viajante (TSP) para um pequeno
    número de locais usando força bruta (testa todas as permutações).
    
    Args:
        grafo (nx.Graph): O grafo da cidade.
        locais_visita (list): Lista de locais, incluindo a 'Restaurante'.
        
    Returns:
        tuple: (melhor sequência de rota, menor custo total)
    """
    if len(locais_visita) < 2:
        return locais_visita, 0
    
    origem = 'Restaurante'
    # Filtra apenas os pontos de entrega (remove a origem)
    pontos_entrega = [loc for loc in locais_visita if loc != origem]
    
    melhor_sequencia = None
    menor_custo = float('inf')

    # Gera todas as permutações possíveis dos locais de entrega
    for perm in itertools.permutations(pontos_entrega):
        custo_atual = 0
        sequencia_atual = [origem] + list(perm)
        
        # Calcula o custo total da sequência (Ponto A -> Ponto B)
        for i in range(len(sequencia_atual) - 1):
            ponto_a = sequencia_atual[i]
            ponto_b = sequencia_atual[i+1]
            
            try:
                # Usa o 'weight' (peso da aresta) do grafo, não A*
                custo_segmento = nx.shortest_path_length(grafo, ponto_a, ponto_b, weight='weight')
                custo_atual += custo_segmento
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                custo_atual = float('inf')
                break
        
        # Atualiza se esta permutação for a melhor até agora
        if custo_atual < menor_custo:
            menor_custo = custo_atual
            melhor_sequencia = sequencia_atual

    return melhor_sequencia, menor_custo
