O script principal que orquestra todo o processo: carregar dados, agrupar, roteirizar e visualizar.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Importa as funções dos nossos módulos
from clustering import agrupar_pedidos
from pathfinding import criar_grafo, otimizar_sequencia_tsp, calcular_rota_a_star

# --- Constantes ---
K_ENTREGADORES = 3 # Defina o número de entregadores (clusters)
PONTO_ORIGEM = 'Restaurante'
OUTPUT_DIR = 'outputs'

def carregar_dados():
    """Carrega todos os arquivos CSV necessários."""
    try:
        mapa_df = pd.read_csv('data/mapa_grafo.csv')
        # Indexa o locais_df por 'local_id' para a heurística
        locais_df = pd.read_csv('data/locais_coordenadas.csv').set_index('local_id')
        pedidos_df = pd.read_csv('data/pedidos_exemplo.csv')
        return mapa_df, locais_df, pedidos_df
    except FileNotFoundError as e:
        print(f"Erro: Arquivo não encontrado. Verifique a pasta /data. {e}")
        return None, None, None

def visualizar_clusters(pedidos_clusterizados, locais_df, kmeans_model):
    """Gera e salva um gráfico de dispersão dos clusters."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    output_path = os.path.join(OUTPUT_DIR, 'clusters_de_entrega.png')
    
    plt.figure(figsize=(12, 8))
    
    # Plota todos os locais (pontos cinza)
    plt.scatter(locais_df['x'], locais_df['y'], c='gray', alpha=0.3, s=50, label='Todos Locais')

    # Plota os pedidos clusterizados
    plot = sns.scatterplot(
        data=pedidos_clusterizados,
        x='x', y='y',
        hue='cluster',
        palette='bright',
        s=150,
        legend='full'
    )
    
    # Plota o Restaurante
    rest_coords = locais_df.loc[PONTO_ORIGEM]
    plt.scatter(rest_coords['x'], rest_coords['y'], c='red', marker='*', s=250, label='Restaurante')
    
    # Adiciona rótulos
    for i, row in pedidos_clusterizados.iterrows():
        plt.text(row['x'] + 0.5, row['y'] + 0.5, row['local_id'], fontsize=9)

    plt.title('Clusters de Entrega (K-Means)')
    plt.xlabel('Coordenada X')
    plt.ylabel('Coordenada Y')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    print(f"\n[+] Gráfico de clusters salvo em '{output_path}'")

def main():
    """Função principal para executar o sistema SORA."""
    print("--- Iniciando SORA (Sistema de Otimização de Rotas por IA) ---")
    
    # 1. Carregar Dados
    mapa_df, locais_df, pedidos_df = carregar_dados()
    if mapa_df is None:
        return

    print(f"\nCarregados {len(pedidos_df)} pedidos, {len(locais_df)} locais e {len(mapa_df)} ruas.")

    # 2. Preparar dados para Clustering
    # Junta pedidos com suas coordenadas (redefinindo o índice para a junção)
    pedidos_com_coords = pedidos_df.merge(locais_df.reset_index(), on='local_id')

    # 3. Executar Clustering (K-Means)
    pedidos_clusterizados, kmeans_model = agrupar_pedidos(pedidos_com_coords, K_ENTREGADORES)
    
    if pedidos_clusterizados.empty:
        print("Nenhum pedido para processar.")
        return

    print("\n--- Resultado do Clustering (K-Means) ---")
    print(pedidos_clusterizados[['pedido_id', 'local_id', 'cluster', 'x', 'y']])
    
    # 4. Visualizar Clusters
    visualizar_clusters(pedidos_clusterizados, locais_df, kmeans_model)

    # 5. Criar Grafo da Cidade
    G = criar_grafo(mapa_df)

    # 6. Calcular Rota para cada Cluster
    print("\n--- Otimização de Rotas (TSP + A*) ---")
    
    num_clusters_reais = pedidos_clusterizados['cluster'].nunique()
    
    for cluster_id in range(num_clusters_reais):
        # Pega locais para este cluster
        locais_cluster = pedidos_clusterizados[pedidos_clusterizados['cluster'] == cluster_id]['local_id'].tolist()
        
        if not locais_cluster:
            print(f"\nCluster {cluster_id}: Sem pedidos.")
            continue
        
        # Adiciona o restaurante como ponto de partida
        locais_visita = [PONTO_ORIGEM] + locais_cluster
        
        print(f"\n===== Entregador {cluster_id + 1} (Cluster {cluster_id}) =====")
        print(f"  Locais a visitar: {locais_visita}")
        
        # 7. Otimizar Sequência (TSP)
        sequencia_otima, custo_total_tsp = otimizar_sequencia_tsp(G, locais_visita)
        
        if sequencia_otima is None:
            print("  Não foi possível encontrar uma rota para este cluster.")
            continue
            
        print(f"  Sequência Otimizada (TSP): {' -> '.join(sequencia_otima)}")
        print(f"  Custo total estimado (TSP): {custo_total_tsp:.2f}")
        
        # 8. Detalhar Caminho (A*)
        print("  Caminho detalhado (A*):")
        custo_total_verificado_a_star = 0
        for i in range(len(sequencia_otima) - 1):
            origem = sequencia_otima[i]
            destino = sequencia_otima[i+1]
            
            # Passamos o locais_df indexado para a heurística
            caminho_segmento, custo_segmento = calcular_rota_a_star(G, locais_df, origem, destino)
            
            if caminho_segmento:
                print(f"    - {origem} -> {destino} (Custo: {custo_segmento:.2f}): {' -> '.join(caminho_segmento)}")
                custo_total_verificado_a_star += custo_segmento
            else:
                print(f"    - {origem} -> {destino}: CAMINHO NÃO ENCONTRADO")
        
        print(f"  Custo total verificado (A*): {custo_total_verificado_a_star:.2f}")
        print("=" * (29 + len(str(cluster_id))))

    print("\n--- Execução do SORA concluída ---")

if __name__ == "__main__":
    main()
