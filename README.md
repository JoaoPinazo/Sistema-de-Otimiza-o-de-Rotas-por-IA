

# Sabor Express - Sistema de Otimização de Rotas por IA (SORA)

Este repositório contém a solução de Inteligência Artificial desenvolvida para a empresa "Sabor Express", focada em otimizar seu processo de delivery de alimentos. O sistema utiliza algoritmos de clustering e busca em grafos para sugerir rotas eficientes, agrupar pedidos e reduzir custos operacionais.

-----

## 1\. 🎯 Descrição do Problema e Objetivos

O "Sabor Express" enfrenta desafios logísticos significativos durante os horários de pico (almoço e jantar). O gerenciamento manual das entregas resulta em:

  * Rotas ineficientes e longos tempos de espera.
  * Atrasos na entrega e insatisfação dos clientes.
  * Aumento no custo de combustível e desgaste dos veículos.
  * Baixa escalabilidade em períodos de alta demanda.

**Desafio Proposto:**
Desenvolver uma solução inteligente (IA) que substitua o planejamento manual, capaz de processar múltiplos pedidos simultâneos e gerar as rotas de entrega mais rápidas e econômicas.

**Objetivos do Projeto:**

  * **Reduzir o tempo médio de entrega** através de roteirização otimizada.
  * **Aumentar a eficiência operacional**, permitindo que cada entregador realize mais entregas por hora.
  * **Reduzir custos** com combustível.
  * **Melhorar a satisfação do cliente** com entregas mais rápidas e previsíveis.

-----

## 2\. 💡 Abordagem Adotada (A Solução SORA)

A solução proposta é um sistema de duas fases que lida com os dois principais gargalos da empresa: **agrupamento** e **roteirização**.

### Fase 1: Agrupamento de Pedidos (Clustering)

Quando a demanda é alta, é ineficiente enviar um entregador para cada pedido. O sistema primeiro agrupa os pedidos pendentes com base em sua proximidade geográfica.

  * **Algoritmo:** **K-Means**.
  * **Como funciona:** O número de "clusters" (K) é definido com base no número de entregadores disponíveis no momento. O K-Means analisa as coordenadas (latitude/longitude) de todos os pedidos prontos e os agrupa em K "zonas" de entrega.
  * **Resultado:** Cada entregador recebe um "lote" de pedidos geograficamente próximos, evitando que diferentes entregadores cruzem a cidade para atender áreas vizinhas.

### Fase 2: Otimização da Rota (Pathfinding)

Após definir *quais* pedidos cada entregador levará (Fase 1), precisamos definir a *ordem* e o *caminho* mais eficientes para entregá-los.

1.  **Definição da Sequência (TSP):** Para o lote de 3-5 pedidos de um entregador, o sistema resolve uma versão simplificada do **Problema do Caixeiro Viajante (TSP)**. Ele calcula a sequência de visitação (Restaurante -\> P2 -\> P4 -\> P1 -\> P3) que minimiza a distância ou o tempo total da viagem.
2.  **Encontrar o Caminho (A\*):** Com a sequência definida, o sistema usa o **Algoritmo A\* (A-Star)** para encontrar o caminho exato nas ruas (o melhor trajeto) entre cada ponto da sequência (ex: do Restaurante ao P2, depois do P2 ao P4...).

-----

## 3\. 🤖 Algoritmos Utilizados

### K-Means (Aprendizado Não Supervisionado)

  * **Propósito:** Clustering (Agrupamento).
  * **Por quê?** É um algoritmo rápido e eficiente para agrupar dados (pedidos) com base na sua localização no espaço (coordenadas geográficas). Ele particiona os N pedidos nos K grupos (entregadores) mais compactos possíveis.

### Grafo Ponderado (Estrutura de Dados)

  * **Propósito:** Representação do Mapa.
  * **Por quê?** A cidade é modelada como um grafo onde:
      * **Nós (Vértices):** Cruzamentos ou locais de entrega (incluindo o Sabor Express).
      * **Arestas (Edges):** As ruas que conectam os nós.
      * **Pesos:** O "custo" de percorrer a aresta. Nosso modelo usa **tempo estimado** ou **distância** (neste projeto, usamos um CSV estático, mas a ideia é que o peso seja dinâmico com base no tráfego).

### Algoritmo A\* (A-Star) (Busca em Grafos)

  * **Propósito:** Pathfinding (Encontrar o Menor Caminho).
  * **Por quê?** O A\* é ideal para mapas. Diferente da Busca em Largura (BFS) ou Profundidade (DFS), que são "cegas", o A\* é uma busca informada. Ele usa uma **heurística** (a distância em linha reta até o destino) para priorizar a exploração de caminhos que "parecem" estar na direção certa, encontrando a rota ótima de forma muito mais rápida.
  * **Fórmula:** $f(n) = g(n) + h(n)$
      * $g(n)$: Custo real do início até o nó $n$.
      * $h(n)$: Custo heurístico (estimado) do nó $n$ até o destino.

-----

## 4\. 🗺️ Diagrama do Grafo e Modelo da Solução

### Modelo Conceitual do Grafo

O grafo abaixo representa uma versão simplificada da área central da cidade. Os nós A, B, C... são pontos de entrega e cruzamentos. O "Sabor Express" é o nó de origem. As arestas têm pesos (ex: 5 minutos) que o A\* usará para calcular a rota.

\`\`

### Fluxo da Solução SORA

Este diagrama ilustra o fluxo de dados desde o recebimento do pedido até a definição da rota para o entregador.

\`\`

-----

## 5\. 📂 Estrutura do Repositório

O projeto está organizado da seguinte forma para facilitar a manutenção e execução:

```
sabor-express-ia/
├── /data/
│   ├── mapa_grafo.csv      # Arquivo com as arestas (ruas) e pesos (distância/tempo)
│   └── pedidos_exemplo.csv # CSV com pedidos (ID, latitude, longitude)
├── /src/
│   ├── clustering.py       # Módulo com a implementação do K-Means
│   ├── pathfinding.py      # Módulo com a implementação do A* e lógica do grafo
│   └── main.py             # Script principal que orquestra a solução
├── /outputs/
│   └── rota_otimizada.png  # Exemplo de output gráfico gerado
├── .gitignore
├── requirements.txt        # Lista de dependências Python
└── README.md               # Esta documentação
```

-----

## 6\. 🚀 Instruções para Execução

### Pré-requisitos

  * Python 3.8 ou superior
  * Pip (Gerenciador de pacotes Python)

### Bibliotecas Necessárias

As principais bibliotecas utilizadas neste projeto são:

  * **Pandas:** Para carregar e manipular os dados dos arquivos CSV.
  * **Scikit-learn:** Para a implementação do algoritmo K-Means.
  * **NetworkX:** Para criar, manipular e analisar a estrutura de grafos.
  * **Matplotlib:** Para visualizar os clusters e as rotas geradas.

### Passo a Passo para Execução

1.  **Clone o repositório:**

    ```bash
    git clone https://github.com/seu-usuario/sabor-express-ia.git
    cd sabor-express-ia
    ```

2.  **Crie um ambiente virtual (Recomendado):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # No Windows: venv\Scripts\activate
    ```

3.  **Instale as dependências:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Execute a simulação:**
    O script `main.py` carrega os dados de `pedidos_exemplo.csv` e o `mapa_grafo.csv`, executa o K-Means para agrupar (supondo K=3 entregadores) e, em seguida, calcula a rota A\* para um dos clusters.

    ```bash
    python src/main.py
    ```

### Outputs Esperados

A execução do script `main.py` irá:

1.  Exibir no console os clusters de entrega definidos.
2.  Mostrar a sequência de rota otimizada (TSP) para o primeiro cluster.
3.  Imprimir o caminho detalhado (A\*) entre os pontos.
4.  Gerar e salvar uma imagem em `/outputs/rota_otimizada.png`, mostrando um gráfico com os pontos de entrega, os clusters e as rotas calculadas.

-----

## 7\. 📈 Análise de Resultados e Limitações

### Eficiência da Solução

  * **K-Means:** É um algoritmo computacionalmente leve (complexidade próxima de $O(N)$ para $N$ pedidos), ideal para uma operação em tempo real.
  * **A\*:** Muito mais eficiente que o BFS/DFS em mapas urbanos. Sua performance depende da qualidade da heurística, mas é comprovadamente ótimo e eficiente para este cenário.
  * **TSP:** É um problema $NP-hard$. No entanto, para o número pequeno de paradas por cluster (ex: 3 a 7), podemos usar algoritmos de força bruta ou aproximações (como o "Vizinho Mais Próximo") que rodam instantaneamente.

### Limitações Encontradas

  * **Grafo Estático:** A solução atual usa um CSV com pesos fixos (distância). Isso não reflete condições reais de tráfego, que mudam dinamicamente (ex: horário de pico vs. madrugada).
  * **Simplicidade do K-Means:** O K-Means otimiza a distância "em linha reta" (Euclidiana) para os clusters, não o tempo de percurso nas ruas.
  * **Restrições Operacionais:** O modelo não considera o tempo de preparo do pedido, a capacidade da mochila do entregador (ex: máximo de 3 pedidos) ou janelas de horário de entrega.

### Sugestões de Melhoria (Trabalhos Futuros)

1.  **Grafo Dinâmico:** Integrar a solução com APIs de mapas (Google Maps, Waze) para obter **pesos de aresta (tempo) em tempo real**, baseados no tráfego.
2.  **Clustering Avançado:** Substituir o K-Means por algoritmos mais robustos como o **DBSCAN** (que pode identificar áreas densas de pedidos) ou uma variante do K-Means que use o tempo de rota (via A\*) como métrica de distância, em vez da distância Euclidiana.
3.  **Modelo Preditivo:** Usar dados históricos de entrega para treinar um modelo de Regressão que **preveja o tempo de entrega** com mais precisão, considerando dia da semana, hora e clima.
4.  **Otimização Multi-Objetivo:** Incluir restrições reais, como a capacidade da mochila do entregador e o tempo máximo que um pedido pode esperar.
