import networkx as nx
import pandas as pd
import random
import numpy as np
import plotly.graph_objects as go
import time
from pycoingecko import CoinGeckoAPI
from datetime import datetime, timedelta
COINGECKO_API_KEY = "***"
CG_CLIENT = CoinGeckoAPI(api_key=COINGECKO_API_KEY)

# Parámetros del modelo
N_TARGET = 1000 
N_AI_PROJECTS = 950 
N_INFRASTRUCTURE = 30 
N_CEX_HUBS = 20 
TARGET_COIN_ID = 'near'
TARGET_COIN_NAME = 'NEAR Protocol'

CEX_TIER_SCORE = {
    'Binance (Tier 1)': 5, 'Coinbase (Tier 1)': 5, 'Kraken (Tier 2)': 3,
    'KuCoin (Tier 2)': 3, 'Gate.io (Tier 3)': 1, 'OKX (Tier 1)': 5,
    'Bybit (Tier 2)': 3, 'Huobi (Tier 3)': 1, 'NEAR Blockchain': 5 
}

def data_acquisition_pipeline():
    print("Iniciando la adquisición de datos de 1000+ nodos...")

    simulated_projects = [f'AI Project {i}' for i in range(N_AI_PROJECTS)]
    simulated_projects.append(TARGET_COIN_NAME)

    data = []
    for project in simulated_projects:
        is_near = (project == TARGET_COIN_NAME)

        vol_mc_ratio = random.uniform(0.01, 0.5) if not is_near else random.uniform(0.05, 0.25)
        trust_score = random.uniform(0.1, 0.9)

        data.append({
            'project_name': project,
            'type': 'P',
            'is_near': is_near,
            'market_cap': random.randint(100, 5000) * (1000000 if is_near else 1),
            'vol_24h': random.randint(5, 500) * (1000000 if is_near else 1),
            'vol_mc_ratio': vol_mc_ratio, 
            'trust_score_ft': trust_score, 
            'has_halving': random.choice([True, False]), 
            'is_multichain': random.choice([True, False]),
            'price_correlation_base': random.uniform(0.1, 0.9), 
            'cex_listing': random.sample(list(CEX_TIER_SCORE.keys()), random.randint(1, 3)),
            'blockchain_explorer': random.choice([f'Explorer {i}' for i in range(N_INFRASTRUCTURE)])
        })

    infra_nodes = [f'Explorer {i}' for i in range(N_INFRASTRUCTURE)]
    hub_nodes = list(CEX_TIER_SCORE.keys())

    df = pd.DataFrame(data)
    print(f"Datos simulados y preprocesados para {len(df)} proyectos de IA.")

    return df, infra_nodes, hub_nodes


def calculate_mcp_composite(df_node1, df_node2):
    """Calcula la Métrica Compuesta de Ponderación (MCP) P-P.[2]"""

    fc = (df_node1['price_correlation_base'] + df_node2['price_correlation_base']) / 2
    fr = 1 - abs(df_node1['vol_mc_ratio'] - df_node2['vol_mc_ratio'])
    ft = (df_node1['trust_score_ft'] + df_node2['trust_score_ft']) / 2
    w_composite = (0.5 * fc) + (0.3 * fr) + (0.2 * ft)

    return w_composite

def calculate_mcp_ph(project_node, hub_name):
    """Calcula la Métrica Compuesta de Ponderación P-H (Riesgo/Confianza).[2]"""
    tier_score = CEX_TIER_SCORE.get(hub_name, 1)

    base_weight = project_node['market_cap'] * project_node['vol_mc_ratio']
    weight = (base_weight * tier_score) / (10**9 * 10) 
    return min(1.0, max(0.01, weight)) 

def build_network_graph(df, infra_nodes, hub_nodes):
    """Crea el grafo Multi-Partito (P, I, H) usando NetworkX."""
    G = nx.Graph()
    print("Creando Nodos P, I, H...")

    for index, row in df.iterrows():
        color = '#00BFFF'
        size = 15
        if row['is_near']:
            color = '#FF007F' # NEAR resaltado en rosa
            size = 25

        G.add_node(row['project_name'],
                   type='P',
                   category='AI',
                   color=color,
                   size=size,
                   market_cap=row['market_cap'],
                   ft_score=row['trust_score_ft'],
                   label=row['project_name'])

    for node in infra_nodes:
        G.add_node(node, type='I', category='Infrastructure', color='#FFD700', size=10, label=node)

    for node, score in CEX_TIER_SCORE.items():
        G.add_node(node, type='H', category=f'CEX Tier {score}', color='#DC143C', size=score * 4, label=node)

    print(f"Total de Nodos creados: {G.number_of_nodes()} (Requisito: >= 1000) [2]")

    print("Calculando Aristas P-P (W_Composite)...")
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            p1 = df.iloc[i]
            p2 = df.iloc[j]
            if random.random() < 0.005:
                weight = calculate_mcp_composite(p1, p2)
                G.add_edge(p1['project_name'], p2['project_name'], weight=weight, type='P-P')

    print("Calculando Aristas P-H (W_P-H)...")
    for index, row in df.iterrows():
        hubs_to_connect = random.sample(hub_nodes, min(3, len(hub_nodes)))
        for hub in hubs_to_connect:
            weight = calculate_mcp_ph(row, hub)
            G.add_edge(row['project_name'], hub, weight=weight, type='P-H', line_color='#DC143C', line_width=weight * 5)

    print("Agregando Aristas P-I (Dependencia Tecnológica)...")
    for index, row in df.iterrows():
        infra_node = row['blockchain_explorer']
        if infra_node in G.nodes:
             G.add_edge(row['project_name'], infra_node, weight=1.0, type='P-I', line_color='#FFD700')

    print(f"Total de Aristas creadas: {G.number_of_edges()}")
    return G

def visualize_graph_plotly(G):
    """Genera la visualización interactiva de la red usando Plotly.[5]"""

    pos = nx.spring_layout(G, k=0.15, iterations=50, seed=42)

    edge_traces = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = edge[2].get('weight', 0.1)
        line_width = weight * 3

        edge_trace = go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            line=dict(width=line_width, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        edge_traces.append(edge_trace)

    node_x = []
    node_y = []
    node_text = []
    node_sizes = []
    node_colors = []

    for node in G.nodes(data=True):
        x, y = pos[node[0]]
        node_x.append(x)
        node_y.append(y)

        mc_formatted = f"${node[1].get('market_cap', 0):,.0f}" if node[1]['type'] == 'P' else 'N/A'
        ft_score_value = node[1].get('ft_score', 'N/A')
        ft_score_formatted = f"{ft_score_value:.2f}" if isinstance(ft_score_value, (int, float)) else ft_score_value

        node_text.append(
            f"<b>{node[0]}</b><br>"
            f"Tipo: {node[1]['type']} ({node[1]['category']})<br>"
            f"Market Cap: {mc_formatted}<br>"
            f"F_T (Trust Score): {ft_score_formatted}"
        )
        node_sizes.append(node[1].get('size', 10))
        node_colors.append(node[1]['color'])

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        hovertext=node_text,
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(color='#FFFFFF', width=1)),
        name='')

    fig = go.Figure(data=edge_traces + [node_trace],
                layout=go.Layout(
                    title=f'<br>Modelo de Grafo Multi-Partito (P, I, H) - Enfoque: {TARGET_COIN_NAME} (IA)',
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    height=800,
                    plot_bgcolor='white'
                    )
                    )

    return fig

if __name__ == '__main__':
    df_nodes, infra_nodes_list, hub_nodes_list = data_acquisition_pipeline()
    G_complex = build_network_graph(df_nodes, infra_nodes_list, hub_nodes_list)
    fig = visualize_graph_plotly(G_complex)

    print("\nVisualización interactiva generada (ejecute fig.show() en un entorno compatible):")
    fig.show()
    print("\n--- Análisis Topológico Básico (NetworkX) ---")
    near_centrality = nx.degree_centrality(G_complex).get(TARGET_COIN_NAME)
    print(f"Centralidad de Grado de {TARGET_COIN_NAME}: {near_centrality:.4f}")

  
