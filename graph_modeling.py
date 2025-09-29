"""
Crypto Graph Model - NEAR AI
============================

Este módulo proporciona herramientas para modelar la interdependencia y 
topología del mercado de criptoactivos usando análisis de grafos.

Autor: Mauricio Castellón
Fecha: 2024
"""

import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pycoingecko import CoinGeckoAPI
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class CryptoGraphModel:
    """
    Clase principal para el modelado de grafos de criptomonedas.
    
    Esta clase permite recolectar datos de criptomonedas, construir grafos
    basados en correlaciones de precios, y visualizar las relaciones entre
    diferentes activos digitales.
    """
    
    def __init__(self):
        """Inicializa el modelo de grafo de criptomonedas."""
        self.cg = CoinGeckoAPI()
        self.crypto_data = None
        self.price_data = None
        self.correlation_matrix = None
        self.graph = None
        self.crypto_list = []
        
    def load_crypto_data(self, crypto_ids: List[str], vs_currency: str = 'usd', 
                        days: int = 30) -> pd.DataFrame:
        """
        Carga datos históricos de precios para las criptomonedas especificadas.
        
        Args:
            crypto_ids: Lista de IDs de criptomonedas (ej: ['bitcoin', 'ethereum'])
            vs_currency: Moneda base para los precios (default: 'usd')
            days: Número de días de datos históricos (default: 30)
            
        Returns:
            DataFrame con datos de precios históricos
        """
        print(f"Cargando datos para {len(crypto_ids)} criptomonedas...")
        
        price_data = {}
        
        for crypto_id in crypto_ids:
            try:
                # Obtener datos históricos de precios
                historical_data = self.cg.get_coin_market_chart_by_id(
                    id=crypto_id,
                    vs_currency=vs_currency,
                    days=days
                )
                
                # Extraer precios y convertir timestamps
                prices = historical_data['prices']
                df_temp = pd.DataFrame(prices, columns=['timestamp', 'price'])
                df_temp['timestamp'] = pd.to_datetime(df_temp['timestamp'], unit='ms')
                df_temp.set_index('timestamp', inplace=True)
                
                price_data[crypto_id] = df_temp['price']
                print(f"✓ Datos cargados para {crypto_id}")
                
            except Exception as e:
                print(f"✗ Error cargando datos para {crypto_id}: {str(e)}")
                continue
        
        # Combinar todos los datos en un DataFrame
        if price_data:
            self.price_data = pd.DataFrame(price_data)
            self.crypto_list = list(price_data.keys())
            print(f"✓ Datos cargados exitosamente para {len(self.crypto_list)} criptomonedas")
            return self.price_data
        else:
            raise ValueError("No se pudieron cargar datos para ninguna criptomoneda")
    
    def calculate_correlations(self, method: str = 'pearson') -> pd.DataFrame:
        """
        Calcula la matriz de correlación entre criptomonedas.
        
        Args:
            method: Método de correlación ('pearson', 'spearman', 'kendall')
            
        Returns:
            Matriz de correlación como DataFrame
        """
        if self.price_data is None:
            raise ValueError("Primero debe cargar los datos con load_crypto_data()")
        
        # Calcular retornos logarítmicos
        returns = np.log(self.price_data / self.price_data.shift(1)).dropna()
        
        # Calcular matriz de correlación
        self.correlation_matrix = returns.corr(method=method)
        
        print(f"✓ Matriz de correlación calculada usando método '{method}'")
        return self.correlation_matrix
    
    def build_correlation_graph(self, threshold: float = 0.5, 
                              correlation_method: str = 'pearson') -> nx.Graph:
        """
        Construye un grafo basado en las correlaciones entre criptomonedas.
        
        Args:
            threshold: Umbral mínimo de correlación para crear una arista
            correlation_method: Método para calcular correlaciones
            
        Returns:
            Grafo NetworkX con las correlaciones
        """
        if self.correlation_matrix is None:
            self.calculate_correlations(method=correlation_method)
        
        # Crear grafo no dirigido
        self.graph = nx.Graph()
        
        # Agregar nodos (criptomonedas)
        for crypto in self.crypto_list:
            self.graph.add_node(crypto)
        
        # Agregar aristas basadas en correlaciones
        for i, crypto1 in enumerate(self.crypto_list):
            for j, crypto2 in enumerate(self.crypto_list[i+1:], i+1):
                correlation = abs(self.correlation_matrix.loc[crypto1, crypto2])
                
                if correlation >= threshold:
                    self.graph.add_edge(
                        crypto1, 
                        crypto2, 
                        weight=correlation,
                        correlation=self.correlation_matrix.loc[crypto1, crypto2]
                    )
        
        print(f"✓ Grafo construido con {len(self.graph.nodes)} nodos y {len(self.graph.edges)} aristas")
        print(f"  Umbral de correlación: {threshold}")
        
        return self.graph
    
    def analyze_graph_metrics(self) -> Dict[str, float]:
        """
        Calcula métricas importantes del grafo.
        
        Returns:
            Diccionario con métricas del grafo
        """
        if self.graph is None:
            raise ValueError("Primero debe construir el grafo con build_correlation_graph()")
        
        metrics = {
            'num_nodes': len(self.graph.nodes),
            'num_edges': len(self.graph.edges),
            'density': nx.density(self.graph),
            'avg_clustering': nx.average_clustering(self.graph),
            'num_connected_components': nx.number_connected_components(self.graph)
        }
        
        # Centralidad solo si el grafo está conectado
        if nx.is_connected(self.graph):
            centrality = nx.degree_centrality(self.graph)
            metrics['most_central_node'] = max(centrality.items(), key=lambda x: x[1])
            metrics['avg_centrality'] = np.mean(list(centrality.values()))
        
        return metrics
    
    def visualize_graph(self, layout: str = 'spring', node_size_factor: int = 1000,
                       show_labels: bool = True, title: str = "Grafo de Correlaciones de Criptomonedas"):
        """
        Visualiza el grafo de correlaciones usando Plotly.
        
        Args:
            layout: Tipo de layout ('spring', 'circular', 'kamada_kawai')
            node_size_factor: Factor de tamaño para los nodos
            show_labels: Si mostrar etiquetas de los nodos
            title: Título del gráfico
        """
        if self.graph is None:
            raise ValueError("Primero debe construir el grafo con build_correlation_graph()")
        
        # Calcular posiciones de los nodos
        if layout == 'spring':
            pos = nx.spring_layout(self.graph, k=1, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(self.graph)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(self.graph)
        else:
            pos = nx.spring_layout(self.graph)
        
        # Extraer coordenadas de nodos
        node_x = [pos[node][0] for node in self.graph.nodes()]
        node_y = [pos[node][1] for node in self.graph.nodes()]
        
        # Extraer coordenadas de aristas
        edge_x = []
        edge_y = []
        edge_info = []
        
        for edge in self.graph.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_info.append(f"{edge[0]} - {edge[1]}: {edge[2]['correlation']:.3f}")
        
        # Crear el gráfico
        fig = go.Figure()
        
        # Agregar aristas
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='lightblue'),
            hoverinfo='none',
            mode='lines',
            name='Correlaciones'
        ))
        
        # Agregar nodos
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text' if show_labels else 'markers',
            hoverinfo='text',
            text=list(self.graph.nodes()) if show_labels else None,
            textposition="middle center",
            marker=dict(
                size=[self.graph.degree(node) * node_size_factor/len(self.graph.nodes()) + 10 
                      for node in self.graph.nodes()],
                color=[self.graph.degree(node) for node in self.graph.nodes()],
                colorscale='Viridis',
                colorbar=dict(title="Grado del Nodo"),
                line=dict(width=2, color='white')
            ),
            name='Criptomonedas'
        )
        
        # Agregar información hover para los nodos
        node_adjacencies = []
        node_text = []
        for node in self.graph.nodes():
            adjacencies = list(self.graph.neighbors(node))
            node_adjacencies.append(len(adjacencies))
            node_text.append(f'{node}<br>Conexiones: {len(adjacencies)}')
        
        node_trace.hovertext = node_text
        fig.add_trace(node_trace)
        
        # Configurar layout
        fig.update_layout(
            title=dict(text=title, x=0.5, font_size=16),
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[
                dict(
                    text="Tamaño del nodo proporcional al número de conexiones",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor='left', yanchor='bottom',
                    font=dict(size=12)
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )
        
        fig.show()
        return fig
    
    def visualize_correlation_heatmap(self, title: str = "Matriz de Correlación de Criptomonedas"):
        """
        Visualiza la matriz de correlación como un mapa de calor.
        
        Args:
            title: Título del gráfico
        """
        if self.correlation_matrix is None:
            raise ValueError("Primero debe calcular correlaciones con calculate_correlations()")
        
        fig = px.imshow(
            self.correlation_matrix,
            labels=dict(x="Criptomoneda", y="Criptomoneda", color="Correlación"),
            x=self.correlation_matrix.columns,
            y=self.correlation_matrix.index,
            color_continuous_scale='RdBu_r',
            title=title
        )
        
        fig.update_layout(
            title=dict(x=0.5, font_size=16),
            width=600,
            height=600
        )
        
        fig.show()
        return fig
    
    def get_top_correlations(self, n: int = 10) -> pd.DataFrame:
        """
        Obtiene las correlaciones más altas entre pares de criptomonedas.
        
        Args:
            n: Número de top correlaciones a retornar
            
        Returns:
            DataFrame con las top correlaciones
        """
        if self.correlation_matrix is None:
            raise ValueError("Primero debe calcular correlaciones con calculate_correlations()")
        
        # Crear lista de correlaciones
        correlations = []
        for i in range(len(self.correlation_matrix.columns)):
            for j in range(i+1, len(self.correlation_matrix.columns)):
                crypto1 = self.correlation_matrix.columns[i]
                crypto2 = self.correlation_matrix.columns[j]
                corr_value = self.correlation_matrix.iloc[i, j]
                
                correlations.append({
                    'Crypto1': crypto1,
                    'Crypto2': crypto2,
                    'Correlation': corr_value,
                    'Abs_Correlation': abs(corr_value)
                })
        
        # Convertir a DataFrame y ordenar
        df_correlations = pd.DataFrame(correlations)
        top_correlations = df_correlations.nlargest(n, 'Abs_Correlation')
        
        return top_correlations[['Crypto1', 'Crypto2', 'Correlation']]


def main():
    """
    Función principal de demostración del uso del CryptoGraphModel.
    """
    print("=== Crypto Graph Model - NEAR AI ===")
    print("Inicializando modelo de grafo de criptomonedas...\n")
    
    # Inicializar modelo
    model = CryptoGraphModel()
    
    # Lista de criptomonedas populares para el análisis
    crypto_list = [
        'bitcoin',
        'ethereum', 
        'cardano',
        'polkadot',
        'chainlink',
        'solana'
    ]
    
    try:
        # Cargar datos
        print("1. Cargando datos de criptomonedas...")
        model.load_crypto_data(crypto_list, days=90)
        
        # Construir grafo
        print("\n2. Construyendo grafo de correlaciones...")
        model.build_correlation_graph(threshold=0.3)
        
        # Analizar métricas
        print("\n3. Analizando métricas del grafo...")
        metrics = model.analyze_graph_metrics()
        
        print("Métricas del Grafo:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        
        # Visualizar grafo
        print("\n4. Generando visualizaciones...")
        model.visualize_graph()
        model.visualize_correlation_heatmap()
        
        # Mostrar top correlaciones
        print("\n5. Top 5 Correlaciones:")
        top_corr = model.get_top_correlations(n=5)
        print(top_corr.to_string(index=False))
        
        print("\n✓ Análisis completado exitosamente!")
        
    except Exception as e:
        print(f"✗ Error durante el análisis: {str(e)}")


if __name__ == "__main__":
    main()