# Crypto-Graph-Model-NEAR-AI

# Modelado de Redes Complejas en Criptomonedas: Enfoque NEAR Protocol (IA)

## 🎯 Objetivo
Este proyecto tiene como objetivo principal **modelar la interdependencia y la topología del mercado de criptoactivos**, enfocándose específicamente en **NEAR Protocol** y su relación con el sector de **Inteligencia Artificial (IA)**.  

El modelo lógico representa un sistema complejo de **más de 1,000 nodos**, que servirá para el posterior desarrollo de **algoritmos de Machine Learning sobre grafos (GML)**.  

---

## 📊 Metodología del Grafo Multi-Partito
El sistema se modela como una **red multi-partita (P-I-H)** para capturar las diversas interconexiones del mercado:

- **Nodos P (Projects):** Proyectos de Inteligencia Artificial, con **NEAR Protocol** como el nodo de interés central.  
- **Nodos I (Infrastructure):** Infraestructuras tecnológicas de soporte (Blockchains, Exploradores, Billeteras).  
- **Nodos H (Hubs):** Exchanges Centralizados (**CEX**) clasificados por *Tier* (liquidez y confianza).  

---

## ⚙️ Ponderación de Aristas (MCP)
La fuerza de las relaciones se cuantifica mediante una **Métrica Compuesta de Ponderación (MCP)**, superando la simple correlación de precios.  

La **MCP** para las aristas **P-P (Proyecto-a-Proyecto)** se calcula como una combinación de:

1. **Factor de Correlación Financiera (F_C):** Basado en retornos de precios históricos.  
2. **Factor de Riesgo Operacional (F_R):** Basado en el ratio *Volumen 24h / Capitalización de Mercado*.  
3. **Factor de Confianza Agregado (F_T):** Derivado de la *Calificación CEX (Tier Score)* y la difusión en redes sociales (*Comunidad*).  

---

## 🛠️ Stack Tecnológico
- **Adquisición de Datos:**  
  Uso conceptual y lógico de la **CoinGecko API** (endpoints `/coins/markets` y `/coins/{id}`) para extraer **17+ atributos actualizados**.  

- **Modelado y Análisis:**  
  - `NetworkX` → Construcción y análisis del grafo.  
  - `iGraph` → Referencia para análisis de grafos a gran escala.  

- **Visualización:**  
  `Plotly` → Generación de un **Force-Directed Layout interactivo** que permite explorar la topología y la fuerza de las aristas ponderadas.  

- **Lenguaje:**  
  `Python`.  
