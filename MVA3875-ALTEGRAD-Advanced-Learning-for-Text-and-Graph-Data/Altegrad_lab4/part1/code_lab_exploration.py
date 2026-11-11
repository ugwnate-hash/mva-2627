"""
Graph Mining - ALTEGRAD - Nov 2024
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

############## Task 1

path_to_data_folder = Path(__file__).resolve().parent.parent 
path_to_data = path_to_data_folder/"datasets"/"CA-HepTh.txt"

print(path_to_data) # Machine invariant path
G = nx.read_edgelist(path_to_data, nodetype=int, comments="#", delimiter='\t', create_using=nx.Graph)

print(f"Number of selfloops: {nx.number_of_selfloops(G)}")
print(f"Number of edges: {len(G.edges)}") 
print(f"Number of nodes: {len(G.nodes)}")
print(2 * len(G.edges) - nx.number_of_selfloops(G))
# Note that the nb of edges in the undirected graph is half from what is being stated in the graph description + selfloops

############## Task 2

connected_components = [c for c in nx.connected_components(G)]
print(f"Number of connected_components: {len(connected_components)}")

if len(connected_components) != 1:
    gcc_nodes_list = list(max(connected_components, key=len))
    gcc = nx.subgraph(G, gcc_nodes_list)

    print(f"Number of edges of the largest connected component: {len(gcc.edges)}") 
    print(f"Number of nodes of the largest connected component: {len(gcc.nodes)}")

    print(f"Fraction of edges of largest connected component: {len(gcc.edges)/len(G.edges)*100:.2f}%") 
    print(f"Fraction of nodes of the largest connected component: {len(gcc.nodes)/len(G.nodes)*100:.2f}%")
else:
    print("The graph is connected!")