"""
Graph Mining - ALTEGRAD - Nov 2024
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye
from random import randint
from sklearn.cluster import KMeans
from pathlib import Path


############## Task 3
# Perform spectral clustering to partition graph G into k clusters

def inv_diagonal_degree_matrix(G):
    diag = G.degree
    D_inv = [1/d[1] for d in diag]
    return np.reshape(D_inv, (-1, 1))

# Tests
G = nx.complete_graph(3)
print(nx.adjacency_matrix(G))
print(inv_diagonal_degree_matrix(G))


def spectral_clustering(G, k):
    
    ##################
    # your code here #
    ##################
    n = len(G.nodes)
    clustering = {}
    A = nx.adjacency_matrix(G)
    D_inv = inv_diagonal_degree_matrix(G)
    L = eye(n) - D_inv*A
    eig_val, eig_vec = eigs(L, k=k, which="SR")
    clusterer = KMeans(n_clusters=k, random_state=0)
    kmeans = clusterer.fit(eig_vec.real)
    for i, node in enumerate(G.nodes):
        clustering[node] = (kmeans.labels_)[i]
    return clustering

############## Task 4

##################
# your code here #
##################

path_to_data_folder = Path(__file__).resolve().parent.parent 
path_to_data = path_to_data_folder/"datasets"/"CA-HepTh.txt"

G = nx.read_edgelist(path_to_data, nodetype=int, comments="#", delimiter='\t', create_using=nx.Graph)
G.remove_edges_from(list(nx.selfloop_edges(G)))
connected_components = [c for c in nx.connected_components(G)]
gcc_nodes_list = list(max(connected_components, key=len))
gcc = nx.subgraph(G, gcc_nodes_list)
clustering = spectral_clustering(gcc,50)


############## Task 5
# Compute modularity value from graph G based on clustering
def modularity(G, clustering):
    
    ##################
    # your code here #
    ##################
    
    
    
    
    return modularity



############## Task 6

##################
# your code here #
##################







