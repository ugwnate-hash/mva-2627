"""Learning on Sets and Graph Generative Models - ALTEGRAD - Nov 2026"""
import random
import numpy as np
import networkx as nx

N_NODES = 24

def set_seed(seed=13):
    random.seed(seed)
    np.random.seed(seed)


def canonical_order(G):
    """Degree-guided BFS; pragmatic *approximate* node ordering, not true canonicalization."""
    deg = dict(G.degree())
    unvisited = set(G.nodes())
    order = []
    while unvisited:
        root = max(unvisited, key=lambda v: (deg[v], -v))
        q = [root]
        unvisited.remove(root)
        while q:
            v = q.pop(0)
            order.append(v)
            nbrs = sorted((u for u in G.neighbors(v) if u in unvisited),
                          key=lambda u: (-deg[u], u))
            for u in nbrs:
                unvisited.remove(u)
                q.append(u)
    return order


def graph_properties(G):
    """Density, triangles per node. Both are continuous targets."""
    # TASK 8: Compute the density and number of triangles per node.
    # Return a float32 NumPy array with shape (2,).
    n = G.number_of_nodes()
    density = nx.density(G)                                  # 2|E| / (n(n-1))
    n_triangles = sum(nx.triangles(G).values()) / 3          # each triangle is counted at its 3 nodes
    return np.array([density, n_triangles / n], dtype=np.float32)


def make_graph(kind, rng):
    n = N_NODES
    s = int(rng.integers(0, 2**31 - 1))
    if kind == 'ER':
        p = float(rng.uniform(0.12, 0.30))
        G = nx.fast_gnp_random_graph(n, p, seed=s)
    elif kind == 'WS':
        k = int(rng.choice([4, 6]))
        beta = float(rng.uniform(0.02, 0.95))
        G = nx.watts_strogatz_graph(n, k, beta, seed=s)
    elif kind == 'SBM':
        pin = float(rng.uniform(0.25, 0.55))
        pout = float(rng.uniform(0.02, 0.12))
        G = nx.stochastic_block_model([12, 12], [[pin, pout], [pout, pin]], seed=s)
    else:
        raise ValueError(kind)
    return nx.convert_node_labels_to_integers(G)


def create_dataset(total=900, seed=13):
    assert total % 3 == 0
    rng = np.random.default_rng(seed)
    graphs, kinds = [], []
    for kind in ['ER', 'WS', 'SBM']:
        for _ in range(total // 3):
            graphs.append(make_graph(kind, rng))
            kinds.append(kind)
    perm = rng.permutation(total)
    A, X, C = [], [], []
    for j in perm:
        G = graphs[j]
        # Compute properties before reordering; they are permutation invariant.
        C.append(graph_properties(G))
        order = canonical_order(G)
        a = nx.to_numpy_array(G, nodelist=order, dtype=np.float32)
        deg = a.sum(axis=1)
        nbr_mean_degree = (a @ deg) / np.maximum(deg, 1)
        x = np.stack([deg / (N_NODES - 1), nbr_mean_degree / (N_NODES - 1)], axis=-1)
        A.append(a)
        X.append(x.astype(np.float32))
    return (np.stack(A), np.stack(X), np.stack(C), np.array(kinds)[perm])


def graph_from_adj(A):
    # Important: Bernoulli sample, not 0.5 threshold, and upper triangle only.
    arr = np.triu(A, 1)
    return nx.from_numpy_array(arr + arr.T)


def summarize(graphs):
    props = np.array([graph_properties(g) for g in graphs])
    return props.mean(0), props.std(0)
