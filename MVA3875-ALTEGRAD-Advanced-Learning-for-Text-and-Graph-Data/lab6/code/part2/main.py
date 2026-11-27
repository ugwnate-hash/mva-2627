"""Learning on Sets and Graph Generative Models - ALTEGRAD - Nov 2026.
Example: python main.py --epochs 40 --graphs 900 --samples 100
"""
import argparse
from pathlib import Path
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from utils import create_dataset, graph_properties, N_NODES
from model import ConditionalGraphVAE, loss_function


# This self-contained helper visualizes sampled graphs and their conditions.
# It is provided in full in both variants; it is not a student TODO.
def plot_conditioned_samples(model, requested, condition_mean, condition_std,
                             filename='conditioned_graph_examples.png',
                             reference_adjacencies=None, seed=13, max_examples=6):
    """Sample and plot graphs, labeling requested vs. achieved conditions.

    Parameters
    ----------
    model : ConditionalGraphVAE
        Trained network. Its sample() method performs Bernoulli edge sampling.
    requested : array of shape (M, 2)
        Raw, unstandardized [density, triangles/node] requests.
    condition_mean, condition_std : arrays of shape (2,)
        Statistics computed on the training split (not the test split).
    filename : str or pathlib.Path
        Figure output path, e.g. 'examples.png' or 'examples.pdf'.
    reference_adjacencies : optional array of shape (M, n, n)
        Real held-out graphs corresponding to the requested properties.
        If provided, figures show reference above generated, side by side.
    seed : int
        Seed for reproducible generation and graph layout.
    max_examples : int
        Maximum number of property requests drawn in this figure.

    Returns
    -------
    generated : ndarray, shape (K, n, n)
    achieved : ndarray, shape (K, 2)
    """
    requested = np.asarray(requested, dtype=np.float32)
    if requested.ndim != 2 or requested.shape[1] != 2:
        raise ValueError('requested must have shape (M, 2): density, triangles/node')
    if len(requested) == 0:
        raise ValueError('Supply at least one requested property pair')

    k = min(len(requested), max_examples)
    requested = requested[:k]
    mean = np.asarray(condition_mean, dtype=np.float32).reshape(2)
    std = np.asarray(condition_std, dtype=np.float32).reshape(2)
    if np.any(std <= 0):
        raise ValueError('Standard deviations must be positive')

    if reference_adjacencies is not None:
        refs = np.asarray(reference_adjacencies)[:k]
        if refs.shape != (k, model.n_nodes, model.n_nodes):
            raise ValueError('reference_adjacencies must have shape (M, n_nodes, n_nodes)')
    else:
        refs = None

    model_device = next(model.parameters()).device
    cond = torch.as_tensor((requested - mean) / std, device=model_device)
    previously_training = model.training
    model.eval()
    try:
        # Isolate random seeds from the training loop and produce stable examples.
        gpu_devices = ([model_device.index if model_device.index is not None else
                        torch.cuda.current_device()] if model_device.type == 'cuda' else [])
        with torch.random.fork_rng(devices=gpu_devices):
            torch.manual_seed(seed)
            with torch.no_grad():
                generated = model.sample(cond).detach().cpu().numpy()
    finally:
        model.train(previously_training)

    achieved = np.stack([graph_properties(nx.from_numpy_array(a)) for a in generated])

    ncols = min(3, k)
    groups = int(np.ceil(k / ncols))
    rows_per_group = 2 if refs is not None else 1
    fig, axs = plt.subplots(groups * rows_per_group, ncols,
                            figsize=(4.4 * ncols, 3.75 * groups * rows_per_group),
                            squeeze=False)
    for ax in axs.flat:
        ax.axis('off')

    for i in range(k):
        r, c = divmod(i, ncols)
        ax_generated = axs[r * rows_per_group + (1 if refs is not None else 0), c]
        g_generated = nx.from_numpy_array(generated[i])
        positions = nx.spring_layout(g_generated, seed=seed + i, iterations=100)
        nx.draw_networkx(g_generated, pos=positions, ax=ax_generated,
                         with_labels=False, node_size=54,
                         node_color='#416a90', edge_color='#7b8895',
                         linewidths=0, width=0.8)
        req_d, req_t = requested[i]
        obs_d, obs_t = achieved[i]
        ax_generated.set_title(
            f'Generated graph {i+1}\n'
            f'Requested: density={req_d:.3f}, triangles/node={req_t:.3f}\n'
            f'Achieved:   density={obs_d:.3f}, triangles/node={obs_t:.3f}',
            fontsize=9.2, pad=8)

        if refs is not None:
            ax_reference = axs[r * rows_per_group, c]
            g_reference = nx.from_numpy_array(refs[i])
            ref_pos = nx.spring_layout(g_reference, seed=seed + i, iterations=100)
            nx.draw_networkx(g_reference, pos=ref_pos, ax=ax_reference,
                             with_labels=False, node_size=54,
                             node_color='#727272', edge_color='#a4a4a4',
                             linewidths=0, width=0.8)
            ax_reference.set_title(
                f'Held-out reference {i+1}\n'
                f'density={req_d:.3f}, triangles/node={req_t:.3f}',
                fontsize=9.2, pad=8)

    fig.suptitle('Property-conditioned graph generation: requested vs. achieved',
                 fontsize=14, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98), h_pad=2.2, w_pad=0.8)
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved graph examples to {filename}')
    return generated, achieved

def plot_requested_vs_achieved(requested, actual, filename):
    names = ("Edge density", "Triangles per node")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for k, ax in enumerate(axes):
        x, y = requested[:, k], actual[:, k]
        lower, upper = min(x.min(), y.min()), max(x.max(), y.max())
        pad = max(1e-3, 0.04 * (upper-lower))
        ax.scatter(x, y, s=18, alpha=0.6)
        ax.plot([lower-pad, upper+pad], [lower-pad, upper+pad], 'k--', linewidth=1)
        ax.set(xlabel='Requested', ylabel='Achieved', title=names[k],
               xlim=(lower-pad, upper+pad), ylim=(lower-pad, upper+pad))
        ax.set_aspect('equal', adjustable='box')
    fig.tight_layout()
    fig.savefig(filename, dpi=160)
    plt.close(fig)


def train(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    A, X, C, kinds = create_dataset(args.graphs, args.seed)
    ids = np.arange(len(A))
    trainval, test = train_test_split(ids, test_size=.15, stratify=kinds, random_state=args.seed)
    train_ids, val_ids = train_test_split(trainval, test_size=.17647, stratify=kinds[trainval], random_state=args.seed)
    c_mean, c_std = C[train_ids].mean(0), C[train_ids].std(0).clip(min=1e-3)
    cond = (C - c_mean) / c_std
    def loader(which, shuffle=False):
        data = TensorDataset(*(torch.tensor(t[which], dtype=torch.float32) for t in (A, X, cond, C)))
        return DataLoader(data, batch_size=args.batch_size, shuffle=shuffle)
    train_loader, val_loader = loader(train_ids, True), loader(val_ids)
    net = ConditionalGraphVAE(n_nodes=N_NODES).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    best_val = np.inf
    best_weights = None
    patience = 0
    for epoch in range(1, args.epochs + 1):
        net.train()
        tr = []
        for a, x, c, c_raw in train_loader:
            a, x, c, c_raw = (t.to(device) for t in (a,x,c,c_raw))
            optimizer.zero_grad()
            logits, mu, lv = net(a,x,c)
            loss, _ = loss_function(net, logits,mu,lv,a,c_raw,
                                    prop_scales=c_std, beta=args.beta,
                                    property_weight=args.property_weight)
            loss.backward()
            optimizer.step()
            tr.append(loss.item())
        net.eval()
        vl = []
        with torch.no_grad():
            for a, x, c, c_raw in val_loader:
                a, x, c, c_raw = (t.to(device) for t in (a,x,c,c_raw))
                logits, mu, lv = net(a,x,c)
                loss, _ = loss_function(net,logits,mu,lv,a,c_raw,
                                        prop_scales=c_std, beta=args.beta,
                                        property_weight=args.property_weight)
                vl.append(loss.item())
        mean_val = np.mean(vl)
        if mean_val < best_val - 1e-4:
            best_val = mean_val
            best_weights = {k: v.detach().cpu().clone() for k,v in net.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if epoch == 1 or epoch % 5 == 0:
            print(f'epoch {epoch:3d}: train {np.mean(tr):.4f} val {mean_val:.4f}')
        if patience >= args.patience:
            break
    net.load_state_dict(best_weights)
    net.eval()
    print('best val loss:', round(best_val,4), 'epoch:',epoch)

    # Task 12: Select held-out property requests. Sample one graph
    # per request from the conditional VAE and compute achieved properties.
    # Compare the two MAEs to a density-matched ER baseline. Print 5 examples.
    # Save requested and achieved arrays to:
    #   f'{args.output_prefix}_results.npz'
    # Call plot_requested_vs_achieved(requested, actual, filename).
    # After completing Task 12, optionally visualize graph samples with
    # the built-in helper in THIS file (no extra Python module required):
    #   n_show = min(args.visualize_samples, len(chosen))
    #   if n_show:
    #       plot_conditioned_samples(
    #           net, requested[:n_show], c_mean, c_std,
    #           f'{args.output_prefix}_graph_samples.png',
    #           reference_adjacencies=A[chosen[:n_show]],
    #           seed=args.seed + 3, max_examples=n_show)
    # This shows each held-out reference graph above a new sampled graph,
    # with requested and achieved properties in the title.
    chosen = test[:args.samples]                                    # held-out requests (test split)
    requested = C[chosen]                                           # raw [density, triangles/node]
    cond_req = torch.tensor((requested - c_mean) / c_std, dtype=torch.float32, device=device)
    generated = net.sample(cond_req).cpu().numpy()                  # Bernoulli edge sampling
    actual = np.stack([graph_properties(nx.from_numpy_array(a)) for a in generated])
    # Density-matched Erdos-Renyi baseline: p = requested density.
    rng = np.random.default_rng(args.seed)
    baseline = np.stack([graph_properties(nx.gnp_random_graph(N_NODES, float(r[0]),
                                                                seed=int(rng.integers(2**31 - 1))))
                         for r in requested])
    mae_model = np.abs(actual - requested).mean(0)
    mae_base = np.abs(baseline - requested).mean(0)
    print(f'{len(chosen)} held-out requests (property_weight={args.property_weight})')
    print(f'cVAE     MAE: density {mae_model[0]:.4f}, triangles/node {mae_model[1]:.4f}')
    print(f'ER base  MAE: density {mae_base[0]:.4f}, triangles/node {mae_base[1]:.4f}')
    print('requested (density, tri/node) -> cVAE achieved | ER baseline')
    for r, a, b in list(zip(requested, actual, baseline))[:5]:
        print(f'  ({r[0]:.3f}, {r[1]:.3f}) -> ({a[0]:.3f}, {a[1]:.3f}) | ({b[0]:.3f}, {b[1]:.3f})')

    # Visualize generated graphs
    results_file = f'{args.output_prefix}_results.npz'
    np.savez(results_file, requested=requested, generated=actual, baseline=baseline,
             property_weight=args.property_weight)
    figure_file = f'{args.output_prefix}_requested_vs_achieved.png'
    plot_requested_vs_achieved(requested, actual, figure_file)
    if args.visualize_samples > 0:
        n_show = min(args.visualize_samples, len(chosen))
        plot_conditioned_samples(
            net, requested[:n_show], c_mean, c_std,
            filename=f'{args.output_prefix}_graph_samples.png',
            reference_adjacencies=A[chosen[:n_show]], seed=args.seed + 3,
            max_examples=n_show,
        )
    print(f'Saved evaluation to {results_file} and {figure_file}')

    return net

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graphs', type=int, default=900)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--samples', type=int, default=100)
    parser.add_argument('--seed', type=int, default=13)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--beta', type=float, default=0.005)
    parser.add_argument('--property-weight', type=float, default=0.02)
    parser.add_argument('--patience', type=int, default=12)
    parser.add_argument('--output-prefix', type=str, default='cvae')
    parser.add_argument('--visualize-samples', type=int, default=6)
    train(parser.parse_args())
