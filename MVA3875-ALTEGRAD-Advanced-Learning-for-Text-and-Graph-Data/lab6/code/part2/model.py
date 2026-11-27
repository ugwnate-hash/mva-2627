"""Learning on Sets and Graph Generative Models - ALTEGRAD - Nov 2026"""
import torch
from torch import nn
import torch.nn.functional as F

class ConditionalGraphVAE(nn.Module):
    def __init__(self, n_nodes=24, input_dim=2, cond_dim=2, hidden=64, latent=16):
        super().__init__()
        self.n_nodes = n_nodes
        self.latent = latent
        self.msg1 = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU())
        self.msg2 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU())
        self.enc_mu = nn.Linear(hidden + cond_dim, latent)
        self.enc_logvar = nn.Linear(hidden + cond_dim, latent)
        self.decoder = nn.Sequential(nn.Linear(latent + cond_dim, hidden * 2), nn.ReLU(),
                                     nn.Linear(hidden * 2, hidden * 2), nn.ReLU(),
                                     nn.Linear(hidden * 2, n_nodes * (n_nodes-1) // 2))
        tri = torch.triu_indices(n_nodes, n_nodes, offset=1)
        self.register_buffer('rows', tri[0])
        self.register_buffer('cols', tri[1])

    def encode(self, A, X, condition):
        # TASK 9
        raise NotImplementedError("Task 9: conditional encoder")

    def reparameterize(self, mu, logvar):
        # Task 9
        # use the posterior mean for deterministic evaluation.
        raise NotImplementedError("Task 9: reparameterization")

    def decode_logits(self, z, condition):
        # Task 9: Concatenate z and the standardized condition and return
        # upper-triangular edge logits (no diagonal entries).
        raise NotImplementedError("Task 10: conditional decoder")

    def logits_to_probability(self, logits):
        # Task 10: Build a (B,n,n) symmetric edge-probability
        # matrix from the upper-triangular logits. Set the diagonal to zero.
        # self.rows and self.cols contain the corresponding indices.
        raise NotImplementedError("Task 10: build symmetric edge probabilities")

    def forward(self, A, X, condition):
        mu, logvar = self.encode(A, X, condition)
        z = self.reparameterize(mu, logvar)
        return self.decode_logits(z, condition), mu, logvar

    @torch.no_grad()
    def sample(self, condition):
        z = torch.randn(condition.size(0), self.latent, device=condition.device)
        P = self.logits_to_probability(self.decode_logits(z, condition))
        tri = torch.triu(torch.bernoulli(P), diagonal=1)
        return tri + tri.transpose(-1, -2)


def loss_function(model, logits, mu, logvar, A, raw_conditions,
                  prop_scales=(0.1, 1.0), beta=0.005, property_weight=0.02):
    # TASK 10: Binary cross entropy on undirected pairs i<j,
    # Return (total_loss, {"recon": ..., "kl": ..., "property": ...}).
    raise NotImplementedError("Task 11: cVAE loss")
