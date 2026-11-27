"""Learning on Sets and Graph Generative Models - ALTEGRAD - Nov 2026"""
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

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
        n = A.size(-1)
        A_tilde = A + torch.eye(n, device=A.device, dtype=A.dtype)          # add self-loops
        d_inv_sqrt = A_tilde.sum(-1).pow(-0.5)                              # (B, n)
        A_bar = d_inv_sqrt.unsqueeze(-1) * A_tilde * d_inv_sqrt.unsqueeze(-2)  # D^-1/2 (A+I) D^-1/2
        H = self.msg1(A_bar @ X)                                            # (B, n, hidden)
        H = self.msg2(A_bar @ H)                                            # (B, n, hidden)
        h_G = H.mean(dim=1)                                                 # mean readout, (B, hidden)
        h = torch.cat([h_G, condition], dim=-1)
        return self.enc_mu(h), self.enc_logvar(h)

    def reparameterize(self, mu, logvar):
        # Task 9
        # use the posterior mean for deterministic evaluation.
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode_logits(self, z, condition):
        # Task 9: Concatenate z and the standardized condition and return
        # upper-triangular edge logits (no diagonal entries).
        return self.decoder(torch.cat([z, condition], dim=-1))             # (B, n(n-1)/2)

    def logits_to_probability(self, logits):
        # Task 10: Build a (B,n,n) symmetric edge-probability
        # matrix from the upper-triangular logits. Set the diagonal to zero.
        # self.rows and self.cols contain the corresponding indices.
        B, n = logits.size(0), self.n_nodes
        P = torch.zeros(B, n, n, device=logits.device, dtype=logits.dtype)
        P[:, self.rows, self.cols] = torch.sigmoid(logits)                  # upper triangle, i < j
        return P + P.transpose(-1, -2)                                      # symmetric, zero diagonal

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
    n = model.n_nodes
    # Reconstruction: BCE on the n(n-1)/2 upper-triangular entries only.
    target = A[:, model.rows, model.cols]
    recon = F.binary_cross_entropy_with_logits(logits, target)
    # KL(q(z|G,c) || N(0, I)) for a diagonal Gaussian, summed over latent dims, averaged over the batch.
    kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1)).mean()
    # Property loss: differentiable expected density and triangles per node from P.
    P = model.logits_to_probability(logits)
    density_hat = torch.sigmoid(logits).sum(dim=-1) * 2 / (n * (n - 1))
    triangles_hat = torch.diagonal(P @ P @ P, dim1=-2, dim2=-1).sum(-1) / (6 * n)
    scales = torch.as_tensor(np.asarray(prop_scales), dtype=logits.dtype, device=logits.device)
    prop = 0.5 * (((density_hat - raw_conditions[:, 0]) / scales[0]).pow(2)
                  + ((triangles_hat - raw_conditions[:, 1]) / scales[1]).pow(2)).mean()
    total = recon + beta * kl + property_weight * prop
    return total, {"recon": recon.item(), "kl": kl.item(), "property": prop.item()}
