"""
Deep Learning on Graphs - ALTEGRAD - Nov 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, hidden_dim_3, n_class, device):
        super(GNN, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc3 = nn.Linear(hidden_dim_2, hidden_dim_3)
        self.fc4 = nn.Linear(hidden_dim_3, n_class)
        self.relu = nn.ReLU()

    def forward(self, x_in, adj, idx):
        
        ############## Task 6
    
        ##################
        # your code here #
        ##################
        
        fc1 = self.fc1
        fc2 = self.fc2
        fc3 = self.fc3
        fc4 = self.fc4
        relu = self.relu

        x = fc1(torch.mm(adj, x_in))
        x = relu(x)
        x = torch.mm(adj, x)
        x = fc2(x)
        idx = idx.unsqueeze(1).repeat(1, x.size(1))
        out = torch.zeros(torch.max(idx)+1, x.size(1)).to(self.device)
        out = out.scatter_add_(0, idx, x) 
        
        ##################
        # your code here #
        ##################

        out = fc4(relu(fc3(out)))

        return F.log_softmax(out, dim=1)
