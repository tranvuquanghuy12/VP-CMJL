import torch
import torch.nn as nn
import torch.nn.functional as F

class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2, nheads=1):
        super(GATLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.nheads = nheads

        self.W = nn.Parameter(torch.zeros(size=(nheads, in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        self.a = nn.Parameter(torch.zeros(size=(nheads, 2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h):
        # h: (batch_size, number_of_nodes, in_features)
        batch_size = h.size(0)
        N = h.size(1)

        # Apply linear transformation W for each head
        # h_prime: (batch, nheads, N, out_features)
        h_prime = torch.einsum('bni,hio->bhno', h, self.W)

        # Attention mechanism
        # Prepare inputs for attention: concatenate all node pairs
        # We need to compute e_ij for all i, j. 
        # a_input: (batch, nheads, N, N, 2*out_features)
        
        # Broadcast to create pairs
        h_prime_i = h_prime.unsqueeze(3).repeat(1, 1, 1, N, 1) # (B, H, N, N, F)
        h_prime_j = h_prime.unsqueeze(2).repeat(1, 1, N, 1, 1) # (B, H, N, N, F)
        
        # Concatenate features
        a_input = torch.cat([h_prime_i, h_prime_j], dim=-1) # (B, H, N, N, 2*F)

        # Apply attention vector 'a'
        # e: (batch, nheads, N, N)
        e = self.leakyrelu(torch.einsum('bhnmf,hfz->bhnm', a_input, self.a).squeeze(-1))

        # Softmax to get attention weights
        attention = F.softmax(e, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        # Apply attention to features
        # h_prime: (batch, nheads, N, out_features)
        # attention: (batch, nheads, N, N)
        # output: (batch, nheads, N, out_features)
        h_new = torch.einsum('bhnm,bhmf->bhnf', attention, h_prime)

        # Average heads (or concat)
        # Here we average for simplicity and keeping dimension
        h_new = h_new.mean(dim=1) 
        
        return F.elu(h_new)

