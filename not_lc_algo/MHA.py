import torch
import torch.nn as nn
import torch.nn.functional as F

class MHA(nn.Module):
    def __init__(self, dim, n_head, dropout=0.1):
        super(MHA, self).__init__()
        self.dim = dim
        self.n_head = n_head
        self.head_dim = self.dim // self.n_head
        assert self.dim == self.head_dim * self.n_head
        self.q_linear = nn.Linear(dim, dim)
        self.k_linear = nn.Linear(dim, dim)
        self.v_linear = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(dim, dim)
    
    def forward(self, x, mask=None):
        b, t, d = x.size()
        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)
        
        Q = Q.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        K = K.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        V = V.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        
        score = torch.matmul(Q, K.transpose(2, 3)) / torch.sqrt(torch.tensor(self.head_dim), dtype=torch.float32)
        
        if mask is None:
            mask = mask.unsqueeze(1)
            score = score.masked_fill(mask == 0, -1e9)
        score = F.softmax(score, dim=-1)
        if self.dropout is not None:
            score = self.dropout(score)
        
        output = torch.matmul(score, V).transpose(1, 2).contiguous().view(b, t, d)
        output = self.fc_out(output)
        return output