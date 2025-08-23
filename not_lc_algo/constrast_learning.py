import torch.nn.functional as F
import torch
import torch.nn as nn

## infoNCE变体
def calCL(Emb1, Emb2, idx, tau):
    emb1 = F.embedding(idx, Emb1)
    emb2 = F.embedding(idx, Emb2)

    norm_emb1 = F.normalize(emb1, p=2, dim=-1)
    norm_emb2 = F.normalize(emb2, p=2, dim=-1)

    similarity_matrix = torch.matmul(norm_emb1, norm_emb2)

    positive_similarity = torch.diag(similarity_matrix)
    positive_term = torch.exp(positive_similarity / tau)
    negtive_term = torch.sum(torch.exp(similarity_matrix / tau), dim=-1)
    
    loss = - torch.sum(torch.log(positive_term / negtive_term))
    return loss