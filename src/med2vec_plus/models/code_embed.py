import torch
import torch.nn as nn

class NonNegEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.weight_raw = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        nn.init.xavier_uniform_(self.weight_raw)

    @property
    def weight(self):
        return torch.nn.functional.softplus(self.weight_raw)

    def forward(self, indices: torch.Tensor):
        return torch.nn.functional.embedding(indices, self.weight)
