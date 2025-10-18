import torch
import torch.nn as nn

class Embedding(nn.Module):
    # stop auto complete unless I wrote for help
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None, *args, **kwargs):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        self.embed_matrix = nn.Parameter(torch.empty(self.num_embeddings, self.embedding_dim, device = self.device, dtype = self.dtype, requires_grad=True))
        nn.init.trunc_normal_(self.embed_matrix, mean = 0, std = 1, a = -3, b = 3)

    def forward(self, token_ids : torch.Tensor) -> torch.Tensor:
        return self.embed_matrix[token_ids]
