import torch
import torch.nn as nn

class Linear(nn.Module):
    # stop auto complete
    def __init__(self, in_features : int, out_features : int, device : torch.device | None =None, dtype : torch.dtype | None =None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        self.W = nn.Parameter(torch.empty(self.out_features, self.in_features, device=self.device, dtype=self.dtype))

        std = (2 / (self.in_features + self.out_features)) ** 0.5
        nn.init.trunc_normal_(self.W, mean = 0, std = std, a = -3 * std, b = 3 * std)

    def forward(self, x):
        return x @ self.W.T