import numpy as np
import torch

def Load_Data(x: np.array, batch_size: int, context_length: int, device: str | torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    idxs = np.random.randint(0, x.shape[0] - context_length, batch_size)
    inputs = np.stack([x[i : i + context_length] for i in idxs])
    outputs = np.stack([x[i + 1 : i + context_length + 1] for i in idxs])
    return torch.tensor(inputs).to(device), torch.tensor(outputs).to(device)



