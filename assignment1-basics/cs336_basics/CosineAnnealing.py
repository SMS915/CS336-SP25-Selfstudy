import torch
from math import pi, cos
from jaxtyping import Float

def CosineAnnealing(t: int, t_warm: int, t_cycle: int, lr_max: float, lr_min: float) -> Float:
    return t * lr_max / t_warm if t < t_warm else lr_min if t > t_cycle else lr_min + (lr_max - lr_min) * (1 + cos((t - t_warm) * pi / (t_cycle - t_warm))) / 2