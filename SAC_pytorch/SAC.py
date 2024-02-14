import torch
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList

# ein notations

from einx import get_at
from einops import rearrange, repeat, reduce, pack, unpack

# helpers

def exists(v):
    return v is not None

# main modules

class Actor(Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError

class Critic(Module):
    """ will be 2 critics, with min of their Qs """

    def __init__(self):
        super().__init__()
        raise NotImplementedError

class Value(Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError

# main class

class SAC(Module):
    def __init__(
        self
    ):
        super().__init__()

    def forward(self, x):
        return x
