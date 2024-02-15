import torch
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList

from beartype import beartype
from beartype.typing import Tuple, List, Optional, Union

# ein notations

from einx import get_at
from einops import rearrange, repeat, reduce, pack, unpack

# EMA for target networks

from ema_pytorch import EMA

# helpers

def exists(v):
    return v is not None

def cast_tuple(t, length = 1):
    return t if isinstance(t, tuple) else ((t,) * length)

# mlp

@beartype
def MLP(
    dim,
    dim_out,
    dim_hiddens: Union[int, Tuple[int, ...]],
    layernorm = False,
    dropout = 0.,
    activation = nn.ReLU
):
    """
    simple mlp for Q and value networks

    following Figure 1 in https://arxiv.org/pdf/2110.02034.pdf for placement of dropouts and layernorm
    however, be aware that Levine in his lecture has ablations that show layernorm alone (without dropout) is sufficient for regularization
    """

    dim_hiddens = cast_tuple(dim_hiddens)

    layers = []

    curr_dim = dim

    for dim_hidden in dim_hiddens:
        layers.append(nn.Linear(curr_dim, dim_hidden))

        layers.append(nn.Dropout(dropout))

        if layernorm:
            layers.append(nn.LayerNorm(dim_hidden))

        layers.append(activation())

        curr_dim = dim_hidden

    # final layer out

    layers.append(nn.Linear(curr_dim, dim_out))

    return nn.Sequential(*layers)

# main modules

class Actor(Module):
    def __init__(
        self,
        dim_state
    ):
        super().__init__()
        raise NotImplementedError

class Critic(Module):
    @beartype
    def __init__(
        self,
        dim_state,
        dim_actions,
        dim_hiddens: Tuple[int, ...] = tuple(),
        layernorm = False,
        dropout = 0.
    ):
        super().__init__()

        self.to_q = MLP(
            dim_state + dim_actions,
            dim_out = 1,
            dim_hiddens = dim_hiddens,
            layernorm = layernorm,
            dropout = dropout
        )

    def forward(
        self,
        state,
        actions
    ):
        state_actions = pack([state, actions], 'b *')

        q_values = self.to_q(state_actions)
        q_values = rearrange('b 1 -> b')

        return q_values

class ValueNetwork(Module):
    @beartype
    def __init__(
        self,
        dim_state,
        dim_hiddens: Tuple[int, ...] = tuple()
    ):
        super().__init__()

        self.to_values = MLP(
            dim_state,
            dim_out= 1,
            dim_hiddens = dim_hiddens
        )

    def forward(
        self,
        states
    ):
        values = self.to_values(states)
        values = rearrange(values, 'b 1 -> b')
        return values

# main class

class SAC(Module):
    def __init__(
        self
    ):
        super().__init__()

    def forward(self, x):
        return x
