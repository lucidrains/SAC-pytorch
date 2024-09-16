from __future__ import annotations
from collections import namedtuple

import torch
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList

from beartype import beartype

# ein notations
# b - batch
# n - number of actions

from einx import get_at
from einops import rearrange, repeat, reduce, pack, unpack

# EMA for target networks

from ema_pytorch import EMA

# constants

ContinuousOutput = namedtuple('ContinuousOutput', ['mu', 'sigma'])

SoftActorOutput = namedtuple('SoftActorOutput', ['continuous', 'discrete'])
SampledSoftActorOutput = namedtuple('SampledSoftActorOutput', ['continuous', 'discrete'])

# helpers

def exists(v):
    return v is not None

def cast_tuple(t, length = 1):
    return t if isinstance(t, tuple) else ((t,) * length)

# tensor helpers

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / temperature) + gumbel_noise(t)).argmax(dim = dim)

# mlp

@beartype
def MLP(
    dim,
    dim_out,
    dim_hiddens: int | tuple[int, ...],
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
    @beartype
    def __init__(
        self,
        *,
        dim_state,
        num_cont_actions,
        num_discrete_actions: tuple[int, ...] = (),
        dim_hiddens: tuple[int, ...] = (),
        eps = 1e-5
    ):
        super().__init__()
        self.eps = eps

        discrete_action_dims = sum(num_discrete_actions)
        cont_action_dims = num_cont_actions * 2

        self.num_discrete_actions = num_discrete_actions
        self.split_dims = (discrete_action_dims, cont_action_dims)

        self.to_actions = MLP(
            dim_state,
            dim_hiddens = dim_hiddens,
            dim_out = discrete_action_dims + cont_action_dims
        )


    def forward(
        self,
        state,
        sample = False,
        discrete_sample_temperature = 1.
    ):
        action_dims = self.to_actions(state)

        discrete_actions, cont_actions = action_dims.split(self.split_dims, dim = -1)
        discrete_action_logits = discrete_actions.split(self.num_discrete_actions)

        mu, sigma = rearrange(cont_actions, '... (n mu_sigma) -> mu_sigma ... n', mu_sigma = 2)
        sigma = sigma.sigmoid().clamp(min = self.eps)

        cont_output = ContinuousOutput(mu, sigma)
        discrete_output = discrete_action_logits

        if not sample:
            return SoftActorOutput(cont_output, discrete_output)

        sampled_cont = mu + sigma * torch.randn_like(sigma)
        sampled_discrete = [gumbel_sample(logits, temperature = discrete_sample_temperature) for logits in discrete_action_logits]

        return SampledSoftActorOutput(sampled_cont, sampled_discrete)

class Critic(Module):
    @beartype
    def __init__(
        self,
        *,
        dim_state,
        num_continuous_actions,
        dim_hiddens: tuple[int, ...] = (),
        layernorm = False,
        dropout = 0.
    ):
        super().__init__()

        self.to_q = MLP(
            dim_state + num_continuous_actions,
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
        state_actions, _ = pack([state, actions], 'b *')

        q_values = self.to_q(state_actions)
        q_values = rearrange('b 1 -> b')

        return q_values

class ValueNetwork(Module):
    @beartype
    def __init__(
        self,
        *,
        dim_state,
        dim_hiddens: tuple[int, ...] = ()
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
    @beartype
    def __init__(
        self
    ):
        super().__init__()

    def forward(self, x):
        return x
