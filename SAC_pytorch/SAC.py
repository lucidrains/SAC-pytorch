from __future__ import annotations

import math
from collections import namedtuple

import torch
from torch import nn, einsum, Tensor, tensor
from torch.distributions import Normal
from torch.nn import Module, ModuleList, Sequential

# tensor typing

from beartype import beartype

# ein notations
# b - batch
# n - number of actions

from einx import get_at
from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange

# EMA for target networks

from ema_pytorch import EMA

# constants

ContinuousOutput = namedtuple('ContinuousOutput', ['mu', 'sigma'])

SoftActorOutput = namedtuple('SoftActorOutput', ['continuous', 'discrete'])
SampledSoftActorOutput = namedtuple('SampledSoftActorOutput', ['continuous', 'discrete', 'continuous_log_prob', 'discrete_log_prob'])

# helpers

def exists(v):
    return v is not None

def compact(arr):
    return [*filter(exists, arr)]

def cast_tuple(t, length = 1):
    return t if isinstance(t, tuple) else ((t,) * length)

# tensor helpers

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.rand_like(t)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    assert temperature > 0.
    return ((t / temperature) + gumbel_noise(t)).argmax(dim = dim)

# helper classes

def Sequential(*modules):
    return nn.Sequential(*filter(exists, modules))

class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

# mlp

class MLP(Module):
    @beartype
    def __init__(
        self,
        dim,
        dim_out,
        dim_hiddens: int | tuple[int, ...],
        layernorm = False,
        dropout = 0.,
        activation = nn.ReLU,
        add_residual = False
    ):
        super().__init__()
        """
        simple mlp for Q and value networks

        following Figure 1 in https://arxiv.org/pdf/2110.02034.pdf for placement of dropouts and layernorm
        however, be aware that Levine in his lecture has ablations that show layernorm alone (without dropout) is sufficient for regularization
        """

        dim_hiddens = cast_tuple(dim_hiddens)

        layers = []

        curr_dim = dim

        for dim_hidden in dim_hiddens:

            layer = Sequential(
                nn.Linear(curr_dim, dim_hidden),
                nn.Dropout(dropout),
                nn.LayerNorm(dim_hidden) if layernorm else None,
                activation()
            )

            if add_residual:
                layer = Residual(layer)

            layers.append(layer)

            curr_dim = dim_hidden

        # final layer out

        layers.append(nn.Linear(curr_dim, dim_out))

        self.layers = ModuleList(layers)

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)

        return x

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

        self.num_cont_actions = num_cont_actions
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
        discrete_sample_temperature = 1.,
        discrete_sample_deterministic = False
    ):
        action_dims = self.to_actions(state)

        discrete_actions, cont_actions = action_dims.split(self.split_dims, dim = -1)
        discrete_action_logits = discrete_actions.split(self.num_discrete_actions, dim = -1)

        mu, sigma = rearrange(cont_actions, '... (n mu_sigma) -> mu_sigma ... n', mu_sigma = 2)
        sigma = sigma.sigmoid().clamp(min = self.eps)

        cont_output = ContinuousOutput(mu, sigma)
        discrete_output = discrete_action_logits

        if not sample:
            return SoftActorOutput(cont_output, discrete_output)

        # handle continuous

        sampled_cont_actions = mu + sigma * torch.randn_like(sigma)
        squashed_cont_actions = sampled_cont_actions.tanh() # tanh squashing

        cont_log_prob = Normal(mu, sigma).log_prob(sampled_cont_actions)
        cont_log_prob = cont_log_prob - log(1. - squashed_cont_actions ** 2, eps = self.eps)

        scaled_squashed_cont_actions = squashed_cont_actions * self.num_cont_actions

        # handle discrete

        sampled_discrete_actions = []
        discrete_log_probs = []

        for logits in discrete_action_logits:

            if discrete_sample_deterministic:
                sampled_action = logits.argmax(dim = -1)
            else:
                sampled_action = gumbel_sample(logits, temperature = discrete_sample_temperature)

            sampled_discrete_actions.append(sampled_action)

            log_probs = logits.log_softmax(dim = -1)
            discrete_log_prob = get_at('... [d], ... -> ...', log_probs, sampled_action)
            discrete_log_probs.append(discrete_log_prob)

        # return all sampled continuous and discrete actions with their associated log prob

        return SampledSoftActorOutput(
            scaled_squashed_cont_actions,
            sampled_discrete_actions,
            cont_log_prob,
            discrete_log_probs
        )

class Critic(Module):
    @beartype
    def __init__(
        self,
        *,
        dim_state,
        num_actions,
        dim_hiddens: tuple[int, ...] = (),
        layernorm = False,
        dropout = 0.,
        num_quantiles: int | None = None,
        quantiles: tuple[float, ...] | None = None
    ):
        super().__init__()
        assert not exists(num_quantiles) or num_quantiles > 0

        self.returning_quantiles = exists(num_quantiles)
        self.num_quantiles = num_quantiles

        self.to_values = MLP(
            dim_state + num_actions,
            dim_out = 1 if not self.returning_quantiles else num_quantiles,
            dim_hiddens = dim_hiddens,
            layernorm = layernorm,
            dropout = dropout
        )

        # excluding 0 and 1 - say 3 quantiles will be 0.25, 0.5, 0.75

        if exists(quantiles):
            quantiles = tensor(quantiles)
        else:
            quantiles = torch.linspace(0., 1., num_quantiles + 2)[1:-1]

        self.register_buffer('quantiles', quantiles)

    def forward(
        self,
        state,
        cont_actions = None
    ):
        pack_input = compact([state, cont_actions])

        mlp_input, _ = pack(pack_input, 'b *')

        values = self.to_values(mlp_input)

        if self.returning_quantiles:
            return values

        return rearrange(values, '... 1 -> ...')

class MultipleCritics(Module):
    @beartype
    def __init__(
        self,
        *critics: Critic
    ):
        super().__init__()
        assert len(critics) > 0
        assert all([not critic.returning_quantiles for critic in critics])

        self.num_critics = len(critics)
        self.critics = ModuleList(critics)

    def forward(
        self,
        states,
        cont_actions = None,
        target_value = None,
    ):
        values = [critic(states, cont_actions) for critic in self.critics], 'this wrapper only allows for non-quantile critics'

        if not exists(target_value):
            min_critic_value = torch.minimum(*values)
            return min_critic_value, values

        losses = [F.mse_loss(values, target) for value in values]
        return losses

class MultipleQuantileCritics(Module):
    @beartype
    def __init__(
        self,
        *critics: Critic,
        frac_atom_keep = 0.75 # will truncate 25% of the top values
    ):
        super().__init__()
        assert len(critics) > 0
        assert all([critic.returning_quantiles for critic in critics]), 'all critics must be returning quantiles'
        assert len(set([critic.num_quantiles for critic in critics])) == 1, 'all critics must have same number of quantiles'

        self.num_critics = len(critics)

        self.critics = ModuleList(critics)
        self.num_atom_keep = frac_atom_keep * self.num_critics * self.num_quantiles

    @property
    def quantiles(self):
        return self.critics[0].quantiles

    @property
    def num_quantiles(self):
        return self.critics[0].num_quantiles

    def forward(
        self,
        states,
        cont_actions = None,
        target_value = None,
    ):
        quantile_atoms = [critic(states, cont_actions) for critic in self.critics]

        if not exists(target_value):
            quantile_atoms = rearrange(quantile_atoms, 'c b ... q -> b ... (q c)')

            # mean of the truncated distribution

            truncated_quantiles = quantile_atoms.topk(self.num_atom_keep, largest = False).values
            truncated_mean = truncated_quantiles.mean()

            return truncated_mean, quantile_atoms

        # quantile regression if training

        target_value = torch.stack(target_value)
        quantile_atoms = torch.stack(quantile_atoms)
        quantiles = self.quantiles

        # quantile regression loss

        error = target_value - quantile_atoms
        losses = torch.maximum(error * quantiles, error * (quantiles - 1.))

        losses = reduce(losses, '... q -> ...', 'sum')
        return losses.mean()

# automatic entropy temperature adjustment
# will account for both continuous and discrete

class LearnedEntropyTemperature(Module):
    def __init__(
        self,
        num_discrete_actions = 0,
        num_cont_action = 0
    ):
        super().__init__()

        self.log_alpha = nn.Parameter(tensor(0.))

        self.has_discrete = num_discrete_actions > 0
        self.has_continuous = num_cont_actions > 0

        self.discrete_entropy_target = 0.98 * math.log(num_discrete_actions)
        self.continuous_entropy_target = num_cont_actions

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def forward(
        self,
        cont_log_prob = None,
        discrete_log_prob = None
    ):
        assert exists(cont_log_prob) or exists(discrete_log_prob)

        alpha = self.alpha

        losses = []

        if exists(discrete_log_prob):
            discrete_entropy_temp_loss = -alpha * (discrete_log_prob + self.discrete_entropy_target).detach()

            losses.append(discrete_entropy_temp_loss)

        if exists(cont_log_prob):
            cont_entropy_temp_loss = -alpha * (cont_log_prob + self.continuous_entropy_target).detach()

            losses.append(cont_entropy_temp_loss)

        return sum(losses).mean()

# main class

class SAC(Module):
    @beartype
    def __init__(
        self
    ):
        super().__init__()

    def forward(self, x):
        return x
