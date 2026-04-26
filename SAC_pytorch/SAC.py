from __future__ import annotations
from functools import partial

import math
from collections import namedtuple

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributions import Normal, Beta
from torch import nn, einsum, Tensor, tensor, cat, stack
from torch.nn import Module, ModuleList, Sequential

from adam_atan2_pytorch import AdoptAtan2 as Adopt

from hl_gauss_pytorch import HLGaussLoss, HLGaussLossFromSupport

# tensor typing

import jaxtyping
from beartype import beartype
from beartype.door import is_bearable

class TorchTyping:
    def __init__(self, abstract_dtype):
        self.abstract_dtype = abstract_dtype

    def __getitem__(self, shapes: str):
        return self.abstract_dtype[Tensor, shapes]

Float = TorchTyping(jaxtyping.Float)
Int   = TorchTyping(jaxtyping.Int)
Bool  = TorchTyping(jaxtyping.Bool)

# ein notations
# b - batch
# n - number of actions
# nc - number of continuous actions
# nd - number of discrete actions
# c - critics
# q - quantiles

from einx import get_at
from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange

# EMA for target networks

from ema_pytorch import EMA

# constants

ContinuousOutput = namedtuple('ContinuousOutput', [
    'mu',
    'sigma'
])

SoftActorOutput = namedtuple('SoftActorOutput', [
    'continuous',
    'discrete'
])

SampledSoftActorOutput = namedtuple('SampledSoftActorOutput', [
    'continuous',
    'continuous_log_prob',
    'continuous_entropy',
    'discrete',
    'discrete_action_logits',
])

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

def compact(arr):
    return [*filter(exists, arr)]

def identity(t):
    return t

def cast_tuple(t, length = 1):
    return t if isinstance(t, tuple) else ((t,) * length)

# tensor helpers

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def l2norm(t):
    return F.normalize(t, p = 2, dim = -1)

def entropy(t, eps = 1e-20):
    prob = t.softmax(dim = -1)
    return (-prob * log(prob, eps = eps)).sum(dim = -1)

def gumbel_noise(t):
    noise = torch.rand_like(t)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    assert temperature > 0.
    return ((t / temperature) + gumbel_noise(t)).argmax(dim = dim)

# expectile regression
# for expectile bellman proposed by https://arxiv.org/abs/2406.04081v1, which obviates need for multi critic for alleviating overestimation bias

def expectile_l2_loss(
    x,
    target,
    tau = 0.5,  # 0.5 would be the classic l2 loss - less would weigh negative higher, and more would weigh positive higher
    reduction = 'mean'
):
    assert 0 <= tau <= 1.
    assert reduction in {'mean', 'none'}

    if tau == 0.5:
        return F.mse_loss(x, target, reduction = reduction)

    diff = x - target

    weight = torch.where(diff < 0, tau, 1. - tau)

    loss = (weight * diff.square())

    if reduction == 'mean':
        loss = loss.mean()

    return loss

# orthogonal residual updates
# https://arxiv.org/abs/2505.11881

def orthog_project(x, y):
    dtype = x.dtype

    if x.device.type != 'mps':
        x, y = x.double(), y.double()

    unit = l2norm(y)
    parallel = (x * unit).sum(dim = -1, keepdim = True) * unit
    orthog = x - parallel

    return orthog.to(dtype)

# FIRE - Frobenius-Isometry Reinitialization
# Han et al. https://arxiv.org/abs/2602.08040

@torch.no_grad()
def apply_fire(
    module,
    num_iters = 20,
    coefs = (1.5, -0.5)
):
    a, b = coefs

    for p in module.parameters():
        if p.ndim != 2:
            continue

        t = p.data
        t_norm = t.norm()

        if t_norm == 0.:
            continue

        t = t / t_norm

        for _ in range(num_iters):
            A = t.T @ t
            t = a * t + b * (t @ A)

        p.data.copy_(t)

# distributed helpers

def is_distributed():
    return dist.is_initialized() and dist.get_world_size() > 1

def maybe_distributed_mean(t):
    if not is_distributed():
        return t

    dist.all_reduce(t)
    t = t / dist.get_world_size()
    return t

# helper classes

def Sequential(*modules):
    return nn.Sequential(*filter(exists, modules))

# RSM Norm (not to be confused with RMSNorm from transformers)
# this was proposed by SimBa https://arxiv.org/abs/2410.09754
# experiments show this to outperform other types of normalization

class RSMNorm(Module):
    def __init__(
        self,
        dim,
        eps = 1e-5,
        time_dilate_factor = 1.,
    ):
        # equation (3) in https://arxiv.org/abs/2410.09754
        super().__init__()
        self.dim = dim
        self.eps = 1e-5

        self.time_dilate_factor = time_dilate_factor

        self.register_buffer('step', tensor(1))
        self.register_buffer('running_mean', torch.zeros(dim))
        self.register_buffer('running_variance', torch.ones(dim))

    def reset_step(self):
        self.step.zero_()

    @property
    def time(self):
        return self.step / self.time_dilate_factor

    def forward(
        self,
        x
    ):
        assert x.shape[-1] == self.dim, f'expected feature dimension of {self.dim} but received {x.shape[-1]}'

        time = self.time.item()
        mean = self.running_mean
        variance = self.running_variance

        normed = (x - mean) / variance.sqrt().clamp(min = self.eps)

        if not self.training:
            return normed

        # update running mean and variance

        with torch.no_grad():

            new_obs_mean = maybe_distributed_mean(reduce(x, '... d -> d', 'mean'))
            delta = new_obs_mean - mean

            new_mean = mean + delta / time
            new_variance = (time - 1) / time * (variance + (delta ** 2) / time)

            self.step.add_(1)
            self.running_mean.copy_(new_mean)
            self.running_variance.copy_(new_variance)

        return normed

# Simplicial Embeddings
# Lavoie et al - https://arxiv.org/abs/2204.00616
# Obando-Ceron et al - https://openreview.net/forum?id=mCpq1GCKxA

class SEM(Module):
    def __init__(
        self,
        dim,
        temperature = 0.1,
        dim_simplex = 8,
        pre_layernorm = False
    ):
        super().__init__()
        assert divisible_by(dim, dim_simplex), f'{dim} must be divisible by {dim_simplex}'

        self.dim = dim
        self.dim_simplex = dim_simplex
        self.temperature = temperature

        self.norm = nn.LayerNorm(dim, bias = False) if pre_layernorm else nn.Identity()

    def forward(
        self,
        t
    ):
        t = self.norm(t)
        t = rearrange(t, '... (l v) -> ... l v', v = self.dim_simplex)
        t = (t / self.temperature).softmax(dim = -1)
        return rearrange(t, '... l v -> ... (l v)')

# SimBa - Kaist + SonyAI research

class ReluSquared(Module):
    def forward(self, x):
        return F.relu(x) ** 2

class SimBa(Module):

    @beartype
    def __init__(
        self,
        dim,
        dim_out,
        dim_hidden = None,
        depth = 3,
        dropout = 0.,
        expansion_factor = 2,
        final_norm = False,
        simplicial_embed = True
    ):
        super().__init__()
        """
        SimBa - https://arxiv.org/abs/2410.09754
        """

        dim_hidden = default(dim_hidden, dim * 2)

        layers = []

        self.proj_in = nn.Linear(dim, dim_hidden)

        dim_inner = dim_hidden * expansion_factor

        for _ in range(depth):

            layer = Sequential(
                nn.LayerNorm(dim_hidden, bias = False),
                nn.Linear(dim_hidden, dim_inner),
                nn.Dropout(dropout),
                ReluSquared(),
                nn.Linear(dim_inner, dim_hidden),
            )

            nn.init.constant_(layer[-1].weight, 1e-5)

            layers.append(layer)

        # final layer out

        self.layers = ModuleList(layers)

        self.final_norm = nn.LayerNorm(dim_hidden) if final_norm else nn.Identity()

        self.simplicial_embed = SEM(dim_hidden, pre_layernorm = not final_norm) if simplicial_embed else nn.Identity()

        self.proj_out = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden * 2),
            nn.LeakyReLU(),
            nn.Linear(dim_hidden * 2, dim_out)
        )

    def forward(self, x):

        x = self.proj_in(x)

        for layer in self.layers:
            layer_out = layer(x)
            x = x + orthog_project(layer_out, x)

        x = self.final_norm(x)

        x = self.simplicial_embed(x)

        return self.proj_out(x)

# distributions

def get_scale(source_range, target_range):
    source_min, source_max = source_range
    target_min, target_max = target_range
    return (target_max - target_min) / (source_max - source_min)

def rescale_from_to(t, source_range, target_range):
    source_min, _ = source_range
    target_min, _ = target_range
    scale = get_scale(source_range, target_range)
    return (t - source_min) * scale + target_min

class SquashedNormal(Module):
    def __init__(self, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.source_range = (-1., 1.)

    def process_params(self, params):
        mu, sigma = rearrange(params, '... (n mu_sigma) -> mu_sigma ... n', mu_sigma = 2)
        sigma = sigma.sigmoid().clamp(min = self.eps)
        return ContinuousOutput(mu, sigma)

    def forward(self, params, reparametrize = False):
        mu, sigma = self.process_params(params)

        if reparametrize:
            sampled = mu + sigma * torch.randn_like(sigma)
        else:
            sampled = torch.normal(mu, sigma)

        squashed = sampled.tanh()

        # log prob

        log_prob = Normal(mu, sigma).log_prob(sampled)
        log_prob = log_prob - 2 * (log(tensor(2.)) - sampled - F.softplus(-2 * sampled))

        # approximate entropy

        entropy = -log_prob

        return squashed, log_prob, entropy, self.source_range

class BetaDistribution(Module):
    def __init__(self, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.source_range = (0., 1.)

    def process_params(self, params):
        alpha, beta = rearrange(params, '... (n alpha_beta) -> alpha_beta ... n', alpha_beta = 2)

        # unimodal

        alpha = F.softplus(alpha) + 1. + self.eps
        beta = F.softplus(beta) + 1. + self.eps
        return ContinuousOutput(alpha, beta)

    def forward(self, params, reparametrize = False):
        alpha, beta = self.process_params(params)

        dist = Beta(alpha, beta)

        if not reparametrize:
            sampled = dist.sample()
        else:
            sampled = dist.rsample()

        # log prob

        sampled_for_log_prob = sampled.clamp(min = self.eps, max = 1. - self.eps)
        log_prob = dist.log_prob(sampled_for_log_prob)

        # entropy

        entropy = dist.entropy()

        return sampled, log_prob, entropy, self.source_range

# main modules

class Actor(Module):
    @beartype
    def __init__(
        self,
        *,
        dim_state,
        num_cont_actions,
        mlp_depth = 3,
        num_discrete_actions: tuple[int, ...] = (),
        dim_hidden = None,
        eps = 1e-5,
        rsmnorm_input = True,
        use_beta = False,
        simplicial_embed = False,
        target_range: tuple[float, float] | None = None,
    ):
        super().__init__()
        self.eps = eps
        self.rsmnorm = RSMNorm(dim_state) if rsmnorm_input else nn.Identity()

        discrete_action_dims = sum(num_discrete_actions)
        cont_action_dims = num_cont_actions * 2

        self.num_cont_actions = num_cont_actions
        self.num_discrete_actions = num_discrete_actions
        self.split_dims = (discrete_action_dims, cont_action_dims)

        self.to_actions = SimBa(
            dim_state,
            depth = mlp_depth,
            dim_hidden = dim_hidden,
            dim_out = discrete_action_dims + cont_action_dims,
            simplicial_embed = simplicial_embed
        )

        # continuous distribution

        cont_klass = BetaDistribution if use_beta else SquashedNormal
        self.cont_dist = cont_klass(eps = eps)

        self.target_range = target_range

    def forward(
        self,
        state: Float['b ...'],
        sample = False,
        cont_reparametrize = False,
        discrete_sample_temperature = 1.,
        discrete_sample_deterministic = False,
    ) -> (
        SoftActorOutput |
        SampledSoftActorOutput
    ):

        state = self.rsmnorm(state)

        action_dims = self.to_actions(state)

        discrete_actions, cont_actions = action_dims.split(self.split_dims, dim = -1)
        discrete_action_logits = discrete_actions.split(self.num_discrete_actions, dim = -1)

        if not sample:
            cont_output = self.cont_dist.process_params(cont_actions)
            return SoftActorOutput(cont_output, discrete_action_logits)

        # handle continuous

        sampled_cont_actions, cont_log_prob, cont_entropy, source_range = self.cont_dist(cont_actions, reparametrize = cont_reparametrize)

        if exists(self.target_range) and self.target_range != source_range:
            sampled_cont_actions = rescale_from_to(sampled_cont_actions, source_range, self.target_range)

            scale = get_scale(source_range, self.target_range)

            cont_log_prob = cont_log_prob - math.log(scale)
            cont_entropy = cont_entropy + math.log(scale)

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
            sampled_cont_actions,
            cont_log_prob,
            cont_entropy,
            stack(sampled_discrete_actions, dim = -1) if len(sampled_discrete_actions) > 0 else None,
            discrete_action_logits
        )

class Critic(Module):
    @beartype
    def __init__(
        self,
        *,
        dim_state,
        num_cont_actions,
        mlp_depth = 3,
        num_discrete_actions: tuple[int, ...] = (),
        dim_hidden = None,
        dropout = 0.,
        dim_out = 1,
        rsmnorm_input = True,
        simplicial_embed = False
    ):
        super().__init__()

        self.rsmnorm = RSMNorm(dim_state) if rsmnorm_input else nn.Identity()

        dim_out = default(dim_out, dim_state)

        num_actions_split = tensor((num_cont_actions, *num_discrete_actions))

        # determine the output dimension of the critic
        # which is the sum of all the actions (continous and discrete), multiplied by the number of quantiles

        critic_dim_out = num_actions_split * dim_out

        self.to_values = SimBa(
            dim_state + num_cont_actions,
            depth = mlp_depth,
            dim_out = critic_dim_out.sum().item(),
            dim_hidden = dim_hidden,
            dropout = dropout,
            simplicial_embed = simplicial_embed
        )

        # save the number of quantiles and the number of actions, for splitting out the output of the critic correctly

        self.num_actions_split = num_actions_split.tolist()

        self.dim_out = dim_out

        # for tensor typing

        self._n = num_cont_actions
        self._o = dim_out

    def forward(
        self,
        state: Float['b ...'],
        cont_actions: Float['b {self._n}'] | None = None
    ) -> tuple[Float['b ...'], ...]:

        state = self.rsmnorm(state)

        pack_input = compact([state, cont_actions])

        mlp_input, _ = pack(pack_input, 'b *')

        values = self.to_values(mlp_input)

        greater_one_output_dim = self.dim_out > 1

        if greater_one_output_dim:
            values = rearrange(values, '... (n o) -> ... n o', o = self.dim_out)

        split_dim = -2 if greater_one_output_dim else -1

        values = values.split(self.num_actions_split, dim = split_dim)

        return values

class MultipleCritics(Module):
    @beartype
    def __init__(
        self,
        *critics: Critic,
        use_softmin = False,
        expectile_l2_loss_tau = 0.5 # regular mse loss if 0.5
    ):
        super().__init__()
        assert len(critics) > 0
        assert all([critic.dim_out == 1 for critic in critics]), 'this wrapper only allows for critics that return a single predicted value per action'

        self.num_critics = len(critics)
        self.one_critic = len(critics) == 1
        self.critics = ModuleList(critics)

        self.use_softmin = use_softmin
        self.expectile_l2_loss_tau = expectile_l2_loss_tau

    def forward(
        self,
        states: Float['b ...'],
        cont_actions: Float['b nc'] | None = None,
        discrete_actions: Int['b nd'] | None = None,
        target_values: Float['b n'] | None = None,
        return_breakdown = False,
        **kwargs
    ):
        critics_values = [critic(states, cont_actions) for critic in self.critics]

        critics_values = [stack(one_value_for_critics) for one_value_for_critics in zip(*critics_values)]

        if not exists(target_values):

            min_critic_values = []

            for critic_values in critics_values:
                if self.use_softmin:
                    values = stack(values, dim = -1)
                    softmin = (-values).softmax(dim = -1)

                    min_critic_value = (softmin * critic_values).sum(dim = -1)

                elif self.one_critic:
                    min_critic_value = critic_values[0]

                else:
                    min_critic_value = torch.minimum(*critic_values)

                min_critic_values.append(min_critic_value)

            if not return_breakdown:
                return min_critic_values

            return min_critic_values, values

        cont_critics_values, *discrete_critics_values = critics_values

        if exists(discrete_actions):
            discrete_critics_values = [get_at('c b [l], b -> c b', discrete_critics_value, discrete_action) for discrete_critics_value, discrete_action in zip(discrete_critics_values, discrete_actions.unbind(dim = -1))]

        values, _ = pack([cont_critics_values, *discrete_critics_values], 'c b *')

        target_values = repeat(target_values, '... -> c ...', c = self.num_critics)

        losses = expectile_l2_loss(values, target_values, tau = self.expectile_l2_loss_tau, reduction = 'none')

        reduced_losses = reduce(losses, 'c b n -> b', 'sum').mean() # sum losses per action per critic, and average over batch

        if not return_breakdown:
            return reduced_losses

        return reduced_losses, losses

class MultipleCriticsWithClassificationLoss(Module):
    @beartype
    def __init__(
        self,
        *critics: Critic,
        hl_gauss_loss: dict | HLGaussLossFromSupport,
        use_softmin = False,
    ):
        super().__init__()
        assert len(critics) > 0
        assert all([critic.dim_out > 1 for critic in critics]), 'the critic must return multiple bins for classification loss'

        self.num_critics = len(critics)
        self.critics = ModuleList(critics)

        self.use_softmin = use_softmin

        if isinstance(hl_gauss_loss, dict):
            hl_gauss_loss = HLGaussLoss(**hl_gauss_loss)

        self.hl_gauss_loss = hl_gauss_loss

    def forward(
        self,
        states: Float['b ...'],
        cont_actions: Float['b nc'] | None = None,
        discrete_actions: Int['b nd'] | None = None,
        target_values: Float['b n'] | None = None,
        return_breakdown = False,
        **kwargs
    ):
        raw_critics_values = [critic(states, cont_actions) for critic in self.critics]

        binned_critics_values = []
        critics_values = []

        for one_value_for_critics in zip(*raw_critics_values):
            stacked_critic_values = stack(one_value_for_critics)

            binned_critics_values.append(stacked_critic_values)

            stacked_critic_values = self.hl_gauss_loss(stacked_critic_values)

            critics_values.append(stacked_critic_values)

        if not exists(target_values):

            min_critic_values = []

            for critic_values in critics_values:
                if self.use_softmin:
                    values = stack(values, dim = -1)
                    softmin = (-values).softmax(dim = -1)

                    min_critic_value = (softmin * critic_values).sum(dim = -1)
                else:
                    min_critic_value = torch.minimum(*critic_values)

                min_critic_values.append(min_critic_value)

            if not return_breakdown:
                return min_critic_values

            return min_critic_values, values

        cont_critics_values, *discrete_critics_values = binned_critics_values

        if exists(discrete_actions):
            discrete_critics_values = [get_at('c b [l] bins, b -> c b bins', discrete_critics_value, discrete_action) for discrete_critics_value, discrete_action in zip(discrete_critics_values, discrete_actions.unbind(dim = -1))]

        values, _ = pack([cont_critics_values, *discrete_critics_values], 'c b * bins')

        target_values = repeat(target_values, '... -> c ...', c = self.num_critics)

        # from "Stop Regressing" paper out of deepmind, Farebrother et al

        cross_entropy_losses = self.hl_gauss_loss(values, target_values, reduction = 'none')

        reduced_losses = reduce(cross_entropy_losses, 'c b n -> b', 'sum').mean() # sum losses per action per critic, and average over batch

        if not return_breakdown:
            return reduced_losses

        return reduced_losses, cross_entropy_losses

class MultipleQuantileCritics(Module):
    @beartype
    def __init__(
        self,
        *critics: Critic,
        quantiles: list[float] | Tensor | None = None,
        frac_atom_keep = 0.75 # will truncate 25% of the top values
    ):
        super().__init__()
        assert len(critics) > 0
        assert all([critic.dim_out > 1 for critic in critics]), 'all critics must be returning greater than one output dimension, assumed as quantiles'
        assert len(set([critic.dim_out for critic in critics])) == 1, 'all critics must have same number of output dimensions for quantiled training'

        self.num_critics = len(critics)

        self.critics = ModuleList(critics)
        self.num_atom_keep = int(frac_atom_keep * self.num_critics * self.num_quantiles)

        # quantiles

        num_quantiles = self.num_quantiles
        quantiles = torch.linspace(0., 1., num_quantiles + 2)[1:-1] # excluding 0 and 1 - say 3 quantiles will be 0.25, 0.5, 0.75

        self.register_buffer('quantiles', quantiles)

    @property
    def num_quantiles(self):
        return self.critics[0].dim_out

    def forward(
        self,
        states: Float['b ...'],
        cont_actions: Float['b nc'] | None = None,
        discrete_actions: Int['b nd'] | None = None,
        target_values: Float['b n q'] | None = None,
        truncate_quantiles_across_critics = False,
        return_breakdown = False
    ):
        critics_quantile_atoms = [critic(states, cont_actions) for critic in self.critics]

        critics_quantile_atoms = [stack(one_value_for_critics) for one_value_for_critics in zip(*critics_quantile_atoms)]

        if not exists(target_values):

            if truncate_quantiles_across_critics:
                truncated_means_per_action = []

                for critic_quantile_atoms in critics_quantile_atoms:
                    quantile_atoms = rearrange(critic_quantile_atoms, 'c b ... q -> b ... (q c)')

                    # mean of the truncated distribution

                    truncated_quantiles = quantile_atoms.topk(self.num_atom_keep, largest = False).values
                    truncated_mean = truncated_quantiles.mean(dim = -1)

                    truncated_means_per_action.append(truncated_mean)

                return_value = truncated_means_per_action
            else:

                min_critic_value_per_action = []

                for critic_quantile_atoms in critics_quantile_atoms:
                    min_critic_value = torch.minimum(*critic_quantile_atoms)
                    min_critic_value_per_action.append(min_critic_value)

                return_value = min_critic_value_per_action

            if not return_breakdown:
                return return_value

            return return_value, critics_quantile_atoms

        cont_quantile_atoms, *discrete_quantile_atoms = critics_quantile_atoms

        if exists(discrete_actions):
            discrete_quantile_atoms = [get_at('c b [l] q, b -> c b q', discrete_quantile_atom, discrete_action) for discrete_quantile_atom, discrete_action in zip(discrete_quantile_atoms, discrete_actions.unbind(dim = -1))]

        quantile_atoms, _ = pack([cont_quantile_atoms, *discrete_quantile_atoms], 'c b * q')

        # quantile regression if training

        quantiles = self.quantiles

        # quantile regression loss

        error = target_values - quantile_atoms
        losses = torch.maximum(error * quantiles, error * (quantiles - 1.))

        reduced_losses = reduce(losses, 'c b n q -> b', 'sum').mean()

        if not return_breakdown:
            return reduced_losses

        return reduced_losses, losses

# automatic entropy temperature adjustment
# will account for both continuous and discrete

class LearnedEntropyTemperature(Module):
    def __init__(
        self,
        num_discrete_actions = 0,
        num_cont_actions = 0
    ):
        super().__init__()

        self.log_alpha = nn.Parameter(tensor(0.))

        self.has_discrete = len(num_discrete_actions) > 0
        self.has_continuous = num_cont_actions > 0

        self.discrete_entropy_targets = [0.98 * math.log(one_num_discrete_actions) for one_num_discrete_actions in num_discrete_actions]
        self.continuous_entropy_target = num_cont_actions

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def forward(
        self,
        cont_entropy: Float['b nc'] | None = None,
        discrete_logits: tuple[Float['b _'], ...] | None = None,
        return_breakdown = False
    ):
        assert exists(cont_entropy) or exists(discrete_logits)

        alpha = self.alpha

        losses = []

        if exists(discrete_logits):

            for one_discrete_logits, discrete_entropy_target in zip(discrete_logits, self.discrete_entropy_targets):
                discrete_entropy = entropy(one_discrete_logits)
                discrete_entropy_temp_loss = -alpha * (discrete_entropy_target - discrete_entropy).detach()

                losses.append(discrete_entropy_temp_loss.mean())

        if exists(cont_entropy):
            cont_entropy_temp_loss = -alpha * (self.continuous_entropy_target - cont_entropy).detach()

            cont_entropy_temp_loss = cont_entropy_temp_loss.mean(dim = 0)
            losses.append(cont_entropy_temp_loss)

        reduced_losses, _ = pack(losses, '*')
        reduced_losses = reduced_losses.sum()

        if not return_breakdown:
            return reduced_losses

        return reduced_losses, losses

# main class

class SAC(Module):
    @beartype
    def __init__(
        self,
        actor: dict | Actor,
        critics: (
            list[dict] |
            list[Critic] |
            MultipleCritics |
            MultipleQuantileCritics |
            MultipleCriticsWithClassificationLoss
        ),
        quantiled_critics = False,
        hl_gauss_loss: dict | HLGaussLossFromSupport | None = None,
        reward_discount_rate = 0.99,
        reward_scale = 1.,
        actor_learning_rate = 3e-4,
        actor_regen_reg_rate = 1e-4,
        critic_target_ema_decay = 0.99,
        critics_learning_rate = 3e-4,
        critics_regen_reg_rate = 1e-4,
        use_minto = False,
        multiple_critics_kwargs: dict = dict(),
        temperature_learning_rate = 3e-4,
        ema_kwargs: dict = dict(),
        actor_update_freq = 2,
        fire_every: int | None = None,
        apply_fire_actor: bool = True,
        apply_fire_critic: bool = True,
        fire_num_iters: int = 20
    ):
        super().__init__()

        # set actor

        if isinstance(actor, dict):
            actor = Actor(**actor)

        self.actor = actor

        self.actor_optimizer = Adopt(
            actor.parameters(),
            lr = actor_learning_rate,
            regen_reg_rate = actor_regen_reg_rate
        )
        # based on the actor hyperparameters, init the learned temperature container

        self.learned_entropy_temperature = LearnedEntropyTemperature(
            num_cont_actions = actor.num_cont_actions,
            num_discrete_actions = actor.num_discrete_actions
        )

        self.temperature_optimizer = Adopt(
            self.learned_entropy_temperature.parameters(),
            lr = temperature_learning_rate,
        )

        # set critics
        # allow for usual double+ critic trick or truncated quantiled critics

        if is_bearable(critics, list[dict]):
            critics = [Critic(**critic) for critic in critics]

        critic_dim_outs = {critic.dim_out for critic in critics}

        assert len(critic_dim_outs) == 1, 'critics must all have the same output dimension'

        critic_dim_out = list(critic_dim_outs)[0]

        critic_kwargs = dict()

        if critic_dim_out == 1:
            critic_klass = MultipleCritics
            critic_kwargs = multiple_critics_kwargs
        elif critic_dim_out > 1 and not quantiled_critics:
            assert exists(hl_gauss_loss), 'hl_gauss_loss must be set'
            critic_klass = MultipleCriticsWithClassificationLoss
            critic_kwargs = dict(hl_gauss_loss = hl_gauss_loss)
        else:
            critic_klass = MultipleQuantileCritics

        if is_bearable(critics, list[Critic]):
            critics = critic_klass(*critics, **critic_kwargs)

        assert isinstance(critics, critic_klass), f'expected {critic_klass.__name__} but received critics wrapped with {type(critics).__name__}'

        self.critics = critics

        self.quantiled_critics = quantiled_critics

        # critic optimizers

        self.critics_optimizer = Adopt(
            critics.parameters(),
            lr = critics_learning_rate,
            regen_reg_rate = critics_regen_reg_rate
        )

        # target critic network

        self.critics_target = EMA(
            critics,
            beta = critic_target_ema_decay,
            include_online_model = False,
            **ema_kwargs
        )

        # minto - taking the minimum of both online and ema

        self.use_minto = use_minto

        # reward related

        self.reward_scale = reward_scale
        self.reward_discount_rate = reward_discount_rate

        # maybe fire

        self.fire_every = fire_every
        self.apply_fire_actor = apply_fire_actor
        self.apply_fire_critic = apply_fire_critic
        self.fire_num_iters = fire_num_iters

        # steps and frequency of actor update

        self.actor_update_freq = actor_update_freq
        self.register_buffer('step', tensor(0))

    @torch.no_grad()
    def apply_fire_(
        self,
        num_iters = 20,
        coefs = (1.5, -0.5)
    ):
        if self.apply_fire_actor:
            apply_fire(self.actor, num_iters = num_iters, coefs = coefs)

        if self.apply_fire_critic:
            for critic in self.critics.critics:
                apply_fire(critic, num_iters = num_iters, coefs = coefs)

            if exists(self.critics_target):
                self.critics_target.copy_params_from_model_to_ema()

    def forward(
        self,
        states: Float['b ...'],
        cont_actions: Float['b nc'],
        discrete_actions: Int['b nd'],
        rewards: Float['b'],
        done: Bool['b'],
        next_states: Float['b ...']
    ):

        rewards = rewards * self.reward_scale

        # bellman equation
        # todo: setup n-step

        entropy_temp = self.learned_entropy_temperature.alpha.detach()

        γ = self.reward_discount_rate
        not_terminal = (~done).float()

        # outputs from actor for the next states

        with torch.no_grad():
            next_actor_output = self.actor(next_states, sample = True)

            next_cont_actions = next_actor_output.continuous
            next_cont_log_prob = next_actor_output.continuous_log_prob
            next_discrete_logits = next_actor_output.discrete_action_logits

            # forward for critic predictions
            # using EMA, but also using online, if Minto is turned on - Ahmed Hendawy et al. https://arxiv.org/abs/2510.02590

            self.critics_target.eval()
            next_cont_q_value, *next_discrete_q_values = self.critics_target(next_states, cont_actions = next_cont_actions)

            if self.use_minto:
                was_training = self.critics.training
                self.critics.eval()

                online_next_cont_q_value, *online_next_discrete_q_values = self.critics(next_states, cont_actions = next_cont_actions)

                next_cont_q_value = torch.minimum(next_cont_q_value, online_next_cont_q_value)
                next_discrete_q_values = tuple(torch.minimum(*ema_and_online) for ema_and_online in zip(next_discrete_q_values, online_next_discrete_q_values))

                self.critics.train(was_training)

        # learned temperature

        learned_entropy_weight = self.learned_entropy_temperature.alpha

        next_soft_state_values = []

        # first handle continuous soft state value

        if exists(next_cont_log_prob):
            next_cont_log_prob = next_cont_log_prob.sum(dim = -1, keepdim = True)

            if self.quantiled_critics:
                next_cont_log_prob = rearrange(next_cont_log_prob, '... -> ... 1')

            cont_soft_state_value = next_cont_q_value - learned_entropy_weight * next_cont_log_prob
            next_soft_state_values.append(cont_soft_state_value)

        # then handle discrete contribution

        if len(next_discrete_q_values) > 0:

            for next_discrete_q_value, next_discrete_logit in zip(next_discrete_q_values, next_discrete_logits):
                next_discrete_prob = next_discrete_logit.softmax(dim = -1)
                next_discrete_log_prob = log(next_discrete_prob)

                if self.quantiled_critics:
                    next_discrete_prob = rearrange(next_discrete_prob, '... -> ... 1')
                    next_discrete_log_prob = rearrange(next_discrete_log_prob, '... -> ... 1')

                discrete_soft_state_value = (next_discrete_prob * (next_discrete_q_value - learned_entropy_weight * next_discrete_log_prob)).sum(dim = 1)
                next_soft_state_values.append(discrete_soft_state_value)

        # quantile vs not

        if self.quantiled_critics:
            next_soft_state_values, _ = pack(next_soft_state_values, 'b * q')
            rewards = rearrange(rewards, 'b -> b 1 1')
            not_terminal = rearrange(not_terminal, 'b -> b 1 1')
        else:
            next_soft_state_values, _ = pack(next_soft_state_values, 'b *')
            rewards = rearrange(rewards, 'b -> b 1')
            not_terminal = rearrange(not_terminal, 'b -> b 1')

        # target q value

        target_q_values = rewards + not_terminal * γ * next_soft_state_values

        critics_losses = self.critics(
            states,
            cont_actions = cont_actions,
            discrete_actions = discrete_actions,
            target_values = target_q_values
        )

        # update the critics

        critics_losses.backward()
        self.critics_optimizer.step()
        self.critics_optimizer.zero_grad()

        if divisible_by(int(self.step.item()), self.actor_update_freq):

            # update the actor

            actor_output = self.actor(states, sample = True, cont_reparametrize = True)

            cont_entropy = actor_output.continuous_entropy
            discrete_logits = actor_output.discrete_action_logits

            cont_q_values, *discrete_q_values = self.critics(
                states,
                cont_actions = actor_output.continuous,
                discrete_actions = actor_output.discrete,
                truncate_quantiles_across_critics = True
            )

            actor_action_losses = []

            if exists(cont_entropy):
                cont_entropy_summed = cont_entropy.sum(dim = -1, keepdim = True)
                cont_action_loss = (-entropy_temp * cont_entropy_summed - cont_q_values).mean()

                actor_action_losses.append(cont_action_loss)

            for discrete_logit, one_discrete_q_value in zip(discrete_logits, discrete_q_values):
                discrete_prob = discrete_logit.softmax(dim = -1)
                discrete_log_prob = log(discrete_prob)

                one_discrete_actor_loss = (discrete_prob * (entropy_temp * discrete_log_prob - one_discrete_q_value)).sum(dim = -1).mean()

                actor_action_losses.append(one_discrete_actor_loss)

            sum(actor_action_losses).backward()
            self.actor_optimizer.step()
            self.actor_optimizer.zero_grad()

            # update the learned entropy temperature

            temperature_loss = self.learned_entropy_temperature(
                cont_entropy = cont_entropy,
                discrete_logits = discrete_logits
            )

            temperature_loss.backward()
            self.temperature_optimizer.step()
            self.temperature_optimizer.zero_grad()

        # increment step, update ema

        self.step.add_(1)

        self.critics_target.update()

        # maybe fire

        step = int(self.step.item())

        if exists(self.fire_every) and step > 0 and divisible_by(step, self.fire_every):
            self.apply_fire_(num_iters = self.fire_num_iters)
