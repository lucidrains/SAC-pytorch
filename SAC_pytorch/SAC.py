from __future__ import annotations

import math
from collections import namedtuple

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributions import Normal
from torch import nn, einsum, Tensor, tensor
from torch.nn import Module, ModuleList, Sequential

from adam_atan2_pytorch.adam_atan2_with_wasserstein_reg import AdamAtan2 as Adam

from hyper_connections import HyperConnections

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
    'discrete',
    'discrete_action_logits',
])

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def compact(arr):
    return [*filter(exists, arr)]

def identity(t):
    return t

def cast_tuple(t, length = 1):
    return t if isinstance(t, tuple) else ((t,) * length)

# tensor helpers

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def entropy(t, eps = 1e-20):
    prob = t.softmax(dim = -1)
    return (-prob * log(prob, eps = eps)).sum(dim = -1)

def gumbel_noise(t):
    noise = torch.rand_like(t)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    assert temperature > 0.
    return ((t / temperature) + gumbel_noise(t)).argmax(dim = dim)

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
        num_residual_streams = 4
    ):
        super().__init__()
        """
        SimBa - https://arxiv.org/abs/2410.09754
        """

        dim_hidden = default(dim_hidden, dim * 2)

        layers = []

        self.proj_in = nn.Linear(dim, dim_hidden)

        dim_inner = dim_hidden * expansion_factor

        init_hyper_conn, self.expand_streams, self.reduce_streams = HyperConnections.get_init_and_expand_reduce_stream_functions(num_residual_streams, disable = num_residual_streams == 1)

        for _ in range(depth):

            layer = Sequential(
                nn.LayerNorm(dim_hidden, bias = False),
                nn.Linear(dim_hidden, dim_inner),
                nn.Dropout(dropout),
                ReluSquared(),
                nn.Linear(dim_inner, dim_hidden),
            )

            nn.init.constant_(layer[-1].weight, 1e-5)

            layers.append(init_hyper_conn(dim = dim_hidden, branch = layer))

        # final layer out

        self.layers = ModuleList(layers)

        self.final_norm = nn.LayerNorm(dim_hidden) if final_norm else nn.Identity()

        self.proj_out = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):

        x = self.proj_in(x)

        x = self.expand_streams(x)

        for layer in self.layers:
            x = layer(x)

        x = self.reduce_streams(x)

        x = self.final_norm(x)
        return self.proj_out(x)

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
        rsmnorm_input = True
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
            dim_out = discrete_action_dims + cont_action_dims
        )


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

        mu, sigma = rearrange(cont_actions, '... (n mu_sigma) -> mu_sigma ... n', mu_sigma = 2)
        sigma = sigma.sigmoid().clamp(min = self.eps)

        cont_output = ContinuousOutput(mu, sigma)
        discrete_output = discrete_action_logits

        if not sample:
            return SoftActorOutput(cont_output, discrete_output)

        # handle continuous

        if cont_reparametrize:
            sampled_cont_actions = mu + sigma * torch.randn_like(sigma)
        else:
            sampled_cont_actions = torch.normal(mu, sigma)

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
            torch.stack(sampled_discrete_actions, dim = -1),
            cont_log_prob,
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
        layernorm = False,
        dropout = 0.,
        num_quantiles: int | None = None,
        quantiles: tuple[float, ...] | None = None,
        rsmnorm_input = True
    ):
        super().__init__()
        assert not exists(num_quantiles) or num_quantiles > 0

        self.rsmnorm = RSMNorm(dim_state) if rsmnorm_input else nn.Identity()

        use_quantiles = exists(num_quantiles)
        self.returning_quantiles = use_quantiles

        num_actions_split = tensor((num_cont_actions, *num_discrete_actions))

        # determine the output dimension of the critic
        # which is the sum of all the actions (continous and discrete), multiplied by the number of quantiles

        critic_dim_out = num_actions_split
        if use_quantiles:
            critic_dim_out = critic_dim_out * num_quantiles

        self.to_values = SimBa(
            dim_state + num_cont_actions,
            depth = mlp_depth,
            dim_out = critic_dim_out.sum().item(),
            dim_hidden = dim_hidden,
            dropout = dropout
        )

        # save the number of quantiles and the number of actions, for splitting out the output of the critic correctly

        self.num_quantiles = num_quantiles
        self.num_actions_split = num_actions_split.tolist()

        # for tensor typing

        self._n = num_cont_actions
        self._q = num_quantiles

        # quantiles

        if exists(quantiles):
            quantiles = tensor(quantiles)
        elif exists(num_quantiles):
            quantiles = torch.linspace(0., 1., num_quantiles + 2)[1:-1] # excluding 0 and 1 - say 3 quantiles will be 0.25, 0.5, 0.75

        self.register_buffer('quantiles', quantiles)

    def forward(
        self,
        state: Float['b ...'],
        cont_actions: Float['b {self._n}'] | None = None
    ) -> tuple[Float['b ...'], ...]:

        state = self.rsmnorm(state)

        pack_input = compact([state, cont_actions])

        mlp_input, _ = pack(pack_input, 'b *')

        values = self.to_values(mlp_input)

        if self.returning_quantiles:
            values = rearrange(values, '... (n q) -> ... n q', q = self.num_quantiles)

        split_dim = -2 if self.returning_quantiles else -1

        values = values.split(self.num_actions_split, dim = split_dim)

        return values

class MultipleCritics(Module):
    @beartype
    def __init__(
        self,
        *critics: Critic,
        use_softmin = False
    ):
        super().__init__()
        assert len(critics) > 0
        assert all([not critic.returning_quantiles for critic in critics]), 'this wrapper only allows for non-quantile critics'

        self.num_critics = len(critics)
        self.critics = ModuleList(critics)

        self.use_softmin = use_softmin

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

        critics_values = [torch.stack(one_value_for_critics) for one_value_for_critics in zip(*critics_values)]

        if not exists(target_values):

            min_critic_values = []

            for critic_values in critics_values:
                if self.use_softmin:
                    values = torch.stack(values, dim = -1)
                    softmin = (-values).softmax(dim = -1)

                    min_critic_value = (softmin * critic_values).sum(dim = -1)
                else:
                    min_critic_value = torch.minimum(*critic_values)

                min_critic_values.append(min_critic_value)

            if not return_breakdown:
                return min_critic_values

            return min_critic_values, values

        cont_critics_values, *discrete_critics_values = critics_values

        discrete_critics_values = [get_at('c b [l], b -> c b', discrete_critics_value, discrete_action) for discrete_critics_value, discrete_action in zip(discrete_critics_values, discrete_actions.unbind(dim = -1))]

        values, _ = pack([cont_critics_values, *discrete_critics_values], 'c b *')

        target_values = repeat(target_values, '... -> c ...', c = self.num_critics)
        losses = F.mse_loss(values, target_values, reduction = 'none')

        reduced_losses = reduce(losses, 'c b n -> b', 'sum').mean() # sum losses per action per critic, and average over batch

        if not return_breakdown:
            return reduced_losses

        return reduced_losses, losses

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
        self.num_atom_keep = int(frac_atom_keep * self.num_critics * self.num_quantiles)

    @property
    def quantiles(self):
        return self.critics[0].quantiles

    @property
    def num_quantiles(self):
        return self.critics[0].num_quantiles

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

        critics_quantile_atoms = [torch.stack(one_value_for_critics) for one_value_for_critics in zip(*critics_quantile_atoms)]

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

        self.discrete_entropy_targets = [0.98 * math.log(one_num_discrete_actions)for one_num_discrete_actions in num_discrete_actions] 
        self.continuous_entropy_target = num_cont_actions

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def forward(
        self,
        cont_log_prob: Float['b nc'] | None = None,
        discrete_logits: tuple[Float['b _'], ...] | None = None,
        return_breakdown = False
    ):
        assert exists(cont_log_prob) or exists(discrete_log_prob)

        alpha = self.alpha

        losses = []

        if exists(discrete_logits):

            for one_discrete_logits, discrete_entropy_target in zip(discrete_logits, self.discrete_entropy_targets):
                discrete_log_prob = one_discrete_logits.log_softmax(dim = -1)
                discrete_entropy_temp_loss = -alpha * (discrete_log_prob + discrete_entropy_target).detach()

                losses.append(discrete_entropy_temp_loss.mean())

        if exists(cont_log_prob):
            cont_entropy_temp_loss = -alpha * (cont_log_prob + self.continuous_entropy_target).detach()

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
            MultipleQuantileCritics
        ),
        quantiled_critics = False,
        reward_discount_rate = 0.99,
        reward_scale = 1.,
        actor_learning_rate = 3e-4,
        actor_regen_reg_rate = 1e-4,
        critic_target_ema_decay = 0.99,
        critics_learning_rate = 3e-4,
        critics_regen_reg_rate = 1e-4,
        temperature_learning_rate = 3e-4,
        ema_kwargs: dict = dict()
    ):
        super().__init__()

        # set actor

        if isinstance(actor, dict):
            actor = Actor(**actor)

        self.actor = actor

        self.actor_optimizer = Adam(
            actor.parameters(),
            lr = actor_learning_rate,
            regen_reg_rate = actor_regen_reg_rate
        )
        # based on the actor hyperparameters, init the leraned temperature container

        self.learned_entropy_temperature = LearnedEntropyTemperature(
            num_cont_actions = actor.num_cont_actions,
            num_discrete_actions = actor.num_discrete_actions
        )

        self.temperature_optimizer = Adam(
            self.learned_entropy_temperature.parameters(),
            lr = temperature_learning_rate,
        )

        # set critics
        # allow for usual double+ critic trick or truncated quantiled critics

        if is_bearable(critics, list[dict]):
            critics = [Critic(**critic) for critic in critics]

        multiple_critic_klass = MultipleCritics if not quantiled_critics else MultipleQuantileCritics

        if is_bearable(critics, list[Critic]):
            critics = multiple_critic_klass(*critics)

        assert isinstance(critics, multiple_critic_klass), f'expected {multiple_critic_klass.__name__} but received critics wrapped with {type(critics).__name__}'

        self.critics = critics

        self.quantiled_critics = quantiled_critics

        # critic optimizers

        self.critics_optimizer = Adam(
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

        # reward related

        self.reward_scale = reward_scale
        self.reward_discount_rate = reward_discount_rate

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

        with torch.no_grad():
            self.critics_target.eval()

            next_cont_q_value, *next_discrete_q_values = self.critics_target(next_states, cont_actions = cont_actions)

            # outputs from actor

            actor_output = self.actor(states, sample = True)

            cont_log_prob = actor_output.continuous_log_prob
            discrete_logits = actor_output.discrete_action_logits

            # learned temperature

            learned_entropy_weight = self.learned_entropy_temperature.alpha

            next_soft_state_values = []

            # first handle continuous soft state value

            if exists(cont_log_prob):
                if self.quantiled_critics:
                    cont_log_prob = rearrange(cont_log_prob, '... -> ... 1')
    
                cont_soft_state_value = next_cont_q_value - learned_entropy_weight * cont_log_prob
                next_soft_state_values.append(cont_soft_state_value)

            # then handle discrete contribution

            if len(next_discrete_q_values) > 0:

                for next_discrete_q_value, discrete_logit in zip(next_discrete_q_values, discrete_logits):
                    discrete_prob = discrete_logit.softmax(dim = -1)
                    discrete_log_prob = log(discrete_prob)

                    if self.quantiled_critics:
                        discrete_prob = rearrange(discrete_prob, '... -> ... 1')
                        discrete_log_prob = rearrange(discrete_log_prob, '... -> ... 1')

                    discrete_soft_state_value = (discrete_prob * (next_discrete_q_value - learned_entropy_weight * discrete_log_prob)).sum(dim = 1)
                    next_soft_state_values.append(discrete_soft_state_value)

        if self.quantiled_critics:
            next_soft_state_values, _ = pack(next_soft_state_values, 'b * q')
        else:
            next_soft_state_values, _ = pack(next_soft_state_values, 'b *')

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

        # update the actor

        actor_output = self.actor(states, sample = True, cont_reparametrize = True)

        cont_log_prob = actor_output.continuous_log_prob
        discrete_logits = actor_output.discrete_action_logits

        cont_q_values, *discrete_q_values = self.critics(
            states,
            cont_actions = actor_output.continuous,
            discrete_actions = actor_output.discrete,
            truncate_quantiles_across_critics = True
        )

        actor_action_losses = []

        if exists(cont_log_prob):
            cont_action_loss = (entropy_temp * cont_log_prob - cont_q_values).sum(dim = -1).mean()

            actor_action_losses.append(cont_action_loss)

        for discrete_logit, one_discrete_q_value in zip(discrete_logits, discrete_q_values):
            discrete_prob = discrete_logit.softmax(dim = -1)
            discrete_log_prob = log(discrete_prob)

            one_discrete_actor_loss = (discrete_prob * (entropy_temp * discrete_log_prob - one_discrete_q_value)).mean()

            actor_action_losses.append(one_discrete_actor_loss)

        sum(actor_action_losses).backward()
        self.actor_optimizer.step()
        self.actor_optimizer.zero_grad()

        # update the learned entropy temperature

        temperature_loss = self.learned_entropy_temperature(
            cont_log_prob = cont_log_prob,
            discrete_logits = discrete_logits
        )

        temperature_loss.backward()
        self.temperature_optimizer.step()
        self.temperature_optimizer.zero_grad()

        # update ema of all critics

        self.critics_target.update()
