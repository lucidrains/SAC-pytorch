from __future__ import annotations

import math
from collections import namedtuple

import torch
from torch import nn, einsum, Tensor, tensor
from torch.distributions import Normal
from torch.nn import Module, ModuleList, Sequential

from adam_atan2_pytorch import AdamAtan2

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
# c - critics
# q - quantiles

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

def identity(t):
    return t

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

def init_(m):
    if not isinstance(m, nn.Linear):
        return

    gain = torch.nn.init.calculate_gain('relu')
    torch.nn.init.orthogonal_(m.weight, gain)

    if not exists(m.bias):
        return

    torch.nn.init.zeros_(m.bias)

# helper classes

def Sequential(*modules):
    return nn.Sequential(*filter(exists, modules))

class Residual(Module):
    @beartype
    def __init__(
        self,
        fn: Module,
        dim_in,
        dim_out
    ):
        super().__init__()
        self.fn = fn
        self.residual_proj = nn.Linear(dim_in, dim_out) if dim_in != dim_out else identity

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + self.residual_proj(x)

class NegativeConcat(Module):
    """
    https://arxiv.org/abs/1706.00388v1
    """
    @beartype
    def __init__(
        self,
        fn: Module
    ):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        fn = self.fn
        return torch.cat((fn(x), -fn(-x)), dim = -1)

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
        negative_concat = True,
        expansion_factor = 2,
        add_residual = True
    ):
        super().__init__()
        """
        simple mlp for Q and value networks

        following Figure 1 in https://arxiv.org/pdf/2110.02034.pdf for placement of dropouts and layernorm
        however, be aware that Levine in his lecture has ablations that show layernorm alone (without dropout) is sufficient for regularization
        """

        out_expansion_factor = (2 if negative_concat else 1) * expansion_factor
        maybe_negative_concat = NegativeConcat if negative_concat else identity

        dim_hiddens = cast_tuple(dim_hiddens)

        layers = []

        curr_dim = dim

        for dim_hidden in dim_hiddens:


            layer = Sequential(
                nn.Linear(curr_dim, dim_hidden * expansion_factor),
                nn.Dropout(dropout),
                nn.LayerNorm(dim_hidden * expansion_factor) if layernorm else None,
                maybe_negative_concat(activation()),
                nn.Linear(dim_hidden * out_expansion_factor, dim_hidden)
            )

            if add_residual:
                layer = Residual(layer, curr_dim, dim_hidden)

            layers.append(layer)

            curr_dim = dim_hidden

        # final layer out

        layers.append(nn.Linear(curr_dim, dim_out))

        self.layers = ModuleList(layers)

        # init

        self.apply(init_)

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
        state: Float['b ...'],
        sample = False,
        cont_reparametrize = False,
        discrete_sample_temperature = 1.,
        discrete_sample_deterministic = False
    ) -> (
        SoftActorOutput |
        SampledSoftActorOutput
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

        self._n = num_actions

        # excluding 0 and 1 - say 3 quantiles will be 0.25, 0.5, 0.75

        if exists(quantiles):
            quantiles = tensor(quantiles)
        elif exists(num_quantiles):
            quantiles = torch.linspace(0., 1., num_quantiles + 2)[1:-1]

        self.register_buffer('quantiles', quantiles)

    def forward(
        self,
        state: Float['b ...'],
        cont_actions: Float['b {self._n}'] | None = None
    ) -> (
        Float['b'] |
        Float['b q']
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
        cont_actions: Float['b n'] | None = None,
        target_value: Float['b'] | None = None,
    ):
        values = [critic(states, cont_actions) for critic in self.critics]

        if not exists(target_value):

            if self.use_softmin:
                values = torch.stack(values, dim = -1)
                softmin = (-values).softmax(dim = -1)

                min_critic_value = (softmin * values).sum(dim = -1)
            else:
                min_critic_value = torch.minimum(*values)

            return min_critic_value, values

        losses = [F.mse_loss(values, target) for value in values]
        return sum(losses), losses

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
        states: Float['b ...'],
        cont_actions: Float['b n'] | None = None,
        target_value: Float['b q'] | None = None,
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
        discrete_log_probs: tuple[Float['b _'], ...] | None = None
    ):
        assert exists(cont_log_prob) or exists(discrete_log_prob)

        alpha = self.alpha

        losses = []

        if exists(discrete_log_prob):

            for discrete_log_prob, discrete_entropy_target in zip(discrete_log_probs, self.discrete_entropy_targets):
                discrete_entropy_temp_loss = -alpha * (discrete_log_prob + discrete_entropy_target).detach()

                losses.append(discrete_entropy_temp_loss)

        if exists(cont_log_prob):
            cont_entropy_temp_loss = -alpha * (cont_log_prob + self.continuous_entropy_target).detach()

            losses.append(cont_entropy_temp_loss)

        return sum(losses).mean()

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
        critic_target_ema_decay = 0.99,
        critics_learning_rate = 3e-4,
        critics_regen_reg_rate = 1e-4,
        ema_kwargs: dict = dict()
    ):
        super().__init__()

        # set actor

        if isinstance(actor, dict):
            actor = Actor(**actor)

        self.actor = actor

        # based on the actor hyperparameters, init the leraned temperature container

        self.learned_entropy_temperature = LearnedEntropyTemperature(
            num_cont_actions = actor.num_cont_actions,
            num_discrete_actions = actor.num_discrete_actions
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

        # critic optimizers

        self.critics_optimizer = AdamAtan2(
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
        cont_actions: Float['b n'],
        rewards: Float['b'],
        done: Bool['b'],
        next_states: Float['b ...']
    ):

        rewards = rewards * self.reward_scale

        # bellman equation
        # todo: setup n-step

        γ = self.discount_factor_gamma
        not_terminal = (~done).float()

        with torch.no_grad():
            self.critics_target.eval()
            next_q_value = self.critics_target(next_states, cont_actions = cont_actions)
            sample_actor_output = self.actor(states, sample = True, cont_reparametrize = True)

            next_soft_state_value = next_q_value - sample_actor_output.continuous_log_prob

        q_value = rewards + not_terminal * γ * next_soft_state_value

        pred_q_value = self.critics(
            states,
            cont_actions = cont_actions,
            target_value = q_value
        )

        # update the critics

        critic_loss.backward()
        self.critics_optimizer.step()
        self.critics_optimizer.zero_grad()

        # update ema of all critics

        self.critics_target.update()

        return 0.
