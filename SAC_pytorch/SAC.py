from __future__ import annotations

import math
from collections import namedtuple

import torch
from torch.distributions import Normal
from torch import nn, einsum, Tensor, tensor
from torch.nn import Module, ModuleList, Sequential

from adam_atan2_pytorch import AdamAtan2 as Adam

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
        discrete_sample_deterministic = False,
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
            cont_log_prob,
            sampled_discrete_actions,
            discrete_action_logits
        )

class Critic(Module):
    @beartype
    def __init__(
        self,
        *,
        dim_state,
        num_cont_actions,
        num_discrete_actions: tuple[int, ...] = (),
        dim_hiddens: tuple[int, ...] = (),
        layernorm = False,
        dropout = 0.,
        num_quantiles: int | None = None,
        quantiles: tuple[float, ...] | None = None
    ):
        super().__init__()
        assert not exists(num_quantiles) or num_quantiles > 0

        use_quantiles = exists(num_quantiles)
        self.returning_quantiles = use_quantiles
        self.num_quantiles = num_quantiles

        num_actions_split = tensor((num_cont_actions, *num_discrete_actions))

        # determine the output dimension of the critic
        # which is the sum of all the actions (continous and discrete), multiplied by the number of quantiles

        critic_dim_out = num_actions_split
        if use_quantiles:
            critic_dim_out = critic_dim_out * num_quantiles

        self.to_values = MLP(
            dim_state + num_cont_actions,
            dim_out = critic_dim_out.sum().item(),
            dim_hiddens = dim_hiddens,
            layernorm = layernorm,
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
    ):
        critic_values = [critic(states, cont_actions) for critic in self.critics]

        critic_values = [torch.stack(one_value_for_critics) for one_value_for_critics in zip(*critic_values)]

        if not exists(target_value):

            if self.use_softmin:
                values = torch.stack(values, dim = -1)
                softmin = (-values).softmax(dim = -1)

                min_critic_value = (softmin * values).sum(dim = -1)
            else:
                min_critic_value = torch.minimum(*values)

            return min_critic_value, values

        cont_critic_values, *discrete_critic_values = critic_values
        discrete_critic_values = [get_at('... [l], ... -> ...', discrete_critic_value, discrete_action) for discrete_critic_value, discrete_action in zip(discrete_critic_values, discrete_actions)]

        values, _ = pack(values, 'c b *')

        losses = F.mse_loss(values, target_values, reduction = 'none')

        reduced_losses = reduce(losses, 'c b n -> b', 'sum').mean() # sum losses per action per critic, and average over batch

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
        cont_actions: Float['b nc'] | None = None,
        discrete_actions: Int['b nd'] | None = None,
        target_values: Float['b n q'] | None = None,
    ):
        critics_quantile_atoms = [critic(states, cont_actions) for critic in self.critics]

        critics_quantile_atoms = [torch.stack(one_value_for_critics) for one_value_for_critics in zip(*critics_quantile_atoms)]

        if not exists(target_value):
            quantile_atoms = rearrange(quantile_atoms, 'c b ... q -> b ... (q c)')

            # mean of the truncated distribution

            truncated_quantiles = quantile_atoms.topk(self.num_atom_keep, largest = False).values
            truncated_mean = truncated_quantiles.mean()

            return truncated_mean, quantile_atoms

        cont_quantile_atoms, *discrete_quantile_atoms = critics_quantile_atoms
        discrete_quantile_atoms = [get_at('... [l] q, ... -> ... q', discrete_quantile_atom, discrete_action) for discrete_quantile_atoms, discrete_action in zip(discrete_quantile_atoms, discrete_actions)]

        quantile_atoms, _ = pack(quantile_atoms, 'c b * q')

        # quantile regression if training

        target_value = torch.stack(target_value)
        quantiles = self.quantiles

        # quantile regression loss

        error = target_value - quantile_atoms
        losses = torch.maximum(error * quantiles, error * (quantiles - 1.))

        reduced_losses = reduce(losses, 'c b n q -> b', 'sum').mean()

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

        γ = self.discount_factor_gamma
        not_terminal = (~done).float()

        with torch.no_grad():
            self.critics_target.eval()

            next_cont_q_value, *next_discrete_q_values = self.critics_target(next_states, cont_actions = cont_actions)

            # outputs from actor

            actor_output = self.actor(states, sample = True, cont_reparametrize = True, return_discrete_entropies = True)

            cont_log_prob = actor_output.continuous_log_prob
            discrete_logits = actor_output.discrete_action_logits

            # handle extra quantile last dimension if needed

            if self.quantiled_critics:
                cont_log_prob = rearrange(cont_log_prob, '... -> ... 1')
                discrete_logits = [rearrange(t, '... -> ... 1') for t in discrete_logits]

            # learned temperature

            learned_entropy_weight = self.learned_entropy_temperature.alpha

            next_soft_state_values = []

            # first handle continuous soft state value

            if exists(cont_log_prob):
                cont_soft_state_value = next_cont_q_value - learned_entropy_weight * cont_log_prob

                next_soft_state_values.append(cont_soft_state_value)

            # then handle discrete contribution

            if len(next_discrete_q_values) > 0:

                for next_discrete_q_value, discrete_logit in zip(next_discrete_q_values, discrete_logits):
                    discrete_prob = discrete_logits.softmax(dim = -1)
                    discrete_logprob = log(discrete_prob)

                    discrete_soft_state_value = (discrete_prob * (next_discrete_q_value - learned_entropy_weight * discrete_log_prob)).sum(dim = -1)

                    next_soft_state_values.append(discrete_soft_state_value)

        next_soft_state_values: Float['b n ...'] = torch.stack(next_soft_state_values, dim = 1)

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

        # update ema of all critics

        self.critics_target.update()
