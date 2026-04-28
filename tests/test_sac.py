import pytest
param = pytest.mark.parametrize

import torch
from torch import nn

@param('use_beta', (False, True))
@param('simplicial_embed', (False, True))
@param('actor_state_recon', (False, True))
@param('critic_state_recon', (False, True))
@param('state_recon_loss_fn', (nn.MSELoss(), nn.SmoothL1Loss()))
@param('state_recon_branch_layer', (-1, 0))
def test_sac(
    use_beta,
    simplicial_embed,
    actor_state_recon,
    critic_state_recon,
    state_recon_loss_fn,
    state_recon_branch_layer
):

    from SAC_pytorch import (
        SAC,
        Actor,
        Critic,
        MultipleCritics
    )

    actor_critic_kwargs = dict(
        dim_state = 5,
        num_cont_actions = 2,
        num_discrete_actions = (5, 5),
        dim_hidden = 24,
        state_recon_branch_layer = state_recon_branch_layer
    )

    critic1 = Critic(
        **actor_critic_kwargs,
        dim_out = 3,
        state_recon = critic_state_recon,
    )

    critic2 = Critic(
        **actor_critic_kwargs,
        dim_out = 3,
        state_recon = critic_state_recon,
    )

    actor = Actor(
        **actor_critic_kwargs,
        use_beta = use_beta,
        simplicial_embed = simplicial_embed,
        target_range = (-2., 2.),
        state_recon = actor_state_recon,
    )

    agent = SAC(
        actor = actor,
        critics = [critic1, critic2],
        quantiled_critics = True,
        fire_every = 5,
        actor_state_recon_loss_weight = 0.5,
        critic_state_recon_loss_weight = 2.0,
        state_recon_loss_fn = state_recon_loss_fn,
    )

    for _ in range(10):
        state = torch.randn(3, 5)

        actor_output = actor(state, sample = True)

        agent(
            states = state,
            cont_actions = actor_output.continuous,
            discrete_actions = actor_output.discrete,
            rewards = torch.randn(3),
            done = torch.zeros(3).bool(),
            next_states = state + 1
        )
