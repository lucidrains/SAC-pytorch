import pytest
param = pytest.mark.parametrize

import torch

@param('use_beta', (False, True))
@param('simplicial_embed', (False, True))
def test_sac(
    use_beta,
    simplicial_embed
):

    from SAC_pytorch import (
        SAC,
        Actor,
        Critic,
        MultipleCritics
    )

    critic1 = Critic(
        dim_state = 5,
        num_cont_actions = 2,
        num_discrete_actions = (5, 5),
        dim_out = 3
    )

    critic2 = Critic(
        dim_state = 5,
        num_cont_actions = 2,
        num_discrete_actions = (5, 5),
        dim_out = 3
    )

    actor = Actor(
        dim_state = 5,
        num_cont_actions = 2,
        num_discrete_actions = (5, 5),
        use_beta = use_beta,
        simplicial_embed = simplicial_embed,
        dim_hidden = 24,
        target_range = (-2., 2.)
    )

    agent = SAC(
        actor = actor,
        critics = [critic1, critic2],
        quantiled_critics = True
    )

    state = torch.randn(3, 5)

    cont_actions, cont_logprob, cont_entropy, discrete, discrete_logprob = actor(state, sample = True)

    agent(
        states = state,
        cont_actions = cont_actions,
        discrete_actions = discrete,
        rewards = torch.randn(1),
        done = torch.zeros(1).bool(),
        next_states = state + 1
    )
