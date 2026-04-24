import pytest
import torch

def test_sac():
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
        num_discrete_actions = (5, 5)
    )

    agent = SAC(
        actor = actor,
        critics = [critic1, critic2],
        quantiled_critics = True
    )

    state = torch.randn(3, 5)
    cont_actions, cont_logprob, discrete, discrete_logprob = actor(state, sample = True)

    agent(
        states = state,
        cont_actions = cont_actions,
        discrete_actions = discrete,
        rewards = torch.randn(1),
        done = torch.zeros(1).bool(),
        next_states = state + 1
    )
