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

# transformer critic unit tests

@param('seq_len', (None, 1, 7))
def test_transformer_critic(seq_len):
    from SAC_pytorch import TransformerCritic

    critic = TransformerCritic(
        dim_state = 5,
        num_cont_actions = 2,
        dim_out = 3,
        max_seq_len = 10
    )

    state = torch.randn(4, 5)

    if seq_len is None:
        actions = torch.randn(4, 2)
        out, = critic(state, actions)
        assert out.shape == (4, 2, 3)
    else:
        actions = torch.randn(4, seq_len, 2)
        out, = critic(state, actions)
        assert out.shape == (4, seq_len, 2, 3)

# end-to-end sac with transformer critic

@param('seq_len', (None, 1, 5))
def test_sac_transformer_critic_e2e(seq_len):
    from SAC_pytorch import SAC, Actor

    critic_kwargs = dict(
        dim_state = 5,
        num_cont_actions = 2,
        dim_out = 3,
        max_seq_len = 10
    )

    agent = SAC(
        actor = Actor(
            dim_state = 5,
            num_cont_actions = 2
        ),
        critics = [critic_kwargs, critic_kwargs],
        transformer_critic = True,
        quantiled_critics = True
    )

    state = torch.randn(4, 5)
    next_state = torch.randn(4, 5)

    if seq_len is None:
        cont_actions = torch.randn(4, 2)
        rewards = torch.randn(4)
    else:
        cont_actions = torch.randn(4, seq_len, 2)
        rewards = torch.randn(4, seq_len)

    done = torch.zeros(4).bool()

    agent(
        states = state,
        cont_actions = cont_actions,
        discrete_actions = None,
        rewards = rewards,
        done = done,
        next_states = next_state
    )

# assertion for mismatched chunk lengths

def test_mismatched_chunk_lengths():
    from SAC_pytorch import SAC, Actor

    critic_kwargs = dict(
        dim_state = 5,
        num_cont_actions = 2,
        dim_out = 3,
        max_seq_len = 10
    )

    agent = SAC(
        actor = Actor(
            dim_state = 5,
            num_cont_actions = 2
        ),
        critics = [critic_kwargs, critic_kwargs],
        transformer_critic = True,
        quantiled_critics = True
    )

    with pytest.raises(AssertionError, match = 'rewards chunk length'):
        agent(
            states = torch.randn(4, 5),
            cont_actions = torch.randn(4, 7, 2),
            discrete_actions = None,
            rewards = torch.randn(4, 6),
            done = torch.zeros(4).bool(),
            next_states = torch.randn(4, 5)
        )

# e2e with memmap replay buffer and n-step dataloader

def test_transformer_critic_with_replay_buffer(tmp_path):
    import numpy as np
    from memmap_replay_buffer import ReplayBuffer
    from SAC_pytorch import SAC, Actor

    dim_state = 5
    num_cont_actions = 2
    n_steps = 8

    # create replay buffer and simulate trajectories

    fields = dict(
        state = ('float', (dim_state,)),
        action = ('float', (num_cont_actions,)),
        reward = ('float', ()),
        done = ('bool', ()),
    )

    rb = ReplayBuffer(
        str(tmp_path / 'replay'),
        max_episodes = 10,
        max_timesteps = 50,
        fields = fields
    )

    for _ in range(5):
        ep_len = np.random.randint(15, 30)
        with rb.one_episode():
            for t in range(ep_len):
                rb.store(
                    state = np.random.randn(dim_state).astype(np.float32),
                    action = np.random.randn(num_cont_actions).astype(np.float32),
                    reward = float(np.random.randn()),
                    done = (t == ep_len - 1),
                )

    # create agent with transformer critic

    critic_kwargs = dict(
        dim_state = dim_state,
        num_cont_actions = num_cont_actions,
        dim_hidden = 32,
        dim_out = 1,
        max_seq_len = n_steps,
    )

    agent = SAC(
        actor = Actor(dim_state = dim_state, num_cont_actions = num_cont_actions),
        critics = [critic_kwargs, critic_kwargs],
        transformer_critic = True,
    )

    # load one batch from the n-step dataloader

    dl = rb.dataloader(
        batch_size = 4,
        n_steps = n_steps,
        current_fields = ('state',),
        next_fields = ('state',),
        sequence_fields = ('action', 'reward', 'done'),
        to_named_tuple = ('state', 'next_state', 'seq_action', 'seq_reward', 'seq_done', 'n_step_lens'),
    )

    batch = next(iter(dl))

    # forward + backward through the agent

    agent(
        states = batch.state,
        cont_actions = batch.seq_action,
        discrete_actions = None,
        rewards = batch.seq_reward,
        done = batch.seq_done,
        next_states = batch.next_state,
    )
