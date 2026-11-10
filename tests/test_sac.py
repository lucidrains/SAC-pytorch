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
        dim_hidden = 16,
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

    for _ in range(5):
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
        dim_hidden = 32,
        depth = 2,
        heads = 4,
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
        dim_hidden = 32,
        depth = 2,
        heads = 4,
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

# discrete-only actions (no continuous actions) - e.g. LunarLander discrete

@param('quantiled_critics', (False, True))
@param('use_minto', (False, True))
def test_sac_discrete_only(quantiled_critics, use_minto):
    from SAC_pytorch import SAC, Actor, Critic

    critic_kwargs = dict(
        dim_state = 5,
        num_cont_actions = 0,
        num_discrete_actions = (4,),
        dim_hidden = 16
    )

    actor = Actor(**critic_kwargs)

    critic_dim_out = 3 if quantiled_critics else 1

    critics = [Critic(**critic_kwargs, dim_out = critic_dim_out), Critic(**critic_kwargs, dim_out = critic_dim_out)]

    agent = SAC(
        actor = actor,
        critics = critics,
        quantiled_critics = quantiled_critics,
        use_minto = use_minto,
        fire_every = 5,
    )

    state = torch.randn(3, 5)

    for _ in range(5):
        actor_output = actor(state, sample = True)

        assert actor_output.continuous is None
        assert actor_output.continuous_log_prob is None
        assert actor_output.continuous_entropy is None
        assert actor_output.discrete.shape == (3, 1)

        agent(
            states = state,
            cont_actions = None,
            discrete_actions = actor_output.discrete,
            rewards = torch.randn(3),
            done = torch.zeros(3).bool(),
            next_states = torch.randn(3, 5)
        )

# assertion for mismatched chunk lengths

def test_mismatched_chunk_lengths():
    from SAC_pytorch import SAC, Actor

    critic_kwargs = dict(
        dim_state = 5,
        num_cont_actions = 2,
        dim_out = 3,
        dim_hidden = 32,
        depth = 2,
        heads = 4,
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
        depth = 2,
        heads = 4,
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

# softmin critics + breakdown returns

def test_multiple_critics_softmin():
    from SAC_pytorch.SAC import MultipleCritics, MultipleCriticsWithClassificationLoss, Critic

    critic_kwargs = dict(dim_state = 3, num_cont_actions = 2, dim_hidden = 8)

    critics = MultipleCritics(
        Critic(**critic_kwargs),
        Critic(**critic_kwargs),
        use_softmin = True
    )

    states = torch.randn(4, 3)
    cont_actions = torch.randn(4, 2)

    min_values = critics(states, cont_actions = cont_actions)
    assert len(min_values) == 1

    min_values, breakdown = critics(states, cont_actions = cont_actions, return_breakdown = True)
    assert len(breakdown) == 1

    # classification loss version

    critics = MultipleCriticsWithClassificationLoss(
        Critic(**{**critic_kwargs, 'dim_out': 5}),
        Critic(**{**critic_kwargs, 'dim_out': 5}),
        hl_gauss_loss = dict(min_value = -10., max_value = 10., num_bins = 5),
        use_softmin = True
    )

    min_values = critics(states, cont_actions = cont_actions)
    assert len(min_values) == 1

# mellowmax / soft mellowmax critic aggregation

def test_mellowmax_operators():
    from SAC_pytorch.SAC import mellowmax, soft_mellowmax_min

    values = torch.randn(4, 1024)

    # temperature -> 0 gives the mean, temperature -> inf gives the max / min

    assert torch.allclose(mellowmax(values, temperature = 1000.), values.amax(dim = 0), atol = 1e-2)
    assert torch.allclose(mellowmax(values, temperature = 0.01), values.mean(dim = 0), atol = 5e-2)
    assert torch.allclose(soft_mellowmax_min(values, temperature = 1000.), values.amin(dim = 0), atol = 1e-2)
    assert torch.allclose(soft_mellowmax_min(values, temperature = 0.01), values.mean(dim = 0), atol = 5e-2)

def test_soft_mellowmax_one_critic():
    from SAC_pytorch import SAC, Actor, Critic
    from SAC_pytorch.SAC import MultipleCritics

    critic_kwargs = dict(dim_state = 3, num_cont_actions = 2, dim_hidden = 8, num_reward_lanes = 2)

    critics = MultipleCritics(
        Critic(**critic_kwargs),
        use_soft_mellowmax = True,
        soft_mellowmax_temperature = 1.
    )

    states = torch.randn(4, 3)
    cont_actions = torch.randn(4, 2)

    aggregated, = critics(states, cont_actions = cont_actions)
    _, breakdown = critics(states, cont_actions = cont_actions, return_breakdown = True)

    lane_values = breakdown[0][0]
    assert torch.all(aggregated >= lane_values.amin(dim = -1) - 1e-6)
    assert torch.all(aggregated <= lane_values.mean(dim = -1) + 1e-6)

    # only allowed with a single critic

    with pytest.raises(AssertionError, match = 'single critic'):
        MultipleCritics(Critic(**critic_kwargs), Critic(**critic_kwargs), use_soft_mellowmax = True)

    # e2e sac with one critic and multiple reward lanes

    agent = SAC(
        actor = Actor(dim_state = 3, num_cont_actions = 2, dim_hidden = 8),
        critics = [dict(**critic_kwargs)],
        use_soft_mellowmax = True,
        soft_mellowmax_temperature = 1.
    )

    agent(
        states = states,
        cont_actions = cont_actions,
        discrete_actions = None,
        rewards = torch.randn(4, 2),
        done = torch.zeros(4).bool(),
        next_states = states
    )

# learned entropy temperature target should be total entropy across action dims

def test_learned_entropy_temperature_target():
    from SAC_pytorch.SAC import LearnedEntropyTemperature

    num_cont_actions = 2

    temperature = LearnedEntropyTemperature(num_cont_actions = num_cont_actions, num_discrete_actions = ())

    # equilibrium should occur when the total entropy over action dims equals the target

    entropy = torch.full((4, num_cont_actions), num_cont_actions / 2)
    loss = temperature(cont_entropy = entropy)
    grad = torch.autograd.grad(loss, temperature.log_alpha)[0]

    assert abs(grad.item()) < 1e-4

# sequence of next states is not supported - should raise clear error

def test_sequence_next_states_error():
    from SAC_pytorch import SAC, Actor

    agent = SAC(
        actor = Actor(dim_state = 5, num_cont_actions = 2),
        critics = [dict(dim_state = 5, num_cont_actions = 2, dim_hidden = 16)] * 2
    )

    with pytest.raises(AssertionError, match = 'next_states'):
        agent(
            states = torch.randn(4, 3, 5),
            cont_actions = torch.randn(4, 3, 2),
            discrete_actions = None,
            rewards = torch.randn(4, 3),
            done = torch.zeros(4).bool(),
            next_states = torch.randn(4, 3, 5),
        )

# test mean_conc_beta integration and scale multiplication

@param('target_range', (None, (-1., 1.), (-2.5, 2.5)))
def test_mean_conc_beta_integration(target_range):
    from SAC_pytorch import Actor
    from SAC_pytorch.SAC import BetaDistribution
    from mean_conc_beta import Beta

    actor = Actor(
        dim_state = 6,
        num_cont_actions = 3,
        use_beta = True,
        target_range = target_range
    )

    assert isinstance(actor.cont_dist, BetaDistribution)
    assert isinstance(actor.cont_dist.beta, Beta)
    assert actor.cont_dist.source_range == (-1., 1.)

    state = torch.randn(16, 6)
    out = actor(state, sample = True)

    scale = 2.5 if target_range == (-2.5, 2.5) else 1.0

    assert out.continuous.shape == (16, 3)
    assert (out.continuous >= -scale).all() and (out.continuous <= scale).all()
    assert not torch.isnan(out.continuous).any()
    assert not torch.isnan(out.continuous_log_prob).any()
    assert not torch.isnan(out.continuous_entropy).any()

    # deterministic mode action
    det_out = actor(state, sample = False)
    assert det_out.continuous.mu.shape == (16, 3)
    assert (det_out.continuous.mu >= -1.0).all() and (det_out.continuous.mu <= 1.0).all()

def test_mean_conc_beta_entropy_depends_on_mean():
    from SAC_pytorch import Actor

    actor = Actor(dim_state = 6, num_cont_actions = 3, use_beta = True)

    params = torch.randn(4, 6, requires_grad = True)
    _, _, entropy, _ = actor.cont_dist(params, reparametrize = True)

    entropy.sum().backward()

    # the policy mean must receive exploration pressure, otherwise it
    # saturates at the action bounds and the entropy collapses

    assert params.grad[..., 0].abs().sum() > 0.

def test_mean_conc_beta_entropy_target():
    from SAC_pytorch import Actor, SAC, Critic

    actor = Actor(dim_state = 6, num_cont_actions = 3, use_beta = True)
    critics = [Critic(dim_state = 6, num_cont_actions = 3, dim_hidden = 16, dim_out = 1) for _ in range(2)]

    agent = SAC(actor = actor, critics = critics)

    assert agent.learned_entropy_temperature.continuous_entropy_target == 3 * actor.cont_dist.beta.target_entropy.item()

# multi-lane reward critics (Hybrid Reward Architecture - van Seijen et al.)

@param('sum_reward_lanes_before_min', (False, True))
@param('quantiled_critics', (False, True))
@param('use_hl_gauss', (False, True))
def test_reward_lanes_sac(
    sum_reward_lanes_before_min,
    quantiled_critics,
    use_hl_gauss
):
    if quantiled_critics and use_hl_gauss:
        return

    from SAC_pytorch import SAC, Actor, Critic

    num_reward_lanes = 3
    dim_out = 4 if (quantiled_critics or use_hl_gauss) else 1

    critic_kwargs = dict(
        dim_state = 5,
        num_cont_actions = 2,
        num_discrete_actions = (4,),
        dim_hidden = 16,
        dim_out = dim_out,
        num_reward_lanes = num_reward_lanes
    )

    actor = Actor(
        dim_state = 5,
        num_cont_actions = 2,
        num_discrete_actions = (4,),
        dim_hidden = 16
    )

    hl_gauss_loss = dict(min_value = -5., max_value = 5., num_bins = dim_out) if use_hl_gauss else None

    agent = SAC(
        actor = actor,
        critics = [Critic(**critic_kwargs), Critic(**critic_kwargs)],
        quantiled_critics = quantiled_critics,
        hl_gauss_loss = hl_gauss_loss,
        sum_reward_lanes_before_min = sum_reward_lanes_before_min
    )

    # default reward lane weights must be all ones

    assert torch.allclose(agent.reward_lane_weights, torch.ones(num_reward_lanes))

    state = torch.randn(4, 5)
    actor_out = actor(state, sample = True)

    # train step with default all-ones weights

    agent(
        states = state,
        cont_actions = actor_out.continuous,
        discrete_actions = actor_out.discrete,
        rewards = torch.randn(4, num_reward_lanes),
        done = torch.zeros(4).bool(),
        next_states = torch.randn(4, 5)
    )

    # train step and policy extraction with retailored lane weights

    agent.set_reward_lane_weights((2.0, 0.5, 1.0))
    assert torch.allclose(agent.reward_lane_weights, torch.tensor([2.0, 0.5, 1.0]))

    agent(
        states = state,
        cont_actions = actor_out.continuous,
        discrete_actions = actor_out.discrete,
        rewards = torch.randn(4, num_reward_lanes),
        done = torch.zeros(4).bool(),
        next_states = torch.randn(4, 5),
        reward_lane_weights = (0.2, 1.5, 0.8)
    )

    # standalone policy extraction with custom lane weights

    loss = agent.update_actor(state, reward_lane_weights = (1.0, 0.0, 2.0))
    assert not torch.isnan(loss)

# end-to-end multi-lane critic aggregation hypotheses and policy retailoring

def test_reward_lanes_e2e():
    from SAC_pytorch import SAC, Actor, Critic

    torch.manual_seed(42)

    dim_state = 4
    num_cont_actions = 2
    num_reward_lanes = 2

    critic_kwargs = dict(
        dim_state = dim_state,
        num_cont_actions = num_cont_actions,
        dim_hidden = 32,
        num_reward_lanes = num_reward_lanes
    )

    agent = SAC(
        actor = Actor(dim_state = dim_state, num_cont_actions = num_cont_actions, dim_hidden = 32),
        critics = [Critic(**critic_kwargs), Critic(**critic_kwargs)],
        actor_learning_rate = 1e-2,
        critics_learning_rate = 1e-2
    )

    states = torch.randn(32, dim_state)
    actions = torch.zeros(32, num_cont_actions)

    # default lane weights are all ones (1., 1.)

    weights = agent.reward_lane_weights
    assert torch.allclose(weights, torch.ones(num_reward_lanes))

    # hypothesis 1 (min_then_sum: per-lane twin min then weighted sum) <= hypothesis 2 (sum_then_min: weighted sum then twin min)
    # by subadditivity of min for non-negative lane weights

    q_min_then_sum, = agent.critics(states, cont_actions = actions, reward_lane_weights = weights, sum_reward_lanes_before_min = False)
    q_sum_then_min, = agent.critics(states, cont_actions = actions, reward_lane_weights = weights, sum_reward_lanes_before_min = True)

    assert (q_min_then_sum <= q_sum_then_min + 1e-6).all()

    # train multi-lane critics end-to-end on two competing reward lanes:
    # lane 0 rewards actions near +1, while lane 1 rewards actions near -1

    for _ in range(80):
        s = torch.randn(64, dim_state)
        a = torch.empty(64, num_cont_actions).uniform_(-1., 1.)

        r_lane0 = a.mean(dim = -1)
        r_lane1 = -a.mean(dim = -1)
        rewards = torch.stack((r_lane0, r_lane1), dim = -1)

        agent(
            states = s,
            cont_actions = a,
            discrete_actions = None,
            rewards = rewards,
            done = torch.ones(64).bool(),
            next_states = s
        )

    # retailor policy extraction toward lane 0 with weights (1., 0.)

    for _ in range(40):
        agent.update_actor(states, reward_lane_weights = (1.0, 0.0))

    act_lane0 = agent.actor(states, sample = False).continuous.mu.mean().item()

    # retailor policy extraction toward lane 1 with weights (0., 1.)

    for _ in range(40):
        agent.update_actor(states, reward_lane_weights = (0.0, 1.0))

    act_lane1 = agent.actor(states, sample = False).continuous.mu.mean().item()

    assert act_lane0 > 0.2 and act_lane1 < -0.2


