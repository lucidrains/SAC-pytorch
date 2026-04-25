# /// script
# dependencies = [
#   "sac-pytorch",
#   "memmap-replay-buffer",
#   "gymnasium[box2d]",
#   "accelerate",
#   "fire",
#   "tqdm",
#   "moviepy"
# ]
# ///

from __future__ import annotations

import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import shutil
from collections import deque

import torch
import gymnasium as gym
import fire
import numpy as np
from tqdm import tqdm
from accelerate import Accelerator

from SAC_pytorch import SAC, Actor, Critic
from memmap_replay_buffer import ReplayBuffer

# functions

def exists(v):
    return v is not None

def divisible_by(num, den):
    return (num % den) == 0

# main

def main(
    continuous:             bool = True,
    episodes:               int = 1500,
    batch_size:             int = 256,
    replay_capacity:        int = 100_000,
    warmup_steps:           int = 1000,
    epochs:                 int = 2,
    learning_rate:          float = 3e-4,
    gamma:                  float = 0.99,
    seed:                   int | None = None,
    log_every:              int = 2,
    clear_buffer:           bool = True,
    record_video_every:     int = 100,
    cpu:                    bool = False,
    max_steps_per_episode:  int = 500,
    update_every_episodes:  int = 25,
    target_reward:          float = 25.,
    rolling_window:         int = 20,
    num_critics:            int = 2,
    expectile_l2_loss_tau:  float = 0.45,
    use_beta:               bool = False
):
    accelerator = Accelerator(cpu = cpu)
    device = accelerator.device

    if exists(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)

    # environment

    env = gym.make('LunarLander-v3', continuous = continuous, render_mode = 'rgb_array')

    if record_video_every > 0:
        video_folder = f"videos_{'cont' if continuous else 'disc'}"

        if clear_buffer and os.path.exists(video_folder):
            shutil.rmtree(video_folder)

        env = gym.wrappers.RecordVideo(
            env,
            video_folder = video_folder,
            episode_trigger = lambda ep: divisible_by(ep, record_video_every),
            disable_logger = True
        )

    state_dim = env.observation_space.shape[0]

    if continuous:
        num_cont_actions = env.action_space.shape[0]
        num_discrete_actions = ()
    else:
        num_cont_actions = 0
        num_discrete_actions = (int(env.action_space.n),)

    # networks

    actor_critic_kwargs = dict(
        rsmnorm_input = False,
        dim_state = state_dim,
        num_cont_actions = num_cont_actions,
        num_discrete_actions = num_discrete_actions,
        dim_hidden = 256,
        mlp_depth = 2
    )

    actor_kwargs = dict(
        **actor_critic_kwargs,
        use_beta = use_beta,
        target_range = (-1., 1.)
    )

    actor = Actor(**actor_kwargs)

    critics = [Critic(**actor_critic_kwargs, dim_out = 1) for _ in range(num_critics)]

    agent = SAC(
        actor = actor,
        critics = critics,
        quantiled_critics = False,
        multiple_critics_kwargs = dict(
            expectile_l2_loss_tau = expectile_l2_loss_tau
        ),
        reward_discount_rate = gamma,
        actor_learning_rate = learning_rate,
        critics_learning_rate = learning_rate,
        temperature_learning_rate = learning_rate,
        actor_regen_reg_rate = 0.,
        critics_regen_reg_rate = 0.,
        reward_scale = 2.
    )

    agent.to(device)

    if continuous:
        agent.learned_entropy_temperature.continuous_entropy_target = num_cont_actions / 2.0

    # replay buffer

    buffer_path = f'./replay_data_{"cont" if continuous else "disc"}'

    if clear_buffer and os.path.exists(buffer_path):
        shutil.rmtree(buffer_path)

    fields = dict(
        state = ('float', (state_dim,)),
        action = ('float', (num_cont_actions,)) if continuous else ('int', (1,)),
        reward = ('float', ()),
        done = ('bool', ()),
        next_state = ('float', (state_dim,))
    )

    replay_buffer = ReplayBuffer(
        buffer_path,
        max_episodes = replay_capacity,
        max_timesteps = max_steps_per_episode,
        fields = fields,
        flush_every_store_step = max_steps_per_episode
    )

    # training loop

    total_steps = 0
    recent_rewards = deque(maxlen = rolling_window)
    recent_steps = deque(maxlen = rolling_window)

    pbar = tqdm(range(episodes), desc = 'Training')

    for episode in pbar:
        reset_kwargs = dict(seed = seed + episode) if exists(seed) else dict()
        state, _ = env.reset(**reset_kwargs)
        episode_reward = 0.

        with replay_buffer.one_episode():
            for t in range(max_steps_per_episode):

                with torch.no_grad():
                    agent.eval()
                    state_t = torch.tensor(state, dtype = torch.float32, device = device).unsqueeze(0)
                    actor_output = agent.actor(state_t, sample = True)
                    agent.train()

                    if continuous:
                        action_raw = actor_output.continuous[0].cpu().numpy()
                        action_env = np.clip(action_raw, -1., 1.)
                        action_store = action_env
                    else:
                        action_raw = actor_output.discrete[0].cpu().numpy()
                        action_env = int(action_raw[0])
                        action_store = action_raw

                next_state, reward, terminated, truncated, _ = env.step(action_env)

                replay_buffer.store(
                    state = state,
                    action = action_store,
                    reward = reward,
                    done = terminated,
                    next_state = next_state
                )

                state = next_state
                episode_reward += reward
                total_steps += 1

                if terminated or truncated:
                    break

        episode_length = t + 1
        recent_rewards.append(episode_reward)
        recent_steps.append(episode_length)

        # batch update every N episodes

        should_update = total_steps > warmup_steps and divisible_by(episode + 1, update_every_episodes)

        if should_update:
            dl = replay_buffer.dataloader(
                batch_size = batch_size,
                timestep_level = True,
                shuffle = True,
                device = device,
                to_named_tuple = ('state', 'action', 'reward', 'done', 'next_state')
            )

            for _ in range(epochs):
                for states, actions, rewards, dones, next_states in tqdm(dl, desc = 'Updating', leave = False):

                    if continuous:
                        cont_actions = actions
                        discrete_actions = None
                    else:
                        cont_actions = None
                        discrete_actions = actions

                    agent(
                        states = states,
                        cont_actions = cont_actions,
                        discrete_actions = discrete_actions,
                        rewards = rewards,
                        done = dones,
                        next_states = next_states
                    )

        # logging

        if divisible_by(episode, log_every):

            avg_reward = np.mean(recent_rewards)
            avg_steps = np.mean(recent_steps)

            pbar.set_postfix(
                reward = f'{avg_reward:.1f}',
                steps = f'{avg_steps:.1f}'
            )

            if len(recent_rewards) == rolling_window and avg_reward > target_reward:
                print(f'\nConverged at episode {episode} — avg reward {avg_reward:.2f}')
                break

    print(f'Training finished. Final Avg Reward (last {rolling_window}): {np.mean(recent_rewards):.2f}')

if __name__ == '__main__':
    fire.Fire(main)
