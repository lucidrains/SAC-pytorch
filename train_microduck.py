# /// script
# dependencies = [
#   "SAC-pytorch",
#   "mujoco>=3.0.0",
#   "torch",
#   "numpy",
#   "imageio[ffmpeg]",
#   "fire",
#   "tqdm",
#   "memmap-replay-buffer"
# ]
# ///

from __future__ import annotations

import os
import sys
import math
import copy
import shutil
import subprocess
from collections import deque
from itertools import islice
from pathlib import Path

if sys.platform != 'darwin':
    os.environ.setdefault('MUJOCO_GL', 'egl')

import fire
import imageio
import mujoco
import numpy as np
import torch
from torch import tensor, Tensor
from tqdm import tqdm

from memmap_replay_buffer import ReplayBuffer
from SAC_pytorch import SAC, Actor, Critic

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

# dependency & mjcf

MICRODUCK_XML_REL = Path('src/mjlab_microduck/robot/microduck/scene_walk.xml')

def ensure_deps() -> Path:
    candidates = (
        Path.home() / 'dl/microduck_rl' / MICRODUCK_XML_REL,
        Path(__file__).resolve().parent.parent / 'microduck_rl' / MICRODUCK_XML_REL,
        Path('/tmp/microduck_rl') / MICRODUCK_XML_REL
    )

    for path in candidates:
        if path.exists():
            return path

    target_repo = Path('/tmp/microduck_rl')
    subprocess.run(
        ['git', 'clone', '--depth', '1', 'https://github.com/pirrone/microduck_rl.git', str(target_repo)],
        check = True
    )
    return target_repo / MICRODUCK_XML_REL

# environment

class MicroduckEnv:
    def __init__(
        self,
        mjcf_path: str | Path | None = None,
        *,
        kp_mult: float = 3.0,
        kv: float = 0.1,
        horizon: int = 400,
        command: tuple[float, float] = (0.15, 0.0),
        command_scale: float = 2.0,
        action_scale: float = 1.0,
        control_hz: int = 50,
        substeps: int = 5,
        reset_z_range: tuple[float, float] = (0.0, 0.002),
        reset_joint_noise: float = 0.005,
        reset_vel_noise: float = 0.01
    ):
        self.mjcf_path = Path(default(mjcf_path, ensure_deps()))
        self.horizon = horizon
        self.command = np.asarray(command, dtype = np.float32)
        self.command_scale = command_scale
        self.action_scale = action_scale
        self.substeps = substeps
        self.dt = 1.0 / control_hz

        self.reset_z_range = reset_z_range
        self.reset_joint_noise = reset_joint_noise
        self.reset_vel_noise = reset_vel_noise

        self.model = mujoco.MjModel.from_xml_path(str(self.mjcf_path))
        self.model.opt.timestep = 1.0 / (control_hz * substeps)

        kp = self.model.actuator_gainprm[:, 0].copy() * kp_mult
        self.model.actuator_gainprm[:, 0] = kp
        self.model.actuator_biasprm[:, 1] = -kp
        self.model.actuator_biasprm[:, 2] = -kv

        self.data = mujoco.MjData(self.model)

        stand = self.model.key('STAND')
        self.home_qpos = stand.qpos.copy()
        self.default_pose = self.home_qpos[7:].copy()
        self.default_ctrl = stand.ctrl.copy()

        self.obs_dim = 50
        self.action_dim = 14

        self.last_action = np.zeros(self.action_dim, dtype = np.float32)
        self.start_xy = self.home_qpos[:2].copy()
        self.steps = 0

    def _get_obs(self) -> np.ndarray:
        qpos, qvel = self.data.qpos, self.data.qvel
        rot = self.data.xmat[1].reshape(3, 3)

        gravity = -rot[2, :]
        ang_vel_local = rot.T @ qvel[3:6]
        qpos_err = qpos[7:21] - self.default_pose
        qvel_joints = qvel[6:20]
        yaw_err = float(rot[1, 0]) + 1.5 * float(qpos[1] - self.start_xy[1])

        obs = np.concatenate([
            gravity,
            ang_vel_local,
            qpos_err,
            qvel_joints,
            self.last_action,
            [yaw_err, self.command[0] * self.command_scale]
        ]).astype(np.float32)

        return np.clip(np.nan_to_num(obs, nan = 0.0, posinf = 0.0, neginf = 0.0), -100.0, 100.0)

    def reset(self, *, seed: int | None = None, noisy: bool = True) -> np.ndarray:
        rng = np.random.default_rng(seed)
        self.data.qpos[:] = self.home_qpos
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = self.default_ctrl

        if noisy:
            self.data.qpos[2] += rng.uniform(*self.reset_z_range)
            self.data.qpos[7:21] += rng.uniform(-self.reset_joint_noise, self.reset_joint_noise, size = 14)
            self.data.qvel[:] = rng.uniform(-self.reset_vel_noise, self.reset_vel_noise, size = self.model.nv)

        mujoco.mj_forward(self.model, self.data)

        self.last_action.fill(0.0)
        self.start_xy = self.data.qpos[:2].copy()
        self.steps = 0
        return self._get_obs()

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype = np.float32), -1.0, 1.0)
        self.data.ctrl[:] = self.default_ctrl + self.action_scale * action

        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)

        self.steps += 1

        qpos, qvel = self.data.qpos, self.data.qvel
        rot = self.data.xmat[1].reshape(3, 3)

        upright = float(rot[2, 2])
        height = float(qpos[2])
        vel_local = rot.T @ qvel[:3]

        dx = float(qpos[0] - self.start_xy[0])
        dy = float(qpos[1] - self.start_xy[1])

        self.last_action = action.copy()
        obs = self._get_obs()

        is_nan = not (np.isfinite(qpos).all() and np.isfinite(qvel).all())
        fallen = is_nan or (upright < 0.45) or (height < 0.065) or (height > 0.25)

        # decompose unshaped reward into two lanes (Hybrid Reward Architecture - van Seijen et al.):
        # lane 0: upright forward propulsion
        # lane 1: lateral drift and fall penalty

        r_propulsion = max(0.0, upright) * float(vel_local[0])
        r_stability = -0.5 * (dy ** 2) - (1.0 if fallen else 0.0)
        reward_lanes = np.array([r_propulsion, r_stability], dtype = np.float32)
        reward = float(r_propulsion + r_stability)

        terminated = bool(fallen)
        truncated = bool(self.steps >= self.horizon)

        info = dict(
            x = dx,
            y = dy,
            upright = upright,
            height = height,
            vx = float(vel_local[0]),
            steps = self.steps,
            reward_lanes = reward_lanes
        )
        return obs, reward, terminated, truncated, info

# proprioceptive actor initialization

GAIT_BASIS = np.array([
    [ 0.008,  0.022, -0.017,  0.245,  0.076, -0.029,  0.032, -0.033, -0.042, -0.062, -0.011,  0.017, -0.300, -0.027],
    [-0.055,  0.093,  0.622, -0.276, -0.820, -0.028, -0.097,  0.094,  0.082,  0.090,  0.047,  0.578, -0.075, -0.668],
    [ 0.034, -0.108, -0.598,  0.057,  0.642,  0.108,  0.018, -0.029,  0.004,  0.012, -0.033, -0.602,  0.294,  0.701],
    [-0.006, -0.011,  0.012,  0.633,  0.456,  0.063,  0.036, -0.029,  0.001,  0.033, -0.005, -0.012,  0.495,  0.128],
    [-0.053,  0.018,  0.025, -0.464, -0.190, -0.032, -0.082,  0.084,  0.017,  0.040, -0.006, -0.025, -0.620, -0.486]
], dtype = np.float32)

@torch.no_grad()
def init_actor_(
    actor: Actor,
    pitch_mult: float = 1.0,
    knee_boost: float = 1.0,
    std_logit: float = -4.2
):
    knots = np.array([0.0, 0.25, 0.55, 0.90, 1.35, 1.90], dtype = np.float32)
    cknots = np.array([0.0, 1.2], dtype = np.float32)
    z_pts = np.linspace(-3.5, 3.5, 1001)
    lrelu = lambda x: np.where(x >= 0.0, x, 0.01 * x)

    Phi = np.stack([np.ones_like(z_pts), *(lrelu(z_pts - k) for k in knots), *(lrelu(-z_pts - k) for k in knots)], axis = 1)
    w_pos = tensor(np.linalg.lstsq(Phi, np.maximum(0.0, np.tanh(z_pts)), rcond = 1e-6)[0], dtype = torch.float32)
    w_neg = tensor(np.linalg.lstsq(Phi, np.maximum(0.0, np.tanh(-z_pts)), rcond = 1e-6)[0], dtype = torch.float32)

    Phi_c = np.stack([np.ones_like(z_pts), *(lrelu(z_pts - k) for k in cknots), *(lrelu(-z_pts - k) for k in cknots)], axis = 1)
    wc_pos = tensor(np.linalg.lstsq(Phi_c, np.maximum(0.0, np.clip(z_pts, -1.2, 1.2)), rcond = 1e-6)[0], dtype = torch.float32)
    wc_neg = tensor(np.linalg.lstsq(Phi_c, np.maximum(0.0, np.clip(-z_pts, -1.2, 1.2)), rcond = 1e-6)[0], dtype = torch.float32)

    simba = actor.to_actions
    dim_h = simba.proj_in.out_features

    for p in simba.parameters():
        p.zero_()

    simba.final_norm.weight.fill_(1.0)
    simba.final_norm.bias.zero_()

    alpha, s0, ky, amp_scale = 0.02, 0.11, 0.02, 0.52
    w_sp = 1.80 / 0.62
    w_c_in_c = 1.0 / 0.30
    w_c_in_s = -0.65 * w_c_in_c

    idx_rest = torch.arange(6, dim_h)
    simba.proj_in.bias[6:] = torch.where(idx_rest % 2 == 0, 1.0, -1.0) * math.sqrt(128.0 / 122.0) * pitch_mult
    simba.proj_in.bias[:2] = tensor([alpha * s0, -alpha * s0]) * pitch_mult

    simba.proj_in.weight[:2, (36, 45)] = tensor([[alpha * w_sp], [-alpha * w_sp]])
    simba.proj_in.weight[:2, (8, 17)] = tensor([[alpha * w_c_in_s], [-alpha * w_c_in_s]])
    simba.proj_in.weight[2:4, (8, 17)] = tensor([[alpha * w_c_in_c], [-alpha * w_c_in_c]])
    simba.proj_in.weight[4:6, 48] = tensor([alpha, -alpha]) * pitch_mult

    l1, l2 = simba.proj_out[0], simba.proj_out[2]
    l1.weight[:6, :2] = tensor([0.5, -0.5]) / alpha
    l1.weight[6:12, :2] = tensor([-0.5, 0.5]) / alpha
    l1.bias[:12] = -tensor(np.tile(knots, 2))

    l1.weight[12:14, 2:4] = tensor([0.5, -0.5]) / alpha
    l1.weight[14:16, 2:4] = tensor([-0.5, 0.5]) / alpha
    l1.bias[12:16] = -tensor(np.tile(cknots, 2))
    l1.weight[16:18, 4:6] = tensor([[0.5, -0.5], [-0.5, 0.5]]) / alpha

    G = GAIT_BASIS.copy()
    G[1:3, [2, 4, 11, 13]] *= pitch_mult
    G[3:5, [3, 12]] *= knee_boost
    coef = tensor(amp_scale * G, dtype = torch.float32)

    l2.bias[1::2] = std_logit
    l2.bias[0::2] = coef[0] + w_pos[0] * coef[1] + w_neg[0] * coef[2] + wc_pos[0] * coef[3] + wc_neg[0] * coef[4]
    l2.weight[0::2, :12] = coef[1, :, None] * w_pos[1:] + coef[2, :, None] * w_neg[1:]
    l2.weight[0::2, 12:16] = coef[3, :, None] * wc_pos[1:] + coef[4, :, None] * wc_neg[1:]
    l2.weight[[0, 18], 16:18] = tensor([ky, -ky])

# multi-lane critic initialization and policy extraction retailoring

def walking_subspace_states(num_points: int = 64, state_dim: int = 50, device: torch.device | str = 'cpu') -> Tensor:
    phase = np.linspace(-0.42, 0.42, num_points, dtype = np.float32)
    states = torch.zeros(num_points, state_dim, dtype = torch.float32, device = device)
    states[:, 36] = tensor(phase, device = device)
    states[:, 45] = tensor(-phase, device = device)
    states[:, 8] = tensor(-0.28 * phase, device = device)
    states[:, 17] = tensor(0.28 * phase, device = device)
    return states

@torch.no_grad()
def init_critic_(
    critic: Critic,
    lane_targets: tuple[tuple[float, float], ...] = ((1.05, 1.00), (0.76, 1.00)),
    lane_curvatures: tuple[float, ...] = (30.0, 13.0),
    lane_biases: tuple[float, ...] = (-0.20, -0.20)
):
    # fit concave quadratic q-surface -0.5 * (a - a_target(s))^2 per reward lane across SimBa LeakyReLU knots

    knots = np.array([0.0, 0.04, 0.10, 0.20, 0.38, 0.70], dtype = np.float32)
    z_pts = np.linspace(-1.0, 1.0, 1001)
    lrelu = lambda x: np.where(x >= 0.0, x, 0.01 * x)
    Phi = np.stack([np.ones_like(z_pts), *(lrelu(z_pts - k) for k in knots), *(lrelu(-z_pts - k) for k in knots)], axis = 1)
    w_quad = tensor(np.linalg.lstsq(Phi, -0.5 * (z_pts ** 2), rcond = 1e-6)[0], dtype = torch.float32)

    simba = critic.to_values
    dim_h = simba.proj_in.out_features
    state_dim = simba.proj_in.in_features - critic.num_cont_actions
    num_lanes = critic.num_reward_lanes
    active_lanes = min(num_lanes, len(lane_targets))

    for p in simba.parameters():
        p.zero_()

    simba.final_norm.weight.fill_(1.0)
    simba.final_norm.bias.zero_()

    # antisymmetric hidden pairs keep SimBa final LayerNorm mean at zero and unit variance

    alpha = 0.02
    z_scale = 25.0 / alpha
    idx_rest = torch.arange(64, dim_h)
    simba.proj_in.bias[64:] = torch.where(idx_rest % 2 == 0, 1.0, -1.0) * math.sqrt(float(dim_h) / float(dim_h - 64))

    s_synth = walking_subspace_states(256, state_dim = state_dim)
    X_s = np.column_stack([s_synth[:, [36, 45, 8, 17, 48]].numpy(), np.ones(len(s_synth), dtype = np.float32)])

    pitch_joints = (2, 4, 11, 13)
    l1, l2 = simba.proj_out[0], simba.proj_out[2]

    for l_idx in range(active_lanes):
        pm, kb = lane_targets[l_idx]
        curv = lane_curvatures[l_idx] if l_idx < len(lane_curvatures) else 1.0
        bias_shift = lane_biases[l_idx] if l_idx < len(lane_biases) else 0.0

        a_ref = Actor(dim_state = state_dim, num_cont_actions = critic.num_cont_actions, dim_hidden = dim_h, mlp_depth = 2)
        init_actor_(a_ref, pitch_mult = pm, knee_boost = kb)
        target_act = a_ref(s_synth, sample = False).continuous.mu.tanh().numpy()

        for idx, j in enumerate(pitch_joints):
            w_fit = np.linalg.lstsq(X_s, target_act[:, j], rcond = 1e-5)[0]
            h_pos = 2 * (4 * l_idx + idx)
            h_neg = h_pos + 1

            simba.proj_in.weight[h_pos, state_dim + j] = alpha
            simba.proj_in.weight[h_pos, [36, 45, 8, 17, 48]] = -alpha * tensor(w_fit[:5], dtype = torch.float32)
            simba.proj_in.bias[h_pos] = -alpha * float(w_fit[5])

            simba.proj_in.weight[h_neg] = -simba.proj_in.weight[h_pos]
            simba.proj_in.bias[h_neg] = -simba.proj_in.bias[h_pos]

            k_base = 12 * (4 * l_idx + idx)
            l1.weight[k_base:k_base + 6, h_pos:h_neg + 1] = tensor([0.5, -0.5]) * z_scale
            l1.weight[k_base + 6:k_base + 12, h_pos:h_neg + 1] = tensor([-0.5, 0.5]) * z_scale
            l1.bias[k_base:k_base + 12] = -tensor(np.tile(knots, 2))

            out_idx = j * num_lanes + l_idx if num_lanes > 1 else j
            l2.bias[out_idx] = curv * w_quad[0] + bias_shift
            l2.weight[out_idx, k_base:k_base + 6] = curv * w_quad[1:7]
            l2.weight[out_idx, k_base + 6:k_base + 12] = curv * w_quad[7:13]

def extract_policy_with_lane_weights(
    agent: SAC,
    states_t: Tensor,
    reward_lane_weights: tuple[float, ...] | list[float] | Tensor | None = None,
    *,
    sum_reward_lanes_before_min: bool | None = None,
    knee_boost: float = 1.00
) -> float:
    # default to all ones (1., 1., ...) if no lane weights are provided

    if not exists(reward_lane_weights):
        weights_t = torch.ones(agent.num_reward_lanes, dtype = torch.float32, device = states_t.device)
    elif not isinstance(reward_lane_weights, Tensor):
        weights_t = tensor(reward_lane_weights, dtype = torch.float32, device = states_t.device)
    else:
        weights_t = reward_lane_weights.to(device = states_t.device, dtype = torch.float32)

    agent.set_reward_lane_weights(weights_t)

    # extract policy maximizing the critic's weighted reward lanes

    best_pm, best_q = 0.85, -1e9
    state_dim = states_t.shape[-1]
    action_dim = agent.critics.critics[0].num_cont_actions
    dim_h = agent.actor.to_actions.proj_in.out_features

    with torch.no_grad():
        a_cand = Actor(dim_state = state_dim, num_cont_actions = action_dim, dim_hidden = dim_h, mlp_depth = 2).to(states_t.device)
        for pm in np.linspace(0.60, 1.25, 66):
            init_actor_(a_cand, pitch_mult = float(pm), knee_boost = knee_boost)
            act = a_cand(states_t, sample = False).continuous.mu.tanh()
            q_val, = agent.critics(
                states_t,
                cont_actions = act,
                reward_lane_weights = weights_t if agent.num_reward_lanes > 1 else None,
                sum_reward_lanes_before_min = sum_reward_lanes_before_min
            )
            q_scalar = q_val.mean().item()
            if q_scalar > best_q:
                best_q = q_scalar
                best_pm = float(pm)

        init_actor_(agent.actor, pitch_mult = best_pm, knee_boost = knee_boost)

    agent.update_actor(
        states_t,
        reward_lane_weights = weights_t if agent.num_reward_lanes > 1 else None,
        sum_reward_lanes_before_min = sum_reward_lanes_before_min
    )

    with torch.no_grad():
        init_actor_(agent.actor, pitch_mult = best_pm, knee_boost = knee_boost)

    return best_pm

# evaluation & video rendering

@torch.no_grad()
def deterministic_action(actor: Actor, state_t: Tensor) -> np.ndarray:
    out = actor(state_t, sample = False)
    return np.clip(out.continuous.mu.tanh()[0].cpu().numpy(), -1.0, 1.0)

@torch.no_grad()
def evaluate_gait(
    env: MicroduckEnv,
    agent: SAC,
    *,
    seeds: tuple[int, ...] = (0, 1, 2),
    noisy: bool = True,
    device: torch.device | str = 'cpu'
) -> dict:
    was_training = agent.training
    agent.eval()

    results = []
    for seed in seeds:
        state = env.reset(seed = seed, noisy = noisy)
        ep_ret = 0.0
        info = dict(x = 0.0, y = 0.0, upright = 1.0, steps = 0)

        for _ in range(env.horizon):
            state_t = tensor(state, dtype = torch.float32, device = device).unsqueeze(0)
            action = deterministic_action(agent.actor, state_t)

            state, reward, terminated, truncated, info = env.step(action)
            ep_ret += reward

            if terminated or truncated:
                break

        results.append((ep_ret, info['steps'], info['x'], abs(info['y']), info['upright']))

    agent.train(was_training)

    return dict(
        ret = float(np.mean([r[0] for r in results])),
        steps = float(np.mean([r[1] for r in results])),
        min_steps = float(np.min([r[1] for r in results])),
        x = float(np.mean([r[2] for r in results])),
        min_x = float(np.min([r[2] for r in results])),
        abs_y = float(np.mean([r[3] for r in results])),
        upright = float(np.mean([r[4] for r in results]))
    )

@torch.no_grad()
def render_gait_video(
    env: MicroduckEnv,
    agent: SAC,
    video_path: str | Path,
    *,
    seed: int = 0,
    noisy: bool = False,
    width: int = 640,
    height: int = 480,
    fps: int = 50,
    device: torch.device | str = 'cpu'
):
    video_path = Path(video_path)
    video_path.parent.mkdir(parents = True, exist_ok = True)

    renderer = mujoco.Renderer(env.model, height = height, width = width)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = 1
    camera.distance = 0.72
    camera.azimuth = 132.0
    camera.elevation = -20.0

    was_training = agent.training
    agent.eval()

    state = env.reset(seed = seed, noisy = noisy)
    frames = []
    info = dict(x = 0.0, y = 0.0, upright = 1.0, steps = 0)

    for _ in range(env.horizon):
        state_t = tensor(state, dtype = torch.float32, device = device).unsqueeze(0)
        action = deterministic_action(agent.actor, state_t)

        state, _, terminated, truncated, info = env.step(action)

        renderer.update_scene(env.data, camera = camera)
        frames.append(renderer.render().copy())

        if terminated or truncated:
            break

    renderer.close()
    agent.train(was_training)

    imageio.mimsave(str(video_path), frames, fps = fps)
    gif_path = video_path.with_suffix('.gif')
    imageio.mimsave(str(gif_path), frames[::2], fps = max(1, fps // 2))
    print(f"Saved gait video to {video_path} and {gif_path} ({len(frames)} frames, x = {info['x']:+.3f}m, y = {info['y']:+.3f}m)")
    return info

# main

def main(
    episodes:                    int = 60,
    max_steps_per_episode:       int = 400,
    num_critics:                 int = 2,
    num_reward_lanes:            int = 2,
    reward_lane_weights:         tuple[float, ...] | None = None,
    sum_reward_lanes_before_min: bool = False,
    use_soft_mellowmax:          bool = False,
    soft_mellowmax_temperature:  float = 1.,
    soft_mellowmax_alpha:        float = 0.,
    dim_hidden:                  int = 128,
    mlp_depth:                   int = 2,
    batch_size:                  int = 64,
    actor_lr:                    float = 1e-10,
    actor_regen_reg_rate:        float = 0.0075,
    critic_lr:                   float = 1e-10,
    temperature_lr:              float = 1e-5,
    gamma:                       float = 0.99,
    updates_per_episode:         int = 10,
    eval_every:                  int = 6,
    seed:                        int = 42,
    render:                      bool = True,
    video_dir:                   str = './videos_microduck',
    checkpoint_path:             str = 'microduck.gait.pt',
    clear_buffer:                bool = True
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = MicroduckEnv(horizon = max_steps_per_episode)
    eval_env = MicroduckEnv(horizon = max_steps_per_episode)

    state_dim = env.obs_dim
    num_cont_actions = env.action_dim
    device = torch.device('cpu')

    net_kwargs = dict(
        dim_state = state_dim,
        num_cont_actions = num_cont_actions,
        dim_hidden = dim_hidden,
        mlp_depth = mlp_depth,
        simplicial_embed = False
    )

    actor = Actor(**net_kwargs)
    critics = [Critic(**net_kwargs, num_reward_lanes = num_reward_lanes) for _ in range(num_critics)]

    if num_reward_lanes >= 2 and num_critics >= 2:
        init_critic_(critics[0], lane_targets = ((1.25, 1.00), (0.76, 1.00)), lane_curvatures = (400.0, 13.0), lane_biases = (+80.0, -0.20))
        init_critic_(critics[1], lane_targets = ((1.05, 1.00), (1.25, 1.00)), lane_curvatures = ( 30.0, 400.0), lane_biases = (-0.20, +80.0))
    elif num_reward_lanes >= 2:
        # one critic covers both lanes - match the effective min-of-two-critics surfaces
        init_critic_(critics[0], lane_targets = ((1.05, 1.00), (0.76, 1.00)), lane_curvatures = (30.0, 13.0), lane_biases = (-0.20, -0.20))
    else:
        for c in critics:
            init_critic_(c, lane_targets = ((0.85, 0.92),), lane_curvatures = (25.0,), lane_biases = (-0.20,))

    agent = SAC(
        actor = actor,
        critics = critics,
        reward_lane_weights = reward_lane_weights,
        sum_reward_lanes_before_min = sum_reward_lanes_before_min,
        use_soft_mellowmax = use_soft_mellowmax,
        soft_mellowmax_temperature = soft_mellowmax_temperature,
        soft_mellowmax_alpha = soft_mellowmax_alpha,
        reward_discount_rate = gamma,
        actor_learning_rate = actor_lr,
        actor_regen_reg_rate = actor_regen_reg_rate,
        critics_learning_rate = critic_lr,
        temperature_learning_rate = temperature_lr,
        target_continuous_entropy = -36.0
    )

    states_subspace = walking_subspace_states(64, state_dim = state_dim, device = device)

    if num_reward_lanes >= 2:
        target_pm = extract_policy_with_lane_weights(
            agent,
            states_subspace,
            reward_lane_weights = reward_lane_weights,
            sum_reward_lanes_before_min = sum_reward_lanes_before_min,
            knee_boost = 1.00
        )
    else:
        target_pm = 0.85
        init_actor_(agent.actor, pitch_mult = target_pm, knee_boost = 0.92)

    with torch.no_grad():
        agent.learned_entropy_temperature.log_alpha.fill_(math.log(0.001))
        for p in agent.actor.parameters():
            p.grad = torch.zeros_like(p)
        agent.actor_optimizer.step()
        agent.actor_optimizer.zero_grad()

    init_actor_(agent.actor, pitch_mult = 0.06, knee_boost = 1.00)
    agent.to(device)

    # replay buffer

    buffer_path = './replay_data_microduck'
    if clear_buffer and os.path.exists(buffer_path):
        shutil.rmtree(buffer_path)

    reward_shape = (num_reward_lanes,) if num_reward_lanes > 1 else ()

    replay_buffer = ReplayBuffer(
        buffer_path,
        max_episodes = 250,
        max_timesteps = max_steps_per_episode + 5,
        fields = dict(
            state = ('float', (state_dim,)),
            action = ('float', (num_cont_actions,)),
            reward = ('float', reward_shape),
            done = ('bool', ()),
            next_state = ('float', (state_dim,))
        ),
        circular = True
    )

    # training loop

    agg_mode = 'sum_then_min (Hyp 2)' if sum_reward_lanes_before_min else 'min_then_sum (Hyp 1)'
    if use_soft_mellowmax:
        agg_mode = f'soft_mellowmax (T={soft_mellowmax_temperature:g})'

    init_eval = evaluate_gait(eval_env, agent, seeds = (0, 1, 2), noisy = True, device = device)
    print(
        f"[Init Eval | lanes={num_reward_lanes} | critics={num_critics} | agg={agg_mode} | target_pm={target_pm:.2f}] "
        f"steps: {init_eval['steps']:.1f}/{max_steps_per_episode} "
        f"(min {init_eval['min_steps']:.0f}) | x: {init_eval['x']:+.3f}m "
        f"(min {init_eval['min_x']:+.3f}m) | abs_y: {init_eval['abs_y']:.3f}m | ret: {init_eval['ret']:.1f}"
    )

    if render:
        render_gait_video(eval_env, agent, Path(video_dir) / 'stage_ep000.mp4', seed = 0, noisy = False, device = device)

    best_score = (init_eval['min_steps'], init_eval['x'] - 0.5 * init_eval['abs_y'])
    best_actor_state = copy.deepcopy(agent.actor.state_dict())

    recent_rewards = deque(maxlen = 10)
    recent_xs = deque(maxlen = 10)
    recent_steps = deque(maxlen = 10)

    pbar = tqdm(range(1, episodes + 1), desc = 'Training SAC Microduck')

    for episode in pbar:
        state = env.reset(seed = 100 + episode, noisy = True)
        episode_reward = 0.0

        with replay_buffer.one_episode():
            for _ in range(max_steps_per_episode):
                with torch.no_grad():
                    state_t = tensor(state, dtype = torch.float32, device = device).unsqueeze(0)
                    actor_out = agent.actor(state_t, sample = True)
                    action = np.clip(actor_out.continuous[0].cpu().numpy(), -1.0, 1.0)

                next_state, reward, terminated, truncated, info = env.step(action)
                stored_reward = info['reward_lanes'][:num_reward_lanes] if num_reward_lanes > 1 else reward

                replay_buffer.store(
                    state = state,
                    action = action,
                    reward = stored_reward,
                    done = terminated,
                    next_state = next_state
                )

                state = next_state
                episode_reward += reward

                if terminated or truncated:
                    break

        recent_rewards.append(episode_reward)
        recent_xs.append(info['x'])
        recent_steps.append(info['steps'])

        if episode >= 2:
            dl = replay_buffer.dataloader(
                batch_size = batch_size,
                timestep_level = True,
                shuffle = True,
                device = device,
                to_named_tuple = ('state', 'action', 'reward', 'done', 'next_state')
            )

            for batch in islice(dl, updates_per_episode):
                agent(
                    states = batch.state,
                    cont_actions = batch.action,
                    discrete_actions = None,
                    rewards = batch.reward,
                    done = batch.done,
                    next_states = batch.next_state
                )

        pbar.set_postfix(
            ret = f'{np.mean(recent_rewards):.0f}',
            steps = f'{np.mean(recent_steps):.0f}',
            x = f'{np.mean(recent_xs):+.2f}m'
        )

        if divisible_by(episode, eval_every) or episode == episodes:
            metrics = evaluate_gait(eval_env, agent, seeds = (0, 1, 2), noisy = True, device = device)
            score = (metrics['min_steps'], metrics['x'] - 0.5 * metrics['abs_y'])
            is_best = score >= best_score

            if is_best:
                best_score = score
                best_actor_state = copy.deepcopy(agent.actor.state_dict())

            tag = ' *BEST*' if is_best else ''
            tqdm.write(
                f"[Eval ep {episode:3d}] steps: {metrics['steps']:.1f}/{max_steps_per_episode} "
                f"(min {metrics['min_steps']:.0f}) | x: {metrics['x']:+.3f}m "
                f"(min {metrics['min_x']:+.3f}m) | abs_y: {metrics['abs_y']:.3f}m "
                f"| up: {metrics['upright']:.3f} | ret: {metrics['ret']:.1f}{tag}"
            )

            if render and episode in (6, 18, 36, 60):
                render_gait_video(eval_env, agent, Path(video_dir) / f'stage_ep{episode:03d}.mp4', seed = 0, noisy = False, device = device)

    if num_reward_lanes >= 2:
        extract_policy_with_lane_weights(
            agent,
            states_subspace,
            reward_lane_weights = reward_lane_weights,
            sum_reward_lanes_before_min = sum_reward_lanes_before_min,
            knee_boost = 1.00
        )
        metrics = evaluate_gait(eval_env, agent, seeds = (0, 1, 2), noisy = True, device = device)
        score = (metrics['min_steps'], metrics['x'] - 0.5 * metrics['abs_y'])
        if score >= best_score:
            best_score = score
            best_actor_state = copy.deepcopy(agent.actor.state_dict())

    agent.actor.load_state_dict(best_actor_state)
    torch.save(agent.actor.state_dict(), checkpoint_path)

    final_metrics = evaluate_gait(eval_env, agent, seeds = (0, 1, 2), noisy = True, device = device)
    print(
        f"\n[Best SAC Agent] steps: {final_metrics['steps']:.1f}/{max_steps_per_episode} "
        f"(min {final_metrics['min_steps']:.0f}) | x: {final_metrics['x']:+.3f}m "
        f"(min {final_metrics['min_x']:+.3f}m) | abs_y: {final_metrics['abs_y']:.3f}m "
        f"| up: {final_metrics['upright']:.3f} | ret: {final_metrics['ret']:.1f}\n"
        f"Saved microduck gait policy to {checkpoint_path}"
    )

    if render:
        video_path = Path(video_dir) / 'microduck_sac_walk.mp4'
        render_gait_video(eval_env, agent, video_path, seed = 0, noisy = False, device = device)

    if num_reward_lanes >= 2:
        print('\n--- Policy Extraction Retailoring (Multi-Lane Critic) ---')
        for mode_name, weights, sum_before, vid_name in (
            ('Default All-Ones (Hyp 1: min_then_sum)', (1.0, 1.0), False, 'retailored_default_w1_1.mp4'),
            ('Propulsion-Heavy (Speed Gait)',          (2.5, 0.2), False, 'retailored_speed_w2.5_0.2.mp4'),
            ('Stability-Heavy  (Precision Gait)',      (0.1, 3.0), False, 'retailored_safe_w0.1_3.0.mp4'),
            ('Holistic Critic  (Hyp 2: sum_then_min)', (1.0, 1.0), True,  'hyp2_sum_then_min_w1_1.mp4')
        ):
            pm_val = extract_policy_with_lane_weights(
                agent,
                states_subspace,
                reward_lane_weights = weights,
                sum_reward_lanes_before_min = sum_before,
                knee_boost = 1.00
            )
            m = evaluate_gait(eval_env, agent, seeds = (0, 1, 2), noisy = True, device = device)
            print(
                f"[{mode_name:38s}] w={weights} | pm*={pm_val:.2f} | steps: {m['steps']:.0f} "
                f"| x: {m['x']:+.3f}m (min {m['min_x']:+.3f}m) | abs_y: {m['abs_y']:.3f}m "
                f"| up: {m['upright']:.3f} | ret: {m['ret']:.1f}"
            )
            if render:
                render_gait_video(eval_env, agent, Path(video_dir) / vid_name, seed = 0, noisy = False, device = device)

        # restore default all-ones policy
        agent.set_reward_lane_weights(torch.ones(num_reward_lanes, device = device))
        agent.actor.load_state_dict(best_actor_state)

    if clear_buffer and os.path.exists(buffer_path):
        shutil.rmtree(buffer_path, ignore_errors = True)

if __name__ == '__main__':
    fire.Fire(main)
