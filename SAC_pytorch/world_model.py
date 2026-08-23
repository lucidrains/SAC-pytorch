from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from einops import rearrange, reduce, repeat

from torch_einops_utils import batched_index_select

if TYPE_CHECKING:
    from SAC_pytorch.SAC import Actor, Critic, MultipleCritics

# world model for test-time tree search
# QWM - Dong et al. https://arxiv.org/abs/2608.17163

class WorldModel(nn.Module):
    """
    World model for test-time tree search on top of Q-learning, following
    "Q-Learning with World Models" (QWM) - Dong et al. https://arxiv.org/abs/2608.17163

    Built on top of the SPR / forward-dynamics model (FDM) embedded within the actor -
    the latent transition + reconstruction head of its SPR branch is the world model here.
    """

    def __init__(
        self,
        *,
        actor,
        critics,
        num_candidates = 8,
        num_world_samples = 1,
        search_depth = 4,
        num_beam = 1,
        num_leaf_actions = 8,
        tree_discount = 0.1
    ):
        """
        N       - num_candidates: candidate actions per state node
        K       - num_world_samples: next states sampled per action (deterministic, so K = 1)
        D       - search_depth: depth of the search tree
        J       - num_beam: surviving paths kept after pruning at each level
        Nleaf   - num_leaf_actions: actions sampled at the leaf, mean-aggregated
        λ       - tree_discount: recursive discount for future imagined values (0.1 in paper)
        """
        super().__init__()

        self.actor = actor
        self.critics = critics

        assert all(critic.spr for critic in critics.critics), 'world model requires the critics to be constructed with spr = True'
        assert actor.spr, 'world model requires the actor to be constructed with spr = True'
        assert search_depth >= 1, 'search_depth must be at least 1'
        assert num_beam <= num_candidates, 'num_beam must be at most num_candidates'
        assert num_leaf_actions >= 1, 'num_leaf_actions must be at least 1'

        self.num_candidates = num_candidates
        self.num_world_samples = num_world_samples
        self.search_depth = search_depth
        self.num_beam = num_beam
        self.num_leaf_actions = num_leaf_actions
        self.tree_discount = tree_discount

    def predict_next_state(self, state, cont_actions):
        """Mψ(s, a) - next state via actor's SPR reconstruction head"""
        return self.actor.spr_next_state(state, cont_actions)

    def reward_at(self, state, cont_actions, embed = None):
        """rψ(s, a) - reward model via actor's SPR branch"""
        return self.actor.spr_reward(state, cont_actions, embed = embed)

    def critic_value(self, state, cont_actions):
        """min over the critic ensemble of Q(s, a) - summed over the per-action-dimension values"""
        min_values, = self.critics(state, cont_actions = cont_actions)
        return reduce(min_values, '... n -> ...', 'sum')

    def value_at(self, states, depth):
        """recursive node value V(d | s) - critic score (eq. 5) plus world-model rollout value (eq. 6)"""
        if depth == self.search_depth:

            # leaf - mean aggregate of Q over Nleaf sampled actions (eq. 7)

            leaf_actions = self.actor(repeat(states, 'b s -> (b n) s', n = self.num_leaf_actions), sample = True).continuous
            leaf_values = self.critic_value(repeat(states, 'b s -> (b n) s', n = self.num_leaf_actions), leaf_actions)
            leaf_values = rearrange(leaf_values, '(b n) -> b n', b = states.shape[0], n = self.num_leaf_actions)
            return reduce(leaf_values, 'b n -> b', 'mean')

        N = self.num_candidates
        J = self.num_beam

        b = states.shape[0]

        # expand each state node with N candidate actions from the policy
        # the actor's SPR branch embedding comes out of the same forward pass

        candidate_output = self.actor(repeat(states, 'b s -> (b n) s', n = N), sample = True)
        candidate_actions = candidate_output.continuous
        candidate_embeds = candidate_output.spr_embed

        candidate_values = self.critic_value(repeat(states, 'b s -> (b n) s', n = N), candidate_actions)
        candidate_values = rearrange(candidate_values, '(b n) -> b n', b = b, n = N)

        # world model rollouts - predicted next states for each candidate action (K = 1 deterministic)

        next_states = self.actor.spr_next_state(repeat(states, 'b s -> (b n) s', n = N), candidate_actions, embed = candidate_embeds)
        next_states = rearrange(next_states, '(b n) s -> b n s', b = b, n = N)

        # reward model - rψ(s_d, a^n) for each candidate action (eq. 6)

        candidate_rewards = self.actor.spr_reward(repeat(states, 'b s -> (b n) s', n = N), candidate_actions, embed = candidate_embeds)
        candidate_rewards = rearrange(candidate_rewards, '(b n) 1 -> b n', b = b, n = N)

        # VQ - max aggregation of the direct Q-values (eq. 5)

        vq = reduce(candidate_values, 'b n -> b', 'max')

        # beam pruning (eq. 12) - keep top-J paths by Q-score and only roll those out

        _, best_indices = candidate_values.topk(J, dim = -1)

        best_next_states = batched_index_select(next_states, best_indices, dim = 1)
        best_next_states = rearrange(best_next_states, 'b j s -> (b j) s')

        best_rewards = batched_index_select(candidate_rewards, best_indices, dim = 1)

        children_values = self.value_at(best_next_states, depth + 1)
        children_values = rearrange(children_values, '(b j) -> b j', b = b, j = J)

        # Vr - reward of the current node plus discounted value of the imagined future (eq. 6),
        # max-aggregated over the surviving paths

        vr = reduce(best_rewards + self.tree_discount * children_values, 'b j -> b', 'max')

        return vq + vr

    @torch.no_grad()
    def tree_search(self, state, return_scores = False):
        """QWM tree search from the given state - root value per candidate action (eq. 9)"""
        N = self.num_candidates

        # root candidates proposed by the policy
        # the actor's SPR branch embedding comes out of the same forward pass

        root_output = self.actor(repeat(state, '1 s -> n s', n = N), sample = True)
        root_actions = root_output.continuous
        root_embeds = root_output.spr_embed

        root_q_values = self.critic_value(repeat(state, '1 s -> n s', n = N), root_actions)

        # reward model at the root - rψ(s0, a^n) for each candidate (eq. 9)

        root_rewards = self.actor.spr_reward(repeat(state, '1 s -> n s', n = N), root_actions, embed = root_embeds)
        root_rewards = rearrange(root_rewards, 'n 1 -> n')

        # world-model predicted next states for each root candidate

        next_states = self.actor.spr_next_state(repeat(state, '1 s -> n s', n = N), root_actions, embed = root_embeds)

        # continuation values - recursively aggregated over the imagined rollouts

        continuation_values = self.value_at(next_states, depth = 1)

        # root value (eq. 9), with the 1/2 folded into λ as in the paper

        tree_scores = root_q_values + root_rewards + self.tree_discount * continuation_values

        if not return_scores:
            return tree_scores

        return tree_scores, root_actions

    @torch.no_grad()
    def select_action(self, state, sample = True, return_scores = False):
        """select action via tree search - argmax at eval, softmax sampling during rollouts"""
        tree_scores, root_actions = self.tree_search(state, return_scores = True)

        if sample:
            index = tree_scores.softmax(dim = -1).multinomial(1)
        else:
            index = tree_scores.topk(1, dim = -1).indices

        action = root_actions.index_select(0, index)

        if not return_scores:
            return action

        return action, tree_scores
