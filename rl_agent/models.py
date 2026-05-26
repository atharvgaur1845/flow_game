"""Neural network models for the hierarchical RL agent.

LowLevelPolicy : actor-critic for real-time gameplay actions
                 Input:  119-dim observation (includes target archetype one-hot)
                 Output: move logits (9), dash logits (2), shop logits (3), value (1)

MetaPolicy     : actor-critic that selects target archetype
                 Input:  16-dim meta-observation
                 Output: archetype logits (5), meta-value (1)

Both use orthogonal initialization and LayerNorm (works with batch size 1
at inference time unlike BatchNorm).
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from rl_agent.config import (
    OBS_DIM, META_OBS_DIM,
    N_MOVE_ACTIONS, N_DASH_ACTIONS, N_SHOP_ACTIONS, N_ARCHETYPES,
)


def _ortho_init(layer: nn.Linear, gain: float = math.sqrt(2)) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class _MLP(nn.Module):
    """Shared trunk: Linear → LayerNorm → ReLU (repeated)."""

    def __init__(self, in_dim: int, hidden: Tuple[int, ...]) -> None:
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers.append(_ortho_init(nn.Linear(prev, h)))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            prev = h
        self.net = nn.Sequential(*layers)
        self.out_dim = prev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LowLevelPolicy(nn.Module):
    """Actor-critic for low-level game actions.

    During play  → sample from (move_dist, dash_dist).
    During shop  → sample from shop_dist.
    The env embeds which mode is active via obs[is_shop flag]; the network
    always computes all heads but the caller uses only the relevant one.
    """

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        hidden: Tuple[int, ...] = (256, 256),
    ) -> None:
        super().__init__()
        self.trunk = _MLP(obs_dim, hidden)
        d = self.trunk.out_dim

        # Actor heads.
        self.move_head = _ortho_init(nn.Linear(d, N_MOVE_ACTIONS), gain=0.01)
        self.dash_head = _ortho_init(nn.Linear(d, N_DASH_ACTIONS), gain=0.01)
        self.shop_head = _ortho_init(nn.Linear(d, N_SHOP_ACTIONS), gain=0.01)

        # Critic.
        self.value_head = _ortho_init(nn.Linear(d, 1), gain=1.0)

    def forward(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (move_logits, dash_logits, shop_logits, value)."""
        h = self.trunk(obs)
        return (
            self.move_head(h),
            self.dash_head(h),
            self.shop_head(h),
            self.value_head(h).squeeze(-1),
        )

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        is_shop: torch.Tensor,          # bool tensor [B]
        action: torch.Tensor | None = None,   # [B, 3] int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample or evaluate actions.

        Returns: actions [B,3], log_prob [B], entropy [B], value [B]
        """
        move_logits, dash_logits, shop_logits, value = self(obs)

        move_dist = Categorical(logits=move_logits)
        dash_dist = Categorical(logits=dash_logits)
        shop_dist = Categorical(logits=shop_logits)

        if action is None:
            move_a = move_dist.sample()
            dash_a = dash_dist.sample()
            shop_a = shop_dist.sample()
            action = torch.stack([move_a, dash_a, shop_a], dim=-1)
        else:
            move_a = action[:, 0]
            dash_a = action[:, 1]
            shop_a = action[:, 2]

        # During shop, log_prob comes from shop_dist; otherwise from move+dash.
        log_prob_play = move_dist.log_prob(move_a) + dash_dist.log_prob(dash_a)
        log_prob_shop = shop_dist.log_prob(shop_a)
        log_prob = torch.where(is_shop, log_prob_shop, log_prob_play)

        entropy_play = move_dist.entropy() + dash_dist.entropy()
        entropy_shop = shop_dist.entropy()
        entropy = torch.where(is_shop, entropy_shop, entropy_play)

        return action, log_prob, entropy, value


class MetaPolicy(nn.Module):
    """Actor-critic that selects target archetype every K low-level steps."""

    def __init__(
        self,
        obs_dim: int = META_OBS_DIM,
        hidden: Tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__()
        self.trunk = _MLP(obs_dim, hidden)
        d = self.trunk.out_dim
        self.arch_head  = _ortho_init(nn.Linear(d, N_ARCHETYPES), gain=0.01)
        self.value_head = _ortho_init(nn.Linear(d, 1), gain=1.0)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (arch_logits [B,5], value [B])."""
        h = self.trunk(obs)
        return self.arch_head(h), self.value_head(h).squeeze(-1)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self(obs)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy  = dist.entropy()
        return action, log_prob, entropy, value
