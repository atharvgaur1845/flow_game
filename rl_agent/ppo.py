"""Shared PPO implementation (actor-critic, GAE, clipped surrogate loss).

Used by both the low-level policy and the meta-policy — they pass in their
own rollout buffers, policy network, and config overrides.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl_agent.config import (
    GAMMA, GAE_LAMBDA, CLIP_EPS, VF_COEF,
    ENT_COEF_LL, MAX_GRAD_NORM, DEVICE,
)


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------

@dataclass
class RolloutBuffer:
    """Stores one batch of experience for a single policy.

    All tensors live on CPU during collection; moved to DEVICE for update.
    """
    capacity: int
    obs_dim:  int
    act_dim:  int          # flat action size (e.g. 3 for LL, 1 for meta)
    extra_dim: int = 0     # extra per-step data (e.g. is_shop flag)

    obs:      np.ndarray = field(init=False)
    actions:  np.ndarray = field(init=False)
    rewards:  np.ndarray = field(init=False)
    values:   np.ndarray = field(init=False)
    log_probs:np.ndarray = field(init=False)
    dones:    np.ndarray = field(init=False)
    extras:   np.ndarray = field(init=False)
    ptr:      int = field(init=False, default=0)

    def __post_init__(self) -> None:
        C, D, A = self.capacity, self.obs_dim, self.act_dim
        self.obs       = np.zeros((C, D), dtype=np.float32)
        self.actions   = np.zeros((C, A), dtype=np.int64)
        self.rewards   = np.zeros(C,      dtype=np.float32)
        self.values    = np.zeros(C,      dtype=np.float32)
        self.log_probs = np.zeros(C,      dtype=np.float32)
        self.dones     = np.zeros(C,      dtype=np.float32)
        if self.extra_dim > 0:
            self.extras = np.zeros((C, self.extra_dim), dtype=np.float32)
        self.ptr = 0

    def add(self, obs, action, reward, value, log_prob, done, extra=None) -> None:
        i = self.ptr
        self.obs[i]       = obs
        self.actions[i]   = action if np.ndim(action) > 0 else [action]
        self.rewards[i]   = reward
        self.values[i]    = value
        self.log_probs[i] = log_prob
        self.dones[i]     = float(done)
        if self.extra_dim > 0 and extra is not None:
            self.extras[i] = extra
        self.ptr += 1

    def is_full(self) -> bool:
        return self.ptr >= self.capacity

    def reset(self) -> None:
        self.ptr = 0

    def compute_returns_and_advantages(
        self, last_value: float, last_done: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """GAE advantage estimation."""
        n = self.ptr
        advantages = np.zeros(n, dtype=np.float32)
        last_gae   = 0.0
        for t in reversed(range(n)):
            next_val  = last_value if t == n - 1 else self.values[t + 1]
            next_done = last_done  if t == n - 1 else self.dones[t + 1]
            delta     = (self.rewards[t]
                         + GAMMA * next_val * (1.0 - next_done)
                         - self.values[t])
            last_gae  = delta + GAMMA * GAE_LAMBDA * (1.0 - next_done) * last_gae
            advantages[t] = last_gae
        returns = advantages + self.values[:n]
        return returns, advantages

    def to_tensors(self, device: torch.device):
        n = self.ptr
        obs       = torch.from_numpy(self.obs[:n]).to(device)
        actions   = torch.from_numpy(self.actions[:n]).to(device)
        log_probs = torch.from_numpy(self.log_probs[:n]).to(device)
        if self.extra_dim > 0:
            extras = torch.from_numpy(self.extras[:n]).to(device)
        else:
            extras = None
        return obs, actions, log_probs, extras


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------

class PPOTrainer:
    """Runs PPO update epochs on a filled RolloutBuffer."""

    def __init__(
        self,
        policy: nn.Module,
        lr: float,
        n_epochs: int,
        n_minibatches: int,
        ent_coef: float = ENT_COEF_LL,
        clip_eps: float = CLIP_EPS,
        vf_coef: float  = VF_COEF,
        max_grad_norm: float = MAX_GRAD_NORM,
        device: torch.device = DEVICE,
    ) -> None:
        self.policy        = policy
        self.optimizer     = optim.Adam(policy.parameters(), lr=lr, eps=1e-5)
        self.n_epochs      = n_epochs
        self.n_minibatches = n_minibatches
        self.ent_coef      = ent_coef
        self.clip_eps      = clip_eps
        self.vf_coef       = vf_coef
        self.max_grad_norm = max_grad_norm
        self.device        = device

    def update(
        self,
        buffer: RolloutBuffer,
        last_value: float,
        last_done: float,
        is_low_level: bool = True,
    ) -> dict:
        """Run PPO update. Returns dict of training metrics."""
        returns, advantages = buffer.compute_returns_and_advantages(
            last_value, last_done
        )
        returns_t    = torch.from_numpy(returns).to(self.device)
        advantages_t = torch.from_numpy(advantages).to(self.device)
        # Normalize advantages.
        advantages_t = (advantages_t - advantages_t.mean()) / (
            advantages_t.std() + 1e-8
        )

        obs_t, actions_t, old_lp_t, extras_t = buffer.to_tensors(self.device)

        n = buffer.ptr
        mb_size = n // self.n_minibatches
        idx = np.arange(n)

        pg_losses, vf_losses, ent_losses = [], [], []
        clip_fracs = []

        for _ in range(self.n_epochs):
            np.random.shuffle(idx)
            for start in range(0, n, mb_size):
                mb_idx = torch.from_numpy(idx[start: start + mb_size]).long().to(
                    self.device
                )
                mb_obs     = obs_t[mb_idx]
                mb_act     = actions_t[mb_idx]
                mb_old_lp  = old_lp_t[mb_idx]
                mb_ret     = returns_t[mb_idx]
                mb_adv     = advantages_t[mb_idx]

                if is_low_level:
                    # is_shop flag stored in extras[:,0]
                    is_shop = (extras_t[mb_idx, 0] > 0.5) if extras_t is not None \
                              else torch.zeros(len(mb_idx), dtype=torch.bool, device=self.device)
                    _, new_lp, entropy, new_val = self.policy.get_action_and_value(
                        mb_obs, is_shop, mb_act
                    )
                else:
                    # Meta-policy: action is scalar archetype
                    _, new_lp, entropy, new_val = self.policy.get_action_and_value(
                        mb_obs, mb_act[:, 0]
                    )

                ratio = (new_lp - mb_old_lp).exp()
                clip_frac = ((ratio - 1.0).abs() > self.clip_eps).float().mean()
                clip_fracs.append(clip_frac.item())

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * ratio.clamp(1.0 - self.clip_eps,
                                                  1.0 + self.clip_eps)
                pg_loss  = torch.max(pg_loss1, pg_loss2).mean()

                vf_loss  = 0.5 * (new_val - mb_ret).pow(2).mean()
                ent_loss = entropy.mean()

                loss = pg_loss + self.vf_coef * vf_loss - self.ent_coef * ent_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                pg_losses.append(pg_loss.item())
                vf_losses.append(vf_loss.item())
                ent_losses.append(ent_loss.item())

        return {
            "pg_loss":    float(np.mean(pg_losses)),
            "vf_loss":    float(np.mean(vf_losses)),
            "ent_loss":   float(np.mean(ent_losses)),
            "clip_frac":  float(np.mean(clip_fracs)),
        }
