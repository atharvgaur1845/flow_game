"""Reward computation helpers — standalone so they can be unit-tested."""
from __future__ import annotations

from rl_agent.config import LAMBDA_ALIGN, SCORE_NORM, SURVIVAL_BONUS, DEATH_PENALTY


def compute_reward(
    score_delta: float,
    pi_current,         # tensor or array of length 5
    target_archetype: int,
    terminated: bool,
) -> float:
    """Compute per-step reward.

    reward = score_delta / SCORE_NORM
           + λ[target] * π[target]
           + SURVIVAL_BONUS (each step)
           + DEATH_PENALTY  (terminal step only)
    """
    alignment  = float(pi_current[target_archetype])
    score_r    = score_delta / SCORE_NORM
    align_r    = LAMBDA_ALIGN[target_archetype] * alignment
    survival_r = SURVIVAL_BONUS
    death_r    = DEATH_PENALTY if terminated else 0.0
    return float(score_r + align_r + survival_r + death_r)


def compute_meta_reward(ll_rewards_sum: float) -> float:
    """Meta-policy reward = sum of low-level rewards over the option period."""
    return float(ll_rewards_sum)
