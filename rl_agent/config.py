"""All RL hyperparameters. Change NUM_ENVS to scale between RTX 4060 → A6000."""
from __future__ import annotations

import torch

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
SIM_DT          = 1 / 30.0          # simulated seconds per env step (30 Hz)
MAX_EP_STEPS    = 50_000            # episode truncation
OBS_DIM         = 119               # flat observation vector size
N_MOVE_ACTIONS  = 9                 # 0=none, 1-8 = 8 compass directions
N_DASH_ACTIONS  = 2                 # 0=no dash, 1=dash
N_SHOP_ACTIONS  = 3                 # choose upgrade 1, 2, or 3
N_ARCHETYPES    = 5                 # Arousal, Tactical, Overload, Flow, Apathy

# ---------------------------------------------------------------------------
# Parallelism — single knob to scale between GPUs
# ---------------------------------------------------------------------------
NUM_ENVS        = 8     # RTX 4060 default; set 64 for A6000 48GB

# ---------------------------------------------------------------------------
# Hierarchical structure
# ---------------------------------------------------------------------------
OPTION_PERIOD_K     = 200           # low-level steps per meta-policy decision
META_OBS_DIM        = 16            # meta-policy input size

# ---------------------------------------------------------------------------
# PPO — shared defaults (overridden per-policy where noted)
# ---------------------------------------------------------------------------
GAMMA           = 0.99
GAE_LAMBDA      = 0.95
CLIP_EPS        = 0.2
VF_COEF         = 0.5
ENT_COEF_LL     = 0.01              # low-level entropy bonus
ENT_COEF_META   = 0.05              # higher exploration for meta-policy
MAX_GRAD_NORM   = 0.5

# Low-level policy
LR_LOW          = 3e-4
LL_ROLLOUT_STEPS    = 2048          # steps per env before update
LL_N_EPOCHS         = 4
LL_N_MINIBATCHES    = 8

# Meta-policy
LR_META         = 1e-4
META_ROLLOUT_STEPS  = 10_240        # = 51 meta-decisions per env
META_N_EPOCHS       = 4
META_N_MINIBATCHES  = 4

# ---------------------------------------------------------------------------
# Training budget
# ---------------------------------------------------------------------------
TOTAL_TIMESTEPS = 10_000_000
CHECKPOINT_EVERY = 100_000         # save checkpoint every N env steps

# ---------------------------------------------------------------------------
# Reward — per-archetype alignment weight λ
# ---------------------------------------------------------------------------
LAMBDA_ALIGN = {
    0: 0.3,   # Arousal
    1: 0.4,   # Tactical
    2: 0.3,   # Overload
    3: 0.5,   # Flow
    4: 0.5,   # Apathy
}
SCORE_NORM      = 500.0             # score delta normalization constant
SURVIVAL_BONUS  = 0.005             # small reward per step alive
DEATH_PENALTY   = -1.0

# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
