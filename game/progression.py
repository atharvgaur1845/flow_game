"""Run state, room progression, and shop upgrades.

All scaling here sits *on top of* the backend outputs. We multiply
E[0..3] by room scaling; we never replace them.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Tuple

from . import config


# Indices into pi (match flow_game.py).
ARCH_FLOW = 3


@dataclass
class RunState:
    # Progress
    room: int = 0
    kills_in_room: int = 0
    total_kills: int = 0
    # Scoring
    score: float = 0.0
    max_flow: float = 0.0
    time_survived: float = 0.0
    rooms_cleared: int = 0
    # Modifiers (shop-driven multipliers; start at identity).
    damage_mult: float = 1.0          # incoming damage scaler
    score_mult_bonus: float = 0.0     # additive to the pi-flow multiplier
    enemy_slow: float = 1.0           # <=1.0, scales enemy speed
    # Current-room bookkeeping.
    room_time_total: float = 0.0
    room_time_remaining: float = 0.0
    room_kills_required: int = 0
    is_boss_room: bool = False

    def start_next_room(self) -> None:
        self.room += 1
        self.kills_in_room = 0
        self.is_boss_room = (self.room % config.BOSS_EVERY_N_ROOMS == 0)
        if self.is_boss_room:
            self.room_time_total = 60.0 + self.room * 4.0
            self.room_kills_required = 1           # kill the boss
        else:
            self.room_time_total = (config.ROOM_BASE_TIME
                                    + self.room * config.ROOM_TIME_STEP)
            self.room_kills_required = (config.ROOM_BASE_KILLS
                                        + self.room * config.ROOM_KILL_STEP)
        self.room_time_remaining = self.room_time_total

    def clear_room(self) -> None:
        self.rooms_cleared += 1
        self.score += (config.SCORE_BOSS_CLEAR if self.is_boss_room
                       else config.SCORE_ROOM_CLEAR)

    def tick(self, pi, dt: float) -> None:
        flow = float(pi[ARCH_FLOW])
        self.max_flow = max(self.max_flow, flow)
        self.time_survived += dt
        # Time-based score: accrues faster when in Flow.
        self.score += config.SCORE_TIME_BASE * (1.0 + flow) * dt

    def score_multiplier(self, pi) -> float:
        return 1.0 + 2.0 * float(pi[ARCH_FLOW]) + self.score_mult_bonus

    def register_kill(self, pi) -> None:
        self.kills_in_room += 1
        self.total_kills += 1
        self.score += config.SCORE_KILL_BASE * self.score_multiplier(pi)

    def room_goal_met(self) -> bool:
        if self.is_boss_room:
            return False  # boss handled separately
        if self.kills_in_room >= self.room_kills_required:
            return True
        if self.room_time_remaining <= 0.0:
            return True
        return False

    def should_open_shop(self) -> bool:
        return (self.room > 0
                and self.room % config.SHOP_EVERY_N_ROOMS == 0
                and not self.is_boss_room)


# Room scaling applied to backend physics outputs.
def scale_enemy_speed(E0: float, room: int, enemy_slow: float) -> float:
    return float(E0) * (1.0 + config.ENEMY_SCALE_SPEED * room) * enemy_slow


def scale_spawn_rate(E1: float, room: int) -> float:
    return float(E1) * (1.0 + config.ENEMY_SCALE_SPAWN * room)


# ---------------------------------------------------------------------------
# Shop
# ---------------------------------------------------------------------------
UPGRADES: List[Tuple[str, str]] = [
    ("More HP",       "+20 max HP, fully heal"),
    ("Shorter Dash",  "Dash cooldown -20%"),
    ("Score Boost",   "+0.25 score multiplier"),
    ("Slow Enemies",  "Enemies move 15% slower"),
    ("Damage Armor",  "-20% damage taken"),
]


def pick_shop_choices(n: int = 3) -> List[int]:
    return random.sample(range(len(UPGRADES)), n)


def apply_upgrade(idx: int, player, run: RunState) -> str:
    name, _ = UPGRADES[idx]
    if name == "More HP":
        player.max_hp += 20
        player.hp = player.max_hp
    elif name == "Shorter Dash":
        player.dash_cd_max = max(0.25, player.dash_cd_max * 0.8)
    elif name == "Score Boost":
        run.score_mult_bonus += 0.25
    elif name == "Slow Enemies":
        run.enemy_slow *= 0.85
    elif name == "Damage Armor":
        run.damage_mult *= 0.8
    return name
