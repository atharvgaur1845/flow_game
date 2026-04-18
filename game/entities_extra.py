"""Enemy variants and Boss entity.

Enemy archetypes:
    chaser - medium HP, medium speed, medium radius (baseline).
    fast   - low HP, high speed, small radius.
    tank   - high HP, low speed, large radius.

Spawn mix is biased by the inferred pi mixture without touching the
backend: we only *read* pi and weight the probabilities.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple

from . import config

# Archetype indices (must match flow_game.py).
ARCH_AROUSAL  = 0
ARCH_TACTICAL = 1
ARCH_OVERLOAD = 2
ARCH_FLOW     = 3
ARCH_APATHY   = 4


@dataclass
class Enemy:
    pos: List[float]
    vel: List[float]
    hp: int = 2
    max_hp: int = 2
    base_speed: float = 1.0
    radius: float = 0.38
    kind: str = "chaser"
    contact_cd: float = 0.0
    contact_cd_max: float = 0.35


def _spawn_at_perimeter(player_pos, world_half: float) -> Tuple[float, float, float, float]:
    angle = random.uniform(0.0, 2 * math.pi)
    r = world_half * 0.95
    x, y = r * math.cos(angle), r * math.sin(angle)
    dx, dy = player_pos[0] - x, player_pos[1] - y
    n = math.hypot(dx, dy) + 1e-6
    return x, y, dx / n, dy / n


def make_enemy(kind: str, player_pos, world_half: float) -> Enemy:
    s = config.ENEMY_STATS[kind]
    x, y, ux, uy = _spawn_at_perimeter(player_pos, world_half)
    return Enemy(
        pos=[x, y],
        vel=[ux * s["speed"], uy * s["speed"]],
        hp=s["hp"],
        max_hp=s["hp"],
        base_speed=s["speed"],
        radius=s["radius"],
        kind=kind,
        contact_cd=0.0,
        contact_cd_max=s["contact_cd"],
    )


def pick_kind(pi) -> str:
    """Weighted sample of enemy kind based on the pi mixture.

    We only read pi. Overload -> more chasers, Tactical -> more fast
    enemies (they punish loose play), Apathy -> more tanks (slow, makes
    the arena feel sludgy), with Arousal nudging toward fast too.
    """
    w_chaser = 1.0 + 1.5 * float(pi[ARCH_OVERLOAD])
    w_fast   = 0.5 + 2.0 * float(pi[ARCH_TACTICAL]) + 0.6 * float(pi[ARCH_AROUSAL])
    w_tank   = 0.3 + 1.8 * float(pi[ARCH_APATHY])
    total = w_chaser + w_fast + w_tank
    r = random.random() * total
    if r < w_chaser:
        return "chaser"
    if r < w_chaser + w_fast:
        return "fast"
    return "tank"


def step_enemies(enemies: List[Enemy], player_pos, speed_mult: float,
                 slow_factor: float, dt: float) -> None:
    """Integrate enemy motion. speed_mult = E[0] * room scaling."""
    for e in enemies:
        dx = player_pos[0] - e.pos[0]
        dy = player_pos[1] - e.pos[1]
        n = math.hypot(dx, dy) + 1e-6
        target_vx = (dx / n) * e.base_speed
        target_vy = (dy / n) * e.base_speed
        e.vel[0] += (target_vx - e.vel[0]) * 0.05
        e.vel[1] += (target_vy - e.vel[1]) * 0.05
        m = speed_mult * slow_factor
        e.pos[0] += e.vel[0] * m * dt
        e.pos[1] += e.vel[1] * m * dt
        e.contact_cd = max(0.0, e.contact_cd - dt)


# ---------------------------------------------------------------------------
# Boss
# ---------------------------------------------------------------------------
@dataclass
class Boss:
    pos: List[float] = field(default_factory=lambda: [0.0, 0.0])
    vel: List[float] = field(default_factory=lambda: [0.0, 0.0])
    hp: int = config.BOSS_HP
    max_hp: int = config.BOSS_HP
    radius: float = config.BOSS_RADIUS
    contact_cd: float = 0.0
    teleport_cd: float = config.BOSS_TELEPORT_CD
    minion_cd: float = config.BOSS_MINION_CD
    attack_cd: float = config.BOSS_ATTACK_CD
    hit_flash: float = 0.0
    phase_label: str = "idle"


def step_boss(boss: Boss, player_pos, pi, world_half: float,
              minions: List[Enemy], dt: float) -> None:
    """Boss AI. Reads pi to select behaviour; mutates boss + minions.

    Behaviour:
        High Arousal  -> fast aggressive pursuit.
        High Overload -> summons minions on cooldown.
        High Flow     -> teleports to flank the player.
        High Tactical -> precise, calculated positioning at ideal range.
    """
    boss.contact_cd = max(0.0, boss.contact_cd - dt)
    boss.teleport_cd = max(0.0, boss.teleport_cd - dt)
    boss.minion_cd = max(0.0, boss.minion_cd - dt)
    boss.attack_cd = max(0.0, boss.attack_cd - dt)
    boss.hit_flash = max(0.0, boss.hit_flash - dt)

    # Determine dominant archetype for phase label only.
    arch = max(range(len(pi)), key=lambda i: float(pi[i]))
    boss.phase_label = ["AROUSAL", "TACTICAL", "OVERLOAD", "FLOW", "APATHY"][arch]

    dx = player_pos[0] - boss.pos[0]
    dy = player_pos[1] - boss.pos[1]
    dist = math.hypot(dx, dy) + 1e-6
    ux, uy = dx / dist, dy / dist

    # Aggression scales with arousal, defense with tactical.
    aggression = 0.4 + 1.8 * float(pi[ARCH_AROUSAL])
    ideal_dist = 4.5 - 2.5 * float(pi[ARCH_AROUSAL]) + 1.5 * float(pi[ARCH_TACTICAL])
    range_err = dist - ideal_dist
    target_vx = ux * range_err * 0.4 * aggression
    target_vy = uy * range_err * 0.4 * aggression

    boss.vel[0] += (target_vx - boss.vel[0]) * 0.08
    boss.vel[1] += (target_vy - boss.vel[1]) * 0.08
    boss.pos[0] = max(-world_half, min(world_half, boss.pos[0] + boss.vel[0] * dt))
    boss.pos[1] = max(-world_half, min(world_half, boss.pos[1] + boss.vel[1] * dt))

    # Flow state -> teleport attacks (flank).
    if float(pi[ARCH_FLOW]) > 0.25 and boss.teleport_cd <= 0.0:
        flank_angle = math.atan2(-dy, -dx) + random.choice([-1, 1]) * math.pi * 0.45
        r = 3.5
        boss.pos[0] = max(-world_half, min(world_half,
                                           player_pos[0] + math.cos(flank_angle) * r))
        boss.pos[1] = max(-world_half, min(world_half,
                                           player_pos[1] + math.sin(flank_angle) * r))
        boss.vel[0] = boss.vel[1] = 0.0
        boss.teleport_cd = config.BOSS_TELEPORT_CD * (1.5 - float(pi[ARCH_FLOW]))

    # Overload state -> spawn minions.
    if float(pi[ARCH_OVERLOAD]) > 0.25 and boss.minion_cd <= 0.0:
        spawn_n = 1 + int(2 * float(pi[ARCH_OVERLOAD]))
        for _ in range(spawn_n):
            if len(minions) >= config.MAX_ENEMIES - 1:
                break
            kind = "fast" if random.random() < 0.5 else "chaser"
            minions.append(make_enemy(kind, player_pos, world_half))
        boss.minion_cd = config.BOSS_MINION_CD * (1.5 - float(pi[ARCH_OVERLOAD]))
