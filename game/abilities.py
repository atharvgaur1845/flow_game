"""Dash ability and per-frame timer decrement."""
from __future__ import annotations

import math

from . import config


def try_dash(player, keys, pygame_mod) -> bool:
    """Trigger a dash if SPACE is held and cooldown has elapsed.

    Returns True if a dash fired this frame.
    """
    if player.dash_cd > 0.0:
        return False
    if not keys[pygame_mod.K_SPACE]:
        return False
    vx, vy = player.vel
    n = math.hypot(vx, vy)
    if n < 0.2:
        # If the player is almost still, dash in their last facing direction.
        fx, fy = player.facing
        player.vel[0] += fx * 6.0
        player.vel[1] += fy * 6.0
    else:
        player.vel[0] *= config.DASH_IMPULSE
        player.vel[1] *= config.DASH_IMPULSE
    player.dash_cd = player.dash_cd_max
    player.invuln  = max(player.invuln, config.DASH_INVULN)
    player.dashing = config.DASH_INVULN
    return True


def update_timers(player, dt: float) -> None:
    player.dash_cd = max(0.0, player.dash_cd - dt)
    player.invuln  = max(0.0, player.invuln  - dt)
    player.dashing = max(0.0, player.dashing - dt)
