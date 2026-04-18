"""Dash ability and per-frame timer decrement."""
from __future__ import annotations

import math

from . import config


def try_dash(player, keys, pygame_mod, input_dir) -> bool:
    """Trigger a dash if SPACE is held and cooldown has elapsed.

    Direction priority: current input -> current velocity -> last facing.
    `input_dir` is the (ix, iy) read from WASD this frame (already
    normalized by the caller, but we re-normalize defensively).

    Returns True if a dash fired this frame.
    """
    if player.dash_cd > 0.0 or not keys[pygame_mod.K_SPACE]:
        return False
    ix, iy = input_dir
    ilen = math.hypot(ix, iy)
    if ilen > 0.1:
        dx, dy = ix / ilen, iy / ilen
    else:
        vx, vy = player.vel
        vlen = math.hypot(vx, vy)
        if vlen > 0.2:
            dx, dy = vx / vlen, vy / vlen
        else:
            dx, dy = player.facing
    # Snap velocity to a uniform dash speed so behaviour is consistent
    # whether the player was already moving fast or standing still.
    cur_speed = math.hypot(player.vel[0], player.vel[1])
    speed = max(cur_speed, 6.0) * config.DASH_IMPULSE * 0.6
    player.vel[0] = dx * speed
    player.vel[1] = dy * speed
    player.dash_dir = [dx, dy]
    player.facing   = [dx, dy]
    player.dash_cd  = player.dash_cd_max
    player.invuln   = max(player.invuln, config.DASH_INVULN)
    player.dashing  = config.DASH_INVULN
    return True


def update_timers(player, dt: float) -> None:
    player.dash_cd = max(0.0, player.dash_cd - dt)
    player.invuln  = max(0.0, player.invuln  - dt)
    player.dashing = max(0.0, player.dashing - dt)
