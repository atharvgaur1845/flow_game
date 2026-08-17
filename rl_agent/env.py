"""Headless FlowGameEnv — Gymnasium-compatible environment for RL training.

Runs the full game logic (physics, enemies, boss, psychological inference,
scoring) without any OpenGL/pygame rendering. Import order matters: we set
FLOW_HEADLESS before importing flow_game so that game/ui.py is never loaded.
"""
from __future__ import annotations

import math
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

# Must be set before any flow_game import to suppress pygame/OpenGL.
os.environ.setdefault("FLOW_HEADLESS", "1")

# Make the repo root importable regardless of working directory.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
from game import abilities, config as gcfg
from game.entities_extra import (
    Boss, Enemy, Pickup,
    PICKUP_BUFF_DURATION, PICKUP_CODE, PICKUP_KINDS, PICKUP_LABEL,
    make_enemy, make_pickup, pick_kind, step_boss, step_enemies,
)
from game.game_state import GameState
from game.progression import (
    UPGRADES, RunState, apply_upgrade, pick_shop_choices,
    scale_enemy_speed, scale_spawn_rate,
)
from flow_game import (
    AttentionStateInference, EnvironmentMapper, ObservationBuilder, Player,
    apply_pickup, decay_buffs,
    ARCH_AROUSAL, ARCH_TACTICAL, ARCH_OVERLOAD, ARCH_FLOW, ARCH_APATHY,
    WORLD_HALF, DASH_TRAIL_TTL, DASH_TRAIL_RADIUS, DASH_TRAIL_MAX,
    SPEED_BOOST_MULT, HEAL_AMOUNT, MAX_HP_BONUS,
    K as N_PI,
)

from rl_agent.config import (
    SIM_DT, MAX_EP_STEPS, OBS_DIM,
    N_MOVE_ACTIONS, N_DASH_ACTIONS, N_SHOP_ACTIONS, N_ARCHETYPES,
)

# Movement direction table: (dx, dy) for actions 0-8.
_MOVE_DIRS: List[Tuple[float, float]] = [
    (0.0,  0.0),   # 0: none
    (0.0,  1.0),   # 1: up
    (0.0, -1.0),   # 2: down
    (-1.0, 0.0),   # 3: left
    (1.0,  0.0),   # 4: right
    (-1.0, 1.0),   # 5: up-left
    (1.0,  1.0),   # 6: up-right
    (-1.0,-1.0),   # 7: down-left
    (1.0, -1.0),   # 8: down-right
]

# Normalize diagonals.
_MOVE_DIRS = [
    (dx / (math.hypot(dx, dy) or 1.0), dy / (math.hypot(dx, dy) or 1.0))
    for dx, dy in _MOVE_DIRS
]

PICKUP_TYPES = list(PICKUP_KINDS)   # ["heal", "dash_boost", "speed_boost", "shield", "max_hp"]
N_UPGRADES   = len(UPGRADES)        # 5


class FlowGameEnv:
    """Single-instance headless Flow Game environment.

    Action format (numpy int array of shape [3]):
        [0] move  : 0-8 (direction, see _MOVE_DIRS)
        [1] dash  : 0 or 1
        [2] shop  : 0-2 (only meaningful when state == SHOP)

    Observation: float32 vector of shape [OBS_DIM].

    target_archetype: int 0-4, set externally by the meta-policy via
        env.set_target(archetype). The observation always includes it.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self._device = torch.device("cpu")   # inference stays on CPU per env
        self._attn   = AttentionStateInference().to(self._device)
        self._mapper = EnvironmentMapper().to(self._device)

        # These are reset on each episode.
        self.obs_builder: ObservationBuilder
        self.player: Player
        self.enemies: List[Enemy]
        self.boss: Optional[Boss]
        self.run: RunState
        self.pickups: List[Pickup]
        self.shop_choices: List[int]
        self.pi: torch.Tensor
        self.E: torch.Tensor
        self.c: torch.Tensor
        self.state: GameState
        self.sim_time: float
        self.spawn_accum: float
        self.pickup_timer: float
        self.step_count: int
        self.target_archetype: int = 0  # meta-policy sets this
        self._prev_score: float = 0.0

        self.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_target(self, archetype: int) -> None:
        self.target_archetype = int(archetype)

    def reset(self) -> np.ndarray:
        self.obs_builder = ObservationBuilder()
        self.player      = Player()
        self.enemies     = []
        self.boss        = None
        self.run         = RunState()
        self.pickups     = []
        self.shop_choices = []
        self.spawn_accum  = 0.0
        self.pickup_timer = gcfg.PICKUP_SPAWN_INTERVAL * 0.4
        self.sim_time     = 0.0
        self.step_count   = 0
        self._prev_score  = 0.0

        K_pi = N_PI
        self.pi = torch.full((K_pi,), 1.0 / K_pi, device=self._device)
        self.E  = torch.tensor([1.0, 0.5, 0.5, 1.0], device=self._device)
        self.c  = torch.zeros(4, device=self._device)

        self.state = GameState.PLAYING
        self.run.start_next_room()

        return self._get_obs()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Advance one simulation step.

        Returns: obs, reward, terminated, truncated, info
        """
        move_act  = int(action[0])
        dash_act  = int(action[1])
        shop_act  = int(action[2])

        dt = SIM_DT

        if self.state == GameState.SHOP:
            self._handle_shop(shop_act)
        elif self.state in (GameState.PLAYING, GameState.BOSS):
            self._physics_step(move_act, dash_act, dt)

        self.step_count += 1
        self.sim_time   += dt

        terminated = (self.player.hp <= 0.0)
        truncated  = (self.step_count >= MAX_EP_STEPS)

        score_delta = self.run.score - self._prev_score
        self._prev_score = self.run.score

        reward = self._compute_reward(score_delta, terminated)
        obs    = self._get_obs()

        info = {
            "score":      self.run.score,
            "room":       self.run.room,
            "pi":         self.pi.numpy().copy(),
            "flow":       float(self.pi[ARCH_FLOW]),
            "state":      GameState.NAMES.get(self.state, str(self.state)),
        }
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Internal: physics / game logic
    # ------------------------------------------------------------------

    def _handle_shop(self, shop_act: int) -> None:
        idx = min(shop_act, len(self.shop_choices) - 1)
        apply_upgrade(self.shop_choices[idx], self.player, self.run)
        self.run.start_next_room()
        if self.run.is_boss_room:
            self.state = GameState.BOSS
            self.boss  = Boss()
            self.boss.pos = [0.0, 4.0]
        else:
            self.state = GameState.PLAYING
        self.enemies     = []
        self.boss        = None if not self.run.is_boss_room else self.boss
        self.spawn_accum = 0.0
        self.pickups     = []
        self.pickup_timer = gcfg.PICKUP_SPAWN_INTERVAL * 0.4

    def _physics_step(self, move_act: int, dash_act: int, dt: float) -> None:
        player  = self.player
        run     = self.run
        pi      = self.pi
        E       = self.E

        # --- Input ---
        raw_ix, raw_iy = _MOVE_DIRS[move_act]
        ilen = math.hypot(raw_ix, raw_iy)
        if ilen > 1e-6:
            tgt_ix, tgt_iy = raw_ix / ilen, raw_iy / ilen
            player.facing  = [tgt_ix, tgt_iy]
        else:
            tgt_ix = tgt_iy = 0.0

        lerp = min(1.0, dt * 18.0)
        player.input_dir[0] += (tgt_ix - player.input_dir[0]) * lerp
        player.input_dir[1] += (tgt_iy - player.input_dir[1]) * lerp
        ix, iy = player.input_dir

        speed_mult = SPEED_BOOST_MULT if "speed_boost" in player.buffs else 1.0
        ax = ix * gcfg.PLAYER_ACCEL * speed_mult
        ay = iy * gcfg.PLAYER_ACCEL * speed_mult

        # Register movement keypresses for APM tracking.
        if ilen > 0.1:
            self.obs_builder.register_keypress()
        if dash_act == 1 and player.dash_cd <= 0.0:
            self.obs_builder.register_keypress()
            self._apply_dash(raw_ix, raw_iy)

        friction = float(max(E[2].item(), 0.05))
        player.vel[0] += ax * dt
        player.vel[1] += ay * dt
        damp = math.exp(-friction * dt)
        player.vel[0] *= damp
        player.vel[1] *= damp

        # Boundary bounce.
        BOUNCE_DAMP = 0.55
        new_x = player.pos[0] + player.vel[0] * dt
        new_y = player.pos[1] + player.vel[1] * dt
        if new_x > WORLD_HALF:
            new_x = WORLD_HALF
            if player.vel[0] > 0.0:
                player.vel[0] = -player.vel[0] * BOUNCE_DAMP
        elif new_x < -WORLD_HALF:
            new_x = -WORLD_HALF
            if player.vel[0] < 0.0:
                player.vel[0] = -player.vel[0] * BOUNCE_DAMP
        if new_y > WORLD_HALF:
            new_y = WORLD_HALF
            if player.vel[1] > 0.0:
                player.vel[1] = -player.vel[1] * BOUNCE_DAMP
        elif new_y < -WORLD_HALF:
            new_y = -WORLD_HALF
            if player.vel[1] < 0.0:
                player.vel[1] = -player.vel[1] * BOUNCE_DAMP
        player.pos[0] = new_x
        player.pos[1] = new_y

        abilities.update_timers(player, dt)
        player.hit_flash = max(0.0, player.hit_flash - dt * 3.0)

        # --- Dash trail ---
        if player.dashing > 0.0:
            if (not player.dash_trail or
                    math.hypot(player.pos[0] - player.dash_trail[-1][0],
                               player.pos[1] - player.dash_trail[-1][1]) > 0.15):
                player.dash_trail.append(
                    (player.pos[0], player.pos[1], self.sim_time + DASH_TRAIL_TTL))
        player.dash_trail = [pt for pt in player.dash_trail if pt[2] > self.sim_time]
        if len(player.dash_trail) > DASH_TRAIL_MAX:
            player.dash_trail = player.dash_trail[-DASH_TRAIL_MAX:]

        # --- Buffs ---
        decay_buffs(player, self.sim_time)

        # --- Pickup spawning ---
        self.pickup_timer -= dt
        if (self.pickup_timer <= 0.0
                and len(self.pickups) < gcfg.PICKUP_MAX_ACTIVE):
            self.pickups.append(make_pickup(player.pos, WORLD_HALF))
            self.pickup_timer = gcfg.PICKUP_SPAWN_INTERVAL

        # --- Pickup collection ---
        remaining: List[Pickup] = []
        for p in self.pickups:
            d = math.hypot(p.pos[0] - player.pos[0], p.pos[1] - player.pos[1])
            if d < (p.radius + 0.45):
                apply_pickup(p, player, self.sim_time)
            else:
                remaining.append(p)
        self.pickups = remaining

        self.obs_builder.tick_apm()

        # --- Enemies / boss spawning & stepping ---
        scaled_spawn = scale_spawn_rate(float(E[1].item()), run.room)
        scaled_speed = scale_enemy_speed(float(E[0].item()), run.room, run.enemy_slow)

        if self.state == GameState.PLAYING:
            self.spawn_accum += max(scaled_spawn, 0.05) * dt
            while self.spawn_accum >= 1.0 and len(self.enemies) < gcfg.MAX_ENEMIES:
                kind = pick_kind(pi)
                self.enemies.append(make_enemy(kind, player.pos, WORLD_HALF))
                self.spawn_accum -= 1.0
            step_enemies(self.enemies, player.pos, scaled_speed, 1.0, dt)
        else:  # BOSS
            if self.boss is not None:
                step_boss(self.boss, player.pos, pi, WORLD_HALF, self.enemies, dt)
            step_enemies(self.enemies, player.pos, scaled_speed, 1.0, dt)

        # --- Collisions: enemies ---
        dashing     = player.dashing > 0.0
        trail       = player.dash_trail
        trail_active = len(trail) >= 1

        survivors: List[Enemy] = []
        for e in self.enemies:
            d = math.hypot(e.pos[0] - player.pos[0], e.pos[1] - player.pos[1])
            if d < (e.radius + 0.35) and e.contact_cd <= 0.0:
                if dashing:
                    e.hp -= 99
                else:
                    e.hp -= 1
                    if player.invuln <= 0.0:
                        if "shield" in player.buffs:
                            player.hit_flash = 0.4
                            player.invuln = 0.25
                        else:
                            dmg = gcfg.DAMAGE_ON_HIT * run.damage_mult
                            player.hp -= dmg
                            player.invuln = gcfg.HIT_INVULN
                            player.hit_flash = 1.0
                e.contact_cd = e.contact_cd_max
            if e.hp > 0 and trail_active and self._killed_by_trail(
                    e.pos[0], e.pos[1], e.radius, trail):
                e.hp = 0
            if e.hp > 0:
                survivors.append(e)
            else:
                run.register_kill(pi)
        self.enemies = survivors

        # --- Collisions: boss ---
        if self.state == GameState.BOSS and self.boss is not None:
            d = math.hypot(self.boss.pos[0] - player.pos[0],
                           self.boss.pos[1] - player.pos[1])
            if d < (self.boss.radius + 0.45) and self.boss.contact_cd <= 0.0:
                if dashing:
                    self.boss.hp -= gcfg.BOSS_DASH_DAMAGE
                    self.boss.hit_flash = 1.0
                else:
                    if player.invuln <= 0.0:
                        if "shield" in player.buffs:
                            player.hit_flash = 0.4
                            player.invuln = 0.3
                        else:
                            dmg = gcfg.BOSS_CONTACT_DAMAGE * run.damage_mult
                            player.hp -= dmg
                            player.invuln = gcfg.HIT_INVULN
                            player.hit_flash = 1.0
                self.boss.contact_cd = 0.4
            if self.boss.hp <= 0:
                run.register_kill(pi)
                run.clear_room()
                self._start_next_room_or_shop()

        # --- Progression ---
        # Clamped at zero so the normalised time-remaining observation stays
        # in [0, 1] once a room drops into overtime.
        run.room_time_remaining = max(0.0, run.room_time_remaining - dt)
        run.tick(pi, dt)
        if self.state == GameState.PLAYING:
            if run.room_goal_met():
                run.clear_room()
                self._start_next_room_or_shop()
            else:
                # Timer expiry drops the room into overtime (halved clear
                # bonus) rather than killing the agent. Episodes now end only
                # on HP loss or MAX_EP_STEPS truncation.
                run.check_overtime()

        if player.hp <= 0.0:
            player.hp = 0.0

        # --- Psychological inference ---
        threat_list = self.enemies + ([self.boss] if self.boss is not None else [])
        o_t = self.obs_builder.build(
            player.vel[0], player.vel[1], threat_list, tuple(player.pos)
        ).to(self._device)
        self.pi = self._attn(o_t, (player.vel[0], player.vel[1]))
        self.E, self.c = self._mapper(self.pi)

    def _apply_dash(self, raw_ix: float, raw_iy: float) -> None:
        """Headless equivalent of abilities.try_dash."""
        player = self.player
        if player.dash_cd > 0.0:
            return
        ilen = math.hypot(raw_ix, raw_iy)
        if ilen > 0.1:
            dx, dy = raw_ix / ilen, raw_iy / ilen
        else:
            vx, vy = player.vel
            vlen = math.hypot(vx, vy)
            if vlen > 0.2:
                dx, dy = vx / vlen, vy / vlen
            else:
                dx, dy = player.facing
        cur_speed  = math.hypot(player.vel[0], player.vel[1])
        imp_mult   = 1.35 if "dash_boost" in player.buffs else 1.0
        speed      = max(cur_speed, 6.0) * gcfg.DASH_IMPULSE * 0.6 * imp_mult
        player.vel[0]   = dx * speed
        player.vel[1]   = dy * speed
        player.dash_dir = [dx, dy]
        player.facing   = [dx, dy]
        dash_dur        = gcfg.DASH_INVULN * (1.6 if "dash_boost" in player.buffs else 1.0)
        player.dash_cd  = player.dash_cd_max
        player.invuln   = max(player.invuln, dash_dur)
        player.dashing  = dash_dur

    @staticmethod
    def _killed_by_trail(ex: float, ey: float, er: float, trail: list) -> bool:
        for j in range(len(trail) - 1):
            ax_, ay_, _ = trail[j]
            bx_, by_, _ = trail[j + 1]
            pax, pay = ex - ax_, ey - ay_
            bax, bay = bx_ - ax_, by_ - ay_
            ab2 = bax * bax + bay * bay + 1e-9
            h   = max(0.0, min(1.0, (pax * bax + pay * bay) / ab2))
            cx, cy = ax_ + h * bax, ay_ + h * bay
            if math.hypot(ex - cx, ey - cy) < (DASH_TRAIL_RADIUS + er):
                return True
        ax_, ay_, _ = trail[-1]
        return math.hypot(ex - ax_, ey - ay_) < (DASH_TRAIL_RADIUS + er)

    def _start_next_room_or_shop(self) -> None:
        self.enemies     = []
        self.boss        = None
        self.spawn_accum = 0.0
        self.pickups     = []
        self.pickup_timer = gcfg.PICKUP_SPAWN_INTERVAL * 0.4
        if self.run.should_open_shop():
            self.shop_choices = pick_shop_choices(3)
            self.state = GameState.SHOP
        else:
            self.run.start_next_room()
            if self.run.is_boss_room:
                self.state = GameState.BOSS
                self.boss  = Boss()
                self.boss.pos = [0.0, 4.0]
            else:
                self.state = GameState.PLAYING

    # ------------------------------------------------------------------
    # Observation construction
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        idx = 0
        player  = self.player
        run     = self.run

        # --- Behavioral features (5) ---
        threat_list = self.enemies + ([self.boss] if self.boss is not None else [])
        o_t = self.obs_builder.build(
            player.vel[0], player.vel[1], threat_list, tuple(player.pos)
        )
        obs[idx:idx + 5] = o_t.numpy()
        idx += 5

        # --- Player state (8) ---
        obs[idx]     = player.pos[0] / WORLD_HALF
        obs[idx + 1] = player.pos[1] / WORLD_HALF
        obs[idx + 2] = player.vel[0] / 10.0
        obs[idx + 3] = player.vel[1] / 10.0
        obs[idx + 4] = player.hp / max(player.max_hp, 1.0)
        obs[idx + 5] = player.dash_cd / max(player.dash_cd_max, 1e-6)
        obs[idx + 6] = float(player.dashing > 0.0)
        obs[idx + 7] = float(player.invuln > 0.0)
        idx += 8

        # --- Room state (4) ---
        if run.room_time_total > 0:
            obs[idx] = run.room_time_remaining / run.room_time_total
        kills_left = max(0, run.room_kills_required - run.kills_in_room)
        obs[idx + 1] = kills_left / max(1, run.room_kills_required)
        obs[idx + 2] = min(run.room / 20.0, 1.0)
        obs[idx + 3] = float(run.is_boss_room)
        idx += 4

        # --- Current π (5) ---
        obs[idx:idx + 5] = self.pi.cpu().numpy()
        idx += 5

        # --- Target archetype one-hot (5) ---
        obs[idx + self.target_archetype] = 1.0
        idx += 5

        # --- Shop flag + offered upgrades one-hot (1 + 15 = 16) ---
        if self.state == GameState.SHOP:
            obs[idx] = 1.0
            for j, choice_idx in enumerate(self.shop_choices[:3]):
                obs[idx + 1 + j * N_UPGRADES + choice_idx] = 1.0
        idx += 16

        # --- Nearest 8 enemies (8 × 6 = 48) ---
        all_threats = (self.enemies + ([self.boss] if self.boss is not None else []))
        sorted_threats = sorted(
            all_threats,
            key=lambda e: math.hypot(e.pos[0] - player.pos[0],
                                      e.pos[1] - player.pos[1])
        )[:8]
        for i, e in enumerate(sorted_threats):
            dx = (e.pos[0] - player.pos[0]) / (2 * WORLD_HALF)
            dy = (e.pos[1] - player.pos[1]) / (2 * WORLD_HALF)
            kind = getattr(e, "kind", "boss")
            base = idx + i * 6
            obs[base]     = dx
            obs[base + 1] = dy
            obs[base + 2] = float(kind == "chaser")
            obs[base + 3] = float(kind == "fast")
            obs[base + 4] = float(kind == "tank")
            obs[base + 5] = e.hp / max(e.max_hp, 1)
        idx += 48

        # --- Nearest 4 pickups (4 × 7 = 28) ---
        sorted_pickups = sorted(
            self.pickups,
            key=lambda p: math.hypot(p.pos[0] - player.pos[0],
                                      p.pos[1] - player.pos[1])
        )[:4]
        for i, p in enumerate(sorted_pickups):
            dx = (p.pos[0] - player.pos[0]) / (2 * WORLD_HALF)
            dy = (p.pos[1] - player.pos[1]) / (2 * WORLD_HALF)
            base = idx + i * 7
            obs[base]     = dx
            obs[base + 1] = dy
            for j, ptype in enumerate(PICKUP_TYPES):
                obs[base + 2 + j] = float(p.kind == ptype)
        idx += 28

        assert idx == OBS_DIM, f"obs index mismatch: {idx} != {OBS_DIM}"
        return obs

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(self, score_delta: float, terminated: bool) -> float:
        from rl_agent.config import LAMBDA_ALIGN, SCORE_NORM, SURVIVAL_BONUS, DEATH_PENALTY
        alignment    = float(self.pi[self.target_archetype])
        score_r      = score_delta / SCORE_NORM
        align_r      = LAMBDA_ALIGN[self.target_archetype] * alignment
        survival_r   = SURVIVAL_BONUS
        death_r      = DEATH_PENALTY if terminated else 0.0
        return float(score_r + align_r + survival_r + death_r)

    # ------------------------------------------------------------------
    # Meta-policy observation
    # ------------------------------------------------------------------

    def get_meta_obs(self, score_delta_last_K: float) -> np.ndarray:
        """16-dim observation for the meta-policy."""
        from rl_agent.config import META_OBS_DIM
        obs = np.zeros(META_OBS_DIM, dtype=np.float32)
        obs[0:5]  = self.pi.cpu().numpy()
        obs[5]    = score_delta_last_K / 500.0
        obs[6]    = min(self.run.room / 20.0, 1.0)
        obs[7]    = (self.run.room_time_remaining / max(self.run.room_time_total, 1.0))
        obs[8]    = max(0, self.run.room_kills_required - self.run.kills_in_room) / max(1, self.run.room_kills_required)
        obs[9]    = self.player.hp / max(self.player.max_hp, 1.0)
        obs[10 + self.target_archetype] = 1.0   # prev target one-hot (5)
        return obs


# ---------------------------------------------------------------------------
# Smoke-test entry point
# ---------------------------------------------------------------------------

def _smoke_test(n_steps: int = 200) -> None:
    env = FlowGameEnv(seed=42)
    obs = env.reset()
    print(f"obs shape: {obs.shape}, dtype: {obs.dtype}")
    total_reward = 0.0
    for i in range(n_steps):
        action = np.array([
            random.randint(0, N_MOVE_ACTIONS - 1),
            random.randint(0, N_DASH_ACTIONS - 1),
            random.randint(0, N_SHOP_ACTIONS - 1),
        ], dtype=np.int32)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            obs = env.reset()
    print(f"Smoke test OK: {n_steps} steps, total_reward={total_reward:.3f}")
    print(f"Final info: {info}")


if __name__ == "__main__":
    _smoke_test()
