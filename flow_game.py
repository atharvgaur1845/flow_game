"""Flow Game — Non-Causal Psychological Adaptation Engine + Roguelite.

Backend (psychological-state inference) is in this file and is NEVER
modified by the gameplay layer:
    * AttentionStateInference
    * EnvironmentMapper
    * ObservationBuilder

The roguelite game (rooms, shop, boss, UI, enemy variety) lives in the
`game/` package and only READS pi, E, c from the backend.

Run:
    python flow_game.py               # fullscreen game (default)
    python flow_game.py --windowed    # windowed
    python flow_game.py --probe       # headless invariant check (no window)
"""
from __future__ import annotations

import argparse
import collections
import ctypes
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Deque, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from game import abilities
from game import config as gcfg
from game import entities_extra as eex
from game import progression as prog
from game.entities_extra import (
    Boss, Enemy, Pickup, PICKUP_BUFF_DURATION, PICKUP_CODE, PICKUP_KINDS,
    PICKUP_LABEL, make_enemy, make_pickup, pick_kind, step_boss, step_enemies,
)
from game.game_state import GameState
from game.progression import (
    UPGRADES, RunState, apply_upgrade, pick_shop_choices,
    scale_enemy_speed, scale_spawn_rate,
)

# Skip pygame/OpenGL UI imports when running headless (e.g. RL training).
_HEADLESS = os.environ.get("FLOW_HEADLESS", "0") == "1"
if not _HEADLESS:
    from game.ui import STATE_COLORS, STATE_NAMES, UIOverlay

# ----------------------------------------------------------------------------
# Hyperparameters / symbol table (backend)
# ----------------------------------------------------------------------------
D = 5
K = 5
P = 4
V_LATENT = 4
D_K = 16
W_HIST = 16
W_FUT = 16
W_TOTAL = W_HIST + W_FUT
EMA_ALPHA = 0.02
APM_DECAY = 0.95
MAX_ENEMIES = gcfg.MAX_ENEMIES
WORLD_HALF = 16.0
DASH_TRAIL_TTL = 1.6          # seconds the kill-trail lingers after dash
DASH_TRAIL_RADIUS = 0.55      # collision radius for trail segments
DASH_TRAIL_MAX = 32           # uniform-array cap, must match shader

# Pickup buff effect magnitudes (mirrors entities_extra constants).
SPEED_BOOST_MULT = 1.4
MAX_HP_BONUS     = 25
HEAL_AMOUNT      = 30


def apply_pickup(p, player, now_t: float) -> str:
    """Apply a Pickup to the player; return the on-screen label."""
    k = p.kind
    if k == "heal":
        player.hp = min(player.max_hp, player.hp + HEAL_AMOUNT)
    elif k == "max_hp":
        if "max_hp" not in player.buffs:
            player.max_hp += MAX_HP_BONUS
            player.hp += MAX_HP_BONUS
        player.buffs["max_hp"] = now_t + PICKUP_BUFF_DURATION["max_hp"]
    else:  # dash_boost / speed_boost / shield
        player.buffs[k] = now_t + PICKUP_BUFF_DURATION[k]
    return PICKUP_LABEL[k]


def decay_buffs(player, now_t: float) -> None:
    """Remove expired buffs; revert reversible effects."""
    expired = [k for k, exp in player.buffs.items() if exp <= now_t]
    for k in expired:
        if k == "max_hp":
            player.max_hp = max(1.0, player.max_hp - MAX_HP_BONUS)
            player.hp = min(player.hp, player.max_hp)
        del player.buffs[k]

ARCH_AROUSAL, ARCH_TACTICAL, ARCH_OVERLOAD, ARCH_FLOW, ARCH_APATHY = 0, 1, 2, 3, 4

# Auto-select GPU when available. Backend tensors are tiny, but we honour
# the request: move modules + per-frame inputs to `DEVICE`.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# GLSL Shaders
# ============================================================================
VERTEX_SHADER = """
#version 330 core
layout(location=0) in vec2 a_pos;
out vec2 v_uv;
void main() {
    v_uv = a_pos * 0.5 + 0.5;
    gl_Position = vec4(a_pos, 0.0, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330 core
in  vec2 v_uv;
out vec4 fragColor;

uniform float u_time;
uniform vec2  u_resolution;
uniform vec2  u_player;
uniform vec2  u_enemies[16];
uniform float u_enemy_radii[16];
uniform int   u_enemy_count;
uniform float u_world_half;
uniform float u_cam_zoom;
uniform float u_c1_aberration;
uniform float u_c2_grain;
uniform float u_c3_warp;
uniform float u_c4_bloom;
uniform vec4  u_pi_top4;

// Boss
uniform float u_boss_active;
uniform vec2  u_boss_pos;
uniform float u_boss_radius;
uniform float u_boss_hp_frac;

// Feedback
uniform vec2  u_shake;
uniform float u_player_flash;   // dash i-frames
uniform float u_hit_flash;      // took damage
uniform float u_dash_phase;     // 0 idle .. 1 dash start
uniform vec2  u_dash_dir;       // unit vector of current/last dash
uniform vec2  u_dash_trail[32]; // lethal dash-trail waypoints (world)
uniform float u_dash_trail_a[32]; // per-point alpha (1 = fresh, 0 = expired)
uniform int   u_dash_trail_count;
uniform vec2  u_pickups[8];     // food / power-up positions (world)
uniform int   u_pickup_kinds[8]; // 0 heal,1 dash,2 speed,3 shield,4 max_hp
uniform int   u_pickup_count;

float sdCircle(vec2 p, float r) { return length(p) - r; }
float hash21(vec2 p) {
    return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

vec2 screenToWorld(vec2 uv) {
    float aspect = u_resolution.x / u_resolution.y;
    vec2 ndc = uv * 2.0 - 1.0;
    ndc.x *= aspect;
    return (ndc * u_world_half) / u_cam_zoom;
}

// SDF for a 2D line segment.
float sdSegment(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a;
    vec2 ba = b - a;
    float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-6), 0.0, 1.0);
    return length(pa - ba * h);
}

// SDF for a 2D rotated diamond (square rotated 45 deg) of radius r.
float sdDiamond(vec2 p, float r) {
    return (abs(p.x) + abs(p.y)) - r;
}

vec3 pickupColor(int kind) {
    if (kind == 0) return vec3(0.30, 1.00, 0.45); // heal       — green
    if (kind == 1) return vec3(0.30, 0.95, 1.00); // dash_boost — cyan
    if (kind == 2) return vec3(1.00, 0.65, 0.20); // speed      — orange
    if (kind == 3) return vec3(0.40, 0.55, 1.00); // shield     — blue
    return                vec3(1.00, 0.95, 0.30); // max_hp     — yellow
}

vec3 sampleScene(vec2 world) {
    vec2 g = world * 0.5;
    float grid = 0.0;
    grid += smoothstep(0.04, 0.0, abs(fract(g.x + u_time * 0.05) - 0.5) - 0.48);
    grid += smoothstep(0.04, 0.0, abs(fract(g.y - u_time * 0.03) - 0.5) - 0.48);
    // Background tint driven by the inferred psychological mixture.
    // Each archetype contributes its own aesthetic palette; the dominant
    // state shifts the entire arena's mood without overpowering the scene.
    vec3 bg_tint = vec3(0.55, 0.12, 0.18) * u_pi_top4.x   // arousal: ember red
                 + vec3(0.08, 0.22, 0.55) * u_pi_top4.y   // tactical: deep blue
                 + vec3(0.42, 0.08, 0.55) * u_pi_top4.z   // overload: violet
                 + vec3(0.05, 0.45, 0.40) * u_pi_top4.w;  // flow: teal
    vec3 bg_low  = vec3(0.02, 0.02, 0.05) + bg_tint * 0.18;
    vec3 bg_high = vec3(0.04, 0.06, 0.12) + bg_tint * 0.55;
    vec3 bg = mix(bg_low, bg_high,
                  smoothstep(u_world_half, 0.0, length(world)));
    vec3 grid_col = mix(vec3(0.08, 0.15, 0.35), bg_tint * 1.6, 0.45);
    vec3 col = bg + grid_col * grid;

    // Arena boundary: square box of half-size u_world_half, drawn as a
    // glowing outline. Anything outside is heavily dimmed so the play
    // area is unmistakable.
    float bx = abs(world.x) - u_world_half;
    float by = abs(world.y) - u_world_half;
    float boundary_sdf = max(bx, by);
    float outline = abs(boundary_sdf) - 0.10;
    float outline_mask = smoothstep(0.08, 0.0, outline);
    vec3 boundary_color = vec3(0.30, 0.85, 1.0)
                        * (0.85 + 0.25 * sin(u_time * 2.5));
    col += boundary_color * outline_mask * 1.4;
    if (boundary_sdf > 0.0) {
        col *= 0.18; // outside the arena: nearly black wash
        col += vec3(0.05, 0.02, 0.08);
    }

    // Lethal dash trail (additive glow on the trail segments).
    for (int i = 0; i < 32; ++i) {
        if (i >= u_dash_trail_count) break;
        float a = u_dash_trail_a[i];
        if (a <= 0.0) continue;
        vec2 tp = u_dash_trail[i];
        float td;
        if (i + 1 < u_dash_trail_count) {
            td = sdSegment(world, tp, u_dash_trail[i + 1]);
        } else {
            td = length(world - tp);
        }
        float core = smoothstep(0.10, -0.04, td - 0.18);
        float glow = exp(-max(td - 0.18, 0.0) * 3.0);
        col += vec3(1.0, 0.85, 1.0) * core * a * 0.85;
        col += vec3(0.7, 0.3, 1.0) * glow * a * 0.35;
    }

    // Pickups: rotating diamond with a soft halo, colour-coded by kind.
    for (int i = 0; i < 8; ++i) {
        if (i >= u_pickup_count) break;
        vec2 pp = u_pickups[i];
        vec3 pc = pickupColor(u_pickup_kinds[i]);
        // Rotate UVs around the pickup centre.
        float t = u_time * 2.2 + float(i) * 0.7;
        float ca = cos(t), sa = sin(t);
        vec2 lp = world - pp;
        vec2 rp = vec2(ca * lp.x - sa * lp.y,
                       sa * lp.x + ca * lp.y);
        float pulse = 0.42 + 0.06 * sin(u_time * 5.0 + float(i));
        float dpkp  = sdDiamond(rp, pulse);
        float coreP = smoothstep(0.04, -0.03, dpkp);
        float glowP = exp(-max(dpkp, 0.0) * 3.5);
        col += pc * coreP * 1.15;
        col += pc * glowP * 0.55;
    }

    // Player.
    float dp = sdCircle(world - u_player, 0.45);
    float player_core = smoothstep(0.02, -0.02, dp);
    float player_glow = exp(-max(dp, 0.0) * 4.0) * (0.6 + 0.8 * u_c4_bloom);
    vec3 player_tint = mix(vec3(0.2, 0.9, 1.0),
                           vec3(1.0, 1.0, 1.0),
                           u_player_flash);
    col += player_tint * player_core * (1.0 + 0.6 * u_dash_phase);
    col += vec3(0.1, 0.6, 1.0) * player_glow * 0.45;

    // --- Trippy "killer" dash effect (purely cosmetic, no hitbox change).
    if (u_dash_phase > 0.0) {
        // 1. Echo trail behind the dash direction.
        for (int i = 1; i <= 4; ++i) {
            float t = float(i) / 4.0;
            vec2 echo_center = u_player - u_dash_dir * t * 0.9;
            float de = sdCircle(world - echo_center, 0.45 * (1.0 - 0.18 * t));
            float ae = (1.0 - t) * u_dash_phase;
            float emask = smoothstep(0.04, -0.02, de);
            col += vec3(0.6, 1.0, 1.0) * emask * ae * 0.55;
        }
        // 2. 4-arm spinning blade SDF around the player.
        vec2 lp = world - u_player;
        float ang = u_time * 14.0;
        float blade = 1e9;
        for (int k = 0; k < 4; ++k) {
            float a = ang + float(k) * 1.5707963;
            vec2 r = vec2(cos(a), sin(a));
            vec2 q = vec2(dot(lp, r), dot(lp, vec2(-r.y, r.x)));
            float d = abs(q.y) - 0.04;
            d = max(d, q.x - 0.95);
            d = max(d, -q.x);
            blade = min(blade, d);
        }
        float blade_mask = smoothstep(0.02, -0.01, blade) * u_dash_phase;
        vec3 blade_tint = mix(vec3(0.4, 1.0, 1.0), vec3(1.0, 0.4, 1.0),
                              0.5 + 0.5 * sin(u_time * 9.0));
        col += blade_tint * blade_mask * 1.4;
        // 3. Outer ripple ring that pulses outward during dash.
        float ring_r = 0.55 + (1.0 - u_dash_phase) * 0.6;
        float ring = abs(length(lp) - ring_r) - 0.03;
        col += vec3(1.0, 0.8, 1.0)
               * smoothstep(0.04, 0.0, ring) * u_dash_phase * 0.6;
    }

    // Enemies.
    float pulse = 0.85 + 0.15 * sin(u_time * 4.0);
    for (int i = 0; i < 16; ++i) {
        if (i >= u_enemy_count) break;
        vec2 ep = u_enemies[i];
        float er = u_enemy_radii[i] * pulse;
        float de = sdCircle(world - ep, er);
        float ec = smoothstep(0.02, -0.02, de);
        float eg = exp(-max(de, 0.0) * 5.0) * (0.5 + 0.9 * u_c4_bloom);
        col += vec3(1.0, 0.25, 0.7) * ec;
        col += vec3(0.9, 0.15, 0.5) * eg * 0.35;
    }

    // Boss.
    if (u_boss_active > 0.5) {
        float br = u_boss_radius * (0.95 + 0.05 * sin(u_time * 2.0));
        float db = sdCircle(world - u_boss_pos, br);
        float bc = smoothstep(0.04, -0.04, db);
        float bg_ = exp(-max(db, 0.0) * 1.8) * (0.9 + 1.2 * u_c4_bloom);
        vec3 boss_base = mix(vec3(0.4, 0.05, 0.1), vec3(1.0, 0.6, 0.1),
                             u_boss_hp_frac);
        col += boss_base * bc;
        col += boss_base * bg_ * 0.45;
        // HP ring.
        float ring_d = abs(length(world - u_boss_pos) - (u_boss_radius + 0.35));
        float ring_mask = smoothstep(0.04, 0.0, ring_d);
        float theta = atan(world.y - u_boss_pos.y, world.x - u_boss_pos.x);
        float a = (theta + 3.14159) / 6.28318;
        float ring = ring_mask * step(a, u_boss_hp_frac);
        col += vec3(1.0, 0.7, 0.2) * ring * 0.9;
    }

    vec3 tint = vec3(0.6, 0.2, 0.3) * u_pi_top4.x
              + vec3(0.15, 0.3, 0.5) * u_pi_top4.y
              + vec3(0.4, 0.1, 0.6) * u_pi_top4.z
              + vec3(0.1, 0.5, 0.4) * u_pi_top4.w;
    col += tint * 0.04;
    return col;
}

void main() {
    // Shake offset applied in screen space.
    vec2 uv = v_uv + u_shake;

    vec2 warped = uv;
    warped.x += sin(uv.y * 10.0 + u_time * 1.3) * u_c3_warp;
    warped.y += cos(uv.x *  8.0 + u_time * 0.9) * u_c3_warp;

    vec2 dir = normalize(warped - 0.5 + 1e-6);
    float r_off = u_c1_aberration;
    vec2 wr = screenToWorld(warped + dir * r_off);
    vec2 wg = screenToWorld(warped);
    vec2 wb = screenToWorld(warped - dir * r_off);
    float r = sampleScene(wr).r;
    float g = sampleScene(wg).g;
    float b = sampleScene(wb).b;
    vec3 col = vec3(r, g, b);

    float n = hash21(warped * u_resolution + u_time * 60.0);
    col += (n - 0.5) * u_c2_grain;
    col += pow(max(col, 0.0), vec3(2.0)) * u_c4_bloom * 0.25;

    // Hit flash: saturate toward red and lift exposure.
    col = mix(col, vec3(1.0, 0.2, 0.2), u_hit_flash * 0.5);

    fragColor = vec4(col, 1.0);
}
"""


# ============================================================================
# Backend: PyTorch modules (UNTOUCHED per spec)
# ============================================================================
def _seeded_(tensor: torch.Tensor, scale: float = 0.3) -> torch.Tensor:
    with torch.no_grad():
        tensor.uniform_(-scale, scale)
    return tensor


class AttentionStateInference(nn.Module):
    """Non-causal self-attention over [history | future-projection] window."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(1337)
        self.W_Q = nn.Parameter(_seeded_(torch.empty(D, D_K)))
        self.W_K = nn.Parameter(_seeded_(torch.empty(D, D_K)))
        self.W_V = nn.Parameter(_seeded_(torch.empty(D, D_K)))
        W_pi = torch.zeros(K, D_K)
        for k in range(K):
            W_pi[k, k] = 1.2
            W_pi[k, (k + 1) % D_K] = 0.5
            W_pi[k, (k + 2) % D_K] = -0.3
        W_pi += 0.05 * torch.randn(K, D_K)
        self.W_pi = nn.Parameter(W_pi)
        self.b_pi = nn.Parameter(torch.zeros(K))
        self.register_buffer("history", torch.zeros(W_HIST, D))
        self.register_buffer("ema_mu", torch.zeros(D))
        self.register_buffer("_hist_filled", torch.zeros(1))
        self._scale = 1.0 / math.sqrt(D_K)

    @torch.no_grad()
    def project_future(self, last_norm_obs: torch.Tensor,
                       velocity_xy: Tuple[float, float]) -> torch.Tensor:
        fut = torch.zeros(W_FUT, D, device=last_norm_obs.device)
        vx, vy = velocity_xy
        v_norm_now = math.hypot(vx, vy)
        decay_v = 0.98
        decay_o = 0.90
        for i in range(W_FUT):
            v_future = v_norm_now * (decay_v ** (i + 1))
            f = last_norm_obs.clone()
            f[0] = v_future - self.ema_mu[0]
            f[1] = last_norm_obs[1] * (decay_o ** (i + 1))
            f[2] = last_norm_obs[2] * (decay_o ** (i + 1))
            f[3] = last_norm_obs[3] * (decay_o ** (i + 1))
            eps = 0.05
            if v_future < eps:
                f[4] = last_norm_obs[4] + (i + 1) * 0.01
            else:
                f[4] = max(last_norm_obs[4] - (i + 1) * 0.02, -0.5)
            fut[i] = f
        return fut

    @torch.no_grad()
    def forward(self, o_t: torch.Tensor,
                velocity_xy: Tuple[float, float]) -> torch.Tensor:
        self.ema_mu.mul_(1 - EMA_ALPHA).add_(o_t, alpha=EMA_ALPHA)
        o_norm = o_t - self.ema_mu
        self.history[:-1] = self.history[1:].clone()
        self.history[-1] = o_norm
        self._hist_filled[0] = min(float(self._hist_filled[0] + 1), W_HIST)
        future = self.project_future(o_norm, velocity_xy)
        O = torch.cat([self.history, future], dim=0)
        Q = O @ self.W_Q
        Kt = O @ self.W_K
        Vt = O @ self.W_V
        A = torch.softmax((Q @ Kt.T) * self._scale, dim=-1)
        Z = A @ Vt
        z_t = Z.mean(dim=0)
        logits = self.W_pi @ z_t + self.b_pi
        return torch.softmax(logits, dim=0)


class EnvironmentMapper(nn.Module):
    """Maps pi_t -> (E_t physics params, c_t disentangled visual latents)."""

    def __init__(self) -> None:
        super().__init__()
        M_values = [
            [1.8, 0.6, 1.4, 1.2, 0.3],
            [1.5, 0.4, 1.8, 1.0, 0.2],
            [0.3, 0.9, 0.7, 0.2, 1.0],
            [1.10, 0.95, 0.85, 1.20, 0.90],
        ]
        self.register_buffer("M", torch.tensor(M_values, dtype=torch.float32))
        W_c1 = torch.zeros(8, K)
        W_c1[0, ARCH_AROUSAL]  = 1.0
        W_c1[1, ARCH_OVERLOAD] = 1.0
        W_c1[2, ARCH_TACTICAL] = 1.0
        W_c1[3, ARCH_APATHY]   = 1.0
        W_c1[4, ARCH_FLOW]     = 1.2
        W_c1[5, ARCH_FLOW]     = 0.9
        W_c1[6, ARCH_AROUSAL]  = 0.8
        W_c1[7, ARCH_FLOW]     = 0.7
        self.register_buffer("W_c1", W_c1)
        W_c2 = torch.zeros(V_LATENT, 8)
        W_c2[0, 0] = 0.018; W_c2[0, 1] = 0.014
        W_c2[1, 2] = 0.35;  W_c2[1, 3] = 0.30
        W_c2[2, 4] = 0.020
        W_c2[3, 5] = 0.9;   W_c2[3, 6] = 0.35
        self.register_buffer("W_c2", W_c2)
        self.register_buffer("noise_std", torch.tensor(0.02))

    @torch.no_grad()
    def forward(self, pi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        eps = torch.randn(P, device=pi.device) * self.noise_std
        E_t = self.M @ pi + eps
        hidden = F.relu(self.W_c1 @ pi)
        c_t = self.W_c2 @ hidden
        return E_t, c_t


class ObservationBuilder:
    """Computes o_t = [v_norm, apm_proxy, dir_variance, threat, idle_time]."""

    def __init__(self) -> None:
        self.apm: float = 0.0
        self.idle_frames: float = 0.0
        self.headings: Deque[float] = collections.deque(maxlen=30)

    def register_keypress(self) -> None:
        self.apm = self.apm * APM_DECAY + 1.0

    def tick_apm(self) -> None:
        self.apm *= APM_DECAY

    def build(self, vx: float, vy: float, enemies: List,
              player_pos: Tuple[float, float]) -> torch.Tensor:
        v_norm = math.hypot(vx, vy)
        if v_norm > 1e-3:
            self.headings.append(math.atan2(vy, vx))
        if len(self.headings) >= 2:
            C = sum(math.cos(h) for h in self.headings) / len(self.headings)
            S = sum(math.sin(h) for h in self.headings) / len(self.headings)
            R = math.hypot(C, S)
            dir_var = 1.0 - R
        else:
            dir_var = 0.0
        if enemies:
            px, py = player_pos
            min_d = min(math.hypot(e.pos[0] - px, e.pos[1] - py)
                        for e in enemies)
        else:
            min_d = 100.0
        threat = 1.0 / (1.0 + min_d)
        if v_norm < 0.05:
            self.idle_frames = min(self.idle_frames + 1.0, 120.0)
        else:
            self.idle_frames = max(self.idle_frames - 2.0, 0.0)
        return torch.tensor([v_norm, self.apm, dir_var, threat,
                             self.idle_frames / 120.0], dtype=torch.float32)


# ============================================================================
# Gameplay-side Player (extended per roguelite spec)
# ============================================================================
@dataclass
class Player:
    pos: List[float]        = field(default_factory=lambda: [0.0, 0.0])
    vel: List[float]        = field(default_factory=lambda: [0.0, 0.0])
    facing: List[float]     = field(default_factory=lambda: [1.0, 0.0])
    hp: float               = float(gcfg.PLAYER_MAX_HP)
    max_hp: float           = float(gcfg.PLAYER_MAX_HP)
    dash_cd: float          = 0.0
    dash_cd_max: float      = gcfg.DASH_COOLDOWN
    invuln: float           = 0.0
    dashing: float          = 0.0
    hit_flash: float        = 0.0
    input_dir: List[float]  = field(default_factory=lambda: [0.0, 0.0])
    dash_dir: List[float]   = field(default_factory=lambda: [1.0, 0.0])
    # Lethal dash-trail waypoints: each entry is (x, y, expires_at_seconds).
    # Enemies whose centre falls within DASH_TRAIL_RADIUS of any consecutive
    # segment are instantly killed.
    dash_trail: list        = field(default_factory=list)
    # Active power-up buffs: kind -> expires_at_seconds.
    buffs: dict             = field(default_factory=dict)


# ============================================================================
# Probe (headless, no display)
# ============================================================================
def run_probe(n_frames: int = 300) -> int:
    attn = AttentionStateInference().to(DEVICE)
    mapper = EnvironmentMapper().to(DEVICE)
    obs = ObservationBuilder()
    player = Player()
    enemies = [make_enemy("chaser", player.pos, WORLD_HALF) for _ in range(3)]

    rng = random.Random(0)
    pi = E = c = None
    for frame in range(n_frames):
        vx = math.sin(frame * 0.1) * 2.0
        vy = math.cos(frame * 0.13) * 2.0
        player.vel = [vx, vy]
        player.pos[0] += vx * 0.016
        player.pos[1] += vy * 0.016
        if rng.random() < 0.3:
            obs.register_keypress()
        obs.tick_apm()
        o_t = obs.build(vx, vy, enemies, tuple(player.pos)).to(DEVICE)
        pi = attn(o_t, (vx, vy))
        E, c = mapper(pi)

        assert pi.shape == (K,)
        s = float(pi.sum())
        assert abs(s - 1.0) < 1e-4, f"pi sum not 1: {s}"
        assert float(pi.min()) >= -1e-6
        assert float(pi.max()) <= 1.0 + 1e-6
        assert E.shape == (P,)
        assert c.shape == (V_LATENT,)

    print(f"[probe] {n_frames} frames OK on device={DEVICE}. "
          f"pi={pi.tolist()} E={E.tolist()} c={c.tolist()}")
    return 0


# ============================================================================
# Shader compile / FS quad helpers
# ============================================================================
def compile_shader_program(vertex_src: str, fragment_src: str):
    from OpenGL.GL import shaders as gl_shaders
    from OpenGL.GL import GL_VERTEX_SHADER, GL_FRAGMENT_SHADER
    vs = gl_shaders.compileShader(vertex_src, GL_VERTEX_SHADER)
    fs = gl_shaders.compileShader(fragment_src, GL_FRAGMENT_SHADER)
    return gl_shaders.compileProgram(vs, fs)


# ============================================================================
# Game (interactive)
# ============================================================================
def run_game(windowed: bool = False) -> int:
    import pygame
    from OpenGL.GL import (
        glGenVertexArrays, glBindVertexArray,
        glGenBuffers, glBindBuffer, glBufferData,
        glEnableVertexAttribArray, glVertexAttribPointer,
        glUseProgram, glGetUniformLocation,
        glUniform1f, glUniform1i, glUniform2f, glUniform2fv, glUniform1fv,
        glUniform1iv,
        glUniform4f, glClear, glClearColor, glDrawArrays, glViewport,
        GL_ARRAY_BUFFER, GL_STATIC_DRAW, GL_FLOAT, GL_FALSE,
        GL_COLOR_BUFFER_BIT, GL_TRIANGLE_STRIP,
    )

    pygame.init()
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(
        pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)

    if gcfg.FULLSCREEN and not windowed:
        info = pygame.display.Info()
        width, height = info.current_w, info.current_h
        flags = pygame.OPENGL | pygame.DOUBLEBUF | pygame.FULLSCREEN
        pygame.display.set_mode((width, height), flags)
    else:
        width, height = gcfg.WINDOWED_SIZE
        flags = pygame.OPENGL | pygame.DOUBLEBUF
        pygame.display.set_mode((width, height), flags)
    pygame.display.set_caption("Flow — Neon Arena")
    pygame.mouse.set_visible(False)

    program = compile_shader_program(VERTEX_SHADER, FRAGMENT_SHADER)
    glUseProgram(program)

    quad = np.array([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
                    dtype=np.float32)
    vao = glGenVertexArrays(1); glBindVertexArray(vao)
    vbo = glGenBuffers(1); glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, quad.nbytes, quad, GL_STATIC_DRAW)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))

    U = {name: glGetUniformLocation(program, name) for name in (
        "u_time", "u_resolution", "u_player", "u_enemies", "u_enemy_radii",
        "u_enemy_count", "u_world_half", "u_cam_zoom",
        "u_c1_aberration", "u_c2_grain", "u_c3_warp", "u_c4_bloom",
        "u_pi_top4",
        "u_boss_active", "u_boss_pos", "u_boss_radius", "u_boss_hp_frac",
        "u_shake", "u_player_flash", "u_hit_flash",
        "u_dash_phase", "u_dash_dir",
        "u_dash_trail", "u_dash_trail_a", "u_dash_trail_count",
        "u_pickups", "u_pickup_kinds", "u_pickup_count",
    )}

    ui = UIOverlay(width, height)

    # Backend.
    attn = AttentionStateInference().to(DEVICE)
    mapper = EnvironmentMapper().to(DEVICE)
    obs = ObservationBuilder()

    # Gameplay state.
    state = GameState.MENU
    player = Player()
    enemies: List[Enemy] = []
    boss: Boss | None = None
    run = RunState()
    shop_choices: List[int] = []
    best_score = 0
    pi = torch.full((K,), 1.0 / K, device=DEVICE)
    E = torch.tensor([1.0, 0.5, 0.5, 1.0], device=DEVICE)
    c = torch.zeros(V_LATENT, device=DEVICE)
    spawn_accum = 0.0
    pickups: List[Pickup] = []
    pickup_timer = gcfg.PICKUP_SPAWN_INTERVAL * 0.4   # first one comes faster
    pickup_msg: Tuple[str, float] = ("", 0.0)        # (label, expires_at)

    def reset_run() -> None:
        nonlocal player, enemies, boss, run, spawn_accum, pickups, pickup_timer, pickup_msg
        player = Player()
        enemies = []
        boss = None
        run = RunState()
        spawn_accum = 0.0
        pickups = []
        pickup_timer = gcfg.PICKUP_SPAWN_INTERVAL * 0.4
        pickup_msg = ("", 0.0)
        run.start_next_room()

    def enter_shop() -> None:
        nonlocal shop_choices
        shop_choices = pick_shop_choices(3)

    def start_next_room_or_shop() -> None:
        nonlocal enemies, boss, spawn_accum, state, pickups, pickup_timer
        enemies = []
        boss = None
        spawn_accum = 0.0
        pickups = []
        pickup_timer = gcfg.PICKUP_SPAWN_INTERVAL * 0.4
        if run.should_open_shop():
            enter_shop()
            state = GameState.SHOP
        else:
            run.start_next_room()
            state = GameState.BOSS if run.is_boss_room else GameState.PLAYING
            if run.is_boss_room:
                nonlocal_boss()

    def nonlocal_boss() -> None:
        nonlocal boss
        boss = Boss()
        boss.pos = [0.0, 4.0]

    clock = pygame.time.Clock()
    t_start = time.time()
    debug = False
    running = True

    while running:
        dt = clock.tick(144) / 1000.0
        dt = min(dt, 0.05)

        # -----------------------------------------------------------------
        # Event handling — branching by state
        # -----------------------------------------------------------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_F1:
                    debug = not debug

                if state == GameState.MENU:
                    if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER,
                                     pygame.K_SPACE):
                        reset_run()
                        state = GameState.PLAYING
                elif state == GameState.SHOP:
                    if event.key in (pygame.K_1, pygame.K_KP1):
                        apply_upgrade(shop_choices[0], player, run)
                        run.start_next_room()
                        state = GameState.BOSS if run.is_boss_room else GameState.PLAYING
                        if run.is_boss_room: nonlocal_boss()
                    elif event.key in (pygame.K_2, pygame.K_KP2):
                        apply_upgrade(shop_choices[1], player, run)
                        run.start_next_room()
                        state = GameState.BOSS if run.is_boss_room else GameState.PLAYING
                        if run.is_boss_room: nonlocal_boss()
                    elif event.key in (pygame.K_3, pygame.K_KP3):
                        apply_upgrade(shop_choices[2], player, run)
                        run.start_next_room()
                        state = GameState.BOSS if run.is_boss_room else GameState.PLAYING
                        if run.is_boss_room: nonlocal_boss()
                elif state == GameState.GAME_OVER:
                    if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        state = GameState.MENU

                if state in (GameState.PLAYING, GameState.BOSS) and event.key in (
                    pygame.K_w, pygame.K_a, pygame.K_s, pygame.K_d, pygame.K_SPACE
                ):
                    obs.register_keypress()

        keys = pygame.key.get_pressed()

        # -----------------------------------------------------------------
        # State-specific updates
        # -----------------------------------------------------------------
        if state in (GameState.PLAYING, GameState.BOSS):
            # --- Input + physics -----------------------------------------
            # Read raw WASD as a vector and normalize so diagonal isn't faster.
            raw_ix = float(keys[pygame.K_d] - keys[pygame.K_a])
            raw_iy = float(keys[pygame.K_w] - keys[pygame.K_s])
            ilen = math.hypot(raw_ix, raw_iy)
            if ilen > 1e-6:
                tgt_ix, tgt_iy = raw_ix / ilen, raw_iy / ilen
                player.facing = [tgt_ix, tgt_iy]
            else:
                tgt_ix = tgt_iy = 0.0
            # Critically-damped lerp toward the target input direction so brief
            # key flicker is smoothed but the response stays snappy (~3 frames).
            lerp = min(1.0, dt * 18.0)
            player.input_dir[0] += (tgt_ix - player.input_dir[0]) * lerp
            player.input_dir[1] += (tgt_iy - player.input_dir[1]) * lerp
            ix, iy = player.input_dir
            speed_mult = SPEED_BOOST_MULT if "speed_boost" in player.buffs else 1.0
            ax = ix * gcfg.PLAYER_ACCEL * speed_mult
            ay = iy * gcfg.PLAYER_ACCEL * speed_mult

            abilities.try_dash(player, keys, pygame, (raw_ix, raw_iy))

            friction = float(max(E[2].item(), 0.05))
            player.vel[0] += ax * dt
            player.vel[1] += ay * dt
            damp = math.exp(-friction * dt)
            player.vel[0] *= damp
            player.vel[1] *= damp
            # Boundary bounce (stable): clamp position to the wall and
            # invert the *outward* velocity component once. We only flip
            # the sign when the player is actually moving outward, which
            # prevents the double-reflection jitter that happens when a
            # high-velocity dash crosses the line on consecutive frames.
            BOUNCE_DAMP = 0.55
            new_x = player.pos[0] + player.vel[0] * dt
            new_y = player.pos[1] + player.vel[1] * dt
            if new_x >  WORLD_HALF:
                new_x = WORLD_HALF
                if player.vel[0] > 0.0:
                    player.vel[0] = -player.vel[0] * BOUNCE_DAMP
            elif new_x < -WORLD_HALF:
                new_x = -WORLD_HALF
                if player.vel[0] < 0.0:
                    player.vel[0] = -player.vel[0] * BOUNCE_DAMP
            if new_y >  WORLD_HALF:
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

            # --- Lethal dash trail bookkeeping --------------------------
            now_t = time.time() - t_start
            if player.dashing > 0.0:
                # Append a waypoint roughly every dt; dedupe if barely moved.
                if (not player.dash_trail
                    or math.hypot(player.pos[0] - player.dash_trail[-1][0],
                                  player.pos[1] - player.dash_trail[-1][1]) > 0.15):
                    player.dash_trail.append(
                        (player.pos[0], player.pos[1], now_t + DASH_TRAIL_TTL))
            # Prune expired and cap length.
            player.dash_trail = [pt for pt in player.dash_trail if pt[2] > now_t]
            if len(player.dash_trail) > DASH_TRAIL_MAX:
                player.dash_trail = player.dash_trail[-DASH_TRAIL_MAX:]

            # --- Buffs (decay expired) -----------------------------------
            decay_buffs(player, now_t)

            # --- Pickup spawning -----------------------------------------
            pickup_timer -= dt
            if (pickup_timer <= 0.0
                    and len(pickups) < gcfg.PICKUP_MAX_ACTIVE):
                pickups.append(make_pickup(player.pos, WORLD_HALF))
                pickup_timer = gcfg.PICKUP_SPAWN_INTERVAL

            # --- Pickup collection ---------------------------------------
            remaining_pickups: List[Pickup] = []
            for p in pickups:
                d = math.hypot(p.pos[0] - player.pos[0],
                               p.pos[1] - player.pos[1])
                if d < (p.radius + 0.45):
                    label = apply_pickup(p, player, now_t)
                    pickup_msg = (label, now_t + 1.6)
                else:
                    remaining_pickups.append(p)
            pickups = remaining_pickups

            obs.tick_apm()

            # --- Enemies / boss ------------------------------------------
            scaled_spawn = scale_spawn_rate(float(E[1].item()), run.room)
            scaled_speed = scale_enemy_speed(float(E[0].item()), run.room,
                                             run.enemy_slow)

            if state == GameState.PLAYING:
                spawn_accum += max(scaled_spawn, 0.05) * dt
                while spawn_accum >= 1.0 and len(enemies) < MAX_ENEMIES:
                    kind = pick_kind(pi)
                    enemies.append(make_enemy(kind, player.pos, WORLD_HALF))
                    spawn_accum -= 1.0
                step_enemies(enemies, player.pos, scaled_speed, 1.0, dt)
            else:  # BOSS
                if boss is not None:
                    step_boss(boss, player.pos, pi, WORLD_HALF, enemies, dt)
                step_enemies(enemies, player.pos, scaled_speed, 1.0, dt)

            # --- Collisions ----------------------------------------------
            dashing = player.dashing > 0.0
            trail = player.dash_trail
            trail_active = len(trail) >= 1

            def _killed_by_trail(ex: float, ey: float, er: float) -> bool:
                if not trail_active:
                    return False
                for j in range(len(trail) - 1):
                    ax_, ay_, _ = trail[j]
                    bx_, by_, _ = trail[j + 1]
                    pax, pay = ex - ax_, ey - ay_
                    bax, bay = bx_ - ax_, by_ - ay_
                    ab2 = bax * bax + bay * bay + 1e-9
                    h = max(0.0, min(1.0, (pax * bax + pay * bay) / ab2))
                    cx, cy = ax_ + h * bax, ay_ + h * bay
                    if math.hypot(ex - cx, ey - cy) < (DASH_TRAIL_RADIUS + er):
                        return True
                # Single-point trail still counts as a kill zone.
                ax_, ay_, _ = trail[-1]
                return math.hypot(ex - ax_, ey - ay_) < (DASH_TRAIL_RADIUS + er)

            survivors: List[Enemy] = []
            for e in enemies:
                d = math.hypot(e.pos[0] - player.pos[0],
                               e.pos[1] - player.pos[1])
                if d < (e.radius + 0.35) and e.contact_cd <= 0.0:
                    if dashing:
                        e.hp -= 99           # dash obliterates enemies
                    else:
                        e.hp -= 1
                        if player.invuln <= 0.0:
                            if "shield" in player.buffs:
                                # Shield absorbs the hit: brief flash, no HP loss.
                                player.hit_flash = 0.4
                                player.invuln = 0.25
                            else:
                                dmg = gcfg.DAMAGE_ON_HIT * run.damage_mult
                                player.hp -= dmg
                                player.invuln = gcfg.HIT_INVULN
                                player.hit_flash = 1.0
                    e.contact_cd = e.contact_cd_max
                # Trail kill: any enemy whose body intersects a recent dash
                # segment dies instantly, even after the dash itself ended.
                if e.hp > 0 and _killed_by_trail(e.pos[0], e.pos[1], e.radius):
                    e.hp = 0
                if e.hp > 0:
                    survivors.append(e)
                else:
                    run.register_kill(pi)
            enemies = survivors

            if state == GameState.BOSS and boss is not None:
                d = math.hypot(boss.pos[0] - player.pos[0],
                               boss.pos[1] - player.pos[1])
                if d < (boss.radius + 0.45) and boss.contact_cd <= 0.0:
                    if dashing:
                        boss.hp -= gcfg.BOSS_DASH_DAMAGE
                        boss.hit_flash = 1.0
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
                    boss.contact_cd = 0.4
                if boss.hp <= 0:
                    # Boss defeated -> run.register_kill once, clear room.
                    run.register_kill(pi)
                    run.clear_room()
                    start_next_room_or_shop()

            # --- Progression ---------------------------------------------
            run.room_time_remaining -= dt
            run.tick(pi, dt)
            if state == GameState.PLAYING:
                if run.room_goal_met():
                    run.clear_room()
                    start_next_room_or_shop()
                elif run.time_expired():
                    player.hp = 0.0   # failed kill quota — death

            if player.hp <= 0.0:
                player.hp = 0.0
                best_score = max(best_score, int(run.score))
                state = GameState.GAME_OVER

            # --- Inference -----------------------------------------------
            threat_list = enemies + ([boss] if boss is not None else [])
            o_t = obs.build(player.vel[0], player.vel[1], threat_list,
                            tuple(player.pos)).to(DEVICE)
            pi = attn(o_t, (player.vel[0], player.vel[1]))
            E, c = mapper(pi)

        # -----------------------------------------------------------------
        # Render
        # -----------------------------------------------------------------
        glViewport(0, 0, width, height)
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        glUseProgram(program)
        glUniform1f(U["u_time"], time.time() - t_start)
        glUniform2f(U["u_resolution"], float(width), float(height))
        glUniform2f(U["u_player"], float(player.pos[0]), float(player.pos[1]))

        pos_flat = np.zeros(MAX_ENEMIES * 2, dtype=np.float32)
        rad_flat = np.zeros(MAX_ENEMIES, dtype=np.float32)
        for i, e in enumerate(enemies[:MAX_ENEMIES]):
            pos_flat[2 * i]     = e.pos[0]
            pos_flat[2 * i + 1] = e.pos[1]
            rad_flat[i] = e.radius
        glUniform2fv(U["u_enemies"], MAX_ENEMIES, pos_flat)
        glUniform1fv(U["u_enemy_radii"], MAX_ENEMIES, rad_flat)
        glUniform1i(U["u_enemy_count"], min(len(enemies), MAX_ENEMIES))

        glUniform1f(U["u_world_half"], float(WORLD_HALF))
        glUniform1f(U["u_cam_zoom"], float(max(E[3].item(), 0.3)))

        c1 = max(float(c[0].item()), 0.0)
        c2 = max(float(c[1].item()), 0.0)
        c3 = max(float(c[2].item()), 0.0)
        c4 = max(float(c[3].item()), 0.0)
        glUniform1f(U["u_c1_aberration"], c1)
        glUniform1f(U["u_c2_grain"],      c2)
        glUniform1f(U["u_c3_warp"],       c3)
        glUniform1f(U["u_c4_bloom"],      c4)
        glUniform4f(U["u_pi_top4"],
                    float(pi[ARCH_AROUSAL]), float(pi[ARCH_TACTICAL]),
                    float(pi[ARCH_OVERLOAD]), float(pi[ARCH_FLOW]))

        if boss is not None:
            glUniform1f(U["u_boss_active"], 1.0)
            glUniform2f(U["u_boss_pos"], float(boss.pos[0]), float(boss.pos[1]))
            glUniform1f(U["u_boss_radius"], float(boss.radius))
            glUniform1f(U["u_boss_hp_frac"], max(0.0, boss.hp / boss.max_hp))
        else:
            glUniform1f(U["u_boss_active"], 0.0)
            glUniform2f(U["u_boss_pos"], 0.0, 0.0)
            glUniform1f(U["u_boss_radius"], 0.0)
            glUniform1f(U["u_boss_hp_frac"], 0.0)

        # Screen shake scales with chromatic aberration + damage flash.
        shake_amp = gcfg.SHAKE_GAIN * c1 + 0.02 * player.hit_flash
        sx = (random.random() - 0.5) * 2.0 * shake_amp
        sy = (random.random() - 0.5) * 2.0 * shake_amp
        glUniform2f(U["u_shake"], sx, sy)
        glUniform1f(U["u_player_flash"], min(1.0, player.dashing * 5.0))
        glUniform1f(U["u_hit_flash"], player.hit_flash)
        dash_phase = (player.dashing / gcfg.DASH_INVULN
                      if gcfg.DASH_INVULN > 0 else 0.0)
        glUniform1f(U["u_dash_phase"], max(0.0, min(1.0, dash_phase)))
        glUniform2f(U["u_dash_dir"],
                    float(player.dash_dir[0]), float(player.dash_dir[1]))

        # Dash trail uniforms: positions packed flat + per-point alpha.
        trail_pos = np.zeros(DASH_TRAIL_MAX * 2, dtype=np.float32)
        trail_a   = np.zeros(DASH_TRAIL_MAX,     dtype=np.float32)
        now_t = time.time() - t_start
        for ti, (tx, ty, te) in enumerate(player.dash_trail[:DASH_TRAIL_MAX]):
            trail_pos[2 * ti]     = tx
            trail_pos[2 * ti + 1] = ty
            ttl_left = max(0.0, te - now_t)
            trail_a[ti] = min(1.0, ttl_left / DASH_TRAIL_TTL)
        glUniform2fv(U["u_dash_trail"],   DASH_TRAIL_MAX, trail_pos)
        glUniform1fv(U["u_dash_trail_a"], DASH_TRAIL_MAX, trail_a)
        glUniform1i(U["u_dash_trail_count"],
                    min(len(player.dash_trail), DASH_TRAIL_MAX))

        # Pickups uniform upload.
        pcap = gcfg.PICKUP_SHADER_CAP
        pickup_pos_flat = np.zeros(pcap * 2, dtype=np.float32)
        pickup_kind_flat = np.zeros(pcap, dtype=np.int32)
        for pi_, p_obj in enumerate(pickups[:pcap]):
            pickup_pos_flat[2 * pi_]     = p_obj.pos[0]
            pickup_pos_flat[2 * pi_ + 1] = p_obj.pos[1]
            pickup_kind_flat[pi_] = PICKUP_CODE.get(p_obj.kind, 0)
        glUniform2fv(U["u_pickups"], pcap, pickup_pos_flat)
        glUniform1iv(U["u_pickup_kinds"], pcap, pickup_kind_flat)
        glUniform1i(U["u_pickup_count"], min(len(pickups), pcap))

        glBindVertexArray(vao)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)

        # --- UI overlay --------------------------------------------------
        if state == GameState.MENU:
            ui.render_menu(best_score)
        elif state == GameState.PLAYING:
            goal = (f"Kill {run.room_kills_required - run.kills_in_room} more  "
                    f"·  {max(0.0, run.room_time_remaining):4.1f}s remaining")
            ui.render_playing(player, run, pi, c, goal, boss=None,
                              now_t=time.time() - t_start,
                              pickup_msg=pickup_msg)
        elif state == GameState.BOSS:
            goal = f"Defeat the BOSS  ·  {max(0.0, run.room_time_remaining):4.1f}s"
            ui.render_playing(player, run, pi, c, goal, boss=boss,
                              now_t=time.time() - t_start,
                              pickup_msg=pickup_msg)
        elif state == GameState.SHOP:
            ui.render_shop(shop_choices, UPGRADES, run)
        elif state == GameState.GAME_OVER:
            ui.render_game_over(run)
        ui.draw()

        pygame.display.flip()

        if debug and state in (GameState.PLAYING, GameState.BOSS):
            names = ["Arousal", "Tactical", "Overload", "Flow", "Apathy"]
            mix = ", ".join(f"{n[:3]}:{float(pi[i]):.2f}" for i, n in enumerate(names))
            print(f"[{mix}] E={[round(float(x),2) for x in E.tolist()]} "
                  f"c={[round(float(x),3) for x in c.tolist()]} "
                  f"enemies={len(enemies)} boss={boss.hp if boss else '-'}")

    pygame.quit()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true",
                    help="Run headless invariant check and exit.")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--windowed", action="store_true",
                    help="Windowed mode instead of fullscreen.")
    args = ap.parse_args()
    if args.probe:
        return run_probe(args.frames)
    return run_game(windowed=args.windowed)


if __name__ == "__main__":
    sys.exit(main())
