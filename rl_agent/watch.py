"""Live watcher: runs the trained agent in the full OpenGL renderer.

Meant to run alongside training in a separate terminal (or launched
automatically with --watch in train.py).

Usage:
    # Auto-detects newest checkpoint every 30s:
    /home/atharv/venv/bin/python3 -m rl_agent.watch

    # Watch a specific checkpoint:
    /home/atharv/venv/bin/python3 -m rl_agent.watch --checkpoint rl_agent/checkpoints/checkpoint_final.pt

    # Force a target archetype (override meta-policy):
    /home/atharv/venv/bin/python3 -m rl_agent.watch --archetype flow

Keys during play:
    ESC         quit
    1-5         manually select target archetype
    R           reload latest checkpoint now
    Space       pause / unpause
"""
from __future__ import annotations

import argparse
import ctypes
import glob
import math
import os
import random
import sys
import time
from typing import List, Optional, Tuple

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Headless env for game logic only — no rendering from that side.
os.environ["FLOW_HEADLESS"] = "1"

import numpy as np
import torch

# Import the game's pure-logic modules.
from game import config as gcfg
from game.game_state import GameState
from game.entities_extra import PICKUP_CODE
from flow_game import (
    VERTEX_SHADER, FRAGMENT_SHADER,
    ARCH_AROUSAL, ARCH_FLOW, WORLD_HALF,
    DASH_TRAIL_TTL, DASH_TRAIL_MAX,
    K as N_PI,
)
from rl_agent.config import OPTION_PERIOD_K, OBS_DIM
from rl_agent.env import FlowGameEnv

ARCH_NAMES  = ["Arousal", "Tactical", "Overload", "Flow", "Apathy"]
ARCH_COLORS = [
    (220, 80,  80),   # Arousal  — red
    (80,  140, 220),  # Tactical — blue
    (160, 80,  220),  # Overload — purple
    (80,  210, 200),  # Flow     — teal
    (160, 160, 160),  # Apathy   — grey
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Watch trained RL agent play Flow Game")
    p.add_argument("--checkpoint",     type=str, default=None,
                   help="Checkpoint .pt file. If omitted, auto-detect newest.")
    p.add_argument("--checkpoint_dir", type=str, default="rl_agent/checkpoints")
    p.add_argument("--archetype",      type=str, default=None,
                   help="Force target archetype: arousal/tactical/overload/flow/apathy")
    p.add_argument("--windowed",       action="store_true")
    p.add_argument("--reload_every",   type=int, default=30,
                   help="Seconds between checkpoint auto-reloads")
    p.add_argument("--fps",            type=int, default=60)
    return p.parse_args()


def _newest_checkpoint(ckpt_dir: str) -> Optional[str]:
    files = glob.glob(os.path.join(ckpt_dir, "checkpoint_*.pt"))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def _load_policies(path: str, device: torch.device):
    from rl_agent.models import LowLevelPolicy, MetaPolicy
    ll   = LowLevelPolicy().to(device)
    meta = MetaPolicy().to(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    ll.load_state_dict(ckpt["ll_policy"])
    meta.load_state_dict(ckpt["meta_policy"])
    ll.eval();  meta.eval()
    return ll, meta


# ---------------------------------------------------------------------------
# Main renderer loop
# ---------------------------------------------------------------------------

def main() -> None:
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    forced_arch = None
    if args.archetype:
        forced_arch = next(
            (i for i, n in enumerate(ARCH_NAMES)
             if n.lower() == args.archetype.lower()), None
        )
        if forced_arch is None:
            print(f"Unknown archetype '{args.archetype}'. "
                  f"Choose from: {[n.lower() for n in ARCH_NAMES]}")
            sys.exit(1)

    # --- Find initial checkpoint -------------------------------------------
    ckpt_path = args.checkpoint or _newest_checkpoint(args.checkpoint_dir)
    if ckpt_path is None:
        print("No checkpoint found. Start training first:\n"
              "  python3 -m rl_agent.train --num_envs 1 --total_steps 50000")
        sys.exit(1)
    print(f"Loading checkpoint: {ckpt_path}")
    ll_policy, meta_policy = _load_policies(ckpt_path, device)
    last_ckpt_path  = ckpt_path
    last_ckpt_mtime = os.path.getmtime(ckpt_path)
    last_reload_t   = time.time()

    # --- pygame + OpenGL init ----------------------------------------------
    import pygame
    from OpenGL.GL import (
        glGenVertexArrays, glBindVertexArray,
        glGenBuffers, glBindBuffer, glBufferData,
        glEnableVertexAttribArray, glVertexAttribPointer,
        glUseProgram, glGetUniformLocation,
        glUniform1f, glUniform1i, glUniform2f, glUniform2fv, glUniform1fv,
        glUniform1iv, glUniform4f,
        glClear, glClearColor, glDrawArrays, glViewport,
        GL_ARRAY_BUFFER, GL_STATIC_DRAW, GL_FLOAT, GL_FALSE,
        GL_COLOR_BUFFER_BIT, GL_TRIANGLE_STRIP,
    )
    from flow_game import compile_shader_program

    pygame.init()
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(
        pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)

    if args.windowed:
        width, height = gcfg.WINDOWED_SIZE
        flags = pygame.OPENGL | pygame.DOUBLEBUF
    else:
        info = pygame.display.Info()
        width, height = info.current_w, info.current_h
        flags = pygame.OPENGL | pygame.DOUBLEBUF | pygame.FULLSCREEN
    pygame.display.set_mode((width, height), flags)
    pygame.display.set_caption("Flow — RL Agent Watching")
    pygame.mouse.set_visible(False)

    os.environ.pop("FLOW_HEADLESS", None)   # rendering is now active

    program = compile_shader_program(VERTEX_SHADER, FRAGMENT_SHADER)
    glUseProgram(program)

    quad = np.array([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    vao = glGenVertexArrays(1); glBindVertexArray(vao)
    vbo = glGenBuffers(1);      glBindBuffer(GL_ARRAY_BUFFER, vbo)
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

    # HUD font (pygame 2D → blit on top of GL).
    pygame.font.init()
    font_large = pygame.font.SysFont("monospace", 22, bold=True)
    font_small = pygame.font.SysFont("monospace", 16)

    # --- Env + agent state --------------------------------------------------
    env          = FlowGameEnv(seed=int(time.time()) % 10000)
    obs          = env.reset()
    target_arch  = forced_arch if forced_arch is not None else 0
    env.set_target(target_arch)
    steps_in_opt = 0
    score_accum  = 0.0
    episode      = 0
    paused       = False
    t0           = time.time()

    clock = pygame.time.Clock()
    running = True

    while running:
        # --- Events -------------------------------------------------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    ckpt = _newest_checkpoint(args.checkpoint_dir)
                    if ckpt:
                        print(f"[watch] reloading {ckpt}")
                        ll_policy, meta_policy = _load_policies(ckpt, device)
                        last_ckpt_path  = ckpt
                        last_ckpt_mtime = os.path.getmtime(ckpt)
                        last_reload_t   = time.time()
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif forced_arch is None:
                    for k, key in enumerate([
                        pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5
                    ]):
                        if event.key == key:
                            target_arch = k
                            env.set_target(target_arch)
                            print(f"[watch] target → {ARCH_NAMES[target_arch]}")

        if paused:
            clock.tick(15)
            continue

        # --- Auto-reload checkpoint ----------------------------------------
        now = time.time()
        if now - last_reload_t >= args.reload_every:
            ckpt = _newest_checkpoint(args.checkpoint_dir)
            if ckpt:
                mtime = os.path.getmtime(ckpt)
                if ckpt != last_ckpt_path or mtime > last_ckpt_mtime:
                    print(f"[watch] auto-reload {ckpt}")
                    ll_policy, meta_policy = _load_policies(ckpt, device)
                    last_ckpt_path  = ckpt
                    last_ckpt_mtime = mtime
            last_reload_t = now

        # --- Meta: re-select archetype every OPTION_PERIOD_K steps ----------
        if forced_arch is None and steps_in_opt >= OPTION_PERIOD_K:
            meta_obs_np = env.get_meta_obs(score_accum)
            meta_obs_t  = torch.from_numpy(meta_obs_np[None]).to(device)
            with torch.no_grad():
                arch_t, _, _, _ = meta_policy.get_action_and_value(meta_obs_t)
            target_arch = int(arch_t[0].item())
            env.set_target(target_arch)
            steps_in_opt = 0
            score_accum  = 0.0

        # --- Low-level policy action -----------------------------------------
        obs_t   = torch.from_numpy(obs[None]).to(device)
        is_shop = torch.tensor([obs[27] > 0.5], dtype=torch.bool, device=device)
        with torch.no_grad():
            action_t, _, _, _ = ll_policy.get_action_and_value(obs_t, is_shop)
        action_np = action_t[0].cpu().numpy()

        obs, reward, terminated, truncated, info = env.step(action_np)
        score_accum  += reward
        steps_in_opt += 1

        if terminated or truncated:
            episode += 1
            print(f"[watch] ep={episode:>3}  score={info['score']:>8,.0f}  "
                  f"rooms={info['room']:>2}  "
                  f"π_flow={info['flow']:.2f}  "
                  f"target={ARCH_NAMES[target_arch]}")
            obs = env.reset()
            steps_in_opt = 0
            score_accum  = 0.0
            t0 = time.time()

        # --- Render -----------------------------------------------------------
        t_now = time.time() - t0
        player  = env.player
        enemies = env.enemies
        boss    = env.boss
        pickups = env.pickups
        pi      = env.pi
        E       = env.E
        c       = env.c

        glViewport(0, 0, width, height)
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(program)

        glUniform1f(U["u_time"],        t_now)
        glUniform2f(U["u_resolution"],  float(width), float(height))
        glUniform2f(U["u_player"],      float(player.pos[0]), float(player.pos[1]))

        pos_flat = np.zeros(gcfg.MAX_ENEMIES * 2, dtype=np.float32)
        rad_flat = np.zeros(gcfg.MAX_ENEMIES,     dtype=np.float32)
        for i, e in enumerate(enemies[:gcfg.MAX_ENEMIES]):
            pos_flat[2*i]   = e.pos[0];  pos_flat[2*i+1] = e.pos[1]
            rad_flat[i]     = e.radius
        glUniform2fv(U["u_enemies"],      gcfg.MAX_ENEMIES, pos_flat)
        glUniform1fv(U["u_enemy_radii"],  gcfg.MAX_ENEMIES, rad_flat)
        glUniform1i(U["u_enemy_count"],   min(len(enemies), gcfg.MAX_ENEMIES))

        glUniform1f(U["u_world_half"],    float(WORLD_HALF))
        glUniform1f(U["u_cam_zoom"],      float(max(E[3].item(), 0.3)))

        c0 = max(float(c[0].item()), 0.0)
        c1 = max(float(c[1].item()), 0.0)
        c2 = max(float(c[2].item()), 0.0)
        c3 = max(float(c[3].item()), 0.0)
        glUniform1f(U["u_c1_aberration"], c0)
        glUniform1f(U["u_c2_grain"],      c1)
        glUniform1f(U["u_c3_warp"],       c2)
        glUniform1f(U["u_c4_bloom"],      c3)

        glUniform4f(U["u_pi_top4"],
                    float(pi[0]), float(pi[1]), float(pi[2]), float(pi[3]))

        if boss is not None:
            glUniform1f(U["u_boss_active"],  1.0)
            glUniform2f(U["u_boss_pos"],     float(boss.pos[0]), float(boss.pos[1]))
            glUniform1f(U["u_boss_radius"],  float(boss.radius))
            glUniform1f(U["u_boss_hp_frac"], max(0.0, boss.hp / boss.max_hp))
        else:
            glUniform1f(U["u_boss_active"],  0.0)
            glUniform2f(U["u_boss_pos"],     0.0, 0.0)
            glUniform1f(U["u_boss_radius"],  0.0)
            glUniform1f(U["u_boss_hp_frac"], 0.0)

        shake_amp = gcfg.SHAKE_GAIN * c0 + 0.02 * player.hit_flash
        glUniform2f(U["u_shake"],
                    (random.random() - 0.5) * 2.0 * shake_amp,
                    (random.random() - 0.5) * 2.0 * shake_amp)
        glUniform1f(U["u_player_flash"],  min(1.0, player.dashing * 5.0))
        glUniform1f(U["u_hit_flash"],     player.hit_flash)
        dash_phase = (player.dashing / gcfg.DASH_INVULN) if gcfg.DASH_INVULN > 0 else 0.0
        glUniform1f(U["u_dash_phase"],    max(0.0, min(1.0, dash_phase)))
        glUniform2f(U["u_dash_dir"],      float(player.dash_dir[0]),
                                          float(player.dash_dir[1]))

        trail_pos = np.zeros(DASH_TRAIL_MAX * 2, dtype=np.float32)
        trail_a   = np.zeros(DASH_TRAIL_MAX,     dtype=np.float32)
        for ti, (tx, ty, te) in enumerate(player.dash_trail[:DASH_TRAIL_MAX]):
            trail_pos[2*ti]   = tx;  trail_pos[2*ti+1] = ty
            ttl_left = max(0.0, te - env.sim_time)
            trail_a[ti] = min(1.0, ttl_left / DASH_TRAIL_TTL)
        glUniform2fv(U["u_dash_trail"],   DASH_TRAIL_MAX, trail_pos)
        glUniform1fv(U["u_dash_trail_a"], DASH_TRAIL_MAX, trail_a)
        glUniform1i(U["u_dash_trail_count"], min(len(player.dash_trail), DASH_TRAIL_MAX))

        pcap = gcfg.PICKUP_SHADER_CAP
        pu_pos  = np.zeros(pcap * 2, dtype=np.float32)
        pu_kind = np.zeros(pcap,     dtype=np.int32)
        for pi_, p_obj in enumerate(pickups[:pcap]):
            pu_pos[2*pi_]   = p_obj.pos[0];  pu_pos[2*pi_+1] = p_obj.pos[1]
            pu_kind[pi_]    = PICKUP_CODE.get(p_obj.kind, 0)
        glUniform2fv(U["u_pickups"],      pcap, pu_pos)
        glUniform1iv(U["u_pickup_kinds"], pcap, pu_kind)
        glUniform1i(U["u_pickup_count"],  min(len(pickups), pcap))

        glBindVertexArray(vao)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)

        # --- HUD overlay (pygame surface → pixel blit not possible on GL
        #     window; use GL_BLEND + texture approach is complex, so we
        #     write text directly as a 2D surface and call glDrawPixels or
        #     just print the stats in the window title as a simple fallback).
        _render_hud(pygame, font_large, font_small, width, height,
                    player, env.run, pi, target_arch, info,
                    last_ckpt_path, episode, U, program,
                    glUniform1f, glUniform2f, glUniform4f)

        pygame.display.flip()
        clock.tick(args.fps)

    pygame.quit()


# ---------------------------------------------------------------------------
# HUD: render stats as window title + a simple pygame overlay texture
# ---------------------------------------------------------------------------

def _render_hud(
    pygame, font_large, font_small, width, height,
    player, run, pi, target_arch, info,
    ckpt_path, episode, U, program,
    glUniform1f, glUniform2f, glUniform4f,
) -> None:
    """Render a text HUD by updating the window caption (always visible)
    and optionally drawing a pygame surface overlay.

    The pygame surface approach on an OpenGL window requires uploading it
    as a texture — we do a lightweight version using glWindowPos2i +
    glDrawPixels which is available in compatibility profile.  If that
    isn't available (core profile), we fall back to the title bar.
    """
    arch_name = ARCH_NAMES[target_arch]
    flow_pct  = int(float(pi[ARCH_FLOW]) * 100)
    arch_pct  = int(float(pi[target_arch]) * 100)
    title = (
        f"Flow RL  |  ep={episode}  score={int(info['score']):,}  "
        f"room={info['room']}  "
        f"target={arch_name}({arch_pct}%)  "
        f"π_flow={flow_pct}%  "
        f"hp={int(player.hp)}/{int(player.max_hp)}"
    )
    pygame.display.set_caption(title)

    # Try to render a small transparent surface with text via glWindowPos
    # (compatibility-profile feature; silently skipped on core profile).
    try:
        from OpenGL.GL import glWindowPos2i, glDrawPixels, GL_RGBA, GL_UNSIGNED_BYTE
        from OpenGL.GL import glEnable, glDisable, GL_BLEND, glBlendFunc
        from OpenGL.GL import GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA

        surf = pygame.Surface((width, 90), pygame.SRCALPHA)
        surf.fill((0, 0, 0, 140))

        # Archetype π bars.
        bar_x = 10
        for i, (name, color) in enumerate(zip(ARCH_NAMES, ARCH_COLORS)):
            val = float(pi[i])
            bar_w = int(val * 160)
            is_target = (i == target_arch)
            rect_col  = color if is_target else tuple(c // 2 for c in color)
            pygame.draw.rect(surf, rect_col,
                             (bar_x + i * 170, 10, bar_w, 18))
            label = font_small.render(
                f"{name[:3]} {int(val*100):>3}%", True,
                (255, 255, 255) if is_target else (160, 160, 160)
            )
            surf.blit(label, (bar_x + i * 170, 30))

        # Score / HP line.
        txt = font_large.render(
            f"Score {int(info['score']):>8,}   HP {int(player.hp)}/{int(player.max_hp)}"
            f"   Room {info['room']}   Target → {ARCH_NAMES[target_arch]}",
            True, (220, 220, 220)
        )
        surf.blit(txt, (10, 58))

        # Upload pygame surface as GL pixels at bottom of screen (so it
        # doesn't obscure the game arena).
        raw = pygame.image.tostring(surf, "RGBA", True)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glWindowPos2i(0, 0)
        glDrawPixels(width, 90, GL_RGBA, GL_UNSIGNED_BYTE, raw)
        glDisable(GL_BLEND)
    except Exception:
        pass   # core profile — title bar only


if __name__ == "__main__":
    main()
