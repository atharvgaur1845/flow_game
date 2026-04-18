# Flow Game — Project Context (init.md)

This document preserves the full specification, design rationale, and
mathematical intent of the project so that any future session inherits
the complete context without re-deriving it from code.

The project now has two layers:

1. **Backend inference engine** (`flow_game.py`, untouchable) — the
   non-causal attention model, environment mapper, and observation
   builder that turn keypresses into a state mixture `pi`, physics
   parameters `E`, and visual latents `c`.
2. **Roguelite game layer** (`game/` package) — a state-machine-driven
   neon arena that only **reads** `pi, E, c` and builds rooms, a shop,
   enemy variety, a boss, collisions, score, and UI on top.

---

## 1. Original System-Role Brief (verbatim)

> **System Role:** You are a Lead AI Research Scientist and Graphics
> Engineer. Your task is to write a complete, runnable Python script for
> a dynamic, mathematically rigorous 2D game.
>
> **Project Overview:** I need a self-contained Python game using
> `pygame`, `PyOpenGL`, and `torch`. The environment and visuals must
> adapt in real-time to the player's psychological state. The game
> observes movements, infers state using a non-causal continuous
> attention model, and dynamically maps this to gameplay physics and
> GLSL shader parameters (producing disentangled "trippy" or "gritty"
> visuals).
>
> **Technical Stack**
> - Backend Math & ML: `torch` (PyTorch), `numpy`. Strictly no fake math;
>   implement actual matrix multiplications, attention mechanisms, and
>   tensor operations.
> - Frontend & Input: `pygame` (game loop, window creation, capturing
>   raw D-dimensional input data).
> - Visuals: `PyOpenGL` using GLSL fragment shaders mapped to
>   disentangled latent-space variables.

### Phase 1 — K-Dimensional State Mixture Simplex
K=5 archetypes: Kinetic Arousal, Tactical Deliberation, Cognitive
Overload, Hypnotic Flow, Apathetic Disengagement. `pi_{k,t}` lies on
the (K−1)-simplex: `0 ≤ pi_k ≤ 1`, `sum_k pi_k = 1`.

### Phase 2 — Non-Causal Inference via Bidirectional Attention
Window `O_t = { o_{t-tau}, ..., o_t, ..., hat o_{t+tau} }` with
`hat o` = deterministic physics-projected future observations.
`pi_t = softmax(W_z · Attention(Q, K, V))`.

### Phase 3 — State-to-Environment Matrix M
`E_t = M · pi_t + eps`, `M ∈ R^{PxK}`, `eps ~ N(0, Sigma)`.

### Phase 4 — Disentangled β-VAE Decoder
β-VAE objective (structural, not trained):
`L = E_q[log p] - β · KL(q || p)`. Latent `c ∈ R^V` controls
aberration / grain / warp / bloom, each independently.

### Observation vector (D=5)
`v_norm`, APM proxy, directional variance, proximity threat, idle time,
EMA-normalized.

### Visual latents (V=4)
`c_1` chromatic aberration, `c_2` grain, `c_3` warp, `c_4` bloom.

### Shader math
SDF circles, `uv.x += sin(uv.y*10 + t) * c_3`,
`sampleScene(uv ± normalize(uv-0.5) * c_1)` for R/B channels,
`fract(sin(dot(uv, vec2(12.9898, 78.233))) * 43758.5453) * c_2`.

---

## 2. Roguelite Game Brief (layer 2)

Adaptive Neon Roguelite Arena — arena survival × roguelite progression.

**Core loop:**
```
Start Run →
  Room 1 → Room 2 → Room 3 → Shop →
  Room 4 → Room 5 (Boss) →
  Repeat with scaling → Death → Score screen
```

**Hard rules followed by the game layer:**
- ❌ Do NOT modify the attention model, mapper, or observation builder.
- ❌ Do NOT inject fake states.
- ❌ Do NOT hardcode difficulty curves that replace backend.
- ✅ Only **scale** outputs (`E[0] *= (1 + 0.1 * room)`) and **interpret** outputs (spawn more fast enemies when `pi[Tactical]` is high).

**Progression targets (from spec):**
- HP bar, score + multiplier, room counter.
- State bars (AROUSAL, TACTICAL, OVERLOAD, FLOW, APATHY).
- Dash (SPACE, i-frames).
- Shop every 3 rooms with 5 possible upgrades.
- Boss every 5 rooms, with behaviour driven by `pi`.
- Enemy variety: Chaser, Fast, Tank.
- Visual feedback: shake on high `c1`, glow pulse on high `c4`, etc.
- Game-over screen showing score, time survived, max flow, rooms cleared.

---

## 3. Symbol Table

| Symbol      | Meaning                              | Value |
|-------------|--------------------------------------|-------|
| `D`         | observation dimensionality           | 5     |
| `K`         | archetype count                      | 5     |
| `W_HIST`    | historical window size               | 16    |
| `W_FUT`     | projected future window size         | 16    |
| `W_TOTAL`   | total attention window               | 32    |
| `d_k`       | attention inner dimension            | 16    |
| `P`         | environment parameters               | 4     |
| `V_LATENT`  | disentangled visual latent dims      | 4     |
| `alpha`     | EMA baseline rate                    | 0.02  |
| `beta`      | β-VAE KL weight (structural only)    | implicit via sparse decoder init |
| `sigma^2`   | env noise variance                   | `0.02^2` |
| `MAX_ENEMIES` | shader uniform array size          | 16    |
| `DEVICE`    | torch device                         | `cuda` if available, else `cpu` |

---

## 4. File Map

```
flow_game/
├── flow_game.py              # backend (untouched) + main loop
├── requirements.txt
├── init.md                   # this document
└── game/
    ├── __init__.py
    ├── config.py             # tunable gameplay constants
    ├── game_state.py         # GameState enum (MENU/PLAYING/SHOP/BOSS/GAME_OVER)
    ├── abilities.py          # dash + timer decrement
    ├── entities_extra.py     # Chaser/Fast/Tank + Boss + AI step functions
    ├── progression.py        # RunState, rooms, shop upgrades
    └── ui.py                 # pygame -> GL overlay (HP, bars, text, shop, menu)
```

### Backend modules (must stay pristine)
- **`AttentionStateInference`** [flow_game.py:220-290](flow_game.py#L220-L290) — maintains history ring buffer + ema_mu, builds non-causal window (history ‖ future-projection), scaled dot-product attention, mean-pools temporally, projects to K-simplex via `W_pi` + softmax.
- **`EnvironmentMapper`** [flow_game.py:293-330](flow_game.py#L293-L330) — hand-tuned `M (4×5)` for physics and sparse ReLU decoder `W_c1 (8×5), W_c2 (4×8)` for disentangled visual latents.
- **`ObservationBuilder`** [flow_game.py:333-373](flow_game.py#L333-L373) — builds `o_t` from velocity, APM decay, circular heading variance, min-enemy distance, idle counter.

### Game layer modules
- **`game/config.py`** — all gameplay constants (HP, dash CD, room timing, enemy stats, boss stats, scoring).
- **`game/game_state.py`** — `GameState` enum (`MENU`, `PLAYING`, `SHOP`, `BOSS`, `GAME_OVER`).
- **`game/abilities.py`** — `try_dash(player, keys, pygame)` fires a 4× velocity impulse + i-frames on SPACE; `update_timers(player, dt)` decrements dash/invuln/dashing every frame.
- **`game/entities_extra.py`** — `Enemy` dataclass (kind, hp, radius, base_speed, contact_cd), `make_enemy`, `pick_kind(pi)` (weighted by `pi[OVERLOAD]` / `pi[TACTICAL]` / `pi[APATHY]`), `step_enemies`, `Boss` dataclass, `step_boss(boss, player_pos, pi, ...)` with AI branches driven by the dominant archetype (Arousal→chase, Flow→teleport, Overload→summon minions, Tactical→hold ideal range).
- **`game/progression.py`** — `RunState` (score, room, multipliers, goal), `scale_enemy_speed(E0, room, slow)` and `scale_spawn_rate(E1, room)` which **multiply** backend outputs (never replace), `UPGRADES` list (5 upgrades), `pick_shop_choices`, `apply_upgrade(idx, player, run)`.
- **`game/ui.py`** — `UIOverlay` renders to a `pygame.Surface` each frame (HP bar, dash bar, score / multiplier / room / kills, state bars for π_k, visual-latent bars for c, shop menu, game-over screen), then uploads the surface to a GL texture and composites it over the scene with alpha blending.

### Main loop (state machine)
[flow_game.py:400-620](flow_game.py#L400-L620) — branches on `GameState`:
- **MENU** — ENTER starts a run.
- **PLAYING** — WASD accel → physics → enemy spawn (rate = `scale_spawn_rate(E[1], room)`), enemy homing (speed = `scale_enemy_speed(E[0], room, run.enemy_slow)`), contact collisions (dash = invuln one-shot; otherwise take `DAMAGE_ON_HIT * run.damage_mult`), score accrues via `SCORE_TIME_BASE * (1 + pi[Flow])`, room goal = kills OR timer.
- **SHOP** — 1/2/3 picks an upgrade, then starts the next room (or boss room if `room % 5 == 0`).
- **BOSS** — `step_boss` drives AI using `pi`; player damages boss only during dash.
- **GAME_OVER** — shows stats; ENTER returns to menu.

### Shader additions
New uniforms in the main fragment shader:
- `u_enemy_radii[16]` — per-enemy radius (lets Fast/Chaser/Tank render at their true sizes).
- `u_boss_active`, `u_boss_pos`, `u_boss_radius`, `u_boss_hp_frac` — draws the boss as a large SDF disc with a dynamically filled HP ring around it.
- `u_shake` — 2D screen-space offset, amplitude = `SHAKE_GAIN * c1 + 0.02 * hit_flash`.
- `u_player_flash` — whitens the player during dash i-frames.
- `u_hit_flash` — reddens the whole frame briefly after taking damage.

---

## 5. Archetype → Effect Table (unchanged)

### `M` (physics)

| Row \ Archetype | Arousal | Tactical | Overload | Flow | Apathy |
|---|---|---|---|---|---|
| Enemy speed    | 1.8 | 0.6 | 1.4 | 1.2 | 0.3 |
| Spawn rate     | 1.5 | 0.4 | 1.8 | 1.0 | 0.2 |
| Friction       | 0.3 | 0.9 | 0.7 | 0.2 | 1.0 |
| Camera zoom    | 1.10 | 0.95 | 0.85 | 1.20 | 0.90 |

### Visual decoder (sparse, axis-aligned)

- `c_1` aberration ← Arousal + Overload.
- `c_2` grain ← Tactical + Apathy.
- `c_3` warp ← Flow.
- `c_4` bloom ← Flow + Arousal.

### Boss AI mapping (interprets `pi`, never replaces backend)

| Dominant π state | Boss behaviour |
|---|---|
| Arousal  | fast aggressive pursuit, ideal range collapses toward 0 |
| Tactical | holds ideal range ~4.5 units, precise movement |
| Overload | summons 1–3 minions on cooldown |
| Flow     | teleports to flank the player every ~2.5 s |
| Apathy   | slow, passive (emerges as low aggression values) |

### Enemy-kind weighting

```
w_chaser = 1.0 + 1.5 * pi[OVERLOAD]
w_fast   = 0.5 + 2.0 * pi[TACTICAL] + 0.6 * pi[AROUSAL]
w_tank   = 0.3 + 1.8 * pi[APATHY]
```

---

## 6. Non-Causality Justification (unchanged)

A standard HMM computes `P(pi_t | o_{1:t})`. Our window is
`[history (16) ‖ future-projection (16)]`, built by forward-integrating
the player's current momentum. The attention softmax mixes past and
future rows when producing `z_t`, so `pi_t` depends on both `o_{<=t}`
and `hat o_{t+1:t+16}` — a bidirectional smoother.

Projection rules:
- `v_norm`: decay 0.98/frame on velocity magnitude.
- `apm, dir_variance, threat`: decay 0.90/frame.
- `idle`: grows if projected `v_future < eps`, else decays.

---

## 7. Controls & Scaling Rules

| Key   | Action |
|-------|--------|
| WASD  | Move |
| SPACE | Dash (4× impulse, 0.2 s i-frames, 1.5 s baseline CD) |
| 1/2/3 | Pick shop upgrade |
| F1    | Toggle debug telemetry in stdout |
| ENTER | Start run / return to menu from game-over |
| ESC   | Quit |

**Scaling (applied on top of E, never replacing):**

```
enemy_speed = E[0] * (1 + 0.10 * room) * run.enemy_slow
spawn_rate  = E[1] * (1 + 0.15 * room)
friction    = E[2]         # used directly
cam_zoom    = E[3]         # used directly
```

---

## 8. Run & Verify

```bash
source ~/venv/bin/activate
pip install -r requirements.txt       # once
python flow_game.py                   # fullscreen game (default)
python flow_game.py --windowed        # windowed (1280x800)
python flow_game.py --probe           # headless invariant check
python flow_game.py --probe --frames 1000
```

GPU is used automatically when `torch.cuda.is_available()`.

Headless probe asserts each frame: `pi.shape == (5,)`,
`|sum(pi)-1| < 1e-4`, `pi ∈ [0,1]`, `E.shape == (4,)`, `c.shape == (4,)`.

Interactive behaviour expectations (60–144 FPS):
- **Idle 5 s** → Apathy rises → friction high, enemies sluggish, grain fades in.
- **Keyboard mashing** → Arousal/Overload rise → chromatic aberration + screen shake, spawn rate climbs.
- **Smooth circular movement 10 s** → Flow rises → screen warps sinusoidally, bloom lifts, friction drops.
- **Boss room** → big central SDF disc with HP ring; kill with dash contacts.
- **Shop** → 3 random upgrades; pick with 1/2/3.

---

## 9. Known Limitations

- Weights are seeded, not trained. `W_Q, W_K, W_V` are uniform-random;
  `W_pi, W_c1, W_c2, M` are hand-constructed.
- β-VAE disentanglement is structural (sparse `W_c2`), not learned.
- Enemy cap = 16 (must match `u_enemies[16]` in the shader).
- `pygame.draw.rect` alpha may not dim the shop/game-over background on
  some drivers — worst case the overlay is just slightly less opaque.
- Screen-space shake adds aliasing at very high `c1`; that's the
  intended "stress" signal.
- GPU usage is mostly symbolic — the backend tensors are so small that
  CPU is actually faster on most systems; the `DEVICE` autodetect is
  there to honour the "use GPU if needed" request, not as a perf win.
