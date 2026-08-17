# Flow Game — Psychological Adaptation Engine + Roguelite Arena

## **Ideated at 3AM by Atharv Sharma | Shoyam Mishra**
## **RL Agent authored by Atharv Sharma**

### ▶ **[Play it in your browser — no install](https://atharvgaur1845.github.io/flow_game/)**

![Flow Game Gameplay](simplescreenrecorder-2026-04-20_04.16.07%20(online-video-cutter.com).gif)

A 2D roguelite arena that **adapts in real-time to your psychological state** using a non-causal attention model and disentangled visual latents. It ships three ways:

| | What it is |
|---|---|
| **[Web build](https://atharvgaur1845.github.io/flow_game/)** | The whole game in a browser tab — WebGL2, procedural adaptive audio, everything saved locally. No install, no server. |
| **Desktop build** | The original Python / PyTorch / OpenGL version. |
| **RL agent** | A **hierarchical reinforcement learning agent** that learns to play in any of the five psychological archetypes on command. |

> The web build is not a reimplementation. The attention weights are dumped straight from the PyTorch module (`torch.manual_seed(1337)`) into [docs/js/weights.js](docs/js/weights.js), so the browser infers the **same π** as the desktop game — verified to 1.4 × 10⁻⁷ by [tools/verify_port.mjs](tools/verify_port.mjs).

---

## 📋 Table of Contents

### Web Build
- [Play / Deploy](#-web-build)
- [What the Web Build Adds](#-what-the-web-build-adds)
- [Adaptive Score](#-the-adaptive-score)
- [Local Data](#-local-data)
- [Web Architecture](#-web-architecture)
- [Verifying the Port](#-verifying-the-port)

### Game
- [Overview](#-overview)
- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Game Mechanics](#-game-mechanics)
- [Controls](#-controls)
- [Architecture](#-architecture)
- [Configuration](#-configuration)
- [The Math](#-understanding-the-math)
- [Visual Aesthetic](#-visual-aesthetic)
- [System Requirements](#-system-requirements)

### RL Agent
- [RL Overview](#-rl-agent-overview)
- [RL Architecture](#-rl-architecture)
- [Observation Space](#-observation-space)
- [Action Space](#-action-space)
- [Reward Design](#-reward-design)
- [Hierarchical Training](#-hierarchical-training)
- [Training Guide](#-training-guide)
- [Watching the Agent](#-watching-the-agent)
- [RL Configuration](#-rl-configuration)
- [Troubleshooting](#-troubleshooting)

---

# PART 0 — THE WEB BUILD

---

## 🌐 Web Build

The entire game runs in a browser tab: WebGL2 for the arena, the attention model in plain JavaScript, and a fully synthesised soundtrack. **No build step, no bundler, no dependencies, no server** — `docs/` is 224 KB of static files and there is not a single asset to download.

### Play

**<https://atharvgaur1845.github.io/flow_game/>**

### Deploy it yourself

Settings → Pages → Source: *Deploy from a branch* → Branch: `main`, folder: **`/docs`**. That is the whole deployment.

### Run it locally

ES modules need a real origin, so `file://` will not work:

```bash
cd docs && python3 -m http.server 8000
# then open http://localhost:8000
```

Append `?debug` to the URL to expose the live game object as `window.__flow`.

### Requirements

WebGL2 (Chrome/Edge 56+, Firefox 51+, Safari 15+). The page detects a missing context and says so instead of showing a black screen.

---

## ✨ What the Web Build Adds

Everything below is additive — the simulation itself is a faithful port, tick for tick.

| Feature | Why |
|---------|-----|
| **Attract mode** | The title screen is a live game. A scripted autopilot sweeps deliberately through all five archetypes, so the arena visibly changes colour and the music changes character before you press anything. |
| **π ribbon** | A scrolling stacked-area river of your last 20 seconds of inferred state, in its own band below the arena. Five bars give you a number; the ribbon gives you the shape of the fight. |
| **Diegetic dash ring** | The dash cooldown orbits the player and flares when it lands, instead of living in a corner bar you have no spare attention for. |
| **Low-HP vignette** | Pulses in time with the heartbeat cue. |
| **Off-screen threat markers** | At Flow's 1.2× zoom a real slice of the arena sits outside the frustum, so enemies out there ride the screen edge. |
| **Contextual onboarding** | Movement keys fade in until you move; the dash prompt appears the first time something gets close and never again. No tutorial, no wall of text. |
| **Run report** | Your whole run's π rendered as one image — a fingerprint of how it actually felt — plus score vs. your own median, and a Wordle-style **run card** for the clipboard. |
| **Psych profile** | Lifetime seconds per archetype across every session you have ever played, surfaced on the menu as *"You play like: TACTICAL."* |
| **Milestones** | 14 one-shot achievements as neon toasts mid-run. |
| **Accessibility** | Reduced motion (kills shake, aberration, grain and warp), a colourblind-safe Okabe-Ito palette, split music/SFX volume, rebindable keys. |
| **Touch controls** | A floating thumbstick that origins wherever your thumb lands, plus a dash pad. |
| **Auto quality** | Frame cost is measured continuously and render resolution is trimmed before anything stutters, with a manual Low/Medium/High/Ultra override. |

### Fixed-timestep simulation

The desktop build steps once per rendered frame at up to 144 Hz, and several gameplay lerps (enemy steering, APM decay, the EMA baseline) are *per-frame* rather than per-second. Browsers hand you anything from 30 to 240 Hz, so the web build runs the simulation on a fixed **120 Hz tick** with free-running rendering. A 60 Hz laptop and a 240 Hz monitor play the same game.

---

## 🎵 The Adaptive Score

There are no audio files. Every sound is synthesised at runtime through the Web Audio API — which is also why the page loads instantly and works offline once cached.

The music is not a loop with a filter on it. Five musical identities are **crossfaded continuously by π**, the same mixture that drives the physics and the shader. Tempo, scale, timbre, filter cutoff, reverb length and note density are all functions of your inferred psychological state:

| State | Musical identity |
|-------|-----------------|
| **Arousal** | Phrygian, 150 BPM driving pulse, saw lead, filter wide open |
| **Tactical** | Bare fourths, 110 BPM, metronomic, tight room |
| **Overload** | Tritone cluster, 132 BPM, heavy detune, stuttering hats |
| **Flow** | Sus2 pads, 96 BPM, long reverb, patient arp |
| **Apathy** | Minor triad, 60 BPM, everything lowpassed to a murmur |

The dominant archetype selects the scale, but only after it has held for ~1.2 s — otherwise the melody rewrites itself several times a second. Everything else blends smoothly. Taking damage sidechain-ducks the whole bed.

On top of that sits a full effects layer: dash whoosh, per-enemy-type kill blips, damage thud, five distinct pickup arpeggios, an FM-synthesised boss clang, room-clear chord, dash-ready click and a low-HP heartbeat.

**Audio unlocks on the same keypress that starts your first run** — which is why there is no "click to enable sound" nag anywhere in the UI.

---

## 💾 Local Data

`localStorage` is the entire backend. Nothing is uploaded, there is no analytics, and there is no network request after the page loads.

| Namespace | Contents |
|-----------|----------|
| `settings` | Volumes, mute, quality, auto-scale, reduced motion, colourblind palette, keybinds |
| `records` | Best score, deepest room, lifetime kills, runs played, seconds played, bosses, longest Flow streak |
| `profile` | Lifetime seconds accumulated in each of the five archetypes |
| `runs` | Last 20 run summaries — feeds the menu sparkline and the "vs. median" comparison |
| `seen` | One-shot flags: onboarding hints and unlocked milestones |

All of it is stored under one versioned key (`flow.web.v1`), and **Settings → Erase All Local Data** wipes it. If the browser blocks storage (private mode, embedded contexts) the game plays normally and says so in Settings.

---

## 🏗️ Web Architecture

```
docs/
├── index.html            ← page shell: HUD, screens, touch layer
├── style.css             ← single stylesheet, no external fonts
└── js/
    ├── main.js           ← entry point, fixed-timestep loop, state machine
    ├── config.js         ← port of game/config.py + backend symbols
    ├── weights.js        ← AUTO-GENERATED tensors dumped from PyTorch
    ├── inference.js      ← AttentionStateInference / EnvironmentMapper / ObservationBuilder
    ├── entities.js       ← port of game/entities_extra.py
    ├── progression.js    ← port of game/progression.py
    ├── player.js         ← Player dataclass + game/abilities.py
    ├── renderer.js       ← WebGL2, GLSL ES 3.00 port of the fragment shader
    ├── audio.js          ← procedural SFX + the π-driven adaptive score
    ├── ribbon.js         ← the π ribbon (live strip + run fingerprint)
    ├── hud.js            ← in-run HUD
    ├── ui.js             ← title, shop, report, settings, milestones
    ├── input.js          ← rebindable keyboard + touch
    ├── storage.js        ← localStorage persistence
    ├── milestones.js     ← achievement definitions
    └── runcard.js        ← shareable run card
```

The **same layer contract** as the desktop build holds: `inference.js` is the backend and is never written to by gameplay; everything else only *reads* π, E and c.

---

## 🔬 Verifying the Port

The web build's inference is not a lookalike — it is the same model, and that is checked rather than asserted.

```bash
# Regenerate the weights from the live PyTorch modules
python3 tools/dump_weights.py > docs/js/weights.js

# Prove the JS matches torch on a 120-frame golden trace
python3 tools/dump_trace.py > /tmp/trace.json
node tools/verify_port.mjs /tmp/trace.json
#   frames        : 120
#   max |dpi|     : 1.398e-7
#   OK — JS inference matches the torch backend.

# Headless smoke test of the whole ported game loop
node tools/sim_smoke.mjs
```

`max |dπ| = 1.4e-7` is float32-vs-float64 noise; there is no algorithmic drift.

`sim_smoke.mjs` runs the ported simulation with a scripted player and asserts the same invariants `flow_game.py --probe` does (π on the simplex, E and c finite) plus gameplay ones: the arena is inescapable, the enemy and trail caps hold, rooms advance, bosses spawn on room 5, and shops never offer duplicates.

---
---

# PART I — THE GAME

---

## 🎮 Overview

Flow Game is a roguelite arena survival game where the environment **adapts in real-time to your psychological state**. Every keypress, every movement, every hesitation is fed into a bidirectional attention model that infers which of five archetypes describes you right now — and the game world reshapes itself accordingly.

- **Real-time psychological state inference** — bidirectional attention over 32-frame window
- **Emergent gameplay** — responds to arousal, tactical thinking, cognitive overload, flow, or apathy
- **Neon aesthetic** — mathematically-driven visuals via GLSL fragment shaders
- **Dynamic enemy distribution** — spawn mix is biased by your inferred state
- **Adaptive boss AI** — boss reads your `pi` mixture and changes tactics mid-fight
- **Roguelite progression** — shops, upgrades, escalating difficulty

### The Five Archetypes

The backend tracks your play as a probability distribution over five psychological states:

| Index | Archetype | Colour | Triggered By |
|-------|-----------|--------|-------------|
| 0 | **Arousal** | Ember red | Fast movement, high APM, close threats |
| 1 | **Tactical** | Deep blue | Precise positioning, measured action rate |
| 2 | **Overload** | Violet | Erratic direction changes, enemy swarms |
| 3 | **Flow** | Teal | Smooth velocity, consistent engagement, rhythm |
| 4 | **Apathy** | Grey | Low speed, low APM, long idle periods |

This mixture `π` (sums to 1.0) continuously drives physics, visuals, enemy behaviour, and boss AI — without any scripted modes or difficulty levels.

---

## ✨ Features

### Backend — Inference Engine
- **Non-Causal Attention** — bidirectional attention over 16 history frames + 16 physics-projected future frames
- **EMA Baseline Tracking** — adapts to each player's individual baseline over time
- **5-State Probability Simplex** — smooth, stable transitions; never jumps between states
- **Deterministic Environment Mapper** — `π → (E physics params, c visual latents)` via hand-tuned matrix
- **Live State Bars** — five colour-coded bars visible in the HUD at all times

### Gameplay
- **3 Enemy Archetypes**
  - *Chaser* — 2 HP, medium speed, balanced threat
  - *Fast* — 1 HP, 2× speed, punishes loose positioning
  - *Tank* — 4 HP, 0.55× speed, arena presence
- **Boss Encounters** — every 5 rooms; 220 HP; only dash damage counts
- **5 Pickup Types** — heal, dash boost, speed boost, shield, max HP
- **Lethal Dash Trail** — segments persist 1.6 s; kill enemies on contact
- **Shop System** — every 3 rooms; 5 upgrade types
- **Scoring** — time-based + kill-based, both amplified by `π[Flow]`

### Visuals
- Procedural animated grid, tinted by archetype mixture
- SDF circles for all entities, glow driven by bloom latent
- Dash effects: echo trail, spinning blade, ripple ring
- Post-process: chromatic aberration, film grain, UV warp, bloom — all derived from `c`
- Screen shake amplitude driven by arousal

---

## 📦 Installation

### Prerequisites
- Python 3.8+
- OpenGL 3.3-capable GPU (integrated is fine for the game)
- CUDA-compatible NVIDIA GPU recommended for RL training

### Step 1 — Clone / Navigate
```bash
cd /path/to/flow_game
```

### Step 2 — Install Dependencies
```bash
pip install -r requirements.txt
```

Installs: `pygame`, `PyOpenGL`, `PyOpenGL_accelerate`, `torch`, `numpy`

### Step 3 — CUDA (for RL training)
```bash
# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 🚀 Quick Start

> Nothing to install if you just want to play: **<https://atharvgaur1845.github.io/flow_game/>**

```bash
# Fullscreen (default)
python flow_game.py

# Windowed
python flow_game.py --windowed

# Headless invariant check (no window)
python flow_game.py --probe
```

### First Run
1. **ENTER** — start run from menu
2. **WASD** — move through the arena
3. **SPACE** — dash (invulnerability + lethal trail)
4. Collect rotating diamond pickups for buffs
5. Kill required enemies to clear each room
6. **1/2/3** at shops to pick upgrades
7. Every 5th room: boss fight — only dashes deal damage

---

## 🎯 Game Mechanics

### Psychological State Inference

Five features are extracted each frame and fed to the attention model:

| Feature | Source | Drives |
|---------|--------|--------|
| `v_norm` | Movement speed magnitude | Arousal ↑ |
| `apm_proxy` | Exponentially-decayed keypress rate | Arousal / Tactical ↑ |
| `dir_variance` | Circular variance of recent headings | Overload ↑ |
| `threat` | `1 / (1 + min_enemy_dist)` | Arousal / Tactical ↑ |
| `idle_time` | Normalised stationary frames (0–120) | Apathy ↑ |

The 32-frame non-causal attention produces `π = softmax(W_π @ mean(Z) + b_π)`.

### Physics Parameters (M Matrix)

`E = M @ π + noise`

| Parameter | Arousal | Tactical | Overload | Flow | Apathy |
|-----------|---------|----------|----------|------|--------|
| Enemy speed mult | 1.8 | 0.6 | 1.4 | 1.2 | 0.3 |
| Spawn rate mult | 1.5 | 0.4 | 1.8 | 1.0 | 0.2 |
| Player friction | 0.3 | 0.9 | 0.7 | 0.2 | 1.0 |
| Camera zoom | 1.10 | 0.95 | 0.85 | 1.20 | 0.90 |

### Visual Latents (c)

```
c = W_c2 @ ReLU(W_c1 @ π)
```

| Latent | Driven By | Shader Effect |
|--------|-----------|---------------|
| `c[0]` | Arousal + Overload | Chromatic aberration |
| `c[1]` | Tactical + Apathy | Film grain |
| `c[2]` | Flow | Sinusoidal UV warp |
| `c[3]` | Flow + Arousal | Additive bloom / glow |

### Boss AI

| Your Dominant State | Boss Behaviour |
|--------------------|----------------|
| Arousal | Fast aggressive pursuit; minimal ideal distance |
| Tactical | Precise range warfare; hovers ~4.5 units away |
| Overload | Spawns 1–3 minions every 4–6 seconds |
| Flow | Teleport-flanks every 2–3 seconds |
| Apathy | Slow, passive; low aggression |

### Room Progression

| Metric | Formula |
|--------|---------|
| Normal room time limit | `30 + room × 2` seconds |
| Kills required | `8 + room × 3` |
| Enemy speed scaling | `E[0] × (1 + 0.15 × room)` |
| Spawn rate scaling | `E[1] × (1 + 0.22 × room)` |
| Boss room time limit | `60 + room × 4` seconds |

> **Rooms clear by kills only.** Running out of time without meeting the kill quota counts as death.

### Scoring

```
kill_score   = 25 × (1.0 + 2.0 × π[Flow] + shop_bonus)
time_score   = 1.0 × (1.0 + π[Flow])  pts/sec
room_clear   = 250 pts
boss_clear   = 2000 pts
```

### Shop Upgrades

| Upgrade | Effect |
|---------|--------|
| More HP | +20 max HP, full heal |
| Shorter Dash | Cooldown ×0.8 (min 0.25 s) |
| Score Boost | +0.25 score multiplier bonus |
| Slow Enemies | Enemy speed ×0.85 |
| Damage Armor | Incoming damage ×0.8 |

---

## 🎮 Controls

| Key | Action |
|-----|--------|
| **WASD** | Move |
| **SPACE** | Dash |
| **1 / 2 / 3** | Shop upgrade |
| **ENTER** | Start / continue |
| **ESC** | Quit |
| **F1** | Debug overlay |

### Dash Details

| Property | Value |
|----------|-------|
| Cooldown | 2.5 s (baseline) |
| Invulnerability | 0.4 s |
| Minimum speed | 6.0 units/s |
| Impulse multiplier | 4.0 × |
| Trail TTL | 1.6 s |
| Trail kill radius | 0.55 units |
| Boss damage per dash | 14 HP |

---

## 🏗️ Architecture

```
flow_game/
├── flow_game.py          ← Backend inference engine + main game loop
├── game/
│   ├── config.py         ← All tunable gameplay constants
│   ├── game_state.py     ← GameState: MENU / PLAYING / SHOP / BOSS / GAME_OVER
│   ├── abilities.py      ← Dash logic and per-frame timer updates
│   ├── entities_extra.py ← Enemy, Boss, Pickup dataclasses + AI
│   ├── progression.py    ← RunState, room scaling, shop, scoring
│   └── ui.py             ← HUD, shop menu, game-over overlay
├── docs/                 ← The web build (GitHub Pages root) — see Part 0
├── tools/                ← Weight dumping + port verification
│   ├── dump_weights.py   ← PyTorch tensors -> docs/js/weights.js
│   ├── dump_trace.py     ← Golden (o_t -> pi) trace from the torch backend
│   ├── verify_port.mjs   ← Asserts the JS port matches torch
│   └── sim_smoke.mjs     ← Headless smoke test of the ported game loop
└── rl_agent/             ← See Part II
```

### Layer Contract

**Backend** (`flow_game.py`) — never modified by gameplay:
- `AttentionStateInference` — produces `π`
- `EnvironmentMapper` — maps `π → (E, c)`
- `ObservationBuilder` — computes the 5-feature observation vector

**Game layer** (`game/`) — only *reads* `π, E, c`:
- Scales outputs: `E' = E × (1 + room_scale)`
- Interprets outputs: "high `π[Overload]` → spawn more chasers"
- Never writes to backend state

---

## ⚙️ Configuration

All constants live in [game/config.py](game/config.py):

```python
# Player
PLAYER_MAX_HP   = 100
DASH_COOLDOWN   = 2.5       # seconds
DASH_INVULN     = 0.4       # i-frame window
DAMAGE_ON_HIT   = 15

# Rooms
ROOM_BASE_KILLS = 8
ROOM_KILL_STEP  = 3         # +per room
ENEMY_SCALE_SPEED = 0.15    # +15% per room
ENEMY_SCALE_SPAWN = 0.22    # +22% per room

# Boss
BOSS_HP             = 220
BOSS_DASH_DAMAGE    = 14

# Scoring
SCORE_KILL_BASE  = 25
SCORE_ROOM_CLEAR = 250
SCORE_BOSS_CLEAR = 2000
```

---

## 🔬 Understanding the Math

### The 5-Simplex

```
π_0 + π_1 + π_2 + π_3 + π_4 = 1.0,    0 ≤ π_k ≤ 1
```

Smooth, continuous transitions — no hard mode switches.

### Non-Causal Self-Attention

```
O  = [history(16) | future_projection(16)]   shape: (32, 5)
Q, K, V = O @ W_Q,  O @ W_K,  O @ W_V
A  = softmax( Q @ K^T / √d_k )
Z  = A @ V
π  = softmax( W_π @ mean(Z, dim=0) + b_π )
```

The future projection extrapolates velocity decay and idle growth — giving the model non-causal "foresight" without any lookahead into actual future inputs.

### Environment Mapping

```
E = M @ π + ε,    ε ~ N(0, 0.02²)
c = W_c2 @ ReLU(W_c1 @ π)
```

`M` is a fixed 4×5 hand-tuned matrix. `W_c1` and `W_c2` are sparse by design, ensuring each visual latent is driven by at most one or two archetypes (disentanglement).

---

## 🎨 Visual Aesthetic

| State | Background | Post-Process |
|-------|-----------|--------------|
| Arousal | Ember red grid | Strong aberration, screen shake |
| Tactical | Deep navy blue | Clean, minimal distortion |
| Overload | Chaotic violet | Heavy grain + sinusoidal warp |
| Flow | Glowing teal | Smooth bloom, slight zoom |
| Apathy | Muted grey | Flat, low contrast |

---

## 🔧 System Requirements

| | Minimum | Recommended |
|-|---------|-------------|
| CPU | i5 / Ryzen 5 | i7 / Ryzen 7 |
| RAM | 4 GB | 8 GB |
| GPU | OpenGL 3.3 (integrated OK) | NVIDIA RTX 3060+ (CUDA) |
| Python | 3.8+ | 3.10+ |
| OS | Linux / macOS / Windows 10 | Ubuntu 22.04 / Windows 11 |

---
---

# PART II — THE RL AGENT

> **Author: Atharv Sharma**

---

## 🤖 RL Agent Overview

A custom **hierarchical reinforcement learning agent** trained with hand-written PPO (no external RL library) that learns to play the game while maintaining a specified psychological archetype.

**Core idea**: the game's inference engine already measures `π` from the player's behaviour. The RL agent is trained to simultaneously **maximise game score** and **keep `π[target]` high** — so telling it "play in Flow state" produces an agent that moves smoothly, engages consistently, and scores efficiently, because that is exactly what the game's psychological model measures as Flow.

### What it can do

- Be told "play in Flow / Arousal / Tactical / Overload / Apathy" and exhibit the corresponding behaviour
- Clear rooms, fight bosses, and make shop decisions autonomously
- Switch target archetypes mid-run as difficulty changes
- Be watched in real-time while training continues in a separate process
- Scale from a single RTX 4060 to an A6000 by changing one config value

---

## 🏗️ RL Architecture

```
flow_game/
└── rl_agent/
    ├── config.py          ← All RL hyperparameters (single source of truth)
    ├── env.py             ← Headless FlowGameEnv — full game logic, no rendering
    ├── rewards.py         ← Reward function (score + alignment + survival)
    ├── models.py          ← LowLevelPolicy and MetaPolicy neural networks
    ├── ppo.py             ← RolloutBuffer + PPOTrainer (GAE, clipped objective)
    ├── hierarchical.py    ← HierarchicalTrainer + ParallelEnvs (worker processes)
    ├── train.py           ← CLI training entry point
    ├── watch.py           ← Live renderer — watches the agent play during training
    └── evaluate.py        ← Post-training evaluation
```

### Two-Level Hierarchy

```
┌─────────────────────────────────────────────────┐
│  META-POLICY  (selects every 200 low-level steps) │
│  Input : 16-dim meta-observation                  │
│  Output: target archetype (0–4)                   │
│  Network: MLP 16 → 128 → 128 → 5                 │
└───────────────────┬─────────────────────────────┘
                    │  target archetype (one-hot)
                    ▼
┌─────────────────────────────────────────────────┐
│  LOW-LEVEL POLICY  (acts every game step)         │
│  Input : 119-dim observation                      │
│  Output: move (9) + dash (2) + shop (3)           │
│  Network: MLP 119 → 256 → 256 → heads            │
└─────────────────────────────────────────────────┘
                    │  action
                    ▼
         FlowGameEnv (headless)
```

Both policies are trained concurrently with independent PPO instances, sharing no weights.

### LowLevelPolicy

```
Shared trunk:  Linear(119, 256) → LayerNorm → ReLU
               Linear(256, 256) → LayerNorm → ReLU
                    │
         ┌──────────┼──────────┐
    move head   dash head   shop head   value head
    (→ 9)       (→ 2)       (→ 3)       (→ 1)
```

During play: `log_prob = log_prob(move) + log_prob(dash)`.  
During shop: `log_prob = log_prob(shop)`.  
The active head is selected by the `is_shop` flag embedded in the observation.

### MetaPolicy

```
Trunk:  Linear(16, 128) → ReLU → Linear(128, 128) → ReLU
               │
      ┌────────┴────────┐
  arch head (→ 5)   value head (→ 1)
```

---

## 📐 Observation Space

**119-dimensional float32 vector** — always the same shape regardless of game state.

| Slice | Dims | Content |
|-------|------|---------|
| `[0:5]` | 5 | Behavioural features: `v_norm, apm, dir_variance, threat, idle_time` (same 5 the game's inference engine uses) |
| `[5:13]` | 8 | Player: `pos_x, pos_y, vel_x, vel_y, hp_norm, dash_cd_norm, is_dashing, has_invuln` |
| `[13:17]` | 4 | Room: `time_remaining_norm, kills_remaining_norm, room_norm, is_boss` |
| `[17:22]` | 5 | Current `π` — live output of the game's inference engine |
| `[22:27]` | 5 | Target archetype one-hot — set by meta-policy |
| `[27]` | 1 | `is_shop` flag |
| `[28:43]` | 15 | Shop upgrade one-hots (3 slots × 5 upgrade types) |
| `[43:91]` | 48 | Nearest 8 enemies: `(dx, dy, is_chaser, is_fast, is_tank, hp_norm)` each |
| `[91:119]` | 28 | Nearest 4 pickups: `(dx, dy, is_heal, is_dash, is_speed, is_shield, is_maxhp)` each |

> The behavioural features `[0:5]` are the exact same inputs the game uses to infer `π`. This design means the agent can observe how its own behaviour looks to the psychological model — closing the loop between "what I do" and "what state the game thinks I'm in".

### Meta-Policy Observation (16-dim)

| Slice | Content |
|-------|---------|
| `[0:5]` | Current `π` |
| `[5]` | Score delta over last option period (normalised) |
| `[6]` | Room progress (normalised, max ~20) |
| `[7]` | Time remaining in room (normalised) |
| `[8]` | Kills remaining (normalised) |
| `[9]` | Player HP (normalised) |
| `[10:15]` | Previous target archetype (one-hot) |
| `[15]` | Padding |

---

## 🎯 Action Space

### During PLAYING / BOSS

| Head | Values | Meaning |
|------|--------|---------|
| **move** | 0–8 | 0=none, 1=up, 2=down, 3=left, 4=right, 5=↖ 6=↗ 7=↙ 8=↘ |
| **dash** | 0–1 | 0=no dash, 1=dash |

Combined log-probability: `log π(move) + log π(dash)`

### During SHOP

| Head | Values | Meaning |
|------|--------|---------|
| **shop** | 0–2 | Choose upgrade slot 1, 2, or 3 |

The network always computes all three heads; the env flags which is active via `is_shop` in the observation.

---

## 🏆 Reward Design

Per-step reward:

```
reward = score_delta / 500.0
       + λ[target] × π[target]
       + 0.005  (survival bonus per step)
       − 1.0    (death penalty, terminal steps only)
```

`score_delta` is the score gained since the previous step.  
`π[target]` is the game's current inference of how well behaviour matches the target archetype.

### Per-Archetype λ Values

| Archetype | λ | Reasoning |
|-----------|---|-----------|
| Arousal | 0.3 | Score already rewards aggression; alignment weight kept lower |
| Tactical | 0.4 | Balanced; positioning pays off directly |
| Overload | 0.3 | Hard to maintain deliberately; lower weight reduces frustration |
| **Flow** | **0.5** | Highest weight — the primary target; score and alignment reinforce each other |
| Apathy | 0.5 | Hardest to maintain (game punishes passivity); higher weight needed |

### Meta-Policy Reward

```
meta_reward = sum of low-level rewards over the 200-step option period
```

The meta-policy is rewarded for picking archetypes that lead to high cumulative performance over a ~7-second window.

---

## 🔄 Hierarchical Training

### Option Framework

```
Every 200 low-level steps (~6.7 simulated seconds):

  1. Meta-policy observes meta_obs (16-dim)
  2. Meta-policy selects target archetype a ∈ {0,1,2,3,4}
  3. For 200 steps:
       obs includes target one-hot
       low-level policy acts → (move, dash, shop)
       reward = score_delta/500 + λ[a] × π[a] + survival
  4. meta_reward = sum(rewards over 200 steps)
  5. Store (meta_obs, a, meta_reward, ...) in meta buffer
```

### PPO Parameters

| | Low-Level | Meta-Policy |
|-|-----------|-------------|
| Rollout steps | 2048 / env | 10,240 / env |
| Epochs | 4 | 4 |
| Minibatches | 8 | 4 |
| Learning rate | 3e-4 | 1e-4 |
| Entropy coeff | 0.01 | 0.05 |
| GAE λ | 0.95 | 0.95 |
| PPO clip ε | 0.2 | 0.2 |
| Gradient clip | 0.5 | 0.5 |

Meta entropy is higher (0.05) to encourage the policy to explore all five archetypes rather than collapsing to one.

### Parallelism

```python
# rl_agent/config.py
NUM_ENVS = 8     # RTX 4060 8 GB  — change this one line to scale
NUM_ENVS = 64    # A6000 48 GB
```

Each env runs in a separate OS process (`spawn` context, CPU-only) and communicates via `multiprocessing.Pipe`. The main process runs policy inference and PPO updates on GPU. Rollout buffers live as numpy arrays on CPU; they are moved to GPU only during the PPO update.

**Memory usage**: ~140 MB VRAM. Models are tiny (~120k parameters total). The CUDA runtime itself accounts for ~100 MB of that figure.

---

## 🚀 Training Guide

### Start Training (RTX 4060, with live watcher window)

```bash
/home/atharv/venv/bin/python3 -m rl_agent.train \
    --num_envs 8 \
    --watch \
    --watch_windowed
```

The `--watch` flag:
1. Runs a 2048-step warm-up to create an initial checkpoint
2. Opens a live game window rendering the agent's play
3. Continues full training; the window auto-reloads every 15 s

### Scale to A6000

```bash
/home/atharv/venv/bin/python3 -m rl_agent.train \
    --num_envs 64 \
    --total_steps 50_000_000
```

### Resume from Checkpoint

```bash
/home/atharv/venv/bin/python3 -m rl_agent.train \
    --num_envs 8 \
    --resume rl_agent/checkpoints/checkpoint_step5000000.pt
```

### Short Smoke Run (verify setup)

```bash
FLOW_HEADLESS=1 /home/atharv/venv/bin/python3 -m rl_agent.train \
    --num_envs 1 \
    --total_steps 4096
```

### Training Output

```
Device  : cuda
Envs    : 8
Steps   : 10,000,000

step=    2,048  eps=  0  sps=  233  ll_pg=-0.0063  ll_vf=0.165  ll_ent=2.882
step=    4,096  eps=  0  sps=  394  ll_pg=-0.0042  ll_vf=0.269  ll_ent=2.879
...
[checkpoint] saved rl_agent/checkpoints/checkpoint_step100000.pt
```

| Field | Meaning |
|-------|---------|
| `sps` | Environment steps per second across all workers |
| `ll_pg` | Low-level PPO policy gradient loss |
| `ll_vf` | Value function loss |
| `ll_ent` | Entropy (higher = more exploration) |
| `meta_pg` | Meta-policy gradient loss (appears after first meta update) |

Checkpoints are saved every 100,000 steps and at completion.

### What to Expect

| Steps | Expected Behaviour |
|-------|--------------------|
| 0–50k | Random movement; occasional room clears |
| 50k–200k | Learns to move and dash consistently; reaches room 3–5 |
| 200k–1M | Learns to fight; room 5–10; boss engagement starts |
| 1M–5M | Archetype alignment emerges; Flow agent becomes smooth |
| 5M–10M | Strong specialisation per archetype; room 10+ reliably |

---

## 👁️ Watching the Agent

### Standalone Watcher

Open a second terminal while training runs:

```bash
/home/atharv/venv/bin/python3 -m rl_agent.watch \
    --windowed \
    --reload_every 15
```

The watcher auto-detects and loads the newest checkpoint every 15 seconds.

### Watcher Keys

| Key | Action |
|-----|--------|
| **1–5** | Force target archetype (1=Arousal … 5=Apathy) |
| **R** | Reload latest checkpoint immediately |
| **Space** | Pause / unpause |
| **ESC** | Quit |

### HUD Overlay

The watcher renders the full game (all shaders, all post-processing) plus a bottom HUD strip showing:
- Live π bar for all five archetypes (target archetype is brighter)
- Score, HP, room number, current target name

### Post-Training Evaluation

```bash
/home/atharv/venv/bin/python3 -m rl_agent.evaluate \
    --checkpoint rl_agent/checkpoints/checkpoint_final.pt \
    --archetype flow \
    --windowed
```

Available archetypes: `arousal`, `tactical`, `overload`, `flow`, `apathy`

---

## ⚙️ RL Configuration

All hyperparameters are in [rl_agent/config.py](rl_agent/config.py):

```python
# --- Scale this to match your GPU ---
NUM_ENVS            = 8          # 8 for RTX 4060 / 64 for A6000

# --- Environment ---
SIM_DT              = 1 / 30.0   # 30 Hz simulation
MAX_EP_STEPS        = 50_000     # episode truncation
OBS_DIM             = 119

# --- Hierarchical ---
OPTION_PERIOD_K     = 200        # low-level steps per meta decision

# --- PPO (low-level) ---
LR_LOW              = 3e-4
LL_ROLLOUT_STEPS    = 2048
LL_N_EPOCHS         = 4
ENT_COEF_LL         = 0.01

# --- PPO (meta) ---
LR_META             = 1e-4
META_ROLLOUT_STEPS  = 10_240
ENT_COEF_META       = 0.05

# --- Reward ---
LAMBDA_ALIGN = {
    0: 0.3,   # Arousal
    1: 0.4,   # Tactical
    2: 0.3,   # Overload
    3: 0.5,   # Flow
    4: 0.5,   # Apathy
}
SCORE_NORM          = 500.0
SURVIVAL_BONUS      = 0.005
DEATH_PENALTY       = -1.0

# --- Training budget ---
TOTAL_TIMESTEPS     = 10_000_000
CHECKPOINT_EVERY    = 100_000
```

---

## 🐛 Troubleshooting

### Game Won't Start
```bash
# Verify dependencies
pip list | grep -E 'pygame|torch|numpy|OpenGL'

# Check OpenGL version
python -c "from OpenGL.GL import glGetString, GL_VERSION; import pygame; pygame.init(); pygame.display.set_mode((1,1), pygame.OPENGL); print(glGetString(GL_VERSION))"
```

### RL Training Hangs at Startup
Workers use `spawn` (not `fork`) to avoid CUDA conflicts. First startup takes 10–20 s while worker processes import torch. If it hangs beyond 60 s:
```bash
# Test env in isolation
FLOW_HEADLESS=1 /home/atharv/venv/bin/python3 -m rl_agent.env
```

### VRAM is Only ~140 MB — Is That Right?
Yes. Models are ~120k parameters (<1 MB of weights). PyTorch's CUDA runtime requires ~100 MB overhead regardless of model size. Rollout buffers are numpy on CPU.

### Agent Stuck at Room 1
Normal before 50k steps. Check the env smoke test to confirm game logic is working:
```bash
FLOW_HEADLESS=1 /home/atharv/venv/bin/python3 -m rl_agent.env
# Should print: Smoke test OK: 200 steps, total_reward=...
```

### Watcher Shows Black Screen
The watcher needs a display. If running over SSH:
```bash
export DISPLAY=:0
/home/atharv/venv/bin/python3 -m rl_agent.watch --windowed
```

### "No checkpoint found" on Watcher Launch
Either run `--watch` with train (auto warm-up), or do a short training run first:
```bash
FLOW_HEADLESS=1 /home/atharv/venv/bin/python3 -m rl_agent.train \
    --num_envs 1 --total_steps 2048
```

---

## 📊 Symbol Reference

### Game Backend

| Symbol | Meaning | Value |
|--------|---------|-------|
| `D` | Observation dimensionality | 5 |
| `K` | Archetype count | 5 |
| `W_HIST` | History frames | 16 |
| `W_FUT` | Future-projection frames | 16 |
| `W_TOTAL` | Total attention window | 32 |
| `D_K` | Attention inner dimension | 16 |
| `P` | Environment physics parameters | 4 |
| `V_LATENT` | Visual latent dimensions | 4 |
| `EMA_ALPHA` | EMA baseline decay | 0.02 |
| `WORLD_HALF` | Arena half-size (units) | 16.0 |

### RL Agent

| Symbol | Meaning | Value |
|--------|---------|-------|
| `OBS_DIM` | Low-level observation size | 119 |
| `META_OBS_DIM` | Meta-policy observation size | 16 |
| `N_ARCHETYPES` | Number of psychological archetypes | 5 |
| `OPTION_PERIOD_K` | Low-level steps per meta decision | 200 |
| `SIM_DT` | Simulated seconds per env step | 1/30 |
| `GAMMA` | Discount factor | 0.99 |
| `GAE_LAMBDA` | GAE smoothing | 0.95 |
| `CLIP_EPS` | PPO surrogate clip | 0.2 |

---

## 🚀 Future Ideas

### Game
1. **Doppelgänger Boss** — a ghost copy with mirrored `π`
2. **Hivemind Swarms** — coordinated enemy formations driven by Overload
3. **Time Dilation** — bullet-time when Flow crosses 0.8
4. **Fractured Reality** — parallel arena overlaid at high Overload
5. **Archetype Loadouts** — unlock unique abilities per dominant state

### RL Agent
1. **Self-play adversarial training** — one agent as the game, one as the player
2. **Curiosity-driven exploration** — intrinsic reward for novel `π` states
3. **Multi-task fine-tuning** — warm-start from one archetype to transfer to others
4. **Imitation learning warm-up** — record human play per archetype as BC seed
5. **Online adaptation** — meta-policy updates in real-time during deployment

---

## 📝 Credits

**Flow Game** — *Ideated at 3AM by Atharv Sharma & Shoyam Mishra*

**RL Agent** — *Atharv Sharma*

Built with:
- PyTorch — inference engine + RL training
- Pygame — windowing, input, HUD
- PyOpenGL — GPU shader rendering
- GLSL — fragment shader (procedural scene)
- NumPy — simulation numerics

---

**Enjoy the Flow.** 🌊✨

*Last Updated: May 2026*
