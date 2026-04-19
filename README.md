# Flow Game — Psychological Adaptation Engine + Roguelite Arena

A revolutionary 2D roguelite game that dynamically adapts to your psychological state in real-time using a **non-causal attention model** and **disentangled visual latents**. Built with PyTorch, Pygame, and advanced GLSL shaders.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Game Mechanics](#game-mechanics)
- [Controls](#controls)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [System Requirements](#system-requirements)

---

## 🎮 Overview

Flow Game is a unique roguelite arena survival game where the environment and difficulty **adapt in real-time to your psychological state**. The game features:

- **Real-time psychological state inference** using a bidirectional attention model
- **Emergent gameplay** that responds to your playing style (arousal, tactical thinking, cognitive overload, flow state, or apathy)
- **Neon aesthetic** with mathematically-driven visuals powered by GLSL shaders
- **Dynamic enemy types** that spawn based on your inferred state
- **Boss encounters** with AI behavior driven by your psychology
- **Roguelite progression** with shops, upgrades, and escalating difficulty

### The Psychology Behind the Game

The backend tracks 5 psychological archetypes:
1. **Arousal (ARCH_AROUSAL)** — Kinetic energy; high movement/actions per minute
2. **Tactical (ARCH_TACTICAL)** — Calculated deliberation; precise positioning and strategy
3. **Overload (ARCH_OVERLOAD)** — Cognitive saturation; system overwhelm
4. **Flow (ARCH_FLOW)** — Optimal engagement; seamless skill-challenge balance
5. **Apathy (ARCH_APATHY)** — Disengagement; low input activity

Your playstyle automatically shifts this 5-dimensional mixture (`pi`), which then transforms:
- **Physics parameters** (enemy speeds, spawn rates, friction, camera zoom)
- **Visual distortions** (chromatic aberration, grain, geometric warp, bloom glow)
- **Boss AI behavior** (pursuit vs. tactical positioning vs. minion summoning vs. teleportation)
- **Enemy type distribution** (fast strikers, tanks, chasers appear based on your state)

---

## ✨ Features

### Backend Features (Inference Engine)
- **Non-causal Attention Model** — Bidirectional attention over historical keypresses (16 frames) + physics-projected future (16 frames)
- **EMA Baseline Tracking** — Continuous adaptation to player's shifting baseline
- **5-State Mixture Simplex** — Probabilistic inference of psychological state
- **Environment Mapping** — Deterministic mapping of `pi` → physics (`E`) and visual latents (`c`)
- **Real-time State Visualization** — 5 color-coded bars showing current archetype mixture

### Gameplay Features
- **3 Enemy Archetypes**
  - *Chaser*: Medium HP, medium speed, balanced threat
  - *Fast*: Low HP, high speed, punishes loose play
  - *Tank*: High HP, low speed, soaks damage
- **Dynamic Boss Encounters** (every 5 rooms)
  - Adaptive AI driven by player's inferred state
  - Arousal → aggressive pursuit
  - Tactical → precise range warfare
  - Overload → minion summoning
  - Flow → teleport flanking
  - Apathy → passive slow threat
- **Power-up System** (5 pickup types)
  - Health restoration
  - Dash burst (extended duration + cooldown reduction)
  - Speed boost (temporary haste)
  - Shield (temporary invulnerability)
  - Max HP increase
- **Dash Mechanic**
  - 4× velocity impulse with invulnerability frames
  - Lethal trail that damages enemies
  - Direction priority: current input → velocity → last facing direction
  - Cooldown scaling via shop upgrades
- **Shop System** (every 3 rooms)
  - More HP — +20 max HP, full heal
  - Shorter Dash — -20% cooldown
  - Score Boost — +0.25 multiplier bonus
  - Slow Enemies — -15% enemy speed
  - Damage Armor — -20% incoming damage
- **Scoring System**
  - Time-based: 1 pt/sec (scales with Flow state)
  - Kill-based: 25 pts per kill (scales with archetype multiplier)
  - Room clear: 250 pts
  - Boss clear: 2000 pts
- **Neon Visuals**
  - Real-time procedural grid animation
  - Dynamic background color tint responding to archetype mixture
  - Glow effects on player, enemies, pickups, boss
  - Dash echo trails with trippy blade effects
  - Screen shake intensity driven by player's arousal
  - Chromatic aberration during high-arousal states
  - Smooth bloom/glow based on overload state

### Game Progression
- **5-Room Cycles**: Alternate between combat rooms and boss rooms
- **Escalating Difficulty**: Enemy speed and spawn rates increase per room
- **Room Objectives**
  - Normal rooms: Kill required enemies OR survive time limit
  - Boss rooms: Defeat the boss to advance
- **Game-Over Screen**: Shows final stats (score, time survived, max flow, rooms cleared)

---

## 📦 Installation

### Prerequisites
- **Python 3.8+**
- **pip** (Python package manager)
- **GPU** (optional, recommended for better performance; CUDA-compatible GPU recommended)

### Step 1: Clone or Navigate to Project
```bash
cd /home/atharv/Desktop/projects/flow_game
```

### Step 2: Install Dependencies

**Option A: Using pip (Recommended)**

```bash
pip install -r requirements.txt
```

This installs:
- `pygame` — Game window, input handling, rendering
- `PyOpenGL` — OpenGL bindings for shader-based rendering
- `PyOpenGL_accelerate` — OpenGL acceleration
- `torch` — PyTorch for the inference engine
- `numpy` — Numerical computations

**Option B: Using conda (Alternative)**

```bash
conda create -n flow_game python=3.8
conda activate flow_game
pip install -r requirements.txt
```

### Step 3: GPU Acceleration (Optional)

If you have an NVIDIA GPU and want to use CUDA for faster inference:

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU only (already installed, but if you need to reinforce)
pip install torch torchvision torchaudio
```

---

## 🚀 Quick Start

### Run the Game

**Fullscreen Mode (Default)**
```bash
python flow_game.py
```

**Windowed Mode**
```bash
python flow_game.py --windowed
```

**Headless Invariant Check (No Window)**
```bash
python flow_game.py --probe
```

### First Launch

1. **Press ENTER** on the menu to start a new run
2. **Use WASD** to move around the arena
3. **Hold SPACE** to dash in the direction you're facing or moving
4. **Collect pickups** (rotating diamonds) to gain temporary buffs
5. **Survive enemy waves** to clear rooms
6. **Every 3 rooms**: Shop appears; press **1/2/3** to pick an upgrade
7. **Every 5 rooms**: Boss encounter; only your dash damages the boss
8. **Reach high score** by staying in Flow state and clearing rooms efficiently

---

## 🎯 Game Mechanics

### Psychological State Inference

Your actions are constantly analyzed:

| Metric | Tracked Via | Effect |
|--------|-------------|--------|
| **Velocity** | Movement speed | Higher = more Arousal |
| **APM (Actions/Min)** | Key presses | Higher = more Tactical or Arousal |
| **Direction Variance** | Turn frequency | Higher = more Overload or Apathy |
| **Threat Proximity** | Nearest enemy distance | Closer = Arousal or Tactical |
| **Idle Time** | Stationary frames | Higher = Apathy |

The non-causal attention model processes these 5 observations over a 32-frame window (16 past + 16 future-projected) to produce a smooth, stable mixture (`pi`) that smoothly transitions between the 5 archetypes.

### Physics Mapping (M Matrix)

The inferred state drives physics:

| Parameter | Arousal | Tactical | Overload | Flow | Apathy |
|-----------|---------|----------|----------|------|--------|
| Enemy Speed | 1.8 | 0.6 | 1.4 | 1.2 | 0.3 |
| Spawn Rate | 1.5 | 0.4 | 1.8 | 1.0 | 0.2 |
| Friction | 0.3 | 0.9 | 0.7 | 0.2 | 1.0 |
| Camera Zoom | 1.10 | 0.95 | 0.85 | 1.20 | 0.90 |

**Example**: High Arousal → enemies spawn faster and move faster; camera zooms in (claustrophobic). High Flow → perfect enemy speed/spawn balance; camera zooms out for spatial awareness.

### Visual Latents (c)

Four independent visual distortion parameters, each driven by archetype mixture:

| Latent | Driven By | Effect |
|--------|-----------|--------|
| `c_1` Aberration | Arousal + Overload | RGB channel splitting; trippy effect |
| `c_2` Grain | Tactical + Apathy | Film grain; grittiness |
| `c_3` Warp | Flow | Sinusoidal UV distortion; wavy reality |
| `c_4` Bloom | Flow + Arousal | Screen glow intensity; ethereal feel |

### Boss AI Behavior

Boss reads your `pi` and adapts:

- **High Arousal** → Fast, aggressive pursuit; closes distance rapidly
- **High Tactical** → Maintains ideal range (~4.5 units); precise, calculated movement
- **High Overload** → Summons 1–3 minions every 4–6 seconds
- **High Flow** → Teleports to flank player every 2–3 seconds
- **High Apathy** → Slow, passive movement; low threat (represents system tiredness)

### Enemy Type Spawning

Enemy distribution is biased by archetype mixture:

| Spawn Weight | Formula |
|---|---|
| Chaser | `1.0 + 1.5 × pi[Overload]` |
| Fast | `0.5 + 2.0 × pi[Tactical] + 0.6 × pi[Arousal]` |
| Tank | `0.3 + 1.8 × pi[Apathy]` |

**Example**: High Tactical state → more Fast enemies (they reward precise play); High Apathy → more Tanks (slow, adds weight).

### Scoring & Multipliers

Each kill grants base 25 points × multiplier:

```
Multiplier = 1.0 + 2.0 × pi[Flow] + shop_bonus
```

Time-based score (background accrual):

```
Points/sec = 1.0 × (1.0 + pi[Flow])
```

---

## 🎮 Controls

| Input | Action |
|-------|--------|
| **WASD** | Move (accelerate) |
| **SPACE** | Dash (invulnerability + lethal trail) |
| **1 / 2 / 3** | Pick shop upgrade |
| **ENTER** | Start run (menu) / Continue (game-over) |
| **ESC** | Exit to menu (when in-game) |

### Movement Physics

- Acceleration: 22 m/s² (configurable)
- Friction: scales with `pi[Tactical]` (high = precise; low = slippery)
- Velocity capped by arena boundaries (±16 units)

### Dash Mechanics

- **Cooldown**: 2.5 sec baseline (reduced by shop upgrade "Shorter Dash")
- **Duration**: 1.0 sec i-frames + dashing window
- **Impulse**: 4× velocity multiplier (6.0 m/s minimum)
- **Dash Boost Pickup**: +40% impulse, +60% duration
- **Direction**: Prioritizes (1) current input, (2) current velocity, (3) last facing
- **Lethal Trail**: Segments remain for 1.6 sec; 0.55 unit collision radius

---

## 🏗️ Architecture

```
flow_game/
├── flow_game.py                    # Backend inference engine + main loop
├── requirements.txt
├── init.md
├── README.md (this file)
└── game/
    ├── __init__.py
    ├── config.py                   # Tunable gameplay constants
    ├── game_state.py               # GameState enum
    ├── abilities.py                # Dash and timer mechanics
    ├── entities_extra.py           # Enemy, Boss, Pickup entities
    ├── progression.py              # Rooms, shop, upgrades, scoring
    └── ui.py                       # HUD, shop menu, game-over screen
```

### Layer Separation

**Backend (flow_game.py)** — UNTOUCHED by gameplay layer
- `AttentionStateInference` — Non-causal bidirectional attention
- `EnvironmentMapper` — Maps `pi` → physics + visual latents
- `ObservationBuilder` — Converts keypresses → observation vector
- Main loop: inference thread, shader uniform uploads, collision detection

**Game Layer (game/)** — ONLY READS `pi, E, c`
- Manages rooms, shops, boss, enemies, scoring
- Scales physics outputs: `E' = E × (1 + room_scale)`
- Interprets outputs: "If `pi[OVERLOAD]` is high, spawn more Tanks"
- Never modifies backend, never injects fake states

### Shader System

**Vertex Shader**: Simple fullscreen quad

**Fragment Shader** (~900 lines):
- SDF circles for player, enemies, boss, pickups
- Animated grid background with archetype-driven tint
- Dash trail segments with additive glow
- Dash visual effects: echo trails, spinning blades, ripple ring
- Chromatic aberration, film grain, UV warp, bloom glow
- Dynamic color mapping per pickup type (heal=green, dash=cyan, speed=orange, shield=blue, max_hp=yellow)
- Dynamic HP ring around boss

---

## ⚙️ Configuration

All gameplay constants are tunable in [game/config.py](game/config.py):

### Display
```python
FULLSCREEN = True           # False for windowed
WINDOWED_SIZE = (1280, 800)
```

### Player Stats
```python
PLAYER_MAX_HP = 100
PLAYER_ACCEL = 22.0
DASH_COOLDOWN = 2.5         # seconds
DASH_INVULN = 1.0           # i-frames duration
DASH_IMPULSE = 4.0
DAMAGE_ON_HIT = 10
```

### Difficulty
```python
ROOM_BASE_TIME = 20.0       # seconds
ROOM_TIME_STEP = 3.0        # +seconds per room
ENEMY_SCALE_SPEED = 0.10    # +10% per room
ENEMY_SCALE_SPAWN = 0.15    # +15% per room
```

### Enemy Stats
```python
ENEMY_STATS = {
    "chaser": dict(hp=2, speed=1.0, radius=0.38),
    "fast":   dict(hp=1, speed=2.0, radius=0.28),
    "tank":   dict(hp=4, speed=0.55, radius=0.55),
}
```

### Boss
```python
BOSS_HP = 220
BOSS_RADIUS = 1.2
BOSS_CONTACT_DAMAGE = 12
BOSS_DASH_DAMAGE = 14
```

### Pickups
```python
PICKUP_MAX_ACTIVE = 6
PICKUP_SPAWN_INTERVAL = 7.0
PICKUP_BUFF_DURATION = {
    "dash_boost":  8.0,
    "speed_boost": 8.0,
    "shield":      5.0,
    "max_hp":      12.0,
}
```

### Scoring
```python
SCORE_TIME_BASE = 1.0       # points/sec
SCORE_KILL_BASE = 25        # per kill
SCORE_ROOM_CLEAR = 250
SCORE_BOSS_CLEAR = 2000
```

---

## 🔧 System Requirements

### Minimum
- **CPU**: Intel i5 / AMD Ryzen 5 or equivalent
- **RAM**: 4 GB
- **GPU**: Any with OpenGL 3.3+ support (integrated is OK)
- **Python**: 3.8+

### Recommended
- **CPU**: Intel i7 / AMD Ryzen 7 or equivalent
- **RAM**: 8 GB
- **GPU**: NVIDIA GTX 1080 / RTX 3060 or equivalent (for CUDA acceleration)
- **Python**: 3.10+

### Tested Platforms
- **Linux**: Ubuntu 20.04+, Fedora 35+
- **macOS**: 11.0+ (may require CPU fallback)
- **Windows**: 10, 11 (with Visual C++ Runtime installed)

---

## 🎓 Understanding the Math

### The 5-Simplex

Archetypes lie on a 4-dimensional hyperplane (5D simplex):
```
π_arousal + π_tactical + π_overload + π_flow + π_apathy = 1.0
0 ≤ π_k ≤ 1.0 for all k
```

Result: Smooth, continuous transitions between psychological states.

### Non-Causal Attention

Bidirectional attention allows the model to:
- **Learn temporal patterns** from recent keypresses
- **Project future behavior** based on inertia and observations
- **Stabilize inference** with a 1-second feedback window (32 frames @ 30fps)

Formula:
```
Q, K, V = O @ W_Q, O @ W_K, O @ W_V
A = softmax((Q @ K^T) / √d_k)
Z = A @ V
π = softmax(W_π @ mean(Z) + b_π)
```

### Environment Mapping

Deterministic, hand-tuned matrix `M` (4×5):
```
E = M @ π + N(0, 0.02²)
```

Each row of `M` tunes a different aspect of the game world (speed, spawn, friction, zoom).

### Visual Latents (β-VAE-Inspired)

Sparse decoder architecture ensures _disentanglement_:
```
c = W_c2 @ ReLU(W_c1 @ π)
```

Result: Each latent (`c_1`, `c_2`, `c_3`, `c_4`) independently controls a visual aspect, with minimal cross-talk.

---

## 🐛 Troubleshooting

### Game Won't Start
- Ensure Python 3.8+ is installed: `python --version`
- Verify dependencies: `pip list | grep -E 'pygame|torch|numpy'`
- Try CPU-only torch: `pip install torch --index-url https://download.pytorch.org/whl/cpu`

### Low FPS / Laggy
- Check GPU utilization: `nvidia-smi` (if NVIDIA)
- Disable fullscreen: `python flow_game.py --windowed`
- Reduce number of enemies in [game/config.py](game/config.py): `MAX_ENEMIES = 8` (from 16)

### Controls Unresponsive
- Ensure Pygame is detecting your keyboard (required on some Linux setups):
  ```bash
  pip install --upgrade pygame
  ```

### Shader Compilation Error
- Verify OpenGL 3.3+ support:
  ```bash
  python -c "import OpenGL; from OpenGL.GL import glGetString, GL_VERSION; print(glGetString(GL_VERSION))"
  ```
- Update GPU drivers

---

## 📊 Symbol Table (Backend)

| Symbol | Meaning | Value |
|--------|---------|-------|
| `D` | Observation dimensionality | 5 |
| `K` | Archetype count | 5 |
| `W_HIST` | Historical window | 16 |
| `W_FUT` | Projected future window | 16 |
| `W_TOTAL` | Total attention window | 32 |
| `D_K` | Attention inner dimension | 16 |
| `P` | Environment parameters | 4 |
| `V_LATENT` | Visual latent dimensions | 4 |
| `EMA_ALPHA` | EMA baseline rate | 0.02 |
| `MAX_ENEMIES` | Shader uniform array size | 16 |
| `WORLD_HALF` | Arena half-size | 16.0 |

---

## 🎨 Visual Aesthetic

The game features a **dynamically-tinted neon arena** where the entire experience adapts to your psychological state:

- **Arousal**: Warm ember reds, sharp contrasts, intense blur/aberration
- **Tactical**: Cool deep blues, clean lines, strategic grid visibility
- **Overload**: Chaotic violets, heavy grain, reality-warping distortion
- **Flow**: Peaceful teals, smooth bloom, balanced aesthetics
- **Apathy**: Muted grays, low contrast, dreamlike haze

---

## 📝 License & Credits

**Flow Game** — Created as an experimental intersection of psychology, machine learning, and game design.

Built with:
- PyTorch (Python ML framework)
- Pygame (Game engine)
- PyOpenGL (GPU rendering)
- GLSL (Fragment shaders)

---

## 🚀 Future Ideas

Some wild creative enhancements planned:

1. **Doppelgänger Boss** — A ghostly copy of the player with mirrored behavior
2. **Hivemind Enemies** — Coordinated swarms with visible threading
3. **Time Dilation** — Archetype-dependent bullet-time mechanics
4. **Philosophical Boss Phases** — Each boss form represents a different archetype
5. **Emotional Contagion** — Boss AI state bleeds into player psychology
6. **Fractured Reality** — Parallel dimensions at high Overload states
7. **AI Allies** — Recruit defeated enemies as allies during Flow state
8. **Archetype Loadouts** — Unlock unique abilities per dominant state
9. **Physics Puzzles** — Rooms with obstacles that adapt to current friction/friction state

---

## 💬 Contact & Feedback

Questions or ideas? Reach out or open an issue in the project repository.

---

**Enjoy the Flow!** 🌊✨

---

*Last Updated: April 2026*
