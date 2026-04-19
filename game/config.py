"""Tunable constants for the roguelite layer.

Backend constants (D, K, W_TOTAL, etc.) live in flow_game.py and are
owned by the inference engine. Everything here is gameplay-facing.
"""
from __future__ import annotations

# --- Display --------------------------------------------------------------
FULLSCREEN         = True      # launch windowed if False; --windowed CLI overrides
WINDOWED_SIZE      = (1280, 800)

# --- Player ---------------------------------------------------------------
PLAYER_MAX_HP      = 100
PLAYER_ACCEL       = 22.0      # base acceleration magnitude
DASH_COOLDOWN      = 2.5       # seconds (baseline; shop upgrades scale it)
DASH_INVULN        = 1.0       # full second of i-frames + dashing window
DASH_IMPULSE       = 4.0
HIT_INVULN         = 0.5
DAMAGE_ON_HIT      = 10

# --- Rooms ----------------------------------------------------------------
ROOM_BASE_TIME     = 20.0      # seconds
ROOM_TIME_STEP     = 3.0       # +seconds per room
ROOM_BASE_KILLS    = 5
ROOM_KILL_STEP     = 2
SHOP_EVERY_N_ROOMS = 3         # after every 3 rooms
BOSS_EVERY_N_ROOMS = 5         # 5th room of every cycle is a boss

# --- Enemies --------------------------------------------------------------
MAX_ENEMIES        = 16        # matches shader uniform array size
ENEMY_SCALE_SPEED  = 0.10      # +10% speed per room (on top of E[0])
ENEMY_SCALE_SPAWN  = 0.15      # +15% spawn rate per room

# Enemy archetype stats.
ENEMY_STATS = {
    "chaser": dict(hp=2, contact_cd=0.35, speed=1.0,  radius=0.38),
    "fast":   dict(hp=1, contact_cd=0.25, speed=2.0,  radius=0.28),
    "tank":   dict(hp=4, contact_cd=0.55, speed=0.55, radius=0.55),
}

# --- Boss -----------------------------------------------------------------
BOSS_HP             = 220
BOSS_RADIUS         = 1.2
BOSS_CONTACT_DAMAGE = 12
BOSS_DASH_DAMAGE    = 14
BOSS_TELEPORT_CD    = 3.0
BOSS_MINION_CD      = 4.5
BOSS_ATTACK_CD      = 1.2

# --- Pickups (food / power-ups) ------------------------------------------
PICKUP_MAX_ACTIVE      = 6        # at most this many on the floor
PICKUP_SHADER_CAP      = 8        # uniform-array size in fragment shader
PICKUP_SPAWN_INTERVAL  = 7.0      # seconds between spawns
PICKUP_RADIUS          = 0.45
PICKUP_MIN_DIST        = 4.0      # away from the player at spawn
ENEMY_MIN_SPAWN_DIST   = 5.5      # enemies never spawn closer than this

# --- Scoring --------------------------------------------------------------
SCORE_TIME_BASE    = 1.0       # points/sec baseline
SCORE_KILL_BASE    = 25        # points per kill (before multiplier)
SCORE_ROOM_CLEAR   = 250
SCORE_BOSS_CLEAR   = 2000

# --- Visual feedback scalars ---------------------------------------------
SHAKE_GAIN         = 0.25      # screen shake amplitude per unit c1
