/* Tunable constants — a direct port of game/config.py plus the backend
 * symbols that live at the top of flow_game.py.
 *
 * Anything the desktop build treats as gameplay-facing is here; the
 * inference-engine numbers (D, K, W_TOTAL, ...) sit in the BACKEND block
 * and must stay in sync with flow_game.py. */

// --- Backend (inference engine) -------------------------------------------
export const D = 5;
export const K = 5;
export const P = 4;
export const V_LATENT = 4;
export const D_K = 16;
export const W_HIST = 16;
export const W_FUT = 16;
export const W_TOTAL = W_HIST + W_FUT;
export const EMA_ALPHA = 0.02;
export const APM_DECAY = 0.95;
export const WORLD_HALF = 16.0;
export const DASH_TRAIL_TTL = 1.6;
export const DASH_TRAIL_RADIUS = 0.55;
export const DASH_TRAIL_MAX = 32;

export const ARCH_AROUSAL = 0;
export const ARCH_TACTICAL = 1;
export const ARCH_OVERLOAD = 2;
export const ARCH_FLOW = 3;
export const ARCH_APATHY = 4;

// --- Player ---------------------------------------------------------------
export const PLAYER_MAX_HP = 100;
export const PLAYER_ACCEL = 22.0;
export const DASH_COOLDOWN = 2.5;
export const DASH_INVULN = 0.4;
export const DASH_IMPULSE = 4.0;
export const HIT_INVULN = 0.3;
export const DAMAGE_ON_HIT = 15;

// --- Rooms ----------------------------------------------------------------
export const ROOM_BASE_TIME = 30.0;
export const ROOM_TIME_STEP = 2.0;
export const ROOM_BASE_KILLS = 8;
export const ROOM_KILL_STEP = 3;
export const SHOP_EVERY_N_ROOMS = 3;
export const BOSS_EVERY_N_ROOMS = 5;

// --- Enemies --------------------------------------------------------------
export const MAX_ENEMIES = 16; // matches the shader uniform array size
export const ENEMY_SCALE_SPEED = 0.15;
export const ENEMY_SCALE_SPAWN = 0.22;

export const ENEMY_STATS = {
  chaser: { hp: 2, contact_cd: 0.35, speed: 1.0, radius: 0.38 },
  fast: { hp: 1, contact_cd: 0.25, speed: 2.0, radius: 0.28 },
  tank: { hp: 4, contact_cd: 0.55, speed: 0.55, radius: 0.55 },
};

// --- Boss -----------------------------------------------------------------
export const BOSS_HP = 220;
export const BOSS_RADIUS = 1.2;
export const BOSS_CONTACT_DAMAGE = 12;
export const BOSS_DASH_DAMAGE = 14;
export const BOSS_TELEPORT_CD = 3.0;
export const BOSS_MINION_CD = 4.5;
export const BOSS_ATTACK_CD = 1.2;

// --- Pickups --------------------------------------------------------------
export const PICKUP_MAX_ACTIVE = 6;
export const PICKUP_SHADER_CAP = 8;
export const PICKUP_SPAWN_INTERVAL = 7.0;
export const PICKUP_RADIUS = 0.45;
export const PICKUP_MIN_DIST = 4.0;
export const ENEMY_MIN_SPAWN_DIST = 5.5;

export const SPEED_BOOST_MULT = 1.4;
export const MAX_HP_BONUS = 25;
export const HEAL_AMOUNT = 30;

// --- Scoring --------------------------------------------------------------
export const SCORE_TIME_BASE = 1.0;
export const SCORE_KILL_BASE = 25;
export const SCORE_ROOM_CLEAR = 250;
export const SCORE_BOSS_CLEAR = 2000;
// Running the room timer out is NOT death in the web build (it is in
// flow_game.py). The room goes into overtime instead and the clear bonus is
// scaled by this — the timer costs you points, never the run.
export const ROOM_OVERTIME_MULT = 0.5;

// --- Visual feedback ------------------------------------------------------
export const SHAKE_GAIN = 0.25;

// --- Web-only -------------------------------------------------------------
// The desktop build steps once per rendered frame at up to 144 Hz. Browsers
// hand us anything from 30 to 240 Hz, and several gameplay lerps (enemy
// steering, APM decay, the EMA baseline) are per-frame rather than per-second,
// so we run the simulation on a fixed tick and let rendering float.
export const SIM_HZ = 120;
export const SIM_DT = 1 / SIM_HZ;
export const MAX_SUBSTEPS = 6;

export const STATE_NAMES = ["AROUSAL", "TACTICAL", "OVERLOAD", "FLOW", "APATHY"];
export const STATE_COLORS = [
  "#ff7878", // arousal  red
  "#78b4ff", // tactical blue
  "#c878ff", // overload purple
  "#78ffc8", // flow     teal
  "#aaaaaa", // apathy   grey
];
