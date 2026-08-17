/* Run state, room progression and shop upgrades — port of
 * game/progression.py.
 *
 * All scaling here sits *on top of* the backend outputs: we multiply
 * E[0..3] by room scaling, we never replace them. */

import * as C from "./config.js";

const ARCH_FLOW = C.ARCH_FLOW;

export class RunState {
  constructor() {
    // Progress
    this.room = 0;
    this.killsInRoom = 0;
    this.totalKills = 0;
    // Scoring
    this.score = 0;
    this.maxFlow = 0;
    this.timeSurvived = 0;
    this.roomsCleared = 0;
    // Modifiers (shop-driven multipliers; start at identity)
    this.damageMult = 1.0;
    this.scoreMultBonus = 0.0;
    this.enemySlow = 1.0;
    // Current-room bookkeeping
    this.roomTimeTotal = 0;
    this.roomTimeRemaining = 0;
    this.roomKillsRequired = 0;
    this.isBossRoom = false;
    this.overtime = false;
    this.overtimeRooms = 0;
    // Web-only telemetry, feeds the run report and the lifetime profile.
    this.piSeconds = [0, 0, 0, 0, 0];
    this.longestFlowStreak = 0;
    this._flowStreak = 0;
    this.upgradesTaken = [];
    this.bossesKilled = 0;
    // "timeout" | "killed" | "abandoned" — the room timer killing you at full
    // HP is the single most confusing thing this game does to a new player,
    // so the report has to be able to name the cause.
    this.deathCause = null;
  }

  startNextRoom() {
    this.room += 1;
    this.killsInRoom = 0;
    this.overtime = false;
    this.isBossRoom = this.room % C.BOSS_EVERY_N_ROOMS === 0;
    if (this.isBossRoom) {
      this.roomTimeTotal = 60.0 + this.room * 4.0;
      this.roomKillsRequired = 1; // kill the boss
    } else {
      this.roomTimeTotal = C.ROOM_BASE_TIME + this.room * C.ROOM_TIME_STEP;
      this.roomKillsRequired = C.ROOM_BASE_KILLS + this.room * C.ROOM_KILL_STEP;
    }
    this.roomTimeRemaining = this.roomTimeTotal;
  }

  clearRoom() {
    this.roomsCleared += 1;
    let base = this.isBossRoom ? C.SCORE_BOSS_CLEAR : C.SCORE_ROOM_CLEAR;
    if (this.overtime) base *= C.ROOM_OVERTIME_MULT;
    this.score += base;
    if (this.isBossRoom) this.bossesKilled += 1;
  }

  tick(pi, dt) {
    const flow = pi[ARCH_FLOW];
    this.maxFlow = Math.max(this.maxFlow, flow);
    this.timeSurvived += dt;
    // Time-based score accrues faster when in Flow.
    this.score += C.SCORE_TIME_BASE * (1.0 + flow) * dt;

    for (let k = 0; k < 5; k++) this.piSeconds[k] += pi[k] * dt;
    // "In Flow" = Flow is both dominant and confident.
    if (flow > 0.45) {
      this._flowStreak += dt;
      this.longestFlowStreak = Math.max(this.longestFlowStreak, this._flowStreak);
    } else {
      this._flowStreak = 0;
    }
  }

  scoreMultiplier(pi) {
    return 1.0 + 2.0 * pi[ARCH_FLOW] + this.scoreMultBonus;
  }

  registerKill(pi) {
    this.killsInRoom += 1;
    this.totalKills += 1;
    this.score += C.SCORE_KILL_BASE * this.scoreMultiplier(pi);
  }

  roomGoalMet() {
    if (this.isBossRoom) return false; // boss handled separately
    return this.killsInRoom >= this.roomKillsRequired;
  }

  /** Flip the room into overtime when its timer runs out.
   *
   *  Running out of time is *not* a death: the room continues with the clear
   *  bonus halved. Clamps the timer at zero so the HUD never shows a negative
   *  countdown. Returns true only on the transition tick. */
  checkOvertime() {
    if (this.isBossRoom || this.overtime) return false;
    if (this.roomTimeRemaining > 0 || this.roomGoalMet()) return false;
    this.roomTimeRemaining = 0;
    this.overtime = true;
    this.overtimeRooms += 1;
    return true;
  }

  shouldOpenShop() {
    return this.room > 0 && this.room % C.SHOP_EVERY_N_ROOMS === 0 && !this.isBossRoom;
  }

  /** Index of the archetype this run spent the most time in. */
  dominantArchetype() {
    let best = 0;
    for (let k = 1; k < 5; k++) if (this.piSeconds[k] > this.piSeconds[best]) best = k;
    return best;
  }
}

// Room scaling applied to backend physics outputs.
export function scaleEnemySpeed(E0, room, enemySlow) {
  return E0 * (1.0 + C.ENEMY_SCALE_SPEED * room) * enemySlow;
}

export function scaleSpawnRate(E1, room) {
  return E1 * (1.0 + C.ENEMY_SCALE_SPAWN * room);
}

// --- Shop ------------------------------------------------------------------

/** `favors` tags each upgrade with the archetype it plays into, purely so the
 *  shop cards can be tinted to match the state that likes them. */
export const UPGRADES = [
  { name: "More HP", desc: "+20 max HP, fully heal", favors: C.ARCH_AROUSAL, icon: "♥" },
  { name: "Shorter Dash", desc: "Dash cooldown −20%", favors: C.ARCH_FLOW, icon: "⇢" },
  { name: "Score Boost", desc: "+0.25 score multiplier", favors: C.ARCH_FLOW, icon: "★" },
  { name: "Slow Enemies", desc: "Enemies move 15% slower", favors: C.ARCH_APATHY, icon: "❄" },
  { name: "Damage Armor", desc: "−20% damage taken", favors: C.ARCH_TACTICAL, icon: "⛊" },
];

export function pickShopChoices(n = 3) {
  const pool = UPGRADES.map((_, i) => i);
  for (let i = pool.length - 1; i > 0; i--) {
    const j = (Math.random() * (i + 1)) | 0;
    [pool[i], pool[j]] = [pool[j], pool[i]];
  }
  return pool.slice(0, n);
}

export function applyUpgrade(idx, player, run) {
  const { name } = UPGRADES[idx];
  if (name === "More HP") {
    player.maxHp += 20;
    player.hp = player.maxHp;
  } else if (name === "Shorter Dash") {
    player.dashCdMax = Math.max(0.25, player.dashCdMax * 0.8);
  } else if (name === "Score Boost") {
    run.scoreMultBonus += 0.25;
  } else if (name === "Slow Enemies") {
    run.enemySlow *= 0.85;
  } else if (name === "Damage Armor") {
    run.damageMult *= 0.8;
  }
  run.upgradesTaken.push(idx);
  return name;
}
