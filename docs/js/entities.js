/* Enemy variants, Boss AI and pickups — port of game/entities_extra.py.
 *
 * Spawn mix is biased by the inferred pi mixture without touching the
 * backend: we only *read* pi and weight the probabilities. */

import * as C from "./config.js";

const { ARCH_AROUSAL, ARCH_TACTICAL, ARCH_OVERLOAD, ARCH_FLOW, ARCH_APATHY } = C;

// --- Enemies ---------------------------------------------------------------

export function makeEnemy(kind, playerPos, worldHalf) {
  const s = C.ENEMY_STATS[kind];
  const [x, y, ux, uy] = safePerimeterSpawn(playerPos, worldHalf);
  return {
    pos: [x, y],
    vel: [ux * s.speed, uy * s.speed],
    hp: s.hp,
    maxHp: s.hp,
    baseSpeed: s.speed,
    radius: s.radius,
    kind,
    contactCd: 0,
    contactCdMax: s.contact_cd,
  };
}

function spawnAtPerimeter(playerPos, worldHalf) {
  const angle = Math.random() * Math.PI * 2;
  const r = worldHalf * 0.95;
  const x = r * Math.cos(angle);
  const y = r * Math.sin(angle);
  const dx = playerPos[0] - x;
  const dy = playerPos[1] - y;
  const n = Math.hypot(dx, dy) + 1e-6;
  return [x, y, dx / n, dy / n];
}

/** Perimeter spawn at least `minDist` from the player, so nothing ever
 *  materialises in the player's lap while they camp a wall. */
function safePerimeterSpawn(playerPos, worldHalf, minDist = C.ENEMY_MIN_SPAWN_DIST, maxTries = 12) {
  let best = null;
  for (let i = 0; i < maxTries; i++) {
    const cand = spawnAtPerimeter(playerPos, worldHalf);
    const d = Math.hypot(cand[0] - playerPos[0], cand[1] - playerPos[1]);
    if (d >= minDist) return cand;
    if (best === null || d > best[4]) best = [...cand, d];
  }
  if (best !== null) return [best[0], best[1], best[2], best[3]];
  // Final fallback: antipode of the player projected onto the perimeter.
  const pn = Math.hypot(playerPos[0], playerPos[1]) + 1e-6;
  const ax = (-playerPos[0] / pn) * worldHalf * 0.95;
  const ay = (-playerPos[1] / pn) * worldHalf * 0.95;
  const dx = playerPos[0] - ax;
  const dy = playerPos[1] - ay;
  const n = Math.hypot(dx, dy) + 1e-6;
  return [ax, ay, dx / n, dy / n];
}

/** Weighted sample of enemy kind from the pi mixture.
 *  Overload -> more chasers, Tactical -> more fast enemies (they punish loose
 *  play), Apathy -> more tanks, with Arousal nudging toward fast too. */
export function pickKind(pi) {
  const wChaser = 1.0 + 1.5 * pi[ARCH_OVERLOAD];
  const wFast = 0.5 + 2.0 * pi[ARCH_TACTICAL] + 0.6 * pi[ARCH_AROUSAL];
  const wTank = 0.3 + 1.8 * pi[ARCH_APATHY];
  const r = Math.random() * (wChaser + wFast + wTank);
  if (r < wChaser) return "chaser";
  if (r < wChaser + wFast) return "fast";
  return "tank";
}

/** Integrate enemy motion. speedMult = E[0] * room scaling. */
export function stepEnemies(enemies, playerPos, speedMult, slowFactor, dt) {
  const m = speedMult * slowFactor;
  for (const e of enemies) {
    const dx = playerPos[0] - e.pos[0];
    const dy = playerPos[1] - e.pos[1];
    const n = Math.hypot(dx, dy) + 1e-6;
    e.vel[0] += ((dx / n) * e.baseSpeed - e.vel[0]) * 0.05;
    e.vel[1] += ((dy / n) * e.baseSpeed - e.vel[1]) * 0.05;
    e.pos[0] += e.vel[0] * m * dt;
    e.pos[1] += e.vel[1] * m * dt;
    e.contactCd = Math.max(0, e.contactCd - dt);
  }
}

// --- Boss ------------------------------------------------------------------

export function makeBoss() {
  return {
    pos: [0, 4],
    vel: [0, 0],
    hp: C.BOSS_HP,
    maxHp: C.BOSS_HP,
    radius: C.BOSS_RADIUS,
    contactCd: 0,
    teleportCd: C.BOSS_TELEPORT_CD,
    minionCd: C.BOSS_MINION_CD,
    attackCd: C.BOSS_ATTACK_CD,
    hitFlash: 0,
    phaseLabel: "idle",
  };
}

/** Boss AI. Reads pi to select behaviour; mutates boss + minions.
 *
 *  High Arousal  -> fast aggressive pursuit.
 *  High Overload -> summons minions on cooldown.
 *  High Flow     -> teleports to flank the player.
 *  High Tactical -> precise positioning at ideal range.
 *
 *  Returns a list of event tags for the audio layer ("teleport", "summon"). */
export function stepBoss(boss, playerPos, pi, worldHalf, minions, dt) {
  const events = [];
  boss.contactCd = Math.max(0, boss.contactCd - dt);
  boss.teleportCd = Math.max(0, boss.teleportCd - dt);
  boss.minionCd = Math.max(0, boss.minionCd - dt);
  boss.attackCd = Math.max(0, boss.attackCd - dt);
  boss.hitFlash = Math.max(0, boss.hitFlash - dt);

  let arch = 0;
  for (let i = 1; i < pi.length; i++) if (pi[i] > pi[arch]) arch = i;
  boss.phaseLabel = C.STATE_NAMES[arch];

  const dx = playerPos[0] - boss.pos[0];
  const dy = playerPos[1] - boss.pos[1];
  const dist = Math.hypot(dx, dy) + 1e-6;
  const ux = dx / dist;
  const uy = dy / dist;

  // Aggression scales with arousal, defense with tactical.
  const aggression = 0.4 + 1.8 * pi[ARCH_AROUSAL];
  const idealDist = 4.5 - 2.5 * pi[ARCH_AROUSAL] + 1.5 * pi[ARCH_TACTICAL];
  const rangeErr = dist - idealDist;
  boss.vel[0] += (ux * rangeErr * 0.4 * aggression - boss.vel[0]) * 0.08;
  boss.vel[1] += (uy * rangeErr * 0.4 * aggression - boss.vel[1]) * 0.08;
  boss.pos[0] = clamp(boss.pos[0] + boss.vel[0] * dt, -worldHalf, worldHalf);
  boss.pos[1] = clamp(boss.pos[1] + boss.vel[1] * dt, -worldHalf, worldHalf);

  // Flow state -> teleport attacks (flank).
  if (pi[ARCH_FLOW] > 0.25 && boss.teleportCd <= 0) {
    const flank = Math.atan2(-dy, -dx) + (Math.random() < 0.5 ? -1 : 1) * Math.PI * 0.45;
    const r = 3.5;
    boss.pos[0] = clamp(playerPos[0] + Math.cos(flank) * r, -worldHalf, worldHalf);
    boss.pos[1] = clamp(playerPos[1] + Math.sin(flank) * r, -worldHalf, worldHalf);
    boss.vel[0] = boss.vel[1] = 0;
    boss.teleportCd = C.BOSS_TELEPORT_CD * (1.5 - pi[ARCH_FLOW]);
    events.push("teleport");
  }

  // Overload state -> spawn minions.
  if (pi[ARCH_OVERLOAD] > 0.25 && boss.minionCd <= 0) {
    const spawnN = 1 + Math.floor(2 * pi[ARCH_OVERLOAD]);
    for (let i = 0; i < spawnN; i++) {
      if (minions.length >= C.MAX_ENEMIES - 1) break;
      minions.push(makeEnemy(Math.random() < 0.5 ? "fast" : "chaser", playerPos, worldHalf));
    }
    boss.minionCd = C.BOSS_MINION_CD * (1.5 - pi[ARCH_OVERLOAD]);
    events.push("summon");
  }
  return events;
}

function clamp(v, lo, hi) {
  return v < lo ? lo : v > hi ? hi : v;
}

// --- Pickups ---------------------------------------------------------------

export const PICKUP_KINDS = ["heal", "dash_boost", "speed_boost", "shield", "max_hp"];
// Stable integer codes for the shader (must match the GLSL switch).
export const PICKUP_CODE = Object.fromEntries(PICKUP_KINDS.map((k, i) => [k, i]));
export const PICKUP_LABEL = {
  heal: "+30 HP",
  dash_boost: "BIG DASH",
  speed_boost: "HASTE",
  shield: "SHIELD",
  max_hp: "+MAX HP",
};
export const PICKUP_BUFF_DURATION = {
  dash_boost: 8.0,
  speed_boost: 8.0,
  shield: 5.0,
  max_hp: 12.0,
};

export function makePickup(playerPos, worldHalf, kind = null, minDist = C.PICKUP_MIN_DIST) {
  if (kind === null) kind = PICKUP_KINDS[(Math.random() * PICKUP_KINDS.length) | 0];
  for (let i = 0; i < 20; i++) {
    const x = (Math.random() * 2 - 1) * worldHalf * 0.85;
    const y = (Math.random() * 2 - 1) * worldHalf * 0.85;
    if (Math.hypot(x - playerPos[0], y - playerPos[1]) >= minDist) {
      return { pos: [x, y], kind, radius: C.PICKUP_RADIUS };
    }
  }
  return {
    pos: [-playerPos[0] * 0.7, -playerPos[1] * 0.7],
    kind,
    radius: C.PICKUP_RADIUS,
  };
}
