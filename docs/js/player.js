/* Player entity, dash ability and buff bookkeeping — ports the Player
 * dataclass from flow_game.py plus game/abilities.py. */

import * as C from "./config.js";
import { PICKUP_BUFF_DURATION, PICKUP_LABEL } from "./entities.js";

export function makePlayer() {
  return {
    pos: [0, 0],
    vel: [0, 0],
    facing: [1, 0],
    hp: C.PLAYER_MAX_HP,
    maxHp: C.PLAYER_MAX_HP,
    dashCd: 0,
    dashCdMax: C.DASH_COOLDOWN,
    invuln: 0,
    dashing: 0,
    hitFlash: 0,
    inputDir: [0, 0],
    dashDir: [1, 0],
    // Lethal dash-trail waypoints: [x, y, expiresAtSeconds]. Enemies whose
    // centre falls within DASH_TRAIL_RADIUS of any consecutive segment die.
    dashTrail: [],
    // Active power-up buffs: kind -> expiresAtSeconds.
    buffs: new Map(),
  };
}

/** Trigger a dash if the dash input is held and the cooldown has elapsed.
 *
 *  Direction priority: current input -> current velocity -> last facing.
 *  Returns true if a dash fired this tick. */
export function tryDash(player, dashHeld, rawIx, rawIy) {
  if (player.dashCd > 0 || !dashHeld) return false;

  let dx, dy;
  const ilen = Math.hypot(rawIx, rawIy);
  if (ilen > 0.1) {
    dx = rawIx / ilen;
    dy = rawIy / ilen;
  } else {
    const vlen = Math.hypot(player.vel[0], player.vel[1]);
    if (vlen > 0.2) {
      dx = player.vel[0] / vlen;
      dy = player.vel[1] / vlen;
    } else {
      [dx, dy] = player.facing;
    }
  }

  // Snap velocity to a uniform dash speed so behaviour is consistent whether
  // the player was already moving fast or standing still.
  const curSpeed = Math.hypot(player.vel[0], player.vel[1]);
  const hasBoost = player.buffs.has("dash_boost");
  const impulseMult = hasBoost ? 1.35 : 1.0;
  const speed = Math.max(curSpeed, 6.0) * C.DASH_IMPULSE * 0.6 * impulseMult;
  player.vel[0] = dx * speed;
  player.vel[1] = dy * speed;
  player.dashDir = [dx, dy];
  player.facing = [dx, dy];

  const dashDur = C.DASH_INVULN * (hasBoost ? 1.6 : 1.0);
  player.dashCd = player.dashCdMax;
  player.invuln = Math.max(player.invuln, dashDur);
  player.dashing = dashDur;
  return true;
}

export function updateTimers(player, dt) {
  player.dashCd = Math.max(0, player.dashCd - dt);
  player.invuln = Math.max(0, player.invuln - dt);
  player.dashing = Math.max(0, player.dashing - dt);
}

/** Apply a Pickup to the player; returns the on-screen label. */
export function applyPickup(p, player, nowT) {
  const k = p.kind;
  if (k === "heal") {
    player.hp = Math.min(player.maxHp, player.hp + C.HEAL_AMOUNT);
  } else if (k === "max_hp") {
    if (!player.buffs.has("max_hp")) {
      player.maxHp += C.MAX_HP_BONUS;
      player.hp += C.MAX_HP_BONUS;
    }
    player.buffs.set("max_hp", nowT + PICKUP_BUFF_DURATION.max_hp);
  } else {
    player.buffs.set(k, nowT + PICKUP_BUFF_DURATION[k]);
  }
  return PICKUP_LABEL[k];
}

/** Remove expired buffs, reverting the reversible ones. */
export function decayBuffs(player, nowT) {
  for (const [k, exp] of player.buffs) {
    if (exp > nowT) continue;
    if (k === "max_hp") {
      player.maxHp = Math.max(1, player.maxHp - C.MAX_HP_BONUS);
      player.hp = Math.min(player.hp, player.maxHp);
    }
    player.buffs.delete(k);
  }
}
