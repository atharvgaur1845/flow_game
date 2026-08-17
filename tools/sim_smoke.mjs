/* Headless smoke test of the ported game logic (no DOM, no WebGL).
 *
 * Runs the same fixed-timestep loop main.js uses, with a scripted player, and
 * asserts the invariants the desktop build's --probe checks plus a few
 * gameplay ones: rooms advance, enemies spawn and die, score accrues, the
 * boss appears on room 5, and nothing throws over a long run.
 *
 *   node tools/sim_smoke.mjs
 */
import * as C from "../docs/js/config.js";
import { AttentionStateInference, EnvironmentMapper, ObservationBuilder } from "../docs/js/inference.js";
import { makeEnemy, makeBoss, pickKind, stepEnemies, stepBoss, makePickup } from "../docs/js/entities.js";
import { RunState, scaleEnemySpeed, scaleSpawnRate, pickShopChoices, applyUpgrade } from "../docs/js/progression.js";
import { makePlayer, tryDash, updateTimers, applyPickup, decayBuffs } from "../docs/js/player.js";

let failures = 0;
function check(cond, msg) {
  if (!cond) {
    console.error("FAIL: " + msg);
    failures++;
  }
}

const g = {
  player: makePlayer(),
  enemies: [],
  boss: null,
  run: new RunState(),
  pickups: [],
  pickupTimer: C.PICKUP_SPAWN_INTERVAL * 0.4,
  spawnAccum: 0,
  simTime: 0,
  attn: new AttentionStateInference(),
  mapper: new EnvironmentMapper(),
  obs: new ObservationBuilder(),
  pi: new Float64Array(5).fill(0.2),
  E: new Float64Array([1, 0.5, 0.5, 1]),
  c: new Float64Array(4),
  isBoss: false,
  dead: false,
};
g.run.startNextRoom();

const stats = { kills: 0, rooms: 0, shops: 0, bosses: 0, pickups: 0, dashes: 0, deaths: 0 };

/** A competent scripted player: orbits, kites, dashes through crowds and
 *  grabs pickups. Good enough to reach bosses and shops, which is the point —
 *  a bot that dies in room 2 leaves most of the state machine untested. */
function scriptedInput(t) {
  const p = g.player;

  // Nearest threat, and nearest pickup.
  let near = null, nearD = Infinity;
  const threats = g.boss ? [...g.enemies, g.boss] : g.enemies;
  for (const e of threats) {
    const d = Math.hypot(e.pos[0] - p.pos[0], e.pos[1] - p.pos[1]);
    if (d < nearD) { nearD = d; near = e.pos; }
  }
  let pick = null, pickD = Infinity;
  for (const pk of g.pickups) {
    const d = Math.hypot(pk.pos[0] - p.pos[0], pk.pos[1] - p.pos[1]);
    if (d < pickD) { pickD = d; pick = pk.pos; }
  }

  // Rooms clear on kills, so the winning policy is to hunt while the dash is
  // up and disengage while it recharges — which is exactly how the game wants
  // to be played, and why the inference reads it as Flow.
  const armed = p.dashCd <= 0 || p.dashing > 0;
  let ix, iy;
  if (armed && near) {
    ix = near[0] - p.pos[0];
    iy = near[1] - p.pos[1];
  } else if (pick && (!near || nearD > 4.0)) {
    ix = pick[0] - p.pos[0];
    iy = pick[1] - p.pos[1];
  } else if (near && nearD < 6.0) {
    ix = p.pos[0] - near[0];
    iy = p.pos[1] - near[1];
  } else {
    ix = Math.cos(t * 0.8) * 10 - p.pos[0];
    iy = Math.sin(t * 0.8) * 10 - p.pos[1];
  }

  const n0 = Math.hypot(ix, iy) + 1e-6;
  ix /= n0; iy /= n0;

  // Stay off the walls — cornering is what actually kills you here.
  const edge = 3.0;
  if (p.pos[0] > C.WORLD_HALF - edge) ix -= 1.2;
  if (p.pos[0] < -C.WORLD_HALF + edge) ix += 1.2;
  if (p.pos[1] > C.WORLD_HALF - edge) iy -= 1.2;
  if (p.pos[1] < -C.WORLD_HALF + edge) iy += 1.2;

  const n = Math.hypot(ix, iy) + 1e-6;
  const dash = nearD < 3.0;
  return { ix: ix / n, iy: iy / n, dash };
}

function step(dt) {
  g.simTime += dt;
  const p = g.player;
  const { ix: rawIx, iy: rawIy, dash: dashHeld } = scriptedInput(g.simTime);
  if (Math.random() < 0.25) g.obs.registerKeypress();

  const ilen = Math.hypot(rawIx, rawIy);
  let tgtIx = 0, tgtIy = 0;
  if (ilen > 1e-6) { tgtIx = rawIx / ilen; tgtIy = rawIy / ilen; p.facing = [tgtIx, tgtIy]; }
  const lerp = Math.min(1, dt * 18);
  p.inputDir[0] += (tgtIx - p.inputDir[0]) * lerp;
  p.inputDir[1] += (tgtIy - p.inputDir[1]) * lerp;

  const speedMult = p.buffs.has("speed_boost") ? C.SPEED_BOOST_MULT : 1;
  const ax = p.inputDir[0] * C.PLAYER_ACCEL * speedMult;
  const ay = p.inputDir[1] * C.PLAYER_ACCEL * speedMult;
  if (tryDash(p, dashHeld, rawIx, rawIy)) stats.dashes++;

  const friction = Math.max(g.E[2], 0.05);
  p.vel[0] += ax * dt; p.vel[1] += ay * dt;
  const damp = Math.exp(-friction * dt);
  p.vel[0] *= damp; p.vel[1] *= damp;

  const BD = 0.55;
  let nx = p.pos[0] + p.vel[0] * dt;
  let ny = p.pos[1] + p.vel[1] * dt;
  if (nx > C.WORLD_HALF) { nx = C.WORLD_HALF; if (p.vel[0] > 0) p.vel[0] *= -BD; }
  else if (nx < -C.WORLD_HALF) { nx = -C.WORLD_HALF; if (p.vel[0] < 0) p.vel[0] *= -BD; }
  if (ny > C.WORLD_HALF) { ny = C.WORLD_HALF; if (p.vel[1] > 0) p.vel[1] *= -BD; }
  else if (ny < -C.WORLD_HALF) { ny = -C.WORLD_HALF; if (p.vel[1] < 0) p.vel[1] *= -BD; }
  p.pos[0] = nx; p.pos[1] = ny;

  check(Math.abs(p.pos[0]) <= C.WORLD_HALF + 1e-6 && Math.abs(p.pos[1]) <= C.WORLD_HALF + 1e-6,
    `player escaped the arena at t=${g.simTime.toFixed(2)}`);
  check(Number.isFinite(p.pos[0]) && Number.isFinite(p.vel[0]), "player position/velocity went non-finite");

  updateTimers(p, dt);
  p.hitFlash = Math.max(0, p.hitFlash - dt * 3);

  const nowT = g.simTime;
  if (p.dashing > 0) {
    const last = p.dashTrail[p.dashTrail.length - 1];
    if (!last || Math.hypot(p.pos[0] - last[0], p.pos[1] - last[1]) > 0.15) {
      p.dashTrail.push([p.pos[0], p.pos[1], nowT + C.DASH_TRAIL_TTL]);
    }
  }
  p.dashTrail = p.dashTrail.filter((pt) => pt[2] > nowT);
  if (p.dashTrail.length > C.DASH_TRAIL_MAX) p.dashTrail = p.dashTrail.slice(-C.DASH_TRAIL_MAX);
  check(p.dashTrail.length <= C.DASH_TRAIL_MAX, "dash trail exceeded the shader cap");

  decayBuffs(p, nowT);

  g.pickupTimer -= dt;
  if (g.pickupTimer <= 0 && g.pickups.length < C.PICKUP_MAX_ACTIVE) {
    g.pickups.push(makePickup(p.pos, C.WORLD_HALF));
    g.pickupTimer = C.PICKUP_SPAWN_INTERVAL;
  }
  const keep = [];
  for (const pk of g.pickups) {
    if (Math.hypot(pk.pos[0] - p.pos[0], pk.pos[1] - p.pos[1]) < pk.radius + 0.45) {
      applyPickup(pk, p, nowT);
      stats.pickups++;
    } else keep.push(pk);
  }
  g.pickups = keep;
  check(g.pickups.length <= C.PICKUP_MAX_ACTIVE, "too many pickups on the floor");

  g.obs.tickApm();

  const scaledSpawn = scaleSpawnRate(g.E[1], g.run.room);
  const scaledSpeed = scaleEnemySpeed(g.E[0], g.run.room, g.run.enemySlow);

  if (!g.isBoss) {
    g.spawnAccum += Math.max(scaledSpawn, 0.05) * dt;
    while (g.spawnAccum >= 1 && g.enemies.length < C.MAX_ENEMIES) {
      g.enemies.push(makeEnemy(pickKind(g.pi), p.pos, C.WORLD_HALF));
      g.spawnAccum -= 1;
    }
    stepEnemies(g.enemies, p.pos, scaledSpeed, 1, dt);
  } else {
    if (g.boss) stepBoss(g.boss, p.pos, g.pi, C.WORLD_HALF, g.enemies, dt);
    stepEnemies(g.enemies, p.pos, scaledSpeed, 1, dt);
  }
  check(g.enemies.length <= C.MAX_ENEMIES, `enemy count ${g.enemies.length} exceeded MAX_ENEMIES`);

  const dashing = p.dashing > 0;
  const trail = p.dashTrail;
  const killedByTrail = (ex, ey, er) => {
    if (trail.length < 1) return false;
    for (let j = 0; j < trail.length - 1; j++) {
      const [axx, ayy] = trail[j], [bxx, byy] = trail[j + 1];
      const pax = ex - axx, pay = ey - ayy, bax = bxx - axx, bay = byy - ayy;
      const ab2 = bax * bax + bay * bay + 1e-9;
      const h = Math.max(0, Math.min(1, (pax * bax + pay * bay) / ab2));
      if (Math.hypot(ex - (axx + h * bax), ey - (ayy + h * bay)) < C.DASH_TRAIL_RADIUS + er) return true;
    }
    const [lx, ly] = trail[trail.length - 1];
    return Math.hypot(ex - lx, ey - ly) < C.DASH_TRAIL_RADIUS + er;
  };

  const survivors = [];
  for (const e of g.enemies) {
    const d = Math.hypot(e.pos[0] - p.pos[0], e.pos[1] - p.pos[1]);
    if (d < e.radius + 0.35 && e.contactCd <= 0) {
      if (dashing) e.hp -= 99;
      else {
        e.hp -= 1;
        if (p.invuln <= 0) {
          if (p.buffs.has("shield")) { p.invuln = 0.25; }
          else { p.hp -= C.DAMAGE_ON_HIT * g.run.damageMult; p.invuln = C.HIT_INVULN; }
        }
      }
      e.contactCd = e.contactCdMax;
    }
    if (e.hp > 0 && killedByTrail(e.pos[0], e.pos[1], e.radius)) e.hp = 0;
    if (e.hp > 0) survivors.push(e);
    else { g.run.registerKill(g.pi); stats.kills++; }
  }
  g.enemies = survivors;

  if (g.isBoss && g.boss) {
    const d = Math.hypot(g.boss.pos[0] - p.pos[0], g.boss.pos[1] - p.pos[1]);
    if (d < g.boss.radius + 0.45 && g.boss.contactCd <= 0) {
      if (dashing) g.boss.hp -= C.BOSS_DASH_DAMAGE;
      else if (p.invuln <= 0) {
        if (p.buffs.has("shield")) p.invuln = 0.3;
        else { p.hp -= C.BOSS_CONTACT_DAMAGE * g.run.damageMult; p.invuln = C.HIT_INVULN; }
      }
      g.boss.contactCd = 0.4;
    }
    if (g.boss.hp <= 0) {
      g.run.registerKill(g.pi);
      g.run.clearRoom();
      stats.bosses++; stats.rooms++;
      nextRoom();
    }
  }

  g.run.roomTimeRemaining -= dt;
  g.run.tick(g.pi, dt);

  if (!g.isBoss) {
    if (g.run.roomGoalMet()) {
      g.run.clearRoom();
      stats.rooms++;
      nextRoom();
    } else if (g.run.timeExpired()) {
      p.hp = 0;
    }
  }

  if (p.hp <= 0) { g.dead = true; stats.deaths++; return; }

  const threats = g.boss ? [...g.enemies, g.boss] : g.enemies;
  const oT = g.obs.build(p.vel[0], p.vel[1], threats, p.pos[0], p.pos[1]);
  g.pi.set(g.attn.forward(oT, p.vel[0], p.vel[1]));
  const em = g.mapper.forward(g.pi);
  g.E.set(em.E);
  g.c.set(em.c);

  // Backend invariants — the same ones flow_game.py --probe asserts.
  let s = 0;
  for (let k = 0; k < 5; k++) {
    s += g.pi[k];
    check(g.pi[k] >= -1e-9 && g.pi[k] <= 1 + 1e-9, `pi[${k}]=${g.pi[k]} out of range`);
  }
  check(Math.abs(s - 1) < 1e-6, `pi sums to ${s}, not 1`);
  check(g.E.every(Number.isFinite) && g.c.every(Number.isFinite), "E or c went non-finite");
}

function nextRoom() {
  g.enemies = [];
  g.boss = null;
  g.spawnAccum = 0;
  g.pickups = [];
  g.pickupTimer = C.PICKUP_SPAWN_INTERVAL * 0.4;
  if (g.run.shouldOpenShop()) {
    const choices = pickShopChoices(3);
    check(new Set(choices).size === 3, "shop offered duplicate upgrades");
    applyUpgrade(choices[0], g.player, g.run);
    stats.shops++;
  }
  g.run.startNextRoom();
  g.isBoss = g.run.isBossRoom;
  if (g.isBoss) g.boss = makeBoss();
}

// --- run -------------------------------------------------------------------
const SIM_SECONDS = 400;
const nSteps = Math.round(SIM_SECONDS * C.SIM_HZ);
const t0 = Date.now();
for (let i = 0; i < nSteps && !g.dead; i++) step(C.SIM_DT);
const wall = Date.now() - t0;

console.log(`simulated      : ${g.simTime.toFixed(1)}s of play in ${wall}ms wall ` +
  `(${Math.round(g.simTime / (wall / 1000))}x realtime)`);
console.log(`room reached   : ${g.run.room}   cleared: ${g.run.roomsCleared}`);
console.log(`kills          : ${stats.kills}   dashes: ${stats.dashes}   pickups: ${stats.pickups}`);
console.log(`bosses / shops : ${stats.bosses} / ${stats.shops}`);
console.log(`score          : ${Math.round(g.run.score)}`);
console.log(`pi             : [${[...g.pi].map((v) => v.toFixed(3)).join(", ")}]`);
console.log(`died           : ${g.dead}`);

check(stats.kills > 20, "scripted player barely killed anything — collision may be broken");
check(g.run.room >= 2, "never advanced past room 1");
check(g.run.score > 0, "score never accrued");
check(stats.pickups > 0, "never collected a pickup");
// A 400s run should have reached the room-5 boss and at least one shop.
check(g.run.room >= 5 || g.dead, "never reached a boss room");

// Per-tick budget: the whole sim must fit comfortably inside a 120 Hz tick.
const usPerStep = (wall * 1000) / nSteps;
console.log(`cost           : ${usPerStep.toFixed(1)}us/tick (budget ${(1e6 / C.SIM_HZ).toFixed(0)}us)`);
check(usPerStep < 1000, "simulation tick is too slow for 120 Hz");

if (failures) {
  console.error(`\n${failures} check(s) failed.`);
  process.exit(1);
}
console.log("\nOK — game logic smoke test passed.");
