/* Flow — Neon Arena, web build. Entry point and game loop.
 *
 * The simulation is a faithful port of flow_game.py's main loop: same order
 * of operations, same constants, and the same inference weights, so the
 * browser build plays like the desktop build rather than merely resembling
 * it. Everything web-specific — attract mode, onboarding, persistence,
 * audio, the report card — is layered on top without touching that order.
 *
 * Fixed 120 Hz simulation, free-running render. The desktop build steps once
 * per frame at up to 144 Hz and several gameplay lerps are per-frame rather
 * than per-second, so a fixed tick is what keeps a 60 Hz laptop and a 240 Hz
 * monitor playing the same game. */

import * as C from "./config.js";
import { AttentionStateInference, EnvironmentMapper, ObservationBuilder } from "./inference.js";
import {
  makeEnemy, makeBoss, pickKind, stepEnemies, stepBoss, makePickup, PICKUP_CODE,
} from "./entities.js";
import {
  RunState, scaleEnemySpeed, scaleSpawnRate, pickShopChoices, applyUpgrade,
} from "./progression.js";
import { makePlayer, tryDash, updateTimers, applyPickup, decayBuffs } from "./player.js";
import { Renderer } from "./renderer.js";
import { AudioEngine } from "./audio.js";
import { store } from "./storage.js";
import { Input, IS_TOUCH } from "./input.js";
import { HUD } from "./hud.js";
import { UI } from "./ui.js";
import { Ribbon, RIBBON_PALETTE, RIBBON_PALETTE_CB } from "./ribbon.js";
import { MilestoneTracker } from "./milestones.js";

const ST = {
  TITLE: "title",     // attract mode running behind the menu
  PLAYING: "playing",
  BOSS: "boss",
  SHOP: "shop",
  PAUSED: "paused",
  REPORT: "report",
};

// ─────────────────────────────── boot ───────────────────────────────

const canvas = document.getElementById("scene");
let renderer;
try {
  renderer = new Renderer(canvas);
} catch (err) {
  document.getElementById("unsupported-detail").textContent = String(err.message || err);
  document.getElementById("screen-title").classList.add("hidden");
  document.getElementById("screen-unsupported").classList.remove("hidden");
  throw err;
}

const audio = new AudioEngine();
const input = new Input(store);
const hud = new HUD();
const ribbon = new Ribbon(document.getElementById("ribbon"));
const tracker = new MilestoneTracker(store, onMilestone);

const g = {
  state: ST.TITLE,
  attract: true,
  player: makePlayer(),
  enemies: [],
  boss: null,
  run: new RunState(),
  pickups: [],
  pickupTimer: C.PICKUP_SPAWN_INTERVAL * 0.4,
  spawnAccum: 0,
  shopChoices: [],
  simTime: 0,
  attn: new AttentionStateInference(),
  mapper: new EnvironmentMapper(),
  obs: new ObservationBuilder(),
  pi: new Float64Array(5).fill(0.2),
  E: new Float64Array([1, 0.5, 0.5, 1]),
  c: new Float64Array(4),
  // presentation-only
  dashReadyPop: 0,
  prevDashCd: 0,
  heartbeatT: 0,
  killSfxBudget: 0,
  piHistory: [],
  piSampleAccum: 0,
  audioAccum: 0,
  bossAnnounced: false,
};

const ui = new UI({
  store, audio, renderer, input, ribbon, tracker,
  on: {
    startRun, toTitle, resume, abandon,
    setMuted, setRibbonVisible,
  },
});

applyStoredSettings();
ui.refreshTitle();
input.attach();
input.onGameKey = () => { if (isPlaying()) g.obs.registerKeypress(); };
if (IS_TOUCH) setupTouch();
beginAttract();

// ─────────────────────────────── settings ───────────────────────────────

function applyStoredSettings() {
  const s = store.settings;
  audio.musicVol = s.musicVolume;
  audio.sfxVol = s.sfxVolume;
  audio.muted = s.muted;
  document.body.classList.toggle("cb", s.colorblind);
  ribbon.setPalette(s.colorblind ? RIBBON_PALETTE_CB : RIBBON_PALETTE);
  // Reserve (or reclaim) the ribbon band before measuring the canvas.
  hud.setRibbonVisible(s.showRibbon);
  renderer.setQuality(s.quality);
  renderer.setAutoScale(s.autoScale);
  ribbon.resize();
  document.getElementById("quick-mute").classList.toggle("muted", s.muted);
}

function setMuted(m) {
  store.setSetting("muted", m);
  audio.setMuted(m);
  document.getElementById("quick-mute").classList.toggle("muted", m);
  const box = document.getElementById("set-mute");
  if (box) box.checked = m;
}

function setRibbonVisible(on) {
  hud.setRibbonVisible(on);
  // The canvas stops above the ribbon band, so showing or hiding it changes
  // the viewport the arena renders into.
  renderer.resize();
  ribbon.resize();
}

// ─────────────────────────────── state flow ───────────────────────────────

function isPlaying() {
  return g.state === ST.PLAYING || g.state === ST.BOSS;
}

function beginAttract() {
  g.attract = true;
  g.state = ST.TITLE;
  resetWorld();
  g.run.startNextRoom();
  hud.show(false);
  ui.show("screen-title");
}

function resetWorld() {
  g.player = makePlayer();
  g.enemies = [];
  g.boss = null;
  g.run = new RunState();
  g.pickups = [];
  g.pickupTimer = C.PICKUP_SPAWN_INTERVAL * 0.4;
  g.spawnAccum = 0;
  g.attn.reset();
  g.obs.reset();
  g.pi.fill(0.2);
  g.piHistory = [];
  g.piSampleAccum = 0;
  g.bossAnnounced = false;
  ribbon.clear();
}

function startRun() {
  audio.unlock();
  audio.startMusic();
  g.attract = false;
  resetWorld();
  g.run.startNextRoom();
  g.state = ST.PLAYING;
  ui.hideAll();
  hud.show(true);
  hud.hint(null);
  g._hintMoveDone = store.hasSeen("hint.move");
  g._hintDashDone = store.hasSeen("hint.dash");
}

function toTitle() {
  audio.stopMusic();
  ui.refreshTitle();
  beginAttract();
}

function pause() {
  if (!isPlaying()) return;
  g._pausedFrom = g.state;
  g.state = ST.PAUSED;
  audio.duck(0.45, 0.5);
  ui.show("screen-pause");
}

function resume() {
  if (g.state !== ST.PAUSED) return;
  g.state = g._pausedFrom || ST.PLAYING;
  ui.hideAll();
}

function abandon() {
  g.player.hp = 0;
  g.state = g._pausedFrom || ST.PLAYING;
  endRun();
}

function endRun() {
  audio.gameOver();
  audio.stopMusic();
  store.commitRun(g.run, g.run.piSeconds);
  g.state = ST.REPORT;
  hud.show(false);
  ui.renderReport(g.run, g.piHistory);
  ui.refreshTitle();
}

function enterShop() {
  g.shopChoices = pickShopChoices(3);
  g.state = ST.SHOP;
  audio.shopOpen();
  audio.duck(0.4, 0.8);
  hud.show(false);
  ui.renderShop(g.shopChoices, g.run, pickUpgrade);
}

function pickUpgrade(slot) {
  if (g.state !== ST.SHOP) return;
  applyUpgrade(g.shopChoices[slot], g.player, g.run);
  audio.uiConfirm();
  g.run.startNextRoom();
  enterRoom();
}

/** Common transition into whatever the freshly-started room is. */
function enterRoom() {
  ui.hideAll();
  hud.show(true);
  if (g.run.isBossRoom) {
    g.state = ST.BOSS;
    g.boss = makeBoss();
    if (!g.attract) {
      audio.bossSpawn();
      if (store.markSeen("hint.boss")) {
        hud.hint("Only <kbd>DASH</kbd> damages the boss");
        setTimeout(() => hud.hint(null), 4500);
      }
    }
  } else {
    g.state = ST.PLAYING;
  }
}

function startNextRoomOrShop() {
  g.enemies = [];
  g.boss = null;
  g.spawnAccum = 0;
  g.pickups = [];
  g.pickupTimer = C.PICKUP_SPAWN_INTERVAL * 0.4;
  if (!g.attract && g.run.shouldOpenShop()) {
    enterShop();
  } else {
    g.run.startNextRoom();
    enterRoom();
  }
}

function onMilestone(m) {
  audio.milestone();
  hud.toast("MILESTONE", m.name, m.desc);
}

// ─────────────────────────────── simulation ───────────────────────────────

/** One fixed simulation tick. Mirrors the per-frame order of flow_game.py. */
function step(dt) {
  g.simTime += dt;
  g.killSfxBudget = Math.min(6, g.killSfxBudget + dt * 30);

  if (!isPlaying() && g.state !== ST.TITLE) return;
  if (g.state === ST.TITLE && !g.attract) return;

  const p = g.player;
  const attract = g.attract;

  // --- input ---------------------------------------------------------------
  let rawIx = 0, rawIy = 0, dashHeld = false;
  if (attract) {
    ({ rawIx, rawIy, dashHeld } = autopilot(g.simTime));
  } else {
    [rawIx, rawIy] = input.moveVector();
    dashHeld = input.dashHeld();
  }

  const ilen = Math.hypot(rawIx, rawIy);
  let tgtIx = 0, tgtIy = 0;
  if (ilen > 1e-6) {
    tgtIx = rawIx / ilen;
    tgtIy = rawIy / ilen;
    p.facing = [tgtIx, tgtIy];
  }
  // Critically-damped lerp toward the target input direction so brief key
  // flicker is smoothed but the response stays snappy (~3 frames).
  const lerp = Math.min(1, dt * 18);
  p.inputDir[0] += (tgtIx - p.inputDir[0]) * lerp;
  p.inputDir[1] += (tgtIy - p.inputDir[1]) * lerp;

  const speedMult = p.buffs.has("speed_boost") ? C.SPEED_BOOST_MULT : 1;
  const ax = p.inputDir[0] * C.PLAYER_ACCEL * speedMult;
  const ay = p.inputDir[1] * C.PLAYER_ACCEL * speedMult;

  if (tryDash(p, dashHeld, rawIx, rawIy)) audio.dash();

  // --- physics -------------------------------------------------------------
  const friction = Math.max(g.E[2], 0.05);
  p.vel[0] += ax * dt;
  p.vel[1] += ay * dt;
  const damp = Math.exp(-friction * dt);
  p.vel[0] *= damp;
  p.vel[1] *= damp;

  // Boundary bounce: clamp to the wall and invert the *outward* velocity
  // component once. Only flipping when actually moving outward avoids the
  // double-reflection jitter a fast dash would otherwise cause.
  const BOUNCE_DAMP = 0.55;
  let nx = p.pos[0] + p.vel[0] * dt;
  let ny = p.pos[1] + p.vel[1] * dt;
  if (nx > C.WORLD_HALF) {
    nx = C.WORLD_HALF;
    if (p.vel[0] > 0) p.vel[0] = -p.vel[0] * BOUNCE_DAMP;
  } else if (nx < -C.WORLD_HALF) {
    nx = -C.WORLD_HALF;
    if (p.vel[0] < 0) p.vel[0] = -p.vel[0] * BOUNCE_DAMP;
  }
  if (ny > C.WORLD_HALF) {
    ny = C.WORLD_HALF;
    if (p.vel[1] > 0) p.vel[1] = -p.vel[1] * BOUNCE_DAMP;
  } else if (ny < -C.WORLD_HALF) {
    ny = -C.WORLD_HALF;
    if (p.vel[1] < 0) p.vel[1] = -p.vel[1] * BOUNCE_DAMP;
  }
  p.pos[0] = nx;
  p.pos[1] = ny;

  g.prevDashCd = p.dashCd;
  updateTimers(p, dt);
  if (g.prevDashCd > 0 && p.dashCd <= 0) {
    g.dashReadyPop = 1;
    if (!attract) audio.dashReady();
  }
  g.dashReadyPop = Math.max(0, g.dashReadyPop - dt * 2.2);
  p.hitFlash = Math.max(0, p.hitFlash - dt * 3);

  // --- lethal dash trail ---------------------------------------------------
  const nowT = g.simTime;
  if (p.dashing > 0) {
    const last = p.dashTrail[p.dashTrail.length - 1];
    if (!last || Math.hypot(p.pos[0] - last[0], p.pos[1] - last[1]) > 0.15) {
      p.dashTrail.push([p.pos[0], p.pos[1], nowT + C.DASH_TRAIL_TTL]);
    }
  }
  p.dashTrail = p.dashTrail.filter((pt) => pt[2] > nowT);
  if (p.dashTrail.length > C.DASH_TRAIL_MAX) {
    p.dashTrail = p.dashTrail.slice(-C.DASH_TRAIL_MAX);
  }

  decayBuffs(p, nowT);

  // --- pickups -------------------------------------------------------------
  g.pickupTimer -= dt;
  if (g.pickupTimer <= 0 && g.pickups.length < C.PICKUP_MAX_ACTIVE) {
    g.pickups.push(makePickup(p.pos, C.WORLD_HALF));
    g.pickupTimer = C.PICKUP_SPAWN_INTERVAL;
  }
  const remaining = [];
  for (const pk of g.pickups) {
    if (Math.hypot(pk.pos[0] - p.pos[0], pk.pos[1] - p.pos[1]) < pk.radius + 0.45) {
      const label = applyPickup(pk, p, nowT);
      if (!attract) {
        audio.pickup(pk.kind);
        hud.banner(label);
      }
    } else {
      remaining.push(pk);
    }
  }
  g.pickups = remaining;

  g.obs.tickApm();

  // --- enemies / boss ------------------------------------------------------
  const scaledSpawn = scaleSpawnRate(g.E[1], g.run.room);
  const scaledSpeed = scaleEnemySpeed(g.E[0], g.run.room, g.run.enemySlow);

  if (g.state !== ST.BOSS) {
    g.spawnAccum += Math.max(scaledSpawn, 0.05) * dt;
    while (g.spawnAccum >= 1 && g.enemies.length < C.MAX_ENEMIES) {
      g.enemies.push(makeEnemy(pickKind(g.pi), p.pos, C.WORLD_HALF));
      g.spawnAccum -= 1;
    }
    stepEnemies(g.enemies, p.pos, scaledSpeed, 1, dt);
  } else {
    if (g.boss) {
      const evts = stepBoss(g.boss, p.pos, g.pi, C.WORLD_HALF, g.enemies, dt);
      if (!attract && evts.includes("teleport")) audio.uiMove();
    }
    stepEnemies(g.enemies, p.pos, scaledSpeed, 1, dt);
  }

  // --- collisions ----------------------------------------------------------
  const dashing = p.dashing > 0;
  const trail = p.dashTrail;

  const killedByTrail = (ex, ey, er) => {
    if (trail.length < 1) return false;
    for (let j = 0; j < trail.length - 1; j++) {
      const [axx, ayy] = trail[j];
      const [bxx, byy] = trail[j + 1];
      const pax = ex - axx, pay = ey - ayy;
      const bax = bxx - axx, bay = byy - ayy;
      const ab2 = bax * bax + bay * bay + 1e-9;
      const h = Math.max(0, Math.min(1, (pax * bax + pay * bay) / ab2));
      if (Math.hypot(ex - (axx + h * bax), ey - (ayy + h * bay)) < C.DASH_TRAIL_RADIUS + er) {
        return true;
      }
    }
    const [lx, ly] = trail[trail.length - 1];
    return Math.hypot(ex - lx, ey - ly) < C.DASH_TRAIL_RADIUS + er;
  };

  const survivors = [];
  for (const e of g.enemies) {
    const d = Math.hypot(e.pos[0] - p.pos[0], e.pos[1] - p.pos[1]);
    if (d < e.radius + 0.35 && e.contactCd <= 0) {
      if (dashing) {
        e.hp -= 99; // dash obliterates enemies
      } else {
        e.hp -= 1;
        if (p.invuln <= 0) {
          if (p.buffs.has("shield")) {
            p.hitFlash = 0.4;
            p.invuln = 0.25;
            if (!attract) audio.shieldBlock();
          } else {
            p.hp -= C.DAMAGE_ON_HIT * g.run.damageMult;
            p.invuln = C.HIT_INVULN;
            p.hitFlash = 1.0;
            if (!attract) audio.hit();
          }
        }
      }
      e.contactCd = e.contactCdMax;
    }
    // Trail kill: any enemy intersecting a recent dash segment dies, even
    // after the dash itself has ended.
    if (e.hp > 0 && killedByTrail(e.pos[0], e.pos[1], e.radius)) e.hp = 0;
    if (e.hp > 0) {
      survivors.push(e);
    } else {
      g.run.registerKill(g.pi);
      if (!attract && g.killSfxBudget >= 1) {
        g.killSfxBudget -= 1;
        audio.kill(e.kind);
      }
    }
  }
  g.enemies = survivors;

  if (g.state === ST.BOSS && g.boss) {
    const d = Math.hypot(g.boss.pos[0] - p.pos[0], g.boss.pos[1] - p.pos[1]);
    if (d < g.boss.radius + 0.45 && g.boss.contactCd <= 0) {
      if (dashing) {
        g.boss.hp -= C.BOSS_DASH_DAMAGE;
        g.boss.hitFlash = 1.0;
        if (!attract) audio.bossHit();
      } else if (p.invuln <= 0) {
        if (p.buffs.has("shield")) {
          p.hitFlash = 0.4;
          p.invuln = 0.3;
          if (!attract) audio.shieldBlock();
        } else {
          p.hp -= C.BOSS_CONTACT_DAMAGE * g.run.damageMult;
          p.invuln = C.HIT_INVULN;
          p.hitFlash = 1.0;
          if (!attract) audio.hit();
        }
      }
      g.boss.contactCd = 0.4;
    }
    if (g.boss.hp <= 0) {
      g.run.registerKill(g.pi);
      g.run.clearRoom();
      if (!attract) audio.bossDeath();
      startNextRoomOrShop();
    }
  }

  // --- progression ---------------------------------------------------------
  g.run.roomTimeRemaining -= dt;
  g.run.tick(g.pi, dt);

  if (attract) {
    // Attract mode never dies and never shops: it just keeps the arena alive.
    p.hp = p.maxHp;
    if (g.run.roomTimeRemaining <= 0) g.run.roomTimeRemaining = g.run.roomTimeTotal;
    if (g.run.roomGoalMet()) {
      g.run.clearRoom();
      g.enemies = [];
      g.spawnAccum = 0;
      g.run.startNextRoom();
      g.state = ST.TITLE;
    }
  } else if (g.state === ST.PLAYING) {
    if (g.run.roomGoalMet()) {
      g.run.clearRoom();
      audio.roomClear();
      startNextRoomOrShop();
    } else if (g.run.timeExpired()) {
      p.hp = 0; // failed the kill quota — death
    }
  }

  if (!attract && p.hp <= 0 && g.state !== ST.REPORT) {
    p.hp = 0;
    endRun();
    return;
  }

  // --- inference -----------------------------------------------------------
  const threats = g.boss ? [...g.enemies, g.boss] : g.enemies;
  const oT = g.obs.build(p.vel[0], p.vel[1], threats, p.pos[0], p.pos[1]);
  g.pi.set(g.attn.forward(oT, p.vel[0], p.vel[1]));
  const em = g.mapper.forward(g.pi);
  g.E.set(em.E);
  g.c.set(em.c);

  // --- web-only per-tick extras -------------------------------------------
  if (!attract) {
    tracker.update(
      { run: g.run, player: p, pi: g.pi, records: store.records, state: g.state, simTime: g.simTime },
      dt);
    updateHints(oT);
    updateHeartbeat(dt);

    g.piSampleAccum += dt;
    if (g.piSampleAccum >= 0.25) {
      g.piSampleAccum = 0;
      g.piHistory.push(Float64Array.from(g.pi));
    }
  }

  ribbon.push(g.pi, dt);

  // Feed the score its state at 20 Hz — matches the smoothing constants in
  // AudioEngine.setState.
  g.audioAccum += dt;
  if (g.audioAccum >= 0.05) {
    g.audioAccum = 0;
    audio.setState(g.pi, musicIntensity(oT));
  }
}

function musicIntensity(oT) {
  const roomPart = Math.min(1, g.run.room / 12) * 0.4;
  const threatPart = Math.min(1, oT[3] * 3.2) * 0.32;
  const bossPart = g.state === ST.BOSS ? 0.24 : 0;
  const hpPart = (1 - g.player.hp / Math.max(g.player.maxHp, 1)) * 0.14;
  return Math.max(0, Math.min(1, 0.18 + roomPart + threatPart + bossPart + hpPart));
}

// ─────────────────────────────── onboarding ───────────────────────────────

function updateHints(oT) {
  const p = g.player;
  if (!g._hintMoveDone) {
    if (Math.hypot(p.vel[0], p.vel[1]) > 1.5) {
      g._hintMoveDone = true;
      store.markSeen("hint.move");
      hud.hint(null);
    } else {
      hud.hint(IS_TOUCH ? "Drag anywhere on the left to move"
        : `<kbd>${input.bindLabel("up")}</kbd> <kbd>${input.bindLabel("left")}</kbd> <kbd>${input.bindLabel("down")}</kbd> <kbd>${input.bindLabel("right")}</kbd> to move`);
      return;
    }
  }
  if (!g._hintDashDone) {
    if (p.dashing > 0) {
      g._hintDashDone = true;
      store.markSeen("hint.dash");
      hud.hint(null);
    } else if (oT[3] > 0.18 && p.dashCd <= 0) {
      // A threat is close and the dash is available — the exact moment the
      // prompt is useful, and never again after it lands.
      hud.hint(IS_TOUCH ? "Tap <kbd>DASH</kbd> — it kills on contact"
        : `<kbd>${input.bindLabel("dash")}</kbd> to dash — it kills on contact`);
    } else {
      hud.hint(null);
    }
  }
}

function updateHeartbeat(dt) {
  const frac = g.player.hp / Math.max(g.player.maxHp, 1);
  if (frac >= 0.35 || g.player.hp <= 0) {
    g.heartbeatT = 0;
    return;
  }
  g.heartbeatT -= dt;
  if (g.heartbeatT <= 0) {
    audio.heartbeat();
    g.heartbeatT = 0.55 + frac * 1.6; // faster as it gets worse
  }
}

// ─────────────────────────────── attract mode ───────────────────────────────

/** Scripted play behind the title screen. The phases are chosen to sweep the
 *  inference engine through all five archetypes, so the arena visibly changes
 *  colour and the score changes character before anyone presses a key. */
function autopilot(t) {
  const cycle = t % 24;
  let rawIx = 0, rawIy = 0, dashHeld = false, apm = 0;

  if (cycle < 6) {
    // Smooth wide circles -> Flow
    rawIx = Math.cos(t * 0.9);
    rawIy = Math.sin(t * 0.9);
    apm = 0.12;
  } else if (cycle < 12) {
    // Fast erratic darting + dashes -> Arousal / Overload
    rawIx = Math.sin(t * 5.3) + Math.cos(t * 2.1);
    rawIy = Math.cos(t * 4.7) - Math.sin(t * 3.3);
    dashHeld = Math.sin(t * 1.7) > 0.86;
    apm = 0.55;
  } else if (cycle < 17) {
    // Near-stillness -> Apathy
    rawIx = Math.sin(t * 0.3) * 0.04;
    rawIy = Math.cos(t * 0.25) * 0.04;
    apm = 0.01;
  } else {
    // Measured stop-start repositioning -> Tactical
    const on = Math.sin(t * 1.4) > 0.1;
    rawIx = on ? Math.cos(t * 0.5) : 0;
    rawIy = on ? Math.sin(t * 0.5) : 0;
    apm = 0.18;
  }

  if (Math.random() < apm) g.obs.registerKeypress();
  return { rawIx, rawIy, dashHeld };
}

// ─────────────────────────────── loop ───────────────────────────────

let lastTs = performance.now();
let accumulator = 0;

function frame(ts) {
  requestAnimationFrame(frame);
  const frameMs = ts - lastTs;
  lastTs = ts;
  const dtReal = Math.min(frameMs / 1000, 0.25);

  accumulator += dtReal;
  let steps = 0;
  while (accumulator >= C.SIM_DT && steps < C.MAX_SUBSTEPS) {
    step(C.SIM_DT);
    accumulator -= C.SIM_DT;
    steps++;
  }
  // If we blew the substep budget (tab restore, heavy GC), drop the backlog
  // rather than spiralling.
  if (steps === C.MAX_SUBSTEPS) accumulator = 0;

  const p = g.player;
  const reduced = store.settings.reducedMotion;
  const hpFrac = p.hp / Math.max(p.maxHp, 1);
  renderer.render({
    time: g.simTime,
    simTime: g.simTime,
    player: p,
    enemies: g.enemies,
    boss: g.boss,
    pickups: g.pickups,
    pickupCodes: PICKUP_CODE,
    pi: g.pi,
    c: g.c,
    E: g.E,
    fx: reduced ? 0 : 1,
    dashReady: p.dashCd <= 0 ? 1 : 1 - p.dashCd / Math.max(p.dashCdMax, 1e-3),
    dashReadyPop: g.dashReadyPop,
    lowHp: hpFrac < 0.35 && p.hp > 0 ? (0.35 - hpFrac) / 0.35 : 0,
    shield: p.buffs.has("shield") ? 1 : 0,
  });
  renderer.observeFrame(frameMs);

  if (isPlaying()) hud.update(g, dtReal);
  // The ribbon keeps streaming on the title screen too, so the menu shows the
  // inference engine actually working on the attract-mode player.
  if (store.settings.showRibbon) ribbon.draw();
  ui.updatePerf(renderer._frameMs, renderer.renderScale);
}

requestAnimationFrame(frame);

// ─────────────────────────────── keyboard routing ───────────────────────────────

const IGNORED_KEYS = new Set([
  "Tab", "CapsLock", "ShiftLeft", "ShiftRight", "ControlLeft", "ControlRight",
  "AltLeft", "AltRight", "MetaLeft", "MetaRight", "ContextMenu", "NumLock",
  "ScrollLock", "Pause", "PrintScreen", "Insert",
]);

window.addEventListener("keydown", (e) => {
  if (input.capturing) return;          // Settings is capturing a rebind
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  // e.target is only an Element when something focusable has focus; it is
  // window/document otherwise, and for synthetic events.
  const tgt = e.target;
  if (tgt instanceof Element && tgt.matches("input, select, textarea, button")) return;

  const code = e.code;

  if (code === "KeyM") {
    setMuted(!store.settings.muted);
    return;
  }

  switch (g.state) {
    case ST.TITLE:
      if (ui.current !== "screen-title") {
        // A sub-screen (settings / how / milestones) is open.
        if (code === "Escape") {
          if (ui.current === "screen-settings") ui.closeSettings();
          else ui.show("screen-title");
        }
        return;
      }
      // Any key starts the run: this is also the gesture that unlocks audio,
      // which is why there is no "click to enable sound" nag anywhere.
      if (IGNORED_KEYS.has(code) || /^F\d{1,2}$/.test(code)) return;
      e.preventDefault();
      startRun();
      break;

    case ST.PLAYING:
    case ST.BOSS:
      if (code === "Escape") {
        e.preventDefault();
        pause();
      }
      break;

    case ST.SHOP:
      if (code === "Digit1" || code === "Numpad1") pickUpgrade(0);
      else if (code === "Digit2" || code === "Numpad2") pickUpgrade(1);
      else if (code === "Digit3" || code === "Numpad3") pickUpgrade(2);
      break;

    case ST.PAUSED:
      if (ui.current === "screen-settings") {
        if (code === "Escape") ui.closeSettings();
        return;
      }
      if (code === "Escape") resume();
      break;

    case ST.REPORT:
      if (code === "Enter" || code === "NumpadEnter" || code === "Space") {
        e.preventDefault();
        startRun();
      } else if (code === "Escape") {
        toTitle();
      }
      break;
  }
}, true);

// Any pointer press on the title also starts the run (and unlocks audio),
// but not when it lands on a menu button, which has its own handler.
document.getElementById("screen-title").addEventListener("pointerdown", (e) => {
  if (e.target.closest("button, a")) return;
  audio.unlock();
  startRun();
});

document.getElementById("quick-mute").addEventListener("click", () => {
  setMuted(!store.settings.muted);
});

// ─────────────────────────────── touch ───────────────────────────────

function setupTouch() {
  const layer = document.getElementById("touch-layer");
  layer.classList.remove("hidden");
  input.attachTouch(
    document.getElementById("stick"),
    document.getElementById("stick-knob"),
    document.getElementById("touch-dash"),
    document.getElementById("touch-surface"));
  // The touch layer must not swallow menu taps.
  const sync = () => {
    layer.style.pointerEvents = isPlaying() ? "auto" : "none";
    layer.style.opacity = isPlaying() ? "1" : "0";
  };
  setInterval(sync, 120);
}

// ─────────────────────────────── window events ───────────────────────────────

let resizeTimer = null;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    renderer.resize();
    ribbon.resize();
  }, 120);
});

document.addEventListener("visibilitychange", () => {
  if (document.hidden && isPlaying()) pause();
});

// Opt-in debug handle: `?debug` exposes the live game object and the state
// transitions, which is how the screenshots and browser tests pose specific
// screens without shipping test hooks in the normal path.
if (new URLSearchParams(location.search).has("debug")) {
  window.__flow = {
    g, ui, hud, audio, renderer, store, input, ribbon, tracker,
    startRun, toTitle, pause, resume, endRun, enterShop, enterRoom,
    startNextRoomOrShop, step,
  };
}

// Prevent the page itself from scrolling or zooming under the game.
document.addEventListener("gesturestart", (e) => e.preventDefault());
window.addEventListener("contextmenu", (e) => {
  if (e.target === canvas) e.preventDefault();
});
