/* In-run HUD.
 *
 * Bars are driven with transform:scaleX so they animate on the compositor
 * and cost nothing per frame; text is diffed against the last written value
 * so we never touch the DOM without a reason. */

import { STATE_NAMES } from "./config.js";
import { PICKUP_BUFF_DURATION } from "./entities.js";

const C_LABELS = ["ABER", "GRAIN", "WARP", "BLOOM"];
const C_SCALES = [0.05, 0.5, 0.05, 1.5];

const BUFF_LABEL = {
  heal: "HEAL",
  dash_boost: "BIG DASH",
  speed_boost: "HASTE",
  shield: "SHIELD",
  max_hp: "+MAX HP",
};

const BUFF_COLOR = {
  heal: "#5ae682",
  dash_boost: "#5ae6ff",
  speed_boost: "#ffaa5a",
  shield: "#82a0ff",
  max_hp: "#ffe664",
};

const $ = (id) => document.getElementById(id);

export class HUD {
  constructor() {
    this.el = {
      hud: $("hud"),
      hpFill: $("hp-fill"),
      hpText: $("hp-text"),
      dashFill: $("dash-fill"),
      dashText: $("dash-text"),
      score: $("score"),
      mult: $("mult"),
      room: $("room"),
      kills: $("kills"),
      goal: $("goal"),
      buffs: $("buffs"),
      piBars: $("pi-bars"),
      cBars: $("c-bars"),
      bossWrap: $("boss-wrap"),
      bossLabel: $("boss-label"),
      bossFill: $("boss-fill"),
      banner: $("pickup-banner"),
      hint: $("hint"),
      toasts: $("toasts"),
      ribbonWrap: $("ribbon-wrap"),
      legend: $("ribbon-legend"),
    };

    this._piFills = [];
    this._piVals = [];
    this._piRows = [];
    this._cFills = [];
    this._last = {};
    this._textAccum = 0;
    this._buffSig = "";
    this._dominant = -1;

    this._buildBars();
  }

  _buildBars() {
    const pi = this.el.piBars;
    pi.innerHTML = "";
    STATE_NAMES.forEach((name, i) => {
      const row = document.createElement("div");
      row.className = "pi-row";
      row.style.color = `var(--${name.toLowerCase()})`;
      row.innerHTML =
        `<span class="name">${name}</span>` +
        `<div class="bar bar-tiny"><i class="fill"></i></div>` +
        `<span class="val">0.00</span>`;
      pi.appendChild(row);
      const fill = row.querySelector(".fill");
      fill.style.background = "currentColor";
      this._piFills.push(fill);
      this._piVals.push(row.querySelector(".val"));
      this._piRows.push(row);
    });

    const c = this.el.cBars;
    c.innerHTML = "";
    C_LABELS.forEach((lbl) => {
      const row = document.createElement("div");
      row.className = "c-row";
      row.innerHTML =
        `<span class="name">${lbl}</span>` +
        `<div class="bar bar-tiny"><i class="fill" style="background:#b4b4ff"></i></div>`;
      c.appendChild(row);
      this._cFills.push(row.querySelector(".fill"));
    });

    this.el.legend.innerHTML = STATE_NAMES.map(
      (n) => `<span><i style="background:var(--${n.toLowerCase()})"></i>${n}</span>`
    ).join("");
  }

  show(on) {
    this.el.hud.classList.toggle("hidden", !on);
    this.el.hud.setAttribute("aria-hidden", String(!on));
  }

  /** Toggling the ribbon also reclaims its band for the arena, so callers
   *  must re-measure the canvas afterwards. */
  setRibbonVisible(on) {
    this.el.ribbonWrap.classList.toggle("hidden", !on);
    document.body.classList.toggle("no-ribbon", !on);
  }

  /** @param g the live game object */
  update(g, dtReal) {
    const { player, run, pi, c } = g;

    scaleX(this.el.hpFill, player.hp / Math.max(player.maxHp, 1));
    const dashReady = player.dashCd <= 0;
    scaleX(this.el.dashFill, dashReady ? 1 : 1 - player.dashCd / Math.max(player.dashCdMax, 1e-3));

    let dominant = 0;
    for (let k = 1; k < 5; k++) if (pi[k] > pi[dominant]) dominant = k;
    for (let k = 0; k < 5; k++) scaleX(this._piFills[k], pi[k]);
    for (let i = 0; i < 4; i++) scaleX(this._cFills[i], Math.max(0, c[i]) / C_SCALES[i]);

    if (dominant !== this._dominant) {
      if (this._dominant >= 0) this._piRows[this._dominant].classList.remove("dominant");
      this._piRows[dominant].classList.add("dominant");
      this._dominant = dominant;
    }

    if (g.boss) {
      this.el.bossWrap.classList.remove("hidden");
      scaleX(this.el.bossFill, Math.max(0, g.boss.hp / g.boss.maxHp));
      this._text("bossLabel", `BOSS — ${g.boss.phaseLabel}`);
    } else {
      this.el.bossWrap.classList.add("hidden");
    }

    // Text is refreshed at 12 Hz: any faster is illegible anyway and it keeps
    // layout work off the frame budget.
    this._textAccum += dtReal;
    if (this._textAccum < 0.08) return;
    this._textAccum = 0;

    this._text("hpText", `${Math.max(0, Math.round(player.hp))} / ${Math.round(player.maxHp)}`);
    this._text("dashText", dashReady ? "READY" : player.dashCd.toFixed(1) + "s");
    this.el.dashText.classList.toggle("ready", dashReady);
    this._text("score", Math.round(run.score).toLocaleString());
    this._text("mult", "×" + run.scoreMultiplier(pi).toFixed(2));
    this._text("room", "ROOM " + run.room);
    this._text("kills", run.totalKills + (run.totalKills === 1 ? " KILL" : " KILLS"));

    for (let k = 0; k < 5; k++) {
      const v = pi[k].toFixed(2);
      if (this._piVals[k].textContent !== v) this._piVals[k].textContent = v;
    }

    const t = Math.max(0, run.roomTimeRemaining);
    const urgent = t < 8 && !run.isBossRoom;
    const goal = run.isBossRoom
      ? `Defeat the BOSS · <span class="${urgent ? "urgent" : ""}">${t.toFixed(1)}s</span>`
      : `Kill ${Math.max(0, run.roomKillsRequired - run.killsInRoom)} more · ` +
        `<span class="${urgent ? "urgent" : ""}">${t.toFixed(1)}s remaining</span>`;
    if (this.el.goal.innerHTML !== goal) this.el.goal.innerHTML = goal;

    this._renderBuffs(player, g.simTime);
  }

  _renderBuffs(player, nowT) {
    const sig = [...player.buffs.keys()].join(",");
    if (sig !== this._buffSig) {
      this._buffSig = sig;
      this.el.buffs.innerHTML = "";
      for (const kind of player.buffs.keys()) {
        const row = document.createElement("div");
        row.className = "buff";
        row.dataset.kind = kind;
        row.style.color = BUFF_COLOR[kind] || "#c8c8c8";
        row.innerHTML =
          `<span class="buff-name">${BUFF_LABEL[kind] || kind}</span>` +
          `<div class="bar bar-tiny"><i class="fill" style="background:currentColor"></i></div>` +
          `<span class="buff-time"></span>`;
        this.el.buffs.appendChild(row);
      }
    }
    for (const row of this.el.buffs.children) {
      const kind = row.dataset.kind;
      const exp = player.buffs.get(kind);
      if (exp === undefined) continue;
      const remain = Math.max(0, exp - nowT);
      const full = PICKUP_BUFF_DURATION[kind] || 8;
      scaleX(row.querySelector(".fill"), remain / full);
      row.querySelector(".buff-time").textContent = remain.toFixed(1) + "s";
    }
  }

  _text(key, value) {
    if (this._last[key] === value) return;
    this._last[key] = value;
    this.el[key].textContent = value;
  }

  banner(label) {
    const el = this.el.banner;
    el.textContent = "+ " + label;
    el.classList.remove("show");
    void el.offsetWidth; // restart the animation
    el.classList.add("show");
  }

  /** Contextual onboarding prompt. Pass null to clear. */
  hint(html) {
    const el = this.el.hint;
    if (!html) {
      el.classList.add("hidden");
      return;
    }
    if (el.innerHTML !== html) el.innerHTML = html;
    el.classList.remove("hidden");
  }

  toast(kicker, name, desc, ttl = 4200) {
    const el = document.createElement("div");
    el.className = "toast";
    el.innerHTML =
      `<div class="t-kicker">${kicker}</div>` +
      `<div class="t-name">${name}</div>` +
      (desc ? `<div class="t-desc">${desc}</div>` : "");
    this.el.toasts.appendChild(el);
    // Several milestones can land on the same kill; never let the stack grow
    // tall enough to cover the arena.
    while (this.el.toasts.children.length > 3) this.el.toasts.firstChild.remove();
    setTimeout(() => {
      el.classList.add("out");
      setTimeout(() => el.remove(), 420);
    }, ttl);
  }
}

function scaleX(el, f) {
  const v = f < 0 ? 0 : f > 1 ? 1 : f;
  el.style.transform = `scaleX(${v.toFixed(4)})`;
}
