/* Screens: title, shop, run report, pause, settings, how-to-play,
 * milestones. Everything reads from and writes to the browser store — there
 * is no server to sync with. */

import { STATE_NAMES } from "./config.js";
import { UPGRADES } from "./progression.js";
import { MILESTONES } from "./milestones.js";
import { RIBBON_PALETTE, RIBBON_PALETTE_CB, renderRunStrip, condense } from "./ribbon.js";
import { ACTIONS, ACTION_LABELS, IS_TOUCH } from "./input.js";
import { buildRunCard, copyText } from "./runcard.js";

const $ = (id) => document.getElementById(id);
const cssVar = (name) => `var(--${name})`;
const archVar = (i) => cssVar(STATE_NAMES[i].toLowerCase());

const SCREENS = [
  "screen-title", "screen-shop", "screen-report", "screen-pause",
  "screen-settings", "screen-how", "screen-milestones", "screen-unsupported",
];

export class UI {
  /** @param ctx { store, audio, renderer, input, ribbon, tracker, on } */
  constructor(ctx) {
    this.ctx = ctx;
    this.current = null;
    this._prevScreen = null;
    this._wireActions();
    this._wireSettings();
    this.renderHow();
  }

  // --- screen stack --------------------------------------------------------

  show(id) {
    for (const s of SCREENS) $(s).classList.toggle("hidden", s !== id);
    this.current = id;
  }

  hideAll() {
    for (const s of SCREENS) $(s).classList.add("hidden");
    this.current = null;
  }

  // --- global button routing ----------------------------------------------

  _wireActions() {
    document.addEventListener("click", (e) => {
      const btn = e.target.closest("[data-action]");
      if (!btn) return;
      const a = btn.dataset.action;
      this.ctx.audio.uiConfirm();
      const on = this.ctx.on;
      switch (a) {
        case "play": case "again": on.startRun(); break;
        case "to-title": on.toTitle(); break;
        case "resume": on.resume(); break;
        case "abandon": on.abandon(); break;
        case "how": this.show("screen-how"); break;
        case "close-how": this.show("screen-title"); break;
        case "milestones": this.renderMilestones(); this.show("screen-milestones"); break;
        case "close-milestones": this.show("screen-title"); break;
        case "settings": this.openSettings(); break;
        case "close-settings": this.closeSettings(); break;
        case "reset-binds": this.ctx.input.resetBinds(); this.renderBinds(); break;
        case "copy-card": this._copyCard(btn); break;
        case "wipe": this._wipe(); break;
      }
    });

    document.querySelectorAll(".btn").forEach((b) =>
      b.addEventListener("mouseenter", () => this.ctx.audio.uiMove()));
  }

  // --- title ---------------------------------------------------------------

  refreshTitle() {
    const { store, tracker } = this.ctx;
    const rec = store.records;

    const hasHistory = rec.totalRuns > 0;
    $("records-row").classList.toggle("hidden", !hasHistory);
    $("rec-best").textContent = rec.bestScore.toLocaleString();
    $("rec-room").textContent = rec.deepestRoom;
    $("rec-runs").textContent = rec.totalRuns;
    $("rec-ms").textContent = `${tracker.unlockedCount()}/${MILESTONES.length}`;

    // "You play like…" — the lifetime archetype mix, the payoff for the game
    // having measured every session the player has ever played.
    const mix = store.profileMix();
    const card = $("profile-card");
    if (mix) {
      card.classList.remove("hidden");
      let dom = 0;
      for (let k = 1; k < 5; k++) if (mix[k] > mix[dom]) dom = k;
      const archEl = $("pc-arch");
      archEl.textContent = STATE_NAMES[dom];
      archEl.style.color = archVar(dom);
      $("pc-mix").innerHTML = mix
        .map((v, k) => `<i style="width:${(v * 100).toFixed(2)}%;background:${archVar(k)}" title="${STATE_NAMES[k]} ${Math.round(v * 100)}%"></i>`)
        .join("");
    } else {
      card.classList.add("hidden");
    }

    const spark = $("sparkline-wrap");
    if (store.runs.length >= 3) {
      spark.classList.remove("hidden");
      this._drawSparkline();
    } else {
      spark.classList.add("hidden");
    }

    $("press-any").textContent = IS_TOUCH ? "tap to begin" : "press any key";
  }

  _drawSparkline() {
    const cv = $("sparkline");
    const runs = this.ctx.store.runs;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = cv.clientWidth || 420;
    const h = 46;
    cv.width = Math.round(w * dpr);
    cv.height = Math.round(h * dpr);
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const max = Math.max(...runs.map((r) => r.score), 1);
    const bw = w / runs.length;
    const pal = this.ctx.store.settings.colorblind ? RIBBON_PALETTE_CB : RIBBON_PALETTE;
    runs.forEach((r, i) => {
      const bh = Math.max(2, (r.score / max) * (h - 8));
      const c = pal[r.dom] || pal[4];
      ctx.fillStyle = `rgba(${c[0]},${c[1]},${c[2]},${i === runs.length - 1 ? 1 : 0.6})`;
      ctx.fillRect(i * bw + bw * 0.18, h - bh, bw * 0.64, bh);
    });
  }

  // --- shop ----------------------------------------------------------------

  renderShop(choices, run, onPick) {
    const wrap = $("shop-cards");
    wrap.innerHTML = "";
    choices.forEach((idx, i) => {
      const u = UPGRADES[idx];
      const card = document.createElement("button");
      card.className = "shop-card";
      card.style.setProperty("--accent", archVar(u.favors));
      card.innerHTML =
        `<span class="idx">${i + 1}</span>` +
        `<div class="icon">${u.icon}</div>` +
        `<div class="name">${u.name}</div>` +
        `<div class="desc">${u.desc}</div>` +
        `<div class="favors">FAVOURS ${STATE_NAMES[u.favors]}</div>`;
      card.addEventListener("click", () => onPick(i));
      card.addEventListener("mouseenter", () => this.ctx.audio.uiMove());
      wrap.appendChild(card);
    });
    $("shop-meta").textContent =
      `ROOM ${run.room}  ·  SCORE ${Math.round(run.score).toLocaleString()}  ·  NEXT: ROOM ${run.room + 1}`;
    this.show("screen-shop");
  }

  // --- run report ----------------------------------------------------------

  renderReport(run, samples) {
    const { store } = this.ctx;
    const score = Math.round(run.score);
    const median = store.medianScore();
    const rec = store.records;

    $("report-title").textContent = score >= rec.bestScore && score > 0 ? "NEW BEST" : "RUN OVER";

    const dom = run.dominantArchetype();
    const total = run.piSeconds.reduce((a, b) => a + b, 0) || 1;
    const pct = Math.round((run.piSeconds[dom] / total) * 100);

    // Name the cause outright. A run that just ends, with no stated reason,
    // reads as the game cheating.
    let cause;
    if (run.deathCause === "abandoned") {
      cause = `<span class="cause">RUN ABANDONED</span>`;
    } else {
      cause =
        `<span class="cause bad">KILLED</span> — you ran out of HP in room ${run.room}.<br>` +
        `<span class="cause-tip">Dash through enemies rather than into them — the trail keeps killing for 1.6s after the dash ends, and you are invulnerable while it runs.</span>`;
    }

    $("report-verdict").innerHTML =
      `<div class="cause-line">${cause}</div>` +
      `You spent most of this run in <b style="color:${archVar(dom)}">${STATE_NAMES[dom]}</b> ` +
      `(${pct}% of ${Math.round(run.timeSurvived)}s).`;

    $("r-score").textContent = score.toLocaleString();
    $("r-room").textContent = run.room;
    $("r-kills").textContent = run.totalKills;
    $("r-time").textContent = Math.round(run.timeSurvived) + "s";
    $("r-flow").textContent = run.maxFlow.toFixed(2);
    $("r-streak").textContent = run.longestFlowStreak.toFixed(0) + "s";

    const cmp = $("r-score-cmp");
    if (median !== null && median > 0) {
      const d = Math.round(((score - median) / median) * 100);
      cmp.textContent = `${d >= 0 ? "+" : ""}${d}% vs median`;
      cmp.classList.toggle("down", d < 0);
    } else {
      cmp.textContent = "";
    }

    // The run's whole pi history as one image: a fingerprint of how it felt.
    const strip = $("report-strip");
    strip.innerHTML = "";
    const pal = store.settings.colorblind ? RIBBON_PALETTE_CB : RIBBON_PALETTE;
    if (samples.length > 1) {
      strip.appendChild(renderRunStrip(condense(samples, 260), strip.clientWidth || 800, 74, pal));
    }
    $("report-legend").innerHTML = STATE_NAMES.map(
      (n, k) => `<span><i style="background:${archVar(k)}"></i>${n}</span>`).join("");

    this._card = buildRunCard(run, samples, location.href.split("#")[0]);
    this.show("screen-report");
  }

  async _copyCard(btn) {
    const ok = await copyText(this._card || "");
    const old = btn.textContent;
    btn.textContent = ok ? "COPIED ✓" : "COPY FAILED";
    setTimeout(() => { btn.textContent = old; }, 1600);
  }

  // --- settings ------------------------------------------------------------

  openSettings() {
    this._prevScreen = this.current;
    this.syncSettings();
    this.renderBinds();
    this.show("screen-settings");
  }

  closeSettings() {
    this.show(this._prevScreen === "screen-pause" ? "screen-pause" : "screen-title");
    if (this._prevScreen === "screen-pause") this._prevScreen = null;
  }

  _wireSettings() {
    const { store, audio, renderer, ribbon } = this.ctx;
    const s = store.settings;

    const bind = (id, event, handler) => $(id).addEventListener(event, handler);

    bind("set-music", "input", (e) => {
      const v = +e.target.value;
      store.setSetting("musicVolume", v);
      audio.setMusicVolume(v);
      $("out-music").textContent = Math.round(v * 100) + "%";
    });
    bind("set-sfx", "input", (e) => {
      const v = +e.target.value;
      store.setSetting("sfxVolume", v);
      audio.setSfxVolume(v);
      $("out-sfx").textContent = Math.round(v * 100) + "%";
    });
    bind("set-sfx", "change", () => audio.pickup("heal"));
    bind("set-mute", "change", (e) => this.ctx.on.setMuted(e.target.checked));

    bind("set-quality", "change", (e) => {
      store.setSetting("quality", e.target.value);
      renderer.setQuality(e.target.value);
    });
    bind("set-autoscale", "change", (e) => {
      store.setSetting("autoScale", e.target.checked);
      renderer.setAutoScale(e.target.checked);
    });
    bind("set-motion", "change", (e) => {
      store.setSetting("reducedMotion", e.target.checked);
    });
    bind("set-cb", "change", (e) => {
      store.setSetting("colorblind", e.target.checked);
      document.body.classList.toggle("cb", e.target.checked);
      ribbon.setPalette(e.target.checked ? RIBBON_PALETTE_CB : RIBBON_PALETTE);
    });
    bind("set-ribbon", "change", (e) => {
      store.setSetting("showRibbon", e.target.checked);
      this.ctx.on.setRibbonVisible(e.target.checked);
    });

    document.body.classList.toggle("cb", s.colorblind);
    $("storage-note").textContent = store.available
      ? "Everything above — plus your records, profile and milestones — is stored only in this browser. Nothing is uploaded."
      : "This browser is blocking local storage, so settings and records will not survive a reload.";
  }

  syncSettings() {
    const s = this.ctx.store.settings;
    $("set-music").value = s.musicVolume;
    $("set-sfx").value = s.sfxVolume;
    $("out-music").textContent = Math.round(s.musicVolume * 100) + "%";
    $("out-sfx").textContent = Math.round(s.sfxVolume * 100) + "%";
    $("set-mute").checked = s.muted;
    $("set-quality").value = s.quality;
    $("set-autoscale").checked = s.autoScale;
    $("set-motion").checked = s.reducedMotion;
    $("set-cb").checked = s.colorblind;
    $("set-ribbon").checked = s.showRibbon;
  }

  /** Live perf readout, refreshed only while the settings panel is open. */
  updatePerf(frameMs, scale) {
    if (this.current !== "screen-settings") return;
    $("out-fps").textContent = `${frameMs.toFixed(1)} ms · ${Math.round(1000 / Math.max(frameMs, 0.1))} fps`;
    $("out-scale").textContent = Math.round(scale * 100) + "%";
  }

  renderBinds() {
    const { input } = this.ctx;
    const wrap = $("binds");
    wrap.innerHTML = "";
    for (const a of ACTIONS) {
      const row = document.createElement("div");
      row.className = "bind-row";
      row.innerHTML = `<span>${ACTION_LABELS[a]}</span>`;
      const btn = document.createElement("button");
      btn.textContent = input.bindLabel(a);
      btn.addEventListener("click", () => {
        input.capturing = a;
        btn.textContent = "PRESS A KEY";
        btn.classList.add("capturing");
        input.onCapture = () => {
          btn.classList.remove("capturing");
          this.renderBinds();
        };
      });
      row.appendChild(btn);
      wrap.appendChild(row);
    }
    if (IS_TOUCH) {
      const note = document.createElement("p");
      note.className = "note";
      note.textContent = "On touch devices: drag anywhere on the left two-thirds to move, tap DASH to dash.";
      wrap.appendChild(note);
    }
  }

  _wipe() {
    if (!confirm("Erase all local data — records, profile, milestones and settings? This cannot be undone.")) return;
    this.ctx.store.reset();
    location.reload();
  }

  // --- static screens ------------------------------------------------------

  renderMilestones() {
    const { store } = this.ctx;
    $("milestone-list").innerHTML = MILESTONES.map((m) => {
      const got = store.hasSeen("ms." + m.id);
      return `<div class="ms ${got ? "" : "locked"}">
        <span class="ms-mark">${got ? "◆" : "◇"}</span>
        <span><span class="ms-name">${got ? m.name : "???"}</span>
        <span class="ms-desc"> — ${m.desc}</span></span>
      </div>`;
    }).join("");
  }

  renderHow() {
    const blurbs = [
      "fast movement, high action rate, close threats",
      "precise positioning, measured action rate",
      "erratic direction changes, enemy swarms",
      "smooth velocity, consistent engagement, rhythm",
      "low speed, low action rate, long idle periods",
    ];
    $("how-archetypes").innerHTML = STATE_NAMES.map(
      (n, k) => `<span style="color:${archVar(k)}"><i style="background:${archVar(k)}"></i>${n}</span>
                 <span style="flex:1 1 100%;margin:-10px 0 4px 14px;color:var(--ink-faint);letter-spacing:0.04em">${blurbs[k]}</span>`
    ).join("");
  }
}
