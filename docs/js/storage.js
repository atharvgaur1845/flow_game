/* Browser-local persistence. There is no server anywhere in this project —
 * localStorage is the whole backend.
 *
 * Four namespaces under a single versioned key so a schema bump can migrate
 * or discard cleanly:
 *
 *   settings   volume, quality, accessibility, keybinds
 *   records    bests, lifetime counters
 *   profile    lifetime seconds spent in each archetype  -> "you play like…"
 *   runs       last 20 run summaries -> the menu sparkline
 *   seen       one-shot flags: onboarding hints, unlocked milestones
 */

const KEY = "flow.web.v1";

const DEFAULTS = {
  settings: {
    musicVolume: 0.55,
    sfxVolume: 0.8,
    muted: false,
    quality: "high",
    autoScale: true,
    reducedMotion: false,
    colorblind: false,
    showRibbon: true,
  },
  records: {
    bestScore: 0,
    deepestRoom: 0,
    totalKills: 0,
    totalRuns: 0,
    totalSeconds: 0,
    bossesKilled: 0,
    longestFlowStreak: 0,
  },
  // Lifetime seconds accumulated in each archetype. The game has been
  // measuring the player across every session they have ever played.
  profile: [0, 0, 0, 0, 0],
  runs: [],
  seen: {},
};

function deepDefault(loaded) {
  const out = structuredClone(DEFAULTS);
  if (!loaded || typeof loaded !== "object") return out;
  for (const k of Object.keys(DEFAULTS)) {
    if (!(k in loaded)) continue;
    if (Array.isArray(DEFAULTS[k])) {
      if (Array.isArray(loaded[k])) out[k] = loaded[k];
    } else if (typeof DEFAULTS[k] === "object") {
      Object.assign(out[k], loaded[k]);
    } else {
      out[k] = loaded[k];
    }
  }
  return out;
}

class Store {
  constructor() {
    this.available = true;
    let raw = null;
    try {
      raw = localStorage.getItem(KEY);
    } catch {
      // Private-browsing or a blocked third-party context. Everything still
      // works, it just will not survive a reload.
      this.available = false;
    }
    let parsed = null;
    try {
      parsed = raw ? JSON.parse(raw) : null;
    } catch {
      parsed = null;
    }
    this.data = deepDefault(parsed);
  }

  save() {
    if (!this.available) return;
    try {
      localStorage.setItem(KEY, JSON.stringify(this.data));
    } catch {
      this.available = false;
    }
  }

  get settings() { return this.data.settings; }
  get records() { return this.data.records; }
  get profile() { return this.data.profile; }
  get runs() { return this.data.runs; }
  get seen() { return this.data.seen; }

  setSetting(key, value) {
    this.data.settings[key] = value;
    this.save();
  }

  /** One-shot flags: returns true the first time only. */
  markSeen(flag) {
    if (this.data.seen[flag]) return false;
    this.data.seen[flag] = Date.now();
    this.save();
    return true;
  }

  hasSeen(flag) {
    return !!this.data.seen[flag];
  }

  /** Fold a finished run into records, profile and history. */
  commitRun(run, piSeconds) {
    const r = this.data.records;
    r.bestScore = Math.max(r.bestScore, Math.round(run.score));
    r.deepestRoom = Math.max(r.deepestRoom, run.room);
    r.totalKills += run.totalKills;
    r.totalRuns += 1;
    r.totalSeconds += run.timeSurvived;
    r.bossesKilled += run.bossesKilled;
    r.longestFlowStreak = Math.max(r.longestFlowStreak, run.longestFlowStreak);

    for (let k = 0; k < 5; k++) this.data.profile[k] += piSeconds[k];

    this.data.runs.push({
      t: Date.now(),
      score: Math.round(run.score),
      room: run.room,
      kills: run.totalKills,
      secs: Math.round(run.timeSurvived),
      dom: run.dominantArchetype(),
      pi: piSeconds.map((v) => +(v / Math.max(run.timeSurvived, 0.001)).toFixed(3)),
    });
    if (this.data.runs.length > 20) this.data.runs = this.data.runs.slice(-20);
    this.save();
  }

  /** Lifetime archetype distribution, normalised. Null until there is
   *  enough play history for the claim to mean anything. */
  profileMix() {
    const total = this.data.profile.reduce((a, b) => a + b, 0);
    if (total < 60) return null; // under a minute of lifetime play
    return this.data.profile.map((v) => v / total);
  }

  medianScore() {
    if (this.data.runs.length < 3) return null;
    const s = this.data.runs.map((r) => r.score).sort((a, b) => a - b);
    const m = s.length >> 1;
    return s.length % 2 ? s[m] : Math.round((s[m - 1] + s[m]) / 2);
  }

  reset() {
    this.data = structuredClone(DEFAULTS);
    this.save();
  }
}

export const store = new Store();
