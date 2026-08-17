/* Milestones — one-shot achievements surfaced as neon toasts mid-run and
 * listed on the menu. Unlock state lives in localStorage under `seen`.
 *
 * Each has a `check(ctx)` evaluated a few times a second during play, where
 * ctx = { run, player, pi, records, state, simTime }. Keep them cheap. */

export const MILESTONES = [
  {
    id: "first_blood", name: "First Blood", desc: "Kill your first enemy.",
    check: (c) => c.run.totalKills >= 1,
  },
  {
    id: "first_room", name: "Threshold", desc: "Clear your first room.",
    check: (c) => c.run.roomsCleared >= 1,
  },
  {
    id: "first_boss", name: "Giant Slayer", desc: "Defeat a boss.",
    check: (c) => c.run.bossesKilled >= 1,
  },
  {
    id: "flow_10", name: "In the Zone", desc: "Hold Flow above 0.45 for 10 seconds.",
    check: (c) => c.run.longestFlowStreak >= 10,
  },
  {
    id: "flow_30", name: "Dissolved", desc: "Hold Flow above 0.45 for 30 seconds.",
    check: (c) => c.run.longestFlowStreak >= 30,
  },
  {
    id: "flow_peak", name: "Peak State", desc: "Push Flow past 0.80.",
    check: (c) => c.pi[3] > 0.8,
  },
  {
    id: "overload_peak", name: "Cognitive Overload", desc: "Push Overload past 0.70.",
    check: (c) => c.pi[2] > 0.7,
  },
  {
    id: "tactician", name: "Tactician", desc: "Push Tactical past 0.70.",
    check: (c) => c.pi[1] > 0.7,
  },
  {
    id: "room_10", name: "Deep Run", desc: "Reach room 10.",
    check: (c) => c.run.room >= 10,
  },
  {
    id: "room_15", name: "Further Still", desc: "Reach room 15.",
    check: (c) => c.run.room >= 15,
  },
  {
    id: "score_10k", name: "Five Figures", desc: "Score 10,000 in a single run.",
    check: (c) => c.run.score >= 10000,
  },
  {
    id: "kills_100", name: "Centurion", desc: "100 kills in a single run.",
    check: (c) => c.run.totalKills >= 100,
  },
  {
    id: "untouched", name: "Untouched", desc: "Clear a room at full HP.",
    check: (c) => c.run.roomsCleared >= 1 && c.player.hp >= c.player.maxHp,
  },
  {
    id: "veteran", name: "Veteran", desc: "Play 25 runs.",
    check: (c) => c.records.totalRuns >= 25,
  },
];

export class MilestoneTracker {
  constructor(store, onUnlock) {
    this.store = store;
    this.onUnlock = onUnlock;
    this._accum = 0;
  }

  /** Call every sim tick; throttles itself to ~4 Hz. */
  update(ctx, dt) {
    this._accum += dt;
    if (this._accum < 0.25) return;
    this._accum = 0;
    for (const m of MILESTONES) {
      if (this.store.hasSeen("ms." + m.id)) continue;
      let ok = false;
      try {
        ok = m.check(ctx);
      } catch {
        ok = false;
      }
      if (ok && this.store.markSeen("ms." + m.id)) this.onUnlock(m);
    }
  }

  unlockedCount() {
    return MILESTONES.filter((m) => this.store.hasSeen("ms." + m.id)).length;
  }
}
