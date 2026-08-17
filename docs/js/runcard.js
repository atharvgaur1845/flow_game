/* The shareable run card.
 *
 * A GitHub Pages link spreads when people have something to paste. This
 * builds a Wordle-style block: the headline numbers plus the run's pi
 * fingerprint as coloured squares, so the shape of how somebody played
 * survives being pasted into a chat window with no images. */

import { STATE_NAMES } from "./config.js";

// Emoji whose colours track the archetype palette closely enough to read.
const SQUARES = ["🟥", "🟦", "🟪", "🟩", "⬜"];

/** Compress a run's pi history into `n` dominant-archetype squares. */
export function fingerprint(samples, n = 12) {
  if (!samples.length) return "";
  const out = [];
  const bucket = samples.length / n;
  for (let i = 0; i < n; i++) {
    const lo = Math.floor(i * bucket);
    const hi = Math.max(lo + 1, Math.floor((i + 1) * bucket));
    const acc = new Float64Array(5);
    for (let j = lo; j < hi; j++) for (let k = 0; k < 5; k++) acc[k] += samples[j][k];
    let best = 0;
    for (let k = 1; k < 5; k++) if (acc[k] > acc[best]) best = k;
    out.push(SQUARES[best]);
  }
  return out.join("");
}

export function buildRunCard(run, samples, url) {
  const dom = run.dominantArchetype();
  const total = run.piSeconds.reduce((a, b) => a + b, 0) || 1;
  const pct = Math.round((run.piSeconds[dom] / total) * 100);
  const lines = [
    `FLOW — Room ${run.room} · ${Math.round(run.score).toLocaleString()}`,
    fingerprint(samples, 12),
    `Dominant: ${STATE_NAMES[dom]} (${pct}%)  ·  ${run.totalKills} kills  ·  ${Math.round(run.timeSurvived)}s`,
  ];
  if (url) lines.push(url);
  return lines.join("\n");
}

/** Clipboard write with a fallback for insecure contexts / older Safari. */
export async function copyText(text) {
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch {
    /* fall through to the legacy path */
  }
  try {
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.setAttribute("readonly", "");
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    const ok = document.execCommand("copy");
    document.body.removeChild(ta);
    return ok;
  } catch {
    return false;
  }
}
