/* The pi ribbon — the signature readout of this build.
 *
 * Five bars tell you a number. A scrolling stacked-area river tells you a
 * story: you watch yourself slide out of Tactical blue into Arousal red as a
 * swarm closes, and you can read the whole shape of a run at a glance
 * afterwards. Same drawing routine serves the live HUD strip and the frozen
 * full-run strip on the report card. */

export const RIBBON_PALETTE = [
  [255, 120, 120], // arousal
  [120, 180, 255], // tactical
  [200, 120, 255], // overload
  [120, 255, 200], // flow
  [170, 170, 170], // apathy
];

// Okabe-Ito derived: distinguishable under all common colour-vision types.
export const RIBBON_PALETTE_CB = [
  [230, 159, 0], // arousal  orange
  [86, 180, 233], // tactical sky
  [204, 121, 167], // overload purple
  [0, 158, 115], // flow     green
  [187, 187, 187], // apathy  grey
];

const rgba = (c, a) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;

/**
 * Draw a stacked-area pi ribbon.
 * @param {CanvasRenderingContext2D} ctx
 * @param {Array<Float64Array|number[]>} samples oldest -> newest
 */
export function drawRibbon(ctx, samples, w, h, opts = {}) {
  const palette = opts.palette || RIBBON_PALETTE;
  const alpha = opts.alpha ?? 0.92;
  const fade = opts.fadeIn ?? false;

  ctx.clearRect(0, 0, w, h);
  const n = samples.length;
  if (n < 2) return;

  // Stack from the bottom up so Flow (index 3) sits mid-band and stays the
  // easiest colour to track peripherally.
  const cum = new Float64Array(n);
  for (let k = 0; k < 5; k++) {
    ctx.beginPath();
    // Top edge, left -> right.
    for (let i = 0; i < n; i++) {
      const x = (i / (n - 1)) * w;
      const yTop = h - (cum[i] + samples[i][k]) * h;
      if (i === 0) ctx.moveTo(x, yTop);
      else ctx.lineTo(x, yTop);
    }
    // Bottom edge, right -> left.
    for (let i = n - 1; i >= 0; i--) {
      const x = (i / (n - 1)) * w;
      ctx.lineTo(x, h - cum[i] * h);
    }
    ctx.closePath();

    const c = palette[k];
    if (fade) {
      const grad = ctx.createLinearGradient(0, 0, w, 0);
      grad.addColorStop(0, rgba(c, 0));
      grad.addColorStop(0.18, rgba(c, alpha));
      grad.addColorStop(1, rgba(c, alpha));
      ctx.fillStyle = grad;
    } else {
      ctx.fillStyle = rgba(c, alpha);
    }
    ctx.fill();

    for (let i = 0; i < n; i++) cum[i] += samples[i][k];
  }

  // A bright leading edge so the newest instant reads as "now".
  if (opts.leadingEdge) {
    let y = h;
    for (let k = 0; k < 5; k++) {
      const seg = samples[n - 1][k] * h;
      ctx.fillStyle = rgba(palette[k], 1);
      ctx.fillRect(w - 3, y - seg, 3, seg);
      y -= seg;
    }
  }
}

/** Live scrolling ribbon bound to a canvas element. */
export class Ribbon {
  constructor(canvas, { windowSeconds = 20, sampleHz = 12 } = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.sampleInterval = 1 / sampleHz;
    this.capacity = Math.ceil(windowSeconds * sampleHz);
    this.samples = [];
    this._accum = 0;
    this.palette = RIBBON_PALETTE;
    this._dirty = false;
    this.resize();
  }

  setPalette(p) {
    this.palette = p;
    this._dirty = true;
  }

  resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = this.canvas.clientWidth || 800;
    const h = this.canvas.clientHeight || 48;
    this.canvas.width = Math.round(w * dpr);
    this.canvas.height = Math.round(h * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.w = w;
    this.h = h;
    this._dirty = true;
  }

  push(pi, dt) {
    this._accum += dt;
    while (this._accum >= this.sampleInterval) {
      this._accum -= this.sampleInterval;
      this.samples.push(Float64Array.from(pi));
      if (this.samples.length > this.capacity) this.samples.shift();
      this._dirty = true;
    }
  }

  clear() {
    this.samples.length = 0;
    this._accum = 0;
    this.ctx.clearRect(0, 0, this.w, this.h);
  }

  draw() {
    if (!this._dirty) return;
    this._dirty = false;
    drawRibbon(this.ctx, this.samples, this.w, this.h, {
      palette: this.palette,
      fadeIn: true,
      leadingEdge: true,
      alpha: 0.9,
    });
  }
}

/** Render a completed run's pi history into an offscreen canvas — the
 *  "fingerprint" strip on the report card. */
export function renderRunStrip(samples, w, h, palette = RIBBON_PALETTE) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const cv = document.createElement("canvas");
  cv.width = Math.round(w * dpr);
  cv.height = Math.round(h * dpr);
  cv.style.width = w + "px";
  cv.style.height = h + "px";
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  drawRibbon(ctx, samples, w, h, { palette, alpha: 0.95 });
  return cv;
}

/** Downsample a long run history to at most `target` columns. */
export function condense(samples, target = 240) {
  if (samples.length <= target) return samples;
  const out = [];
  const bucket = samples.length / target;
  for (let i = 0; i < target; i++) {
    const lo = Math.floor(i * bucket);
    const hi = Math.max(lo + 1, Math.floor((i + 1) * bucket));
    const acc = new Float64Array(5);
    for (let j = lo; j < hi; j++) for (let k = 0; k < 5; k++) acc[k] += samples[j][k];
    for (let k = 0; k < 5; k++) acc[k] /= hi - lo;
    out.push(acc);
  }
  return out;
}
