/* Backend: psychological-state inference.
 *
 * Straight port of AttentionStateInference / EnvironmentMapper /
 * ObservationBuilder from flow_game.py. The weights in weights.js are dumped
 * from the torch modules (seed 1337), so this produces the same pi as the
 * desktop build for the same input sequence — it is the same model, not an
 * approximation of it.
 *
 * Everything here is tiny (32x5 attention), so plain arrays beat any matrix
 * library: a full forward pass is ~40k flops, well under a microsecond. */

import { W_Q, W_K, W_V, W_PI, B_PI, M, W_C1, W_C2 } from "./weights.js";
import {
  D, K, D_K, W_HIST, W_FUT, W_TOTAL, EMA_ALPHA, APM_DECAY, P, V_LATENT,
} from "./config.js";

// --- small dense-matrix helpers -------------------------------------------

/** rows(A) x cols(B) matmul where A is [n][m] and B is [m][p]. */
function matmul(A, B, n, m, p, out) {
  for (let i = 0; i < n; i++) {
    const ai = A[i];
    const oi = out[i];
    for (let j = 0; j < p; j++) {
      let s = 0;
      for (let k = 0; k < m; k++) s += ai[k] * B[k][j];
      oi[j] = s;
    }
  }
  return out;
}

/** Matrix-vector product for W [n][m] and v [m]. */
function matvec(W, v, n, m, out) {
  for (let i = 0; i < n; i++) {
    const wi = W[i];
    let s = 0;
    for (let j = 0; j < m; j++) s += wi[j] * v[j];
    out[i] = s;
  }
  return out;
}

function softmaxInPlace(v, n) {
  let mx = -Infinity;
  for (let i = 0; i < n; i++) if (v[i] > mx) mx = v[i];
  let sum = 0;
  for (let i = 0; i < n; i++) {
    v[i] = Math.exp(v[i] - mx);
    sum += v[i];
  }
  const inv = 1 / sum;
  for (let i = 0; i < n; i++) v[i] *= inv;
  return v;
}

function zeros2d(n, m) {
  const a = new Array(n);
  for (let i = 0; i < n; i++) a[i] = new Float64Array(m);
  return a;
}

/** Box-Muller standard normal. */
let _spare = null;
function randn() {
  if (_spare !== null) {
    const s = _spare;
    _spare = null;
    return s;
  }
  let u = 0, v = 0, s = 0;
  do {
    u = Math.random() * 2 - 1;
    v = Math.random() * 2 - 1;
    s = u * u + v * v;
  } while (s >= 1 || s === 0);
  const f = Math.sqrt((-2 * Math.log(s)) / s);
  _spare = v * f;
  return u * f;
}

// --- AttentionStateInference ----------------------------------------------

export class AttentionStateInference {
  constructor() {
    this.history = zeros2d(W_HIST, D);
    this.emaMu = new Float64Array(D);
    this.histFilled = 0;
    this._scale = 1 / Math.sqrt(D_K);

    // Preallocated scratch — forward() runs every sim tick and must not
    // generate garbage.
    this._O = zeros2d(W_TOTAL, D);
    this._Q = zeros2d(W_TOTAL, D_K);
    this._Kt = zeros2d(W_TOTAL, D_K);
    this._V = zeros2d(W_TOTAL, D_K);
    this._A = zeros2d(W_TOTAL, W_TOTAL);
    this._Z = zeros2d(W_TOTAL, D_K);
    this._z = new Float64Array(D_K);
    this._logits = new Float64Array(K);
    this._oNorm = new Float64Array(D);
    this.pi = new Float64Array(K).fill(1 / K);
  }

  /** Extrapolate velocity decay and idle growth W_FUT frames ahead. */
  _projectFuture(oNorm, vx, vy) {
    const vNormNow = Math.hypot(vx, vy);
    const decayV = 0.98;
    const decayO = 0.9;
    for (let i = 0; i < W_FUT; i++) {
      const row = this._O[W_HIST + i];
      const vFuture = vNormNow * Math.pow(decayV, i + 1);
      const d = Math.pow(decayO, i + 1);
      row[0] = vFuture - this.emaMu[0];
      row[1] = oNorm[1] * d;
      row[2] = oNorm[2] * d;
      row[3] = oNorm[3] * d;
      row[4] = vFuture < 0.05
        ? oNorm[4] + (i + 1) * 0.01
        : Math.max(oNorm[4] - (i + 1) * 0.02, -0.5);
    }
  }

  /**
   * @param {Float64Array|number[]} oT 5-dim raw observation
   * @returns {Float64Array} pi over the 5-simplex
   */
  forward(oT, vx, vy) {
    // EMA baseline tracks each player's personal norm.
    for (let i = 0; i < D; i++) {
      this.emaMu[i] = this.emaMu[i] * (1 - EMA_ALPHA) + oT[i] * EMA_ALPHA;
      this._oNorm[i] = oT[i] - this.emaMu[i];
    }

    // Shift the history window and append.
    const oldest = this.history[0];
    for (let t = 0; t < W_HIST - 1; t++) this.history[t] = this.history[t + 1];
    this.history[W_HIST - 1] = oldest;
    oldest.set(this._oNorm);
    this.histFilled = Math.min(this.histFilled + 1, W_HIST);

    // O = [history | future-projection], shape (32, 5)
    for (let t = 0; t < W_HIST; t++) this._O[t].set(this.history[t]);
    this._projectFuture(this._oNorm, vx, vy);

    matmul(this._O, W_Q, W_TOTAL, D, D_K, this._Q);
    matmul(this._O, W_K, W_TOTAL, D, D_K, this._Kt);
    matmul(this._O, W_V, W_TOTAL, D, D_K, this._V);

    // A = softmax(Q Kᵀ / sqrt(d_k))
    for (let i = 0; i < W_TOTAL; i++) {
      const qi = this._Q[i];
      const ai = this._A[i];
      for (let j = 0; j < W_TOTAL; j++) {
        const kj = this._Kt[j];
        let s = 0;
        for (let d = 0; d < D_K; d++) s += qi[d] * kj[d];
        ai[j] = s * this._scale;
      }
      softmaxInPlace(ai, W_TOTAL);
    }

    // Z = A V, then mean-pool over the window.
    this._z.fill(0);
    for (let i = 0; i < W_TOTAL; i++) {
      const ai = this._A[i];
      const zi = this._Z[i];
      for (let d = 0; d < D_K; d++) {
        let s = 0;
        for (let j = 0; j < W_TOTAL; j++) s += ai[j] * this._V[j][d];
        zi[d] = s;
        this._z[d] += s;
      }
    }
    for (let d = 0; d < D_K; d++) this._z[d] /= W_TOTAL;

    matvec(W_PI, this._z, K, D_K, this._logits);
    for (let k = 0; k < K; k++) this._logits[k] += B_PI[k];
    softmaxInPlace(this._logits, K);
    this.pi.set(this._logits);
    return this.pi;
  }

  reset() {
    for (const row of this.history) row.fill(0);
    this.emaMu.fill(0);
    this.histFilled = 0;
    this.pi.fill(1 / K);
  }
}

// --- EnvironmentMapper -----------------------------------------------------

export class EnvironmentMapper {
  constructor() {
    this.E = new Float64Array(P);
    this.c = new Float64Array(V_LATENT);
    this._hidden = new Float64Array(8);
    this.noiseStd = 0.02;
  }

  /** pi -> (E physics params, c disentangled visual latents). */
  forward(pi) {
    matvec(M, pi, P, K, this.E);
    for (let i = 0; i < P; i++) this.E[i] += randn() * this.noiseStd;
    matvec(W_C1, pi, 8, K, this._hidden);
    for (let i = 0; i < 8; i++) if (this._hidden[i] < 0) this._hidden[i] = 0;
    matvec(W_C2, this._hidden, V_LATENT, 8, this.c);
    return { E: this.E, c: this.c };
  }
}

// --- ObservationBuilder ----------------------------------------------------

/** Computes o_t = [v_norm, apm_proxy, dir_variance, threat, idle_time]. */
export class ObservationBuilder {
  constructor() {
    this.apm = 0;
    this.idleFrames = 0;
    this.headings = [];
    this.maxHeadings = 30;
    this._o = new Float64Array(D);
  }

  registerKeypress() {
    this.apm = this.apm * APM_DECAY + 1.0;
  }

  tickApm() {
    this.apm *= APM_DECAY;
  }

  /** @param threats array of entities with a `pos` [x, y] */
  build(vx, vy, threats, px, py) {
    const vNorm = Math.hypot(vx, vy);
    if (vNorm > 1e-3) {
      this.headings.push(Math.atan2(vy, vx));
      if (this.headings.length > this.maxHeadings) this.headings.shift();
    }

    let dirVar = 0;
    if (this.headings.length >= 2) {
      let C = 0, S = 0;
      for (const h of this.headings) {
        C += Math.cos(h);
        S += Math.sin(h);
      }
      C /= this.headings.length;
      S /= this.headings.length;
      dirVar = 1 - Math.hypot(C, S);
    }

    let minD = 100.0;
    for (const e of threats) {
      const d = Math.hypot(e.pos[0] - px, e.pos[1] - py);
      if (d < minD) minD = d;
    }
    const threat = 1 / (1 + minD);

    if (vNorm < 0.05) this.idleFrames = Math.min(this.idleFrames + 1, 120);
    else this.idleFrames = Math.max(this.idleFrames - 2, 0);

    this._o[0] = vNorm;
    this._o[1] = this.apm;
    this._o[2] = dirVar;
    this._o[3] = threat;
    this._o[4] = this.idleFrames / 120;
    return this._o;
  }

  reset() {
    this.apm = 0;
    this.idleFrames = 0;
    this.headings.length = 0;
  }
}
