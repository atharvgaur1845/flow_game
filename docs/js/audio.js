/* Procedural audio engine.
 *
 * Everything is synthesised in the browser — there is not a single audio
 * file in this project, so a GitHub Pages link is instant and works offline
 * once cached.
 *
 * The interesting half is the score. Rather than looping a track, five
 * musical identities are crossfaded by the same pi mixture that drives the
 * physics and the shader: tempo, scale, timbre, filter cutoff and note
 * density are all continuous functions of the inferred psychological state.
 * When the game decides you have slipped from Tactical into Arousal, you
 * hear it happen before you have consciously noticed.
 *
 *   Arousal   Phrygian, 150 BPM, saw lead, filter wide open
 *   Tactical  bare fourths, 110 BPM, metronomic, tight room
 *   Overload  tritone cluster, 132 BPM, detuned, stuttering hats
 *   Flow      sus2 pads, 96 BPM, long reverb, slow arp
 *   Apathy    minor triad, 60 BPM, everything lowpassed to a murmur
 */

const A1 = 55.0; // root of the whole score

/** Per-archetype musical identity. Blended, never switched abruptly. */
const VOICES = [
  { // 0 AROUSAL
    scale: [0, 1, 3, 5, 7, 8, 10], chord: [0, 3, 7, 10], bpm: 150,
    cutoff: 3200, density: 0.85, wave: "sawtooth", detune: 8,
    padGain: 0.20, arpGain: 0.16, hatGain: 0.18, subGain: 0.30, reverb: 0.18,
  },
  { // 1 TACTICAL
    scale: [0, 2, 5, 7, 10], chord: [0, 7, 12, 17], bpm: 110,
    cutoff: 1500, density: 0.42, wave: "triangle", detune: 2,
    padGain: 0.17, arpGain: 0.10, hatGain: 0.10, subGain: 0.22, reverb: 0.12,
  },
  { // 2 OVERLOAD
    scale: [0, 1, 6, 7, 11], chord: [0, 1, 6, 7], bpm: 132,
    cutoff: 2400, density: 0.95, wave: "square", detune: 26,
    padGain: 0.16, arpGain: 0.15, hatGain: 0.22, subGain: 0.26, reverb: 0.30,
  },
  { // 3 FLOW
    scale: [0, 2, 4, 7, 9], chord: [0, 7, 14, 16], bpm: 96,
    cutoff: 2100, density: 0.55, wave: "triangle", detune: 4,
    padGain: 0.26, arpGain: 0.20, hatGain: 0.08, subGain: 0.24, reverb: 0.55,
  },
  { // 4 APATHY
    scale: [0, 3, 7, 10], chord: [0, 3, 7], bpm: 60,
    cutoff: 520, density: 0.18, wave: "sine", detune: 1,
    padGain: 0.14, arpGain: 0.04, hatGain: 0.02, subGain: 0.18, reverb: 0.22,
  },
];

const semi = (n) => Math.pow(2, n / 12);

export class AudioEngine {
  constructor() {
    this.ctx = null;
    this.ready = false;
    this.musicVol = 0.55;
    this.sfxVol = 0.8;
    this.muted = false;
    this._schedTimer = null;
    this._nextNoteTime = 0;
    this._step = 0;
    this._pi = [0.2, 0.2, 0.2, 0.2, 0.2];
    this._intensity = 0; // 0 menu .. 1 boss fight
    this._dominant = 3;
    this._domHold = 0;
    this._musicOn = false;
  }

  /** Must be called from a user gesture — browsers refuse otherwise. */
  unlock() {
    if (this.ctx) {
      if (this.ctx.state === "suspended") this.ctx.resume();
      return;
    }
    const Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) return;
    const ctx = new Ctx({ latencyHint: "interactive" });
    this.ctx = ctx;

    // --- master chain -----------------------------------------------------
    this.master = ctx.createGain();
    this.master.gain.value = this.muted ? 0 : 1;
    const comp = ctx.createDynamicsCompressor();
    comp.threshold.value = -14;
    comp.knee.value = 24;
    comp.ratio.value = 8;
    comp.attack.value = 0.004;
    comp.release.value = 0.18;
    this.master.connect(comp).connect(ctx.destination);

    // Shared reverb — one convolver fed by per-bus sends.
    this.reverb = ctx.createConvolver();
    this.reverb.buffer = this._makeImpulse(2.8, 2.4);
    this.reverbGain = ctx.createGain();
    this.reverbGain.gain.value = 0.9;
    this.reverb.connect(this.reverbGain).connect(this.master);

    this.sfxBus = ctx.createGain();
    this.sfxBus.gain.value = this.sfxVol;
    this.sfxBus.connect(this.master);
    this.sfxSend = ctx.createGain();
    this.sfxSend.gain.value = 0.16;
    this.sfxBus.connect(this.sfxSend).connect(this.reverb);

    this.musicBus = ctx.createGain();
    this.musicBus.gain.value = this.musicVol;
    // Music runs through a lowpass we sweep from pi — the single most
    // effective "the game is reading you" cue in the whole mix.
    this.musicFilter = ctx.createBiquadFilter();
    this.musicFilter.type = "lowpass";
    this.musicFilter.frequency.value = 1200;
    this.musicFilter.Q.value = 0.7;
    // Duck gain is separate so hits can sidechain without fighting the
    // user's volume setting.
    this.musicDuck = ctx.createGain();
    this.musicDuck.gain.value = 1;
    this.musicFilter.connect(this.musicDuck).connect(this.musicBus);
    this.musicBus.connect(this.master);
    this.musicSend = ctx.createGain();
    this.musicSend.gain.value = 0.3;
    this.musicBus.connect(this.musicSend).connect(this.reverb);

    this._noise = this._makeNoise(2.0);
    this._buildDrone();
    this.ready = true;
    if (ctx.state === "suspended") ctx.resume();
  }

  // --- buffers -------------------------------------------------------------

  _makeImpulse(seconds, decay) {
    const ctx = this.ctx;
    const len = Math.floor(ctx.sampleRate * seconds);
    const buf = ctx.createBuffer(2, len, ctx.sampleRate);
    for (let ch = 0; ch < 2; ch++) {
      const d = buf.getChannelData(ch);
      for (let i = 0; i < len; i++) {
        // Slight pre-delay shaping keeps the tail from smearing transients.
        const t = i / len;
        d[i] = (Math.random() * 2 - 1) * Math.pow(1 - t, decay) * (t < 0.002 ? t / 0.002 : 1);
      }
    }
    return buf;
  }

  _makeNoise(seconds) {
    const ctx = this.ctx;
    const len = Math.floor(ctx.sampleRate * seconds);
    const buf = ctx.createBuffer(1, len, ctx.sampleRate);
    const d = buf.getChannelData(0);
    for (let i = 0; i < len; i++) d[i] = Math.random() * 2 - 1;
    return buf;
  }

  // --- settings ------------------------------------------------------------

  setMuted(m) {
    this.muted = m;
    if (this.ready) this.master.gain.setTargetAtTime(m ? 0 : 1, this.ctx.currentTime, 0.02);
  }

  setMusicVolume(v) {
    this.musicVol = v;
    if (this.ready) this.musicBus.gain.setTargetAtTime(v, this.ctx.currentTime, 0.05);
  }

  setSfxVolume(v) {
    this.sfxVol = v;
    if (this.ready) this.sfxBus.gain.setTargetAtTime(v, this.ctx.currentTime, 0.05);
  }

  // --- drone ---------------------------------------------------------------

  _buildDrone() {
    const ctx = this.ctx;
    this.droneGain = ctx.createGain();
    this.droneGain.gain.value = 0;
    this.droneGain.connect(this.musicFilter);

    this.droneOscs = [];
    // Sub + two detuned saws a fifth apart: the harmonic floor of the score.
    const specs = [
      { type: "sine", mult: 1, detune: 0, gain: 0.9 },
      { type: "sawtooth", mult: 2, detune: -6, gain: 0.16 },
      { type: "sawtooth", mult: 3, detune: 7, gain: 0.1 },
    ];
    for (const sp of specs) {
      const o = ctx.createOscillator();
      o.type = sp.type;
      o.frequency.value = A1 * sp.mult;
      o.detune.value = sp.detune;
      const g = ctx.createGain();
      g.gain.value = sp.gain;
      o.connect(g).connect(this.droneGain);
      o.start();
      this.droneOscs.push({ osc: o, gain: g, spec: sp });
    }

    // A slow noise bed that only really opens up under Overload.
    this.bedSrc = ctx.createBufferSource();
    this.bedSrc.buffer = this._noise;
    this.bedSrc.loop = true;
    this.bedFilter = ctx.createBiquadFilter();
    this.bedFilter.type = "bandpass";
    this.bedFilter.frequency.value = 700;
    this.bedFilter.Q.value = 0.6;
    this.bedGain = ctx.createGain();
    this.bedGain.gain.value = 0;
    this.bedSrc.connect(this.bedFilter).connect(this.bedGain).connect(this.musicFilter);
    this.bedSrc.start();
  }

  // --- transport -----------------------------------------------------------

  startMusic() {
    if (!this.ready || this._musicOn) return;
    this._musicOn = true;
    this._nextNoteTime = this.ctx.currentTime + 0.1;
    this._step = 0;
    this.droneGain.gain.setTargetAtTime(0.16, this.ctx.currentTime, 1.2);
    // Lookahead scheduling: a timer this coarse would sound sloppy if it
    // triggered notes directly, so it only queues them ~120 ms ahead and the
    // audio clock does the timing.
    this._schedTimer = setInterval(() => this._scheduler(), 25);
  }

  stopMusic() {
    if (!this.ready) return;
    this._musicOn = false;
    clearInterval(this._schedTimer);
    this._schedTimer = null;
    this.droneGain.gain.setTargetAtTime(0, this.ctx.currentTime, 0.4);
    this.bedGain.gain.setTargetAtTime(0, this.ctx.currentTime, 0.4);
  }

  /** Feed the live psychological mixture + a 0..1 tension scalar. */
  setState(pi, intensity) {
    this._pi = pi;
    this._intensity = intensity;
    if (!this.ready || !this._musicOn) return;

    const t = this.ctx.currentTime;
    const b = this._blend();

    // Dominant archetype picks the scale, but only after it has held for a
    // moment — otherwise the melody rewrites itself several times a second.
    let dom = 0;
    for (let k = 1; k < 5; k++) if (pi[k] > pi[dom]) dom = k;
    if (dom === this._dominant) this._domHold = 0;
    else {
      this._domHold += 0.05;
      if (this._domHold > 1.2) {
        this._dominant = dom;
        this._domHold = 0;
      }
    }

    const cutoff = b.cutoff * (0.55 + 0.65 * intensity);
    this.musicFilter.frequency.setTargetAtTime(
      Math.max(180, Math.min(9000, cutoff)), t, 0.35);
    this.droneGain.gain.setTargetAtTime(b.subGain * (0.5 + 0.5 * intensity), t, 0.5);
    this.bedGain.gain.setTargetAtTime(0.035 * pi[2] * (0.4 + intensity), t, 0.6);
    this.bedFilter.frequency.setTargetAtTime(400 + 1600 * pi[2], t, 0.8);
    this.reverbGain.gain.setTargetAtTime(0.35 + 1.1 * b.reverb, t, 0.6);
    for (const d of this.droneOscs) {
      d.osc.detune.setTargetAtTime(d.spec.detune * (1 + 5 * pi[2]), t, 0.5);
    }
  }

  /** pi-weighted blend of the five musical identities. */
  _blend() {
    const pi = this._pi;
    const out = { bpm: 0, cutoff: 0, density: 0, detune: 0, padGain: 0, arpGain: 0, hatGain: 0, subGain: 0, reverb: 0 };
    for (let k = 0; k < 5; k++) {
      const v = VOICES[k];
      const w = pi[k];
      out.bpm += v.bpm * w;
      out.cutoff += v.cutoff * w;
      out.density += v.density * w;
      out.detune += v.detune * w;
      out.padGain += v.padGain * w;
      out.arpGain += v.arpGain * w;
      out.hatGain += v.hatGain * w;
      out.subGain += v.subGain * w;
      out.reverb += v.reverb * w;
    }
    return out;
  }

  _scheduler() {
    if (!this._musicOn) return;
    const ctx = this.ctx;
    const b = this._blend();
    const stepDur = 60 / Math.max(40, b.bpm) / 4; // sixteenth notes
    while (this._nextNoteTime < ctx.currentTime + 0.12) {
      this._scheduleStep(this._step, this._nextNoteTime, b);
      this._nextNoteTime += stepDur;
      this._step = (this._step + 1) % 32;
    }
  }

  _scheduleStep(step, t, b) {
    const voice = VOICES[this._dominant];
    const inten = this._intensity;
    const bar16 = step % 16;

    // Kick on the downbeats — the heartbeat of the arena.
    if (bar16 === 0 || (bar16 === 8 && inten > 0.3)) {
      this._kick(t, 0.55 + 0.35 * inten);
    }

    // Hats: density and stutter both rise with Overload.
    if (b.hatGain > 0.03 && Math.random() < b.density) {
      this._hat(t, b.hatGain * (0.5 + 0.6 * inten), this._pi[2]);
    }

    // Bass pluck on the off-eighths.
    if (bar16 % 4 === 2 && inten > 0.15) {
      const deg = voice.scale[(step * 3) % voice.scale.length];
      this._pluck(t, A1 * semi(deg), 0.16 * (0.4 + inten), b, voice);
    }

    // Pad chord every bar, voiced by the dominant archetype.
    if (bar16 === 0) {
      this._pad(t, voice, b, 60 / Math.max(40, b.bpm) * 4);
    }

    // Arp — the melodic layer. Flow makes it wide and patient, Arousal makes
    // it a scramble.
    if (b.arpGain > 0.05 && step % 2 === 0 && Math.random() < b.density * 0.8) {
      const deg = voice.scale[(step / 2) % voice.scale.length];
      const oct = 2 + ((step >> 3) % 2);
      this._arp(t, A1 * semi(deg) * Math.pow(2, oct), b.arpGain * (0.35 + 0.65 * inten), voice, b);
    }
  }

  // --- musical voices ------------------------------------------------------

  _kick(t, gain) {
    const ctx = this.ctx;
    const o = ctx.createOscillator();
    const g = ctx.createGain();
    o.type = "sine";
    o.frequency.setValueAtTime(150, t);
    o.frequency.exponentialRampToValueAtTime(38, t + 0.12);
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(gain * 0.5, t + 0.005);
    g.gain.exponentialRampToValueAtTime(0.0001, t + 0.28);
    o.connect(g).connect(this.musicFilter);
    o.start(t);
    o.stop(t + 0.3);
  }

  _hat(t, gain, overload) {
    const ctx = this.ctx;
    const s = ctx.createBufferSource();
    s.buffer = this._noise;
    s.playbackRate.value = 1 + Math.random() * 0.4;
    const f = ctx.createBiquadFilter();
    f.type = "highpass";
    f.frequency.value = 6000 - 2000 * overload;
    const g = ctx.createGain();
    const dur = 0.03 + Math.random() * 0.04 * (1 + overload);
    g.gain.setValueAtTime(gain * 0.35, t);
    g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
    s.connect(f).connect(g).connect(this.musicFilter);
    s.start(t, Math.random() * 1.5);
    s.stop(t + dur + 0.01);
  }

  _pluck(t, freq, gain, b, voice) {
    const ctx = this.ctx;
    const o = ctx.createOscillator();
    o.type = voice.wave;
    o.frequency.value = freq;
    o.detune.value = (Math.random() - 0.5) * b.detune * 2;
    const f = ctx.createBiquadFilter();
    f.type = "lowpass";
    f.frequency.setValueAtTime(freq * 8, t);
    f.frequency.exponentialRampToValueAtTime(freq * 2, t + 0.2);
    const g = ctx.createGain();
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(gain, t + 0.008);
    g.gain.exponentialRampToValueAtTime(0.0001, t + 0.26);
    o.connect(f).connect(g).connect(this.musicFilter);
    o.start(t);
    o.stop(t + 0.3);
  }

  _pad(t, voice, b, dur) {
    const ctx = this.ctx;
    for (const iv of voice.chord) {
      for (const det of [-b.detune, b.detune]) {
        const o = ctx.createOscillator();
        o.type = voice.wave === "square" ? "sawtooth" : voice.wave;
        o.frequency.value = A1 * 2 * semi(iv);
        o.detune.value = det;
        const g = ctx.createGain();
        g.gain.setValueAtTime(0.0001, t);
        g.gain.linearRampToValueAtTime(b.padGain * 0.09, t + dur * 0.35);
        g.gain.linearRampToValueAtTime(0.0001, t + dur);
        o.connect(g).connect(this.musicFilter);
        o.start(t);
        o.stop(t + dur + 0.05);
      }
    }
  }

  _arp(t, freq, gain, voice, b) {
    const ctx = this.ctx;
    const o = ctx.createOscillator();
    o.type = voice.wave;
    o.frequency.value = freq;
    o.detune.value = (Math.random() - 0.5) * b.detune;
    const g = ctx.createGain();
    const dur = 0.12 + 0.25 * b.reverb;
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(gain * 0.16, t + 0.01);
    g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
    o.connect(g).connect(this.musicFilter);
    o.start(t);
    o.stop(t + dur + 0.02);
  }

  /** Sidechain-style duck, used on damage and boss impacts. */
  duck(amount = 0.5, release = 0.35) {
    if (!this.ready) return;
    const t = this.ctx.currentTime;
    this.musicDuck.gain.cancelScheduledValues(t);
    this.musicDuck.gain.setValueAtTime(1 - amount, t);
    this.musicDuck.gain.setTargetAtTime(1, t + 0.02, release);
  }

  // --- sfx primitives ------------------------------------------------------

  _tone({ type = "sine", f0, f1, dur, gain = 0.3, t0 = 0, dest = null, curve = "exp" }) {
    const ctx = this.ctx;
    const t = ctx.currentTime + t0;
    const o = ctx.createOscillator();
    o.type = type;
    o.frequency.setValueAtTime(f0, t);
    if (f1 && f1 !== f0) {
      if (curve === "exp") o.frequency.exponentialRampToValueAtTime(Math.max(1, f1), t + dur);
      else o.frequency.linearRampToValueAtTime(f1, t + dur);
    }
    const g = ctx.createGain();
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(gain, t + Math.min(0.012, dur * 0.2));
    g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
    o.connect(g).connect(dest || this.sfxBus);
    o.start(t);
    o.stop(t + dur + 0.02);
    return { osc: o, gain: g, t };
  }

  _noiseBurst({ dur = 0.2, gain = 0.3, f0 = 2000, f1 = 400, type = "bandpass", q = 1.2, t0 = 0 }) {
    const ctx = this.ctx;
    const t = ctx.currentTime + t0;
    const s = ctx.createBufferSource();
    s.buffer = this._noise;
    s.playbackRate.value = 1;
    const f = ctx.createBiquadFilter();
    f.type = type;
    f.Q.value = q;
    f.frequency.setValueAtTime(f0, t);
    f.frequency.exponentialRampToValueAtTime(Math.max(40, f1), t + dur);
    const g = ctx.createGain();
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(gain, t + 0.008);
    g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
    s.connect(f).connect(g).connect(this.sfxBus);
    s.start(t, Math.random() * 1.5);
    s.stop(t + dur + 0.02);
  }

  // --- game sfx ------------------------------------------------------------

  dash() {
    if (!this.ready) return;
    this._noiseBurst({ dur: 0.28, gain: 0.34, f0: 5200, f1: 320, q: 1.6 });
    this._tone({ type: "sawtooth", f0: 720, f1: 130, dur: 0.24, gain: 0.16 });
    this._tone({ type: "sine", f0: 1400, f1: 260, dur: 0.16, gain: 0.1 });
    this.duck(0.22, 0.14);
  }

  dashReady() {
    if (!this.ready) return;
    this._tone({ type: "triangle", f0: 1180, f1: 1180, dur: 0.05, gain: 0.11 });
    this._tone({ type: "triangle", f0: 1770, f1: 1770, dur: 0.07, gain: 0.08, t0: 0.045 });
  }

  kill(kind) {
    if (!this.ready) return;
    if (kind === "fast") {
      this._tone({ type: "square", f0: 1500, f1: 420, dur: 0.09, gain: 0.16 });
      this._noiseBurst({ dur: 0.07, gain: 0.14, f0: 6000, f1: 2200, q: 2 });
    } else if (kind === "tank") {
      this._tone({ type: "sawtooth", f0: 260, f1: 60, dur: 0.3, gain: 0.24 });
      this._noiseBurst({ dur: 0.26, gain: 0.2, f0: 1200, f1: 120, q: 0.8 });
    } else {
      this._tone({ type: "square", f0: 760, f1: 200, dur: 0.14, gain: 0.18 });
      this._noiseBurst({ dur: 0.12, gain: 0.14, f0: 3000, f1: 700, q: 1.4 });
    }
  }

  hit() {
    if (!this.ready) return;
    this._tone({ type: "sine", f0: 190, f1: 45, dur: 0.32, gain: 0.42 });
    this._noiseBurst({ dur: 0.2, gain: 0.3, f0: 900, f1: 90, type: "lowpass", q: 1.1 });
    this.duck(0.6, 0.3);
  }

  shieldBlock() {
    if (!this.ready) return;
    this._tone({ type: "triangle", f0: 900, f1: 1500, dur: 0.14, gain: 0.2 });
    this._noiseBurst({ dur: 0.12, gain: 0.12, f0: 3000, f1: 5000, q: 3 });
  }

  pickup(kind) {
    if (!this.ready) return;
    // Each pickup gets its own three-note signature, so you can identify
    // what you grabbed without taking your eyes off the swarm.
    const sigs = {
      heal: { notes: [0, 4, 7], type: "sine", base: 523 },
      dash_boost: { notes: [0, 7, 12], type: "triangle", base: 660 },
      speed_boost: { notes: [0, 2, 7], type: "square", base: 587 },
      shield: { notes: [0, 5, 9], type: "triangle", base: 440 },
      max_hp: { notes: [0, 4, 9], type: "sine", base: 698 },
    };
    const s = sigs[kind] || sigs.heal;
    s.notes.forEach((n, i) => {
      this._tone({
        type: s.type, f0: s.base * semi(n), f1: s.base * semi(n),
        dur: 0.16, gain: 0.15, t0: i * 0.055,
      });
    });
  }

  roomClear() {
    if (!this.ready) return;
    [0, 4, 7, 12].forEach((n, i) => {
      this._tone({ type: "triangle", f0: 440 * semi(n), f1: 440 * semi(n), dur: 0.5, gain: 0.14, t0: i * 0.07 });
    });
    this._noiseBurst({ dur: 0.5, gain: 0.1, f0: 400, f1: 6000, q: 0.7 });
  }

  bossSpawn() {
    if (!this.ready) return;
    this._tone({ type: "sawtooth", f0: 110, f1: 32, dur: 1.6, gain: 0.4 });
    this._noiseBurst({ dur: 1.4, gain: 0.22, f0: 200, f1: 4000, q: 0.6 });
    this.duck(0.5, 0.9);
  }

  bossHit() {
    if (!this.ready) return;
    // FM clang: a non-integer modulator ratio is what makes metal sound like
    // metal rather than like a bell.
    const ctx = this.ctx;
    const t = ctx.currentTime;
    const carrier = ctx.createOscillator();
    const mod = ctx.createOscillator();
    const modGain = ctx.createGain();
    carrier.type = "sine";
    carrier.frequency.value = 420;
    mod.type = "square";
    mod.frequency.value = 420 * 2.37;
    modGain.gain.setValueAtTime(1400, t);
    modGain.gain.exponentialRampToValueAtTime(20, t + 0.35);
    mod.connect(modGain).connect(carrier.frequency);
    const g = ctx.createGain();
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(0.3, t + 0.004);
    g.gain.exponentialRampToValueAtTime(0.0001, t + 0.4);
    carrier.connect(g).connect(this.sfxBus);
    carrier.start(t); mod.start(t);
    carrier.stop(t + 0.45); mod.stop(t + 0.45);
    this.duck(0.3, 0.2);
  }

  bossDeath() {
    if (!this.ready) return;
    this._tone({ type: "sawtooth", f0: 320, f1: 28, dur: 2.2, gain: 0.42 });
    this._noiseBurst({ dur: 1.8, gain: 0.34, f0: 4000, f1: 60, type: "lowpass", q: 0.9 });
    [0, 7, 12, 19].forEach((n, i) =>
      this._tone({ type: "triangle", f0: 440 * semi(n), f1: 440 * semi(n), dur: 1.2, gain: 0.12, t0: 0.3 + i * 0.1 }));
    this.duck(0.7, 1.2);
  }

  gameOver() {
    if (!this.ready) return;
    [0, -3, -7, -12].forEach((n, i) =>
      this._tone({ type: "sawtooth", f0: 330 * semi(n), f1: 330 * semi(n) * 0.98, dur: 1.6, gain: 0.13, t0: i * 0.14, curve: "lin" }));
    this._tone({ type: "sine", f0: 90, f1: 40, dur: 2.4, gain: 0.3 });
  }

  /** Room-timer countdown. Pitch rises as the remaining seconds fall, so the
   *  urgency is audible without looking at the clock. */
  timerTick(secondsLeft) {
    if (!this.ready) return;
    const f = 520 + (10 - Math.min(secondsLeft, 10)) * 62;
    const loud = secondsLeft <= 5;
    this._tone({ type: "square", f0: f, f1: f, dur: loud ? 0.09 : 0.06, gain: loud ? 0.17 : 0.09 });
    if (loud) this._tone({ type: "triangle", f0: f * 2, f1: f * 2, dur: 0.07, gain: 0.07 });
  }

  /** Room dropped into overtime: a downward resolve, not an alarm. Nothing
   *  bad happened to the player, they just lost half the clear bonus. */
  overtime() {
    if (!this.ready) return;
    [0, -2, -5].forEach((n, i) =>
      this._tone({ type: "triangle", f0: 440 * semi(n), f1: 440 * semi(n), dur: 0.5, gain: 0.12, t0: i * 0.11 }));
    this._tone({ type: "sine", f0: 160, f1: 96, dur: 0.7, gain: 0.18 });
  }

  heartbeat() {
    if (!this.ready) return;
    this._tone({ type: "sine", f0: 90, f1: 44, dur: 0.16, gain: 0.34 });
    this._tone({ type: "sine", f0: 78, f1: 38, dur: 0.2, gain: 0.24, t0: 0.19 });
  }

  milestone() {
    if (!this.ready) return;
    [0, 7, 12, 16, 19].forEach((n, i) =>
      this._tone({ type: "triangle", f0: 660 * semi(n), f1: 660 * semi(n), dur: 0.45, gain: 0.13, t0: i * 0.06 }));
  }

  uiMove() {
    if (!this.ready) return;
    this._tone({ type: "square", f0: 880, f1: 880, dur: 0.03, gain: 0.06 });
  }

  uiConfirm() {
    if (!this.ready) return;
    this._tone({ type: "triangle", f0: 660, f1: 990, dur: 0.12, gain: 0.14 });
  }

  shopOpen() {
    if (!this.ready) return;
    [0, 5, 9, 14].forEach((n, i) =>
      this._tone({ type: "sine", f0: 330 * semi(n), f1: 330 * semi(n), dur: 0.9, gain: 0.1, t0: i * 0.08 }));
  }
}
