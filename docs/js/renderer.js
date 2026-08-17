/* WebGL2 renderer.
 *
 * The fragment shader is a port of the GLSL 330 shader in flow_game.py to
 * GLSL ES 3.00, plus three additions that only make sense for a browser
 * build where nobody has read a manual first:
 *
 *   - a dash-cooldown ring orbiting the player (diegetic replacement for
 *     glancing at a corner bar),
 *   - a low-HP vignette that pulses with the heartbeat cue,
 *   - edge markers for enemies outside the camera frustum (at Flow's 1.2x
 *     zoom a meaningful slice of the arena is off-screen).
 *
 * Quality tiers are compile-time #defines rather than uniforms so the low
 * tiers genuinely skip work instead of branching around it. */

import * as C from "./config.js";

const VERT = `#version 300 es
layout(location=0) in vec2 a_pos;
out vec2 v_uv;
void main() {
    v_uv = a_pos * 0.5 + 0.5;
    gl_Position = vec4(a_pos, 0.0, 1.0);
}`;

const FRAG = `#version 300 es
precision highp float;
precision highp int;

in  vec2 v_uv;
out vec4 fragColor;

uniform float u_time;
uniform vec2  u_resolution;
uniform vec2  u_player;
uniform vec2  u_enemies[16];
uniform float u_enemy_radii[16];
uniform int   u_enemy_count;
uniform float u_world_half;
uniform float u_cam_zoom;
uniform float u_c1_aberration;
uniform float u_c2_grain;
uniform float u_c3_warp;
uniform float u_c4_bloom;
uniform vec4  u_pi_top4;

// Boss
uniform float u_boss_active;
uniform vec2  u_boss_pos;
uniform float u_boss_radius;
uniform float u_boss_hp_frac;

// Feedback
uniform vec2  u_shake;
uniform float u_player_flash;    // dash i-frames
uniform float u_hit_flash;       // took damage
uniform float u_dash_phase;      // 0 idle .. 1 dash start
uniform vec2  u_dash_dir;        // unit vector of current/last dash
uniform vec2  u_dash_trail[32];  // lethal dash-trail waypoints (world)
uniform float u_dash_trail_a[32];// per-point alpha (1 = fresh, 0 = expired)
uniform int   u_dash_trail_count;
uniform vec2  u_pickups[8];      // food / power-up positions (world)
uniform int   u_pickup_kinds[8]; // 0 heal,1 dash,2 speed,3 shield,4 max_hp
uniform int   u_pickup_count;

// Web additions
uniform float u_dash_ready;      // 0..1 cooldown fill
uniform float u_dash_ready_pop;  // decaying flash when it hits 1.0
uniform float u_low_hp;          // 0..1, ramps in below 35% HP
uniform float u_shield;          // 0..1 shield buff strength
uniform float u_fx;              // global post-fx scale (0 = reduced motion)

float sdCircle(vec2 p, float r) { return length(p) - r; }
float hash21(vec2 p) {
    return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

vec2 screenToWorld(vec2 uv) {
    float aspect = u_resolution.x / u_resolution.y;
    vec2 ndc = uv * 2.0 - 1.0;
    ndc.x *= aspect;
    return (ndc * u_world_half) / u_cam_zoom;
}

vec2 worldToUv(vec2 w) {
    float aspect = u_resolution.x / u_resolution.y;
    vec2 ndc = (w * u_cam_zoom) / u_world_half;
    ndc.x /= aspect;
    return ndc * 0.5 + 0.5;
}

// SDF for a 2D line segment.
float sdSegment(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a;
    vec2 ba = b - a;
    float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-6), 0.0, 1.0);
    return length(pa - ba * h);
}

// SDF for a 2D rotated diamond (square rotated 45 deg) of radius r.
float sdDiamond(vec2 p, float r) {
    return (abs(p.x) + abs(p.y)) - r;
}

vec3 pickupColor(int kind) {
    if (kind == 0) return vec3(0.30, 1.00, 0.45); // heal       — green
    if (kind == 1) return vec3(0.30, 0.95, 1.00); // dash_boost — cyan
    if (kind == 2) return vec3(1.00, 0.65, 0.20); // speed      — orange
    if (kind == 3) return vec3(0.40, 0.55, 1.00); // shield     — blue
    return                vec3(1.00, 0.95, 0.30); // max_hp     — yellow
}

vec3 sampleScene(vec2 world) {
    vec2 g = world * 0.5;
    float grid = 0.0;
    grid += smoothstep(0.04, 0.0, abs(fract(g.x + u_time * 0.05) - 0.5) - 0.48);
    grid += smoothstep(0.04, 0.0, abs(fract(g.y - u_time * 0.03) - 0.5) - 0.48);
    // Background tint driven by the inferred psychological mixture.
    vec3 bg_tint = vec3(0.55, 0.12, 0.18) * u_pi_top4.x   // arousal: ember red
                 + vec3(0.08, 0.22, 0.55) * u_pi_top4.y   // tactical: deep blue
                 + vec3(0.42, 0.08, 0.55) * u_pi_top4.z   // overload: violet
                 + vec3(0.05, 0.45, 0.40) * u_pi_top4.w;  // flow: teal
    vec3 bg_low  = vec3(0.02, 0.02, 0.05) + bg_tint * 0.18;
    vec3 bg_high = vec3(0.04, 0.06, 0.12) + bg_tint * 0.55;
    vec3 bg = mix(bg_low, bg_high,
                  smoothstep(u_world_half, 0.0, length(world)));
    vec3 grid_col = mix(vec3(0.08, 0.15, 0.35), bg_tint * 1.6, 0.45);
    vec3 col = bg + grid_col * grid;

    // Arena boundary: glowing outline; outside is heavily dimmed.
    float bx = abs(world.x) - u_world_half;
    float by = abs(world.y) - u_world_half;
    float boundary_sdf = max(bx, by);
    float outline = abs(boundary_sdf) - 0.10;
    float outline_mask = smoothstep(0.08, 0.0, outline);
    vec3 boundary_color = vec3(0.30, 0.85, 1.0)
                        * (0.85 + 0.25 * sin(u_time * 2.5));
    col += boundary_color * outline_mask * 1.4;
    if (boundary_sdf > 0.0) {
        col *= 0.18;
        col += vec3(0.05, 0.02, 0.08);
    }

    // Lethal dash trail (additive glow on the trail segments).
    for (int i = 0; i < TRAIL_STEPS; ++i) {
        if (i >= u_dash_trail_count) break;
        float a = u_dash_trail_a[i];
        if (a <= 0.0) continue;
        vec2 tp = u_dash_trail[i];
        float td;
        if (i + 1 < u_dash_trail_count) {
            td = sdSegment(world, tp, u_dash_trail[i + 1]);
        } else {
            td = length(world - tp);
        }
        float core = smoothstep(0.10, -0.04, td - 0.18);
        float glow = exp(-max(td - 0.18, 0.0) * 3.0);
        col += vec3(1.0, 0.85, 1.0) * core * a * 0.85;
        col += vec3(0.7, 0.3, 1.0) * glow * a * 0.35;
    }

    // Pickups: rotating diamond with a soft halo, colour-coded by kind.
    for (int i = 0; i < 8; ++i) {
        if (i >= u_pickup_count) break;
        vec2 pp = u_pickups[i];
        vec3 pc = pickupColor(u_pickup_kinds[i]);
        float t = u_time * 2.2 + float(i) * 0.7;
        float ca = cos(t), sa = sin(t);
        vec2 lp = world - pp;
        vec2 rp = vec2(ca * lp.x - sa * lp.y,
                       sa * lp.x + ca * lp.y);
        float pulse = 0.42 + 0.06 * sin(u_time * 5.0 + float(i));
        float dpkp  = sdDiamond(rp, pulse);
        float coreP = smoothstep(0.04, -0.03, dpkp);
        float glowP = exp(-max(dpkp, 0.0) * 3.5);
        col += pc * coreP * 1.15;
        col += pc * glowP * 0.55;
    }

    // Player.
    float dp = sdCircle(world - u_player, 0.45);
    float player_core = smoothstep(0.02, -0.02, dp);
    float player_glow = exp(-max(dp, 0.0) * 4.0) * (0.6 + 0.8 * u_c4_bloom);
    vec3 player_tint = mix(vec3(0.2, 0.9, 1.0),
                           vec3(1.0, 1.0, 1.0),
                           u_player_flash);
    col += player_tint * player_core * (1.0 + 0.6 * u_dash_phase);
    col += vec3(0.1, 0.6, 1.0) * player_glow * 0.45;

    // --- Dash-cooldown ring (web addition) --------------------------------
    // A thin arc orbiting the player that sweeps as the cooldown refills and
    // flares white the instant it completes.
    {
        vec2 lp = world - u_player;
        float r = length(lp);
        float ring_d = abs(r - 0.78) - 0.035;
        float theta = atan(lp.y, lp.x);
        float a = (theta + 3.14159265) / 6.28318531;
        float filled = step(a, u_dash_ready);
        float ring_mask = smoothstep(0.045, 0.0, ring_d);
        vec3 ring_col = mix(vec3(0.35, 0.55, 0.75), vec3(0.5, 1.0, 1.0),
                            u_dash_ready);
        col += ring_col * ring_mask * (0.18 + 0.72 * filled);
        // Ready pop: a full bright halo that decays.
        col += vec3(0.7, 1.0, 1.0) * ring_mask * u_dash_ready_pop * 1.2;
    }

    // --- Shield bubble (web addition) -------------------------------------
    if (u_shield > 0.0) {
        vec2 lp = world - u_player;
        float sd = abs(length(lp) - (0.72 + 0.04 * sin(u_time * 6.0))) - 0.03;
        col += vec3(0.45, 0.6, 1.0) * smoothstep(0.05, 0.0, sd) * u_shield * 0.9;
    }

    // --- Trippy "killer" dash effect (cosmetic, no hitbox change) ---------
    if (u_dash_phase > 0.0) {
        // 1. Echo trail behind the dash direction.
        for (int i = 1; i <= 4; ++i) {
            float t = float(i) / 4.0;
            vec2 echo_center = u_player - u_dash_dir * t * 0.9;
            float de = sdCircle(world - echo_center, 0.45 * (1.0 - 0.18 * t));
            float ae = (1.0 - t) * u_dash_phase;
            float emask = smoothstep(0.04, -0.02, de);
            col += vec3(0.6, 1.0, 1.0) * emask * ae * 0.55;
        }
        // 2. 4-arm spinning blade SDF around the player.
        vec2 lp = world - u_player;
        float ang = u_time * 14.0;
        float blade = 1e9;
        for (int k = 0; k < 4; ++k) {
            float a = ang + float(k) * 1.5707963;
            vec2 r = vec2(cos(a), sin(a));
            vec2 q = vec2(dot(lp, r), dot(lp, vec2(-r.y, r.x)));
            float d = abs(q.y) - 0.04;
            d = max(d, q.x - 0.95);
            d = max(d, -q.x);
            blade = min(blade, d);
        }
        float blade_mask = smoothstep(0.02, -0.01, blade) * u_dash_phase;
        vec3 blade_tint = mix(vec3(0.4, 1.0, 1.0), vec3(1.0, 0.4, 1.0),
                              0.5 + 0.5 * sin(u_time * 9.0));
        col += blade_tint * blade_mask * 1.4;
        // 3. Outer ripple ring that pulses outward during dash.
        float ring_r = 0.55 + (1.0 - u_dash_phase) * 0.6;
        float ring = abs(length(lp) - ring_r) - 0.03;
        col += vec3(1.0, 0.8, 1.0)
               * smoothstep(0.04, 0.0, ring) * u_dash_phase * 0.6;
    }

    // Enemies.
    float pulse = 0.85 + 0.15 * sin(u_time * 4.0);
    for (int i = 0; i < 16; ++i) {
        if (i >= u_enemy_count) break;
        vec2 ep = u_enemies[i];
        float er = u_enemy_radii[i] * pulse;
        float de = sdCircle(world - ep, er);
        float ec = smoothstep(0.02, -0.02, de);
        float eg = exp(-max(de, 0.0) * 5.0) * (0.5 + 0.9 * u_c4_bloom);
        col += vec3(1.0, 0.25, 0.7) * ec;
        col += vec3(0.9, 0.15, 0.5) * eg * 0.35;
    }

    // Boss.
    if (u_boss_active > 0.5) {
        float br = u_boss_radius * (0.95 + 0.05 * sin(u_time * 2.0));
        float db = sdCircle(world - u_boss_pos, br);
        float bc = smoothstep(0.04, -0.04, db);
        float bg_ = exp(-max(db, 0.0) * 1.8) * (0.9 + 1.2 * u_c4_bloom);
        vec3 boss_base = mix(vec3(0.4, 0.05, 0.1), vec3(1.0, 0.6, 0.1),
                             u_boss_hp_frac);
        col += boss_base * bc;
        col += boss_base * bg_ * 0.45;
        // HP ring.
        float ring_d = abs(length(world - u_boss_pos) - (u_boss_radius + 0.35));
        float ring_mask = smoothstep(0.04, 0.0, ring_d);
        float theta = atan(world.y - u_boss_pos.y, world.x - u_boss_pos.x);
        float a = (theta + 3.14159) / 6.28318;
        float ring = ring_mask * step(a, u_boss_hp_frac);
        col += vec3(1.0, 0.7, 0.2) * ring * 0.9;
    }

    vec3 tint = vec3(0.6, 0.2, 0.3) * u_pi_top4.x
              + vec3(0.15, 0.3, 0.5) * u_pi_top4.y
              + vec3(0.4, 0.1, 0.6) * u_pi_top4.z
              + vec3(0.1, 0.5, 0.4) * u_pi_top4.w;
    col += tint * 0.04;
    return col;
}

void main() {
    // Shake offset applied in screen space.
    vec2 uv = v_uv + u_shake;

    vec2 warped = uv;
    warped.x += sin(uv.y * 10.0 + u_time * 1.3) * u_c3_warp;
    warped.y += cos(uv.x *  8.0 + u_time * 0.9) * u_c3_warp;

#if ABERRATION
    vec2 dir = normalize(warped - 0.5 + 1e-6);
    float r_off = u_c1_aberration;
    vec2 wr = screenToWorld(warped + dir * r_off);
    vec2 wg = screenToWorld(warped);
    vec2 wb = screenToWorld(warped - dir * r_off);
    float r = sampleScene(wr).r;
    float g = sampleScene(wg).g;
    float b = sampleScene(wb).b;
    vec3 col = vec3(r, g, b);
#else
    vec3 col = sampleScene(screenToWorld(warped));
#endif

    float n = hash21(warped * u_resolution + u_time * 60.0);
    col += (n - 0.5) * u_c2_grain;
    col += pow(max(col, 0.0), vec3(2.0)) * u_c4_bloom * 0.25;

#if EDGE_MARKERS
    // Off-camera enemy markers: at Flow's 1.2x zoom part of the arena sits
    // outside the frustum, so blips ride the screen edge pointing at threats.
    for (int i = 0; i < 16; ++i) {
        if (i >= u_enemy_count) break;
        vec2 euv = worldToUv(u_enemies[i]);
        if (euv.x > 0.02 && euv.x < 0.98 && euv.y > 0.02 && euv.y < 0.98) continue;
        vec2 clamped = clamp(euv, vec2(0.022), vec2(0.978));
        vec2 d = (uv - clamped) * vec2(u_resolution.x / u_resolution.y, 1.0);
        float m = smoothstep(0.016, 0.004, length(d));
        col += vec3(1.0, 0.3, 0.6) * m * 0.85;
    }
#endif

    // Hit flash: saturate toward red and lift exposure.
    col = mix(col, vec3(1.0, 0.2, 0.2), u_hit_flash * 0.5);

    // Low-HP vignette (web addition): a red pulse creeping in from the edges,
    // beating in time with the heartbeat cue in the audio layer.
    if (u_low_hp > 0.0) {
        float vig = smoothstep(0.25, 0.85, length(v_uv - 0.5) * 1.35);
        float beat = 0.62 + 0.38 * sin(u_time * 6.5);
        col = mix(col, vec3(0.55, 0.02, 0.06), vig * u_low_hp * beat * 0.75);
    }

    fragColor = vec4(col, 1.0);
}`;

export const QUALITY_LEVELS = ["low", "medium", "high", "ultra"];

const QUALITY_DEFS = {
  low: { TRAIL_STEPS: 8, ABERRATION: 0, EDGE_MARKERS: 0, scale: 0.55 },
  medium: { TRAIL_STEPS: 16, ABERRATION: 0, EDGE_MARKERS: 1, scale: 0.75 },
  high: { TRAIL_STEPS: 32, ABERRATION: 1, EDGE_MARKERS: 1, scale: 1.0 },
  ultra: { TRAIL_STEPS: 32, ABERRATION: 1, EDGE_MARKERS: 1, scale: 1.0 },
};

const UNIFORM_NAMES = [
  "u_time", "u_resolution", "u_player", "u_enemies", "u_enemy_radii",
  "u_enemy_count", "u_world_half", "u_cam_zoom",
  "u_c1_aberration", "u_c2_grain", "u_c3_warp", "u_c4_bloom", "u_pi_top4",
  "u_boss_active", "u_boss_pos", "u_boss_radius", "u_boss_hp_frac",
  "u_shake", "u_player_flash", "u_hit_flash", "u_dash_phase", "u_dash_dir",
  "u_dash_trail", "u_dash_trail_a", "u_dash_trail_count",
  "u_pickups", "u_pickup_kinds", "u_pickup_count",
  "u_dash_ready", "u_dash_ready_pop", "u_low_hp", "u_shield", "u_fx",
];

export class Renderer {
  constructor(canvas) {
    this.canvas = canvas;
    const opts = { alpha: false, antialias: false, depth: false, powerPreference: "high-performance" };
    this.gl = canvas.getContext("webgl2", opts);
    if (!this.gl) throw new Error("WebGL2 unavailable");

    const gl = this.gl;
    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);
    const vbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

    // Scratch buffers for the uniform arrays, allocated once.
    this._enemyPos = new Float32Array(C.MAX_ENEMIES * 2);
    this._enemyRad = new Float32Array(C.MAX_ENEMIES);
    this._trailPos = new Float32Array(C.DASH_TRAIL_MAX * 2);
    this._trailA = new Float32Array(C.DASH_TRAIL_MAX);
    this._pickupPos = new Float32Array(C.PICKUP_SHADER_CAP * 2);
    this._pickupKind = new Int32Array(C.PICKUP_SHADER_CAP);

    this.quality = "high";
    this.renderScale = 1.0;
    this.autoScale = true;
    this._program = null;
    this.setQuality("high");

    // Adaptive resolution: a slow EMA of frame cost, nudged once a second so
    // it never oscillates visibly.
    this._frameMs = 16.7;
    this._adaptCooldown = 0;
    this._dpr = Math.min(window.devicePixelRatio || 1, 2);
  }

  setQuality(level) {
    if (!QUALITY_LEVELS.includes(level)) level = "high";
    this.quality = level;
    const defs = QUALITY_DEFS[level];
    const header = Object.entries(defs)
      .filter(([k]) => k !== "scale")
      .map(([k, v]) => `#define ${k} ${v}`)
      .join("\n");
    const frag = FRAG.replace("precision highp float;", `precision highp float;\n${header}`);

    const gl = this.gl;
    const prog = compileProgram(gl, VERT, frag);
    if (this._program) gl.deleteProgram(this._program);
    this._program = prog;
    gl.useProgram(prog);
    this.U = {};
    for (const n of UNIFORM_NAMES) this.U[n] = gl.getUniformLocation(prog, n);
    this.baseScale = defs.scale;
    this.renderScale = defs.scale;
    this.resize();
  }

  setAutoScale(on) {
    this.autoScale = on;
    if (!on) this.renderScale = this.baseScale;
  }

  resize() {
    const gl = this.gl;
    const cssW = this.canvas.clientWidth || window.innerWidth;
    const cssH = this.canvas.clientHeight || window.innerHeight;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    this._dpr = dpr;
    const w = Math.max(320, Math.round(cssW * dpr * this.renderScale));
    const h = Math.max(240, Math.round(cssH * dpr * this.renderScale));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
    gl.viewport(0, 0, w, h);
  }

  /** Feed measured frame cost; silently trims resolution to protect 60fps. */
  observeFrame(ms) {
    this._frameMs += (ms - this._frameMs) * 0.05;
    if (!this.autoScale) return;
    this._adaptCooldown -= ms;
    if (this._adaptCooldown > 0) return;
    this._adaptCooldown = 1000;
    const min = 0.4;
    if (this._frameMs > 21 && this.renderScale > min) {
      this.renderScale = Math.max(min, this.renderScale - 0.12);
      this.resize();
    } else if (this._frameMs < 12.5 && this.renderScale < this.baseScale) {
      this.renderScale = Math.min(this.baseScale, this.renderScale + 0.08);
      this.resize();
    }
  }

  render(s) {
    const gl = this.gl;
    const U = this.U;
    gl.useProgram(this._program);
    gl.bindVertexArray(this.vao);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.uniform1f(U.u_time, s.time);
    gl.uniform2f(U.u_resolution, this.canvas.width, this.canvas.height);
    gl.uniform2f(U.u_player, s.player.pos[0], s.player.pos[1]);

    const nE = Math.min(s.enemies.length, C.MAX_ENEMIES);
    this._enemyPos.fill(0);
    this._enemyRad.fill(0);
    for (let i = 0; i < nE; i++) {
      this._enemyPos[2 * i] = s.enemies[i].pos[0];
      this._enemyPos[2 * i + 1] = s.enemies[i].pos[1];
      this._enemyRad[i] = s.enemies[i].radius;
    }
    gl.uniform2fv(U.u_enemies, this._enemyPos);
    gl.uniform1fv(U.u_enemy_radii, this._enemyRad);
    gl.uniform1i(U.u_enemy_count, nE);

    gl.uniform1f(U.u_world_half, C.WORLD_HALF);
    gl.uniform1f(U.u_cam_zoom, Math.max(s.E[3], 0.3));

    const fx = s.fx;
    gl.uniform1f(U.u_c1_aberration, Math.max(s.c[0], 0) * fx);
    gl.uniform1f(U.u_c2_grain, Math.max(s.c[1], 0) * fx);
    gl.uniform1f(U.u_c3_warp, Math.max(s.c[2], 0) * fx);
    gl.uniform1f(U.u_c4_bloom, Math.max(s.c[3], 0));
    gl.uniform4f(U.u_pi_top4, s.pi[0], s.pi[1], s.pi[2], s.pi[3]);

    if (s.boss) {
      gl.uniform1f(U.u_boss_active, 1);
      gl.uniform2f(U.u_boss_pos, s.boss.pos[0], s.boss.pos[1]);
      gl.uniform1f(U.u_boss_radius, s.boss.radius);
      gl.uniform1f(U.u_boss_hp_frac, Math.max(0, s.boss.hp / s.boss.maxHp));
    } else {
      gl.uniform1f(U.u_boss_active, 0);
      gl.uniform2f(U.u_boss_pos, 0, 0);
      gl.uniform1f(U.u_boss_radius, 0);
      gl.uniform1f(U.u_boss_hp_frac, 0);
    }

    // Screen shake scales with chromatic aberration + damage flash.
    const shakeAmp = (C.SHAKE_GAIN * Math.max(s.c[0], 0) + 0.02 * s.player.hitFlash) * fx;
    gl.uniform2f(U.u_shake,
      (Math.random() - 0.5) * 2 * shakeAmp,
      (Math.random() - 0.5) * 2 * shakeAmp);
    gl.uniform1f(U.u_player_flash, Math.min(1, s.player.dashing * 5));
    gl.uniform1f(U.u_hit_flash, s.player.hitFlash);
    const dashPhase = C.DASH_INVULN > 0 ? s.player.dashing / C.DASH_INVULN : 0;
    gl.uniform1f(U.u_dash_phase, Math.max(0, Math.min(1, dashPhase)));
    gl.uniform2f(U.u_dash_dir, s.player.dashDir[0], s.player.dashDir[1]);

    const trail = s.player.dashTrail;
    const nT = Math.min(trail.length, C.DASH_TRAIL_MAX);
    this._trailPos.fill(0);
    this._trailA.fill(0);
    for (let i = 0; i < nT; i++) {
      this._trailPos[2 * i] = trail[i][0];
      this._trailPos[2 * i + 1] = trail[i][1];
      this._trailA[i] = Math.min(1, Math.max(0, trail[i][2] - s.simTime) / C.DASH_TRAIL_TTL);
    }
    gl.uniform2fv(U.u_dash_trail, this._trailPos);
    gl.uniform1fv(U.u_dash_trail_a, this._trailA);
    gl.uniform1i(U.u_dash_trail_count, nT);

    const nP = Math.min(s.pickups.length, C.PICKUP_SHADER_CAP);
    this._pickupPos.fill(0);
    this._pickupKind.fill(0);
    for (let i = 0; i < nP; i++) {
      this._pickupPos[2 * i] = s.pickups[i].pos[0];
      this._pickupPos[2 * i + 1] = s.pickups[i].pos[1];
      this._pickupKind[i] = s.pickupCodes[s.pickups[i].kind] ?? 0;
    }
    gl.uniform2fv(U.u_pickups, this._pickupPos);
    gl.uniform1iv(U.u_pickup_kinds, this._pickupKind);
    gl.uniform1i(U.u_pickup_count, nP);

    gl.uniform1f(U.u_dash_ready, s.dashReady);
    gl.uniform1f(U.u_dash_ready_pop, s.dashReadyPop);
    gl.uniform1f(U.u_low_hp, s.lowHp * fx);
    gl.uniform1f(U.u_shield, s.shield);
    gl.uniform1f(U.u_fx, fx);

    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }
}

function compileProgram(gl, vsSrc, fsSrc) {
  const vs = compileShader(gl, gl.VERTEX_SHADER, vsSrc);
  const fs = compileShader(gl, gl.FRAGMENT_SHADER, fsSrc);
  const prog = gl.createProgram();
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    throw new Error("Shader link failed: " + gl.getProgramInfoLog(prog));
  }
  gl.deleteShader(vs);
  gl.deleteShader(fs);
  return prog;
}

function compileShader(gl, type, src) {
  const sh = gl.createShader(type);
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    throw new Error("Shader compile failed: " + gl.getShaderInfoLog(sh));
  }
  return sh;
}
