/* Input: keyboard with rebindable actions, plus touch controls.
 *
 * A link that gets shared lands on phones as often as desktops, so the touch
 * path is a first-class citizen rather than an afterthought: a floating
 * thumbstick that origins wherever the thumb lands, and a dash button. */

export const ACTIONS = ["up", "down", "left", "right", "dash"];

export const DEFAULT_BINDS = {
  up: ["KeyW", "ArrowUp"],
  down: ["KeyS", "ArrowDown"],
  left: ["KeyA", "ArrowLeft"],
  right: ["KeyD", "ArrowRight"],
  dash: ["Space", "ShiftLeft"],
};

export const ACTION_LABELS = {
  up: "Move up", down: "Move down", left: "Move left", right: "Move right", dash: "Dash",
};

export class Input {
  constructor(store) {
    this.store = store;
    this.binds = normaliseBinds(store.settings.binds);
    this.down = new Set();
    this.touch = { active: false, dx: 0, dy: 0, dash: false, id: null, ox: 0, oy: 0 };
    this.onGameKey = null; // fired on keydown of any bound gameplay key (APM)
    this.capturing = null; // action name while rebinding
    this.onCapture = null;
    this._bound = false;
  }

  attach(target = window) {
    if (this._bound) return;
    this._bound = true;
    target.addEventListener("keydown", (e) => this._onKeyDown(e));
    target.addEventListener("keyup", (e) => this.down.delete(e.code));
    target.addEventListener("blur", () => this.down.clear());
    document.addEventListener("visibilitychange", () => this.down.clear());
  }

  _onKeyDown(e) {
    if (this.capturing) {
      e.preventDefault();
      if (e.code !== "Escape") this.rebind(this.capturing, e.code);
      const a = this.capturing;
      this.capturing = null;
      if (this.onCapture) this.onCapture(a);
      return;
    }
    if (e.repeat) return;
    this.down.add(e.code);
    // Space and the arrows scroll the page otherwise, which is instantly
    // fatal to a full-viewport game.
    if (this._isBound(e.code)) {
      e.preventDefault();
      if (this.onGameKey) this.onGameKey(e.code);
    }
  }

  _isBound(code) {
    for (const a of ACTIONS) if (this.binds[a].includes(code)) return true;
    return false;
  }

  isDown(action) {
    for (const code of this.binds[action]) if (this.down.has(code)) return true;
    return false;
  }

  rebind(action, code) {
    // A key can only drive one action; steal it from whoever had it.
    for (const a of ACTIONS) {
      this.binds[a] = this.binds[a].filter((c) => c !== code);
      if (this.binds[a].length === 0 && a !== action) {
        this.binds[a] = [...DEFAULT_BINDS[a]].filter((c) => c !== code);
      }
    }
    this.binds[action] = [code];
    this.store.setSetting("binds", this.binds);
  }

  resetBinds() {
    this.binds = structuredClone(DEFAULT_BINDS);
    this.store.setSetting("binds", this.binds);
  }

  bindLabel(action) {
    return this.binds[action].map(prettyCode).join(" / ");
  }

  /** Combined keyboard + touch movement vector, unnormalised. */
  moveVector() {
    let x = (this.isDown("right") ? 1 : 0) - (this.isDown("left") ? 1 : 0);
    let y = (this.isDown("up") ? 1 : 0) - (this.isDown("down") ? 1 : 0);
    if (this.touch.active && (this.touch.dx !== 0 || this.touch.dy !== 0)) {
      x = this.touch.dx;
      y = this.touch.dy;
    }
    return [x, y];
  }

  dashHeld() {
    return this.isDown("dash") || this.touch.dash;
  }

  // --- touch ---------------------------------------------------------------

  attachTouch(stickEl, knobEl, dashEl, surfaceEl) {
    const RADIUS = 56;

    const place = (x, y) => {
      stickEl.style.left = x + "px";
      stickEl.style.top = y + "px";
      stickEl.classList.add("active");
    };

    surfaceEl.addEventListener("touchstart", (e) => {
      for (const t of e.changedTouches) {
        // Right third of the screen is the dash pad; the rest is the stick.
        if (t.clientX > window.innerWidth * 0.66) continue;
        if (this.touch.id !== null) continue;
        this.touch.id = t.identifier;
        this.touch.active = true;
        this.touch.ox = t.clientX;
        this.touch.oy = t.clientY;
        place(t.clientX, t.clientY);
        knobEl.style.transform = "translate(-50%, -50%)";
      }
      e.preventDefault();
    }, { passive: false });

    surfaceEl.addEventListener("touchmove", (e) => {
      for (const t of e.changedTouches) {
        if (t.identifier !== this.touch.id) continue;
        let dx = t.clientX - this.touch.ox;
        let dy = t.clientY - this.touch.oy;
        const len = Math.hypot(dx, dy);
        const clamped = Math.min(len, RADIUS);
        const nx = len > 1e-3 ? (dx / len) * (clamped / RADIUS) : 0;
        const ny = len > 1e-3 ? (dy / len) * (clamped / RADIUS) : 0;
        this.touch.dx = nx;
        this.touch.dy = -ny; // screen y is inverted relative to world y
        knobEl.style.transform =
          `translate(calc(-50% + ${nx * RADIUS}px), calc(-50% + ${ny * RADIUS}px))`;
      }
      e.preventDefault();
    }, { passive: false });

    const end = (e) => {
      for (const t of e.changedTouches) {
        if (t.identifier !== this.touch.id) continue;
        this.touch.id = null;
        this.touch.active = false;
        this.touch.dx = this.touch.dy = 0;
        stickEl.classList.remove("active");
        knobEl.style.transform = "translate(-50%, -50%)";
      }
    };
    surfaceEl.addEventListener("touchend", end);
    surfaceEl.addEventListener("touchcancel", end);

    const dashOn = (e) => { this.touch.dash = true; if (this.onGameKey) this.onGameKey("Space"); e.preventDefault(); };
    const dashOff = (e) => { this.touch.dash = false; e.preventDefault(); };
    dashEl.addEventListener("touchstart", dashOn, { passive: false });
    dashEl.addEventListener("touchend", dashOff, { passive: false });
    dashEl.addEventListener("touchcancel", dashOff, { passive: false });
  }
}

function normaliseBinds(saved) {
  const out = structuredClone(DEFAULT_BINDS);
  if (!saved || typeof saved !== "object") return out;
  for (const a of ACTIONS) {
    if (Array.isArray(saved[a]) && saved[a].length) out[a] = saved[a].slice(0, 2);
  }
  return out;
}

const PRETTY = {
  Space: "SPACE",
  ArrowUp: "↑", ArrowDown: "↓", ArrowLeft: "←", ArrowRight: "→",
  ShiftLeft: "L-SHIFT", ShiftRight: "R-SHIFT",
  ControlLeft: "L-CTRL", ControlRight: "R-CTRL",
  AltLeft: "L-ALT", AltRight: "R-ALT",
};

export function prettyCode(code) {
  if (!code) return "—";
  if (PRETTY[code]) return PRETTY[code];
  if (code.startsWith("Key")) return code.slice(3);
  if (code.startsWith("Digit")) return code.slice(5);
  if (code.startsWith("Numpad")) return "NUM " + code.slice(6);
  return code.toUpperCase();
}

export const IS_TOUCH = (() => {
  return (("ontouchstart" in window) || navigator.maxTouchPoints > 0) &&
    window.matchMedia("(pointer: coarse)").matches;
})();
