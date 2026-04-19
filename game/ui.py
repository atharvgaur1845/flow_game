"""HUD overlay: render text and bars on a pygame Surface, then upload
that Surface to a GL texture and composite it over the scene.

This approach keeps all pygame 2D rendering off the display window
(which is in OpenGL mode and cannot be blit into directly) while still
letting us use pygame.font for crisp text.
"""
from __future__ import annotations

import ctypes

import numpy as np
import pygame
from OpenGL.GL import (
    GL_ARRAY_BUFFER, GL_BLEND, GL_CLAMP_TO_EDGE, GL_COLOR_BUFFER_BIT,
    GL_FALSE, GL_FLOAT, GL_FRAGMENT_SHADER, GL_LINEAR, GL_ONE_MINUS_SRC_ALPHA,
    GL_RGBA, GL_SRC_ALPHA, GL_STATIC_DRAW, GL_TEXTURE0, GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER, GL_TEXTURE_MIN_FILTER, GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T, GL_TRIANGLE_STRIP, GL_UNSIGNED_BYTE, GL_VERTEX_SHADER,
    glActiveTexture, glBindBuffer, glBindTexture, glBindVertexArray,
    glBlendFunc, glBufferData, glDisable, glDrawArrays, glEnable,
    glEnableVertexAttribArray, glGenBuffers, glGenTextures, glGenVertexArrays,
    glGetUniformLocation, glTexImage2D, glTexParameteri, glUniform1i,
    glUseProgram, glVertexAttribPointer,
)
from OpenGL.GL.shaders import compileProgram, compileShader


_VS = """
#version 330 core
layout(location=0) in vec2 a_pos;
out vec2 v_uv;
void main() {
    v_uv = vec2(a_pos.x * 0.5 + 0.5, 1.0 - (a_pos.y * 0.5 + 0.5));
    gl_Position = vec4(a_pos, 0.0, 1.0);
}
"""

_FS = """
#version 330 core
in  vec2 v_uv;
out vec4 fragColor;
uniform sampler2D u_tex;
void main() { fragColor = texture(u_tex, v_uv); }
"""


STATE_NAMES = ("AROUSAL", "TACTICAL", "OVERLOAD", "FLOW", "APATHY")
STATE_COLORS = (
    (255, 120, 120),  # arousal  red
    (120, 180, 255),  # tactical blue
    (200, 120, 255),  # overload purple
    (120, 255, 200),  # flow     teal
    (170, 170, 170),  # apathy   grey
)


class UIOverlay:
    def __init__(self, width: int, height: int) -> None:
        self.w, self.h = width, height
        self.prog = compileProgram(
            compileShader(_VS, GL_VERTEX_SHADER),
            compileShader(_FS, GL_FRAGMENT_SHADER),
        )
        quad = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype=np.float32)
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, quad.nbytes, quad, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))

        self.tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

        self._loc_tex = glGetUniformLocation(self.prog, "u_tex")
        # Fonts scale with the smaller of width/height to stay readable on 4K.
        scale = max(1.0, min(width, height) / 800.0)
        self.font_small = pygame.font.SysFont("monospace", int(16 * scale), bold=True)
        self.font       = pygame.font.SysFont("monospace", int(20 * scale), bold=True)
        self.font_big   = pygame.font.SysFont("monospace", int(40 * scale), bold=True)
        self.surf = pygame.Surface((width, height), pygame.SRCALPHA)

    # --- primitive helpers ------------------------------------------------
    def _text(self, txt, x, y, color=(230, 240, 255), big=False, small=False):
        f = self.font_big if big else (self.font_small if small else self.font)
        img = f.render(txt, True, color)
        self.surf.blit(img, (int(x), int(y)))
        return img.get_width()

    def _bar(self, x, y, w, h, fraction, fg, bg=(32, 32, 48), border=(210, 215, 230)):
        fraction = max(0.0, min(1.0, fraction))
        pygame.draw.rect(self.surf, bg, (int(x), int(y), int(w), int(h)))
        pygame.draw.rect(self.surf, fg,
                         (int(x), int(y), int(w * fraction), int(h)))
        pygame.draw.rect(self.surf, border,
                         (int(x), int(y), int(w), int(h)), 1)

    # --- screens ----------------------------------------------------------
    def render_menu(self, best_score: int) -> None:
        self.surf.fill((0, 0, 0, 0))
        cx = self.w // 2
        self._text("FLOW — NEON ARENA", cx - 280, self.h // 2 - 160, big=True)
        self._text("An adaptive roguelite driven by your own movement.",
                   cx - 280, self.h // 2 - 90, (200, 210, 230))
        lines = [
            "WASD   move",
            "SPACE  dash  (brief i-frames, damages boss)",
            "1/2/3  pick upgrade in shop",
            "F1     debug telemetry",
            "ESC    quit",
        ]
        for i, ln in enumerate(lines):
            self._text(ln, cx - 200, self.h // 2 - 30 + i * 28, (190, 200, 220))
        self._text("Press ENTER to start",
                   cx - 140, self.h // 2 + 140, (180, 230, 180))
        if best_score > 0:
            self._text(f"Best: {best_score}", cx - 50, self.h // 2 + 180,
                       (220, 200, 140))

    def render_playing(self, player, run, pi, c_vec, goal_text: str,
                       *, boss=None, now_t: float = 0.0,
                       pickup_msg=("", 0.0)) -> None:
        self.surf.fill((0, 0, 0, 0))
        # HP
        self._text("HP", 24, 22)
        self._bar(70, 22, 320, 22, player.hp / max(player.max_hp, 1),
                  (220, 70, 90))
        self._text(f"{max(int(player.hp),0)} / {int(player.max_hp)}", 400, 22)
        # Dash
        dash_ready = player.dash_cd <= 0.0
        dash_frac = (1.0 - player.dash_cd / max(player.dash_cd_max, 1e-3)
                     if not dash_ready else 1.0)
        self._text("DASH", 24, 54, small=True)
        self._bar(70, 56, 160, 10, dash_frac,
                  (120, 220, 255) if dash_ready else (90, 140, 180))
        if dash_ready:
            self._text("READY", 240, 54, (140, 240, 200), small=True)

        # Score / multiplier / room
        mult = run.score_multiplier(pi)
        self._text(f"SCORE {int(run.score):>7}", 24, 80)
        self._text(f"x{mult:4.2f}", 260, 80, (180, 230, 180))
        self._text(f"ROOM {run.room}", 370, 80, (255, 210, 120))
        self._text(f"KILLS {run.total_kills}", 490, 80, (220, 220, 240))
        self._text(goal_text, 24, 108, (180, 190, 210), small=True)

        # State bars on the right.
        bar_x = self.w - 300
        self._text("INFERRED STATE (π)", bar_x, 22, (200, 210, 230), small=True)
        for i, name in enumerate(STATE_NAMES):
            y = 44 + i * 24
            self._text(name, bar_x, y, STATE_COLORS[i], small=True)
            self._bar(bar_x + 90, y + 4, 180, 14, float(pi[i]), STATE_COLORS[i])

        # Visual latents (c) lower right.
        cy = self.h - 110
        labels = ("ABER", "GRAIN", "WARP", "BLOOM")
        scales = (0.05, 0.5, 0.05, 1.5)
        self._text("VISUAL LATENTS (c)", bar_x, cy - 20, (200, 210, 230), small=True)
        for i, lbl in enumerate(labels):
            val = max(0.0, float(c_vec[i]))
            y = cy + i * 18
            self._text(lbl, bar_x, y, small=True)
            self._bar(bar_x + 90, y + 4, 180, 10, val / scales[i], (180, 180, 255))

        # Boss HP bar at the top-center.
        if boss is not None:
            w = self.w // 2
            x = (self.w - w) // 2
            self._text(f"BOSS — {boss.phase_label}", x, 14, (255, 160, 200), small=True)
            self._bar(x, 32, w, 16, boss.hp / max(boss.max_hp, 1), (255, 80, 140))

        # Active buffs (left column, below the room goal).
        if getattr(player, "buffs", None):
            self._text("BUFFS", 24, 138, (200, 230, 200), small=True)
            BUFF_COLOR = {
                "heal":         (90, 230, 130),
                "dash_boost":   (90, 230, 255),
                "speed_boost":  (255, 170, 90),
                "shield":       (130, 160, 255),
                "max_hp":       (255, 230, 100),
            }
            BUFF_LABEL = {
                "heal":         "HEAL",
                "dash_boost":   "BIG DASH",
                "speed_boost":  "HASTE",
                "shield":       "SHIELD",
                "max_hp":       "+MAX HP",
            }
            BUFF_DUR = {
                "dash_boost": 8.0, "speed_boost": 8.0,
                "shield": 5.0, "max_hp": 12.0,
            }
            row = 0
            for kind, exp in player.buffs.items():
                remain = max(0.0, exp - now_t)
                col = BUFF_COLOR.get(kind, (200, 200, 200))
                lbl = BUFF_LABEL.get(kind, kind.upper())
                full = BUFF_DUR.get(kind, 8.0)
                y = 158 + row * 22
                self._text(f"{lbl}  {remain:4.1f}s", 24, y, col, small=True)
                self._bar(180, y + 4, 120, 10, remain / full, col)
                row += 1

        # Pickup banner (centered top, fades over its TTL).
        msg, exp = pickup_msg
        if msg and now_t < exp:
            self._text(f"+ {msg}", self.w // 2 - 80, 70,
                       (255, 230, 140), big=True)

    def render_shop(self, choices, upgrade_list, run) -> None:
        self.surf.fill((0, 0, 0, 0))
        cx = self.w // 2
        # Dim background wash.
        pygame.draw.rect(self.surf, (0, 0, 0, 160),
                         (0, 0, self.w, self.h))
        self._text("SHOP — press 1, 2, or 3", cx - 240, 120, big=True)
        for i, idx in enumerate(choices):
            name, desc = upgrade_list[idx]
            y = 240 + i * 110
            self._text(f"{i + 1}. {name}", cx - 260, y, (255, 230, 140))
            self._text(desc, cx - 240, y + 40, (200, 210, 230), small=True)
        self._text(f"Room {run.room}   Score {int(run.score)}",
                   24, self.h - 44, (180, 190, 210), small=True)

    def render_game_over(self, run) -> None:
        self.surf.fill((0, 0, 0, 0))
        # Dim background wash.
        pygame.draw.rect(self.surf, (0, 0, 0, 180), (0, 0, self.w, self.h))
        cx = self.w // 2
        self._text("GAME OVER", cx - 160, self.h // 2 - 180,
                   (255, 120, 120), big=True)
        lines = [
            f"Score           {int(run.score)}",
            f"Time survived   {run.time_survived:6.1f} s",
            f"Max flow (π_4)  {run.max_flow:6.2f}",
            f"Rooms cleared   {run.rooms_cleared}",
            f"Total kills     {run.total_kills}",
        ]
        for i, line in enumerate(lines):
            self._text(line, cx - 200, self.h // 2 - 60 + i * 36)
        self._text("Press ENTER for menu · ESC to quit",
                   cx - 260, self.h // 2 + 160, (180, 230, 180))

    # --- GL upload + draw -------------------------------------------------
    def draw(self) -> None:
        raw = pygame.image.tostring(self.surf, "RGBA", False)
        glBindTexture(GL_TEXTURE_2D, self.tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.w, self.h, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, raw)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glUseProgram(self.prog)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.tex)
        glUniform1i(self._loc_tex, 0)
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glDisable(GL_BLEND)
