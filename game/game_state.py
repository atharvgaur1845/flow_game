"""Top-level game state machine."""
from __future__ import annotations


class GameState:
    MENU       = 0
    PLAYING    = 1
    SHOP       = 2
    BOSS       = 3
    GAME_OVER  = 4

    NAMES = {
        MENU: "MENU", PLAYING: "PLAYING", SHOP: "SHOP",
        BOSS: "BOSS", GAME_OVER: "GAME_OVER",
    }
