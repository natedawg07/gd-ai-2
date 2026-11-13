"""A tiny Geometry Dash-like simulator for testing the agent without the game.

It speaks the same backend protocol as the mod (level message + state
dicts), and uses the same action meaning and the same 2 updates per step at
1/60 s, so GDEnv, the observation code, and the DQN are all exercised
exactly as they are against the real game.

Layout: cube section with single/double/triple spikes -> ship portal -> ship
corridor with hazard pillars -> cube portal -> a few more spikes -> end.
"""

import numpy as np

from obs import BLOCK, FLOOR_Y

DT = 1.0 / 60.0
FRAME_SKIP = 2
SPEED = 311.58             # units/s at normal speed
GRAVITY = 2600.0           # cube gravity, units/s^2
JUMP_V = 590.0             # cube jump velocity -> ~2.2 block jump
SHIP_ACCEL = 1500.0
SHIP_VMAX = 400.0
FLOOR_TOP = FLOOR_Y - BLOCK / 2      # 90
CEIL_Y = FLOOR_Y + 8 * BLOCK          # ship's highest centre position
CEIL_TOP = CEIL_Y + BLOCK / 2

HAZARD, SHIP_PORTAL, CUBE_PORTAL = 2, 5, 6


def spike(x):
    """GD spikes have a hitbox much smaller than their 30x30 sprite."""
    return [HAZARD, x - 5.0, FLOOR_TOP + 2.0, 10.0, 18.0]


def build_level():
    objs = []
    for x, n in [(600, 1), (900, 2), (1250, 1), (1550, 2), (1900, 3), (2250, 1)]:
        for k in range(n):
            objs.append(spike(x + k * BLOCK))
    objs.append([SHIP_PORTAL, 2500 - 17, FLOOR_Y - 45, 34, 90])
    gap_centres = [180, 240, 300, 240, 180, 150]
    for i, c in enumerate(gap_centres):
        x = 2750 + i * 240
        objs.append([HAZARD, x, FLOOR_TOP, BLOCK, (c - 60) - FLOOR_TOP])   # floor pillar
        objs.append([HAZARD, x, c + 60, BLOCK, CEIL_TOP - (c + 60)])       # ceiling pillar
    objs.append([CUBE_PORTAL, 4250 - 17, FLOOR_Y - 45, 34, 90])
    for x, n in [(4550, 1), (4800, 2)]:
        for k in range(n):
            objs.append(spike(x + k * BLOCK))
    return objs, 5100.0


class FakeGame:
    def __init__(self, seed=0):
        self.objects, self.length = build_level()
        arr = np.asarray(self.objects, dtype=np.float64)
        self.types = arr[:, 0].astype(int)
        self.hx0, self.hy0 = arr[:, 1], arr[:, 2]
        self.hx1, self.hy1 = arr[:, 1] + arr[:, 3], arr[:, 2] + arr[:, 4]
        self.is_hazard = self.types == HAZARD
        self.ship_from = min(o[1] for o in self.objects if o[0] == SHIP_PORTAL)
        self.ship_to = min(o[1] for o in self.objects if o[0] == CUBE_PORTAL)
        self.sent_level = False
        self._reset_player()

    def _reset_player(self):
        self.x, self.y, self.vy = 0.0, FLOOR_Y, 0.0
        self.held = False
        self.dead = False
        self.win = False

    @property
    def mode(self):
        return 1 if self.ship_from <= self.x < self.ship_to else 0

    @property
    def on_ground(self):
        return self.y <= FLOOR_Y + 1e-6 and self.vy <= 0

    def _substep(self, down):
        self.held = down
        if self.mode == 0:
            if down and self.on_ground:
                self.vy = JUMP_V
            self.vy -= GRAVITY * DT
        else:
            self.vy += (SHIP_ACCEL if down else -SHIP_ACCEL) * DT
            self.vy = float(np.clip(self.vy, -SHIP_VMAX, SHIP_VMAX))
        self.y += self.vy * DT
        if self.y <= FLOOR_Y:
            self.y, self.vy = FLOOR_Y, 0.0
        if self.mode == 1 and self.y >= CEIL_Y:
            self.y, self.vy = CEIL_Y, 0.0
        self.x += SPEED * DT

        px0, px1 = self.x - BLOCK / 2, self.x + BLOCK / 2
        py0, py1 = self.y - BLOCK / 2, self.y + BLOCK / 2
        hit = (self.is_hazard & (self.hx0 < px1) & (self.hx1 > px0)
               & (self.hy0 < py1) & (self.hy1 > py0))
        if hit.any():
            self.dead = True
        if self.x >= self.length:
            self.win = True

    def _state(self):
        return {"type": "state", "x": self.x, "y": self.y, "vy": self.vy / 60.0,
                "ground": int(self.on_ground), "mode": self.mode, "upside": 0,
                "held": int(self.held), "speed": 0.9,
                "pct": min(100.0, 100.0 * self.x / self.length),
                "dead": int(self.dead), "win": int(self.win)}

    # ── backend interface ──────────────────────────────────────────────────
    def reset(self):
        self._reset_player()
        level = None
        if not self.sent_level:
            level = {"type": "level", "length": self.length, "objects": self.objects}
            self.sent_level = True
        return level, self._state()

    def step(self, action):
        if self.dead or self.win:
            return self._state()
        for i in range(FRAME_SKIP):
            down = action == 2 or (action == 1 and i == 0)
            if action == 1 and i == 0 and self.held:
                self.held = False   # a click needs a fresh press
            self._substep(down)
            if self.dead or self.win:
                break
        return self._state()

    def close(self):
        pass
