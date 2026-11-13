"""Turn raw game state into what the agent sees.

Two parts:
  grid  (3, 20, 32) float32 in {0,1}: what's around the player
        channel 0 = solid (blocks, slopes), 1 = hazard (spikes, saws),
        2 = interactive (pads, rings, portals)
        x: 1 block behind to 15 blocks ahead, y: 5 blocks below to 5 above,
        in half-block cells. Row 0 is the bottom, column 0 is behind the player.
  vec   (14,) float32: the player's own state.

There is no absolute x position: the agent has to react to what's ahead
instead of memorising "jump at x=1015".
"""

import numpy as np

BLOCK = 30.0                 # one GD grid block in game units
CELL = BLOCK / 2             # grid resolution
BEHIND_BLOCKS = 1
AHEAD_BLOCKS = 15
HALF_HEIGHT_BLOCKS = 5
GRID_W = int((BEHIND_BLOCKS + AHEAD_BLOCKS) * BLOCK / CELL)   # 32
GRID_H = int(2 * HALF_HEIGHT_BLOCKS * BLOCK / CELL)           # 20
N_CHANNELS = 3
GRID_SHAPE = (N_CHANNELS, GRID_H, GRID_W)

FLOOR_Y = 105.0              # player centre y when standing on the floor
N_MODES = 8                  # cube, ship, ball, ufo, wave, robot, spider, swing
VEC_DIM = 3 + N_MODES + 3

# GameObjectType ids (Geode Enums.hpp) -> grid channel
SOLID_TYPES = {0, 21, 25}             # Solid, Breakable, Slope
HAZARD_TYPES = {2, 47}                # Hazard, AnimatedHazard
CH_SOLID, CH_HAZARD, CH_INTERACT = 0, 1, 2


def channel_for(obj_type: int) -> int:
    if obj_type in SOLID_TYPES:
        return CH_SOLID
    if obj_type in HAZARD_TYPES:
        return CH_HAZARD
    return CH_INTERACT


class LevelGeometry:
    """Static level hitboxes, sorted by left edge for fast window queries."""

    def __init__(self, objects, length):
        self.length = float(length)
        arr = np.asarray(objects, dtype=np.float32).reshape(-1, 5)
        order = np.argsort(arr[:, 1], kind="stable")
        arr = arr[order]
        self.channel = np.array([channel_for(int(t)) for t in arr[:, 0]], dtype=np.int64)
        self.x0 = arr[:, 1]
        self.y0 = arr[:, 2]
        self.x1 = arr[:, 1] + arr[:, 3]
        self.y1 = arr[:, 2] + arr[:, 4]
        widths = arr[:, 3]
        self.max_w = float(widths.max()) if len(widths) else 0.0

    def __len__(self):
        return len(self.x0)

    @classmethod
    def from_message(cls, msg):
        return cls(msg.get("objects", []), msg["length"])

    def counts(self):
        return {name: int((self.channel == ch).sum())
                for name, ch in (("solid", CH_SOLID), ("hazard", CH_HAZARD),
                                 ("interactive", CH_INTERACT))}


def build_grid(geom: LevelGeometry, px: float, py: float) -> np.ndarray:
    grid = np.zeros(GRID_SHAPE, dtype=np.float32)
    if geom is None or len(geom) == 0:
        return grid
    left = px - BEHIND_BLOCKS * BLOCK
    right = px + AHEAD_BLOCKS * BLOCK
    bottom = py - HALF_HEIGHT_BLOCKS * BLOCK
    top = py + HALF_HEIGHT_BLOCKS * BLOCK

    # Candidates: left edge before the window's right side, and not so far
    # left that even the widest object couldn't reach into the window.
    lo = np.searchsorted(geom.x0, left - geom.max_w, side="left")
    hi = np.searchsorted(geom.x0, right, side="left")
    if hi <= lo:
        return grid
    idx = np.arange(lo, hi)
    m = (geom.x1[idx] > left) & (geom.y1[idx] > bottom) & (geom.y0[idx] < top)
    idx = idx[m]

    for i in idx:
        cx0 = max(0, int(np.floor((geom.x0[i] - left) / CELL)))
        cx1 = min(GRID_W, int(np.ceil((geom.x1[i] - left) / CELL)))
        cy0 = max(0, int(np.floor((geom.y0[i] - bottom) / CELL)))
        cy1 = min(GRID_H, int(np.ceil((geom.y1[i] - bottom) / CELL)))
        if cx1 > cx0 and cy1 > cy0:
            grid[geom.channel[i], cy0:cy1, cx0:cx1] = 1.0
    return grid


def player_vec(state) -> np.ndarray:
    v = np.zeros(VEC_DIM, dtype=np.float32)
    v[0] = (state["y"] - FLOOR_Y) / BLOCK / HALF_HEIGHT_BLOCKS   # height in ~[-1, 2]
    v[1] = state["vy"] / 10.0
    v[2] = float(state["ground"])
    mode = int(state.get("mode", 0))
    if 0 <= mode < N_MODES:
        v[3 + mode] = 1.0
    v[3 + N_MODES] = float(state.get("upside", 0))
    v[4 + N_MODES] = float(state.get("held", 0))
    v[5 + N_MODES] = float(state.get("speed", 0.9))
    return v


def make_obs(geom, state):
    return {"grid": build_grid(geom, state["x"], state["y"]),
            "vec": player_vec(state)}
