import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from obs import (BLOCK, CELL, CH_HAZARD, CH_INTERACT, CH_SOLID, FLOOR_Y, GRID_SHAPE,
                 HALF_HEIGHT_BLOCKS, LevelGeometry, build_grid, channel_for, player_vec)


def test_channels():
    assert channel_for(0) == CH_SOLID
    assert channel_for(25) == CH_SOLID
    assert channel_for(2) == CH_HAZARD
    assert channel_for(47) == CH_HAZARD
    assert channel_for(5) == CH_INTERACT   # ship portal
    assert channel_for(11) == CH_INTERACT  # yellow orb


def test_spike_lands_in_expected_cell():
    # Spike hitbox 10x18, left edge 3 blocks ahead of the player, on the floor.
    px, py = 1000.0, FLOOR_Y
    spike_x = px + 3 * BLOCK
    geom = LevelGeometry([[2, spike_x, 92.0, 10.0, 18.0]], length=5000)
    g = build_grid(geom, px, py)
    assert g.shape == GRID_SHAPE
    left = px - BLOCK
    bottom = py - HALF_HEIGHT_BLOCKS * BLOCK
    col = int((spike_x - left) // CELL)          # 4 blocks from the window's left = col 8
    row = int((92.0 - bottom) // CELL)
    assert col == 8
    assert g[CH_HAZARD, row, col] == 1.0
    assert g[CH_SOLID].sum() == 0 and g[CH_INTERACT].sum() == 0
    # Only the cells the hitbox covers are set.
    assert g[CH_HAZARD].sum() == 1 * 2   # 1 column wide (10 < 15), 2 rows tall (18 > 15)


def test_objects_outside_window_ignored():
    px, py = 1000.0, FLOOR_Y
    geom = LevelGeometry([
        [2, px - 200, 92, 10, 18],          # far behind
        [2, px + 20 * BLOCK, 92, 10, 18],   # beyond lookahead
        [0, px + 2 * BLOCK, py + 10 * BLOCK, 30, 30],  # far above
    ], length=5000)
    assert build_grid(geom, px, py).sum() == 0


def test_wide_object_starting_behind_is_seen():
    px, py = 1000.0, FLOOR_Y
    # A long block that starts well behind the player but extends ahead.
    geom = LevelGeometry([[0, px - 300, 60, 600, 30],
                          [2, px + 50, 92, 10, 18]], length=5000)
    g = build_grid(geom, px, py)
    assert g[CH_SOLID].sum() > 0
    assert g[CH_HAZARD].sum() > 0


def test_player_vec_mode_onehot():
    v = player_vec({"y": FLOOR_Y, "vy": 0.0, "ground": 1, "mode": 1, "upside": 0,
                    "held": 1, "speed": 0.9})
    assert v[0] == 0.0 and v[2] == 1.0
    assert v[3 + 1] == 1.0 and v[3:11].sum() == 1.0
    assert v[12] == 1.0 and np.isclose(v[13], 0.9)
