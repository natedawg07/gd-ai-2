import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from live import LivePublisher, read_snapshot
from obs import GRID_SHAPE, VEC_DIM


class StubAgent:
    def q_values(self, obs):
        return np.array([0.5, 1.5, -2.0], dtype=np.float32)


def test_publish_and_read_roundtrip():
    path = os.path.join(tempfile.mkdtemp(), "live.npz")
    grid = np.zeros(GRID_SHAPE, dtype=np.float32)
    grid[1, 3, 7] = 1.0
    obs = {"grid": grid, "vec": np.arange(VEC_DIM, dtype=np.float32)}
    pub = LivePublisher(path, interval=0.0)
    pub.maybe_publish(StubAgent(), obs, 1, True, 0.42, ep=7, step=1234, greedy=False,
                      progress=12.5, best_greedy=30.0)
    s = read_snapshot(path)
    assert s is not None
    assert s["grid"].shape == GRID_SHAPE and s["grid"][1, 3, 7] == 1
    assert np.allclose(s["q"], [0.5, 1.5, -2.0])
    assert s["action"] == 1 and s["explored"] is True
    assert abs(s["eps"] - 0.42) < 1e-6
    assert s["ep"] == 7 and s["step"] == 1234 and s["greedy"] is False
    assert not os.path.exists(path + ".tmp")


def test_throttled():
    path = os.path.join(tempfile.mkdtemp(), "live.npz")
    pub = LivePublisher(path, interval=60.0)
    obs = {"grid": np.zeros(GRID_SHAPE), "vec": np.zeros(VEC_DIM)}
    pub.maybe_publish(StubAgent(), obs, 0, False, 0.1, ep=1)
    first = os.path.getmtime(path)
    pub.maybe_publish(StubAgent(), obs, 2, False, 0.1, ep=2)   # within interval: skipped
    assert read_snapshot(path)["ep"] == 1 and os.path.getmtime(path) == first


def test_missing_file_reads_none():
    assert read_snapshot("/nonexistent/live.npz") is None
