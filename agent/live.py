"""Share what the agent is seeing/thinking with the eyes.py window.

train.py / play.py call LivePublisher.maybe_publish() every step; about 10
times a second it writes a tiny snapshot (the observation grid, player
vector, Q-values, chosen action, explore/exploit, epsilon, ...) to
runs/<run>/live.npz. The write is atomic (temp file + rename), so the viewer
never reads half a file, and the cost to training is ~one extra forward pass
per 0.1 s.
"""

import os
import time
import zipfile

import numpy as np


class LivePublisher:
    def __init__(self, path, interval=0.1):
        self.path = path
        self.tmp = path + ".tmp"
        self.interval = interval
        self._last = 0.0

    def due(self):
        return time.monotonic() - self._last >= self.interval

    def maybe_publish(self, agent, obs, action, explored, eps, **info):
        """Publish if `interval` has passed. `agent` provides q_values(obs)."""
        if not self.due():
            return
        self._last = time.monotonic()
        q = agent.q_values(obs)
        snap = {
            "grid": np.asarray(obs["grid"], dtype=np.uint8),
            "vec": np.asarray(obs["vec"], dtype=np.float32),
            "q": np.asarray(q, dtype=np.float32),
            "action": np.int64(action),
            "explored": np.bool_(explored),
            "eps": np.float32(eps),
            "time": np.float64(time.time()),
        }
        for k, v in info.items():
            snap[k] = np.asarray(v)
        try:
            with open(self.tmp, "wb") as f:
                np.savez(f, **snap)
            os.replace(self.tmp, self.path)
        except OSError:
            pass  # never let the viewer feature break training


def read_snapshot(path):
    """Returns a dict of plain Python/numpy values, or None if unavailable."""
    try:
        with np.load(path, allow_pickle=False) as z:
            out = {}
            for k in z.files:
                v = z[k]
                out[k] = v.item() if v.ndim == 0 else v
            return out
    except (OSError, ValueError, EOFError, KeyError, zipfile.BadZipFile):
        return None
