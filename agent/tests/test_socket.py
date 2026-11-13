"""End-to-end check of the socket protocol, with a stand-in for the mod.

The stand-in connects like the mod does, answers {"cmd":...} lines with
level/state messages produced by FakeGame, and can drop the connection to
test reconnecting.
"""

import json
import os
import socket
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import fake_config
from fake_env import FakeGame
from gd_env import GDEnv, SocketBackend


def fake_mod(path, max_commands, sessions=1):
    """Connect `sessions` times; serve up to max_commands per connection."""
    for _ in range(sessions):
        game = FakeGame()
        while True:
            try:
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.connect(path)
                break
            except OSError:
                time.sleep(0.01)
        f = s.makefile("rwb")
        for _ in range(max_commands):
            line = f.readline()
            if not line:
                break
            cmd = json.loads(line)
            if cmd["cmd"] == "reset":
                level, state = game.reset()
                if level:
                    f.write((json.dumps(level) + "\n").encode())
            else:
                state = game.step(cmd["a"])
            f.write((json.dumps(state) + "\n").encode())
            f.flush()
        f.close()
        s.close()


def test_protocol_and_reconnect():
    path = os.path.join(tempfile.mkdtemp(), "t.sock")
    cfg = fake_config()
    env = GDEnv(SocketBackend(path), cfg)
    # first connection dies after 1 reset + 9 steps; the second one is healthy
    t = threading.Thread(target=fake_mod, args=(path, 10, 2), daemon=True)
    t.start()

    obs, info = env.reset()
    assert env.geom is not None and len(env.geom) == 27
    for _ in range(9):
        obs, r, term, trunc, info = env.step(0)
        assert not term and not trunc
    # the stand-in hangs up: the episode is truncated, not a death
    obs, r, term, trunc, info = env.step(0)
    assert trunc and not term and info.get("disconnected")
    # reset waits for the reconnect and gets a fresh level message
    obs, info = env.reset()
    assert info["x"] == 0.0
    obs, r, term, trunc, info = env.step(1)
    assert not term
    env.close()
    t.join(timeout=2)
