import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import fake_config
from gd_env import make_env


@pytest.fixture
def env():
    e = make_env("fake", fake_config())
    yield e
    e.close()


def test_spaces_match(env):
    obs, info = env.reset()
    assert env.observation_space.contains(obs)
    obs, r, term, trunc, info = env.step(0)
    assert env.observation_space.contains(obs)


def test_doing_nothing_dies_at_first_spike(env):
    env.reset()
    total, steps = 0.0, 0
    while True:
        _, r, term, trunc, info = env.step(0)
        total += r
        steps += 1
        if term or trunc:
            break
    assert info["dead"] and not info["win"]
    # Dies near the first spike at x=600 (out of 5100)
    assert 10.0 < info["pct"] < 13.0
    # Progress reward sums to the % reached, plus the death penalty
    assert total == pytest.approx(info["pct"] + env.cfg.death_penalty, abs=0.5)


def test_hold_makes_ship_climb_and_release_falls(env):
    env.reset()
    g = env.backend
    g.x = g.ship_from + 5    # teleport into the ship section
    _, _, _, _, info = env.step(2)
    assert info["mode"] == 1
    y0 = g.y
    for _ in range(5):
        env.step(2)
    assert g.y > y0          # holding climbs
    y1 = g.y
    for _ in range(25):      # the ship has momentum: it keeps rising briefly
        env.step(0)
    assert g.y < y1          # releasing falls


def test_click_jumps_once_hold_keeps_jumping(env):
    env.reset()
    g = env.backend
    env.step(1)
    assert g.vy > 0           # click on the ground jumps
    while not g.on_ground:
        env.step(0)
    env.step(0)
    assert g.on_ground        # no second jump without input
    # hold: lands and immediately jumps again
    jumps = 0
    was_ground = True
    for _ in range(60):
        env.step(2)
        if was_ground and g.vy > 0:
            jumps += 1
        was_ground = g.on_ground or g.y <= 105.0 + 1e-6
        if g.dead:
            break
    assert jumps >= 1


def test_time_reward_mode():
    cfg = fake_config()
    cfg.reward_mode = "time"
    e = make_env("fake", cfg)
    e.reset()
    _, r, _, _, _ = e.step(0)
    assert r == pytest.approx(cfg.time_reward)
    e.close()


def test_scripted_policy_can_win():
    """The fake level is beatable with the three actions (so it's a fair test)."""
    e = make_env("fake", fake_config())
    obs, info = e.reset()
    g = e.backend
    while True:
        if g.mode == 0:
            # jump when the nearest spike ahead is ~1.5 blocks away (the
            # triple spike only clears if the jump starts 27-45 units early)
            ahead = g.hx0[(g.hx0 > g.x - 10) & g.is_hazard]
            a = 1 if len(ahead) and ahead.min() - g.x < 45 and g.on_ground else 0
        else:
            # ship: steer toward the centre of the next gap
            nxt = np.where(g.is_hazard & (g.hx1 > g.x - 15))[0]
            if len(nxt):
                ceil_i = nxt[np.argmax(g.hy0[nxt[:2]])] if len(nxt) >= 2 else nxt[0]
                target = g.hy0[ceil_i] - 60
            else:
                target = 200.0
            a = 2 if g.y < target - g.vy * 0.15 else 0
        obs, r, term, trunc, info = e.step(a)
        if term or trunc:
            break
    e.close()
    assert info["win"], f"scripted policy died at {info['pct']:.1f}%"
