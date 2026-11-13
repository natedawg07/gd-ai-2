import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from dqn import DQNAgent, NStepBuffer, ReplayBuffer, double_dqn_target, epsilon_at
from obs import GRID_SHAPE, VEC_DIM


def fake_obs(k):
    g = np.zeros(GRID_SHAPE, dtype=np.float32)
    g.flat[k % g.size] = 1.0
    return {"grid": g, "vec": np.full(VEC_DIM, float(k), dtype=np.float32)}


def test_epsilon_schedule():
    assert epsilon_at(0, 1.0, 0.05, 100) == 1.0
    assert epsilon_at(50, 1.0, 0.05, 100) == pytest.approx(0.525)
    assert epsilon_at(100, 1.0, 0.05, 100) == 0.05
    assert epsilon_at(10_000, 1.0, 0.05, 100) == 0.05


def test_nstep_returns_mid_episode():
    nb = NStepBuffer(n=3, gamma=0.5)
    out = []
    for t in range(4):
        out += nb.push(fake_obs(t), t % 3, 1.0, fake_obs(t + 1), False, False)
    # First emission after 3 steps: s0 -> s3, R = 1 + .5 + .25, discount .125
    assert len(out) == 2
    obs, a, R, nxt, done, disc = out[0]
    assert obs["vec"][0] == 0 and nxt["vec"][0] == 3
    assert R == pytest.approx(1.75) and disc == pytest.approx(0.125) and done == 0.0


def test_nstep_flush_on_death():
    nb = NStepBuffer(n=3, gamma=0.5)
    nb.push(fake_obs(0), 0, 1.0, fake_obs(1), False, False)
    out = nb.push(fake_obs(1), 1, -10.0, fake_obs(2), True, False)
    assert len(out) == 2
    _, _, R0, _, d0, disc0 = out[0]
    _, _, R1, _, d1, disc1 = out[1]
    assert R0 == pytest.approx(1.0 + 0.5 * -10.0) and disc0 == pytest.approx(0.25)
    assert R1 == pytest.approx(-10.0) and disc1 == pytest.approx(0.5)
    assert d0 == d1 == 1.0
    assert len(nb.q) == 0


def test_nstep_truncation_is_not_done():
    nb = NStepBuffer(n=3, gamma=0.9)
    out = nb.push(fake_obs(0), 0, 1.0, fake_obs(1), False, True)
    assert len(out) == 1 and out[0][4] == 0.0


def test_double_dqn_target():
    reward = torch.tensor([1.0, 1.0])
    discount = torch.tensor([0.9, 0.9])
    done = torch.tensor([0.0, 1.0])
    q_online = torch.tensor([[0.0, 5.0, 1.0], [0.0, 5.0, 1.0]])  # online picks action 1
    q_target = torch.tensor([[9.0, 2.0, 7.0], [9.0, 2.0, 7.0]])  # target scores it 2.0
    y = double_dqn_target(reward, discount, done, q_online, q_target)
    assert y[0].item() == pytest.approx(1.0 + 0.9 * 2.0)   # not 9.0: that's plain DQN
    assert y[1].item() == pytest.approx(1.0)              # done: no bootstrap


def test_replay_roundtrip():
    rb = ReplayBuffer(10, GRID_SHAPE, VEC_DIM)
    for k in range(15):   # wraps around
        rb.add(fake_obs(k), k % 3, float(k), fake_obs(k + 1), 0.0, 0.99)
    assert len(rb) == 10
    b = rb.sample(32)
    assert b["grid"].shape == (32, *GRID_SHAPE)
    for i in range(32):
        k = int(b["vec"][i, 0])
        assert b["reward"][i] == k and k >= 5
        assert b["grid"][i].sum() == 1 and b["grid"][i].flat[k % b["grid"][i].size] == 1
        assert b["next_vec"][i, 0] == k + 1


def test_select_action_explore_vs_exploit():
    cfg = Config(device="cpu", explore_probs=(0.0, 0.0, 1.0))
    agent = DQNAgent(GRID_SHAPE, VEC_DIM, 3, cfg)
    obs = fake_obs(3)
    # eps = 1: always explore, and explore_probs force "hold"
    assert all(agent.select_action(obs, 1.0) == (2, True) for _ in range(20))
    # eps = 0: always exploit = argmax Q
    best = int(agent.q_values(obs).argmax())
    assert all(agent.select_action(obs, 0.0) == (best, False) for _ in range(20))


def test_flying_modes_explore_with_their_own_probs():
    cfg = Config(device="cpu", explore_probs=(1.0, 0.0, 0.0), explore_probs_fly=(0.0, 0.0, 1.0))
    agent = DQNAgent(GRID_SHAPE, VEC_DIM, 3, cfg)
    for mode, expected in [(0, 0), (1, 2), (2, 0), (3, 2), (4, 2), (5, 0), (6, 0), (7, 2)]:
        obs = fake_obs(0)
        obs["vec"][:] = 0.0
        obs["vec"][3 + mode] = 1.0
        assert agent.select_action(obs, 1.0)[0] == expected, f"mode {mode}"


def test_update_reduces_loss_on_fixed_batch():
    cfg = Config(device="cpu", lr=1e-3, target_update=10_000)
    agent = DQNAgent(GRID_SHAPE, VEC_DIM, 3, cfg)
    rb = ReplayBuffer(64, GRID_SHAPE, VEC_DIM)
    for k in range(64):
        rb.add(fake_obs(k), k % 3, float(k % 5), fake_obs(k + 1), 1.0, 0.99)
    batch = rb.sample(64)
    first, _ = agent.update(batch)
    for _ in range(200):
        last, _ = agent.update(batch)
    assert last < first * 0.2
