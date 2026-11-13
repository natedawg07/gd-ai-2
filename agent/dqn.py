"""Double DQN, written out by hand.

The pieces, in the order the agent uses them:

  1. select_action  – epsilon-greedy. With probability eps, EXPLORE (random
                      action); otherwise EXPLOIT (the action with the highest
                      predicted value Q(s, a)).
  2. NStepBuffer    – turns single steps into n-step transitions, so a bad
                      jump gets blamed even if the death comes 3 steps later.
  3. ReplayBuffer   – remembers past transitions; training samples random
                      batches so consecutive, highly correlated frames don't
                      dominate each update.
  4. update         – one gradient step towards the Double DQN target:
                        y = R + gamma^n * (1 - done) * Q_target(s', argmax_a Q_online(s', a))
                      The online net picks the next action, the slower target
                      net scores it. Splitting the two jobs reduces the
                      over-estimation plain DQN suffers from.
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

FLYING_MODES = {1, 3, 4, 7}   # ship, ufo, wave, swing


def pick_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    # On this Mac, PyTorch's CPU build has no oneDNN, so conv backward passes
    # are ~15x slower on CPU (~50 ms/update) than on the Apple GPU (~3 ms).
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Exploration schedule ───────────────────────────────────────────────────

def epsilon_at(step, start, end, decay_steps):
    """Linear decay from start to end over decay_steps, then constant."""
    if step >= decay_steps:
        return end
    return start + (end - start) * (step / decay_steps)


def is_flying(vec):
    """Gamemode one-hot lives at vec[3:11]: cube, ship, ball, ufo, wave, robot,
    spider, swing. Ship/ufo/wave/swing fly; the others are ground modes."""
    return int(np.argmax(vec[3:11])) in FLYING_MODES


# ── Network ────────────────────────────────────────────────────────────────

class QNetwork(nn.Module):
    """Grid -> small CNN, player vector -> MLP, joined -> one Q-value per action.

    No pooling layers: exact horizontal distance to a spike is what decides
    when to jump, so we only downsample once with a strided conv.
    """

    def __init__(self, grid_shape, vec_dim, n_actions):
        super().__init__()
        c, h, w = grid_shape
        self.conv = nn.Sequential(
            nn.Conv2d(c, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        conv_out = 32 * ((h + 1) // 2) * ((w + 1) // 2)
        self.vec = nn.Sequential(nn.Linear(vec_dim, 64), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(conv_out + 64, 256), nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, grid, vec):
        return self.head(torch.cat([self.conv(grid), self.vec(vec)], dim=1))


# ── n-step returns ─────────────────────────────────────────────────────────

class NStepBuffer:
    """Collects single steps and emits n-step transitions:

        (s_t, a_t, R = r_t + g*r_{t+1} + ... + g^(k-1)*r_{t+k-1}, s_{t+k}, done, g^k)

    k = n normally, fewer at the end of an episode. done is True only if the
    episode really ended (death/win), not when it was cut off by a time limit,
    so truncated episodes still bootstrap from s_{t+k}.
    """

    def __init__(self, n, gamma):
        self.n = n
        self.gamma = gamma
        self.q = deque()

    def push(self, obs, action, reward, next_obs, terminated, truncated):
        self.q.append((obs, action, reward))
        out = []
        if len(self.q) == self.n:
            out.append(self._emit(next_obs, terminated))
            self.q.popleft()
        if terminated or truncated:
            while self.q:
                out.append(self._emit(next_obs, terminated))
                self.q.popleft()
        return out

    def _emit(self, next_obs, terminated):
        ret, disc = 0.0, 1.0
        for _, _, r in self.q:
            ret += disc * r
            disc *= self.gamma
        obs, action, _ = self.q[0]
        return obs, action, ret, next_obs, float(terminated), disc

    def clear(self):
        self.q.clear()


# ── Replay memory ──────────────────────────────────────────────────────────

class ReplayBuffer:
    """Ring buffer. Grids are 0/1, so they're bit-packed (8x smaller): 200k
    transitions take ~100 MB instead of ~770 MB."""

    def __init__(self, capacity, grid_shape, vec_dim, seed=0):
        self.capacity = capacity
        self.grid_shape = grid_shape
        self.grid_size = int(np.prod(grid_shape))
        packed = (self.grid_size + 7) // 8
        self.grid = np.zeros((capacity, packed), dtype=np.uint8)
        self.next_grid = np.zeros((capacity, packed), dtype=np.uint8)
        self.vec = np.zeros((capacity, vec_dim), dtype=np.float32)
        self.next_vec = np.zeros((capacity, vec_dim), dtype=np.float32)
        self.action = np.zeros(capacity, dtype=np.int64)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.discount = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def _pack(self, g):
        return np.packbits(np.asarray(g, dtype=np.uint8).reshape(-1))

    def _unpack(self, packed):
        flat = np.unpackbits(packed, axis=1, count=self.grid_size)
        return flat.reshape((-1, *self.grid_shape)).astype(np.float32)

    def add(self, obs, action, ret, next_obs, done, discount):
        i = self.pos
        self.grid[i] = self._pack(obs["grid"])
        self.vec[i] = obs["vec"]
        self.next_grid[i] = self._pack(next_obs["grid"])
        self.next_vec[i] = next_obs["vec"]
        self.action[i] = action
        self.reward[i] = ret
        self.done[i] = done
        self.discount[i] = discount
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = self.rng.integers(0, self.size, size=batch_size)
        return {
            "grid": self._unpack(self.grid[idx]),
            "vec": self.vec[idx],
            "action": self.action[idx],
            "reward": self.reward[idx],
            "next_grid": self._unpack(self.next_grid[idx]),
            "next_vec": self.next_vec[idx],
            "done": self.done[idx],
            "discount": self.discount[idx],
        }

    def __len__(self):
        return self.size


# ── The agent ──────────────────────────────────────────────────────────────

def double_dqn_target(reward, discount, done, next_q_online, next_q_target):
    """y = R + discount * (1 - done) * Q_target(s', argmax_a Q_online(s', a))."""
    best_next = next_q_online.argmax(dim=1, keepdim=True)
    next_value = next_q_target.gather(1, best_next).squeeze(1)
    return reward + discount * (1.0 - done) * next_value


class DQNAgent:
    def __init__(self, grid_shape, vec_dim, n_actions, cfg):
        self.cfg = cfg
        self.n_actions = n_actions
        self.device = pick_device(cfg.device)
        self.online = QNetwork(grid_shape, vec_dim, n_actions).to(self.device)
        self.target = QNetwork(grid_shape, vec_dim, n_actions).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        # Acting needs one tiny forward pass per step; on the GPU that's mostly
        # launch/sync overhead (~1.7 ms vs ~0.1 ms on CPU). So actions come from
        # a CPU copy of the online net, refreshed every `actor_sync` updates.
        if self.device.type == "cpu":
            self.actor = self.online
        else:
            self.actor = QNetwork(grid_shape, vec_dim, n_actions)
            self.actor.eval()
            self._sync_actor()
        self.opt = torch.optim.Adam(self.online.parameters(), lr=cfg.lr)
        self.explore_p = np.asarray(cfg.explore_probs, dtype=np.float64)
        self.explore_p /= self.explore_p.sum()
        self.explore_p_fly = np.asarray(cfg.explore_probs_fly, dtype=np.float64)
        self.explore_p_fly /= self.explore_p_fly.sum()
        self.rng = random.Random(cfg.seed)
        self.np_rng = np.random.default_rng(cfg.seed)
        self.updates = 0

    # 1. explore or exploit
    def select_action(self, obs, eps):
        """Returns (action, explored)."""
        if self.rng.random() < eps:
            p = self.explore_p_fly if is_flying(obs["vec"]) else self.explore_p
            return int(self.np_rng.choice(self.n_actions, p=p)), True
        return int(self.q_values(obs).argmax()), False

    @torch.no_grad()
    def q_values(self, obs):
        g = torch.as_tensor(obs["grid"]).unsqueeze(0)
        v = torch.as_tensor(obs["vec"]).unsqueeze(0)
        return self.actor(g, v).squeeze(0).numpy()

    def _sync_actor(self):
        if self.actor is not self.online:
            self.actor.load_state_dict({k: t.detach().cpu() for k, t in
                                        self.online.state_dict().items()})

    # 4. learn from a batch
    def update(self, batch):
        dev = self.device
        grid = torch.as_tensor(batch["grid"], device=dev)
        vec = torch.as_tensor(batch["vec"], device=dev)
        action = torch.as_tensor(batch["action"], device=dev)
        reward = torch.as_tensor(batch["reward"], device=dev)
        next_grid = torch.as_tensor(batch["next_grid"], device=dev)
        next_vec = torch.as_tensor(batch["next_vec"], device=dev)
        done = torch.as_tensor(batch["done"], device=dev)
        discount = torch.as_tensor(batch["discount"], device=dev)

        q = self.online(grid, vec).gather(1, action.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            y = double_dqn_target(reward, discount, done,
                                  self.online(next_grid, next_vec),
                                  self.target(next_grid, next_vec))
        loss = F.smooth_l1_loss(q, y)   # Huber: robust to the big death/win rewards

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.cfg.grad_clip)
        self.opt.step()

        self.updates += 1
        if self.updates % self.cfg.target_update == 0:
            self.target.load_state_dict(self.online.state_dict())
        if self.updates % self.cfg.actor_sync == 0:
            self._sync_actor()
        return float(loss.item()), float(q.mean().item())

    # ── save / load ────────────────────────────────────────────────────────
    def state_dict(self):
        return {"online": self.online.state_dict(), "target": self.target.state_dict(),
                "opt": self.opt.state_dict(), "updates": self.updates}

    def load_state_dict(self, d, weights_only=False):
        self.online.load_state_dict(d["online"])
        self.target.load_state_dict(d.get("target", d["online"]))
        if not weights_only and "opt" in d:
            self.opt.load_state_dict(d["opt"])
            self.updates = d.get("updates", 0)
        self._sync_actor()
