"""Every tunable number in one place."""

from dataclasses import dataclass, asdict


@dataclass
class Config:
    # ── Environment ────────────────────────────────────────────────────────
    sock_path: str = "/tmp/gd_ai2.sock"
    max_steps: int = 20_000          # truncate an episode after this many steps
    # Reward: "progress" = 100 * dx / level_length per step (episode total = % reached)
    #         "time"     = time_reward per step survived
    reward_mode: str = "progress"
    time_reward: float = 0.1
    death_penalty: float = -10.0     # added on the step you die (0 to disable)
    win_bonus: float = 100.0         # added on the step you finish the level

    # ── Exploration vs exploitation (epsilon-greedy) ──────────────────────
    eps_start: float = 1.0           # explore 100% of the time at first...
    eps_end: float = 0.05            # ...down to 5%
    eps_decay_steps: int = 150_000   # linear decay length
    # When exploring, sample actions with these probabilities
    # [nothing, click, hold]. On the ground (cube/ball/robot/spider), uniform
    # random play jumps 2/3 of the time and dies at the first spike, so it
    # leans toward "do nothing"...
    explore_probs: tuple = (0.6, 0.2, 0.2)
    # ...but in flying modes (ship/ufo/wave/swing) that same bias makes the
    # ship sink: only ~7% of random flights reached the first gap in the
    # simulator, vs ~41% with uniform actions. So flying explores uniformly.
    explore_probs_fly: tuple = (1 / 3, 1 / 3, 1 / 3)
    eval_every: int = 10             # every Nth episode is greedy (eps = 0)

    # ── DQN ────────────────────────────────────────────────────────────────
    gamma: float = 0.99
    n_step: int = 3
    lr: float = 1e-4
    batch_size: int = 64
    buffer_size: int = 200_000
    learning_starts: int = 5_000     # random-ish steps before the first update
    train_every: int = 1             # gradient steps happen every N env steps
    target_update: int = 2_000       # hard-copy online -> target every N updates
    grad_clip: float = 10.0
    device: str = "auto"             # "auto" | "cpu" | "mps" | "cuda"
    actor_sync: int = 100            # refresh the CPU acting copy every N updates

    # ── Bookkeeping ────────────────────────────────────────────────────────
    total_steps: int = 1_000_000
    ckpt_every: int = 25             # episodes between checkpoints
    seed: int = 0

    def to_dict(self):
        return asdict(self)


def fake_config() -> Config:
    """Smaller settings for the offline sanity-check environment."""
    return Config(
        max_steps=2_000,
        eps_decay_steps=30_000,
        learning_starts=2_000,
        buffer_size=50_000,
        total_steps=80_000,
        eval_every=5,
        target_update=1_000,
        lr=2.5e-4,
    )
