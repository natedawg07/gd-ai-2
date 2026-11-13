#!/usr/bin/env python3
"""Watch a trained agent play with pure exploitation (eps = 0).

    python play.py                    # runs/gd/best.pt on the real game
    python play.py --env fake         # runs/fake/best.pt on the simulator
    python play.py --model runs/gd/checkpoint.pt --episodes 5
    python play.py --speed 0.5        # half speed, easier to follow in eyes.py

Pair it with `python eyes.py --run <env>` to watch what it's thinking.
"""

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config, fake_config   # noqa: E402
from dqn import DQNAgent                  # noqa: E402
from gd_env import ACTIONS, make_env      # noqa: E402
from live import LivePublisher            # noqa: E402
from obs import GRID_SHAPE, VEC_DIM       # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=["gd", "fake"], default="gd")
    p.add_argument("--model", default=None)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--speed", type=float, default=1.0,
                   help="1 = real time (30 decisions/s, like the mod); 0 = as fast as possible")
    args = p.parse_args()
    step_time = (1.0 / 30.0) / args.speed if args.speed > 0 else 0.0

    cfg = fake_config() if args.env == "fake" else Config()
    path = args.model or os.path.join(ROOT, "runs", args.env, "best.pt")
    env = make_env(args.env, cfg)
    agent = DQNAgent(GRID_SHAPE, VEC_DIM, env.action_space.n, cfg)
    ck = torch.load(path, map_location=agent.device, weights_only=False)
    agent.load_state_dict(ck["agent"], weights_only=True)
    print(f"[play] {path} (trained {ck['meta']['total_steps']:,} steps)")
    run_dir = os.path.join(ROOT, "runs", args.env)
    os.makedirs(run_dir, exist_ok=True)
    live = LivePublisher(os.path.join(run_dir, "live.npz"))   # eyes.py --run <env>
    best = float(ck["meta"].get("best_greedy", 0.0))

    try:
        for ep in range(1, args.episodes + 1):
            obs, info = env.reset()
            counts = [0] * len(ACTIONS)
            step = 0
            while True:
                action, _ = agent.select_action(obs, eps=0.0)
                live.maybe_publish(agent, obs, action, False, 0.0, ep=ep, step=step, greedy=True,
                                   progress=float(info.get("pct", 0.0)), best_greedy=best)
                step += 1
                counts[action] += 1
                t0 = time.monotonic()
                obs, _, terminated, truncated, info = env.step(action)
                if step_time:
                    time.sleep(max(0.0, step_time - (time.monotonic() - t0)))
                if terminated or truncated:
                    break
            result = "WIN" if info.get("win") else f"died at {info['pct']:.2f}%"
            print(f"episode {ep}: {result} | actions {dict(zip(ACTIONS, counts))}")
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
