#!/usr/bin/env python3
"""Train the Double DQN agent.

    python train.py                 # real game: open Stereo Madness in GD
    python train.py --env fake      # offline simulator (sanity check)
    python train.py --resume        # continue from runs/<name>/checkpoint.pt

Ctrl+C saves a checkpoint and exits.

Every `eval_every`-th episode is played greedily (eps = 0, pure exploitation);
its progress is the honest "how good is the agent now" number. All other
episodes use epsilon-greedy, so some of their deaths are exploration.
"""

import argparse
import json
import os
import sys
import time
from collections import deque

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config, fake_config          # noqa: E402
from dqn import DQNAgent, NStepBuffer, ReplayBuffer, epsilon_at  # noqa: E402
from gd_env import ACTIONS, make_env            # noqa: E402
from live import LivePublisher                  # noqa: E402
from obs import GRID_SHAPE, VEC_DIM             # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=["gd", "fake"], default="gd")
    p.add_argument("--run", default=None, help="run name (default: env name)")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--steps", type=int, default=None, help="override total_steps")
    p.add_argument("--reward", choices=["progress", "time"], default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--quiet", action="store_true", help="only print greedy episodes")
    return p.parse_args()


def save(path, agent, meta, cfg):
    tmp = path + ".tmp"
    torch.save({"agent": agent.state_dict(), "meta": meta, "config": cfg.to_dict()}, tmp)
    os.replace(tmp, path)


def main():
    args = parse_args()
    cfg = fake_config() if args.env == "fake" else Config()
    if args.steps:
        cfg.total_steps = args.steps
    if args.reward:
        cfg.reward_mode = args.reward
    if args.device:
        cfg.device = args.device

    run_dir = os.path.join(ROOT, "runs", args.run or args.env)
    os.makedirs(run_dir, exist_ok=True)
    ckpt_path = os.path.join(run_dir, "checkpoint.pt")
    best_path = os.path.join(run_dir, "best.pt")
    log_path = os.path.join(run_dir, "episodes.jsonl")
    live = LivePublisher(os.path.join(run_dir, "live.npz"))   # feeds eyes.py

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    env = make_env(args.env, cfg)
    agent = DQNAgent(GRID_SHAPE, VEC_DIM, env.action_space.n, cfg)
    buffer = ReplayBuffer(cfg.buffer_size, GRID_SHAPE, VEC_DIM, seed=cfg.seed)
    nstep = NStepBuffer(cfg.n_step, cfg.gamma)

    meta = {"total_steps": 0, "episode": 0, "best_progress": 0.0, "best_greedy": 0.0,
            "wall_time": 0.0}
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=agent.device, weights_only=False)
        agent.load_state_dict(ck["agent"])
        meta.update(ck["meta"])
        print(f"[train] resumed from episode {meta['episode']}, step {meta['total_steps']} "
              f"(replay memory starts empty and refills)")
    elif os.path.exists(log_path):
        os.remove(log_path)  # fresh run, fresh log

    print(f"[train] env={args.env} reward={cfg.reward_mode} device={agent.device} "
          f"run_dir={os.path.relpath(run_dir, ROOT)}")
    print(f"[train] exploration: eps {cfg.eps_start} -> {cfg.eps_end} over "
          f"{cfg.eps_decay_steps:,} steps, greedy eval every {cfg.eval_every} episodes")

    recent = deque(maxlen=20)
    start_time = time.time() - meta["wall_time"]
    learn_from = meta["total_steps"] + cfg.learning_starts  # warm up (again) after resume

    def checkpoint():
        meta["wall_time"] = time.time() - start_time
        save(ckpt_path, agent, meta, cfg)

    try:
        while meta["total_steps"] < cfg.total_steps:
            meta["episode"] += 1
            ep = meta["episode"]
            greedy = ep % cfg.eval_every == 0
            obs, info = env.reset()
            nstep.clear()

            ep_reward, ep_steps, explored = 0.0, 0, 0
            counts = [0] * len(ACTIONS)
            losses, qs = [], []
            t0 = time.time()
            while True:
                eps = 0.0 if greedy else epsilon_at(meta["total_steps"], cfg.eps_start,
                                                     cfg.eps_end, cfg.eps_decay_steps)
                action, was_random = agent.select_action(obs, eps)
                live.maybe_publish(agent, obs, action, was_random, eps, ep=ep,
                                   step=meta["total_steps"], greedy=greedy,
                                   progress=float(info.get("pct", 0.0)),
                                   best_greedy=meta["best_greedy"])
                next_obs, reward, terminated, truncated, info = env.step(action)

                for tr in nstep.push(obs, action, reward, next_obs, terminated, truncated):
                    buffer.add(*tr)

                meta["total_steps"] += 1
                if (meta["total_steps"] >= learn_from and len(buffer) >= cfg.batch_size
                        and meta["total_steps"] % cfg.train_every == 0):
                    loss, q = agent.update(buffer.sample(cfg.batch_size))
                    losses.append(loss)
                    qs.append(q)

                ep_reward += reward
                ep_steps += 1
                explored += was_random
                counts[action] += 1
                obs = next_obs
                if terminated or truncated:
                    break

            progress = float(min(100.0, info["pct"]))
            meta["best_progress"] = max(meta["best_progress"], progress)
            # >= so that once it can win, best.pt tracks the latest winning model
            if greedy and progress >= meta["best_greedy"]:
                meta["best_greedy"] = progress
                save(best_path, agent, meta, cfg)
            recent.append(progress)
            elapsed = time.time() - start_time
            sps = ep_steps / max(1e-6, time.time() - t0)

            entry = {
                "ep": ep, "greedy": greedy, "progress": round(progress, 2),
                "reward": round(ep_reward, 3), "length": ep_steps,
                "steps": meta["total_steps"], "eps": round(eps, 4),
                "explored": round(explored / max(1, ep_steps), 3),
                "actions": counts, "win": bool(info.get("win")),
                "loss": round(float(np.mean(losses)), 5) if losses else None,
                "q": round(float(np.mean(qs)), 3) if qs else None,
                "best": round(meta["best_progress"], 2),
                "best_greedy": round(meta["best_greedy"], 2),
                "avg20": round(float(np.mean(recent)), 2),
                "elapsed": round(elapsed, 1), "sps": round(sps, 1),
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")

            if greedy or not args.quiet:
                tag = "GREEDY " if greedy else "explore"
                print(f"ep {ep:5d} {tag} | {progress:6.2f}%{' WIN' if entry['win'] else '    '}"
                      f" | len {ep_steps:5d} | eps {eps:.3f} | random {entry['explored']:.0%}"
                      f" | a {counts} | best {meta['best_progress']:.1f}%"
                      f" greedy-best {meta['best_greedy']:.1f}% | avg20 {entry['avg20']:.1f}%"
                      f" | {sps:.0f} sps | step {meta['total_steps']:,}")

            if ep % cfg.ckpt_every == 0:
                checkpoint()
    except KeyboardInterrupt:
        print("\n[train] stopping")
    finally:
        checkpoint()
        env.close()
        print(f"[train] saved {os.path.relpath(ckpt_path, ROOT)}; "
              f"best greedy progress {meta['best_greedy']:.1f}%")


if __name__ == "__main__":
    main()
