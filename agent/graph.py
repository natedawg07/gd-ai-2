#!/usr/bin/env python3
"""Live learning curves from runs/<run>/episodes.jsonl.

    python graph.py                 # live window for runs/gd
    python graph.py --run fake      # the simulator run
    python graph.py --save out.png  # render once to a file and exit

Top:    level progress per episode. Grey dots are training episodes (with
        exploration), the blue line is greedy evaluation (pure exploitation).
Middle: epsilon (chance of a random action) and the fraction of actions that
        actually were random. This is exploration fading into exploitation.
Bottom: what the agent chose: nothing / click / hold, as shares of each episode.
"""

import argparse
import json
import os
import time

import matplotlib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WINDOW = 20


def read(path):
    eps = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    eps.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # partial last line while train.py is writing
    return eps


def rolling(v, w):
    v = np.asarray(v, dtype=float)
    if len(v) == 0:
        return v
    c = np.cumsum(np.insert(v, 0, 0.0))
    out = np.empty(len(v))
    for i in range(len(v)):
        lo = max(0, i - w + 1)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out


def draw(fig, axes, eps, title):
    for ax in axes:
        ax.clear()
    ax_p, ax_e, ax_a = axes
    if not eps:
        fig.suptitle(f"{title}: waiting for episodes...")
        return
    ep = np.array([e["ep"] for e in eps])
    prog = np.array([e["progress"] for e in eps])
    greedy = np.array([e["greedy"] for e in eps])
    train = ~greedy

    ax_p.scatter(ep[train], prog[train], s=6, c="#9aa0a6", alpha=0.5, label="training (ε-greedy)")
    ax_p.plot(ep[train], rolling(prog[train], WINDOW), c="#5f6368", lw=1, label=f"training avg{WINDOW}")
    if greedy.any():
        ax_p.plot(ep[greedy], prog[greedy], c="#1a73e8", lw=2, marker="o", ms=3,
                  label="greedy eval (ε=0)")
    wins = np.array([e.get("win", False) for e in eps])
    if wins.any():
        ax_p.scatter(ep[wins], prog[wins], marker="*", s=80, c="#f9ab00", zorder=5, label="win")
    ax_p.set_ylabel("level progress %")
    ax_p.set_ylim(0, 105)
    ax_p.legend(loc="upper left", fontsize=8)
    ax_p.grid(alpha=0.3)

    ax_e.plot(ep, [e["eps"] if not e["greedy"] else np.nan for e in eps], c="#d93025", lw=1.5,
              label="ε (explore probability)")
    ax_e.plot(ep[train], rolling([e["explored"] for e in eps if not e["greedy"]], WINDOW),
              c="#e37400", lw=1, ls="--", label="actually random")
    ax_e.set_ylabel("exploration")
    ax_e.set_ylim(0, 1.05)
    ax_e.legend(loc="upper right", fontsize=8)
    ax_e.grid(alpha=0.3)

    acts = np.array([e["actions"] for e in eps], dtype=float)
    share = acts / np.maximum(1, acts.sum(1, keepdims=True))
    ax_a.stackplot(ep, rolling(share[:, 0], WINDOW), rolling(share[:, 1], WINDOW),
                   rolling(share[:, 2], WINDOW), labels=["nothing", "click", "hold"],
                   colors=["#dadce0", "#34a853", "#a142f4"], alpha=0.9)
    ax_a.set_ylabel("action share")
    ax_a.set_xlabel("episode")
    ax_a.set_ylim(0, 1)
    ax_a.legend(loc="upper right", fontsize=8)

    last = eps[-1]
    fig.suptitle(f"{title}: {last['ep']} episodes, {last['steps']:,} steps | "
                 f"best greedy {last['best_greedy']:.1f}% | best {last['best']:.1f}%")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="gd")
    p.add_argument("--save", default=None)
    args = p.parse_args()
    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = os.path.join(ROOT, "runs", args.run, "episodes.jsonl")
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1.3, 1.3]})
    if args.save:
        draw(fig, axes, read(path), args.run)
        fig.tight_layout()
        fig.savefig(args.save, dpi=110)
        print(f"saved {args.save}")
        return

    plt.ion()
    last_n = -1
    try:
        while plt.fignum_exists(fig.number):
            eps = read(path)
            if len(eps) != last_n:
                draw(fig, axes, eps, args.run)
                fig.tight_layout()
                last_n = len(eps)
            plt.pause(2.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
