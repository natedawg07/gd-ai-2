#!/usr/bin/env python3
"""Agent's-eye view: what the agent sees and what it's thinking, live.

    python eyes.py                 # follow runs/gd (train.py or play.py)
    python eyes.py --run fake      # follow the simulator run
    python eyes.py --save out.png  # render the latest snapshot once and exit

Top left:     the exact 32x20 grid the network gets as input, in blocks
              relative to the player (blue square). Red = hazard,
              grey = solid, amber = pad/orb/portal. Dashed line = floor.
Top right:    Q(s, a) for nothing / click / hold: the agent's estimate of
              future reward for each action. Bold outline = the action it
              took. The title says whether that move was EXPLORE (random,
              chosen with probability eps) or EXPLOIT (highest Q).
Bottom left:  the three Q-values over the current episode. Watch Q(click)
              overtake Q(nothing) as a spike gets close.
Bottom right: episode, epsilon, progress.
"""

import argparse
import os
import sys
import time
from collections import deque

import matplotlib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from live import read_snapshot                                        # noqa: E402
from obs import (AHEAD_BLOCKS, BEHIND_BLOCKS, CH_HAZARD, CH_INTERACT,  # noqa: E402
                 CH_SOLID, HALF_HEIGHT_BLOCKS, N_MODES)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACTIONS = ("nothing", "click", "hold")
ACTION_COLORS = ("#9aa0a6", "#34a853", "#a142f4")
MODES = ("cube", "ship", "ball", "ufo", "wave", "robot", "spider", "swing")
HISTORY = 300
STALE_SEC = 3.0

RGB = {CH_SOLID: (0.37, 0.39, 0.41), CH_INTERACT: (0.98, 0.67, 0.0), CH_HAZARD: (0.85, 0.19, 0.15)}


def grid_to_rgb(grid):
    img = np.ones((grid.shape[1], grid.shape[2], 3), dtype=np.float32)
    for ch in (CH_SOLID, CH_INTERACT, CH_HAZARD):   # hazards drawn last, on top
        img[grid[ch] > 0] = RGB[ch]
    return img


class EyesWindow:
    def __init__(self, plt, title):
        from matplotlib.patches import Rectangle
        self.plt = plt
        self.fig = plt.figure(figsize=(13, 7))
        self.fig.canvas.manager.set_window_title(f"GD AI 2: agent's-eye view ({title})")
        gs = self.fig.add_gridspec(2, 3, height_ratios=[3, 1.5], width_ratios=[1.6, 1.6, 1.1],
                                   hspace=0.35, wspace=0.3)
        self.ax_grid = self.fig.add_subplot(gs[0, :2])
        self.ax_q = self.fig.add_subplot(gs[0, 2])
        self.ax_hist = self.fig.add_subplot(gs[1, :2])
        self.ax_text = self.fig.add_subplot(gs[1, 2])

        # grid
        extent = [-BEHIND_BLOCKS, AHEAD_BLOCKS, -HALF_HEIGHT_BLOCKS, HALF_HEIGHT_BLOCKS]
        self.im = self.ax_grid.imshow(np.ones((20, 32, 3)), origin="lower", extent=extent,
                                      interpolation="nearest")
        self.ax_grid.set_xticks(range(-BEHIND_BLOCKS, AHEAD_BLOCKS + 1))
        self.ax_grid.set_yticks(range(-HALF_HEIGHT_BLOCKS, HALF_HEIGHT_BLOCKS + 1))
        self.ax_grid.grid(color="#dadce0", lw=0.5)
        self.ax_grid.set_xlabel("blocks ahead of the player")
        self.ax_grid.set_ylabel("blocks above/below")
        self.player = Rectangle((-0.5, -0.5), 1, 1, fill=True, fc=(0.1, 0.45, 0.91, 0.15),
                                ec="#1a73e8", lw=2.5, zorder=5)
        self.ax_grid.add_patch(self.player)
        self.floor = self.ax_grid.axhline(-0.5, color="#1a73e8", ls="--", lw=1, alpha=0.6)

        # Q bars
        self.bars = self.ax_q.bar(range(3), [0, 0, 0], color=ACTION_COLORS, alpha=0.5)
        self.bar_labels = [self.ax_q.text(i, 0, "", ha="center", va="bottom", fontsize=10)
                           for i in range(3)]
        self.ax_q.set_xticks(range(3), ACTIONS)
        self.ax_q.axhline(0, color="#5f6368", lw=0.8)
        self.ax_q.set_ylabel("Q(s, a): expected future reward")

        # Q history
        self.hist = deque(maxlen=HISTORY)
        self.lines = [self.ax_hist.plot([], [], color=c, lw=1.5, label=a)[0]
                      for a, c in zip(ACTIONS, ACTION_COLORS)]
        self.explore_marks = self.ax_hist.scatter([], [], marker="|", s=40, c="#e37400",
                                                  label="explored")
        self.ax_hist.legend(loc="upper left", fontsize=8, ncol=4)
        self.ax_hist.set_title("Q-values during this episode", fontsize=10)
        self.ax_hist.set_xlabel("snapshots (~10 per second)")
        self.ax_hist.grid(alpha=0.3)

        # status text
        self.ax_text.axis("off")
        self.status = self.ax_text.text(0.0, 1.0, "waiting for train.py / play.py...",
                                        va="top", ha="left", family="monospace", fontsize=11,
                                        transform=self.ax_text.transAxes)
        self.last_ep = None

    def update(self, s):
        grid, vec, q = s["grid"], s["vec"], s["q"]
        action, explored = int(s["action"]), bool(s["explored"])
        greedy = bool(s.get("greedy", False))
        mode = MODES[int(np.argmax(vec[3:3 + N_MODES]))]
        held = vec[4 + N_MODES] > 0.5
        on_ground = vec[2] > 0.5

        # grid + player
        self.im.set_data(grid_to_rgb(grid))
        self.player.set_facecolor((0.1, 0.45, 0.91, 0.55 if held else 0.12))
        floor_y = -0.5 - 5.0 * float(vec[0])
        self.floor.set_ydata([floor_y, floor_y])
        self.floor.set_visible(-HALF_HEIGHT_BLOCKS < floor_y < HALF_HEIGHT_BLOCKS)
        self.ax_grid.set_title(
            f"What the agent sees  |  mode: {mode}  |  {'on ground' if on_ground else 'in the air'}"
            f"  |  button {'HELD' if held else 'up'}", fontsize=11)

        # Q bars
        lo, hi = min(0.0, float(q.min())), max(0.0, float(q.max()))
        pad = max(0.5, 0.15 * (hi - lo))
        self.ax_q.set_ylim(lo - pad, hi + pad)
        for i, (bar, lab) in enumerate(zip(self.bars, self.bar_labels)):
            bar.set_height(float(q[i]))
            chosen = i == action
            bar.set_alpha(1.0 if chosen else 0.35)
            bar.set_edgecolor("black" if chosen else "none")
            bar.set_linewidth(2.5 if chosen else 0)
            lab.set_position((i, max(float(q[i]), 0.0)))
            lab.set_text(f"{q[i]:.2f}")
        if explored:
            self.ax_q.set_title(f"EXPLORE: random action ({ACTIONS[action]})",
                                color="#e37400", fontsize=11, fontweight="bold")
        else:
            self.ax_q.set_title(f"EXPLOIT: highest Q ({ACTIONS[action]})"
                                + ("  [greedy eval]" if greedy else ""),
                                color="#1a73e8", fontsize=11, fontweight="bold")

        # history (reset each episode)
        ep = s.get("ep")
        if ep != self.last_ep:
            self.hist.clear()
            self.last_ep = ep
        self.hist.append((q.copy(), explored))
        hist = list(self.hist)
        qs = np.array([h[0] for h in hist])
        xs = np.arange(len(qs))
        for i, line in enumerate(self.lines):
            line.set_data(xs, qs[:, i])
        ex = [(x, qs[x].max()) for x in xs if hist[x][1]]
        self.explore_marks.set_offsets(np.array(ex) if ex else np.empty((0, 2)))
        self.ax_hist.set_xlim(0, max(10, len(qs)))
        self.ax_hist.set_ylim(qs.min() - 0.5, qs.max() + 0.5)
        self.update_status(s)

    def update_status(self, s):
        greedy = bool(s.get("greedy", False))
        age = time.time() - float(s.get("time", time.time()))
        stale = f"\n\n(no update for {age:.0f}s:\n paused or stopped?)" if age > STALE_SEC else ""
        self.status.set_text(
            f"episode   {s.get('ep', '?')}"
            f"{'  (greedy)' if greedy else ''}\n"
            f"step      {int(s.get('step', 0)):,}\n"
            f"epsilon   {float(s['eps']):.3f}\n"
            f"  -> explores {float(s['eps']):.0%} of moves\n"
            f"progress  {float(s.get('progress', 0)):.1f}%\n"
            f"best      {float(s.get('best_greedy', 0)):.1f}% (greedy)"
            f"{stale}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="gd")
    p.add_argument("--save", default=None)
    args = p.parse_args()
    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = os.path.join(ROOT, "runs", args.run, "live.npz")
    win = EyesWindow(plt, args.run)

    if args.save:
        s = read_snapshot(path)
        if s is None:
            sys.exit(f"no snapshot at {path}; is train.py running?")
        win.update(s)
        win.fig.savefig(args.save, dpi=100, bbox_inches="tight")
        print(f"saved {args.save}")
        return

    plt.ion()
    plt.show()
    last_mtime = None
    try:
        while plt.fignum_exists(win.fig.number):
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                mtime = None
            if mtime and mtime != last_mtime:
                s = read_snapshot(path)
                if s is not None:
                    win.update(s)
                    last_mtime = mtime
            elif mtime and time.time() - mtime > STALE_SEC:
                s = read_snapshot(path)   # only refresh the "no update for Ns" note
                if s is not None:
                    win.update_status(s)
            plt.pause(0.08)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
