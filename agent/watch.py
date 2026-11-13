#!/usr/bin/env python3
"""Terminal dashboard for a training run.

    python watch.py               # runs/gd
    python watch.py --run fake
"""

import argparse
import json
import os
import time

from rich import box
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BARS = "▁▂▃▄▅▆▇█"


def read(path):
    out = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def spark(values, width=60, lo=0.0, hi=100.0):
    values = values[-width:]
    if not values:
        return ""
    return "".join(BARS[min(len(BARS) - 1, int((v - lo) / max(1e-9, hi - lo) * len(BARS)))]
                   for v in values)


def render(eps, run):
    if not eps:
        return Panel(f"waiting for runs/{run}/episodes.jsonl ... start train.py", title="GD AI 2")
    last = eps[-1]
    greedy = [e for e in eps if e["greedy"]]
    train = [e for e in eps if not e["greedy"]]
    h, rem = divmod(int(last["elapsed"]), 3600)
    m, s = divmod(rem, 60)
    head = (f"[bold]episodes[/] {last['ep']}   [bold]steps[/] {last['steps']:,}   "
            f"[bold]ε[/] {train[-1]['eps'] if train else 0:.3f}   "
            f"[bold green]best greedy[/] {last['best_greedy']:.1f}%   "
            f"[bold]best any[/] {last['best']:.1f}%   [dim]{h}h{m:02d}m{s:02d}s, "
            f"{last['sps']:.0f} steps/s[/]")
    curves = (f"[blue]greedy (exploit)[/]  {spark([e['progress'] for e in greedy])}\n"
              f"[white]training (ε-greedy)[/] {spark([e['progress'] for e in train])}\n"
              f"[red]ε[/]                  {spark([e['eps'] for e in train], lo=0, hi=1)}")
    t = Table(box=box.SIMPLE_HEAVY, pad_edge=False)
    for col in ("ep", "mode", "progress", "len", "ε", "random", "nothing/click/hold"):
        t.add_column(col, justify="right")
    for e in eps[-15:]:
        prog = f"{e['progress']:.1f}%" + (" ★" if e.get("win") else "")
        t.add_row(str(e["ep"]), "[blue]greedy[/]" if e["greedy"] else "explore", prog,
                  str(e["length"]), f"{e['eps']:.2f}", f"{e['explored']:.0%}",
                  "/".join(str(a) for a in e["actions"]))
    return Group(Panel(head, title=f"GD AI 2: runs/{run}"), Panel(curves, title="progress %"),
                 Panel(t, title="recent episodes"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="gd")
    args = p.parse_args()
    path = os.path.join(ROOT, "runs", args.run, "episodes.jsonl")
    try:
        with Live(render(read(path), args.run), refresh_per_second=1, screen=True) as live:
            while True:
                time.sleep(2.0)
                live.update(render(read(path), args.run))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
