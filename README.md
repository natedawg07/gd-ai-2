# GD AI 2

A Geometry Dash agent that learns to play by trial and error, using a hand-written **Double DQN** with an **epsilon-greedy** schedule (explore first, exploit later).

```
 Geometry Dash + Geode mod (mod/)                         Python agent (agent/)
 ┌──────────────────────────────┐   /tmp/gd_ai2.sock    ┌───────────────────────────┐
 │ game frozen until a command  │ <── {"cmd":"step"} ── │ DQN picks: nothing/click/ │
 │ applies input (handleButton) │                       │ hold (epsilon-greedy)     │
 │ runs 2 fixed 1/60 s updates  │ ── {"type":"state"} ─>│ reward, replay, learn     │
 │ reports death / win / pct    │                       │                           │
 └──────────────────────────────┘                       └───────────────────────────┘
```

The game runs in **lockstep** with the agent: it only advances when Python sends an action, so the same state plus the same action always gives the same result. Inputs and restarts happen inside the game, so there's no keyboard input, no window focus, and no `sleep()` calls.

## Setup (once)

```bash
cd ~/"GD AI 2"
/opt/homebrew/bin/python3.14 -m venv .venv && .venv/bin/pip install -r agent/requirements.txt
cd mod && GEODE_SDK=/Users/Shared/Geode/sdk geode build --ninja   # builds + installs the mod into the standalone GD copy
```

If the old `GD RL Bridge` mod ever shows up in **Geode → Mods**, disable it. It hooks the same functions.

## Run

1. Check that learning works without the game (a few minutes):
   `.venv/bin/python agent/train.py --env fake`
2. Train on the real game:
   - Start GD with `tools/launch_gd.sh`, or double-click `~/Games/Geometry Dash 2.2081/Launch Geometry Dash 2.2081.command`. Both start the standalone copy, see "Which Geometry Dash to run" below.
   - In settings, mute the music (training runs faster than real time, so it drifts out of sync).
   - Open **Stereo Madness**.
   - Run `.venv/bin/python agent/train.py`. The game connects on its own and starts.
   - Ctrl+C saves and stops. `--resume` continues.
3. Watch it learn, in a second terminal (add `--run fake` for the simulator):
   - `.venv/bin/python agent/graph.py`: live learning curves (progress, ε decay, action mix).
   - `.venv/bin/python agent/eyes.py`: **agent's-eye view**, updated about 10 times a second. It shows the exact grid the network sees, the live Q-values for nothing/click/hold with the chosen action outlined, whether each move was EXPLORE (random) or EXPLOIT (highest Q), and the Q-values over the current episode.
   - `.venv/bin/python agent/watch.py`: terminal dashboard.
4. Watch the best model play greedily:
   `.venv/bin/python agent/play.py` (real time by default; `--speed 0.5` for half speed). Pair it with `eyes.py` to see its decisions.
   On the simulator: `.venv/bin/python agent/play.py --env fake --model runs/fake-v2/checkpoint.pt` and `.venv/bin/python agent/eyes.py --run fake`.

### Hands-off training session

Run each of these in its own terminal tab, in this order (the trainer and viewers wait for the game):

```bash
cd ~/"GD AI 2" && caffeinate -di .venv/bin/python agent/train.py   # keeps the Mac awake
cd ~/"GD AI 2" && .venv/bin/python agent/graph.py                   # learning curves
cd ~/"GD AI 2" && .venv/bin/python agent/eyes.py                    # agent's-eye view
cd ~/"GD AI 2" && tools/launch_gd.sh                                # then open Stereo Madness
```

- While the agent is in control, the mod ignores GD's "pause when the window loses focus", so you can click other windows without stopping training. Esc still pauses.
- Don't minimize GD. A minimized app can stop drawing frames, and the game only advances when it draws.
- Training stops by itself after 1M steps. Ctrl+C in the trainer tab saves and stops at any time, and `--resume` continues.

Each run is saved in `runs/<env>/`: `episodes.jsonl`, `checkpoint.pt` (to resume), and `best.pt` (the best greedy result).

## How it learns

| Piece | Where | What it does |
|---|---|---|
| Actions | `gd_env.py` | `0` nothing (button up), `1` click (press then release), `2` hold (button down the whole step). About 30 decisions per second. |
| Observation | `obs.py` | A 32×20 grid around the player, 1 block behind to 15 ahead, with channels for solid / hazard / interactive. Plus the player's height, vertical speed, on-ground, gamemode, gravity and button state. No absolute x, so it can't memorize positions. |
| Reward | `gd_env.py` | `progress` (default): `100·Δx/level_length` per step, so an episode's total reward = % reached. `time`: +0.1 per step alive (`--reward time`). Death −10, finishing +100. |
| **Exploration vs exploitation** | `dqn.py: select_action`, `epsilon_at` | With probability ε it explores (random action, weighted 60% nothing / 20% click / 20% hold). Otherwise it exploits (`argmax Q`). ε goes from 1.0 to 0.05 over 150k steps. Every 10th episode is fully greedy (ε=0); that's the honest skill measure. |
| Memory | `dqn.py: ReplayBuffer` | 200k bit-packed transitions; training samples random batches. |
| Credit | `dqn.py: NStepBuffer` | 3-step returns, so the jump that caused a death gets the blame. |
| Learning | `dqn.py: update` | Double DQN target `R + γⁿ·Q_target(s', argmax Q_online(s'))`, Huber loss, Adam. |

All hyperparameters are in `agent/config.py`.

## Tests

```bash
.venv/bin/python -m pytest agent/tests
```
They cover the grid (a spike lands in the right cell), n-step returns, the epsilon schedule, the Double DQN target, the replay buffer, the simulator's physics, a scripted run that proves the simulator level is beatable, and the socket protocol end to end, including reconnecting after a dropped connection (with a stand-in for the mod).

## Check on the first in-game run

The mod builds against GD 2.2081, but these behaviors can only be confirmed inside the game:
- Click makes the cube jump once, hold makes the ship climb (the `handleButton` timing).
- After a death the level does **not** restart by itself (`delayedResetLevel` is blocked) and only restarts when the agent resets.
- The first `[env] level loaded: N objects {...}` line shows hazards, not thousands of decorations.
- Pausing GD (Esc) just pauses training. Python waits and prints a message every 10 s.
- After a win (`levelComplete` is skipped while the agent controls the game), the next reset starts a clean attempt.

Known limitation: level geometry is read once per level, so objects moved by triggers aren't tracked. Stereo Madness doesn't need it.

## Which Geometry Dash to run

Use the **standalone copy** at `~/Games/Geometry Dash 2.2081/Geometry Dash.app`. It's GD 2.2081 with Geode 5.11.2 and this mod, outside Steam's folder, so Steam updates can't touch it.
- **Start it** with `tools/launch_gd.sh`. Steam needs to be running and signed in; the script opens Steam if it isn't. Opening the `.app` from Finder doesn't work, because a Steam game started outside Steam quits and asks Steam to start its own copy. The script avoids that by giving the game its Steam app id (`steam_appid.txt` + `SteamAppId=322170`).
- **Mod builds** go there automatically: the Geode CLI's current profile is "GD 2.2081 standalone". The old Steam profile is still listed as `CryptoNateVa` (`geode profile switch CryptoNateVa`).
- **Steam's own copy** (`steamapps/common/Geometry Dash`) also has Geode right now, but the next Steam update will replace it. That doesn't matter for this project.

## If the Geode button disappears (Steam updated GD)

This only affects Steam's copy. You only need these steps if the standalone copy is lost and has to be rebuilt.

Steam's current GD build (buildid 26002420, Oct 19 2026) is newer than **2.2081**, the version Geode and this mod target. A Steam update, or "Verify integrity", replaces the app and removes Geode. To recover:

1. **Check the version.** Run `dwarfdump --uuid ".../Geometry Dash.app/Contents/MacOS/Geometry Dash"`. 2.2081 is arm64 `ABCA9EC8-DDEE-317A-BC32-3F41A842311D` and x86_64 `3FBF9F72-5A83-3918-A82A-37BD3C019D7A`.
2. **If it's not 2.2081, downgrade.** Open the Steam console (`steam://open/console`) and run `download_depot 322170 322172 8780295549609362522`. Then copy the downloaded `Geometry Dash.app` (from `Steam.AppBundle/Steam/Contents/MacOS/steamapps/content/app_322170/depot_322172/`) over `steamapps/common/Geometry Dash/Geometry Dash.app`.
3. **Reinstall Geode 5.11.2.** The official `geode-installer-v5.11.2-mac.pkg` fails on this Mac: its script hits an AppleScript bug at the file picker (error -1708). Do what its script does by hand, using the files from the package's `Payload`:
   - in `Contents/Frameworks/`, rename `libfmod.dylib` to `restore_fmod.dylib`;
   - copy in `Geode.dylib`, `GeodeBootstrapper.dylib` and Geode's `libfmod.dylib`;
   - copy `resources/*` to `Contents/geode/resources/geode.loader/`.
4. **Reinstall the mod:** copy `mod/build/nathanlee.gd-ai-2.geode` into `Contents/geode/mods/`.
5. **Make a fresh standalone copy:** run `ditto` on the fixed app into `~/Games/Geometry Dash 2.2081/`, then start it with `tools/launch_gd.sh`.
