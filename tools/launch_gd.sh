#!/bin/sh
# Launch the standalone Geometry Dash 2.2081 + Geode copy that lives outside
# Steam's folder, so Steam updates can't overwrite it.
#
#   tools/launch_gd.sh
#   GD_APP="/path/to/Geometry Dash.app" tools/launch_gd.sh
#
# Steam must be running and signed in (GD uses it for your account).
# Starting the .app from Finder doesn't work: a Steam game launched outside
# Steam quits and asks Steam to start it, and Steam would start its own
# copy. Telling the game its Steam app id (322170) through steam_appid.txt
# in its working directory and the SteamAppId variable skips that step.

APP="${GD_APP:-$HOME/Games/Geometry Dash 2.2081/Geometry Dash.app}"
BIN_DIR="$APP/Contents/MacOS"

if [ ! -x "$BIN_DIR/Geometry Dash" ]; then
    echo "Not found: $APP" >&2
    exit 1
fi
if pgrep -x "Geometry Dash" >/dev/null; then
    echo "Geometry Dash is already running." >&2
    exit 1
fi
if ! pgrep -x steam_osx >/dev/null; then
    echo "Starting Steam first..."
    open -a Steam
    i=0
    while ! pgrep -x steam_osx >/dev/null && [ $i -lt 60 ]; do sleep 1; i=$((i + 1)); done
    sleep 5   # give it a moment to sign in
fi

[ -f "$BIN_DIR/steam_appid.txt" ] || echo 322170 > "$BIN_DIR/steam_appid.txt"
cd "$BIN_DIR" || exit 1
SteamAppId=322170 SteamGameId=322170 nohup "./Geometry Dash" >/dev/null 2>&1 &
echo "Launched $APP"
