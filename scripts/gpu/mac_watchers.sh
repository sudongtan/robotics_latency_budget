#!/usr/bin/env bash
# Mac: run the session's two watchers in the background, detached from any window (D38).
#   bash scripts/gpu/mac_watchers.sh start    # run FROM Terminal.app (see below)
#   bash scripts/gpu/mac_watchers.sh status   # running? pids, last pull age, last health line
#   bash scripts/gpu/mac_watchers.sh stop     # after the final verified pull
# Watchers: scripts/gpu/sync_down.sh --watch 300 (pull runs/, under caffeinate) and
# scripts/gpu/health.py --watch 300. Started with nohup, they survive closing the Terminal
# window, VS Code or Claude Code; they do not survive a reboot or a dead battery (then run
# `start` again; the first pull catches up).
#
# Why not a LaunchAgent: macOS privacy protection (TCC) blocks launchd jobs from ~/Documents,
# where the repo lives ("Operation not permitted"), unless bash and python get Full Disk Access.
# Started from Terminal.app, the processes keep Terminal's Documents permission.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
DIR="$REPO/runs/_watchers"
cd "$REPO"

alive() { [[ -f "$DIR/$1.pid" ]] && kill -0 "$(cat "$DIR/$1.pid")" 2>/dev/null; }

start_one() {  # name, command...
  local name="$1"; shift
  if alive "$name"; then echo "$name already running (pid $(cat "$DIR/$name.pid"))"; return; fi
  nohup "$@" >> "$DIR/$name.log" 2>&1 < /dev/null &
  echo $! > "$DIR/$name.pid"
  disown
  echo "started $name (pid $!, log runs/_watchers/$name.log)"
}

case "${1:-status}" in
  start)
    mkdir -p "$DIR"
    if pgrep -f "sync_down.sh --watch|health.py --watch" >/dev/null; then
      others="$(pgrep -f "sync_down.sh --watch|health.py --watch" | tr '\n' ' ')"
      own="$(cat "$DIR"/*.pid 2>/dev/null | tr '\n' ' ' || true)"
      echo "note: watcher processes already running (pids $others); own: ${own:-none}."
      echo "      Stop the Terminal-window copies (Ctrl-C) to avoid running two of each."
    fi
    start_one syncdown bash scripts/gpu/sync_down.sh --watch 300
    start_one health uv run python scripts/gpu/health.py --watch 300
    ;;
  stop)
    for n in syncdown health; do
      if alive "$n"; then
        pid="$(cat "$DIR/$n.pid")"
        pkill -P "$pid" 2>/dev/null || true  # children (caffeinate, python)
        kill "$pid" 2>/dev/null || true
        echo "stopped $n (pid $pid)"
      else
        echo "$n not running"
      fi
      rm -f "$DIR/$n.pid"
    done
    pkill -f "caffeinate -i .*sync_down.sh" 2>/dev/null || true
    ;;
  status)
    for n in syncdown health; do
      if alive "$n"; then echo "$n: running (pid $(cat "$DIR/$n.pid"))"; else echo "$n: NOT running"; fi
    done
    if [[ -f runs/.last_pull ]]; then
      echo "last successful pull: $(( ($(date +%s) - $(stat -f %m runs/.last_pull)) / 60 )) min ago"
    fi
    [[ -f runs/_health.log ]] && tail -1 runs/_health.log
    ;;
  *) echo "usage: mac_watchers.sh start|status|stop" >&2; exit 2 ;;
esac
