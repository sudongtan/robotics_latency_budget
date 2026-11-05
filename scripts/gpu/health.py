"""Health monitor for session S, run on the Mac next to `sync_down.sh --watch` (D32).

Every `--watch` seconds it probes the GPU box over ssh and checks the Mac, then reports:
- ALERT (also a macOS notification, once per new problem; "resolved" when it clears)
- INFO  (sweep events: gate result, sweep done) and a one-line status summary.
Everything is appended to runs/_health.log.

    uv run python scripts/gpu/health.py --watch 300     # during the session
    uv run python scripts/gpu/health.py                 # one check

Checks (thresholds are flags):
- box unreachable (2 probes in a row) -> box probably died: see the restart procedure (D31)
- box disk: free < --min-free-gb or used > --max-disk-pct
- box RAM available < --min-ram-gb; GPU memory > --max-gpu-mem-pct
- GPU utilization < --min-gpu-util for --low-util-checks probes in a row while a job runs
- sweep: a job FAILED, STEP/GATE FAILED, runner process gone before SWEEP DONE, current job log
  silent for > --stall-min minutes
- Mac: free disk at runs/ < --min-free-gb; no successful pull (runs/.last_pull) for
  > --pull-max-min minutes while the sweep runs
Uses GPU_HOST / GPU_DIR / GPU_SSH_CONFIG like the other scripts (scripts/gpu/common.sh).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
STATE = RUNS / ".health_state.json"
LOG = RUNS / "_health.log"

# Runs on the box; prints key=value lines and the status file as "S:<line>".
PROBE = r"""
cd "$1" 2>/dev/null || { echo "no_repo=1"; exit 0; }
df -Pk . | awk 'NR==2 {printf "disk_free_gb=%.1f\ndisk_used_pct=%d\n", $4/1048576, $5+0}'
awk '/MemAvailable/ {printf "ram_avail_gb=%.1f\n", $2/1048576}' /proc/meminfo
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits \
    | head -1 \
    | awk -F', *' '{print "gpu_util=" $1; print "gpu_mem_used=" $2; print "gpu_mem_total=" $3}'
fi
st=$(ls -t runs/_sweeps/*/status 2>/dev/null | head -1)
if [ -n "$st" ]; then
  echo "status_file=$st"
  pgrep -f "_sweeps/.*/run.sh" >/dev/null && echo "runner=1" || echo "runner=0"
  log=$(ls -t "$(dirname "$st")"/job*.log 2>/dev/null | head -1)
  [ -n "$log" ] && echo "log_age_s=$(( $(date +%s) - $(stat -c %Y "$log") ))"
  sed 's/^/S:/' "$st"
fi
"""


@dataclasses.dataclass
class Limits:
    min_free_gb: float = 10.0
    max_disk_pct: float = 85.0
    min_ram_gb: float = 2.0
    max_gpu_mem_pct: float = 97.0
    min_gpu_util: float = 10.0
    low_util_checks: int = 3
    stall_min: float = 30.0
    pull_max_min: float = 15.0


def parse_probe(text: str) -> dict:
    out: dict = {"status": []}
    for line in text.splitlines():
        if line.startswith("S:"):
            out["status"].append(line[2:])
        elif "=" in line:
            k, v = line.split("=", 1)
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out


def assess(probe: dict | None, mac: dict, state: dict, lim: Limits) -> tuple[set, list, list]:
    """Pure check. Returns (active alert messages, info messages, summary parts) and updates
    `state` (counters, number of status lines already seen)."""
    alerts: set[str] = set()
    infos: list[str] = []
    summary: list[str] = []

    if mac["free_gb"] < lim.min_free_gb:
        alerts.add(f"Mac disk low: {mac['free_gb']:.0f} GB free at runs/")

    if probe is None:
        state["unreachable"] = state.get("unreachable", 0) + 1
        if state["unreachable"] >= 2:
            alerts.add("GPU box unreachable (2 checks): it may have died; see restart (D31)")
        summary.append("box: no answer")
        return alerts, infos, summary
    state["unreachable"] = 0
    if probe.get("no_repo"):
        alerts.add("GPU box reachable but repo missing (push_code.sh not run?)")
        return alerts, infos, summary

    free, used = probe.get("disk_free_gb"), probe.get("disk_used_pct")
    if free is not None:
        summary.append(f"box disk {free:.0f} GB free ({used:.0f}% used)")
        if free < lim.min_free_gb or used > lim.max_disk_pct:
            alerts.add(f"GPU box disk low: {free:.0f} GB free, {used:.0f}% used")
    ram = probe.get("ram_avail_gb")
    if ram is not None:
        summary.append(f"RAM {ram:.0f} GB avail")
        if ram < lim.min_ram_gb:
            alerts.add(f"GPU box RAM low: {ram:.1f} GB available")
    if "gpu_util" in probe:
        mem_pct = 100.0 * probe["gpu_mem_used"] / max(probe["gpu_mem_total"], 1.0)
        summary.append(f"GPU {probe['gpu_util']:.0f}% util, mem {mem_pct:.0f}%")
        if mem_pct > lim.max_gpu_mem_pct:
            alerts.add(f"GPU memory nearly full: {mem_pct:.0f}%")

    status = probe["status"]
    done = any("SWEEP DONE" in s for s in status)
    running = bool(probe.get("runner")) and not done
    seen = state.get("status_seen", {}).get(probe.get("status_file", ""), 0)
    for line in status[seen:]:
        if "GATE FAILED" in line or "STEP FAILED" in line:
            alerts.add(f"sweep stopped: {line}")
        elif "FAILED(" in line:
            infos.append(f"job failed (queue continues): {line}")
        elif "SWEEP DONE" in line:
            infos.append("sweep done: run the final sync_down.sh and verify")
    gate_lines = [s for s in status if "[GATE]" in s]
    if gate_lines:
        i = status.index(gate_lines[-1])
        if i + 1 < len(status) and status[i + 1].split()[1] == "ok" and i + 1 >= seen:
            infos.append("gate PASSED: main sweep continues")
    state.setdefault("status_seen", {})[probe.get("status_file", "")] = len(status)
    n_done = sum(1 for s in status if " ok " in f" {s} ")
    n_fail = sum(1 for s in status if "FAILED(" in s)
    if status:
        summary.append(f"jobs ok {n_done}, failed {n_fail}" + (", DONE" if done else ""))

    if status and not done and not probe.get("runner"):
        alerts.add("sweep runner not running but sweep not done (box rebooted? relaunch, D31)")
    if running:
        age = probe.get("log_age_s")
        if age is not None and age > lim.stall_min * 60:
            alerts.add(f"current job silent for {age / 60:.0f} min (stalled?)")
        if mac["last_pull_min"] is not None and mac["last_pull_min"] > lim.pull_max_min:
            alerts.add(
                f"Mac has not pulled for {mac['last_pull_min']:.0f} min: "
                "is sync_down --watch running?"
            )
        if "gpu_util" in probe:
            low = probe["gpu_util"] < lim.min_gpu_util
            state["low_util"] = state.get("low_util", 0) + 1 if low else 0
            if state["low_util"] >= lim.low_util_checks:
                alerts.add(
                    f"GPU nearly idle ({probe['gpu_util']:.0f}%) for {state['low_util']} checks"
                )
    return alerts, infos, summary


def probe_box() -> dict | None:
    host = os.environ.get("GPU_HOST", "gpu")
    gdir = os.environ.get("GPU_DIR", "latency_budget")
    ssh = ["ssh", "-o", "ConnectTimeout=15", "-o", "BatchMode=yes"]
    if os.environ.get("GPU_SSH_CONFIG"):
        ssh += ["-F", os.environ["GPU_SSH_CONFIG"]]
    try:
        r = subprocess.run(
            ssh + [host, "bash", "-s", "--", gdir],
            input=PROBE,
            text=True,
            capture_output=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        return None
    return parse_probe(r.stdout) if r.returncode == 0 else None


def mac_state() -> dict:
    RUNS.mkdir(exist_ok=True)
    stamp = RUNS / ".last_pull"
    age = (time.time() - stamp.stat().st_mtime) / 60 if stamp.exists() else None
    return {"free_gb": shutil.disk_usage(RUNS).free / 2**30, "last_pull_min": age}


def notify(msg: str) -> None:
    if sys.platform == "darwin" and os.environ.get("HEALTH_NOTIFY", "1") == "1":
        text = msg.replace('"', "'")
        subprocess.run(
            ["osascript", "-e", f'display notification "{text}" with title "latency-budget"'],
            check=False,
        )


def check_once(lim: Limits) -> None:
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    alerts, infos, summary = assess(probe_box(), mac_state(), state, lim)
    prev = set(state.get("active", []))
    now = time.strftime("%F %T")
    lines = [f"{now} status: " + "; ".join(summary)]
    for a in sorted(alerts - prev):
        lines.append(f"{now} ALERT {a}")
        notify(a)
    for a in sorted(alerts & prev):
        lines.append(f"{now} still: {a}")
    for a in sorted(prev - alerts):
        lines.append(f"{now} resolved: {a}")
    for i in infos:
        lines.append(f"{now} INFO {i}")
        notify(i)
    state["active"] = sorted(alerts)
    STATE.write_text(json.dumps(state))
    with LOG.open("a") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--watch", type=float, default=0, help="seconds between checks (0 = once)")
    for f in dataclasses.fields(Limits):
        ap.add_argument("--" + f.name.replace("_", "-"), type=type(f.default), default=f.default)
    args = ap.parse_args()
    lim = Limits(**{f.name: getattr(args, f.name) for f in dataclasses.fields(Limits)})
    while True:
        check_once(lim)
        if not args.watch:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
