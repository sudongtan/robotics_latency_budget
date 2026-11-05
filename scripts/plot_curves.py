"""Learning curves for one or more runs: writes runs/<id>/curves.png for each run, and a
combined plot when several are given.

    uv run python scripts/plot_curves.py runs/smoke_A_mlp_s0_* runs/smoke_D_* runs/smoke_E_*
    uv run python scripts/plot_curves.py runs/smoke_* --out runs/smoke_curves.png
"""

from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PANELS = [
    ("metrics.jsonl", "rollout/reward_per_step", "train reward / step"),
    ("metrics.jsonl", "rollout/episode_length", "train episode length (completed)"),
    ("eval.jsonl", "eval/episode_return", "eval return (deterministic)"),
    ("metrics.jsonl", "policy/std_mean", "policy std"),
]


def _read(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _plot(ax_row, run: pathlib.Path, label: str) -> None:
    for ax, (fname, key, title) in zip(ax_row, PANELS, strict=True):
        rows = [r for r in _read(run / fname) if r.get(key) is not None]
        ax.plot(
            [r["env_steps"] for r in rows],
            [r[key] for r in rows],
            label=label,
            marker="o" if fname == "eval.jsonl" else None,
        )
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("env steps", fontsize=8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path, default=None)
    args = ap.parse_args()
    for run in args.runs:
        fig, axes = plt.subplots(1, len(PANELS), figsize=(16, 3.2))
        _plot(axes, run, run.name)
        fig.suptitle(run.name, fontsize=10)
        fig.tight_layout()
        fig.savefig(run / "curves.png", dpi=110)
        plt.close(fig)
    if len(args.runs) > 1:
        fig, axes = plt.subplots(1, len(PANELS), figsize=(16, 3.4))
        for run in args.runs:
            _plot(axes, run, run.name)
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        out = args.out or args.runs[0].parent / "curves_combined.png"
        fig.savefig(out, dpi=110)
        print("wrote", out)


if __name__ == "__main__":
    main()
