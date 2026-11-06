"""Paper figures from `runs/_analysis/` (written by scripts/analyze.py) and `runs/<id>/` (R5).

    uv run python scripts/analyze.py && uv run python scripts/make_figures.py

Writes docs/figures/*.png. Captions: docs/results.md. Curves are IQM over seeds with 95 %
stratified-bootstrap bands (configs/analysis/default.yaml: seed, n_boot).
"""

from __future__ import annotations

import json
import pathlib

import matplotlib
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from latrl.analysis import stats as S  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
AN = ROOT / "runs" / "_analysis"
OUT = ROOT / "docs" / "figures"

# Categorical slots in fixed order (dataviz reference palette); colour follows the condition.
COLOR = {
    "A": "#2a78d6",
    "C": "#eb6834",
    "D": "#1baf7a",
    "E": "#eda100",
    "F": "#e87ba4",
    "B20": "#008300",
    "B40": "#6250d6",
    "B60": "#e34948",
}
LABEL = {
    "A": "A  0 ms, MLP",
    "C": "C  random, MLP",
    "D": "D  random, history",
    "E": "E  random, GRU",
    "F": "F  random, MLP + true latency",
    "B20": "B20  fixed 20 ms",
    "B40": "B40  fixed 40 ms",
    "B60": "B60  fixed 60 ms",
}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "text.color": INK,
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
    }
)


def cond_of(run_id: pd.Series) -> pd.Series:
    return run_id.str.replace(r"_(mlp|mlppriv|hist\d+|gru)_s\d+_[0-9a-f]+$", "", regex=True)


def iqm_band(ps: pd.DataFrame, cond: str, axis: str, cfg: dict, rng) -> pd.DataFrame:
    sub = ps[(ps["cond"] == cond) & (ps["axis"] == axis)]
    tab = sub.pivot(index="run_id", columns="x_ms", values="success") * 100
    rows = []
    for x in tab.columns:
        e = S.iqm_ci(tab[x].to_numpy(), cfg["n_boot"], rng, cfg["ci"])
        rows.append({"x": x, "iqm": e.point, "lo": e.lo, "hi": e.hi})
    return pd.DataFrame(rows)


def shade_ranges(ax) -> None:
    ax.axvspan(0, 80, color="#efeee9", zorder=0, lw=0)
    ax.axvspan(5, 30, facecolor="none", edgecolor="#c9c8c1", hatch="////", lw=0, zorder=0)
    ax.text(79, 101, "training range (C–F)", ha="right", va="bottom", fontsize=7, color=INK2)
    ax.text(17.5, 101, "typical\nproprio.", ha="center", va="bottom", fontsize=7, color=INK2)


def fig_hero(ps, pr, cfg) -> None:
    rng = np.random.default_rng(cfg["seed"])
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True, constrained_layout=True)
    panels = [("A", "C", "D", "E", "F"), ("A", "B20", "B40", "B60")]
    titles = ["Main conditions", "Fixed-delay specialists"]
    for ax, conds, title in zip(axes, panels, titles, strict=True):
        shade_ranges(ax)
        for c in conds:
            b = iqm_band(ps, c, "action", cfg, rng)
            n = int((pr["cond"] == c).sum())
            ax.fill_between(b["x"], b["lo"], b["hi"], color=COLOR[c], alpha=0.18, lw=0)
            ax.plot(b["x"], b["iqm"], color=COLOR[c], lw=2, label=f"{LABEL[c]}  (n={n})")
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlabel("test action latency (ms)")
        ax.set_xlim(0, 120)
        ax.set_ylim(0, 108)
        ax.set_xticks(range(0, 121, 20))
        ax.legend(loc="upper right", fontsize=7.5, bbox_to_anchor=(1.0, 0.93))
    axes[0].set_ylabel("success rate (%)  IQM over seeds, 95 % CI")
    fig.savefig(OUT / "fig1_success_vs_action_latency.png", dpi=200)
    plt.close(fig)


def fig_learning(pr, cfg) -> None:
    """Final return per seed (left) and eval-return curves (right): walk vs stand."""
    order = [
        "A",
        "B20",
        "B40",
        "B60",
        "C",
        "D",
        "E",
        "F",
        "C_obs",
        "D_obs",
        "E_obs",
        "C_tm",
        "D_tm",
        "E_tm",
        "DH2",
        "DH8",
    ]
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(10, 3.6), constrained_layout=True, gridspec_kw={"width_ratios": [1.7, 1]}
    )
    rng = np.random.default_rng(cfg["seed"])
    for i, c in enumerate(order):
        v = pr[pr["cond"] == c]["final_return"].to_numpy()
        x = i + rng.uniform(-0.18, 0.18, v.size)
        ax.scatter(x, v, s=22, color=COLOR.get(c, INK2), edgecolor=SURFACE, lw=0.8, zorder=3)
        walk = int((v > 25).sum())
        ax.text(i, 32.4, f"{walk}/{v.size}", ha="center", fontsize=7, color=INK2)
    ax.axhline(25, color=INK2, lw=0.8, ls="--")
    ax.text(len(order) - 0.5, 25.3, "walks", ha="right", va="bottom", fontsize=7, color=INK2)
    ax.text(len(order) - 0.5, 24.7, "stands", ha="right", va="top", fontsize=7, color=INK2)
    ax.set_xticks(range(len(order)), order, rotation=45, ha="right")
    ax.set_ylabel("final eval return (200M steps)")
    ax.set_ylim(17, 33.5)
    ax.set_title("Training outcome per seed (walking seeds / seeds)", loc="left", fontsize=10)
    ax.grid(axis="x", visible=False)

    for c in ("A", "C", "E"):
        for r in pr[pr["cond"] == c].itertuples():
            ev = [json.loads(line) for line in (ROOT / "runs" / r.run_id / "eval.jsonl").open()]
            ax2.plot(
                [e["env_steps"] / 1e6 for e in ev],
                [e["eval/episode_return"] for e in ev],
                color=COLOR[c],
                lw=1.2,
                alpha=0.85,
                label=LABEL[c] if r.seed == 0 else None,
            )
    ax2.set_xlabel("environment steps (M)")
    ax2.set_ylabel("eval return")
    ax2.set_title("Learning curves, every seed", loc="left", fontsize=10)
    ax2.legend(fontsize=7.5, loc="lower right")
    fig.savefig(OUT / "fig2_training_outcome.png", dpi=200)
    plt.close(fig)


def fig_channels(ps, pr, cfg) -> None:
    """H5: each policy on the action and the observation axis (20 ms grid)."""
    rng = np.random.default_rng(cfg["seed"])
    pols = [("C", "E"), ("C_obs", "E_obs")]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), sharey=True, constrained_layout=True)
    names = {
        "C": "C (action-trained)",
        "E": "E (action-trained)",
        "C_obs": "C_obs (obs-trained)",
        "E_obs": "E_obs (obs-trained)",
    }
    colors = {"C": COLOR["C"], "E": COLOR["E"], "C_obs": COLOR["C"], "E_obs": COLOR["E"]}
    for ax, group in zip(axes, pols, strict=True):
        for p in group:
            for axis, ls in (("action", "-"), ("obs", "--")):
                b = iqm_band(ps, p, axis, cfg, rng)
                b = b[np.isclose(b["x"] % 20, 0)]
                ax.plot(
                    b["x"],
                    b["iqm"],
                    color=colors[p],
                    ls=ls,
                    lw=2,
                    marker="o",
                    ms=4,
                    label=f"{names[p]}, {axis} latency",
                )
        ax.set_xlabel("test latency (ms), 20 ms grid")
        ax.set_xticks(range(0, 121, 20))
        ax.set_ylim(0, 105)
        ax.legend(fontsize=7.5, loc="lower left")
    axes[0].set_ylabel("success rate (%)  IQM over seeds")
    axes[0].set_title("Trained on action delay", loc="left", fontsize=10)
    axes[1].set_title("Trained on observation delay", loc="left", fontsize=10)
    fig.savefig(OUT / "fig3_action_vs_obs_channel.png", dpi=200)
    plt.close(fig)


def main() -> None:
    cfg = OmegaConf.to_container(OmegaConf.load(ROOT / "configs/analysis/default.yaml"))
    OUT.mkdir(parents=True, exist_ok=True)
    ps = pd.read_parquet(AN / "per_setting.parquet")
    ps["cond"] = cond_of(ps["run_id"])
    pr = pd.read_csv(AN / "per_run.csv")
    fig_hero(ps, pr, cfg)
    fig_learning(pr, cfg)
    fig_channels(ps, pr, cfg)
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
