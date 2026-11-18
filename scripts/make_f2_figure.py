"""F2 headline figure from runs/_analysis_f2/ (written by scripts/analyze_f2.py).

    uv run python scripts/analyze_f2.py && uv run python scripts/make_f2_figure.py

Left: success vs. test action latency, IQM over all 10 seeds with 95 % bootstrap bands.
Right: L50 per seed, walking seeds filled, standing seeds hollow (no selection in the left panel).
Writes docs/figures/f2_success_vs_action_latency.png. Same palette and style as make_figures.py.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import make_figures as mf  # noqa: E402  (sets matplotlib style)

plt = mf.plt
ROOT = pathlib.Path(__file__).resolve().parents[1]
AN = ROOT / "runs" / "_analysis_f2"
CONDS = ["A", "C_r40", "D_r40", "E_r40", "F_r40"]
BASE = {c: c.split("_")[0] for c in CONDS}  # colour follows the method
LABEL = {
    "A": "A  0 ms, MLP",
    "C_r40": "C  random 0–40 ms, MLP",
    "D_r40": "D  random 0–40 ms, history",
    "E_r40": "E  random 0–40 ms, GRU",
    "F_r40": "F  random 0–40 ms, MLP + true latency",
}


def main() -> None:
    cfg = OmegaConf.to_container(OmegaConf.load(ROOT / "configs/analysis/default.yaml"))
    ps = pd.read_parquet(AN / "per_setting.parquet")
    ps["cond"] = mf.cond_of(ps["run_id"])
    pr = pd.read_csv(AN / "per_run.csv")
    rng = np.random.default_rng(cfg["seed"])

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(10, 3.7), constrained_layout=True, gridspec_kw={"width_ratios": [1.8, 1]}
    )
    ax.axvspan(0, 40, color="#efeee9", zorder=0, lw=0)
    ax.text(39, 101, "training range (C–F)", ha="right", va="bottom", fontsize=7, color=mf.INK2)
    for c in CONDS:
        b = mf.iqm_band(ps, c, "action", cfg, rng)
        g = pr[pr["cond"] == c]
        col = mf.COLOR[BASE[c]]
        ax.fill_between(b["x"], b["lo"], b["hi"], color=col, alpha=0.18, lw=0)
        lab = f"{LABEL[c]}  ({int(g['walks'].sum())}/{len(g)} walk)"
        ax.plot(b["x"], b["iqm"], color=col, lw=2, label=lab)
    ax.set_xlabel("test action latency (ms)")
    ax.set_ylabel("success rate (%)  IQM over all seeds, 95 % CI")
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 108)
    ax.set_xticks(range(0, 121, 20))
    ax.legend(loc="upper right", fontsize=7.5, bbox_to_anchor=(1.0, 0.93))
    ax.set_title("Success vs. delay, trained on 0–40 ms (F2)", loc="left", fontsize=10)

    for i, c in enumerate(CONDS):
        g = pr[pr["cond"] == c]
        col = mf.COLOR[BASE[c]]
        jit = np.linspace(-0.18, 0.18, len(g))
        for (_, r), j in zip(g.sort_values("seed").iterrows(), jit, strict=True):
            ax2.scatter(
                i + j,
                r["L50"],
                s=34,
                color=col if r["walks"] else mf.SURFACE,
                edgecolor=col,
                linewidth=1.4,
                zorder=3,
            )
    ax2.set_xticks(range(len(CONDS)), [c.split("_")[0] for c in CONDS])
    ax2.set_ylabel("L50 per seed (ms)")
    ax2.set_ylim(-4, 80)
    ax2.set_title("Per seed: filled = walks, hollow = stands", loc="left", fontsize=10)

    out = mf.OUT / "f2_success_vs_action_latency.png"
    fig.savefig(out, dpi=200)
    print("wrote", out)


if __name__ == "__main__":
    main()
