"""Shared helpers for the modelling scripts: data loading, temporal split, plot style."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C

RESULTS = C.ROOT / "results"
FIGS = RESULTS / "figures"
RESULTS.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)

TEST_SEASONS = (2022, 2023, 2024)     # train on 2008-2021, test on the last three seasons
SEED = 42

# reference categorical palette (fixed order) + text/surface tokens
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TEXT, TEXT2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT2, "xtick.color": TEXT2, "ytick.color": TEXT2,
    "text.color": TEXT, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
    "axes.titlesize": 11, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.frameon": False, "legend.fontsize": 8, "lines.linewidth": 2, "figure.dpi": 130,
    "axes.axisbelow": True, "font.family": "DejaVu Sans",
})


def load_matches(usable_only=True) -> pd.DataFrame:
    f = pd.read_csv(C.PROCESSED / "match_features.csv", parse_dates=["date"])
    f = f.sort_values(["date", "match_id"]).reset_index(drop=True).copy()
    # leakage-safe "recent scoring climate": mean first-innings total of the previous
    # 60 usable matches (captures era shifts such as the 2023 Impact Player rule)
    u = f["usable"] == 1
    roll = f.loc[u, "inn1_total"].rolling(60, min_periods=10).mean().shift(1)
    f["recent_league_inn1"] = roll.reindex(f.index).ffill()
    f["recent_league_inn1"] = f["recent_league_inn1"].fillna(f["league_avg_inn1_prior"])
    if usable_only:
        f = f[u].reset_index(drop=True)
    return f


def split(f: pd.DataFrame):
    test = f["season"].isin(TEST_SEASONS)
    return f[~test].copy(), f[test].copy()


def save_json(obj, name):
    with open(RESULTS / name, "w") as fh:
        json.dump(obj, fh, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else str(o))


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(FIGS / name, bbox_inches="tight")
    plt.close(fig)
