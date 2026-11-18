"""Step 2b: player-season feature table (one row per player per season).

Columns cover batting (volume, average, strike rate, boundary/dot %, phase splits,
entry point) and bowling (economy, strike rate, phase share of overs, death/pp economy).
These are what the PCA + clustering stage uses to separate roles.

Output: data/processed/player_season.csv  (plus player_career.csv, all seasons pooled)
Run:    python src/features_player.py   (after clean.py)
"""
import numpy as np
import pandas as pd

import config as C

MIN_BALLS_FACED = 60       # flags for a "qualified" batting season
MIN_BALLS_BOWLED = 120     # = 20 overs


def safe_div(a, b, scale=1.0):
    return np.where(b > 0, scale * a / np.where(b > 0, b, 1), np.nan)


def batting(d: pd.DataFrame, keys) -> pd.DataFrame:
    # entry point: legal-ball number of the innings when the batter first appears
    # (as striker or non-striker). 0 = opener, ~60 = came in at the halfway mark.
    ap = pd.concat([
        d[["match_id", "inning", "batter", "legal_ball_no", "is_legal"]],
        d[["match_id", "inning", "non_striker", "legal_ball_no", "is_legal"]]
        .rename(columns={"non_striker": "batter"})])
    ap["ball_before"] = ap["legal_ball_no"] - ap["is_legal"]   # balls already bowled
    entry = ap.groupby(["match_id", "inning", "batter"])["ball_before"].min().rename("entry_ball")
    order = entry.reset_index()
    order["bat_pos"] = order.groupby(["match_id", "inning"])["entry_ball"].rank(method="first")

    inn = d.groupby(["match_id", "inning", "batter"]).agg(
        runs=("batsman_runs", "sum"), balls=("faced_by_batter", "sum"),
        fours=("is_four", "sum"), sixes=("is_six", "sum"),
        dots=("is_dot", "sum")).reset_index()
    outs = (d[d["is_out"] == 1].groupby(["match_id", "inning", "player_dismissed"]).size()
            .rename("out").reset_index().rename(columns={"player_dismissed": "batter"}))
    inn = order.merge(inn, on=["match_id", "inning", "batter"], how="left").merge(
        outs, on=["match_id", "inning", "batter"], how="left").fillna(
        {"runs": 0, "balls": 0, "fours": 0, "sixes": 0, "dots": 0, "out": 0})
    inn = inn.merge(d[["match_id", "season"]].drop_duplicates(), on="match_id")
    inn["fifty"] = ((inn["runs"] >= 50) & (inn["runs"] < 100)).astype(int)
    inn["hundred"] = (inn["runs"] >= 100).astype(int)

    g = inn.rename(columns={"batter": "player"}).groupby(keys).agg(
        bat_innings=("match_id", "count"), runs=("runs", "sum"), balls_faced=("balls", "sum"),
        outs=("out", "sum"), fours=("fours", "sum"), sixes=("sixes", "sum"),
        dots_faced=("dots", "sum"), fifties=("fifty", "sum"), hundreds=("hundred", "sum"),
        avg_entry_ball=("entry_ball", "mean"), avg_bat_pos=("bat_pos", "mean"),
        highest=("runs", "max"))
    g["bat_avg"] = safe_div(g["runs"], g["outs"])            # NaN if never dismissed
    g["bat_sr"] = safe_div(g["runs"], g["balls_faced"], 100)
    g["balls_per_inn"] = safe_div(g["balls_faced"], g["bat_innings"])
    g["boundary_pct"] = safe_div(g["fours"] + g["sixes"], g["balls_faced"], 100)
    g["boundary_run_pct"] = safe_div(4 * g["fours"] + 6 * g["sixes"], g["runs"], 100)
    g["six_pct"] = safe_div(g["sixes"], g["balls_faced"], 100)
    g["dot_pct_bat"] = safe_div(g["dots_faced"], g["balls_faced"], 100)

    # phase splits: strike rate in each phase and share of balls faced there
    ph = d.groupby(keys_to_cols(keys, "batter") + ["phase"]).agg(
        r=("batsman_runs", "sum"), b=("faced_by_batter", "sum")).reset_index()
    ph = ph.rename(columns={"batter": "player"}).pivot_table(
        index=keys, columns="phase", values=["r", "b"], fill_value=0)
    for p in C.PHASES:
        b = ph[("b", p)] if ("b", p) in ph else 0
        r = ph[("r", p)] if ("r", p) in ph else 0
        g[f"sr_{p}"] = pd.Series(safe_div(r, b, 100), index=ph.index).reindex(g.index)
        g[f"share_balls_{p}"] = pd.Series(b, index=ph.index).reindex(g.index) / g["balls_faced"]
    return g


def bowling(d: pd.DataFrame, keys) -> pd.DataFrame:
    g = d.rename(columns={"bowler": "player"}).groupby(keys).agg(
        bowl_innings=("match_id", "nunique"), balls_bowled=("is_legal", "sum"),
        runs_conceded=("bowler_runs", "sum"), wickets=("is_bowler_wicket", "sum"),
        dots_bowled=("is_dot", "sum"), wides=("is_wide", "sum"), noballs=("is_noball", "sum"),
        fours_conceded=("is_four", "sum"), sixes_conceded=("is_six", "sum"))
    g["overs"] = g["balls_bowled"] / 6
    g["economy"] = safe_div(g["runs_conceded"], g["balls_bowled"], 6)
    g["bowl_avg"] = safe_div(g["runs_conceded"], g["wickets"])
    g["bowl_sr"] = safe_div(g["balls_bowled"], g["wickets"])
    g["dot_pct_bowl"] = safe_div(g["dots_bowled"], g["balls_bowled"], 100)
    g["boundary_pct_conceded"] = safe_div(g["fours_conceded"] + g["sixes_conceded"],
                                          g["balls_bowled"], 100)
    g["overs_per_inn"] = safe_div(g["overs"], g["bowl_innings"])

    ph = d.groupby(keys_to_cols(keys, "bowler") + ["phase"]).agg(
        r=("bowler_runs", "sum"), b=("is_legal", "sum"), w=("is_bowler_wicket", "sum")).reset_index()
    ph = ph.rename(columns={"bowler": "player"}).pivot_table(
        index=keys, columns="phase", values=["r", "b", "w"], fill_value=0)
    for p in C.PHASES:
        b = ph[("b", p)] if ("b", p) in ph else 0
        g[f"econ_{p}"] = pd.Series(safe_div(ph[("r", p)], b, 6), index=ph.index).reindex(g.index)
        g[f"share_overs_{p}"] = pd.Series(b, index=ph.index).reindex(g.index) / g["balls_bowled"]
        g[f"wkts_{p}"] = ph[("w", p)].reindex(g.index)
    return g


def keys_to_cols(keys, role_col):
    return [role_col if k == "player" else k for k in keys]


def teams_played_for(d, keys):
    t = pd.concat([
        d[["season", "batter", "batting_team"]].set_axis(["season", "player", "team"], axis=1),
        d[["season", "bowler", "bowling_team"]].set_axis(["season", "player", "team"], axis=1)])
    return t.groupby(keys)["team"].agg(lambda s: " / ".join(sorted(set(s))))


def build(d, keys):
    bat, bowl = batting(d, keys), bowling(d, keys)
    out = bat.join(bowl, how="outer")
    mp = pd.concat([d[["match_id", "season", "batter"]].rename(columns={"batter": "player"}),
                    d[["match_id", "season", "non_striker"]].rename(columns={"non_striker": "player"}),
                    d[["match_id", "season", "bowler"]].rename(columns={"bowler": "player"})])
    out.insert(0, "matches", mp.drop_duplicates().groupby(keys)["match_id"].nunique())
    out.insert(1, "teams", teams_played_for(d, keys))
    count_cols = ["bat_innings", "runs", "balls_faced", "bowl_innings", "balls_bowled", "wickets"]
    out[count_cols] = out[count_cols].fillna(0)
    out["qualified_bat"] = (out["balls_faced"] >= MIN_BALLS_FACED).astype(int)
    out["qualified_bowl"] = (out["balls_bowled"] >= MIN_BALLS_BOWLED).astype(int)
    return out.reset_index()


def main():
    d = pd.read_csv(C.PROCESSED / "deliveries_clean.csv", low_memory=False)
    ps = build(d, ["season", "player"])
    pc = build(d, ["player"])
    ps.to_csv(C.PROCESSED / "player_season.csv", index=False)
    pc.to_csv(C.PROCESSED / "player_career.csv", index=False)
    print(f"player_season: {ps.shape[0]} rows x {ps.shape[1]} cols | "
          f"qualified bat: {ps['qualified_bat'].sum()}  qualified bowl: {ps['qualified_bowl'].sum()}")
    print(f"player_career: {pc.shape[0]} players")


if __name__ == "__main__":
    main()
