"""Step 2a: match-level feature table (one row per match).

Every historical feature is computed ONLY from matches played before the current one
(the loop updates its state after reading the features), so there's no leakage.
Orientation: team "A" = side batting first, team "B" = side batting second.
Batting order is fixed at the toss, so it counts as pre-match information.

Output: data/processed/match_features.csv
Run:    python src/features_match.py   (after clean.py)
"""
from collections import defaultdict, deque

import numpy as np
import pandas as pd

import config as C

HOME_CITIES = {
    "Chennai Super Kings": {"Chennai"}, "Delhi Capitals": {"Delhi"},
    "Deccan Chargers": {"Hyderabad"}, "Sunrisers Hyderabad": {"Hyderabad"},
    "Gujarat Lions": {"Rajkot"}, "Gujarat Titans": {"Ahmedabad"},
    "Kochi Tuskers Kerala": {"Kochi"}, "Kolkata Knight Riders": {"Kolkata"},
    "Lucknow Super Giants": {"Lucknow"}, "Mumbai Indians": {"Mumbai"},
    "Pune Warriors": {"Pune"}, "Rising Pune Supergiant": {"Pune"},
    "Punjab Kings": {"Chandigarh", "Mohali", "Dharamsala"},
    "Rajasthan Royals": {"Jaipur"}, "Royal Challengers Bengaluru": {"Bengaluru"},
}

ELO_K, ELO_START, ELO_SEASON_REVERT = 20.0, 1500.0, 1 / 3
VENUE_PRIOR_K = 5          # pseudo-matches of shrinkage toward the league average
PLAYER_WINDOW = 10         # innings used for a player's recent form


# ---------------------------------------------------------------- innings snapshots
def innings_snapshots(d: pd.DataFrame) -> pd.DataFrame:
    """Runs / wickets / boundaries after n overs, plus runs in the 3 overs before n."""
    out = []
    for n in C.SNAPSHOT_OVERS:
        upto = d[d["over"] < n]
        last3 = d[(d["over"] >= n - 3) & (d["over"] < n)]
        g = upto.groupby(["match_id", "inning"]).agg(
            runs=("total_runs", "sum"), wkts=("is_wicket", "sum"),
            fours=("is_four", "sum"), sixes=("is_six", "sum"), dots=("is_dot", "sum"),
            extras=("extra_runs", "sum"), balls=("is_legal", "sum"))
        g["last3"] = last3.groupby(["match_id", "inning"])["total_runs"].sum()
        g["rr"] = g["runs"] / (g["balls"] / 6).clip(lower=1 / 6)
        g = g.drop(columns="balls")
        g.columns = [f"{c}_{n}" for c in g.columns]
        out.append(g)
    s = pd.concat(out, axis=1).fillna(0).reset_index()
    w = s.pivot(index="match_id", columns="inning")
    w.columns = [f"inn{i}_{c}" for c, i in w.columns]
    return w.reset_index()


# ---------------------------------------------------------------- player-level inputs
def player_innings(d: pd.DataFrame):
    """Per-match batting & bowling lines, plus batting order (first appearance)."""
    bat = d.groupby(["match_id", "batting_team", "batter"]).agg(
        runs=("batsman_runs", "sum"), balls=("faced_by_batter", "sum")).reset_index()
    # batting order = row at which a player first appears as striker or non-striker
    ap = pd.concat([
        d[["match_id", "batting_team", "batter"]].assign(idx=d.index),
        d[["match_id", "batting_team", "non_striker"]].rename(columns={"non_striker": "batter"})
        .assign(idx=d.index)])
    order = ap.groupby(["match_id", "batting_team", "batter"])["idx"].min().reset_index()
    order["pos"] = order.groupby(["match_id", "batting_team"])["idx"].rank(method="first")
    bat = bat.merge(order.drop(columns="idx"), on=["match_id", "batting_team", "batter"])
    bowl = d.groupby(["match_id", "bowling_team", "bowler"]).agg(
        runs=("bowler_runs", "sum"), balls=("is_legal", "sum"),
        wkts=("is_bowler_wicket", "sum")).reset_index()
    return bat, bowl


def lineup_strength(batters, bowlers, bat_hist, bowl_hist):
    """Pooled recent form of a lineup: batting SR, runs/innings; bowling economy, SR."""
    r = b = n = 0
    for p in batters:
        for runs, balls in bat_hist.get(p, ()):
            r, b, n = r + runs, b + balls, n + 1
    br = bb = bw = 0
    for p in bowlers:
        for runs, balls, w in bowl_hist.get(p, ()):
            br, bb, bw = br + runs, bb + balls, bw + w
    return {
        "lineup_bat_sr": 100 * r / b if b >= 30 else np.nan,
        "lineup_bat_rpi": r / n if n >= 5 else np.nan,
        "lineup_bowl_econ": 6 * br / bb if bb >= 60 else np.nan,
        "lineup_bowl_sr": bb / bw if bw >= 3 else np.nan,
    }


# ---------------------------------------------------------------- main loop
def build(m: pd.DataFrame, d: pd.DataFrame) -> pd.DataFrame:
    bat, bowl = player_innings(d)
    bat_by_match = {k: g for k, g in bat.groupby("match_id")}
    bowl_by_match = {k: g for k, g in bowl.groupby("match_id")}

    results = defaultdict(list)                 # team -> list of 1/0 (decided matches)
    scored = defaultdict(lambda: deque(maxlen=5))
    conceded = defaultdict(lambda: deque(maxlen=5))
    season_rec = defaultdict(lambda: [0, 0])    # (team, season) -> [wins, played]
    h2h = defaultdict(lambda: [0, 0])           # (team, opp) -> [wins, played]
    elo = defaultdict(lambda: ELO_START)
    elo_season = {}
    venue_inn1 = defaultdict(list)
    venue_bf = defaultdict(list)                # 1 if bat-first side won
    league_inn1, league_bf = [], []
    bat_hist = defaultdict(lambda: deque(maxlen=PLAYER_WINDOW))
    bowl_hist = defaultdict(lambda: deque(maxlen=PLAYER_WINDOW))
    prev_lineup = {}                            # team -> (top-7 batters, bowlers)

    rows = []
    for r in m.itertuples(index=False):
        A, B, season = r.bat_first_team, r.bat_second_team, r.season
        if pd.isna(A):            # abandoned without a ball bowled
            continue
        for t in (A, B):          # pull Elo a third of the way back to 1500 each new season
            if elo_season.get(t) != season:
                if t in elo_season:
                    elo[t] = elo[t] + ELO_SEASON_REVERT * (ELO_START - elo[t])
                elo_season[t] = season

        f = {"match_id": r.match_id}
        for side, t in (("A", A), ("B", B)):
            res = results[t]
            f[f"{side}_n_prior"] = len(res)
            for k in C.FORM_WINDOWS:
                f[f"{side}_form{k}"] = np.mean(res[-k:]) if res else np.nan
            w, p = season_rec[(t, season)]
            f[f"{side}_season_winrate"] = w / p if p else np.nan
            f[f"{side}_avg_scored5"] = np.mean(scored[t]) if scored[t] else np.nan
            f[f"{side}_avg_conceded5"] = np.mean(conceded[t]) if conceded[t] else np.nan
            f[f"{side}_elo"] = elo[t]
            f[f"{side}_home"] = int(r.city in HOME_CITIES.get(t, set()))
            bats, bowls = prev_lineup.get(t, ((), ()))
            for k, v in lineup_strength(bats, bowls, bat_hist, bowl_hist).items():
                f[f"{side}_{k}"] = v

        hw, hp = h2h[(A, B)]
        f["h2h_n"] = hp
        f["h2h_A_winrate"] = hw / hp if hp else np.nan
        f["h2h_A_winrate_smooth"] = (hw + 1) / (hp + 2)        # Laplace smoothing

        L1 = np.mean(league_inn1) if league_inn1 else 160.0
        LB = np.mean(league_bf) if league_bf else 0.5
        vi, vb = venue_inn1[r.venue], venue_bf[r.venue]
        f["venue_n_prior"] = len(vb)
        f["venue_avg_inn1"] = np.mean(vi) if vi else np.nan
        f["venue_avg_inn1_smooth"] = (np.sum(vi) + VENUE_PRIOR_K * L1) / (len(vi) + VENUE_PRIOR_K)
        f["venue_batfirst_winrate"] = np.mean(vb) if vb else np.nan
        f["venue_batfirst_winrate_smooth"] = (np.sum(vb) + VENUE_PRIOR_K * LB) / (len(vb) + VENUE_PRIOR_K)
        f["league_avg_inn1_prior"] = L1
        rows.append(f)

        # ---------- update state with this match's outcome (after features are read)
        if not pd.isna(r.winner):
            a_won = int(r.winner == A)
            results[A].append(a_won); results[B].append(1 - a_won)
            for t, wv in ((A, a_won), (B, 1 - a_won)):
                season_rec[(t, season)][0] += wv; season_rec[(t, season)][1] += 1
            h2h[(A, B)][0] += a_won; h2h[(A, B)][1] += 1
            h2h[(B, A)][0] += 1 - a_won; h2h[(B, A)][1] += 1
            ea = 1 / (1 + 10 ** ((elo[B] - elo[A]) / 400))
            elo[A] += ELO_K * (a_won - ea); elo[B] -= ELO_K * (a_won - ea)
        if r.usable:
            venue_inn1[r.venue].append(r.inn1_runs); league_inn1.append(r.inn1_runs)
            venue_bf[r.venue].append(r.bat_first_won); league_bf.append(r.bat_first_won)
            scored[A].append(r.inn1_runs); conceded[A].append(r.inn2_runs)
            scored[B].append(r.inn2_runs); conceded[B].append(r.inn1_runs)
        mb = bat_by_match.get(r.match_id)
        mw = bowl_by_match.get(r.match_id)
        if mb is not None:
            for x in mb.itertuples(index=False):
                bat_hist[x.batter].append((x.runs, x.balls))
            for t in (A, B):
                top7 = mb[mb["batting_team"] == t].sort_values("pos")["batter"].head(7).tolist()
                bw = mw[mw["bowling_team"] == t]["bowler"].tolist() if mw is not None else []
                prev_lineup[t] = (top7, bw)
        if mw is not None:
            for x in mw.itertuples(index=False):
                bowl_hist[x.bowler].append((x.runs, x.balls, x.wkts))

    return pd.DataFrame(rows)


def main():
    m = pd.read_csv(C.PROCESSED / "matches_clean.csv", parse_dates=["date"])
    d = pd.read_csv(C.PROCESSED / "deliveries_clean.csv", low_memory=False)

    hist = build(m, d)
    snaps = innings_snapshots(d)

    keep = ["match_id", "season", "date", "match_type", "is_playoff", "venue", "city",
            "bat_first_team", "bat_second_team", "toss_winner", "toss_decision",
            "toss_winner_bats_first", "played_abroad", "impact_player_era",
            "usable", "is_dl", "is_reduced_overs", "is_tie", "is_no_result",
            "inn1_runs", "inn1_wkts", "inn2_runs", "inn2_wkts", "target_runs",
            "winner", "bat_first_won", "toss_winner_won"]
    f = m[keep].merge(hist, on="match_id", how="inner").merge(snaps, on="match_id", how="left")

    # convenience differences (A minus B) for the classifiers
    for c in ["form5", "form10", "season_winrate", "avg_scored5", "avg_conceded5", "elo",
              "home", "lineup_bat_sr", "lineup_bat_rpi", "lineup_bowl_econ", "lineup_bowl_sr"]:
        f[f"diff_{c}"] = f[f"A_{c}"] - f[f"B_{c}"]
    f["toss_decision_field"] = (f["toss_decision"] == "field").astype(int)
    # chase state: runs still needed and required rate after n overs of the 2nd innings
    for n in C.SNAPSHOT_OVERS:
        f[f"inn2_need_{n}"] = f["target_runs"] - f[f"inn2_runs_{n}"]
        f[f"inn2_req_rr_{n}"] = f[f"inn2_need_{n}"] / (20 - n)
        # chase already over by over n: the state reveals the result, so flag it
        f[f"inn2_done_by_{n}"] = ((f["inn2_runs_{0}".format(n)] >= f["target_runs"])
                                  | (f[f"inn2_wkts_{n}"] >= 10)).astype(int)
    f = f.rename(columns={"inn1_runs": "inn1_total", "inn2_runs": "inn2_total"})
    f.to_csv(C.PROCESSED / "match_features.csv", index=False)

    print(f"match_features: {f.shape[0]} rows x {f.shape[1]} cols  (usable: {int(f['usable'].sum())})")
    print("  missing share in key pre-match features (first matches of each team are NaN):")
    for c in ["A_form5", "h2h_A_winrate", "venue_avg_inn1", "A_lineup_bat_sr", "A_lineup_bowl_econ"]:
        print(f"    {c:24s} {f[c].isna().mean():.1%}")


if __name__ == "__main__":
    main()
