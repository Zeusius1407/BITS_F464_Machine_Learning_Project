"""Step 1: clean the raw Kaggle IPL tables.

Outputs (data/processed/):
  matches_clean.csv     one row per match, canonical teams/venues, result flags
  deliveries_clean.csv  one row per ball (innings 1-2 only), legal-ball and credit flags
  super_overs.csv       super-over balls, kept separately (not used in stats)

Run:  python src/clean.py
"""
import numpy as np
import pandas as pd

import config as C


def canon_team(s: pd.Series) -> pd.Series:
    s = s.replace(C.TEAM_ALIASES)
    if C.MERGE_DECCAN_INTO_SRH:
        s = s.replace({"Deccan Chargers": "Sunrisers Hyderabad"})
    return s


def canon_venue(v: str) -> str:
    v = v.strip()
    # strip trailing ", City" parts repeatedly (e.g. "..., Mohali, Chandigarh")
    while "," in v and v.rsplit(",", 1)[1].strip() in C.KNOWN_CITY_SUFFIXES:
        v = v.rsplit(",", 1)[0].strip()
    return C.VENUE_ALIASES.get(v, v)


def clean_matches(m: pd.DataFrame) -> pd.DataFrame:
    m = m.rename(columns={"id": "match_id"}).copy()
    m["date"] = pd.to_datetime(m["date"])
    # "2007/08" style labels -> calendar year the season was played in
    m["season"] = m["date"].dt.year
    for c in ["team1", "team2", "toss_winner", "winner"]:
        m[c] = canon_team(m[c])
    m["venue"] = m["venue"].map(canon_venue)

    # fill missing city from the venue's most common city, then a manual fallback
    city_of_venue = m.dropna(subset=["city"]).groupby("venue")["city"].agg(lambda x: x.mode()[0])
    m["city"] = m["city"].fillna(m["venue"].map(city_of_venue))
    m["city"] = m["city"].fillna(m["venue"].map({
        "Dubai International Cricket Stadium": "Dubai",
        "Sharjah Cricket Stadium": "Sharjah",
    }))
    m["city"] = m["city"].replace({"Bangalore": "Bengaluru"})

    m["is_no_result"] = (m["result"] == "no result").astype(int)
    m["is_dl"] = (m["method"] == "D/L").astype(int)
    m["is_reduced_overs"] = (m["target_overs"].fillna(0) < 20).astype(int)
    m["is_tie"] = (m["result"] == "tie").astype(int)      # winner = super-over winner
    m["is_playoff"] = (m["match_type"] != "League").astype(int)
    m["played_abroad"] = m["season"].isin([2009, 2014, 2020, 2021]) & m["city"].isin(
        ["Cape Town", "Port Elizabeth", "Durban", "Centurion", "East London", "Johannesburg",
         "Kimberley", "Bloemfontein", "Abu Dhabi", "Dubai", "Sharjah"])
    m["played_abroad"] = m["played_abroad"].astype(int)
    m["impact_player_era"] = (m["season"] >= C.IMPACT_PLAYER_FROM).astype(int)
    # a clean, full 20-over, decided match: what the models should train on
    m["usable"] = ((m["is_no_result"] == 0) & (m["is_dl"] == 0)
                   & (m["is_reduced_overs"] == 0)).astype(int)
    m = m.drop(columns=["method", "umpire1", "umpire2"])
    return m.sort_values(["date", "match_id"]).reset_index(drop=True)


def clean_deliveries(d: pd.DataFrame):
    d = d.copy()
    for c in ["batting_team", "bowling_team"]:
        d[c] = canon_team(d[c])
    for c in ["extras_type", "player_dismissed", "dismissal_kind", "fielder"]:
        d[c] = d[c].replace("NA", np.nan)

    sup = d[d["inning"] > 2].copy()
    d = d[d["inning"] <= 2].copy()

    et = d["extras_type"]
    d["is_wide"] = (et == "wides").astype(int)
    d["is_noball"] = (et == "noballs").astype(int)
    d["is_legal"] = ((d["is_wide"] == 0) & (d["is_noball"] == 0)).astype(int)
    d["faced_by_batter"] = (d["is_wide"] == 0).astype(int)          # no-balls count as faced
    # runs charged to the bowler: everything except byes / leg-byes / penalty
    d["bowler_runs"] = d["total_runs"] - np.where(et.isin(["byes", "legbyes", "penalty"]),
                                                  d["extra_runs"], 0)
    dk = d["dismissal_kind"]
    d["is_bowler_wicket"] = ((d["is_wicket"] == 1) & ~dk.isin(C.NON_BOWLER_WICKETS)).astype(int)
    d["is_out"] = ((d["is_wicket"] == 1) & ~dk.isin(C.NOT_AN_OUT)).astype(int)
    d["is_four"] = (d["batsman_runs"] == 4).astype(int)
    d["is_six"] = (d["batsman_runs"] == 6).astype(int)
    d["is_dot"] = ((d["total_runs"] == 0) & (d["is_legal"] == 1)).astype(int)
    d["phase"] = pd.cut(d["over"], bins=[-1, 5, 14, 19], labels=["pp", "mid", "death"]).astype(str)
    d = d.sort_values(["match_id", "inning", "over", "ball"], kind="stable").reset_index(drop=True)
    # running count of legal balls in the innings (1..120) = position in the innings
    d["legal_ball_no"] = d.groupby(["match_id", "inning"])["is_legal"].cumsum()
    return d, sup


def add_innings_summary(m: pd.DataFrame, d: pd.DataFrame) -> pd.DataFrame:
    g = d.groupby(["match_id", "inning"]).agg(
        team=("batting_team", "first"), runs=("total_runs", "sum"),
        wkts=("is_wicket", "sum"), legal_balls=("is_legal", "sum")).reset_index()
    w = g.pivot(index="match_id", columns="inning")
    w.columns = [f"inn{i}_{c}" for c, i in w.columns]
    w = w.rename(columns={"inn1_team": "bat_first_team", "inn2_team": "bat_second_team"})
    m = m.merge(w.reset_index(), on="match_id", how="left")
    # matches with no 2nd innings (abandoned): infer chasing side from team1/team2
    miss = m["bat_second_team"].isna() & m["bat_first_team"].notna()
    m.loc[miss, "bat_second_team"] = np.where(m.loc[miss, "team1"] == m.loc[miss, "bat_first_team"],
                                              m.loc[miss, "team2"], m.loc[miss, "team1"])
    m["toss_winner_bats_first"] = (m["toss_winner"] == m["bat_first_team"]).astype(int)
    m["bat_first_won"] = np.where(m["winner"].isna(), np.nan,
                                  (m["winner"] == m["bat_first_team"]).astype(float))
    m["toss_winner_won"] = np.where(m["winner"].isna(), np.nan,
                                    (m["winner"] == m["toss_winner"]).astype(float))
    return m


def sanity_checks(m: pd.DataFrame) -> None:
    # toss decision must agree with who actually batted first
    expected = np.where(m["toss_decision"] == "bat", 1, 0)
    bad = (m["bat_first_team"].notna()) & (m["toss_winner_bats_first"] != expected)
    print(f"  toss/batting-order mismatches: {int(bad.sum())}")
    # first-innings total + 1 should equal the target in full matches
    full = m[(m["usable"] == 1)]
    off = (full["inn1_runs"] + 1 != full["target_runs"]).sum()
    print(f"  usable matches where inn1_runs+1 != target_runs: {int(off)}")


def main():
    C.PROCESSED.mkdir(parents=True, exist_ok=True)
    m_raw = pd.read_csv(C.RAW / "matches.csv")
    d_raw = pd.read_csv(C.RAW / "deliveries.csv", low_memory=False)

    m = clean_matches(m_raw)
    d, sup = clean_deliveries(d_raw)
    m = add_innings_summary(m, d)
    d = d.merge(m[["match_id", "season", "date", "venue"]], on="match_id", how="left")

    print("Cleaning summary")
    print(f"  matches: {len(m)}  | usable (decided, full 20 overs, no D/L): {m['usable'].sum()}")
    print(f"  no result: {m['is_no_result'].sum()}  D/L: {m['is_dl'].sum()}  "
          f"reduced overs: {m['is_reduced_overs'].sum()}  ties: {m['is_tie'].sum()}")
    print(f"  teams: {m['team1'].nunique()}  venues: {m['venue'].nunique()} (raw {m_raw['venue'].nunique()})")
    print(f"  deliveries kept: {len(d)}  super-over balls set aside: {len(sup)}")
    sanity_checks(m)

    m.to_csv(C.PROCESSED / "matches_clean.csv", index=False)
    d.to_csv(C.PROCESSED / "deliveries_clean.csv", index=False)
    sup.to_csv(C.PROCESSED / "super_overs.csv", index=False)


if __name__ == "__main__":
    main()
