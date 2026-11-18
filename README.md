# IPL Match & Player Analytics: data pipeline

Turns the raw Kaggle IPL tables (2008–2024) into two modelling-ready datasets:

- **match_features.csv**: one row per match, for the regression and classification work. This covers the first-innings total, the winner, and toss/venue.
- **player_season.csv**: one row per player per season, for PCA and clustering of player roles.

## Run

```bash
pip install -r requirements.txt
./run_all.sh                   # everything, ~70 s
# or step by step:
python src/clean.py            # -> matches_clean.csv, deliveries_clean.csv, super_overs.csv
python src/features_match.py   # -> match_features.csv
python src/features_player.py  # -> player_season.csv, player_career.csv
python src/models_innings.py   # first-innings total regression
python src/models_winner.py    # LogReg / KNN / CART / PCA+LogReg winner models
python src/toss_bayes.py       # toss & venue tests, Bayesian bat-or-field rule
python src/player_roles.py     # PCA + K-Means / Ward player roles + validation
```
Results go to `results/` and `results/figures/`. **See `REPORT.md` for the findings.**
Put `matches.csv` and `deliveries.csv` in `data/raw/`. The full run takes about 10 seconds. Outputs go to `data/processed/`.

## Cleaning decisions (`src/clean.py`, `src/config.py`)

| Issue | What was done |
|---|---|
| Season labels like `2007/08`, `2020/21` | `season` = calendar year from `date` (2008 … 2024) |
| Franchise renames | Delhi Daredevils → Delhi Capitals, Kings XI Punjab → Punjab Kings, RC Bangalore → RC Bengaluru, Rising Pune Supergiants → Supergiant |
| Deccan Chargers vs SRH | Kept **separate** (different franchises). Set `MERGE_DECCAN_INTO_SRH = True` in `config.py` to merge them |
| 58 raw venue strings | Normalised to 36 grounds (city suffixes stripped; Feroz Shah Kotla → Arun Jaitley, Motera → Narendra Modi, Subrata Roy Sahara → MCA Pune, etc.) |
| Missing `city` (51 rows) | Filled from the venue |
| No result (5), D/L (21), reduced-overs (33 total) | Kept, but flagged. **`usable = 1`** marks the 1,061 decided, full 20-over, non-D/L matches. Train on these |
| Ties (14) | Kept. `winner` = super-over winner, `is_tie = 1` |
| Super-over balls (161) | Removed from all stats and saved to `super_overs.csv` |
| Legal balls | Wides and no-balls are not legal. A batter's balls faced excludes wides only |
| Bowler credit | Runs conceded exclude byes, leg-byes and penalty runs. Bowler wickets exclude run out, retired hurt/out and obstructing the field |
| Overs | 0-indexed in the raw data. Phases: powerplay = overs 0–5, middle = 6–14, death = 15–19 |

Built-in checks (printed when the script runs): 0 toss/batting-order mismatches, and first-innings total + 1 equals `target_runs` in every usable match.
Spot checks: each season's top run-scorer and top wicket-taker from `player_season.csv` match the official Orange and Purple Cap figures for 2008–2024.

## match_features.csv

**Orientation:** **A = team batting first, B = team batting second.** Batting order is fixed at the toss, so it counts as pre-match information. Main targets: `inn1_total` (regression) and `bat_first_won` (classification).

**No leakage:** every historical feature is computed only from matches *before* the current one. A form value recomputed from scratch matched the pipeline's. The first matches of each team, venue and pairing have NaN history, about 1–8% depending on the feature. Impute these, or drop the 2008 season. Use a **season-based train/test split**, not a random one.

| Group | Columns |
|---|---|
| Context | `season, date, match_type, is_playoff, venue, city, played_abroad, impact_player_era` |
| Toss | `toss_winner, toss_decision, toss_decision_field, toss_winner_bats_first` |
| Team form (A_ / B_) | `form5, form10` (win rate, last 5/10), `season_winrate`, `avg_scored5, avg_conceded5`, `elo`, `home`, `n_prior` |
| Player form (A_ / B_) | `lineup_bat_sr, lineup_bat_rpi, lineup_bowl_econ, lineup_bowl_sr`: pooled last-10-innings form of the team's **previous-match** top-7 batters and bowlers. The previous lineup is used because the current scorecard would leak information |
| Differences | `diff_*` = A minus B for each of the above |
| Head-to-head | `h2h_n, h2h_A_winrate, h2h_A_winrate_smooth` (Laplace smoothing) |
| Venue | `venue_n_prior, venue_avg_inn1, venue_batfirst_winrate` plus `_smooth` versions, shrunk toward the league average with 5 pseudo-matches. Use the smoothed versions for grounds with few matches |
| 1st-innings state (n = 6, 10, 15) | `inn1_{runs,wkts,rr,last3,fours,sixes,dots,extras}_{n}` |
| 2nd-innings state | the same set for `inn2_`, plus `inn2_need_{n}`, `inn2_req_rr_{n}`, `inn2_done_by_{n}`. That last one flags chases already over by over n, which you should exclude from in-play models |
| Outcomes | `inn1_total, inn1_wkts, inn2_total, inn2_wkts, target_runs, winner, bat_first_won, toss_winner_won` |

## player_season.csv  (`player_career.csv` has the same columns with all seasons pooled)

| Group | Columns |
|---|---|
| ID | `season, player, teams, matches` |
| Batting | `bat_innings, runs, balls_faced, outs, bat_avg, bat_sr, balls_per_inn, boundary_pct, boundary_run_pct, six_pct, dot_pct_bat, fifties, hundreds, highest` |
| Batting role signals | `avg_bat_pos`, `avg_entry_ball` (legal balls already bowled when the batter arrived: 0 = opener, 90+ = finisher), `sr_pp/mid/death`, `share_balls_pp/mid/death` |
| Bowling | `bowl_innings, balls_bowled, overs, runs_conceded, wickets, economy, bowl_avg, bowl_sr, dot_pct_bowl, boundary_pct_conceded, overs_per_inn, wides, noballs` |
| Bowling role signals | `econ_pp/mid/death`, `share_overs_pp/mid/death`, `wkts_pp/mid/death` |
| Filters | `qualified_bat` (≥60 balls faced), `qualified_bowl` (≥120 balls bowled) |

`bat_avg` is NaN when a player was never dismissed, and phase SRs are NaN when no balls were faced in that phase. Impute these or filter on the qualified flags before PCA.

## Early numbers (usable matches)
- Team batting first won 46.1%. The toss winner won 50.7%, so the toss alone looks close to a coin-flip. Venue-level analysis (Bayesian) is where any signal will show up.
- Correlation of first-innings total with runs after 6, 10 and 15 overs: 0.58, 0.73, 0.89. With the smoothed venue average alone: 0.12.
- The pre-match Elo favourite wins only 52.9%. Expect modest pre-match accuracy.
