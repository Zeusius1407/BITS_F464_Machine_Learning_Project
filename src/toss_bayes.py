"""Q3 / Q4 / Objective iv: toss, venue and winning; Bayesian bat-or-field decision.

1. Frequentist tests: does winning the toss help? Does it depend on the decision, team or venue?
   (binomial test, chi-square, logistic regression with a likelihood-ratio test for
   the venue x decision interaction)
2. Bayesian decision theory: for each venue, the bat-first win probability p_v gets a
   Beta prior fitted to all venues (empirical Bayes). Its posterior gives
   E[win | bat] = E[p_v] and E[win | field] = 1 - E[p_v]. Under 0-1 loss the Bayes action
   is the one with the higher posterior expected win probability. P(p_v < 0.5) is
   reported as the confidence that fielding is better.
3. Back-test: for 2015-2024, recommend using only earlier seasons (recency-weighted) and
   check whether toss winners who followed the rule won more often.

Run: python src/toss_bayes.py
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

import common as K

RECENCY_HALF_LIFE = 4.0      # seasons; older matches count half as much every 4 seasons
MIN_VENUE_MATCHES = 15
WEAK_PRIOR = 10.0            # pseudo-matches for the weakly-informative prior


def fit_beta_prior(wins, n):
    """Method-of-moments Beta prior across venues, corrected for binomial noise."""
    p = wins / n
    m = np.average(p, weights=n)
    v_obs = np.average((p - m) ** 2, weights=n)
    v_bin = np.mean(m * (1 - m) / n)
    v_true = max(v_obs - v_bin, 1e-4)
    strength = max(m * (1 - m) / v_true - 1, 2.0)
    return m * strength, (1 - m) * strength


def venue_posteriors(df, weights=None, prior_strength=None):
    """prior_strength=None -> empirical-Bayes prior; a number -> weak prior of that many
    pseudo-matches centred on the league-wide bat-first win rate."""
    w = np.ones(len(df)) if weights is None else weights
    g = (df.assign(w=w, ww=w * df["bat_first_won"]).groupby("venue")
         .agg(n_eff=("w", "sum"), wins_eff=("ww", "sum"), n=("w", "size")))
    if prior_strength is None:
        a0, b0 = fit_beta_prior(g["wins_eff"].values, g["n_eff"].values)
    else:
        m = g["wins_eff"].sum() / g["n_eff"].sum()
        a0, b0 = m * prior_strength, (1 - m) * prior_strength
    g["a"], g["b"] = a0 + g["wins_eff"], b0 + g["n_eff"] - g["wins_eff"]
    g["post_mean_batfirst"] = g["a"] / (g["a"] + g["b"])
    g["ci_low"] = stats.beta.ppf(0.05, g["a"], g["b"])
    g["ci_high"] = stats.beta.ppf(0.95, g["a"], g["b"])
    g["P_field_better"] = stats.beta.cdf(0.5, g["a"], g["b"])
    g["bayes_action"] = np.where(g["post_mean_batfirst"] >= 0.5, "bat", "field")
    g["confidence"] = np.where(g["bayes_action"] == "field", g["P_field_better"], 1 - g["P_field_better"])
    return g, (a0, b0)


def main():
    f = K.load_matches()
    out = {}

    # ---------------- 1. does the toss matter?
    n = len(f); k = int(f["toss_winner_won"].sum())
    bt = stats.binomtest(k, n, 0.5)
    out["toss_winner_win_rate"] = k / n
    out["toss_binom_p"] = bt.pvalue
    by_dec = f.groupby("toss_decision")["toss_winner_won"].agg(["mean", "size"])
    ct = pd.crosstab(f["toss_decision"], f["toss_winner_won"])
    chi_dec = stats.chi2_contingency(ct)
    out["toss_win_rate_by_decision"] = by_dec.to_dict()
    out["chi2_decision_p"] = chi_dec.pvalue
    # bat-first win rate over time
    by_season = f.groupby("season").agg(bat_first_win=("bat_first_won", "mean"),
                                        field_choice=("toss_decision_field", "mean"),
                                        toss_win=("toss_winner_won", "mean"), n=("match_id", "size"))
    # toss -> win, by team (team that won the toss)
    by_team = f.groupby("toss_winner")["toss_winner_won"].agg(["mean", "size"]).sort_values("mean")
    ct_team = pd.crosstab(f["toss_winner"], f["toss_winner_won"])
    out["chi2_team_p"] = stats.chi2_contingency(ct_team).pvalue
    # home advantage (either side at home, other not)
    h = f[(f["A_home"] + f["B_home"]) == 1]
    home_won = np.where(h["A_home"] == 1, h["bat_first_won"], 1 - h["bat_first_won"])
    out["home_win_rate"] = home_won.mean(); out["home_n"] = len(h)
    out["home_binom_p"] = stats.binomtest(int(home_won.sum()), len(h), 0.5).pvalue

    # logistic regression: toss winner wins ~ decision * venue (+ era)
    big = f["venue"].value_counts()
    f["venue_g"] = f["venue"].where(f["venue"].map(big) >= 30, "Other")
    f["era"] = pd.cut(f["season"], [2007, 2014, 2019, 2024], labels=["2008-14", "2015-19", "2020-24"]).astype(str)
    d = f.assign(y=f["toss_winner_won"].astype(int), field=f["toss_decision_field"])
    m0 = smf.logit("y ~ field + C(venue_g) + C(era)", d).fit(disp=0)
    m1 = smf.logit("y ~ field * C(venue_g) + C(era)", d).fit(disp=0)
    m2 = smf.logit("y ~ field * C(era) + C(venue_g)", d).fit(disp=0)
    lr = 2 * (m1.llf - m0.llf); dfree = m1.df_model - m0.df_model
    lr2 = 2 * (m2.llf - m0.llf); df2 = m2.df_model - m0.df_model
    out["lrt_venue_x_decision"] = {"LR": lr, "df": dfree, "p": stats.chi2.sf(lr, dfree)}
    out["lrt_era_x_decision"] = {"LR": lr2, "df": df2, "p": stats.chi2.sf(lr2, df2)}
    out["logit_main_effects"] = m0.summary2().tables[1][["Coef.", "P>|z|"]].round(4).to_dict()

    # ---------------- 2. Bayesian decision per venue (all seasons, recency-weighted)
    last = f["season"].max()
    wts = 0.5 ** ((last - f["season"]) / RECENCY_HALF_LIFE)
    # (a) empirical Bayes: the prior is learned from how much venues really differ
    post_eb, prior_eb = venue_posteriors(f, wts.values)
    # (b) weak prior: 10 pseudo-matches at the league rate, so each venue's own data speaks
    post, prior = venue_posteriors(f, wts.values, prior_strength=WEAK_PRIOR)
    post["eb_post_mean"] = post_eb["post_mean_batfirst"]
    post["eb_action"] = post_eb["bayes_action"]
    out["beta_prior_empirical_bayes"] = prior_eb
    out["eb_prior_strength_pseudo_matches"] = sum(prior_eb)
    out["beta_prior_weak"] = prior
    table = post[post["n"] >= MIN_VENUE_MATCHES].sort_values("post_mean_batfirst")
    table.round(3).to_csv(K.RESULTS / "venue_bayes_decisions.csv")

    # ---------------- 3. back-test: recommend from earlier seasons only
    out["backtest"] = {}
    for label, strength in (("empirical_bayes", None), ("weak_prior", WEAK_PRIOR)):
        bt_rows = []
        for s_ in range(2015, last + 1):
            hist, cur = f[f["season"] < s_], f[f["season"] == s_]
            w = 0.5 ** ((s_ - 1 - hist["season"]) / RECENCY_HALF_LIFE)
            p, (a0, b0) = venue_posteriors(hist, w.values, prior_strength=strength)
            pm = cur["venue"].map(p["post_mean_batfirst"]).fillna(a0 / (a0 + b0))
            bt_rows.append(cur.assign(rec=np.where(pm >= 0.5, "bat", "field"), post_mean=pm.values))
        b = pd.concat(bt_rows)
        b["followed"] = (b["toss_decision"] == b["rec"]).astype(int)
        # did the side that took the recommended role win? (independent of who won the toss)
        b["rec_role_won"] = np.where(b["rec"] == "bat", b["bat_first_won"], 1 - b["bat_first_won"])
        summ = b.groupby("followed")["toss_winner_won"].agg(["mean", "size"])
        ct_f = pd.crosstab(b["followed"], b["toss_winner_won"])
        out["backtest"][label] = {
            "matches": len(b), "share_recommend_field": (b["rec"] == "field").mean(),
            "share_rule_followed": b["followed"].mean(),
            "toss_winner_win_rate_followed": summ.loc[1, "mean"], "n_followed": int(summ.loc[1, "size"]),
            "toss_winner_win_rate_not_followed": summ.loc[0, "mean"], "n_not_followed": int(summ.loc[0, "size"]),
            "chi2_p": stats.chi2_contingency(ct_f).pvalue,
            "recommended_role_win_rate": b["rec_role_won"].mean(),
            "always_field_role_win_rate": 1 - b["bat_first_won"].mean(),
        }
    K.save_json(out, "toss_bayes.json")
    by_season.round(3).to_csv(K.RESULTS / "toss_by_season.csv")
    by_team.round(3).to_csv(K.RESULTS / "toss_by_team.csv")
    for kk in ["toss_winner_win_rate", "toss_binom_p", "chi2_decision_p", "chi2_team_p",
               "home_win_rate", "home_n", "home_binom_p", "lrt_venue_x_decision", "lrt_era_x_decision",
               "beta_prior_empirical_bayes", "eb_prior_strength_pseudo_matches", "backtest"]:
        print(kk, out[kk])
    print(by_dec)
    print(table[["n", "post_mean_batfirst", "ci_low", "ci_high", "P_field_better", "bayes_action",
                 "eb_post_mean"]].round(3).to_string())

    # ---------------- figures
    t = table.copy()
    fig, ax = K.plt.subplots(figsize=(7.5, 0.32 * len(t) + 1.2))
    y = np.arange(len(t))
    col = [K.SERIES[0] if a == "field" else K.SERIES[1] for a in t["bayes_action"]]
    ax.hlines(y, t["ci_low"], t["ci_high"], color=col, lw=2)
    ax.scatter(t["post_mean_batfirst"], y, color=col, s=36, zorder=3, edgecolor=K.SURFACE, lw=1)
    ax.axvline(0.5, color=K.TEXT2, lw=.8, ls="--")
    ax.set_yticks(y); ax.set_yticklabels([f"{v}  (n={int(nn)})" for v, nn in zip(t.index, t["n"])])
    ax.scatter(t["eb_post_mean"], y, marker="|", s=90, color=K.TEXT2, zorder=4, label="Empirical-Bayes mean")
    ax.legend(loc="lower right")
    ax.set_xlabel("Posterior P(team batting first wins), weak prior, recency-weighted, 90% CI")
    ax.set_title("Bayes action by venue: blue = field first, orange = bat first")
    K.savefig(fig, "venue_bayes_posteriors.png")

    fig, ax = K.plt.subplots(figsize=(7, 3.2))
    ax.plot(by_season.index, by_season["field_choice"], marker="o", ms=5, color=K.SERIES[0],
            label="Toss winners choosing to field")
    ax.plot(by_season.index, 1 - by_season["bat_first_win"], marker="o", ms=5, color=K.SERIES[2],
            label="Chasing side won")
    ax.plot(by_season.index, by_season["toss_win"], marker="o", ms=5, color=K.SERIES[1],
            label="Toss winner won")
    ax.axhline(0.5, color=K.TEXT2, lw=.8, ls="--")
    ax.set_ylim(0.2, 1); ax.set_ylabel("Share of matches"); ax.legend(loc="lower left", ncol=1)
    ax.set_title("Captains field far more than the results justify")
    K.savefig(fig, "toss_trends_by_season.png")


if __name__ == "__main__":
    main()
