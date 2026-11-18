"""Q2 / Q5 / Objective iii: predict the match winner (target = bat_first_won).

Three information stages:
  pre_match     : after the toss (form, Elo, venue, head-to-head, lineups, toss)
  innings_break : + first-innings total and wickets
  chase_10      : + chase state after 10 overs of the second innings
Models: Logistic Regression, KNN, CART, PCA + Logistic Regression, against two baselines.
Hyper-parameters are tuned with GroupKFold over training seasons (2008-21); the final
evaluation is on the held-out 2022-24 seasons with bootstrap 95% CIs.

Run: python src/models_winner.py
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

import common as K

PRE = ["diff_form5", "diff_form10", "diff_season_winrate", "diff_elo", "A_home", "B_home",
       "h2h_A_winrate_smooth", "venue_batfirst_winrate_smooth", "venue_avg_inn1_smooth",
       # toss_decision_field is dropped: it is exactly 1 - toss_winner_bats_first
       "toss_winner_bats_first", "diff_lineup_bat_sr",
       "diff_lineup_bat_rpi", "diff_lineup_bowl_econ", "diff_avg_scored5",
       "diff_avg_conceded5", "is_playoff", "recent_league_inn1"]
BREAK = PRE + ["inn1_total", "inn1_wkts", "par_diff_venue", "par_diff_recent"]
CHASE = BREAK + ["inn2_need_10", "inn2_wkts_10", "inn2_req_rr_10"]
STAGES = {"pre_match": PRE, "innings_break": BREAK, "chase_10": CHASE}
# KNN suffers when many weak features dilute the distance; this compact set keeps only
# the match-state features, so "similar match" means similar score at the break.
KNN_CORE = ["inn1_total", "inn1_wkts", "par_diff_venue", "par_diff_recent",
            "venue_batfirst_winrate_smooth"]


def prep(f):
    f = f.copy()
    f["par_diff_venue"] = f["inn1_total"] - f["venue_avg_inn1_smooth"]
    f["par_diff_recent"] = f["inn1_total"] - f["recent_league_inn1"]
    return f


def models():
    base = [("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]
    return {
        "Logistic Regression": (Pipeline(base + [("m", LogisticRegression(max_iter=2000))]),
                                {"m__C": [0.003, 0.01, 0.03, 0.1, 0.3, 1, 3]}),
        "KNN": (Pipeline(base + [("m", KNeighborsClassifier())]),
                {"m__n_neighbors": [5, 15, 25, 45, 75, 115], "m__weights": ["uniform", "distance"],
                 "m__metric": ["euclidean", "manhattan"]}),
        "CART": (Pipeline(base + [("m", DecisionTreeClassifier(random_state=K.SEED))]),
                 {"m__max_depth": [2, 3, 4, 5, 6], "m__min_samples_leaf": [10, 20, 40, 80],
                  "m__criterion": ["gini", "entropy"]}),
        "PCA + Logistic": (Pipeline(base + [("pca", PCA(random_state=K.SEED)),
                                            ("m", LogisticRegression(max_iter=2000))]),
                           {"pca__n_components": [2, 3, 5, 8], "m__C": [0.01, 0.1, 1]}),
    }


def boot_ci(y, p, fn, n=2000, seed=K.SEED):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    stats = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        stats.append(fn(y[i], p[i]))
    return np.percentile(stats, [2.5, 97.5])


def evaluate(y, prob):
    pred = (prob >= 0.5).astype(int)
    lo, hi = boot_ci(y, pred, accuracy_score)
    return {"accuracy": accuracy_score(y, pred), "acc_ci_low": lo, "acc_ci_high": hi,
            "roc_auc": roc_auc_score(y, prob), "brier": brier_score_loss(y, prob),
            "log_loss": log_loss(y, np.clip(prob, 1e-6, 1 - 1e-6))}


def main():
    f = prep(K.load_matches())
    tr_all, te_all = K.split(f)
    rows, fitted, perm = [], {}, {}
    cv = GroupKFold(n_splits=5)

    for stage, cols in STAGES.items():
        tr, te = tr_all, te_all
        if stage == "chase_10":          # drop chases already decided by over 10
            tr, te = tr[tr["inn2_done_by_10"] == 0], te[te["inn2_done_by_10"] == 0]
        Xtr, ytr, Xte, yte = tr[cols], tr["bat_first_won"].astype(int), te[cols], te["bat_first_won"].astype(int)

        # baselines
        maj = float(ytr.mean() >= 0.5)
        rows.append({"stage": stage, "model": "Baseline: majority class",
                     **evaluate(yte, np.full(len(yte), ytr.mean())), "best_params": f"always {int(maj)}"})
        elo_p = 1 / (1 + 10 ** (-te["diff_elo"] / 400))
        rows.append({"stage": stage, "model": "Baseline: Elo favourite", **evaluate(yte, elo_p.values),
                     "best_params": ""})

        for name, (pipe, grid) in models().items():
            gs = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc", n_jobs=-1)
            gs.fit(Xtr, ytr, groups=tr["season"])
            prob = gs.best_estimator_.predict_proba(Xte)[:, 1]
            rows.append({"stage": stage, "model": name, **evaluate(yte, prob),
                         "cv_auc": gs.best_score_, "best_params": str(gs.best_params_)})
            fitted[(stage, name)] = gs.best_estimator_
            if stage in ("pre_match", "innings_break"):
                pi = permutation_importance(gs.best_estimator_, Xte, yte, scoring="roc_auc",
                                            n_repeats=30, random_state=K.SEED)
                perm[(stage, name)] = pd.Series(pi.importances_mean, index=cols)
        if stage == "innings_break":
            pipe, grid = models()["KNN"]
            gs = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc", n_jobs=-1)
            gs.fit(tr[KNN_CORE], ytr, groups=tr["season"])
            prob = gs.best_estimator_.predict_proba(te[KNN_CORE])[:, 1]
            rows.append({"stage": stage, "model": "KNN (5 core features)", **evaluate(yte, prob),
                         "cv_auc": gs.best_score_, "best_params": str(gs.best_params_)})
            fitted[(stage, "KNN core")] = gs.best_estimator_

    res = pd.DataFrame(rows)
    res.to_csv(K.RESULTS / "winner_model_comparison.csv", index=False)
    show = res[["stage", "model", "accuracy", "acc_ci_low", "acc_ci_high", "roc_auc", "brier", "best_params"]]
    print(show.round(3).to_string(index=False))

    # ---- interpretable logistic regression (statsmodels) for p-values, pre-match
    tr = tr_all
    imp = SimpleImputer(strategy="median").fit(tr[PRE])
    Z = pd.DataFrame(StandardScaler().fit_transform(imp.transform(tr[PRE])), columns=PRE)
    logit = sm.Logit(tr["bat_first_won"].astype(int).values, sm.add_constant(Z)).fit(disp=0)
    lt = pd.DataFrame({"std_coef": logit.params, "odds_ratio_per_SD": np.exp(logit.params),
                       "p_value": logit.pvalues}).drop("const").sort_values("p_value")
    # par_diff_* are exact linear combinations of inn1_total and the averages, so they
    # are left out of the statsmodels fit (they would make the p-values undefined)
    BREAK_SM = [c for c in BREAK if not c.startswith("par_diff")]
    Zb = pd.DataFrame(StandardScaler().fit_transform(
        SimpleImputer(strategy="median").fit_transform(prep(tr)[BREAK_SM])), columns=BREAK_SM)
    logit_b = sm.Logit(tr["bat_first_won"].astype(int).values, sm.add_constant(Zb)).fit(disp=0)
    ltb = pd.DataFrame({"std_coef": logit_b.params, "odds_ratio_per_SD": np.exp(logit_b.params),
                        "p_value": logit_b.pvalues}).drop("const").sort_values("p_value")

    pm = pd.DataFrame({f"{s} | {m}": v for (s, m), v in perm.items()})
    pm.to_csv(K.RESULTS / "winner_permutation_importance.csv")
    with open(K.RESULTS / "winner_logit_and_tree.txt", "w") as fh:
        fh.write("=== Logistic regression, pre-match features (standardised, train 2008-21) ===\n")
        fh.write(lt.round(4).to_string() + "\n\n")
        fh.write("=== Logistic regression, innings-break features ===\n")
        fh.write(ltb.round(4).to_string() + "\n\n")
        for stage in ("pre_match", "innings_break"):
            tree = fitted[(stage, "CART")].named_steps["m"]
            fh.write(f"=== CART rules, {stage} (inputs are standardised) ===\n")
            fh.write(export_text(tree, feature_names=STAGES[stage], decimals=2) + "\n")
    print(lt.round(3).head(8))

    # ---- KNN similarity: 5 most similar historical matches for one 2024 match at the break
    knn = fitted[("innings_break", "KNN core")]
    Xtr_t = knn[:-1].transform(tr_all[KNN_CORE]); Xte_t = knn[:-1].transform(te_all[KNN_CORE])
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=5, metric=knn.named_steps["m"].metric).fit(Xtr_t)
    q = te_all.index.get_loc(te_all[te_all["season"] == 2024].index[-1])   # the 2024 final
    dist, idx = nn.kneighbors(Xte_t[[q]])
    qrow = te_all.iloc[q]
    sim = tr_all.iloc[idx[0]][["date", "venue", "bat_first_team", "bat_second_team", "inn1_total",
                               "inn1_wkts", "winner", "bat_first_won"]].assign(distance=dist[0])
    with open(K.RESULTS / "knn_similar_matches.txt", "w") as fh:
        fh.write(f"Query: {qrow.date.date()} {qrow.bat_first_team} {int(qrow.inn1_total)}/{int(qrow.inn1_wkts)} "
                 f"vs {qrow.bat_second_team} at {qrow.venue}; actual winner: {qrow.winner}\n")
        fh.write(f"KNN P(bat-first wins) = {knn.predict_proba(te_all[KNN_CORE].iloc[[q]])[0,1]:.2f}\n\n")
        fh.write(sim.to_string(index=False))
    print(open(K.RESULTS / "knn_similar_matches.txt").read())

    # ---- figures
    order = ["Baseline: majority class", "Baseline: Elo favourite", "Logistic Regression",
             "KNN", "KNN (5 core features)", "CART", "PCA + Logistic"]
    fig, axes = K.plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, metric, title in ((axes[0], "accuracy", "Test accuracy (2022-24)"),
                              (axes[1], "roc_auc", "Test ROC-AUC (2022-24)")):
        w = 0.25
        for j, stage in enumerate(STAGES):
            d = res[res.stage == stage].set_index("model").reindex(order)
            x = np.arange(len(order)) + (j - 1) * w
            ax.barh(x, d[metric], height=w - 0.03, color=K.SERIES[j], label=stage.replace("_", " "))
            if metric == "accuracy":
                ax.errorbar(d[metric], x, xerr=[d[metric] - d.acc_ci_low, d.acc_ci_high - d[metric]],
                            fmt="none", ecolor=K.TEXT2, elinewidth=.8, capsize=2)
        ax.axvline(0.5, color=K.TEXT2, lw=.8, ls="--")
        ax.set_yticks(np.arange(len(order))); ax.set_yticklabels(order)
        ax.set_xlim(0.35, 1.0); ax.set_title(title); ax.invert_yaxis()
    axes[0].legend(loc="lower right", title="Information available")
    K.savefig(fig, "winner_model_comparison.png")

    best = res[(res.stage == "pre_match") & ~res.model.str.startswith("Baseline")].sort_values("roc_auc").iloc[-1].model
    for stage in ("pre_match", "innings_break"):
        s = perm[(stage, best if stage == "pre_match" else "Logistic Regression")].sort_values().tail(10)
        fig, ax = K.plt.subplots(figsize=(6, 3.6))
        ax.barh(s.index, s.values, color=K.SERIES[0], height=.6)
        ax.axvline(0, color=K.TEXT2, lw=.8)
        ax.set_xlabel("Drop in test ROC-AUC when the feature is shuffled")
        ax.set_title(f"Most important features: {stage.replace('_', ' ')} "
                     f"({best if stage == 'pre_match' else 'Logistic Regression'})")
        K.savefig(fig, f"winner_importance_{stage}.png")

    tree = fitted[("innings_break", "CART")].named_steps["m"]
    fig, ax = K.plt.subplots(figsize=(13, 6))
    plot_tree(tree, feature_names=BREAK, class_names=["chase wins", "bat-first wins"], filled=False,
              impurity=False, proportion=True, rounded=True, fontsize=7, max_depth=3, ax=ax)
    ax.set_title("CART at the innings break (top 3 levels; thresholds are standardised)")
    K.savefig(fig, "cart_tree_innings_break.png")
    K.save_json({"comparison": res.to_dict("records"), "best_pre_match_model": best,
                 "logit_pre_match": lt.to_dict()}, "winner_models.json")


if __name__ == "__main__":
    main()
