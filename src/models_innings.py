"""Q1 / Objective ii: predict the first-innings total.

For each snapshot (0 = pre-match, 6, 10, 15 overs):
  * baseline : runs_n + current run rate * overs left (the "projected score" on TV)
  * OLS      : linear regression on score state + venue + era + team/lineup form
Train 2008-2021, test 2022-2024. Reports MAE / RMSE / R^2, standardised coefficients
with p-values (statsmodels), and permutation importance on the test seasons.

Run: python src/models_innings.py
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import common as K

# impact_player_era is left out: it is 0 for every training season (constant column).
# The era shift is carried by recent_league_inn1 instead.
CONTEXT = ["venue_avg_inn1_smooth", "recent_league_inn1", "played_abroad",
           "A_lineup_bat_sr", "A_lineup_bat_rpi", "B_lineup_bowl_econ", "A_avg_scored5",
           "B_avg_conceded5", "A_home", "is_playoff"]


def features(f, n, centers=None):
    X = f[CONTEXT].copy()
    if n:
        X[f"runs_{n}"] = f[f"inn1_runs_{n}"]
        X[f"wkts_{n}"] = f[f"inn1_wkts_{n}"]
        X[f"last3_{n}"] = f[f"inn1_last3_{n}"]
        X[f"sixes_{n}"] = f[f"inn1_sixes_{n}"]
        # wickets in hand amplify how much of the current rate can be sustained.
        # Both terms are centred on training means so the main effects stay interpretable.
        rr_c, w_c = centers if centers else (0.0, 0.0)
        X[f"rr_x_wkts_in_hand_{n}"] = (f[f"inn1_rr_{n}"] - rr_c) * ((10 - f[f"inn1_wkts_{n}"]) - w_c)
    return X


def metrics(y, p):
    return {"MAE": mean_absolute_error(y, p), "RMSE": mean_squared_error(y, p) ** 0.5,
            "R2": r2_score(y, p)}


def main():
    f = K.load_matches()
    tr, te = K.split(f)
    rows, coef_tables, perm_tables, preds, bias = [], {}, {}, {}, {}
    for n in (0, 6, 10, 15):
        cen = ((tr[f"inn1_rr_{n}"].mean(), (10 - tr[f"inn1_wkts_{n}"]).mean()) if n else None)
        Xtr, Xte = features(tr, n, cen), features(te, n, cen)
        ytr, yte = tr["inn1_total"], te["inn1_total"]
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LinearRegression())
        model.fit(Xtr, ytr)
        p = model.predict(Xte)
        rows.append({"overs": n, "model": "Linear regression", **metrics(yte, p)})
        preds[n] = p
        bias[n] = (yte - p).groupby(te["season"]).mean().round(2).to_dict()
        if n:
            base = te[f"inn1_runs_{n}"] + te[f"inn1_rr_{n}"] * (20 - n)
            rows.append({"overs": n, "model": "Run-rate projection (baseline)", **metrics(yte, base)})
        else:
            base = np.full(len(te), ytr.mean())
            rows.append({"overs": n, "model": "Train mean (baseline)", **metrics(yte, base)})

        # standardised OLS coefficients + p-values on the training seasons
        imp = SimpleImputer(strategy="median").fit(Xtr)
        Z = pd.DataFrame(StandardScaler().fit_transform(imp.transform(Xtr)), columns=Xtr.columns)
        ols = sm.OLS(ytr.values, sm.add_constant(Z)).fit()
        coef_tables[n] = pd.DataFrame({"std_coef": ols.params.drop("const"),
                                       "p_value": ols.pvalues.drop("const")}).sort_values(
            "std_coef", key=abs, ascending=False)
        pi = permutation_importance(model, Xte, yte, scoring="neg_root_mean_squared_error",
                                    n_repeats=30, random_state=K.SEED)
        perm_tables[n] = pd.Series(pi.importances_mean, index=Xte.columns).sort_values(ascending=False)

    res = pd.DataFrame(rows)
    res.to_csv(K.RESULTS / "innings_regression_metrics.csv", index=False)
    with open(K.RESULTS / "innings_regression_coefficients.txt", "w") as fh:
        for n in coef_tables:
            fh.write(f"\n=== {n} overs: standardised coefficients (runs per 1 SD), train 2008-21 ===\n")
            fh.write(coef_tables[n].round(4).to_string() + "\n")
            fh.write(f"--- permutation importance on test (RMSE increase, runs) ---\n")
            fh.write(perm_tables[n].round(3).to_string() + "\n")
    print(res.round(3).to_string(index=False))
    print(coef_tables[10].round(3).head(8))
    print("mean residual (actual - predicted) by test season:", bias)

    # ---- figure 1: RMSE by overs, model vs baseline
    fig, ax = K.plt.subplots(figsize=(6, 3.4))
    lr = res[res.model == "Linear regression"]
    bl = res[res.model != "Linear regression"]
    ax.plot(lr.overs, lr.RMSE, marker="o", ms=6, color=K.SERIES[0], label="Linear regression")
    ax.plot(bl.overs, bl.RMSE, marker="o", ms=6, color=K.SERIES[1], label="Baseline (mean / run-rate projection)")
    for d, c in ((lr, K.SERIES[0]), (bl, K.SERIES[1])):
        for x, y in zip(d.overs, d.RMSE):
            ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=7.5, color=K.TEXT2)
    ax.set_xticks([0, 6, 10, 15]); ax.set_xlabel("Overs bowled when the prediction is made")
    ax.set_ylabel("Test RMSE (runs), 2022-24"); ax.set_title("First-innings total: error shrinks as overs pass")
    ax.legend(loc="upper right"); ax.set_ylim(0, None)
    K.savefig(fig, "innings_rmse_by_overs.png")

    # ---- figure 2: predicted vs actual at 10 overs
    fig, ax = K.plt.subplots(figsize=(4.4, 4.2))
    ax.scatter(te["inn1_total"], preds[10], s=14, color=K.SERIES[0], alpha=.6, edgecolor=K.SURFACE, lw=.5)
    lims = [70, 290]; ax.plot(lims, lims, color=K.TEXT2, lw=1, ls="--")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Actual first-innings total"); ax.set_ylabel("Predicted after 10 overs")
    ax.set_title("Predicted vs actual, 10 overs (test)")
    K.savefig(fig, "innings_pred_vs_actual_10.png")

    # ---- figure 3: standardised coefficients at 10 overs
    ct = coef_tables[10].iloc[::-1]
    fig, ax = K.plt.subplots(figsize=(6, 4))
    colors = [K.SERIES[0] if v > 0 else K.SERIES[1] for v in ct.std_coef]
    ax.barh(ct.index, ct.std_coef, color=colors, height=.6)
    ax.axvline(0, color=K.TEXT2, lw=.8)
    ax.set_xlabel("Runs added per +1 SD of the feature (train)")
    ax.set_title("What explains the total at 10 overs")
    K.savefig(fig, "innings_coefficients_10.png")
    K.save_json({"metrics": res.to_dict("records"),
                 "coef_10": coef_tables[10].to_dict(), "coef_0": coef_tables[0].to_dict(),
                 "coef_15": coef_tables[15].to_dict(), "perm_10": perm_tables[10].to_dict(),
                 "residual_bias_by_season": bias},
                "innings_regression.json")


if __name__ == "__main__":
    main()
