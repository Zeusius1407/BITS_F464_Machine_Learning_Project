"""Q5 / Objectives v-vii: PCA and clustering of player-seasons into playing roles.

Batting  : player-seasons with >= 100 balls faced.  Bowling: >= 120 balls bowled.
Pipeline : standardise -> PCA (summary + plots) -> K-Means and Ward hierarchical clustering
           -> clusters named by rules on their centroids -> each player's modal role ->
           compared with a hand-labelled reference set of well-known players.

Run: python src/player_roles.py
"""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler

import common as K
import config as C

BAT_RAW = ["bat_avg_adj", "bat_sr", "boundary_pct", "six_pct", "dot_pct_bat", "balls_per_inn",
           "avg_entry_ball", "share_balls_pp", "share_balls_death", "sr_pp", "sr_death"]
# Scoring rates rose a lot from 2008 to 2024, so rate stats are divided by that season's
# median ("_rel" = relative to the season). Without this, early-era batters cluster
# together just for being from an earlier era.
ERA_ADJ = ["bat_avg_adj", "bat_sr", "boundary_pct", "six_pct", "dot_pct_bat", "sr_pp", "sr_death"]
BAT_FEATS = [f"{c}_rel" if c in ERA_ADJ else c for c in BAT_RAW]
BOWL_FEATS = ["economy", "bowl_sr_adj", "dot_pct_bowl", "boundary_pct_conceded", "share_overs_pp",
              "share_overs_mid", "share_overs_death", "overs_per_inn"]
K_BAT, K_BOWL = 5, 4

# Reference roles for well-known players (consensus cricket knowledge, by the author).
# Review and extend these before relying on the validation numbers.
REF_BAT = {
    "Aggressive opener": ["CH Gayle", "V Sehwag", "DA Warner", "JC Buttler", "PP Shaw", "YBK Jaiswal",
                          "TM Head", "Abhishek Sharma", "BB McCullum", "AC Gilchrist", "Q de Kock",
                          "PD Salt", "Ishan Kishan", "JM Bairstow"],
    "Anchor": ["V Kohli", "KL Rahul", "F du Plessis", "S Dhawan", "AM Rahane", "KS Williamson",
               "Shubman Gill", "RD Gaikwad", "G Gambhir", "SR Tendulkar", "MK Pandey", "SE Marsh"],
    "Finisher / power-hitter": ["MS Dhoni", "KA Pollard", "AD Russell", "HH Pandya", "DA Miller",
                                "R Tewatia", "KD Karthik", "RK Singh", "M Shahrukh Khan", "Abdul Samad",
                                "YK Pathan", "T Stubbs", "H Klaasen"],
}
REF_BOWL = {
    "Powerplay / new ball": ["B Kumar", "DL Chahar", "TA Boult", "Mohammed Siraj", "Mohammed Shami",
                             "P Kumar", "Sandeep Sharma", "UT Yadav", "Z Khan", "I Sharma"],
    "Middle overs": ["YS Chahal", "Rashid Khan", "SP Narine", "A Mishra", "PP Chawla", "Kuldeep Yadav",
                     "R Ashwin", "Harbhajan Singh", "Imran Tahir", "RA Jadeja", "Ravi Bishnoi",
                     "AR Patel", "CV Varun"],
    "Death overs": ["DJ Bravo", "SL Malinga", "HV Patel", "T Natarajan", "Arshdeep Singh", "AJ Tye",
                    "M Pathirana", "CJ Jordan"],
}


def prep(ps):
    bat = ps[ps["balls_faced"] >= 100].copy()
    bat["bat_avg_adj"] = bat["runs"] / bat["outs"].clip(lower=1)        # not-outs capped
    for c in ["sr_pp", "sr_death"]:
        bat[c] = bat[c].fillna(bat["bat_sr"])
    for c in ERA_ADJ:
        bat[f"{c}_rel"] = bat[c] / bat.groupby("season")[c].transform("median")
    bowl = ps[ps["balls_bowled"] >= 120].copy()
    bowl["bowl_sr_adj"] = bowl["balls_bowled"] / bowl["wickets"].clip(lower=1)
    return bat.reset_index(drop=True), bowl.reset_index(drop=True)


def name_bat(cent):
    """Rules on centroids: late entry -> finisher (high SR) or middle order; early entry ->
    anchor (highest average), aggressive opener (highest SR), conservative top order."""
    names, late = {}, cent[cent["avg_entry_ball"] >= 40]
    early = cent[cent["avg_entry_ball"] < 40]
    if len(late):
        f = late["bat_sr"].idxmax(); names[f] = "Finisher / power-hitter"
        for i in late.index.drop(f):
            names[i] = "Middle-order accumulator"
    rest = early.copy()
    if len(rest):
        a = rest["bat_avg_adj"].idxmax(); names[a] = "Anchor"; rest = rest.drop(a)
    if len(rest):
        o = rest["bat_sr"].idxmax(); names[o] = "Aggressive opener"; rest = rest.drop(o)
    for i in rest.index:
        names[i] = "Conservative top order"
    return names


def name_bowl(cent):
    names, rest = {}, cent.copy()
    p = rest["overs_per_inn"].idxmin(); names[p] = "Part-time / change bowler"; rest = rest.drop(p)
    for col, label in (("share_overs_pp", "Powerplay / new ball"), ("share_overs_mid", "Middle overs")):
        i = rest[col].idxmax(); names[i] = label; rest = rest.drop(i)
    for i in rest.index:
        names[i] = "Death overs"
    return names


def run(df, feats, k, namer, ref, tag, raw_feats=None, figures=True):
    raw_feats = raw_feats or feats
    desc = list(dict.fromkeys(feats + raw_feats))
    X = StandardScaler().fit_transform(df[feats])
    pca = PCA(random_state=K.SEED).fit(X)
    Z = pca.transform(X)
    load = pd.DataFrame(pca.components_[:3].T, index=feats, columns=["PC1", "PC2", "PC3"])

    sel = []
    for kk in range(2, 9):
        km_ = KMeans(kk, n_init=20, random_state=K.SEED).fit(X)
        hc_ = AgglomerativeClustering(kk, linkage="ward").fit(X)
        sel.append({"k": kk, "inertia": km_.inertia_, "silhouette_kmeans": silhouette_score(X, km_.labels_),
                    "silhouette_ward": silhouette_score(X, hc_.labels_),
                    "ARI_kmeans_vs_ward": adjusted_rand_score(km_.labels_, hc_.labels_)})
    sel = pd.DataFrame(sel)

    km = KMeans(k, n_init=50, random_state=K.SEED).fit(X)
    hc = AgglomerativeClustering(k, linkage="ward").fit(X)
    n90 = int(np.searchsorted(pca.explained_variance_ratio_.cumsum(), 0.90) + 1)
    km_pc = KMeans(k, n_init=50, random_state=K.SEED).fit(Z[:, :n90])     # clustering in PC space

    df = df.copy()
    for col, lab in (("km", km.labels_), ("hc", hc.labels_)):
        cent = df.assign(_c=lab).groupby("_c")[desc].mean()
        df[f"role_{col}"] = pd.Series(lab).map(namer(cent)).values
    df["PC1"], df["PC2"] = Z[:, 0], Z[:, 1]
    profile = df.groupby("role_km")[raw_feats].mean()      # readable units
    profile["n_player_seasons"] = df["role_km"].value_counts()

    # each player's modal role (weighted by volume) and comparison with the references
    wcol = "balls_faced" if tag.startswith("batting") else "balls_bowled"
    modal = {}
    for col in ("role_km", "role_hc"):
        modal[col] = (df.groupby(["player", col])[wcol].sum().reset_index()
                      .sort_values(wcol).groupby("player").tail(1).set_index("player")[col])
    refs = pd.DataFrame([(p, r) for r, ps_ in ref.items() for p in ps_], columns=["player", "reference"])
    refs["found"] = refs["player"].isin(modal["role_km"].index)
    rf = refs[refs["found"]].copy()
    rf["kmeans_role"] = rf["player"].map(modal["role_km"])
    rf["ward_role"] = rf["player"].map(modal["role_hc"])
    val = {
        "reference_players_found": int(rf.shape[0]), "missing": refs.loc[~refs["found"], "player"].tolist(),
        "kmeans_agreement": (rf["kmeans_role"] == rf["reference"]).mean(),
        "ward_agreement": (rf["ward_role"] == rf["reference"]).mean(),
        "kmeans_ARI_vs_reference": adjusted_rand_score(rf["reference"], rf["kmeans_role"]),
        "kmeans_NMI_vs_reference": normalized_mutual_info_score(rf["reference"], rf["kmeans_role"]),
        "ARI_kmeans_vs_ward": adjusted_rand_score(df["role_km"], df["role_hc"]),
        "ARI_kmeans_raw_vs_pcspace": adjusted_rand_score(km.labels_, km_pc.labels_),
        "pcs_for_90pct_variance": n90,
        "silhouette_kmeans": silhouette_score(X, km.labels_),
        "explained_variance_ratio": pca.explained_variance_ratio_.round(4).tolist(),
    }
    conf = pd.crosstab(rf["reference"], rf["kmeans_role"])

    if not figures:           # sensitivity runs: metrics only, no files
        return val, profile, conf, sel, load
    # outputs
    df.to_csv(K.RESULTS / f"{tag}_roles_player_season.csv", index=False)
    rf.to_csv(K.RESULTS / f"{tag}_role_validation.csv", index=False)
    with open(K.RESULTS / f"{tag}_roles_summary.txt", "w") as fh:
        fh.write(f"=== {tag}: {len(df)} player-seasons, k={k} ===\n\nPCA loadings\n{load.round(3)}\n\n")
        fh.write(f"Explained variance: {val['explained_variance_ratio']}\n\n")
        fh.write(f"Choosing k\n{sel.round(3).to_string(index=False)}\n\n")
        fh.write(f"K-Means role profiles (feature means)\n{profile.round(2).T.to_string()}\n\n")
        for r in profile.index:
            top = df[df.role_km == r].sort_values(wcol, ascending=False)
            ex = top.drop_duplicates("player")["player"].head(10).tolist()
            fh.write(f"{r}: {', '.join(ex)}\n")
        fh.write(f"\nValidation vs reference labels\n{ {k_: v for k_, v in val.items() if k_ != 'explained_variance_ratio'} }\n")
        fh.write(f"\nConfusion (reference rows x K-Means role columns)\n{conf.to_string()}\n")
        fh.write(f"\nMismatches\n{rf[rf.kmeans_role != rf.reference].to_string(index=False)}\n")

    # figures: scree, PC scatter, dendrogram
    roles = list(profile.index)
    cmap = {r: K.SERIES[i] for i, r in enumerate(roles)}
    fig, axes = K.plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1, 2]})
    ev = pca.explained_variance_ratio_
    axes[0].bar(np.arange(1, len(ev) + 1), ev, color=K.SERIES[0], width=.6)
    axes[0].plot(np.arange(1, len(ev) + 1), ev.cumsum(), color=K.SERIES[1], marker="o", ms=5, label="cumulative")
    axes[0].set_xlabel("Principal component"); axes[0].set_ylabel("Share of variance")
    axes[0].set_title(f"{tag.title()} PCA: scree"); axes[0].legend(loc="center right")
    ax = axes[1]
    for r in roles:
        d = df[df.role_km == r]
        ax.scatter(d.PC1, d.PC2, s=12, color=cmap[r], alpha=.65, label=f"{r} ({len(d)})", edgecolor="none")
    # label the highest-volume player-season of each role (one per role keeps it legible)
    star = df.sort_values(wcol, ascending=False).groupby("role_km").head(1)
    for _, s in star.iterrows():
        ax.annotate(f"{s.player} '{str(s.season)[2:]}", (s.PC1, s.PC2), fontsize=6.5, color=K.TEXT,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel(f"PC1 ({ev[0]:.0%})"); ax.set_ylabel(f"PC2 ({ev[1]:.0%})")
    ax.set_title(f"{tag.title()} player-seasons on the first two PCs, coloured by K-Means role")
    ax.legend(loc="best", fontsize=7, markerscale=1.5)
    K.savefig(fig, f"{tag}_pca_roles.png")

    fig, ax = K.plt.subplots(figsize=(9, 3.6))
    dendrogram(linkage(X, "ward"), truncate_mode="lastp", p=30, ax=ax, color_threshold=None,
               above_threshold_color=K.SERIES[0], leaf_font_size=7)
    ax.set_title(f"{tag.title()}: Ward hierarchical clustering (last 30 merges)")
    ax.set_ylabel("Ward distance"); ax.grid(False)
    K.savefig(fig, f"{tag}_dendrogram.png")
    return val, profile, conf, sel, load


def main():
    ps = pd.read_csv(C.PROCESSED / "player_season.csv")
    bat, bowl = prep(ps)
    out = {}
    pd.set_option("display.width", 200)
    # sensitivity check: batting clusters on raw (not era-adjusted) features
    v_raw = run(bat, BAT_RAW, K_BAT, name_bat, REF_BAT, "batting_unadjusted", figures=False)[0]
    out["batting_unadjusted"] = {k_: v_raw[k_] for k_ in ("kmeans_agreement", "kmeans_ARI_vs_reference",
                                                         "silhouette_kmeans")}
    for tag, df, feats, k, namer, ref, raw in (
            ("batting", bat, BAT_FEATS, K_BAT, name_bat, REF_BAT, BAT_RAW),
            ("bowling", bowl, BOWL_FEATS, K_BOWL, name_bowl, REF_BOWL, None)):
        val, prof, conf, sel, load = run(df, feats, k, namer, ref, tag, raw)
        out[tag] = val
        print(f"\n===== {tag} =====")
        print(load.round(2)); print(prof.round(2).T); print(conf)
        print({k_: (round(v, 3) if isinstance(v, float) else v) for k_, v in val.items()})
    K.save_json(out, "player_roles.json")


if __name__ == "__main__":
    main()
