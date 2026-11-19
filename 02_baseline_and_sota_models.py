"""02_baseline_and_sota_models.py
Experimento predictivo + auditoria de fuga por duplicados.

Protocolos (mismos modelos, 5 folds x 20 repeticiones = 100 evaluaciones cada uno):
  P1 raw-naive : 1059 filas crudas, StratifiedKFold aleatorio (practica comun en la literatura).
  P2 raw-group : 1059 filas crudas, StratifiedGroupKFold por group_id (copias nunca cruzan train/test).
  P3 unique    : 83 perfiles unicos, RepeatedStratifiedKFold (protocolo principal del paper).

Metrica primaria (a priori): balanced accuracy (P3). Secundarias: accuracy, F1-macro, MCC, QWK,
log-loss, Brier multiclase. Escalado dentro de Pipeline (ajustado solo en train de cada fold).

Salidas: results/fold_scores.csv, results/oof_predictions.parquet,
         tables/table_02_main_results_cv, tables/table_06_leakage_audit, figures/fig_02_*.
"""
import json
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score,
                             log_loss, matthews_corrcoef)

import utils as U
from cf_lib import FEATURES, make_classifier

warnings.filterwarnings("ignore")
MODELS = ["Dummy", "DomainRule", "LR", "kNN", "SVM-RBF", "DT", "RF", "XGBoost", "LightGBM", "MLP"]

uniq = pd.read_csv(U.DATA_PATH)
raw = pd.read_csv(U.RAW_GROUPED_PATH)
DATA = {
    "unique": (uniq[FEATURES].to_numpy(float), uniq["grade"].to_numpy()),
    "raw-naive": (raw[FEATURES].to_numpy(float), raw["grade"].to_numpy()),
    "raw-group": (raw[FEATURES].to_numpy(float), raw["grade"].to_numpy()),
}
SPLITS = {
    "unique": json.load(open(U.RESULTS_DIR / "splits_unique.json")),
    "raw-naive": json.load(open(U.RESULTS_DIR / "splits_raw_naive.json")),
    "raw-group": json.load(open(U.RESULTS_DIR / "splits_raw_group.json")),
}


def brier_mc(y, P):
    Y = np.eye(3)[y]
    return float(((P - Y) ** 2).sum(1).mean())


def run_one(model, protocol, sp):
    X, y = DATA[protocol]
    tr, te = np.array(sp["train"]), np.array(sp["test"])
    clf = make_classifier(model)
    if model == "RF":
        clf.set_params(n_jobs=1)
    clf.fit(X[tr], y[tr])
    yp = clf.predict(X[te])
    P = clf.predict_proba(X[te])
    if P.shape[1] < 3:  # clase ausente en train (no ocurre con estratificacion, salvaguarda)
        full = np.zeros((len(te), 3)); full[:, clf.classes_] = P; P = full
    P = np.clip(P, 1e-6, 1); P = P / P.sum(1, keepdims=True)
    yt = y[te]
    rec = {"model": model, "protocol": protocol, "repeat": sp["repeat"], "fold": sp["fold"],
           "n_train": len(tr), "n_test": len(te),
           "accuracy": accuracy_score(yt, yp), "balanced_accuracy": balanced_accuracy_score(yt, yp),
           "f1_macro": f1_score(yt, yp, average="macro"), "mcc": matthews_corrcoef(yt, yp),
           "qwk": cohen_kappa_score(yt, yp, weights="quadratic"),
           "log_loss": log_loss(yt, P, labels=[0, 1, 2]), "brier": brier_mc(yt, P)}
    oof = None
    if protocol == "unique":
        oof = pd.DataFrame({"model": model, "repeat": sp["repeat"], "fold": sp["fold"], "idx": te,
                            "y_true": yt, "y_pred": yp, "p_low": P[:, 0], "p_medium": P[:, 1], "p_high": P[:, 2]})
    return rec, oof


if __name__ == "__main__":
    t0 = time.time()
    jobs = [(m, p, sp) for p in SPLITS for m in MODELS for sp in SPLITS[p]]
    out = Parallel(n_jobs=-2, verbose=0)(delayed(run_one)(m, p, sp) for m, p, sp in jobs)
    scores = pd.DataFrame([o[0] for o in out])
    oof = pd.concat([o[1] for o in out if o[1] is not None], ignore_index=True)
    scores.to_csv(U.RESULTS_DIR / "fold_scores.csv", index=False)
    oof.to_parquet(U.RESULTS_DIR / "oof_predictions.parquet", index=False)

    # ---------------- Tabla 2: resultados principales (P3 unique) ----------------
    METRICS = [("balanced_accuracy", True), ("f1_macro", True), ("mcc", True), ("qwk", True),
               ("accuracy", True), ("log_loss", False), ("brier", False)]
    su = scores[scores.protocol == "unique"]
    rows, tex = [], []
    for m in MODELS:
        s = su[su.model == m]
        r, t = {"Model": m}, {"Model": m}
        for met, _ in METRICS:
            lo, hi = U.bootstrap_ci(s[met].values)
            r[f"{met}_mean"], r[f"{met}_sd"] = s[met].mean(), s[met].std(ddof=1)
            r[f"{met}_ci_lo"], r[f"{met}_ci_hi"] = lo, hi
            t[met] = U.fmt_mean_sd(s[met].values)
        t["BA 95% CI"] = U.fmt_ci(r["balanced_accuracy_ci_lo"], r["balanced_accuracy_ci_hi"])
        rows.append(r); tex.append(t)
    T2 = pd.DataFrame(rows); T2t = pd.DataFrame(tex)
    for met, hib in METRICS:
        U.highlight(T2t, T2[f"{met}_mean"], met, hib)          # mejor en negrita, segundo subrayado
    T2t.columns = ["Model"] + [U.arrow(n, h) for n, (_, h) in zip(
        ["BA", "F1-macro", "MCC", "QWK", "Accuracy", "Log-loss", "Brier"], METRICS)] + ["BA 95% CI"]
    U.save_table(T2, "table_02_main_results_cv", df_tex=T2t)

    # ---------------- Tabla 6: auditoria de fuga ----------------
    rows, tex = [], []
    for m in MODELS:
        r, t = {"Model": m}, {"Model": m}
        for p in ("raw-naive", "raw-group", "unique"):
            s = scores[(scores.model == m) & (scores.protocol == p)]
            r[f"acc_{p}"] = s.accuracy.mean(); r[f"acc_sd_{p}"] = s.accuracy.std(ddof=1)
            r[f"ba_{p}"] = s.balanced_accuracy.mean()
            t[p] = U.fmt_mean_sd(s.accuracy.values)
        # inflacion: diferencia por repeticion (media de folds) naive - group
        a = scores[(scores.model == m) & (scores.protocol == "raw-naive")].groupby("repeat").accuracy.mean()
        b = scores[(scores.model == m) & (scores.protocol == "raw-group")].groupby("repeat").accuracy.mean()
        d = (a - b).values
        lo, hi = U.bootstrap_ci(d)
        stat, p = U.wilcoxon_safe(a.values, b.values)
        r.update({"inflation_mean": d.mean(), "inflation_ci_lo": lo, "inflation_ci_hi": hi,
                  "wilcoxon_W": stat, "wilcoxon_p": p, "rank_biserial": U.rank_biserial_wilcoxon(a.values, b.values)})
        t["Inflation (naive - group)"] = f"{d.mean():.3f} {U.fmt_ci(lo, hi)}"
        rows.append(r); tex.append(t)
    T6 = pd.DataFrame(rows)
    T6["wilcoxon_p_holm"] = U.holm(T6.wilcoxon_p.fillna(1).values)
    T6t = pd.DataFrame(tex)
    T6t["p (Holm)"] = [U.fmt_p(p) for p in T6.wilcoxon_p_holm]
    T6t.columns = ["Model", "Acc. raw-naive", "Acc. raw-group", "Acc. unique", "Inflation [95% CI]", "p (Holm)"]
    U.save_table(T6, "table_06_leakage_audit", df_tex=T6t)

    # ---------------- Figura 2 ----------------
    fig, axes = plt.subplots(1, 2, figsize=(U.DOUBLE_COL, 3.0), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    order = [m for m in MODELS if m != "Dummy"]
    yy = np.arange(len(order))
    style = {"raw-naive": ("#D55E00", "o", "Raw, random K-fold"),
             "raw-group": ("#0072B2", "s", "Raw, group K-fold"),
             "unique": ("#009E73", "^", "Unique profiles")}
    for j, (p, (c, mk, lab)) in enumerate(style.items()):
        mu, lo, hi = [], [], []
        for m in order:
            v = scores[(scores.model == m) & (scores.protocol == p)].accuracy.values
            l, h = U.bootstrap_ci(v); mu.append(v.mean()); lo.append(v.mean() - l); hi.append(h - v.mean())
        ax.errorbar(mu, yy + (j - 1) * 0.22, xerr=[lo, hi], fmt=mk, color=c, ms=4, capsize=2, lw=0.9, label=lab)
    ax.set_yticks(yy, order); ax.invert_yaxis(); ax.set_xlabel("Accuracy (mean, 95% bootstrap CI)")
    ax.legend(loc="lower center", fontsize=8, ncol=3, bbox_to_anchor=(0.5, 1.0), handletextpad=0.2,
              columnspacing=0.8)
    U.add_panel_label(ax, "a", x=-0.28)
    ax = axes[1]
    med_order = su.groupby("model").balanced_accuracy.mean().sort_values(ascending=False).index.tolist()
    import seaborn as sns
    sns.boxplot(data=su, y="model", x="balanced_accuracy", order=med_order, ax=ax, color="#BBBBBB",
                fliersize=0, width=0.6, linewidth=0.7)
    sns.stripplot(data=su, y="model", x="balanced_accuracy", order=med_order, ax=ax, size=1.6,
                  color="#0072B2", alpha=0.45, jitter=0.2)
    means = su.groupby("model").balanced_accuracy.mean().reindex(med_order)
    ax.scatter(means.values, np.arange(len(med_order)), marker="D", color="#D55E00", s=12, zorder=5, label="Mean")
    ax.axvline(1 / 3, color="k", ls=":", lw=0.7)
    ax.set_xlabel("Balanced accuracy (unique, 5×20 CV)"); ax.set_ylabel("")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=8)
    U.add_panel_label(ax, "b", x=-0.28)
    fig.tight_layout()
    U.save_figure(fig, "fig_02_main_performance_comparison")
    print(T2[["Model", "balanced_accuracy_mean", "balanced_accuracy_sd", "f1_macro_mean", "brier_mean"]].round(3).to_string())
    print(T6[["Model", "acc_raw-naive", "acc_raw-group", "acc_unique", "inflation_mean", "wilcoxon_p_holm"]].round(4).to_string())
    print(f"02 OK in {time.time() - t0:.1f}s")
