"""04_statistical_tests.py
Pruebas estadisticas robustas.

Prediccion (P3 unique, 5x20):
  * Friedman + Nemenyi (CD) sobre BA por fold (10 modelos).
  * Referencia = LightGBM (mejor ML): Wilcoxon sobre medias por repeticion (n = 20) y t-test corregido de
    Nadeau-Bengio sobre los 100 folds; Holm; rank-biserial r. McNemar sobre OOF de la repeticion 0.
Contrafactuales (5 clasificadores, 5x10):
  * Unidad estadistica primaria = INSTANCIA (promedio sobre clasificadores y repeticiones; conservador).
    Robustez: unidad = (clasificador, instancia).
  * Friedman + Nemenyi sobre validez al implementar (endpoint primario, infactible = 0).
  * Causal-SCM vs cada baseline: Wilcoxon pareado, Holm por familia de metrica, delta de Cliff,
    r de rango-biserial, IC bootstrap 95% de la diferencia media pareada.
  * Secundario: subconjunto donde el recurso causal es factible (estratificacion declarada post hoc).

Salidas: tables/table_04_statistical_significance, tables/table_05_counterfactual_results,
         tables/table_05b_counterfactual_by_classifier, figures/fig_06_counterfactual_recourse,
         figures/fig_08_critical_difference
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

import utils as U

scores = pd.read_csv(U.RESULTS_DIR / "fold_scores.csv")
oof = pd.read_parquet(U.RESULTS_DIR / "oof_predictions.parquet")
cf = pd.read_parquet(U.RESULTS_DIR / "cf_instances.parquet")
REF = "LightGBM"
PROPOSED = "Causal-SCM"
METHODS = ["NUN", "Wachter-grid", "DiCE-random", "Actionable-indep", "Causal-SCM"]
rows = []

# =============================================================================
# 1. Prediccion
# =============================================================================
su = scores[scores.protocol == "unique"]
M = su.pivot_table(index=["repeat", "fold"], columns="model", values="balanced_accuracy")
fr_stat, fr_p, pred_ranks, pred_cd, pred_n = U.friedman_nemenyi(M)
rows.append({"family": "Prediction (BA)", "comparison": f"Friedman across {M.shape[1]} models", "metric": "balanced_accuracy",
             "n": pred_n, "statistic": fr_stat, "p": fr_p, "p_holm": np.nan, "effect": np.nan,
             "effect_type": f"Nemenyi CD = {pred_cd:.3f}", "diff_mean": np.nan, "diff_ci_lo": np.nan, "diff_ci_hi": np.nan})
rep = su.groupby(["model", "repeat"]).balanced_accuracy.mean().unstack(0)
pred_rows = []
for m in M.columns:
    if m == REF:
        continue
    d_fold = (M[REF] - M[m]).values
    t_nb, p_nb = U.nadeau_bengio_ttest(d_fold, n_train=66, n_test=17)
    W, p_w = U.wilcoxon_safe(rep[REF].values, rep[m].values)
    lo, hi = U.bootstrap_ci(rep[REF].values - rep[m].values)
    pred_rows.append({"family": "Prediction (BA)", "comparison": f"{REF} vs {m}", "metric": "balanced_accuracy",
                      "n": len(rep), "statistic": W, "p": p_w, "effect": U.rank_biserial_wilcoxon(rep[REF], rep[m]),
                      "effect_type": "rank-biserial r", "diff_mean": float((rep[REF] - rep[m]).mean()),
                      "diff_ci_lo": lo, "diff_ci_hi": hi, "t_nadeau_bengio": t_nb, "p_nadeau_bengio": p_nb})
pr = pd.DataFrame(pred_rows)
pr["p_holm"] = U.holm(pr.p.values)
pr["p_nadeau_bengio_holm"] = U.holm(pr.p_nadeau_bengio.fillna(1).values)
# McNemar (OOF, repeticion 0)
o0 = oof[oof.repeat == 0].pivot_table(index="idx", columns="model", values="y_pred")
yt = oof[(oof.repeat == 0)].drop_duplicates("idx").set_index("idx").y_true.reindex(o0.index)
mc = []
for m in o0.columns:
    if m == REF:
        continue
    a, b = o0[REF] == yt, o0[m] == yt
    tab = [[int((a & b).sum()), int((a & ~b).sum())], [int((~a & b).sum()), int((~a & ~b).sum())]]
    r = mcnemar(tab, exact=True)
    mc.append({"model": m, "b_ref_only": tab[0][1], "c_other_only": tab[1][0], "p_mcnemar": r.pvalue})
mc = pd.DataFrame(mc)
mc["p_mcnemar_holm"] = U.holm(mc.p_mcnemar.values)
pr = pr.merge(mc.rename(columns={"model": "m"}).assign(comparison=lambda d: REF + " vs " + d.m).drop(columns="m"),
              on="comparison", how="left")
rows += pr.to_dict("records")

# =============================================================================
# 2. Contrafactuales
# =============================================================================
key = ["classifier", "repeat", "fold", "idx"]
cf["false_promise"] = ((cf.nominal_valid == 1) & (cf.impl_validity < 0.5)).astype(float)
cf["promise_gap"] = cf.nominal_valid.fillna(0) - cf.impl_validity
cfeas = cf[cf.method == PROPOSED][key + ["feasible"]].rename(columns={"feasible": "causal_feasible"})
cf = cf.merge(cfeas, on=key)
cf.to_parquet(U.RESULTS_DIR / "cf_instances_annotated.parquet", index=False)

CF_METRICS = [  # (metrica, mayor_es_mejor, subconjunto)
    ("impl_validity", True, "all"),
    ("false_promise", False, "all"),
    ("promise_gap", False, "all"),
    ("impl_validity", True, "causal-feasible"),
    ("action_cost", False, "causal-feasible"),
    ("causal_inconsistency", False, "causal-feasible"),
    ("lof_score", False, "causal-feasible"),
]


def blocks(df, metric, unit):
    by = ["idx"] if unit == "instance" else ["classifier", "idx"]
    return df.pivot_table(index=by, columns="method", values=metric, aggfunc="mean")


cf_rows = []
friedman_cf = {}
for unit in ("instance", "classifier×instance"):
    for metric, hib, subset in CF_METRICS:
        d = cf if subset == "all" else cf[cf.causal_feasible == 1]
        B = blocks(d, metric, unit).dropna()
        fam = f"CF [{unit}] {metric} ({subset})"
        if subset == "all" and metric == "impl_validity":
            s, p, rk, cd, n = U.friedman_nemenyi(B[METHODS], higher_is_better=True)
            friedman_cf[unit] = (rk, cd, n, s, p)
            cf_rows.append({"family": fam, "comparison": "Friedman across 5 methods", "metric": metric, "n": n,
                            "statistic": s, "p": p, "effect": np.nan, "effect_type": f"Nemenyi CD = {cd:.3f}"})
        fam_rows = []
        for m in METHODS:
            if m == PROPOSED:
                continue
            x, z = B[PROPOSED].values, B[m].values
            W, p = U.wilcoxon_safe(x, z)
            lo, hi = U.bootstrap_ci(x - z)
            delta = U.cliffs_delta(x, z)
            fam_rows.append({"family": fam, "comparison": f"{PROPOSED} vs {m}", "metric": metric, "n": len(B),
                             "statistic": W, "p": p, "effect": delta, "effect_type": "Cliff's delta",
                             "magnitude": U.cliffs_magnitude(delta), "rank_biserial": U.rank_biserial_wilcoxon(x, z),
                             "diff_mean": float((x - z).mean()), "diff_ci_lo": lo, "diff_ci_hi": hi,
                             "proposed_mean": float(x.mean()), "baseline_mean": float(z.mean()),
                             "favours_proposed": bool((x - z).mean() > 0) == hib})
        fr = pd.DataFrame(fam_rows)
        fr["p_holm"] = U.holm(fr.p.values)
        cf_rows += fr.to_dict("records")
rows += cf_rows
T4 = pd.DataFrame(rows)
T4.to_csv(U.RESULTS_DIR / "all_statistical_tests.csv", index=False)

# Version compacta para el paper: prediccion (vs ref) + CF a nivel instancia
show = T4[(T4.family.str.startswith("Prediction")) | (T4.family.str.contains(r"\[instance\]"))].copy()
def interp(r):
    if "Friedman" in r.comparison:
        return "differences among methods" if r.p < U.ALPHA else "no global difference"
    ph = r.p_holm
    if pd.isna(ph):
        return ""
    if ph >= U.ALPHA:
        return "n.s."
    fav = r.get("favours_proposed")
    if r.family.startswith("Prediction"):
        return f"{REF} better" if r.diff_mean > 0 else f"{REF} worse"
    return "favours Causal-SCM" if fav else "favours baseline"
show["interpretation"] = show.apply(interp, axis=1)
T4t = pd.DataFrame({
    "Family": show.family.str.replace("CF [instance] ", "CF: ", regex=False).str.replace("Prediction (BA)", "Pred. BA", regex=False),
    "Comparison": show.comparison,
    "n": show.n.astype(int),
    "Stat.": show.statistic.map(lambda v: f"{v:.2f}" if pd.notna(v) else "--"),
    "p": show.p.map(U.fmt_p), "p (Holm)": show.p_holm.map(U.fmt_p),
    "Effect": [f"{e:+.3f}" if pd.notna(e) else t for e, t in zip(show.effect, show.effect_type)],
    "Diff. [95% CI]": [f"{d:+.3f} {U.fmt_ci(l, h)}" if pd.notna(d) else "--"
                       for d, l, h in zip(show.diff_mean, show.diff_ci_lo, show.diff_ci_hi)],
    "Interpretation": show.interpretation,
})
U.save_table(show, "table_04_statistical_significance", df_tex=T4t)

# =============================================================================
# 3. Tabla 5: metricas de contrafactuales por metodo (unidad = instancia)
# =============================================================================
def per_inst(df, col):
    return df.groupby("idx")[col].mean()

t5, t5t = [], []
for m in METHODS:
    d = cf[cf.method == m]
    df_ = d[d.causal_feasible == 1]
    iv = per_inst(d, "impl_validity"); lo, hi = U.bootstrap_ci(iv.values)
    ivf = per_inst(df_, "impl_validity"); lof, hif = U.bootstrap_ci(ivf.values)
    r = {"Method": m, "n_rows": len(d), "n_instances": d.idx.nunique(),
         "feasibility": d.feasible.mean(), "nominal_validity": d.nominal_valid.mean(),
         "impl_validity_mean": iv.mean(), "impl_validity_sd": iv.std(ddof=1), "impl_validity_ci_lo": lo, "impl_validity_ci_hi": hi,
         "impl_validity_causal_feasible": ivf.mean(), "impl_validity_cf_ci_lo": lof, "impl_validity_cf_ci_hi": hif,
         "false_promise_rate": d.false_promise.mean(), "promise_gap": d.promise_gap.mean(),
         "action_cost": d.action_cost.mean(), "distance_l1": d.distance_l1.mean(), "sparsity": d.sparsity.mean(),
         "actionability_violation": d.actionability_violation.mean(), "causal_inconsistency": d.causal_inconsistency.mean(),
         "lof_outlier_rate": d.lof_outlier.mean(), "nn_distance": d.nn_distance.mean()}
    t5.append(r)
    t5t.append({"Method": m, "Feasible": f"{r['feasibility']:.3f}", "Promised": f"{r['nominal_validity']:.3f}",
                "Impl. validity [95% CI]": f"{r['impl_validity_mean']:.3f} {U.fmt_ci(lo, hi)}",
                "Impl. | causal-feasible": f"{r['impl_validity_causal_feasible']:.3f} {U.fmt_ci(lof, hif)}",
                "False promise": f"{r['false_promise_rate']:.3f}", "Action cost": f"{r['action_cost']:.3f}",
                "L1": f"{r['distance_l1']:.3f}", "Sparsity": f"{r['sparsity']:.2f}",
                "Non-action. edits": f"{r['actionability_violation']:.3f}",
                "Causal incons.": f"{r['causal_inconsistency']:.3f}", "LOF outlier": f"{r['lof_outlier_rate']:.3f}"})
T5 = pd.DataFrame(t5); T5t = pd.DataFrame(t5t)
for col, src, hib in (("Impl. validity [95% CI]", "impl_validity_mean", True),
                      ("Impl. | causal-feasible", "impl_validity_causal_feasible", True),
                      ("False promise", "false_promise_rate", False), ("Action cost", "action_cost", False),
                      ("Non-action. edits", "actionability_violation", False),
                      ("Causal incons.", "causal_inconsistency", False), ("LOF outlier", "lof_outlier_rate", False)):
    U.highlight(T5t, T5[src], col, hib)                      # mejor en negrita, segundo subrayado
    T5t = T5t.rename(columns={col: U.arrow(col, hib)})
U.save_table(T5, "table_05_counterfactual_results", df_tex=T5t)

T5b = cf.groupby(["classifier", "method"]).agg(impl_validity=("impl_validity", "mean"), feasibility=("feasible", "mean"),
                                              false_promise=("false_promise", "mean")).reset_index()
T5b = T5b.merge(cf[cf.causal_feasible == 1].groupby(["classifier", "method"]).impl_validity.mean()
                .rename("impl_validity_causal_feasible").reset_index(), on=["classifier", "method"])
T5bt = T5b.pivot(index="classifier", columns="method", values="impl_validity_causal_feasible")[METHODS].map(lambda v: f"{v:.3f}").reset_index()
T5bt.columns = ["Classifier"] + METHODS
U.save_table(T5b, "table_05b_counterfactual_by_classifier", df_tex=T5bt)

# =============================================================================
# 4. Figura 6: contrafactuales
# =============================================================================
COL = dict(zip(METHODS, ["#56B4E9", "#E69F00", "#CC79A7", "#009E73", "#0072B2"]))
MK = dict(zip(METHODS, ["o", "s", "D", "v", "^"]))
fig, axes = plt.subplots(2, 2, figsize=(U.DOUBLE_COL, 5.2))
ax = axes[0, 0]
for i, m in enumerate(METHODS):
    r = T5.set_index("Method").loc[m]
    ax.plot([r.nominal_validity, r.impl_validity_mean], [i, i], color="#888888", lw=1.2, zorder=1)
    ax.scatter(r.nominal_validity, i, marker="o", facecolor="white", edgecolor="k", s=30, zorder=2,
               label="Promised (nominal)" if i == 0 else None)
    ax.errorbar(r.impl_validity_mean, i, xerr=[[r.impl_validity_mean - r.impl_validity_ci_lo],
                                               [r.impl_validity_ci_hi - r.impl_validity_mean]],
                fmt=MK[m], color=COL[m], ms=6, capsize=2, zorder=3, label="Delivered (implemented)" if i == 0 else None)
ax.set_yticks(range(len(METHODS)), METHODS); ax.set_xlim(-0.02, 1.05); ax.set_xlabel("Validity")
ax.set_ylim(-0.5, len(METHODS) - 0.5 + 0.9)
ax.legend(loc="upper center", fontsize=8, ncol=2, bbox_to_anchor=(0.5, 1.02))
U.add_panel_label(ax, "a", x=-0.42)

ax = axes[0, 1]
w = 0.38
for j, (sub, lab, hatch) in enumerate(((1, "Actionable recourse exists", ""), (0, "No actionable recourse", "//"))):
    vals, los, his = [], [], []
    for m in METHODS:
        v = per_inst(cf[(cf.method == m) & (cf.causal_feasible == sub)], "impl_validity")
        l, h = U.bootstrap_ci(v.values); vals.append(v.mean()); los.append(v.mean() - l); his.append(h - v.mean())
    ax.bar(np.arange(len(METHODS)) + (j - 0.5) * w, vals, w, yerr=[los, his], capsize=2,
           color=[COL[m] for m in METHODS], hatch=hatch, edgecolor="k", lw=0.4, label=lab,
           alpha=1.0 if sub == 1 else 0.55)
ax.set_xticks(range(len(METHODS)), METHODS, fontsize=8, rotation=30, ha="right", rotation_mode="anchor")
ax.set_ylabel("Implementation validity"); ax.set_ylim(0, 1.05); ax.legend(fontsize=8, loc="upper left")
U.add_panel_label(ax, "b")

ax = axes[1, 0]
H = T5b.pivot(index="classifier", columns="method", values="false_promise")[METHODS]
im = ax.imshow(H.values, cmap="cividis_r", vmin=0, vmax=1, aspect="auto")
for i in range(H.shape[0]):
    for j in range(H.shape[1]):
        ax.text(j, i, f"{H.values[i, j]:.2f}", ha="center", va="center", fontsize=8,
                color="white" if H.values[i, j] > 0.5 else "black")
ax.set_xticks(range(len(METHODS)), METHODS, fontsize=8, rotation=30, ha="right", rotation_mode="anchor")
ax.set_yticks(range(H.shape[0]), H.index); ax.grid(False)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("False-promise rate")
U.add_panel_label(ax, "c", x=-0.3)

ax = axes[1, 1]
fe = cf[cf.causal_feasible == 1]
for m in METHODS:
    d = fe[fe.method == m].groupby("idx").agg(c=("action_cost", "mean"), l=("lof_score", "mean"))
    ax.scatter(d.c, d.l, s=10, marker=MK[m], color=COL[m], alpha=0.65, label=m, edgecolor="none")
ax.axhline(1.0, color="k", ls=":", lw=0.7)
ax.set_xlabel("Action cost (normalised L1, actionable)"); ax.set_ylabel("LOF score (lower = more plausible)")
ax.legend(fontsize=8, loc="upper right", markerscale=1.5)
U.add_panel_label(ax, "d")
fig.tight_layout()
U.save_figure(fig, "fig_06_counterfactual_recourse", despine=True)

# =============================================================================
# 5. Figura 8: diagramas de diferencia critica
# =============================================================================
fig, axes = plt.subplots(2, 1, figsize=(U.DOUBLE_COL, 4.4), gridspec_kw={"height_ratios": [1.25, 1]})
U.plot_cd_diagram(axes[0], pred_ranks, pred_cd)
U.add_panel_label(axes[0], "a", x=-0.12, y=0.95)
rk, cd, n, _, _ = friedman_cf["instance"]
U.plot_cd_diagram(axes[1], rk, cd)
U.add_panel_label(axes[1], "b", x=-0.12, y=0.95)
U.save_figure(fig, "fig_08_critical_difference", despine=False)

json.dump({"pred_friedman": [fr_stat, fr_p, pred_cd, pred_n], "pred_ranks": pred_ranks.to_dict(),
           "cf_friedman_instance": [friedman_cf["instance"][3], friedman_cf["instance"][4], cd, n],
           "cf_ranks_instance": rk.to_dict()}, open(U.RESULTS_DIR / "friedman_summary.json", "w", encoding="utf-8"), indent=2)
pd.set_option("display.width", 250)
print(f"Pred Friedman chi2={fr_stat:.2f} p={fr_p:.2e} CD={pred_cd:.3f}"); print(pred_ranks.round(2).to_string())
print(pr[["comparison", "diff_mean", "p", "p_holm", "p_nadeau_bengio", "p_nadeau_bengio_holm", "p_mcnemar_holm"]].round(4).to_string())
print(f"CF Friedman (instance) chi2={friedman_cf['instance'][3]:.2f} p={friedman_cf['instance'][4]:.2e} CD={cd:.3f} n={n}")
print(rk.round(2).to_string())
print(T4[T4.family.str.contains("CF")][["family", "comparison", "n", "p_holm", "effect", "diff_mean", "diff_ci_lo", "diff_ci_hi"]].round(4).to_string())
print(T5.round(3).T.to_string())
print("04 OK")
