"""07_external_statistics.py
Estadistica, tablas y figuras de la validacion externa + resumen cruzado de los 3 casos de calidad de leche.
Los datos NO se combinan: cada dataset es un caso de estudio independiente (muestra, vaca, granja); lo que se
compara entre casos es el comportamiento de los metodos.

Unidad estadistica (independencia): Milk Quality = perfil unico; heat_stress = VACA; scc_management = GRANJA.

Salidas:
  tables/table_10_external_cohorts, table_11_external_predictive, table_12_external_counterfactual,
  table_13_external_significance, table_14_cross_dataset_summary, table_15_external_dag_checks
  figures/fig_09_external_counterfactuals, fig_10_cross_dataset_summary, fig_11_external_sensitivity,
  fig_12_external_dags
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import utils as U
from cf_lib import SPECS, dag_edges

PROC = U.PROJECT_DIR / "data" / "processed"
METHODS = ["NUN", "Wachter-grid", "DiCE-random", "Actionable-indep", "Causal-SCM"]
PROPOSED = "Causal-SCM"
COL = dict(zip(METHODS, ["#56B4E9", "#E69F00", "#CC79A7", "#009E73", "#0072B2"]))
MK = dict(zip(METHODS, ["o", "s", "D", "v", "^"]))
EXT = ["heat_stress", "scc_management"]
LABEL = {"milk_quality": "Milk sample (grade)", "heat_stress": "Cow (SCC, heat stress)",
         "scc_management": "Farm (SCC, management)"}
SHORT = {"milk_quality": "Sample", "heat_stress": "Cow", "scc_management": "Farm"}
UNIT = {"milk_quality": "profile", "heat_stress": "cow", "scc_management": "farm"}


def load_cf(name):
    if name == "milk_quality":
        cf = pd.read_parquet(U.RESULTS_DIR / "cf_instances.parquet")
        cf["unit"] = cf["idx"]
    else:
        cf = pd.read_parquet(U.RESULTS_DIR / f"ext_{name}_cf_instances.parquet")
    key = ["classifier", "repeat", "fold", "idx"]
    cf["false_promise"] = ((cf.nominal_valid == 1) & (cf.impl_validity < 0.5)).astype(float)
    cf["promise_gap"] = cf.nominal_valid.fillna(0) - cf.impl_validity
    if "direction_violation" not in cf:
        cf["direction_violation"] = 0.0
    feas = cf[cf.method == PROPOSED][key + ["feasible"]].rename(columns={"feasible": "causal_feasible"})
    return cf.merge(feas, on=key)


CF = {n: load_cf(n) for n in ["milk_quality"] + EXT}

# =============================================================================
# Tabla 10: cohortes externas
# =============================================================================
rows = []
for name in EXT:
    d = pd.read_csv(PROC / f"{name}_ml_ready.csv")
    sp = SPECS[name]
    rows.append({"Case": LABEL[name], "Rows": len(d), "Independent units": d["cow_id"].nunique() if name == "heat_stress" else len(d),
                 "Unit": UNIT[name], "Quality target": "SCC > 200k cells/mL", "Prevalence": d.scc_high.mean(),
                 "Features": ", ".join(sp.features), "Actionable (direction)": ", ".join(f"{a} ({sp.directions[a]})" for a in sp.actionable),
                 "Immutable / structural": ", ".join(f for f in sp.features if f in sp.exogenous and f not in sp.actionable),
                 "Validation": "StratifiedGroupKFold by cow" if name == "heat_stress" else "RepeatedStratifiedKFold"})
T10 = pd.DataFrame(rows)
T10t = T10.copy(); T10t["Prevalence"] = T10t.Prevalence.map(lambda v: f"{v:.3f}")
U.save_table(T10, "table_10_external_cohorts", df_tex=T10t)

# =============================================================================
# Tabla 11: prediccion externa (+ fuga por medidas repetidas)
# =============================================================================
rows, tex = [], []
for name in EXT:
    s = pd.read_csv(U.RESULTS_DIR / f"ext_{name}_pred_scores.csv")
    main = "grouped" if name == "heat_stress" else "stratified"
    for m in s.model.unique():
        v = s[(s.model == m) & (s.protocol == main)]
        r = {"dataset": name, "model": m}
        t = {"Case": SHORT[name], "Model": m}
        for met in ("balanced_accuracy", "roc_auc", "f1_macro", "mcc", "brier"):
            lo, hi = U.bootstrap_ci(v[met].values)
            r.update({f"{met}_mean": v[met].mean(), f"{met}_sd": v[met].std(ddof=1), f"{met}_ci_lo": lo, f"{met}_ci_hi": hi})
            t[met] = U.fmt_mean_sd(v[met].values)
        if name == "heat_stress":
            lk = s[(s.model == m) & (s.protocol == "row-wise (leaky)")]
            a = lk.groupby("repeat").balanced_accuracy.mean(); b = v.groupby("repeat").balanced_accuracy.mean()
            d = (a - b).values; lo, hi = U.bootstrap_ci(d); _, p = U.wilcoxon_safe(a.values, b.values)
            r.update({"ba_rowwise_mean": lk.balanced_accuracy.mean(), "inflation": d.mean(), "inflation_ci_lo": lo,
                      "inflation_ci_hi": hi, "inflation_p": p})
            t["BA row-wise (leaky)"] = U.fmt_mean_sd(lk.balanced_accuracy.values)
            t["Inflation [95% CI]"] = f"{d.mean():+.3f} {U.fmt_ci(lo, hi)}"
        else:
            t["BA row-wise (leaky)"] = "--"; t["Inflation [95% CI]"] = "--"
        rows.append(r); tex.append(t)
T11 = pd.DataFrame(rows)
if "inflation_p" in T11:
    msk = T11.inflation_p.notna()
    T11.loc[msk, "inflation_p_holm"] = U.holm(T11.loc[msk, "inflation_p"].values)
T11t = pd.DataFrame(tex)
for met, hib in (("balanced_accuracy", True), ("roc_auc", True), ("f1_macro", True), ("mcc", True), ("brier", False)):
    U.highlight(T11t, T11[f"{met}_mean"], met, hib, groups=T11.dataset)
T11t.columns = ["Case", "Model", U.arrow("BA", True), U.arrow("AUC", True), U.arrow("F1-macro", True),
                U.arrow("MCC", True), U.arrow("Brier", False), "BA row-wise (leaky)", "Inflation [95% CI]"]
U.save_table(T11, "table_11_external_predictive", df_tex=T11t)

# =============================================================================
# Tabla 12 y 14: contrafactuales por metodo
# =============================================================================
def unit_mean(df, col):
    return df.groupby("unit")[col].mean()


def cf_summary(name):
    cf = CF[name]
    out = []
    for m in METHODS:
        d = cf[cf.method == m]
        f = d[d.causal_feasible == 1]
        iv = unit_mean(d, "impl_validity"); lo, hi = U.bootstrap_ci(iv.values)
        ivf = unit_mean(f, "impl_validity"); lof, hif = U.bootstrap_ci(ivf.values) if len(ivf) else (np.nan, np.nan)
        fp = unit_mean(d, "false_promise"); fpl, fph = U.bootstrap_ci(fp.values)
        out.append({"dataset": name, "method": m, "n_rows": len(d), "n_units": d.unit.nunique(),
                    "feasibility": d.feasible.mean(), "nominal_validity": d.nominal_valid.mean(),
                    "impl_validity": iv.mean(), "impl_ci_lo": lo, "impl_ci_hi": hi,
                    "impl_validity_causal_feasible": ivf.mean() if len(ivf) else np.nan, "impl_cf_ci_lo": lof, "impl_cf_ci_hi": hif,
                    "false_promise": fp.mean(), "fp_ci_lo": fpl, "fp_ci_hi": fph,
                    "action_cost": d.action_cost.mean(), "distance_l1": d.distance_l1.mean(),
                    "actionability_violation": d.actionability_violation.mean(),
                    "direction_violation": d.direction_violation.mean(),
                    "causal_inconsistency": d.causal_inconsistency.mean(), "lof_outlier": d.lof_outlier.mean()})
    return out


SUM = pd.DataFrame([r for n in ["milk_quality"] + EXT for r in cf_summary(n)])
SUM.to_csv(U.RESULTS_DIR / "cross_dataset_cf_summary.csv", index=False)
ext = SUM[SUM.dataset.isin(EXT)]
T12t = pd.DataFrame({
    "Case": ext.dataset.map(SHORT), "Method": ext.method,
    "Feasible": ext.feasibility.map(lambda v: f"{v:.3f}"), "Promised": ext.nominal_validity.map(lambda v: f"{v:.3f}"),
    "Impl. validity [95% CI]": [f"{a:.3f} {U.fmt_ci(l, h)}" for a, l, h in zip(ext.impl_validity, ext.impl_ci_lo, ext.impl_ci_hi)],
    "Impl. | causal-feasible": [f"{a:.3f} {U.fmt_ci(l, h)}" for a, l, h in zip(ext.impl_validity_causal_feasible, ext.impl_cf_ci_lo, ext.impl_cf_ci_hi)],
    "False promise": ext.false_promise.map(lambda v: f"{v:.3f}"),
    "Non-action. edits": ext.actionability_violation.map(lambda v: f"{v:.3f}"),
    "Direction viol.": ext.direction_violation.map(lambda v: f"{v:.3f}"),
    "Causal incons.": ext.causal_inconsistency.map(lambda v: f"{v:.3f}"),
    "Action cost": ext.action_cost.map(lambda v: f"{v:.3f}")}).reset_index(drop=True)
ext = ext.reset_index(drop=True)
for col, src, hib in (("Impl. validity [95% CI]", "impl_validity", True),
                      ("Impl. | causal-feasible", "impl_validity_causal_feasible", True),
                      ("False promise", "false_promise", False), ("Non-action. edits", "actionability_violation", False),
                      ("Direction viol.", "direction_violation", False), ("Causal incons.", "causal_inconsistency", False),
                      ("Action cost", "action_cost", False)):
    U.highlight(T12t, ext[src], col, hib, groups=ext.dataset)
    T12t = T12t.rename(columns={col: U.arrow(col, hib)})
U.save_table(ext, "table_12_external_counterfactual", df_tex=T12t)

# =============================================================================
# Tabla 13: significancia (unidad independiente) + Friedman
# =============================================================================
FAM = [("impl_validity", True, "all"), ("false_promise", False, "all"), ("impl_validity", True, "causal-feasible"),
       ("action_cost", False, "causal-feasible"), ("causal_inconsistency", False, "causal-feasible")]
rows = []
fried = {}
for name in ["milk_quality"] + EXT:
    cf = CF[name]
    for metric, hib, subset in FAM:
        d = cf if subset == "all" else cf[cf.causal_feasible == 1]
        B = d.pivot_table(index="unit", columns="method", values=metric, aggfunc="mean").dropna()
        if len(B) < 6:
            continue
        if metric == "impl_validity" and subset == "all":
            s, p, rk, cd, n = U.friedman_nemenyi(B[METHODS])
            fried[name] = {"chi2": s, "p": p, "cd": cd, "n": n, "ranks": rk.to_dict()}
            rows.append({"dataset": name, "family": f"{metric} ({subset})", "comparison": "Friedman (5 methods)",
                         "n": n, "statistic": s, "p": p, "effect": np.nan, "effect_type": f"Nemenyi CD = {cd:.3f}"})
        fam = []
        for m in METHODS:
            if m == PROPOSED:
                continue
            x, z = B[PROPOSED].values, B[m].values
            W, p = U.wilcoxon_safe(x, z); lo, hi = U.bootstrap_ci(x - z); dlt = U.cliffs_delta(x, z)
            rec = {"dataset": name, "family": f"{metric} ({subset})", "comparison": f"{PROPOSED} vs {m}",
                   "n": len(B), "statistic": W, "p": p, "effect": dlt, "effect_type": "Cliff's delta",
                   "magnitude": U.cliffs_magnitude(dlt), "diff_mean": float((x - z).mean()), "diff_ci_lo": lo,
                   "diff_ci_hi": hi, "favours_proposed": bool(((x - z).mean() > 0) == hib)}
            if metric == "impl_validity" and subset == "all":
                # R2.1: equivalencia formal (TOST pareado no parametrico, margen +-0.05 declarado a priori)
                pt, _, _, (l90, h90) = U.tost_wilcoxon(x, z, margin=0.05)
                rec.update({"tost_p_margin_0.05": pt, "diff_ci90_lo": l90, "diff_ci90_hi": h90})
            fam.append(rec)
        fam = pd.DataFrame(fam); fam["p_holm"] = U.holm(fam.p.values)
        rows += fam.to_dict("records")
T13 = pd.DataFrame(rows)
# R2.3: Holm global sobre las comparaciones confirmatorias (validez al implementar y promesas falsas; 3 casos x 4)
conf = T13.family.isin(["impl_validity (all)", "false_promise (all)"]) & ~T13.comparison.str.startswith("Friedman")
T13.loc[conf, "p_holm_global"] = U.holm(T13.loc[conf, "p"].values)
T13.to_csv(U.RESULTS_DIR / "cross_dataset_tests.csv", index=False)
json.dump(fried, open(U.RESULTS_DIR / "cross_dataset_friedman.json", "w", encoding="utf-8"), indent=2)
show = T13[T13.dataset.isin(EXT)].copy()


def interp(r):
    if "Friedman" in r.comparison:
        return "differences among methods" if r.p < U.ALPHA else "no global difference"
    if pd.isna(r.p_holm) or r.p_holm >= U.ALPHA:
        return "n.s."
    return "favours Causal-SCM" if r.favours_proposed else "favours baseline"


show["interpretation"] = show.apply(interp, axis=1)
T13t = pd.DataFrame({"Case": show.dataset.map(SHORT), "Family": show.family, "Comparison": show.comparison,
                     "n": show.n.astype(int), "p": show.p.map(U.fmt_p), "p (Holm)": show.p_holm.map(U.fmt_p),
                     "p (Holm, global)": show.p_holm_global.map(U.fmt_p),
                     "Effect": [f"{e:+.3f}" if pd.notna(e) else t for e, t in zip(show.effect, show.effect_type)],
                     "Diff. [95% CI]": [f"{d:+.3f} {U.fmt_ci(l, h)}" if pd.notna(d) else "--"
                                        for d, l, h in zip(show.diff_mean, show.diff_ci_lo, show.diff_ci_hi)],
                     "TOST p ($\\pm$0.05)": show["tost_p_margin_0.05"].map(U.fmt_p),
                     "Interpretation": show.interpretation})
U.save_table(show, "table_13_external_significance", df_tex=T13t)
# Tabla 13b: comparaciones confirmatorias de los 3 casos (incluye Muestra) con Holm global y TOST
cft = T13[conf].copy()
T13b = pd.DataFrame({"Case": cft.dataset.map(SHORT), "Endpoint": cft.family.str.replace(" (all)", "", regex=False),
                     "Comparison": cft.comparison, "n": cft.n.astype(int),
                     "Diff. [95% CI]": [f"{d:+.3f} {U.fmt_ci(l, h)}" for d, l, h in zip(cft.diff_mean, cft.diff_ci_lo, cft.diff_ci_hi)],
                     "Cliff's $\\delta$": cft.effect.map(lambda v: f"{v:+.3f}"),
                     "p (Holm, family)": cft.p_holm.map(U.fmt_p), "p (Holm, global)": cft.p_holm_global.map(U.fmt_p),
                     "TOST p ($\\pm$0.05)": cft["tost_p_margin_0.05"].map(U.fmt_p),
                     "Conclusion": ["superior" if (pg < U.ALPHA and f) else "inferior" if pg < U.ALPHA else
                                    ("equivalent" if (pd.notna(pt) and pt < U.ALPHA) else "inconclusive")
                                    for pg, f, pt in zip(cft.p_holm_global, cft.favours_proposed, cft["tost_p_margin_0.05"])]})
U.save_table(cft.assign(conclusion=T13b.Conclusion.values), "table_13b_confirmatory_tests", df_tex=T13b)

# Tabla 14: resumen cruzado (3 casos x 5 metodos)
T14 = SUM[["dataset", "method", "n_units", "feasibility", "nominal_validity", "impl_validity",
           "impl_validity_causal_feasible", "false_promise", "actionability_violation", "direction_violation"]].copy()
T14t = pd.DataFrame({"Case": T14.dataset.map(SHORT), "Method": T14.method, "Units": T14.n_units,
                     "Promised": T14.nominal_validity.map(lambda v: f"{v:.2f}"),
                     "Delivered": T14.impl_validity.map(lambda v: f"{v:.2f}"),
                     "Delivered | feasible": T14.impl_validity_causal_feasible.map(lambda v: f"{v:.2f}"),
                     "False promise": T14.false_promise.map(lambda v: f"{v:.2f}"),
                     "Infeasible edits": (T14.actionability_violation.fillna(0) + 0).map(lambda v: f"{v:.2f}"),
                     "Feasible": T14.feasibility.map(lambda v: f"{v:.2f}")})
for col, src, hib in (("Delivered", "impl_validity", True), ("Delivered | feasible", "impl_validity_causal_feasible", True),
                      ("False promise", "false_promise", False), ("Infeasible edits", "actionability_violation", False)):
    U.highlight(T14t, T14[src], col, hib, groups=T14.dataset)
    T14t = T14t.rename(columns={col: U.arrow(col, hib)})
U.save_table(T14, "table_14_cross_dataset_summary", df_tex=T14t)

# =============================================================================
# Tabla 15: chequeo de DAGs externos + sensibilidad al DAG
# =============================================================================
rows = []
for name in EXT:
    t = pd.read_csv(U.RESULTS_DIR / f"ext_{name}_dag_tests.csv")
    info = json.load(open(U.RESULTS_DIR / f"ext_{name}_pc.json", encoding="utf-8"))
    ds = pd.read_csv(U.RESULTS_DIR / f"ext_{name}_dag_sensitivity.csv")
    ds["false_promise"] = ((ds.nominal_valid == 1) & (ds.impl_validity < 0.5)).astype(float)
    cond = ds[ds.feasible == 1].groupby(["method_dag", "world_dag"]).impl_validity.mean().unstack()
    diag = np.mean([cond.loc[k, k] for k in cond.index if k in cond.columns])
    off = np.nanmean([cond.loc[a, b] for a in cond.index for b in cond.columns if a != b])
    exp_row = cond.loc["Expert"].drop("Expert").mean()
    rows.append({"dataset": name, "n_implications": len(t), "n_rejected_holm": int(t.rejected_holm.sum()),
                 "rejected": "; ".join(t[t.rejected_holm].implication), "pc_skeleton": "; ".join(f"{a}–{b}" for a, b in info["pc_skeleton"]),
                 "shd_expert_vs_pc": info["shd_skeleton"], "validity_matched_dag": diag,
                 "validity_mismatched_dag": off, "expert_validity_under_other_worlds": exp_row})
    cond.to_csv(U.RESULTS_DIR / f"ext_{name}_dag_matrix.csv")
T15 = pd.DataFrame(rows)
T15t = T15.copy()
for c in ("validity_matched_dag", "validity_mismatched_dag", "expert_validity_under_other_worlds"):
    T15t[c] = T15t[c].map(lambda v: f"{v:.3f}")
T15t["dataset"] = T15t.dataset.map(SHORT)
T15t.columns = ["Case", "Implications", "Rejected (Holm)", "Rejected implications", "PC skeleton", "SHD",
                "Valid | feasible (matched DAG)", "Valid | feasible (mismatched)", "Expert DAG under other worlds"]
U.save_table(T15, "table_15_external_dag_checks", df_tex=T15t)

# =============================================================================
# Figura 9: contrafactuales externos (2 filas x 3 paneles)
# =============================================================================
fig, axes = plt.subplots(2, 3, figsize=(U.DOUBLE_COL, 5.0))
labs = iter("abcdef")
for r, name in enumerate(EXT):
    S = SUM[SUM.dataset == name].set_index("method")
    cf = CF[name]
    ax = axes[r, 0]
    for i, m in enumerate(METHODS):
        s = S.loc[m]
        ax.plot([s.nominal_validity, s.impl_validity], [i, i], color="#888888", lw=1.1, zorder=1)
        ax.scatter(s.nominal_validity, i, facecolor="white", edgecolor="k", s=24, zorder=2,
                   label="Promised" if i == 0 else None)
        ax.errorbar(s.impl_validity, i, xerr=[[s.impl_validity - s.impl_ci_lo], [s.impl_ci_hi - s.impl_validity]],
                    fmt=MK[m], color=COL[m], ms=5, capsize=2, zorder=3, label="Delivered" if i == 0 else None)
    ax.set_yticks(range(len(METHODS)), METHODS if True else []); ax.set_xlim(-0.03, 1.05)
    ax.set_ylim(-0.6, len(METHODS) - 0.4 + 0.8)
    ax.set_xlabel("Validity"); ax.legend(fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.03))
    ax.text(-0.02, 1.08, SHORT[name], transform=ax.transAxes, fontsize=9, style="italic", ha="left")
    U.add_panel_label(ax, next(labs), x=-0.62)
    ax = axes[r, 1]
    w = 0.38
    for j, (sub, lab, hatch, al) in enumerate(((1, "Recourse exists", "", 1.0), (0, "No recourse", "//", 0.55))):
        vals, los, his = [], [], []
        for m in METHODS:
            v = unit_mean(cf[(cf.method == m) & (cf.causal_feasible == sub)], "impl_validity")
            if len(v) == 0:
                vals.append(0); los.append(0); his.append(0); continue
            l, h = U.bootstrap_ci(v.values); vals.append(v.mean()); los.append(v.mean() - l); his.append(h - v.mean())
        ax.bar(np.arange(len(METHODS)) + (j - 0.5) * w, vals, w, yerr=[los, his], capsize=1.5,
               color=[COL[m] for m in METHODS], hatch=hatch, edgecolor="k", lw=0.4, alpha=al, label=lab)
    ax.set_xticks(range(len(METHODS)), ["NUN", "Wach.", "DiCE", "Act.", "Causal"], fontsize=8, rotation=40,
                  ha="right", rotation_mode="anchor")
    ax.set_ylim(0, 1.42); ax.set_yticks(np.arange(0, 1.01, 0.2)); ax.set_ylabel("Impl. validity")
    ax.legend(fontsize=8, loc="upper left", ncol=1, handlelength=1.2, borderaxespad=0.2)
    U.add_panel_label(ax, next(labs), x=-0.3)
    ax = axes[r, 2]
    x = np.arange(len(METHODS))
    ax.bar(x - 0.27, S.loc[METHODS, "false_promise"], 0.27, color="#D55E00", edgecolor="k", lw=0.4, label="False promise")
    ax.bar(x, S.loc[METHODS, "actionability_violation"], 0.27, color="#999999", edgecolor="k", lw=0.4, label="Non-actionable edits")
    ax.bar(x + 0.27, S.loc[METHODS, "direction_violation"], 0.27, color="#F0E442", edgecolor="k", lw=0.4, label="Direction violation")
    ax.set_xticks(x, ["NUN", "Wach.", "DiCE", "Act.", "Causal"], fontsize=8, rotation=40, ha="right",
                  rotation_mode="anchor")
    ax.set_ylim(0, 1.62); ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("Rate"); ax.legend(fontsize=8, loc="upper right", handlelength=1.2, borderaxespad=0.2)
    U.add_panel_label(ax, next(labs), x=-0.3)
fig.set_size_inches(U.DOUBLE_COL, 5.6)
fig.tight_layout()
U.save_figure(fig, "fig_09_external_counterfactuals")

# =============================================================================
# Figura 10: resumen cruzado de los 3 casos
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(U.DOUBLE_COL, 2.7))
cases = ["milk_quality"] + EXT
xx = np.arange(len(cases))
w = 0.16
for k, (col, ylab, lo_c, hi_c) in enumerate((("false_promise", "False-promise rate", "fp_ci_lo", "fp_ci_hi"),
                                             ("impl_validity_causal_feasible", "Impl. validity | recourse exists", "impl_cf_ci_lo", "impl_cf_ci_hi"),
                                             ("impl_validity", "Impl. validity (all)", "impl_ci_lo", "impl_ci_hi"))):
    ax = axes[k]
    for i, m in enumerate(METHODS):
        s = SUM[SUM.method == m].set_index("dataset").reindex(cases)
        ax.bar(xx + (i - 2) * w, s[col], w, yerr=[s[col] - s[lo_c], s[hi_c] - s[col]], capsize=1.2,
               color=COL[m], edgecolor="k", lw=0.3, label=m, error_kw={"lw": 0.6})
    ax.set_xticks(xx, [SHORT[c] for c in cases]); ax.set_ylabel(ylab); ax.set_ylim(0, 1.08)
    U.add_panel_label(ax, "abc"[k], x=-0.28)
axes[0].legend(fontsize=8, loc="upper center", ncol=5, bbox_to_anchor=(1.85, 1.2), handlelength=1.2, columnspacing=0.8)
fig.tight_layout()
fig.subplots_adjust(wspace=0.55)
U.save_figure(fig, "fig_10_cross_dataset_summary")

# =============================================================================
# Figura 11: sensibilidad externa (tau + matriz DAG)
# =============================================================================
fig = plt.figure(figsize=(U.DOUBLE_COL, 5.0))
gs = fig.add_gridspec(2, 2, hspace=0.6, wspace=0.85)
for r, name in enumerate(EXT):
    t = pd.read_csv(U.RESULTS_DIR / f"ext_{name}_tau.csv")
    t["false_promise"] = ((t.nominal_valid == 1) & (t.impl_validity < 0.5)).astype(float)
    tp = t[t.abduction == "posterior"].groupby("tau").agg(feas=("feasible", "mean"), impl=("impl_validity", "mean"),
                                                          fp=("false_promise", "mean"))
    tp["cond"] = t[(t.abduction == "posterior") & (t.feasible == 1)].groupby("tau").impl_validity.mean()
    pr = t[t.abduction == "prior"]
    ax = fig.add_subplot(gs[r, 0])
    ax.plot(tp.index, tp.cond, "-o", color="#009E73", ms=3.5, label="Validity | feasible")
    ax.plot(tp.index, tp.feas, "--s", color="#0072B2", ms=3.5, label="Feasibility")
    ax.plot(tp.index, tp.impl, "-.^", color="#E69F00", ms=3.5, label="Impl. validity (all)")
    ax.plot(tp.index, tp.fp, ":D", color="#D55E00", ms=3.5, label="False promise")
    ax.scatter([0.8], [pr.feasible.mean()], marker="x", color="k", s=25, zorder=5, label="Feasibility (prior abduction)")
    ax.set_xlabel("Validity threshold τ"); ax.set_ylabel(f"Rate ({SHORT[name]})"); ax.set_ylim(-0.03, 1.05)
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(0.0, 0.33), handlelength=1.6,
              labelspacing=0.25, borderaxespad=0.2)
    U.add_panel_label(ax, "ac"[r], x=-0.25)
    ax = fig.add_subplot(gs[r, 1])
    cond = pd.read_csv(U.RESULTS_DIR / f"ext_{name}_dag_matrix.csv", index_col=0)
    order = list(cond.index)
    cond = cond.reindex(index=order, columns=order)
    im = ax.imshow(cond.values, cmap="cividis", vmin=0, vmax=1, aspect="auto")
    for i in range(len(order)):
        for j in range(len(order)):
            v = cond.values[i, j]
            ax.text(j, i, "--" if np.isnan(v) else f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if (np.isnan(v) or v < 0.6) else "black")
    ax.set_xticks(range(len(order)), order, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(order)), order, fontsize=8); ax.grid(False)
    ax.set_xlabel("World DAG", fontsize=8); ax.set_ylabel("Method DAG", fontsize=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("Validity | feasible", fontsize=8)
    U.add_panel_label(ax, "bd"[r], x=-0.5)
U.save_figure(fig, "fig_11_external_sensitivity", despine=False)

# =============================================================================
# Figura 12: DAGs externos (experto + skeleton PC)
# =============================================================================
POS = {"heat_stress": {"thi": (0, 1.0), "hf_pct": (0, -1.0), "rectal_temp_c": (1.3, 0.45), "resp_bpm": (1.3, 1.6),
                       "pulse_bpm": (2.6, 0.8), "dmi_kg": (2.6, -1.0)},
       "scc_management": {"ps": (0, 1.6), "tm": (1.0, 1.6), "mp": (2.0, 2.4), "ml": (2.0, 1.0), "its": (3.0, 1.9),
                          "cmt": (3.0, 0.6), "ppd": (4.0, 1.0), "td": (3.0, -0.6), "efj": (0, 0.0), "tw": (1.0, 0.0),
                          "pv": (1.9, -0.6)}}
NAMES = {"thi": "THI", "hf_pct": "%HF", "rectal_temp_c": "RT", "resp_bpm": "RR", "pulse_bpm": "PR", "dmi_kg": "DMI"}
fig, axes = plt.subplots(1, 2, figsize=(U.DOUBLE_COL, 2.8), gridspec_kw={"width_ratios": [1, 1.35]})
for ax, name, lab in zip(axes, EXT, "ab"):
    sp = SPECS[name]
    info = json.load(open(U.RESULTS_DIR / f"ext_{name}_pc.json", encoding="utf-8"))
    pos = POS[name]
    exp = dag_edges(sp.dag)
    pc_sk = {tuple(e) for e in info["pc_skeleton"]}
    for n, (px, py) in pos.items():
        c = "#E69F00" if n in sp.actionable else ("#BBBBBB" if n in sp.exogenous else "#56B4E9")
        ax.scatter(px, py, s=520, color=c, edgecolor="k", zorder=3, lw=0.7)
        ax.text(px, py, NAMES.get(n, n.upper()), ha="center", va="center", fontsize=8, zorder=4)
    for a, b in exp:
        in_pc = (a, b) in pc_sk or (b, a) in pc_sk
        ax.annotate("", xy=pos[b], xytext=pos[a], arrowprops=dict(arrowstyle="-|>", color="k", lw=1.3 if in_pc else 0.8,
                                                                  shrinkA=10, shrinkB=10))
    for a, b in pc_sk:
        if (a, b) not in exp and (b, a) not in exp:
            ax.annotate("", xy=pos[b], xytext=pos[a], arrowprops=dict(arrowstyle="-", color="#D55E00", lw=0.9, ls="--",
                                                                      shrinkA=10, shrinkB=10))
    xs, ys = zip(*pos.values())
    ax.set_xlim(min(xs) - 0.5, max(xs) + 0.5); ax.set_ylim(min(ys) - 0.6, max(ys) + 0.5); ax.axis("off")
    U.add_panel_label(ax, lab, x=0.0, y=0.98)
U.save_figure(fig, "fig_12_external_dags", despine=False)

pd.set_option("display.width", 250)
print(T11[["dataset", "model", "balanced_accuracy_mean", "roc_auc_mean"] + (["inflation"] if "inflation" in T11 else [])].round(3).to_string())
print(SUM[["dataset", "method", "n_units", "feasibility", "nominal_validity", "impl_validity", "impl_validity_causal_feasible",
           "false_promise", "actionability_violation", "direction_violation", "action_cost"]].round(3).to_string())
print(T13[["dataset", "family", "comparison", "n", "p_holm", "effect", "diff_mean", "diff_ci_lo", "diff_ci_hi"]].round(4).to_string())
print(json.dumps(fried, indent=1, default=float))
print(T15.drop(columns=["rejected"]).round(3).to_string())
print("07 OK")
