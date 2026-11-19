"""05_explainability_and_error.py
* SHAP (TreeExplainer) de LightGBM calculado sobre los folds de TEST (5x4 primeras repeticiones).
* Importancia por permutacion (caida de BA) en folds de test (5x20).
* Matriz de confusion OOF normalizada y diagrama de fiabilidad (clase High) + Brier.
* Casos de estudio: contrafactuales de cada metodo para 3 perfiles representativos (LightGBM, repeticion 0).

Salidas: figures/fig_04_explainability_shap_or_latent, figures/fig_05_error_calibration_or_case_study,
         tables/table_09_case_studies, results/shap_values.parquet, results/permutation_importance.csv
"""
import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix

import utils as U
from cf_lib import FEATURES, make_classifier

warnings.filterwarnings("ignore")
MODEL = "LightGBM"
uniq = pd.read_csv(U.DATA_PATH)
X, y = uniq[FEATURES].to_numpy(float), uniq["grade"].to_numpy()
splits = json.load(open(U.RESULTS_DIR / "splits_unique.json"))

# --------------------------- SHAP + permutacion ---------------------------
shap_rows, perm_rows = [], []
for sp in splits:
    tr, te = np.array(sp["train"]), np.array(sp["test"])
    m = make_classifier(MODEL).fit(X[tr], y[tr])
    pi = permutation_importance(m, X[te], y[te], scoring="balanced_accuracy", n_repeats=10,
                                random_state=U.SEED)
    for f, mu in zip(FEATURES, pi.importances_mean):
        perm_rows.append({"repeat": sp["repeat"], "fold": sp["fold"], "feature": f, "importance": mu})
    if sp["repeat"] < 4:
        sv = shap.TreeExplainer(m).shap_values(X[te])
        sv = np.stack(sv, axis=-1) if isinstance(sv, list) else sv          # (n, d, classes)
        for k, i in enumerate(te):
            for j, f in enumerate(FEATURES):
                shap_rows.append({"repeat": sp["repeat"], "idx": int(i), "feature": f, "value": X[i, j],
                                  **{f"shap_{c}": float(sv[k, j, c]) for c in range(3)}})
SH = pd.DataFrame(shap_rows); SH.to_parquet(U.RESULTS_DIR / "shap_values.parquet", index=False)
PI = pd.DataFrame(perm_rows); PI.to_csv(U.RESULTS_DIR / "permutation_importance.csv", index=False)

fig, axes = plt.subplots(1, 3, figsize=(U.DOUBLE_COL, 2.8), gridspec_kw={"width_ratios": [1, 1.25, 1]})
ax = axes[0]
imp = SH.assign(**{f"a{c}": SH[f"shap_{c}"].abs() for c in range(3)}).groupby("feature")[["a0", "a1", "a2"]].mean()
imp = imp.loc[imp.sum(1).sort_values().index]
left = np.zeros(len(imp))
for c in range(3):
    ax.barh(imp.index, imp[f"a{c}"], left=left, color=U.CLASS_COLORS[c], label=U.CLASS_NAMES[c], edgecolor="k", lw=0.3)
    left += imp[f"a{c}"].values
ax.set_xlabel("mean |SHAP| (log-odds)"); ax.legend(fontsize=8, loc="lower right")
ax.set_xlim(0, imp.sum(1).max() * 1.55)
U.add_panel_label(ax, "a", x=-0.45)

ax = axes[1]  # beeswarm manual para la clase High
order = imp.index[::-1].tolist()
rng = np.random.default_rng(U.SEED)
for r, f in enumerate(order):
    s = SH[SH.feature == f]
    v = s.value.values
    vn = (v - v.min()) / (np.ptp(v) if np.ptp(v) > 0 else 1)
    sc = ax.scatter(s.shap_2, r + rng.uniform(-0.28, 0.28, len(s)), c=vn, cmap="cividis", s=3, alpha=0.7, lw=0)
ax.set_yticks(range(len(order)), order); ax.invert_yaxis(); ax.axvline(0, color="k", lw=0.6)
ax.set_xlabel("SHAP value for class High")
cb = fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.02); cb.set_label("Feature value (low → high)"); cb.set_ticks([])
U.add_panel_label(ax, "b", x=-0.3)

ax = axes[2]
pis = PI.groupby("feature").importance
pm = pis.mean().sort_values()
ci = {f: U.bootstrap_ci(PI[PI.feature == f].importance.values) for f in pm.index}
ax.errorbar(pm.values, range(len(pm)), xerr=[[pm[f] - ci[f][0] for f in pm.index], [ci[f][1] - pm[f] for f in pm.index]],
            fmt="o", color="#0072B2", ms=4, capsize=2)
ax.set_yticks(range(len(pm)), pm.index); ax.axvline(0, color="k", lw=0.6, ls="--")
ax.set_xlabel("Permutation ΔBA")
U.add_panel_label(ax, "c", x=-0.45)
fig.tight_layout()
U.save_figure(fig, "fig_04_explainability_shap_or_latent")

# --------------------------- Error y calibracion ---------------------------
oof = pd.read_parquet(U.RESULTS_DIR / "oof_predictions.parquet")
fig, axes = plt.subplots(1, 3, figsize=(U.DOUBLE_COL, 2.5), gridspec_kw={"width_ratios": [1, 1, 1.1]})
for ax, mname, lab in ((axes[0], MODEL, "a"), (axes[1], "DomainRule", "b")):
    o = oof[oof.model == mname]
    cm = confusion_matrix(o.y_true, o.y_pred, labels=[0, 1, 2], normalize="true")
    ax.imshow(cm, cmap="cividis", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if cm[i, j] < 0.5 else "black")
    ax.set_xticks(range(3), U.CLASS_NAMES); ax.set_yticks(range(3), U.CLASS_NAMES); ax.grid(False)
    ax.set_xlabel(f"Predicted ({mname})"); ax.set_ylabel("True")
    U.add_panel_label(ax, lab, x=-0.42)
ax = axes[2]
brier = {}
for mname, c, mk in ((MODEL, "#0072B2", "o"), ("XGBoost", "#009E73", "s"), ("RF", "#E69F00", "^"), ("LR", "#CC79A7", "D")):
    o = oof[oof.model == mname]
    yt = (o.y_true == 2).astype(int).values
    fp, mp = calibration_curve(yt, o.p_high.values, n_bins=8, strategy="quantile")
    ax.plot(mp, fp, marker=mk, color=c, ms=3.5, lw=1, label=mname)
    brier[mname] = float(np.mean((o.p_high.values - yt) ** 2))
ax.plot([0, 1], [0, 1], "k:", lw=0.8)
ax.set_xlabel("Predicted P(High)"); ax.set_ylabel("Observed frequency"); ax.legend(fontsize=8, loc="upper left")
U.add_panel_label(ax, "c", x=-0.3)
fig.tight_layout()
U.save_figure(fig, "fig_05_error_calibration_or_case_study", despine=False)
json.dump(brier, open(U.RESULTS_DIR / "brier_high_ovr.json", "w", encoding="utf-8"), indent=2)

# --------------------------- Casos de estudio ---------------------------
cf = pd.read_parquet(U.RESULTS_DIR / "cf_instances_annotated.parquet")
c0 = cf[(cf.classifier == MODEL) & (cf.repeat == 0)]
cases = []
feas = c0[(c0.method == "Causal-SCM") & (c0.feasible == 1)].sort_values("idx")
infeas_ph = c0[(c0.method == "Causal-SCM") & (c0.feasible == 0) & (c0.idx.map(lambda i: not 6.4 <= X[i, 0] <= 6.8))]
infeas_sens = c0[(c0.method == "Causal-SCM") & (c0.feasible == 0) & (c0.idx.map(lambda i: 6.4 <= X[i, 0] <= 6.8))]
for lab, d in (("A: recourse exists", feas), ("B: abnormal pH", infeas_ph), ("C: sensory defects", infeas_sens)):
    if len(d):
        cases.append((lab, int(d.idx.iloc[0])))
rows, tex = [], []
for lab, i in cases:
    rows.append({"case": lab, "idx": i, "method": "Factual", **dict(zip(FEATURES, X[i])), "grade_true": int(y[i]),
                 "impl_validity": np.nan})
    for m in ["NUN", "Wachter-grid", "DiCE-random", "Actionable-indep", "Causal-SCM"]:
        r = c0[(c0.idx == i) & (c0.method == m)].iloc[0]
        vals = r.cf_first if isinstance(r.cf_first, (list, np.ndarray)) and r.feasible == 1 else [np.nan] * 7
        rows.append({"case": lab, "idx": i, "method": m, **dict(zip(FEATURES, vals)), "grade_true": int(y[i]),
                     "impl_validity": r.impl_validity})
T9 = pd.DataFrame(rows)
T9t = T9.copy()
for f in FEATURES:
    T9t[f] = [("--" if pd.isna(v) else (f"{v:.1f}" if f in ("ph",) else f"{v:.0f}") if f != "ph" else ("--" if pd.isna(v) else f"{v:.2f}"))
              for v in T9[f]]
T9t["impl_validity"] = T9.impl_validity.map(lambda v: "--" if pd.isna(v) else f"{v:.2f}")
T9t = T9t.drop(columns=["idx", "grade_true"])
T9t.columns = ["Case", "Method", "pH", "Temp.", "Taste", "Odor", "Fat", "Turb.", "Colour", "Impl. validity"]
U.save_table(T9, "table_09_case_studies", df_tex=T9t)
print(imp.sum(1).sort_values(ascending=False).round(3).to_string())
print(pm.sort_values(ascending=False).round(4).to_string())
print("Brier(High, OvR):", {k: round(v, 4) for k, v in brier.items()})
print(T9.round(3).to_string())
print("05 OK")
