"""01_data_splitting_and_prep.py
Carga de datos, definicion X/y/grupos, esquemas de CV anti-leakage, tabla de cohorte y figura 1.

Salidas: results/splits_unique.json, results/splits_raw_naive.json, results/splits_raw_group.json,
         tables/table_01_cohort_summary.(csv|tex), figures/fig_01_data_distribution_or_manifold.(pdf|png)
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold, StratifiedKFold

import utils as U
from cf_lib import FEATURES

uniq = pd.read_csv(U.DATA_PATH)
raw = pd.read_csv(U.RAW_GROUPED_PATH)
X_u, y_u = uniq[FEATURES].to_numpy(float), uniq["grade"].to_numpy()
X_r, y_r, g_r = raw[FEATURES].to_numpy(float), raw["grade"].to_numpy(), raw["group_id"].to_numpy()
assert len(uniq) == 83 and len(raw) == 1059
# Columnas excluidas de X: group_id, n_copies (multiplicidad -> fuga), features derivadas (redundantes)

# --------------------------------------------------------------------------
# Esquemas de CV (indices persistidos para que 02-05 usen exactamente los mismos folds)
# --------------------------------------------------------------------------
def dump(splits, name):
    with open(U.RESULTS_DIR / name, "w", encoding="utf-8") as f:
        json.dump([{"repeat": r, "fold": k, "train": tr.tolist(), "test": te.tolist()} for r, k, tr, te in splits], f)

rskf = RepeatedStratifiedKFold(n_splits=U.PRED_SPLITS, n_repeats=U.PRED_REPEATS, random_state=U.SEED)
dump([(i // U.PRED_SPLITS, i % U.PRED_SPLITS, tr, te) for i, (tr, te) in enumerate(rskf.split(X_u, y_u))],
     "splits_unique.json")

naive, grouped = [], []
for r in range(U.PRED_REPEATS):
    skf = StratifiedKFold(n_splits=U.PRED_SPLITS, shuffle=True, random_state=U.SEED + r)
    for k, (tr, te) in enumerate(skf.split(X_r, y_r)):
        naive.append((r, k, tr, te))
    sgkf = StratifiedGroupKFold(n_splits=U.PRED_SPLITS, shuffle=True, random_state=U.SEED + r)
    for k, (tr, te) in enumerate(sgkf.split(X_r, y_r, g_r)):
        assert not set(g_r[tr]) & set(g_r[te])
        grouped.append((r, k, tr, te))
dump(naive, "splits_raw_naive.json")
dump(grouped, "splits_raw_group.json")

leak = [np.isin(g_r[te], g_r[tr]).mean() for _, _, tr, te in naive]
leak_info = {"naive_test_rows_with_twin_mean": float(np.mean(leak)), "naive_min": float(np.min(leak)),
             "naive_max": float(np.max(leak)), "grouped_test_rows_with_twin": 0.0}
json.dump(leak_info, open(U.RESULTS_DIR / "leakage_overlap.json", "w", encoding="utf-8"), indent=2)

rcf = RepeatedStratifiedKFold(n_splits=U.CF_SPLITS, n_repeats=U.CF_REPEATS, random_state=U.SEED)
dump([(i // U.CF_SPLITS, i % U.CF_SPLITS, tr, te) for i, (tr, te) in enumerate(rcf.split(X_u, y_u))],
     "splits_cf.json")

# --------------------------------------------------------------------------
# Tabla 1: resumen de cohorte (unicos), por clase, + conteo crudo
# --------------------------------------------------------------------------
rows = []
labels = {0: "Low", 1: "Medium", 2: "High"}
def cell(s, f):
    if f in ("taste", "odor", "fat", "turbidity"):
        return s.mean(), f"{100 * s.mean():.1f}\\%"
    return s.median(), f"{s.median():.1f} [{s.quantile(.25):.1f}--{s.quantile(.75):.1f}]"

csv_rows, tex_rows = [], []
n_row = {"Variable": "n unique profiles (raw rows)"}
n_tex = dict(n_row)
for g in (0, 1, 2):
    nu, nr = int((y_u == g).sum()), int((y_r == g).sum())
    n_row[labels[g]] = nu; n_tex[labels[g]] = f"{nu} ({nr})"
n_row["All"] = len(y_u); n_tex["All"] = f"{len(y_u)} ({len(y_r)})"
csv_rows.append(n_row); tex_rows.append(n_tex)
from scipy import stats
for f in FEATURES:
    r_csv, r_tex = {"Variable": f}, {"Variable": f}
    for g in (0, 1, 2):
        v, t = cell(uniq.loc[uniq.grade == g, f], f)
        r_csv[labels[g]], r_tex[labels[g]] = v, t
    v, t = cell(uniq[f], f)
    r_csv["All"], r_tex["All"] = v, t
    if f in ("taste", "odor", "fat", "turbidity"):
        chi2, p, _, _ = stats.chi2_contingency(pd.crosstab(uniq[f], uniq.grade), correction=False)
        r_csv["test"], r_tex["test"] = "chi2", "$\\chi^2$"
    else:
        _, p = stats.kruskal(*[uniq.loc[uniq.grade == g, f] for g in (0, 1, 2)])
        r_csv["test"], r_tex["test"] = "Kruskal-Wallis", "KW"
    r_csv["p_value"], r_tex["p"] = p, U.fmt_p(p)
    csv_rows.append(r_csv); tex_rows.append(r_tex)
n_tex.update({"test": "", "p": ""})
U.save_table(pd.DataFrame(csv_rows), "table_01_cohort_summary", df_tex=pd.DataFrame(tex_rows))

# --------------------------------------------------------------------------
# Figura 1: (a) clases crudo vs unicos, (b) multiplicidad, (c) pH vs temperatura
# --------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(U.DOUBLE_COL, 2.5), gridspec_kw={"width_ratios": [1, 1, 1.25]})
ax = axes[0]
pr = pd.Series(y_r).value_counts(normalize=True).sort_index()
pu = pd.Series(y_u).value_counts(normalize=True).sort_index()
xx = np.arange(3); w = 0.38
ax.bar(xx - w / 2, pr.values, w, color="#999999", label="Raw (n = 1059)", hatch="//", edgecolor="k", lw=0.4)
ax.bar(xx + w / 2, pu.values, w, color=[U.CLASS_COLORS[i] for i in range(3)], label="Unique (n = 83)",
       edgecolor="k", lw=0.4)
ax.set_xticks(xx, U.CLASS_NAMES); ax.set_ylabel("Proportion"); ax.legend(loc="upper right")
ax.set_ylim(0, 0.55)
U.add_panel_label(ax, "a")

ax = axes[1]
mult = raw.groupby("group_id").agg(n=("n_copies", "first"), grade=("grade", "first"))
bins = np.arange(0, 56, 5)
for g in (0, 1, 2):
    ax.hist(mult.loc[mult.grade == g, "n"], bins=bins, histtype="step", lw=1.4, color=U.CLASS_COLORS[g],
            label=U.CLASS_NAMES[g], ls=["-", "--", ":"][g])
ax.set_xlabel("Copies per unique profile"); ax.set_ylabel("Unique profiles"); ax.legend()
U.add_panel_label(ax, "b")

ax = axes[2]
ax.add_patch(Rectangle((30, 6.4), 15, 0.4, color="#009E73", alpha=0.15, lw=0))
jit = np.random.default_rng(U.SEED)
for g, mk in zip((0, 1, 2), ("o", "s", "^")):
    s = uniq[uniq.grade == g]
    ax.scatter(s.temperature + jit.uniform(-0.6, 0.6, len(s)), s.ph + jit.uniform(-0.05, 0.05, len(s)),
               c=U.CLASS_COLORS[g], marker=mk, s=16, label=U.CLASS_NAMES[g], edgecolor="k", lw=0.3, alpha=0.9)
ax.axvline(45, color="k", ls="--", lw=0.7); ax.axhline(6.4, color="k", ls=":", lw=0.6); ax.axhline(6.8, color="k", ls=":", lw=0.6)
ax.set_xlabel("Temperature (°C)"); ax.set_ylabel("pH"); ax.legend(loc="lower right", handletextpad=0.2)
ax.set_xlim(30, 93)
U.add_panel_label(ax, "c")
fig.tight_layout()
U.save_figure(fig, "fig_01_data_distribution_or_manifold")
print("01 OK", leak_info)
