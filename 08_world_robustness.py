"""08_world_robustness.py
Respuesta a la revision por pares (ronda 1).

(A) R1.1 / R2.2 - Evaluacion no circular: los 5 metodos se evaluan en un MUNDO de otra clase funcional
    (SCM con ecuaciones Random Forest, mismo DAG, misma abduccion), mientras Causal-SCM sigue usando su
    SCM LINEAL ajustado en el fold de entrenamiento. Mismos folds, clasificadores y semillas que el analisis
    principal (los contrafactuales generados son identicos; solo cambia el mundo de implementacion).
(B) R1.2 - Fidelidad de DiCE: libreria oficial dice-ml 0.12 (metodo random, 4 CFs) vs. la reimplementacion
    DiCE-random, caso Muestra, clasificador RF, 5x10 folds, mismo evaluador y mundo lineal.

Unidad estadistica: perfil / vaca / granja. Pruebas: Wilcoxon pareado + Holm (por familia), Cliff's delta,
IC 95% bootstrap, TOST (margen 0.05) para la fidelidad.

Salidas: results/robust_forest_<caso>.parquet, results/dice_official_milk_rf.parquet,
         tables/table_16_world_model_robustness, table_17_world_model_tests, table_18_dice_fidelity,
         figures/fig_13_world_model_robustness
"""
import json
import logging
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import utils as U
from cf_lib import SCM, SPECS, CFEvaluator, make_classifier, run_cf_fold

warnings.filterwarnings("ignore")
PROC = U.PROJECT_DIR / "data" / "processed"
METHODS = ["NUN", "Wachter-grid", "DiCE-random", "Actionable-indep", "Causal-SCM"]
CLFS = ["LR", "MLP", "RF", "XGBoost", "LightGBM"]
CASES = {"milk_quality": ("Sample", U.DATA_PATH, "grade", None, "splits_cf.json", 0),
         "heat_stress": ("Cow", PROC / "heat_stress_ml_ready.csv", "scc_high", "cow_id", "ext_heat_stress_splits_cf.json", 1),
         "scc_management": ("Farm", PROC / "scc_management_ml_ready.csv", "scc_high", "farm_id", "ext_scc_management_splits_cf.json", 2)}
COL = dict(zip(METHODS, ["#56B4E9", "#E69F00", "#CC79A7", "#009E73", "#0072B2"]))


def seed_of(di, ci, r, f):
    base = U.SEED * 1000 + ci * 100 + r * 10 + f
    return base if di == 0 else base + di * 10000


def annotate(cf):
    key = ["classifier", "repeat", "fold", "idx"]
    cf["false_promise"] = ((cf.nominal_valid == 1) & (cf.impl_validity < 0.5)).astype(float)
    feas = cf[cf.method == "Causal-SCM"][key + ["feasible"]].rename(columns={"feasible": "causal_feasible"})
    return cf.merge(feas, on=key)


# =============================================================================
# (A) Mundo forest
# =============================================================================
def run_forest(name):
    label, path, target, unit_col, split_file, di = CASES[name]
    spec = SPECS[name]
    d = pd.read_csv(path)
    X, y = d[spec.features].to_numpy(float), d[target].to_numpy()
    world = SCM(spec.dag, spec, kind="forest").fit(X)
    splits = json.load(open(U.RESULTS_DIR / split_file))
    out = Parallel(n_jobs=-2)(
        delayed(run_cf_fold)(c, X, y, np.array(sp["train"]), np.array(sp["test"]), world,
                             seed_of(di, ci, sp["repeat"], sp["fold"]), tau=U.CF_TAU, m_samples=U.CF_M_SAMPLES,
                             m_eval=U.EVAL_M_SAMPLES, spec=spec, extra={"repeat": sp["repeat"], "fold": sp["fold"]})
        for ci, c in enumerate(CLFS) for sp in splits)
    cf = pd.DataFrame([r for rows in out for r in rows])
    cf["unit"] = cf["idx"] if unit_col is None else d[unit_col].to_numpy()[cf.idx.values]
    cf = annotate(cf)
    cf.to_parquet(U.RESULTS_DIR / f"robust_forest_{name}.parquet", index=False)
    return cf


def unit_summary(cf, name, world):
    rows = []
    for m in METHODS:
        dm = cf[cf.method == m]
        iv = dm.groupby("unit").impl_validity.mean()
        fp = dm.groupby("unit").false_promise.mean()
        ivf = dm[dm.causal_feasible == 1].groupby("unit").impl_validity.mean()
        lo, hi = U.bootstrap_ci(iv.values)
        rows.append({"dataset": name, "world": world, "method": m, "impl_validity": iv.mean(), "impl_ci_lo": lo,
                     "impl_ci_hi": hi, "false_promise": fp.mean(), "impl_validity_causal_feasible": ivf.mean()})
    return rows


def tests(cf, name):
    out = []
    for metric, hib, subset in (("impl_validity", True, "all"), ("false_promise", False, "all"),
                                ("impl_validity", True, "causal-feasible")):
        d = cf if subset == "all" else cf[cf.causal_feasible == 1]
        B = d.pivot_table(index="unit", columns="method", values=metric, aggfunc="mean").dropna()
        fam = []
        for m in METHODS[:-1]:
            x, z = B["Causal-SCM"].values, B[m].values
            W, p = U.wilcoxon_safe(x, z)
            lo, hi = U.bootstrap_ci(x - z)
            dl = U.cliffs_delta(x, z)
            fam.append({"dataset": name, "family": f"{metric} ({subset})", "comparison": f"Causal-SCM vs {m}",
                        "n": len(B), "p": p, "cliffs_delta": dl, "magnitude": U.cliffs_magnitude(dl),
                        "diff_mean": float((x - z).mean()), "diff_ci_lo": lo, "diff_ci_hi": hi,
                        "favours_proposed": bool(((x - z).mean() > 0) == hib)})
        fam = pd.DataFrame(fam)
        fam["p_holm"] = U.holm(fam.p.values)
        out += fam.to_dict("records")
    return out


# =============================================================================
# (B) DiCE oficial
# =============================================================================
def dice_fold(sp, seed):
    import dice_ml
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import FunctionTransformer
    logging.disable(logging.WARNING)
    spec = SPECS["milk_quality"]
    d = pd.read_csv(U.DATA_PATH)
    X, y = d[spec.features].to_numpy(float), d["grade"].to_numpy()
    tr, te = np.array(sp["train"]), np.array(sp["test"])
    clf = make_classifier("RF"); clf.set_params(n_jobs=1)
    model = make_pipeline(FunctionTransformer(lambda Z: np.asarray(Z, float)), clf).fit(X[tr], y[tr])
    cat = sorted(spec.binary)
    dtr = d.iloc[tr][spec.features].copy(); dtr["grade"] = y[tr]
    for c in cat:
        dtr[c] = dtr[c].astype(int).astype(str)
    D = dice_ml.Data(dataframe=dtr, continuous_features=[f for f in spec.features if f not in spec.binary],
                     outcome_name="grade")
    exp = dice_ml.Dice(D, dice_ml.Model(model=model, backend="sklearn"), method="random")
    world = SCM(spec.dag, spec).fit(X)
    rng = np.random.default_rng(seed)
    ev = CFEvaluator(model, X[tr], world, rng, target=2, m_eval=U.EVAL_M_SAMPLES, spec=spec)
    rows = []
    for i, p in zip(te, model.predict(X[te])):
        if p == 2:
            continue
        q = d.iloc[[i]][spec.features].copy()
        for c in cat:
            q[c] = q[c].astype(int).astype(str)
        base = {"classifier": "RF", "repeat": sp["repeat"], "fold": sp["fold"], "idx": int(i), "method": "DiCE-official"}
        try:
            r = exp.generate_counterfactuals(q, total_CFs=4, desired_class=2, random_seed=int(seed % 2**31 + i),
                                             verbose=False)
            cfs = r.cf_examples_list[0].final_cfs_df
        except Exception:
            cfs = None
        if cfs is None or len(cfs) == 0:
            rows.append({**base, "feasible": 0.0, "impl_validity": 0.0, "nominal_valid": 0.0})
            continue
        C = cfs[spec.features].astype(float).to_numpy()
        evs = ev.evaluate_many(X[i], list(C))
        agg = {k: float(np.mean([e[k] for e in evs])) for k in evs[0]}
        rows.append({**base, "feasible": 1.0, **agg})
    return rows


if __name__ == "__main__":
    t0 = time.time()
    summ, tst, CF = [], [], {}
    for name in CASES:
        cf = run_forest(name)
        CF[name] = cf
        summ += unit_summary(cf, name, "forest")
        tst += tests(cf, name)
        print(f"[{name}] forest world OK ({time.time() - t0:.0f}s)")
    lin = pd.read_csv(U.RESULTS_DIR / "cross_dataset_cf_summary.csv")
    lin = lin.assign(world="linear")[["dataset", "world", "method", "impl_validity", "impl_ci_lo", "impl_ci_hi",
                                      "false_promise", "impl_validity_causal_feasible"]]
    S = pd.concat([lin, pd.DataFrame(summ)], ignore_index=True)
    S.to_csv(U.RESULTS_DIR / "world_model_robustness_summary.csv", index=False)

    # ---- Tabla 16 ----
    W = S.pivot_table(index=["dataset", "method"], columns="world",
                      values=["impl_validity", "false_promise", "impl_validity_causal_feasible"]).reset_index()
    W.columns = ["_".join([c for c in col if c]) for col in W.columns]
    order = {n: i for i, n in enumerate(CASES)}
    W["o1"] = W.dataset.map(order); W["o2"] = W.method.map({m: i for i, m in enumerate(METHODS)})
    W = W.sort_values(["o1", "o2"]).drop(columns=["o1", "o2"]).reset_index(drop=True)
    T16t = pd.DataFrame({"Case": W.dataset.map({k: v[0] for k, v in CASES.items()}), "Method": W.method})
    spec_cols = [("impl_validity_linear", "Delivered (linear world)", True), ("impl_validity_forest", "Delivered (forest world)", True),
                 ("impl_validity_causal_feasible_linear", "Delivered | recourse (linear)", True),
                 ("impl_validity_causal_feasible_forest", "Delivered | recourse (forest)", True),
                 ("false_promise_linear", "False promise (linear)", False), ("false_promise_forest", "False promise (forest)", False)]
    for c, lab, hib in spec_cols:
        T16t[U.arrow(lab, hib)] = W[c].map(lambda v: f"{v:.3f}")
        U.highlight(T16t, W[c], U.arrow(lab, hib), hib, groups=W.dataset)
    U.save_table(W, "table_16_world_model_robustness", df_tex=T16t)

    # ---- Tabla 17 ----
    T17 = pd.DataFrame(tst)
    T17t = pd.DataFrame({"Case": T17.dataset.map({k: v[0] for k, v in CASES.items()}), "Family": T17.family,
                         "Comparison": T17.comparison, "n": T17.n,
                         "Diff. [95% CI]": [f"{a:+.3f} {U.fmt_ci(l, h)}" for a, l, h in zip(T17.diff_mean, T17.diff_ci_lo, T17.diff_ci_hi)],
                         "Cliff's $\\delta$": T17.cliffs_delta.map(lambda v: f"{v:+.3f}"),
                         "p (Holm)": T17.p_holm.map(U.fmt_p),
                         "Interpretation": ["n.s." if p >= U.ALPHA else ("favours Causal-SCM" if f else "favours baseline")
                                            for p, f in zip(T17.p_holm, T17.favours_proposed)]})
    U.save_table(T17, "table_17_world_model_tests", df_tex=T17t)
    print(T17[["dataset", "family", "comparison", "n", "diff_mean", "diff_ci_lo", "diff_ci_hi", "cliffs_delta", "p_holm"]].round(4).to_string())

    # ---- (B) DiCE oficial ----
    splits = json.load(open(U.RESULTS_DIR / "splits_cf.json"))
    ci = CLFS.index("RF")
    res = Parallel(n_jobs=-2)(delayed(dice_fold)(sp, seed_of(0, ci, sp["repeat"], sp["fold"])) for sp in splits)
    off = pd.DataFrame([r for rows in res for r in rows])
    off.to_parquet(U.RESULTS_DIR / "dice_official_milk_rf.parquet", index=False)
    ours = pd.read_parquet(U.RESULTS_DIR / "cf_instances.parquet")
    ours = ours[(ours.classifier == "RF") & (ours.method == "DiCE-random")]
    both = pd.concat([ours.assign(method="DiCE-random (ours)"), off], ignore_index=True)
    both["false_promise"] = ((both.nominal_valid == 1) & (both.impl_validity < 0.5)).astype(float)
    mets = [("feasible", None), ("nominal_valid", None), ("impl_validity", True), ("false_promise", False),
            ("actionability_violation", False), ("sparsity", None), ("distance_l1", None), ("lof_outlier", False)]
    U_ = {m: both[both.method == m].groupby("idx")[[k for k, _ in mets]].mean() for m in both.method.unique()}
    common = U_["DiCE-random (ours)"].index.intersection(U_["DiCE-official"].index)
    rows = []
    for k, _ in mets:
        a, b = U_["DiCE-random (ours)"].loc[common, k].values, U_["DiCE-official"].loc[common, k].values
        _, p = U.wilcoxon_safe(a, b)
        pt, _, _, (lo90, hi90) = U.tost_wilcoxon(a, b, margin=0.05) if k in ("impl_validity", "false_promise") else (np.nan, 0, 0, (np.nan, np.nan))
        rows.append({"metric": k, "n_profiles": len(common), "ours_mean": a.mean(), "official_mean": b.mean(),
                     "diff_mean": (a - b).mean(), "wilcoxon_p": p, "tost_p_margin_0.05": pt, "diff_ci90_lo": lo90, "diff_ci90_hi": hi90})
    T18 = pd.DataFrame(rows)
    T18t = pd.DataFrame({"Metric": T18.metric, "n": T18.n_profiles, "DiCE-random (ours)": T18.ours_mean.map(lambda v: f"{v:.3f}"),
                         "dice-ml 0.12 (official)": T18.official_mean.map(lambda v: f"{v:.3f}"),
                         "Diff.": T18.diff_mean.map(lambda v: f"{v:+.3f}"), "Wilcoxon p": T18.wilcoxon_p.map(U.fmt_p),
                         "TOST p ($\\pm$0.05)": T18["tost_p_margin_0.05"].map(U.fmt_p),
                         "90% CI diff.": [U.fmt_ci(l, h) if pd.notna(l) else "--" for l, h in zip(T18.diff_ci90_lo, T18.diff_ci90_hi)]})
    U.save_table(T18, "table_18_dice_fidelity", df_tex=T18t)
    print(T18.round(4).to_string())

    # ---- Figura 13 ----
    fig, axes = plt.subplots(2, 3, figsize=(U.DOUBLE_COL, 4.8), sharey="row")
    for j, name in enumerate(CASES):
        for r, (col, ylab) in enumerate((("false_promise", "False-promise rate"),
                                         ("impl_validity_causal_feasible", "Delivered | recourse exists"))):
            ax = axes[r, j]
            for i, m in enumerate(METHODS):
                s = S[(S.dataset == name) & (S.method == m)].set_index("world")[col]
                ax.plot([i - 0.15, i + 0.15], [s["linear"], s["forest"]], color="#888888", lw=0.9, zorder=1)
                ax.scatter(i - 0.15, s["linear"], marker="o", facecolor="white", edgecolor=COL[m], s=30, lw=1.3, zorder=2,
                           label="Linear world" if i == 0 else None)
                ax.scatter(i + 0.15, s["forest"], marker="s", color=COL[m], s=30, zorder=3,
                           label="Forest world" if i == 0 else None)
            ax.set_xticks(range(len(METHODS)), ["NUN", "Wach.", "DiCE", "Act.", "Causal"], fontsize=8)
            ax.set_ylim(-0.04, 1.06)
            if j == 0:
                ax.set_ylabel(ylab)
            if r == 0:
                ax.text(0.5, 1.03, CASES[name][0], transform=ax.transAxes, ha="center", fontsize=9, style="italic")
            U.add_panel_label(ax, "abcdef"[r * 3 + j], x=-0.2 if j else -0.38)
    axes[0, 0].legend(fontsize=8, loc="center left")
    fig.tight_layout()
    U.save_figure(fig, "fig_13_world_model_robustness")
    print(f"08 OK ({time.time() - t0:.0f}s)")
