"""03_ablation_and_sensitivity.py
(a) Ablacion de grupos de features del mejor modelo ML (LightGBM) con los mismos folds (5x20).
(b) Sensibilidad del recurso causal al umbral tau y al modo de abduccion (posterior vs prior).
(c) Sensibilidad al DAG (idea C2): DAG del metodo x DAG del mundo.
(d) Chequeo del DAG experto: implicaciones testeables (Markov local, LR tests + Holm) y PC (causal-learn).

Salidas: tables/table_03_ablation_study, table_07_dag_sensitivity, table_08_dag_testable_implications,
         results/ablation_scores.csv, results/tau_sensitivity.csv, results/dag_sensitivity.csv, results/pc_dag.json,
         figures/fig_03_ablation_and_sensitivity, figures/fig_07_causal_dag
"""
import json
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

import utils as U
from cf_lib import (BINARY, DAG_EXPERT, FEATURES, IDX, SCM, TOPO_ORDER, _parent_design, dag_edges,
                    dag_variants, make_classifier, run_cf_fold)

warnings.filterwarnings("ignore")
t0 = time.time()
uniq = pd.read_csv(U.DATA_PATH)
X, y = uniq[FEATURES].to_numpy(float), uniq["grade"].to_numpy()
splits_u = json.load(open(U.RESULTS_DIR / "splits_unique.json"))
splits_cf = json.load(open(U.RESULTS_DIR / "splits_cf.json"))
ABL_MODEL = "LightGBM"

# =============================================================================
# (a) Ablacion de features
# =============================================================================
DERIVED = ["ph_deviation", "ph_out_of_range", "temp_out_of_range", "sensory_score", "physical_score", "optimal_count"]
GROUPS = {
    "Full (7 base)": FEATURES,
    "– pH": [f for f in FEATURES if f != "ph"],
    "– temperature": [f for f in FEATURES if f != "temperature"],
    "– sensory (taste, odor)": [f for f in FEATURES if f not in ("taste", "odor")],
    "– physical (fat, turbidity)": [f for f in FEATURES if f not in ("fat", "turbidity")],
    "– colour": [f for f in FEATURES if f != "colour"],
    "+ domain features": FEATURES + DERIVED,
}


def abl_one(gname, cols, sp):
    Xg = uniq[cols].to_numpy(float)
    tr, te = np.array(sp["train"]), np.array(sp["test"])
    m = make_classifier(ABL_MODEL).fit(Xg[tr], y[tr])
    return {"variant": gname, "repeat": sp["repeat"], "fold": sp["fold"], "n_train": len(tr), "n_test": len(te),
            "balanced_accuracy": balanced_accuracy_score(y[te], m.predict(Xg[te]))}


abl = pd.DataFrame(Parallel(n_jobs=-2)(delayed(abl_one)(g, c, sp) for g, c in GROUPS.items() for sp in splits_u))
abl.to_csv(U.RESULTS_DIR / "ablation_scores.csv", index=False)
full = abl[abl.variant == "Full (7 base)"].sort_values(["repeat", "fold"]).balanced_accuracy.values
abl_rows = []
for g in GROUPS:
    v = abl[abl.variant == g].sort_values(["repeat", "fold"]).balanced_accuracy.values
    d = v - full
    lo, hi = U.bootstrap_ci(d)
    t, p_nb = U.nadeau_bengio_ttest(d, n_train=66, n_test=17) if g != "Full (7 base)" else (np.nan, np.nan)
    abl_rows.append({"block": "Feature ablation (LightGBM, BA)", "variant": g, "mean": v.mean(), "sd": v.std(ddof=1),
                     "delta": d.mean(), "delta_ci_lo": lo, "delta_ci_hi": hi, "p_nadeau_bengio": p_nb})

# =============================================================================
# (b) Sensibilidad tau y abduccion del recurso causal (5 clasificadores, 5x10)
# =============================================================================
world = SCM(DAG_EXPERT).fit(X)
CLFS = ["LR", "MLP", "RF", "XGBoost", "LightGBM"]
TAUS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
jobs = []
for ci, c in enumerate(CLFS):
    for sp in splits_cf:
        seed = U.SEED * 1000 + ci * 100 + sp["repeat"] * 10 + sp["fold"]
        for tau in TAUS:
            jobs.append(dict(c=c, sp=sp, seed=seed, tau=tau, abd="posterior"))
        jobs.append(dict(c=c, sp=sp, seed=seed, tau=U.CF_TAU, abd="prior"))
res = Parallel(n_jobs=-2)(
    delayed(run_cf_fold)(j["c"], X, y, np.array(j["sp"]["train"]), np.array(j["sp"]["test"]), world, j["seed"],
                         methods=["Causal-SCM"], tau=j["tau"], abduction=j["abd"], m_samples=U.CF_M_SAMPLES,
                         m_eval=U.EVAL_M_SAMPLES,
                         extra={"repeat": j["sp"]["repeat"], "fold": j["sp"]["fold"], "tau": j["tau"], "abduction": j["abd"]})
    for j in jobs)
tau_df = pd.DataFrame([r for rows in res for r in rows])
tau_df["false_promise"] = ((tau_df.nominal_valid == 1) & (tau_df.impl_validity < 0.5)).astype(float)
tau_df.to_csv(U.RESULTS_DIR / "tau_sensitivity.csv", index=False)


def inst_mean(df, col):
    return df.groupby(["classifier", "idx"])[col].mean()


for (tau, abd), g in tau_df.groupby(["tau", "abduction"]):
    if abd == "prior" or True:
        v = inst_mean(g, "impl_validity")
        lo, hi = U.bootstrap_ci(v.values)
        feas = g.feasible.mean()
        cond = g.loc[g.feasible == 1, "impl_validity"].mean()
        abl_rows.append({"block": "Causal recourse (impl. validity)",
                         "variant": f"tau = {tau:.2f}, abduction = {abd}", "mean": v.mean(), "sd": v.std(ddof=1),
                         "delta": np.nan, "delta_ci_lo": lo, "delta_ci_hi": hi, "p_nadeau_bengio": np.nan,
                         "feasibility": feas, "validity_if_feasible": cond,
                         "false_promise": g.false_promise.mean(),
                         "action_cost": g.action_cost.mean()})
T3 = pd.DataFrame(abl_rows)
T3t = pd.DataFrame({
    "Block": ["Feature ablation" if b.startswith("Feature") else "Causal recourse" for b in T3.block],
    "Variant": T3.variant,
    "Score": [U.fmt_mean_sd([m, m]) if False else f"{m:.3f} $\\pm$ {s:.3f}" for m, s in zip(T3["mean"], T3["sd"])],
    "Delta / 95% CI": [f"{d:+.3f} {U.fmt_ci(l, h)}" if not np.isnan(d) else U.fmt_ci(l, h)
                       for d, l, h in zip(T3.delta, T3.delta_ci_lo, T3.delta_ci_hi)],
    "p (NB)": [U.fmt_p(p) for p in T3.p_nadeau_bengio],
    "Feasible": [f"{v:.3f}" if not pd.isna(v) else "--" for v in T3.get("feasibility", pd.Series([np.nan] * len(T3)))],
    "Valid | feasible": [f"{v:.3f}" if not pd.isna(v) else "--" for v in T3.get("validity_if_feasible", pd.Series([np.nan] * len(T3)))],
    "False promise": [f"{v:.3f}" if not pd.isna(v) else "--" for v in T3.get("false_promise", pd.Series([np.nan] * len(T3)))],
})
U.save_table(T3, "table_03_ablation_study", df_tex=T3t)

# =============================================================================
# (d) Implicaciones testeables del DAG experto + PC
# =============================================================================
def descendants(dag, node):
    ch = {c for c, ps in dag.items() if node in ps}
    out = set(ch)
    for c in ch:
        out |= descendants(dag, c)
    return out


def lr_test(node, parents, z):
    yv = X[:, IDX[node]]
    P = {p: X[:, IDX[p]] for p in FEATURES}
    Z0 = _parent_design(parents, P) if parents else np.zeros((len(yv), 0))
    Z1 = np.column_stack([Z0, _parent_design([z], P)])
    df_ = Z1.shape[1] - Z0.shape[1]
    if node in BINARY:
        def ll(Z):
            if Z.shape[1] == 0:
                p = yv.mean(); return np.sum(yv * np.log(p) + (1 - yv) * np.log(1 - p))
            m = LogisticRegression(C=1e6, max_iter=10000).fit(StandardScaler().fit_transform(Z), yv)
            pr = np.clip(m.predict_proba(StandardScaler().fit_transform(Z))[:, 1], 1e-12, 1 - 1e-12)
            return np.sum(yv * np.log(pr) + (1 - yv) * np.log(1 - pr))
    else:
        def ll(Z):
            n = len(yv)
            r = yv - (LinearRegression().fit(Z, yv).predict(Z) if Z.shape[1] else yv.mean())
            s2 = np.mean(r ** 2)
            return -0.5 * n * (np.log(2 * np.pi * s2) + 1)
    stat = 2 * (ll(Z1) - ll(Z0))
    return max(stat, 0.0), df_, float(stats.chi2.sf(max(stat, 0.0), df_))


ci_rows = []
full_dag = {**{n: [] for n in ("temperature", "fat")}, **DAG_EXPERT}
for node in TOPO_ORDER:
    pa = full_dag[node]
    nd = [v for v in TOPO_ORDER if v != node and v not in descendants(full_dag, node) and v not in pa]
    for z in nd:
        if TOPO_ORDER.index(z) > TOPO_ORDER.index(node) and z in ("temperature", "fat"):
            continue
        if TOPO_ORDER.index(z) > TOPO_ORDER.index(node):
            continue  # cada par se testea una vez (local Markov en orden topologico)
        stat, df_, p = lr_test(node, pa, z)
        ci_rows.append({"implication": f"{node} ⊥ {z} | {{{', '.join(pa) or '∅'}}}", "node": node, "z": z,
                        "LR_chi2": stat, "df": df_, "p": p})
T8 = pd.DataFrame(ci_rows)
T8["p_holm"] = U.holm(T8.p.values)
T8["rejected_holm"] = T8.p_holm < U.ALPHA
T8t = pd.DataFrame({"Implication": T8.implication.str.replace("⊥", "$\\perp$").str.replace("∅", "$\\emptyset$")
                    .str.replace("{", "\\{", regex=False).str.replace("}", "\\}", regex=False),
                    "LR $\\chi^2$": T8.LR_chi2.map(lambda v: f"{v:.2f}"), "df": T8["df"],
                    "p": T8.p.map(U.fmt_p), "p (Holm)": T8.p_holm.map(U.fmt_p),
                    "Rejected": T8.rejected_holm.map({True: "yes", False: "no"})})
U.save_table(T8, "table_08_dag_testable_implications", df_tex=T8t)

# PC (causal-learn), Fisher-z, alpha = 0.05; aristas hacia temperature/fat prohibidas
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.graph.GraphNode import GraphNode

Xs = StandardScaler().fit_transform(X)
nodes = [GraphNode(f"X{i + 1}") for i in range(len(FEATURES))]
bk = BackgroundKnowledge()
for r in ("temperature", "fat"):
    for f in FEATURES:
        if f != r:
            bk.add_forbidden_by_node(nodes[IDX[f]], nodes[IDX[r]])
cg = pc(Xs, alpha=0.05, indep_test="fisherz", stable=True, background_knowledge=bk, show_progress=False, node_names=None)
G = cg.G.graph
skeleton = set()
for i in range(len(FEATURES)):
    for j in range(i + 1, len(FEATURES)):
        if G[i, j] != 0 or G[j, i] != 0:
            a, b = FEATURES[i], FEATURES[j]
            # orientacion por orden de dominio (skeleton de PC + orden causal temporal)
            a, b = sorted([a, b], key=TOPO_ORDER.index)
            skeleton.add((a, b))
pc_dag = {n: [] for n in DAG_EXPERT}
for a, b in skeleton:
    if b in pc_dag:
        pc_dag[b].append(a)
json.dump({"skeleton": sorted(skeleton), "pc_dag": pc_dag}, open(U.RESULTS_DIR / "pc_dag.json", "w", encoding="utf-8"), indent=2)
exp_skel = {tuple(sorted(e, key=TOPO_ORDER.index)) for e in dag_edges(DAG_EXPERT)}
shd = len(exp_skel ^ skeleton)

# =============================================================================
# (c) Sensibilidad al DAG: DAG del metodo x DAG del mundo (LightGBM y RF, 5x10)
# =============================================================================
VAR = dag_variants(pc_dag)
worlds = {k: SCM(v).fit(X) for k, v in VAR.items()}
jobs = []
for ci, c in enumerate(["RF", "LightGBM"]):
    for sp in splits_cf:
        seed = U.SEED * 1000 + ci * 100 + sp["repeat"] * 10 + sp["fold"]
        for md in VAR:
            for wd in VAR:
                jobs.append((c, sp, seed, md, wd))
res = Parallel(n_jobs=-2)(
    delayed(run_cf_fold)(c, X, y, np.array(sp["train"]), np.array(sp["test"]), worlds[wd], seed,
                         methods=["Causal-SCM"], dag=VAR[md], tau=U.CF_TAU, m_samples=U.CF_M_SAMPLES,
                         m_eval=U.EVAL_M_SAMPLES, extra={"repeat": sp["repeat"], "fold": sp["fold"],
                                                         "method_dag": md, "world_dag": wd})
    for c, sp, seed, md, wd in jobs)
dag_df = pd.DataFrame([r for rows in res for r in rows])
dag_df["false_promise"] = ((dag_df.nominal_valid == 1) & (dag_df.impl_validity < 0.5)).astype(float)
dag_df.to_csv(U.RESULTS_DIR / "dag_sensitivity.csv", index=False)
T7 = dag_df.groupby(["method_dag", "world_dag"]).agg(impl_validity=("impl_validity", "mean"),
                                                   feasibility=("feasible", "mean"),
                                                   false_promise=("false_promise", "mean")).reset_index()
cond = dag_df[dag_df.feasible == 1].groupby(["method_dag", "world_dag"]).impl_validity.mean().rename("validity_if_feasible")
T7 = T7.merge(cond.reset_index(), on=["method_dag", "world_dag"], how="left")
T7["shd_method_vs_world"] = [len({tuple(sorted(e, key=TOPO_ORDER.index)) for e in dag_edges(VAR[a])} ^
                                 {tuple(sorted(e, key=TOPO_ORDER.index)) for e in dag_edges(VAR[b])})
                             for a, b in zip(T7.method_dag, T7.world_dag)]
T7t = T7.copy()
for c in ("impl_validity", "feasibility", "false_promise", "validity_if_feasible"):
    T7t[c] = T7t[c].map(lambda v: f"{v:.3f}")
T7t.columns = ["Method DAG", "World DAG", "Impl. validity", "Feasible", "False promise", "Valid | feasible", "SHD"]
U.save_table(T7, "table_07_dag_sensitivity", df_tex=T7t)

# =============================================================================
# Figura 3
# =============================================================================
fig = plt.figure(figsize=(U.DOUBLE_COL, 5.4))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.55, wspace=0.42)
ax = fig.add_subplot(gs[0, 0])
fa = T3[T3.block.str.startswith("Feature") & (T3.variant != "Full (7 base)")]
yy = np.arange(len(fa))
ax.errorbar(fa.delta, yy, xerr=[fa.delta - fa.delta_ci_lo, fa.delta_ci_hi - fa.delta], fmt="o", color="#0072B2",
            ms=4, capsize=2, lw=0.9)
ax.axvline(0, color="k", lw=0.7, ls="--")
ax.set_yticks(yy, fa.variant); ax.invert_yaxis(); ax.set_xlabel("ΔBA vs. full model (95% CI)")
U.add_panel_label(ax, "a", x=-0.62)

ax = fig.add_subplot(gs[0, 1])
tp = tau_df[tau_df.abduction == "posterior"].groupby("tau").agg(
    feas=("feasible", "mean"), impl=("impl_validity", "mean"), fp=("false_promise", "mean"))
tp["cond"] = tau_df[(tau_df.abduction == "posterior") & (tau_df.feasible == 1)].groupby("tau").impl_validity.mean()
ax.plot(tp.index, tp.cond, "-o", color="#009E73", ms=4, label="Validity | feasible")
ax.plot(tp.index, tp.feas, "--s", color="#0072B2", ms=4, label="Feasibility")
ax.plot(tp.index, tp.impl, "-.^", color="#E69F00", ms=4, label="Impl. validity (all)")
ax.plot(tp.index, tp.fp, ":D", color="#D55E00", ms=4, label="False-promise rate")
ax.set_xlabel("Validity threshold τ"); ax.set_ylabel("Rate"); ax.set_ylim(-0.02, 1.02)
ax.legend(fontsize=8, loc="center right", bbox_to_anchor=(1.0, 0.64))
U.add_panel_label(ax, "b")

ax = fig.add_subplot(gs[1, :])
order = list(VAR)
H = T7.pivot(index="method_dag", columns="world_dag", values="validity_if_feasible").reindex(index=order, columns=order)
F = T7.pivot(index="method_dag", columns="world_dag", values="feasibility").reindex(index=order, columns=order)
im = ax.imshow(H.values, cmap="cividis", vmin=0, vmax=1, aspect="auto")
for i in range(len(order)):
    for j in range(len(order)):
        v = H.values[i, j]
        ax.text(j, i, f"{v:.2f}\n({F.values[i, j]:.2f})", ha="center", va="center", fontsize=8,
                color="white" if v < 0.6 else "black")
ax.set_xticks(range(len(order)), order); ax.set_yticks(range(len(order)), order)
ax.set_xlabel("World (implementation) DAG"); ax.set_ylabel("Method DAG"); ax.grid(False)
cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01); cb.set_label("Validity | feasible")
U.add_panel_label(ax, "c", x=-0.14)
U.save_figure(fig, "fig_03_ablation_and_sensitivity", despine=False)

# =============================================================================
# Figura 7: DAG experto + skeleton PC
# =============================================================================
pos = {"temperature": (0, 1.2), "fat": (0, -1.2), "ph": (1.1, 0.2), "turbidity": (1.4, -1.0),
       "odor": (2.6, 1.6), "taste": (2.6, 0.2), "colour": (2.6, -1.9)}
short = {"temperature": "Temp.", "fat": "Fat", "ph": "pH", "turbidity": "Turb.", "odor": "Odor",
         "taste": "Taste", "colour": "Colour"}
fig, axes = plt.subplots(1, 2, figsize=(U.DOUBLE_COL, 2.7))
for k, (ax, edges, lab) in enumerate(zip(axes, [dag_edges(DAG_EXPERT), {(a, b) for a, b in skeleton}],
                                          ["a", "b"])):
    for n, (px, py) in pos.items():
        act = n in ("temperature", "fat")
        ax.scatter(px, py, s=750, color="#E69F00" if act else "#56B4E9", edgecolor="k", zorder=3, lw=0.8)
        ax.text(px, py, short[n], ha="center", va="center", fontsize=8, zorder=4)
    for a, b in edges:
        (x0, y0), (x1, y1) = pos[a], pos[b]
        in_exp = (a, b) in dag_edges(DAG_EXPERT)
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color="k" if in_exp else "#D55E00", lw=1.0,
                                    ls="-" if in_exp else "--", shrinkA=13, shrinkB=13))
    ax.set_xlim(-0.5, 3.1); ax.set_ylim(-2.35, 2.05); ax.axis("off")
    U.add_panel_label(ax, lab, x=0.0, y=0.98)
U.save_figure(fig, "fig_07_causal_dag", despine=False)

print(T3[["block", "variant", "mean", "delta", "delta_ci_lo", "delta_ci_hi", "p_nadeau_bengio"]].round(4).to_string())
print(T3[["variant", "feasibility", "validity_if_feasible", "false_promise"]].dropna().round(3).to_string())
print(T8.round(4).to_string())
print("PC skeleton:", sorted(skeleton), "SHD vs expert:", shd)
print(T7.round(3).to_string())
print(f"03 OK in {time.time() - t0:.1f}s")
