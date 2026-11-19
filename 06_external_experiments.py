"""06_external_experiments.py
Validacion externa en dos datasets reales (Mendeley, CC BY 4.0):

  heat_stress     750 muestras / 50 vacas (medidas repetidas) -> CV agrupada por vaca (StratifiedGroupKFold).
                  Target: CCS > 200k (1). Recurso deseado: CCS <= 200k (0).
                  Accionables: THI (solo bajar: enfriamiento) y DMI (solo subir: alimentacion). Inmutable: %HF.
  scc_management  90 granjas -> RepeatedStratifiedKFold. Target: CCS > 200k (1). Recurso deseado: 0.
                  Accionables: 6 practicas de higiene (solo adoptar). Estructurales (sistema, ordeño
                  mecanizado, sala, linea) e its sin ordeño mecanico: no accionables/no factibles.

Bloques por dataset:
  (1) Prediccion: 9 modelos x 5x20; heat_stress ademas auditoria de fuga (KFold por fila vs agrupado por vaca).
  (2) Contrafactuales: 5 metodos x 5 clasificadores (heat 5x5 grupos; scc 5x10).
  (3) Sensibilidad del recurso causal: tau y abduccion prior.
  (4) DAG: implicaciones testeables (Markov local, LR + Holm), PC (causal-learn) y matriz DAG-metodo x DAG-mundo.

Salidas: results/ext_<dataset>_*.{csv,parquet,json}
"""
import json
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, brier_score_loss, f1_score, log_loss, matthews_corrcoef,
                             roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

import utils as U
from cf_lib import SCM, SPECS, _parent_design, dag_edges, make_classifier, run_cf_fold

warnings.filterwarnings("ignore")
PROC = U.PROJECT_DIR / "data" / "processed"
MODELS = ["Dummy", "LR", "kNN", "SVM-RBF", "DT", "RF", "XGBoost", "LightGBM", "MLP"]
CF_CLFS = ["LR", "MLP", "RF", "XGBoost", "LightGBM"]
TAUS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
CONFIG = {
    "heat_stress": {"file": "heat_stress_ml_ready.csv", "group": "cow_id", "cf_repeats": 5,
                    "extra_dags": {"No THI→DMI": {**SPECS["heat_stress"].dag, "dmi_kg": ["hf_pct"]}}},
    "scc_management": {"file": "scc_management_ml_ready.csv", "group": None, "cf_repeats": 10,
                       "extra_dags": {"No structure→routine": {**SPECS["scc_management"].dag,
                                                               "cmt": [], "ppd": ["cmt"]}}},
}


def load(name):
    spec, cfg = SPECS[name], CONFIG[name]
    d = pd.read_csv(PROC / cfg["file"])
    X = d[spec.features].to_numpy(float)
    y = d["scc_high"].to_numpy()
    g = d[cfg["group"]].to_numpy() if cfg["group"] else None
    return d, X, y, g


def make_splits(y, g, repeats, grouped=True):
    out = []
    for r in range(repeats):
        if g is not None and grouped:
            cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=U.SEED + r)
            it = cv.split(np.zeros(len(y)), y, g)
        else:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=U.SEED + r)
            it = cv.split(np.zeros(len(y)), y)
        for k, (tr, te) in enumerate(it):
            if g is not None and grouped:
                assert not set(g[tr]) & set(g[te])
            out.append({"repeat": r, "fold": k, "train": tr.tolist(), "test": te.tolist()})
    return out


def pred_one(model, X, y, sp, protocol):
    tr, te = np.array(sp["train"]), np.array(sp["test"])
    clf = make_classifier(model)
    if model == "RF":
        clf.set_params(n_jobs=1)
    clf.fit(X[tr], y[tr])
    yp = clf.predict(X[te])
    p1 = np.clip(clf.predict_proba(X[te])[:, 1], 1e-6, 1 - 1e-6)
    yt = y[te]
    return {"model": model, "protocol": protocol, "repeat": sp["repeat"], "fold": sp["fold"],
            "n_train": len(tr), "n_test": len(te),
            "balanced_accuracy": balanced_accuracy_score(yt, yp), "roc_auc": roc_auc_score(yt, p1),
            "f1_macro": f1_score(yt, yp, average="macro"), "mcc": matthews_corrcoef(yt, yp),
            "brier": brier_score_loss(yt, p1), "log_loss": log_loss(yt, p1, labels=[0, 1])}


def lr_test(spec, X, node, parents, z):
    yv = X[:, spec.idx[node]]
    P = {p: X[:, spec.idx[p]] for p in spec.features}
    Z0 = _parent_design(parents, P, spec) if parents else np.zeros((len(yv), 0))
    Z1 = np.column_stack([Z0, _parent_design([z], P, spec)])
    df_ = Z1.shape[1] - Z0.shape[1]
    if node in spec.binary:
        def ll(Z):
            if Z.shape[1] == 0:
                p = yv.mean(); return np.sum(yv * np.log(p) + (1 - yv) * np.log(1 - p))
            Zs = StandardScaler().fit_transform(Z)
            pr = np.clip(LogisticRegression(C=1e6, max_iter=10000).fit(Zs, yv).predict_proba(Zs)[:, 1], 1e-12, 1 - 1e-12)
            return np.sum(yv * np.log(pr) + (1 - yv) * np.log(1 - pr))
    else:
        def ll(Z):
            r = yv - (LinearRegression().fit(Z, yv).predict(Z) if Z.shape[1] else yv.mean())
            return -0.5 * len(yv) * (np.log(2 * np.pi * np.mean(r ** 2)) + 1)
    stat = max(2 * (ll(Z1) - ll(Z0)), 0.0)
    return stat, df_, float(stats.chi2.sf(stat, df_))


def descendants(dag, node):
    ch = {c for c, ps in dag.items() if node in ps}
    out = set(ch)
    for c in ch:
        out |= descendants(dag, c)
    return out


def dag_checks(name, spec, X):
    full = {**{n: [] for n in spec.exogenous}, **spec.dag}
    rows = []
    for node in spec.topo:
        pa = full[node]
        for z in spec.topo:
            if z == node or z in pa or z in descendants(full, node) or spec.topo.index(z) > spec.topo.index(node):
                continue
            stat, df_, p = lr_test(spec, X, node, pa, z)
            rows.append({"dataset": name, "implication": f"{node} ⊥ {z} | {{{', '.join(pa) or '∅'}}}",
                         "node": node, "z": z, "LR_chi2": stat, "df": df_, "p": p})
    T = pd.DataFrame(rows)
    T["p_holm"] = U.holm(T.p.values)
    T["rejected_holm"] = T.p_holm < U.ALPHA
    # PC (Fisher-z), aristas hacia nodos exogenos prohibidas; orientacion por orden topologico de dominio
    from causallearn.graph.GraphNode import GraphNode
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
    Xs = StandardScaler().fit_transform(X)
    nodes = [GraphNode(f"X{i + 1}") for i in range(len(spec.features))]
    bk = BackgroundKnowledge()
    for r in spec.exogenous:
        for f in spec.features:
            if f != r:
                bk.add_forbidden_by_node(nodes[spec.idx[f]], nodes[spec.idx[r]])
    cg = pc(Xs, alpha=0.05, indep_test="fisherz", stable=True, background_knowledge=bk, show_progress=False)
    G = cg.G.graph
    skel = set()
    for i in range(len(spec.features)):
        for j in range(i + 1, len(spec.features)):
            if G[i, j] != 0 or G[j, i] != 0:
                a, b = sorted([spec.features[i], spec.features[j]], key=spec.topo.index)
                skel.add((a, b))
    pc_dag = {n: [] for n in spec.endogenous}
    for a, b in skel:
        if b in pc_dag:
            pc_dag[b].append(a)
    exp = {tuple(sorted(e, key=spec.topo.index)) for e in dag_edges(spec.dag)}
    info = {"pc_skeleton": sorted(skel), "pc_dag": pc_dag, "expert_edges": sorted(exp),
            "shd_skeleton": len(exp ^ skel), "n_implications": len(T), "n_rejected_holm": int(T.rejected_holm.sum())}
    return T, info


def run_dataset(name, di):
    t0 = time.time()
    spec, cfg = SPECS[name], CONFIG[name]
    d, X, y, g = load(name)
    pre = f"ext_{name}"
    # ---------------- (1) prediccion ----------------
    splits = make_splits(y, g, 20, grouped=True)
    jobs = [(m, sp, "grouped" if g is not None else "stratified") for m in MODELS for sp in splits]
    if g is not None:
        naive = make_splits(y, g, 20, grouped=False)
        jobs += [(m, sp, "row-wise (leaky)") for m in MODELS for sp in naive]
    scores = pd.DataFrame(Parallel(n_jobs=-2)(delayed(pred_one)(m, X, y, sp, pr) for m, sp, pr in jobs))
    scores.to_csv(U.RESULTS_DIR / f"{pre}_pred_scores.csv", index=False)
    print(f"[{name}] prediccion OK ({time.time() - t0:.0f}s)")
    print(scores.groupby(["protocol", "model"])[["balanced_accuracy", "roc_auc"]].mean().round(3).to_string())

    # ---------------- (2) contrafactuales ----------------
    world = SCM(spec.dag, spec).fit(X)
    cf_splits = make_splits(y, g, cfg["cf_repeats"], grouped=True)
    json.dump(cf_splits, open(U.RESULTS_DIR / f"{pre}_splits_cf.json", "w", encoding="utf-8"))
    seeds = {(ci, sp["repeat"], sp["fold"]): U.SEED * 1000 + di * 10000 + ci * 100 + sp["repeat"] * 10 + sp["fold"]
             for ci in range(len(CF_CLFS)) for sp in cf_splits}
    out = Parallel(n_jobs=-2)(
        delayed(run_cf_fold)(c, X, y, np.array(sp["train"]), np.array(sp["test"]), world,
                             seeds[(ci, sp["repeat"], sp["fold"])], tau=U.CF_TAU, m_samples=U.CF_M_SAMPLES,
                             m_eval=U.EVAL_M_SAMPLES, spec=spec,
                             extra={"repeat": sp["repeat"], "fold": sp["fold"]})
        for ci, c in enumerate(CF_CLFS) for sp in cf_splits)
    cf = pd.DataFrame([r for rows in out for r in rows])
    if g is not None:
        cf["unit"] = g[cf.idx.values]
    else:
        cf["unit"] = d["farm_id"].to_numpy()[cf.idx.values]
    cf.to_parquet(U.RESULTS_DIR / f"{pre}_cf_instances.parquet", index=False)
    print(f"[{name}] contrafactuales OK rows={len(cf)} ({time.time() - t0:.0f}s)")
    print(cf.groupby("method")[["feasible", "nominal_valid", "impl_validity", "action_cost", "actionability_violation",
                                "direction_violation", "causal_inconsistency"]].mean().round(3).to_string())

    # ---------------- (3) tau y abduccion ----------------
    jobs = []
    for ci, c in enumerate(CF_CLFS):
        for sp in cf_splits:
            s = seeds[(ci, sp["repeat"], sp["fold"])]
            jobs += [(c, sp, s, t, "posterior") for t in TAUS] + [(c, sp, s, U.CF_TAU, "prior")]
    res = Parallel(n_jobs=-2)(
        delayed(run_cf_fold)(c, X, y, np.array(sp["train"]), np.array(sp["test"]), world, s, methods=["Causal-SCM"],
                             tau=t, abduction=a, m_samples=U.CF_M_SAMPLES, m_eval=U.EVAL_M_SAMPLES, spec=spec,
                             extra={"repeat": sp["repeat"], "fold": sp["fold"], "tau": t, "abduction": a})
        for c, sp, s, t, a in jobs)
    tau_df = pd.DataFrame([r for rows in res for r in rows])
    tau_df.to_csv(U.RESULTS_DIR / f"{pre}_tau.csv", index=False)
    print(f"[{name}] tau OK ({time.time() - t0:.0f}s)")

    # ---------------- (4) DAG ----------------
    T, info = dag_checks(name, spec, X)
    T.to_csv(U.RESULTS_DIR / f"{pre}_dag_tests.csv", index=False)
    json.dump(info, open(U.RESULTS_DIR / f"{pre}_pc.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"[{name}] DAG checks: {info['n_rejected_holm']}/{info['n_implications']} rechazadas; PC: {info['pc_skeleton']}")
    VAR = spec.dag_variants(info["pc_dag"], cfg["extra_dags"])
    worlds = {k: SCM(v, spec).fit(X) for k, v in VAR.items()}
    sub = [sp for sp in cf_splits if sp["repeat"] < 3]
    ci = CF_CLFS.index("LightGBM")
    res = Parallel(n_jobs=-2)(
        delayed(run_cf_fold)("LightGBM", X, y, np.array(sp["train"]), np.array(sp["test"]), worlds[wd],
                             seeds[(ci, sp["repeat"], sp["fold"])], methods=["Causal-SCM"], dag=VAR[md],
                             tau=U.CF_TAU, m_samples=U.CF_M_SAMPLES, m_eval=U.EVAL_M_SAMPLES, spec=spec,
                             extra={"repeat": sp["repeat"], "fold": sp["fold"], "method_dag": md, "world_dag": wd})
        for sp in sub for md in VAR for wd in VAR)
    dag_df = pd.DataFrame([r for rows in res for r in rows])
    dag_df.to_csv(U.RESULTS_DIR / f"{pre}_dag_sensitivity.csv", index=False)
    json.dump({k: v for k, v in VAR.items()}, open(U.RESULTS_DIR / f"{pre}_dag_variants.json", "w", encoding="utf-8"), indent=2,
              ensure_ascii=False)
    print(f"[{name}] DAG sensitivity OK ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    for di, name in enumerate(CONFIG, start=1):
        run_dataset(name, di)
    print("06 OK")
