"""09_noncircular_validation.py
Ronda 4 de revision: evaluacion no circular del recurso.

(A) Las mismas recomendaciones de los 5 metodos (mismos folds, clasificadores y semillas que 02b/06) se
    implementan en un 2x2 de mundos (clase funcional x grafo) y bajo confusion no medida:
        lin_expert    SCM lineal, DAG experto            (mundo original)
        lin_dense     SCM lineal, grafo saturado         (no asume el DAG experto; solo el orden causal)
        forest_expert SCM Random Forest, DAG experto
        forest_dense  SCM Random Forest, grafo saturado  (ni la forma funcional ni el grafo del metodo)
        att25..att100 lin_expert con atenuacion gamma    (gamma = fraccion de la asociacion que NO es causal)
    Caso Vaca: variante exploratoria de Causal-SCM con abduccion del efecto persistente de la vaca.
    En cada mundo: validez segun el modelo de decision, validez segun el DESENLACE (mecanismo estructural
    de log-CCS o del grado, dos clases funcionales) y oraculo (mejor accion admisible).
    Caso Vaca: ademas, validez OBSERVADA en las transiciones reales de la misma vaca entre fases de THI.
(B) Validacion del simulador con datos reales (caso Vaca): SCM ajustados sin la vaca; se predice el estado
    fisiologico de cada muestra bajo do(THI = a) y se compara con lo observado en esa vaca en la fase a.

Salidas: results/val_<caso>.parquet, results/val_within_cow_transitions.parquet
Uso: python 09_noncircular_validation.py [--pilot]
"""
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import utils as U
from cf_lib import SCM, SPECS
from cf_validation import OutcomeModel, dense_dag, run_cf_fold_multi

warnings.filterwarnings("ignore")
PROC = U.PROJECT_DIR / "data" / "processed"
CLFS = ["LR", "MLP", "RF", "XGBoost", "LightGBM"]
GAMMAS = {"att25": 0.25, "att50": 0.50, "att75": 0.75, "att100": 1.00}
SCC_THR = float(np.log10(200_000)) + 1e-6   # CCS = 200k exactos cuenta como deseado (scc_high = CCS > 200k)
CASES = {"milk_quality": ("Sample", U.DATA_PATH, "grade", None, "splits_cf.json", 0),
         "heat_stress": ("Cow", PROC / "heat_stress_ml_ready.csv", "scc_high", "cow_id", "ext_heat_stress_splits_cf.json", 1),
         "scc_management": ("Farm", PROC / "scc_management_ml_ready.csv", "scc_high", "farm_id", "ext_scc_management_splits_cf.json", 2)}
PILOT = "--pilot" in sys.argv


def seed_of(di, ci, r, f):
    base = U.SEED * 1000 + ci * 100 + r * 10 + f
    return base if di == 0 else base + di * 10000


class CowObserved:
    """Transiciones reales: la misma vaca observada en la fase de THI recomendada (sin simulador).

    obs_exact = 1 si la parte accionable de la recomendacion es solo un cambio de THI (o ninguno): entonces
    el estado implementado se observa exactamente. Si ademas pide cambiar el DMI, solo se observa la parte
    de THI (obs_exact = 0). obs_model: el clasificador sobre los estados reales; obs_real: CCS real <= 200k.
    """

    def __init__(self, d, X, y, spec):
        self.cow, self.thi, self.X, self.y = d.cow_id.to_numpy(), d.thi.to_numpy(), X, y
        self.it, self.idm = spec.idx["thi"], spec.idx["dmi_kg"]

    def __call__(self, i, cf, x, clf, target):
        a = int(round(cf[self.it]))
        rows = np.where((self.cow == self.cow[i]) & (self.thi == a))[0]
        null = a == self.thi[i]
        if null:
            rows = rows[rows != i]
        return {"obs_exact": float(abs(cf[self.idm] - x[self.idm]) < 1e-9), "obs_null": float(null),
                "obs_model": float((clf.predict(self.X[rows]) == target).mean()),
                "obs_real": float((self.y[rows] == target).mean())}


def run_case(name):
    label, path, target, unit_col, split_file, di = CASES[name]
    spec = SPECS[name]
    d = pd.read_csv(path)
    X, y = d[spec.features].to_numpy(float), d[target].to_numpy()
    lin_e = SCM(spec.dag, spec).fit(X)
    worlds = {"lin_expert": (lin_e, 0.0),
              "lin_dense": (SCM(dense_dag(spec), spec).fit(X), 0.0),
              "forest_expert": (SCM(spec.dag, spec, kind="forest").fit(X), 0.0),
              "forest_dense": (SCM(dense_dag(spec), spec, kind="forest").fit(X), 0.0),
              **{k: (lin_e, g) for k, g in GAMMAS.items()}}
    if name == "milk_quality":
        yout = (y == spec.target).astype(int)
        outcomes = {k: OutcomeModel(spec, k, continuous=False).fit(X, yout) for k in ("linear", "forest")}
    else:
        yout = d["scc_log"].to_numpy(float)
        assert ((yout > SCC_THR) == (y == 1)).all()
        outcomes = {k: OutcomeModel(spec, k, continuous=True, threshold=SCC_THR).fit(X, yout) for k in ("linear", "forest")}
    obs = CowObserved(d, X, y, spec) if name == "heat_stress" else None
    splits = json.load(open(U.RESULTS_DIR / split_file))
    if PILOT:
        splits = [s for s in splits if s["repeat"] == 0][:2]
    out = Parallel(n_jobs=-2)(
        delayed(run_cf_fold_multi)(c, X, y, np.array(sp["train"]), np.array(sp["test"]), lin_e, worlds, outcomes,
                                   yout, seed_of(di, ci, sp["repeat"], sp["fold"]), spec, tau=U.CF_TAU,
                                   m_samples=U.CF_M_SAMPLES, m_eval=U.EVAL_M_SAMPLES,
                                   extra={"repeat": sp["repeat"], "fold": sp["fold"]}, observed_fn=obs,
                                   groups=d[unit_col].to_numpy() if name == "heat_stress" else None)
        for ci, c in enumerate(CLFS if not PILOT else CLFS[:2]) for sp in splits)
    cf = pd.DataFrame([r for rows in out for r in rows])
    cf["unit"] = cf["idx"] if unit_col is None else d[unit_col].to_numpy()[cf.idx.values]
    cf["cf_first"] = cf["cf_first"].map(lambda v: json.dumps(v) if isinstance(v, list) else None)
    cf.to_parquet(U.RESULTS_DIR / f"val_{name}{'_pilot' if PILOT else ''}.parquet", index=False)
    return cf


# =============================================================================
# (B) Validacion del simulador con transiciones reales (caso Vaca)
# =============================================================================
def transitions_fold(sp):
    spec = SPECS["heat_stress"]
    d = pd.read_csv(PROC / "heat_stress_ml_ready.csv")
    X = d[spec.features].to_numpy(float)
    cow, thi = d.cow_id.to_numpy(), d.thi.to_numpy()
    tr, te = np.array(sp["train"]), np.array(sp["test"])
    grp = SCM(spec.dag, spec).fit(X[tr]).set_groups(X, cow, tr)
    sims = {"lin_expert": SCM(spec.dag, spec).fit(X[tr]), "lin_dense": SCM(dense_dag(spec), spec).fit(X[tr]),
            "forest_expert": SCM(spec.dag, spec, kind="forest").fit(X[tr]),
            "forest_dense": SCM(dense_dag(spec), spec, kind="forest").fit(X[tr])}
    it = spec.idx["thi"]
    act = {a: j for j, a in enumerate(spec.actionable)}
    med = ["rectal_temp_c", "resp_bpm", "pulse_bpm", "dmi_kg"]
    rng = np.random.default_rng(U.SEED)
    rows = []
    for i in te:
        x = X[i]
        for a in (0, 1, 2):
            if a == thi[i]:
                continue
            obs = X[(cow == cow[i]) & (thi == a)].mean(0)
            A = np.array([[x[spec.idx[f]] for f in spec.actionable]], float)
            A[0, act["thi"]] = a
            preds = {"No change (independence)": x}
            for k, w in sims.items():
                preds[k] = w.counterfactual_batch(x, A, w.abduct(x, 1, rng))[0, 0]
            # sin abduccion: media poblacional de la ecuacion estructural (ruido = 0)
            w = sims["lin_expert"]
            preds["lin_expert, no abduction"] = w.counterfactual_batch(x, A, {n: np.zeros(1) for n in w.dag})[0, 0]
            # efecto persistente de la vaca (contraido), sin el ruido de la replica
            preds["lin_expert, cow-level abduction"] = grp.counterfactual_batch(x, A, grp.abduct(x, 1, rng, mode="group_mean"))[0, 0]
            for k, v in preds.items():
                for m in med:
                    j = spec.idx[m]
                    rows.append({"repeat": sp["repeat"], "fold": sp["fold"], "idx": int(i), "cow_id": int(cow[i]),
                                 "thi_from": int(thi[i]), "thi_to": a, "predictor": k, "mediator": m,
                                 "factual": x[j], "pred": float(v[j]), "obs": float(obs[j])})
    return rows


if __name__ == "__main__":
    t0 = time.time()
    for name in CASES:
        cf = run_case(name)
        cols = ["impl_validity", "impl__lin_expert", "impl__lin_dense", "impl__forest_expert", "impl__forest_dense",
                "impl__att50", "out_linear__lin_expert", "out_forest__forest_dense"]
        print(f"[{name}] rows={len(cf)} ({time.time() - t0:.0f}s)")
        print(cf.groupby("method")[cols].mean().round(3).to_string(), flush=True)
    splits = json.load(open(U.RESULTS_DIR / "ext_heat_stress_splits_cf.json"))
    if PILOT:
        splits = splits[:2]
    res = Parallel(n_jobs=-2)(delayed(transitions_fold)(sp) for sp in splits)
    T = pd.DataFrame([r for rows in res for r in rows])
    T.to_parquet(U.RESULTS_DIR / f"val_within_cow_transitions{'_pilot' if PILOT else ''}.parquet", index=False)
    print(f"09 OK ({time.time() - t0:.0f}s)")
