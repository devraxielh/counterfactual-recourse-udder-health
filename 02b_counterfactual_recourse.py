"""02b_counterfactual_recourse.py
Experimento central (idea C1): contrafactuales causales (Causal-SCM, propuesto) vs. no causales.

* Instancias explicadas: perfiles de test (5 folds x 10 repeticiones sobre los 83 unicos) cuya
  prediccion != High. Objetivo del recurso: High.
* Clasificadores (sistema de decision): LR, MLP, RF, XGBoost, LightGBM, ajustados por fold.
* SCM del metodo: DAG experto ajustado SOLO en train del fold.
* SCM-mundo (simulador de implementacion): DAG experto ajustado en los 83 perfiles; las acciones
  recomendadas sobre variables accionables (temperature, fat) se propagan con ruido abducido
  (posterior) y se re-clasifican -> validez al implementar (endpoint primario, decidido a priori).

Salida: results/cf_instances.parquet
"""
import json
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import utils as U
from cf_lib import DAG_EXPERT, FEATURES, SCM, run_cf_fold

warnings.filterwarnings("ignore")
CF_CLASSIFIERS = ["LR", "MLP", "RF", "XGBoost", "LightGBM"]

if __name__ == "__main__":
    t0 = time.time()
    uniq = pd.read_csv(U.DATA_PATH)
    X, y = uniq[FEATURES].to_numpy(float), uniq["grade"].to_numpy()
    world = SCM(DAG_EXPERT).fit(X)
    splits = json.load(open(U.RESULTS_DIR / "splits_cf.json"))
    jobs = []
    for ci, c in enumerate(CF_CLASSIFIERS):
        for sp in splits:
            seed = U.SEED * 1000 + ci * 100 + sp["repeat"] * 10 + sp["fold"]
            jobs.append((c, sp, seed))
    out = Parallel(n_jobs=-2)(
        delayed(run_cf_fold)(c, X, y, np.array(sp["train"]), np.array(sp["test"]), world, seed,
                             tau=U.CF_TAU, m_samples=U.CF_M_SAMPLES, m_eval=U.EVAL_M_SAMPLES,
                             extra={"repeat": sp["repeat"], "fold": sp["fold"]})
        for c, sp, seed in jobs)
    df = pd.DataFrame([r for rows in out for r in rows])
    df.to_parquet(U.RESULTS_DIR / "cf_instances.parquet", index=False)
    summ = df.groupby("method")[["feasible", "nominal_valid", "impl_validity", "action_cost", "distance_l1",
                                 "sparsity", "actionability_violation", "causal_inconsistency",
                                 "lof_outlier"]].mean().round(3)
    print(summ.to_string())
    print(df.groupby(["classifier", "method"]).impl_validity.mean().unstack().round(3).to_string())
    print(f"02b OK rows={len(df)} in {time.time() - t0:.1f}s")
