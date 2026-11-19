"""cf_validation.py - Evaluacion no circular del recurso (ronda 4 de revision).

Responde a cuatro objeciones sobre la evaluacion original (un unico mundo simulado, con el mismo DAG que
el metodo, y exito definido por el clasificador):

* OutcomeModel: mecanismo estructural del DESENLACE (calidad real, no la prediccion del modelo de decision).
  Continuo (log-CCS; exito = CCS <= 200k) con ruido aditivo y abduccion exacta del residuo de la unidad, o
  binario (grado High) con umbral logistico y abduccion posterior truncada. Dos clases funcionales.
* run_cf_fold_multi: genera los contrafactuales UNA vez (misma secuencia de numeros aleatorios que
  cf_lib.run_cf_fold, de modo que las recomendaciones son las mismas) y las implementa en varios mundos:
  distinto grafo (Dense: solo asume el orden causal), distinta clase funcional (forest) y distinta fuerza
  causal (atenuacion gamma = fraccion de la asociacion que es confusion). En cada mundo calcula tambien el
  ORACULO: la mejor accion admisible, independiente de cualquier metodo (existe o no un recurso).
* observed_fn: gancho para comparar con transiciones REALES (caso Vaca: la misma vaca observada en las tres
  fases de THI), sin simulador.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit, logit
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cf_lib import (CF_METHODS, SCM, SEED, CFContext, CFEvaluator, Spec, _parent_design, cf_causal_scm,
                    make_classifier)


GROUP_VARIANT = "Causal-SCM (unit-level noise)"


def dense_dag(spec: Spec) -> dict:
    """Grafo saturado: cada nodo endogeno depende de todos sus predecesores en el orden causal."""
    return {n: spec.topo[:i] for i, n in enumerate(spec.topo) if n in spec.endogenous}


class OutcomeModel:
    """Ecuacion estructural del desenlace Y := g(X) + u, ajustada sobre todas las unidades.

    continuous=True : y = log10(CCS); deseado si y_cf <= threshold. u = residuo exacto de la unidad.
    continuous=False: y = 1 si la unidad esta en la clase deseada; y_cf = 1[logit p(x_cf) + u > 0], con u
                      logistico truncado a la region compatible con el y observado (abduccion posterior).
    """

    def __init__(self, spec: Spec, kind: str = "linear", continuous: bool = True, threshold: float | None = None):
        self.spec, self.kind, self.continuous, self.threshold = spec, kind, continuous, threshold

    def _design(self, X):
        X = np.asarray(X, float)
        return _parent_design(self.spec.features, {f: X[:, self.spec.idx[f]] for f in self.spec.features}, self.spec)

    def fit(self, X, y):
        Z = self._design(X)
        if self.continuous:
            self.m_ = (RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=SEED, n_jobs=1)
                       if self.kind == "forest" else make_pipeline(StandardScaler(), Ridge(alpha=1.0))).fit(Z, y)
        else:
            self.m_ = (RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=SEED, n_jobs=1)
                       if self.kind == "forest" else
                       make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))).fit(Z, y)
        return self

    def score(self, X):
        Z = self._design(X)
        if self.continuous:
            return self.m_.predict(Z)
        if self.kind == "forest":
            return logit(np.clip(self.m_.predict_proba(Z)[:, 1], 0.01, 0.99))
        return self.m_.decision_function(Z)

    def abduct(self, x, y_i, M, rng):
        s = self.score(x[None, :])[0]
        if self.continuous:
            return np.full(M, y_i - s)
        F = np.clip(expit(-s), 1e-9, 1 - 1e-9)
        v = rng.uniform(size=M)
        q = F + v * (1 - F) if y_i == 1 else v * F
        return logit(np.clip(q, 1e-12, 1 - 1e-12))

    def desired(self, S, u):
        """S: (K, M, D) estados implementados; u: (M,). Devuelve (K,) = P(desenlace deseado)."""
        K, M, D = S.shape
        s = self.score(S.reshape(-1, D)).reshape(K, M) + u[None, :]
        return ((s <= self.threshold) if self.continuous else (s > 0)).mean(1)


def run_cf_fold_multi(clf_name, X, y, tr, te, world_main, worlds, outcomes, yout, seed, spec: Spec,
                      tau=0.8, m_samples=100, m_eval=200, extra=None, observed_fn=None, oracle_worlds=None,
                      groups=None):
    """Una fila por (instancia explicada, metodo).

    worlds   : {nombre: (SCM ajustado, gamma)}; outcomes: {nombre: OutcomeModel}; yout: desenlace por fila.
    Columnas : metricas originales en world_main + impl__<mundo> (validez segun el modelo de decision),
               out_<desenlace>__<mundo> (validez segun el desenlace) y oracle_* (mejor accion admisible).
    Infactible => todas las validezes = 0 (misma convencion que run_cf_fold).
    groups   : si se da (medidas repetidas), añade la variante exploratoria GROUP_VARIANT: Causal-SCM con
               abduccion del efecto persistente de la unidad en vez del residuo de la replica. Usa un
               flujo aleatorio propio, de modo que los 5 metodos originales no cambian.
    """
    methods = list(CF_METHODS)
    target = spec.target
    oracle_worlds = list(worlds) if oracle_worlds is None else oracle_worlds
    rng = np.random.default_rng(seed)                    # misma secuencia que cf_lib.run_cf_fold
    rng2 = np.random.default_rng(seed + 7_000_003)       # flujo independiente para los mundos adicionales
    clf = make_classifier(clf_name, seed=SEED)
    if clf_name == "RF":
        clf.set_params(n_jobs=1)
    clf.fit(X[tr], y[tr])
    scm = SCM(spec.dag, spec).fit(X[tr])
    ctx = CFContext(clf, X[tr], y[tr], scm, rng, target=target, tau=tau, m_samples=m_samples, spec=spec)
    ev = CFEvaluator(clf, X[tr], world_main, rng, target=target, m_eval=m_eval, spec=spec)
    a_idx = [spec.idx[f] for f in spec.actionable]
    D = X.shape[1]
    if groups is not None:
        rng3 = np.random.default_rng(seed + 9_000_011)
        scm_g = SCM(spec.dag, spec).fit(X[tr]).set_groups(X, groups, tr)
        ctx_g = CFContext(clf, X[tr], y[tr], scm_g, rng3, target=target, tau=tau, m_samples=m_samples,
                          abduction="group", spec=spec)
        ev_g = CFEvaluator(clf, X[tr], world_main, rng3, target=target, m_eval=m_eval, spec=spec)
        methods = methods + [GROUP_VARIANT]
    rows = []
    for i, p in zip(te, clf.predict(X[te])):
        if p == target:
            continue
        x = X[i]
        inst, cfs_by = [], {}
        for mname in methods:
            info = {}
            if mname == "Causal-SCM":
                cfs, info = cf_causal_scm(x, ctx, return_info=True)
            elif mname == GROUP_VARIANT:
                cfs, info = cf_causal_scm(x, ctx_g, return_info=True)
            else:
                cfs = CF_METHODS[mname](x, ctx)
            base = {"classifier": clf_name, "idx": int(i), "y_true": int(y[i]), "y_pred": int(p), "method": mname,
                    "feasible": float(len(cfs) > 0), "n_cfs": len(cfs), **info, **(extra or {})}
            if cfs:
                evs = (ev_g if mname == GROUP_VARIANT else ev).evaluate_many(x, cfs, causal=mname.startswith("Causal-SCM"))
                base.update({k: float(np.mean([e[k] for e in evs])) for k in evs[0]})
                base["cf_first"] = cfs[0].round(4).tolist()
            else:
                base["impl_validity"] = 0.0
            cfs_by[mname] = [np.asarray(c, float) for c in cfs]
            inst.append(base)
        # ---------- mundos adicionales: mismas recomendaciones, mismo ruido para todos los metodos ----------
        allowed = ctx.allowed_actions(x)
        owner, A = [], []
        for mname in methods:
            for c in cfs_by[mname]:
                owner.append(mname); A.append(c[a_idx])
        n_cf = len(A)
        A = np.vstack([np.array(A).reshape(n_cf, len(a_idx)), allowed]) if (n_cf or len(allowed)) else None
        owner = np.array(owner)
        u_out = {o: om.abduct(x, yout[i], m_eval, rng2) for o, om in outcomes.items()}
        U_cache, common = {}, {"n_allowed_actions": int(len(allowed))}
        per = {m: {} for m in methods}
        for wname, (w, gamma) in worlds.items():
            if A is None:
                continue
            if id(w) not in U_cache:
                U_cache[id(w)] = w.abduct(x, m_eval, rng2, mode="posterior")
            S = w.counterfactual_batch(x, A, U_cache[id(w)], attenuation=gamma)       # (K, M, D)
            K = S.shape[0]
            impl = (clf.predict(S.reshape(-1, D)).reshape(K, m_eval) == target).mean(1)
            outs = {o: om.desired(S, u_out[o]) for o, om in outcomes.items()}
            for mname in methods:
                sel = np.where(owner == mname)[0]
                if len(sel):
                    per[mname][f"impl__{wname}"] = float(impl[sel].mean())
                    for o in outs:
                        per[mname][f"out_{o}__{wname}"] = float(outs[o][sel].mean())
            if wname in oracle_worlds:
                # cota superior: mejor accion admisible de la rejilla o cualquier recomendacion evaluada
                common[f"oracle_impl__{wname}"] = float(impl.max())
                common[f"oracle_grid_impl__{wname}"] = float(impl[n_cf:].max()) if len(allowed) else 0.0
                for o in outs:
                    common[f"oracle_out_{o}__{wname}"] = float(outs[o].max())
                    common[f"oracle_grid_out_{o}__{wname}"] = float(outs[o][n_cf:].max()) if len(allowed) else 0.0
        for base in inst:
            mname = base["method"]
            for wname in worlds:
                base.setdefault(f"impl__{wname}", 0.0)
                base[f"impl__{wname}"] = per[mname].get(f"impl__{wname}", 0.0)
                for o in outcomes:
                    base[f"out_{o}__{wname}"] = per[mname].get(f"out_{o}__{wname}", 0.0)
            base.update(common)
            if observed_fn is not None:
                obs = [observed_fn(i, c, x, clf, target) for c in cfs_by[mname]]
                if obs:
                    base.update({k: float(np.mean([o[k] for o in obs])) for k in obs[0]})
                else:
                    base.update({"obs_exact": 1.0, "obs_null": 1.0, "obs_model": 0.0, "obs_real": 0.0})
            rows.append(base)
    return rows
