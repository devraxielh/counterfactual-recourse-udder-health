"""cf_lib.py - Modelos, SCM y generadores de contrafactuales (generico por dataset via `Spec`).

Componentes
-----------
* Spec: especificacion de un dataset (features, binarias, accionables con direccion permitida, exogenas,
  orden topologico, DAG experto, transformaciones de padres, restricciones de factibilidad, clase objetivo).
  MILK_SPEC reproduce exactamente la configuracion original del paper (Milk Quality).
* make_classifier(name): fabrica de clasificadores (Pipeline con escalado ajustado dentro del fold).
* DomainRuleClassifier: regla de dominio de Milk Quality (k aprendido SOLO en el fold de entrenamiento).
* SCM: modelo causal estructural aditivo/logistico sobre un DAG; abduccion del ruido (continuas: residuo
  exacto; binarias: logistica truncada posterior); do() sobre cualquier nodo accionable (corta sus padres).
* Generadores: NUN, Wachter-grid, DiCE-random, Actionable-indep, Causal-SCM (propuesto).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.special import expit, logit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor, NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

SEED = 42


# ============================================================================
# Especificacion de dataset
# ============================================================================
@dataclass
class Spec:
    name: str
    features: list
    binary: set
    actionable: list                       # orden = columnas de la matriz de acciones
    directions: dict                       # feature -> "any" | "increase" | "decrease"
    exogenous: list                        # nodos sin ecuacion estructural (raices)
    topo: list                             # orden topologico de TODOS los features
    dag: dict                              # padres de cada nodo endogeno (DAG experto)
    target: int                            # clase deseada del recurso
    transforms: dict = field(default_factory=dict)   # padre -> f(v) -> lista de columnas
    action_filter: Callable | None = None  # f(x, actions) -> mascara de acciones fisicamente posibles
    max_unique_grid: int = 20              # features con mas valores unicos se discretizan por cuantiles
    n_quantiles: int = 9
    include_status_quo: bool = True        # ronda 4: cada palanca puede quedarse en su valor factual

    def __post_init__(self):
        self.idx = {f: i for i, f in enumerate(self.features)}
        self.non_actionable = [f for f in self.features if f not in self.actionable]
        self.endogenous = [f for f in self.features if f not in self.exogenous]   # orden = features (RNG)

    def dag_variants(self, pc_dag: dict | None = None, extra: dict | None = None) -> dict:
        v = {"Expert": self.dag}
        if extra:
            v.update(extra)
        v["Empty"] = {k: [] for k in self.endogenous}
        v["Dense"] = {n: self.topo[:i] for i, n in enumerate(self.topo) if n in self.endogenous}
        if pc_dag is not None:
            v["PC (data)"] = pc_dag
        return v


# ----------------------------------------------------------------------------
# Milk Quality (configuracion original)
# ----------------------------------------------------------------------------
FEATURES = ["ph", "temperature", "taste", "odor", "fat", "turbidity", "colour"]
IDX = {f: i for i, f in enumerate(FEATURES)}
ACTIONABLE = ["temperature", "fat"]
NON_ACTIONABLE = [f for f in FEATURES if f not in ACTIONABLE]
BINARY = {"taste", "odor", "fat", "turbidity"}
PH_REF = 6.6
PH_RANGE = (6.4, 6.8)
TEMP_MAX = 45
DAG_EXPERT = {
    "ph": ["temperature"],
    "turbidity": ["ph", "fat"],
    "odor": ["temperature", "ph"],
    "taste": ["temperature", "ph", "fat"],
    "colour": ["fat", "turbidity"],
}
TOPO_ORDER = ["temperature", "fat", "ph", "turbidity", "odor", "taste", "colour"]

MILK_SPEC = Spec(
    name="milk_quality", features=FEATURES, binary=BINARY, actionable=ACTIONABLE,
    directions={"temperature": "any", "fat": "any"}, exogenous=["temperature", "fat"], topo=TOPO_ORDER,
    dag=DAG_EXPERT, target=2,
    transforms={"ph": lambda v: [v, np.abs(v - PH_REF)],                  # efecto no monotonico del pH
                "temperature": lambda v: [v, (v - 40.0) ** 2 / 100.0]},   # efecto no lineal de la temperatura
)


def dag_variants(pc_dag: dict | None = None) -> dict:
    """Variantes del DAG de Milk Quality (usadas en 03_ablation_and_sensitivity.py)."""
    v = {
        "Expert": DAG_EXPERT,
        "No T→pH": {**DAG_EXPERT, "ph": []},
        "No T→sensory": {**DAG_EXPERT, "odor": ["ph"], "taste": ["ph", "fat"]},
        "Empty": {k: [] for k in DAG_EXPERT},
        "Dense": {n: TOPO_ORDER[:i] for i, n in enumerate(TOPO_ORDER) if i >= 2},
    }
    if pc_dag is not None:
        v["PC (data)"] = pc_dag
    return v


def dag_edges(dag: dict) -> set:
    return {(p, c) for c, ps in dag.items() for p in ps}


# ----------------------------------------------------------------------------
# Heat stress (Mendeley 954f6g36sb) - unidad: muestra (vaca x fase THI x replica)
# ----------------------------------------------------------------------------
HEAT_FEATURES = ["thi", "hf_pct", "rectal_temp_c", "pulse_bpm", "resp_bpm", "dmi_kg"]
HEAT_DAG = {
    # carga termica -> termorregulacion (Kadzere et al. 2002; Polsky & von Keyserlingk 2017)
    "rectal_temp_c": ["thi", "hf_pct"],
    "resp_bpm": ["thi", "hf_pct"],
    "pulse_bpm": ["thi", "hf_pct", "rectal_temp_c"],
    # ingesta: potencial genetico (HF) y depresion por calor (West 2003)
    "dmi_kg": ["thi", "hf_pct", "rectal_temp_c"],
}
HEAT_SPEC = Spec(
    name="heat_stress", features=HEAT_FEATURES, binary=set(),
    actionable=["thi", "dmi_kg"],                       # enfriamiento (ventilacion/aspersion) y alimentacion
    directions={"thi": "decrease", "dmi_kg": "increase"},
    exogenous=["thi", "hf_pct"],
    topo=["thi", "hf_pct", "rectal_temp_c", "resp_bpm", "pulse_bpm", "dmi_kg"],
    dag=HEAT_DAG, target=0,                             # deseado: CCS <= 200k
)

# ----------------------------------------------------------------------------
# SCC management practices (Mendeley xbk6x9rrnb) - unidad: granja
# ----------------------------------------------------------------------------
SCC_FEATURES = ["ps", "tm", "mp", "ml", "efj", "cmt", "pv", "tw", "ppd", "its", "td"]
SCC_DAG = {
    "tm": ["ps"],                 # sistema intensivo (pasto irrigado) -> ordeño mecanizado
    "mp": ["tm"],                 # ordeño mecanizado -> sala de ordeño
    "ml": ["tm", "mp"],           # linea de ordeño requiere infraestructura
    "its": ["tm"],                # inmersion de pezoneras solo existe con ordeño mecanico
    "cmt": ["ml"],                # rutina organizada (linea) -> deteccion con CMT
    "ppd": ["ml", "cmt"],         # paquete de rutina de ordeño (NMC 10-point plan)
    "td": ["cmt"],                # deteccion -> tratamiento de vacas secas/afectadas
}


def _scc_filter(x, actions):
    """La inmersion de pezoneras (its) solo es posible si el ordeño es mecanizado (tm = 1)."""
    its_col = SCC_SPEC.actionable.index("its")
    if x[SCC_SPEC.idx["tm"]] == 1:
        return np.ones(len(actions), bool)
    return actions[:, its_col] <= x[SCC_SPEC.idx["its"]]


SCC_SPEC = Spec(
    name="scc_management", features=SCC_FEATURES, binary=set(SCC_FEATURES),
    actionable=["efj", "cmt", "tw", "ppd", "its", "td"],        # practicas de higiene de bajo costo
    directions={f: "increase" for f in ["efj", "cmt", "tw", "ppd", "its", "td"]},   # solo adopcion
    exogenous=["ps", "efj", "pv", "tw"],
    topo=["ps", "efj", "pv", "tw", "tm", "mp", "ml", "its", "cmt", "ppd", "td"],
    dag=SCC_DAG, target=0, action_filter=_scc_filter,           # deseado: CCS <= 200k
)

SPECS = {"milk_quality": MILK_SPEC, "heat_stress": HEAT_SPEC, "scc_management": SCC_SPEC}


# ============================================================================
# Clasificadores
# ============================================================================
class DomainRuleClassifier(BaseEstimator, ClassifierMixin):
    """Regla de dominio interpretable (Milk Quality); el umbral k se aprende en el fold de entrenamiento."""

    def fit(self, X, y):
        X = np.asarray(X, float)
        self.classes_ = np.array([0, 1, 2])
        best = (-1, 2)
        for k in (1, 2, 3, 4):
            self.k_ = k
            s = balanced_accuracy_score(y, self.predict(X))
            if s > best[0]:
                best = (s, k)
        self.k_ = best[1]
        return self

    def predict(self, X):
        X = np.asarray(X, float)
        ph, t = X[:, IDX["ph"]], X[:, IDX["temperature"]]
        oc = X[:, [IDX["taste"], IDX["odor"], IDX["fat"], IDX["turbidity"]]].sum(1)
        low = (ph < PH_RANGE[0]) | (ph > PH_RANGE[1]) | (t > TEMP_MAX)
        return np.where(low, 0, np.where(oc >= self.k_, 2, 1))

    def predict_proba(self, X):
        p = self.predict(X)
        P = np.full((len(p), 3), 0.01)
        P[np.arange(len(p)), p] = 0.98
        return P


def make_classifier(name: str, seed: int = SEED):
    if name == "Dummy":
        return DummyClassifier(strategy="most_frequent")
    if name == "DomainRule":
        return DomainRuleClassifier()
    if name == "LR":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
    if name == "kNN":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
    if name == "SVM-RBF":
        return make_pipeline(StandardScaler(), SVC(C=1.0, gamma="scale", probability=True, random_state=seed))
    if name == "DT":
        return DecisionTreeClassifier(random_state=seed)
    if name == "RF":
        return RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=-1)
    if name == "XGBoost":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.1, subsample=0.8,
                             colsample_bytree=0.8, random_state=seed, n_jobs=4, verbosity=0)
    if name == "LightGBM":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=8, min_child_samples=5,
                              random_state=seed, verbose=-1, n_jobs=1)
    if name == "MLP":
        return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32, 16), alpha=1e-3,
                                                             max_iter=3000, random_state=seed))
    raise ValueError(name)


# ============================================================================
# SCM
# ============================================================================
def _parent_design(parents: list[str], P: dict, spec: Spec = MILK_SPEC) -> np.ndarray:
    cols = []
    n = len(next(iter(P.values()))) if P else 0
    for p in parents:
        v = np.asarray(P[p], float)
        f = spec.transforms.get(p)
        cols += f(v) if f else [v]
    return np.column_stack(cols) if cols else np.zeros((n, 0))


class SCM:
    """SCM con ruido aditivo (continuas) y umbral logistico (binarias).

    kind="linear" (por defecto; el que usa Causal-SCM): ridge / regresion logistica.
    kind="forest" (mundo alternativo para la robustez R1.1): Random Forest con el mismo DAG y el mismo
    mecanismo de ruido (aditivo / umbral logistico sobre logit(p_RF)), abduccion posterior identica.
    """

    def __init__(self, dag: dict, spec: Spec = MILK_SPEC, kind: str = "linear"):
        self.spec = spec
        self.kind = kind
        self.dag = {n: list(dag.get(n, [])) for n in spec.endogenous}

    def fit(self, X: np.ndarray):
        from sklearn.ensemble import RandomForestRegressor
        X = np.asarray(X, float)
        I, B = self.spec.idx, self.spec.binary
        self.models_, self.bounds_, self.resid_ = {}, {}, {}
        for node, pa in self.dag.items():
            y = X[:, I[node]]
            self.bounds_[node] = (y.min(), y.max())
            P = {p: X[:, I[p]] for p in pa}
            Z = _parent_design(pa, P, self.spec) if pa else np.zeros((len(y), 0))
            if node in B:
                if len(np.unique(y)) < 2 or Z.shape[1] == 0:
                    pbar = np.clip(y.mean(), 1e-3, 1 - 1e-3)
                    self.models_[node] = ("const", float(logit(pbar)))
                elif self.kind == "forest":
                    m = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, random_state=SEED,
                                               n_jobs=1).fit(Z, y)
                    self.models_[node] = ("rf_clf", m)
                else:
                    m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)).fit(Z, y)
                    self.models_[node] = ("logit", m)
            else:
                if Z.shape[1] == 0:
                    self.models_[node] = ("const", float(y.mean()))
                    self.resid_[node] = y - y.mean()
                else:
                    if self.kind == "forest":
                        m = RandomForestRegressor(n_estimators=200, min_samples_leaf=5, random_state=SEED,
                                                  n_jobs=1, oob_score=True).fit(Z, y)
                        self.models_[node] = ("rf_reg", m)
                        self.resid_[node] = y - m.oob_prediction_
                    else:
                        m = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(Z, y)
                        self.models_[node] = ("ridge", m)
                        self.resid_[node] = y - m.predict(Z)
        return self

    def _mean(self, node: str, P: dict, n: int) -> np.ndarray:
        kind, m = self.models_[node]
        if kind == "const":
            return np.full(n, m)
        Z = _parent_design(self.dag[node], P, self.spec)
        if kind == "logit":
            return m.decision_function(Z)
        if kind == "rf_clf":
            return logit(np.clip(m.predict_proba(Z)[:, 1], 0.01, 0.99))
        return m.predict(Z)

    def set_groups(self, X: np.ndarray, groups: np.ndarray, fit_rows: np.ndarray):
        """Ruido persistente de la unidad (p. ej. la vaca) para mode="group" / "group_mean" (ronda 5).

        El residuo de cada nodo continuo se descompone en un efecto de grupo y ruido de la replica. El
        efecto de grupo es la media de los residuos del grupo contraida por su fiabilidad,
        lam = s2_entre / (s2_entre + s2_intra / n_g); las varianzas se estiman SOLO con fit_rows (train).
        Solo usa las features del grupo (su historial), nunca el desenlace."""
        X, groups = np.asarray(X, float), np.asarray(groups)
        I = self.spec.idx
        self._gkey = {X[i].tobytes(): groups[i] for i in range(len(X))}
        self.group_u_, self.within_ = {}, {}
        P = {f: X[:, I[f]] for f in self.spec.features}
        fit = np.zeros(len(X), bool); fit[fit_rows] = True
        for node in self.dag:
            if node in self.spec.binary:
                continue
            r = X[:, I[node]] - self._mean(node, P, len(X))
            ug = np.unique(groups)
            mean = {g: r[groups == g].mean() for g in ug}
            n = {g: int((groups == g).sum()) for g in ug}
            gf = np.unique(groups[fit])
            within = np.concatenate([r[fit & (groups == g)] - r[fit & (groups == g)].mean() for g in gf])
            nbar = np.mean([(fit & (groups == g)).sum() for g in gf])
            s2w = within.var(ddof=len(gf)) if len(within) > len(gf) else 0.0
            s2b = max(np.var([r[fit & (groups == g)].mean() for g in gf], ddof=1) - s2w / nbar, 0.0)
            self.group_u_[node] = {g: (s2b / (s2b + s2w / n[g]) if s2b + s2w > 0 else 0.0) * mean[g] for g in ug}
            self.within_[node] = within
        return self

    def abduct(self, x: np.ndarray, M: int, rng, mode: str = "posterior") -> dict:
        """Ruido exogeno para la instancia factual x (M muestras).

        mode="group": efecto persistente de la unidad + ruido de replica remuestreado (nodos continuos);
        mode="group_mean": solo el efecto persistente (valor esperado). Binarios: posterior truncado."""
        I, B = self.spec.idx, self.spec.binary
        U = {}
        P = {f: np.array([x[I[f]]]) for f in self.spec.features}
        for node in self.dag:
            mu = self._mean(node, P, 1)[0]
            if mode in ("group", "group_mean") and node not in B:
                ug = self.group_u_[node][self._gkey[np.asarray(x, float).tobytes()]]
                U[node] = np.full(M, ug) + (rng.choice(self.within_[node], size=M, replace=True) if mode == "group" else 0.0)
                continue
            if node in B:
                if mode == "prior":
                    U[node] = rng.logistic(size=M)
                    continue
                thr = -mu
                Fthr = np.clip(expit(thr), 1e-9, 1 - 1e-9)
                v = rng.uniform(size=M)
                q = Fthr + v * (1 - Fthr) if x[I[node]] == 1 else v * Fthr
                U[node] = logit(np.clip(q, 1e-12, 1 - 1e-12))
            else:
                if mode == "prior":
                    U[node] = rng.choice(self.resid_[node], size=M, replace=True)
                else:
                    U[node] = np.full(M, x[I[node]] - mu)
        return U

    def counterfactual_batch(self, x: np.ndarray, actions: np.ndarray, U: dict,
                             attenuation: float = 0.0) -> np.ndarray:
        """actions: (K, n_actionable) valores objetivo en el orden spec.actionable. Devuelve (K, M, D).

        attenuation = gamma en [0, 1] (analisis de sensibilidad a confusion no medida, ronda 4): solo una
        fraccion (1 - gamma) de la respuesta estructural estimada es causal; el resto se atribuye a un
        confusor y NO responde a la intervencion: mu = mu(factual) + (1 - gamma) * (mu(cf) - mu(factual)).
        gamma = 0 reproduce exactamente el comportamiento original; gamma = 1 deja inmoviles los descendientes.

        Semantica de intervencion DISPERSA (Karimi et al., 2021): solo se interviene do(a) sobre las
        palancas cuyo valor objetivo difiere del factual; las palancas no modificadas siguen su ecuacion
        estructural y pueden responder a los cambios aguas arriba (p. ej. enfriar -> sube el DMI)."""
        sp = self.spec
        K = actions.shape[0]
        M = len(next(iter(U.values())))
        act_col = {a: j for j, a in enumerate(sp.actionable)}
        vals = {}
        Pf = {f: np.array([float(x[sp.idx[f]])]) for f in sp.features} if attenuation else None
        for node in sp.topo:
            if node in sp.exogenous:
                struct = np.full((K, M), float(x[sp.idx[node]]))
            else:
                P = {p: vals[p].ravel() for p in self.dag[node]}
                mu = self._mean(node, P, K * M).reshape(K, M)
                if attenuation:
                    mu_f = self._mean(node, Pf, 1)[0]
                    mu = mu_f + (1.0 - attenuation) * (mu - mu_f)
                u = U[node][None, :]
                if node in sp.binary:
                    struct = (mu + u > 0).astype(float)
                else:
                    lo, hi = self.bounds_[node]
                    struct = np.clip(mu + u, lo, hi)
            if node in act_col:
                target = actions[:, [act_col[node]]].astype(float)
                changed = np.abs(target - x[sp.idx[node]]) > 1e-9          # (K, 1)
                vals[node] = np.where(changed, np.repeat(target, M, axis=1), struct)   # do() corta sus padres
            else:
                vals[node] = struct
        return np.stack([vals[f] for f in sp.features], axis=-1)


# ============================================================================
# Utilidades de contrafactuales
# ============================================================================
def feature_levels(col: np.ndarray, spec: Spec) -> np.ndarray:
    u = np.unique(col)
    if len(u) <= spec.max_unique_grid:
        return u
    q = np.linspace(0.05, 0.95, spec.n_quantiles)
    return np.unique(np.quantile(col, q, method="nearest"))


def build_grid(Xtr: np.ndarray, spec: Spec = MILK_SPEC) -> np.ndarray:
    vals = [feature_levels(Xtr[:, j], spec) for j in range(Xtr.shape[1])]
    return np.array(list(itertools.product(*vals)), dtype=float)


def norm_l1(x: np.ndarray, C: np.ndarray, ranges: np.ndarray) -> np.ndarray:
    return (np.abs(C - x) / ranges).sum(axis=-1)


def action_cost(x: np.ndarray, C: np.ndarray, ranges: np.ndarray, spec: Spec = MILK_SPEC) -> np.ndarray:
    a = [spec.idx[f] for f in spec.actionable]
    return (np.abs(C[..., a] - x[a]) / ranges[a]).sum(axis=-1)


def representative(samples: np.ndarray, spec: Spec = MILK_SPEC) -> np.ndarray:
    """Punto representativo de muestras (M, D): moda en binarias, media en continuas."""
    r = samples.mean(axis=0)
    for f in spec.binary:
        r[spec.idx[f]] = float(r[spec.idx[f]] >= 0.5)
    return r


class CFContext:
    """Todo lo que un generador necesita dentro de un fold."""

    def __init__(self, clf, Xtr, ytr, scm_train, rng, target=2, tau=0.8, m_samples=100,
                 abduction="posterior", spec: Spec = MILK_SPEC):
        self.spec = spec
        self.clf, self.Xtr, self.ytr = clf, Xtr, ytr
        self.scm, self.rng, self.target, self.tau = scm_train, rng, target, tau
        self.m, self.abduction = m_samples, abduction
        self.ranges = np.ptp(Xtr, axis=0)
        self.ranges[self.ranges == 0] = 1.0
        self.grid = build_grid(Xtr, spec)
        self.grid_pred = clf.predict(self.grid)
        self.tr_pred = clf.predict(Xtr)
        self.uniq = [np.unique(Xtr[:, j]) for j in range(Xtr.shape[1])]
        levels = [feature_levels(Xtr[:, spec.idx[a]], spec) for a in spec.actionable]
        self._levels = levels
        self.actions = np.array(list(itertools.product(*levels)), float)

    def allowed_actions(self, x: np.ndarray) -> np.ndarray:
        """Acciones que respetan direccion (solo subir / solo bajar) y factibilidad fisica."""
        sp = self.spec
        actions = self.actions
        if sp.include_status_quo:
            # Sin esto, una palanca continua discretizada por cuantiles (p. ej. DMI) no puede quedarse en
            # su valor actual y toda recomendacion la mueve; con esto existe la accion "solo enfriar".
            lv = [np.union1d(self._levels[j], [x[sp.idx[a]]]) for j, a in enumerate(sp.actionable)]
            actions = np.array(list(itertools.product(*lv)), float)
        mask = np.ones(len(actions), bool)
        for j, a in enumerate(sp.actionable):
            d, xv = sp.directions.get(a, "any"), x[sp.idx[a]]
            if d == "increase":
                mask &= actions[:, j] >= xv - 1e-12
            elif d == "decrease":
                mask &= actions[:, j] <= xv + 1e-12
        if sp.action_filter is not None:
            mask &= sp.action_filter(x, actions)
        return actions[mask]


def _actions_points(x, actions, spec: Spec = MILK_SPEC):
    P = np.repeat(x[None, :], len(actions), axis=0)
    for j, a in enumerate(spec.actionable):
        P[:, spec.idx[a]] = actions[:, j]
    return P


def cf_nun(x, ctx: CFContext):
    mask = (ctx.tr_pred == ctx.target) & (ctx.ytr == ctx.target)
    C = ctx.Xtr[mask]
    if len(C) == 0:
        return []
    c = norm_l1(x, C, ctx.ranges)
    return [C[np.argmin(c)]]


def cf_wachter_grid(x, ctx: CFContext):
    C = ctx.grid[ctx.grid_pred == ctx.target]
    if len(C) == 0:
        return []
    c = norm_l1(x, C, ctx.ranges)
    sp = (np.abs(C - x) > 1e-9).sum(1)
    order = np.lexsort((sp, c))
    return [C[order[0]]]


def cf_dice_random(x, ctx: CFContext, n_cand: int = 3000, k: int = 4):
    rng = ctx.rng
    D = len(ctx.spec.features)
    C = np.repeat(x[None, :], n_cand, axis=0)
    nfeat = rng.integers(1, 4, size=n_cand)
    for i in range(n_cand):
        feats = rng.choice(D, size=nfeat[i], replace=False)
        for j in feats:
            C[i, j] = rng.choice(ctx.uniq[j])
    C = np.unique(C, axis=0)
    C = C[ctx.clf.predict(C) == ctx.target]
    if len(C) == 0:
        return []
    c = norm_l1(x, C, ctx.ranges)
    pool = C[np.argsort(c)[:50]]
    chosen = [pool[0]]
    while len(chosen) < min(k, len(pool)):
        d = np.min([norm_l1(s, pool, ctx.ranges) for s in chosen], axis=0)
        d[[any(np.allclose(p, s) for s in chosen) for p in pool]] = -1
        chosen.append(pool[np.argmax(d)])
    return chosen


def cf_actionable_indep(x, ctx: CFContext):
    """Cambia solo features accionables (con direccion permitida) asumiendo independencia."""
    acts = ctx.allowed_actions(x)
    if len(acts) == 0:
        return []
    C = _actions_points(x, acts, ctx.spec)
    C = C[ctx.clf.predict(C) == ctx.target]
    if len(C) == 0:
        return []
    c = norm_l1(x, C, ctx.ranges)
    return [C[np.argmin(c)]]


def cf_causal_scm(x, ctx: CFContext, return_info: bool = False):
    """Recurso causal: min costo de accion s.a. P(f(x_SCF) = target) >= tau."""
    acts = ctx.allowed_actions(x)
    U = ctx.scm.abduct(x, ctx.m, ctx.rng, mode=ctx.abduction)
    if len(acts) == 0:
        return ([], {"p_valid_best": 0.0}) if return_info else []
    S = ctx.scm.counterfactual_batch(x, acts, U)            # (K, M, D)
    K, M, D = S.shape
    pred = ctx.clf.predict(S.reshape(-1, D)).reshape(K, M)
    pval = (pred == ctx.target).mean(1)
    cost = action_cost(x, _actions_points(x, acts, ctx.spec), ctx.ranges, ctx.spec)
    feas = np.where(pval >= ctx.tau)[0]
    if len(feas) == 0:
        return ([], {"p_valid_best": float(pval.max())}) if return_info else []
    order = feas[np.lexsort((-pval[feas], cost[feas]))]
    best = order[0]
    rep = representative(S[best], ctx.spec)
    # coordenadas accionables = la ACCION recomendada (intervenidas) o el valor factual (no intervenidas);
    # las respuestas estructurales de palancas no intervenidas no son acciones ni cuentan como costo.
    for j, a in enumerate(ctx.spec.actionable):
        k = ctx.spec.idx[a]
        rep[k] = acts[best, j] if abs(acts[best, j] - x[k]) > 1e-9 else x[k]
    info = {"p_valid_train_scm": float(pval[best]), "p_valid_best": float(pval.max())}
    return ([rep], info) if return_info else [rep]


CF_METHODS = {
    "NUN": cf_nun,
    "Wachter-grid": cf_wachter_grid,
    "DiCE-random": cf_dice_random,
    "Actionable-indep": cf_actionable_indep,
    "Causal-SCM": cf_causal_scm,
}


# ============================================================================
# Evaluacion de un contrafactual
# ============================================================================
class CFEvaluator:
    def __init__(self, clf, Xtr, world_scm: SCM, rng, target=2, m_eval=200, spec: Spec = MILK_SPEC):
        self.spec = spec
        self.clf, self.world, self.rng, self.target, self.m = clf, world_scm, rng, target, m_eval
        self.scaler = StandardScaler().fit(Xtr)
        Z = self.scaler.transform(Xtr)
        self.lof = LocalOutlierFactor(n_neighbors=min(10, len(Z) - 1), novelty=True).fit(Z)
        self.nn = NearestNeighbors(n_neighbors=1).fit(Z)
        self.ranges = np.ptp(Xtr, axis=0)
        self.ranges[self.ranges == 0] = 1.0

    def evaluate_many(self, x: np.ndarray, cfs: list, causal: bool = False) -> list[dict]:
        """Evalua varios CFs de una instancia con una sola llamada a predict.
        causal=True: los cambios en no accionables son consecuencias del SCM, no ediciones directas."""
        if not cfs:
            return []
        sp = self.spec
        C = np.asarray(cfs, float)
        a_idx = [sp.idx[f] for f in sp.actionable]
        na_idx = [sp.idx[f] for f in sp.non_actionable]
        # Implementacion: solo se ejecutan las acciones sobre variables accionables, en el SCM-mundo
        U = self.world.abduct(x, self.m, self.rng, mode="posterior")
        S = self.world.counterfactual_batch(x, C[:, a_idx], U)        # (n, M, D)
        n, M, D = S.shape
        pred = self.clf.predict(np.vstack([S.reshape(-1, D), C]))
        impl = (pred[: n * M].reshape(n, M) == self.target).mean(1)
        nominal = pred[n * M:] == self.target
        Z = self.scaler.transform(C)
        lof = -self.lof.score_samples(Z)
        lof_out = self.lof.predict(Z) == -1
        nn = self.nn.kneighbors(Z)[0][:, 0]
        out = []
        for i, cf in enumerate(C):
            diff = np.abs(cf - x) > 1e-9
            world_rep = representative(S[i], sp)
            incons = []
            for f in sp.non_actionable:
                j = sp.idx[f]
                if f in sp.binary:
                    incons.append(float(world_rep[j] != cf[j]))
                else:
                    incons.append(float(abs(world_rep[j] - cf[j]) / self.ranges[j] > 0.05))
            # violacion de direccion (p. ej. recomendar SUBIR el THI o dejar de aplicar una practica)
            dir_viol = 0.0
            for j, f in zip(a_idx, sp.actionable):
                d = sp.directions.get(f, "any")
                if (d == "increase" and cf[j] < x[j] - 1e-9) or (d == "decrease" and cf[j] > x[j] + 1e-9):
                    dir_viol = 1.0
            out.append({
                "nominal_valid": float(nominal[i]),
                "impl_validity": float(impl[i]),
                "distance_l1": float((np.abs(cf - x) / self.ranges).sum()),
                "action_cost": float((np.abs(cf[a_idx] - x[a_idx]) / self.ranges[a_idx]).sum()),
                "sparsity": int(diff.sum()),
                "n_nonactionable_edits": 0 if causal else int(diff[na_idx].sum()),
                "actionability_violation": 0.0 if causal else float(diff[na_idx].any()),
                "direction_violation": dir_viol,
                "causal_inconsistency": float(np.mean(incons)) if incons else 0.0,
                "lof_score": float(lof[i]),
                "lof_outlier": float(lof_out[i]),
                "nn_distance": float(nn[i]),
            })
        return out


# ============================================================================
# Ejecucion de un fold completo del experimento de contrafactuales
# ============================================================================
def run_cf_fold(clf_name, X, y, tr, te, world_scm, seed, methods=None, dag=None, tau=0.8,
                m_samples=100, abduction="posterior", m_eval=200, target=None, extra=None,
                spec: Spec = MILK_SPEC):
    """Devuelve una fila por (instancia explicada, metodo). Infactible => impl_validity = 0."""
    methods = methods or list(CF_METHODS)
    dag = dag if dag is not None else spec.dag
    target = spec.target if target is None else target
    rng = np.random.default_rng(seed)
    clf = make_classifier(clf_name, seed=SEED)
    if clf_name == "RF":
        clf.set_params(n_jobs=1)
    clf.fit(X[tr], y[tr])
    scm = SCM(dag, spec).fit(X[tr])
    ctx = CFContext(clf, X[tr], y[tr], scm, rng, target=target, tau=tau, m_samples=m_samples,
                    abduction=abduction, spec=spec)
    ev = CFEvaluator(clf, X[tr], world_scm, rng, target=target, m_eval=m_eval, spec=spec)
    pred_te = clf.predict(X[te])
    rows = []
    for i, p in zip(te, pred_te):
        if p == target:
            continue
        x = X[i]
        for mname in methods:
            info = {}
            if mname == "Causal-SCM":
                cfs, info = cf_causal_scm(x, ctx, return_info=True)
            else:
                cfs = CF_METHODS[mname](x, ctx)
            base = {"classifier": clf_name, "idx": int(i), "y_true": int(y[i]), "y_pred": int(p),
                    "method": mname, "feasible": float(len(cfs) > 0), "n_cfs": len(cfs), **info}
            if extra:
                base.update(extra)
            if not cfs:
                rows.append({**base, "impl_validity": 0.0})
                continue
            evs = ev.evaluate_many(x, cfs, causal=(mname == "Causal-SCM"))
            agg = {k: float(np.mean([e[k] for e in evs])) for k in evs[0]}
            rows.append({**base, **agg, "cf_first": cfs[0].round(4).tolist()})
    return rows
