"""10_noncircular_statistics.py
Estadistica, tablas y figuras de la evaluacion no circular (ronda 4). Lee results/val_*.parquet (script 09).

Unidad estadistica: perfil (Sample), vaca (Cow), granja (Farm). Todas las tasas son medias sobre unidades
independientes con IC 95% bootstrap; pruebas de Wilcoxon pareadas con Holm por familia y delta de Cliff.

Salidas:
  tables/table_19_noncircular_worlds        validez entregada y promesas falsas en el 2x2 de mundos
  tables/table_20_oracle_decomposition      descomposicion con el oraculo (existe / no existe recurso)
  tables/table_21_outcome_validity          exito segun el clasificador vs. segun el desenlace
  tables/table_22_confounding_attenuation   sensibilidad a confusion no medida (gamma)
  tables/table_23_within_cow_simulator      el simulador frente a transiciones reales de la misma vaca
  tables/table_24_real_transition_validity  validez observada (sin simulador) por metodo, caso Vaca
  tables/table_25_noncircular_tests         pruebas de hipotesis
  figures/fig_14_noncircular_worlds, fig_15_oracle_and_confounding, fig_16_real_transitions,
  figures/fig_17_outcome_validity
"""
import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

import utils as U
from cf_lib import SPECS, _parent_design
from cf_validation import GROUP_VARIANT

PROC = U.PROJECT_DIR / "data" / "processed"
METHODS = ["NUN", "Wachter-grid", "DiCE-random", "Actionable-indep", "Causal-SCM"]
UNCON = METHODS[:3]
MSHORT = dict(zip(METHODS, ["NUN", "Wach.", "DiCE", "Act.", "Causal"]))
COL = dict(zip(METHODS, ["#56B4E9", "#E69F00", "#CC79A7", "#009E73", "#0072B2"]))
CASES = {"heat_stress": "Cow", "milk_quality": "Sample", "scc_management": "Farm"}     # Vaca primero: caso principal
WORLDS = {"lin_expert": "Linear, expert DAG", "lin_dense": "Linear, saturated", "forest_expert": "Forest, expert DAG",
          "forest_dense": "Forest, saturated"}
WSHORT = {"lin_expert": "L/E", "lin_dense": "L/S", "forest_expert": "F/E", "forest_dense": "F/S"}
GAM = {"lin_expert": 0.0, "att25": 0.25, "att50": 0.5, "att75": 0.75, "att100": 1.0}
REF = "forest_dense"          # mundo de referencia: no comparte con el metodo ni forma funcional ni grafo
RNG = np.random.default_rng(U.SEED)
SUF = os.environ.get("VAL_SUFFIX", "")      # "_pilot" para probar con la salida de 09 --pilot


def load(name):
    cf = pd.read_parquet(U.RESULTS_DIR / f"val_{name}{SUF}.parquet")
    cf["nominal_valid"] = cf.nominal_valid.fillna(0.0)
    cf["direction_violation"] = cf.direction_violation.fillna(0.0)
    cf["action_cost"] = cf.action_cost.fillna(0.0)
    for w in list(WORLDS) + list(GAM):
        cf[f"fp__{w}"] = ((cf.nominal_valid == 1) & (cf[f"impl__{w}"] < 0.5)).astype(float)
        cf[f"exists__{w}"] = (cf[f"oracle_grid_impl__{w}"] >= 0.5).astype(float)
    for o in ("linear", "forest"):
        for w in ("lin_expert", REF):
            cf[f"ofp_{o}__{w}"] = ((cf.nominal_valid == 1) & (cf[f"out_{o}__{w}"] < 0.5)).astype(float)
            cf[f"illusory_{o}__{w}"] = ((cf[f"impl__{w}"] >= 0.5) & (cf[f"out_{o}__{w}"] < 0.5)).astype(float)
            cf[f"outadm_{o}__{w}"] = np.where(cf.direction_violation > 0, 0.0, cf[f"out_{o}__{w}"])
    cf["null_levers"] = ((cf.feasible == 1) & (cf.action_cost < 1e-9)).astype(float)
    return cf


def um(d, col):
    """Media por unidad independiente."""
    return d.groupby("unit")[col].mean()


def mci(v):
    v = np.asarray(v, float)
    lo, hi = U.bootstrap_ci(v)
    return float(np.mean(v)), lo, hi


def fmt(m, lo, hi, d=3):
    return f"{m:.{d}f} {U.fmt_ci(lo, hi, d)}"


def paired(cf, col, a, b, subset=None):
    d = cf if subset is None else cf[subset]
    B = d.pivot_table(index="unit", columns="method", values=col, aggfunc="mean")[[a, b]].dropna()
    x, z = B[a].values, B[b].values
    _, p = U.wilcoxon_safe(x, z)
    lo, hi = U.bootstrap_ci(x - z)
    dl = U.cliffs_delta(x, z)
    return {"n": len(B), "diff_mean": float((x - z).mean()), "diff_ci_lo": lo, "diff_ci_hi": hi, "p": p,
            "cliffs_delta": dl, "magnitude": U.cliffs_magnitude(dl)}


CF = {n: load(n) for n in CASES}

# =============================================================================
# Tabla 19: 2x2 de mundos
# =============================================================================
rows = []
for name, lab in CASES.items():
    for m in METHODS:
        d = CF[name][CF[name].method == m]
        r = {"dataset": name, "case": lab, "method": m, "n_units": d.unit.nunique()}
        for w in WORLDS:
            r[f"delivered_{w}"], r[f"delivered_{w}_lo"], r[f"delivered_{w}_hi"] = mci(um(d, f"impl__{w}"))
            r[f"false_promise_{w}"], r[f"false_promise_{w}_lo"], r[f"false_promise_{w}_hi"] = mci(um(d, f"fp__{w}"))
        rows.append(r)
T19 = pd.DataFrame(rows)
T19t = pd.DataFrame({"Case": T19.case, "Method": T19.method})
for kind, hib in (("delivered", True), ("false_promise", False)):
    for w in WORLDS:
        c = U.arrow(f"{'Deliv.' if hib else 'False pr.'} {WSHORT[w]}", hib)
        T19t[c] = T19[f"{kind}_{w}"].map(lambda v: f"{v:.3f}")
        U.highlight(T19t, T19[f"{kind}_{w}"], c, hib, groups=T19.dataset)
U.save_table(T19, "table_19_noncircular_worlds", df_tex=T19t)

# =============================================================================
# Tabla 20: descomposicion con el oraculo (mundo de referencia)
# =============================================================================
rows = []
for name, lab in CASES.items():
    cf = CF[name]
    ex = cf[f"exists__{REF}"] == 1
    share = um(cf[cf.method == "NUN"], f"exists__{REF}")
    for m in METHODS:
        d, de, dn = cf[cf.method == m], cf[(cf.method == m) & ex], cf[(cf.method == m) & ~ex]
        fp = d[d[f"fp__{REF}"] == 1]
        r = {"dataset": name, "case": lab, "method": m, "recourse_exists_share": share.mean(),
             "delivered_given_exists": um(de, f"impl__{REF}").mean(),
             "regret_given_exists": (um(de, f"oracle_impl__{REF}") - um(de, f"impl__{REF}")).mean(),
             "false_promise_given_exists": um(de, f"fp__{REF}").mean(),
             "false_promise_given_none": um(dn, f"fp__{REF}").mean() if len(dn) else np.nan,
             "abstains_given_none": (1 - um(dn, "feasible")).mean() if len(dn) else np.nan,
             "misses_given_exists": (1 - um(de, "feasible")).mean(),
             }
        # Composicion de las promesas falsas a nivel de unidad (misma agregacion que el resto de la tabla
        # y que la figura 15a-c): parte de la tasa media por unidad que corresponde a cada causa.
        exi_, fp_ = d[f"exists__{REF}"], d[f"fp__{REF}"]
        tot = um(d, f"fp__{REF}").mean()
        part = lambda mask: um(d.assign(v=fp_ * mask.astype(float)), "v").mean() / tot if tot > 0 else np.nan
        r["fp_share_no_recourse"] = part(exi_ == 0)
        r["fp_share_direction_violation"] = part((exi_ == 1) & (d.direction_violation > 0))
        r["fp_share_null_levers"] = part((exi_ == 1) & (d.direction_violation == 0) & (d.null_levers == 1))
        r["fp_share_wrong_action"] = (1 - r["fp_share_no_recourse"] - r["fp_share_direction_violation"] - r["fp_share_null_levers"]
                                      if tot > 0 else np.nan)
        rows.append(r)
T20 = pd.DataFrame(rows)
f2 = lambda v: "--" if pd.isna(v) else f"{v:.3f}"
T20t = pd.DataFrame({"Case": T20.case, "Method": T20.method, "Recourse exists": T20.recourse_exists_share.map(f2)})
for c, src, hib in (("Delivered | exists", "delivered_given_exists", True), ("Regret | exists", "regret_given_exists", False),
                    ("False pr. | exists", "false_promise_given_exists", False),
                    ("False pr. | none", "false_promise_given_none", False),
                    ("Abstains | none", "abstains_given_none", True)):
    T20t[U.arrow(c, hib)] = T20[src].map(f2)
    U.highlight(T20t, T20[src], U.arrow(c, hib), hib, groups=T20.dataset)
for c, src in (("FP: no recourse", "fp_share_no_recourse"), ("FP: wrong direction", "fp_share_direction_violation"),
               ("FP: no lever moved", "fp_share_null_levers"), ("FP: wrong action", "fp_share_wrong_action")):
    T20t[c] = T20[src].map(f2)
U.save_table(T20, "table_20_oracle_decomposition", df_tex=T20t)

# =============================================================================
# Tabla 21: clasificador vs. desenlace
# =============================================================================
rows = []
for name, lab in CASES.items():
    for m in METHODS:
        d = CF[name][CF[name].method == m]
        r = {"dataset": name, "case": lab, "method": m, "direction_violation": um(d, "direction_violation").mean()}
        for w in ("lin_expert", REF):
            r[f"model_validity_{w}"] = um(d, f"impl__{w}").mean()
            for o in ("linear", "forest"):
                r[f"outcome_validity_{o}_{w}"], r[f"outcome_validity_{o}_{w}_lo"], r[f"outcome_validity_{o}_{w}_hi"] = \
                    mci(um(d, f"out_{o}__{w}"))
                r[f"outcome_validity_admissible_{o}_{w}"] = um(d, f"outadm_{o}__{w}").mean()
                r[f"illusory_recourse_{o}_{w}"] = um(d, f"illusory_{o}__{w}").mean()
                r[f"outcome_false_promise_{o}_{w}"] = um(d, f"ofp_{o}__{w}").mean()
        rows.append(r)
T21 = pd.DataFrame(rows)
T21t = pd.DataFrame({"Case": T21.case, "Method": T21.method})
for c, src, hib in (("Model validity", f"model_validity_{REF}", True),
                    ("Outcome validity (linear)", f"outcome_validity_linear_{REF}", True),
                    ("Outcome validity (forest)", f"outcome_validity_forest_{REF}", True),
                    ("Admissible outcome validity (forest)", f"outcome_validity_admissible_forest_{REF}", True),
                    ("Illusory recourse (forest)", f"illusory_recourse_forest_{REF}", False),
                    ("Outcome false promise (forest)", f"outcome_false_promise_forest_{REF}", False),
                    ("Direction violations", "direction_violation", False)):
    T21t[U.arrow(c, hib)] = T21[src].map(lambda v: f"{v:.3f}")
    U.highlight(T21t, T21[src], U.arrow(c, hib), hib, groups=T21.dataset)
U.save_table(T21, "table_21_outcome_validity", df_tex=T21t)

# =============================================================================
# Tabla 22: atenuacion (confusion no medida)
# =============================================================================
rows = []
for name, lab in CASES.items():
    cf = CF[name]
    for w, g in GAM.items():
        r = {"dataset": name, "case": lab, "gamma": g}
        for m in METHODS:
            d = cf[cf.method == m]
            r[f"delivered_{m}"] = um(d, f"impl__{w}").mean()
            r[f"false_promise_{m}"] = um(d, f"fp__{w}").mean()
        t = paired(cf, f"impl__{w}", "Causal-SCM", "Actionable-indep")
        r.update({"diff_causal_vs_actionable": t["diff_mean"], "diff_ci_lo": t["diff_ci_lo"], "diff_ci_hi": t["diff_ci_hi"], "p": t["p"]})
        t2 = paired(cf, f"fp__{w}", "Causal-SCM", "Actionable-indep")
        r.update({"fp_diff_causal_vs_actionable": t2["diff_mean"], "fp_diff_ci_lo": t2["diff_ci_lo"],
                  "fp_diff_ci_hi": t2["diff_ci_hi"], "fp_p": t2["p"]})
        rows.append(r)
T22 = pd.DataFrame(rows)
T22["p_holm"] = np.concatenate([U.holm(g.p.values) for _, g in T22.groupby("dataset", sort=False)])
T22["fp_p_holm"] = np.concatenate([U.holm(g.fp_p.values) for _, g in T22.groupby("dataset", sort=False)])
T22t = pd.DataFrame({"Case": T22.case, "$\\gamma$": T22.gamma.map(lambda v: f"{v:.2f}"),
                     "Deliv. best unconstr.": T22[[f"delivered_{m}" for m in UNCON]].max(1).map(lambda v: f"{v:.3f}"),
                     "Deliv. Actionable": T22["delivered_Actionable-indep"].map(lambda v: f"{v:.3f}"),
                     "Deliv. Causal": T22["delivered_Causal-SCM"].map(lambda v: f"{v:.3f}"),
                     "Causal $-$ Act. [95% CI]": [f"{a:+.3f} {U.fmt_ci(l, h)}" for a, l, h in zip(T22.diff_causal_vs_actionable, T22.diff_ci_lo, T22.diff_ci_hi)],
                     "p (Holm)": T22.p_holm.map(U.fmt_p),
                     "False pr. Actionable": T22["false_promise_Actionable-indep"].map(lambda v: f"{v:.3f}"),
                     "False pr. Causal": T22["false_promise_Causal-SCM"].map(lambda v: f"{v:.3f}"),
                     "FP diff. p (Holm)": T22.fp_p_holm.map(U.fmt_p)})
U.save_table(T22, "table_22_confounding_attenuation", df_tex=T22t)

# =============================================================================
# Tabla 23: simulador vs. transiciones reales de la misma vaca
# =============================================================================
TR = pd.read_parquet(U.RESULTS_DIR / f"val_within_cow_transitions{SUF}.parquet")
dh = pd.read_csv(PROC / "heat_stress_ml_ready.csv")
MED = ["rectal_temp_c", "resp_bpm", "pulse_bpm", "dmi_kg"]
MLAB = {"rectal_temp_c": "Rectal temp.", "resp_bpm": "Respiration", "pulse_bpm": "Pulse", "dmi_kg": "DMI"}
rng_ = {m: float(np.ptp(dh[m])) for m in MED}
TR["err"] = (TR.pred - TR.obs).abs() / TR.mediator.map(rng_)
TR["dp"], TR["do"] = TR.pred - TR.factual, TR.obs - TR.factual
COOL = TR[TR.thi_to < TR.thi_from]
PRED = ["No change (independence)", "lin_expert, no abduction", "lin_expert", "lin_expert, cow-level abduction", "lin_dense",
        "forest_expert", "forest_dense"]
PLAB = {"No change (independence)": "No change (independence)", "lin_expert, no abduction": "Linear SCM, no abduction",
        "lin_expert": "Linear SCM, expert DAG", "lin_expert, cow-level abduction": "Linear SCM, cow-level abduction", "lin_dense": "Linear SCM, saturated", "forest_expert": "Forest SCM, expert DAG",
        "forest_dense": "Forest SCM, saturated"}


def slope_ci(d):
    """Fraccion realizada del efecto predicho: sum(dp*do)/sum(dp^2); IC 95% bootstrap por vaca."""
    g = d.assign(a=d.dp * d["do"], b=d.dp ** 2).groupby("cow_id")[["a", "b"]].sum()
    if g.b.sum() < 1e-12:
        return np.nan, np.nan, np.nan
    idx = RNG.integers(0, len(g), size=(U.N_BOOTSTRAP, len(g)))
    bs = g.a.values[idx].sum(1) / g.b.values[idx].sum(1)
    return float(g.a.sum() / g.b.sum()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


E = COOL.groupby(["predictor", "mediator", "cow_id"]).err.mean().unstack("mediator")
E["mean"] = E[MED].mean(1)
rows, fam = [], []
for p in PRED:
    r = {"predictor": PLAB[p], "transitions": "cooling (THI decrease)", "n_cows": int(E.loc[p].shape[0])}
    for m in MED + ["mean"]:
        r[f"nmae_{m}"], r[f"nmae_{m}_lo"], r[f"nmae_{m}_hi"] = mci(E.loc[p][m].values)
    for m in MED:
        r[f"realised_fraction_{m}"], r[f"realised_fraction_{m}_lo"], r[f"realised_fraction_{m}_hi"] = \
            slope_ci(COOL[(COOL.predictor == p) & (COOL.mediator == m)])
    if p != PRED[0]:
        x, z = E.loc[p]["mean"], E.loc[PRED[0]]["mean"]
        r["p_vs_no_change"] = U.wilcoxon_safe(x.values, z.loc[x.index].values)[1]
        r["cliffs_delta_vs_no_change"] = U.cliffs_delta(x.values, z.loc[x.index].values)
    rows.append(r)
T23 = pd.DataFrame(rows)
T23.loc[T23.p_vs_no_change.notna(), "p_holm_vs_no_change"] = U.holm(T23.p_vs_no_change.dropna().values)
T23t = pd.DataFrame({"Predictor of the post-cooling state": T23.predictor})
for m in MED + ["mean"]:
    c = U.arrow(f"nMAE {MLAB.get(m, 'mean')}", False)
    T23t[c] = T23[f"nmae_{m}"].map(lambda v: f"{v:.3f}")
    U.highlight(T23t, T23[f"nmae_{m}"], c, False)
T23t["Realised fraction RT / RR / PR / DMI"] = [
    "--" if pd.isna(r["realised_fraction_rectal_temp_c"]) else " / ".join(f"{r[f'realised_fraction_{m}']:.2f}" for m in MED)
    for _, r in T23.iterrows()]
T23t["p vs. no change (Holm)"] = T23.p_holm_vs_no_change.map(U.fmt_p)
U.save_table(T23, "table_23_within_cow_simulator", df_tex=T23t)

# =============================================================================
# Tabla 24: validez observada en transiciones reales (Vaca)
# =============================================================================
def add_obs(c):
    c = c.copy()
    c["real_fp"] = ((c.nominal_valid == 1) & (c.obs_real < 0.5)).astype(float)
    c["obs_fp"] = ((c.nominal_valid == 1) & (c.obs_model < 0.5)).astype(float)
    c["sim_gap_lin"] = (c["impl__lin_expert"] - c.obs_model).abs()
    c["sim_gap_ref"] = (c[f"impl__{REF}"] - c.obs_model).abs()
    return c


CF["heat_stress"] = add_obs(CF["heat_stress"])
M24 = METHODS + [GROUP_VARIANT]
rows = []
cow = CF["heat_stress"]
for m in M24:
    d = cow[cow.method == m]
    ex = d[(d.obs_exact == 1) & (d.feasible == 1)]
    r = {"method": m, "n_cows": d.unit.nunique(), "feasible": um(d, "feasible").mean(),
         "lever_part_thi_only": float(d[d.feasible == 1].obs_exact.mean()),
         "promised": um(d, "nominal_valid").mean(), "simulated_lin_expert": um(d, "impl__lin_expert").mean(),
         f"simulated_{REF}": um(d, f"impl__{REF}").mean(),
         "false_promise_lin_expert": um(d, "fp__lin_expert").mean(), f"false_promise_{REF}": um(d, f"fp__{REF}").mean()}
    r["observed_model_validity"], r["observed_model_validity_lo"], r["observed_model_validity_hi"] = mci(um(d, "obs_model"))
    r["observed_real_validity"], r["observed_real_validity_lo"], r["observed_real_validity_hi"] = mci(um(d, "obs_real"))
    r["observed_false_promise_model"] = um(d, "obs_fp").mean()
    r["observed_false_promise_real"] = um(d, "real_fp").mean()
    r["exact_n_cows"] = ex.unit.nunique()
    r["exact_method_own_predicted_validity"] = um(ex, "p_valid_train_scm").mean() if m.startswith("Causal-SCM") and len(ex) else np.nan
    r["exact_observed_model_validity"] = um(ex, "obs_model").mean() if len(ex) else np.nan
    r["exact_observed_real_validity"] = um(ex, "obs_real").mean() if len(ex) else np.nan
    r["exact_simulated_lin_expert"] = um(ex, "impl__lin_expert").mean() if len(ex) else np.nan
    r[f"exact_simulated_{REF}"] = um(ex, f"impl__{REF}").mean() if len(ex) else np.nan
    r["exact_simulator_abs_gap_lin_expert"] = um(ex, "sim_gap_lin").mean() if len(ex) else np.nan
    r[f"exact_simulator_abs_gap_{REF}"] = um(ex, "sim_gap_ref").mean() if len(ex) else np.nan
    rows.append(r)
T24 = pd.DataFrame(rows)
T24t = pd.DataFrame({"Method": T24.method, "Offered": T24.feasible.map(lambda v: f"{v:.3f}"),
                     "Simulated (L/E)": T24.simulated_lin_expert.map(lambda v: f"{v:.3f}"),
                     "Simulated (F/S)": T24[f"simulated_{REF}"].map(lambda v: f"{v:.3f}")})
for c, src, hib in (("Observed, classifier", "observed_model_validity", True), ("Observed, real SCC", "observed_real_validity", True),
                    ("Observed false pr., classifier", "observed_false_promise_model", False),
                    ("Observed false pr., real SCC", "observed_false_promise_real", False)):
    T24t[U.arrow(c, hib)] = T24[src].map(lambda v: f"{v:.3f}")
    U.highlight(T24t, T24[src], U.arrow(c, hib), hib)
T24t["Cooling-only share"] = T24.lever_part_thi_only.map(lambda v: f"{v:.3f}")
T24t["Cooling-only: simulated (L/E) / observed"] = [f"{a:.3f} / {b:.3f}" if pd.notna(a) else "--"
                                                    for a, b in zip(T24.exact_simulated_lin_expert, T24.exact_observed_model_validity)]
U.save_table(T24, "table_24_real_transition_validity", df_tex=T24t)

# =============================================================================
# Tabla 25: pruebas
# =============================================================================
FAM = [(f"impl__{REF}", "Delivered validity, forest/saturated world", True, None),
       (f"fp__{REF}", "False promise, forest/saturated world", False, None),
       (f"impl__{REF}", "Delivered validity | recourse exists (oracle)", True, f"exists__{REF}"),
       (f"fp__{REF}", "False promise | recourse exists (oracle)", False, f"exists__{REF}"),
       (f"out_forest__{REF}", "Outcome validity (forest outcome)", True, None),
       (f"outadm_forest__{REF}", "Admissible outcome validity (forest outcome)", True, None),
       ("obs_model", "Observed validity, classifier (real transitions)", True, None),
       ("obs_real", "Observed validity, real SCC (real transitions)", True, None),
       ("real_fp", "Observed false promise, real SCC (real transitions)", False, None)]
rows = []
for name, lab, cf in [(n, l, CF[n]) for n, l in CASES.items()]:
    for col, famname, hib, sub in FAM:
        if col not in cf:
            continue
        mask = None if sub is None else (cf[sub] == 1)
        fam = []
        for a, others in (("Causal-SCM", METHODS[:4]), ("Actionable-indep", UNCON)):
            for b in others:
                t = paired(cf, col, a, b, mask)
                fam.append({"dataset": name, "case": lab, "family": famname, "comparison": f"{a} vs {b}", **t,
                            "favours_first": bool((t["diff_mean"] > 0) == hib)})
        fam = pd.DataFrame(fam)
        fam["p_holm"] = U.holm(fam.p.values)
        rows += fam.to_dict("records")
# Exploratorio (ronda 5): abduccion del efecto persistente de la vaca frente a la del residuo de la replica
fam = []
for col, famname, hib in ((f"impl__{REF}", "Delivered validity, forest/saturated world", True), (f"fp__{REF}", "False promise, forest/saturated world", False),
                          ("obs_model", "Observed validity, classifier (real transitions)", True),
                          ("obs_real", "Observed validity, real SCC (real transitions)", True),
                          ("real_fp", "Observed false promise, real SCC (real transitions)", False)):
    for b in ("Causal-SCM", "Actionable-indep"):
        t = paired(CF["heat_stress"], col, GROUP_VARIANT, b)
        fam.append({"dataset": "heat_stress", "case": "Cow", "family": "Exploratory: " + famname, "comparison": f"{GROUP_VARIANT} vs {b}",
                    **t, "favours_first": bool((t["diff_mean"] > 0) == hib)})
fam = pd.DataFrame(fam)
fam["p_holm"] = U.holm(fam.p.values)
rows += fam.to_dict("records")
T25 = pd.DataFrame(rows)
T25["conclusion"] = ["n.s." if p >= U.ALPHA else ("favours first" if f else "favours second") for p, f in zip(T25.p_holm, T25.favours_first)]
T25t = pd.DataFrame({"Case": T25.case, "Family": T25.family, "Comparison": T25.comparison, "n": T25.n,
                     "Diff. [95% CI]": [f"{a:+.3f} {U.fmt_ci(l, h)}" for a, l, h in zip(T25.diff_mean, T25.diff_ci_lo, T25.diff_ci_hi)],
                     "Cliff's $\\delta$": T25.cliffs_delta.map(lambda v: f"{v:+.3f}"), "p (Holm)": T25.p_holm.map(U.fmt_p),
                     "Conclusion": T25.conclusion})
U.save_table(T25, "table_25_noncircular_tests", df_tex=T25t)

# =============================================================================
# Tabla 26: cobertura y precision (la abstencion cuenta aparte) e indice de acoplamiento
# =============================================================================
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PATHS = {"milk_quality": (U.DATA_PATH, None), "heat_stress": (PROC / "heat_stress_ml_ready.csv", "cow_id"),
         "scc_management": (PROC / "scc_management_ml_ready.csv", None)}


def coupling_index(name):
    """Cuanto explican las palancas a sus hijos en el DAG, fuera de muestra (CV agrupada por unidad).
    Por nodo: habilidad con todos los padres menos habilidad sin los padres-palanca
    (R2 en continuas; 1 - logloss/logloss nulo en binarias). Se calcula sin ejecutar ningun recurso."""
    spec = SPECS[name]
    path, gcol = PATHS[name]
    d = pd.read_csv(path)
    X = d[spec.features].to_numpy(float)
    g = d[gcol].to_numpy() if gcol else np.arange(len(d))
    out = []
    for node, pa in spec.dag.items():
        lev = [q for q in pa if q in spec.actionable]
        if not lev:
            continue
        yv = X[:, spec.idx[node]]
        P = {f: X[:, spec.idx[f]] for f in spec.features}
        sk = []
        for parents in (pa, [q for q in pa if q not in lev]):
            Z = _parent_design(parents, P, spec) if parents else np.zeros((len(yv), 0))
            num = den = 0.0
            for tr, te in GroupKFold(n_splits=5).split(Z, yv, g):
                if node in spec.binary:
                    p0 = np.clip(yv[tr].mean(), 1e-3, 1 - 1e-3)
                    if Z.shape[1] and len(np.unique(yv[tr])) > 1:
                        pr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(Z[tr], yv[tr]).predict_proba(Z[te])[:, 1]
                    else:
                        pr = np.full(len(te), p0)
                    pr = np.clip(pr, 1e-3, 1 - 1e-3)
                    num += -(yv[te] * np.log(pr) + (1 - yv[te]) * np.log(1 - pr)).sum()
                    den += -(yv[te] * np.log(p0) + (1 - yv[te]) * np.log(1 - p0)).sum()
                else:
                    mu = yv[tr].mean()
                    pr = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(Z[tr], yv[tr]).predict(Z[te]) if Z.shape[1] else np.full(len(te), mu)
                    num += ((yv[te] - pr) ** 2).sum(); den += ((yv[te] - mu) ** 2).sum()
            sk.append(1 - num / den)
        out.append({"node": node, "delta_skill": sk[0] - sk[1], "actionable": node in spec.actionable})
    return pd.DataFrame(out)


rows = []
for name, lab in CASES.items():
    cf = CF[name]
    cpl = coupling_index(name)
    gain = paired(cf, f"impl__{REF}", "Causal-SCM", "Actionable-indep")
    for m in METHODS:
        d = cf[cf.method == m]
        f = d[d.feasible == 1]
        rows.append({"dataset": name, "case": lab, "method": m, "coverage": um(d, "feasible").mean(),
                     "precision_classifier": um(f, f"impl__{REF}").mean(), "precision_outcome": um(f, f"out_forest__{REF}").mean(),
                     "false_promise_given_offered": um(f, f"fp__{REF}").mean(),
                     "unconditional_classifier": um(d, f"impl__{REF}").mean(), "unconditional_outcome": um(d, f"out_forest__{REF}").mean(),
                     "coupling_index": cpl.delta_skill.mean(), "coupling_nodes": len(cpl),
                     # la propagacion solo puede abrir recursos a traves de hijos que NO se pueden fijar directamente
                     "coupling_nonactionable": cpl[~cpl.actionable].delta_skill.mean() if (~cpl.actionable).any() else 0.0,
                     "coupling_nonactionable_nodes": int((~cpl.actionable).sum()),
                     "coupling_max_node": cpl.loc[cpl.delta_skill.idxmax(), "node"], "coupling_max": cpl.delta_skill.max(),
                     "causal_gain_vs_actionable": gain["diff_mean"], "causal_gain_ci_lo": gain["diff_ci_lo"], "causal_gain_ci_hi": gain["diff_ci_hi"]})
T26 = pd.DataFrame(rows)
T26t = pd.DataFrame({"Case": T26.case, "Coupling (all children)": T26.coupling_index.map(lambda v: f"{v:.3f}"),
                     "Coupling (non-actionable children)": T26.coupling_nonactionable.map(lambda v: f"{v:.3f}"),
                     "Causal gain [95% CI]": [f"{a:+.3f} {U.fmt_ci(l, h)}" for a, l, h in zip(T26.causal_gain_vs_actionable, T26.causal_gain_ci_lo, T26.causal_gain_ci_hi)],
                     "Method": T26.method})
for c, src, hib in (("Coverage", "coverage", None), ("Precision, classifier", "precision_classifier", True),
                    ("Precision, outcome", "precision_outcome", True), ("False pr. | offered", "false_promise_given_offered", False)):
    T26t[U.arrow(c, hib)] = T26[src].map(lambda v: f"{v:.3f}")
    if hib is not None:
        U.highlight(T26t, T26[src], U.arrow(c, hib), hib, groups=T26.dataset)
U.save_table(T26, "table_26_coverage_and_coupling", df_tex=T26t)

# =============================================================================
# Figura 14: 2x2 de mundos
# =============================================================================
WM = dict(zip(WORLDS, [("o", False), ("s", False), ("o", True), ("s", True)]))
fig, axes = plt.subplots(2, 3, figsize=(U.DOUBLE_COL, 4.9), sharey="row")
for j, (name, lab) in enumerate(CASES.items()):
    t = T19[T19.dataset == name].set_index("method")
    for r, (kind, ylab) in enumerate((("delivered", "Delivered validity"), ("false_promise", "False-promise rate"))):
        ax = axes[r, j]
        for i, m in enumerate(METHODS):
            for k, w in enumerate(WORLDS):
                xx = i + (k - 1.5) * 0.17
                v, lo, hi = t.loc[m, f"{kind}_{w}"], t.loc[m, f"{kind}_{w}_lo"], t.loc[m, f"{kind}_{w}_hi"]
                mk, fill = WM[w]
                ax.plot([xx, xx], [lo, hi], color=COL[m], lw=0.9, zorder=1)
                ax.scatter(xx, v, marker=mk, s=22, lw=1.1, zorder=2, edgecolor=COL[m], facecolor=COL[m] if fill else "white")
        ax.set_xticks(range(len(METHODS)), [MSHORT[m] for m in METHODS], fontsize=8)
        ax.set_ylim(-0.04, 1.06)
        if j == 0:
            ax.set_ylabel(ylab)
        if r == 0:
            ax.text(0.5, 1.03, lab, transform=ax.transAxes, ha="center", fontsize=9, style="italic")
        U.add_panel_label(ax, "abcdef"[r * 3 + j], x=-0.2 if j else -0.38)
h = [Line2D([], [], marker=WM[w][0], ls="", ms=5, mec="k", mfc="k" if WM[w][1] else "white") for w in WORLDS]
l = list(WORLDS.values())
fig.legend(h, l, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.03), handletextpad=0.2, columnspacing=1.2)
fig.tight_layout(rect=(0, 0.03, 1, 1))
U.save_figure(fig, "fig_14_noncircular_worlds")

# =============================================================================
# Figura 15: descomposicion con el oraculo y atenuacion
# =============================================================================
fig, axes = plt.subplots(2, 3, figsize=(U.DOUBLE_COL, 4.9), sharey="row")
for j, (name, lab) in enumerate(CASES.items()):
    cf = CF[name]
    ax = axes[0, j]
    for i, m in enumerate(METHODS):
        d = cf[cf.method == m]
        none = um(d.assign(v=d[f"fp__{REF}"] * (1 - d[f"exists__{REF}"])), "v").mean()
        exi = um(d.assign(v=d[f"fp__{REF}"] * d[f"exists__{REF}"]), "v").mean()
        ax.bar(i, none, color=COL[m], width=0.7, label="No actionable recourse exists" if i == 0 else None)
        ax.bar(i, exi, bottom=none, color="white", edgecolor=COL[m], hatch="////", width=0.7, lw=0.9,
               label="Recourse exists, not delivered" if i == 0 else None)
    ax.set_xticks(range(len(METHODS)), [MSHORT[m] for m in METHODS], fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.text(0.5, 1.03, lab, transform=ax.transAxes, ha="center", fontsize=9, style="italic")
    if j == 0:
        ax.set_ylabel("False-promise rate")
    U.add_panel_label(ax, "abc"[j], x=-0.2 if j else -0.38)
    ax = axes[1, j]
    t = T22[T22.dataset == name]
    for m in METHODS:
        ax.plot(t.gamma, t[f"delivered_{m}"], color=COL[m], marker="o", ms=3.5, lw=1.3, label=m if j == 0 else None)
    ax.set_xlabel(r"Confounded share $\gamma$")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", ".25", ".5", ".75", "1"])
    ax.set_ylim(-0.04, 1.06)
    if j == 0:
        ax.set_ylabel("Delivered validity")
    U.add_panel_label(ax, "def"[j], x=-0.2 if j else -0.38)
h = [Patch(facecolor="#777777", edgecolor="k"), Patch(facecolor="white", edgecolor="k", hatch="////")]
axes[0, 2].legend(h, ["No actionable recourse exists", "Recourse exists, not delivered"], fontsize=8, loc="upper right")
hm, lm = axes[1, 0].get_legend_handles_labels()
fig.legend(hm, lm, loc="lower center", ncol=5, fontsize=8, bbox_to_anchor=(0.5, -0.03), columnspacing=1.2, handletextpad=0.3)
fig.tight_layout(rect=(0, 0.03, 1, 1))
U.save_figure(fig, "fig_15_oracle_and_confounding")

# =============================================================================
# Figura 16: transiciones reales (Vaca)
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(U.DOUBLE_COL, 3.3), gridspec_kw={"width_ratios": [1, 0.8, 1.7]})
ax = axes[0]
MC = dict(zip(MED, ["#D55E00", "#0072B2", "#009E73", "#CC79A7"]))
d = COOL[COOL.predictor == "lin_expert"].assign(dpn=lambda x: x.dp / x.mediator.map(rng_), don=lambda x: x["do"] / x.mediator.map(rng_))
g = d.groupby(["mediator", "cow_id", "thi_from", "thi_to"])[["dpn", "don"]].mean().reset_index()
for m in MED:
    s = g[g.mediator == m]
    ax.scatter(s.dpn, s.don, s=7, color=MC[m], alpha=0.55, lw=0, label=MLAB[m])
lim = [min(g.dpn.min(), g.don.min()) - 0.03, max(g.dpn.max(), g.don.max()) + 0.03]
ax.plot(lim, lim, color="k", lw=0.8, ls="--")
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("Predicted change"); ax.set_ylabel("Observed change (same cow)")
ax.legend(fontsize=8, loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, handletextpad=0.0, columnspacing=0.6, labelspacing=0.2, markerscale=1.5)
U.add_panel_label(ax, "a", x=-0.45, y=1.2)
ax = axes[1]
order = PRED
vals = [T23.set_index("predictor").loc[PLAB[p]] for p in order]
ax.barh(range(len(order)), [v["nmae_mean"] for v in vals],
        xerr=[[v["nmae_mean"] - v["nmae_mean_lo"] for v in vals], [v["nmae_mean_hi"] - v["nmae_mean"] for v in vals]],
        color=["#999999", "#BBBBBB", "#0072B2", "#D55E00", "#56B4E9", "#009E73", "#8FD3B6"], error_kw={"lw": 0.9}, height=0.7)
ax.set_yticks(range(len(order)), ["No change", "Linear, no abd.", "Linear, replicate abd.", "Linear, cow abd.", "Linear, satur.", "Forest, expert", "Forest, satur."], fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("Normalised MAE")
U.add_panel_label(ax, "b", x=-0.95, y=1.2)
ax = axes[2]
t = T24.set_index("method")
bars = [("simulated_lin_expert", "Simulated (linear, expert)", "white", "//"), (f"simulated_{REF}", "Simulated (forest, saturated)", "white", ".."),
        ("observed_model_validity", "Observed, classifier", None, None), ("observed_real_validity", "Observed, real SCC", None, "xx")]
for k, (c, lab_, face, hatch) in enumerate(bars):
    for i, m in enumerate(METHODS):
        ax.bar(i + (k - 1.5) * 0.2, t.loc[m, c], width=0.2, color=(COL[m] if face is None else face), edgecolor=COL[m] if face else "k",
               hatch=hatch, lw=0.6, alpha=0.6 if (face is None and hatch) else 1.0, label=lab_ if i == 4 else None)
ax.set_xticks(range(len(METHODS)), [MSHORT[m] for m in METHODS], fontsize=8)
ax.set_ylim(0, 1.0); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_ylabel("Delivered validity")
ax.legend(fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, columnspacing=0.8, handletextpad=0.3, handlelength=1.2)
U.add_panel_label(ax, "c", x=-0.2, y=1.2)
fig.tight_layout(w_pad=1.2)
U.save_figure(fig, "fig_16_real_transitions")

# =============================================================================
# Figura 17: clasificador vs. desenlace
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(U.DOUBLE_COL, 2.6), sharey=True)
for j, (name, lab) in enumerate(CASES.items()):
    ax = axes[j]
    t = T21[T21.dataset == name].set_index("method")
    for i, m in enumerate(METHODS):
        a, b, c = t.loc[m, f"model_validity_{REF}"], t.loc[m, f"outcome_validity_forest_{REF}"], t.loc[m, f"outcome_validity_admissible_forest_{REF}"]
        ax.plot([i - 0.2, i, i + 0.2], [a, b, c], color="#888888", lw=0.8, zorder=1)
        ax.scatter(i - 0.2, a, marker="o", s=26, facecolor="white", edgecolor=COL[m], lw=1.2, zorder=2, label="Decision model flips" if i == 0 else None)
        ax.scatter(i, b, marker="s", s=26, color=COL[m], zorder=3, label="Outcome reaches target" if i == 0 else None)
        ax.scatter(i + 0.2, c, marker="^", s=26, color=COL[m], zorder=3, label="... through admissible actions" if i == 0 else None)
    ax.set_xticks(range(len(METHODS)), [MSHORT[m] for m in METHODS], fontsize=8)
    ax.set_ylim(-0.04, 1.06)
    ax.text(0.5, 1.03, lab, transform=ax.transAxes, ha="center", fontsize=9, style="italic")
    if j == 0:
        ax.set_ylabel("Delivered validity")
    U.add_panel_label(ax, "abc"[j], x=-0.2 if j else -0.38)
h = [Line2D([], [], marker=mk, ls="", ms=5, mec="k", mfc=fc) for mk, fc in (("o", "white"), ("s", "k"), ("^", "k"))]
fig.legend(h, ["Decision model predicts the target", "Outcome reaches the target", "Outcome reaches the target through admissible actions"],
           fontsize=8, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06), handletextpad=0.2, columnspacing=1.2)
fig.tight_layout(rect=(0, 0.04, 1, 1))
U.save_figure(fig, "fig_17_outcome_validity")

pd.set_option("display.width", 250)
print(T19[["case", "method"] + [f"delivered_{w}" for w in WORLDS] + [f"false_promise_{w}" for w in WORLDS]].round(3).to_string())
print(T20.drop(columns=["dataset"]).round(3).to_string())
print(T21[["case", "method", f"model_validity_{REF}", f"outcome_validity_linear_{REF}", f"outcome_validity_forest_{REF}",
           f"outcome_validity_admissible_forest_{REF}", f"illusory_recourse_forest_{REF}", "direction_violation"]].round(3).to_string())
print(T22[["case", "gamma", "delivered_Actionable-indep", "delivered_Causal-SCM", "diff_causal_vs_actionable", "diff_ci_lo", "diff_ci_hi",
           "p_holm", "false_promise_Actionable-indep", "false_promise_Causal-SCM", "fp_p_holm"]].round(4).to_string())
print(T23[["predictor"] + [f"nmae_{m}" for m in MED + ["mean"]] + [f"realised_fraction_{m}" for m in MED] + ["p_holm_vs_no_change"]].round(3).to_string())
print(T24.round(3).T.to_string())
print(T26.round(3).to_string())
print(T25[["case", "family", "comparison", "n", "diff_mean", "diff_ci_lo", "diff_ci_hi", "cliffs_delta", "p_holm", "conclusion"]].round(4).to_string())
print("10 OK")
