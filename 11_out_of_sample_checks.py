"""11_out_of_sample_checks.py
Ronda 6 de revision: comprobaciones fuera de muestra de los tres hallazgos exploratorios de la ronda 5.

(A) Abduccion a nivel de vaca, grupos geneticos no vistos (caso Vaca; 5 grupos de 10 vacas).
    Las 10 particiones "3 grupos de desarrollo / 2 de evaluacion". En cada una:
      A1. el nivel de abduccion (ninguna / replica / vaca) se ELIGE solo con los grupos de desarrollo
          (CV interna agrupada por vaca, minimo error del estado post-enfriamiento) y se evalua en los
          grupos de evaluacion con un SCM ajustado solo en desarrollo;
      A2. clasificador, SCM y componentes de varianza se ajustan solo en desarrollo; las recomendaciones
          de Actionable-indep, Causal-SCM y la variante se comparan con las transiciones reales de las
          vacas de evaluacion. Ademas: comparacion sobre el MISMO subconjunto de instancias.
(B) Indice de acoplamiento por grupo genetico (dosis-respuesta dentro del caso Vaca).
(C) Puente con ensayos de enfriamiento publicados (data/external/cooling_trials.csv): que fraccion del
    contraste natural T2 -> T0 de este dataset reproducen, y que ganancia causal corresponde a esa
    eficacia en el eje gamma de la Tabla 22.

Salidas: results/oos_*.parquet|csv, tables/table_27_unseen_groups_validation, table_28_coupling_by_group,
         table_29_cooling_trials_bridge, figures/fig_18_out_of_sample_checks
"""
import itertools
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import utils as U
from cf_lib import SCM, SPECS, _parent_design
from cf_validation import GROUP_VARIANT, run_cf_fold_multi

warnings.filterwarnings("ignore")
PROC = U.PROJECT_DIR / "data" / "processed"
SPEC = SPECS["heat_stress"]
CLFS = ["LR", "MLP", "RF", "XGBoost", "LightGBM"]
MED = ["rectal_temp_c", "resp_bpm", "pulse_bpm", "dmi_kg"]
MODES = {"none": "No abduction", "replicate": "Replicate-level (original)", "cow": "Cow-level"}
M3 = ["Actionable-indep", "Causal-SCM", GROUP_VARIANT]
COL = {"Actionable-indep": "#009E73", "Causal-SCM": "#0072B2", GROUP_VARIANT: "#D55E00"}
D = pd.read_csv(PROC / "heat_stress_ml_ready.csv")
X, Y = D[SPEC.features].to_numpy(float), D["scc_high"].to_numpy()
COW, THI, HF = D.cow_id.to_numpy(), D.thi.to_numpy(), D.hf_pct.to_numpy()
GROUPS = sorted(np.unique(HF))
SPLITS = [{"split": k, "eval_groups": list(c)} for k, c in enumerate(itertools.combinations(GROUPS, 2))]


class CowObserved:
    """Igual que en 09: estado y CCS reales de la misma vaca en la fase de THI recomendada."""

    def __init__(self):
        self.it, self.idm = SPEC.idx["thi"], SPEC.idx["dmi_kg"]

    def __call__(self, i, cf, x, clf, target):
        a = int(round(cf[self.it]))
        rows = np.where((COW == COW[i]) & (THI == a))[0]
        null = a == THI[i]
        if null:
            rows = rows[rows != i]
        return {"obs_exact": float(abs(cf[self.idm] - x[self.idm]) < 1e-9), "obs_null": float(null),
                "obs_model": float((clf.predict(X[rows]) == target).mean()), "obs_real": float((Y[rows] == target).mean())}


def cooling_errors(scm, rows, rng_norm):
    """Error normalizado del estado post-enfriamiento (media de las 4 variables) por vaca y modo."""
    act = {a: j for j, a in enumerate(SPEC.actionable)}
    rng = np.random.default_rng(U.SEED)
    out = []
    for i in rows:
        x = X[i]
        for a in range(int(THI[i])):
            obs = X[(COW == COW[i]) & (THI == a)].mean(0)
            A = np.array([[x[SPEC.idx[f]] for f in SPEC.actionable]], float)
            A[0, act["thi"]] = a
            U_ = {"none": {n: np.zeros(1) for n in scm.dag}, "replicate": scm.abduct(x, 1, rng),
                  "cow": scm.abduct(x, 1, rng, mode="group_mean")}
            for m, u in U_.items():
                p = scm.counterfactual_batch(x, A, u)[0, 0]
                out.append({"cow_id": int(COW[i]), "mode": m,
                            "err": float(np.mean([abs(p[SPEC.idx[v]] - obs[SPEC.idx[v]]) / rng_norm[v] for v in MED]))})
    return pd.DataFrame(out)


def selection_split(sp):
    ev = np.isin(HF, sp["eval_groups"])
    dev, te = np.where(~ev)[0], np.where(ev)[0]
    rng_norm = {v: float(np.ptp(X[dev, SPEC.idx[v]])) for v in MED}
    inner = []
    for tr_i, te_i in GroupKFold(n_splits=5).split(dev, groups=COW[dev]):
        s = SCM(SPEC.dag, SPEC).fit(X[dev[tr_i]]).set_groups(X, COW, dev[tr_i])
        inner.append(cooling_errors(s, dev[te_i], rng_norm))
    inner = pd.concat(inner).groupby(["mode", "cow_id"]).err.mean().groupby("mode").mean()
    s = SCM(SPEC.dag, SPEC).fit(X[dev]).set_groups(X, COW, dev)
    e = cooling_errors(s, te, rng_norm).groupby(["mode", "cow_id"]).err.mean().reset_index()
    e["split"], e["selected_on_dev"] = sp["split"], inner.idxmin()
    for m in MODES:
        e[f"dev_err_{m}"] = inner[m]
    return e


def coupling_nonactionable(rows):
    """Indice de acoplamiento hacia hijos no accionables (definicion de 10_noncircular_statistics)."""
    g = COW[rows]
    P = {f: X[rows, SPEC.idx[f]] for f in SPEC.features}
    out = []
    for node, pa in SPEC.dag.items():
        lev = [q for q in pa if q in SPEC.actionable]
        if not lev or node in SPEC.actionable:
            continue
        yv = X[rows, SPEC.idx[node]]
        sk = []
        for parents in (pa, [q for q in pa if q not in lev]):
            Z = _parent_design(parents, P, SPEC) if parents else np.zeros((len(yv), 0))
            num = den = 0.0
            for tr, te in GroupKFold(n_splits=5).split(Z, yv, g):
                mu = yv[tr].mean()
                pr = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(Z[tr], yv[tr]).predict(Z[te]) if Z.shape[1] else np.full(len(te), mu)
                num += ((yv[te] - pr) ** 2).sum(); den += ((yv[te] - mu) ** 2).sum()
            sk.append(1 - num / den)
        out.append(sk[0] - sk[1])
    return float(np.mean(out))


def paired(df, col, a, b):
    B = df.pivot_table(index="unit", columns="method", values=col, aggfunc="mean")[[a, b]].dropna()
    x, z = B[a].values, B[b].values
    lo, hi = U.bootstrap_ci(x - z)
    return {"n": len(B), "a": float(x.mean()), "b": float(z.mean()), "diff": float((x - z).mean()), "lo": lo, "hi": hi,
            "p": U.wilcoxon_safe(x, z)[1], "cliffs_delta": U.cliffs_delta(x, z)}


if __name__ == "__main__":
    t0 = time.time()
    # ------------------------------------------------------------------ (A1) seleccion del nivel de abduccion
    sel = pd.concat(Parallel(n_jobs=-2)(delayed(selection_split)(sp) for sp in SPLITS), ignore_index=True)
    sel.to_csv(U.RESULTS_DIR / "oos_abduction_selection.csv", index=False)
    picked = sel.groupby("split").selected_on_dev.first()
    print("Nivel elegido en desarrollo:", picked.value_counts().to_dict())
    # ------------------------------------------------------------------ (A2) recurso en grupos no vistos
    world = SCM(SPEC.dag, SPEC).fit(X)
    obs = CowObserved()
    jobs = []
    for ci, c in enumerate(CLFS):
        for sp in SPLITS:
            ev = np.isin(HF, sp["eval_groups"])
            jobs.append((c, np.where(~ev)[0], np.where(ev)[0], U.SEED * 1000 + 50000 + ci * 100 + sp["split"], sp["split"]))
    out = Parallel(n_jobs=-2)(
        delayed(run_cf_fold_multi)(c, X, Y, tr, te, world, {"lin_expert": (world, 0.0)}, {}, None, seed, SPEC,
                                   tau=U.CF_TAU, m_samples=U.CF_M_SAMPLES, m_eval=U.EVAL_M_SAMPLES,
                                   extra={"split": k}, observed_fn=obs, groups=COW)
        for c, tr, te, seed, k in jobs)
    cf = pd.DataFrame([r for rows in out for r in rows])
    cf["unit"] = COW[cf.idx.values]
    cf["hf_pct"] = HF[cf.idx.values]
    cf["nominal_valid"] = cf.nominal_valid.fillna(0.0)
    cf["real_fp"] = ((cf.nominal_valid == 1) & (cf.obs_real < 0.5)).astype(float)
    cf = cf.drop(columns=["cf_first"])
    cf.to_parquet(U.RESULTS_DIR / "oos_unseen_groups_cf.parquet", index=False)
    print(f"(A) OK ({time.time() - t0:.0f}s)")

    ind = pd.read_parquet(U.RESULTS_DIR / "val_heat_stress.parquet")           # dentro de distribucion (paso 09)
    ind["nominal_valid"] = ind.nominal_valid.fillna(0.0)
    ind["real_fp"] = ((ind.nominal_valid == 1) & (ind.obs_real < 0.5)).astype(float)
    ind["hf_pct"] = HF[ind.idx.values]

    def same_subset(df, key):
        """Instancias en las que Causal-SCM y la variante dan AMBAS una recomendacion de solo enfriar."""
        s = df[df.method.isin(["Causal-SCM", GROUP_VARIANT]) & (df.feasible == 1) & (df.obs_exact == 1)]
        k = s.groupby(key).method.nunique()
        return s.set_index(key).loc[k[k == 2].index].reset_index()

    rows = []
    for label, df, key in (("Unseen genetic groups", cf, ["classifier", "split", "idx"]),
                           ("Within distribution (step 09)", ind, ["classifier", "repeat", "fold", "idx"])):
        for m in M3:
            d = df[df.method == m]
            r = {"evaluation": label, "method": m, "n_cows": d.unit.nunique(), "offered": d.groupby("unit").feasible.mean().mean()}
            for c in ("obs_model", "obs_real", "real_fp"):
                v = d.groupby("unit")[c].mean().values
                r[c] = v.mean(); r[c + "_lo"], r[c + "_hi"] = U.bootstrap_ci(v)
            rows.append(r)
        ss = same_subset(df, key)
        for c, lab in (("obs_model", "classifier"), ("obs_real", "real SCC"), ("real_fp", "false promise, real SCC")):
            for sub_lab, dd in (("all instances", df), ("same cooling-only instances", ss)):
                t = paired(dd, c, GROUP_VARIANT, "Causal-SCM")
                rows.append({"evaluation": label, "method": f"Variant - Causal-SCM: {lab}, {sub_lab}", "n_cows": t["n"],
                             "diff": t["diff"], "diff_lo": t["lo"], "diff_hi": t["hi"], "p": t["p"],
                             "variant_mean": t["a"], "causal_mean": t["b"], "cliffs_delta": t["cliffs_delta"]})
        for c, lab in (("obs_model", "classifier"), ("obs_real", "real SCC")):
            t = paired(df, c, "Causal-SCM", "Actionable-indep")
            rows.append({"evaluation": label, "method": f"Causal-SCM - Actionable-indep: {lab}, all instances", "n_cows": t["n"],
                         "diff": t["diff"], "diff_lo": t["lo"], "diff_hi": t["hi"], "p": t["p"],
                         "variant_mean": t["a"], "causal_mean": t["b"], "cliffs_delta": t["cliffs_delta"]})
    # eleccion del nivel de abduccion
    ev_err = sel.groupby(["mode", "cow_id"]).err.mean().unstack("mode")
    for m, lab in MODES.items():
        v = ev_err[m].values
        lo, hi = U.bootstrap_ci(v)
        rows.append({"evaluation": "Unseen genetic groups", "method": f"State prediction error: {lab}", "n_cows": len(v),
                     "obs_model": np.nan, "diff": v.mean(), "diff_lo": lo, "diff_hi": hi,
                     "p": U.wilcoxon_safe(v, ev_err["replicate"].values)[1] if m != "replicate" else np.nan})
    rows.append({"evaluation": "Unseen genetic groups", "method": "Splits in which the development groups selected cow-level abduction",
                 "n_cows": len(SPLITS), "diff": float((picked == "cow").mean())})
    T27 = pd.DataFrame(rows)
    T27.loc[T27.p.notna(), "p_holm"] = U.holm(T27.p.dropna().values)
    f3 = lambda v: "--" if pd.isna(v) else f"{v:.3f}"
    T27t = pd.DataFrame({
        "Evaluation": T27.evaluation, "Quantity": T27.method, "Cows": T27.n_cows,
        "Offered": T27.offered.map(f3),
        "Observed, classifier [95% CI]": [f"{a:.3f} {U.fmt_ci(l, h)}" if pd.notna(a) else "--" for a, l, h in zip(T27.obs_model, T27.obs_model_lo, T27.obs_model_hi)],
        "Observed, real SCC [95% CI]": [f"{a:.3f} {U.fmt_ci(l, h)}" if pd.notna(a) else "--" for a, l, h in zip(T27.obs_real, T27.obs_real_lo, T27.obs_real_hi)],
        "Observed false pr.": T27.real_fp.map(f3),
        "Value or difference [95% CI]": [f"{a:+.3f} {U.fmt_ci(l, h)}" if pd.notna(l) else f3(a) for a, l, h in zip(T27["diff"], T27.diff_lo, T27.diff_hi)],
        "p (Holm)": T27.p_holm.map(U.fmt_p)})
    U.save_table(T27, "table_27_unseen_groups_validation", df_tex=T27t)

    # ------------------------------------------------------------------ (B) acoplamiento por grupo genetico
    rows = []
    for g in GROUPS:
        r = {"hf_pct": g, "n_cows": int(D[HF == g].cow_id.nunique()), "coupling_nonactionable": coupling_nonactionable(np.where(HF == g)[0]),
             "prevalence_scc_high": float(Y[HF == g].mean())}
        d = ind[ind.hf_pct == g]
        for c, lab in (("impl__forest_dense", "gain_simulated_fs"), ("obs_model", "gain_observed_classifier"), ("obs_real", "gain_observed_real_scc")):
            if d.unit.nunique():
                t = paired(d, c, "Causal-SCM", "Actionable-indep")
                r[lab], r[lab + "_lo"], r[lab + "_hi"] = t["diff"], t["lo"], t["hi"]
        r["instances_explained"] = int((d.method == "Causal-SCM").sum())
        rows.append(r)
    T28 = pd.DataFrame(rows)
    rho = {c: stats.spearmanr(T28.coupling_nonactionable, T28[c]) for c in ("gain_simulated_fs", "gain_observed_classifier", "gain_observed_real_scc")}
    for c, s in rho.items():
        T28[f"spearman_{c}"] = s.statistic
        T28[f"spearman_p_{c}"] = s.pvalue
    T28t = pd.DataFrame({"HF genetics (%)": T28.hf_pct.map(lambda v: f"{v:g}"), "Cows": T28.n_cows,
                         "SCC > 200k": T28.prevalence_scc_high.map(lambda v: f"{v:.2f}"),
                         "Coupling (non-actionable children)": T28.coupling_nonactionable.map(lambda v: f"{v:.3f}"),
                         "Causal gain, simulated F/S [95% CI]": [f"{a:+.3f} {U.fmt_ci(l, h)}" for a, l, h in zip(T28.gain_simulated_fs, T28.gain_simulated_fs_lo, T28.gain_simulated_fs_hi)],
                         "Causal gain, observed real SCC [95% CI]": [f"{a:+.3f} {U.fmt_ci(l, h)}" for a, l, h in zip(T28.gain_observed_real_scc, T28.gain_observed_real_scc_lo, T28.gain_observed_real_scc_hi)]})
    U.save_table(T28, "table_28_coupling_by_group", df_tex=T28t)

    # ------------------------------------------------------------------ (C) ensayos de enfriamiento
    L = pd.read_csv(U.PROJECT_DIR / "data" / "external" / "cooling_trials.csv")
    hot, cool = D[THI == 2], D[THI == 0]
    nat = {v: {"abs": float(cool[v].mean() - hot[v].mean()), "rel": float(cool[v].mean() / hot[v].mean() - 1)}
           for v in ["resp_bpm", "rectal_temp_c", "pulse_bpm", "dmi_kg", "scc_k"]}
    L["trial_abs"] = L.abs_diff.where(L.abs_diff.notna(), L.treated - L.control)
    L["trial_rel"] = L.rel_diff.where(L.rel_diff.notna(), L.treated / L.control - 1)
    L["natural_abs"] = L.variable.map(lambda v: nat[v]["abs"])
    L["natural_rel"] = L.variable.map(lambda v: nat[v]["rel"])
    L["ratio_abs"] = L.trial_abs / L.natural_abs
    # la temperatura rectal solo se compara en absoluto (un cambio relativo en grados Celsius no tiene sentido)
    L["ratio_rel"] = np.where(L.variable == "rectal_temp_c", np.nan, L.trial_rel / L.natural_rel)
    L["efficacy"] = L.ratio_rel.where(L.ratio_rel.notna(), L.ratio_abs)
    L["implied_gamma"] = (1 - L.efficacy).clip(lower=0.0)
    T22 = pd.read_csv(U.TABLES_DIR / "table_22_confounding_attenuation.csv")
    cw = T22[T22.dataset == "heat_stress"].sort_values("gamma")
    L["causal_gain_at_gamma"] = np.interp(L.implied_gamma, cw.gamma, cw.diff_causal_vs_actionable)
    L["causal_false_promise_at_gamma"] = np.interp(L.implied_gamma, cw.gamma, cw["false_promise_Causal-SCM"])
    T29t = pd.DataFrame({"Study": [f"{s} {y}" for s, y in zip(L.study, L.year)], "Cooling contrast": L.cooling,
                         "Variable": L.variable.map({"resp_bpm": "Respiration", "rectal_temp_c": "Rectal temp.", "pulse_bpm": "Pulse", "dmi_kg": "DMI", "scc_k": "SCC"}),
                         "Trial effect": [f"{a:+.2f}" if pd.notna(a) else f"{100 * r:+.0f}%" for a, r in zip(L.trial_abs, L.trial_rel)],
                         "Natural T2 to T0 here": [f"{a:+.2f} ({100 * r:+.0f}%)" for a, r in zip(L.natural_abs, L.natural_rel)],
                         "Efficacy (trial / natural)": L.efficacy.map(lambda v: f"{v:.2f}"),
                         "Implied $\\gamma$": L.implied_gamma.map(lambda v: f"{v:.2f}"),
                         "Causal gain at that $\\gamma$": L.causal_gain_at_gamma.map(lambda v: f"{v:+.3f}")})
    U.save_table(L.fillna({"doi": "n/a"}), "table_29_cooling_trials_bridge", df_tex=T29t)

    # ------------------------------------------------------------------ Figura 18
    fig, axes = plt.subplots(1, 3, figsize=(U.DOUBLE_COL, 2.9), gridspec_kw={"width_ratios": [0.85, 1.25, 1.2]})
    ax = axes[0]
    for i, (m, lab) in enumerate(MODES.items()):
        v = ev_err[m].values
        lo, hi = U.bootstrap_ci(v)
        ax.bar(i, v.mean(), yerr=[[v.mean() - lo], [hi - v.mean()]], color=["#BBBBBB", "#0072B2", "#D55E00"][i], width=0.7, error_kw={"lw": 0.9})
    ax.set_xticks(range(3), ["None", "Replicate", "Cow"], fontsize=8)
    ax.set_xlabel("Abducted noise"); ax.set_ylabel("Normalised MAE,\nunseen genetic groups")
    U.add_panel_label(ax, "a", x=-0.5)
    ax = axes[1]
    for k, (lab, df, hatch) in enumerate((("Within distribution", ind, None), ("Unseen genetic groups", cf, "///"))):
        for i, m in enumerate(M3):
            v = df[df.method == m].groupby("unit").obs_real.mean().values
            lo, hi = U.bootstrap_ci(v)
            ax.bar(i + (k - 0.5) * 0.38, v.mean(), yerr=[[v.mean() - lo], [hi - v.mean()]], width=0.36, color=COL[m] if hatch is None else "white",
                   edgecolor=COL[m], hatch=hatch, lw=0.8, error_kw={"lw": 0.9})
    ax.set_xticks(range(3), ["Actionable", "Causal", "Causal,\ncow noise"], fontsize=8)
    ax.set_ylabel("Observed validity (real SCC)"); ax.set_ylim(0, 1.0)
    from matplotlib.patches import Patch
    ax.legend([Patch(facecolor="#777777", edgecolor="k"), Patch(facecolor="white", edgecolor="k", hatch="///")],
              ["Within distribution", "Unseen genetic groups"], fontsize=8, loc="upper left")
    U.add_panel_label(ax, "b", x=-0.3)
    ax = axes[2]
    ax.plot(cw.gamma, cw.diff_causal_vs_actionable, color="#0072B2", marker="o", ms=3.5, lw=1.3)
    ax.fill_between(cw.gamma, cw.diff_ci_lo, cw.diff_ci_hi, color="#0072B2", alpha=0.18, lw=0)
    ax.axhline(0, color="k", lw=0.7)
    g = L.implied_gamma.dropna()
    ax.axvspan(g.min(), g.max(), color="#E69F00", alpha=0.35, lw=0, label="Published cooling trials")
    ax.plot(g, np.full(len(g), cw.diff_ci_lo.min() - 0.02), "|", color="#D55E00", ms=7, mew=1.2)
    ax.set_xlabel(r"Effect not reproduced, $\gamma$")
    ax.legend(fontsize=8, loc="upper right", handlelength=1.2)
    ax.set_ylabel("Causal gain over\nActionable-indep")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", ".25", ".5", ".75", "1"])
    U.add_panel_label(ax, "c", x=-0.42)
    fig.tight_layout()
    U.save_figure(fig, "fig_18_out_of_sample_checks")

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(T27[["evaluation", "method", "n_cows", "offered", "obs_model", "obs_real", "real_fp", "diff", "diff_lo", "diff_hi", "p_holm"]].round(3).to_string())
    print(T28.round(3).T.to_string())
    print(L[["study", "variable", "trial_abs", "trial_rel", "natural_abs", "natural_rel", "ratio_abs", "ratio_rel", "efficacy", "implied_gamma", "causal_gain_at_gamma"]].round(3).to_string())
    print(f"11 OK ({time.time() - t0:.0f}s)")
