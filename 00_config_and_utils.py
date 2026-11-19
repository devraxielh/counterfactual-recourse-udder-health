"""00_config_and_utils.py - configuracion global, estilo Q1 y utilidades compartidas.

Paper: Causally-constrained counterfactual explanations for actionable milk-quality recourse.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# ----------------------------------------------------------------------------
# Reproducibilidad
# ----------------------------------------------------------------------------
SEED = 42
N_BOOTSTRAP = 2000
ALPHA = 0.05
# Prediccion (unicos): 5 folds x 20 repeticiones = 100 evaluaciones
PRED_SPLITS, PRED_REPEATS = 5, 20
# Contrafactuales: 5 folds x 10 repeticiones
CF_SPLITS, CF_REPEATS = 5, 10
CF_M_SAMPLES = 100      # muestras de ruido posterior para recurso causal
EVAL_M_SAMPLES = 200    # muestras para validez al implementar
CF_TAU = 0.8            # umbral de validez esperada del recurso causal
TARGET_CLASS = 2        # high


def set_global_seed(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


set_global_seed()

# ----------------------------------------------------------------------------
# Rutas
# ----------------------------------------------------------------------------
SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPTS_DIR.parent
DATA_PATH = PROJECT_DIR / "dataset_ml_ready.csv"
RAW_GROUPED_PATH = PROJECT_DIR / "dataset_raw_grouped.csv"
RESULTS_DIR = SCRIPTS_DIR / "results"
TABLES_DIR = SCRIPTS_DIR / "tables"
FIGURES_DIR = SCRIPTS_DIR / "figures"
for _d in (RESULTS_DIR, TABLES_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["Low", "Medium", "High"]

# ----------------------------------------------------------------------------
# Estilo grafico Q1 (sin titulos)
# ----------------------------------------------------------------------------
OKABE_ITO = ["#E69F00", "#56B4E9", "#009E73", "#F0E442",
             "#0072B2", "#D55E00", "#CC79A7", "#000000"]
CLASS_COLORS = {0: "#D55E00", 1: "#E69F00", 2: "#0072B2"}
SINGLE_COL = 3.5
DOUBLE_COL = 7.0


def set_publication_style(family: str = "serif") -> None:
    sns.set_theme(context="paper", style="ticks", palette=OKABE_ITO)
    plt.rcParams.update({
        "font.family": family,
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "stix" if family == "serif" else "dejavusans",
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "figure.dpi": 150,
    })


set_publication_style()


def add_panel_label(ax, label: str, x: float = -0.14, y: float = 1.04) -> None:
    ax.text(x, y, f"({label})", transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="bottom", ha="left")


def _strip_titles(fig) -> None:
    if getattr(fig, "_suptitle", None) is not None:
        fig._suptitle.set_text("")
    for ax in fig.get_axes():
        ax.set_title("")
        ax.set_title("", loc="left")
        ax.set_title("", loc="right")


def save_figure(fig, name: str, formats=("pdf", "png"), dpi: int = 600, despine: bool = True) -> list[Path]:
    """Guarda en vectorial + PNG 600 DPI, eliminando cualquier titulo residual."""
    _strip_titles(fig)
    if despine:
        for ax in fig.get_axes():
            sns.despine(ax=ax)
    paths = []
    for fmt in formats:
        p = FIGURES_DIR / f"{name}.{fmt}"
        for attempt in range(6):  # reintentos: OneDrive puede bloquear el archivo durante la sincronizacion
            try:
                fig.savefig(p, dpi=dpi, bbox_inches="tight")
                break
            except OSError:
                if attempt == 5:
                    raise
                import time
                time.sleep(2 * (attempt + 1))
        paths.append(p)
    plt.close(fig)
    return paths


# ----------------------------------------------------------------------------
# Tablas
# ----------------------------------------------------------------------------
def _latex_escape(s) -> str:
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("#", r"\#"), ("_", r"\_"), ("$", r"\$")):
        s = s.replace(a, b)
    return s


def highlight(df_tex: pd.DataFrame, values: pd.Series, col: str, higher_is_better: bool,
              groups: pd.Series | None = None) -> None:
    """Marca en df_tex[col] el mejor valor en \\textbf{} y el segundo en \\underline{} (por grupo si se da)."""
    vals = pd.Series(np.asarray(values, float), index=df_tex.index)
    grp = pd.Series(0, index=df_tex.index) if groups is None else pd.Series(np.asarray(groups), index=df_tex.index)
    for _, idx in vals.groupby(grp).groups.items():
        v = vals.loc[idx].dropna()
        if len(v) == 0:
            continue
        order = v.sort_values(ascending=not higher_is_better)
        uniq = order.unique()
        best = order.index[order == uniq[0]]
        for i in best:
            df_tex.loc[i, col] = "\\textbf{" + str(df_tex.loc[i, col]) + "}"
        if len(uniq) > 1:
            for i in order.index[order == uniq[1]]:
                df_tex.loc[i, col] = "\\underline{" + str(df_tex.loc[i, col]) + "}"


def arrow(name: str, higher_is_better: bool | None) -> str:
    """Encabezado con direccion de la metrica: ↑ mayor es mejor, ↓ menor es mejor."""
    if higher_is_better is None:
        return name
    return f"{name} $\\uparrow$" if higher_is_better else f"{name} $\\downarrow$"


def save_table(df: pd.DataFrame, name: str, df_tex: pd.DataFrame | None = None,
               index: bool = False, column_format: str | None = None) -> None:
    """CSV con valores completos (celdas no aplicables = 'n/a') + .tex booktabs (df_tex = version formateada)."""
    df.astype(object).where(pd.notna(df), "n/a").to_csv(TABLES_DIR / f"{name}.csv", index=index)
    tex_df = (df_tex if df_tex is not None else df).copy()
    # encabezados: se escapa el texto fuera de $...$ y se respeta el modo matematico ($\uparrow$, $\chi^2$...)
    def _esc_header(c):
        parts = str(c).split("$")
        return "$".join(p if i % 2 else _latex_escape(p) for i, p in enumerate(parts))
    tex_df.columns = [_esc_header(c) for c in tex_df.columns]
    for c in tex_df.select_dtypes(include="object").columns:
        tex_df[c] = tex_df[c].map(lambda v: v if "\\" in str(v) else _latex_escape(v))
    tex = tex_df.to_latex(index=index, escape=False, column_format=column_format,
                          float_format="%.3f")
    (TABLES_DIR / f"{name}.tex").write_text(_unicode_to_latex(tex), encoding="utf-8")


_UNICODE_LATEX = {"→": r"$\rightarrow$", "⊥": r"$\perp$", "∅": r"$\emptyset$", "–": "--", "—": "---",
                  "Δ": r"$\Delta$", "τ": r"$\tau$", "δ": r"$\delta$", "≤": r"$\le$", "≥": r"$\ge$",
                  "±": r"$\pm$", "µ": r"$\mu$", "°": r"$^\circ$", "×": r"$\times$", "↑": r"$\uparrow$",
                  "↓": r"$\downarrow$"}


def _unicode_to_latex(s: str) -> str:
    """Sustituye caracteres Unicode por comandos LaTeX para que compile con pdflatex sin paquetes extra."""
    for a, b in _UNICODE_LATEX.items():
        s = s.replace(a, b)
    return s


def fmt_mean_sd(values, decimals: int = 3) -> str:
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return "--"
    sd = v.std(ddof=1) if v.size > 1 else 0.0
    return f"{v.mean():.{decimals}f} $\\pm$ {sd:.{decimals}f}"


def fmt_ci(lo: float, hi: float, decimals: int = 3) -> str:
    return f"[{lo:.{decimals}f}, {hi:.{decimals}f}]"


def fmt_p(p: float) -> str:
    if p is None or np.isnan(p):
        return "--"
    return f"{p:.2e}" if p < 1e-3 else f"{p:.4f}"


# ----------------------------------------------------------------------------
# Estadistica
# ----------------------------------------------------------------------------
def bootstrap_ci(values, stat=np.mean, n_boot: int = N_BOOTSTRAP,
                 alpha: float = ALPHA, seed: int = SEED) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    boots = stat(v[idx], axis=1) if stat in (np.mean, np.median) else np.array([stat(v[i]) for i in idx])
    return float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def rank_biserial_wilcoxon(x, y) -> float:
    from scipy.stats import rankdata
    d = np.asarray(x, float) - np.asarray(y, float)
    d = d[d != 0]
    if d.size == 0:
        return 0.0
    r = rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def cliffs_delta(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    gt = (x[:, None] > y[None, :]).sum()
    lt = (x[:, None] < y[None, :]).sum()
    return float((gt - lt) / (x.size * y.size))


def cliffs_magnitude(d: float) -> str:
    a = abs(d)
    return "negligible" if a < 0.147 else "small" if a < 0.33 else "medium" if a < 0.474 else "large"


def wilcoxon_safe(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if np.allclose(x - y, 0):
        return np.nan, 1.0
    res = stats.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
    return float(res.statistic), float(res.pvalue)


def tost_wilcoxon(x, y, margin: float = 0.05):
    """TOST pareado no parametrico (dos Wilcoxon unilaterales) para equivalencia |media(x - y)| < margin.
    Devuelve (p_tost, p_lower, p_upper, IC 90% bootstrap de la diferencia media)."""
    d = np.asarray(x, float) - np.asarray(y, float)
    def one(v, alt):
        if np.allclose(v, 0):
            return 1.0
        return float(stats.wilcoxon(v, zero_method="wilcox", alternative=alt).pvalue)
    p_lo = one(d + margin, "greater")     # H0: diff <= -margin
    p_hi = one(d - margin, "less")        # H0: diff >= +margin
    lo, hi = bootstrap_ci(d, alpha=0.10)
    return max(p_lo, p_hi), p_lo, p_hi, (lo, hi)


def holm(pvals) -> np.ndarray:
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def nadeau_bengio_ttest(diffs, n_train: int, n_test: int):
    """t-test corregido para CV repetida (Nadeau & Bengio, 2003; Bouckaert & Frank, 2004)."""
    d = np.asarray(diffs, float)
    k = d.size
    var = d.var(ddof=1)
    if var == 0:
        return np.nan, 1.0
    t = d.mean() / np.sqrt((1.0 / k + n_test / n_train) * var)
    p = 2 * stats.t.sf(abs(t), df=k - 1)
    return float(t), float(p)


def friedman_nemenyi(score_matrix: pd.DataFrame, higher_is_better: bool = True):
    """score_matrix: filas = bloques (instancias/folds), columnas = metodos."""
    M = score_matrix.dropna()
    stat, p = stats.friedmanchisquare(*[M[c].values for c in M.columns])
    ranks = M.rank(axis=1, ascending=not higher_is_better).mean()
    k, n = M.shape[1], M.shape[0]
    q = stats.studentized_range.ppf(1 - ALPHA, k, np.inf) / np.sqrt(2)
    cd = q * np.sqrt(k * (k + 1) / (6.0 * n))
    return float(stat), float(p), ranks.sort_values(), float(cd), n


def plot_cd_diagram(ax, ranks: pd.Series, cd: float, label_fmt=str) -> None:
    """Diagrama de diferencia critica (Demsar, 2006) sin titulo."""
    ranks = ranks.sort_values()
    k = len(ranks)
    lo, hi = 1, k
    ax.set_xlim(lo - 0.3, hi + 0.3)
    ax.set_ylim(0, 1)
    ax.invert_xaxis()
    ax.axis("off")
    y0 = 0.82
    ax.hlines(y0, lo, hi, color="k", lw=1)
    for r in range(lo, hi + 1):
        ax.vlines(r, y0, y0 + 0.04, color="k", lw=1)
        ax.text(r, y0 + 0.06, str(r), ha="center", va="bottom", fontsize=8)
    # CD bar
    ax.hlines(0.97, hi, hi - cd, color="k", lw=1.5)
    ax.text(hi - cd / 2, 0.985, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=8)
    names = list(ranks.index)
    half = int(np.ceil(k / 2))
    for i, (name, r) in enumerate(ranks.items()):
        if i < half:
            yy = y0 - 0.12 - 0.11 * i
            ax.plot([r, r, lo - 0.25], [y0, yy, yy], color="k", lw=0.7)
            # eje invertido: ha="left" coloca el texto a la derecha (fuera de la linea)
            ax.text(lo - 0.3, yy, f"{label_fmt(name)} ({r:.2f})", ha="left", va="center", fontsize=8)
        else:
            j = k - 1 - i
            yy = y0 - 0.12 - 0.11 * j
            ax.plot([r, r, hi + 0.25], [y0, yy, yy], color="k", lw=0.7)
            ax.text(hi + 0.3, yy, f"({r:.2f}) {label_fmt(name)}", ha="right", va="center", fontsize=8)
    # grupos no significativos
    vals = ranks.values
    groups = []
    for i in range(k):
        j = i
        while j + 1 < k and vals[j + 1] - vals[i] <= cd:
            j += 1
        if j > i and not any(g[0] <= i and g[1] >= j for g in groups):
            groups.append((i, j))
    for gi, (i, j) in enumerate(groups):
        yy = y0 - 0.05 - 0.035 * gi
        ax.hlines(yy, vals[i] - 0.03, vals[j] + 0.03, color="k", lw=2.2)
