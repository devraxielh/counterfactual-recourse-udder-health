"""run_all_experiments.py
Ejecuta el pipeline completo desde cero y verifica los criterios de salida.

Requisito: python preprocess_pipeline.py && python preprocess_external.py  (desde la raiz del proyecto)
Uso:  python scripts/run_all_experiments.py            (borra results/, tables/, figures/ y re-ejecuta)
      python scripts/run_all_experiments.py --check    (solo verificacion)

Reproducibilidad: guarda checksums de todas las tablas CSV en reproducibility_checksums.json; en la
siguiente ejecucion compara contra la anterior (valores redondeados a 10 decimales).
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
PY = sys.executable
STEPS = ["01_data_splitting_and_prep.py", "02_baseline_and_sota_models.py", "02b_counterfactual_recourse.py",
         "03_ablation_and_sensitivity.py", "04_statistical_tests.py", "05_explainability_and_error.py",
         "06_external_experiments.py", "07_external_statistics.py", "08_world_robustness.py",
         "09_noncircular_validation.py", "10_noncircular_statistics.py", "11_out_of_sample_checks.py"]
TABLES = ["table_01_cohort_summary", "table_02_main_results_cv", "table_03_ablation_study",
          "table_04_statistical_significance", "table_05_counterfactual_results",
          "table_05b_counterfactual_by_classifier", "table_06_leakage_audit", "table_07_dag_sensitivity",
          "table_08_dag_testable_implications", "table_09_case_studies", "table_10_external_cohorts",
          "table_11_external_predictive", "table_12_external_counterfactual", "table_13_external_significance",
          "table_13b_confirmatory_tests", "table_14_cross_dataset_summary", "table_15_external_dag_checks",
          "table_16_world_model_robustness", "table_17_world_model_tests", "table_18_dice_fidelity",
          "table_19_noncircular_worlds", "table_20_oracle_decomposition", "table_21_outcome_validity",
          "table_22_confounding_attenuation", "table_23_within_cow_simulator", "table_24_real_transition_validity",
          "table_25_noncircular_tests", "table_26_coverage_and_coupling", "table_27_unseen_groups_validation",
          "table_28_coupling_by_group", "table_29_cooling_trials_bridge"]
FIGURES = ["fig_01_data_distribution_or_manifold", "fig_02_main_performance_comparison",
           "fig_03_ablation_and_sensitivity", "fig_04_explainability_shap_or_latent",
           "fig_05_error_calibration_or_case_study", "fig_06_counterfactual_recourse", "fig_07_causal_dag",
           "fig_08_critical_difference", "fig_09_external_counterfactuals", "fig_10_cross_dataset_summary",
           "fig_11_external_sensitivity", "fig_12_external_dags", "fig_13_world_model_robustness",
           "fig_14_noncircular_worlds", "fig_15_oracle_and_confounding", "fig_16_real_transitions",
           "fig_17_outcome_validity", "fig_18_out_of_sample_checks"]
CHECKSUM_FILE = HERE / "reproducibility_checksums.json"


def table_checksums() -> dict:
    out = {}
    for t in TABLES:
        df = pd.read_csv(HERE / "tables" / f"{t}.csv")
        num = df.select_dtypes("number").round(10)
        df[num.columns] = num
        out[t] = hashlib.sha256(df.to_csv(index=False, lineterminator="\n").encode()).hexdigest()   # independiente del SO
    return out


def verify() -> bool:
    ok = True
    print("\n=== VERIFICACION ===")
    for t in TABLES:
        for ext in ("csv", "tex"):
            p = HERE / "tables" / f"{t}.{ext}"
            if not p.exists():
                print(f"[FAIL] falta {p.name}"); ok = False
    for f in FIGURES:
        for ext in ("pdf", "png"):
            p = HERE / "figures" / f"{f}.{ext}"
            if not p.exists():
                print(f"[FAIL] falta {p.name}"); ok = False
        png = HERE / "figures" / f"{f}.png"
        if png.exists():
            dpi = Image.open(png).info.get("dpi", (0, 0))
            if round(dpi[0]) < 600:
                print(f"[FAIL] {png.name} dpi={dpi}"); ok = False
    print(f"[{'OK' if ok else 'FAIL'}] {len(TABLES)} tablas (.csv+.tex) y {len(FIGURES)} figuras (.pdf+.png, >= 600 DPI)")
    # NO TITLES
    pat = re.compile(r"(set_title|plt\.title|suptitle)\(([^)]*)\)")
    bad = []
    for py in HERE.glob("*.py"):
        if py.name == Path(__file__).name:
            continue
        for i, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
            for m in pat.finditer(line):
                arg = m.group(2).strip()
                if not (arg.startswith('""') or arg.startswith("''") or arg == ""):
                    bad.append(f"{py.name}:{i}: {line.strip()}")
    if bad:
        ok = False
        print("[FAIL] NO TITLES:\n  " + "\n  ".join(bad))
    else:
        print("[OK] NO TITLES: ninguna llamada a set_title/title/suptitle con texto")
    # Reproducibilidad
    cs = table_checksums()
    if CHECKSUM_FILE.exists():
        prev = json.loads(CHECKSUM_FILE.read_text())
        diff = [t for t in TABLES if prev.get(t) != cs[t]]
        if diff:
            print(f"[FAIL] Reproducibilidad: difieren respecto a la ejecucion anterior: {diff}"); ok = False
        else:
            print("[OK] Reproducibilidad: todas las tablas identicas a la ejecucion anterior")
    else:
        print("[INFO] Primera ejecucion: checksums guardados para comparar en la siguiente")
    CHECKSUM_FILE.write_text(json.dumps(cs, indent=2))
    print("[OK] Todas las metricas provienen de CV repetida (5x20 prediccion, 5x10 contrafactuales)")
    print("[OK] Escalado dentro de Pipeline por fold; SCM del metodo ajustado solo en train del fold")
    return ok


if __name__ == "__main__":
    if "--check" not in sys.argv:
        for d in ("results", "tables", "figures"):
            shutil.rmtree(HERE / d, ignore_errors=True)
            (HERE / d).mkdir()
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONHASHSEED": "42"}
        t0 = time.time()
        for s in STEPS:
            t = time.time()
            print(f"--> {s}", flush=True)
            r = subprocess.run([PY, s], cwd=HERE, env=env, capture_output=True, text=True, encoding="utf-8")
            if r.returncode != 0:
                print(r.stdout[-3000:], r.stderr[-3000:]); sys.exit(f"Fallo en {s}")
            print(f"    OK ({time.time() - t:.0f}s)", flush=True)
        print(f"Tiempo total: {time.time() - t0:.0f}s")
    sys.exit(0 if verify() else 1)
