# Fuentes de datos crudos

| Carpeta | Fuente | Licencia | Descarga |
|---|---|---|---|
| `milknew.csv` | Kaggle `cpluzshrijayan/milkquality` (Milk Quality Prediction) | ver Kaggle | 2026-11-11 |
| `heat_stress_bd/` | Mendeley Data 954f6g36sb v2: "Physiological responses, Dry matter Intake, milk yield, composition and blood metabolites of HF Cross cows" (Savar, Bangladesh, 2024) — https://data.mendeley.com/datasets/954f6g36sb | CC BY 4.0 | 2026-11-11 |
| `scc_management/` | Mendeley Data xbk6x9rrnb v1: Data for "Management practices and correlation of somatic cell levels in bovine milk" — https://data.mendeley.com/datasets/xbk6x9rrnb | CC BY 4.0 | 2026-11-11 |

Notas conocidas:
* heat_stress_bd: `Animal ID` en `physiological_responses.xlsx` esta desfasado en -1 respecto a los otros dos archivos (offset constante en las 50 vacas); los nombres de grupo genetico usan `HF_50` vs `HF50`.
* scc_management: las variables `CCS_x` valen 1 cuando la CCS EXCEDE el limite (contrario a la descripcion de la hoja Summary); verificado contra `CCS` (100 % de coincidencia).
