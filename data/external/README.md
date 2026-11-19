# Datos externos

`cooling_trials.csv`: efectos de ensayos de enfriamiento publicados sobre fisiología, consumo y CCS de vacas lecheras.
Lo usa `scripts/11_out_of_sample_checks.py` (tabla 29).

Verificación del 2026-11-18 (columna `verified`):
- **full text**: D'Emilio et al. 2017 y Rezazadeh Baei 2006, leídos en el PDF de acceso abierto.
- **abstract at publisher**: Brouk et al. 2003, resumen en la biblioteca de ASABE.
- **abstract only**: Perano et al. 2015 y Smith et al. 2006. El resumen coincide en Elicit y Europe PMC, pero el sitio
  de la revista rechazó la descarga automática del texto completo. **Pendiente de verificar a mano.**

Cambios respecto a la primera versión (tomada de resúmenes):
- D'Emilio et al. 2018 (CIGR Journal) se eliminó: informa el mismo ensayo y las mismas cifras que el periodo P1 de
  D'Emilio et al. 2017.
- D'Emilio et al. 2017: el resumen dice "cerca de 50 frente a 70"; el texto da 56,4 frente a 70,1 en P1. Se usa P1.
- Rezazadeh Baei 2006: valores con dos decimales de la tabla 2. El grupo control también tenía 6 ventiladores y mojado
  seis veces al día, así que el contraste es "más enfriamiento frente a algo de enfriamiento".
