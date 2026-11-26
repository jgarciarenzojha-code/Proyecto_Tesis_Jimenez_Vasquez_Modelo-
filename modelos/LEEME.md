# Carpeta modelos: qué es cada archivo

## Modelo final de la tesis (NO renombrar ni mover: los usan los scripts, la web y el Word)
| Archivo | Qué es | Script que lo creó |
|---|---|---|
| mejor_modelo_v2_regularizada_140_90.pt | CNN 1 · Residual | 23 |
| mejor_modelo_v3_multiescala_140_90.pt | CNN 2 · Multiescala | 30 |
| mejor_modelo_multitarea_140_90.pt | CNN 3 · Multitarea | 31 |
| **mejor_modelo_ensemble_140_90.pt** | **CNN-Ensemble: las 3 CNN + umbrales en un solo archivo. Es el que usa la app web.** | 37 (con los resultados del 35) |

El ensemble (script 35) no es un entrenamiento aparte: carga las tres CNN de arriba y promedia sus salidas.

## Experimentos complementarios (no se usan en la web; respaldan la Discusión)
| Archivo | Qué es | Resultado en validación |
|---|---|---|
| experimento_v4_unificada\mejor_modelo_v4_unificada_140_90.pt | Una sola red que une las tres (script 39) | 0,727, descartada |
| experimento_200_pacientes\mejor_modelo_v2_ampliado_45.pt | CNN residual con 200 pacientes más (script 45) | 0,732 |
| experimento_200_pacientes\mejor_modelo_v3_ampliado_45.pt | CNN multiescala con 200 pacientes más (script 45) | 0,718 |
| experimento_200_pacientes\mejor_modelo_multitarea_ampliado_45.pt | CNN multitarea con 200 pacientes más (script 45) | 0,727 |

Los 3 modelos ampliados están en la subcarpeta experimento_200_pacientes (no tienen relación con el modelo final).
Ensemble ampliado: 0,741 frente a 0,739 del final (no significativo; script 46). Se conservó el modelo final.

## Borrados (25-nov-2026)
Modelos de las pruebas iniciales (scripts 11–19: CNN base, z-score y V2 con el criterio anterior).
No forman parte del modelo final ni del Word; si se vuelven a correr esos scripts, se crean de nuevo.
