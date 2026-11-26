# Carpeta resultados: una carpeta por cada CNN, el ensemble y los experimentos

Tipos: **.json / .csv** = métricas (se abren en VS Code o Excel) · **historial_*.csv** = métricas por época ·
**log_*.txt** = salida de la terminal · **.npz / .npy** = predicciones por segmento (datos para los scripts).

## 1_cnn_residual\  →  CNN 1 · Residual (scripts 23–26)
| Archivo | Qué contiene | Script |
|---|---|---|
| historial_cnn_v2_regularizada_140_90.csv | Pérdida y ROC-AUC de cada época (mejor: época 3, AUC 0,723) | 23 |
| probabilidades_validation_v2_140_90.npy · etiquetas_validation_v2_140_90.npy | Sus predicciones en validación | 24 |
| metricas_umbral_por_sujeto_140_90.csv | Sus umbrales evaluados por sujeto | 25 |
| metricas_finales_calfree.csv · probabilidades_calfree_final.npy · etiquetas_calfree_final.npy | Su evaluación inicial en CalFree (AUC 0,678) | 26 |

## 2_cnn_multiescala\  →  CNN 2 · Multiescala (script 30)
| historial_cnn_v3_multiescala_140_90.csv | Pérdida y ROC-AUC de cada época (mejor: época 2, AUC 0,722) | 30 |
|---|---|---|

## 3_cnn_multitarea\  →  CNN 3 · Multitarea (script 31)
| historial_cnn_multitarea_140_90.csv | Pérdida, ROC-AUC y error de PAS/PAD por época (mejor: época 3, AUC 0,726) | 31 |
|---|---|---|

## 4_cnn_ensemble\  →  el modelo final: los números del Word
| Archivo | Qué contiene | Script | En el Word |
|---|---|---|---|
| **ensemble_35.json** / .csv | AUC en validación del ensemble (0,739) y de cada CNN, umbrales 0,4698 y 0,3039 | 35 | Tablas VI y VII |
| predicciones_val_ensemble_35.npz | Probabilidad de cada segmento de validación (ensemble y cada CNN) | 35 | Fig. 8 |
| **evaluacion_final_ensemble_calfree_36.json** / .csv | Evaluación final en CalFree: AUC 0,695, IC 95 %, matriz de confusión, sensibilidad, etc. | 36 | Tabla VII |
| probabilidades_calfree_ensemble_36.npz | Probabilidad de cada segmento de CalFree (ensemble y cada CNN) | 36 | Fig. 7 y 8 |
| comparacion_ensemble_vs_cnn_calfree_41.json / .csv · log_41 | Ensemble frente a cada CNN en CalFree | 41 | Tabla VI |
| analisis_severidad_47.json / .csv · log_47 | Desempeño por grado de HTA (grado 1 y grado ≥ 2) | 47 | Tabla VIII |

Las métricas de cada CNN en validación y en CalFree están dentro de los archivos del ensemble
(35 y 41), porque se calcularon todas juntas para poder compararlas.

## 5_experimentos\  →  pruebas complementarias que no cambiaron el modelo final
| Subcarpeta | Experimento | Resultado |
|---|---|---|
| demografia\ | Edad, sexo e IMC (scripts 32 y 33) | ROC-AUC 0,52: no predice |
| calibracion\ | Calibración personal por sujeto (script 34) | Factibilidad |
| v4_unificada\ | Una sola red que une las tres (script 39) | 0,727 en validación: descartada |
| 200_pacientes\ | 200 pacientes más de VitalDB (scripts 42–46) | 0,741 frente a 0,739: sin mejora significativa |

Las rutas de todos los scripts ya apuntan a estas carpetas (25-nov-2026).
La carpeta auditoria_ecg_40\ la creó un script aparte (40_extraer_muestra_ecg_auditoria.py, en la carpeta principal).
