# Scripts de la tesis: guía rápida

Los scripts conservan su número original, que indica el **orden en que se hicieron**.
Las carpetas los agrupan por **para qué sirven**. Todos usan rutas absolutas
(`E:\Tesis_PulseDB\...`), así que se ejecutan igual desde cualquier carpeta.

> **Modelo final de la tesis:** CNN-Ensemble (carpeta 4, script 35), evaluado en
> CalFree con el script 36, empaquetado para la web con el 37 y graficado con el 40.

---

## 1_revision_de_datos: conocer y verificar el dataset
| Script | Qué hace |
|---|---|
| 01_verificar_sujetos | Cuenta los sujetos de Train y CalFree. |
| 02_analizar_presion | Estadísticas de la presión sistólica y diastólica. |
| 03_analizar_clases | Cuántos segmentos son normales y cuántos hipertensos. |
| 04_analizar_por_sujeto | Cuántos segmentos y de qué clase tiene cada sujeto. |
| 07_verificar_calidad_ecg | Revisa que las señales ECG no tengan valores vacíos o planos. |
| 14_analizar_ecg_train | Compara las señales ECG de la clase 0 y la clase 1. |
| 20a_inspeccionar_subject | Revisa cómo vienen guardados los ID de sujeto. |
| 27_revisar_variables_demograficas | Revisa la edad, el sexo, el peso y la talla del dataset. |
| 28_analizar_edades_y_criterio_adulto | Distribución de edades (población adulta). |
| 29_analizar_presiones_extremos | Busca valores de presión extremos o raros. |
| 32_auditar_variables_demograficas | Auditoría completa de las variables demográficas. |
| 38_verificar_dataset_pulsedb | Verificación general del dataset PulseDB utilizado. |

## 2_preparacion_de_datos: separar entrenamiento y validación
| Script | Qué hace |
|---|---|
| 05_preparar_division | Divide por sujeto: 1 034 para entrenamiento y 259 para validación. |
| 06_verificar_clases_split | Comprueba la proporción de clases en cada parte. |
| 08_probar_dataset_pytorch | Prueba la carga de los datos en PyTorch. |
| 44_preparar_sujetos_nuevos | Plan B: arma los segmentos de los 200 pacientes nuevos (máx. 360 por paciente, al azar). |

## 3_pruebas_y_modelos_iniciales: primeras versiones (reemplazadas)
Sirven como historial del desarrollo; **no forman parte del modelo final**.
| Script | Qué hace |
|---|---|
| 09_probar_cnn | Prueba que la primera CNN funcione (sin datos reales). |
| 10_prueba_entrenamiento_real | Entrenamiento corto de prueba. |
| 11_entrenar_validar_cnn | Primer entrenamiento completo de la CNN base. |
| 12_continuar_entrenamiento | Continúa ese entrenamiento con *early stopping*. |
| 13_analizar_umbral | Umbrales de la CNN base. |
| 15_entrenar_cnn_zscore | CNN con normalización z-score por segmento. |
| 16_probar_cnn_v2 | Prueba la arquitectura residual V2 (sin datos reales). |
| 17_entrenar_cnn_v2 | Primer entrenamiento de la CNN V2. |
| 18_entrenar_cnn_v2_regularizada | CNN V2 con más regularización (criterio anterior). |
| 19_analizar_umbral_v2_regularizada | Umbrales de esa CNN V2. |
| 20_analizar_por_sujeto_validation | Resultados por sujeto en validación. |
| 21_analizar_umbral_por_sujeto | Umbrales evaluados por sujeto. |
| 22_comparar_criterios_presion | Compara criterios de presión y justifica el uso de 140/90. |

## 4_modelos_finales_140_90: entrenamiento del modelo final
| Script | Qué hace |
|---|---|
| 23_entrenar_cnn_v2_140_90 | Entrena la **CNN 1D residual** (criterio 140/90). |
| 30_entrenar_cnn_v3_multiescala | Entrena la **CNN 1D multiescala**. |
| 31_entrenar_cnn_multitarea | Entrena la **CNN 1D multitarea** (clase + PAS + PAD). |
| 35_ensemble_cnn | Combina las tres en el **CNN-Ensemble** y elige los umbrales en validación. |

## 5_umbrales_y_evaluacion: umbrales y prueba final en CalFree
| Script | Qué hace |
|---|---|
| 24_analizar_umbral_140_90 | Umbrales de la CNN residual 140/90. |
| 24b_completar_umbral_140_90 | Completa la tabla de umbrales anterior. |
| 25_analizar_umbrales_por_sujeto_140_90 | Umbrales evaluados por sujeto. |
| 26_evaluacion_final_calfree | Evaluación en CalFree de la CNN residual sola. |
| 36_evaluacion_final_ensemble_calfree | **Evaluación final del CNN-Ensemble en CalFree** (resultados de la tesis). |
| 41_comparar_ensemble_vs_cnn_calfree | Ensemble frente a cada CNN en CalFree (Tabla VI); usa los resultados del 36, sin reentrenar. |
| 47_analisis_por_severidad | Análisis por severidad (grado 1 y grado ≥ 2, ESC 2024) en validación y CalFree; usa las predicciones del 35 y el 36. |

## 6_experimentos_complementarios: pruebas que no mejoraron el modelo
| Script | Qué hace |
|---|---|
| 33_modelo_a_demografico | Modelo solo con edad, sexo e IMC (ROC-AUC 0,52). |
| 34_experimento_calibracion | Experimento de calibración personal por sujeto. |
| 39_entrenar_cnn_v4_unificada | Una sola red unificada V4 (0,727 en validación; descartada). |

## 7_app_web: pasar el modelo a la aplicación web
| Script | Qué hace |
|---|---|
| 37_empaquetar_ensemble | Junta las tres CNN en un archivo que usa la app web. |

## 8_figuras: figuras para el documento
| Script | Qué hace |
|---|---|
| 40_figuras_formato_paf | Figuras 6, 7 y 8 en formato PAF (300 dpi, coma decimal). |

---
Copia de la organización anterior (todos en una sola carpeta):
`E:\Tesis_PulseDB\scripts_respaldo_2026-11-21_antes_ordenar`

## 9_modelos_ampliados_plan_b: experimento con 200 pacientes nuevos (no mejoró)
| Script | Qué hace |
|---|---|
| 45_entrenar_cnn_ampliada | Reentrena cada CNN con los 1034 pacientes originales + 200 nuevos (misma configuración). |
| 46_ensemble_ampliado_validacion | Compara el ensemble ampliado con el actual en validación (0,741 frente a 0,739; no significativo). |

(Los datos nuevos se preparan con 2_preparacion_de_datos/44_preparar_sujetos_nuevos.)
