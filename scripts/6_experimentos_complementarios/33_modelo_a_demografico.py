import os
import json
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

# ============================================================
# PUNTO 33 - MODELO A: SOLO VARIABLES DEMOGRÁFICAS
#
# Regresión logística con edad, sexo e IMC (sin ECG).
# Sirve como línea base para responder:
#   ¿Cuánto se predice SOLO con demografía?
#
# Se compara con el Modelo B (CNN V2, solo ECG) usando las
# probabilidades de validación ya guardadas en el punto 24
# (no se reentrena la CNN).
#
# Reglas:
# - Imputación y normalización con estadísticas SOLO de Train.
# - Selección únicamente con Validation.
# - CalFree NO se utiliza.
# - Intervalos de confianza 95 % con bootstrap POR SUJETO.
#
# Requiere: procesado/demografia_32.npz (punto 32)
# Guarda (archivos NUEVOS):
# - resultados/modelo_a_demografico_33.csv
# - resultados/modelo_a_demografico_33.json
# - resultados/probabilidades_validation_modelo_a_33.npy
# ============================================================

CARPETA_PROCESADO = r"E:\Tesis_PulseDB\procesado"
CARPETA_RESULTADOS = r"E:\Tesis_PulseDB\resultados"

RUTA_DEMO = os.path.join(CARPETA_PROCESADO, "demografia_32.npz")
RUTA_PROB_B = os.path.join(CARPETA_RESULTADOS, "1_cnn_residual", "probabilidades_validation_v2_140_90.npy")
RUTA_Y_B = os.path.join(CARPETA_RESULTADOS, "1_cnn_residual", "etiquetas_validation_v2_140_90.npy")

RUTA_CSV = os.path.join(CARPETA_RESULTADOS, "5_experimentos", "demografia", "modelo_a_demografico_33.csv")
RUTA_JSON = os.path.join(CARPETA_RESULTADOS, "5_experimentos", "demografia", "modelo_a_demografico_33.json")
RUTA_PROB_A = os.path.join(CARPETA_RESULTADOS, "5_experimentos", "demografia", "probabilidades_validation_modelo_a_33.npy")

SEED = 42
N_BOOTSTRAP = 1000

# Conjuntos de variables a comparar (el principal es edad + sexo + IMC).
CONJUNTOS = {
    "edad": ["edad"],
    "edad + sexo": ["edad", "sexo_masculino"],
    "edad + sexo + IMC (principal)": ["edad", "sexo_masculino", "imc"],
    "edad + sexo + peso + talla": ["edad", "sexo_masculino", "peso", "talla"],
}

PRINCIPAL = "edad + sexo + IMC (principal)"

print("=" * 78)
print("PUNTO 33 - MODELO A: REGRESIÓN LOGÍSTICA CON DEMOGRAFÍA")
print("=" * 78)

# ============================================================
# DATOS
# ============================================================

demo = np.load(RUTA_DEMO, allow_pickle=True)

indices_train = np.load(os.path.join(CARPETA_PROCESADO, "indices_train.npy")).astype(np.int64)
indices_val = np.load(os.path.join(CARPETA_PROCESADO, "indices_val.npy")).astype(np.int64)
y_train = np.load(os.path.join(CARPETA_PROCESADO, "etiquetas_train_140_90.npy")).astype(int)
y_val = np.load(os.path.join(CARPETA_PROCESADO, "etiquetas_val_140_90.npy")).astype(int)

sujeto_val = demo["sujeto"][indices_val]

# Modelo B (CNN V2, solo ECG): probabilidades de validación del punto 24.
prob_b = np.load(RUTA_PROB_B).astype(np.float64)
y_b = np.load(RUTA_Y_B).astype(int)

if len(prob_b) != len(y_val) or not np.array_equal(y_b, y_val):
    raise ValueError("Las probabilidades del Modelo B no están alineadas con indices_val.")

print(f"Train: {len(y_train):,} segmentos | Validation: {len(y_val):,} segmentos "
      f"({len(np.unique(sujeto_val))} sujetos)")


def matriz(variables, idx):
    return np.column_stack([demo[v][idx].astype(np.float64) for v in variables])


# ============================================================
# BOOTSTRAP POR SUJETO
# ============================================================

sujetos_unicos, codigo_sujeto = np.unique(sujeto_val, return_inverse=True)
posiciones_por_sujeto = [np.flatnonzero(codigo_sujeto == k) for k in range(len(sujetos_unicos))]

rng = np.random.default_rng(SEED)
remuestreos = [
    np.concatenate([posiciones_por_sujeto[k] for k in rng.integers(0, len(sujetos_unicos), len(sujetos_unicos))])
    for _ in range(N_BOOTSTRAP)
]


def bootstrap_auc(prob, y):
    valores = []
    for idx in remuestreos:
        if y[idx].min() != y[idx].max():
            valores.append(roc_auc_score(y[idx], prob[idx]))
    return np.percentile(valores, [2.5, 97.5])


def bootstrap_diferencia(prob_1, prob_2, y):
    """IC 95 % de AUC(prob_1) - AUC(prob_2), remuestreando sujetos (pareado)."""
    valores = []
    for idx in remuestreos:
        if y[idx].min() != y[idx].max():
            valores.append(roc_auc_score(y[idx], prob_1[idx]) - roc_auc_score(y[idx], prob_2[idx]))
    valores = np.asarray(valores)
    return np.percentile(valores, [2.5, 97.5]), float(np.mean(valores <= 0))


# ============================================================
# ENTRENAMIENTO DE CADA CONJUNTO
# ============================================================

filas = []
modelos = {}

for nombre, variables in CONJUNTOS.items():

    X_tr = matriz(variables, indices_train)
    X_va = matriz(variables, indices_val)

    # Imputación con la mediana de Train (si hubiera faltantes).
    medianas = np.nanmedian(X_tr, axis=0)
    for j in range(X_tr.shape[1]):
        X_tr[~np.isfinite(X_tr[:, j]), j] = medianas[j]
        X_va[~np.isfinite(X_va[:, j]), j] = medianas[j]

    # Normalización con media y desviación de Train.
    media = X_tr.mean(axis=0)
    std = X_tr.std(axis=0)
    std[std < 1e-9] = 1.0

    Z_tr = (X_tr - media) / std
    Z_va = (X_va - media) / std

    modelo = LogisticRegression(class_weight="balanced", max_iter=1000)
    modelo.fit(Z_tr, y_train)

    prob_tr = modelo.predict_proba(Z_tr)[:, 1]
    prob_va = modelo.predict_proba(Z_va)[:, 1]

    auc_tr = roc_auc_score(y_train, prob_tr)
    auc_va = roc_auc_score(y_val, prob_va)
    pr_va = average_precision_score(y_val, prob_va)
    ic = bootstrap_auc(prob_va, y_val)

    print(
        f"\n{nombre}\n"
        f"  AUC train = {auc_tr:.4f} | AUC val = {auc_va:.4f} "
        f"[IC95 {ic[0]:.4f} – {ic[1]:.4f}] | PR-AUC val = {pr_va:.4f}"
    )
    for v, c in zip(variables, modelo.coef_[0]):
        print(f"    coeficiente {v:<16} {c:+.4f}  (variable estandarizada)")

    filas.append({
        "modelo": f"A: {nombre}",
        "variables": ", ".join(variables),
        "auc_train": auc_tr,
        "auc_val": auc_va,
        "auc_val_ic_inf": ic[0],
        "auc_val_ic_sup": ic[1],
        "pr_auc_val": pr_va,
    })

    modelos[nombre] = {
        "variables": variables,
        "medianas_train": medianas.tolist(),
        "media_train": media.tolist(),
        "std_train": std.tolist(),
        "coeficientes": modelo.coef_[0].tolist(),
        "intercepto": float(modelo.intercept_[0]),
        "prob_val": prob_va,
    }

# ============================================================
# COMPARACIÓN CON EL MODELO B (CNN V2, SOLO ECG)
# ============================================================

auc_b = roc_auc_score(y_val, prob_b)
pr_b = average_precision_score(y_val, prob_b)
ic_b = bootstrap_auc(prob_b, y_val)

filas.append({
    "modelo": "B: CNN V2 (solo ECG)",
    "variables": "ECG",
    "auc_train": np.nan,
    "auc_val": auc_b,
    "auc_val_ic_inf": ic_b[0],
    "auc_val_ic_sup": ic_b[1],
    "pr_auc_val": pr_b,
})

prob_a = modelos[PRINCIPAL]["prob_val"]
auc_a = roc_auc_score(y_val, prob_a)
ic_dif, p_b_no_supera = bootstrap_diferencia(prob_b, prob_a, y_val)

# Correlación entre las predicciones: si es baja, ECG y demografía
# aportan información distinta y un modelo híbrido podría sumar ambas.
correlacion = float(np.corrcoef(prob_a, prob_b)[0, 1])

print("\n" + "=" * 78)
print("COMPARACIÓN EN VALIDATION (mismos segmentos, mismos sujetos)")
print("=" * 78)
print(f"Modelo A ({PRINCIPAL}): AUC = {auc_a:.4f}")
print(f"Modelo B (CNN V2, solo ECG):              AUC = {auc_b:.4f}  [IC95 {ic_b[0]:.4f} – {ic_b[1]:.4f}]")
print(f"Diferencia B − A: {auc_b - auc_a:+.4f}  [IC95 {ic_dif[0]:+.4f} – {ic_dif[1]:+.4f}]")
print(f"Proporción de remuestreos donde B no supera a A: {p_b_no_supera:.3f}")
print(f"Correlación entre probabilidades A y B: {correlacion:.3f}")

ic_a = bootstrap_auc(prob_a, y_val)

if ic_a[0] <= 0.5:
    # La correlación baja no significa "información complementaria"
    # si el Modelo A no supera al azar.
    lectura = ("La demografía sola NO supera al azar (el IC 95 % incluye 0.5): "
               "su aporte esperado en un modelo híbrido es bajo.")
elif correlacion < 0.5:
    lectura = ("Correlación baja y Modelo A mejor que el azar: ECG y demografía "
               "capturan información distinta; un modelo híbrido podría mejorar.")
else:
    lectura = ("Correlación alta: ECG y demografía capturan información parecida; "
               "la mejora del Modelo C podría ser limitada.")
print(f"Lectura: {lectura}")

# ============================================================
# GUARDAR
# ============================================================

pd.DataFrame(filas).to_csv(RUTA_CSV, index=False)
np.save(RUTA_PROB_A, prob_a.astype(np.float32))

with open(RUTA_JSON, "w", encoding="utf-8") as f:
    json.dump(
        {
            "principal": PRINCIPAL,
            "modelos": {k: {kk: vv for kk, vv in v.items() if kk != "prob_val"} for k, v in modelos.items()},
            "comparacion": {
                "auc_a": float(auc_a),
                "auc_b": float(auc_b),
                "diferencia_b_menos_a": float(auc_b - auc_a),
                "ic95_diferencia": [float(ic_dif[0]), float(ic_dif[1])],
                "correlacion_a_b": correlacion,
                "lectura": lectura,
            },
            "bootstrap": {"remuestreos": N_BOOTSTRAP, "unidad": "sujeto", "seed": SEED},
        },
        f,
        ensure_ascii=False,
        indent=2,
    )

print("\n" + "=" * 78)
print("PUNTO 33 FINALIZADO")
print("=" * 78)
print(f"Tabla: {RUTA_CSV}")
print(f"Detalle: {RUTA_JSON}")
print(
    "\nIMPORTANTE:"
    "\n- La CNN no fue reentrenada (se usan las probabilidades del punto 24)."
    "\n- Normalización e imputación solo con Train."
    "\n- CalFree NO fue utilizado."
)
