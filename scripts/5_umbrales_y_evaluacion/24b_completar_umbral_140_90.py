import os
import numpy as np

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve
)

# =========================================================
# RUTAS
# =========================================================

CARPETA_RESULTADOS = r"E:\Tesis_PulseDB\resultados"

RUTA_PROBS = os.path.join(
    CARPETA_RESULTADOS,
    "1_cnn_residual", "probabilidades_validation_v2_140_90.npy"
)

RUTA_LABELS = os.path.join(
    CARPETA_RESULTADOS,
    "1_cnn_residual", "etiquetas_validation_v2_140_90.npy"
)

# =========================================================
# CARGAR RESULTADOS YA CALCULADOS
# =========================================================

probabilidades = np.load(
    RUTA_PROBS
)

etiquetas = np.load(
    RUTA_LABELS
).astype(int)

print("=" * 70)
print("PUNTO 24B - COMPLETAR ANÁLISIS DE UMBRAL 140/90")
print("=" * 70)

print(
    f"\nSegmentos: {len(etiquetas):,}"
)

print(
    f"Positivos: {np.sum(etiquetas == 1):,}"
)

print(
    f"Negativos: {np.sum(etiquetas == 0):,}"
)

# =========================================================
# FUNCIÓN DE MÉTRICAS
# =========================================================

def evaluar(umbral):

    pred = (
        probabilidades >= umbral
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        etiquetas,
        pred
    ).ravel()

    accuracy = accuracy_score(
        etiquetas,
        pred
    )

    precision = precision_score(
        etiquetas,
        pred,
        zero_division=0
    )

    sensibilidad = recall_score(
        etiquetas,
        pred,
        zero_division=0
    )

    especificidad = (
        tn / (tn + fp)
    )

    f1 = f1_score(
        etiquetas,
        pred,
        zero_division=0
    )

    youden = (
        sensibilidad
        + especificidad
        - 1
    )

    return {
        "umbral": umbral,
        "accuracy": accuracy,
        "precision": precision,
        "sensibilidad": sensibilidad,
        "especificidad": especificidad,
        "f1": f1,
        "youden": youden,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    }


def mostrar(nombre, r):

    print("\n" + "-" * 70)

    print(nombre)

    print("-" * 70)

    print(
        f"Umbral:       {r['umbral']:.4f}"
    )

    print(
        f"Accuracy:     {r['accuracy']:.4f}"
    )

    print(
        f"Precision:    {r['precision']:.4f}"
    )

    print(
        f"Sensibilidad: {r['sensibilidad']:.4f}"
    )

    print(
        f"Especificidad:{r['especificidad']: .4f}"
    )

    print(
        f"F1:           {r['f1']:.4f}"
    )

    print(
        f"Youden:       {r['youden']:.4f}"
    )

    print(
        f"TN={r['tn']:,} | "
        f"FP={r['fp']:,} | "
        f"FN={r['fn']:,} | "
        f"TP={r['tp']:,}"
    )

# =========================================================
# CURVA ROC
# =========================================================

fpr, tpr, thresholds = roc_curve(
    etiquetas,
    probabilidades
)

# =========================================================
# MAYOR ESPECIFICIDAD MANTENIENDO
# SENSIBILIDAD >= 80 %
# =========================================================

indices_validos = np.where(
    tpr >= 0.80
)[0]

if len(indices_validos) == 0:

    print(
        "\nNo existe un umbral con "
        "sensibilidad >= 80%."
    )

else:

    # Dentro de los puntos con sensibilidad >= 80%,
    # buscamos el menor FPR = mayor especificidad.

    mejor_indice = indices_validos[
        np.argmin(
            fpr[indices_validos]
        )
    ]

    umbral_sens80 = thresholds[
        mejor_indice
    ]

    resultado_sens80 = evaluar(
        umbral_sens80
    )

    mostrar(
        "SENSIBILIDAD >= 80% "
        "CON MAYOR ESPECIFICIDAD",
        resultado_sens80
    )

# =========================================================
# REFERENCIAS IMPORTANTES
# =========================================================

mostrar(
    "UMBRAL 0.50",
    evaluar(0.50)
)

mostrar(
    "UMBRAL YOUDEN 0.6052",
    evaluar(0.6052)
)

mostrar(
    "UMBRAL F1 0.7695",
    evaluar(0.7695)
)

# =========================================================
# ALGUNOS PUNTOS INTERMEDIOS
# =========================================================

print("\n" + "=" * 70)
print("REFERENCIA DE UMBRALES")
print("=" * 70)

for umbral in [
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80
]:

    r = evaluar(
        umbral
    )

    print(
        f"{umbral:.2f} | "
        f"Prec={r['precision']:.4f} | "
        f"Sens={r['sensibilidad']:.4f} | "
        f"Spec={r['especificidad']:.4f} | "
        f"F1={r['f1']:.4f}"
    )

print("\nIMPORTANTE:")
print(
    "Se utilizaron probabilidades ya generadas "
    "sobre Validation."
)

print(
    "No se volvió a entrenar el modelo."
)

print(
    "CalFree NO fue utilizado."
)

print(
    "\nPUNTO 24B FINALIZADO."
)