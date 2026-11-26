import os
import h5py
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score
)

# =========================================================
# RUTAS
# =========================================================

RUTA_MAT = r"E:\Tesis_PulseDB\data_train\VitalDB_Train_Subset.mat"
CARPETA_PROCESADO = r"E:\Tesis_PulseDB\procesado"
CARPETA_RESULTADOS = r"E:\Tesis_PulseDB\resultados"

RUTA_INDICES = os.path.join(
    CARPETA_PROCESADO,
    "indices_val.npy"
)

RUTA_PROB = os.path.join(
    CARPETA_RESULTADOS,
    "probabilidades_validation_v2_regularizada.npy"
)

RUTA_ETIQUETAS = os.path.join(
    CARPETA_RESULTADOS,
    "etiquetas_validation_v2_regularizada.npy"
)

RUTA_SALIDA = os.path.join(
    CARPETA_RESULTADOS,
    "analisis_por_sujeto_validation.csv"
)

# =========================================================
# CARGAR DATOS
# =========================================================

print("=" * 55)
print("PUNTO 20 - ANÁLISIS POR SUJETO EN VALIDATION")
print("=" * 55)

indices = np.load(RUTA_INDICES)
probabilidades = np.load(RUTA_PROB)
etiquetas = np.load(RUTA_ETIQUETAS)

print(f"Segmentos: {len(indices):,}")
print(f"Probabilidades: {len(probabilidades):,}")
print(f"Etiquetas: {len(etiquetas):,}")

if not (
    len(indices)
    == len(probabilidades)
    == len(etiquetas)
):
    raise ValueError(
        "Los tamaños de índices, probabilidades "
        "y etiquetas no coinciden."
    )

# =========================================================
# OBTENER SUBJECT ID
# =========================================================

print("\nLeyendo identificadores de sujetos...")

sujetos = []

with h5py.File(RUTA_MAT, "r") as f:

    subject_dataset = f["Subset"]["Subject"]

    total = len(indices)

    for posicion, indice_real in enumerate(indices, start=1):

        referencia = subject_dataset[0, indice_real]

        datos_subject = np.asarray(
            f[referencia]
        ).flatten()

        subject_id = "".join(
            chr(int(codigo))
            for codigo in datos_subject
            if int(codigo) != 0
        )

        sujetos.append(subject_id)

        if posicion % 10000 == 0:
            print(
                f"Subjects leídos: "
                f"{posicion:,}/{total:,}"
            )

sujetos = np.asarray(
    sujetos,
    dtype=str
)

print(
    f"Sujetos únicos encontrados: "
    f"{len(np.unique(sujetos)):,}"
)

print(
    "Primeros IDs:",
    np.unique(sujetos)[:5]
)

# =========================================================
# DATAFRAME POR SEGMENTO
# =========================================================

df = pd.DataFrame({
    "subject": sujetos,
    "y_real": etiquetas.astype(int),
    "probabilidad": probabilidades
})

# =========================================================
# MÉTRICAS POR SUJETO
# =========================================================

filas = []

for subject, grupo in df.groupby("subject"):

    y = grupo["y_real"].to_numpy()
    p = grupo["probabilidad"].to_numpy()

    positivos = int(y.sum())
    negativos = int(len(y) - positivos)

    porcentaje_positivo = (
        positivos / len(y)
    )

    prob_media = float(
        np.mean(p)
    )

    prob_mediana = float(
        np.median(p)
    )

    # AUC solo puede calcularse si el sujeto
    # tiene ambas clases.
    if len(np.unique(y)) == 2:

        auc = float(
            roc_auc_score(y, p)
        )

        pr_auc = float(
            average_precision_score(y, p)
        )

    else:

        auc = np.nan
        pr_auc = np.nan

    filas.append({
        "subject": subject,
        "segmentos": len(y),
        "positivos": positivos,
        "negativos": negativos,
        "porcentaje_positivo": porcentaje_positivo,
        "probabilidad_media": prob_media,
        "probabilidad_mediana": prob_mediana,
        "roc_auc_sujeto": auc,
        "pr_auc_sujeto": pr_auc
    })

resultado = pd.DataFrame(filas)

resultado.to_csv(
    RUTA_SALIDA,
    index=False
)

# =========================================================
# RESUMEN
# =========================================================

solo_negativos = (
    resultado["positivos"] == 0
).sum()

solo_positivos = (
    resultado["negativos"] == 0
).sum()

mixtos = (
    (resultado["positivos"] > 0)
    &
    (resultado["negativos"] > 0)
).sum()

auc_validos = (
    resultado["roc_auc_sujeto"]
    .dropna()
)

pr_validos = (
    resultado["pr_auc_sujeto"]
    .dropna()
)

print("\n" + "=" * 55)
print("COMPOSICIÓN DE SUJETOS")
print("=" * 55)

print(
    f"Total sujetos: "
    f"{len(resultado):,}"
)

print(
    f"Solo negativos: "
    f"{solo_negativos:,}"
)

print(
    f"Solo positivos: "
    f"{solo_positivos:,}"
)

print(
    f"Mixtos: "
    f"{mixtos:,}"
)

print("\n" + "=" * 55)
print("ROC-AUC DENTRO DE CADA SUJETO MIXTO")
print("=" * 55)

print(
    f"Sujetos con AUC calculable: "
    f"{len(auc_validos):,}"
)

if len(auc_validos) > 0:

    print(
        f"Media ROC-AUC: "
        f"{auc_validos.mean():.4f}"
    )

    print(
        f"Mediana ROC-AUC: "
        f"{auc_validos.median():.4f}"
    )

    print(
        f"Desviación estándar: "
        f"{auc_validos.std():.4f}"
    )

    print(
        f"Mínimo: "
        f"{auc_validos.min():.4f}"
    )

    print(
        f"Percentil 25: "
        f"{auc_validos.quantile(0.25):.4f}"
    )

    print(
        f"Percentil 75: "
        f"{auc_validos.quantile(0.75):.4f}"
    )

    print(
        f"Máximo: "
        f"{auc_validos.max():.4f}"
    )

    print(
        f"Sujetos AUC >= 0.50: "
        f"{(auc_validos >= 0.50).sum():,} "
        f"({(auc_validos >= 0.50).mean()*100:.2f}%)"
    )

    print(
        f"Sujetos AUC >= 0.60: "
        f"{(auc_validos >= 0.60).sum():,} "
        f"({(auc_validos >= 0.60).mean()*100:.2f}%)"
    )

    print(
        f"Sujetos AUC >= 0.70: "
        f"{(auc_validos >= 0.70).sum():,} "
        f"({(auc_validos >= 0.70).mean()*100:.2f}%)"
    )

print("\n" + "=" * 55)
print("PR-AUC DENTRO DE CADA SUJETO MIXTO")
print("=" * 55)

if len(pr_validos) > 0:

    print(
        f"Media PR-AUC: "
        f"{pr_validos.mean():.4f}"
    )

    print(
        f"Mediana PR-AUC: "
        f"{pr_validos.median():.4f}"
    )

print("\n" + "=" * 55)
print("PROBABILIDAD MEDIA POR SUJETO")
print("=" * 55)

print(
    resultado[
        "probabilidad_media"
    ].describe()
)

print("\nArchivo guardado:")
print(RUTA_SALIDA)

print("\nIMPORTANTE:")
print(
    "Este análisis utiliza únicamente Validation."
)
print(
    "CalFree continúa completamente sin utilizar."
)

print("\nANÁLISIS FINALIZADO.")