import os
import h5py
import numpy as np
import pandas as pd

# =========================================================
# CONFIGURACIÓN
# =========================================================

RUTA_MAT = r"E:\Tesis_PulseDB\data_train\VitalDB_Train_Subset.mat"

CARPETA_PROCESADO = r"E:\Tesis_PulseDB\procesado"
CARPETA_RESULTADOS = r"E:\Tesis_PulseDB\resultados"

RUTA_INDICES_VAL = os.path.join(
    CARPETA_PROCESADO,
    "indices_val.npy"
)

RUTA_PROBS = os.path.join(
    CARPETA_RESULTADOS,
    "1_cnn_residual", "probabilidades_validation_v2_140_90.npy"
)

RUTA_LABELS = os.path.join(
    CARPETA_RESULTADOS,
    "1_cnn_residual", "etiquetas_validation_v2_140_90.npy"
)

RUTA_SALIDA = os.path.join(
    CARPETA_RESULTADOS,
    "1_cnn_residual", "metricas_umbral_por_sujeto_140_90.csv"
)

# Umbrales que queremos comparar
UMBRALES = [
    0.4262,  # Sensibilidad global >= 80%
    0.5000,  # Umbral convencional
    0.6052,  # Mejor Youden
    0.7695   # Mejor F1
]

print("=" * 75)
print("PUNTO 25 - ANÁLISIS DE UMBRALES POR SUJETO - 140/90")
print("=" * 75)

# =========================================================
# CARGAR DATOS
# =========================================================

indices_val = np.load(
    RUTA_INDICES_VAL
)

probabilidades = np.load(
    RUTA_PROBS
)

etiquetas = np.load(
    RUTA_LABELS
).astype(int)

print(f"\nSegmentos Validation: {len(indices_val):,}")
print(f"Probabilidades:       {len(probabilidades):,}")
print(f"Etiquetas:            {len(etiquetas):,}")

if not (
    len(indices_val)
    == len(probabilidades)
    == len(etiquetas)
):
    raise ValueError(
        "Las longitudes de índices, probabilidades "
        "y etiquetas no coinciden."
    )

# =========================================================
# OBTENER SUJETO DE CADA SEGMENTO
# =========================================================

print("\nLeyendo identificadores de sujetos...")

with h5py.File(RUTA_MAT, "r") as f:

    subject_dataset = f["Subset"]["Subject"]

    referencias = np.asarray(
        subject_dataset[
            0,
            indices_val
        ]
    ).reshape(-1)

    sujetos_lista = []

    for ref in referencias:

        valor = np.asarray(
            f[ref]
        ).reshape(-1)

        # Convertir el identificador referenciado
        # a una representación comparable.
        if valor.dtype.kind in ("u", "i"):

            try:
                sujeto = "".join(
                    chr(int(x))
                    for x in valor
                    if int(x) != 0
                )

            except Exception:
                sujeto = str(
                    valor.tolist()
                )

        else:

            sujeto = str(
                valor.tolist()
            )

        sujetos_lista.append(
            sujeto
        )

sujetos = np.asarray(
    sujetos_lista,
    dtype=str
)

print(
    f"Sujetos únicos encontrados: "
    f"{len(np.unique(sujetos))}"
)

# =========================================================
# ANALIZAR CADA UMBRAL
# =========================================================

filas = []

for umbral in UMBRALES:

    print("\n" + "=" * 75)
    print(f"UMBRAL {umbral:.4f}")
    print("=" * 75)

    sensibilidades = []
    especificidades = []

    sujetos_con_positivos = 0
    sujetos_con_negativos = 0

    for sujeto in np.unique(sujetos):

        mascara = (
            sujetos == sujeto
        )

        y_true = etiquetas[mascara]

        probs = probabilidades[mascara]

        y_pred = (
            probs >= umbral
        ).astype(int)

        positivos = np.sum(
            y_true == 1
        )

        negativos = np.sum(
            y_true == 0
        )

        # ---------------------------------------------
        # SENSIBILIDAD
        # ---------------------------------------------

        if positivos > 0:

            sujetos_con_positivos += 1

            tp = np.sum(
                (y_true == 1)
                &
                (y_pred == 1)
            )

            fn = np.sum(
                (y_true == 1)
                &
                (y_pred == 0)
            )

            sensibilidad = (
                tp / (tp + fn)
            )

            sensibilidades.append(
                sensibilidad
            )

        # ---------------------------------------------
        # ESPECIFICIDAD
        # ---------------------------------------------

        if negativos > 0:

            sujetos_con_negativos += 1

            tn = np.sum(
                (y_true == 0)
                &
                (y_pred == 0)
            )

            fp = np.sum(
                (y_true == 0)
                &
                (y_pred == 1)
            )

            especificidad = (
                tn / (tn + fp)
            )

            especificidades.append(
                especificidad
            )

    sensibilidades = np.asarray(
        sensibilidades
    )

    especificidades = np.asarray(
        especificidades
    )

    # =====================================================
    # RESUMEN
    # =====================================================

    sens_media = np.mean(
        sensibilidades
    )

    sens_mediana = np.median(
        sensibilidades
    )

    spec_media = np.mean(
        especificidades
    )

    spec_mediana = np.median(
        especificidades
    )

    porcentaje_sens50 = (
        np.mean(
            sensibilidades >= 0.50
        )
        * 100
    )

    porcentaje_sens70 = (
        np.mean(
            sensibilidades >= 0.70
        )
        * 100
    )

    porcentaje_sens80 = (
        np.mean(
            sensibilidades >= 0.80
        )
        * 100
    )

    print(
        f"Sujetos con segmentos positivos: "
        f"{sujetos_con_positivos}"
    )

    print(
        f"Sujetos con segmentos negativos: "
        f"{sujetos_con_negativos}"
    )

    print("\nSensibilidad por sujeto:")

    print(
        f"Media:   {sens_media:.4f}"
    )

    print(
        f"Mediana: {sens_mediana:.4f}"
    )

    print("\nEspecificidad por sujeto:")

    print(
        f"Media:   {spec_media:.4f}"
    )

    print(
        f"Mediana: {spec_mediana:.4f}"
    )

    print(
        "\nPorcentaje de sujetos positivos "
        "con sensibilidad >= 50%:"
    )

    print(
        f"{porcentaje_sens50:.2f}%"
    )

    print(
        "Porcentaje de sujetos positivos "
        "con sensibilidad >= 70%:"
    )

    print(
        f"{porcentaje_sens70:.2f}%"
    )

    print(
        "Porcentaje de sujetos positivos "
        "con sensibilidad >= 80%:"
    )

    print(
        f"{porcentaje_sens80:.2f}%"
    )

    filas.append({
        "umbral": umbral,
        "sujetos_con_positivos":
            sujetos_con_positivos,
        "sujetos_con_negativos":
            sujetos_con_negativos,
        "sensibilidad_media_sujeto":
            sens_media,
        "sensibilidad_mediana_sujeto":
            sens_mediana,
        "especificidad_media_sujeto":
            spec_media,
        "especificidad_mediana_sujeto":
            spec_mediana,
        "porcentaje_sujetos_sens_ge_50":
            porcentaje_sens50,
        "porcentaje_sujetos_sens_ge_70":
            porcentaje_sens70,
        "porcentaje_sujetos_sens_ge_80":
            porcentaje_sens80
    })

# =========================================================
# GUARDAR
# =========================================================

df = pd.DataFrame(
    filas
)

df.to_csv(
    RUTA_SALIDA,
    index=False
)

print("\n" + "=" * 75)
print("RESUMEN FINAL")
print("=" * 75)

print(
    df.to_string(
        index=False
    )
)

print(
    f"\nResultados guardados en:\n"
    f"{RUTA_SALIDA}"
)

print(
    "\nIMPORTANTE:"
)

print(
    "Este análisis utilizó únicamente Validation."
)

print(
    "No se realizó ningún entrenamiento."
)

print(
    "CalFree NO fue utilizado para seleccionar "
    "el umbral."
)

print(
    "\nPUNTO 25 FINALIZADO."
)