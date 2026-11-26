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

RUTA_INDICES = os.path.join(
    CARPETA_PROCESADO,
    "indices_val.npy"
)

RUTA_PROBABILIDADES = os.path.join(
    CARPETA_RESULTADOS,
    "probabilidades_validation_v2_regularizada.npy"
)

RUTA_ETIQUETAS = os.path.join(
    CARPETA_RESULTADOS,
    "etiquetas_validation_v2_regularizada.npy"
)

RUTA_SALIDA = os.path.join(
    CARPETA_RESULTADOS,
    "metricas_umbral_por_sujeto_validation.csv"
)

# Umbrales importantes encontrados en Punto 19
UMBRALES = [
    0.2145,  # sensibilidad aproximada 80 %
    0.30,
    0.40,
    0.50,
    0.5546,  # mejor Youden
    0.60,
    0.70
]

# =========================================================
# CARGAR DATOS
# =========================================================

print("=" * 65)
print("PUNTO 21 - ANÁLISIS DE UMBRALES ENTRE SUJETOS")
print("=" * 65)

indices = np.load(RUTA_INDICES)
probabilidades = np.load(RUTA_PROBABILIDADES)
etiquetas = np.load(RUTA_ETIQUETAS).astype(int)

print(f"Segmentos Validation: {len(indices):,}")

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
# RECUPERAR SUBJECT IDs
# =========================================================

print("\nRecuperando identificadores de sujetos...")

sujetos = []

with h5py.File(RUTA_MAT, "r") as f:

    subject_dataset = f["Subset"]["Subject"]

    total = len(indices)

    for posicion, indice_real in enumerate(
        indices,
        start=1
    ):

        referencia = subject_dataset[
            0,
            indice_real
        ]

        datos_subject = np.asarray(
            f[referencia]
        ).flatten()

        subject_id = "".join(
            chr(int(codigo))
            for codigo in datos_subject
            if int(codigo) != 0
        )

        sujetos.append(subject_id)

        if posicion % 20000 == 0:
            print(
                f"Leídos: "
                f"{posicion:,}/{total:,}"
            )

sujetos = np.asarray(
    sujetos,
    dtype=str
)

print(
    "Sujetos únicos:",
    len(np.unique(sujetos))
)

# =========================================================
# DATAFRAME
# =========================================================

df = pd.DataFrame({
    "subject": sujetos,
    "y_real": etiquetas,
    "probabilidad": probabilidades
})

# =========================================================
# FUNCIÓN DE MÉTRICAS POR SUJETO
# =========================================================

def metricas_sujeto(grupo, umbral):

    y = grupo["y_real"].to_numpy()

    p = grupo[
        "probabilidad"
    ].to_numpy()

    pred = (
        p >= umbral
    ).astype(int)

    tp = np.sum(
        (y == 1) & (pred == 1)
    )

    fn = np.sum(
        (y == 1) & (pred == 0)
    )

    tn = np.sum(
        (y == 0) & (pred == 0)
    )

    fp = np.sum(
        (y == 0) & (pred == 1)
    )

    sensibilidad = np.nan

    if (tp + fn) > 0:
        sensibilidad = (
            tp / (tp + fn)
        )

    especificidad = np.nan

    if (tn + fp) > 0:
        especificidad = (
            tn / (tn + fp)
        )

    return (
        sensibilidad,
        especificidad
    )

# =========================================================
# ANALIZAR CADA UMBRAL
# =========================================================

resultados = []

print("\n" + "=" * 65)
print("RESULTADOS")
print("=" * 65)

for umbral in UMBRALES:

    sensibilidades = []
    especificidades = []

    for _, grupo in df.groupby(
        "subject"
    ):

        sens, esp = metricas_sujeto(
            grupo,
            umbral
        )

        if not np.isnan(sens):
            sensibilidades.append(sens)

        if not np.isnan(esp):
            especificidades.append(esp)

    sensibilidades = np.asarray(
        sensibilidades
    )

    especificidades = np.asarray(
        especificidades
    )

    sens_media = np.mean(
        sensibilidades
    )

    sens_mediana = np.median(
        sensibilidades
    )

    esp_media = np.mean(
        especificidades
    )

    esp_mediana = np.median(
        especificidades
    )

    sujetos_sens_50 = np.mean(
        sensibilidades >= 0.50
    )

    sujetos_sens_70 = np.mean(
        sensibilidades >= 0.70
    )

    sujetos_sens_80 = np.mean(
        sensibilidades >= 0.80
    )

    resultados.append({
        "umbral": umbral,
        "sensibilidad_media_sujeto":
            sens_media,
        "sensibilidad_mediana_sujeto":
            sens_mediana,
        "especificidad_media_sujeto":
            esp_media,
        "especificidad_mediana_sujeto":
            esp_mediana,
        "proporcion_sujetos_sens_ge_50":
            sujetos_sens_50,
        "proporcion_sujetos_sens_ge_70":
            sujetos_sens_70,
        "proporcion_sujetos_sens_ge_80":
            sujetos_sens_80
    })

    print(
        f"\nUMBRAL {umbral:.4f}"
    )

    print(
        f"Sensibilidad media por sujeto:   "
        f"{sens_media:.4f}"
    )

    print(
        f"Sensibilidad mediana por sujeto: "
        f"{sens_mediana:.4f}"
    )

    print(
        f"Especificidad media por sujeto:  "
        f"{esp_media:.4f}"
    )

    print(
        f"Especificidad mediana por sujeto:"
        f" {esp_mediana:.4f}"
    )

    print(
        f"Sujetos con sensibilidad >=50%: "
        f"{sujetos_sens_50*100:.2f}%"
    )

    print(
        f"Sujetos con sensibilidad >=70%: "
        f"{sujetos_sens_70*100:.2f}%"
    )

    print(
        f"Sujetos con sensibilidad >=80%: "
        f"{sujetos_sens_80*100:.2f}%"
    )

# =========================================================
# GUARDAR
# =========================================================

resultado_df = pd.DataFrame(
    resultados
)

resultado_df.to_csv(
    RUTA_SALIDA,
    index=False
)

print("\n" + "=" * 65)
print("RESUMEN")
print("=" * 65)

print(
    resultado_df.to_string(
        index=False
    )
)

print("\nArchivo guardado:")
print(RUTA_SALIDA)

print("\nIMPORTANTE:")
print(
    "Este análisis utiliza exclusivamente Validation."
)

print(
    "CalFree continúa sin utilizar."
)

print("\nPUNTO 21 FINALIZADO.")