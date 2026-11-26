import os
import h5py
import numpy as np

# =========================================================
# CONFIGURACIÓN
# =========================================================

RUTA_MAT = r"E:\Tesis_PulseDB\data_train\VitalDB_Train_Subset.mat"
CARPETA_PROCESADO = r"E:\Tesis_PulseDB\procesado"

RUTA_INDICES_TRAIN = os.path.join(
    CARPETA_PROCESADO,
    "indices_train.npy"
)

RUTA_ETIQUETAS_TRAIN = os.path.join(
    CARPETA_PROCESADO,
    "etiquetas_train.npy"
)

# Analizaremos una muestra suficientemente grande
# sin cargar los 372 mil ECG completos en RAM.
N_MUESTRA = 10000

SEMILLA = 42

rng = np.random.default_rng(SEMILLA)


# =========================================================
# CARGAR ÍNDICES Y ETIQUETAS
# =========================================================

indices_train = np.load(
    RUTA_INDICES_TRAIN
)

etiquetas_train = np.load(
    RUTA_ETIQUETAS_TRAIN
)

print("========================================")
print("ANÁLISIS ECG - TRAIN")
print("========================================")

print(
    "Segmentos Train disponibles:",
    len(indices_train)
)

print(
    "Positivos:",
    int(etiquetas_train.sum())
)

print(
    "Negativos:",
    int(
        len(etiquetas_train)
        - etiquetas_train.sum()
    )
)


# =========================================================
# SELECCIONAR MUESTRA BALANCEADA
# =========================================================

posiciones_positivas = np.where(
    etiquetas_train == 1
)[0]

posiciones_negativas = np.where(
    etiquetas_train == 0
)[0]

n_por_clase = N_MUESTRA // 2

muestra_pos = rng.choice(
    posiciones_positivas,
    size=n_por_clase,
    replace=False
)

muestra_neg = rng.choice(
    posiciones_negativas,
    size=n_por_clase,
    replace=False
)

posiciones_muestra = np.concatenate(
    [
        muestra_neg,
        muestra_pos
    ]
)

rng.shuffle(
    posiciones_muestra
)

indices_reales = indices_train[
    posiciones_muestra
]

etiquetas_muestra = etiquetas_train[
    posiciones_muestra
]

print("\n===== MUESTRA =====")

print(
    "Total:",
    len(indices_reales)
)

print(
    "Clase 0:",
    int(
        np.sum(etiquetas_muestra == 0)
    )
)

print(
    "Clase 1:",
    int(
        np.sum(etiquetas_muestra == 1)
    )
)


# =========================================================
# EXTRAER ESTADÍSTICAS
# =========================================================

medias = []
desviaciones = []
minimos = []
maximos = []
rangos = []

# Diferencia absoluta entre muestras consecutivas.
# Nos da una medida simple de cuánto cambia la señal.
variacion_media = []

# Porcentaje de valores exactamente en 0 o 1.
# Es útil porque vimos que las señales están en [0, 1].
porcentaje_ceros = []
porcentaje_unos = []


print("\nLeyendo ECG del archivo .mat...")

with h5py.File(
    RUTA_MAT,
    "r"
) as archivo:

    signals = archivo[
        "Subset"
    ]["Signals"]

    for contador, indice_real in enumerate(
        indices_reales,
        start=1
    ):

        ecg = np.asarray(
            signals[:, 0, indice_real],
            dtype=np.float32
        )

        media = np.mean(ecg)
        std = np.std(ecg)

        minimo = np.min(ecg)
        maximo = np.max(ecg)

        rango = maximo - minimo

        diferencias = np.abs(
            np.diff(ecg)
        )

        var_media = np.mean(
            diferencias
        )

        pct_cero = np.mean(
            ecg == 0.0
        )

        pct_uno = np.mean(
            ecg == 1.0
        )

        medias.append(media)
        desviaciones.append(std)
        minimos.append(minimo)
        maximos.append(maximo)
        rangos.append(rango)
        variacion_media.append(var_media)
        porcentaje_ceros.append(pct_cero)
        porcentaje_unos.append(pct_uno)

        if contador % 2000 == 0:

            print(
                f"Procesados: "
                f"{contador}/{len(indices_reales)}"
            )


# =========================================================
# CONVERTIR A NUMPY
# =========================================================

medias = np.asarray(medias)
desviaciones = np.asarray(desviaciones)
minimos = np.asarray(minimos)
maximos = np.asarray(maximos)
rangos = np.asarray(rangos)

variacion_media = np.asarray(
    variacion_media
)

porcentaje_ceros = np.asarray(
    porcentaje_ceros
)

porcentaje_unos = np.asarray(
    porcentaje_unos
)


# =========================================================
# FUNCIÓN DE RESUMEN
# =========================================================

def resumen(nombre, valores):

    print(f"\n{nombre}")

    print(
        f"Media:       "
        f"{np.mean(valores):.6f}"
    )

    print(
        f"Desv. Std:   "
        f"{np.std(valores):.6f}"
    )

    print(
        f"Mínimo:      "
        f"{np.min(valores):.6f}"
    )

    print(
        f"P25:         "
        f"{np.percentile(valores, 25):.6f}"
    )

    print(
        f"Mediana:     "
        f"{np.median(valores):.6f}"
    )

    print(
        f"P75:         "
        f"{np.percentile(valores, 75):.6f}"
    )

    print(
        f"Máximo:      "
        f"{np.max(valores):.6f}"
    )


# =========================================================
# RESULTADOS GENERALES
# =========================================================

print("\n========================================")
print("ESTADÍSTICAS GENERALES")
print("========================================")

resumen(
    "Media de cada ECG:",
    medias
)

resumen(
    "Desviación estándar de cada ECG:",
    desviaciones
)

resumen(
    "Rango (máximo - mínimo):",
    rangos
)

resumen(
    "Variación media entre muestras consecutivas:",
    variacion_media
)

resumen(
    "Proporción de valores exactamente 0:",
    porcentaje_ceros
)

resumen(
    "Proporción de valores exactamente 1:",
    porcentaje_unos
)


# =========================================================
# COMPARACIÓN CLASE 0 VS CLASE 1
# =========================================================

mask0 = (
    etiquetas_muestra == 0
)

mask1 = (
    etiquetas_muestra == 1
)


def comparar(
    nombre,
    valores
):

    valores0 = valores[mask0]
    valores1 = valores[mask1]

    print(f"\n{nombre}")

    print(
        "Clase 0 | "
        f"media={np.mean(valores0):.6f} | "
        f"mediana={np.median(valores0):.6f}"
    )

    print(
        "Clase 1 | "
        f"media={np.mean(valores1):.6f} | "
        f"mediana={np.median(valores1):.6f}"
    )


print("\n========================================")
print("COMPARACIÓN CLASE 0 VS CLASE 1")
print("========================================")

comparar(
    "Media del ECG:",
    medias
)

comparar(
    "Desviación estándar:",
    desviaciones
)

comparar(
    "Rango:",
    rangos
)

comparar(
    "Variación media:",
    variacion_media
)

comparar(
    "Valores exactamente 0:",
    porcentaje_ceros
)

comparar(
    "Valores exactamente 1:",
    porcentaje_unos
)


# =========================================================
# POSIBLES SEÑALES PROBLEMÁTICAS
# =========================================================

std_muy_baja = np.sum(
    desviaciones < 0.01
)

rango_muy_bajo = np.sum(
    rangos < 0.10
)

muchos_ceros = np.sum(
    porcentaje_ceros > 0.10
)

muchos_unos = np.sum(
    porcentaje_unos > 0.10
)


print("\n========================================")
print("COMPROBACIONES BÁSICAS")
print("========================================")

print(
    "ECG con std < 0.01:",
    std_muy_baja,
    "/",
    len(desviaciones)
)

print(
    "ECG con rango < 0.10:",
    rango_muy_bajo,
    "/",
    len(rangos)
)

print(
    "ECG con >10% valores en 0:",
    muchos_ceros,
    "/",
    len(porcentaje_ceros)
)

print(
    "ECG con >10% valores en 1:",
    muchos_unos,
    "/",
    len(porcentaje_unos)
)


# =========================================================
# EFECTO DE ESTANDARIZACIÓN POR SEGMENTO
# =========================================================

print("\n========================================")
print("PRUEBA DE ESTANDARIZACIÓN")
print("========================================")

ejemplos = indices_reales[:5]

with h5py.File(
    RUTA_MAT,
    "r"
) as archivo:

    signals = archivo[
        "Subset"
    ]["Signals"]

    for numero, indice_real in enumerate(
        ejemplos,
        start=1
    ):

        ecg = np.asarray(
            signals[:, 0, indice_real],
            dtype=np.float32
        )

        media_original = np.mean(ecg)
        std_original = np.std(ecg)

        ecg_z = (
            ecg - media_original
        ) / (
            std_original + 1e-8
        )

        print(
            f"\nECG {numero}"
        )

        print(
            f"Original -> "
            f"media={np.mean(ecg):.6f}, "
            f"std={np.std(ecg):.6f}, "
            f"min={np.min(ecg):.6f}, "
            f"max={np.max(ecg):.6f}"
        )

        print(
            f"Z-score  -> "
            f"media={np.mean(ecg_z):.6f}, "
            f"std={np.std(ecg_z):.6f}, "
            f"min={np.min(ecg_z):.6f}, "
            f"max={np.max(ecg_z):.6f}"
        )


print("\n========================================")
print("ANÁLISIS TERMINADO")
print("========================================")