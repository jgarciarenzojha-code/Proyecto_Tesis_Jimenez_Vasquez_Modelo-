import h5py
import numpy as np
import os

RUTA_TRAIN = r"E:\Tesis_PulseDB\data_train\VitalDB_Train_Subset.mat"
CARPETA_PROCESADO = r"E:\Tesis_PulseDB\procesado"

indices_train = np.load(
    os.path.join(CARPETA_PROCESADO, "indices_train.npy")
)

indices_val = np.load(
    os.path.join(CARPETA_PROCESADO, "indices_val.npy")
)


with h5py.File(RUTA_TRAIN, "r") as f:

    sbp = f["Subset"]["SBP"][0, :]
    dbp = f["Subset"]["DBP"][0, :]


def analizar(nombre, indices):

    sbp_subset = sbp[indices]
    dbp_subset = dbp[indices]

    # Etiqueta provisional >=130/80
    etiquetas = (
        (sbp_subset >= 130) |
        (dbp_subset >= 80)
    ).astype(np.int8)

    positivos = np.sum(etiquetas == 1)
    negativos = np.sum(etiquetas == 0)

    total = len(etiquetas)

    print(f"\n===== {nombre} =====")
    print("Total segmentos:", total)

    print("\nClase 0 - Fuera de rango:")
    print(negativos)
    print(
        "Porcentaje:",
        round((negativos / total) * 100, 2),
        "%"
    )

    print("\nClase 1 - Rango hipertensivo:")
    print(positivos)
    print(
        "Porcentaje:",
        round((positivos / total) * 100, 2),
        "%"
    )

    print("\nRelación negativos / positivos:")
    print(round(negativos / positivos, 4))

    return etiquetas


etiquetas_train = analizar(
    "TRAIN",
    indices_train
)

etiquetas_val = analizar(
    "VALIDATION",
    indices_val
)


# Guardamos las etiquetas provisionales
np.save(
    os.path.join(
        CARPETA_PROCESADO,
        "etiquetas_train.npy"
    ),
    etiquetas_train
)

np.save(
    os.path.join(
        CARPETA_PROCESADO,
        "etiquetas_val.npy"
    ),
    etiquetas_val
)


# pos_weight SOLO usando TRAIN
positivos_train = np.sum(etiquetas_train == 1)
negativos_train = np.sum(etiquetas_train == 0)

pos_weight = negativos_train / positivos_train

print("\n===== PESO DE CLASE =====")
print("pos_weight TRAIN:", round(pos_weight, 4))

print("\nEtiquetas guardadas correctamente.")