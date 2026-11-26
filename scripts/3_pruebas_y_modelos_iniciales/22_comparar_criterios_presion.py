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

RUTA_INDICES_VAL = os.path.join(
    CARPETA_PROCESADO,
    "indices_val.npy"
)

# IMPORTANTE:
# No se sobrescriben las etiquetas actuales de 130/80.
# Las nuevas etiquetas se guardarán con nombres diferentes.

RUTA_ETIQUETAS_TRAIN_140 = os.path.join(
    CARPETA_PROCESADO,
    "etiquetas_train_140_90.npy"
)

RUTA_ETIQUETAS_VAL_140 = os.path.join(
    CARPETA_PROCESADO,
    "etiquetas_val_140_90.npy"
)

print("=" * 65)
print("PUNTO 22 - COMPARACIÓN DE CRITERIOS DE PRESIÓN")
print("=" * 65)

print("\nCriterio A:")
print("SBP >= 130 OR DBP >= 80")

print("\nCriterio B:")
print("SBP >= 140 OR DBP >= 90")

# =========================================================
# CARGAR ÍNDICES
# =========================================================

indices_train = np.load(
    RUTA_INDICES_TRAIN
)

indices_val = np.load(
    RUTA_INDICES_VAL
)

print("\n===== DIVISIÓN EXISTENTE =====")

print(
    f"Train:      {len(indices_train):,}"
)

print(
    f"Validation: {len(indices_val):,}"
)

# =========================================================
# LEER SBP Y DBP
# =========================================================

print("\nLeyendo SBP y DBP...")

with h5py.File(RUTA_MAT, "r") as f:

    sbp_dataset = f["Subset"]["SBP"]
    dbp_dataset = f["Subset"]["DBP"]

    # Leemos todos los valores una sola vez.
    sbp = np.asarray(
        sbp_dataset
    ).squeeze().astype(np.float32)

    dbp = np.asarray(
        dbp_dataset
    ).squeeze().astype(np.float32)

print(
    f"Total registros de presión: "
    f"{len(sbp):,}"
)

if len(sbp) != len(dbp):
    raise ValueError(
        "SBP y DBP tienen tamaños diferentes."
    )

# =========================================================
# CREAR ETIQUETAS GLOBALES
# =========================================================

etiquetas_130 = (
    (sbp >= 130.0)
    |
    (dbp >= 80.0)
).astype(np.int8)

etiquetas_140 = (
    (sbp >= 140.0)
    |
    (dbp >= 90.0)
).astype(np.int8)

# =========================================================
# EXTRAER TRAIN Y VALIDATION
# =========================================================

y_train_130 = etiquetas_130[
    indices_train
]

y_val_130 = etiquetas_130[
    indices_val
]

y_train_140 = etiquetas_140[
    indices_train
]

y_val_140 = etiquetas_140[
    indices_val
]

# =========================================================
# FUNCIÓN DE RESUMEN
# =========================================================

def resumen(nombre, etiquetas):

    total = len(etiquetas)

    positivos = int(
        np.sum(etiquetas == 1)
    )

    negativos = int(
        np.sum(etiquetas == 0)
    )

    porcentaje_positivos = (
        positivos / total * 100
    )

    porcentaje_negativos = (
        negativos / total * 100
    )

    ratio = (
        negativos / positivos
        if positivos > 0
        else np.inf
    )

    print(
        f"\n--- {nombre} ---"
    )

    print(
        f"Total:      {total:,}"
    )

    print(
        f"Negativos:  {negativos:,} "
        f"({porcentaje_negativos:.2f}%)"
    )

    print(
        f"Positivos:  {positivos:,} "
        f"({porcentaje_positivos:.2f}%)"
    )

    print(
        f"Ratio neg/pos: {ratio:.4f}"
    )

    return {
        "total": total,
        "positivos": positivos,
        "negativos": negativos,
        "porcentaje_positivos":
            porcentaje_positivos,
        "ratio": ratio
    }

# =========================================================
# RESULTADOS 130/80
# =========================================================

print("\n" + "=" * 65)
print("CRITERIO A - 130/80")
print("=" * 65)

resultado_train_130 = resumen(
    "TRAIN 130/80",
    y_train_130
)

resultado_val_130 = resumen(
    "VALIDATION 130/80",
    y_val_130
)

# =========================================================
# RESULTADOS 140/90
# =========================================================

print("\n" + "=" * 65)
print("CRITERIO B - 140/90")
print("=" * 65)

resultado_train_140 = resumen(
    "TRAIN 140/90",
    y_train_140
)

resultado_val_140 = resumen(
    "VALIDATION 140/90",
    y_val_140
)

# =========================================================
# CAMBIO ENTRE CRITERIOS
# =========================================================

print("\n" + "=" * 65)
print("CAMBIO DE ETIQUETAS")
print("=" * 65)

# Como 140/90 es un criterio más restrictivo,
# algunos positivos 130/80 pasan a ser negativos.

cambios_train = np.sum(
    y_train_130 != y_train_140
)

cambios_val = np.sum(
    y_val_130 != y_val_140
)

positivos_130_a_negativos_140_train = np.sum(
    (y_train_130 == 1)
    &
    (y_train_140 == 0)
)

positivos_130_a_negativos_140_val = np.sum(
    (y_val_130 == 1)
    &
    (y_val_140 == 0)
)

print(
    f"Train - etiquetas diferentes: "
    f"{cambios_train:,} "
    f"({cambios_train / len(y_train_130) * 100:.2f}%)"
)

print(
    f"Validation - etiquetas diferentes: "
    f"{cambios_val:,} "
    f"({cambios_val / len(y_val_130) * 100:.2f}%)"
)

print(
    "\nPositivos con 130/80 que dejan "
    "de ser positivos con 140/90:"
)

print(
    f"Train:      "
    f"{positivos_130_a_negativos_140_train:,}"
)

print(
    f"Validation: "
    f"{positivos_130_a_negativos_140_val:,}"
)

# =========================================================
# POS_WEIGHT PARA 140/90
# =========================================================

pos_weight_140 = (
    resultado_train_140["negativos"]
    /
    resultado_train_140["positivos"]
)

print("\n" + "=" * 65)
print("POS_WEIGHT PARA UN POSIBLE MODELO 140/90")
print("=" * 65)

print(
    f"pos_weight calculado SOLO con Train: "
    f"{pos_weight_140:.4f}"
)

print(
    "\nEste valor NO se está usando todavía."
)

print(
    "Solo se calcula para conocer "
    "el desbalance del criterio 140/90."
)

# =========================================================
# GUARDAR ETIQUETAS 140/90
# =========================================================

np.save(
    RUTA_ETIQUETAS_TRAIN_140,
    y_train_140
)

np.save(
    RUTA_ETIQUETAS_VAL_140,
    y_val_140
)

print("\n" + "=" * 65)
print("ARCHIVOS NUEVOS GUARDADOS")
print("=" * 65)

print(
    RUTA_ETIQUETAS_TRAIN_140
)

print(
    RUTA_ETIQUETAS_VAL_140
)

print("\nIMPORTANTE:")

print(
    "Las etiquetas originales 130/80 "
    "NO fueron modificadas."
)

print(
    "No se entrenó ningún modelo."
)

print(
    "CalFree NO fue utilizado."
)

print("\nPUNTO 22 FINALIZADO.")