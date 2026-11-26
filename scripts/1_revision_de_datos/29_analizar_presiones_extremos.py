import os
import h5py
import numpy as np

# ============================================================
# RUTAS
# ============================================================

RUTA_MAT = (
    r"E:\Tesis_PulseDB\data_train"
    r"\VitalDB_Train_Subset.mat"
)

CARPETA_PROCESADO = r"E:\Tesis_PulseDB\procesado"

RUTA_TRAIN = os.path.join(
    CARPETA_PROCESADO,
    "indices_train.npy"
)

RUTA_VAL = os.path.join(
    CARPETA_PROCESADO,
    "indices_val.npy"
)

# ============================================================
# CARGAR ÍNDICES
# ============================================================

print("=" * 75)
print("PUNTO 29 - ANÁLISIS DE PRESIÓN ARTERIAL Y VALORES EXTREMOS")
print("=" * 75)

indices_train = np.load(RUTA_TRAIN)
indices_val = np.load(RUTA_VAL)

# ============================================================
# CARGAR SBP Y DBP
# ============================================================

with h5py.File(RUTA_MAT, "r") as f:

    subset = f["Subset"]

    sbp = np.asarray(
        subset["SBP"]
    ).reshape(-1).astype(float)

    dbp = np.asarray(
        subset["DBP"]
    ).reshape(-1).astype(float)

print(f"\nSegmentos totales del archivo: {len(sbp):,}")

# ============================================================
# FUNCIÓN DE ANÁLISIS
# ============================================================

def analizar(nombre, indices):

    s = sbp[indices]
    d = dbp[indices]

    positivos = (
        (s >= 140)
        |
        (d >= 90)
    )

    negativos = ~positivos

    # --------------------------------------------------------
    # Valores potencialmente extremos
    # Solo se cuentan. NO se eliminan.
    # --------------------------------------------------------

    sbp_muy_baja = s < 70
    sbp_muy_alta = s > 220

    dbp_muy_baja = d < 40
    dbp_muy_alta = d > 130

    cualquier_extremo = (
        sbp_muy_baja
        |
        sbp_muy_alta
        |
        dbp_muy_baja
        |
        dbp_muy_alta
    )

    # --------------------------------------------------------
    # Casos cercanos al umbral 140/90
    # --------------------------------------------------------

    cerca_sbp = (
        (s >= 135)
        &
        (s < 145)
    )

    cerca_dbp = (
        (d >= 85)
        &
        (d < 95)
    )

    cerca_umbral = (
        cerca_sbp
        |
        cerca_dbp
    )

    print("\n" + "=" * 75)
    print(nombre)
    print("=" * 75)

    print(f"Segmentos: {len(indices):,}")

    print("\nSBP:")
    print(f"  Mínimo:  {s.min():.2f}")
    print(f"  P1:      {np.percentile(s, 1):.2f}")
    print(f"  P5:      {np.percentile(s, 5):.2f}")
    print(f"  Mediana: {np.median(s):.2f}")
    print(f"  P95:     {np.percentile(s, 95):.2f}")
    print(f"  P99:     {np.percentile(s, 99):.2f}")
    print(f"  Máximo:  {s.max():.2f}")

    print("\nDBP:")
    print(f"  Mínimo:  {d.min():.2f}")
    print(f"  P1:      {np.percentile(d, 1):.2f}")
    print(f"  P5:      {np.percentile(d, 5):.2f}")
    print(f"  Mediana: {np.median(d):.2f}")
    print(f"  P95:     {np.percentile(d, 95):.2f}")
    print(f"  P99:     {np.percentile(d, 99):.2f}")
    print(f"  Máximo:  {d.max():.2f}")

    print("\nCLASIFICACIÓN 140/90:")

    print(
        f"  Negativos: {negativos.sum():,} "
        f"({negativos.mean()*100:.2f}%)"
    )

    print(
        f"  Positivos: {positivos.sum():,} "
        f"({positivos.mean()*100:.2f}%)"
    )

    print("\nVALORES POTENCIALMENTE EXTREMOS:")

    print(
        f"  SBP < 70:  {sbp_muy_baja.sum():,} "
        f"({sbp_muy_baja.mean()*100:.3f}%)"
    )

    print(
        f"  SBP > 220: {sbp_muy_alta.sum():,} "
        f"({sbp_muy_alta.mean()*100:.3f}%)"
    )

    print(
        f"  DBP < 40:  {dbp_muy_baja.sum():,} "
        f"({dbp_muy_baja.mean()*100:.3f}%)"
    )

    print(
        f"  DBP > 130: {dbp_muy_alta.sum():,} "
        f"({dbp_muy_alta.mean()*100:.3f}%)"
    )

    print(
        f"\n  Con al menos un valor extremo: "
        f"{cualquier_extremo.sum():,} "
        f"({cualquier_extremo.mean()*100:.3f}%)"
    )

    print("\nCERCA DEL UMBRAL:")

    print(
        f"  SBP entre 135 y <145: "
        f"{cerca_sbp.sum():,}"
    )

    print(
        f"  DBP entre 85 y <95: "
        f"{cerca_dbp.sum():,}"
    )

    print(
        f"  Cerca de 140/90 por alguna medida: "
        f"{cerca_umbral.sum():,} "
        f"({cerca_umbral.mean()*100:.2f}%)"
    )

    # --------------------------------------------------------
    # Positivos según qué componente activa la etiqueta
    # --------------------------------------------------------

    solo_sbp = (
        (s >= 140)
        &
        (d < 90)
    )

    solo_dbp = (
        (s < 140)
        &
        (d >= 90)
    )

    ambos = (
        (s >= 140)
        &
        (d >= 90)
    )

    print("\nORIGEN DE LOS POSITIVOS:")

    print(
        f"  Solo SBP >= 140: "
        f"{solo_sbp.sum():,}"
    )

    print(
        f"  Solo DBP >= 90: "
        f"{solo_dbp.sum():,}"
    )

    print(
        f"  Ambos >= umbral: "
        f"{ambos.sum():,}"
    )


# ============================================================
# EJECUTAR
# ============================================================

analizar(
    "TRAIN",
    indices_train
)

analizar(
    "VALIDATION",
    indices_val
)

print("\n" + "=" * 75)
print("PUNTO 29 FINALIZADO")
print("=" * 75)

print(
    "\nIMPORTANTE:"
    "\n- No se eliminó ningún segmento."
    "\n- No se modificaron las etiquetas."
    "\n- No se entrenó ningún modelo."
    "\n- CalFree no fue utilizado."
)