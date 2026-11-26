import os
import h5py
import numpy as np

RUTA_MAT = (
    r"E:\Tesis_PulseDB\data_train"
    r"\VitalDB_Train_Subset.mat"
)

CARPETA_PROCESADO = (
    r"E:\Tesis_PulseDB\procesado"
)

RUTA_TRAIN = os.path.join(
    CARPETA_PROCESADO,
    "indices_train.npy"
)

RUTA_VAL = os.path.join(
    CARPETA_PROCESADO,
    "indices_val.npy"
)

print("=" * 75)
print("PUNTO 28 - ANÁLISIS DE EDAD Y POBLACIÓN ADULTA")
print("=" * 75)

indices_train = np.load(RUTA_TRAIN)
indices_val = np.load(RUTA_VAL)

with h5py.File(RUTA_MAT, "r") as f:

    subset = f["Subset"]

    age = np.asarray(
        subset["Age"]
    ).reshape(-1).astype(float)

    sbp = np.asarray(
        subset["SBP"]
    ).reshape(-1).astype(float)

    dbp = np.asarray(
        subset["DBP"]
    ).reshape(-1).astype(float)

    # -----------------------------------------------------
    # Recuperar Subject correctamente desde referencias HDF5
    # -----------------------------------------------------

    refs = np.asarray(
        subset["Subject"]
    ).reshape(-1)

    sujetos = []

    for ref in refs:

        valor = np.asarray(
            f[ref]
        ).reshape(-1)

        if valor.dtype.kind in ("u", "i"):

            try:
                sujeto = "".join(
                    chr(int(x))
                    for x in valor
                    if int(x) != 0
                )
            except Exception:
                sujeto = str(valor.tolist())

        else:
            sujeto = str(valor.tolist())

        sujetos.append(sujeto)

sujetos = np.asarray(
    sujetos,
    dtype=str
)

print(f"\nSegmentos totales: {len(age):,}")
print(f"Sujetos totales:   {len(np.unique(sujetos)):,}")

# =========================================================
# FUNCIÓN DE ANÁLISIS
# =========================================================

def analizar(nombre, indices):

    edades = age[indices]
    sujetos_split = sujetos[indices]
    sbp_split = sbp[indices]
    dbp_split = dbp[indices]

    menores = edades < 18
    adultos = edades >= 18

    sujetos_unicos = np.unique(
        sujetos_split
    )

    sujetos_menores = np.unique(
        sujetos_split[menores]
    )

    sujetos_adultos = np.unique(
        sujetos_split[adultos]
    )

    positivos_140_90 = (
        (sbp_split >= 140)
        |
        (dbp_split >= 90)
    )

    positivos_adultos = (
        positivos_140_90
        &
        adultos
    )

    print("\n" + "=" * 75)
    print(nombre)
    print("=" * 75)

    print(
        f"Segmentos: {len(indices):,}"
    )

    print(
        f"Sujetos únicos: "
        f"{len(sujetos_unicos):,}"
    )

    print(
        f"\nEdad mínima: "
        f"{edades.min():.2f}"
    )

    print(
        f"Edad máxima: "
        f"{edades.max():.2f}"
    )

    print(
        f"Edad media: "
        f"{edades.mean():.2f}"
    )

    print(
        f"\nSegmentos <18 años: "
        f"{menores.sum():,} "
        f"({menores.mean()*100:.2f}%)"
    )

    print(
        f"Segmentos >=18 años: "
        f"{adultos.sum():,} "
        f"({adultos.mean()*100:.2f}%)"
    )

    print(
        f"\nSujetos con segmentos <18: "
        f"{len(sujetos_menores):,}"
    )

    print(
        f"Sujetos con segmentos >=18: "
        f"{len(sujetos_adultos):,}"
    )

    print(
        f"\nPositivos 140/90 totales: "
        f"{positivos_140_90.sum():,}"
    )

    print(
        f"Positivos 140/90 en adultos: "
        f"{positivos_adultos.sum():,}"
    )

    # Distribución por rangos de edad
    rangos = [
        (0, 18),
        (18, 30),
        (30, 45),
        (45, 60),
        (60, 75),
        (75, 200)
    ]

    print("\nDistribución por edad:")

    for minimo, maximo in rangos:

        mascara = (
            (edades >= minimo)
            &
            (edades < maximo)
        )

        print(
            f"{minimo:>3}-{maximo:<3}: "
            f"{mascara.sum():,} segmentos"
        )


analizar(
    "TRAIN",
    indices_train
)

analizar(
    "VALIDATION",
    indices_val
)

print("\n" + "=" * 75)
print("PUNTO 28 FINALIZADO")
print("=" * 75)

print(
    "\nEste análisis NO modifica ningún dataset,"
)

print(
    "NO entrena ningún modelo y NO utiliza CalFree."
)