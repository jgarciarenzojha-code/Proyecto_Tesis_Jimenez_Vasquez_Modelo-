import h5py
import numpy as np
from collections import defaultdict

RUTA_TRAIN = r"E:\Tesis_PulseDB\data_train\VitalDB_Train_Subset.mat"
RUTA_CALFREE = r"E:\Tesis_PulseDB\data_calfree\VitalDB_CalFree_Test_Subset.mat"

UMBRAL_SBP = 130
UMBRAL_DBP = 80


def obtener_nombre_sujeto(f, ref):
    datos = f[ref][()].flatten()
    return "".join(chr(int(x)) for x in datos)


def analizar(nombre, ruta):

    print(f"\n===== {nombre} =====")

    with h5py.File(ruta, "r") as f:

        subset = f["Subset"]

        sbp = subset["SBP"][0, :]
        dbp = subset["DBP"][0, :]
        referencias = subset["Subject"][0, :]

        estadisticas = defaultdict(
            lambda: {"total": 0, "hta": 0, "no_hta": 0}
        )

        print("Procesando segmentos por sujeto...")

        for i, ref in enumerate(referencias):

            sujeto = obtener_nombre_sujeto(f, ref)

            es_hta = (
                sbp[i] >= UMBRAL_SBP
                or dbp[i] >= UMBRAL_DBP
            )

            estadisticas[sujeto]["total"] += 1

            if es_hta:
                estadisticas[sujeto]["hta"] += 1
            else:
                estadisticas[sujeto]["no_hta"] += 1

    solo_hta = 0
    solo_no_hta = 0
    mixtos = 0

    porcentajes_hta = []
    segmentos_por_sujeto = []

    for sujeto, datos in estadisticas.items():

        total = datos["total"]
        hta = datos["hta"]
        no_hta = datos["no_hta"]

        porcentaje = (hta / total) * 100

        porcentajes_hta.append(porcentaje)
        segmentos_por_sujeto.append(total)

        if hta > 0 and no_hta == 0:
            solo_hta += 1

        elif no_hta > 0 and hta == 0:
            solo_no_hta += 1

        else:
            mixtos += 1

    print("Sujetos totales:", len(estadisticas))

    print("\n--- Comportamiento de los sujetos ---")
    print("Solo segmentos >=130/80:", solo_hta)
    print("Solo segmentos <130/80:", solo_no_hta)
    print("Sujetos con ambos tipos:", mixtos)

    print("\n--- Segmentos por sujeto ---")
    print("Mínimo:", np.min(segmentos_por_sujeto))
    print("Máximo:", np.max(segmentos_por_sujeto))
    print("Mediana:", np.median(segmentos_por_sujeto))
    print("Media:", round(np.mean(segmentos_por_sujeto), 2))

    print("\n--- % de segmentos >=130/80 por sujeto ---")
    print("Mínimo:", round(np.min(porcentajes_hta), 2))
    print("Máximo:", round(np.max(porcentajes_hta), 2))
    print("Mediana:", round(np.median(porcentajes_hta), 2))
    print("Media:", round(np.mean(porcentajes_hta), 2))

    print("\nPercentiles:")
    print(
        np.percentile(
            porcentajes_hta,
            [0, 25, 50, 75, 100]
        )
    )


analizar("TRAIN", RUTA_TRAIN)
analizar("CALFREE TEST", RUTA_CALFREE)