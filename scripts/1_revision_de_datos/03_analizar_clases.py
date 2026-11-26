import h5py
import numpy as np

RUTA_TRAIN = r"E:\Tesis_PulseDB\data_train\VitalDB_Train_Subset.mat"
RUTA_CALFREE = r"E:\Tesis_PulseDB\data_calfree\VitalDB_CalFree_Test_Subset.mat"


def analizar(nombre, ruta):

    with h5py.File(ruta, "r") as f:

        sbp = f["Subset"]["SBP"][0, :]
        dbp = f["Subset"]["DBP"][0, :]

    total = len(sbp)

    # Criterio AHA/ACC 2025: estadio 1 o superior
    hta_130_80 = (sbp >= 130) | (dbp >= 80)

    # Criterio más conservador: 140/90
    hta_140_90 = (sbp >= 140) | (dbp >= 90)

    print(f"\n===== {nombre} =====")
    print("Total segmentos:", total)

    print("\n--- Criterio >=130/80 ---")
    print("Rango hipertensivo:", np.sum(hta_130_80))
    print("Fuera de rango hipertensivo:", np.sum(~hta_130_80))
    print("Porcentaje hipertensivo:",
          round(np.mean(hta_130_80) * 100, 2), "%")

    print("\n--- Criterio >=140/90 ---")
    print("Rango hipertensivo:", np.sum(hta_140_90))
    print("Fuera de rango hipertensivo:", np.sum(~hta_140_90))
    print("Porcentaje hipertensivo:",
          round(np.mean(hta_140_90) * 100, 2), "%")


analizar("TRAIN", RUTA_TRAIN)
analizar("CALFREE TEST", RUTA_CALFREE)