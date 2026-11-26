import h5py
import numpy as np

RUTA_TRAIN = r"E:\Tesis_PulseDB\data_train\VitalDB_Train_Subset.mat"
RUTA_CALFREE = r"E:\Tesis_PulseDB\data_calfree\VitalDB_CalFree_Test_Subset.mat"


def analizar(nombre, ruta):

    print(f"\n===== {nombre} =====")

    with h5py.File(ruta, "r") as f:

        sbp = f["Subset"]["SBP"][0, :]
        dbp = f["Subset"]["DBP"][0, :]

        print("Número de segmentos:", len(sbp))

        print("\n--- SBP ---")
        print("Mínimo:", np.min(sbp))
        print("Máximo:", np.max(sbp))
        print("Media:", np.mean(sbp))
        print("Mediana:", np.median(sbp))
        print("Desviación estándar:", np.std(sbp))

        print("\n--- DBP ---")
        print("Mínimo:", np.min(dbp))
        print("Máximo:", np.max(dbp))
        print("Media:", np.mean(dbp))
        print("Mediana:", np.median(dbp))
        print("Desviación estándar:", np.std(dbp))

        print("\n--- Calidad básica ---")
        print("NaN en SBP:", np.isnan(sbp).sum())
        print("NaN en DBP:", np.isnan(dbp).sum())

        print("\nPercentiles SBP:")
        print(np.percentile(sbp, [1, 5, 25, 50, 75, 95, 99]))

        print("\nPercentiles DBP:")
        print(np.percentile(dbp, [1, 5, 25, 50, 75, 95, 99]))


analizar("TRAIN", RUTA_TRAIN)
analizar("CALFREE TEST", RUTA_CALFREE)