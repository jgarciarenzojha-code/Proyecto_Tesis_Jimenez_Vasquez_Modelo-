import h5py
import numpy as np

RUTA_TRAIN = r"E:\Tesis_PulseDB\data_train\VitalDB_Train_Subset.mat"

BLOQUE = 5000

with h5py.File(RUTA_TRAIN, "r") as f:

    signals = f["Subset"]["Signals"]

    total = signals.shape[2]

    print("Forma de Signals:", signals.shape)
    print("Total segmentos:", total)
    print("\nAnalizando canal ECG...")

    segmentos_nan = 0
    segmentos_inf = 0
    segmentos_constantes = 0

    minimo_global = np.inf
    maximo_global = -np.inf

    for inicio in range(0, total, BLOQUE):

        fin = min(inicio + BLOQUE, total)

        # Canal 0 = ECG
        ecg = signals[:, 0, inicio:fin]

        # Cada columna corresponde a un segmento
        tiene_nan = np.any(np.isnan(ecg), axis=0)
        tiene_inf = np.any(np.isinf(ecg), axis=0)

        segmentos_nan += np.sum(tiene_nan)
        segmentos_inf += np.sum(tiene_inf)

        # Rango de cada segmento
        rango = np.ptp(ecg, axis=0)

        segmentos_constantes += np.sum(rango == 0)

        valores_validos = ecg[np.isfinite(ecg)]

        if len(valores_validos) > 0:
            minimo_global = min(
                minimo_global,
                np.min(valores_validos)
            )

            maximo_global = max(
                maximo_global,
                np.max(valores_validos)
            )

        print(
            f"\rProcesados: {fin}/{total}",
            end=""
        )

print("\n\n===== CALIDAD ECG =====")

print("Segmentos con NaN:", segmentos_nan)
print("Segmentos con infinito:", segmentos_inf)
print("Segmentos constantes:", segmentos_constantes)

print("\nValor mínimo ECG:", minimo_global)
print("Valor máximo ECG:", maximo_global)

print("\nAnálisis terminado.")