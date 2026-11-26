import h5py
import numpy as np
import os

RUTA_TRAIN = r"E:\Tesis_PulseDB\data_train\VitalDB_Train_Subset.mat"
CARPETA_SALIDA = r"E:\Tesis_PulseDB\procesado"

SEED = 42
PORCENTAJE_TRAIN = 0.80

os.makedirs(CARPETA_SALIDA, exist_ok=True)


def obtener_nombre_sujeto(f, ref):
    datos = f[ref][()].flatten()
    return "".join(chr(int(x)) for x in datos)


print("Leyendo sujetos...")

with h5py.File(RUTA_TRAIN, "r") as f:

    referencias = f["Subset"]["Subject"][0, :]

    sujetos_segmentos = {}

    for i, ref in enumerate(referencias):

        sujeto = obtener_nombre_sujeto(f, ref)

        if sujeto not in sujetos_segmentos:
            sujetos_segmentos[sujeto] = []

        sujetos_segmentos[sujeto].append(i)


sujetos = np.array(list(sujetos_segmentos.keys()))

print("Sujetos encontrados:", len(sujetos))

# Mezcla reproducible
rng = np.random.default_rng(SEED)
rng.shuffle(sujetos)

n_train = int(len(sujetos) * PORCENTAJE_TRAIN)

sujetos_train = sujetos[:n_train]
sujetos_val = sujetos[n_train:]

set_train = set(sujetos_train)
set_val = set(sujetos_val)

indices_train = []
indices_val = []

for sujeto, indices in sujetos_segmentos.items():

    if sujeto in set_train:
        indices_train.extend(indices)

    elif sujeto in set_val:
        indices_val.extend(indices)


indices_train = np.array(indices_train, dtype=np.int64)
indices_val = np.array(indices_val, dtype=np.int64)


# Guardar
np.save(
    os.path.join(CARPETA_SALIDA, "indices_train.npy"),
    indices_train
)

np.save(
    os.path.join(CARPETA_SALIDA, "indices_val.npy"),
    indices_val
)

np.save(
    os.path.join(CARPETA_SALIDA, "sujetos_train.npy"),
    sujetos_train
)

np.save(
    os.path.join(CARPETA_SALIDA, "sujetos_val.npy"),
    sujetos_val
)


print("\n===== DIVISIÓN COMPLETADA =====")

print("Sujetos TRAIN:", len(sujetos_train))
print("Sujetos VALIDATION:", len(sujetos_val))

print("\nSegmentos TRAIN:", len(indices_train))
print("Segmentos VALIDATION:", len(indices_val))

print(
    "\nSujetos compartidos:",
    len(set(sujetos_train) & set(sujetos_val))
)

print("\nArchivos guardados en:")
print(CARPETA_SALIDA)