import h5py
import numpy as np

RUTA_MAT = r"E:\Tesis_PulseDB\data_train\VitalDB_Train_Subset.mat"
RUTA_INDICES = r"E:\Tesis_PulseDB\procesado\indices_val.npy"

indices = np.load(RUTA_INDICES)

print("=" * 60)
print("INSPECCIÓN DEL CAMPO SUBJECT")
print("=" * 60)

with h5py.File(RUTA_MAT, "r") as f:

    subject_dataset = f["Subset"]["Subject"]

    print("Shape Subject:", subject_dataset.shape)
    print("Dtype Subject:", subject_dataset.dtype)

    print("\nPrimeros 5 segmentos de Validation:")

    for posicion, indice_real in enumerate(indices[:5], start=1):

        referencia = subject_dataset[0, indice_real]

        print("\n--------------------------------")
        print("Posición:", posicion)
        print("Índice real:", indice_real)
        print("Tipo referencia:", type(referencia))
        print("Referencia:", referencia)

        if referencia:

            objeto = f[referencia]

            print("Objeto referenciado:", objeto)
            print("Shape:", objeto.shape)
            print("Dtype:", objeto.dtype)

            datos = np.asarray(objeto)

            print("Datos:")
            print(datos)

print("\nINSPECCIÓN TERMINADA")