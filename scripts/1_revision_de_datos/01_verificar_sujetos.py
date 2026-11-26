import h5py

RUTA_TRAIN = r"E:\Tesis_PulseDB\data_train\VitalDB_Train_Subset.mat"
RUTA_CALFREE = r"E:\Tesis_PulseDB\data_calfree\VitalDB_CalFree_Test_Subset.mat"


def obtener_sujetos(archivo):
    referencias = archivo["Subset"]["Subject"][0, :]

    sujetos = []

    for ref in referencias:
        datos = archivo[ref][()].flatten()
        nombre = "".join(chr(int(x)) for x in datos)
        sujetos.append(nombre)

    return sujetos


print("Abriendo datasets...")

with h5py.File(RUTA_TRAIN, "r") as train_file, \
     h5py.File(RUTA_CALFREE, "r") as calfree_file:

    print("Leyendo sujetos de Train...")
    train = obtener_sujetos(train_file)

    print("Leyendo sujetos de CalFree...")
    calfree = obtener_sujetos(calfree_file)


train_unicos = set(train)
calfree_unicos = set(calfree)

repetidos = train_unicos.intersection(calfree_unicos)


print("\n===== RESULTADOS =====")

print("TRAIN segmentos:", len(train))
print("TRAIN sujetos únicos:", len(train_unicos))

print("\nCALFREE segmentos:", len(calfree))
print("CALFREE sujetos únicos:", len(calfree_unicos))

print("\nSujetos presentes en ambos:", len(repetidos))

if len(repetidos) == 0:
    print("CORRECTO: Train y CalFree no comparten sujetos.")
else:
    print("ADVERTENCIA: existen sujetos compartidos.")
    print("Sujetos repetidos:", sorted(repetidos))

print("\nPrimeros 5 sujetos Train:")
print(train[:5])

print("\nPrimeros 5 sujetos CalFree:")
print(calfree[:5])