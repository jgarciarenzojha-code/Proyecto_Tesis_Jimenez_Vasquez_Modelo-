import h5py
import numpy as np

RUTA_TRAIN = (
    r"E:\Tesis_PulseDB\data_train"
    r"\VitalDB_Train_Subset.mat"
)

RUTA_CALFREE = (
    r"E:\Tesis_PulseDB\data_calfree"
    r"\VitalDB_CalFree_Test_Subset.mat"
)

VARIABLES = [
    "Age",
    "BMI",
    "Gender",
    "Height",
    "Weight"
]


def analizar_archivo(nombre, ruta):

    print("\n" + "=" * 75)
    print(nombre)
    print("=" * 75)

    with h5py.File(ruta, "r") as f:

        subset = f["Subset"]

        print(
            f"Campos disponibles:\n"
            f"{list(subset.keys())}"
        )

        for variable in VARIABLES:

            print("\n" + "-" * 60)
            print(f"VARIABLE: {variable}")
            print("-" * 60)

            if variable not in subset:

                print("NO EXISTE")
                continue

            datos = np.asarray(
                subset[variable]
            ).reshape(-1)

            print(
                f"Cantidad: {len(datos):,}"
            )

            print(
                f"Tipo: {datos.dtype}"
            )

            # Variables numéricas
            if np.issubdtype(
                datos.dtype,
                np.number
            ):

                datos = datos.astype(
                    np.float64
                )

                nan_count = np.isnan(
                    datos
                ).sum()

                inf_count = np.isinf(
                    datos
                ).sum()

                validos = datos[
                    np.isfinite(datos)
                ]

                print(
                    f"NaN: {nan_count:,}"
                )

                print(
                    f"Inf: {inf_count:,}"
                )

                if len(validos) > 0:

                    print(
                        f"Mínimo: "
                        f"{validos.min():.4f}"
                    )

                    print(
                        f"Máximo: "
                        f"{validos.max():.4f}"
                    )

                    print(
                        f"Media: "
                        f"{validos.mean():.4f}"
                    )

                    print(
                        f"Mediana: "
                        f"{np.median(validos):.4f}"
                    )

                    valores_unicos = (
                        np.unique(validos)
                    )

                    print(
                        f"Valores únicos: "
                        f"{len(valores_unicos):,}"
                    )

                    if len(
                        valores_unicos
                    ) <= 20:

                        print(
                            "Valores:"
                        )

                        print(
                            valores_unicos
                        )

            else:

                print(
                    "Variable no numérica "
                    "o almacenada como referencia."
                )


print("=" * 75)
print("PUNTO 27 - REVISIÓN DE VARIABLES DEMOGRÁFICAS")
print("=" * 75)

analizar_archivo(
    "TRAIN",
    RUTA_TRAIN
)

analizar_archivo(
    "CALFREE",
    RUTA_CALFREE
)

print("\n" + "=" * 75)
print("PUNTO 27 FINALIZADO")
print("=" * 75)

print(
    "\nIMPORTANTE: "
    "Este script NO entrena ningún modelo."
)

print(
    "Solo inspecciona las variables disponibles."
)