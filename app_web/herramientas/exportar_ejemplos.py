"""
Exporta registros de ejemplo desde VitalDB CalFree (conjunto de prueba)
para demostrar la aplicación web.

- Solo LEE el archivo .mat (no lo modifica).
- Cada ejemplo = N segmentos consecutivos del MISMO sujeto (60 s por defecto).
- Se guardan como CSV (una muestra por línea) en app_web/ejemplos/.
- Los valores de SBP/DBP del dataset se guardan solo como referencia,
  para comparar después del análisis. El modelo NO los recibe.

Ejecutar desde la carpeta app_web:
    ..\\.venv\\Scripts\\python.exe herramientas\\exportar_ejemplos.py
"""

import json
from pathlib import Path

import h5py
import numpy as np


RAIZ = Path(__file__).resolve().parent.parent
RUTA_MAT = RAIZ.parent / "data_calfree" / "VitalDB_CalFree_Test_Subset.mat"
CARPETA_SALIDA = RAIZ / "ejemplos"

SEGMENTOS_POR_EJEMPLO = 6           # 6 x 10 s = 60 s
EJEMPLOS_POSITIVOS = 3
EJEMPLOS_NEGATIVOS = 3
SEED = 7


def decodificar(f, ref):

    valores = np.asarray(f[ref]).reshape(-1)
    return "".join(chr(int(v)) for v in valores if int(v) != 0)


def main():

    CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)

    with h5py.File(RUTA_MAT, "r") as f:

        s = f["Subset"]

        sbp = np.asarray(s["SBP"]).reshape(-1)
        dbp = np.asarray(s["DBP"]).reshape(-1)
        edad = np.asarray(s["Age"]).reshape(-1)
        peso = np.asarray(s["Weight"]).reshape(-1)
        talla = np.asarray(s["Height"]).reshape(-1)
        refs_sujeto = np.asarray(s["Subject"]).reshape(-1)
        refs_sexo = np.asarray(s["Gender"]).reshape(-1)

        etiqueta = ((sbp >= 140) | (dbp >= 90)).astype(int)

        total = len(sbp)
        n = SEGMENTOS_POR_EJEMPLO
        rng = np.random.default_rng(SEED)

        elegidos = {"positivo": [], "negativo": []}
        sujetos_usados = set()

        for inicio in rng.permutation(total - n):

            if (
                len(elegidos["positivo"]) >= EJEMPLOS_POSITIVOS
                and len(elegidos["negativo"]) >= EJEMPLOS_NEGATIVOS
            ):
                break

            idx = np.arange(inicio, inicio + n)
            positivos = int(etiqueta[idx].sum())

            # Bloques claramente positivos o claramente negativos.
            if positivos == n:
                clase = "positivo"
            elif positivos == 0:
                clase = "negativo"
            else:
                continue

            cupo = EJEMPLOS_POSITIVOS if clase == "positivo" else EJEMPLOS_NEGATIVOS
            if len(elegidos[clase]) >= cupo:
                continue

            sujetos = {decodificar(f, refs_sujeto[i]) for i in idx}
            if len(sujetos) != 1:
                continue

            sujeto = sujetos.pop()
            if sujeto in sujetos_usados:
                continue

            sujetos_usados.add(sujeto)
            elegidos[clase].append((idx, sujeto))

        indice = []
        orden = elegidos["positivo"] + elegidos["negativo"]

        # Mezclar para que el orden no revele la clase.
        orden = [orden[i] for i in rng.permutation(len(orden))]

        for numero, (idx, sujeto) in enumerate(orden, start=1):

            ecg = np.concatenate(
                [np.asarray(s["Signals"][:, 0, int(i)]) for i in idx]
            ).astype(np.float32)

            nombre_archivo = f"registro_prueba_{numero:02d}.csv"

            with open(CARPETA_SALIDA / nombre_archivo, "w", encoding="utf-8") as out:
                out.write("ecg\n")
                out.write("\n".join(f"{v:.6f}" for v in ecg))
                out.write("\n")

            i0 = int(idx[0])
            sexo = decodificar(f, refs_sexo[i0]).strip().upper()[:1]

            indice.append({
                "id": f"ej{numero:02d}",
                "nombre": f"Registro de prueba {numero:02d}",
                "archivo": nombre_archivo,
                "duracion_s": n * 10,
                "fuente": "VitalDB CalFree (conjunto de prueba independiente)",
                "demograficos": {
                    "edad": round(float(edad[i0]), 0),
                    "sexo": sexo if sexo in ("M", "F") else None,
                    "peso": round(float(peso[i0]), 1),
                    "talla": round(float(talla[i0]), 1),
                },
                "referencia": {
                    "sbp_promedio": round(float(sbp[idx].mean()), 1),
                    "dbp_promedio": round(float(dbp[idx].mean()), 1),
                    "segmentos_140_90": int(etiqueta[idx].sum()),
                    "segmentos_total": n,
                },
            })

            print(f"{nombre_archivo}: sujeto={sujeto} clase_ref="
                  f"{'>=140/90' if etiqueta[idx].all() else '<140/90'}")

    with open(CARPETA_SALIDA / "ejemplos.json", "w", encoding="utf-8") as out:
        json.dump(indice, out, ensure_ascii=False, indent=2)

    print(f"\n{len(indice)} ejemplos guardados en {CARPETA_SALIDA}")


if __name__ == "__main__":
    main()
