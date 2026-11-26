"""
Genera archivos .mat de prueba para la aplicación web.

- ejemplos_mat/registro_prueba_0X.mat: la MISMA señal de cada CSV de
  app_web/ejemplos/ (60 s), guardada como vector "ecg" (MATLAB v5).
  Al subirlos deben dar el mismo resultado que el registro de ejemplo.
- ejemplos_mat/sujeto_calfree_completo.mat: los 400 segmentos de un
  sujeto de VitalDB CalFree como matriz "ecg" [400 x 1250]
  (la app analiza como máximo 360 segmentos = 1 hora).

Solo LEE los CSV y el .mat de CalFree (no los modifica).

Ejecutar desde la carpeta app_web:
    ..\\.venv\\Scripts\\python.exe herramientas\\exportar_ejemplos_mat.py
"""

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import scipy.io


RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from backend.modelo.preprocesamiento import leer_csv_ecg  # noqa: E402


CARPETA_EJEMPLOS = RAIZ / "ejemplos"
CARPETA_SALIDA = RAIZ / "ejemplos_mat"
RUTA_CALFREE = RAIZ.parent / "data_calfree" / "VitalDB_CalFree_Test_Subset.mat"

FRECUENCIA_HZ = 125
SEGMENTOS_POR_SUJETO = 400


def main():

    CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)

    ejemplos = json.loads((CARPETA_EJEMPLOS / "ejemplos.json").read_text(encoding="utf-8"))

    for e in ejemplos:
        ecg = leer_csv_ecg((CARPETA_EJEMPLOS / e["archivo"]).read_bytes())
        destino = CARPETA_SALIDA / Path(e["archivo"]).with_suffix(".mat").name
        scipy.io.savemat(destino, {"ecg": ecg, "fs": FRECUENCIA_HZ}, do_compression=True)
        print(f"  {destino.name:<28} {len(ecg):>7,} muestras  ({destino.stat().st_size / 1024:.0f} KB)")

    # Un sujeto completo de CalFree (los 400 primeros segmentos = primer sujeto).
    with h5py.File(RUTA_CALFREE, "r") as f:
        ecg = np.asarray(f["Subset"]["Signals"][:, 0, :SEGMENTOS_POR_SUJETO]).T   # [400, 1250]

    destino = CARPETA_SALIDA / "sujeto_calfree_completo.mat"
    scipy.io.savemat(destino, {"ecg": ecg, "fs": FRECUENCIA_HZ}, do_compression=True)
    print(f"  {destino.name:<28} {ecg.size:>7,} muestras  ({destino.stat().st_size / 1e6:.1f} MB)")

    print(f"\nArchivos guardados en: {CARPETA_SALIDA}")


if __name__ == "__main__":
    main()
