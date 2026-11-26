import os
import time
import h5py
import numpy as np
import pandas as pd

# ============================================================
# PUNTO 44 - PREPARAR LOS SEGMENTOS DE LOS SUJETOS NUEVOS (PLAN B)
#
# Toma los 200 sujetos nuevos verificados en el punto 43 y arma un
# archivo con sus segmentos para sumarlos al ENTRENAMIENTO.
#
# Mismo criterio que PulseDB usó para el subconjunto de entrenamiento:
# - Señal ECG_F (verificado en el punto 43: idéntica a Signals[:, 0]).
# - Como máximo 360 segmentos por sujeto, elegidos al azar (semilla 42).
#   El subconjunto original tiene exactamente 360 por sujeto, tomados al azar.
# - Etiqueta 140/90: PAS >= 140 o PAD >= 90 (la PAS/PAD de cada segmento).
#
# El archivo de control p000789 (sujeto de entrenamiento) se EXCLUYE.
# Validación y CalFree no se tocan.
#
# Salida:
# - procesado/sujetos_nuevos_44.npz  (ecg, sbp, dbp, etiqueta, sujeto)
# - resultados/preparacion_sujetos_nuevos_44.txt
# ============================================================

RAIZ = r"E:\Tesis_PulseDB"
CARPETA = os.path.join(RAIZ, "data_nuevos")
RUTA_VERIF = os.path.join(RAIZ, "resultados", "5_experimentos", "200_pacientes", "verificacion_sujetos_nuevos_43.csv")
RUTA_NPZ = os.path.join(RAIZ, "procesado", "sujetos_nuevos_44.npz")
RUTA_TXT = os.path.join(RAIZ, "resultados", "5_experimentos", "200_pacientes", "preparacion_sujetos_nuevos_44.txt")

SEED = 42
MAX_POR_SUJETO = 360

print("=" * 78)
print("PUNTO 44 - PREPARAR SEGMENTOS DE LOS SUJETOS NUEVOS")
print("=" * 78)

verif = pd.read_csv(RUTA_VERIF)
aptos = verif[(verif.lectura == "OK") & (verif.conjunto == "nuevo") & verif.coincide_info
              & (verif.con_nan == 0) & (verif.planos == 0) & (verif.include_flag_0 == 0)]
assert len(aptos) == 200, f"Se esperaban 200 sujetos aptos, hay {len(aptos)}"

train = set(np.load(os.path.join(RAIZ, "procesado", "sujetos_train.npy"), allow_pickle=True).astype(str))
val = set(np.load(os.path.join(RAIZ, "procesado", "sujetos_val.npy"), allow_pickle=True).astype(str))
assert not (set(aptos.sujeto) & (train | val)), "Un sujeto nuevo aparece en entrenamiento o validación"

rng = np.random.default_rng(SEED)
ecg_l, sbp_l, dbp_l, suj_l = [], [], [], []
inicio = time.time()
for a, sujeto in zip(aptos.archivo, aptos.sujeto):
    with h5py.File(os.path.join(CARPETA, a), "r") as f:
        w = f["Subj_Wins"]
        n = w["ECG_F"].shape[1]
        elegidos = np.sort(rng.choice(n, MAX_POR_SUJETO, replace=False)) if n > MAX_POR_SUJETO else np.arange(n)
        leer = lambda k, i: np.asarray(f[w[k][0, i]]).reshape(-1)
        ecg_l.append(np.stack([leer("ECG_F", i) for i in elegidos]).astype(np.float32))
        sbp_l.append(np.array([leer("SegSBP", i)[0] for i in elegidos], dtype=np.float32))
        dbp_l.append(np.array([leer("SegDBP", i)[0] for i in elegidos], dtype=np.float32))
        suj_l.append(np.full(len(elegidos), sujeto))

ecg = np.concatenate(ecg_l)
sbp = np.concatenate(sbp_l)
dbp = np.concatenate(dbp_l)
sujeto = np.concatenate(suj_l)
etiqueta = ((sbp >= 140) | (dbp >= 90)).astype(np.int8)
assert ecg.shape[1] == 1250 and np.isfinite(ecg).all()

np.savez(RUTA_NPZ, ecg=ecg, sbp=sbp, dbp=dbp, etiqueta=etiqueta, sujeto=sujeto)

# Comparación con el entrenamiento original
y_tr = np.load(os.path.join(RAIZ, "procesado", "etiquetas_train_140_90.npy"))
total = len(y_tr) + len(etiqueta)
pos = int(y_tr.sum() + etiqueta.sum())
lineas = [
    f"Sujetos nuevos: {len(aptos)} | segmentos tomados: {len(etiqueta):,} "
    f"(tope {MAX_POR_SUJETO} por sujeto, semilla {SEED}) | lectura en {time.time() - inicio:.0f} s",
    f"Segmentos nuevos con HTA 140/90: {int(etiqueta.sum()):,} ({etiqueta.mean():.3f})",
    f"Sujetos con el tope de {MAX_POR_SUJETO}: {int(sum(len(s) == MAX_POR_SUJETO for s in suj_l))}",
    "",
    f"Entrenamiento original: 1034 sujetos | {len(y_tr):,} segmentos | positivos {int(y_tr.sum()):,} ({y_tr.mean():.3f})",
    f"Entrenamiento ampliado: {1034 + len(aptos)} sujetos | {total:,} segmentos | positivos {pos:,} ({pos / total:.3f})",
    "",
    "Validación (259 sujetos) y CalFree (144) no cambian.",
    f"Guardado: {RUTA_NPZ}",
]
resumen = "\n".join(lineas)
print("\n" + resumen)
open(RUTA_TXT, "w", encoding="utf-8").write(resumen + "\n")
