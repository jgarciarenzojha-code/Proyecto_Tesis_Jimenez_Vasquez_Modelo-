import os
import json
import time
import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# ============================================================
# PUNTO 41 - COMPARACIÓN DEL ENSEMBLE CON CADA CNN EN CALFREE
# (Tabla VI de la tesis)
#
# - NO entrena ni carga ningún modelo: usa los logits de las tres
#   CNN que el punto 36 ya guardó (mismos pesos del ensemble final).
# - Bootstrap por sujeto con los MISMOS parámetros del punto 36
#   (2000 remuestreos, semilla 42, mismo procedimiento), por lo que
#   el IC del ensemble y de la CNN V2 reproducen exactamente el 36.
# - Es un análisis descriptivo: ninguna decisión (modelo, umbrales)
#   se toma con estos resultados.
#
# Salidas:
# - resultados/4_cnn_ensemble/comparacion_ensemble_vs_cnn_calfree_41.json
# - resultados/4_cnn_ensemble/comparacion_ensemble_vs_cnn_calfree_41.csv
# ============================================================

RAIZ = r"E:\Tesis_PulseDB"
RUTA_MAT = os.path.join(RAIZ, "data_calfree", "VitalDB_CalFree_Test_Subset.mat")
RUTA_NPZ = os.path.join(RAIZ, "resultados", "4_cnn_ensemble", "probabilidades_calfree_ensemble_36.npz")
RUTA_36 = os.path.join(RAIZ, "resultados", "4_cnn_ensemble", "evaluacion_final_ensemble_calfree_36.json")
RUTA_JSON = os.path.join(RAIZ, "resultados", "4_cnn_ensemble", "comparacion_ensemble_vs_cnn_calfree_41.json")
RUTA_CSV = os.path.join(RAIZ, "resultados", "4_cnn_ensemble", "comparacion_ensemble_vs_cnn_calfree_41.csv")

SEED = 42
N_BOOTSTRAP = 2000   # igual que el punto 36

print("=" * 78)
print("PUNTO 41 - ENSEMBLE FRENTE A CADA CNN EN CALFREE (TABLA VI)")
print("=" * 78)

datos = np.load(RUTA_NPZ)
y = datos["etiquetas"].astype(int)
sigmoide = lambda z: 1 / (1 + np.exp(-z))
puntajes = {
    "CNN 1D residual (V2)": datos["logit_v2"].astype(np.float64),
    "CNN 1D multiescala (V3)": datos["logit_v3"].astype(np.float64),
    "CNN 1D multitarea": datos["logit_multitarea"].astype(np.float64),
    "CNN-Ensemble": datos["prob_ensemble"].astype(np.float64),
}

# Control: el promedio de logits debe reproducir la probabilidad guardada del ensemble.
logit_ens = np.mean([datos["logit_v2"], datos["logit_v3"], datos["logit_multitarea"]], axis=0)
assert np.max(np.abs(sigmoide(logit_ens.astype(np.float64)) - puntajes["CNN-Ensemble"])) < 1e-6

# ------------------------------------------------------------
# Sujeto de cada segmento (misma lectura que el punto 36)
# ------------------------------------------------------------


def decodificar(f, ref):
    v = np.asarray(f[ref]).reshape(-1)
    return "".join(chr(int(x)) for x in v if int(x) != 0).strip()


inicio = time.time()
with h5py.File(RUTA_MAT, "r") as f:
    refs = np.asarray(f["Subset"]["Subject"]).reshape(-1)
    n = len(refs)
    assert n == len(y)
    cache = {}

    def suj(i):
        if i not in cache:
            cache[i] = decodificar(f, refs[i])
        return cache[i]

    sujeto = np.empty(n, dtype=object)
    i = 0
    while i < n:
        actual, paso, hi = suj(i), 1, i + 1
        while hi < n and suj(hi) == actual:
            paso *= 2
            hi = i + paso
        hi = min(hi, n)
        lo = i + paso // 2 if paso > 1 else i
        while hi - lo > 1:
            medio = (lo + hi) // 2
            if suj(medio) == actual:
                lo = medio
            else:
                hi = medio
        sujeto[i:hi] = actual
        i = hi

    muestra = np.random.default_rng(SEED).choice(n, 2000, replace=False)
    if sum(decodificar(f, refs[k]) != sujeto[k] for k in muestra):
        raise RuntimeError("Sujetos no contiguos en CalFree; revisar lectura.")
sujeto = sujeto.astype(str)
print(f"Segmentos: {n:,} | sujetos: {len(np.unique(sujeto))} | lectura en {time.time() - inicio:.0f} s")

# ------------------------------------------------------------
# Bootstrap por sujeto (idéntico al punto 36)
# ------------------------------------------------------------

sujetos_unicos, codigo = np.unique(sujeto, return_inverse=True)
pos_suj = [np.flatnonzero(codigo == k) for k in range(len(sujetos_unicos))]
rng = np.random.default_rng(SEED)
remuestreos = [
    np.concatenate([pos_suj[k] for k in rng.integers(0, len(pos_suj), len(pos_suj))])
    for _ in range(N_BOOTSTRAP)
]
validos = [i for i in remuestreos if y[i].min() != y[i].max()]

auc_boot = {m: np.array([roc_auc_score(y[i], p[i]) for i in validos]) for m, p in puntajes.items()}
auc = {m: roc_auc_score(y, p) for m, p in puntajes.items()}

filas = []
print(f"\nROC-AUC en CalFree (IC 95 %, bootstrap por sujeto, {N_BOOTSTRAP} remuestreos, semilla {SEED})")
for m in puntajes:
    lo, hi = np.percentile(auc_boot[m], [2.5, 97.5])
    fila = {"modelo": m, "roc_auc": auc[m], "ic95_inf": lo, "ic95_sup": hi}
    if m != "CNN-Ensemble":
        d = auc_boot["CNN-Ensemble"] - auc_boot[m]
        dlo, dhi = np.percentile(d, [2.5, 97.5])
        fila.update({"dif_ensemble_menos_cnn": auc["CNN-Ensemble"] - auc[m],
                     "dif_ic95_inf": dlo, "dif_ic95_sup": dhi,
                     "significativa": bool(dlo > 0 or dhi < 0)})
    filas.append(fila)
    extra = ""
    if "dif_ensemble_menos_cnn" in fila:
        extra = (f" | ensemble − red = {fila['dif_ensemble_menos_cnn']:+.4f} "
                 f"[{fila['dif_ic95_inf']:+.4f}, {fila['dif_ic95_sup']:+.4f}]"
                 f" {'significativa' if fila['significativa'] else 'no significativa'}")
    print(f"{m:<26} {auc[m]:.4f} [{lo:.4f} – {hi:.4f}]{extra}")

# Control: debe coincidir con el punto 36 (ensemble y V2).
r36 = json.load(open(RUTA_36, encoding="utf-8"))
ens, v2 = filas[3], filas[0]
assert abs(ens["roc_auc"] - r36["ensemble"]["roc_auc"]) < 1e-6
assert np.allclose([ens["ic95_inf"], ens["ic95_sup"]], r36["ensemble"]["ic95"], atol=1e-6)
assert np.allclose([v2["ic95_inf"], v2["ic95_sup"]], r36["cnn_v2"]["ic95"], atol=1e-6)
assert np.allclose([v2["dif_ic95_inf"], v2["dif_ic95_sup"]], r36["mejora_ensemble_sobre_v2"]["ic95"], atol=1e-6)
print("\nControl: el ensemble y la CNN V2 reproducen exactamente los resultados del punto 36.")

pd.DataFrame(filas).to_csv(RUTA_CSV, index=False, encoding="utf-8-sig")
json.dump({
    "descripcion": "Comparación del CNN-Ensemble con cada CNN en CalFree (Tabla VI). "
                   "Análisis descriptivo; el modelo y los umbrales se eligieron solo con validación.",
    "fuente_logits": "resultados/4_cnn_ensemble/probabilidades_calfree_ensemble_36.npz (mismos pesos del ensemble final)",
    "bootstrap": {"tipo": "por sujeto", "remuestreos": N_BOOTSTRAP, "validos": len(validos), "semilla": SEED},
    "segmentos": int(n), "sujetos": int(len(sujetos_unicos)),
    "resultados": filas,
}, open(RUTA_JSON, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

print(f"\nGuardado: {RUTA_JSON}\nGuardado: {RUTA_CSV}")
print("No se entrenó ni se modificó ningún modelo.")
