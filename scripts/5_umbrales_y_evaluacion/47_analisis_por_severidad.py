import os
import json
import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# ============================================================
# PUNTO 47 - ANÁLISIS COMPLEMENTARIO POR SEVERIDAD DE HTA
#
# Modelo FINAL (CNN-Ensemble), SIN reentrenar ni volver a inferir:
# usa las probabilidades ya guardadas por los puntos 35 (validación)
# y 36 (CalFree), es decir, los mismos pesos del resultado principal.
#
# Grado por SEGMENTO (la PAS/PAD de referencia de cada segmento es la
# misma con la que se definió la etiqueta 140/90), guía ESC 2024:
#   - No elevada (clase negativa): PAS < 140 y PAD < 90
#   - HTA grado 1: PAS 140–159 y/o PAD 90–99 (y no grado 2)
#   - HTA grado >= 2: PAS >= 160 y/o PAD >= 100
#
# Métricas:
#   - AUC grado 1 vs no elevada y AUC grado >= 2 vs no elevada.
#   - AUC grado >= 2 vs resto (no elevada + grado 1): definición del
#     análisis exploratorio previo que dio 0,814 en validación.
#   - Sensibilidad por grado con los umbrales fijados en validación
#     (0,4698 y 0,3039), sin recalibrar; especificidad en no elevada.
#   - IC 95 %: bootstrap por sujeto (validación 1000, CalFree 2000,
#     semilla 42, igual que los puntos 35 y 36).
#
# CalFree se usa solo para describir; no se elige nada con él.
#
# Salidas: resultados/4_cnn_ensemble/analisis_severidad_47.csv / .json
# ============================================================

RAIZ = r"E:\Tesis_PulseDB"
RES = os.path.join(RAIZ, "resultados")
PROC = os.path.join(RAIZ, "procesado")
SEED = 42
UMBRALES = {"Youden 0,4698": 0.4698, "Sens. 80 % 0,3039": 0.3039}
GRUPOS = {0: "No elevada", 1: "HTA grado 1", 2: "HTA grado >= 2"}


def grado(sbp, dbp):
    g = np.zeros(len(sbp), dtype=int)
    g[(sbp >= 140) | (dbp >= 90)] = 1
    g[(sbp >= 160) | (dbp >= 100)] = 2
    return g


def texto(f, ref):
    v = np.asarray(f[ref]).reshape(-1)
    return "".join(chr(int(x)) for x in v if int(x) != 0).strip()


def sujetos_contiguos(f, refs):
    n = len(refs)
    cache = {}

    def suj(i):
        if i not in cache:
            cache[i] = texto(f, refs[i])
        return cache[i]

    out = np.empty(n, dtype=object)
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
        out[i:hi] = actual
        i = hi
    muestra = np.random.default_rng(SEED).choice(n, min(2000, n), replace=False)
    if sum(texto(f, refs[k]) != out[k] for k in muestra):
        raise RuntimeError("Sujetos no contiguos; revisar lectura.")
    return out.astype(str)


# ------------------------------------------------------------
# Validación (punto 35)
# ------------------------------------------------------------
idx_val = np.load(os.path.join(PROC, "indices_val.npy"))
with h5py.File(os.path.join(RAIZ, "data_train", "VitalDB_Train_Subset.mat"), "r") as f:
    sbp_v = np.asarray(f["Subset"]["SBP"]).reshape(-1)[idx_val].astype(np.float64)
    dbp_v = np.asarray(f["Subset"]["DBP"]).reshape(-1)[idx_val].astype(np.float64)
suj_v = np.load(os.path.join(PROC, "demografia_32.npz"), allow_pickle=True)["sujeto"][idx_val].astype(str)
p_v = np.load(os.path.join(RES, "4_cnn_ensemble", "predicciones_val_ensemble_35.npz"))["prob_ensemble"].astype(np.float64)
y_v = np.load(os.path.join(PROC, "etiquetas_val_140_90.npy")).astype(int)

# ------------------------------------------------------------
# CalFree (punto 36)
# ------------------------------------------------------------
with h5py.File(os.path.join(RAIZ, "data_calfree", "VitalDB_CalFree_Test_Subset.mat"), "r") as f:
    s = f["Subset"]
    sbp_c = np.asarray(s["SBP"]).reshape(-1).astype(np.float64)
    dbp_c = np.asarray(s["DBP"]).reshape(-1).astype(np.float64)
    suj_c = sujetos_contiguos(f, np.asarray(s["Subject"]).reshape(-1))
cal = np.load(os.path.join(RES, "4_cnn_ensemble", "probabilidades_calfree_ensemble_36.npz"))
p_c = cal["prob_ensemble"].astype(np.float64)
y_c = cal["etiquetas"].astype(int)

# Controles: mismo modelo y mismas etiquetas que el resultado principal.
for nombre, y, sbp, dbp, p, ref in [("validación", y_v, sbp_v, dbp_v, p_v, 0.7388),
                                     ("CalFree", y_c, sbp_c, dbp_c, p_c, 0.6946)]:
    assert np.array_equal(y, ((sbp >= 140) | (dbp >= 90)).astype(int)), f"Etiquetas distintas en {nombre}"
    assert abs(roc_auc_score(y, p) - ref) < 5e-4, f"AUC distinto del principal en {nombre}"
assert len(np.unique(suj_v)) == 259 and len(np.unique(suj_c)) == 144


# ------------------------------------------------------------
# Análisis
# ------------------------------------------------------------
def analizar(nombre, sbp, dbp, p, sujeto, n_boot):
    g = grado(sbp, dbp)
    _, cod = np.unique(sujeto, return_inverse=True)
    pos = [np.flatnonzero(cod == k) for k in range(cod.max() + 1)]
    rng = np.random.default_rng(SEED)
    rem = [np.concatenate([pos[k] for k in rng.integers(0, len(pos), len(pos))]) for _ in range(n_boot)]

    filas = []
    # Composición
    for k, etiqueta in GRUPOS.items():
        m = g == k
        filas.append({"conjunto": nombre, "metrica": "composicion", "grupo": etiqueta,
                      "segmentos": int(m.sum()), "sujetos": int(len(np.unique(sujeto[m]))),
                      "poco_fiable": bool(len(np.unique(sujeto[m])) < 10)})

    # AUC
    def auc_par(idx, positivo, negativo):
        gi, pi = g[idx], p[idx]
        sel = np.isin(gi, positivo + negativo)
        yy = np.isin(gi[sel], positivo).astype(int)
        return roc_auc_score(yy, pi[sel]) if 0 < yy.sum() < len(yy) else np.nan

    todos = np.arange(len(p))
    for etiqueta, positivo, negativo in [("Grado 1 vs no elevada", [1], [0]),
                                         ("Grado >= 2 vs no elevada", [2], [0]),
                                         ("Grado >= 2 vs resto (def. del 0,814)", [2], [0, 1]),
                                         ("HTA total vs no elevada (principal)", [1, 2], [0])]:
        auc = auc_par(todos, positivo, negativo)
        vals = np.array([auc_par(i, positivo, negativo) for i in rem])
        vals = vals[~np.isnan(vals)]
        lo, hi = np.percentile(vals, [2.5, 97.5])
        suj_pos = len(np.unique(sujeto[np.isin(g, positivo)]))
        filas.append({"conjunto": nombre, "metrica": "auc", "grupo": etiqueta, "valor": auc,
                      "ic95_inf": lo, "ic95_sup": hi, "remuestreos_validos": int(len(vals)),
                      "sujetos_positivos": int(suj_pos), "poco_fiable": bool(suj_pos < 10)})

    # Diferencia pareada grado >= 2 − grado 1 (ambos vs no elevada)
    d = np.array([auc_par(i, [2], [0]) - auc_par(i, [1], [0]) for i in rem])
    d = d[~np.isnan(d)]
    lo, hi = np.percentile(d, [2.5, 97.5])
    filas.append({"conjunto": nombre, "metrica": "diferencia_auc", "grupo": "Grado >= 2 − grado 1 (vs no elevada)",
                  "valor": auc_par(todos, [2], [0]) - auc_par(todos, [1], [0]), "ic95_inf": lo, "ic95_sup": hi,
                  "significativa": bool(lo > 0 or hi < 0)})

    # Sensibilidad por grado y especificidad en no elevada
    for unombre, u in UMBRALES.items():
        for k, etiqueta in GRUPOS.items():
            def tasa(idx):
                m = g[idx] == k
                if not m.any():
                    return np.nan
                acierto = p[idx][m] >= u if k > 0 else p[idx][m] < u
                return acierto.mean()
            vals = np.array([tasa(i) for i in rem])
            vals = vals[~np.isnan(vals)]
            lo, hi = np.percentile(vals, [2.5, 97.5])
            filas.append({"conjunto": nombre, "metrica": f"{'sensibilidad' if k > 0 else 'especificidad'} | {unombre}",
                          "grupo": etiqueta, "valor": tasa(todos), "ic95_inf": lo, "ic95_sup": hi})
    return filas


filas = analizar("Validación", sbp_v, dbp_v, p_v, suj_v, 1000) + analizar("CalFree", sbp_c, dbp_c, p_c, suj_c, 2000)
df = pd.DataFrame(filas)
df.to_csv(os.path.join(RES, "4_cnn_ensemble", "analisis_severidad_47.csv"), index=False, encoding="utf-8-sig")
json.dump({"descripcion": "Análisis complementario por severidad (grado por segmento, ESC 2024). Modelo final "
                          "CNN-Ensemble; probabilidades de los puntos 35 y 36; umbrales fijados en validación.",
           "bootstrap": {"tipo": "por sujeto", "validacion": 1000, "calfree": 2000, "semilla": SEED},
           "resultados": filas},
          open(os.path.join(RES, "4_cnn_ensemble", "analisis_severidad_47.json"), "w", encoding="utf-8"),
          indent=2, ensure_ascii=False, default=float)

pd.set_option("display.width", 200)
for c in ["Validación", "CalFree"]:
    print("=" * 78 + f"\n{c}\n" + "=" * 78)
    print(df[(df.conjunto == c)].drop(columns="conjunto").to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print("\nControles OK: mismas etiquetas y mismo AUC que el resultado principal (0,7388 y 0,6946).")
print("No se entrenó ni se volvió a inferir ningún modelo. CalFree solo se usó para describir.")
