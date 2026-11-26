import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

# ============================================================
# PUNTO 46 - ENSEMBLE AMPLIADO FRENTE AL ENSEMBLE ACTUAL (VALIDACIÓN)
#
# Combina las tres CNN reentrenadas con los sujetos nuevos (punto 45),
# igual que el punto 35: promedio de logits con pesos iguales.
# Lo compara con el ensemble actual en los MISMOS 259 sujetos de
# validación, con bootstrap por sujeto (1000 remuestreos, semilla 42,
# igual que el punto 35) y diferencias pareadas.
#
# CalFree NO se usa. No modifica ningún modelo.
#
# Salida: resultados/ensemble_ampliado_validacion_46.json / .csv
# ============================================================

RAIZ = r"E:\Tesis_PulseDB"
RES = os.path.join(RAIZ, "resultados")
SEED = 42
N_BOOTSTRAP = 1000
DATOS = "ampliado"

y = np.load(os.path.join(RAIZ, "procesado", "etiquetas_val_140_90.npy")).astype(int)
sujeto = np.load(os.path.join(RAIZ, "procesado", "demografia_32.npz"), allow_pickle=True)["sujeto"][
    np.load(os.path.join(RAIZ, "procesado", "indices_val.npy"))].astype(str)
assert len(np.unique(sujeto)) == 259

actual = np.load(os.path.join(RES, "4_cnn_ensemble", "predicciones_val_ensemble_35.npz"))
sig = lambda z: 1 / (1 + np.exp(-z))
redes = ["v2", "v3", "multitarea"]
nuevos = {r: np.load(os.path.join(RES, "5_experimentos", "200_pacientes", f"logits_val_{r}_{DATOS}_45.npy")).astype(np.float64) for r in redes}

puntajes = {
    "Ensemble actual": actual["prob_ensemble"].astype(np.float64),
    **{f"CNN {r} actual": actual[f"logit_{r}"].astype(np.float64) for r in redes},
    **{f"CNN {r} ampliada": nuevos[r] for r in redes},
    "Ensemble ampliado": sig(np.mean([nuevos[r] for r in redes], axis=0)),
}
assert abs(roc_auc_score(y, puntajes["Ensemble actual"]) - 0.7388) < 5e-4

_, codigo = np.unique(sujeto, return_inverse=True)
pos = [np.flatnonzero(codigo == k) for k in range(codigo.max() + 1)]
rng = np.random.default_rng(SEED)
rem = [np.concatenate([pos[k] for k in rng.integers(0, len(pos), len(pos))]) for _ in range(N_BOOTSTRAP)]
rem = [i for i in rem if y[i].min() != y[i].max()]
boot = {m: np.array([roc_auc_score(y[i], p[i]) for i in rem]) for m, p in puntajes.items()}

filas = []
print("=" * 78)
print("PUNTO 46 - VALIDACIÓN: ENSEMBLE AMPLIADO FRENTE AL ACTUAL")
print("=" * 78)
for m, p in puntajes.items():
    auc = roc_auc_score(y, p)
    lo, hi = np.percentile(boot[m], [2.5, 97.5])
    filas.append({"modelo": m, "roc_auc": auc, "ic95_inf": lo, "ic95_sup": hi, "pr_auc": average_precision_score(y, p)})
    print(f"{m:<26} AUC {auc:.4f} [{lo:.4f} – {hi:.4f}]  PR-AUC {filas[-1]['pr_auc']:.4f}")

comparaciones = [("Ensemble ampliado", "Ensemble actual")] + [(f"CNN {r} ampliada", f"CNN {r} actual") for r in redes]
difs = []
print("\nDiferencias pareadas por sujeto (nuevo − actual):")
for a, b in comparaciones:
    d = boot[a] - boot[b]
    lo, hi = np.percentile(d, [2.5, 97.5])
    dif = roc_auc_score(y, puntajes[a]) - roc_auc_score(y, puntajes[b])
    difs.append({"comparacion": f"{a} − {b}", "dif": dif, "ic95_inf": lo, "ic95_sup": hi,
                 "prob_mejora": float((d > 0).mean()), "significativa": bool(lo > 0 or hi < 0)})
    print(f"  {a} − {b}: {dif:+.4f} [{lo:+.4f}, {hi:+.4f}] | P(mejora) {difs[-1]['prob_mejora']:.2f}"
          f" | {'significativa' if difs[-1]['significativa'] else 'no significativa'}")

pd.DataFrame(filas).to_csv(os.path.join(RES, "5_experimentos", "200_pacientes", "ensemble_ampliado_validacion_46.csv"), index=False, encoding="utf-8-sig")
json.dump({"bootstrap": {"tipo": "por sujeto", "remuestreos": N_BOOTSTRAP, "semilla": SEED},
           "resultados": filas, "diferencias": difs,
           "nota": "Solo validación (259 sujetos). CalFree no usado."},
          open(os.path.join(RES, "5_experimentos", "200_pacientes", "ensemble_ampliado_validacion_46.json"), "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)
print("\nCalFree NO fue utilizado. Ningún modelo fue modificado.")
