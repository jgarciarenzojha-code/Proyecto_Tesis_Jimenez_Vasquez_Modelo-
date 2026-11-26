import os
import json
import time
import h5py
import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, confusion_matrix
)

# ============================================================
# PUNTO 36 - EVALUACIÓN FINAL DEL ENSEMBLE EN CALFREE
#
# "Examen final": 144 personas que el modelo NUNCA vio
# (VitalDB_CalFree_Test_Subset).
#
# Configuración CONGELADA en el punto 35 (solo con Validation):
# - Ensemble = promedio de logits de CNN V2 + V3 + multitarea.
# - Umbral Youden y umbral de sensibilidad >= 80 %.
# Aquí NO se ajusta nada: solo se mide.
#
# Transparencia: CalFree ya había sido usado una vez (punto 26)
# para evaluar la CNN V2. Ninguna decisión del ensemble usó CalFree.
#
# Guarda (archivos NUEVOS):
# - resultados/4_cnn_ensemble/evaluacion_final_ensemble_calfree_36.csv
# - resultados/4_cnn_ensemble/evaluacion_final_ensemble_calfree_36.json
# - resultados/4_cnn_ensemble/probabilidades_calfree_ensemble_36.npz
# - figuras/roc_ensemble_calfree_36.png
# - figuras/matrices_confusion_calfree_36.png
# ============================================================

RAIZ = r"E:\Tesis_PulseDB"
RUTA_MAT = os.path.join(RAIZ, "data_calfree", "VitalDB_CalFree_Test_Subset.mat")
RUTA_CONFIG = os.path.join(RAIZ, "resultados", "4_cnn_ensemble", "ensemble_35.json")
RUTA_PRED_VAL = os.path.join(RAIZ, "resultados", "4_cnn_ensemble", "predicciones_val_ensemble_35.npz")
CARPETA_MODELOS = os.path.join(RAIZ, "modelos")
CARPETA_RESULTADOS = os.path.join(RAIZ, "resultados")
CARPETA_FIGURAS = os.path.join(RAIZ, "figuras")

RUTA_CSV = os.path.join(CARPETA_RESULTADOS, "4_cnn_ensemble", "evaluacion_final_ensemble_calfree_36.csv")
RUTA_JSON = os.path.join(CARPETA_RESULTADOS, "4_cnn_ensemble", "evaluacion_final_ensemble_calfree_36.json")
RUTA_NPZ = os.path.join(CARPETA_RESULTADOS, "4_cnn_ensemble", "probabilidades_calfree_ensemble_36.npz")
RUTA_FIG_ROC = os.path.join(CARPETA_FIGURAS, "roc_ensemble_calfree_36.png")
RUTA_FIG_CM = os.path.join(CARPETA_FIGURAS, "matrices_confusion_calfree_36.png")

SEED = 42
N_BOOTSTRAP = 2000

print("=" * 78)
print("PUNTO 36 - EVALUACIÓN FINAL DEL ENSEMBLE EN CALFREE")
print("=" * 78)

# ------------------------------------------------------------
# Arquitecturas: copias exactas de las del punto 35
# (no se importa ese script para no volver a ejecutarlo).
# ------------------------------------------------------------
import torch.nn as nn


class BloqueResidual(nn.Module):

    def __init__(self, cin, cout, k, stride=1):
        super().__init__()
        p = k // 2
        self.conv1 = nn.Conv1d(cin, cout, k, stride=stride, padding=p, bias=False)
        self.bn1 = nn.BatchNorm1d(cout)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(cout, cout, k, stride=1, padding=p, bias=False)
        self.bn2 = nn.BatchNorm1d(cout)
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(nn.Conv1d(cin, cout, 1, stride=stride, bias=False), nn.BatchNorm1d(cout))
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class CNNV2(nn.Module):

    def __init__(self):
        super().__init__()
        self.entrada = nn.Sequential(nn.Conv1d(1, 32, 15, stride=2, padding=7, bias=False), nn.BatchNorm1d(32), nn.ReLU())
        self.block1 = BloqueResidual(32, 32, 7, 1)
        self.block2 = BloqueResidual(32, 64, 7, 2)
        self.block3 = BloqueResidual(64, 128, 5, 2)
        self.block4 = BloqueResidual(128, 256, 3, 2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.clasificador = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.4), nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.pool(self.block4(self.block3(self.block2(self.block1(self.entrada(x))))))
        return self.clasificador(x).squeeze(1)


class BloqueMultiEscala(nn.Module):

    def __init__(self, cin, c=32):
        super().__init__()

        def rama(k):
            return nn.Sequential(nn.Conv1d(cin, c, k, padding=k // 2, bias=False), nn.BatchNorm1d(c), nn.ReLU())

        self.branch_3, self.branch_7, self.branch_15, self.branch_31 = rama(3), rama(7), rama(15), rama(31)

    def forward(self, x):
        return torch.cat([self.branch_3(x), self.branch_7(x), self.branch_15(x), self.branch_31(x)], dim=1)


class CNNV3(nn.Module):

    def __init__(self):
        super().__init__()
        self.entry = nn.Sequential(nn.Conv1d(1, 32, 15, stride=2, padding=7, bias=False), nn.BatchNorm1d(32), nn.ReLU())
        self.multiscale = BloqueMultiEscala(32, 32)
        self.res1 = BloqueResidual(128, 128, 7, 2)
        self.res2 = BloqueResidual(128, 256, 5, 2)
        self.res3 = BloqueResidual(256, 256, 3, 2)
        self.pool = nn.AdaptiveAvgPool1d(4)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.40), nn.Linear(256 * 4, 128), nn.ReLU(), nn.Dropout(0.30), nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.pool(self.res3(self.res2(self.res1(self.multiscale(self.entry(x))))))
        return self.classifier(x).squeeze(1)


class CNNMultiTarea(nn.Module):

    def __init__(self):
        super().__init__()
        self.entry = nn.Sequential(nn.Conv1d(1, 32, 15, stride=2, padding=7, bias=False), nn.BatchNorm1d(32), nn.ReLU())
        self.block1 = BloqueResidual(32, 32, 7, 1)
        self.block2 = BloqueResidual(32, 64, 7, 2)
        self.block3 = BloqueResidual(64, 128, 5, 2)
        self.block4 = BloqueResidual(128, 256, 3, 2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.shared = nn.Sequential(nn.Flatten(), nn.Dropout(0.40), nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.30))
        self.head_class = nn.Linear(64, 1)
        self.head_sbp = nn.Linear(64, 1)
        self.head_dbp = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(self.block4(self.block3(self.block2(self.block1(self.entry(x))))))
        return self.head_class(self.shared(x)).squeeze(1)


CLASES = {"v2": CNNV2, "v3": CNNV3, "multitarea": CNNMultiTarea}

# ============================================================
# CONFIGURACIÓN CONGELADA (punto 35)
# ============================================================

with open(RUTA_CONFIG, "r", encoding="utf-8") as f:
    config = json.load(f)

UMBRALES = {
    "Youden": float(config["umbral_youden"]["umbral"]),
    "Sensibilidad >= 80 %": float(config["umbral_sensibilidad_80"]["umbral"]),
}

print("Configuración congelada (punto 35):")
for n, u in UMBRALES.items():
    print(f"  Umbral {n}: {u:.4f}")
print(f"  AUC en Validation: {config['auc_val']:.4f}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {device}")

redes = {}
for nombre, archivo in config["modelos"].items():
    red = CLASES[nombre]().to(device)
    ckpt = torch.load(os.path.join(CARPETA_MODELOS, archivo), map_location=device, weights_only=False)
    red.load_state_dict(ckpt["model_state_dict"])
    red.eval()
    redes[nombre] = red

# ============================================================
# DATOS DE CALFREE
# ============================================================


def decodificar(f, ref):
    v = np.asarray(f[ref]).reshape(-1)
    return "".join(chr(int(x)) for x in v if int(x) != 0).strip()


print("\nLeyendo CalFree ...")

with h5py.File(RUTA_MAT, "r") as f:

    s = f["Subset"]
    sbp = np.asarray(s["SBP"]).reshape(-1).astype(np.float64)
    dbp = np.asarray(s["DBP"]).reshape(-1).astype(np.float64)
    refs = np.asarray(s["Subject"]).reshape(-1)
    n = len(sbp)

    # Sujetos: bloques contiguos (búsqueda binaria + verificación aleatoria).
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
    errores = sum(decodificar(f, refs[k]) != sujeto[k] for k in muestra)
    if errores:
        raise RuntimeError("Sujetos no contiguos en CalFree; revisar lectura.")
    sujeto = sujeto.astype(str)

    y = ((sbp >= 140) | (dbp >= 90)).astype(int)
    print(f"Segmentos: {n:,} | sujetos: {len(np.unique(sujeto))} | "
          f"prevalencia 140/90: {y.mean():.4f} (verificación de sujetos: 0 errores en 2000)")

    # --------------------------------------------------------
    # Inferencia por tramos contiguos
    # --------------------------------------------------------
    signals = s["Signals"]
    logits = {nm: np.empty(n) for nm in redes}
    TAM = 400
    inicio = time.time()

    with torch.no_grad():
        for a in range(0, n, TAM):
            b = min(a + TAM, n)
            bloque = np.asarray(signals[:, 0, a:b], dtype=np.float32).T
            std = bloque.std(axis=1, keepdims=True)
            std[std < 1e-6] = 1.0
            x = torch.from_numpy((bloque - bloque.mean(axis=1, keepdims=True)) / std).unsqueeze(1).to(device)
            for nm, red in redes.items():
                logits[nm][a:b] = red(x).cpu().numpy()

    print(f"Inferencia de 3 CNN terminada en {time.time() - inicio:.0f} s")

sigmoide = lambda z: 1 / (1 + np.exp(-z))
prob_ens = sigmoide(np.mean([logits[nm] for nm in redes], axis=0))
prob_v2 = sigmoide(logits["v2"])

# Control: la V2 debe reproducir el resultado del punto 26.
print(f"Control: AUC CNN V2 en CalFree = {roc_auc_score(y, prob_v2):.4f} (punto 26 reportó 0.6781)")

# ============================================================
# MÉTRICAS CON IC 95 % (bootstrap por sujeto)
# ============================================================

sujetos_unicos, codigo = np.unique(sujeto, return_inverse=True)
pos_suj = [np.flatnonzero(codigo == k) for k in range(len(sujetos_unicos))]
rng = np.random.default_rng(SEED)
remuestreos = [
    np.concatenate([pos_suj[k] for k in rng.integers(0, len(pos_suj), len(pos_suj))])
    for _ in range(N_BOOTSTRAP)
]


def ic(func):
    vals = [func(i) for i in remuestreos if y[i].min() != y[i].max()]
    return np.percentile(vals, [2.5, 97.5])


def metricas_umbral(prob, umbral, idx=None):
    yy = y if idx is None else y[idx]
    pp = prob if idx is None else prob[idx]
    pred = (pp >= umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(yy, pred, labels=[0, 1]).ravel()
    return {
        "sensibilidad": tp / (tp + fn) if tp + fn else 0.0,
        "especificidad": tn / (tn + fp) if tn + fp else 0.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "exactitud": (tp + tn) / len(pred),
        "f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


print("\n" + "=" * 78)
print("RESULTADO FINAL EN CALFREE (144 personas nunca vistas)")
print("=" * 78)

auc_ens = roc_auc_score(y, prob_ens)
auc_v2 = roc_auc_score(y, prob_v2)
ic_ens = ic(lambda i: roc_auc_score(y[i], prob_ens[i]))
ic_v2 = ic(lambda i: roc_auc_score(y[i], prob_v2[i]))
ic_dif = ic(lambda i: roc_auc_score(y[i], prob_ens[i]) - roc_auc_score(y[i], prob_v2[i]))
pr_ens = average_precision_score(y, prob_ens)

print(f"Ensemble  ROC-AUC = {auc_ens:.4f}  [IC95 {ic_ens[0]:.4f} – {ic_ens[1]:.4f}]  PR-AUC = {pr_ens:.4f}")
print(f"CNN V2    ROC-AUC = {auc_v2:.4f}  [IC95 {ic_v2[0]:.4f} – {ic_v2[1]:.4f}]")
print(f"Mejora ensemble − V2 = {auc_ens - auc_v2:+.4f}  [IC95 {ic_dif[0]:+.4f} – {ic_dif[1]:+.4f}]")
print(f"Referencia: AUC del ensemble en Validation = {config['auc_val']:.4f}")

filas = [{
    "modelo": "Ensemble (V2 + V3 + multitarea)", "umbral": "—", "roc_auc": auc_ens,
    "roc_auc_ic_inf": ic_ens[0], "roc_auc_ic_sup": ic_ens[1], "pr_auc": pr_ens,
}, {
    "modelo": "CNN V2", "umbral": "—", "roc_auc": auc_v2,
    "roc_auc_ic_inf": ic_v2[0], "roc_auc_ic_sup": ic_v2[1], "pr_auc": average_precision_score(y, prob_v2),
}]

resultados_umbral = {}

for nombre, umbral in UMBRALES.items():

    m = metricas_umbral(prob_ens, umbral)
    ic_sens = ic(lambda i: metricas_umbral(prob_ens, umbral, i)["sensibilidad"])
    ic_esp = ic(lambda i: metricas_umbral(prob_ens, umbral, i)["especificidad"])
    m["ic_sensibilidad"] = ic_sens.tolist()
    m["ic_especificidad"] = ic_esp.tolist()
    resultados_umbral[nombre] = {"umbral": umbral, **m}

    print(
        f"\nUmbral {nombre} ({umbral:.4f}):\n"
        f"  Sensibilidad  = {m['sensibilidad']:.4f}  [IC95 {ic_sens[0]:.4f} – {ic_sens[1]:.4f}]\n"
        f"  Especificidad = {m['especificidad']:.4f}  [IC95 {ic_esp[0]:.4f} – {ic_esp[1]:.4f}]\n"
        f"  Precisión     = {m['precision']:.4f}\n"
        f"  Exactitud     = {m['exactitud']:.4f}\n"
        f"  F1            = {m['f1']:.4f}\n"
        f"  TN={m['tn']:,}  FP={m['fp']:,}  FN={m['fn']:,}  TP={m['tp']:,}"
    )

    filas.append({"modelo": "Ensemble", "umbral": f"{nombre} ({umbral:.4f})", "roc_auc": auc_ens, **m})

# ============================================================
# ANÁLISIS COMPLEMENTARIO POR PERSONA
# ============================================================

print("\n" + "=" * 78)
print("ANÁLISIS COMPLEMENTARIO POR PERSONA (promedio de sus segmentos)")
print("=" * 78)

cnt = np.bincount(codigo)
media = lambda v: np.bincount(codigo, weights=v) / cnt
prob_persona = media(prob_ens)
y_persona = ((media(sbp) >= 140) | (media(dbp) >= 90)).astype(int)

persona = {"personas": int(len(sujetos_unicos)), "positivas": int(y_persona.sum())}

if 0 < y_persona.sum() < len(y_persona):
    auc_p = roc_auc_score(y_persona, prob_persona)
    vals = []
    r2 = np.random.default_rng(SEED)
    for _ in range(N_BOOTSTRAP):
        k = r2.integers(0, len(y_persona), len(y_persona))
        if 0 < y_persona[k].sum() < len(k):
            vals.append(roc_auc_score(y_persona[k], prob_persona[k]))
    ic_p = np.percentile(vals, [2.5, 97.5])
    persona.update({"auc": float(auc_p), "ic95": ic_p.tolist()})
    print(f"Personas: {len(y_persona)} | positivas (presión promedio >= 140/90): {y_persona.sum()}")
    print(f"AUC por persona = {auc_p:.4f}  [IC95 {ic_p[0]:.4f} – {ic_p[1]:.4f}]")
    print("NOTA: pocas personas positivas → intervalo amplio; usar solo como análisis complementario.")
else:
    print("No hay suficientes personas positivas y negativas para calcular el AUC por persona.")

# ============================================================
# FIGURAS PARA EL INFORME
# ============================================================

os.makedirs(CARPETA_FIGURAS, exist_ok=True)

pred_val = np.load(RUTA_PRED_VAL)
y_val = np.load(os.path.join(RAIZ, "procesado", "etiquetas_val_140_90.npy")).astype(int)

fig, ax = plt.subplots(figsize=(6.4, 6.0), dpi=200)
for etiqueta, yy, pp, color in [
    (f"Validación (AUC = {config['auc_val']:.3f})", y_val, pred_val["prob_ensemble"], "#2563eb"),
    (f"CalFree – prueba final (AUC = {auc_ens:.3f})", y, prob_ens, "#dc2626"),
]:
    fpr, tpr, _ = roc_curve(yy, pp)
    ax.plot(fpr, tpr, color=color, lw=2.2, label=etiqueta)

for nombre, umbral in UMBRALES.items():
    m = resultados_umbral[nombre]
    ax.scatter(1 - m["especificidad"], m["sensibilidad"], s=55, zorder=5, color="#111827",
               marker="o" if nombre == "Youden" else "s")
    ax.annotate(f"{nombre} ({umbral:.2f})", (1 - m["especificidad"], m["sensibilidad"]),
                textcoords="offset points", xytext=(8, -12), fontsize=8.5)

ax.plot([0, 1], [0, 1], ls="--", color="#94a3b8", lw=1, label="Azar (AUC = 0.5)")
ax.set_xlabel("1 − Especificidad (tasa de falsos positivos)")
ax.set_ylabel("Sensibilidad (tasa de verdaderos positivos)")
ax.set_title("Curva ROC – Ensemble de CNN (criterio 140/90)")
ax.legend(loc="lower right", fontsize=8.5)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(RUTA_FIG_ROC)
plt.close(fig)

fig, ejes = plt.subplots(1, 2, figsize=(10, 4.4), dpi=200)
for ax, (nombre, umbral) in zip(ejes, UMBRALES.items()):
    m = resultados_umbral[nombre]
    matriz = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    ax.imshow(matriz, cmap="Blues")
    for (r, c), v in np.ndenumerate(matriz):
        ax.text(c, r, f"{v:,}\n({v / matriz.sum() * 100:.1f} %)", ha="center", va="center",
                color="white" if v > matriz.max() * 0.5 else "#111827", fontsize=10)
    ax.set_xticks([0, 1], ["Predicho: < 140/90", "Predicho: ≥ 140/90"])
    ax.set_yticks([0, 1], ["Real: < 140/90", "Real: ≥ 140/90"])
    ax.set_title(f"Umbral {nombre} = {umbral:.4f}\nSens {m['sensibilidad']:.2f} · Espec {m['especificidad']:.2f}",
                 fontsize=10)
fig.suptitle("Matrices de confusión – Ensemble en CalFree (57 600 segmentos, 144 personas)", fontsize=11)
fig.tight_layout()
fig.savefig(RUTA_FIG_CM)
plt.close(fig)

# ============================================================
# GUARDAR
# ============================================================

pd.DataFrame(filas).to_csv(RUTA_CSV, index=False)
np.savez_compressed(RUTA_NPZ, prob_ensemble=prob_ens.astype(np.float32), etiquetas=y.astype(np.int8),
                    **{f"logit_{nm}": logits[nm].astype(np.float32) for nm in redes})

with open(RUTA_JSON, "w", encoding="utf-8") as f:
    json.dump(
        {
            "dataset": "VitalDB_CalFree_Test_Subset",
            "segmentos": int(n),
            "sujetos": int(len(sujetos_unicos)),
            "prevalencia": float(y.mean()),
            "ensemble": {"roc_auc": float(auc_ens), "ic95": ic_ens.tolist(), "pr_auc": float(pr_ens)},
            "cnn_v2": {"roc_auc": float(auc_v2), "ic95": ic_v2.tolist()},
            "mejora_ensemble_sobre_v2": {"dif": float(auc_ens - auc_v2), "ic95": ic_dif.tolist()},
            "umbrales": resultados_umbral,
            "por_persona": persona,
            "transparencia": "CalFree fue usado una vez antes (punto 26) para la V2; "
                             "ninguna decisión del ensemble usó CalFree.",
        },
        f, ensure_ascii=False, indent=2, default=float,
    )

print("\n" + "=" * 78)
print("PUNTO 36 FINALIZADO")
print("=" * 78)
print(f"Tabla:  {RUTA_CSV}")
print(f"Figuras: {RUTA_FIG_ROC}\n         {RUTA_FIG_CM}")
print(
    "\nIMPORTANTE:"
    "\n- Configuración y umbrales congelados del punto 35 (sin ajustes en CalFree)."
    "\n- Evaluación realizada UNA sola vez."
)
