import os
import json
import time
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, confusion_matrix
)

# ============================================================
# PUNTO 35 - ENSEMBLE DE CNN (V2 + V3 + MULTITAREA), SOLO ECG
#
# No se entrena ninguna red: se combinan las 3 CNN ya entrenadas
# (puntos 23, 30 y 31) promediando sus logits con PESOS IGUALES
# (sin ajustar pesos, para no sobreajustar Validation).
#
# - Predicciones en Validation de las 3 CNN.
# - AUC con IC 95 % (bootstrap por sujeto) y comparación pareada vs V2.
# - Umbrales elegidos SOLO con Validation:
#     · Youden (equilibrio sensibilidad/especificidad)
#     · Sensibilidad global >= 80 %
# - CalFree NO se usa (queda para la evaluación final, una sola vez).
#
# Guarda (archivos NUEVOS):
# - resultados/4_cnn_ensemble/predicciones_val_ensemble_35.npz
# - resultados/4_cnn_ensemble/ensemble_35.csv
# - resultados/4_cnn_ensemble/ensemble_35.json   (umbrales congelados para CalFree y la app)
# ============================================================

RUTA_MAT = r"E:\Tesis_PulseDB\data_train\VitalDB_Train_Subset.mat"
CARPETA_PROCESADO = r"E:\Tesis_PulseDB\procesado"
CARPETA_RESULTADOS = r"E:\Tesis_PulseDB\resultados"
CARPETA_MODELOS = r"E:\Tesis_PulseDB\modelos"

MODELOS = {
    "v2": "mejor_modelo_v2_regularizada_140_90.pt",
    "v3": "mejor_modelo_v3_multiescala_140_90.pt",
    "multitarea": "mejor_modelo_multitarea_140_90.pt",
}

RUTA_NPZ = os.path.join(CARPETA_RESULTADOS, "4_cnn_ensemble", "predicciones_val_ensemble_35.npz")
RUTA_CSV = os.path.join(CARPETA_RESULTADOS, "4_cnn_ensemble", "ensemble_35.csv")
RUTA_JSON = os.path.join(CARPETA_RESULTADOS, "4_cnn_ensemble", "ensemble_35.json")

SEED = 42
N_BOOTSTRAP = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 78)
print("PUNTO 35 - ENSEMBLE DE CNN (V2 + V3 + MULTITAREA)")
print("=" * 78)
print(f"Dispositivo: {device}")

# ============================================================
# ARQUITECTURAS (copias exactas de los puntos 23, 30 y 31)
# ============================================================

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
        return self.head_class(self.shared(x)).squeeze(1)      # solo clasificación


CLASES = {"v2": CNNV2, "v3": CNNV3, "multitarea": CNNMultiTarea}

# ============================================================
# DATOS
# ============================================================

indices_val = np.load(os.path.join(CARPETA_PROCESADO, "indices_val.npy")).astype(np.int64)
y_val = np.load(os.path.join(CARPETA_PROCESADO, "etiquetas_val_140_90.npy")).astype(int)
sujeto = np.load(os.path.join(CARPETA_PROCESADO, "demografia_32.npz"), allow_pickle=True)["sujeto"][indices_val]

# ============================================================
# INFERENCIA (se lee un sujeto por lote: tramos contiguos del archivo)
# ============================================================

redes = {}
for nombre, archivo in MODELOS.items():
    red = CLASES[nombre]().to(device)
    ckpt = torch.load(os.path.join(CARPETA_MODELOS, archivo), map_location=device, weights_only=False)
    red.load_state_dict(ckpt["model_state_dict"])
    red.eval()
    redes[nombre] = red

logits = {n: np.empty(len(indices_val)) for n in redes}
tramos = np.split(np.arange(len(indices_val)), np.flatnonzero(np.diff(indices_val) != 1) + 1)

print(f"\nInferencia de 3 CNN en {len(indices_val):,} segmentos ({len(tramos)} tramos) ...")
inicio = time.time()

with h5py.File(RUTA_MAT, "r") as f, torch.no_grad():
    signals = f["Subset"]["Signals"]

    for pos in tramos:
        bloque = np.asarray(signals[:, 0, indices_val[pos[0]]:indices_val[pos[-1]] + 1], dtype=np.float32).T
        std = bloque.std(axis=1, keepdims=True)
        std[std < 1e-6] = 1.0
        x = torch.from_numpy((bloque - bloque.mean(axis=1, keepdims=True)) / std).unsqueeze(1).to(device)

        for nombre, red in redes.items():
            logits[nombre][pos] = red(x).cpu().numpy()

print(f"Inferencia terminada en {time.time() - inicio:.0f} s")

sigmoide = lambda z: 1 / (1 + np.exp(-z))

# Ensemble: promedio de logits con pesos iguales.
logit_ens = np.mean([logits[n] for n in redes], axis=0)
prob_ens = sigmoide(logit_ens)

# ============================================================
# MÉTRICAS CON IC (bootstrap por sujeto)
# ============================================================

sujetos_unicos, codigo = np.unique(sujeto, return_inverse=True)
pos_suj = [np.flatnonzero(codigo == k) for k in range(len(sujetos_unicos))]
rng = np.random.default_rng(SEED)
remuestreos = [
    np.concatenate([pos_suj[k] for k in rng.integers(0, len(pos_suj), len(pos_suj))])
    for _ in range(N_BOOTSTRAP)
]


def auc_ic(score):
    vals = [roc_auc_score(y_val[i], score[i]) for i in remuestreos if y_val[i].min() != y_val[i].max()]
    return roc_auc_score(y_val, score), *np.percentile(vals, [2.5, 97.5])


print("\n" + "=" * 78)
print("AUC EN VALIDATION")
print("=" * 78)

filas = []
puntajes = {**{f"CNN {n}": sigmoide(logits[n]) for n in redes}, "Ensemble (V2 + V3 + multitarea)": prob_ens}

for nombre, score in puntajes.items():
    auc, lo, hi = auc_ic(score)
    pr = average_precision_score(y_val, score)
    filas.append({"modelo": nombre, "auc": auc, "ic_inf": lo, "ic_sup": hi, "pr_auc": pr})
    print(f"{nombre:<34} AUC = {auc:.4f}  [IC95 {lo:.4f} – {hi:.4f}]  PR-AUC = {pr:.4f}")

difs = [
    roc_auc_score(y_val[i], prob_ens[i]) - roc_auc_score(y_val[i], logits["v2"][i])
    for i in remuestreos if y_val[i].min() != y_val[i].max()
]
dif = roc_auc_score(y_val, prob_ens) - roc_auc_score(y_val, logits["v2"])
ic_dif = np.percentile(difs, [2.5, 97.5])
print(f"\nMejora del ensemble sobre la V2: {dif:+.4f}  [IC95 {ic_dif[0]:+.4f} – {ic_dif[1]:+.4f}]")

# ============================================================
# UMBRALES (solo Validation)
# ============================================================

fpr, tpr, thr = roc_curve(y_val, prob_ens)

i_youden = int(np.argmax(tpr - fpr))
umbral_youden = float(thr[i_youden])

i_sens80 = int(np.argmax(tpr >= 0.80))
umbral_sens80 = float(thr[i_sens80])


def metricas(umbral):
    pred = (prob_ens >= umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_val, pred, labels=[0, 1]).ravel()
    return {
        "umbral": umbral,
        "sensibilidad": tp / (tp + fn),
        "especificidad": tn / (tn + fp),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "exactitud": (tp + tn) / len(pred),
        "f1": 2 * tp / (2 * tp + fp + fn),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


m_youden = metricas(umbral_youden)
m_sens80 = metricas(umbral_sens80)

print("\n" + "=" * 78)
print("UMBRALES DEL ENSEMBLE (elegidos solo con Validation)")
print("=" * 78)
for nombre, m in [("Youden", m_youden), ("Sensibilidad >= 80 %", m_sens80)]:
    print(
        f"{nombre:<22} umbral = {m['umbral']:.4f} | sens = {m['sensibilidad']:.4f} | "
        f"espec = {m['especificidad']:.4f} | precisión = {m['precision']:.4f} | F1 = {m['f1']:.4f}"
    )

# ============================================================
# GUARDAR
# ============================================================

np.savez_compressed(
    RUTA_NPZ,
    prob_ensemble=prob_ens.astype(np.float32),
    **{f"logit_{n}": logits[n].astype(np.float32) for n in redes},
)
pd.DataFrame(filas).to_csv(RUTA_CSV, index=False)

with open(RUTA_JSON, "w", encoding="utf-8") as fj:
    json.dump(
        {
            "modelos": MODELOS,
            "combinacion": "promedio de logits con pesos iguales; probabilidad = sigmoide(promedio)",
            "preprocesamiento": "z-score por segmento (std < 1e-6 -> 1)",
            "auc_val": float(roc_auc_score(y_val, prob_ens)),
            "mejora_sobre_v2": {"dif": float(dif), "ic95": ic_dif.tolist()},
            "umbral_youden": m_youden,
            "umbral_sensibilidad_80": m_sens80,
            "nota": "Umbrales congelados: elegidos solo con Validation; CalFree no usado.",
        },
        fj,
        ensure_ascii=False,
        indent=2,
        default=float,
    )

print("\n" + "=" * 78)
print("PUNTO 35 FINALIZADO")
print("=" * 78)
print(f"Tabla: {RUTA_CSV}")
print(f"Umbrales congelados: {RUTA_JSON}")
print(
    "\nIMPORTANTE:"
    "\n- Ninguna CNN fue reentrenada; pesos iguales (sin ajuste en Validation)."
    "\n- Umbrales elegidos solo con Validation."
    "\n- CalFree NO fue utilizado."
)
