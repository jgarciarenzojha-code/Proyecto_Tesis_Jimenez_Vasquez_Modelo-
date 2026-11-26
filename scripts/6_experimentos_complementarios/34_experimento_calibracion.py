import os
import json
import time
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

# ============================================================
# PUNTO 34 - EXPERIMENTO DE FACTIBILIDAD: CALIBRACIÓN PERSONAL
#
# Pregunta: si el especialista registra UNA medición de presión
# de referencia (tensiómetro) al inicio, ¿mejora la detección?
#
# Simulación en VALIDATION (sin reentrenar ninguna CNN):
# - Para cada sujeto, su PRIMER segmento es la "calibración":
#   se conoce su SBP/DBP real (como si se midiera con tensiómetro).
# - Se evalúan TODOS sus demás segmentos (la calibración se excluye).
#
# Variantes comparadas (mismos segmentos, mismos sujetos):
#   B : CNN V2 sola, sin referencia (situación actual).
#   R : SOLO la presión de referencia (sin ECG)  ← control clave.
#   K1: CNN multitarea calibrada: SBP_est = SBP_ref + (pred(t) − pred(t0)).
#   K2: combinación (referencia + CNN V2 + multitarea calibrada) con
#       regresión logística; validación cruzada de 5 grupos POR SUJETO.
#
# Reglas: CalFree NO se usa; IC 95 % con bootstrap por sujeto;
# solo se crean archivos NUEVOS.
# ============================================================

RUTA_MAT = r"E:\Tesis_PulseDB\data_train\VitalDB_Train_Subset.mat"
CARPETA_PROCESADO = r"E:\Tesis_PulseDB\procesado"
CARPETA_RESULTADOS = r"E:\Tesis_PulseDB\resultados"
RUTA_MULTITAREA = r"E:\Tesis_PulseDB\modelos\mejor_modelo_multitarea_140_90.pt"

RUTA_PRED_MT = os.path.join(CARPETA_RESULTADOS, "5_experimentos", "calibracion", "predicciones_val_multitarea_34.npz")
RUTA_CSV = os.path.join(CARPETA_RESULTADOS, "5_experimentos", "calibracion", "experimento_calibracion_34.csv")
RUTA_JSON = os.path.join(CARPETA_RESULTADOS, "5_experimentos", "calibracion", "experimento_calibracion_34.json")

SEED = 42
N_BOOTSTRAP = 1000
N_GRUPOS_CV = 5
BATCH = 512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 78)
print("PUNTO 34 - EXPERIMENTO DE FACTIBILIDAD: CALIBRACIÓN PERSONAL")
print("=" * 78)
print(f"Dispositivo: {device}")

# ============================================================
# DATOS DE VALIDATION
# ============================================================

indices_val = np.load(os.path.join(CARPETA_PROCESADO, "indices_val.npy")).astype(np.int64)
y_val = np.load(os.path.join(CARPETA_PROCESADO, "etiquetas_val_140_90.npy")).astype(int)
demo = np.load(os.path.join(CARPETA_PROCESADO, "demografia_32.npz"), allow_pickle=True)

sujeto = demo["sujeto"][indices_val]
sbp = demo["sbp"][indices_val].astype(np.float64)
dbp = demo["dbp"][indices_val].astype(np.float64)

prob_v2 = np.load(os.path.join(CARPETA_RESULTADOS, "1_cnn_residual", "probabilidades_validation_v2_140_90.npy")).astype(np.float64)
assert np.array_equal(
    np.load(os.path.join(CARPETA_RESULTADOS, "1_cnn_residual", "etiquetas_validation_v2_140_90.npy")).astype(int), y_val
)

# ============================================================
# CNN MULTITAREA (copia exacta de la arquitectura del punto 31)
# ============================================================

class ResidualBlock(nn.Module):

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


class CNNMultiTask(nn.Module):

    def __init__(self):
        super().__init__()
        self.entry = nn.Sequential(nn.Conv1d(1, 32, 15, stride=2, padding=7, bias=False), nn.BatchNorm1d(32), nn.ReLU())
        self.block1 = ResidualBlock(32, 32, 7, 1)
        self.block2 = ResidualBlock(32, 64, 7, 2)
        self.block3 = ResidualBlock(64, 128, 5, 2)
        self.block4 = ResidualBlock(128, 256, 3, 2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.shared = nn.Sequential(nn.Flatten(), nn.Dropout(0.40), nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.30))
        self.head_class = nn.Linear(64, 1)
        self.head_sbp = nn.Linear(64, 1)
        self.head_dbp = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(self.block4(self.block3(self.block2(self.block1(self.entry(x))))))
        f = self.shared(x)
        return self.head_class(f).squeeze(1), self.head_sbp(f).squeeze(1), self.head_dbp(f).squeeze(1)


# ============================================================
# INFERENCIA DE LA MULTITAREA EN VALIDATION (o carga si ya existe)
# ============================================================

if os.path.exists(RUTA_PRED_MT):

    print(f"\nUsando predicciones guardadas: {RUTA_PRED_MT}")
    guardado = np.load(RUTA_PRED_MT)
    prob_mt, sbp_pred, dbp_pred = guardado["prob"], guardado["sbp_mmHg"], guardado["dbp_mmHg"]

else:

    ckpt = torch.load(RUTA_MULTITAREA, map_location=device, weights_only=False)
    modelo = CNNMultiTask().to(device)
    modelo.load_state_dict(ckpt["model_state_dict"])
    modelo.eval()

    print(f"\nInferencia de la CNN multitarea en {len(indices_val):,} segmentos de validation ...")
    inicio = time.time()

    prob_mt = np.empty(len(indices_val))
    sbp_pred = np.empty(len(indices_val))
    dbp_pred = np.empty(len(indices_val))

    # Cada sujeto ocupa un tramo contiguo del archivo: se lee un sujeto por lote.
    cortes = np.flatnonzero(np.diff(indices_val) != 1) + 1
    tramos = np.split(np.arange(len(indices_val)), cortes)

    with h5py.File(RUTA_MAT, "r") as f, torch.no_grad():
        signals = f["Subset"]["Signals"]

        for n_tramo, pos in enumerate(tramos):
            a = int(pos[0])
            idx = indices_val[pos]
            lo, hi = int(idx[0]), int(idx[-1]) + 1
            bloque = np.asarray(signals[:, 0, lo:hi], dtype=np.float32).T   # [n, 1250]

            media = bloque.mean(axis=1, keepdims=True)
            std = bloque.std(axis=1, keepdims=True)
            std[std < 1e-6] = 1.0
            x = torch.from_numpy((bloque - media) / std).unsqueeze(1).to(device)

            logit, s_n, d_n = modelo(x)
            prob_mt[a:a + len(idx)] = torch.sigmoid(logit).cpu().numpy()
            sbp_pred[a:a + len(idx)] = s_n.cpu().numpy() * ckpt["sbp_std"] + ckpt["sbp_mean"]
            dbp_pred[a:a + len(idx)] = d_n.cpu().numpy() * ckpt["dbp_std"] + ckpt["dbp_mean"]

            if n_tramo % 40 == 0:
                print(f"  {a + len(idx):,}/{len(indices_val):,}")

    print(f"Inferencia terminada en {time.time() - inicio:.0f} s")
    np.savez_compressed(RUTA_PRED_MT, prob=prob_mt, sbp_mmHg=sbp_pred, dbp_mmHg=dbp_pred)

auc_mt_check = roc_auc_score(y_val, prob_mt)
print(f"Control: AUC de la multitarea en validation = {auc_mt_check:.4f} (punto 31 reportó 0.7263)")

# ============================================================
# PROTOCOLO DE CALIBRACIÓN
# ============================================================

sujetos_unicos, codigo = np.unique(sujeto, return_inverse=True)
n_suj = len(sujetos_unicos)

# Primer segmento (en orden del archivo) de cada sujeto = calibración.
primero = np.array([np.flatnonzero(codigo == k)[0] for k in range(n_suj)])
es_calibracion = np.zeros(len(y_val), dtype=bool)
es_calibracion[primero] = True

ref_sbp = sbp[primero][codigo]
ref_dbp = dbp[primero][codigo]
ref_sbp_pred = sbp_pred[primero][codigo]
ref_dbp_pred = dbp_pred[primero][codigo]
ref_logit_v2 = np.log(prob_v2[primero] / (1 - prob_v2[primero]))[codigo]

# Estimación calibrada (método clásico de presión sin brazalete).
sbp_est = ref_sbp + (sbp_pred - ref_sbp_pred)
dbp_est = ref_dbp + (dbp_pred - ref_dbp_pred)

logit_v2 = np.log(np.clip(prob_v2, 1e-6, 1 - 1e-6) / np.clip(1 - prob_v2, 1e-6, 1))

ev = ~es_calibracion                       # se evalúa sin el segmento de calibración
y_ev = y_val[ev]
cod_ev = codigo[ev]

print(f"\nSujetos: {n_suj} | segmentos evaluados: {ev.sum():,} (se excluyen {n_suj} de calibración)")


# ============================================================
# BOOTSTRAP POR SUJETO
# ============================================================

rng = np.random.default_rng(SEED)
pos_por_sujeto = [np.flatnonzero(cod_ev == k) for k in range(n_suj)]
remuestreos = [
    np.concatenate([pos_por_sujeto[k] for k in rng.integers(0, n_suj, n_suj)])
    for _ in range(N_BOOTSTRAP)
]


def con_ic(score):
    auc = roc_auc_score(y_ev, score)
    vals = [roc_auc_score(y_ev[i], score[i]) for i in remuestreos if y_ev[i].min() != y_ev[i].max()]
    return auc, np.percentile(vals, 2.5), np.percentile(vals, 97.5), average_precision_score(y_ev, score)


def ic_diferencia(score_1, score_2):
    vals = [
        roc_auc_score(y_ev[i], score_1[i]) - roc_auc_score(y_ev[i], score_2[i])
        for i in remuestreos if y_ev[i].min() != y_ev[i].max()
    ]
    return np.percentile(vals, [2.5, 97.5])


def validacion_cruzada_por_sujeto(X):
    """Regresión logística; cada sujeto se predice con un modelo que no lo vio."""

    grupos = np.random.default_rng(SEED).permutation(n_suj) % N_GRUPOS_CV
    g_seg = grupos[cod_ev]
    pred = np.empty(len(y_ev))

    for g in range(N_GRUPOS_CV):
        tr, te = g_seg != g, g_seg == g
        media, std = X[tr].mean(axis=0), X[tr].std(axis=0) + 1e-9
        m = LogisticRegression(class_weight="balanced", max_iter=2000)
        m.fit((X[tr] - media) / std, y_ev[tr])
        pred[te] = m.predict_proba((X[te] - media) / std)[:, 1]

    return pred


# ============================================================
# VARIANTES
# ============================================================

variantes = {}

variantes["B: CNN V2 sola (sin referencia)"] = prob_v2[ev]

X_r = np.column_stack([ref_sbp[ev], ref_dbp[ev]])
variantes["R: solo presión de referencia (sin ECG)"] = validacion_cruzada_por_sujeto(X_r)

puntaje_k1 = np.maximum((sbp_est[ev] - 140) / 15.0, (dbp_est[ev] - 90) / 10.0)
variantes["K1: multitarea calibrada (regla 140/90)"] = puntaje_k1

X_k2 = np.column_stack([
    ref_sbp[ev], ref_dbp[ev],
    sbp_est[ev], dbp_est[ev],
    logit_v2[ev], logit_v2[ev] - ref_logit_v2[ev],
])
variantes["K2: referencia + CNN V2 + multitarea calibrada"] = validacion_cruzada_por_sujeto(X_k2)

print("\n" + "=" * 78)
print("RESULTADOS EN VALIDATION (segmentos sin el de calibración)")
print("=" * 78)

filas = []
for nombre, score in variantes.items():
    auc, lo, hi, pr = con_ic(score)
    filas.append({"variante": nombre, "auc": auc, "ic_inf": lo, "ic_sup": hi, "pr_auc": pr})
    print(f"{nombre:<52} AUC = {auc:.4f}  [IC95 {lo:.4f} – {hi:.4f}]  PR-AUC = {pr:.4f}")

r = variantes["R: solo presión de referencia (sin ECG)"]
k2 = variantes["K2: referencia + CNN V2 + multitarea calibrada"]
b = variantes["B: CNN V2 sola (sin referencia)"]

d_k2_r = ic_diferencia(k2, r)
d_k2_b = ic_diferencia(k2, b)

auc_r, auc_k2, auc_b = roc_auc_score(y_ev, r), roc_auc_score(y_ev, k2), roc_auc_score(y_ev, b)

print("\nAporte del ECG cuando ya se conoce la referencia:")
print(f"  K2 − R = {auc_k2 - auc_r:+.4f}  [IC95 {d_k2_r[0]:+.4f} – {d_k2_r[1]:+.4f}]")
print("Mejora de la calibración frente a la situación actual:")
print(f"  K2 − B = {auc_k2 - auc_b:+.4f}  [IC95 {d_k2_b[0]:+.4f} – {d_k2_b[1]:+.4f}]")

# MAE de la estimación calibrada (referencia para la discusión).
mae_sbp = float(np.mean(np.abs(sbp_est[ev] - sbp[ev])))
mae_dbp = float(np.mean(np.abs(dbp_est[ev] - dbp[ev])))
print(f"\nError de la presión estimada calibrada: MAE SBP = {mae_sbp:.2f} mmHg | MAE DBP = {mae_dbp:.2f} mmHg")

pd.DataFrame(filas).to_csv(RUTA_CSV, index=False)

with open(RUTA_JSON, "w", encoding="utf-8") as f:
    json.dump(
        {
            "protocolo": "primer segmento de cada sujeto = calibración; se evalúan los demás",
            "sujetos": int(n_suj),
            "segmentos_evaluados": int(ev.sum()),
            "variantes": filas,
            "k2_menos_r": {"dif": auc_k2 - auc_r, "ic95": d_k2_r.tolist()},
            "k2_menos_b": {"dif": auc_k2 - auc_b, "ic95": d_k2_b.tolist()},
            "mae_sbp_calibrado": mae_sbp,
            "mae_dbp_calibrado": mae_dbp,
            "control_auc_multitarea": float(auc_mt_check),
        },
        f,
        ensure_ascii=False,
        indent=2,
    )

print("\n" + "=" * 78)
print("PUNTO 34 FINALIZADO")
print("=" * 78)
print(f"Tabla: {RUTA_CSV}")
print(
    "\nIMPORTANTE:"
    "\n- Experimento de factibilidad: ninguna CNN fue reentrenada."
    "\n- Combinaciones evaluadas con validación cruzada por sujeto."
    "\n- CalFree NO fue utilizado."
)
