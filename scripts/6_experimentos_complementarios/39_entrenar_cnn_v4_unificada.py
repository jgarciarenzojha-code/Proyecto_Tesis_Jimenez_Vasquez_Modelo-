import os
import sys
import json
import time
import copy
import h5py
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, confusion_matrix

# ============================================================
# PUNTO 39 - CNN V4 UNIFICADA (UNA SOLA RED)
#
# Una única CNN 1D que integra, dentro de la misma red y entrenada
# de extremo a extremo, las ideas que funcionaron en las versiones
# anteriores:
#   - módulo multiescala (filtros 3, 7, 15 y 31) de la V3;
#   - bloques residuales de la V2;
#   - cabezas auxiliares SBP/DBP de la multitarea (solo en el
#     entrenamiento; la aplicación usa únicamente la clasificación);
#   - promedio móvil exponencial de los pesos (EMA), que suaviza
#     las oscilaciones de validación entre épocas sin crear otra red.
#
# Mismo protocolo que los puntos 23, 30 y 31: mismos índices por
# sujeto, mismas etiquetas 140/90, z-score por segmento,
# BCEWithLogitsLoss + pos_weight, selección SOLO con Validation.
# CalFree NO se usa.
#
# Regla de decisión (fijada antes de entrenar): la V4 reemplaza al
# ensemble del punto 35 solo si su ROC-AUC en Validation es >= 0.7388.
# Si no, se descarta y el modelo final sigue siendo el ensemble.
# ============================================================

RAIZ = r"E:\Tesis_PulseDB"
RUTA_MAT = os.path.join(RAIZ, "data_train", "VitalDB_Train_Subset.mat")
PROCESADO = os.path.join(RAIZ, "procesado")
MODELOS = os.path.join(RAIZ, "modelos")
RESULTADOS = os.path.join(RAIZ, "resultados")

RUTA_MODELO = os.path.join(MODELOS, "experimento_v4_unificada", "mejor_modelo_v4_unificada_140_90.pt")
RUTA_HISTORIAL = os.path.join(RESULTADOS, "5_experimentos", "v4_unificada", "historial_cnn_v4_unificada_140_90.csv")
RUTA_NPZ = os.path.join(RESULTADOS, "5_experimentos", "v4_unificada", "predicciones_val_v4_39.npz")
RUTA_JSON = os.path.join(RESULTADOS, "5_experimentos", "v4_unificada", "cnn_v4_unificada_39.json")

SEED = 42
BATCH_SIZE = 128
MAX_EPOCHS = 15
PATIENCE = 4
LR = 0.001
WEIGHT_DECAY = 1e-4
LAMBDA_SBP = 0.15                   # igual que el punto 31
LAMBDA_DBP = 0.10
EMA_DECAY = 0.999
RUIDO_STD = 0.03                    # aumento de datos: ruido gaussiano leve
PROB_RUIDO = 0.5
BLOQUE_LECTURA = 20000
N_BOOTSTRAP = 1000

# "--solo-evaluar": no entrena; compara el modelo ya guardado (mismo resultado).
SOLO_EVALUAR = "--solo-evaluar" in sys.argv

AUC_ENSEMBLE = 0.7388               # punto 35 (Validation)
AUC_V2 = 0.7227

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 78)
print("PUNTO 39 - CNN V4 UNIFICADA (residual + multiescala + multitarea + EMA)")
print("=" * 78)
print(f"Dispositivo: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

if os.path.exists(RUTA_MODELO) and not SOLO_EVALUAR:
    print(f"AVISO: {RUTA_MODELO} ya existe y será reemplazado (archivo propio del punto 39).")

# ============================================================
# DATOS (mismos archivos que los puntos 23, 30 y 31)
# ============================================================

indices_train = np.load(os.path.join(PROCESADO, "indices_train.npy")).astype(np.int64)
indices_val = np.load(os.path.join(PROCESADO, "indices_val.npy")).astype(np.int64)
y_train = np.load(os.path.join(PROCESADO, "etiquetas_train_140_90.npy")).astype(np.float32)
y_val = np.load(os.path.join(PROCESADO, "etiquetas_val_140_90.npy")).astype(np.float32)
sujeto_val = np.load(os.path.join(PROCESADO, "demografia_32.npz"), allow_pickle=True)["sujeto"][indices_val]

pos_weight_value = float((y_train == 0).sum() / (y_train == 1).sum())

inicio = time.time()
with h5py.File(RUTA_MAT, "r") as f:
    s = f["Subset"]
    sbp = np.asarray(s["SBP"]).reshape(-1).astype(np.float32)
    dbp = np.asarray(s["DBP"]).reshape(-1).astype(np.float32)

    # Todo el canal ECG en memoria (≈ 2.3 GB en float32), con z-score por
    # segmento ya aplicado: el mismo preprocesamiento que la aplicación web.
    signals = s["Signals"]
    total = signals.shape[2]
    ecg = np.empty((total, signals.shape[0]), dtype=np.float32)
    for a in range(0, total, BLOQUE_LECTURA):
        b = min(a + BLOQUE_LECTURA, total)
        bloque = np.asarray(signals[:, 0, a:b], dtype=np.float32).T
        std = bloque.std(axis=1, keepdims=True)
        std[std < 1e-6] = 1.0
        ecg[a:b] = (bloque - bloque.mean(axis=1, keepdims=True)) / std

print(f"ECG cargado y normalizado: {ecg.shape} en {time.time() - inicio:.0f} s")

# Objetivos auxiliares normalizados SOLO con Train.
SBP_MEAN, SBP_STD = float(sbp[indices_train].mean()), float(sbp[indices_train].std())
DBP_MEAN, DBP_STD = float(dbp[indices_train].mean()), float(dbp[indices_train].std())

print(f"Train: {len(y_train):,} | Validation: {len(y_val):,} | pos_weight: {pos_weight_value:.4f}")


class ECGDataset(Dataset):

    def __init__(self, indices, etiquetas, aumentar):
        self.indices = indices
        self.etiquetas = etiquetas
        self.sbp = (sbp[indices] - SBP_MEAN) / SBP_STD
        self.dbp = (dbp[indices] - DBP_MEAN) / DBP_STD
        self.aumentar = aumentar

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x = ecg[self.indices[i]]
        if self.aumentar and np.random.rand() < PROB_RUIDO:
            x = x + np.random.normal(0, RUIDO_STD, x.shape).astype(np.float32)
        return (
            torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0),
            torch.tensor(self.etiquetas[i]),
            torch.tensor(self.sbp[i]),
            torch.tensor(self.dbp[i]),
        )


generador = torch.Generator().manual_seed(SEED)
loader_train = DataLoader(ECGDataset(indices_train, y_train, True), batch_size=BATCH_SIZE,
                          shuffle=True, generator=generador, pin_memory=True)
loader_val = DataLoader(ECGDataset(indices_val, y_val, False), batch_size=512, shuffle=False, pin_memory=True)

# ============================================================
# ARQUITECTURA: UNA SOLA RED
# ============================================================


class BloqueResidual(nn.Module):

    def __init__(self, cin, cout, k, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(cin, cout, k, stride=stride, padding=k // 2, bias=False)
        self.bn1 = nn.BatchNorm1d(cout)
        self.conv2 = nn.Conv1d(cout, cout, k, padding=k // 2, bias=False)
        self.bn2 = nn.BatchNorm1d(cout)
        self.relu = nn.ReLU(inplace=True)
        self.atajo = (
            nn.Sequential(nn.Conv1d(cin, cout, 1, stride=stride, bias=False), nn.BatchNorm1d(cout))
            if stride != 1 or cin != cout else nn.Identity()
        )

    def forward(self, x):
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return self.relu(y + self.atajo(x))


class BloqueMultiEscala(nn.Module):
    """Cuatro convoluciones en paralelo (3, 7, 15 y 31 muestras) concatenadas."""

    def __init__(self, cin, c=32):
        super().__init__()
        self.ramas = nn.ModuleList(
            nn.Sequential(nn.Conv1d(cin, c, k, padding=k // 2, bias=False), nn.BatchNorm1d(c), nn.ReLU(inplace=True))
            for k in (3, 7, 15, 31)
        )

    def forward(self, x):
        return torch.cat([r(x) for r in self.ramas], dim=1)


class CNNV4Unificada(nn.Module):
    """
    ECG [1 x 1250] -> Conv inicial -> multiescala -> 3 bloques residuales
    -> pooling global -> capa compartida -> clasificación (+ SBP/DBP auxiliares).
    """

    def __init__(self):
        super().__init__()
        self.entrada = nn.Sequential(
            nn.Conv1d(1, 32, 15, stride=2, padding=7, bias=False), nn.BatchNorm1d(32), nn.ReLU(inplace=True)
        )                                                      # 32 x 625
        self.multiescala = BloqueMultiEscala(32, 32)           # 128 x 625
        self.res1 = BloqueResidual(128, 128, 7, stride=2)      # 128 x 313
        self.res2 = BloqueResidual(128, 256, 5, stride=2)      # 256 x 157
        self.res3 = BloqueResidual(256, 256, 3, stride=2)      # 256 x 79
        self.pool = nn.AdaptiveAvgPool1d(1)                    # 256 x 1
        self.compartida = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.40), nn.Linear(256, 64), nn.ReLU(inplace=True), nn.Dropout(0.30)
        )
        self.cabeza_clase = nn.Linear(64, 1)
        self.cabeza_sbp = nn.Linear(64, 1)                     # solo entrenamiento
        self.cabeza_dbp = nn.Linear(64, 1)                     # solo entrenamiento

    def forward(self, x, auxiliares=False):
        x = self.multiescala(self.entrada(x))
        x = self.pool(self.res3(self.res2(self.res1(x))))
        z = self.compartida(x)
        logit = self.cabeza_clase(z).squeeze(1)
        if not auxiliares:
            return logit
        return logit, self.cabeza_sbp(z).squeeze(1), self.cabeza_dbp(z).squeeze(1)


modelo = CNNV4Unificada().to(device)
ema = copy.deepcopy(modelo).eval()
for p in ema.parameters():
    p.requires_grad_(False)

print(f"Parámetros: {sum(p.numel() for p in modelo.parameters()):,} "
      f"(en inferencia: {sum(p.numel() for n, p in modelo.named_parameters() if 'sbp' not in n and 'dbp' not in n):,})")


@torch.no_grad()
def actualizar_ema():
    for pe, pm in zip(ema.parameters(), modelo.parameters()):
        pe.mul_(EMA_DECAY).add_(pm.detach(), alpha=1 - EMA_DECAY)
    for be, bm in zip(ema.buffers(), modelo.buffers()):
        be.copy_(bm)


@torch.no_grad()
def predecir(red, loader):
    red.eval()
    salida = []
    for x, *_ in loader:
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            salida.append(red(x.to(device, non_blocking=True)).float().cpu().numpy())
    return np.concatenate(salida)


perdida_clase = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value], device=device))
perdida_reg = nn.SmoothL1Loss()
optimizador = torch.optim.AdamW(modelo.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
planificador = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizador, mode="max", factor=0.5, patience=2)
escalador = torch.amp.GradScaler(enabled=device.type == "cuda")

# ============================================================
# ENTRENAMIENTO (selección por ROC-AUC de Validation del modelo EMA)
# ============================================================

mejor_auc, sin_mejora, historial = -np.inf, 0, []

for epoca in range(1, 1 if SOLO_EVALUAR else MAX_EPOCHS + 1):

    t0 = time.time()
    modelo.train()
    perdidas = []

    for x, y, ys, yd in loader_train:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        ys, yd = ys.to(device, non_blocking=True), yd.to(device, non_blocking=True)
        optimizador.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logit, ps, pd_ = modelo(x, auxiliares=True)
            perdida = (perdida_clase(logit.float(), y)
                       + LAMBDA_SBP * perdida_reg(ps.float(), ys)
                       + LAMBDA_DBP * perdida_reg(pd_.float(), yd))
        escalador.scale(perdida).backward()
        escalador.unscale_(optimizador)
        torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=5.0)
        escalador.step(optimizador)
        escalador.update()
        actualizar_ema()
        perdidas.append(perdida.item())

    auc_red = roc_auc_score(y_val, predecir(modelo, loader_val))
    logit_ema = predecir(ema, loader_val)
    auc_ema = roc_auc_score(y_val, logit_ema)
    pr_ema = average_precision_score(y_val, logit_ema)
    planificador.step(auc_ema)

    historial.append({
        "epoch": epoca, "train_loss": float(np.mean(perdidas)), "roc_auc_ema": auc_ema,
        "pr_auc_ema": pr_ema, "roc_auc_sin_ema": auc_red, "lr": optimizador.param_groups[0]["lr"],
    })
    pd.DataFrame(historial).to_csv(RUTA_HISTORIAL, index=False)

    print(f"Época {epoca:>2}/{MAX_EPOCHS} ({(time.time() - t0) / 60:.1f} min) | pérdida train {np.mean(perdidas):.4f} | "
          f"AUC val EMA {auc_ema:.4f} (sin EMA {auc_red:.4f}) | PR-AUC {pr_ema:.4f} | "
          f"lr {optimizador.param_groups[0]['lr']:.6f}", flush=True)

    if auc_ema > mejor_auc:
        mejor_auc, sin_mejora = auc_ema, 0
        torch.save({
            "epoch": epoca,
            "model_state_dict": ema.state_dict(),
            "val_roc_auc": float(auc_ema),
            "criterion": "SBP >= 140 OR DBP >= 90",
            "input": "ECG only",
            "preprocessing": "per-segment z-score (std < 1e-6 -> 1)",
            "architecture": "CNN V4 unificada: multiescala + residual + cabezas auxiliares SBP/DBP; pesos EMA",
            "ema_decay": EMA_DECAY,
            "sbp_mean": SBP_MEAN, "sbp_std": SBP_STD, "dbp_mean": DBP_MEAN, "dbp_std": DBP_STD,
            "seed": SEED,
        }, RUTA_MODELO)
        print("  >>> nuevo mejor modelo guardado")
    else:
        sin_mejora += 1
        print(f"  sin mejora: {sin_mejora}/{PATIENCE}")
        if sin_mejora >= PATIENCE:
            print("EARLY STOPPING")
            break

# ============================================================
# COMPARACIÓN FINAL CON EL ENSEMBLE (mismos segmentos, bootstrap por sujeto)
# ============================================================

ckpt = torch.load(RUTA_MODELO, map_location=device, weights_only=True)
ema.load_state_dict(ckpt["model_state_dict"])
logit_v4 = predecir(ema, loader_val)
prob_v4 = 1 / (1 + np.exp(-logit_v4.astype(np.float64)))

previas = np.load(os.path.join(RESULTADOS, "4_cnn_ensemble", "predicciones_val_ensemble_35.npz"))
prob_ens = previas["prob_ensemble"].astype(np.float64)

# Comprobación de alineación: la V2 guardada debe reproducir su AUC conocido.
auc_v2_guardada = roc_auc_score(y_val, previas["logit_v2"])
if abs(auc_v2_guardada - AUC_V2) > 0.002:
    raise RuntimeError(f"Predicciones del punto 35 desalineadas (AUC V2 = {auc_v2_guardada:.4f}).")

_, codigo = np.unique(sujeto_val, return_inverse=True)
pos_suj = [np.flatnonzero(codigo == k) for k in range(codigo.max() + 1)]
rng = np.random.default_rng(SEED)
remuestreos = [np.concatenate([pos_suj[k] for k in rng.integers(0, len(pos_suj), len(pos_suj))])
               for _ in range(N_BOOTSTRAP)]
remuestreos = [i for i in remuestreos if y_val[i].min() != y_val[i].max()]

auc_v4 = roc_auc_score(y_val, prob_v4)
auc_ens = roc_auc_score(y_val, prob_ens)
ic_v4 = np.percentile([roc_auc_score(y_val[i], prob_v4[i]) for i in remuestreos], [2.5, 97.5])
difs = [roc_auc_score(y_val[i], prob_v4[i]) - roc_auc_score(y_val[i], prob_ens[i]) for i in remuestreos]
ic_dif = np.percentile(difs, [2.5, 97.5])

# Umbrales (solo Validation), por si la V4 resulta elegida.
fpr, tpr, thr = roc_curve(y_val, prob_v4)
umbral_youden = float(thr[int(np.argmax(tpr - fpr))])
umbral_sens80 = float(thr[int(np.argmax(tpr >= 0.80))])


def metricas(umbral):
    tn, fp, fn, tp = confusion_matrix(y_val, (prob_v4 >= umbral).astype(int), labels=[0, 1]).ravel()
    return {"umbral": umbral, "sensibilidad": tp / (tp + fn), "especificidad": tn / (tn + fp),
            "precision": tp / (tp + fp) if tp + fp else 0.0, "exactitud": (tp + tn) / len(y_val)}


elegida = auc_v4 >= AUC_ENSEMBLE

print("\n" + "=" * 78)
print("RESULTADO EN VALIDATION (CalFree no se usó)")
print("=" * 78)
print(f"Mejor época: {ckpt['epoch']}")
print(f"CNN V4 unificada  AUC = {auc_v4:.4f}  [IC95 {ic_v4[0]:.4f} - {ic_v4[1]:.4f}]  "
      f"PR-AUC = {average_precision_score(y_val, prob_v4):.4f}")
print(f"Ensemble (pto 35) AUC = {auc_ens:.4f}")
print(f"Diferencia V4 - ensemble: {auc_v4 - auc_ens:+.4f}  [IC95 {ic_dif[0]:+.4f} - {ic_dif[1]:+.4f}]")
for nombre, u in [("Youden", umbral_youden), ("Sensibilidad >= 80 %", umbral_sens80)]:
    m = metricas(u)
    print(f"Umbral {nombre:<21} {u:.4f} | sens {m['sensibilidad']:.4f} | espec {m['especificidad']:.4f}")

print("\nDECISIÓN (regla fijada antes de entrenar):")
print("  La V4 IGUALA o SUPERA al ensemble: pasa a ser candidata a modelo final (una sola red)."
      if elegida else
      "  La V4 NO alcanza al ensemble: se descarta y el modelo final sigue siendo el ensemble.")

np.savez_compressed(RUTA_NPZ, logit_v4=logit_v4.astype(np.float32), prob_v4=prob_v4.astype(np.float32))
with open(RUTA_JSON, "w", encoding="utf-8") as fj:
    json.dump({
        "auc_val": auc_v4, "ic95": ic_v4.tolist(), "auc_ensemble": auc_ens,
        "diferencia_vs_ensemble": {"dif": auc_v4 - auc_ens, "ic95": ic_dif.tolist()},
        "mejor_epoca": int(ckpt["epoch"]), "umbral_youden": metricas(umbral_youden),
        "umbral_sensibilidad_80": metricas(umbral_sens80), "elegida": bool(elegida),
        "regla": f"se elige solo si AUC val >= {AUC_ENSEMBLE} (ensemble del punto 35)",
    }, fj, ensure_ascii=False, indent=2, default=float)

print(f"\nModelo: {RUTA_MODELO}\nHistorial: {RUTA_HISTORIAL}\nResumen: {RUTA_JSON}")
print("IMPORTANTE: solo se usó Validation; CalFree NO fue utilizado; ningún script anterior fue modificado.")
