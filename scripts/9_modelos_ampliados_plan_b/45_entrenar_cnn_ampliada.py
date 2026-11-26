import os
import sys
import time
import json
import argparse
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score

# ============================================================
# PUNTO 45 - REENTRENAR UNA CNN CON LOS SUJETOS NUEVOS (PLAN B)
#
# Uso:
#   python 45_entrenar_cnn_ampliada.py --modelo v2|v3|multitarea [--datos ampliado|original]
#
# - "ampliado": entrenamiento original (1034 sujetos) + 200 sujetos
#   nuevos del punto 44.  "original": solo los 1034 (control, para
#   medir cuánto varía el resultado solo por volver a entrenar).
# - Validación: los MISMOS 259 sujetos de siempre. CalFree NO se usa.
# - Misma arquitectura e hiperparámetros que los scripts 23, 30 y 31:
#   batch 128, máx. 15 épocas, paciencia 4, AdamW lr 1e-3 wd 1e-4,
#   ReduceLROnPlateau(max, 0.5, 2), BCE con pos_weight = neg/pos de
#   entrenamiento, z-score por segmento, semilla 42; V3 y multitarea con
#   clip de gradiente 5.0; multitarea con λ_PAS 0.15 y λ_PAD 0.10.
# - Las arquitecturas se importan de la app web (las mismas clases que
#   cargan los pesos del modelo actual).
# - No modifica los modelos actuales: guarda archivos nuevos (*_45.pt).
# ============================================================

RAIZ = r"E:\Tesis_PulseDB"
sys.path.insert(0, os.path.join(RAIZ, "app_web"))
from backend.modelo.arquitecturas import CNNV2, CNNV3MultiEscala, CNNMultiTarea  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--modelo", required=True, choices=["v2", "v3", "multitarea"])
parser.add_argument("--datos", default="ampliado", choices=["ampliado", "original"])
args = parser.parse_args()

PROCESADO = os.path.join(RAIZ, "procesado")
RUTA_MAT = os.path.join(RAIZ, "data_train", "VitalDB_Train_Subset.mat")
RUTA_NUEVOS = os.path.join(PROCESADO, "sujetos_nuevos_44.npz")
NOMBRE = f"{args.modelo}_{args.datos}_45"
RUTA_MODELO = os.path.join(RAIZ, "modelos", "experimento_200_pacientes", f"mejor_modelo_{NOMBRE}.pt")
RUTA_HIST = os.path.join(RAIZ, "resultados", "5_experimentos", "200_pacientes", f"historial_{NOMBRE}.csv")
RUTA_PRED = os.path.join(RAIZ, "resultados", "5_experimentos", "200_pacientes", f"logits_val_{NOMBRE}.npy")
RUTA_JSON = os.path.join(RAIZ, "resultados", "5_experimentos", "200_pacientes", f"resultado_{NOMBRE}.json")

SEED = 42
BATCH_SIZE = 128
MAX_EPOCHS = 15
PATIENCE = 4
LR = 1e-3
WEIGHT_DECAY = 1e-4
CLIP = None if args.modelo == "v2" else 5.0
MIN_LR = 1e-5 if args.modelo == "v2" else 0.0
LAMBDA_SBP, LAMBDA_DBP = 0.15, 0.10

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 78)
print(f"PUNTO 45 - CNN {args.modelo.upper()} | datos: {args.datos} | {device}")
print("=" * 78)


def zscore(bloque):
    std = bloque.std(axis=1, keepdims=True)
    std[std < 1e-6] = 1.0
    return ((bloque - bloque.mean(axis=1, keepdims=True)) / std).astype(np.float32)


# ------------------------------------------------------------
# Datos (todo en memoria, z-score por segmento)
# ------------------------------------------------------------
idx_train = np.load(os.path.join(PROCESADO, "indices_train.npy"))
idx_val = np.load(os.path.join(PROCESADO, "indices_val.npy"))
y_train = np.load(os.path.join(PROCESADO, "etiquetas_train_140_90.npy")).astype(np.float32)
y_val = np.load(os.path.join(PROCESADO, "etiquetas_val_140_90.npy")).astype(np.float32)

inicio = time.time()
with h5py.File(RUTA_MAT, "r") as f:
    s = f["Subset"]
    sbp_all = np.asarray(s["SBP"]).reshape(-1).astype(np.float32)
    dbp_all = np.asarray(s["DBP"]).reshape(-1).astype(np.float32)
    signals = s["Signals"]
    total = signals.shape[2]
    ecg_all = np.empty((total, signals.shape[0]), dtype=np.float32)
    for a in range(0, total, 20000):
        b = min(a + 20000, total)
        ecg_all[a:b] = zscore(np.asarray(signals[:, 0, a:b], dtype=np.float32).T)

# Control: las etiquetas guardadas deben coincidir con PAS/PAD del subconjunto.
assert np.array_equal(y_train, ((sbp_all[idx_train] >= 140) | (dbp_all[idx_train] >= 90)).astype(np.float32))
assert np.array_equal(y_val, ((sbp_all[idx_val] >= 140) | (dbp_all[idx_val] >= 90)).astype(np.float32))

x_tr, s_tr, d_tr = ecg_all[idx_train], sbp_all[idx_train], dbp_all[idx_train]
x_va = ecg_all[idx_val]
del ecg_all

if args.datos == "ampliado":
    nuevos = np.load(RUTA_NUEVOS, allow_pickle=True)
    x_tr = np.concatenate([x_tr, zscore(nuevos["ecg"])])
    y_train = np.concatenate([y_train, nuevos["etiqueta"].astype(np.float32)])
    s_tr = np.concatenate([s_tr, nuevos["sbp"]])
    d_tr = np.concatenate([d_tr, nuevos["dbp"]])
print(f"Datos en memoria en {time.time() - inicio:.0f} s | train {len(y_train):,} "
      f"(positivos {y_train.mean():.3f}) | validación {len(y_val):,}")

pos_weight_value = float((y_train == 0).sum() / (y_train == 1).sum())
SBP_MEAN, SBP_STD = float(s_tr.mean()), float(s_tr.std())
DBP_MEAN, DBP_STD = float(d_tr.mean()), float(d_tr.std())
s_tr_n = ((s_tr - SBP_MEAN) / SBP_STD).astype(np.float32)
d_tr_n = ((d_tr - DBP_MEAN) / DBP_STD).astype(np.float32)
print(f"pos_weight (solo train): {pos_weight_value:.4f}")

# ------------------------------------------------------------
# Modelo
# ------------------------------------------------------------


class MultiTareaEntrenamiento(CNNMultiTarea):
    """Misma red de la app; en entrenamiento devuelve las tres cabezas."""

    def forward(self, x):
        for capa in (self.entry, self.block1, self.block2, self.block3, self.block4, self.pool):
            x = capa(x)
        h = self.shared(x)
        return self.head_class(h).squeeze(1), self.head_sbp(h).squeeze(1), self.head_dbp(h).squeeze(1)


modelo = {"v2": CNNV2, "v3": CNNV3MultiEscala, "multitarea": MultiTareaEntrenamiento}[args.modelo]().to(device)
print(f"Parámetros: {sum(p.numel() for p in modelo.parameters()):,}")

criterio = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value], device=device))
reg = nn.SmoothL1Loss()
optim = torch.optim.AdamW(modelo.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="max", factor=0.5, patience=2, min_lr=MIN_LR)


def logits_de(salida):
    return salida[0] if isinstance(salida, tuple) else salida


def validar():
    modelo.eval()
    out = []
    with torch.no_grad():
        for a in range(0, len(x_va), 512):
            xb = torch.from_numpy(x_va[a:a + 512]).unsqueeze(1).to(device)
            out.append(logits_de(modelo(xb)).float().cpu().numpy())
    z = np.concatenate(out)
    return z, roc_auc_score(y_val, z), average_precision_score(y_val, z)


# ------------------------------------------------------------
# Entrenamiento
# ------------------------------------------------------------
gen = np.random.default_rng(SEED)
mejor_auc, sin_mejora, historial = -np.inf, 0, []
for epoca in range(1, MAX_EPOCHS + 1):
    t0 = time.time()
    modelo.train()
    orden = gen.permutation(len(y_train))
    perdidas = []
    for a in range(0, len(orden), BATCH_SIZE):
        k = np.sort(orden[a:a + BATCH_SIZE])
        xb = torch.from_numpy(x_tr[k]).unsqueeze(1).to(device, non_blocking=True)
        yb = torch.from_numpy(y_train[k]).to(device)
        optim.zero_grad(set_to_none=True)
        salida = modelo(xb)
        perdida = criterio(logits_de(salida), yb)
        if args.modelo == "multitarea":
            perdida = (perdida
                       + LAMBDA_SBP * reg(salida[1], torch.from_numpy(s_tr_n[k]).to(device))
                       + LAMBDA_DBP * reg(salida[2], torch.from_numpy(d_tr_n[k]).to(device)))
        perdida.backward()
        if CLIP:
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=CLIP)
        optim.step()
        perdidas.append(perdida.item())

    z, auc, pr = validar()
    sched.step(auc)
    lr = optim.param_groups[0]["lr"]
    historial.append({"epoch": epoca, "train_loss": float(np.mean(perdidas)), "roc_auc": auc, "pr_auc": pr,
                      "learning_rate": lr, "minutos": (time.time() - t0) / 60})
    marca = ""
    if auc > mejor_auc:
        mejor_auc, sin_mejora, marca = auc, 0, "  <- mejor"
        torch.save({"epoch": epoca, "model_state_dict": modelo.state_dict(), "roc_auc": auc, "pr_auc": pr,
                    "pos_weight": pos_weight_value, "datos": args.datos, "zscore": True,
                    "sbp_mean": SBP_MEAN, "sbp_std": SBP_STD, "dbp_mean": DBP_MEAN, "dbp_std": DBP_STD},
                   RUTA_MODELO)
        np.save(RUTA_PRED, z)
    else:
        sin_mejora += 1
    print(f"Época {epoca:2d}/{MAX_EPOCHS} ({(time.time() - t0) / 60:.1f} min) | pérdida {np.mean(perdidas):.4f} | "
          f"AUC val {auc:.4f} | PR-AUC {pr:.4f} | lr {lr:.6f}{marca}", flush=True)
    pd.DataFrame(historial).to_csv(RUTA_HIST, index=False)
    if sin_mejora >= PATIENCE:
        print("Early stopping.")
        break

ref = {"v2": 0.7227, "v3": 0.7221, "multitarea": 0.7263}
json.dump({"modelo": args.modelo, "datos": args.datos, "mejor_auc_val": mejor_auc,
           "mejor_epoca": int(pd.DataFrame(historial).roc_auc.idxmax() + 1), "train_segmentos": int(len(y_train)),
           "pos_weight": pos_weight_value}, open(RUTA_JSON, "w", encoding="utf-8"), indent=2)
print(f"\nMejor AUC en validación: {mejor_auc:.4f} (modelo actual de la app: ~{ref[args.modelo]})")
print(f"Guardado: {RUTA_MODELO}\nCalFree NO fue utilizado. Los modelos actuales no se modificaron.")
