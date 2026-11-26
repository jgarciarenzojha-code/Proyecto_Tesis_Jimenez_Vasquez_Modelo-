import os
import h5py
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

# =========================================================
# CONFIGURACIÓN
# =========================================================

RUTA_MAT = r"E:\Tesis_PulseDB\data_train\VitalDB_Train_Subset.mat"

CARPETA_PROCESADO = r"E:\Tesis_PulseDB\procesado"
CARPETA_MODELOS = r"E:\Tesis_PulseDB\modelos"
CARPETA_RESULTADOS = r"E:\Tesis_PulseDB\resultados"

RUTA_INDICES_TRAIN = os.path.join(
    CARPETA_PROCESADO,
    "indices_train.npy"
)

RUTA_INDICES_VAL = os.path.join(
    CARPETA_PROCESADO,
    "indices_val.npy"
)

RUTA_LABELS_TRAIN = os.path.join(
    CARPETA_PROCESADO,
    "etiquetas_train_140_90.npy"
)

RUTA_LABELS_VAL = os.path.join(
    CARPETA_PROCESADO,
    "etiquetas_val_140_90.npy"
)

RUTA_MODELO = os.path.join(
    CARPETA_MODELOS,
    "mejor_modelo_v2_regularizada_140_90.pt"
)

RUTA_HISTORIAL = os.path.join(
    CARPETA_RESULTADOS,
    "1_cnn_residual", "historial_cnn_v2_regularizada_140_90.csv"
)

BATCH_SIZE = 128
MAX_EPOCHS = 15
PATIENCE = 4

LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001

torch.manual_seed(42)
np.random.seed(42)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 70)
print("PUNTO 23 - CNN V2 REGULARIZADA - CRITERIO 140/90")
print("=" * 70)

print(f"Dispositivo: {device}")

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

# =========================================================
# CARGAR ÍNDICES Y ETIQUETAS
# =========================================================

indices_train = np.load(
    RUTA_INDICES_TRAIN
)

indices_val = np.load(
    RUTA_INDICES_VAL
)

labels_train = np.load(
    RUTA_LABELS_TRAIN
).astype(np.float32)

labels_val = np.load(
    RUTA_LABELS_VAL
).astype(np.float32)

print("\nDatos:")
print(
    f"Train:      {len(indices_train):,}"
)
print(
    f"Validation: {len(indices_val):,}"
)

positivos_train = int(
    np.sum(labels_train == 1)
)

negativos_train = int(
    np.sum(labels_train == 0)
)

positivos_val = int(
    np.sum(labels_val == 1)
)

negativos_val = int(
    np.sum(labels_val == 0)
)

print("\nDistribución 140/90:")

print(
    f"Train -> Negativos: {negativos_train:,} | "
    f"Positivos: {positivos_train:,}"
)

print(
    f"Validation -> Negativos: {negativos_val:,} | "
    f"Positivos: {positivos_val:,}"
)

# =========================================================
# POS_WEIGHT
# =========================================================

pos_weight_value = (
    negativos_train /
    positivos_train
)

print(
    f"\npos_weight calculado SOLO con Train: "
    f"{pos_weight_value:.4f}"
)

pos_weight = torch.tensor(
    [pos_weight_value],
    dtype=torch.float32,
    device=device
)

# =========================================================
# DATASET
# =========================================================

class ECGDataset(Dataset):

    def __init__(
        self,
        ruta_mat,
        indices,
        labels
    ):
        self.ruta_mat = ruta_mat
        self.indices = indices
        self.labels = labels

        self.archivo = None
        self.signals = None

    def __len__(self):
        return len(self.indices)

    def _abrir_archivo(self):

        if self.archivo is None:

            self.archivo = h5py.File(
                self.ruta_mat,
                "r"
            )

            self.signals = (
                self.archivo["Subset"]["Signals"]
            )

    def __getitem__(self, idx):

        self._abrir_archivo()

        indice_real = int(
            self.indices[idx]
        )

        # Canal 0 = ECG
        ecg = np.asarray(
            self.signals[
                :,
                0,
                indice_real
            ],
            dtype=np.float32
        )

        # ---------------------------------------------
        # Z-SCORE POR SEGMENTO
        # ---------------------------------------------

        media = ecg.mean()
        desviacion = ecg.std()

        if desviacion > 1e-8:
            ecg = (
                (ecg - media)
                / desviacion
            )
        else:
            ecg = ecg - media

        ecg = np.expand_dims(
            ecg,
            axis=0
        )

        label = np.float32(
            self.labels[idx]
        )

        return (
            torch.from_numpy(ecg),
            torch.tensor(
                label,
                dtype=torch.float32
            )
        )

# =========================================================
# DATALOADERS
# =========================================================

dataset_train = ECGDataset(
    RUTA_MAT,
    indices_train,
    labels_train
)

dataset_val = ECGDataset(
    RUTA_MAT,
    indices_val,
    labels_val
)

loader_train = DataLoader(
    dataset_train,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

loader_val = DataLoader(
    dataset_val,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

print(
    f"\nBatches Train: "
    f"{len(loader_train):,}"
)

print(
    f"Batches Validation: "
    f"{len(loader_val):,}"
)

# =========================================================
# BLOQUE RESIDUAL
# =========================================================

class ResidualBlock1D(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1
    ):
        super().__init__()

        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )

        self.bn1 = nn.BatchNorm1d(
            out_channels
        )

        self.relu = nn.ReLU()

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False
        )

        self.bn2 = nn.BatchNorm1d(
            out_channels
        )

        if (
            stride != 1
            or in_channels != out_channels
        ):
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm1d(
                    out_channels
                )
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):

        identidad = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identidad
        out = self.relu(out)

        return out

# =========================================================
# CNN V2
# =========================================================

class CNNV2(nn.Module):

    def __init__(self):
        super().__init__()

        self.entrada = nn.Sequential(
            nn.Conv1d(
                1,
                32,
                kernel_size=15,
                stride=2,
                padding=7,
                bias=False
            ),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        self.block1 = ResidualBlock1D(
            32,
            32,
            kernel_size=7,
            stride=1
        )

        self.block2 = ResidualBlock1D(
            32,
            64,
            kernel_size=7,
            stride=2
        )

        self.block3 = ResidualBlock1D(
            64,
            128,
            kernel_size=5,
            stride=2
        )

        self.block4 = ResidualBlock1D(
            128,
            256,
            kernel_size=3,
            stride=2
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.clasificador = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(
                256,
                64
            ),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(
                64,
                1
            )
        )

    def forward(self, x):

        x = self.entrada(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.pool(x)

        x = self.clasificador(x)

        return x.squeeze(1)

# =========================================================
# MODELO
# =========================================================

modelo = CNNV2().to(device)

parametros = sum(
    p.numel()
    for p in modelo.parameters()
)

print(
    f"\nParámetros del modelo: "
    f"{parametros:,}"
)

criterion = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight
)

optimizer = torch.optim.AdamW(
    modelo.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

scheduler = (
    torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-5
    )
)

# =========================================================
# VALIDACIÓN
# =========================================================

def validar():

    modelo.eval()

    losses = []

    todas_probs = []
    todas_labels = []

    with torch.no_grad():

        for x, y in loader_val:

            x = x.to(
                device,
                non_blocking=True
            )

            y = y.to(
                device,
                non_blocking=True
            )

            logits = modelo(x)

            loss = criterion(
                logits,
                y
            )

            losses.append(
                loss.item()
            )

            probs = torch.sigmoid(
                logits
            )

            todas_probs.extend(
                probs.cpu().numpy()
            )

            todas_labels.extend(
                y.cpu().numpy()
            )

    todas_probs = np.asarray(
        todas_probs
    )

    todas_labels = np.asarray(
        todas_labels
    ).astype(int)

    predicciones = (
        todas_probs >= 0.50
    ).astype(int)

    acc = accuracy_score(
        todas_labels,
        predicciones
    )

    precision = precision_score(
        todas_labels,
        predicciones,
        zero_division=0
    )

    sensibilidad = recall_score(
        todas_labels,
        predicciones,
        zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        todas_labels,
        predicciones
    ).ravel()

    especificidad = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    f1 = f1_score(
        todas_labels,
        predicciones,
        zero_division=0
    )

    auc = roc_auc_score(
        todas_labels,
        todas_probs
    )

    pr_auc = average_precision_score(
        todas_labels,
        todas_probs
    )

    return {
        "loss": np.mean(losses),
        "accuracy": acc,
        "precision": precision,
        "sensibilidad": sensibilidad,
        "especificidad": especificidad,
        "f1": f1,
        "auc": auc,
        "pr_auc": pr_auc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    }

# =========================================================
# ENTRENAMIENTO
# =========================================================

mejor_auc = -np.inf
epocas_sin_mejora = 0

historial = []

print("\n" + "=" * 70)
print("INICIANDO ENTRENAMIENTO")
print("=" * 70)

for epoch in range(
    1,
    MAX_EPOCHS + 1
):

    modelo.train()

    train_losses = []

    for batch_idx, (x, y) in enumerate(
        loader_train,
        start=1
    ):

        x = x.to(
            device,
            non_blocking=True
        )

        y = y.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = modelo(x)

        loss = criterion(
            logits,
            y
        )

        loss.backward()

        optimizer.step()

        train_losses.append(
            loss.item()
        )

        if batch_idx % 500 == 0:

            print(
                f"Epoch {epoch:02d} | "
                f"Batch "
                f"{batch_idx:,}/"
                f"{len(loader_train):,} | "
                f"Loss actual: "
                f"{loss.item():.4f}"
            )

    train_loss = np.mean(
        train_losses
    )

    metricas = validar()

    auc_actual = metricas["auc"]

    scheduler.step(
        auc_actual
    )

    lr_actual = optimizer.param_groups[
        0
    ]["lr"]

    historial.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": metricas["loss"],
        "accuracy": metricas["accuracy"],
        "precision": metricas["precision"],
        "sensibilidad":
            metricas["sensibilidad"],
        "especificidad":
            metricas["especificidad"],
        "f1": metricas["f1"],
        "roc_auc": metricas["auc"],
        "pr_auc": metricas["pr_auc"],
        "tn": metricas["tn"],
        "fp": metricas["fp"],
        "fn": metricas["fn"],
        "tp": metricas["tp"],
        "learning_rate": lr_actual
    })

    print("\n" + "-" * 70)

    print(
        f"Epoch {epoch:02d}"
    )

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Val Loss:   "
        f"{metricas['loss']:.4f}"
    )

    print(
        f"Accuracy:   "
        f"{metricas['accuracy']:.4f}"
    )

    print(
        f"Precision:  "
        f"{metricas['precision']:.4f}"
    )

    print(
        f"Sensibilidad: "
        f"{metricas['sensibilidad']:.4f}"
    )

    print(
        f"Especificidad: "
        f"{metricas['especificidad']:.4f}"
    )

    print(
        f"F1:         "
        f"{metricas['f1']:.4f}"
    )

    print(
        f"ROC-AUC:    "
        f"{metricas['auc']:.4f}"
    )

    print(
        f"PR-AUC:     "
        f"{metricas['pr_auc']:.4f}"
    )

    print(
        f"TN={metricas['tn']:,} | "
        f"FP={metricas['fp']:,} | "
        f"FN={metricas['fn']:,} | "
        f"TP={metricas['tp']:,}"
    )

    print(
        f"Learning rate: {lr_actual}"
    )

    # =====================================================
    # GUARDAR MEJOR MODELO
    # =====================================================

    if auc_actual > mejor_auc:

        mejor_auc = auc_actual
        epocas_sin_mejora = 0

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict":
                    modelo.state_dict(),
                "optimizer_state_dict":
                    optimizer.state_dict(),
                "roc_auc":
                    metricas["auc"],
                "pr_auc":
                    metricas["pr_auc"],
                "criterio":
                    "SBP >= 140 OR DBP >= 90",
                "pos_weight":
                    pos_weight_value,
                "weight_decay":
                    WEIGHT_DECAY,
                "zscore":
                    True
            },
            RUTA_MODELO
        )

        print(
            ">>> NUEVO MEJOR MODELO GUARDADO"
        )

    else:

        epocas_sin_mejora += 1

        print(
            f">>> Sin mejora: "
            f"{epocas_sin_mejora}/"
            f"{PATIENCE}"
        )

    pd.DataFrame(
        historial
    ).to_csv(
        RUTA_HISTORIAL,
        index=False
    )

    if epocas_sin_mejora >= PATIENCE:

        print(
            "\nEARLY STOPPING ACTIVADO."
        )

        break

# =========================================================
# FINAL
# =========================================================

print("\n" + "=" * 70)
print("ENTRENAMIENTO FINALIZADO")
print("=" * 70)

print(
    f"Mejor ROC-AUC Validation: "
    f"{mejor_auc:.4f}"
)

print(
    "\nMejor modelo:"
)

print(
    RUTA_MODELO
)

print(
    "\nHistorial:"
)

print(
    RUTA_HISTORIAL
)

print(
    "\nCriterio utilizado:"
)

print(
    "SBP >= 140 OR DBP >= 90"
)

print(
    "\nCalFree NO fue utilizado."
)

print(
    "\nPUNTO 23 FINALIZADO."
)