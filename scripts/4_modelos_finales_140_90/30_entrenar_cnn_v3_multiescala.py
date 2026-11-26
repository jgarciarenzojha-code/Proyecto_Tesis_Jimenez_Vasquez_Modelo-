import os
import time
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

# ============================================================
# CONFIGURACIÓN
# ============================================================

RUTA_MAT = (
    r"E:\Tesis_PulseDB\data_train"
    r"\VitalDB_Train_Subset.mat"
)

CARPETA_PROCESADO = r"E:\Tesis_PulseDB\procesado"
CARPETA_MODELOS = r"E:\Tesis_PulseDB\modelos"
CARPETA_RESULTADOS = r"E:\Tesis_PulseDB\resultados"

os.makedirs(CARPETA_MODELOS, exist_ok=True)
os.makedirs(CARPETA_RESULTADOS, exist_ok=True)

RUTA_INDICES_TRAIN = os.path.join(
    CARPETA_PROCESADO,
    "indices_train.npy"
)

RUTA_INDICES_VAL = os.path.join(
    CARPETA_PROCESADO,
    "indices_val.npy"
)

RUTA_Y_TRAIN = os.path.join(
    CARPETA_PROCESADO,
    "etiquetas_train_140_90.npy"
)

RUTA_Y_VAL = os.path.join(
    CARPETA_PROCESADO,
    "etiquetas_val_140_90.npy"
)

RUTA_MODELO = os.path.join(
    CARPETA_MODELOS,
    "mejor_modelo_v3_multiescala_140_90.pt"
)

RUTA_HISTORIAL = os.path.join(
    CARPETA_RESULTADOS,
    "2_cnn_multiescala", "historial_cnn_v3_multiescala_140_90.csv"
)

SEED = 42
BATCH_SIZE = 128
MAX_EPOCHS = 15
PATIENCE = 4
LR = 0.001
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0

np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("=" * 78)
print("PUNTO 30 - CNN V3 MULTIESCALA")
print("=" * 78)

print(f"\nDispositivo: {device}")

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

# ============================================================
# CARGAR ÍNDICES Y ETIQUETAS
# ============================================================

indices_train = np.load(
    RUTA_INDICES_TRAIN
).astype(np.int64)

indices_val = np.load(
    RUTA_INDICES_VAL
).astype(np.int64)

y_train = np.load(
    RUTA_Y_TRAIN
).astype(np.float32)

y_val = np.load(
    RUTA_Y_VAL
).astype(np.float32)

assert len(indices_train) == len(y_train)
assert len(indices_val) == len(y_val)

neg_train = int((y_train == 0).sum())
pos_train = int((y_train == 1).sum())

pos_weight_value = (
    neg_train / pos_train
)

print("\nDATOS:")
print(
    f"Train:      {len(y_train):,} segmentos"
)
print(
    f"Validation: {len(y_val):,} segmentos"
)

print(
    f"Train negativos: {neg_train:,}"
)
print(
    f"Train positivos: {pos_train:,}"
)

print(
    f"pos_weight: {pos_weight_value:.4f}"
)

# ============================================================
# DATASET HDF5
# ============================================================

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

    def _abrir(self):

        if self.archivo is None:

            self.archivo = h5py.File(
                self.ruta_mat,
                "r"
            )

            self.signals = (
                self.archivo["Subset"]["Signals"]
            )

    def __len__(self):

        return len(self.indices)

    def __getitem__(self, idx):

        self._abrir()

        indice_real = int(
            self.indices[idx]
        )

        # Signals:
        # [1250, 3, segmentos]
        # Canal 0 = ECG
        ecg = np.asarray(
            self.signals[
                :,
                0,
                indice_real
            ],
            dtype=np.float32
        )

        # ----------------------------------------------
        # Z-SCORE POR SEGMENTO
        # ----------------------------------------------

        media = ecg.mean()
        std = ecg.std()

        if std < 1e-6:
            std = 1.0

        ecg = (
            (ecg - media) / std
        ).astype(np.float32)

        ecg = torch.from_numpy(
            ecg
        ).unsqueeze(0)

        label = torch.tensor(
            self.labels[idx],
            dtype=torch.float32
        )

        return ecg, label


dataset_train = ECGDataset(
    RUTA_MAT,
    indices_train,
    y_train
)

dataset_val = ECGDataset(
    RUTA_MAT,
    indices_val,
    y_val
)

loader_train = DataLoader(
    dataset_train,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

loader_val = DataLoader(
    dataset_val,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

# ============================================================
# BLOQUE RESIDUAL
# ============================================================

class ResidualBlock(nn.Module):

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

        self.relu = nn.ReLU(
            inplace=True
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

        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)

        return out

# ============================================================
# BLOQUE MULTIESCALA
# ============================================================

class MultiScaleBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        branch_channels=32
    ):

        super().__init__()

        # Diferentes ventanas temporales
        self.branch_3 = nn.Sequential(
            nn.Conv1d(
                in_channels,
                branch_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm1d(
                branch_channels
            ),
            nn.ReLU(inplace=True)
        )

        self.branch_7 = nn.Sequential(
            nn.Conv1d(
                in_channels,
                branch_channels,
                kernel_size=7,
                padding=3,
                bias=False
            ),
            nn.BatchNorm1d(
                branch_channels
            ),
            nn.ReLU(inplace=True)
        )

        self.branch_15 = nn.Sequential(
            nn.Conv1d(
                in_channels,
                branch_channels,
                kernel_size=15,
                padding=7,
                bias=False
            ),
            nn.BatchNorm1d(
                branch_channels
            ),
            nn.ReLU(inplace=True)
        )

        self.branch_31 = nn.Sequential(
            nn.Conv1d(
                in_channels,
                branch_channels,
                kernel_size=31,
                padding=15,
                bias=False
            ),
            nn.BatchNorm1d(
                branch_channels
            ),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        b1 = self.branch_3(x)
        b2 = self.branch_7(x)
        b3 = self.branch_15(x)
        b4 = self.branch_31(x)

        return torch.cat(
            [b1, b2, b3, b4],
            dim=1
        )

# ============================================================
# CNN V3
# ============================================================

class CNNV3MultiEscala(nn.Module):

    def __init__(self):

        super().__init__()

        # Entrada ECG
        self.entry = nn.Sequential(

            nn.Conv1d(
                1,
                32,
                kernel_size=15,
                stride=2,
                padding=7,
                bias=False
            ),

            nn.BatchNorm1d(32),

            nn.ReLU(inplace=True)
        )

        # Extracción multiescala
        # 4 ramas x 32 = 128 canales
        self.multiscale = MultiScaleBlock(
            32,
            branch_channels=32
        )

        # Bloques residuales posteriores
        self.res1 = ResidualBlock(
            128,
            128,
            kernel_size=7,
            stride=2
        )

        self.res2 = ResidualBlock(
            128,
            256,
            kernel_size=5,
            stride=2
        )

        self.res3 = ResidualBlock(
            256,
            256,
            kernel_size=3,
            stride=2
        )

        # Mantener algo más de información temporal:
        # no reducimos directamente a longitud 1.
        self.pool = nn.AdaptiveAvgPool1d(
            4
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Dropout(0.40),

            nn.Linear(
                256 * 4,
                128
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(0.30),

            nn.Linear(
                128,
                1
            )
        )

    def forward(self, x):

        x = self.entry(x)

        x = self.multiscale(x)

        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)

        x = self.pool(x)

        x = self.classifier(x)

        return x.squeeze(1)


model = CNNV3MultiEscala().to(
    device
)

parametros = sum(
    p.numel()
    for p in model.parameters()
)

print("\nMODELO:")
print(
    f"Parámetros entrenables: "
    f"{parametros:,}"
)

# ============================================================
# LOSS / OPTIMIZADOR
# ============================================================

pos_weight = torch.tensor(
    [pos_weight_value],
    dtype=torch.float32,
    device=device
)

criterion = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)

# ============================================================
# EVALUACIÓN
# ============================================================

def evaluar(modelo, loader):

    modelo.eval()

    losses = []
    etiquetas = []
    probabilidades = []

    with torch.no_grad():

        for x, y in loader:

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

            probs = torch.sigmoid(
                logits
            )

            losses.append(
                loss.item()
            )

            etiquetas.append(
                y.cpu().numpy()
            )

            probabilidades.append(
                probs.cpu().numpy()
            )

    y_true = np.concatenate(
        etiquetas
    )

    y_prob = np.concatenate(
        probabilidades
    )

    y_pred = (
        y_prob >= 0.5
    ).astype(int)

    auc = roc_auc_score(
        y_true,
        y_prob
    )

    pr_auc = average_precision_score(
        y_true,
        y_prob
    )

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    sensitivity = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    return {
        "loss": float(np.mean(losses)),
        "accuracy": accuracy,
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "roc_auc": auc,
        "pr_auc": pr_auc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }

# ============================================================
# ENTRENAMIENTO
# ============================================================

best_auc = -np.inf
epochs_without_improvement = 0
historial = []

print("\n" + "=" * 78)
print("INICIANDO ENTRENAMIENTO")
print("=" * 78)

for epoch in range(
    1,
    MAX_EPOCHS + 1
):

    inicio = time.time()

    model.train()

    train_losses = []

    for x, y in loader_train:

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

        logits = model(x)

        loss = criterion(
            logits,
            y
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0
        )

        optimizer.step()

        train_losses.append(
            loss.item()
        )

    metricas = evaluar(
        model,
        loader_val
    )

    train_loss = float(
        np.mean(train_losses)
    )

    scheduler.step(
        metricas["roc_auc"]
    )

    lr_actual = optimizer.param_groups[0]["lr"]

    duracion = (
        time.time() - inicio
    ) / 60

    fila = {
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": metricas["loss"],
        "accuracy": metricas["accuracy"],
        "precision": metricas["precision"],
        "sensitivity": metricas["sensitivity"],
        "specificity": metricas["specificity"],
        "f1": metricas["f1"],
        "roc_auc": metricas["roc_auc"],
        "pr_auc": metricas["pr_auc"],
        "lr": lr_actual
    }

    historial.append(fila)

    print("\n" + "-" * 78)

    print(
        f"Época {epoch}/{MAX_EPOCHS} "
        f"- {duracion:.1f} min"
    )

    print(
        f"Train loss: {train_loss:.4f}"
    )

    print(
        f"Val loss:   {metricas['loss']:.4f}"
    )

    print(
        f"Accuracy:    {metricas['accuracy']:.4f}"
    )

    print(
        f"Precision:   {metricas['precision']:.4f}"
    )

    print(
        f"Sensitivity: {metricas['sensitivity']:.4f}"
    )

    print(
        f"Specificity: {metricas['specificity']:.4f}"
    )

    print(
        f"F1:          {metricas['f1']:.4f}"
    )

    print(
        f"ROC-AUC:     {metricas['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC:      {metricas['pr_auc']:.4f}"
    )

    print(
        f"TN={metricas['tn']:,} "
        f"FP={metricas['fp']:,} "
        f"FN={metricas['fn']:,} "
        f"TP={metricas['tp']:,}"
    )

    print(
        f"Learning rate: {lr_actual:.6f}"
    )

    # --------------------------------------------------------
    # GUARDAR MEJOR MODELO SEGÚN ROC-AUC
    # --------------------------------------------------------

    if metricas["roc_auc"] > best_auc:

        best_auc = metricas["roc_auc"]

        epochs_without_improvement = 0

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict":
                    model.state_dict(),
                "optimizer_state_dict":
                    optimizer.state_dict(),
                "val_roc_auc":
                    metricas["roc_auc"],
                "val_pr_auc":
                    metricas["pr_auc"],
                "criterion":
                    "SBP >= 140 OR DBP >= 90",
                "input":
                    "ECG only",
                "preprocessing":
                    "per-segment z-score",
                "architecture":
                    "CNN V3 multiscale residual",
                "seed":
                    SEED
            },
            RUTA_MODELO
        )

        print(
            ">>> NUEVO MEJOR MODELO GUARDADO"
        )

    else:

        epochs_without_improvement += 1

        print(
            "Sin mejora de ROC-AUC: "
            f"{epochs_without_improvement}/"
            f"{PATIENCE}"
        )

    # Guardar historial en cada época
    pd.DataFrame(
        historial
    ).to_csv(
        RUTA_HISTORIAL,
        index=False
    )

    if (
        epochs_without_improvement
        >= PATIENCE
    ):

        print(
            "\nEARLY STOPPING ACTIVADO."
        )

        break

# ============================================================
# RESUMEN
# ============================================================

historial_df = pd.DataFrame(
    historial
)

mejor_fila = historial_df.loc[
    historial_df["roc_auc"].idxmax()
]

print("\n" + "=" * 78)
print("PUNTO 30 FINALIZADO")
print("=" * 78)

print(
    f"\nMejor época: "
    f"{int(mejor_fila['epoch'])}"
)

print(
    f"Mejor Validation ROC-AUC: "
    f"{mejor_fila['roc_auc']:.4f}"
)

print(
    f"Validation PR-AUC: "
    f"{mejor_fila['pr_auc']:.4f}"
)

print("\nBASELINE V2:")
print("ROC-AUC: 0.7227")
print("PR-AUC:  0.2688")

diferencia = (
    mejor_fila["roc_auc"]
    - 0.7227
)

print(
    f"\nDiferencia AUC V3 - V2: "
    f"{diferencia:+.4f}"
)

print(
    f"\nModelo guardado en:\n"
    f"{RUTA_MODELO}"
)

print(
    f"\nHistorial guardado en:\n"
    f"{RUTA_HISTORIAL}"
)

print(
    "\nIMPORTANTE:"
    "\nCalFree NO fue utilizado."
    "\nLa selección se realizó únicamente "
    "con Validation."
)