import os
import csv
import h5py
import numpy as np
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

os.makedirs(CARPETA_MODELOS, exist_ok=True)
os.makedirs(CARPETA_RESULTADOS, exist_ok=True)

RUTA_INDICES_TRAIN = os.path.join(
    CARPETA_PROCESADO, "indices_train.npy"
)

RUTA_ETIQUETAS_TRAIN = os.path.join(
    CARPETA_PROCESADO, "etiquetas_train.npy"
)

RUTA_INDICES_VAL = os.path.join(
    CARPETA_PROCESADO, "indices_val.npy"
)

RUTA_ETIQUETAS_VAL = os.path.join(
    CARPETA_PROCESADO, "etiquetas_val.npy"
)

RUTA_MODELO = os.path.join(
    CARPETA_MODELOS,
    "mejor_modelo_v2_zscore.pt"
)

RUTA_CSV = os.path.join(
    CARPETA_RESULTADOS,
    "historial_cnn_v2_zscore.csv"
)

BATCH_SIZE = 128
MAX_EPOCHS = 15
PATIENCE = 4

LEARNING_RATE = 0.001
POS_WEIGHT = 3.4682

# Solo para métricas descriptivas durante entrenamiento.
# NO significa que 0.50 será el umbral final.
UMBRAL = 0.50

SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("========================================")
print("CNN V2 + Z-SCORE")
print("========================================")

print("Dispositivo:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# =========================================================
# DATASET
# =========================================================

class ECGDataset(Dataset):

    def __init__(
        self,
        ruta_mat,
        ruta_indices,
        ruta_etiquetas
    ):

        self.ruta_mat = ruta_mat

        self.indices = np.load(
            ruta_indices
        )

        self.etiquetas = np.load(
            ruta_etiquetas
        )

        self.archivo = None
        self.signals = None

    def __len__(self):

        return len(self.indices)

    def _abrir(self):

        if self.archivo is None:

            self.archivo = h5py.File(
                self.ruta_mat,
                "r"
            )

            self.signals = (
                self.archivo["Subset"]["Signals"]
            )

    def __getitem__(self, idx):

        self._abrir()

        indice_real = self.indices[idx]

        ecg = np.asarray(
            self.signals[:, 0, indice_real],
            dtype=np.float32
        )

        # -----------------------------------------
        # Z-SCORE POR SEGMENTO
        # -----------------------------------------

        media = np.mean(ecg)
        std = np.std(ecg)

        ecg = (
            ecg - media
        ) / (
            std + 1e-8
        )

        ecg = np.expand_dims(
            ecg,
            axis=0
        )

        etiqueta = np.float32(
            self.etiquetas[idx]
        )

        return (
            torch.from_numpy(ecg),
            torch.tensor(
                etiqueta,
                dtype=torch.float32
            )
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

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = x + identidad
        x = self.relu(x)

        return x


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

        self.bloque1 = ResidualBlock1D(
            32,
            32,
            kernel_size=7,
            stride=1
        )

        self.bloque2 = ResidualBlock1D(
            32,
            64,
            kernel_size=7,
            stride=2
        )

        self.bloque3 = ResidualBlock1D(
            64,
            128,
            kernel_size=5,
            stride=2
        )

        self.bloque4 = ResidualBlock1D(
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

        x = self.bloque1(x)
        x = self.bloque2(x)
        x = self.bloque3(x)
        x = self.bloque4(x)

        x = self.pool(x)

        x = self.clasificador(x)

        return x.squeeze(1)


# =========================================================
# DATA LOADERS
# =========================================================

train_dataset = ECGDataset(
    RUTA_MAT,
    RUTA_INDICES_TRAIN,
    RUTA_ETIQUETAS_TRAIN
)

val_dataset = ECGDataset(
    RUTA_MAT,
    RUTA_INDICES_VAL,
    RUTA_ETIQUETAS_VAL
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

print("\n===== DATOS =====")

print(
    "Train:",
    len(train_dataset)
)

print(
    "Validation:",
    len(val_dataset)
)

print(
    "Batch size:",
    BATCH_SIZE
)

print(
    "Batches Train:",
    len(train_loader)
)

print(
    "Batches Validation:",
    len(val_loader)
)

print(
    "Preprocesamiento: Z-score por segmento"
)


# =========================================================
# MODELO / LOSS / OPTIMIZER
# =========================================================

modelo = CNNV2().to(device)

total_parametros = sum(
    p.numel()
    for p in modelo.parameters()
)

print(
    "\nParámetros CNN V2:",
    f"{total_parametros:,}"
)

pos_weight = torch.tensor(
    [POS_WEIGHT],
    dtype=torch.float32,
    device=device
)

criterion = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight
)

optimizer = torch.optim.Adam(
    modelo.parameters(),
    lr=LEARNING_RATE
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2,
    min_lr=0.00001
)


# =========================================================
# VALIDACIÓN
# =========================================================

def validar():

    modelo.eval()

    loss_total = 0.0

    reales = []
    probabilidades = []

    with torch.no_grad():

        for ecg, etiquetas in val_loader:

            ecg = ecg.to(
                device,
                non_blocking=True
            )

            etiquetas = etiquetas.to(
                device,
                non_blocking=True
            )

            logits = modelo(ecg)

            loss = criterion(
                logits,
                etiquetas
            )

            loss_total += (
                loss.item()
                * ecg.size(0)
            )

            probs = torch.sigmoid(
                logits
            )

            reales.extend(
                etiquetas
                .cpu()
                .numpy()
            )

            probabilidades.extend(
                probs
                .cpu()
                .numpy()
            )

    reales = np.asarray(
        reales,
        dtype=np.int8
    )

    probabilidades = np.asarray(
        probabilidades,
        dtype=np.float32
    )

    predicciones = (
        probabilidades >= UMBRAL
    ).astype(np.int8)

    tn, fp, fn, tp = confusion_matrix(
        reales,
        predicciones,
        labels=[0, 1]
    ).ravel()

    sensibilidad = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    especificidad = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    return {

        "loss":
            loss_total / len(val_dataset),

        "accuracy":
            accuracy_score(
                reales,
                predicciones
            ),

        "precision":
            precision_score(
                reales,
                predicciones,
                zero_division=0
            ),

        "sensibilidad":
            sensibilidad,

        "especificidad":
            especificidad,

        "f1":
            f1_score(
                reales,
                predicciones,
                zero_division=0
            ),

        "roc_auc":
            roc_auc_score(
                reales,
                probabilidades
            ),

        "pr_auc":
            average_precision_score(
                reales,
                probabilidades
            ),

        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }


# =========================================================
# CREAR CSV
# =========================================================

columnas_csv = [
    "epoca",
    "learning_rate",
    "train_loss",
    "validation_loss",
    "accuracy",
    "precision",
    "sensibilidad",
    "especificidad",
    "f1",
    "roc_auc",
    "pr_auc",
    "tn",
    "fp",
    "fn",
    "tp"
]

with open(
    RUTA_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as archivo_csv:

    escritor = csv.DictWriter(
        archivo_csv,
        fieldnames=columnas_csv
    )

    escritor.writeheader()


# =========================================================
# ENTRENAMIENTO
# =========================================================

mejor_auc = -1.0
mejor_epoca = 0
sin_mejora = 0

print("\n========================================")
print("INICIO DEL ENTRENAMIENTO REAL")
print("========================================")

for epoch in range(
    1,
    MAX_EPOCHS + 1
):

    modelo.train()

    train_loss_total = 0.0

    for batch_idx, (
        ecg,
        etiquetas
    ) in enumerate(
        train_loader,
        start=1
    ):

        ecg = ecg.to(
            device,
            non_blocking=True
        )

        etiquetas = etiquetas.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = modelo(ecg)

        loss = criterion(
            logits,
            etiquetas
        )

        loss.backward()

        optimizer.step()

        train_loss_total += (
            loss.item()
            * ecg.size(0)
        )

        if batch_idx % 500 == 0:

            print(
                f"Época {epoch}/{MAX_EPOCHS}"
                f" | Batch "
                f"{batch_idx}/{len(train_loader)}"
            )

    train_loss = (
        train_loss_total
        / len(train_dataset)
    )

    metricas = validar()

    # LR usado durante la época que acaba de terminar
    lr_epoca = (
        optimizer.param_groups[0]["lr"]
    )

    # Scheduler observa AUC de validación
    scheduler.step(
        metricas["roc_auc"]
    )

    print("\n----------------------------------------")
    print(
        f"ÉPOCA {epoch}/{MAX_EPOCHS}"
    )
    print("----------------------------------------")

    print(
        f"Learning Rate:     {lr_epoca:.6f}"
    )

    print(
        f"Train Loss:        {train_loss:.4f}"
    )

    print(
        f"Validation Loss:   "
        f"{metricas['loss']:.4f}"
    )

    print(
        f"Accuracy:          "
        f"{metricas['accuracy']:.4f}"
    )

    print(
        f"Precision:         "
        f"{metricas['precision']:.4f}"
    )

    print(
        f"Sensibilidad:      "
        f"{metricas['sensibilidad']:.4f}"
    )

    print(
        f"Especificidad:     "
        f"{metricas['especificidad']:.4f}"
    )

    print(
        f"F1-score:          "
        f"{metricas['f1']:.4f}"
    )

    print(
        f"ROC-AUC:           "
        f"{metricas['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC:            "
        f"{metricas['pr_auc']:.4f}"
    )

    print("\nMatriz de confusión:")

    print(
        f"TN: {metricas['tn']}  "
        f"FP: {metricas['fp']}"
    )

    print(
        f"FN: {metricas['fn']}  "
        f"TP: {metricas['tp']}"
    )


    # =====================================================
    # GUARDAR RESULTADOS EN CSV
    # =====================================================

    fila = {

        "epoca": epoch,

        "learning_rate":
            lr_epoca,

        "train_loss":
            train_loss,

        "validation_loss":
            metricas["loss"],

        "accuracy":
            metricas["accuracy"],

        "precision":
            metricas["precision"],

        "sensibilidad":
            metricas["sensibilidad"],

        "especificidad":
            metricas["especificidad"],

        "f1":
            metricas["f1"],

        "roc_auc":
            metricas["roc_auc"],

        "pr_auc":
            metricas["pr_auc"],

        "tn":
            metricas["tn"],

        "fp":
            metricas["fp"],

        "fn":
            metricas["fn"],

        "tp":
            metricas["tp"]
    }

    with open(
        RUTA_CSV,
        "a",
        newline="",
        encoding="utf-8"
    ) as archivo_csv:

        escritor = csv.DictWriter(
            archivo_csv,
            fieldnames=columnas_csv
        )

        escritor.writerow(
            fila
        )


    # =====================================================
    # MEJOR MODELO
    # =====================================================

    if (
        metricas["roc_auc"]
        > mejor_auc
    ):

        mejor_auc = (
            metricas["roc_auc"]
        )

        mejor_epoca = epoch
        sin_mejora = 0

        torch.save(
            {
                "epoch":
                    epoch,

                "model_state_dict":
                    modelo.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "val_auc":
                    metricas["roc_auc"],

                "val_pr_auc":
                    metricas["pr_auc"],

                "val_accuracy":
                    metricas["accuracy"],

                "val_precision":
                    metricas["precision"],

                "val_sensibilidad":
                    metricas["sensibilidad"],

                "val_especificidad":
                    metricas["especificidad"],

                "val_f1":
                    metricas["f1"],

                "pos_weight":
                    POS_WEIGHT,

                "threshold":
                    UMBRAL,

                "preprocessing":
                    "zscore_por_segmento",

                "architecture":
                    "CNNV2_residual"
            },
            RUTA_MODELO
        )

        print(
            "\n>>> NUEVO MEJOR MODELO CNN V2 GUARDADO"
        )

        print(
            f">>> ROC-AUC: "
            f"{mejor_auc:.4f}"
        )

    else:

        sin_mejora += 1

        print(
            f"\nSin mejora: "
            f"{sin_mejora}/{PATIENCE}"
        )


    # =====================================================
    # EARLY STOPPING
    # =====================================================

    if sin_mejora >= PATIENCE:

        print("\n========================================")
        print("EARLY STOPPING ACTIVADO")
        print("========================================")

        break


# =========================================================
# FINAL
# =========================================================

print("\n========================================")
print("ENTRENAMIENTO CNN V2 FINALIZADO")
print("========================================")

print(
    "Mejor época:",
    mejor_epoca
)

print(
    "Mejor ROC-AUC Validation:",
    round(mejor_auc, 4)
)

print(
    "\nModelo guardado:"
)

print(
    RUTA_MODELO
)

print(
    "\nHistorial guardado:"
)

print(
    RUTA_CSV
)

print("\n========================================")
print("COMPARACIÓN")
print("========================================")

print(
    "CNN v1 original:          0.6385"
)

print(
    "CNN v1 + Z-score:         0.6442"
)

print(
    "CNN v2 + Z-score:        ",
    round(mejor_auc, 4)
)

print("\nCalFree NO fue utilizado.")