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
    confusion_matrix,
    mean_absolute_error
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
    "mejor_modelo_multitarea_140_90.pt"
)

RUTA_HISTORIAL = os.path.join(
    CARPETA_RESULTADOS,
    "3_cnn_multitarea", "historial_cnn_multitarea_140_90.csv"
)

SEED = 42
BATCH_SIZE = 128
MAX_EPOCHS = 15
PATIENCE = 4

LR = 0.001
WEIGHT_DECAY = 1e-4

# Peso de las tareas auxiliares.
# La clasificación continúa siendo la tarea principal.
LAMBDA_SBP = 0.15
LAMBDA_DBP = 0.10

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
print("PUNTO 31 - CNN MULTITAREA: ECG -> CLASIFICACION + SBP + DBP")
print("=" * 78)

print(f"\nDispositivo: {device}")

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

# ============================================================
# ÍNDICES Y ETIQUETAS
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

neg_train = int(
    (y_train == 0).sum()
)

pos_train = int(
    (y_train == 1).sum()
)

pos_weight_value = (
    neg_train / pos_train
)

# ============================================================
# CARGAR SBP / DBP
# ============================================================

with h5py.File(RUTA_MAT, "r") as f:

    sbp_total = np.asarray(
        f["Subset"]["SBP"]
    ).reshape(-1).astype(np.float32)

    dbp_total = np.asarray(
        f["Subset"]["DBP"]
    ).reshape(-1).astype(np.float32)

sbp_train = sbp_total[
    indices_train
]

dbp_train = dbp_total[
    indices_train
]

sbp_val = sbp_total[
    indices_val
]

dbp_val = dbp_total[
    indices_val
]

# ============================================================
# NORMALIZACIÓN DE TARGETS CONTINUOS
#
# IMPORTANTE:
# media/std calculados SOLO con Train.
# Validation no participa.
# ============================================================

SBP_MEAN = float(
    sbp_train.mean()
)

SBP_STD = float(
    sbp_train.std()
)

DBP_MEAN = float(
    dbp_train.mean()
)

DBP_STD = float(
    dbp_train.std()
)

print("\nDATOS:")

print(
    f"Train:      {len(y_train):,} segmentos"
)

print(
    f"Validation: {len(y_val):,} segmentos"
)

print(
    f"Negativos Train: {neg_train:,}"
)

print(
    f"Positivos Train: {pos_train:,}"
)

print(
    f"pos_weight: {pos_weight_value:.4f}"
)

print("\nNORMALIZACIÓN AUXILIAR:")

print(
    f"SBP Train: media={SBP_MEAN:.4f}, "
    f"std={SBP_STD:.4f}"
)

print(
    f"DBP Train: media={DBP_MEAN:.4f}, "
    f"std={DBP_STD:.4f}"
)

# ============================================================
# DATASET
# ============================================================

class ECGMultiTaskDataset(Dataset):

    def __init__(
        self,
        ruta_mat,
        indices,
        labels,
        sbp,
        dbp
    ):

        self.ruta_mat = ruta_mat
        self.indices = indices
        self.labels = labels
        self.sbp = sbp
        self.dbp = dbp

        self.archivo = None
        self.signals = None

    def _abrir(self):

        if self.archivo is None:

            self.archivo = h5py.File(
                self.ruta_mat,
                "r"
            )

            self.signals = (
                self.archivo[
                    "Subset"
                ]["Signals"]
            )

    def __len__(self):

        return len(
            self.indices
        )

    def __getitem__(self, idx):

        self._abrir()

        indice_real = int(
            self.indices[idx]
        )

        # --------------------------------------------
        # ÚNICA ENTRADA DEL MODELO = ECG
        # --------------------------------------------

        ecg = np.asarray(
            self.signals[
                :,
                0,
                indice_real
            ],
            dtype=np.float32
        )

        # Z-score por segmento ECG
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

        # Etiqueta principal
        label = torch.tensor(
            self.labels[idx],
            dtype=torch.float32
        )

        # Targets auxiliares normalizados
        sbp_norm = (
            self.sbp[idx]
            - SBP_MEAN
        ) / SBP_STD

        dbp_norm = (
            self.dbp[idx]
            - DBP_MEAN
        ) / DBP_STD

        sbp_target = torch.tensor(
            sbp_norm,
            dtype=torch.float32
        )

        dbp_target = torch.tensor(
            dbp_norm,
            dtype=torch.float32
        )

        return (
            ecg,
            label,
            sbp_target,
            dbp_target
        )


dataset_train = ECGMultiTaskDataset(
    RUTA_MAT,
    indices_train,
    y_train,
    sbp_train,
    dbp_train
)

dataset_val = ECGMultiTaskDataset(
    RUTA_MAT,
    indices_val,
    y_val,
    sbp_val,
    dbp_val
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

        self.relu = nn.ReLU(
            inplace=True
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

        identity = self.shortcut(
            x
        )

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)

        return out

# ============================================================
# MODELO MULTITAREA
# ============================================================

class CNNMultiTask(nn.Module):

    def __init__(self):

        super().__init__()

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

        self.block1 = ResidualBlock(
            32,
            32,
            kernel_size=7,
            stride=1
        )

        self.block2 = ResidualBlock(
            32,
            64,
            kernel_size=7,
            stride=2
        )

        self.block3 = ResidualBlock(
            64,
            128,
            kernel_size=5,
            stride=2
        )

        self.block4 = ResidualBlock(
            128,
            256,
            kernel_size=3,
            stride=2
        )

        self.pool = nn.AdaptiveAvgPool1d(
            1
        )

        self.shared = nn.Sequential(

            nn.Flatten(),

            nn.Dropout(0.40),

            nn.Linear(
                256,
                64
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(0.30)
        )

        # Salida principal
        self.head_class = nn.Linear(
            64,
            1
        )

        # Salidas auxiliares
        self.head_sbp = nn.Linear(
            64,
            1
        )

        self.head_dbp = nn.Linear(
            64,
            1
        )

    def forward(self, x):

        x = self.entry(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.pool(x)

        features = self.shared(x)

        logits = self.head_class(
            features
        ).squeeze(1)

        sbp = self.head_sbp(
            features
        ).squeeze(1)

        dbp = self.head_dbp(
            features
        ).squeeze(1)

        return logits, sbp, dbp


model = CNNMultiTask().to(
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

print(
    "Entrada: ECG únicamente"
)

print(
    "Salida principal: clasificación 140/90"
)

print(
    "Tareas auxiliares: SBP y DBP"
)

# ============================================================
# FUNCIONES DE PÉRDIDA
# ============================================================

pos_weight = torch.tensor(
    [pos_weight_value],
    dtype=torch.float32,
    device=device
)

loss_class = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight
)

# SmoothL1 es menos sensible a valores extremos
# que MSE para las tareas auxiliares.
loss_regression = nn.SmoothL1Loss()

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

    class_losses = []

    y_true_list = []
    y_prob_list = []

    sbp_true_list = []
    sbp_pred_list = []

    dbp_true_list = []
    dbp_pred_list = []

    with torch.no_grad():

        for (
            x,
            y,
            sbp_target,
            dbp_target
        ) in loader:

            x = x.to(
                device,
                non_blocking=True
            )

            y = y.to(
                device,
                non_blocking=True
            )

            sbp_target = sbp_target.to(
                device,
                non_blocking=True
            )

            dbp_target = dbp_target.to(
                device,
                non_blocking=True
            )

            (
                logits,
                sbp_pred,
                dbp_pred
            ) = model(x)

            lc = loss_class(
                logits,
                y
            )

            probs = torch.sigmoid(
                logits
            )

            class_losses.append(
                lc.item()
            )

            y_true_list.append(
                y.cpu().numpy()
            )

            y_prob_list.append(
                probs.cpu().numpy()
            )

            sbp_true_list.append(
                sbp_target.cpu().numpy()
            )

            sbp_pred_list.append(
                sbp_pred.cpu().numpy()
            )

            dbp_true_list.append(
                dbp_target.cpu().numpy()
            )

            dbp_pred_list.append(
                dbp_pred.cpu().numpy()
            )

    y_true = np.concatenate(
        y_true_list
    )

    y_prob = np.concatenate(
        y_prob_list
    )

    y_pred = (
        y_prob >= 0.5
    ).astype(int)

    sbp_true_norm = np.concatenate(
        sbp_true_list
    )

    sbp_pred_norm = np.concatenate(
        sbp_pred_list
    )

    dbp_true_norm = np.concatenate(
        dbp_true_list
    )

    dbp_pred_norm = np.concatenate(
        dbp_pred_list
    )

    # Volver a mmHg solamente para
    # evaluar las tareas auxiliares.
    sbp_true_mmHg = (
        sbp_true_norm * SBP_STD
        + SBP_MEAN
    )

    sbp_pred_mmHg = (
        sbp_pred_norm * SBP_STD
        + SBP_MEAN
    )

    dbp_true_mmHg = (
        dbp_true_norm * DBP_STD
        + DBP_MEAN
    )

    dbp_pred_mmHg = (
        dbp_pred_norm * DBP_STD
        + DBP_MEAN
    )

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

    sbp_mae = mean_absolute_error(
        sbp_true_mmHg,
        sbp_pred_mmHg
    )

    dbp_mae = mean_absolute_error(
        dbp_true_mmHg,
        dbp_pred_mmHg
    )

    return {
        "class_loss":
            float(np.mean(class_losses)),
        "accuracy":
            accuracy,
        "precision":
            precision,
        "sensitivity":
            sensitivity,
        "specificity":
            specificity,
        "f1":
            f1,
        "roc_auc":
            auc,
        "pr_auc":
            pr_auc,
        "sbp_mae":
            sbp_mae,
        "dbp_mae":
            dbp_mae,
        "tn":
            int(tn),
        "fp":
            int(fp),
        "fn":
            int(fn),
        "tp":
            int(tp)
    }

# ============================================================
# ENTRENAMIENTO
# ============================================================

best_auc = -np.inf
epochs_without_improvement = 0
historial = []

print("\n" + "=" * 78)
print("INICIANDO ENTRENAMIENTO MULTITAREA")
print("=" * 78)

for epoch in range(
    1,
    MAX_EPOCHS + 1
):

    inicio = time.time()

    model.train()

    total_losses = []
    class_losses = []
    sbp_losses = []
    dbp_losses = []

    for (
        x,
        y,
        sbp_target,
        dbp_target
    ) in loader_train:

        x = x.to(
            device,
            non_blocking=True
        )

        y = y.to(
            device,
            non_blocking=True
        )

        sbp_target = sbp_target.to(
            device,
            non_blocking=True
        )

        dbp_target = dbp_target.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        (
            logits,
            sbp_pred,
            dbp_pred
        ) = model(x)

        lc = loss_class(
            logits,
            y
        )

        ls = loss_regression(
            sbp_pred,
            sbp_target
        )

        ld = loss_regression(
            dbp_pred,
            dbp_target
        )

        # Clasificación = tarea principal
        loss = (
            lc
            + LAMBDA_SBP * ls
            + LAMBDA_DBP * ld
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0
        )

        optimizer.step()

        total_losses.append(
            loss.item()
        )

        class_losses.append(
            lc.item()
        )

        sbp_losses.append(
            ls.item()
        )

        dbp_losses.append(
            ld.item()
        )

    metricas = evaluar(
        model,
        loader_val
    )

    scheduler.step(
        metricas["roc_auc"]
    )

    lr_actual = (
        optimizer.param_groups[0]["lr"]
    )

    duracion = (
        time.time() - inicio
    ) / 60

    fila = {
        "epoch":
            epoch,
        "train_total_loss":
            np.mean(total_losses),
        "train_class_loss":
            np.mean(class_losses),
        "train_sbp_loss":
            np.mean(sbp_losses),
        "train_dbp_loss":
            np.mean(dbp_losses),
        "val_class_loss":
            metricas["class_loss"],
        "accuracy":
            metricas["accuracy"],
        "precision":
            metricas["precision"],
        "sensitivity":
            metricas["sensitivity"],
        "specificity":
            metricas["specificity"],
        "f1":
            metricas["f1"],
        "roc_auc":
            metricas["roc_auc"],
        "pr_auc":
            metricas["pr_auc"],
        "sbp_mae":
            metricas["sbp_mae"],
        "dbp_mae":
            metricas["dbp_mae"],
        "lr":
            lr_actual
    }

    historial.append(
        fila
    )

    print("\n" + "-" * 78)

    print(
        f"Época {epoch}/{MAX_EPOCHS} "
        f"- {duracion:.1f} min"
    )

    print(
        f"Train total loss: "
        f"{np.mean(total_losses):.4f}"
    )

    print(
        f"Train class loss: "
        f"{np.mean(class_losses):.4f}"
    )

    print(
        f"Val class loss:   "
        f"{metricas['class_loss']:.4f}"
    )

    print(
        f"Accuracy:    "
        f"{metricas['accuracy']:.4f}"
    )

    print(
        f"Precision:   "
        f"{metricas['precision']:.4f}"
    )

    print(
        f"Sensitivity: "
        f"{metricas['sensitivity']:.4f}"
    )

    print(
        f"Specificity: "
        f"{metricas['specificity']:.4f}"
    )

    print(
        f"F1:          "
        f"{metricas['f1']:.4f}"
    )

    print(
        f"ROC-AUC:     "
        f"{metricas['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC:      "
        f"{metricas['pr_auc']:.4f}"
    )

    print(
        f"SBP MAE:     "
        f"{metricas['sbp_mae']:.2f} mmHg"
    )

    print(
        f"DBP MAE:     "
        f"{metricas['dbp_mae']:.2f} mmHg"
    )

    print(
        f"TN={metricas['tn']:,} "
        f"FP={metricas['fp']:,} "
        f"FN={metricas['fn']:,} "
        f"TP={metricas['tp']:,}"
    )

    print(
        f"Learning rate: "
        f"{lr_actual:.6f}"
    )

    # ========================================================
    # GUARDAR MEJOR MODELO SEGÚN ROC-AUC DE CLASIFICACIÓN
    # ========================================================

    if metricas["roc_auc"] > best_auc:

        best_auc = metricas[
            "roc_auc"
        ]

        epochs_without_improvement = 0

        torch.save(
            {
                "epoch":
                    epoch,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "val_roc_auc":
                    metricas["roc_auc"],

                "val_pr_auc":
                    metricas["pr_auc"],

                "val_sbp_mae":
                    metricas["sbp_mae"],

                "val_dbp_mae":
                    metricas["dbp_mae"],

                "criterion":
                    "SBP >= 140 OR DBP >= 90",

                "input":
                    "ECG only",

                "preprocessing":
                    "per-segment z-score",

                "architecture":
                    "CNN residual multitask",

                "auxiliary_tasks":
                    "SBP and DBP regression",

                "lambda_sbp":
                    LAMBDA_SBP,

                "lambda_dbp":
                    LAMBDA_DBP,

                "sbp_mean":
                    SBP_MEAN,

                "sbp_std":
                    SBP_STD,

                "dbp_mean":
                    DBP_MEAN,

                "dbp_std":
                    DBP_STD,

                "seed":
                    SEED
            },
            RUTA_MODELO
        )

        print(
            ">>> NUEVO MEJOR MODELO "
            "MULTITAREA GUARDADO"
        )

    else:

        epochs_without_improvement += 1

        print(
            "Sin mejora de ROC-AUC: "
            f"{epochs_without_improvement}/"
            f"{PATIENCE}"
        )

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
# RESUMEN FINAL
# ============================================================

historial_df = pd.DataFrame(
    historial
)

mejor_fila = historial_df.loc[
    historial_df["roc_auc"].idxmax()
]

print("\n" + "=" * 78)
print("PUNTO 31 FINALIZADO")
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

print(
    f"SBP MAE auxiliar: "
    f"{mejor_fila['sbp_mae']:.2f} mmHg"
)

print(
    f"DBP MAE auxiliar: "
    f"{mejor_fila['dbp_mae']:.2f} mmHg"
)

print("\nCOMPARACIÓN:")

print(
    "CNN V2 ROC-AUC: "
    "0.7227"
)

print(
    "CNN V3 ROC-AUC: "
    "0.7221"
)

print(
    "CNN multitarea ROC-AUC: "
    f"{mejor_fila['roc_auc']:.4f}"
)

diferencia = (
    mejor_fila["roc_auc"]
    - 0.7227
)

print(
    f"\nDiferencia multitarea - V2: "
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
    "\n- La única entrada del modelo fue ECG."
    "\n- SBP y DBP se utilizaron únicamente "
    "como supervisión auxiliar."
    "\n- CalFree NO fue utilizado."
    "\n- El modelo se seleccionó únicamente "
    "mediante Validation ROC-AUC."
)