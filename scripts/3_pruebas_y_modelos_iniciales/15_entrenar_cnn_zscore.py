import os
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

os.makedirs(CARPETA_MODELOS, exist_ok=True)

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

# Guardamos este experimento aparte.
# NO sobrescribe mejor_modelo.pt.
RUTA_MODELO_ZSCORE = os.path.join(
    CARPETA_MODELOS,
    "mejor_modelo_zscore.pt"
)

BATCH_SIZE = 64
MAX_EPOCHS = 12
PATIENCE = 4

LEARNING_RATE = 0.001
POS_WEIGHT = 3.4682
UMBRAL = 0.5

SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("===== DISPOSITIVO =====")
print("Usando:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# =========================================================
# DATASET CON Z-SCORE
# =========================================================

class ECGDatasetZScore(Dataset):

    def __init__(
        self,
        ruta_mat,
        ruta_indices,
        ruta_etiquetas
    ):

        self.ruta_mat = ruta_mat
        self.indices = np.load(ruta_indices)
        self.etiquetas = np.load(ruta_etiquetas)

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

        indice_real = self.indices[idx]

        ecg = np.asarray(
            self.signals[:, 0, indice_real],
            dtype=np.float32
        )

        # =================================================
        # ÚNICO CAMBIO PRINCIPAL DEL EXPERIMENTO:
        # Z-SCORE POR CADA SEGMENTO ECG
        # =================================================

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
            torch.tensor(etiqueta)
        )


# =========================================================
# MISMA CNN V1
# =========================================================

class CNN1D(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv1d(
                1, 32,
                kernel_size=7,
                padding=3
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(
                32, 64,
                kernel_size=5,
                padding=2
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(
                64, 128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        return x.squeeze(1)


# =========================================================
# DATASETS
# =========================================================

train_dataset = ECGDatasetZScore(
    RUTA_MAT,
    RUTA_INDICES_TRAIN,
    RUTA_ETIQUETAS_TRAIN
)

val_dataset = ECGDatasetZScore(
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
print("Train:", len(train_dataset))
print("Validation:", len(val_dataset))

print("\nPreprocesamiento:")
print("Z-score individual por segmento ECG")


# =========================================================
# MODELO
# =========================================================

modelo = CNN1D().to(device)

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
                loss.item() * ecg.size(0)
            )

            probs = torch.sigmoid(logits)

            reales.extend(
                etiquetas.cpu().numpy()
            )

            probabilidades.extend(
                probs.cpu().numpy()
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

    matriz = confusion_matrix(
        reales,
        predicciones,
        labels=[0, 1]
    )

    tn, fp, fn, tp = matriz.ravel()

    sensibilidad = recall_score(
        reales,
        predicciones,
        zero_division=0
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

        "auc":
            roc_auc_score(
                reales,
                probabilidades
            ),

        "pr_auc":
            average_precision_score(
                reales,
                probabilidades
            ),

        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    }


# =========================================================
# ENTRENAMIENTO
# =========================================================

mejor_auc = -1.0
mejor_epoca = 0
sin_mejora = 0

print("\n========================================")
print("ENTRENAMIENTO CNN + Z-SCORE")
print("========================================")

for epoch in range(MAX_EPOCHS):

    numero_epoca = epoch + 1

    modelo.train()

    loss_total = 0.0

    for batch_idx, (ecg, etiquetas) in enumerate(
        train_loader
    ):

        ecg = ecg.to(
            device,
            non_blocking=True
        )

        etiquetas = etiquetas.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad()

        logits = modelo(ecg)

        loss = criterion(
            logits,
            etiquetas
        )

        loss.backward()

        optimizer.step()

        loss_total += (
            loss.item() * ecg.size(0)
        )

        if (batch_idx + 1) % 1000 == 0:

            print(
                f"Época {numero_epoca}/{MAX_EPOCHS}"
                f" | Batch "
                f"{batch_idx + 1}/{len(train_loader)}"
            )

    train_loss = (
        loss_total /
        len(train_dataset)
    )

    metricas = validar()

    scheduler.step(
        metricas["auc"]
    )

    lr_actual = (
        optimizer.param_groups[0]["lr"]
    )

    print("\n----------------------------------------")
    print(
        f"ÉPOCA {numero_epoca}/{MAX_EPOCHS}"
    )
    print("----------------------------------------")

    print(
        f"Learning Rate:     {lr_actual:.6f}"
    )

    print(
        f"Train Loss:        {train_loss:.4f}"
    )

    print(
        f"Validation Loss:   {metricas['loss']:.4f}"
    )

    print(
        f"Accuracy:          {metricas['accuracy']:.4f}"
    )

    print(
        f"Precision:         {metricas['precision']:.4f}"
    )

    print(
        f"Sensibilidad:      {metricas['sensibilidad']:.4f}"
    )

    print(
        f"Especificidad:     {metricas['especificidad']:.4f}"
    )

    print(
        f"F1-score:          {metricas['f1']:.4f}"
    )

    print(
        f"ROC-AUC:           {metricas['auc']:.4f}"
    )

    print(
        f"PR-AUC:            {metricas['pr_auc']:.4f}"
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
    # GUARDAR MEJOR MODELO
    # =====================================================

    if metricas["auc"] > mejor_auc:

        mejor_auc = metricas["auc"]
        mejor_epoca = numero_epoca
        sin_mejora = 0

        torch.save(
            {
                "epoch":
                    numero_epoca,

                "model_state_dict":
                    modelo.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "val_auc":
                    metricas["auc"],

                "val_pr_auc":
                    metricas["pr_auc"],

                "val_f1":
                    metricas["f1"],

                "val_sensibilidad":
                    metricas["sensibilidad"],

                "val_especificidad":
                    metricas["especificidad"],

                "pos_weight":
                    POS_WEIGHT,

                "threshold":
                    UMBRAL,

                "preprocessing":
                    "zscore_por_segmento"
            },
            RUTA_MODELO_ZSCORE
        )

        print(
            "\n>>> NUEVO MEJOR MODELO Z-SCORE GUARDADO"
        )

        print(
            f">>> ROC-AUC: {mejor_auc:.4f}"
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
# RESULTADO FINAL
# =========================================================

print("\n========================================")
print("EXPERIMENTO Z-SCORE FINALIZADO")
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
    "Modelo guardado en:"
)

print(
    RUTA_MODELO_ZSCORE
)

print("\nComparación pendiente:")

print(
    "Baseline ECG original ROC-AUC: 0.6385"
)

print(
    "CNN + Z-score ROC-AUC:",
    round(mejor_auc, 4)
)