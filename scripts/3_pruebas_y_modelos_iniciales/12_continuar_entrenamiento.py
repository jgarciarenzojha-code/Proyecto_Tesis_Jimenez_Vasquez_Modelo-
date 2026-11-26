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
    confusion_matrix
)

# =========================================================
# CONFIGURACIÓN
# =========================================================

RUTA_MAT = r"E:\Tesis_PulseDB\data_train\VitalDB_Train_Subset.mat"
CARPETA_PROCESADO = r"E:\Tesis_PulseDB\procesado"
CARPETA_MODELOS = r"E:\Tesis_PulseDB\modelos"

RUTA_MODELO = os.path.join(
    CARPETA_MODELOS,
    "mejor_modelo.pt"
)

RUTA_INDICES_TRAIN = os.path.join(
    CARPETA_PROCESADO,
    "indices_train.npy"
)

RUTA_ETIQUETAS_TRAIN = os.path.join(
    CARPETA_PROCESADO,
    "etiquetas_train.npy"
)

RUTA_INDICES_VAL = os.path.join(
    CARPETA_PROCESADO,
    "indices_val.npy"
)

RUTA_ETIQUETAS_VAL = os.path.join(
    CARPETA_PROCESADO,
    "etiquetas_val.npy"
)

BATCH_SIZE = 64

# Máximo de épocas totales
MAX_EPOCHS = 15

# Si AUC no mejora durante 4 épocas consecutivas, detenemos
PATIENCE = 4

POS_WEIGHT = 3.4682
UMBRAL = 0.5

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("===== DISPOSITIVO =====")
print("Usando:", device)

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

        ecg = self.signals[
            :, 0, indice_real
        ]

        ecg = np.asarray(
            ecg,
            dtype=np.float32
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
# CNN
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
# DATOS
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
print("Train:", len(train_dataset))
print("Validation:", len(val_dataset))


# =========================================================
# MODELO Y CHECKPOINT
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
    lr=0.001
)

checkpoint = torch.load(
    RUTA_MODELO,
    map_location=device,
    weights_only=False
)

modelo.load_state_dict(
    checkpoint["model_state_dict"]
)

optimizer.load_state_dict(
    checkpoint["optimizer_state_dict"]
)

epoca_inicial = checkpoint["epoch"]
mejor_auc = checkpoint["val_auc"]

print("\n===== CHECKPOINT =====")
print("Continuando desde época:", epoca_inicial)
print("Mejor ROC-AUC actual:", round(mejor_auc, 4))


# =========================================================
# SCHEDULER
# =========================================================

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

    reales = np.array(
        reales,
        dtype=np.int8
    )

    probabilidades = np.array(
        probabilidades
    )

    predicciones = (
        probabilidades >= UMBRAL
    ).astype(np.int8)

    accuracy = accuracy_score(
        reales,
        predicciones
    )

    precision = precision_score(
        reales,
        predicciones,
        zero_division=0
    )

    sensibilidad = recall_score(
        reales,
        predicciones,
        zero_division=0
    )

    f1 = f1_score(
        reales,
        predicciones,
        zero_division=0
    )

    auc = roc_auc_score(
        reales,
        probabilidades
    )

    matriz = confusion_matrix(
        reales,
        predicciones,
        labels=[0, 1]
    )

    tn, fp, fn, tp = matriz.ravel()

    especificidad = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    val_loss = (
        loss_total /
        len(val_dataset)
    )

    return {
        "loss": val_loss,
        "accuracy": accuracy,
        "precision": precision,
        "sensibilidad": sensibilidad,
        "especificidad": especificidad,
        "f1": f1,
        "auc": auc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    }


# =========================================================
# CONTINUAR ENTRENAMIENTO
# =========================================================

sin_mejora = 0

print("\n===== CONTINUANDO ENTRENAMIENTO =====")

for epoch in range(
    epoca_inicial,
    MAX_EPOCHS
):

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
                f" | Batch {batch_idx + 1}/{len(train_loader)}"
            )

    train_loss = (
        loss_total /
        len(train_dataset)
    )

    metricas = validar()

    # Scheduler observa ROC-AUC
    scheduler.step(
        metricas["auc"]
    )

    lr_actual = optimizer.param_groups[0]["lr"]

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
    # MEJOR MODELO
    # =====================================================

    if metricas["auc"] > mejor_auc:

        mejor_auc = metricas["auc"]

        sin_mejora = 0

        torch.save(
            {
                "epoch": numero_epoca,

                "model_state_dict":
                    modelo.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "val_auc":
                    metricas["auc"],

                "val_f1":
                    metricas["f1"],

                "val_sensibilidad":
                    metricas["sensibilidad"],

                "val_especificidad":
                    metricas["especificidad"],

                "pos_weight":
                    POS_WEIGHT,

                "threshold":
                    UMBRAL
            },
            RUTA_MODELO
        )

        print(
            "\n>>> NUEVO MEJOR MODELO GUARDADO"
        )

        print(
            f">>> ROC-AUC: {mejor_auc:.4f}"
        )

    else:

        sin_mejora += 1

        print(
            f"\nSin mejora: {sin_mejora}/{PATIENCE}"
        )


    # =====================================================
    # EARLY STOPPING
    # =====================================================

    if sin_mejora >= PATIENCE:

        print("\n========================================")
        print("EARLY STOPPING ACTIVADO")
        print("========================================")

        break


print("\n========================================")
print("ENTRENAMIENTO FINALIZADO")
print("========================================")

print(
    "Mejor ROC-AUC Validation:",
    round(mejor_auc, 4)
)

print(
    "Mejor modelo:"
)

print(
    RUTA_MODELO
)