import os
import h5py
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# =========================================================
# CONFIGURACIÓN
# =========================================================

RUTA_MAT = r"E:\Tesis_PulseDB\data_train\VitalDB_Train_Subset.mat"

CARPETA_PROCESADO = r"E:\Tesis_PulseDB\procesado"
CARPETA_MODELOS = r"E:\Tesis_PulseDB\modelos"
CARPETA_RESULTADOS = r"E:\Tesis_PulseDB\resultados"

RUTA_INDICES_VAL = os.path.join(
    CARPETA_PROCESADO,
    "indices_val.npy"
)

RUTA_LABELS_VAL = os.path.join(
    CARPETA_PROCESADO,
    "etiquetas_val_140_90.npy"
)

RUTA_MODELO = os.path.join(
    CARPETA_MODELOS,
    "mejor_modelo_v2_regularizada_140_90.pt"
)

RUTA_PROBS = os.path.join(
    CARPETA_RESULTADOS,
    "1_cnn_residual", "probabilidades_validation_v2_140_90.npy"
)

RUTA_LABELS_SALIDA = os.path.join(
    CARPETA_RESULTADOS,
    "1_cnn_residual", "etiquetas_validation_v2_140_90.npy"
)

BATCH_SIZE = 128

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 70)
print("PUNTO 24 - ANÁLISIS DE UMBRAL - MODELO 140/90")
print("=" * 70)

print(f"Dispositivo: {device}")

# =========================================================
# DATOS
# =========================================================

indices_val = np.load(
    RUTA_INDICES_VAL
)

labels_val = np.load(
    RUTA_LABELS_VAL
).astype(np.float32)

print(
    f"\nSegmentos Validation: "
    f"{len(indices_val):,}"
)

print(
    f"Positivos: "
    f"{int(labels_val.sum()):,}"
)

print(
    f"Negativos: "
    f"{int(len(labels_val) - labels_val.sum()):,}"
)

print(
    f"Prevalencia positiva: "
    f"{labels_val.mean():.4f}"
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

        indice_real = int(
            self.indices[idx]
        )

        ecg = np.asarray(
            self.signals[
                :,
                0,
                indice_real
            ],
            dtype=np.float32
        )

        media = ecg.mean()
        std = ecg.std()

        if std > 1e-8:
            ecg = (
                (ecg - media)
                / std
            )
        else:
            ecg = ecg - media

        ecg = np.expand_dims(
            ecg,
            axis=0
        )

        return (
            torch.from_numpy(ecg),
            torch.tensor(
                self.labels[idx],
                dtype=torch.float32
            )
        )

loader = DataLoader(
    ECGDataset(
        RUTA_MAT,
        indices_val,
        labels_val
    ),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

# =========================================================
# MODELO
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
            kernel_size,
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
            kernel_size,
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
                    1,
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


class CNNV2(nn.Module):

    def __init__(self):
        super().__init__()

        self.entrada = nn.Sequential(
            nn.Conv1d(
                1,
                32,
                15,
                stride=2,
                padding=7,
                bias=False
            ),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        self.block1 = ResidualBlock1D(
            32, 32, 7, 1
        )

        self.block2 = ResidualBlock1D(
            32, 64, 7, 2
        )

        self.block3 = ResidualBlock1D(
            64, 128, 5, 2
        )

        self.block4 = ResidualBlock1D(
            128, 256, 3, 2
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.clasificador = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
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

modelo = CNNV2().to(device)

checkpoint = torch.load(
    RUTA_MODELO,
    map_location=device,
    weights_only=False
)

modelo.load_state_dict(
    checkpoint["model_state_dict"]
)

modelo.eval()

print("\nCheckpoint cargado:")

print(
    f"Época: {checkpoint['epoch']}"
)

print(
    f"ROC-AUC guardado: "
    f"{checkpoint['roc_auc']:.4f}"
)

print(
    f"PR-AUC guardado: "
    f"{checkpoint['pr_auc']:.4f}"
)

# =========================================================
# INFERENCIA VALIDATION
# =========================================================

probabilidades = []
etiquetas = []

print("\nCalculando probabilidades...")

with torch.no_grad():

    for x, y in loader:

        x = x.to(
            device,
            non_blocking=True
        )

        logits = modelo(x)

        probs = torch.sigmoid(
            logits
        )

        probabilidades.extend(
            probs.cpu().numpy()
        )

        etiquetas.extend(
            y.numpy()
        )

probabilidades = np.asarray(
    probabilidades
)

etiquetas = np.asarray(
    etiquetas
).astype(int)

np.save(
    RUTA_PROBS,
    probabilidades
)

np.save(
    RUTA_LABELS_SALIDA,
    etiquetas
)

# =========================================================
# MÉTRICAS GENERALES
# =========================================================

roc_auc = roc_auc_score(
    etiquetas,
    probabilidades
)

pr_auc = average_precision_score(
    etiquetas,
    probabilidades
)

print("\n" + "=" * 70)
print("MÉTRICAS GENERALES")
print("=" * 70)

print(
    f"ROC-AUC: {roc_auc:.4f}"
)

print(
    f"PR-AUC:  {pr_auc:.4f}"
)

print(
    f"Probabilidad mínima: "
    f"{probabilidades.min():.4f}"
)

print(
    f"Probabilidad máxima: "
    f"{probabilidades.max():.4f}"
)

print(
    f"Probabilidad media: "
    f"{probabilidades.mean():.4f}"
)

# =========================================================
# FUNCIÓN PARA EVALUAR UMBRAL
# =========================================================

def evaluar_umbral(umbral):

    pred = (
        probabilidades >= umbral
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        etiquetas,
        pred
    ).ravel()

    acc = accuracy_score(
        etiquetas,
        pred
    )

    precision = precision_score(
        etiquetas,
        pred,
        zero_division=0
    )

    sensibilidad = recall_score(
        etiquetas,
        pred,
        zero_division=0
    )

    especificidad = (
        tn / (tn + fp)
    )

    f1 = f1_score(
        etiquetas,
        pred,
        zero_division=0
    )

    youden = (
        sensibilidad
        + especificidad
        - 1
    )

    return {
        "umbral": umbral,
        "accuracy": acc,
        "precision": precision,
        "sensibilidad": sensibilidad,
        "especificidad": especificidad,
        "f1": f1,
        "youden": youden,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    }


def imprimir(nombre, r):

    print(
        f"\n--- {nombre} ---"
    )

    print(
        f"Umbral:       "
        f"{r['umbral']:.4f}"
    )

    print(
        f"Accuracy:     "
        f"{r['accuracy']:.4f}"
    )

    print(
        f"Precision:    "
        f"{r['precision']:.4f}"
    )

    print(
        f"Sensibilidad: "
        f"{r['sensibilidad']:.4f}"
    )

    print(
        f"Especificidad:"
        f" {r['especificidad']:.4f}"
    )

    print(
        f"F1:           "
        f"{r['f1']:.4f}"
    )

    print(
        f"Youden:       "
        f"{r['youden']:.4f}"
    )

    print(
        f"TN={r['tn']:,} | "
        f"FP={r['fp']:,} | "
        f"FN={r['fn']:,} | "
        f"TP={r['tp']:,}"
    )

# =========================================================
# UMBRAL 0.50
# =========================================================

resultado_050 = evaluar_umbral(
    0.50
)

imprimir(
    "UMBRAL BASE 0.50",
    resultado_050
)

# =========================================================
# MEJOR YOUDEN
# =========================================================

fpr, tpr, thresholds = roc_curve(
    etiquetas,
    probabilidades
)

youden_values = (
    tpr - fpr
)

indice_youden = np.argmax(
    youden_values
)

umbral_youden = thresholds[
    indice_youden
]

resultado_youden = evaluar_umbral(
    umbral_youden
)

imprimir(
    "MEJOR ÍNDICE DE YOUDEN",
    resultado_youden
)

# =========================================================
# MEJOR F1
# =========================================================

precision_curve, recall_curve, pr_thresholds = (
    precision_recall_curve(
        etiquetas,
        probabilidades
    )
)

f1_values = (
    2
    * precision_curve[:-1]
    * recall_curve[:-1]
    /
    (
        precision_curve[:-1]
        + recall_curve[:-1]
        + 1e-12
    )
)

indice_f1 = np.argmax(
    f1_values
)

umbral_f1 = pr_thresholds[
    indice_f1
]

resultado_f1 = evaluar_umbral(
    umbral_f1
)

imprimir(
    "MEJOR F1",
    resultado_f1
)

# =========================================================
# SENSIBILIDAD >= 80 %
# =========================================================

mejor_sens80 = None

for umbral in np.unique(
    probabilidades
):

    r = evaluar_umbral(
        umbral
    )

    if r["sensibilidad"] >= 0.80:

        if (
            mejor_sens80 is None
            or
            r["especificidad"]
            >
            mejor_sens80["especificidad"]
        ):
            mejor_sens80 = r

if mejor_sens80 is not None:

    imprimir(
        "SENSIBILIDAD >= 80% "
        "CON MAYOR ESPECIFICIDAD",
        mejor_sens80
    )

# =========================================================
# UMBRALES EXPLORATORIOS
# =========================================================

print("\n" + "=" * 70)
print("UMBRALES EXPLORATORIOS")
print("=" * 70)

for umbral in [
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80
]:

    r = evaluar_umbral(
        umbral
    )

    print(
        f"{umbral:.2f} | "
        f"Prec={r['precision']:.4f} | "
        f"Sens={r['sensibilidad']:.4f} | "
        f"Spec={r['especificidad']:.4f} | "
        f"F1={r['f1']:.4f}"
    )

print("\nArchivos guardados:")

print(RUTA_PROBS)
print(RUTA_LABELS_SALIDA)

print(
    "\nCalFree NO fue utilizado."
)

print(
    "\nPUNTO 24 FINALIZADO."
)