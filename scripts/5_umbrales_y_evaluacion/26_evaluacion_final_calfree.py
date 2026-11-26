import os
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# =========================================================
# CONFIGURACIÓN FINAL CONGELADA
# =========================================================

RUTA_MAT = (
    r"E:\Tesis_PulseDB\data_calfree"
    r"\VitalDB_CalFree_Test_Subset.mat"
)

RUTA_MODELO = (
    r"E:\Tesis_PulseDB\modelos"
    r"\mejor_modelo_v2_regularizada_140_90.pt"
)

CARPETA_RESULTADOS = (
    r"E:\Tesis_PulseDB\resultados"
)

RUTA_PROBS = os.path.join(
    CARPETA_RESULTADOS,
    "1_cnn_residual", "probabilidades_calfree_final.npy"
)

RUTA_LABELS = os.path.join(
    CARPETA_RESULTADOS,
    "1_cnn_residual", "etiquetas_calfree_final.npy"
)

RUTA_METRICAS = os.path.join(
    CARPETA_RESULTADOS,
    "1_cnn_residual", "metricas_finales_calfree.csv"
)

# IMPORTANTE:
# Este umbral fue seleccionado previamente
# SOLO utilizando Validation mediante Youden.

UMBRAL_FINAL = 0.6052

BATCH_SIZE = 128

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 75)
print("PUNTO 26 - EVALUACIÓN FINAL INDEPENDIENTE EN CALFREE")
print("=" * 75)

print(f"\nDispositivo: {device}")

if torch.cuda.is_available():
    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

print("\nCONFIGURACIÓN CONGELADA:")
print("Modelo: CNN V2 residual regularizada")
print("Entrada: ECG")
print("Preprocesamiento: Z-score por segmento")
print("Criterio: SBP >= 140 OR DBP >= 90")
print(f"Umbral final: {UMBRAL_FINAL:.4f}")
print("CalFree: evaluación independiente")

# =========================================================
# LEER SBP Y DBP DE CALFREE
# =========================================================

print("\nLeyendo etiquetas de CalFree...")

with h5py.File(
    RUTA_MAT,
    "r"
) as f:

    subset = f["Subset"]

    sbp = np.asarray(
        subset["SBP"]
    ).reshape(-1).astype(
        np.float32
    )

    dbp = np.asarray(
        subset["DBP"]
    ).reshape(-1).astype(
        np.float32
    )

    total_segmentos = (
        subset["Signals"].shape[2]
    )

print(
    f"Segmentos CalFree: "
    f"{total_segmentos:,}"
)

if (
    len(sbp) != total_segmentos
    or len(dbp) != total_segmentos
):
    raise ValueError(
        "SBP/DBP no coincide con el número "
        "de segmentos de CalFree."
    )

# =========================================================
# ETIQUETAS 140/90
# =========================================================

labels = (
    (sbp >= 140.0)
    |
    (dbp >= 90.0)
).astype(np.float32)

positivos = int(
    labels.sum()
)

negativos = int(
    len(labels) - positivos
)

prevalencia = float(
    labels.mean()
)

print("\nDistribución CalFree 140/90:")

print(
    f"Negativos: {negativos:,}"
)

print(
    f"Positivos: {positivos:,}"
)

print(
    f"Prevalencia positiva: "
    f"{prevalencia:.4f}"
)

# =========================================================
# DATASET
# =========================================================

class CalFreeDataset(Dataset):

    def __init__(
        self,
        ruta_mat,
        labels
    ):

        self.ruta_mat = ruta_mat
        self.labels = labels

        self.archivo = None
        self.signals = None

    def __len__(self):
        return len(self.labels)

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

    def __getitem__(self, idx):

        self._abrir()

        # Canal 0 = ECG
        ecg = np.asarray(
            self.signals[
                :,
                0,
                idx
            ],
            dtype=np.float32
        )

        # Mismo preprocesamiento utilizado
        # durante entrenamiento/Validation.
        media = ecg.mean()
        std = ecg.std()

        if std > 1e-8:

            ecg = (
                (ecg - media)
                / std
            )

        else:

            ecg = (
                ecg - media
            )

        ecg = np.expand_dims(
            ecg,
            axis=0
        )

        label = torch.tensor(
            self.labels[idx],
            dtype=torch.float32
        )

        return (
            torch.from_numpy(ecg),
            label
        )


dataset = CalFreeDataset(
    RUTA_MAT,
    labels
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

print(
    f"\nBatches CalFree: "
    f"{len(loader):,}"
)

# =========================================================
# ARQUITECTURA EXACTA CNN V2
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

        padding = (
            kernel_size // 2
        )

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
            or
            in_channels != out_channels
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

            self.shortcut = (
                nn.Identity()
            )

    def forward(self, x):

        identidad = (
            self.shortcut(x)
        )

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = (
            x + identidad
        )

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
            32,
            32,
            7,
            1
        )

        self.block2 = ResidualBlock1D(
            32,
            64,
            7,
            2
        )

        self.block3 = ResidualBlock1D(
            64,
            128,
            5,
            2
        )

        self.block4 = ResidualBlock1D(
            128,
            256,
            3,
            2
        )

        self.pool = (
            nn.AdaptiveAvgPool1d(1)
        )

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
# CARGAR CHECKPOINT FINAL
# =========================================================

modelo = CNNV2().to(
    device
)

checkpoint = torch.load(
    RUTA_MODELO,
    map_location=device,
    weights_only=False
)

modelo.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)

modelo.eval()

print("\nCheckpoint cargado correctamente.")

print(
    f"Época seleccionada: "
    f"{checkpoint['epoch']}"
)

print(
    f"ROC-AUC Validation del checkpoint: "
    f"{checkpoint['roc_auc']:.4f}"
)

print(
    f"PR-AUC Validation del checkpoint: "
    f"{checkpoint['pr_auc']:.4f}"
)

# =========================================================
# INFERENCIA FINAL
# =========================================================

print(
    "\nIniciando evaluación FINAL "
    "sobre CalFree..."
)

probabilidades = []
etiquetas_reales = []

with torch.no_grad():

    for numero_batch, (x, y) in enumerate(
        loader,
        start=1
    ):

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

        etiquetas_reales.extend(
            y.numpy()
        )

        if (
            numero_batch % 100 == 0
            or numero_batch == len(loader)
        ):

            print(
                f"Batch "
                f"{numero_batch:,}"
                f"/{len(loader):,}"
            )

probabilidades = np.asarray(
    probabilidades,
    dtype=np.float32
)

etiquetas_reales = np.asarray(
    etiquetas_reales,
    dtype=int
)

# =========================================================
# GUARDAR PREDICCIONES
# =========================================================

np.save(
    RUTA_PROBS,
    probabilidades
)

np.save(
    RUTA_LABELS,
    etiquetas_reales
)

# =========================================================
# MÉTRICAS INDEPENDIENTES
# =========================================================

roc_auc = roc_auc_score(
    etiquetas_reales,
    probabilidades
)

pr_auc = average_precision_score(
    etiquetas_reales,
    probabilidades
)

predicciones = (
    probabilidades
    >= UMBRAL_FINAL
).astype(int)

tn, fp, fn, tp = confusion_matrix(
    etiquetas_reales,
    predicciones
).ravel()

accuracy = accuracy_score(
    etiquetas_reales,
    predicciones
)

precision = precision_score(
    etiquetas_reales,
    predicciones,
    zero_division=0
)

sensibilidad = recall_score(
    etiquetas_reales,
    predicciones,
    zero_division=0
)

especificidad = (
    tn / (tn + fp)
)

f1 = f1_score(
    etiquetas_reales,
    predicciones,
    zero_division=0
)

youden = (
    sensibilidad
    + especificidad
    - 1
)

# =========================================================
# RESULTADO FINAL
# =========================================================

print("\n" + "=" * 75)
print("RESULTADOS FINALES - CALFREE")
print("=" * 75)

print(
    f"\nROC-AUC:       "
    f"{roc_auc:.4f}"
)

print(
    f"PR-AUC:        "
    f"{pr_auc:.4f}"
)

print(
    f"Umbral fijo:   "
    f"{UMBRAL_FINAL:.4f}"
)

print(
    f"Accuracy:      "
    f"{accuracy:.4f}"
)

print(
    f"Precision:     "
    f"{precision:.4f}"
)

print(
    f"Sensibilidad:  "
    f"{sensibilidad:.4f}"
)

print(
    f"Especificidad: "
    f"{especificidad:.4f}"
)

print(
    f"F1:            "
    f"{f1:.4f}"
)

print(
    f"Youden:        "
    f"{youden:.4f}"
)

print("\nMatriz de confusión:")

print(
    f"TN={tn:,} | "
    f"FP={fp:,} | "
    f"FN={fn:,} | "
    f"TP={tp:,}"
)

# =========================================================
# GUARDAR RESUMEN
# =========================================================

df_resultados = pd.DataFrame(
    [{
        "dataset":
            "VitalDB_CalFree_Test_Subset",
        "criterio":
            "SBP >= 140 OR DBP >= 90",
        "umbral":
            UMBRAL_FINAL,
        "roc_auc":
            roc_auc,
        "pr_auc":
            pr_auc,
        "accuracy":
            accuracy,
        "precision":
            precision,
        "sensibilidad":
            sensibilidad,
        "especificidad":
            especificidad,
        "f1":
            f1,
        "youden":
            youden,
        "tn":
            tn,
        "fp":
            fp,
        "fn":
            fn,
        "tp":
            tp,
        "prevalencia":
            prevalencia
    }]
)

df_resultados.to_csv(
    RUTA_METRICAS,
    index=False
)

print("\nArchivos guardados:")

print(
    RUTA_PROBS
)

print(
    RUTA_LABELS
)

print(
    RUTA_METRICAS
)

print("\n" + "=" * 75)

print(
    "CALFREE FUE UTILIZADO ÚNICAMENTE "
    "PARA LA EVALUACIÓN FINAL."
)

print(
    "NO SE AJUSTÓ EL MODELO NI EL UMBRAL "
    "CON CALFREE."
)

print("=" * 75)

print(
    "\nPUNTO 26 FINALIZADO."
)