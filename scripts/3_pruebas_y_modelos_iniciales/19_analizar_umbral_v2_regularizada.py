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

RUTA_ETIQUETAS_VAL = os.path.join(
    CARPETA_PROCESADO,
    "etiquetas_val.npy"
)

RUTA_MODELO = os.path.join(
    CARPETA_MODELOS,
    "mejor_modelo_v2_regularizada.pt"
)

RUTA_PROBABILIDADES = os.path.join(
    CARPETA_RESULTADOS,
    "probabilidades_validation_v2_regularizada.npy"
)

RUTA_REALES = os.path.join(
    CARPETA_RESULTADOS,
    "etiquetas_validation_v2_regularizada.npy"
)

BATCH_SIZE = 128

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("==============================================")
print("PUNTO 19 - ANÁLISIS DE UMBRAL")
print("CNN V2 REGULARIZADA")
print("==============================================")

print("Dispositivo:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


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

        # Z-score por segmento
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
            32, 32,
            kernel_size=7,
            stride=1
        )

        self.bloque2 = ResidualBlock1D(
            32, 64,
            kernel_size=7,
            stride=2
        )

        self.bloque3 = ResidualBlock1D(
            64, 128,
            kernel_size=5,
            stride=2
        )

        self.bloque4 = ResidualBlock1D(
            128, 256,
            kernel_size=3,
            stride=2
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

        x = self.bloque1(x)
        x = self.bloque2(x)
        x = self.bloque3(x)
        x = self.bloque4(x)

        x = self.pool(x)

        x = self.clasificador(x)

        return x.squeeze(1)


# =========================================================
# CARGAR MODELO GANADOR
# =========================================================

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

print("\n===== CHECKPOINT =====")
print(
    "Época guardada:",
    checkpoint["epoch"]
)

print(
    "ROC-AUC guardado:",
    checkpoint["val_auc"]
)

print(
    "PR-AUC guardado:",
    checkpoint["val_pr_auc"]
)

print(
    "Weight decay:",
    checkpoint["weight_decay"]
)


# =========================================================
# VALIDATION
# =========================================================

val_dataset = ECGDataset(
    RUTA_MAT,
    RUTA_INDICES_VAL,
    RUTA_ETIQUETAS_VAL
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

reales = []
probabilidades = []

print("\nCalculando probabilidades de Validation...")

with torch.no_grad():

    for batch_idx, (
        ecg,
        etiquetas
    ) in enumerate(
        val_loader,
        start=1
    ):

        ecg = ecg.to(
            device,
            non_blocking=True
        )

        logits = modelo(ecg)

        probs = torch.sigmoid(
            logits
        )

        probabilidades.extend(
            probs.cpu().numpy()
        )

        reales.extend(
            etiquetas.numpy()
        )

        if batch_idx % 200 == 0:

            print(
                f"Batch "
                f"{batch_idx}/{len(val_loader)}"
            )


reales = np.asarray(
    reales,
    dtype=np.int8
)

probabilidades = np.asarray(
    probabilidades,
    dtype=np.float32
)

np.save(
    RUTA_PROBABILIDADES,
    probabilidades
)

np.save(
    RUTA_REALES,
    reales
)


# =========================================================
# MÉTRICAS GENERALES
# =========================================================

roc_auc = roc_auc_score(
    reales,
    probabilidades
)

pr_auc = average_precision_score(
    reales,
    probabilidades
)

print("\n==============================================")
print("MÉTRICAS INDEPENDIENTES DEL UMBRAL")
print("==============================================")

print(
    f"Segmentos Validation: "
    f"{len(reales):,}"
)

print(
    f"Positivos: "
    f"{np.sum(reales == 1):,}"
)

print(
    f"Negativos: "
    f"{np.sum(reales == 0):,}"
)

print(
    f"Prevalencia positiva: "
    f"{np.mean(reales):.4f}"
)

print(
    f"ROC-AUC: "
    f"{roc_auc:.4f}"
)

print(
    f"PR-AUC: "
    f"{pr_auc:.4f}"
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
    ).astype(np.int8)

    tn, fp, fn, tp = confusion_matrix(
        reales,
        pred,
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
        "umbral": umbral,

        "accuracy":
            accuracy_score(
                reales,
                pred
            ),

        "precision":
            precision_score(
                reales,
                pred,
                zero_division=0
            ),

        "sensibilidad":
            sensibilidad,

        "especificidad":
            especificidad,

        "f1":
            f1_score(
                reales,
                pred,
                zero_division=0
            ),

        "youden":
            sensibilidad
            + especificidad
            - 1,

        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }


def mostrar(nombre, resultado):

    print(
        f"\n===== {nombre} ====="
    )

    print(
        f"Umbral:       "
        f"{resultado['umbral']:.4f}"
    )

    print(
        f"Accuracy:     "
        f"{resultado['accuracy']:.4f}"
    )

    print(
        f"Precision:    "
        f"{resultado['precision']:.4f}"
    )

    print(
        f"Sensibilidad: "
        f"{resultado['sensibilidad']:.4f}"
    )

    print(
        f"Especificidad:"
        f" {resultado['especificidad']:.4f}"
    )

    print(
        f"F1-score:     "
        f"{resultado['f1']:.4f}"
    )

    print(
        f"Youden:       "
        f"{resultado['youden']:.4f}"
    )

    print(
        f"TN={resultado['tn']} "
        f"FP={resultado['fp']} "
        f"FN={resultado['fn']} "
        f"TP={resultado['tp']}"
    )


# =========================================================
# UMBRAL 0.50
# =========================================================

resultado_050 = evaluar_umbral(
    0.50
)

mostrar(
    "UMBRAL 0.50",
    resultado_050
)


# =========================================================
# MEJOR YOUDEN
# =========================================================

fpr, tpr, thresholds = roc_curve(
    reales,
    probabilidades
)

youden = (
    tpr - fpr
)

indice_youden = np.argmax(
    youden
)

umbral_youden = thresholds[
    indice_youden
]

resultado_youden = evaluar_umbral(
    umbral_youden
)

mostrar(
    "MEJOR ÍNDICE DE YOUDEN",
    resultado_youden
)


# =========================================================
# MEJOR F1
# =========================================================

precision_curve, recall_curve, thresholds_pr = (
    precision_recall_curve(
        reales,
        probabilidades
    )
)

f1_curve = (
    2
    * precision_curve[:-1]
    * recall_curve[:-1]
    / (
        precision_curve[:-1]
        + recall_curve[:-1]
        + 1e-12
    )
)

indice_f1 = np.argmax(
    f1_curve
)

umbral_f1 = thresholds_pr[
    indice_f1
]

resultado_f1 = evaluar_umbral(
    umbral_f1
)

mostrar(
    "MEJOR F1",
    resultado_f1
)


# =========================================================
# SENSIBILIDAD >= 80 %
# =========================================================

candidatos = []

# Analizamos los umbrales únicos producidos por ROC
for umbral in thresholds:

    if not np.isfinite(umbral):
        continue

    resultado = evaluar_umbral(
        umbral
    )

    if (
        resultado["sensibilidad"]
        >= 0.80
    ):
        candidatos.append(
            resultado
        )

if len(candidatos) > 0:

    # Entre los que alcanzan sensibilidad >=80%,
    # elegimos el de mayor especificidad.
    resultado_sens80 = max(
        candidatos,
        key=lambda x: x["especificidad"]
    )

    mostrar(
        "SENSIBILIDAD >= 80% "
        "(MÁXIMA ESPECIFICIDAD)",
        resultado_sens80
    )

else:

    print(
        "\nNo existe un umbral "
        "con sensibilidad >= 80%."
    )


# =========================================================
# TABLA DE UMBRALES FIJOS
# =========================================================

print("\n==============================================")
print("TABLA EXPLORATORIA DE UMBRALES")
print("==============================================")

print(
    "Umbral | Precision | Sensib. | "
    "Especif. | F1"
)

for umbral in np.arange(
    0.20,
    0.81,
    0.05
):

    r = evaluar_umbral(
        float(umbral)
    )

    print(
        f"{umbral:6.2f} | "
        f"{r['precision']:.4f}    | "
        f"{r['sensibilidad']:.4f}  | "
        f"{r['especificidad']:.4f}   | "
        f"{r['f1']:.4f}"
    )


print("\n==============================================")
print("ANÁLISIS TERMINADO")
print("==============================================")

print(
    "Probabilidades guardadas en:"
)

print(
    RUTA_PROBABILIDADES
)

print(
    "\nEtiquetas guardadas en:"
)

print(
    RUTA_REALES
)

print(
    "\nIMPORTANTE: todavía NO se utilizó CalFree."
)