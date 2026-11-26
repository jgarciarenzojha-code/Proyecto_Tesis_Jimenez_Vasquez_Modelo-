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
    "mejor_modelo.pt"
)

BATCH_SIZE = 64

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
# MODELO
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
# CARGAR VALIDATION
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

print("\n===== VALIDATION =====")
print("Segmentos:", len(val_dataset))


# =========================================================
# CARGAR MEJOR MODELO
# =========================================================

modelo = CNN1D().to(device)

checkpoint = torch.load(
    RUTA_MODELO,
    map_location=device,
    weights_only=False
)

modelo.load_state_dict(
    checkpoint["model_state_dict"]
)

modelo.eval()

print("\n===== MODELO =====")
print("Época guardada:", checkpoint["epoch"])
print(
    "ROC-AUC guardado:",
    round(float(checkpoint["val_auc"]), 4)
)


# =========================================================
# GENERAR PROBABILIDADES EN VALIDATION
# =========================================================

reales = []
probabilidades = []

print("\nGenerando predicciones de Validation...")

with torch.no_grad():

    for batch_idx, (ecg, etiquetas) in enumerate(
        val_loader
    ):

        ecg = ecg.to(
            device,
            non_blocking=True
        )

        logits = modelo(ecg)

        probs = torch.sigmoid(logits)

        reales.extend(
            etiquetas.numpy()
        )

        probabilidades.extend(
            probs.cpu().numpy()
        )

        if (batch_idx + 1) % 500 == 0:

            print(
                f"Batch {batch_idx + 1}/{len(val_loader)}"
            )


reales = np.asarray(
    reales,
    dtype=np.int8
)

probabilidades = np.asarray(
    probabilidades,
    dtype=np.float32
)


# =========================================================
# MÉTRICAS INDEPENDIENTES DEL UMBRAL
# =========================================================

roc_auc = roc_auc_score(
    reales,
    probabilidades
)

pr_auc = average_precision_score(
    reales,
    probabilidades
)

prevalencia = reales.mean()

print("\n========================================")
print("MÉTRICAS GENERALES DE VALIDATION")
print("========================================")

print(
    f"Positivos reales: {reales.sum()}/{len(reales)}"
)

print(
    f"Prevalencia positiva: {prevalencia:.4f}"
)

print(
    f"ROC-AUC: {roc_auc:.4f}"
)

print(
    f"PR-AUC (Average Precision): {pr_auc:.4f}"
)

print(
    f"Probabilidad mínima: {probabilidades.min():.4f}"
)

print(
    f"Probabilidad máxima: {probabilidades.max():.4f}"
)

print(
    f"Probabilidad media: {probabilidades.mean():.4f}"
)


# =========================================================
# FUNCIÓN PARA CALCULAR MÉTRICAS POR UMBRAL
# =========================================================

def calcular_metricas(umbral):

    pred = (
        probabilidades >= umbral
    ).astype(np.int8)

    matriz = confusion_matrix(
        reales,
        pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = matriz.ravel()

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

    accuracy = accuracy_score(
        reales,
        pred
    )

    precision = precision_score(
        reales,
        pred,
        zero_division=0
    )

    f1 = f1_score(
        reales,
        pred,
        zero_division=0
    )

    youden = (
        sensibilidad +
        especificidad -
        1
    )

    return {
        "umbral": umbral,
        "accuracy": accuracy,
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


# =========================================================
# TABLA DE UMBRALES
# =========================================================

print("\n========================================")
print("COMPARACIÓN DE UMBRALES")
print("========================================")

print(
    "Umbral | Acc    | Prec   | Sens   | "
    "Esp    | F1     | Youden"
)

print("-" * 67)

resultados = []

for umbral in np.arange(
    0.10,
    0.91,
    0.05
):

    m = calcular_metricas(
        float(umbral)
    )

    resultados.append(m)

    print(
        f"{m['umbral']:.2f}   | "
        f"{m['accuracy']:.4f} | "
        f"{m['precision']:.4f} | "
        f"{m['sensibilidad']:.4f} | "
        f"{m['especificidad']:.4f} | "
        f"{m['f1']:.4f} | "
        f"{m['youden']:.4f}"
    )


# =========================================================
# MEJOR UMBRAL SEGÚN YOUDEN
# BÚSQUEDA FINA
# =========================================================

mejor_youden = None

for umbral in np.arange(
    0.05,
    0.951,
    0.001
):

    m = calcular_metricas(
        float(umbral)
    )

    if (
        mejor_youden is None
        or m["youden"] > mejor_youden["youden"]
    ):

        mejor_youden = m


# =========================================================
# MEJOR UMBRAL SEGÚN F1
# =========================================================

mejor_f1 = None

for umbral in np.arange(
    0.05,
    0.951,
    0.001
):

    m = calcular_metricas(
        float(umbral)
    )

    if (
        mejor_f1 is None
        or m["f1"] > mejor_f1["f1"]
    ):

        mejor_f1 = m


# =========================================================
# UMBRAL PARA SENSIBILIDAD >= 80%
# CON MAYOR ESPECIFICIDAD
# =========================================================

candidatos_sens80 = []

for umbral in np.arange(
    0.05,
    0.951,
    0.001
):

    m = calcular_metricas(
        float(umbral)
    )

    if m["sensibilidad"] >= 0.80:
        candidatos_sens80.append(m)

if candidatos_sens80:

    sens80 = max(
        candidatos_sens80,
        key=lambda x: x["especificidad"]
    )

else:
    sens80 = None


# =========================================================
# RESULTADOS IMPORTANTES
# =========================================================

def mostrar_resultado(
    titulo,
    m
):

    print("\n" + titulo)

    print(
        f"Umbral:        {m['umbral']:.3f}"
    )

    print(
        f"Accuracy:      {m['accuracy']:.4f}"
    )

    print(
        f"Precision:     {m['precision']:.4f}"
    )

    print(
        f"Sensibilidad:  {m['sensibilidad']:.4f}"
    )

    print(
        f"Especificidad: {m['especificidad']:.4f}"
    )

    print(
        f"F1-score:      {m['f1']:.4f}"
    )

    print(
        f"Youden:        {m['youden']:.4f}"
    )

    print(
        f"TN={m['tn']} "
        f"FP={m['fp']} "
        f"FN={m['fn']} "
        f"TP={m['tp']}"
    )


print("\n========================================")
print("UMBRALES CANDIDATOS")
print("========================================")

mostrar_resultado(
    "1. Mejor índice de Youden:",
    mejor_youden
)

mostrar_resultado(
    "2. Mejor F1-score:",
    mejor_f1
)

if sens80 is not None:

    mostrar_resultado(
        "3. Sensibilidad >= 80% "
        "con mayor especificidad:",
        sens80
    )

else:

    print(
        "\nNo se encontró umbral "
        "con sensibilidad >= 80%."
    )


print("\n========================================")
print("ANÁLISIS FINALIZADO")
print("========================================")