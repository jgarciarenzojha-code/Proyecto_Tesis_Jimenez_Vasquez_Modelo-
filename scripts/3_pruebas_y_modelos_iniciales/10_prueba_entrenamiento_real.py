import h5py
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader, Subset


# =========================================================
# CONFIGURACIÓN
# =========================================================

RUTA_MAT = r"E:\Tesis_PulseDB\data_train\VitalDB_Train_Subset.mat"
RUTA_INDICES = r"E:\Tesis_PulseDB\procesado\indices_train.npy"
RUTA_ETIQUETAS = r"E:\Tesis_PulseDB\procesado\etiquetas_train.npy"

BATCH_SIZE = 64

# Solo para esta prueba técnica
MAX_BATCHES = 200

POS_WEIGHT = 3.4682

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

        # Canal 0 = ECG
        ecg = self.signals[
            :, 0, indice_real
        ]

        ecg = np.asarray(
            ecg,
            dtype=np.float32
        )

        # [1, 1250]
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
# CNN 1D
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

            nn.Linear(
                128,
                1
            )
        )

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        return x.squeeze(1)


# =========================================================
# CREAR DATASET
# =========================================================

dataset_completo = ECGDataset(
    RUTA_MAT,
    RUTA_INDICES,
    RUTA_ETIQUETAS
)

print("\n===== DATASET =====")
print(
    "Segmentos Train disponibles:",
    len(dataset_completo)
)


# =========================================================
# SUBCONJUNTO PARA PRUEBA
# =========================================================

# 200 batches x 64 = máximo 12,800 ECG
cantidad_prueba = min(
    MAX_BATCHES * BATCH_SIZE,
    len(dataset_completo)
)

rng = np.random.default_rng(42)

indices_prueba = rng.choice(
    len(dataset_completo),
    size=cantidad_prueba,
    replace=False
)

dataset_prueba = Subset(
    dataset_completo,
    indices_prueba
)

loader = DataLoader(
    dataset_prueba,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

print(
    "Segmentos usados en prueba:",
    len(dataset_prueba)
)


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
    lr=0.001
)


# =========================================================
# ENTRENAMIENTO DE PRUEBA
# =========================================================

print("\n===== ENTRENAMIENTO DE PRUEBA =====")

modelo.train()

loss_acumulado = 0.0

for batch_idx, (ecg, etiquetas) in enumerate(loader):

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

    loss_acumulado += loss.item()

    if (batch_idx + 1) % 20 == 0:

        loss_promedio = (
            loss_acumulado /
            (batch_idx + 1)
        )

        print(
            f"Batch {batch_idx + 1:3d}/{len(loader)}"
            f" | Loss promedio: {loss_promedio:.4f}"
            f" | Loss actual: {loss.item():.4f}"
        )


# =========================================================
# RESULTADO
# =========================================================

loss_final = (
    loss_acumulado /
    len(loader)
)

print("\n===== RESULTADO =====")

print(
    "Batches procesados:",
    len(loader)
)

print(
    "ECG procesados:",
    len(dataset_prueba)
)

print(
    "Loss promedio:",
    round(loss_final, 4)
)

if torch.cuda.is_available():

    memoria = (
        torch.cuda.max_memory_allocated()
        / 1024**2
    )

    print(
        "Memoria máxima GPU:",
        round(memoria, 2),
        "MB"
    )

print(
    "\nPrueba de entrenamiento real completada."
)