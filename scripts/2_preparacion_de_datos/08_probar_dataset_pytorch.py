import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

RUTA_MAT = r"E:\Tesis_PulseDB\data_train\VitalDB_Train_Subset.mat"
RUTA_INDICES = r"E:\Tesis_PulseDB\procesado\indices_train.npy"
RUTA_ETIQUETAS = r"E:\Tesis_PulseDB\procesado\etiquetas_train.npy"


class ECGDataset(Dataset):

    def __init__(self, ruta_mat, ruta_indices, ruta_etiquetas):
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

            self.signals = self.archivo["Subset"]["Signals"]

    def __getitem__(self, idx):

        self._abrir_archivo()

        indice_real = self.indices[idx]

        # Canal 0 = ECG
        ecg = self.signals[:, 0, indice_real]

        # Convertir a float32
        ecg = np.asarray(
            ecg,
            dtype=np.float32
        )

        # Forma para Conv1D:
        # [canales, longitud]
        ecg = np.expand_dims(ecg, axis=0)

        etiqueta = np.float32(
            self.etiquetas[idx]
        )

        return (
            torch.from_numpy(ecg),
            torch.tensor(etiqueta)
        )


dataset = ECGDataset(
    RUTA_MAT,
    RUTA_INDICES,
    RUTA_ETIQUETAS
)

print("===== DATASET =====")
print("Total segmentos:", len(dataset))


# Probar un segmento
ecg, etiqueta = dataset[0]

print("\n===== PRIMER SEGMENTO =====")
print("Forma ECG:", ecg.shape)
print("Tipo ECG:", ecg.dtype)
print("Etiqueta:", etiqueta.item())

print("Mínimo:", ecg.min().item())
print("Máximo:", ecg.max().item())
print("Media:", ecg.mean().item())


# Probar DataLoader
loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=0
)

ecg_batch, etiquetas_batch = next(iter(loader))

print("\n===== PRIMER BATCH =====")
print("Forma ECG:", ecg_batch.shape)
print("Forma etiquetas:", etiquetas_batch.shape)

print(
    "Positivos en batch:",
    int(etiquetas_batch.sum().item())
)

print(
    "Negativos en batch:",
    int(
        len(etiquetas_batch)
        - etiquetas_batch.sum().item()
    )
)

print("\nDataset PyTorch funcionando correctamente.")