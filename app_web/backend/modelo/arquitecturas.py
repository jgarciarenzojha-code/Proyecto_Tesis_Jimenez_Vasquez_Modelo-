"""
Arquitecturas CNN usadas en la tesis.

Son copias EXACTAS de las clases definidas en los scripts de
entrenamiento (los nombres de capas deben coincidir con los
checkpoints .pt para que load_state_dict funcione):

- cnn_v2         -> scripts/23 y 26 (CNN V2 residual regularizada)
- cnn_v3         -> scripts/30      (CNN V3 multiescala)
- cnn_multitarea -> scripts/31      (CNN multitarea: clase + SBP + DBP)

Para agregar un modelo nuevo (por ejemplo, ECG + variables
demográficas), se define su clase aquí y se registra en
ARQUITECTURAS. La aplicación web no necesita otros cambios.
"""

import torch
import torch.nn as nn


# ============================================================
# BLOQUE RESIDUAL (común a V2, V3 y multitarea)
# ============================================================

class BloqueResidual(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1):

        super().__init__()

        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            stride=1, padding=padding, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels, out_channels, 1,
                    stride=stride, bias=False
                ),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):

        identidad = self.shortcut(x)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        return self.relu(x + identidad)


# ============================================================
# CNN V2 (scripts 23 / 26)
# ============================================================

class CNNV2(nn.Module):

    def __init__(self):

        super().__init__()

        self.entrada = nn.Sequential(
            nn.Conv1d(1, 32, 15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        self.block1 = BloqueResidual(32, 32, 7, 1)
        self.block2 = BloqueResidual(32, 64, 7, 2)
        self.block3 = BloqueResidual(64, 128, 5, 2)
        self.block4 = BloqueResidual(128, 256, 3, 2)

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

        return self.clasificador(x).squeeze(1)


# ============================================================
# CNN V3 MULTIESCALA (script 30)
# ============================================================

class BloqueMultiEscala(nn.Module):

    def __init__(self, in_channels, branch_channels=32):

        super().__init__()

        def rama(kernel):
            return nn.Sequential(
                nn.Conv1d(
                    in_channels, branch_channels, kernel_size=kernel,
                    padding=kernel // 2, bias=False
                ),
                nn.BatchNorm1d(branch_channels),
                nn.ReLU()
            )

        self.branch_3 = rama(3)
        self.branch_7 = rama(7)
        self.branch_15 = rama(15)
        self.branch_31 = rama(31)

    def forward(self, x):

        return torch.cat(
            [
                self.branch_3(x),
                self.branch_7(x),
                self.branch_15(x),
                self.branch_31(x)
            ],
            dim=1
        )


class CNNV3MultiEscala(nn.Module):

    def __init__(self):

        super().__init__()

        self.entry = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        self.multiscale = BloqueMultiEscala(32, branch_channels=32)

        self.res1 = BloqueResidual(128, 128, kernel_size=7, stride=2)
        self.res2 = BloqueResidual(128, 256, kernel_size=5, stride=2)
        self.res3 = BloqueResidual(256, 256, kernel_size=3, stride=2)

        self.pool = nn.AdaptiveAvgPool1d(4)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.40),
            nn.Linear(256 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(128, 1)
        )

    def forward(self, x):

        x = self.entry(x)
        x = self.multiscale(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.pool(x)

        return self.classifier(x).squeeze(1)


# ============================================================
# CNN MULTITAREA (script 31)
# Solo se usa la salida de clasificación en la aplicación.
# ============================================================

class CNNMultiTarea(nn.Module):

    def __init__(self):

        super().__init__()

        self.entry = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        self.block1 = BloqueResidual(32, 32, kernel_size=7, stride=1)
        self.block2 = BloqueResidual(32, 64, kernel_size=7, stride=2)
        self.block3 = BloqueResidual(64, 128, kernel_size=5, stride=2)
        self.block4 = BloqueResidual(128, 256, kernel_size=3, stride=2)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.40),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.30)
        )

        self.head_class = nn.Linear(64, 1)
        self.head_sbp = nn.Linear(64, 1)
        self.head_dbp = nn.Linear(64, 1)

    def forward(self, x):

        x = self.entry(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool(x)

        features = self.shared(x)

        # Para la aplicación solo interesa el logit de clasificación.
        return self.head_class(features).squeeze(1)


# ============================================================
# ENSEMBLE (scripts 35-37)
# Promedio de los logits de V2, V3 y multitarea (pesos iguales).
# Se carga desde modelos/mejor_modelo_ensemble_140_90.pt
# ============================================================

class EnsembleCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.miembros = nn.ModuleDict({
            "v2": CNNV2(),
            "v3": CNNV3MultiEscala(),
            "multitarea": CNNMultiTarea(),
        })

    def cargar_paquete(self, paquete):
        """Carga los pesos de cada miembro desde el archivo del script 37."""

        for nombre, datos in paquete["miembros"].items():
            self.miembros[nombre].load_state_dict(datos["model_state_dict"])

    def forward(self, x):

        logits = torch.stack([m(x) for m in self.miembros.values()], dim=0)
        return logits.mean(dim=0)


# ============================================================
# REGISTRO DE ARQUITECTURAS
# El nombre es el que se escribe en config_modelo.json
# ============================================================

ARQUITECTURAS = {
    "cnn_v2": CNNV2,
    "cnn_v3": CNNV3MultiEscala,
    "cnn_multitarea": CNNMultiTarea,
    "ensemble": EnsembleCNN,
}
