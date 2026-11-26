import torch
import torch.nn as nn

# =========================================================
# CONFIGURACIÓN
# =========================================================

BATCH_SIZE = 128
LONGITUD_ECG = 1250

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("========================================")
print("PRUEBA TÉCNICA CNN V2")
print("========================================")

print("Dispositivo:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# =========================================================
# BLOQUE RESIDUAL 1D
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

        # Si cambia el número de canales o el tamaño temporal,
        # adaptamos la conexión residual.
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

        # Entrada:
        # [batch, 1, 1250]

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
            32,
            32,
            kernel_size=7,
            stride=1
        )

        self.bloque2 = ResidualBlock1D(
            32,
            64,
            kernel_size=7,
            stride=2
        )

        self.bloque3 = ResidualBlock1D(
            64,
            128,
            kernel_size=5,
            stride=2
        )

        self.bloque4 = ResidualBlock1D(
            128,
            256,
            kernel_size=3,
            stride=2
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

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

        x = self.bloque1(x)
        x = self.bloque2(x)
        x = self.bloque3(x)
        x = self.bloque4(x)

        x = self.pool(x)

        x = self.clasificador(x)

        return x.squeeze(1)


# =========================================================
# CREAR MODELO
# =========================================================

modelo = CNNV2().to(device)

print("\n===== ARQUITECTURA =====")
print(modelo)


# =========================================================
# CONTAR PARÁMETROS
# =========================================================

parametros_totales = sum(
    p.numel()
    for p in modelo.parameters()
)

parametros_entrenables = sum(
    p.numel()
    for p in modelo.parameters()
    if p.requires_grad
)

print("\n===== PARÁMETROS =====")

print(
    "Parámetros totales:",
    f"{parametros_totales:,}"
)

print(
    "Parámetros entrenables:",
    f"{parametros_entrenables:,}"
)


# =========================================================
# CREAR BATCH SIMULADO
# =========================================================

x = torch.randn(
    BATCH_SIZE,
    1,
    LONGITUD_ECG,
    device=device
)

y = torch.randint(
    0,
    2,
    (BATCH_SIZE,),
    device=device
).float()

print("\n===== ENTRADA =====")

print(
    "Shape ECG:",
    x.shape
)

print(
    "Shape etiquetas:",
    y.shape
)


# =========================================================
# FORWARD
# =========================================================

if torch.cuda.is_available():

    torch.cuda.reset_peak_memory_stats()

logits = modelo(x)

print("\n===== SALIDA =====")

print(
    "Shape logits:",
    logits.shape
)

probabilidades = torch.sigmoid(
    logits
)

print(
    "Probabilidad mínima:",
    probabilidades.min().item()
)

print(
    "Probabilidad máxima:",
    probabilidades.max().item()
)


# =========================================================
# LOSS
# =========================================================

pos_weight = torch.tensor(
    [3.4682],
    dtype=torch.float32,
    device=device
)

criterion = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight
)

loss = criterion(
    logits,
    y
)

print("\n===== LOSS =====")

print(
    "Loss de prueba:",
    loss.item()
)


# =========================================================
# BACKPROPAGATION
# =========================================================

loss.backward()

print(
    "Forward + backward completado correctamente."
)


# =========================================================
# MEMORIA GPU
# =========================================================

if torch.cuda.is_available():

    memoria_max = (
        torch.cuda.max_memory_allocated()
        / 1024**2
    )

    print("\n===== GPU =====")

    print(
        f"Memoria GPU máxima usada: "
        f"{memoria_max:.2f} MB"
    )


print("\n========================================")
print("PRUEBA CNN V2 TERMINADA")
print("========================================")