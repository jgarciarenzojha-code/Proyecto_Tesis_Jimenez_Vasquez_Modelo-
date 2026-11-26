import torch
import torch.nn as nn


# ==============================
# CONFIGURACIÓN
# ==============================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("===== DISPOSITIVO =====")
print("Usando:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ==============================
# CNN 1D
# ==============================

class CNN1D(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv1d(
                in_channels=1,
                out_channels=32,
                kernel_size=7,
                padding=3
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(
                32,
                64,
                kernel_size=5,
                padding=2
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(
                64,
                128,
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

        # Dejamos logits:
        # NO aplicar sigmoid aquí
        return x.squeeze(1)


# ==============================
# CREAR MODELO
# ==============================

modelo = CNN1D().to(device)

print("\n===== MODELO =====")
print(modelo)


# ==============================
# LOSS CON DESBALANCE
# ==============================

pos_weight = torch.tensor(
    [3.4682],
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


# ==============================
# BATCH FICTICIO
# ==============================

X = torch.rand(
    64,
    1,
    1250,
    device=device
)

y = torch.randint(
    0,
    2,
    (64,),
    device=device
).float()


# ==============================
# UNA ITERACIÓN
# ==============================

optimizer.zero_grad()

logits = modelo(X)

loss = criterion(
    logits,
    y
)

loss.backward()

optimizer.step()


# ==============================
# RESULTADOS
# ==============================

probabilidades = torch.sigmoid(logits)

print("\n===== PRUEBA CNN =====")

print(
    "Forma entrada:",
    X.shape
)

print(
    "Forma salida:",
    logits.shape
)

print(
    "Loss:",
    loss.item()
)

print(
    "Probabilidad mínima:",
    probabilidades.min().item()
)

print(
    "Probabilidad máxima:",
    probabilidades.max().item()
)

print("\nForward + Backward completados.")

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