"""
Carga del modelo y generación de resultados.

Todo lo que depende del modelo (arquitectura, checkpoint, umbral,
niveles, métricas) se lee desde config_modelo.json. Para cambiar
de modelo solo se edita ese archivo.
"""

import json
from pathlib import Path

import numpy as np
import torch

from .arquitecturas import ARQUITECTURAS
from .preprocesamiento import PREPROCESAMIENTOS, segmentar


MENSAJES = {
    "Bajo": (
        "Sin alerta: no se identificó un patrón asociado a hipertensión "
        "arterial en los segmentos analizados."
    ),
    "Moderado": (
        "Alerta intermedia de hipertensión arterial. Se sugiere repetir el "
        "análisis con un registro más largo o realizar un control de presión arterial."
    ),
    "Alto": (
        "Alerta de hipertensión arterial: se identificó un patrón asociado a "
        "hipertensión arterial en la señal ECG analizada. Se recomienda "
        "confirmar con un control de presión arterial."
    ),
}

AVISO = (
    "Este resultado es generado por un modelo computacional con fines de "
    "apoyo y no constituye un diagnóstico médico. Ante cualquier duda, "
    "se recomienda consultar a un profesional de la salud."
)


class Predictor:

    def __init__(self, ruta_config: Path):

        self.ruta_config = Path(ruta_config).resolve()

        with open(self.ruta_config, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        c = self.config

        if c["arquitectura"] not in ARQUITECTURAS:
            raise ValueError(
                f"Arquitectura desconocida en config: {c['arquitectura']}. "
                f"Opciones: {list(ARQUITECTURAS)}"
            )

        if c["preprocesamiento"] not in PREPROCESAMIENTOS:
            raise ValueError(
                f"Preprocesamiento desconocido: {c['preprocesamiento']}"
            )

        self.preprocesar = PREPROCESAMIENTOS[c["preprocesamiento"]]
        self.muestras_por_segmento = int(c["muestras_por_segmento"])
        self.max_segmentos = int(c.get("max_segmentos", 360))
        self.umbral = float(c["umbral_decision"])
        self.niveles = c["niveles"]

        # Ruta del checkpoint relativa a la ubicación del config.
        self.ruta_checkpoint = (
            self.ruta_config.parent / c["ruta_checkpoint"]
        ).resolve()

        if not self.ruta_checkpoint.exists():
            raise FileNotFoundError(
                f"No se encontró el modelo: {self.ruta_checkpoint}"
            )

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.modelo = ARQUITECTURAS[c["arquitectura"]]().to(self.device)

        # Carga segura: solo tensores y tipos básicos (no ejecuta código).
        checkpoint = torch.load(
            self.ruta_checkpoint,
            map_location=self.device,
            weights_only=True
        )

        if checkpoint.get("tipo") == "ensemble":
            # Paquete del script 37: pesos de varias CNN.
            self.modelo.cargar_paquete(checkpoint)
        else:
            self.modelo.load_state_dict(checkpoint["model_state_dict"])
        self.modelo.eval()

        self.epoca_checkpoint = checkpoint.get("epoch")

    # --------------------------------------------------------

    def nivel_de(self, probabilidad: float) -> str:

        for n in self.niveles:
            if n["desde"] <= probabilidad < n["hasta"]:
                return n["nivel"]

        # probabilidad == 1.0 cae en el último nivel
        return self.niveles[-1]["nivel"]

    # --------------------------------------------------------

    @torch.no_grad()
    def predecir_senal(self, senal: np.ndarray) -> dict:
        """
        Analiza una señal ECG completa:
        segmenta -> z-score -> CNN -> probabilidad por segmento -> promedio.
        """

        segmentos, descartadas = segmentar(
            senal, self.muestras_por_segmento, self.max_segmentos
        )

        x = self.preprocesar(segmentos)
        x = torch.from_numpy(x).unsqueeze(1).to(self.device)  # [N, 1, L]

        probs = []
        for inicio in range(0, len(x), 128):
            logits = self.modelo(x[inicio:inicio + 128])
            probs.append(torch.sigmoid(logits).cpu().numpy())

        probs = np.concatenate(probs).astype(float)

        # Agregación: promedio de las probabilidades de todos los segmentos.
        prob_final = float(probs.mean())
        nivel = self.nivel_de(prob_final)

        return {
            "probabilidad": round(prob_final, 4),
            "nivel": nivel,
            "supera_umbral": prob_final >= self.umbral,
            "umbral": self.umbral,
            "mensaje": MENSAJES[nivel],
            "aviso": AVISO,
            "segmentos": [
                {
                    "indice": i + 1,
                    "inicio_s": round(
                        i * self.muestras_por_segmento
                        / self.config["frecuencia_muestreo_hz"], 1
                    ),
                    "probabilidad": round(float(p), 4),
                    "nivel": self.nivel_de(float(p)),
                }
                for i, p in enumerate(probs)
            ],
            "n_segmentos": int(len(probs)),
            "muestras_descartadas": int(descartadas),
            "duracion_analizada_s": round(
                len(probs) * self.muestras_por_segmento
                / self.config["frecuencia_muestreo_hz"], 1
            ),
        }

    # --------------------------------------------------------

    def info(self) -> dict:
        """Información pública del modelo para la pantalla 'Modelo'."""

        c = self.config

        return {
            "nombre": c["nombre"],
            "version": c["version"],
            "arquitectura": c["arquitectura"],
            "descripcion": c.get("descripcion", ""),
            "entradas": c.get("entradas", []),
            "usa_variables_demograficas": c.get(
                "usa_variables_demograficas", False
            ),
            "criterio_etiqueta": c["criterio_etiqueta"],
            "frecuencia_muestreo_hz": c["frecuencia_muestreo_hz"],
            "muestras_por_segmento": self.muestras_por_segmento,
            "preprocesamiento": c["preprocesamiento"],
            "agregacion": c.get("agregacion", "promedio"),
            "umbral_decision": self.umbral,
            "niveles": self.niveles,
            "origen_umbrales": c.get("origen_umbrales", ""),
            "metricas": c.get("metricas", {}),
            "archivo_checkpoint": self.ruta_checkpoint.name,
            "epoca_checkpoint": self.epoca_checkpoint,
            "computo": "GPU (CUDA)" if self.device.type == "cuda" else "CPU",
            "estado": c.get("estado", ""),
        }
