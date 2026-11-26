import os
import json
import numpy as np
import torch

# ============================================================
# PUNTO 37 - EMPAQUETAR EL ENSEMBLE EN UN SOLO ARCHIVO
#
# El ensemble (punto 35) combina 3 CNN ya entrenadas. Este script
# NO entrena nada: reúne en un único archivo
#     modelos/mejor_modelo_ensemble_140_90.pt
# los pesos de las 3 redes, la regla de combinación, los umbrales
# congelados (Validation) y los resultados en CalFree (punto 36).
#
# Verifica que el paquete reproduzca exactamente las
# probabilidades guardadas en el punto 36.
# ============================================================

RAIZ = r"E:\Tesis_PulseDB"
CARPETA_MODELOS = os.path.join(RAIZ, "modelos")
RUTA_SALIDA = os.path.join(CARPETA_MODELOS, "mejor_modelo_ensemble_140_90.pt")

config = json.load(open(os.path.join(RAIZ, "resultados", "4_cnn_ensemble", "ensemble_35.json"), encoding="utf-8"))
final = json.load(open(os.path.join(RAIZ, "resultados", "4_cnn_ensemble", "evaluacion_final_ensemble_calfree_36.json"), encoding="utf-8"))

print("=" * 78)
print("PUNTO 37 - EMPAQUETAR EL ENSEMBLE")
print("=" * 78)

if os.path.exists(RUTA_SALIDA):
    print(f"AVISO: {RUTA_SALIDA} ya existe y será reemplazado (archivo propio del punto 37).")

paquete = {
    "tipo": "ensemble",
    "nombre": "Ensemble de CNN (V2 + V3 + multitarea)",
    "criterio": "SBP >= 140 OR DBP >= 90",
    "entrada": "ECG (1 derivación), 10 s a 125 Hz = 1250 muestras",
    "preprocesamiento": config["preprocesamiento"],
    "combinacion": config["combinacion"],
    "miembros": {},
    "umbral_youden": config["umbral_youden"]["umbral"],
    "umbral_sensibilidad_80": config["umbral_sensibilidad_80"]["umbral"],
    "auc_validacion": config["auc_val"],
    "auc_calfree": final["ensemble"]["roc_auc"],
    "auc_calfree_ic95": final["ensemble"]["ic95"],
    "nota": "Umbrales elegidos solo con Validation; CalFree evaluado una sola vez (punto 36).",
}

for nombre, archivo in config["modelos"].items():
    ckpt = torch.load(os.path.join(CARPETA_MODELOS, archivo), map_location="cpu", weights_only=False)
    paquete["miembros"][nombre] = {
        "archivo_origen": archivo,
        "model_state_dict": ckpt["model_state_dict"],
    }
    print(f"  incluido: {nombre:<11} <- {archivo}")

torch.save(paquete, RUTA_SALIDA)
print(f"\nGuardado: {RUTA_SALIDA} ({os.path.getsize(RUTA_SALIDA) / 1e6:.1f} MB)")

# ============================================================
# VERIFICACIÓN: el paquete reproduce las probabilidades del punto 36
# ============================================================

# Las arquitecturas se importan del MISMO módulo que usa la aplicación web
# (app_web/backend/modelo/arquitecturas.py), para garantizar que sean idénticas.
import sys
import h5py

sys.path.insert(0, os.path.join(RAIZ, "app_web"))
from backend.modelo.arquitecturas import EnsembleCNN

# Carga segura: solo tensores y tipos básicos.
cargado = torch.load(RUTA_SALIDA, map_location="cpu", weights_only=True)
ensemble = EnsembleCNN()
ensemble.cargar_paquete(cargado)
ensemble.eval()

with h5py.File(os.path.join(RAIZ, "data_calfree", "VitalDB_CalFree_Test_Subset.mat"), "r") as f:
    bloque = np.asarray(f["Subset"]["Signals"][:, 0, 0:400], dtype=np.float32).T

std = bloque.std(axis=1, keepdims=True)
std[std < 1e-6] = 1.0
x = torch.from_numpy((bloque - bloque.mean(axis=1, keepdims=True)) / std).unsqueeze(1)

with torch.no_grad():
    logit = ensemble(x).numpy()
prob = 1 / (1 + np.exp(-logit))

referencia = np.load(os.path.join(RAIZ, "resultados", "4_cnn_ensemble", "probabilidades_calfree_ensemble_36.npz"))["prob_ensemble"][:400]
diferencia = float(np.max(np.abs(prob - referencia)))

print(f"Verificación (400 segmentos de CalFree): diferencia máxima con el punto 36 = {diferencia:.2e}")
if diferencia > 1e-3:
    raise RuntimeError("El paquete no reproduce las predicciones del punto 36.")

print("\n" + "=" * 78)
print("PUNTO 37 FINALIZADO")
print("=" * 78)
print(f"AUC Validation: {cargado['auc_validacion']:.4f} | AUC CalFree: {cargado['auc_calfree']:.4f}")
print(f"Umbral Youden: {cargado['umbral_youden']:.4f} | Umbral sens >= 80 %: {cargado['umbral_sensibilidad_80']:.4f}")
print("\nIMPORTANTE: no se entrenó nada; los 3 archivos .pt originales no se modificaron.")
