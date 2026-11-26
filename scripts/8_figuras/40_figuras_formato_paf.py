import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

# ============================================================
# PUNTO 40 - FIGURAS EN EL FORMATO DEL PAF
#


RAIZ = r"E:\Tesis_PulseDB"
RESULTADOS = os.path.join(RAIZ, "resultados")
SALIDA = os.path.join(RAIZ, "figuras", "paf")
os.makedirs(SALIDA, exist_ok=True)

DPI = 300
plt.rcParams.update({"font.family": "Arial", "font.size": 11})


def dec(x, d):
    """Número con coma decimal."""
    return f"{x:.{d}f}".replace(".", ",")


def miles(n):
    """Entero con espacio como separador de miles."""
    return f"{int(n):,}".replace(",", " ")


def ejes_con_coma(ax, dx=1, dy=2):
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: dec(v, dx)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: dec(v, dy)))


config = json.load(open(os.path.join(RESULTADOS, "4_cnn_ensemble", "ensemble_35.json"), encoding="utf-8"))
U_YOUDEN = config["umbral_youden"]["umbral"]
U_SENS80 = config["umbral_sensibilidad_80"]["umbral"]

val = np.load(os.path.join(RESULTADOS, "4_cnn_ensemble", "predicciones_val_ensemble_35.npz"))
p_val = val["prob_ensemble"].astype(np.float64)
y_val = np.load(os.path.join(RAIZ, "procesado", "etiquetas_val_140_90.npy")).astype(int)

cal = np.load(os.path.join(RESULTADOS, "4_cnn_ensemble", "probabilidades_calfree_ensemble_36.npz"))
p_cal = cal["prob_ensemble"].astype(np.float64)
y_cal = cal["etiquetas"].astype(int)


def metricas(y, p, u):
    tn, fp, fn, tp = confusion_matrix(y, (p >= u).astype(int), labels=[0, 1]).ravel()
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp, "sens": tp / (tp + fn), "espec": tn / (tn + fp)}


# Comprobación: deben coincidir con la evaluación final (punto 36).
final = json.load(open(os.path.join(RESULTADOS, "4_cnn_ensemble", "evaluacion_final_ensemble_calfree_36.json"), encoding="utf-8"))
m_y = metricas(y_cal, p_cal, U_YOUDEN)
assert (m_y["tn"], m_y["fp"], m_y["fn"], m_y["tp"]) == tuple(final["umbrales"]["Youden"][k] for k in ("tn", "fp", "fn", "tp"))
assert abs(roc_auc_score(y_cal, p_cal) - final["ensemble"]["roc_auc"]) < 1e-6   # float32 al guardar

print("=" * 70)
print("PUNTO 40 - FIGURAS EN FORMATO PAF (300 dpi, coma decimal)")
print("=" * 70)

# ============================================================
# FIG. 6 - CURVAS DE ENTRENAMIENTO
# ============================================================

v2 = pd.read_csv(os.path.join(RESULTADOS, "1_cnn_residual", "historial_cnn_v2_regularizada_140_90.csv"))
v3 = pd.read_csv(os.path.join(RESULTADOS, "2_cnn_multiescala", "historial_cnn_v3_multiescala_140_90.csv"))
mt = pd.read_csv(os.path.join(RESULTADOS, "3_cnn_multitarea", "historial_cnn_multitarea_140_90.csv"))

fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
for d, nombre, color in [(v2, "CNN 1D residual", "#1d4ed8"), (v3, "CNN 1D multiescala", "#0e7490"),
                         (mt, "CNN 1D multitarea", "#6d28d9")]:
    a.plot(d.epoch, d.roc_auc, "-o", color=color, lw=2, ms=5, label=nombre)
    i = d.roc_auc.idxmax()
    a.plot(d.epoch[i], d.roc_auc[i], "*", color=color, ms=15, mec="black", mew=0.6)
a.axhline(config["auc_val"], color="#be123c", ls="--", lw=1.6, label=f"CNN-Ensemble ({dec(config['auc_val'], 4)})")
a.set_xlabel("Época")
a.set_ylabel("ROC-AUC (validación)")
a.set_title("(a) ROC-AUC en validación por época")
a.set_ylim(0.65, 0.76)
a.grid(alpha=0.3)
a.legend(fontsize=9, loc="lower right")

b.plot(v2.epoch, v2.train_loss, "-o", color="#1d4ed8", lw=2, ms=5, label="Entrenamiento")
b.plot(v2.epoch, v2.val_loss, "-s", color="#b45309", lw=2, ms=5, label="Validación")
mejor = v2.epoch[v2.roc_auc.idxmax()]
b.axvline(mejor, color="gray", ls=":", lw=1.4)
b.text(mejor + 0.1, max(v2.val_loss) * 0.99, "mejor época", fontsize=9, color="gray", va="top")
b.set_xlabel("Época")
b.set_ylabel("Pérdida (BCE ponderada)")
b.set_title("(b) Pérdida de la CNN 1D residual")
b.grid(alpha=0.3)
b.legend(fontsize=9)

for ax, dy in ((a, 2), (b, 1)):
    ax.set_xticks(range(1, 8))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _, dy=dy: dec(v, dy)))
fig.tight_layout()
fig.savefig(os.path.join(SALIDA, "fig6_curvas_entrenamiento.png"), dpi=DPI, bbox_inches="tight")
plt.close(fig)
print("Fig. 6 guardada")

# ============================================================
# FIG. 7 - MATRICES DE CONFUSIÓN EN CALFREE
# ============================================================

fig, ejes = plt.subplots(1, 2, figsize=(10, 4.4))
paneles = [("(a) Umbral de Youden", U_YOUDEN), ("(b) Umbral de sensibilidad ≥ 80 %", U_SENS80)]
for ax, (nombre, umbral) in zip(ejes, paneles):
    m = metricas(y_cal, p_cal, umbral)
    matriz = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    ax.imshow(matriz, cmap="Blues")
    for (r, c), v in np.ndenumerate(matriz):
        ax.text(c, r, f"{miles(v)}\n({dec(v / matriz.sum() * 100, 1)} %)", ha="center", va="center",
                color="white" if v > matriz.max() * 0.5 else "#111827", fontsize=10)
    ax.set_xticks([0, 1], ["Predicho: < 140/90", "Predicho: ≥ 140/90"])
    ax.set_yticks([0, 1], ["Real: < 140/90", "Real: ≥ 140/90"])
    ax.set_title(f"{nombre} = {dec(umbral, 4)}\nSens. {dec(m['sens'], 2)} · Espec. {dec(m['espec'], 2)}", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(SALIDA, "fig7_matrices_confusion_calfree.png"), dpi=DPI, bbox_inches="tight")
plt.close(fig)
print("Fig. 7 guardada")

# ============================================================
# FIG. 8 - CURVA ROC (VALIDACIÓN Y CALFREE)
# ============================================================

fig, ax = plt.subplots(figsize=(6.4, 6.0))
for etiqueta, yy, pp, color in [
    (f"Validación (AUC = {dec(roc_auc_score(y_val, p_val), 3)})", y_val, p_val, "#2563eb"),
    (f"CalFree – prueba final (AUC = {dec(roc_auc_score(y_cal, p_cal), 3)})", y_cal, p_cal, "#dc2626"),
]:
    fpr, tpr, _ = roc_curve(yy, pp)
    ax.plot(fpr, tpr, color=color, lw=2.2, label=etiqueta)

for texto, umbral, marcador in [(f"Umbral de Youden ({dec(U_YOUDEN, 4)})", U_YOUDEN, "o"),
                                (f"Umbral {dec(U_SENS80, 4)} (sens. ≥ 80 % en validación)", U_SENS80, "s")]:
    m = metricas(y_cal, p_cal, umbral)
    ax.scatter(1 - m["espec"], m["sens"], s=55, zorder=5, color="#111827", marker=marcador)
    ax.annotate(texto, (1 - m["espec"], m["sens"]), textcoords="offset points", xytext=(8, -12), fontsize=8.5)

ax.plot([0, 1], [0, 1], ls="--", color="#94a3b8", lw=1, label="Azar (AUC = 0,5)")
ax.set_xlabel("1 − Especificidad (tasa de falsos positivos)")
ax.set_ylabel("Sensibilidad (tasa de verdaderos positivos)")
ejes_con_coma(ax, 1, 1)
ax.legend(loc="lower right", fontsize=8.5)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(SALIDA, "fig8_curva_roc.png"), dpi=DPI, bbox_inches="tight")
plt.close(fig)
print("Fig. 8 guardada")

print(f"\nFiguras en: {SALIDA}")
print("Datos verificados contra el punto 36 (matriz de confusión y ROC-AUC idénticos).")
