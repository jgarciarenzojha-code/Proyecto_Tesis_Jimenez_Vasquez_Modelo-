import os
import time
import h5py
import numpy as np

# ============================================================
# PUNTO 38 - VERIFICACIÓN DEL DATASET PulseDB
#
# Solo lectura: NO modifica ni crea archivos.
#
# Comprueba:
# 1. Sujetos y segmentos de Train, Validation y CalFree.
# 2. Que Train y Validation no compartan índices ni sujetos.
# 3. Que los índices estén dentro del rango.
# 4. Etiquetas 140/90 (y que coincidan con las guardadas).
# 5. Calidad de TODAS las señales ECG (NaN/Inf, señal plana).
# 6. Que ningún sujeto de Train/Validation aparezca en CalFree.
#
# Basado en la propuesta de revisión externa, con lectura rápida
# de sujetos (bloques contiguos + verificación aleatoria).
# ============================================================

RAIZ = r"E:\Tesis_PulseDB"
RUTA_TRAIN = os.path.join(RAIZ, "data_train", "VitalDB_Train_Subset.mat")
RUTA_CALFREE = os.path.join(RAIZ, "data_calfree", "VitalDB_CalFree_Test_Subset.mat")
PROCESADO = os.path.join(RAIZ, "procesado")

RUTA_INDICES_TRAIN = os.path.join(PROCESADO, "indices_train.npy")
RUTA_INDICES_VAL = os.path.join(PROCESADO, "indices_val.npy")
RUTA_LABELS_TRAIN = os.path.join(PROCESADO, "etiquetas_train_140_90.npy")
RUTA_LABELS_VAL = os.path.join(PROCESADO, "etiquetas_val_140_90.npy")

SEED = 42
MUESTRA_VERIFICACION = 3000
BLOQUE_LECTURA = 4000


def cargar_npy(ruta):
    if not os.path.isfile(ruta):
        raise FileNotFoundError(f"No existe: {ruta}")
    return np.load(ruta, allow_pickle=False)


def decodificar(f, ref):
    datos = np.asarray(f[ref]).reshape(-1)
    return "".join(chr(int(x)) for x in datos if int(x) != 0).strip()


def sujetos_de_archivo(f, nombre):
    """
    Sujeto de CADA segmento del archivo.
    Los segmentos de cada sujeto están contiguos: se localiza el final
    de cada bloque con búsqueda binaria (pocas lecturas) y luego se
    VERIFICA con una muestra aleatoria decodificada directamente.
    """

    refs = np.asarray(f["Subset"]["Subject"]).reshape(-1)
    n = len(refs)
    cache = {}

    def suj(i):
        if i not in cache:
            cache[i] = decodificar(f, refs[i])
        return cache[i]

    inicio = time.time()
    sujetos = np.empty(n, dtype=object)
    i = 0

    while i < n:
        actual, paso, hi = suj(i), 1, i + 1
        while hi < n and suj(hi) == actual:
            paso *= 2
            hi = i + paso
        hi = min(hi, n)
        lo = i + paso // 2 if paso > 1 else i
        while hi - lo > 1:
            medio = (lo + hi) // 2
            if suj(medio) == actual:
                lo = medio
            else:
                hi = medio
        sujetos[i:hi] = actual
        i = hi

    muestra = np.random.default_rng(SEED).choice(n, size=min(MUESTRA_VERIFICACION, n), replace=False)
    errores = sum(decodificar(f, refs[k]) != sujetos[k] for k in muestra)

    print(f"  {nombre}: sujetos identificados en {time.time() - inicio:.1f} s; "
          f"verificación con {len(muestra):,} segmentos al azar: {errores} errores")

    if errores:
        raise RuntimeError(f"{nombre}: los sujetos no están en bloques contiguos.")

    return sujetos.astype(str)


def comprobar_indices(indices, total, nombre):
    if indices.ndim != 1 or len(indices) == 0:
        raise ValueError(f"{nombre}: índices vacíos o con forma incorrecta")
    if not np.issubdtype(indices.dtype, np.integer):
        raise ValueError(f"{nombre}: índices no enteros")
    if indices.min() < 0 or indices.max() >= total:
        raise ValueError(f"{nombre}: índices fuera de rango")

    repetidos = len(indices) - len(np.unique(indices))
    print(f"{nombre}: {len(indices):,} segmentos | índice mín {indices.min():,} | "
          f"máx {indices.max():,} | repetidos {repetidos:,}")
    return repetidos


def resumen_etiquetas(y, nombre):
    positivos = int(np.sum(y == 1))
    negativos = int(np.sum(y == 0))
    print(f"  {nombre:<11} negativos (0): {negativos:>8,} | positivos (1): {positivos:>7,} | "
          f"prevalencia positiva: {100 * positivos / len(y):.2f} %")


def verificar_etiquetas(sbp, dbp, indices, ruta_labels, nombre):
    calculadas = ((sbp[indices] >= 140) | (dbp[indices] >= 90)).astype(np.int8)
    resumen_etiquetas(calculadas, nombre)

    guardadas = cargar_npy(ruta_labels).reshape(-1)
    if len(guardadas) != len(calculadas):
        print(f"  ERROR {nombre}: longitud de etiquetas diferente")
        return False

    diferencias = int(np.sum(guardadas.astype(np.int8) != calculadas))
    print(f"  {nombre:<11} diferencias con etiquetas guardadas: {diferencias:,}")
    return diferencias == 0


def verificar_senales(f, nombre):
    """Revisa el canal ECG (0) de TODOS los segmentos del archivo."""

    signals = f["Subset"]["Signals"]
    n = signals.shape[2]
    no_finitos, planos, std_min = 0, 0, np.inf
    inicio = time.time()

    for a in range(0, n, BLOQUE_LECTURA):
        b = min(a + BLOQUE_LECTURA, n)
        ecg = np.asarray(signals[:, 0, a:b], dtype=np.float64)       # [1250, bloque]
        finitos = np.isfinite(ecg).all(axis=0)
        no_finitos += int((~finitos).sum())
        std = ecg[:, finitos].std(axis=0)
        planos += int((std < 1e-6).sum())
        if std.size:
            std_min = min(std_min, float(std.min()))

    print(f"  {nombre}: forma de Signals {signals.shape} (muestras, canales, segmentos)")
    print(f"  {nombre}: revisados {n:,} segmentos en {time.time() - inicio:.0f} s | "
          f"con NaN/Inf: {no_finitos} | planos (std < 1e-6): {planos} | std mínima: {std_min:.4f}")
    return no_finitos, planos


def main():

    print("=" * 74)
    print("PUNTO 38 - VERIFICACIÓN DEL DATASET PulseDB (solo lectura)")
    print("=" * 74)

    indices_train = cargar_npy(RUTA_INDICES_TRAIN)
    indices_val = cargar_npy(RUTA_INDICES_VAL)

    # ========================================================
    # TRAIN SUBSET (entrenamiento + validación)
    # ========================================================

    with h5py.File(RUTA_TRAIN, "r") as f:

        s = f["Subset"]
        sbp = np.asarray(s["SBP"]).reshape(-1)
        dbp = np.asarray(s["DBP"]).reshape(-1)
        total = len(sbp)

        print(f"\n[1] VitalDB_Train_Subset: {total:,} segmentos")
        comprobar_indices(indices_train, total, "TRAIN")
        comprobar_indices(indices_val, total, "VALIDATION")

        print("\n[2] ÍNDICES")
        compartidos_idx = np.intersect1d(indices_train, indices_val)
        union_idx = np.union1d(indices_train, indices_val)
        print(f"Índices compartidos Train/Validation: {len(compartidos_idx):,}")
        print(f"Índices utilizados: {len(union_idx):,} | sin asignar: {total - len(union_idx):,}")

        print("\n[3] SUJETOS")
        sujeto = sujetos_de_archivo(f, "Train Subset")
        sujetos_train = set(sujeto[indices_train])
        sujetos_val = set(sujeto[indices_val])
        compartidos = sujetos_train & sujetos_val

        seg_por_sujeto = np.unique(np.unique(sujeto, return_counts=True)[1])
        print(f"Sujetos TRAIN: {len(sujetos_train):,} | VALIDATION: {len(sujetos_val):,} | "
              f"compartidos: {len(compartidos):,}")
        print(f"Segmentos por sujeto (valores distintos): {seg_por_sujeto.tolist()}")
        print("OK: Train y Validation no comparten sujetos" if not compartidos
              else f"ALERTA: sujetos compartidos, ej.: {sorted(compartidos)[:10]}")

        print("\n[4] ETIQUETAS 140/90")
        ok_train = verificar_etiquetas(sbp, dbp, indices_train, RUTA_LABELS_TRAIN, "TRAIN")
        ok_val = verificar_etiquetas(sbp, dbp, indices_val, RUTA_LABELS_VAL, "VALIDATION")

        print("\n[5] CALIDAD DE SEÑALES ECG (todos los segmentos)")
        verificar_senales(f, "Train Subset")

    # ========================================================
    # CALFREE (prueba independiente)
    # ========================================================

    with h5py.File(RUTA_CALFREE, "r") as f:

        s = f["Subset"]
        sbp_c = np.asarray(s["SBP"]).reshape(-1)
        dbp_c = np.asarray(s["DBP"]).reshape(-1)
        total_c = len(sbp_c)

        print(f"\n[6] VitalDB_CalFree_Test_Subset: {total_c:,} segmentos")
        sujeto_c = sujetos_de_archivo(f, "CalFree")
        sujetos_calfree = set(sujeto_c)
        seg_c = np.unique(np.unique(sujeto_c, return_counts=True)[1])
        print(f"Sujetos CalFree: {len(sujetos_calfree):,} | segmentos por sujeto: {seg_c.tolist()}")
        resumen_etiquetas(((sbp_c >= 140) | (dbp_c >= 90)).astype(int), "CALFREE")
        verificar_senales(f, "CalFree")

    # ========================================================
    # INDEPENDENCIA ENTRE CONJUNTOS
    # ========================================================

    print("\n[7] SUPERPOSICIÓN CON CALFREE (por identificador de sujeto)")
    train_c = sujetos_train & sujetos_calfree
    val_c = sujetos_val & sujetos_calfree
    print(f"TRAIN ∩ CALFREE: {len(train_c)} | VALIDATION ∩ CALFREE: {len(val_c)}")
    print("OK: no se encontraron identificadores compartidos" if not (train_c or val_c)
          else "ALERTA: identificadores de sujeto compartidos")
    print("Nota: esto compara identificadores; personas repetidas con distinto ID no serían detectadas.")

    print("\n[8] RESUMEN FINAL")
    print(f"Sujetos únicos: {len(sujetos_train | sujetos_val | sujetos_calfree):,}")
    print(f"Segmentos utilizados: {len(indices_train) + len(indices_val) + total_c:,}")
    print(f"Train/Validation sin sujetos compartidos: {not compartidos}")
    print(f"Etiquetas TRAIN correctas: {ok_train} | Etiquetas VALIDATION correctas: {ok_val}")

    print("\n" + "=" * 74)
    print("VERIFICACIÓN FINALIZADA (ningún archivo fue modificado)")
    print("=" * 74)


if __name__ == "__main__":
    main()
