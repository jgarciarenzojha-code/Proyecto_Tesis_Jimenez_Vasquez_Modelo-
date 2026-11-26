import os
import time
import json
import h5py
import numpy as np
import pandas as pd

from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score

# ============================================================
# PUNTO 32 - AUDITORÍA DE VARIABLES DEMOGRÁFICAS
#
# Objetivo: decidir qué variables demográficas pueden
# combinarse con la CNN (modelo híbrido) sin comprometer
# la validez del experimento.
#
# Este script SOLO LEE datos:
# - No entrena modelos.
# - No usa CalFree.
# - No modifica archivos existentes.
#
# Revisa:
# 1. Faltantes y valores fuera de rango.
# 2. Constancia de cada variable dentro de cada sujeto.
# 3. Distribución Train vs Validation.
# 4. Relación univariada con la etiqueta 140/90 (AUC).
# 5. Continuidad temporal entre segmentos consecutivos.
#
# Guarda (archivos NUEVOS):
# - procesado/demografia_32.npz
# - resultados/auditoria_demografica_32.csv
# - resultados/auditoria_demografica_32.json
# ============================================================

RUTA_MAT = (
    r"E:\Tesis_PulseDB\data_train"
    r"\VitalDB_Train_Subset.mat"
)

CARPETA_PROCESADO = r"E:\Tesis_PulseDB\procesado"
CARPETA_RESULTADOS = r"E:\Tesis_PulseDB\resultados"

RUTA_SALIDA_NPZ = os.path.join(CARPETA_PROCESADO, "demografia_32.npz")
RUTA_SALIDA_CSV = os.path.join(CARPETA_RESULTADOS, "5_experimentos", "demografia", "auditoria_demografica_32.csv")
RUTA_SALIDA_JSON = os.path.join(CARPETA_RESULTADOS, "5_experimentos", "demografia", "auditoria_demografica_32.json")

# Rangos plausibles en adultos (solo para detectar errores).
RANGOS = {
    "edad": (0, 110),
    "peso": (25, 250),
    "talla": (100, 220),
    "imc": (10, 70),
}

# Pares de segmentos consecutivos a revisar para continuidad.
PARES_CONTINUIDAD = 3000
SEED = 42

print("=" * 78)
print("PUNTO 32 - AUDITORÍA DE VARIABLES DEMOGRÁFICAS")
print("=" * 78)

for ruta in [RUTA_SALIDA_NPZ, RUTA_SALIDA_CSV, RUTA_SALIDA_JSON]:
    if os.path.exists(ruta):
        print(f"AVISO: {ruta} ya existe y será reemplazado (archivo propio del punto 32).")

# ============================================================
# ÍNDICES Y ETIQUETAS (división por sujeto ya existente)
# ============================================================

indices_train = np.load(os.path.join(CARPETA_PROCESADO, "indices_train.npy")).astype(np.int64)
indices_val = np.load(os.path.join(CARPETA_PROCESADO, "indices_val.npy")).astype(np.int64)

y_train = np.load(os.path.join(CARPETA_PROCESADO, "etiquetas_train_140_90.npy")).astype(int)
y_val = np.load(os.path.join(CARPETA_PROCESADO, "etiquetas_val_140_90.npy")).astype(int)

assert len(indices_train) == len(y_train)
assert len(indices_val) == len(y_val)


# ============================================================
# LECTURA DE VARIABLES
# ============================================================

def decodificar(f, ref):
    """Convierte una referencia HDF5 de MATLAB (char) en texto."""

    valores = np.asarray(f[ref]).reshape(-1)
    return "".join(chr(int(v)) for v in valores if int(v) != 0).strip()


def sujetos_por_bloques(f, refs_sujeto, refs_sexo, muestras_verificacion=3000):
    """
    Decodificar ~466 000 referencias una por una tarda más de una hora.
    Como los segmentos de cada sujeto están contiguos en el archivo, se
    localiza el final de cada bloque con búsqueda exponencial + binaria
    (pocas lecturas por sujeto) y el sexo se lee una vez por bloque.
    Después se VERIFICA con una muestra aleatoria decodificada directamente.
    """

    inicio = time.time()
    n = len(refs_sujeto)
    cache = {}

    def suj(i):
        if i not in cache:
            cache[i] = decodificar(f, refs_sujeto[i])
        return cache[i]

    sujeto = np.empty(n, dtype=object)
    sexo = np.empty(n, dtype=object)

    i = 0
    bloques = 0
    while i < n:
        actual = suj(i)

        # Búsqueda exponencial del primer índice con otro sujeto.
        paso = 1
        hi = i + 1
        while hi < n and suj(hi) == actual:
            paso *= 2
            hi = i + paso
        hi = min(hi, n)
        lo = i + paso // 2 if paso > 1 else i

        # Búsqueda binaria entre lo (mismo sujeto) y hi (otro o fin).
        while hi - lo > 1:
            medio = (lo + hi) // 2
            if suj(medio) == actual:
                lo = medio
            else:
                hi = medio

        sujeto[i:hi] = actual
        sexo[i:hi] = decodificar(f, refs_sexo[i])
        bloques += 1
        i = hi

        if bloques % 200 == 0:
            print(f"  bloques: {bloques:,} | segmentos: {i:,}/{n:,}")

    print(f"  {bloques:,} bloques en {time.time() - inicio:.1f} s "
          f"({len(cache):,} lecturas de sujeto)")

    # Verificación independiente con una muestra aleatoria.
    rng = np.random.default_rng(SEED)
    muestra = rng.choice(n, size=min(muestras_verificacion, n), replace=False)
    errores_sujeto = sum(decodificar(f, refs_sujeto[k]) != sujeto[k] for k in muestra)
    errores_sexo = sum(decodificar(f, refs_sexo[k]) != sexo[k] for k in muestra)

    print(f"  Verificación con {len(muestra):,} segmentos al azar: "
          f"errores sujeto = {errores_sujeto}, errores sexo = {errores_sexo}")

    if errores_sujeto or errores_sexo:
        raise RuntimeError("Los bloques por sujeto no son contiguos; no se puede usar este atajo.")

    return sujeto.astype(str), sexo.astype(str), bloques


print("\nLeyendo variables del archivo .mat ...")

with h5py.File(RUTA_MAT, "r") as f:

    s = f["Subset"]

    print(f"Campos disponibles: {list(s.keys())}")

    edad = np.asarray(s["Age"]).reshape(-1).astype(np.float64)
    peso = np.asarray(s["Weight"]).reshape(-1).astype(np.float64)
    talla = np.asarray(s["Height"]).reshape(-1).astype(np.float64)
    imc = np.asarray(s["BMI"]).reshape(-1).astype(np.float64)
    sbp = np.asarray(s["SBP"]).reshape(-1).astype(np.float64)
    dbp = np.asarray(s["DBP"]).reshape(-1).astype(np.float64)

    refs_sexo = np.asarray(s["Gender"]).reshape(-1)
    refs_sujeto = np.asarray(s["Subject"]).reshape(-1)

    print("\nIdentificando sujeto y sexo de cada segmento ...")
    sujeto, sexo_txt, n_bloques = sujetos_por_bloques(f, refs_sujeto, refs_sexo)

    total = len(edad)

    # --------------------------------------------------------
    # Continuidad temporal: salto en el borde entre segmentos
    # consecutivos del mismo sujeto vs. variación interna.
    # --------------------------------------------------------

    print("\nRevisando continuidad entre segmentos consecutivos ...")

    rng = np.random.default_rng(SEED)
    candidatos = indices_train[:-1]
    candidatos = candidatos[np.isin(candidatos + 1, indices_train)]
    candidatos = candidatos[sujeto[candidatos] == sujeto[candidatos + 1]]
    muestra = np.sort(rng.choice(candidatos, size=min(PARES_CONTINUIDAD, len(candidatos)), replace=False))

    saltos_borde = []
    saltos_internos = []

    signals = s["Signals"]

    for i in muestra:
        a = np.asarray(signals[:, 0, int(i)], dtype=np.float64)
        b = np.asarray(signals[:, 0, int(i) + 1], dtype=np.float64)

        # Diferencia típica entre muestras vecinas dentro del segmento.
        tipico = np.median(np.abs(np.diff(a))) + 1e-9

        saltos_borde.append(abs(b[0] - a[-1]) / tipico)
        saltos_internos.append(np.percentile(np.abs(np.diff(a)), 95) / tipico)

saltos_borde = np.asarray(saltos_borde)
saltos_internos = np.asarray(saltos_internos)

print(f"Segmentos totales en el archivo: {total:,}")

# ============================================================
# SEXO -> 0/1
# ============================================================

valores_sexo = sorted(set(sexo_txt))
print(f"\nValores de sexo encontrados: {valores_sexo}")

mapa_sexo = {}
for v in valores_sexo:
    letra = v[:1].upper()
    if letra == "M":
        mapa_sexo[v] = 1.0       # masculino
    elif letra == "F":
        mapa_sexo[v] = 0.0       # femenino
    else:
        mapa_sexo[v] = np.nan    # desconocido

sexo = np.array([mapa_sexo[v] for v in sexo_txt], dtype=np.float64)

VARIABLES = {
    "edad": edad,
    "sexo_masculino": sexo,
    "peso": peso,
    "talla": talla,
    "imc": imc,
}

# IMC recalculado para verificar consistencia con peso y talla.
imc_calculado = peso / (talla / 100.0) ** 2

filas = []
reporte = {"segmentos_totales": int(total)}

# ============================================================
# 1. FALTANTES Y RANGOS
# ============================================================

print("\n" + "=" * 78)
print("1. FALTANTES Y VALORES FUERA DE RANGO")
print("=" * 78)

for nombre, datos in VARIABLES.items():

    for split, idx in [("train", indices_train), ("val", indices_val)]:

        v = datos[idx]
        faltantes = int(np.sum(~np.isfinite(v)))
        validos = v[np.isfinite(v)]

        fuera = 0
        clave_rango = nombre if nombre in RANGOS else None
        if clave_rango:
            lo, hi = RANGOS[clave_rango]
            fuera = int(np.sum((validos < lo) | (validos > hi)))

        fila = {
            "variable": nombre,
            "split": split,
            "segmentos": len(v),
            "faltantes": faltantes,
            "pct_faltantes": faltantes / len(v) * 100,
            "fuera_de_rango": fuera,
            "min": float(validos.min()) if len(validos) else np.nan,
            "max": float(validos.max()) if len(validos) else np.nan,
            "media": float(validos.mean()) if len(validos) else np.nan,
            "std": float(validos.std()) if len(validos) else np.nan,
            "mediana": float(np.median(validos)) if len(validos) else np.nan,
        }
        filas.append(fila)

        print(
            f"{nombre:<15} {split:<5} faltantes={faltantes:>6,} "
            f"({fila['pct_faltantes']:.2f} %)  fuera_rango={fuera:>6,}  "
            f"min={fila['min']:.1f}  max={fila['max']:.1f}  "
            f"media={fila['media']:.2f}  std={fila['std']:.2f}"
        )

dif_imc = np.abs(imc - imc_calculado)
dif_imc = dif_imc[np.isfinite(dif_imc)]
print(
    f"\nConsistencia IMC vs peso/talla²: diferencia mediana = {np.median(dif_imc):.3f}, "
    f"percentil 99 = {np.percentile(dif_imc, 99):.3f}"
)
reporte["imc_diferencia_mediana"] = float(np.median(dif_imc))
reporte["imc_diferencia_p99"] = float(np.percentile(dif_imc, 99))

# Menores de 18 años
for split, idx in [("train", indices_train), ("val", indices_val)]:
    menores = edad[idx] < 18
    suj_menores = np.unique(sujeto[idx][menores])
    print(
        f"Menores de 18 en {split}: {menores.sum():,} segmentos "
        f"({menores.mean() * 100:.2f} %), {len(suj_menores)} sujetos"
    )
    reporte[f"menores18_segmentos_{split}"] = int(menores.sum())
    reporte[f"menores18_sujetos_{split}"] = int(len(suj_menores))

# ============================================================
# 2. CONSTANCIA DENTRO DE CADA SUJETO
# ============================================================

print("\n" + "=" * 78)
print("2. ¿LA VARIABLE ES CONSTANTE DENTRO DE CADA SUJETO?")
print("=" * 78)

df_todo = pd.DataFrame({k: v for k, v in VARIABLES.items()})
df_todo["sujeto"] = sujeto

idx_usados = np.concatenate([indices_train, indices_val])
grupos = df_todo.iloc[idx_usados].groupby("sujeto")

n_sujetos = grupos.ngroups
print(f"Sujetos (train + val): {n_sujetos:,}")

reporte["constancia"] = {}

for nombre in VARIABLES:
    unicos = grupos[nombre].nunique(dropna=True)
    variables = int((unicos > 1).sum())
    print(f"{nombre:<15} sujetos con más de un valor: {variables:,} de {n_sujetos:,}")
    reporte["constancia"][nombre] = variables

# ============================================================
# 3. DISTRIBUCIÓN TRAIN vs VALIDATION (a nivel de SUJETO)
# ============================================================

print("\n" + "=" * 78)
print("3. DISTRIBUCIÓN TRAIN vs VALIDATION (un valor por sujeto)")
print("=" * 78)

sujetos_train = np.unique(sujeto[indices_train])
sujetos_val = np.unique(sujeto[indices_val])

compartidos = set(sujetos_train) & set(sujetos_val)
print(f"Sujetos train: {len(sujetos_train):,} | val: {len(sujetos_val):,} | compartidos: {len(compartidos)}")
reporte["sujetos_train"] = int(len(sujetos_train))
reporte["sujetos_val"] = int(len(sujetos_val))
reporte["sujetos_compartidos"] = int(len(compartidos))

por_sujeto = df_todo.iloc[idx_usados].groupby("sujeto").first()

reporte["ks"] = {}

for nombre in VARIABLES:
    a = por_sujeto.loc[por_sujeto.index.isin(sujetos_train), nombre].dropna()
    b = por_sujeto.loc[por_sujeto.index.isin(sujetos_val), nombre].dropna()

    if nombre == "sexo_masculino":
        print(f"{nombre:<15} % masculino  train={a.mean() * 100:.1f} %  val={b.mean() * 100:.1f} %")
        reporte["ks"][nombre] = {"train_pct": float(a.mean() * 100), "val_pct": float(b.mean() * 100)}
        continue

    estadistico, p = ks_2samp(a, b)
    print(
        f"{nombre:<15} train media={a.mean():.2f}  val media={b.mean():.2f}  "
        f"KS={estadistico:.3f}  p={p:.3f}"
    )
    reporte["ks"][nombre] = {"ks": float(estadistico), "p": float(p),
                             "media_train": float(a.mean()), "media_val": float(b.mean())}

# ============================================================
# 4. RELACIÓN UNIVARIADA CON LA ETIQUETA 140/90
# ============================================================

print("\n" + "=" * 78)
print("4. RELACIÓN DE CADA VARIABLE CON LA ETIQUETA 140/90")
print("   (AUC de la variable sola; 0.5 = sin relación)")
print("=" * 78)

reporte["auc_univariado"] = {}

for nombre, datos in VARIABLES.items():

    resultado = {}

    for split, idx, y in [("train", indices_train, y_train), ("val", indices_val, y_val)]:
        v = datos[idx]
        ok = np.isfinite(v)
        auc = roc_auc_score(y[ok], v[ok])
        resultado[split] = auc

    direccion = "mayor valor → más positivos" if resultado["train"] >= 0.5 else "menor valor → más positivos"
    fuerza = max(resultado["val"], 1 - resultado["val"])

    print(
        f"{nombre:<15} AUC train={resultado['train']:.4f}  AUC val={resultado['val']:.4f}  "
        f"(capacidad en val: {fuerza:.4f}; {direccion})"
    )
    reporte["auc_univariado"][nombre] = {
        "train": float(resultado["train"]), "val": float(resultado["val"]),
        "capacidad_val": float(fuerza),
    }

print("\nProporción de positivos 140/90 por rango de edad (train):")

bins = [0, 18, 30, 45, 60, 75, 120]
etiquetas_bins = ["<18", "18-29", "30-44", "45-59", "60-74", "75+"]
rangos_edad = pd.cut(edad[indices_train], bins=bins, labels=etiquetas_bins, right=False)
tabla_edad = (
    pd.DataFrame({"rango": rangos_edad, "y": y_train})
    .groupby("rango", observed=False)["y"]
    .agg(["count", "mean"])
)
for rango, fila in tabla_edad.iterrows():
    print(f"  {rango:<6} segmentos={int(fila['count']):>8,}  positivos={fila['mean'] * 100:5.1f} %")

reporte["positivos_por_edad_train"] = {
    str(k): {"segmentos": int(v["count"]), "pct_positivos": float(v["mean"] * 100)}
    for k, v in tabla_edad.iterrows()
}

print("\nProporción de positivos 140/90 por sexo (train):")
for valor, etiqueta in [(0.0, "Femenino"), (1.0, "Masculino")]:
    m = sexo[indices_train] == valor
    print(f"  {etiqueta:<10} segmentos={m.sum():>8,}  positivos={y_train[m].mean() * 100:5.1f} %")

# ============================================================
# 5. CONTINUIDAD TEMPORAL ENTRE SEGMENTOS CONSECUTIVOS
# ============================================================

print("\n" + "=" * 78)
print("5. ¿LOS SEGMENTOS CONSECUTIVOS SON CONTINUOS EN EL TIEMPO?")
print("=" * 78)

# Si la grabación fuera continua, el salto en el borde (último punto
# del segmento i -> primer punto del segmento i+1) sería del mismo
# orden que los saltos internos entre muestras vecinas.
proporcion_borde_normal = float(np.mean(saltos_borde <= saltos_internos))

print(f"Pares revisados (mismo sujeto, índices consecutivos): {len(saltos_borde):,}")
print(f"Salto en el borde (mediana, en unidades típicas): {np.median(saltos_borde):.2f}")
print(f"Salto interno p95 (mediana, en unidades típicas):  {np.median(saltos_internos):.2f}")
print(f"Pares cuyo borde parece continuo: {proporcion_borde_normal * 100:.1f} %")

if proporcion_borde_normal >= 0.8:
    conclusion = (
        "La mayoría de segmentos consecutivos parecen continuos: se puede "
        "describir la agregación como un registro continuo (verificar en el informe)."
    )
else:
    conclusion = (
        "Los segmentos consecutivos NO parecen continuos: la agregación debe "
        "describirse como 'varios segmentos del mismo sujeto', no como un "
        "registro continuo de 60 s."
    )

print(f"Conclusión: {conclusion}")

reporte["continuidad"] = {
    "pares": int(len(saltos_borde)),
    "salto_borde_mediana": float(np.median(saltos_borde)),
    "salto_interno_p95_mediana": float(np.median(saltos_internos)),
    "pct_bordes_continuos": proporcion_borde_normal * 100,
    "conclusion": conclusion,
}

# ============================================================
# GUARDAR
# ============================================================

np.savez_compressed(
    RUTA_SALIDA_NPZ,
    # Variables para TODO el archivo de Train (indexar con indices_train / indices_val)
    edad=edad.astype(np.float32),
    sexo_masculino=sexo.astype(np.float32),
    peso=peso.astype(np.float32),
    talla=talla.astype(np.float32),
    imc=imc.astype(np.float32),
    sbp=sbp.astype(np.float32),
    dbp=dbp.astype(np.float32),
    sujeto=sujeto,
)

pd.DataFrame(filas).to_csv(RUTA_SALIDA_CSV, index=False)

with open(RUTA_SALIDA_JSON, "w", encoding="utf-8") as fjson:
    json.dump(reporte, fjson, ensure_ascii=False, indent=2)

print("\n" + "=" * 78)
print("PUNTO 32 FINALIZADO")
print("=" * 78)
print(f"Variables guardadas en: {RUTA_SALIDA_NPZ}")
print(f"Resumen CSV:            {RUTA_SALIDA_CSV}")
print(f"Resumen JSON:           {RUTA_SALIDA_JSON}")
print(
    "\nIMPORTANTE:"
    "\n- No se entrenó ningún modelo."
    "\n- CalFree NO fue utilizado."
    "\n- Ningún archivo existente fue modificado."
)
