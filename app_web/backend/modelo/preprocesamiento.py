"""
Lectura y preprocesamiento de la señal ECG.

Reproduce el mismo preprocesamiento usado en el entrenamiento:
- Segmentos de 1250 muestras (10 s a 125 Hz).
- Z-score por segmento (media 0, desviación 1).
"""

import re

import numpy as np


class ErrorSenal(ValueError):
    """Error de validación de la señal, con mensaje legible para el usuario."""


def leer_csv_ecg(contenido: bytes) -> np.ndarray:
    """
    Convierte el contenido de un archivo CSV/TXT en un vector de muestras.

    Acepta:
    - Una columna con un valor por línea.
    - Una fila con valores separados por coma, punto y coma, espacio o tab.
    - Encabezado opcional (por ejemplo "ecg"); el texto no numérico se ignora.
    - Varias columnas: se usa la primera columna numérica.
    """

    try:
        texto = contenido.decode("utf-8-sig")
    except UnicodeDecodeError:
        texto = contenido.decode("latin-1")

    lineas = [l.strip() for l in texto.splitlines() if l.strip()]

    if not lineas:
        raise ErrorSenal("El archivo está vacío.")

    separador = re.compile(r"[;,\t ]+")
    valores = []

    if len(lineas) == 1:
        # Todo en una sola fila.
        candidatos = separador.split(lineas[0])
    else:
        # Una muestra por línea (se toma la primera columna numérica).
        candidatos = []
        for linea in lineas:
            partes = separador.split(linea)
            numericas = [p for p in partes if _es_numero(p)]
            if numericas:
                candidatos.append(numericas[0])

    for c in candidatos:
        if _es_numero(c):
            valores.append(float(c))

    if not valores:
        raise ErrorSenal("No se encontraron valores numéricos en el archivo.")

    senal = np.asarray(valores, dtype=np.float32)

    if not np.all(np.isfinite(senal)):
        raise ErrorSenal("La señal contiene valores no válidos (NaN o infinito).")

    return senal


# Límite de muestras que se leen de un .mat (protege la memoria del servidor
# ante archivos comprimidos muy grandes). 2 700 000 muestras = 6 h a 125 Hz,
# igual que max_segmentos (2160) de config_modelo.json.
MAX_MUESTRAS_MAT = 2_700_000
# Elementos máximos de una variable .mat (≈ 300 MB en double).
MAX_ELEMENTOS_VARIABLE = 40_000_000
CABECERA_HDF5 = b"\x89HDF\r\n\x1a\n"


def leer_mat_ecg(contenido: bytes, muestras_por_segmento: int = 1250) -> np.ndarray:
    """
    Extrae la señal ECG de un archivo MATLAB (.mat).

    Acepta:
    - Formato v7.3 (HDF5) con la estructura de PulseDB: Subset/Signals,
      canal 0 = ECG. Los segmentos de 10 s se unen en orden.
    - Formato v7.3 de los archivos por sujeto de PulseDB (Subj_Wins/ECG_F).
    - Formatos v4/v5/v7 con una variable cuyo nombre contenga "ecg"
      o, si no la hay, con una única variable numérica.
    - Vector (1×N o N×1), matriz de segmentos (N×1250 o 1250×N) o
      matriz multicanal con pocas columnas (se usa el primer canal).
    """

    if contenido[:8] == CABECERA_HDF5 or contenido[512:520] == CABECERA_HDF5:
        datos = _leer_mat_hdf5(contenido, muestras_por_segmento)
    else:
        datos = _leer_mat_clasico(contenido)

    senal = _a_vector(np.asarray(datos), muestras_por_segmento)

    if senal.size == 0:
        raise ErrorSenal("La señal ECG del archivo .mat está vacía.")

    senal = senal[:MAX_MUESTRAS_MAT].astype(np.float32)

    if not np.all(np.isfinite(senal)):
        raise ErrorSenal("La señal contiene valores no válidos (NaN o infinito).")

    return senal


def _leer_mat_clasico(contenido: bytes) -> np.ndarray:

    import io
    import scipy.io

    try:
        # whosmat solo lee las cabeceras: permite rechazar variables
        # enormes antes de descomprimirlas.
        variables = scipy.io.whosmat(io.BytesIO(contenido))
    except Exception:
        raise ErrorSenal("El archivo .mat no es válido o está dañado.")

    numericas = [
        (nombre, forma) for nombre, forma, clase in variables
        if clase in ("double", "single", "int8", "int16", "int32", "int64",
                     "uint8", "uint16", "uint32", "uint64")
    ]
    elegida = _elegir_variable(numericas)

    if int(np.prod(elegida[1])) > MAX_ELEMENTOS_VARIABLE:
        raise ErrorSenal("La variable ECG del archivo .mat es demasiado grande.")

    try:
        mat = scipy.io.loadmat(io.BytesIO(contenido), variable_names=[elegida[0]])
    except Exception:
        raise ErrorSenal("No se pudo leer la variable ECG del archivo .mat.")

    return mat[elegida[0]]


def _leer_mat_hdf5(contenido: bytes, muestras_por_segmento: int) -> np.ndarray:

    import io
    import h5py

    try:
        f = h5py.File(io.BytesIO(contenido), "r")
    except Exception:
        raise ErrorSenal("El archivo .mat (v7.3) no es válido o está dañado.")

    with f:

        # Estructura de PulseDB: Signals [1250, 3, N] (canal 0 = ECG).
        if "Subset" in f and "Signals" in f["Subset"]:
            signals = f["Subset"]["Signals"]
            if signals.ndim != 3 or signals.shape[0] != muestras_por_segmento:
                raise ErrorSenal("La estructura Subset/Signals no tiene el formato de PulseDB.")
            n = min(signals.shape[2], MAX_MUESTRAS_MAT // muestras_por_segmento)
            return np.asarray(signals[:, 0, :n]).T          # [N, 1250]

        # Archivo por sujeto de PulseDB (Segment_Files): Subj_Wins/ECG_F,
        # una referencia por ventana de 10 s con el ECG ya filtrado.
        if "Subj_Wins" in f and "ECG_F" in f["Subj_Wins"]:
            refs = np.asarray(f["Subj_Wins"]["ECG_F"]).reshape(-1)
            n = min(len(refs), MAX_MUESTRAS_MAT // muestras_por_segmento)
            try:
                ventanas = [np.asarray(f[r], dtype=np.float32).reshape(-1) for r in refs[:n]]
            except Exception:
                raise ErrorSenal("No se pudieron leer las ventanas ECG_F del archivo .mat.")
            if any(len(v) != muestras_por_segmento for v in ventanas):
                raise ErrorSenal("Las ventanas ECG_F no tienen el formato de PulseDB (1250 muestras).")
            return np.stack(ventanas)                        # [N, 1250]

        numericas = [
            (nombre, d.shape) for nombre, d in f.items()
            if isinstance(d, h5py.Dataset) and d.dtype.kind in "fiu"
        ]
        elegida = _elegir_variable(numericas)

        if int(np.prod(elegida[1])) > MAX_ELEMENTOS_VARIABLE:
            raise ErrorSenal("La variable ECG del archivo .mat es demasiado grande.")

        # HDF5 guarda las matrices de MATLAB con las dimensiones invertidas.
        return np.asarray(f[elegida[0]]).T


def _elegir_variable(numericas: list):
    """Variable con 'ecg' en el nombre; si no la hay, la única numérica."""

    if not numericas:
        raise ErrorSenal("No se encontró una señal ECG en el archivo .mat.")

    con_ecg = [v for v in numericas if "ecg" in v[0].lower()]
    if con_ecg:
        return con_ecg[0]

    if len(numericas) == 1:
        return numericas[0]

    nombres = ", ".join(v[0] for v in numericas[:8])
    raise ErrorSenal(
        "El archivo .mat tiene varias variables y ninguna se llama 'ecg' "
        f"({nombres}). Renombre la variable de la señal ECG como 'ecg'."
    )


def _a_vector(datos: np.ndarray, muestras_por_segmento: int) -> np.ndarray:

    datos = np.squeeze(datos)

    if datos.ndim == 1:
        return datos

    if datos.ndim == 2:
        filas, columnas = datos.shape
        if columnas == muestras_por_segmento:
            return datos.reshape(-1)                         # N segmentos × 1250
        if filas == muestras_por_segmento:
            return datos.T.reshape(-1)                       # 1250 × N segmentos
        if columnas <= 16 < filas:
            return datos[:, 0]                               # muestras × canales
        if filas <= 16 < columnas:
            return datos[0, :]                               # canales × muestras

    raise ErrorSenal(
        f"No se reconoce la forma de la señal ECG en el .mat {datos.shape}. "
        "Se espera un vector o una matriz de segmentos de "
        f"{muestras_por_segmento} muestras."
    )


def leer_demograficos_mat(contenido: bytes) -> dict:
    """
    Edad, sexo, peso y talla guardados en un .mat de PulseDB (Subj_Wins o
    Subset), para completar el formulario. Lectura de mejor esfuerzo: si el
    archivo no los trae, devuelve un diccionario vacío y nunca lanza error.
    """

    import io
    import h5py

    if contenido[:8] != CABECERA_HDF5 and contenido[512:520] != CABECERA_HDF5:
        return {}

    datos = {}

    try:
        with h5py.File(io.BytesIO(contenido), "r") as f:

            if "Subj_Wins" in f:
                grupo, por_referencia = f["Subj_Wins"], True     # 1 ref por ventana
            elif "Subset" in f:
                grupo, por_referencia = f["Subset"], False       # 1 valor por segmento
            else:
                return {}

            def primero(campo):
                valor = np.asarray(grupo[campo]).reshape(-1)[0]
                return np.asarray(f[valor]).reshape(-1) if por_referencia else np.atleast_1d(valor)

            for campo, clave in (("Age", "edad"), ("Weight", "peso"), ("Height", "talla")):
                if campo in grupo:
                    valor = float(primero(campo)[0])
                    if np.isfinite(valor):
                        datos[clave] = round(valor, 1)

            if "Gender" in grupo:
                valor = np.asarray(grupo["Gender"]).reshape(-1)[0]
                codigos = np.asarray(f[valor]).reshape(-1)        # texto: 'F' o 'M'
                sexo = "".join(chr(int(c)) for c in codigos if int(c) != 0).strip().upper()
                if sexo in ("F", "M"):
                    datos["sexo"] = sexo
    except Exception:
        return datos

    return datos


def leer_senal(contenido: bytes, nombre_archivo: str = "") -> np.ndarray:
    """Lee la señal según la extensión: .mat (MATLAB) o .csv/.txt (texto)."""

    if nombre_archivo.lower().endswith(".mat"):
        return leer_mat_ecg(contenido)

    return leer_csv_ecg(contenido)


def _es_numero(texto: str) -> bool:

    try:
        float(texto)
        return True
    except ValueError:
        return False


def segmentar(senal: np.ndarray, muestras_por_segmento: int, max_segmentos: int):
    """
    Divide la señal en segmentos consecutivos sin solapamiento.

    Devuelve (segmentos [N, L], muestras_descartadas).
    Las muestras sobrantes al final (menos de un segmento) se descartan.
    """

    n_segmentos = len(senal) // muestras_por_segmento

    if n_segmentos == 0:
        raise ErrorSenal(
            f"La señal tiene {len(senal)} muestras; se necesitan al menos "
            f"{muestras_por_segmento} (un segmento completo)."
        )

    n_segmentos = min(n_segmentos, max_segmentos)
    usadas = n_segmentos * muestras_por_segmento

    segmentos = senal[:usadas].reshape(n_segmentos, muestras_por_segmento)

    return segmentos, len(senal) - usadas


def zscore_por_segmento(segmentos: np.ndarray) -> np.ndarray:
    """Z-score independiente para cada segmento (igual que en entrenamiento)."""

    media = segmentos.mean(axis=1, keepdims=True)
    std = segmentos.std(axis=1, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)

    return ((segmentos - media) / std).astype(np.float32)


PREPROCESAMIENTOS = {
    "zscore_por_segmento": zscore_por_segmento,
}
