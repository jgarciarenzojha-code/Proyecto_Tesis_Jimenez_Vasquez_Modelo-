"""
Dispositivo IoT SIMULADO: reproduce un registro ECG como si fuera un equipo
de adquisición y envía al servidor un segmento de 10 s cada cierto tiempo
por HTTP (POST /api/dispositivo/segmento). Al terminar, cierra la sesión y el
resultado queda guardado en el historial del especialista asignado.

No usa ningún sensor físico: lee un archivo (.mat de PulseDB, .csv o .txt).
Sirve para validar la integración con la aplicación web; un dispositivo real
solo tendría que enviar los segmentos con el mismo formato.

Ejecutar desde la carpeta app_web (con la aplicación en marcha):

    ..\\.venv\\Scripts\\python.exe herramientas\\dispositivo_simulado.py ^
        --archivo ejemplos_mat\\registro_prueba_01.mat --token <TOKEN>

Opciones útiles:
    --intervalo 10        segundos entre envíos (10 = tiempo real; 1 = acelerado)
    --max-segmentos 30    detener después de N segmentos
    --codigo REG-001      código de registro al guardar la sesión
    --url http://127.0.0.1:8001
El token también puede pasarse con la variable de entorno TOKEN_DISPOSITIVO.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from backend.modelo.preprocesamiento import ErrorSenal, leer_senal  # noqa: E402

FRECUENCIA_HZ = 125
MUESTRAS_POR_SEGMENTO = 1250          # 10 s a 125 Hz


def enviar(url, token, datos):
    """POST JSON con el token del dispositivo. Devuelve el JSON de respuesta."""

    solicitud = urllib.request.Request(
        url,
        data=json.dumps(datos).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(solicitud, timeout=30) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            detalle = json.loads(e.read().decode("utf-8")).get("detail", "")
        except Exception:
            detalle = ""
        raise RuntimeError(f"El servidor respondió {e.code}: {detalle}") from None
    except urllib.error.URLError as e:
        raise RuntimeError(f"No se pudo conectar con el servidor ({e.reason}). ¿Está en marcha la aplicación?") from None


def main():

    parser = argparse.ArgumentParser(description="Dispositivo IoT simulado (envía segmentos ECG por HTTP)")
    parser.add_argument("--archivo", required=True, help="registro ECG: .mat (PulseDB), .csv o .txt")
    parser.add_argument("--token", default=os.environ.get("TOKEN_DISPOSITIVO"), help="token del dispositivo")
    parser.add_argument("--url", default="http://127.0.0.1:8001", help="dirección de la aplicación")
    parser.add_argument("--intervalo", type=float, default=10.0, help="segundos entre envíos")
    parser.add_argument("--max-segmentos", type=int, default=0, help="0 = todo el registro")
    parser.add_argument("--codigo", default=None, help="código de registro al guardar la sesión")
    args = parser.parse_args()

    if not args.token:
        sys.exit("Falta el token: use --token o la variable de entorno TOKEN_DISPOSITIVO.")

    ruta = Path(args.archivo)
    try:
        senal = leer_senal(ruta.read_bytes(), ruta.name)
    except (OSError, ErrorSenal) as e:
        sys.exit(f"No se pudo leer el registro: {e}")

    total = len(senal) // MUESTRAS_POR_SEGMENTO
    if args.max_segmentos > 0:
        total = min(total, args.max_segmentos)
    if total == 0:
        sys.exit("El registro tiene menos de 10 s de señal.")

    sesion = datetime.now().strftime("S%Y%m%d-%H%M%S")
    base = args.url.rstrip("/")

    print("=" * 72)
    print("DISPOSITIVO IoT SIMULADO · envío de segmentos ECG por HTTP")
    print("=" * 72)
    print(f"Registro: {ruta.name} | segmentos a enviar: {total} ({total * 10 // 60} min {total * 10 % 60} s)")
    print(f"Servidor: {base} | sesión: {sesion} | intervalo: {args.intervalo:g} s")
    print("-" * 72)

    enviados = 0
    try:
        for i in range(total):
            segmento = senal[i * MUESTRAS_POR_SEGMENTO:(i + 1) * MUESTRAS_POR_SEGMENTO]
            r = enviar(f"{base}/api/dispositivo/segmento", args.token, {
                "sesion": sesion,
                "indice": i,
                "muestras": [round(float(v), 5) for v in segmento],
                "frecuencia_hz": FRECUENCIA_HZ,
            })
            enviados += 1
            print(f"[{datetime.now():%H:%M:%S}] segmento {i + 1:>3}/{total} enviado → "
                  f"prob. {r['probabilidad']:.4f} ({r['nivel']:<8}) | promedio sesión {r['promedio_sesion']:.4f} "
                  f"({r['nivel_sesion']})", flush=True)
            if i < total - 1:
                time.sleep(args.intervalo)
    except KeyboardInterrupt:
        print("\nEnvío detenido por el usuario.")
    except RuntimeError as e:
        print(f"\nERROR: {e}")

    if enviados == 0:
        sys.exit("No se envió ningún segmento.")

    print("-" * 72)
    try:
        fin = enviar(f"{base}/api/dispositivo/finalizar", args.token, {"sesion": sesion, "codigo_registro": args.codigo})
        print(f"Sesión finalizada y guardada en el historial: {fin['codigo_registro']} (análisis #{fin['id']})")
        print(f"Resultado: {fin['n_segmentos']} segmentos · probabilidad promedio {fin['probabilidad']:.4f} · nivel {fin['nivel']}")
    except RuntimeError as e:
        print(f"No se pudo finalizar la sesión: {e}")


if __name__ == "__main__":
    main()
