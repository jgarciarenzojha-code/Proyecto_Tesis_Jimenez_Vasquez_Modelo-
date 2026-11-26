"""
Gestión de dispositivos IoT (simulados) desde la terminal.

Cada dispositivo tiene un código, un nombre, un especialista asignado (que lo
verá en la pestaña "Monitoreo") y un token secreto con el que envía los
segmentos ECG. El token se muestra UNA sola vez al crearlo o renovarlo; en la
base de datos solo se guarda su hash.

Ejecutar desde la carpeta app_web:

    # Listar dispositivos
    ..\\.venv\\Scripts\\python.exe herramientas\\gestionar_dispositivos.py listar

    # Registrar un dispositivo asignado a un especialista
    ..\\.venv\\Scripts\\python.exe herramientas\\gestionar_dispositivos.py crear ECG-SIM-01 "Dispositivo ECG simulado 01" especialista01

    # Generar un token nuevo (el anterior deja de funcionar)
    ..\\.venv\\Scripts\\python.exe herramientas\\gestionar_dispositivos.py renovar ECG-SIM-01

    # Desactivar / reactivar
    ..\\.venv\\Scripts\\python.exe herramientas\\gestionar_dispositivos.py desactivar ECG-SIM-01
    ..\\.venv\\Scripts\\python.exe herramientas\\gestionar_dispositivos.py activar ECG-SIM-01
"""

import argparse
import re
import sqlite3
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from backend import seguridad as seg              # noqa: E402
from backend.basedatos import BaseDatos            # noqa: E402

RUTA_DB = RAIZ / "datos" / "historial.db"
PATRON_CODIGO = re.compile(r"^[A-Za-z0-9_-]{3,30}$")


def mostrar_token(codigo, token):
    print(f"\nToken del dispositivo {codigo} (guárdelo; no se volverá a mostrar):\n\n    {token}\n")
    print("Úselo con: herramientas\\dispositivo_simulado.py --token <token> ...")


def main():

    parser = argparse.ArgumentParser(description="Gestión de dispositivos IoT simulados")
    sub = parser.add_subparsers(dest="comando", required=True)

    sub.add_parser("listar")

    p = sub.add_parser("crear")
    p.add_argument("codigo")
    p.add_argument("nombre")
    p.add_argument("usuario", help="usuario del especialista que verá el dispositivo")

    for comando in ("renovar", "desactivar", "activar"):
        sub.add_parser(comando).add_argument("codigo")

    args = parser.parse_args()
    bd = BaseDatos(RUTA_DB)

    if args.comando == "listar":
        dispositivos = bd.listar_dispositivos()
        if not dispositivos:
            print("No hay dispositivos registrados.")
        for d in dispositivos:
            estado = "activo" if d["activo"] else "inactivo"
            print(f"{d['codigo']:<14} {d['nombre']:<32} {estado:<9} "
                  f"asignado a: {d['especialista'] or '—':<22} último contacto: {d['ultimo_contacto'] or '—'}")

    elif args.comando == "crear":
        if not PATRON_CODIGO.match(args.codigo):
            sys.exit("El código solo admite letras, números, '-' y '_' (3 a 30 caracteres).")
        usuario = bd.usuario_por_nombre(args.usuario)
        if usuario is None:
            sys.exit(f"No existe el usuario '{args.usuario}'.")
        token = seg.nuevo_token()
        try:
            bd.crear_dispositivo(args.codigo, args.nombre, seg.hash_token(token), usuario["id"])
        except sqlite3.IntegrityError:
            sys.exit(f"Ya existe un dispositivo con el código '{args.codigo}'.")
        bd.auditar("dispositivo_creado", None, f"{args.codigo} → {usuario['usuario']}")
        print(f"Dispositivo {args.codigo} registrado y asignado a {usuario['nombre']}.")
        mostrar_token(args.codigo, token)

    elif args.comando == "renovar":
        token = seg.nuevo_token()
        if not bd.actualizar_dispositivo(args.codigo, hash_token=seg.hash_token(token)):
            sys.exit(f"No existe el dispositivo '{args.codigo}'.")
        bd.auditar("dispositivo_token_renovado", None, args.codigo)
        mostrar_token(args.codigo, token)

    else:
        activo = 1 if args.comando == "activar" else 0
        if not bd.actualizar_dispositivo(args.codigo, activo=activo):
            sys.exit(f"No existe el dispositivo '{args.codigo}'.")
        bd.auditar(f"dispositivo_{args.comando}do", None, args.codigo)
        print(f"Dispositivo {args.codigo} {'activado' if activo else 'desactivado'}.")


if __name__ == "__main__":
    main()
