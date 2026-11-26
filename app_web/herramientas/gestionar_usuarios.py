"""
Gestión de usuarios desde la terminal (útil para crear el primer administrador
o recuperar el acceso). Normalmente los especialistas se crean desde /admin.

Ejecutar desde la carpeta app_web:

    # Listar usuarios
    ..\\.venv\\Scripts\\python.exe herramientas\\gestionar_usuarios.py listar

    # Crear usuario (pide la contraseña sin mostrarla en pantalla)
    ..\\.venv\\Scripts\\python.exe herramientas\\gestionar_usuarios.py crear renzo "Renzo Jiménez" administrador

    # Crear usuario con contraseña generada automáticamente
    ..\\.venv\\Scripts\\python.exe herramientas\\gestionar_usuarios.py crear esp01 "Especialista 01" especialista --generar

    # Cambiar la contraseña de un usuario existente
    ..\\.venv\\Scripts\\python.exe herramientas\\gestionar_usuarios.py contrasena renzo

La contraseña nunca se guarda en texto plano: solo su hash scrypt.
"""

import argparse
import getpass
import os
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from backend import seguridad as seg              # noqa: E402
from backend.basedatos import ROLES, BaseDatos     # noqa: E402

RUTA_DB = RAIZ / "datos" / "historial.db"


def leer_contrasena(generar: bool) -> str:

    if generar:
        return seg.generar_contrasena()

    # Permite pasarla por variable de entorno (sin escribirla en archivos).
    desde_entorno = os.environ.pop("CONTRASENA_NUEVA", None)
    if desde_entorno:
        return desde_entorno

    primera = getpass.getpass("Contraseña: ")
    segunda = getpass.getpass("Repita la contraseña: ")
    if primera != segunda:
        sys.exit("Las contraseñas no coinciden.")
    return primera


def main():

    parser = argparse.ArgumentParser(description="Gestión de usuarios de la aplicación web")
    sub = parser.add_subparsers(dest="comando", required=True)

    sub.add_parser("listar")

    p_crear = sub.add_parser("crear")
    p_crear.add_argument("usuario")
    p_crear.add_argument("nombre")
    p_crear.add_argument("rol", choices=ROLES)
    p_crear.add_argument("--generar", action="store_true", help="generar contraseña aleatoria")

    p_clave = sub.add_parser("contrasena")
    p_clave.add_argument("usuario")
    p_clave.add_argument("--generar", action="store_true")

    args = parser.parse_args()
    bd = BaseDatos(RUTA_DB)

    if args.comando == "listar":
        for u in bd.listar_usuarios():
            estado = "activo" if u["activo"] else "inactivo"
            print(f"{u['id']:>3}  {u['usuario']:<16} {u['rol']:<14} {estado:<9} {u['nombre']}")
        return

    if args.comando == "crear":

        if bd.usuario_por_nombre(args.usuario):
            sys.exit(f"El usuario '{args.usuario}' ya existe.")

        contrasena = leer_contrasena(args.generar)
        error = seg.validar_contrasena_nueva(contrasena)
        if error:
            sys.exit(error)

        bd.crear_usuario(args.usuario, args.nombre, args.rol, seg.hash_contrasena(contrasena))
        bd.auditar("usuario_creado", None, f"{args.usuario} ({args.rol}) desde terminal")

        print(f"Usuario '{args.usuario}' creado con rol {args.rol}.")
        if args.generar:
            print(f"Contraseña generada (guárdela ahora, no se volverá a mostrar): {contrasena}")
        return

    if args.comando == "contrasena":

        usuario = bd.usuario_por_nombre(args.usuario)
        if usuario is None:
            sys.exit(f"El usuario '{args.usuario}' no existe.")

        contrasena = leer_contrasena(args.generar)
        error = seg.validar_contrasena_nueva(contrasena)
        if error:
            sys.exit(error)

        bd.actualizar_usuario(usuario["id"], hash_contrasena=seg.hash_contrasena(contrasena))
        bd.cerrar_sesiones_de_usuario(usuario["id"])
        bd.auditar("contrasena_restablecida", None, f"{args.usuario} desde terminal")

        print(f"Contraseña de '{args.usuario}' actualizada.")
        if args.generar:
            print(f"Nueva contraseña: {contrasena}")


if __name__ == "__main__":
    main()
