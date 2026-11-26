"""
Seguridad: contraseñas, sesiones y límite de intentos.

- Contraseñas: hash scrypt con sal aleatoria (hashlib, sin dependencias).
  En la base de datos nunca se guarda la contraseña en texto plano.
- Sesiones: token aleatorio en una cookie HttpOnly; en la base de datos
  solo se guarda el SHA-256 del token.
- Inicio de sesión: bloqueo temporal tras varios intentos fallidos.
"""

import base64
import hashlib
import hmac
import secrets
import time


# ------------------------------------------------------------
# Contraseñas
# ------------------------------------------------------------

SCRYPT_N, SCRYPT_R, SCRYPT_P, SCRYPT_LEN = 2 ** 14, 8, 1, 32
LONGITUD_MINIMA = 8


def hash_contrasena(contrasena: str) -> str:

    sal = secrets.token_bytes(16)
    clave = hashlib.scrypt(
        contrasena.encode("utf-8"), salt=sal,
        n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P, dklen=SCRYPT_LEN,
    )
    b64 = lambda b: base64.b64encode(b).decode("ascii")

    return f"scrypt${SCRYPT_N}${SCRYPT_R}${SCRYPT_P}${b64(sal)}${b64(clave)}"


def verificar_contrasena(contrasena: str, almacenado: str) -> bool:

    try:
        _, n, r, p, sal, clave = almacenado.split("$")
        calculada = hashlib.scrypt(
            contrasena.encode("utf-8"), salt=base64.b64decode(sal),
            n=int(n), r=int(r), p=int(p), dklen=len(base64.b64decode(clave)),
        )
        return hmac.compare_digest(calculada, base64.b64decode(clave))
    except Exception:
        return False


def validar_contrasena_nueva(contrasena: str):
    """Devuelve un mensaje de error o None si la contraseña es aceptable."""

    if len(contrasena) < LONGITUD_MINIMA:
        return f"La contraseña debe tener al menos {LONGITUD_MINIMA} caracteres."
    if not any(c.isalpha() for c in contrasena) or not any(c.isdigit() for c in contrasena):
        return "La contraseña debe combinar letras y números."
    return None


def generar_contrasena(longitud: int = 12) -> str:
    """Contraseña aleatoria legible (sin caracteres confusos como 0/O, 1/l)."""

    letras = "abcdefghjkmnpqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ"
    digitos = "23456789"
    simbolos = "#@$%*"

    while True:
        clave = [secrets.choice(letras + digitos) for _ in range(longitud - 2)]
        clave += [secrets.choice(digitos), secrets.choice(simbolos)]
        secrets.SystemRandom().shuffle(clave)
        clave = "".join(clave)
        if any(c.isupper() for c in clave) and any(c.islower() for c in clave):
            return clave


# ------------------------------------------------------------
# Sesiones
# ------------------------------------------------------------

NOMBRE_COOKIE = "sesion_ecg"
HORAS_SESION = 8


def nuevo_token() -> str:
    return secrets.token_urlsafe(32)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


# ------------------------------------------------------------
# Límite de intentos de inicio de sesión (en memoria)
# ------------------------------------------------------------

MAX_INTENTOS = 5
BLOQUEO_SEGUNDOS = 5 * 60

_intentos = {}   # usuario -> (fallos, bloqueado_hasta)


def segundos_bloqueado(usuario: str) -> int:

    fallos, hasta = _intentos.get(usuario.lower(), (0, 0))
    restante = int(hasta - time.time())
    return restante if restante > 0 else 0


def registrar_fallo(usuario: str):

    clave = usuario.lower()
    fallos, _ = _intentos.get(clave, (0, 0))
    fallos += 1

    if fallos >= MAX_INTENTOS:
        _intentos[clave] = (0, time.time() + BLOQUEO_SEGUNDOS)
    else:
        _intentos[clave] = (fallos, 0)


def limpiar_intentos(usuario: str):
    _intentos.pop(usuario.lower(), None)
