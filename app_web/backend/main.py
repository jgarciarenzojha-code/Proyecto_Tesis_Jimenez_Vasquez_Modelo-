"""
API y páginas de la aplicación web.

Páginas (mismo servidor, rutas distintas):
    /login  -> inicio de sesión
    /app    -> panel del especialista (análisis de señales ECG)
    /admin  -> dashboard del administrador (solo rol administrador)

Flujo de análisis: el especialista sube un archivo ECG (o elige un registro
de ejemplo) -> el servidor segmenta y normaliza la señal -> la CNN estima la
probabilidad por segmento -> se promedia -> se asigna un nivel de alerta.

Ejecutar desde la carpeta app_web:
    ..\\.venv\\Scripts\\python.exe -m uvicorn backend.main:app --port 8001
"""

import csv
import io
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import seguridad as seg
from .basedatos import ROLES, BaseDatos
from .modelo.predictor import Predictor
from .modelo.preprocesamiento import ErrorSenal, leer_demograficos_mat, leer_senal


RAIZ = Path(__file__).resolve().parent.parent          # app_web/
RUTA_CONFIG = RAIZ / "config_modelo.json"
CARPETA_PAGINAS = RAIZ / "frontend"
CARPETA_STATIC = RAIZ / "frontend" / "static"
CARPETA_EJEMPLOS = RAIZ / "ejemplos"
RUTA_DB = RAIZ / "datos" / "historial.db"

TAMANO_MAX_ARCHIVO = 300 * 1024 * 1024                 # 300 MB
SEGMENTOS_EN_RESPUESTA = 12                             # para la gráfica
CABECERA_ORIGEN = "x-solicitud"                         # protección CSRF
VALOR_ORIGEN = "app-ecg"

PATRON_USUARIO = re.compile(r"^[A-Za-z0-9._-]{3,30}$")
PATRON_CODIGO = re.compile(r"^[A-Za-z0-9 _-]{1,30}$")


predictor = Predictor(RUTA_CONFIG)
bd = BaseDatos(RUTA_DB)

# Usuario de demostración: se crea SOLO si la base de datos no tiene ningún
# usuario (primera vez que se levanta la aplicación en otra computadora).
# Usuario: especialista. Solo se guarda el hash scrypt de la contraseña;
# la contraseña está indicada en el README y conviene cambiarla.
USUARIO_DEMO = "especialista"
HASH_CONTRASENA_DEMO = "scrypt$16384$8$1$BMuZ7WiUkjeqF1pS16hIUQ==$vfjIQimfW9meWyn3Ka+w7KFsgPSrl3puIjGzWA4JR0Y="

if not bd.listar_usuarios():
    bd.crear_usuario(USUARIO_DEMO, "Especialista de demostración", "especialista", HASH_CONTRASENA_DEMO)
    bd.auditar("usuario_demo_creado", None, USUARIO_DEMO)

app = FastAPI(
    title="Sistema de alerta temprana - ECG",
    description=(
        "API para la clasificación de señales ECG mediante una CNN "
        "(criterio de hipertensión arterial 140/90)."
    ),
    version="2.0.0",
)


# ============================================================
# SESIÓN Y PERMISOS
# ============================================================

def usuario_actual(request: Request):
    """Usuario de la sesión actual o None."""

    token = request.cookies.get(seg.NOMBRE_COOKIE)
    if not token:
        return None
    return bd.usuario_de_sesion(seg.hash_token(token))


def requiere_usuario(request: Request):

    usuario = usuario_actual(request)
    if usuario is None:
        raise HTTPException(401, "Sesión no iniciada o expirada.")
    return usuario


def requiere_admin(usuario=Depends(requiere_usuario)):

    if usuario["rol"] != "administrador":
        raise HTTPException(403, "Acceso solo para administradores.")
    return usuario


def verificar_origen(request: Request):
    """
    Protección CSRF para operaciones que modifican datos: la página
    envía una cabecera propia que un formulario de otro sitio no puede enviar.
    """

    if request.headers.get(CABECERA_ORIGEN) != VALOR_ORIGEN:
        raise HTTPException(403, "Solicitud no permitida.")


def publico(usuario):
    """Datos del usuario que se pueden enviar al navegador."""

    return {
        "id": usuario["id"],
        "usuario": usuario["usuario"],
        "nombre": usuario["nombre"],
        "rol": usuario["rol"],
    }


def destino_por_rol(usuario):
    return "/admin" if usuario["rol"] == "administrador" else "/app"


# ============================================================
# PÁGINAS
# ============================================================

def pagina(nombre):

    respuesta = FileResponse(CARPETA_PAGINAS / nombre)
    respuesta.headers["Cache-Control"] = "no-store"
    return respuesta


@app.get("/")
def inicio(request: Request):

    usuario = usuario_actual(request)
    return RedirectResponse(destino_por_rol(usuario) if usuario else "/login", 303)


@app.get("/login")
def pagina_login(request: Request):

    usuario = usuario_actual(request)
    if usuario:
        return RedirectResponse(destino_por_rol(usuario), 303)
    return pagina("login.html")


@app.get("/app")
def pagina_app(request: Request):

    if usuario_actual(request) is None:
        return RedirectResponse("/login", 303)
    return pagina("app.html")


@app.get("/admin")
def pagina_admin(request: Request):

    usuario = usuario_actual(request)
    if usuario is None:
        return RedirectResponse("/login", 303)
    if usuario["rol"] != "administrador":
        return RedirectResponse("/app", 303)
    return pagina("admin.html")


# ============================================================
# AUTENTICACIÓN
# ============================================================

class DatosLogin(BaseModel):
    usuario: str
    contrasena: str


class DatosCambioContrasena(BaseModel):
    actual: str
    nueva: str


@app.post("/api/auth/login", dependencies=[Depends(verificar_origen)])
def login(datos: DatosLogin, response: Response):

    nombre = datos.usuario.strip()

    espera = seg.segundos_bloqueado(nombre)
    if espera:
        raise HTTPException(
            429, f"Demasiados intentos fallidos. Intente de nuevo en {espera // 60 + 1} min."
        )

    usuario = bd.usuario_por_nombre(nombre) if nombre else None

    valido = (
        usuario is not None
        and usuario["activo"] == 1
        and seg.verificar_contrasena(datos.contrasena, usuario["hash_contrasena"])
    )

    if not valido:
        seg.registrar_fallo(nombre)
        bd.auditar("inicio_sesion_fallido", None, f"usuario: {nombre[:40]}")
        # Mensaje genérico: no revela si el usuario existe.
        raise HTTPException(401, "Usuario o contraseña incorrectos.")

    seg.limpiar_intentos(nombre)

    token = seg.nuevo_token()
    bd.crear_sesion(seg.hash_token(token), usuario["id"], seg.HORAS_SESION)
    bd.actualizar_usuario(usuario["id"], ultimo_acceso=datetime.now().isoformat(timespec="seconds"))
    bd.auditar("inicio_sesion", usuario)

    response.set_cookie(
        seg.NOMBRE_COOKIE,
        token,
        max_age=seg.HORAS_SESION * 3600,
        httponly=True,
        samesite="lax",
        path="/",
    )

    return {"usuario": publico(usuario), "destino": destino_por_rol(usuario)}


@app.post("/api/auth/logout", dependencies=[Depends(verificar_origen)])
def logout(request: Request, response: Response):

    token = request.cookies.get(seg.NOMBRE_COOKIE)
    usuario = usuario_actual(request)

    if token:
        bd.cerrar_sesion(seg.hash_token(token))
    if usuario:
        bd.auditar("cierre_sesion", usuario)

    response.delete_cookie(seg.NOMBRE_COOKIE, path="/")
    return {"ok": True}


@app.get("/api/auth/yo")
def yo(usuario=Depends(requiere_usuario)):
    return publico(usuario)


@app.post("/api/auth/cambiar-contrasena", dependencies=[Depends(verificar_origen)])
def cambiar_contrasena(datos: DatosCambioContrasena, usuario=Depends(requiere_usuario)):

    if not seg.verificar_contrasena(datos.actual, usuario["hash_contrasena"]):
        raise HTTPException(400, "La contraseña actual no es correcta.")

    error = seg.validar_contrasena_nueva(datos.nueva)
    if error:
        raise HTTPException(400, error)

    bd.actualizar_usuario(usuario["id"], hash_contrasena=seg.hash_contrasena(datos.nueva))
    bd.auditar("cambio_contrasena", usuario)
    return {"ok": True}


# ============================================================
# ANÁLISIS (especialista y administrador)
# ============================================================

def cargar_indice_ejemplos():

    ruta = CARPETA_EJEMPLOS / "ejemplos.json"
    if not ruta.exists():
        return []
    with open(ruta, "r", encoding="utf-8") as f:
        return json.load(f)


def _numero_opcional(valor, nombre, minimo, maximo):

    if valor is None or str(valor).strip() == "":
        return None

    try:
        numero = float(str(valor).replace(",", "."))
    except ValueError:
        raise HTTPException(400, f"El campo '{nombre}' debe ser numérico.")

    if not (minimo <= numero <= maximo):
        raise HTTPException(400, f"El campo '{nombre}' debe estar entre {minimo} y {maximo}.")

    return numero


def validar_demograficos(edad, sexo, peso, talla):

    datos = {
        "edad": _numero_opcional(edad, "edad", 0, 120),
        "peso": _numero_opcional(peso, "peso (kg)", 2, 350),
        "talla": _numero_opcional(talla, "talla (cm)", 40, 250),
        "sexo": None,
        "imc": None,
    }

    if sexo and sexo.strip():
        s = sexo.strip().upper()
        if s not in ("M", "F"):
            raise HTTPException(400, "El campo 'sexo' debe ser M o F.")
        datos["sexo"] = s

    if datos["peso"] and datos["talla"]:
        datos["imc"] = round(datos["peso"] / (datos["talla"] / 100) ** 2, 1)

    return datos


@app.get("/api/estado")
def estado():
    return {"estado": "ok", "version": predictor.config["version"]}


@app.get("/api/modelo")
def informacion_modelo(usuario=Depends(requiere_usuario)):
    return predictor.info()


@app.get("/api/ejemplos")
def listar_ejemplos(usuario=Depends(requiere_usuario)):
    return cargar_indice_ejemplos()


@app.post("/api/leer-demograficos", dependencies=[Depends(verificar_origen)])
async def leer_demograficos(
    archivo: UploadFile = File(...),
    usuario=Depends(requiere_usuario),
):
    """Edad, sexo, peso y talla que trae un .mat de PulseDB (para el formulario)."""

    if not (archivo.filename or "").lower().endswith(".mat"):
        return {}

    contenido = await archivo.read()
    if len(contenido) > TAMANO_MAX_ARCHIVO:
        raise HTTPException(400, "El archivo supera el tamaño máximo (300 MB).")

    datos = leer_demograficos_mat(contenido)

    # Solo se devuelven valores dentro de los rangos que acepta el formulario.
    validos = {}
    for clave, valor in datos.items():
        campos = dict.fromkeys(("edad", "sexo", "peso", "talla"))
        campos[clave] = str(valor)
        try:
            validos[clave] = validar_demograficos(**campos)[clave]
        except HTTPException:
            pass
    return validos


@app.post("/api/analizar", dependencies=[Depends(verificar_origen)])
async def analizar(
    archivo: Optional[UploadFile] = File(None),
    ejemplo_id: Optional[str] = Form(None),
    codigo_registro: Optional[str] = Form(None),
    edad: Optional[str] = Form(None),
    sexo: Optional[str] = Form(None),
    peso: Optional[str] = Form(None),
    talla: Optional[str] = Form(None),
    usuario=Depends(requiere_usuario),
):

    demograficos = validar_demograficos(edad, sexo, peso, talla)

    codigo = (codigo_registro or "").strip() or None
    if codigo and not PATRON_CODIGO.match(codigo):
        raise HTTPException(
            400, "El código de registro solo admite letras, números, espacios, '-' y '_' (máx. 30)."
        )

    referencia = None

    if archivo is not None and archivo.filename:

        contenido = await archivo.read()
        if len(contenido) > TAMANO_MAX_ARCHIVO:
            raise HTTPException(400, "El archivo supera el tamaño máximo (300 MB).")
        origen = f"Archivo: {archivo.filename[:120]}"
        nombre_archivo = archivo.filename
        if not nombre_archivo.lower().endswith((".csv", ".txt", ".mat")):
            raise HTTPException(400, "Formato no admitido. Use un archivo .csv, .txt o .mat.")

    elif ejemplo_id:

        ejemplo = next((e for e in cargar_indice_ejemplos() if e["id"] == ejemplo_id), None)
        if ejemplo is None:
            raise HTTPException(404, "Registro de ejemplo no encontrado.")
        contenido = (CARPETA_EJEMPLOS / ejemplo["archivo"]).read_bytes()
        origen = f"Ejemplo: {ejemplo['nombre']}"
        nombre_archivo = ejemplo["archivo"]
        referencia = ejemplo.get("referencia")

    else:
        raise HTTPException(400, "Debe subir un archivo ECG o seleccionar un registro de ejemplo.")

    try:
        senal = leer_senal(contenido, nombre_archivo)
        resultado = predictor.predecir_senal(senal)
    except ErrorSenal as e:
        raise HTTPException(400, str(e))

    id_analisis, codigo = bd.guardar_analisis(
        usuario["id"], codigo, origen, demograficos, resultado, predictor.config["version"]
    )
    bd.auditar("analisis", usuario, f"#{id_analisis} {codigo} · {resultado['nivel']}")

    L = predictor.muestras_por_segmento
    n_vista = min(resultado["n_segmentos"], SEGMENTOS_EN_RESPUESTA)

    return {
        "id": id_analisis,
        "codigo_registro": codigo,
        "origen": origen,
        "demograficos": demograficos,
        "demograficos_usados_por_modelo": predictor.config.get("usa_variables_demograficas", False),
        "referencia": referencia,
        "resultado": resultado,
        "senal_vista": np.round(senal[: n_vista * L], 4).tolist(),
        "segmentos_en_vista": n_vista,
        "frecuencia_muestreo_hz": predictor.config["frecuencia_muestreo_hz"],
        "modelo": {"nombre": predictor.config["nombre"], "version": predictor.config["version"]},
    }


@app.get("/api/mis-analisis")
def mis_analisis(usuario=Depends(requiere_usuario)):
    """Cada usuario ve solo sus propios análisis."""
    return bd.listar_analisis(usuario_id=usuario["id"])


# ============================================================
# DISPOSITIVO IoT (SIMULADO)
#
# Un dispositivo registrado envía segmentos ECG de 10 s por HTTP con su
# token (cabecera "Authorization: Bearer <token>"). Cada segmento se
# analiza con el mismo modelo que la carga de archivos. En esta versión
# el dispositivo es un cliente simulado (herramientas/dispositivo_simulado.py)
# que reproduce registros de PulseDB; no se usa un sensor físico.
# ============================================================

PATRON_SESION = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
SEGUNDOS_EN_LINEA = 30          # sin contacto por más tiempo = desconectado


def requiere_dispositivo(request: Request):

    cabecera = request.headers.get("authorization", "")
    if not cabecera.lower().startswith("bearer "):
        raise HTTPException(401, "Falta el token del dispositivo.")
    dispositivo = bd.dispositivo_por_token(seg.hash_token(cabecera[7:].strip()))
    if dispositivo is None:
        raise HTTPException(401, "Token de dispositivo no válido o dispositivo desactivado.")
    return dispositivo


class DatosSegmento(BaseModel):
    sesion: str
    indice: int
    muestras: list[float]
    frecuencia_hz: int


class DatosFinSesion(BaseModel):
    sesion: str
    codigo_registro: Optional[str] = None


def validar_sesion(sesion):
    if not PATRON_SESION.match(sesion or ""):
        raise HTTPException(400, "Identificador de sesión no válido (letras, números, '-' y '_', máx. 40).")


def resumen_lecturas(lecturas):
    """Promedio y nivel de una sesión a partir de sus segmentos."""

    if not lecturas:
        return None, None
    promedio = float(np.mean([l["probabilidad"] for l in lecturas]))
    return round(promedio, 4), predictor.nivel_de(promedio)


@app.post("/api/dispositivo/segmento")
def recibir_segmento(datos: DatosSegmento, dispositivo=Depends(requiere_dispositivo)):

    validar_sesion(datos.sesion)

    L = predictor.muestras_por_segmento
    fs = predictor.config["frecuencia_muestreo_hz"]

    if datos.frecuencia_hz != fs:
        raise HTTPException(400, f"La frecuencia de muestreo debe ser {fs} Hz.")
    if len(datos.muestras) != L:
        raise HTTPException(400, f"Cada segmento debe tener {L} muestras (10 s a {fs} Hz).")
    if not 0 <= datos.indice < 100000:
        raise HTTPException(400, "Índice de segmento fuera de rango.")

    senal = np.asarray(datos.muestras, dtype=np.float32)
    if not np.all(np.isfinite(senal)):
        raise HTTPException(400, "El segmento contiene valores no válidos.")

    resultado = predictor.predecir_senal(senal)
    bd.guardar_lectura(
        dispositivo["id"], datos.sesion, datos.indice, resultado["probabilidad"],
        resultado["nivel"], np.round(senal, 4).tolist(),
    )

    promedio, nivel = resumen_lecturas(bd.sesion_dispositivo(dispositivo["id"], datos.sesion)["lecturas"])

    return {
        "dispositivo": dispositivo["codigo"],
        "indice": datos.indice,
        "probabilidad": resultado["probabilidad"],
        "nivel": resultado["nivel"],
        "promedio_sesion": promedio,
        "nivel_sesion": nivel,
    }


@app.post("/api/dispositivo/finalizar")
def finalizar_sesion(datos: DatosFinSesion, dispositivo=Depends(requiere_dispositivo)):
    """Cierra la sesión y la guarda como un análisis en el historial del especialista."""

    validar_sesion(datos.sesion)

    codigo = (datos.codigo_registro or "").strip() or None
    if codigo and not PATRON_CODIGO.match(codigo):
        raise HTTPException(400, "Código de registro no válido.")

    sesion = bd.sesion_dispositivo(dispositivo["id"], datos.sesion)
    if sesion is None or not sesion["lecturas"]:
        raise HTTPException(404, "La sesión no tiene segmentos recibidos.")
    if sesion["finalizada"]:
        raise HTTPException(409, "La sesión ya fue finalizada.")

    lecturas = sesion["lecturas"]
    promedio, nivel = resumen_lecturas(lecturas)
    fs = predictor.config["frecuencia_muestreo_hz"]
    L = predictor.muestras_por_segmento

    resultado = {
        "probabilidad": promedio,
        "nivel": nivel,
        "umbral": predictor.umbral,
        "n_segmentos": len(lecturas),
        "duracion_analizada_s": round(len(lecturas) * L / fs, 1),
        "segmentos": [
            {"indice": l["indice"] + 1, "inicio_s": round(l["indice"] * L / fs, 1),
             "probabilidad": l["probabilidad"], "nivel": l["nivel"]}
            for l in lecturas
        ],
    }

    demograficos = {"edad": None, "sexo": None, "peso": None, "talla": None, "imc": None}
    origen = f"Dispositivo IoT simulado: {dispositivo['codigo']}"
    usuario = bd.usuario_por_id(dispositivo["usuario_id"]) if dispositivo["usuario_id"] else None

    id_analisis, codigo = bd.guardar_analisis(
        dispositivo["usuario_id"], codigo, origen, demograficos, resultado, predictor.config["version"]
    )
    bd.finalizar_sesion_dispositivo(dispositivo["id"], datos.sesion, id_analisis)
    bd.auditar("analisis_dispositivo", usuario, f"#{id_analisis} {codigo} · {dispositivo['codigo']} · {nivel}")

    return {
        "id": id_analisis, "codigo_registro": codigo, "n_segmentos": len(lecturas),
        "probabilidad": promedio, "nivel": nivel,
    }


def en_linea(ultimo_contacto):
    if not ultimo_contacto:
        return False
    return (datetime.now() - datetime.fromisoformat(ultimo_contacto)).total_seconds() <= SEGUNDOS_EN_LINEA


@app.get("/api/monitoreo/dispositivos")
def monitoreo_dispositivos(usuario=Depends(requiere_usuario)):
    """Dispositivos asignados al usuario (el administrador ve todos)."""

    propios = None if usuario["rol"] == "administrador" else usuario["id"]
    return [
        {**d, "en_linea": en_linea(d["ultimo_contacto"])}
        for d in bd.listar_dispositivos(usuario_id=propios) if d["activo"]
    ]


@app.get("/api/monitoreo/dispositivos/{dispositivo_id}")
def monitoreo_dispositivo(dispositivo_id: int, usuario=Depends(requiere_usuario)):
    """Última sesión del dispositivo: resultado por segmento y señal del último segmento."""

    dispositivo = bd.dispositivo_por_id(dispositivo_id)
    if dispositivo is None or not dispositivo["activo"]:
        raise HTTPException(404, "Dispositivo no encontrado.")
    if usuario["rol"] != "administrador" and dispositivo["usuario_id"] != usuario["id"]:
        raise HTTPException(403, "Este dispositivo no está asignado a tu cuenta.")

    sesion = bd.sesion_dispositivo(dispositivo_id)
    promedio, nivel = resumen_lecturas(sesion["lecturas"]) if sesion else (None, None)

    return {
        "dispositivo": {"id": dispositivo["id"], "codigo": dispositivo["codigo"], "nombre": dispositivo["nombre"],
                        "ultimo_contacto": dispositivo["ultimo_contacto"],
                        "en_linea": en_linea(dispositivo["ultimo_contacto"])},
        "sesion": None if sesion is None else {
            "sesion": sesion["sesion"], "iniciada": sesion["iniciada"], "finalizada": sesion["finalizada"],
            "analisis_id": sesion["analisis_id"], "lecturas": sesion["lecturas"], "ultima": sesion["ultima"],
            "promedio": promedio, "nivel": nivel,
        },
        "frecuencia_muestreo_hz": predictor.config["frecuencia_muestreo_hz"],
    }


# ============================================================
# ADMINISTRACIÓN (solo administrador)
# ============================================================

class DatosNuevoUsuario(BaseModel):
    usuario: str
    nombre: str
    rol: str = "especialista"
    contrasena: Optional[str] = None


class DatosEdicionUsuario(BaseModel):
    nombre: Optional[str] = None
    activo: Optional[bool] = None


@app.get("/api/admin/resumen")
def admin_resumen(admin=Depends(requiere_admin)):
    return bd.resumen(dias=30)


@app.get("/api/admin/analisis")
def admin_analisis(
    usuario_id: Optional[int] = None,
    desde: Optional[str] = None,
    hasta: Optional[str] = None,
    nivel: Optional[str] = None,
    admin=Depends(requiere_admin),
):
    return bd.listar_analisis(usuario_id=usuario_id, desde=desde, hasta=hasta, nivel=nivel)


@app.get("/api/admin/analisis.csv")
def admin_analisis_csv(
    usuario_id: Optional[int] = None,
    desde: Optional[str] = None,
    hasta: Optional[str] = None,
    nivel: Optional[str] = None,
    admin=Depends(requiere_admin),
):

    filas = bd.listar_analisis(
        usuario_id=usuario_id, desde=desde, hasta=hasta, nivel=nivel, limite=100000
    )
    columnas = [
        "id", "fecha", "codigo_registro", "especialista", "origen", "edad", "sexo",
        "peso", "talla", "imc", "n_segmentos", "probabilidad", "nivel", "version_modelo",
    ]

    salida = io.StringIO()
    escritor = csv.DictWriter(salida, fieldnames=columnas, extrasaction="ignore", delimiter=";")
    escritor.writeheader()
    escritor.writerows(filas)

    bd.auditar("exportacion_csv", admin, f"{len(filas)} filas")

    nombre = f"analisis_{datetime.now():%Y%m%d_%H%M}.csv"
    return Response(
        content="﻿" + salida.getvalue(),            # BOM para Excel
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{nombre}"'},
    )


@app.delete("/api/admin/analisis/{analisis_id}", dependencies=[Depends(verificar_origen)])
def admin_eliminar_analisis(analisis_id: int, admin=Depends(requiere_admin)):

    if not bd.eliminar_analisis(analisis_id):
        raise HTTPException(404, "Análisis no encontrado.")
    bd.auditar("analisis_eliminado", admin, f"#{analisis_id}")
    return {"eliminado": analisis_id}


@app.get("/api/admin/usuarios")
def admin_usuarios(admin=Depends(requiere_admin)):
    return bd.listar_usuarios()


@app.post("/api/admin/usuarios", dependencies=[Depends(verificar_origen)])
def admin_crear_usuario(datos: DatosNuevoUsuario, admin=Depends(requiere_admin)):

    usuario = datos.usuario.strip()
    nombre = datos.nombre.strip()

    if not PATRON_USUARIO.match(usuario):
        raise HTTPException(400, "El usuario debe tener 3–30 caracteres: letras, números, '.', '-' o '_'.")
    if not (2 <= len(nombre) <= 80):
        raise HTTPException(400, "Ingrese un nombre válido.")
    if datos.rol not in ROLES:
        raise HTTPException(400, "Rol no válido.")
    if bd.usuario_por_nombre(usuario):
        raise HTTPException(400, "Ese nombre de usuario ya existe.")

    contrasena = datos.contrasena or seg.generar_contrasena()
    error = seg.validar_contrasena_nueva(contrasena)
    if error:
        raise HTTPException(400, error)

    nuevo_id = bd.crear_usuario(usuario, nombre, datos.rol, seg.hash_contrasena(contrasena))
    bd.auditar("usuario_creado", admin, f"{usuario} ({datos.rol})")

    # La contraseña inicial se muestra UNA sola vez al administrador.
    return {"id": nuevo_id, "usuario": usuario, "contrasena_inicial": contrasena}


@app.patch("/api/admin/usuarios/{usuario_id}", dependencies=[Depends(verificar_origen)])
def admin_editar_usuario(usuario_id: int, datos: DatosEdicionUsuario, admin=Depends(requiere_admin)):

    objetivo = bd.usuario_por_id(usuario_id)
    if objetivo is None:
        raise HTTPException(404, "Usuario no encontrado.")

    cambios = {}

    if datos.nombre is not None:
        if not (2 <= len(datos.nombre.strip()) <= 80):
            raise HTTPException(400, "Ingrese un nombre válido.")
        cambios["nombre"] = datos.nombre.strip()

    if datos.activo is not None:
        if not datos.activo:
            if objetivo["id"] == admin["id"]:
                raise HTTPException(400, "No puede desactivar su propia cuenta.")
            if objetivo["rol"] == "administrador" and bd.contar_administradores_activos() <= 1:
                raise HTTPException(400, "Debe existir al menos un administrador activo.")
        cambios["activo"] = 1 if datos.activo else 0

    bd.actualizar_usuario(usuario_id, **cambios)

    if cambios.get("activo") == 0:
        bd.cerrar_sesiones_de_usuario(usuario_id)

    bd.auditar(
        "usuario_editado", admin,
        f"{objetivo['usuario']}: " + ", ".join(f"{k}={v}" for k, v in cambios.items()),
    )
    return {"ok": True}


@app.post("/api/admin/usuarios/{usuario_id}/restablecer", dependencies=[Depends(verificar_origen)])
def admin_restablecer_contrasena(usuario_id: int, admin=Depends(requiere_admin)):

    objetivo = bd.usuario_por_id(usuario_id)
    if objetivo is None:
        raise HTTPException(404, "Usuario no encontrado.")

    nueva = seg.generar_contrasena()
    bd.actualizar_usuario(usuario_id, hash_contrasena=seg.hash_contrasena(nueva))
    bd.cerrar_sesiones_de_usuario(usuario_id)
    bd.auditar("contrasena_restablecida", admin, objetivo["usuario"])

    return {"usuario": objetivo["usuario"], "contrasena_temporal": nueva}


@app.get("/api/admin/auditoria")
def admin_auditoria(admin=Depends(requiere_admin)):
    return bd.listar_auditoria()


# ============================================================
# ARCHIVOS ESTÁTICOS (CSS, JS)
# ============================================================

@app.middleware("http")
async def sin_cache_estaticos(request: Request, call_next):
    """El navegador debe revalidar CSS/JS para no usar versiones antiguas."""

    respuesta = await call_next(request)
    if request.url.path.startswith("/static/"):
        respuesta.headers["Cache-Control"] = "no-cache"
    return respuesta


app.mount("/static", StaticFiles(directory=CARPETA_STATIC), name="static")
