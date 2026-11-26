"""
Base de datos SQLite de la aplicación (archivo app_web/datos/historial.db).

Tablas:
- usuarios   : cuentas (administrador / especialista), contraseña con hash.
- sesiones   : sesiones activas (se guarda el hash del token, no el token).
- analisis   : resumen de cada análisis (no se guarda la señal completa).
- auditoria  : eventos del sistema (inicios de sesión, análisis, cambios de usuarios).
- dispositivos, sesiones_dispositivo, lecturas_dispositivo : dispositivo IoT
  (simulado) que envía segmentos ECG de 10 s y resultado de cada segmento.
"""

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path


ROLES = ("administrador", "especialista")


def ahora() -> str:
    return datetime.now().isoformat(timespec="seconds")


class BaseDatos:

    def __init__(self, ruta_db: Path):

        self.ruta_db = Path(ruta_db)
        self.ruta_db.parent.mkdir(parents=True, exist_ok=True)
        self._crear_esquema()

    def _conectar(self):

        con = sqlite3.connect(self.ruta_db)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")
        return con

    # ========================================================
    # ESQUEMA
    # ========================================================

    def _crear_esquema(self):

        with self._conectar() as con:

            con.executescript(
                """
                CREATE TABLE IF NOT EXISTS usuarios (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    usuario TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    nombre TEXT NOT NULL,
                    rol TEXT NOT NULL CHECK (rol IN ('administrador', 'especialista')),
                    hash_contrasena TEXT NOT NULL,
                    activo INTEGER NOT NULL DEFAULT 1,
                    creado TEXT NOT NULL,
                    ultimo_acceso TEXT
                );

                CREATE TABLE IF NOT EXISTS sesiones (
                    hash_token TEXT PRIMARY KEY,
                    usuario_id INTEGER NOT NULL REFERENCES usuarios(id) ON DELETE CASCADE,
                    creada TEXT NOT NULL,
                    expira TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS analisis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fecha TEXT NOT NULL,
                    origen TEXT NOT NULL,
                    edad REAL,
                    sexo TEXT,
                    peso REAL,
                    talla REAL,
                    imc REAL,
                    n_segmentos INTEGER NOT NULL,
                    probabilidad REAL NOT NULL,
                    nivel TEXT NOT NULL,
                    version_modelo TEXT NOT NULL,
                    detalle_json TEXT
                );

                CREATE TABLE IF NOT EXISTS auditoria (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fecha TEXT NOT NULL,
                    usuario_id INTEGER,
                    usuario TEXT,
                    evento TEXT NOT NULL,
                    detalle TEXT
                );

                CREATE TABLE IF NOT EXISTS dispositivos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    codigo TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    nombre TEXT NOT NULL,
                    hash_token TEXT NOT NULL UNIQUE,
                    usuario_id INTEGER REFERENCES usuarios(id) ON DELETE SET NULL,
                    activo INTEGER NOT NULL DEFAULT 1,
                    creado TEXT NOT NULL,
                    ultimo_contacto TEXT
                );

                CREATE TABLE IF NOT EXISTS sesiones_dispositivo (
                    dispositivo_id INTEGER NOT NULL REFERENCES dispositivos(id) ON DELETE CASCADE,
                    sesion TEXT NOT NULL,
                    iniciada TEXT NOT NULL,
                    finalizada TEXT,
                    analisis_id INTEGER,
                    PRIMARY KEY (dispositivo_id, sesion)
                );

                CREATE TABLE IF NOT EXISTS lecturas_dispositivo (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dispositivo_id INTEGER NOT NULL REFERENCES dispositivos(id) ON DELETE CASCADE,
                    sesion TEXT NOT NULL,
                    indice INTEGER NOT NULL,
                    fecha TEXT NOT NULL,
                    probabilidad REAL NOT NULL,
                    nivel TEXT NOT NULL,
                    muestras_json TEXT NOT NULL,
                    UNIQUE (dispositivo_id, sesion, indice)
                );
                """
            )

            # Migración: columnas nuevas en una tabla 'analisis' anterior.
            columnas = {f["name"] for f in con.execute("PRAGMA table_info(analisis)")}

            if "usuario_id" not in columnas:
                con.execute("ALTER TABLE analisis ADD COLUMN usuario_id INTEGER")
            if "codigo_registro" not in columnas:
                con.execute("ALTER TABLE analisis ADD COLUMN codigo_registro TEXT")

    # ========================================================
    # USUARIOS
    # ========================================================

    def crear_usuario(self, usuario, nombre, rol, hash_contrasena) -> int:

        if rol not in ROLES:
            raise ValueError(f"Rol no válido: {rol}")

        with self._conectar() as con:
            cur = con.execute(
                """
                INSERT INTO usuarios (usuario, nombre, rol, hash_contrasena, creado)
                VALUES (?, ?, ?, ?, ?)
                """,
                (usuario.strip(), nombre.strip(), rol, hash_contrasena, ahora()),
            )
            return cur.lastrowid

    def usuario_por_nombre(self, usuario):

        with self._conectar() as con:
            fila = con.execute(
                "SELECT * FROM usuarios WHERE usuario = ?", (usuario.strip(),)
            ).fetchone()
        return dict(fila) if fila else None

    def usuario_por_id(self, usuario_id):

        with self._conectar() as con:
            fila = con.execute(
                "SELECT * FROM usuarios WHERE id = ?", (usuario_id,)
            ).fetchone()
        return dict(fila) if fila else None

    def listar_usuarios(self):

        with self._conectar() as con:
            filas = con.execute(
                """
                SELECT u.id, u.usuario, u.nombre, u.rol, u.activo, u.creado,
                       u.ultimo_acceso,
                       (SELECT COUNT(*) FROM analisis a WHERE a.usuario_id = u.id)
                           AS total_analisis
                FROM usuarios u
                ORDER BY u.rol, u.nombre
                """
            ).fetchall()
        return [dict(f) for f in filas]

    def contar_administradores_activos(self) -> int:

        with self._conectar() as con:
            return con.execute(
                "SELECT COUNT(*) FROM usuarios WHERE rol = 'administrador' AND activo = 1"
            ).fetchone()[0]

    def actualizar_usuario(self, usuario_id, **campos):

        permitidos = {"nombre", "activo", "hash_contrasena", "ultimo_acceso"}
        campos = {k: v for k, v in campos.items() if k in permitidos}

        if not campos:
            return

        asignaciones = ", ".join(f"{k} = ?" for k in campos)

        with self._conectar() as con:
            con.execute(
                f"UPDATE usuarios SET {asignaciones} WHERE id = ?",
                (*campos.values(), usuario_id),
            )

    # ========================================================
    # SESIONES
    # ========================================================

    def crear_sesion(self, hash_token, usuario_id, horas):

        expira = (datetime.now() + timedelta(hours=horas)).isoformat(timespec="seconds")

        with self._conectar() as con:
            con.execute("DELETE FROM sesiones WHERE expira < ?", (ahora(),))
            con.execute(
                "INSERT INTO sesiones (hash_token, usuario_id, creada, expira) VALUES (?, ?, ?, ?)",
                (hash_token, usuario_id, ahora(), expira),
            )

    def usuario_de_sesion(self, hash_token):

        with self._conectar() as con:
            fila = con.execute(
                """
                SELECT u.* FROM sesiones s
                JOIN usuarios u ON u.id = s.usuario_id
                WHERE s.hash_token = ? AND s.expira > ? AND u.activo = 1
                """,
                (hash_token, ahora()),
            ).fetchone()
        return dict(fila) if fila else None

    def cerrar_sesion(self, hash_token):

        with self._conectar() as con:
            con.execute("DELETE FROM sesiones WHERE hash_token = ?", (hash_token,))

    def cerrar_sesiones_de_usuario(self, usuario_id):

        with self._conectar() as con:
            con.execute("DELETE FROM sesiones WHERE usuario_id = ?", (usuario_id,))

    # ========================================================
    # ANÁLISIS
    # ========================================================

    def guardar_analisis(self, usuario_id, codigo_registro, origen,
                         demograficos, resultado, version_modelo):
        """Guarda el análisis y devuelve (id, codigo_registro)."""

        detalle = {
            "segmentos": resultado["segmentos"],
            "umbral": resultado["umbral"],
            "duracion_analizada_s": resultado["duracion_analizada_s"],
        }

        with self._conectar() as con:
            cur = con.execute(
                """
                INSERT INTO analisis (
                    fecha, usuario_id, codigo_registro, origen, edad, sexo, peso,
                    talla, imc, n_segmentos, probabilidad, nivel, version_modelo,
                    detalle_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ahora(), usuario_id, codigo_registro, origen,
                    demograficos.get("edad"), demograficos.get("sexo"),
                    demograficos.get("peso"), demograficos.get("talla"),
                    demograficos.get("imc"), resultado["n_segmentos"],
                    resultado["probabilidad"], resultado["nivel"],
                    version_modelo, json.dumps(detalle),
                ),
            )
            nuevo_id = cur.lastrowid

            # Código automático si el especialista no ingresó uno.
            if not codigo_registro:
                codigo_registro = f"REG-{nuevo_id:05d}"
                con.execute(
                    "UPDATE analisis SET codigo_registro = ? WHERE id = ?",
                    (codigo_registro, nuevo_id),
                )

        return nuevo_id, codigo_registro

    def listar_analisis(self, usuario_id=None, desde=None, hasta=None,
                        nivel=None, limite=1000):

        condiciones, valores = [], []

        if usuario_id is not None:
            condiciones.append("a.usuario_id = ?")
            valores.append(usuario_id)
        if desde:
            condiciones.append("a.fecha >= ?")
            valores.append(desde)
        if hasta:
            condiciones.append("a.fecha <= ?")
            valores.append(hasta + "T23:59:59")
        if nivel:
            condiciones.append("a.nivel = ?")
            valores.append(nivel)

        where = ("WHERE " + " AND ".join(condiciones)) if condiciones else ""

        with self._conectar() as con:
            filas = con.execute(
                f"""
                SELECT a.id, a.fecha, a.codigo_registro, a.origen, a.edad, a.sexo,
                       a.peso, a.talla, a.imc, a.n_segmentos, a.probabilidad,
                       a.nivel, a.version_modelo, a.usuario_id,
                       COALESCE(u.nombre, 'Antes del inicio de sesión') AS especialista
                FROM analisis a
                LEFT JOIN usuarios u ON u.id = a.usuario_id
                {where}
                ORDER BY a.id DESC
                LIMIT ?
                """,
                (*valores, limite),
            ).fetchall()

        return [dict(f) for f in filas]

    def eliminar_analisis(self, analisis_id) -> bool:

        with self._conectar() as con:
            cur = con.execute("DELETE FROM analisis WHERE id = ?", (analisis_id,))
            return cur.rowcount > 0

    def resumen(self, dias=30):

        desde = (datetime.now() - timedelta(days=dias - 1)).date().isoformat()

        with self._conectar() as con:

            total = con.execute("SELECT COUNT(*) FROM analisis").fetchone()[0]

            por_nivel = {
                f["nivel"]: f["n"]
                for f in con.execute(
                    "SELECT nivel, COUNT(*) AS n FROM analisis GROUP BY nivel"
                )
            }

            por_dia = {
                f["dia"]: f["n"]
                for f in con.execute(
                    """
                    SELECT substr(fecha, 1, 10) AS dia, COUNT(*) AS n
                    FROM analisis WHERE fecha >= ? GROUP BY dia
                    """,
                    (desde,),
                )
            }

            por_especialista = [
                dict(f)
                for f in con.execute(
                    """
                    SELECT COALESCE(u.nombre, 'Antes del inicio de sesión') AS especialista,
                           COUNT(*) AS total,
                           SUM(a.nivel = 'Bajo') AS bajo,
                           SUM(a.nivel = 'Moderado') AS moderado,
                           SUM(a.nivel = 'Alto') AS alto,
                           MAX(a.fecha) AS ultimo
                    FROM analisis a LEFT JOIN usuarios u ON u.id = a.usuario_id
                    GROUP BY a.usuario_id ORDER BY total DESC
                    """
                )
            ]

            usuarios_activos = con.execute(
                "SELECT COUNT(*) FROM usuarios WHERE activo = 1 AND rol = 'especialista'"
            ).fetchone()[0]

            prob_media = con.execute(
                "SELECT AVG(probabilidad) FROM analisis"
            ).fetchone()[0]

        # Serie diaria completa (días sin análisis = 0).
        serie = []
        for i in range(dias):
            dia = (datetime.now() - timedelta(days=dias - 1 - i)).date().isoformat()
            serie.append({"dia": dia, "total": por_dia.get(dia, 0)})

        return {
            "total": total,
            "por_nivel": {n: por_nivel.get(n, 0) for n in ("Bajo", "Moderado", "Alto")},
            "por_dia": serie,
            "por_especialista": por_especialista,
            "especialistas_activos": usuarios_activos,
            "probabilidad_media": prob_media,
        }

    # ========================================================
    # DISPOSITIVOS IoT (SIMULADOS)
    # ========================================================

    def crear_dispositivo(self, codigo, nombre, hash_token, usuario_id) -> int:

        with self._conectar() as con:
            cur = con.execute(
                """
                INSERT INTO dispositivos (codigo, nombre, hash_token, usuario_id, creado)
                VALUES (?, ?, ?, ?, ?)
                """,
                (codigo.strip(), nombre.strip(), hash_token, usuario_id, ahora()),
            )
            return cur.lastrowid

    def dispositivo_por_token(self, hash_token):

        with self._conectar() as con:
            fila = con.execute(
                "SELECT * FROM dispositivos WHERE hash_token = ? AND activo = 1", (hash_token,)
            ).fetchone()
        return dict(fila) if fila else None

    def dispositivo_por_id(self, dispositivo_id):

        with self._conectar() as con:
            fila = con.execute(
                "SELECT * FROM dispositivos WHERE id = ?", (dispositivo_id,)
            ).fetchone()
        return dict(fila) if fila else None

    def listar_dispositivos(self, usuario_id=None):

        where, valores = ("WHERE d.usuario_id = ?", (usuario_id,)) if usuario_id is not None else ("", ())

        with self._conectar() as con:
            filas = con.execute(
                f"""
                SELECT d.id, d.codigo, d.nombre, d.usuario_id, d.activo, d.creado,
                       d.ultimo_contacto, u.nombre AS especialista
                FROM dispositivos d LEFT JOIN usuarios u ON u.id = d.usuario_id
                {where}
                ORDER BY d.codigo
                """,
                valores,
            ).fetchall()
        return [dict(f) for f in filas]

    def actualizar_dispositivo(self, codigo, **campos):

        permitidos = {"activo", "hash_token", "usuario_id"}
        campos = {k: v for k, v in campos.items() if k in permitidos}
        if not campos:
            return False

        asignaciones = ", ".join(f"{k} = ?" for k in campos)
        with self._conectar() as con:
            cur = con.execute(
                f"UPDATE dispositivos SET {asignaciones} WHERE codigo = ?",
                (*campos.values(), codigo),
            )
            return cur.rowcount > 0

    def guardar_lectura(self, dispositivo_id, sesion, indice, probabilidad, nivel, muestras):
        """Guarda el resultado de un segmento recibido (reenviar el mismo índice lo reemplaza)."""

        with self._conectar() as con:
            con.execute(
                "INSERT OR IGNORE INTO sesiones_dispositivo (dispositivo_id, sesion, iniciada) VALUES (?, ?, ?)",
                (dispositivo_id, sesion, ahora()),
            )
            con.execute(
                """
                INSERT OR REPLACE INTO lecturas_dispositivo
                    (dispositivo_id, sesion, indice, fecha, probabilidad, nivel, muestras_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (dispositivo_id, sesion, indice, ahora(), probabilidad, nivel, json.dumps(muestras)),
            )
            con.execute(
                "UPDATE dispositivos SET ultimo_contacto = ? WHERE id = ?", (ahora(), dispositivo_id)
            )

    def sesion_dispositivo(self, dispositivo_id, sesion=None):
        """Datos de una sesión (por defecto, la última) con sus lecturas en orden."""

        with self._conectar() as con:

            if sesion is None:
                fila = con.execute(
                    """
                    SELECT * FROM sesiones_dispositivo WHERE dispositivo_id = ?
                    ORDER BY iniciada DESC, rowid DESC LIMIT 1
                    """,
                    (dispositivo_id,),
                ).fetchone()
            else:
                fila = con.execute(
                    "SELECT * FROM sesiones_dispositivo WHERE dispositivo_id = ? AND sesion = ?",
                    (dispositivo_id, sesion),
                ).fetchone()

            if fila is None:
                return None

            lecturas = con.execute(
                """
                SELECT indice, fecha, probabilidad, nivel FROM lecturas_dispositivo
                WHERE dispositivo_id = ? AND sesion = ? ORDER BY indice
                """,
                (dispositivo_id, fila["sesion"]),
            ).fetchall()

            ultima = con.execute(
                """
                SELECT indice, muestras_json FROM lecturas_dispositivo
                WHERE dispositivo_id = ? AND sesion = ? ORDER BY id DESC LIMIT 1
                """,
                (dispositivo_id, fila["sesion"]),
            ).fetchone()

        return {
            **dict(fila),
            "lecturas": [dict(l) for l in lecturas],
            "ultima": {"indice": ultima["indice"], "muestras": json.loads(ultima["muestras_json"])} if ultima else None,
        }

    def finalizar_sesion_dispositivo(self, dispositivo_id, sesion, analisis_id):

        with self._conectar() as con:
            con.execute(
                """
                UPDATE sesiones_dispositivo SET finalizada = ?, analisis_id = ?
                WHERE dispositivo_id = ? AND sesion = ?
                """,
                (ahora(), analisis_id, dispositivo_id, sesion),
            )

    # ========================================================
    # AUDITORÍA
    # ========================================================

    def auditar(self, evento, usuario=None, detalle=None):

        with self._conectar() as con:
            con.execute(
                "INSERT INTO auditoria (fecha, usuario_id, usuario, evento, detalle) VALUES (?, ?, ?, ?, ?)",
                (
                    ahora(),
                    usuario["id"] if usuario else None,
                    usuario["usuario"] if usuario else None,
                    evento,
                    detalle,
                ),
            )

    def listar_auditoria(self, limite=300):

        with self._conectar() as con:
            filas = con.execute(
                "SELECT * FROM auditoria ORDER BY id DESC LIMIT ?", (limite,)
            ).fetchall()
        return [dict(f) for f in filas]
