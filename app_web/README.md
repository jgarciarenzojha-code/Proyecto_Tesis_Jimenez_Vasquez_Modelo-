# Aplicación web: Alerta Temprana ECG

Aplicación web que integra el modelo CNN de la tesis. El usuario carga una señal ECG
(o elige un registro de ejemplo), el servidor la procesa con el modelo y la página
muestra la probabilidad estimada y un nivel de alerta (Bajo / Moderado / Alto).

> Herramienta de apoyo con fines académicos. No constituye un diagnóstico médico.

## Cómo ejecutarla

Doble clic en `iniciar_app.bat`, o bien desde esta carpeta:

```
..\.venv\Scripts\python.exe -m uvicorn backend.main:app --port 8001
```

Luego abrir en el navegador: **http://localhost:8001**

La documentación automática de la API está en http://localhost:8001/docs

## Acceso y roles

| Ruta | Quién | Qué hace |
|---|---|---|
| `/login` | Todos | Inicio de sesión |
| `/app` | Especialista y administrador | Análisis de señales ECG y "Mis análisis" (solo los propios) |
| `/admin` | Solo administrador | Resumen, todos los análisis (filtros + CSV), usuarios, modelo y auditoría |

- No hay registro libre: el administrador crea las cuentas desde `/admin` → Usuarios.
- Contraseñas guardadas con hash scrypt; sesión en cookie HttpOnly que expira a las 8 h.
- Bloqueo de 5 min tras 5 intentos fallidos; protección CSRF; auditoría de cada acción.
- Recuperar acceso o crear usuarios desde la terminal:

```
..\.venv\Scripts\python.exe herramientas\gestionar_usuarios.py listar
..\.venv\Scripts\python.exe herramientas\gestionar_usuarios.py contrasena renzo
..\.venv\Scripts\python.exe herramientas\gestionar_usuarios.py crear usuario "Nombre" especialista --generar
```

## Estructura

```
app_web/
├── config_modelo.json      ← modelo activo, umbral, niveles y métricas
├── iniciar_app.bat
├── backend/
│   ├── main.py             ← páginas y API (FastAPI) con control de acceso por rol
│   ├── seguridad.py        ← hash de contraseñas, sesiones, límite de intentos
│   ├── basedatos.py        ← SQLite: usuarios, sesiones, análisis, auditoría
│   └── modelo/
│       ├── arquitecturas.py    ← CNN V2, V3 y multitarea (copias exactas de los scripts)
│       ├── preprocesamiento.py ← lectura CSV, segmentación, z-score
│       └── predictor.py        ← carga del modelo, predicción, niveles
├── frontend/
│   ├── login.html, app.html, admin.html   ← páginas (servidas solo con sesión válida)
│   └── static/             ← estilos.css, comun.js, app.js, admin.js
├── ejemplos/               ← registros de CalFree para demostración
├── herramientas/
│   ├── exportar_ejemplos.py
│   └── gestionar_usuarios.py
└── datos/historial.db      ← base de datos local
```

## Funcionamiento

1. **Entrada:** archivo con la señal ECG a 125 Hz, de hasta 300 MB. Mínimo 10 s (1250 muestras);
   se analiza el registro completo hasta 6 horas (2160 segmentos). Opcional: edad, sexo, peso y talla.
   - **CSV/TXT:** una muestra por línea o separadas por comas.
   - **MAT (MATLAB v5/v7/v7.3):** variable `ecg` (vector o matriz de segmentos N×1250),
     la estructura de PulseDB `Subset/Signals` (canal 0 = ECG) o un archivo por sujeto
     de PulseDB (`Subj_Wins`, se usa `ECG_F`, el ECG filtrado).
     Archivos de prueba: `herramientas/exportar_ejemplos_mat.py` → `ejemplos_mat/`.
2. **Segmentación:** la señal se divide en segmentos de 10 s (1250 muestras).
3. **Preprocesamiento:** z-score por segmento (igual que en el entrenamiento).
4. **Modelo:** la CNN estima una probabilidad por segmento.
5. **Agregación:** el resultado final es el promedio de los segmentos.
6. **Nivel de alerta:** según los umbrales seleccionados en validación:
   - Bajo: < 0.4262 · Moderado: 0.4262 – 0.6052 · Alto: ≥ 0.6052

## Dispositivo IoT simulado (pestaña "Monitoreo IoT")

No se usa ningún sensor físico. Un cliente de software reproduce un registro de
PulseDB y envía un segmento ECG de 10 s cada cierto tiempo por HTTP, como lo haría
un equipo de adquisición. La pestaña **Monitoreo IoT** muestra cada segmento y su
resultado en tiempo real; al terminar, la sesión se guarda en "Mis análisis".

```
cd E:\Tesis_PulseDB\app_web
# 1. Registrar el dispositivo y asignarlo a un especialista (muestra el token una sola vez)
..\.venv\Scripts\python.exe herramientas\gestionar_dispositivos.py crear ECG-SIM-01 "Dispositivo ECG simulado 01" especialista01
# 2. Con la aplicación en marcha, enviar un registro (un segmento cada 10 s)
..\.venv\Scripts\python.exe herramientas\dispositivo_simulado.py --archivo ejemplos_mat\registro_prueba_01.mat --token <TOKEN>
```

Opciones del simulador: `--intervalo 2` (más rápido), `--max-segmentos 30`, `--codigo REG-001`.
API: `POST /api/dispositivo/segmento` (cabecera `Authorization: Bearer <token>`,
JSON con `sesion`, `indice`, `muestras` [1250 valores] y `frecuencia_hz` = 125) y
`POST /api/dispositivo/finalizar`. Un dispositivo real solo tendría que enviar este formato.

## Cambiar de modelo (cuando se entrene el modelo mejorado)

Solo se edita `config_modelo.json`:

- `arquitectura`: `cnn_v2`, `cnn_v3` o `cnn_multitarea` (o una nueva registrada en
  `backend/modelo/arquitecturas.py`).
- `ruta_checkpoint`: ruta al archivo `.pt` (relativa a esta carpeta).
- `umbral_decision` y `niveles`: los obtenidos en **validación** para ese modelo.
- `metricas`: los resultados reales del nuevo modelo.

Luego se reinicia el servidor. La página no necesita cambios.

## Verificación

Las probabilidades que entrega la aplicación coinciden con las del script
`26_evaluacion_final_calfree.py` (diferencia máxima 0.00005 por redondeo),
lo que confirma que el modelo y el preprocesamiento son idénticos.

## Registros de ejemplo

`ejemplos/` contiene 6 registros de 60 s del conjunto de prueba independiente
VitalDB CalFree (no usado en entrenamiento). Para regenerarlos:

```
..\.venv\Scripts\python.exe herramientas\exportar_ejemplos.py
```
