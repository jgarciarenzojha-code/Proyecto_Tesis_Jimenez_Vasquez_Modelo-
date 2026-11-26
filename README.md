# Detección temprana de hipertensión arterial mediante señales ECG (IoT + CNN)

Código, modelos entrenados y aplicación web de la tesis
**"Desarrollo de un modelo basado en IoT y CNN para la detección temprana de hipertensión arterial mediante señales ECG"**
(Escuela Profesional de Ingeniería de Sistemas, Universidad Señor de Sipán).

- **Datos:** PulseDB v2.0, subconjuntos de VitalDB (ECG de una derivación, segmentos de 10 s a 125 Hz).
- **Etiqueta:** hipertensión si PAS ≥ 140 mmHg o PAD ≥ 90 mmHg (por segmento).
- **División por sujeto:** 1 034 entrenamiento · 259 validación · 144 prueba independiente (CalFree).
- **Modelo final:** CNN-Ensemble (CNN 1D residual + multiescala + multitarea; promedio de logits).
- **Resultados:** ROC-AUC 0,739 en validación y 0,695 en CalFree.
- **Aplicación web:** FastAPI + HTML/CSS/JavaScript + SQLite, con un dispositivo IoT **simulado**.

> Herramienta de apoyo con fines académicos. **No constituye un diagnóstico médico.**

## Contenido
| Carpeta | Contenido |
|---|---|
| `scripts/` | Scripts del modelo, ordenados por función. Índice: `scripts/LEEME.md` |
| `app_web/` | Aplicación web (backend, frontend, herramientas y registros de ejemplo) |
| `modelos/` | Las 3 CNN entrenadas y el ensemble (`mejor_modelo_ensemble_140_90.pt`, el que usa la web). Guía: `modelos/LEEME.md` |
| `procesado/` | División por sujeto (índices) y etiquetas 140/90 |
| `resultados/` | Una carpeta por CNN, `4_cnn_ensemble` (resultados de la tesis) y `5_experimentos`. Guía: `resultados/LEEME.md` |
| `figuras/` | Figuras del documento (`figuras/paf`: versión final a 300 dpi) |
| `Guia_tecnica_scripts_modelo_app.docx` | Explicación de los scripts, el modelo y la aplicación web |

**No se incluyen** los datasets de PulseDB (más de 14 GB) ni la base de datos de usuarios de la aplicación.
Los datasets están disponibles públicamente en PulseDB (Wang et al., 2023) y VitalDB (Lee et al., 2022).

## Instalación
Se usó Python 3.11. Los scripts usan la ruta fija `E:\Tesis_PulseDB\...`: clonar o copiar el repositorio en
`E:\Tesis_PulseDB` (o cambiar la línea `RAIZ = ...` de cada script).

```
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install torch numpy h5py scipy scikit-learn pandas matplotlib
.venv\Scripts\python.exe -m pip install -r app_web\requirements.txt
```

Versiones usadas: torch 2.9.0 (CUDA 12.8), numpy 2.4.6, h5py 3.16.0, scipy 1.17.1, scikit-learn 1.9.1,
pandas 3.0.6, matplotlib 3.11.2, FastAPI 0.142.4, Uvicorn 0.54.0.

## Aplicación web (no necesita los datasets)
```
cd app_web
..\.venv\Scripts\python.exe -m uvicorn backend.main:app --port 8001
```

Abrir http://127.0.0.1:8001 e iniciar sesión con el **usuario de demostración**, que se crea
automáticamente la primera vez que se levanta la aplicación (cuando la base de datos está vacía):

| Usuario | Contraseña | Rol |
|---|---|---|
| `especialista` | `especialista123` | Especialista (análisis, historial y Monitoreo IoT) |

> Es una cuenta de demostración para uso local. Se recomienda cambiar la contraseña:
> `..\.venv\Scripts\python.exe herramientas\gestionar_usuarios.py contrasena especialista`

Para administrar usuarios (panel /admin) se puede crear un administrador:
`..\.venv\Scripts\python.exe herramientas\gestionar_usuarios.py crear admin "Administrador" administrador`

Hay registros de prueba en `app_web/ejemplos` y `app_web/ejemplos_mat`.

### Dispositivo IoT simulado
No se usa un sensor físico: un programa en Python lee un registro ECG real y envía segmentos de 10 s al
servidor por HTTP, autenticado con un token. Primero se registra el dispositivo (muestra el token una sola vez):

```
..\.venv\Scripts\python.exe herramientas\gestionar_dispositivos.py crear ECG-SIM-01 "Dispositivo ECG simulado 01" especialista
```

Con el servidor en marcha, en otra terminal (PowerShell):

```
$env:TOKEN_DISPOSITIVO = "<token mostrado al crear el dispositivo>"
..\.venv\Scripts\python.exe herramientas\dispositivo_simulado.py --archivo ejemplos_mat\registro_prueba_01.mat
```

El avance se ve en la pestaña **Monitoreo IoT**. Más detalles en `app_web/README.md`.

## Reproducir los resultados
Con los datasets en `data_train/` y `data_calfree/`, el flujo del modelo final es:
`05` (división por sujeto) → `23`, `30`, `31` (entrenamiento de las tres CNN) → `35` (ensemble y umbrales en validación)
→ `36` (evaluación final en CalFree) → `37` (paquete para la web) → `40` (figuras).
Las figuras se pueden regenerar sin los datasets: `python scripts\8_figuras\40_figuras_formato_paf.py`.

## Autores
Jiménez García Renzo Jhair · Vásquez Bustamante Gian Anthony
Asesor: Ing. Tuesta Monteza Víctor Alexci
