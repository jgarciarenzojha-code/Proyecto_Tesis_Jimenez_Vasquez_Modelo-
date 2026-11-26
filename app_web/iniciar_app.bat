@echo off
REM Inicia la aplicacion web usando el entorno virtual del proyecto.
REM Luego abrir en el navegador: http://localhost:8001

cd /d "%~dp0"
echo Iniciando servidor... (Ctrl+C para detener)
echo Abrir en el navegador: http://localhost:8001
"..\.venv\Scripts\python.exe" -m uvicorn backend.main:app --host 127.0.0.1 --port 8001
pause
