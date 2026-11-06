@echo off
rem Abre la pagina web del proyecto en http://localhost:8000
rem Doble clic para iniciar. Cierra esta ventana para detener el servidor.

cd /d "%~dp0docs"
echo.
echo  Pagina de la GAN en http://localhost:8000
echo  (no cierres esta ventana mientras la estes viendo)
echo.

rem Abre el navegador 2 segundos despues, cuando el servidor ya esta listo
start "" powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 2; Start-Process 'http://localhost:8000'"

rem En este computador "python" abre la Microsoft Store; "py" es el lanzador real
where py >nul 2>nul
if %errorlevel%==0 (
    py -m http.server 8000
) else (
    python -m http.server 8000
)
pause
