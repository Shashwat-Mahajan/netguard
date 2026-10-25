@echo off
setlocal
if "%~1"=="" (
  echo usage: scripts\run_zeek.bat data\file.pcap
  exit /b 1
)
if not exist "%~1" (
  echo not found: %~1
  exit /b 1
)
if not exist data\zeek mkdir data\zeek
del /q data\zeek\*.log 2>nul
set "P=%~1"
set "P=%P:\=/%"
docker run --rm -v "%CD%:/work" -w /work/data/zeek zeek/zeek:latest zeek -C -r /work/%P% LogAscii::use_json=T protocols/ssl/validate-certs protocols/ssh/detect-bruteforcing protocols/ftp/detect-bruteforcing protocols/http/detect-sql-injection
echo.
echo done: data\zeek
dir /b data\zeek