@echo off
rem Usage: scripts\run_suricata.bat data\test.pcap
rem Needs Docker Desktop (Linux containers). Output: data\suricata\eve.json
if "%~1"=="" (echo usage: run_suricata.bat file.pcap & exit /b 1)
if not exist data\suricata mkdir data\suricata
del /q data\suricata\eve.json 2>nul
docker run --rm ^
  -v "%cd%\%~1:/pcap/in.pcap:ro" ^
  -v "%cd%\config\suricata:/rules:ro" ^
  -v "%cd%\data\suricata:/logs" ^
    jasonish/suricata:latest -r /pcap/in.pcap -S /rules/local.rules -k none -l /logs ^
  --set app-layer.protocols.tls.ja3-fingerprints=yes
echo.
if exist data\suricata\eve.json (echo done: data\suricata\eve.json) else (echo no eve.json produced)