@echo off
rem Usage: scripts\update_et_rules.bat
rem Downloads the ET Open ruleset (needs internet) into data\et\suricata.rules
if not exist data\et mkdir data\et
docker run --rm --entrypoint suricata-update ^
  -v "%cd%\data\et:/et" ^
  jasonish/suricata:latest -o /et --no-test --no-reload
echo.
if exist data\et\suricata.rules (echo done: data\et\suricata.rules) else (echo no rules produced)