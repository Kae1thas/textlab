# Сквозной тест API на синтетических данных (во временной папке, рабочая база и модели не затрагиваются).
Set-Location $PSScriptRoot
$env:PYTHONIOENCODING = 'utf-8'
$py = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $py)) { throw 'Нет .venv — сначала запустите start.ps1' }
& $py -m pip install -q --disable-pip-version-check -r requirements-dev.txt
& $py tests\smoke_test.py @args
