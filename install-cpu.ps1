# Установка PyTorch (только CPU, ~200 МБ) и transformers — для компьютера без видеокарты NVIDIA.
# Нужна, чтобы в «Проверке текста» работали две BERT-модели. Обучать BERT на CPU не стоит — это очень долго.
# Сначала один раз запустите start.ps1, чтобы создалось окружение .venv.
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$venvPy = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $venvPy)) { throw 'Нет .venv — сначала запустите start.ps1' }

Write-Host 'Устанавливаю PyTorch (CPU)...'
& $venvPy -m pip install --disable-pip-version-check torch --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { throw 'Не удалось установить PyTorch' }
& $venvPy -m pip install --disable-pip-version-check -r requirements-gpu.txt
if ($LASTEXITCODE -ne 0) { throw 'Не удалось установить transformers' }
& $venvPy -c "import torch, transformers; print('PyTorch', torch.__version__, '| transformers', transformers.__version__)"
Write-Host 'Готово. Перезапустите start.ps1.'
