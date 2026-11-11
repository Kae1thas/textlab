# Установка PyTorch с CUDA и transformers для моделей "BERT fine-tuning" и "Эмбеддинги BERT + LogReg".
# Для RTX 50xx (Blackwell) нужны сборки под CUDA 12.8+, поэтому индекс cu128.
# Сначала один раз запустите start.ps1, чтобы создалось окружение .venv.
param([string]$Cuda = 'cu128')
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$venvPy = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $venvPy)) { throw 'Нет .venv — сначала запустите start.ps1' }

Write-Host "Устанавливаю PyTorch ($Cuda), это ~3 ГБ..."
& $venvPy -m pip install --disable-pip-version-check torch --index-url "https://download.pytorch.org/whl/$Cuda"
if ($LASTEXITCODE -ne 0) { throw 'Не удалось установить PyTorch' }
& $venvPy -m pip install --disable-pip-version-check -r requirements-gpu.txt
if ($LASTEXITCODE -ne 0) { throw 'Не удалось установить transformers' }

& $venvPy -c "import torch; print('PyTorch', torch.__version__, '| CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'GPU не найден')"
Write-Host ''
Write-Host 'Проверка обучения на синтетике (скачает rubert-tiny2 ~120 МБ):'
Write-Host '  .venv\Scripts\python.exe tests\transformer_check.py'
Write-Host 'Перезапустите start.ps1 — модели-трансформеры станут доступны в интерфейсе.'
