# TextLab: создание окружения, установка зависимостей, загрузка датасетов, запуск сервера.
# Запуск: правый клик -> "Выполнить с помощью PowerShell" или в консоли: .\start.ps1
param([int]$Port = 8000, [switch]$NoBrowser, [switch]$SkipData)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$venvPy = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'

function Find-Python {
    foreach ($cmd in @(@('py', '-3.12'), @('py', '-3.11'), @('py', '-3'), @('python'))) {
        try {
            $exe = $cmd[0]; $rest = @($cmd | Select-Object -Skip 1)
            $v = & $exe @rest -c "import sys; print(sys.version_info >= (3, 10))" 2>$null
            if ($LASTEXITCODE -eq 0 -and $v -eq 'True') { return ,$cmd }
        } catch {}
    }
    throw 'Не найден Python 3.10+. Установите Python 3.12 с python.org (галочка "Add to PATH").'
}

if (-not (Test-Path $venvPy)) {
    $py = Find-Python
    Write-Host "Создаю окружение .venv ($($py -join ' '))..."
    $exe = $py[0]; $rest = @($py | Select-Object -Skip 1)
    & $exe @rest -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Не удалось создать .venv' }
    # В некоторых сборках Python venv создаётся без pip
    & $venvPy -m pip --version *> $null
    if ($LASTEXITCODE -ne 0) { & $venvPy -m ensurepip --upgrade | Out-Null }
}

# Зависимости ставятся заново только при изменении requirements.txt.
$hash = (Get-FileHash requirements.txt -Algorithm SHA256).Hash
$marker = Join-Path $PSScriptRoot '.venv\.requirements-hash'
if (-not (Test-Path $marker) -or (Get-Content $marker -Raw).Trim() -ne $hash) {
    Write-Host 'Устанавливаю зависимости...'
    & $venvPy -m pip install --disable-pip-version-check -q -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'pip install завершился с ошибкой' }
    Set-Content -Path $marker -Value $hash -Encoding ascii
}

if (-not $SkipData) {
    Write-Host 'Проверяю датасеты...'
    & $venvPy prepare_data.py
}

$url = "http://127.0.0.1:$Port"
Write-Host "TextLab: $url   (API: $url/docs, остановка: Ctrl+C)"
if (-not $NoBrowser) {
    Start-Job -ScriptBlock { param($u) Start-Sleep 3; Start-Process $u } -ArgumentList $url | Out-Null
}
& $venvPy -m uvicorn backend:app --host 127.0.0.1 --port $Port
