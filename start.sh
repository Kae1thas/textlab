#!/usr/bin/env bash
# TextLab для Linux/macOS: создание окружения, установка зависимостей, загрузка датасетов, запуск сервера.
# Использование: ./start.sh [порт]      Пропустить загрузку данных: SKIP_DATA=1 ./start.sh
set -euo pipefail
cd "$(dirname "$0")"
PORT="${1:-8000}"
PY="${PYTHON:-python3}"

if [ ! -x .venv/bin/python ]; then
  echo "Создаю окружение .venv..."
  "$PY" -m venv .venv
  .venv/bin/python -m pip --version >/dev/null 2>&1 || .venv/bin/python -m ensurepip --upgrade >/dev/null
fi

HASH="$( (sha256sum requirements.txt 2>/dev/null || shasum -a 256 requirements.txt) | cut -d' ' -f1)"
if [ "$(cat .venv/.requirements-hash 2>/dev/null || true)" != "$HASH" ]; then
  echo "Устанавливаю зависимости..."
  .venv/bin/python -m pip install --disable-pip-version-check -q -r requirements.txt
  echo "$HASH" > .venv/.requirements-hash
fi

if [ -z "${SKIP_DATA:-}" ]; then
  echo "Проверяю датасеты..."
  .venv/bin/python prepare_data.py
fi

echo "TextLab: http://127.0.0.1:$PORT   (API: /docs, остановка: Ctrl+C)"
exec .venv/bin/python -m uvicorn backend:app --host 127.0.0.1 --port "$PORT"
