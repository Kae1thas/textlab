"""Проверка трансформерных моделей (после install-gpu.ps1).

Запуск: .venv\\Scripts\\python.exe tests\\transformer_check.py
Скачивает cointegrated/rubert-tiny2 (~120 МБ) при первом запуске и обучает
emb_lr и bert на 600 синтетических текстах. На RTX-видеокарте ~1 минута.
"""
import os
import sys
import tempfile
import threading
from pathlib import Path

os.environ['TEXTLAB_HOME'] = tempfile.mkdtemp(prefix='textlab-tr-')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import core  # noqa: E402
import neural  # noqa: E402
from smoke_data import make_rows  # noqa: E402

ok, reason = neural.available()
print('PyTorch:', ok, reason)
print(neural.environment())
assert ok, reason

texts, labels = make_rows(600)
path = Path(os.environ['TEXTLAB_HOME']) / 'synthetic.csv'
import csv  # noqa: E402
with open(path, 'w', encoding='utf-8', newline='') as f:
    w = csv.writer(f)
    w.writerow(['text', 'label'])
    w.writerows(zip(texts, labels))
ds = {'id': 't', 'name': 't', 'kind': 'csv', 'path': str(path)}
cfg = core.RunConfig(dataset=ds, models=['svm', 'emb_lr', 'bert'], repeats=2, max_rows=600,
                     transformer_epochs=2, transformer_max_length=64)
rows = []
out = core.run_experiment(cfg, 'trcheck', core.RunCallbacks(
    progress=lambda f, m: print(f'  {f:5.0%} {m}'), result=rows.append, cancel=threading.Event()))
for r in rows:
    print(f"{r['model']:<28} rep {r['repeat']} F1 {r['f1_macro']:.3f} train {r['train_seconds']:.1f}s "
          f"GPU {r['gpu_peak_mb'] or 0:.0f} MB")
art = core.MODELS_DIR / 'trcheck_bert'
m = neural.load_artifact(art)
print('загрузка сохранённой модели:', m.predict(['команда забила гол в матче']))
print('OK')
