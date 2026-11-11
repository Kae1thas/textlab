"""Сквозная проверка API TextLab на синтетических данных.

Запуск из папки проекта:  .venv\\Scripts\\python.exe tests\\smoke_test.py
Тест работает во временном каталоге (TEXTLAB_HOME) и не трогает вашу базу и модели.
Опционально: --db путь\\к\\старой\\textlab.sqlite3 — проверить миграцию копии базы.
"""
import argparse
import csv
import io
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--db', help='копия существующей базы для проверки миграции')
ap.add_argument('--keep', action='store_true', help='не удалять временный каталог')
args = ap.parse_args()

home = Path(tempfile.mkdtemp(prefix='textlab-test-'))
os.environ['TEXTLAB_HOME'] = str(home)
if args.db:
    shutil.copy(args.db, home / 'textlab.sqlite3')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

import backend  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from smoke_data import make_rows  # noqa: E402


def synthetic_csv(n=1500, seed=0) -> bytes:
    texts, labels = make_rows(n, seed)
    buf = io.StringIO()
    w = csv.writer(buf, delimiter=';')
    w.writerow(['Отзыв', 'Категория', 'id'])
    for i, (t, l) in enumerate(zip(texts, labels)):
        w.writerow([t, l, i])
    return buf.getvalue().encode('utf-8')


def wait(client, eid, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        d = client.get(f'/api/experiments/{eid}').json()
        if d['status'] not in ('queued', 'running'):
            return d
        time.sleep(0.3)
    raise TimeoutError(eid)


with TestClient(backend.app) as client:
    assert client.get('/api/health').json()['status'] == 'ok'
    if args.db:
        old = client.get('/api/experiments').json()
        print(f'миграция: {len(old)} экспериментов прочитано')
        for e in [e for e in old if e['status'] == 'completed'][:4]:
            d = client.get(f'/api/experiments/{e["id"]}').json()
            assert d['summary'], d
    models = client.get('/api/models').json()
    assert {m['key'] for m in models['models']} >= {'dummy', 'nb', 'cnb', 'lr', 'svm', 'mlp', 'bert', 'emb_lr'}

    # Загрузка CSV с ';', русскими заголовками и автоопределением столбцов
    r = client.post('/api/datasets/upload', files={'file': ('synthetic.csv', synthetic_csv(), 'text/csv')},
                    data={'name': 'Синтетика'})
    assert r.status_code == 200, r.text
    ds = r.json()
    assert ds['rows'] == 1500 and len(ds['class_counts']) == 4, ds
    print('загрузка:', ds['description'], ds['class_counts'])

    bad = client.post('/api/datasets/upload', files={'file': ('x.csv', b'a,b\n1,2\n', 'text/csv')})
    assert bad.status_code == 400

    # Валидация
    assert client.post('/api/experiments', json={'dataset_id': ds['id'], 'models': ['xx']}).status_code == 400
    assert client.post('/api/experiments', json={'dataset_id': ds['id'], 'repeats': 50}).status_code == 400

    # Серия из 3 повторов, пять моделей, подвыборка 1200 строк
    t0 = time.time()
    r = client.post('/api/experiments', json={'dataset_id': ds['id'], 'models': ['nb', 'cnb', 'lr', 'svm', 'mlp'],
                                              'repeats': 3, 'max_rows': 1200, 'test_size': 0.25, 'seed': 7})
    assert r.status_code == 202, r.text
    eid = r.json()['id']
    d = wait(client, eid)
    assert d['status'] == 'completed', d.get('error')
    print(f'эксперимент: {time.time() - t0:.1f} c, train {d["train_rows"]} / test {d["test_rows"]}, {d["sample_note"]}')
    assert d['train_rows'] + d['test_rows'] == 1200
    assert len(d['runs']) == 15 and len(d['summary']) == 5 and len(d['comparisons']) == 10
    for s in d['summary']:
        f = s['f1_macro']
        assert s['repeats'] == 3 and f['n'] == 3 and f['std'] is not None
        assert s['confusion'] and s['artifact'], s['model']
        print(f"  {s['model']:<26} F1 {f['mean']:.3f} ± {f['std']:.3f}  MCC {s['mcc']['mean']:.3f}  "
              f"train {s['train_seconds']['mean']:.2f}s  warn={len(s['warnings'])}")
    seeds = sorted({r['seed'] for r in d['runs']})
    assert seeds == [7, 8, 9], seeds
    c0 = d['comparisons'][0]
    assert c0['p_value'] is not None and c0['mcnemar_p'] is not None and c0['p_holm'] >= c0['p_value'] - 1e-12
    print('  сравнение:', c0['model_a'], 'vs', c0['model_b'], 'p=%.3f (Holm %.3f), McNemar p=%.3f'
          % (c0['p_value'], c0['p_holm'], c0['mcnemar_p']))

    # Кривая обучения: общий тест, вложенные обучающие выборки, baseline
    assert client.post('/api/experiments', json={'dataset_id': ds['id'], 'models': ['lr'], 'max_rows': 1000,
                                                 'train_sizes': [200, 5000]}).status_code == 400
    r = client.post('/api/experiments', json={'dataset_id': ds['id'], 'models': ['dummy', 'lr', 'svm'], 'repeats': 2,
                                              'max_rows': 1500, 'test_size': 0.2, 'train_sizes': [1000, 200, 500]})
    assert r.status_code == 202, r.text
    dc = wait(client, r.json()['id'])
    assert dc['status'] == 'completed', dc.get('error')
    assert dc['train_sizes'] == [200, 500, 1000] and len(dc['curve']) == 9 and len(dc['runs']) == 18
    assert all(s['repeats'] == 2 for s in dc['summary']) and dc['train_rows'] == 1000
    f = {(c['model_key'], c['train_size']): c['f1_macro']['mean'] for c in dc['curve']}
    print('  кривая LR:', [round(f[('lr', n)], 3) for n in (200, 500, 1000)], ' baseline:', round(f[('dummy', 1000)], 3))
    assert f[('lr', 1000)] > f[('lr', 200)] and f[('dummy', 1000)] < 0.3
    dummy = next(s for s in dc['summary'] if s['model_key'] == 'dummy')
    assert dummy['accuracy']['mean'] > 0.4  # доля самого частого класса ~45 %
    ins = client.post('/api/inspect', json={'experiment_id': dc['id'], 'text': 'гол'}).json()
    assert any(x['model'].startswith('Baseline') and x['prediction'] == 'спорт' for x in ins['results'])
    assert client.delete(f'/api/experiments/{dc["id"]}').json()['artifacts_removed'] == 3

    lst = client.get('/api/experiments').json()
    assert lst[0]['id'] == eid and lst[0]['n_models'] == 5

    for kind in ('runs', 'summary', 'comparisons'):
        r = client.get(f'/api/experiments/{eid}/export.csv?kind={kind}')
        assert r.status_code == 200 and r.text.startswith('﻿'), kind

    ins = client.post('/api/inspect', json={'experiment_id': eid, 'text': 'команда забила гол в матче чемпионата'}).json()
    assert len(ins['results']) == 5
    preds = {x['model']: x.get('prediction') for x in ins['results']}
    print('  проверка текста:', preds)
    assert sum(p == 'спорт' for p in preds.values()) >= 4

    an = client.get('/api/analysis').json()
    assert any(a['dataset'] == 'Синтетика' for a in an)

    # Отмена: ставим длинный эксперимент и сразу останавливаем
    r = client.post('/api/experiments', json={'dataset_id': ds['id'], 'models': ['lr', 'svm'], 'repeats': 20, 'max_rows': 1500})
    eid2 = r.json()['id']
    time.sleep(0.5)
    client.post(f'/api/experiments/{eid2}/cancel')
    d2 = wait(client, eid2)
    assert d2['status'] == 'cancelled', d2['status']
    print('отмена: ok, успело', len(d2['runs']), 'прогонов')

    # Трансформер без torch даёт понятную ошибку, а не падение сервера
    bert = next(m for m in models['models'] if m['key'] == 'bert')
    if not bert['available']:
        r = client.post('/api/experiments', json={'dataset_id': ds['id'], 'models': ['bert']})
        d3 = wait(client, r.json()['id'])
        assert d3['status'] == 'failed' and 'PyTorch' in d3['error'], d3
        print('без torch:', d3['error'])

    assert client.delete(f'/api/experiments/{eid}').json()['artifacts_removed'] == 5
    assert not list((home / 'model_store').glob(f'{eid}*'))
    assert client.delete(f'/api/datasets/{ds["id"]}').status_code == 200

print('OK — все проверки пройдены')
if not args.keep:
    shutil.rmtree(home, ignore_errors=True)
else:
    print('каталог теста:', home)
