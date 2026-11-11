"""HTTP-сервер TextLab: FastAPI + SQLite + фоновая очередь экспериментов."""
from __future__ import annotations

import csv
import io
import json
import queue
import shutil
import sqlite3
import threading
import traceback
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

import core
from core import DATA, HOME, MODELS_DIR, MODEL_REGISTRY, ROOT, UPLOADS

DB = HOME / 'textlab.sqlite3'
for p in (DATA, UPLOADS, MODELS_DIR, core.CACHE):
    p.mkdir(parents=True, exist_ok=True)


def now():
    return datetime.now(timezone.utc).isoformat()


# ------------------------------------------------------------------ БД

def conn():
    c = sqlite3.connect(DB, timeout=30)
    c.row_factory = sqlite3.Row
    c.execute('PRAGMA foreign_keys=ON')
    return c


SCHEMA = '''
CREATE TABLE IF NOT EXISTS datasets(
  id TEXT PRIMARY KEY, name TEXT NOT NULL, path TEXT NOT NULL, source TEXT, rows INTEGER,
  classes TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS experiments(
  id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL, dataset_name TEXT NOT NULL, created_at TEXT,
  seed INTEGER, test_size REAL, train_rows INTEGER, test_rows INTEGER, status TEXT, ram_delta_mb REAL);
CREATE TABLE IF NOT EXISTS results(
  id INTEGER PRIMARY KEY AUTOINCREMENT, experiment_id TEXT, model TEXT, accuracy REAL,
  precision_macro REAL, recall_macro REAL, f1_macro REAL, f1_weighted REAL, train_seconds REAL,
  inference_ms REAL, model_bytes INTEGER, confusion TEXT, labels TEXT, artifact TEXT);
CREATE TABLE IF NOT EXISTS comparisons(
  id INTEGER PRIMARY KEY AUTOINCREMENT, experiment_id TEXT NOT NULL, model_a TEXT, model_b TEXT,
  key_a TEXT, key_b TEXT, metric TEXT, mean_diff REAL, repeats INTEGER, t_stat REAL, p_value REAL,
  p_holm REAL, mcnemar_b INTEGER, mcnemar_c INTEGER, mcnemar_p REAL, mcnemar_p_holm REAL);
CREATE INDEX IF NOT EXISTS idx_results_exp ON results(experiment_id);
CREATE INDEX IF NOT EXISTS idx_cmp_exp ON comparisons(experiment_id);
'''

# Колонки, добавленные после первой версии: старые базы дополняются ALTER TABLE.
MIGRATIONS = {
    'datasets': {'kind': 'TEXT', 'language': 'TEXT', 'description': 'TEXT', 'class_counts': 'TEXT'},
    'experiments': {'repeats': 'INTEGER DEFAULT 1', 'max_rows': 'INTEGER', 'models': 'TEXT', 'config': 'TEXT',
                    'progress': 'REAL', 'message': 'TEXT', 'error': 'TEXT', 'started_at': 'TEXT',
                    'finished_at': 'TEXT', 'environment': 'TEXT', 'sample_note': 'TEXT', 'class_counts': 'TEXT'},
    'results': {'model_key': 'TEXT', 'repeat': 'INTEGER DEFAULT 0', 'seed': 'INTEGER',
                'balanced_accuracy': 'REAL', 'mcc': 'REAL', 'kappa': 'REAL', 'vectorize_seconds': 'REAL',
                'peak_ram_mb': 'REAL', 'gpu_peak_mb': 'REAL', 'params': 'TEXT', 'warnings': 'TEXT',
                'per_class': 'TEXT', 'train_size': 'INTEGER'},
}

NAME_TO_KEY = {
    'Multinomial Naive Bayes': 'nb', 'Logistic Regression': 'lr', 'Linear SVM': 'svm', 'MLP Neural Network': 'mlp',
}


def init_db():
    with conn() as c:
        c.execute('PRAGMA journal_mode=WAL')
        c.executescript(SCHEMA)
        for table, cols in MIGRATIONS.items():
            have = {r['name'] for r in c.execute(f'PRAGMA table_info({table})')}
            for col, typ in cols.items():
                if col not in have:
                    c.execute(f'ALTER TABLE {table} ADD COLUMN "{col}" {typ}')
        # Записи первой версии: один повтор, ключ модели по имени.
        for name, key in NAME_TO_KEY.items():
            c.execute('UPDATE results SET model_key=? WHERE model_key IS NULL AND model=?', (key, name))
        c.execute('UPDATE results SET "repeat"=0 WHERE "repeat" IS NULL')
        c.execute('UPDATE results SET seed=(SELECT seed FROM experiments e WHERE e.id=results.experiment_id) WHERE seed IS NULL')
        c.execute('UPDATE experiments SET repeats=1 WHERE repeats IS NULL')
        c.execute("UPDATE experiments SET progress=1 WHERE status='completed' AND progress IS NULL")
        c.execute('UPDATE results SET train_size=(SELECT train_rows FROM experiments e WHERE e.id=results.experiment_id) '
                  'WHERE train_size IS NULL')
        # Задачи, прерванные остановкой сервера.
        c.execute("UPDATE experiments SET status='interrupted', message='Сервер был остановлен во время обучения' "
                  "WHERE status IN ('queued','running')")


def register_builtin_datasets():
    with conn() as c:
        for d in core.BUILTIN_DATASETS:
            fp = DATA / d['file']
            if not fp.exists():
                c.execute('DELETE FROM datasets WHERE id=?', (d['id'],))
                continue
            row = c.execute('SELECT path, class_counts FROM datasets WHERE id=?', (d['id'],)).fetchone()
            if row and row['class_counts'] and row['path'] == str(fp):
                c.execute('UPDATE datasets SET name=?, source=?, kind=?, language=?, description=? WHERE id=?',
                          (d['name'], d['source'], d['kind'], d['language'], d['description'], d['id']))
                continue
            if d['kind'] == 'lenta':
                prof = core.lenta_profile()
                counts = prof['class_counts'] if prof else {}
                rows = prof['eligible_rows'] if prof else None
            else:
                _, labels = core.read_text_label_csv(fp)
                counts = core.class_counts(labels)
                rows = len(labels)
            c.execute('INSERT OR REPLACE INTO datasets(id,name,path,source,rows,classes,created_at,kind,language,description,class_counts) '
                      'VALUES(?,?,?,?,?,?,?,?,?,?,?)',
                      (d['id'], d['name'], str(fp), d['source'], rows, json.dumps(list(counts), ensure_ascii=False),
                       now(), d['kind'], d['language'], d['description'], json.dumps(counts, ensure_ascii=False)))
        # Пользовательские загрузки первой версии без счётчиков классов.
        for r in c.execute("SELECT id, path FROM datasets WHERE class_counts IS NULL").fetchall():
            if Path(r['path']).exists():
                _, labels = core.read_text_label_csv(Path(r['path']))
                c.execute("UPDATE datasets SET class_counts=?, kind='csv', source=COALESCE(source,'Пользовательская загрузка') WHERE id=?",
                          (json.dumps(core.class_counts(labels), ensure_ascii=False), r['id']))


def dataset_row(r) -> dict:
    d = dict(r)
    d['class_counts'] = json.loads(d['class_counts']) if d.get('class_counts') else {}
    d['classes'] = list(d['class_counts']) or json.loads(d.get('classes') or '[]')
    d['kind'] = d.get('kind') or 'csv'
    d['builtin'] = any(b['id'] == d['id'] for b in core.BUILTIN_DATASETS)
    d['available'] = Path(d['path']).exists()
    return d


def get_dataset(did: str) -> dict:
    with conn() as c:
        r = c.execute('SELECT * FROM datasets WHERE id=?', (did,)).fetchone()
    if not r:
        raise HTTPException(404, 'Датасет не найден')
    return dataset_row(r)


# ------------------------------------------------------------------ очередь

JOBS: queue.Queue = queue.Queue()
CANCEL: dict[str, threading.Event] = {}
CURRENT: dict[str, str | None] = {'id': None}


def set_status(eid, **fields):
    keys = ', '.join(f'{k}=?' for k in fields)
    with conn() as c:
        c.execute(f'UPDATE experiments SET {keys} WHERE id=?', (*fields.values(), eid))


def save_result(eid: str, row: dict):
    with conn() as c:
        c.execute('''INSERT INTO results(experiment_id, model, model_key, "repeat", seed, accuracy, balanced_accuracy,
                     precision_macro, recall_macro, f1_macro, f1_weighted, mcc, kappa, train_seconds,
                     vectorize_seconds, inference_ms, peak_ram_mb, gpu_peak_mb, model_bytes, confusion, labels,
                     artifact, params, warnings, per_class, train_size)
                     VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                  (eid, row['model'], row['model_key'], row['repeat'], row['seed'], row['accuracy'],
                   row['balanced_accuracy'], row['precision_macro'], row['recall_macro'], row['f1_macro'],
                   row['f1_weighted'], row['mcc'], row['kappa'], row['train_seconds'], row.get('vectorize_seconds'),
                   row['inference_ms'], row.get('peak_ram_mb'), row.get('gpu_peak_mb'), row.get('model_bytes'),
                   json.dumps(row['confusion']) if row.get('confusion') is not None else None,
                   json.dumps(row['labels'], ensure_ascii=False), row.get('artifact'),
                   json.dumps(row.get('params'), ensure_ascii=False, default=str),
                   json.dumps(row.get('warnings') or [], ensure_ascii=False),
                   json.dumps(row['per_class'], ensure_ascii=False) if row.get('per_class') else None,
                   row.get('train_size')))


def worker():
    while True:
        eid = JOBS.get()
        cancel = CANCEL.setdefault(eid, threading.Event())
        try:
            with conn() as c:
                e = c.execute('SELECT * FROM experiments WHERE id=?', (eid,)).fetchone()
            if e is None or cancel.is_set():
                if e is not None:
                    set_status(eid, status='cancelled', message='Отменён до запуска', finished_at=now())
                continue
            CURRENT['id'] = eid
            cfg_raw = json.loads(e['config'])
            ds = get_dataset(e['dataset_id'])
            cfg = core.RunConfig(dataset=ds, **cfg_raw)
            set_status(eid, status='running', started_at=now(), progress=0.0, message='Запуск')

            def progress(frac, msg):
                set_status(eid, progress=round(frac, 4), message=msg)

            out = core.run_experiment(cfg, eid, core.RunCallbacks(
                progress=progress, result=lambda row: save_result(eid, row), cancel=cancel))
            with conn() as c:
                for r in out['comparisons']:
                    c.execute('''INSERT INTO comparisons(experiment_id, model_a, model_b, key_a, key_b, metric, mean_diff,
                                 repeats, t_stat, p_value, p_holm, mcnemar_b, mcnemar_c, mcnemar_p, mcnemar_p_holm)
                                 VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                              (eid, r['model_a'], r['model_b'], r['key_a'], r['key_b'], r['metric'], r['mean_diff'],
                               r['repeats'], r['t_stat'], r['p_value'], r['p_holm'], r.get('mcnemar_b'),
                               r.get('mcnemar_c'), r.get('mcnemar_p'), r.get('mcnemar_p_holm')))
            set_status(eid, status='completed', progress=1.0, message='Готово', finished_at=now(),
                       train_rows=out['train_rows'], test_rows=out['test_rows'], sample_note=out['sample_note'],
                       class_counts=json.dumps(out['class_counts'], ensure_ascii=False))
        except core.Cancelled:
            set_status(eid, status='cancelled', message='Остановлен пользователем', finished_at=now())
        except Exception as exc:
            if not isinstance(exc, ValueError):
                traceback.print_exc()
            msg = str(exc) if isinstance(exc, ValueError) else f'{type(exc).__name__}: {exc}'
            set_status(eid, status='failed', error=msg[:2000], message='Ошибка', finished_at=now())
        finally:
            CURRENT['id'] = None
            CANCEL.pop(eid, None)
            JOBS.task_done()


@asynccontextmanager
async def lifespan(app):
    init_db()
    register_builtin_datasets()
    threading.Thread(target=worker, daemon=True, name='textlab-worker').start()
    yield


app = FastAPI(title='TextLab', version='2.0.0', lifespan=lifespan,
              description='Сравнительный анализ алгоритмов машинного обучения в задачах классификации текстов')


# ------------------------------------------------------------------ страницы

STATIC_DIR = ROOT / 'static'
STATIC = {'styles.css': 'text/css', 'backend-ui.js': 'text/javascript'}


@app.get('/', include_in_schema=False)
def home():
    # no-cache: после обновления версии браузер не возьмёт старые скрипты из кэша
    return FileResponse(STATIC_DIR / 'index.html', headers={'Cache-Control': 'no-cache'})


@app.get('/assets/{name}', include_in_schema=False)
def asset(name: str):
    # Отдаём только файлы интерфейса, а не всю папку проекта (там база и модели).
    if name not in STATIC:
        raise HTTPException(404)
    return FileResponse(STATIC_DIR / name, media_type=STATIC[name], headers={'Cache-Control': 'no-cache'})


# ------------------------------------------------------------------ датасеты

@app.get('/api/datasets')
def datasets():
    with conn() as c:
        rows = c.execute("SELECT * FROM datasets ORDER BY (source='Пользовательская загрузка'), name").fetchall()
    return [dataset_row(r) for r in rows]


@app.get('/api/datasets/missing')
def missing_datasets():
    """Встроенные наборы, которых ещё нет в data/ (подсказка для интерфейса)."""
    return [{'id': d['id'], 'name': d['name'], 'file': d['file']} for d in core.BUILTIN_DATASETS
            if not (DATA / d['file']).exists()]


@app.post('/api/datasets/upload')
async def upload_dataset(file: UploadFile = File(...), text_column: str = Form(''), label_column: str = Form(''),
                         name: str = Form('')):
    raw = await file.read()
    try:
        try:
            dec = raw.decode('utf-8-sig')
        except UnicodeDecodeError:
            dec = raw.decode('cp1251')
        dialect = core.sniff_dialect(dec[:20000])
        reader = csv.DictReader(io.StringIO(dec), dialect=dialect)
        headers = reader.fieldnames or []
        tcol = text_column or core.guess_column(headers, core.TEXT_ALIASES)
        lcol = label_column or core.guess_column(headers, core.LABEL_ALIASES)
        if not tcol or not lcol or tcol not in headers or lcol not in headers:
            raise ValueError(f'Не найдены столбцы текста и метки. Столбцы файла: {", ".join(headers)}')
        if tcol == lcol:
            raise ValueError('Столбцы текста и метки должны различаться')
        rows = [((r.get(tcol) or '').strip(), (r.get(lcol) or '').strip()) for r in reader]
        rows = [r for r in rows if r[0] and r[1]]
        counts = core.class_counts(l for _, l in rows)
        if len(rows) < 50 or len(counts) < 2:
            raise ValueError('Нужно минимум 50 размеченных строк и два класса')
        if len(counts) > 200:
            raise ValueError(f'Слишком много классов ({len(counts)}). Похоже, выбран не тот столбец метки.')
    except ValueError as e:
        raise HTTPException(400, str(e))
    did = 'u' + uuid.uuid4().hex[:9]
    path = UPLOADS / f'{did}.csv'
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['text', 'label'])
        w.writerows(rows)
    title = name.strip() or Path(file.filename or 'dataset.csv').stem
    with conn() as c:
        c.execute('INSERT INTO datasets(id,name,path,source,rows,classes,created_at,kind,language,description,class_counts) '
                  'VALUES(?,?,?,?,?,?,?,?,?,?,?)',
                  (did, title, str(path), 'Пользовательская загрузка', len(rows),
                   json.dumps(list(counts), ensure_ascii=False), now(), 'csv', None,
                   f'Файл {file.filename}: столбцы «{tcol}» → текст, «{lcol}» → метка',
                   json.dumps(counts, ensure_ascii=False)))
    return get_dataset(did)


@app.delete('/api/datasets/{did}')
def delete_dataset(did: str):
    d = get_dataset(did)
    if d['builtin']:
        raise HTTPException(400, 'Встроенный набор удаляется удалением файла из data/')
    with conn() as c:
        c.execute('DELETE FROM datasets WHERE id=?', (did,))
    Path(d['path']).unlink(missing_ok=True)
    return {'deleted': did}


# ------------------------------------------------------------------ модели

@app.get('/api/models')
def models():
    try:
        import neural
        ok, reason = neural.available()
    except Exception as e:
        ok, reason = False, str(e)
    out = []
    for key, m in MODEL_REGISTRY.items():
        needs = m.get('requires') == 'torch'
        out.append({'key': key, 'name': m['name'], 'family': m['family'], 'features': m['features'],
                    'params': m['params'], 'note': m.get('note'), 'max_train_rows': m.get('max_train_rows'),
                    'available': ok if needs else True, 'reason': reason if needs else None})
    return {'models': out, 'transformer_presets': core.TRANSFORMER_PRESETS}


@app.get('/api/system')
def system():
    return core.environment_info()


# ------------------------------------------------------------------ эксперименты

class RunSpec(BaseModel):
    dataset_id: str
    models: list[str] = Field(default_factory=lambda: ['nb', 'lr', 'svm'])
    test_size: float = 0.2
    seed: int = 42
    repeats: int = 1
    max_rows: int = 50_000
    transformer_model: str = 'cointegrated/rubert-tiny2'
    transformer_epochs: int = 3
    transformer_max_length: int = 256
    train_sizes: list[int] | None = None


@app.post('/api/experiments', status_code=202)
def create_experiment(spec: RunSpec):
    ds = get_dataset(spec.dataset_id)
    if not ds['available']:
        raise HTTPException(400, 'Файл датасета не найден на диске')
    if not 0.1 <= spec.test_size <= 0.4:
        raise HTTPException(400, 'Доля теста должна быть от 0.1 до 0.4')
    if not 500 <= spec.max_rows <= 500_000:
        raise HTTPException(400, 'Объём выборки должен быть от 500 до 500 000 строк')
    if not 1 <= spec.repeats <= 20:
        raise HTTPException(400, 'Число повторов — от 1 до 20')
    if not 0 <= spec.seed <= 2**31 - 100:
        raise HTTPException(400, 'Недопустимый seed')
    if not 1 <= spec.transformer_epochs <= 20 or not 16 <= spec.transformer_max_length <= 512:
        raise HTTPException(400, 'Эпох: 1–20, длина последовательности: 16–512')
    if spec.train_sizes:
        sizes = sorted(set(spec.train_sizes))
        if len(sizes) < 2 or len(sizes) > 12 or sizes[0] < 20:
            raise HTTPException(400, 'Кривая обучения: от 2 до 12 размеров, каждый не меньше 20')
        avail = int(min(spec.max_rows, ds['rows'] or spec.max_rows) * (1 - spec.test_size))
        if sizes[-1] > avail:
            raise HTTPException(400, f'Наибольший размер обучения {sizes[-1]:,} больше обучающей части выборки (~{avail:,}). '
                                     'Увеличьте «Строк в выборке».'.replace(',', ' '))
        spec.train_sizes = sizes
    models_ = list(dict.fromkeys(spec.models))
    unknown = [m for m in models_ if m not in MODEL_REGISTRY]
    if unknown or not models_:
        raise HTTPException(400, f'Неизвестные модели: {unknown}' if unknown else 'Выберите хотя бы одну модель')
    for m in models_:
        lim = MODEL_REGISTRY[m].get('max_train_rows')
        rows_avail = min(spec.max_rows, ds['rows'] or spec.max_rows)
        largest = spec.train_sizes[-1] if spec.train_sizes else rows_avail * (1 - spec.test_size)
        if lim and largest > lim:
            raise HTTPException(400, f'{MODEL_REGISTRY[m]["name"]}: не более {lim:,} обучающих строк'.replace(',', ' '))
    eid = uuid.uuid4().hex
    cfg = spec.model_dump(exclude={'dataset_id'})
    cfg['models'] = models_
    with conn() as c:
        c.execute('''INSERT INTO experiments(id, dataset_id, dataset_name, created_at, seed, test_size, status,
                     repeats, max_rows, models, config, progress, message, environment)
                     VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                  (eid, ds['id'], ds['name'], now(), spec.seed, spec.test_size, 'queued', spec.repeats,
                   spec.max_rows, json.dumps(models_), json.dumps(cfg, ensure_ascii=False), 0.0, 'В очереди',
                   json.dumps(core.environment_info(), ensure_ascii=False)))
    CANCEL[eid] = threading.Event()
    JOBS.put(eid)
    return {'id': eid, 'status': 'queued'}


METRICS = core.METRIC_KEYS + ['train_seconds', 'inference_ms', 'peak_ram_mb', 'gpu_peak_mb']


def aggregate(results: list[dict]) -> list[dict]:
    by: dict[str, list[dict]] = {}
    for r in results:
        by.setdefault(r['model_key'] or r['model'], []).append(r)
    out = []
    for key, rows in by.items():
        rows.sort(key=lambda r: r['repeat'] or 0)
        first = rows[0]
        s = {'model_key': key, 'model': first['model'], 'repeats': len(rows),
             'model_bytes': first.get('model_bytes'), 'artifact': first.get('artifact'),
             'warnings': sorted({w for r in rows for w in json.loads(r.get('warnings') or '[]')}),
             'params': json.loads(first['params']) if first.get('params') else None,
             'labels': json.loads(first['labels']) if first.get('labels') else [],
             'confusion': json.loads(first['confusion']) if first.get('confusion') else None,
             'per_class': json.loads(first['per_class']) if first.get('per_class') else None}
        for m in METRICS:
            s[m] = core.summarize([r.get(m) for r in rows])
        out.append(s)
    out.sort(key=lambda s: -(s['f1_macro']['mean'] or 0))
    return out


def experiment_brief(e) -> dict:
    d = dict(e)
    d['models'] = json.loads(d['models']) if d.get('models') else None
    d['config'] = json.loads(d['config']) if d.get('config') else None
    for k in ('environment', 'class_counts'):
        d[k] = json.loads(d[k]) if d.get(k) else None
    return d


@app.get('/api/experiments')
def list_experiments(limit: int = 200):
    with conn() as c:
        exps = c.execute('SELECT * FROM experiments ORDER BY created_at DESC LIMIT ?', (limit,)).fetchall()
        out = []
        for e in exps:
            d = experiment_brief(e)
            best = c.execute('''SELECT model, AVG(f1_macro) f1, COUNT(*) n FROM results WHERE experiment_id=?
                                AND COALESCE(train_size, 0)=(SELECT COALESCE(MAX(train_size), 0) FROM results WHERE experiment_id=?)
                                GROUP BY COALESCE(model_key, model) ORDER BY f1 DESC''', (e['id'], e['id'])).fetchall()
            d['n_models'] = len(best)
            d['best_model'] = best[0]['model'] if best else None
            d['best_f1'] = best[0]['f1'] if best else None
            d.pop('environment', None)
            out.append(d)
    return out


@app.get('/api/experiments/{eid}')
def get_experiment(eid: str):
    with conn() as c:
        e = c.execute('SELECT * FROM experiments WHERE id=?', (eid,)).fetchone()
        if not e:
            raise HTTPException(404, 'Эксперимент не найден')
        results = [dict(r) for r in c.execute('SELECT * FROM results WHERE experiment_id=? ORDER BY "repeat", id', (eid,))]
        cmps = [dict(r) for r in c.execute('SELECT * FROM comparisons WHERE experiment_id=? ORDER BY id', (eid,))]
    d = experiment_brief(e)
    sizes = sorted({r['train_size'] for r in results if r.get('train_size')})
    top = sizes[-1] if sizes else None
    d['train_sizes'] = sizes
    d['summary'] = aggregate([r for r in results if top is None or r.get('train_size') in (top, None)])
    # Кривая обучения: метрики по каждому размеру обучающей выборки.
    d['curve'] = []
    if len(sizes) > 1:
        for size in sizes:
            for s in aggregate([r for r in results if r.get('train_size') == size]):
                d['curve'].append({'train_size': size, 'model_key': s['model_key'], 'model': s['model'],
                                   **{m: s[m] for m in METRICS}})
    d['runs'] = [{k: r.get(k) for k in ['model', 'model_key', 'repeat', 'seed', 'train_size', *METRICS]} for r in results]
    d['comparisons'] = cmps
    d['queue_position'] = None
    if d['status'] == 'queued':
        d['queue_position'] = list(JOBS.queue).index(eid) + 1 if eid in list(JOBS.queue) else None
    return d


@app.post('/api/experiments/{eid}/cancel')
def cancel_experiment(eid: str):
    ev = CANCEL.get(eid)
    if ev is None:
        raise HTTPException(400, 'Эксперимент не выполняется')
    ev.set()
    return {'id': eid, 'cancelling': True}


@app.delete('/api/experiments/{eid}')
def delete_experiment(eid: str):
    if CURRENT['id'] == eid:
        raise HTTPException(400, 'Сначала остановите эксперимент')
    with conn() as c:
        arts = [r['artifact'] for r in c.execute('SELECT artifact FROM results WHERE experiment_id=?', (eid,)) if r['artifact']]
        c.execute('DELETE FROM results WHERE experiment_id=?', (eid,))
        c.execute('DELETE FROM comparisons WHERE experiment_id=?', (eid,))
        n = c.execute('DELETE FROM experiments WHERE id=?', (eid,)).rowcount
    if eid in CANCEL:
        CANCEL[eid].set()
    for a in arts:
        p = MODELS_DIR / Path(a).name
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        else:
            p.unlink(missing_ok=True)
    if not n:
        raise HTTPException(404, 'Эксперимент не найден')
    return {'deleted': eid, 'artifacts_removed': len(arts)}


@app.get('/api/experiments/{eid}/export.csv')
def export_experiment(eid: str, kind: str = 'runs'):
    d = get_experiment(eid)
    buf = io.StringIO()
    w = csv.writer(buf, delimiter=';')
    if kind == 'summary':
        w.writerow(['model', 'repeats'] + [f'{m}_{s}' for m in METRICS for s in ('mean', 'std', 'ci95')] + ['model_bytes'])
        for s in d['summary']:
            w.writerow([s['model'], s['repeats']] + [s[m][x] for m in METRICS for x in ('mean', 'std', 'ci95')] + [s['model_bytes']])
    elif kind == 'comparisons':
        cols = ['model_a', 'model_b', 'metric', 'mean_diff', 'repeats', 't_stat', 'p_value', 'p_holm',
                'mcnemar_b', 'mcnemar_c', 'mcnemar_p', 'mcnemar_p_holm']
        w.writerow(cols)
        for r in d['comparisons']:
            w.writerow([r[c] for c in cols])
    else:
        cols = ['model', 'train_size', 'repeat', 'seed', *METRICS]
        w.writerow(cols)
        for r in d['runs']:
            w.writerow([r[c] for c in cols])
    data = '﻿' + buf.getvalue()  # BOM, чтобы Excel открыл кириллицу
    return Response(data, media_type='text/csv; charset=utf-8',
                    headers={'Content-Disposition': f'attachment; filename="textlab-{eid[:8]}-{kind}.csv"'})


@app.get('/api/analysis')
def analysis():
    """Сводка по завершённым экспериментам: датасет × объём обучения × модель."""
    with conn() as c:
        rows = c.execute('''SELECT e.dataset_name, e.train_rows, r.* FROM results r JOIN experiments e ON e.id=r.experiment_id
                            WHERE e.status='completed' ''').fetchall()
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        r = dict(r)
        groups.setdefault((r['dataset_name'], r['train_size'] or r['train_rows'], r['model_key'] or r['model']), []).append(r)
    out = []
    for (ds, n, _), items in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0, kv[0][2])):
        s = {'dataset': ds, 'train_rows': n, 'model': items[0]['model'], 'runs': len(items),
             'experiments': len({i['experiment_id'] for i in items})}
        for m in METRICS:
            s[m] = core.summarize([i.get(m) for i in items])
        out.append(s)
    return out


# ------------------------------------------------------------------ проверка текста

class InspectSpec(BaseModel):
    experiment_id: str
    text: str


@app.post('/api/inspect')
def inspect_text(spec: InspectSpec):
    import joblib
    import numpy as np
    text = spec.text.strip()
    if not text or len(text) > 30000:
        raise HTTPException(400, 'Введите текст длиной от 1 до 30 000 символов.')
    with conn() as c:
        records = c.execute('SELECT model, model_key, artifact FROM results WHERE experiment_id=? AND artifact IS NOT NULL',
                            (spec.experiment_id,)).fetchall()
    if not records:
        raise HTTPException(404, 'У эксперимента нет сохранённых моделей.')
    output = []
    for record in records:
        path = MODELS_DIR / Path(record['artifact']).name
        if not path.exists():
            output.append({'model': record['model'], 'error': 'Файл модели отсутствует. Повторите обучение.'})
            continue
        try:
            if path.is_dir():
                import neural
                m = neural.load_artifact(path)
                probs = m.predict_proba([text])[0]
                classes = m.classes_
                scores = sorted([{'label': l, 'value': float(v)} for l, v in zip(classes, probs)],
                                key=lambda v: v['value'], reverse=True)
                output.append({'model': record['model'], 'prediction': scores[0]['label'], 'scores': scores[:5],
                               'score_kind': 'Вероятность модели (softmax, без калибровки)', 'contributions': [],
                               'explanation': 'Для трансформеров вклад отдельных слов не рассчитывается.', 'known_terms': None})
                continue
            pipe = joblib.load(path)
            if record['model_key'] == 'dummy':
                clf = pipe.named_steps['classifier']
                p = clf.predict_proba([[0]])[0]
                scores = sorted([{'label': str(l), 'value': float(v)} for l, v in zip(clf.classes_, p)],
                                key=lambda v: v['value'], reverse=True)
                output.append({'model': record['model'], 'prediction': scores[0]['label'], 'scores': scores[:5],
                               'score_kind': 'Доля класса в обучающей выборке', 'contributions': [],
                               'explanation': 'Baseline не смотрит на текст: всегда отвечает самым частым классом.',
                               'known_terms': None})
                continue
            vectorizer = pipe.named_steps['tfidf']
            clf = pipe.named_steps['classifier']
            x = vectorizer.transform([text])
            predicted = str(pipe.predict([text])[0])
            classes = [str(v) for v in clf.classes_]
            index = classes.index(predicted)
            kind = 'Оценка класса (не вероятность)'
            if hasattr(clf, 'predict_proba'):
                values = clf.predict_proba(x)[0]
                kind = 'Вероятность модели (без проверки калибровки)'
            else:
                values = np.asarray(clf.decision_function(x)).ravel()
                if len(classes) == 2:
                    values = np.array([-values[0], values[0]])
            scores = sorted([{'label': l, 'value': float(v)} for l, v in zip(classes, values)],
                            key=lambda v: v['value'], reverse=True)
            contributions, weights = [], None
            explanation = 'Для MLP вклад слов не рассчитывается.'
            if hasattr(clf, 'coef_') and not hasattr(clf, 'feature_log_prob_'):
                weights = clf.coef_[index] if clf.coef_.shape[0] > 1 else clf.coef_[0] * (1 if index == 1 else -1)
                explanation = 'Вклад в оценку выбранного класса: TF-IDF × вес. Свободный член не показан; это не SHAP.'
            elif hasattr(clf, 'feature_log_prob_'):
                flp = clf.feature_log_prob_
                if flp.shape[0] > 1:
                    weights = flp[index] - np.mean(np.delete(flp, index, axis=0), axis=0)
                    explanation = 'TF-IDF × отличие логарифма вероятности слова от среднего других классов. Это не SHAP.'
            if weights is not None:
                names = vectorizer.get_feature_names_out()
                contributions = sorted([{'term': str(names[i]), 'value': float(v * weights[i])}
                                        for i, v in zip(x.indices, x.data)], key=lambda v: abs(v['value']), reverse=True)[:16]
            output.append({'model': record['model'], 'prediction': predicted, 'scores': scores[:5], 'score_kind': kind,
                           'contributions': contributions, 'explanation': explanation, 'known_terms': int(x.nnz)})
        except Exception as exc:
            output.append({'model': record['model'], 'error': 'Не удалось применить модель: ' + str(exc)})
    return {'results': output}


@app.get('/api/health')
def health():
    with conn() as c:
        n = c.execute('SELECT COUNT(*) FROM datasets').fetchone()[0]
    return {'status': 'ok', 'db': str(DB), 'datasets': n, 'running': CURRENT['id'], 'queued': JOBS.qsize()}
