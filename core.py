"""Ядро TextLab: датасеты, модели, метрики, статистические тесты.

Модуль не зависит от FastAPI и базы данных, поэтому его можно
использовать из скриптов и тестов напрямую.
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
import platform
import random
import sys
import threading
import time
import warnings
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import psutil
from scipy import stats
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score,
                             confusion_matrix, matthews_corrcoef, precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent
# Каталог данных можно вынести (например, для тестов) переменной TEXTLAB_HOME.
HOME = Path(os.environ.get('TEXTLAB_HOME', ROOT)).resolve()
DATA = HOME / 'data'
CACHE = DATA / 'cache'
UPLOADS = DATA / 'uploads'
MODELS_DIR = HOME / 'model_store'

# Длинные новости Lenta превышают стандартный лимит поля csv (128 КБ).
csv.field_size_limit(2**31 - 1)


class Cancelled(Exception):
    """Эксперимент остановлен пользователем."""


# ---------------------------------------------------------------- датасеты

BUILTIN_DATASETS = [
    {
        'id': 'sms-spam', 'name': 'UCI SMS Spam Collection', 'file': 'sms_spam.csv', 'kind': 'csv',
        'language': 'en', 'source': 'UCI ML Repository (CC BY 4.0)',
        'description': 'SMS-сообщения, 2 класса (ham/spam), сильный дисбаланс ~87/13',
    },
    {
        'id': '20newsgroups', 'name': '20 Newsgroups', 'file': '20newsgroups.csv', 'kind': 'csv',
        'language': 'en', 'source': 'scikit-learn / UCI archive',
        'description': 'Сообщения конференций Usenet, 20 тем, классы почти сбалансированы',
    },
    {
        'id': 'rureviews', 'name': 'RuReviews — отзывы о товарах', 'file': 'rureviews.csv', 'kind': 'csv',
        'language': 'ru', 'source': 'github.com/sismetanin/rureviews (Apache-2.0)',
        'description': 'Отзывы на одежду и аксессуары, 3 класса тональности, сбалансирован',
    },
    {
        'id': 'lenta', 'name': 'Lenta.ru — новости (15 тем)', 'file': 'lenta-ru-news.csv', 'kind': 'lenta',
        'language': 'ru', 'source': 'Lenta.ru, корпус yutkin/Lenta.Ru-News-Dataset',
        'description': 'Новости 1999–2019, 15 основных рубрик, сильный дисбаланс (Россия 22% … 69-я параллель 0.2%)',
    },
]

LENTA_MIN_CLASS = 1000
TEXT_ALIASES = ['text', 'review', 'content', 'message', 'body', 'sentence', 'document', 'текст', 'отзыв']
LABEL_ALIASES = ['label', 'category', 'class', 'sentiment', 'topic', 'target', 'y', 'метка', 'класс', 'категория']


def sniff_dialect(sample: str):
    try:
        return csv.Sniffer().sniff(sample, delimiters=',;\t|')
    except csv.Error:
        return csv.excel


def guess_column(headers: list[str], aliases: list[str]) -> str | None:
    low = {h.lower().strip(): h for h in headers}
    for a in aliases:
        if a in low:
            return low[a]
    return None


def read_text_label_csv(path: Path) -> tuple[list[str], list[str]]:
    """Читает подготовленный CSV со столбцами text,label."""
    texts, labels = [], []
    with open(path, encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            t = (row.get('text') or '').strip()
            l = (row.get('label') or '').strip()
            if t and l:
                texts.append(t)
                labels.append(l)
    return texts, labels


def class_counts(labels: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for l in labels:
        out[l] = out.get(l, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))


def lenta_profile() -> dict | None:
    p = DATA / 'lenta_profile.json'
    if p.exists():
        return json.loads(p.read_text(encoding='utf-8'))
    return None


def lenta_eligible_topics() -> set[str]:
    prof = lenta_profile()
    if prof:
        return set(prof['class_counts'])
    # Профиля нет: считаем распределение одним проходом.
    counts: dict[str, int] = {}
    with open(DATA / 'lenta-ru-news.csv', encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            t = (row.get('topic') or '').strip()
            if t:
                counts[t] = counts.get(t, 0) + 1
    return {k for k, v in counts.items() if v >= LENTA_MIN_CLASS}


def lenta_sample(max_rows: int, seed: int, progress: Callable[[str], None] | None = None,
                 cancel: threading.Event | None = None) -> tuple[list[str], list[str], bool]:
    """Случайная (reservoir) выборка из Lenta с кэшированием на диск.

    Повторный запуск с теми же max_rows и seed берёт выборку из data/cache
    за секунды вместо полного прохода по 2-гигабайтному CSV.
    Возвращает (texts, labels, from_cache).
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = CACHE / f'lenta_n{max_rows}_s{seed}.csv'
    if cache.exists():
        t, l = read_text_label_csv(cache)
        return t, l, True
    eligible = lenta_eligible_topics()
    rng = random.Random(seed)
    reservoir: list[tuple[str, str]] = []
    seen = 0
    src = DATA / 'lenta-ru-news.csv'
    total = src.stat().st_size
    with open(src, encoding='utf-8-sig', newline='') as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i % 20000 == 0:
                if cancel is not None and cancel.is_set():
                    raise Cancelled()
                if progress:
                    try:
                        pos = f.buffer.tell() / total  # type: ignore[attr-defined]
                    except Exception:
                        pos = 0
                    progress(f'Чтение Lenta: {i:,} строк ({pos:.0%})'.replace(',', ' '))
            label = (row.get('topic') or '').strip()
            if label not in eligible:
                continue
            txt = ((row.get('title') or '') + '. ' + (row.get('text') or '')).strip()
            if len(txt) < 20:
                continue
            seen += 1
            if len(reservoir) < max_rows:
                reservoir.append((txt, label))
            else:
                j = rng.randrange(seen)
                if j < max_rows:
                    reservoir[j] = (txt, label)
    tmp = cache.with_suffix('.tmp')
    with open(tmp, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['text', 'label'])
        w.writerows(reservoir)
    os.replace(tmp, cache)
    return [r[0] for r in reservoir], [r[1] for r in reservoir], False


def stratified_subsample(texts, labels, max_rows, seed):
    if len(texts) <= max_rows:
        return list(texts), list(labels)
    counts = class_counts(labels)
    keep = {k for k, v in counts.items() if v >= 2}
    idx = [i for i, l in enumerate(labels) if l in keep]
    sub, _ = train_test_split(idx, train_size=max_rows, random_state=seed, stratify=[labels[i] for i in idx])
    sub.sort()
    return [texts[i] for i in sub], [labels[i] for i in sub]


def load_dataset(ds: dict, max_rows: int, seed: int, progress=None, cancel=None) -> tuple[list[str], list[str], str]:
    """Возвращает тексты, метки и пояснение, как получена выборка."""
    if ds['kind'] == 'lenta':
        t, l, cached = lenta_sample(max_rows, seed, progress, cancel)
        note = f'reservoir-выборка {len(t):,} из Lenta' + (' (из кэша)' if cached else '')
        return t, l, note.replace(',', ' ')
    t, l = read_text_label_csv(Path(ds['path']))
    n0 = len(t)
    t, l = stratified_subsample(t, l, max_rows, seed)
    note = f'весь набор, {n0:,} строк' if len(t) == n0 else f'стратифицированная выборка {len(t):,} из {n0:,}'
    return t, l, note.replace(',', ' ')


# ---------------------------------------------------------------- модели

TFIDF_MAIN = dict(ngram_range=(1, 2), min_df=2, max_features=100_000, sublinear_tf=True)
TFIDF_MLP = dict(ngram_range=(1, 2), min_df=2, max_features=20_000, sublinear_tf=True)

MODEL_REGISTRY = {
    'dummy': {
        'name': 'Baseline: самый частый класс', 'family': 'baseline', 'features': 'не использует текст',
        'params': {'strategy': 'most_frequent'},
        'note': 'Нижняя граница: всегда предсказывает самый частый класс обучающей выборки',
    },
    'nb': {
        'name': 'Multinomial Naive Bayes', 'family': 'classic', 'features': 'TF-IDF 1–2-граммы, 100k',
        'params': {'alpha': 0.5}, 'vectorizer': TFIDF_MAIN,
    },
    'cnb': {
        'name': 'Complement Naive Bayes', 'family': 'classic', 'features': 'TF-IDF 1–2-граммы, 100k',
        'params': {'alpha': 0.5}, 'vectorizer': TFIDF_MAIN,
        'note': 'Вариант NB для несбалансированных классов (Rennie et al., 2003)',
    },
    'lr': {
        'name': 'Logistic Regression', 'family': 'classic', 'features': 'TF-IDF 1–2-граммы, 100k',
        'params': {'solver': 'lbfgs', 'C': 1.0, 'max_iter': 1000, 'class_weight': 'balanced'}, 'vectorizer': TFIDF_MAIN,
    },
    'svm': {
        'name': 'Linear SVM', 'family': 'classic', 'features': 'TF-IDF 1–2-граммы, 100k',
        'params': {'C': 1.0, 'max_iter': 5000, 'class_weight': 'balanced'}, 'vectorizer': TFIDF_MAIN,
    },
    'mlp': {
        'name': 'MLP (1 скрытый слой)', 'family': 'neural', 'features': 'TF-IDF 1–2-граммы, 20k',
        'params': {'hidden_layer_sizes': [128], 'max_iter': 50, 'early_stopping': True}, 'vectorizer': TFIDF_MLP,
        'max_train_rows': 150_000,
    },
    'emb_lr': {
        'name': 'Эмбеддинги BERT + LogReg', 'family': 'transformer', 'features': 'mean-pooling замороженного трансформера',
        'params': {'max_length': 256, 'batch_size': 64, 'C': 1.0}, 'requires': 'torch',
    },
    'bert': {
        'name': 'BERT fine-tuning', 'family': 'transformer', 'features': 'дообучение всей сети',
        'params': {'max_length': 256, 'epochs': 3, 'batch_size': 32, 'lr': 5e-5, 'warmup': 0.1, 'weight_decay': 0.01},
        'requires': 'torch',
    },
}

TRANSFORMER_PRESETS = [
    {'id': 'cointegrated/rubert-tiny2', 'note': '29M параметров, ru+en, быстрый — по умолчанию'},
    {'id': 'DeepPavlov/rubert-base-cased', 'note': '180M, русский, точнее и в ~6 раз медленнее'},
    {'id': 'distilbert-base-multilingual-cased', 'note': '135M, мультиязычный'},
    {'id': 'xlm-roberta-base', 'note': '280M, мультиязычный, самый тяжёлый'},
]


class NoFeatures(BaseEstimator, TransformerMixin):
    """Пустое признаковое пространство для baseline: текст модели не нужен."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return sparse.csr_matrix((len(X), 1))

    def get_feature_names_out(self, input_features=None):
        return np.array(['const'])


def make_classifier(key: str, seed: int):
    p = MODEL_REGISTRY[key]['params']
    if key == 'dummy':
        return DummyClassifier(strategy=p['strategy'])
    if key == 'nb':
        return MultinomialNB(alpha=p['alpha'])
    if key == 'cnb':
        return ComplementNB(alpha=p['alpha'])
    if key == 'lr':
        return LogisticRegression(solver=p['solver'], C=p['C'], max_iter=p['max_iter'],
                                  class_weight=p['class_weight'], random_state=seed)
    if key == 'svm':
        return LinearSVC(C=p['C'], max_iter=p['max_iter'], class_weight=p['class_weight'], random_state=seed)
    if key == 'mlp':
        return MLPClassifier(hidden_layer_sizes=tuple(p['hidden_layer_sizes']), max_iter=p['max_iter'],
                             early_stopping=p['early_stopping'], random_state=seed)
    raise KeyError(key)


# ---------------------------------------------------------------- метрики

METRIC_KEYS = ['accuracy', 'balanced_accuracy', 'precision_macro', 'recall_macro', 'f1_macro',
               'f1_weighted', 'mcc', 'kappa']
COST_KEYS = ['train_seconds', 'inference_ms', 'peak_ram_mb', 'gpu_peak_mb', 'model_bytes']


def compute_metrics(y_true, y_pred, labels) -> dict:
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average='macro', zero_division=0)
    _, _, fw, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average='weighted', zero_division=0)
    pc, rc, fc, sc = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'precision_macro': float(p), 'recall_macro': float(r), 'f1_macro': float(f), 'f1_weighted': float(fw),
        'mcc': float(matthews_corrcoef(y_true, y_pred)),
        'kappa': float(cohen_kappa_score(y_true, y_pred)),
        'confusion': confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        'per_class': [{'label': l, 'precision': float(a), 'recall': float(b), 'f1': float(c), 'support': int(d)}
                      for l, a, b, c, d in zip(labels, pc, rc, fc, sc)],
    }


def summarize(values: list[float | None]) -> dict:
    v = [x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))]
    n = len(v)
    if not n:
        return {'mean': None, 'std': None, 'ci95': None, 'n': 0}
    mean = float(np.mean(v))
    if n == 1:
        return {'mean': mean, 'std': None, 'ci95': None, 'n': 1}
    std = float(np.std(v, ddof=1))
    ci = float(stats.t.ppf(0.975, n - 1) * std / math.sqrt(n))
    return {'mean': mean, 'std': std, 'ci95': ci, 'n': n}


# ---------------------------------------------------------------- статистика

def corrected_resampled_ttest(diffs: list[float], n_train: int, n_test: int) -> tuple[float | None, float | None]:
    """Скорректированный t-тест для повторных разбиений (Nadeau & Bengio, 2003).

    Обычный парный t-тест на повторных hold-out разбиениях занижает p-value,
    потому что обучающие выборки пересекаются; поправка n_test/n_train
    увеличивает дисперсию.
    """
    J = len(diffs)
    if J < 2:
        return None, None
    d = np.asarray(diffs, dtype=float)
    var = d.var(ddof=1)
    if var == 0:
        return (None, 1.0) if d.mean() == 0 else (None, 0.0)
    t = d.mean() / math.sqrt((1 / J + n_test / n_train) * var)
    p = 2 * stats.t.sf(abs(t), J - 1)
    return float(t), float(p)


def mcnemar_test(y_true, pred_a, pred_b) -> tuple[int, int, float]:
    """Тест Макнемара на одном общем тестовом наборе.

    b — A прав, B ошибся; c — наоборот. При b+c < 25 используется точный
    биномиальный тест, иначе χ² с поправкой на непрерывность.
    """
    y = np.asarray(y_true)
    a_ok = np.asarray(pred_a) == y
    b_ok = np.asarray(pred_b) == y
    b = int(np.sum(a_ok & ~b_ok))
    c = int(np.sum(~a_ok & b_ok))
    if b + c == 0:
        return b, c, 1.0
    if b + c < 25:
        return b, c, float(stats.binomtest(min(b, c), b + c, 0.5).pvalue)
    chi = (abs(b - c) - 1) ** 2 / (b + c)
    return b, c, float(stats.chi2.sf(chi, 1))


def holm(pvalues: list[float | None]) -> list[float | None]:
    idx = [i for i, p in enumerate(pvalues) if p is not None]
    m = len(idx)
    out: list[float | None] = [None] * len(pvalues)
    running = 0.0
    for rank, i in enumerate(sorted(idx, key=lambda i: pvalues[i])):
        adj = min(1.0, (m - rank) * pvalues[i])
        running = max(running, adj)
        out[i] = running
    return out


def pairwise_comparisons(per_repeat_f1: dict[str, list[float]], predictions0: dict[str, list],
                         y_test0: list, n_train: int, n_test: int, names: dict[str, str]) -> list[dict]:
    keys = list(per_repeat_f1)
    rows = []
    for a, b in combinations(keys, 2):
        fa, fb = per_repeat_f1[a], per_repeat_f1[b]
        diffs = [x - y for x, y in zip(fa, fb)]
        t, p = corrected_resampled_ttest(diffs, n_train, n_test)
        row = {'model_a': names[a], 'model_b': names[b], 'key_a': a, 'key_b': b,
               'metric': 'f1_macro', 'mean_diff': float(np.mean(diffs)), 'repeats': len(diffs),
               't_stat': t, 'p_value': p}
        if a in predictions0 and b in predictions0:
            nb, nc, pm = mcnemar_test(y_test0, predictions0[a], predictions0[b])
            row.update({'mcnemar_b': nb, 'mcnemar_c': nc, 'mcnemar_p': pm})
        rows.append(row)
    adj = holm([r['p_value'] for r in rows])
    adjm = holm([r.get('mcnemar_p') for r in rows])
    for r, p1, p2 in zip(rows, adj, adjm):
        r['p_holm'] = p1
        r['mcnemar_p_holm'] = p2
    return rows


# ---------------------------------------------------------------- замеры

class PeakMemory:
    """Пиковый прирост RSS процесса во время блока (опрос каждые 20 мс)."""

    def __init__(self, interval=0.02):
        self.proc = psutil.Process()
        self.interval = interval
        self.peak = 0
        self._stop = threading.Event()

    def __enter__(self):
        self.base = self.proc.memory_info().rss
        self.peak = self.base
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def _run(self):
        while not self._stop.wait(self.interval):
            try:
                self.peak = max(self.peak, self.proc.memory_info().rss)
            except Exception:
                return

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()
        self.peak = max(self.peak, self.proc.memory_info().rss)

    @property
    def delta_mb(self) -> float:
        return (self.peak - self.base) / 1024 ** 2


def warning_texts(records) -> list[str]:
    out = []
    for r in records:
        if issubclass(r.category, ConvergenceWarning):
            msg = str(r.message).split('\n')[0][:200]
            if msg not in out:
                out.append(msg)
    return out


def environment_info() -> dict:
    import sklearn
    info = {
        'python': sys.version.split()[0], 'platform': platform.platform(),
        'cpu': platform.processor() or platform.machine(), 'cpu_cores': psutil.cpu_count(logical=False),
        'cpu_threads': psutil.cpu_count(), 'ram_gb': round(psutil.virtual_memory().total / 1024 ** 3, 1),
        'sklearn': sklearn.__version__, 'numpy': np.__version__,
    }
    try:
        import neural
        info.update(neural.environment())
    except Exception:
        pass
    return info


# ---------------------------------------------------------------- прогон

@dataclass
class RunConfig:
    dataset: dict
    models: list[str]
    test_size: float = 0.2
    seed: int = 42
    repeats: int = 1
    max_rows: int = 50_000
    transformer_model: str = 'cointegrated/rubert-tiny2'
    transformer_epochs: int = 3
    transformer_max_length: int = 256
    # Режим кривой обучения: размеры вложенных обучающих подвыборок при общем тесте.
    train_sizes: list[int] | None = None


@dataclass
class RunCallbacks:
    progress: Callable[[float, str], None] = lambda frac, msg: None
    result: Callable[[dict], None] = lambda row: None
    cancel: threading.Event = field(default_factory=threading.Event)


def check_cancel(cb: RunCallbacks):
    if cb.cancel.is_set():
        raise Cancelled()


def fit_classic(key, spec, Xtr, ytr, Xte, rseed, vec_cache):
    """Обучение модели на TF-IDF (или без признаков для baseline). Возвращает (pipeline, pred, замеры)."""
    with PeakMemory() as mem, warnings.catch_warnings(record=True) as wrec:
        warnings.simplefilter('always', ConvergenceWarning)
        if 'vectorizer' not in spec:
            vec, vec_seconds = NoFeatures(), 0.0
            Atr = vec.transform(Xtr)
        else:
            vkey = json.dumps(spec['vectorizer'], sort_keys=True, default=list)
            if vkey in vec_cache:
                vec, Atr, vec_seconds = vec_cache[vkey]
            else:
                t0 = time.perf_counter()
                vec = TfidfVectorizer(**spec['vectorizer'])
                Atr = vec.fit_transform(Xtr)
                vec_seconds = time.perf_counter() - t0
                vec_cache[vkey] = (vec, Atr, vec_seconds)
        clf = make_classifier(key, rseed)
        t0 = time.perf_counter()
        clf.fit(Atr, ytr)
        fit_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = clf.predict(vec.transform(Xte))
    infer = time.perf_counter() - t0
    stats_ = {'train_seconds': vec_seconds + fit_seconds, 'vectorize_seconds': vec_seconds,
              'inference_ms': infer * 1000 / len(Xte), 'peak_ram_mb': mem.delta_mb, 'warnings': warning_texts(wrec)}
    return Pipeline([('tfidf', vec), ('classifier', clf)]), pred, stats_


def run_experiment(cfg: RunConfig, eid: str, cb: RunCallbacks) -> dict:
    """Повторный стратифицированный hold-out.

    На каждом повторе r используется seed+r для разбиения; все модели
    повтора обучаются и тестируются на одном и том же разбиении, поэтому
    сравнения парные. Модели первого повтора сохраняются на диск.

    В режиме кривой обучения (train_sizes) тестовый набор повтора общий,
    а обучающие выборки вложены друг в друга: первые n элементов одной
    случайной перестановки обучающей части. Сравнения моделей и сохранённые
    модели относятся к наибольшему размеру.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    cb.progress(0.0, 'Загрузка данных')
    texts, labels, sample_note = load_dataset(cfg.dataset, cfg.max_rows, cfg.seed,
                                              progress=lambda m: cb.progress(0.0, m), cancel=cb.cancel)
    counts = class_counts(labels)
    rare = [k for k, v in counts.items() if v < 2]
    if rare:
        keep = [i for i, l in enumerate(labels) if l not in rare]
        texts = [texts[i] for i in keep]
        labels = [labels[i] for i in keep]
    label_list = sorted(set(labels))
    if len(label_list) < 2:
        raise ValueError('В выборке меньше двух классов.')
    n_test = int(math.ceil(len(texts) * cfg.test_size))
    if n_test < len(label_list):
        raise ValueError(f'Тестовая выборка ({n_test}) меньше числа классов ({len(label_list)}). Увеличьте объём или долю теста.')
    pool = len(texts) - n_test
    sizes: list[int | None] = [None]
    if cfg.train_sizes:
        sizes = sorted(set(int(x) for x in cfg.train_sizes))
        if sizes[-1] > pool:
            raise ValueError(f'Наибольший размер обучения ({sizes[-1]:,}) больше обучающей части выборки ({pool:,}). '
                             f'Увеличьте «Строк в выборке» или уменьшите размеры.'.replace(',', ' '))
        if sizes[0] < len(label_list):
            raise ValueError('Размер обучения меньше числа классов.')
    for key in cfg.models:
        limit = MODEL_REGISTRY[key].get('max_train_rows')
        largest = sizes[-1] or pool
        if limit and largest > limit:
            raise ValueError(f'{MODEL_REGISTRY[key]["name"]}: не более {limit:,} обучающих строк.'.replace(',', ' '))

    need_torch = [k for k in cfg.models if MODEL_REGISTRY[k].get('requires') == 'torch']
    if need_torch:
        import neural
        ok, reason = neural.available()
        if not ok:
            raise ValueError('Для моделей ' + ', '.join(MODEL_REGISTRY[k]['name'] for k in need_torch) + ' нужен PyTorch: ' + reason)

    steps = cfg.repeats * len(cfg.models) * len(sizes)
    step = 0
    per_repeat_f1: dict[str, list[float]] = {k: [] for k in cfg.models}
    predictions0: dict[str, list] = {}
    y_test0: list = []
    split_sizes = (0, 0)
    curve = len(sizes) > 1

    for rep in range(cfg.repeats):
        rseed = cfg.seed + rep
        Xpool, Xte, ypool, yte = train_test_split(texts, labels, test_size=cfg.test_size, random_state=rseed, stratify=labels)
        perm = np.random.RandomState(rseed).permutation(len(Xpool))
        if rep == 0:
            y_test0 = list(yte)
        for size in sizes:
            if size is None:
                Xtr, ytr = Xpool, ypool
            else:
                idx = perm[:size]
                Xtr, ytr = [Xpool[i] for i in idx], [ypool[i] for i in idx]
            final = size == sizes[-1]
            split_sizes = (len(Xtr), len(Xte))
            vec_cache: dict[str, tuple] = {}
            for key in cfg.models:
                check_cancel(cb)
                spec = MODEL_REGISTRY[key]
                base_frac = step / steps
                where = f'Повтор {rep + 1}/{cfg.repeats}' + ((', train ' + f'{len(Xtr):,}'.replace(',', ' ')) if curve else '')
                cb.progress(base_frac, f'{where}: {spec["name"]}')
                row = {'model_key': key, 'model': spec['name'], 'repeat': rep, 'seed': rseed, 'train_size': len(Xtr),
                       'train_rows': len(Xtr), 'test_rows': len(Xte), 'gpu_peak_mb': None,
                       'params': {**spec['params'], **({'vectorizer': {k: list(v) if isinstance(v, tuple) else v
                                                                       for k, v in spec['vectorizer'].items()}}
                                                         if 'vectorizer' in spec else {})}}
                save = rep == 0 and final
                artifact = None
                if spec['family'] in ('classic', 'neural', 'baseline'):
                    pipe, pred, st = fit_classic(key, spec, Xtr, ytr, Xte, rseed, vec_cache)
                    row.update(st)
                    if save:
                        import joblib
                        path = MODELS_DIR / f'{eid}_{key}.joblib'
                        joblib.dump(pipe, path)
                        artifact = path.name
                        row['model_bytes'] = path.stat().st_size
                else:
                    import neural

                    def sub_progress(frac, msg, _b=base_frac, _w=where, _n=spec['name']):
                        cb.progress(_b + frac / steps, f'{_w}: {_n} — {msg}')
                    p = dict(spec['params'])
                    p['max_length'] = cfg.transformer_max_length
                    if key == 'bert':
                        p['epochs'] = cfg.transformer_epochs
                    row['params'] = {**p, 'model_name': cfg.transformer_model}
                    out = neural.train_and_evaluate(key, cfg.transformer_model, p, Xtr, ytr, Xte, rseed,
                                                    save_dir=(MODELS_DIR / f'{eid}_{key}') if save else None,
                                                    progress=sub_progress, cancel=cb.cancel)
                    pred = out['pred']
                    row.update({k: out[k] for k in ('train_seconds', 'inference_ms', 'peak_ram_mb', 'gpu_peak_mb')})
                    row['vectorize_seconds'] = None
                    row['warnings'] = out.get('warnings', [])
                    if save:
                        artifact = out['artifact']
                        row['model_bytes'] = out['model_bytes']
                row.update(compute_metrics(yte, pred, label_list))
                row['labels'] = label_list
                row['artifact'] = artifact
                if not (rep == 0 and final):
                    row['model_bytes'] = None
                    row['confusion'] = None
                    row['per_class'] = None
                if final:
                    per_repeat_f1[key].append(row['f1_macro'])
                    if rep == 0:
                        predictions0[key] = [str(x) for x in pred]
                cb.result(row)
                step += 1

    names = {k: MODEL_REGISTRY[k]['name'] for k in cfg.models}
    comparisons = pairwise_comparisons(per_repeat_f1, predictions0, y_test0, split_sizes[0], split_sizes[1], names) \
        if len(cfg.models) > 1 else []
    cb.progress(1.0, 'Готово')
    return {'train_rows': split_sizes[0], 'test_rows': split_sizes[1], 'sample_note': sample_note,
            'class_counts': class_counts(labels), 'comparisons': comparisons}
