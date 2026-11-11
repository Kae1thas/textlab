"""Трансформерные модели TextLab (опционально, нужен PyTorch + transformers).

Два способа использовать предобученную модель:
  * emb_lr — трансформер заморожен, тексты кодируются mean-pooling эмбеддингами,
    поверх обучается логистическая регрессия (дёшево, без дообучения);
  * bert   — fine-tuning всей сети с классификационной головой.

Если библиотек нет, остальная часть комплекса работает без них.
"""
from __future__ import annotations

import json
import math
import random
import shutil
import threading
import time
from pathlib import Path

import numpy as np


class _Cancelled(Exception):
    pass


def _import():
    import torch  # noqa: F401
    import transformers  # noqa: F401
    return torch, transformers


def available() -> tuple[bool, str]:
    try:
        torch, _ = _import()
    except Exception as e:  # ImportError, ошибки DLL на Windows
        return False, f'не установлен ({type(e).__name__}). Запустите install-gpu.ps1.'
    if not torch.cuda.is_available():
        return True, 'PyTorch без CUDA: обучение пойдёт на CPU и будет очень медленным.'
    return True, 'CUDA: ' + torch.cuda.get_device_name(0)


def environment() -> dict:
    try:
        torch, transformers = _import()
    except Exception:
        return {'torch': None}
    info = {'torch': torch.__version__, 'transformers': transformers.__version__, 'cuda': torch.cuda.is_available()}
    if torch.cuda.is_available():
        info['gpu'] = torch.cuda.get_device_name(0)
        info['gpu_memory_gb'] = round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 1)
        info['cuda_version'] = torch.version.cuda
    return info


def _pretrained(cls, name, **kw):
    """Загрузка из локального кэша HuggingFace; в сеть — только если модели там нет.

    Без этого каждая загрузка делает сетевой запрос к Hub для проверки версии,
    и при сбое сети/VPN запрос может зависнуть, останавливая всю очередь.
    """
    try:
        return cls.from_pretrained(name, local_files_only=True, **kw)
    except Exception:
        return cls.from_pretrained(name, **kw)


_ENCODERS: dict = {}


def _encoder(name):
    # Энкодер для emb_lr переиспользуется между прогонами внутри процесса.
    if name not in _ENCODERS:
        from transformers import AutoModel, AutoTokenizer
        _ENCODERS.clear()
        _ENCODERS[name] = (_pretrained(AutoTokenizer, name), _pretrained(AutoModel, name))
    return _ENCODERS[name]


def _seed_all(seed: int):
    import torch
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device():
    import torch
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _autocast(device):
    import torch
    if device.type == 'cuda':
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast('cuda', dtype=dtype)
    return torch.autocast('cpu', enabled=False)


def _dir_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob('*') if p.is_file())


def _length_order(texts):
    # Сортировка по длине уменьшает паддинг при инференсе.
    return sorted(range(len(texts)), key=lambda i: len(texts[i]))


# ------------------------------------------------------------ эмбеддинги

def encode(model_name: str, texts: list[str], max_length: int, batch_size: int, cancel=None, progress=None,
           tokenizer=None, model=None) -> np.ndarray:
    import torch
    device = _device()
    if tokenizer is None or model is None:
        tokenizer, model = _encoder(model_name)
    model.to(device).eval()
    order = _length_order(texts)
    out = [None] * len(texts)
    n_batches = math.ceil(len(texts) / batch_size)
    with torch.inference_mode():
        for bi in range(n_batches):
            if cancel is not None and cancel.is_set():
                raise _Cancelled()
            idx = order[bi * batch_size:(bi + 1) * batch_size]
            enc = tokenizer([texts[i] for i in idx], truncation=True, max_length=max_length,
                            padding=True, return_tensors='pt').to(device)
            with _autocast(device):
                hidden = model(**enc).last_hidden_state
            mask = enc['attention_mask'].unsqueeze(-1).to(hidden.dtype)
            emb = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            emb = torch.nn.functional.normalize(emb.float(), dim=-1).cpu().numpy()
            for j, i in enumerate(idx):
                out[i] = emb[j]
            if progress and bi % 20 == 0:
                progress(bi / n_batches)
    return np.vstack(out)


class EmbeddingLogReg:
    family = 'emb_lr'

    def __init__(self, model_name, max_length=256, batch_size=64, C=1.0, seed=42):
        self.model_name, self.max_length, self.batch_size, self.C, self.seed = model_name, max_length, batch_size, C, seed
        self._tok = self._enc = None

    def _load_encoder(self):
        if self._enc is None:
            self._tok, self._enc = _encoder(self.model_name)
        return self._tok, self._enc

    def embed(self, texts, cancel=None, progress=None):
        tok, enc = self._load_encoder()
        return encode(self.model_name, texts, self.max_length, self.batch_size, cancel, progress, tok, enc)

    def fit(self, texts, labels, cancel=None, progress=None):
        from sklearn.linear_model import LogisticRegression
        E = self.embed(texts, cancel, (lambda f: progress(0.9 * f, 'кодирование')) if progress else None)
        if progress:
            progress(0.9, 'логистическая регрессия')
        self.clf = LogisticRegression(C=self.C, max_iter=2000, class_weight='balanced', random_state=self.seed)
        self.clf.fit(E, labels)
        self.classes_ = [str(c) for c in self.clf.classes_]
        return self

    def predict_proba(self, texts):
        return self.clf.predict_proba(self.embed(texts))

    def predict(self, texts):
        p = self.predict_proba(texts)
        return np.asarray(self.classes_)[p.argmax(1)]

    def save(self, path: Path):
        import joblib
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.clf, path / 'logreg.joblib')
        (path / 'textlab.json').write_text(json.dumps({
            'family': 'emb_lr', 'model_name': self.model_name, 'max_length': self.max_length,
            'batch_size': self.batch_size, 'classes': self.classes_}, ensure_ascii=False), encoding='utf-8')

    @classmethod
    def load(cls, path: Path):
        import joblib
        meta = json.loads((path / 'textlab.json').read_text(encoding='utf-8'))
        obj = cls(meta['model_name'], meta['max_length'], meta['batch_size'])
        obj.clf = joblib.load(path / 'logreg.joblib')
        obj.classes_ = meta['classes']
        return obj


# ------------------------------------------------------------ fine-tuning

class FineTunedClassifier:
    family = 'bert'

    def __init__(self, model_name, max_length=256, epochs=3, batch_size=32, lr=5e-5, warmup=0.1,
                 weight_decay=0.01, seed=42):
        self.model_name, self.max_length, self.epochs = model_name, max_length, epochs
        self.batch_size, self.lr, self.warmup, self.weight_decay, self.seed = batch_size, lr, warmup, weight_decay, seed

    def fit(self, texts, labels, cancel=None, progress=None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
        _seed_all(self.seed)
        device = _device()
        self.classes_ = sorted({str(l) for l in labels})
        l2i = {l: i for i, l in enumerate(self.classes_)}
        y = torch.tensor([l2i[str(l)] for l in labels])
        self.tokenizer = _pretrained(AutoTokenizer, self.model_name)
        self.model = _pretrained(
            AutoModelForSequenceClassification, self.model_name, num_labels=len(self.classes_),
            id2label=dict(enumerate(self.classes_)), label2id=l2i, ignore_mismatched_sizes=True)
        self.model.to(device)
        # Веса классов как class_weight='balanced' у линейных моделей.
        counts = torch.bincount(y, minlength=len(self.classes_)).float()
        weights = (len(y) / (len(self.classes_) * counts.clamp(min=1))).to(device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
        no_decay = ('bias', 'LayerNorm.weight', 'layer_norm.weight')
        groups = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
        ]
        opt = torch.optim.AdamW(groups, lr=self.lr)
        steps_per_epoch = math.ceil(len(texts) / self.batch_size)
        total = steps_per_epoch * self.epochs
        sched = get_linear_schedule_with_warmup(opt, int(self.warmup * total), total)
        gen = torch.Generator().manual_seed(self.seed)
        step = 0
        self.history = []
        self.model.train()
        for epoch in range(self.epochs):
            perm = torch.randperm(len(texts), generator=gen).tolist()
            running = 0.0
            for bi in range(steps_per_epoch):
                if cancel is not None and cancel.is_set():
                    raise _Cancelled()
                idx = perm[bi * self.batch_size:(bi + 1) * self.batch_size]
                enc = self.tokenizer([texts[i] for i in idx], truncation=True, max_length=self.max_length,
                                     padding=True, return_tensors='pt').to(device)
                with _autocast(device):
                    logits = self.model(**enc).logits
                loss = loss_fn(logits.float(), y[idx].to(device))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                sched.step()
                running += loss.item()
                step += 1
                if progress and step % 20 == 0:
                    progress(step / total, f'эпоха {epoch + 1}/{self.epochs}, loss {running / (bi + 1):.3f}')
            self.history.append(running / steps_per_epoch)
        self.model.eval()
        return self

    def predict_proba(self, texts, batch_size=128):
        import torch
        device = _device()
        self.model.to(device).eval()
        order = _length_order(texts)
        out = np.zeros((len(texts), len(self.classes_)), dtype=np.float32)
        with torch.inference_mode():
            for s in range(0, len(texts), batch_size):
                idx = order[s:s + batch_size]
                enc = self.tokenizer([texts[i] for i in idx], truncation=True, max_length=self.max_length,
                                     padding=True, return_tensors='pt').to(device)
                with _autocast(device):
                    logits = self.model(**enc).logits
                out[idx] = torch.softmax(logits.float(), -1).cpu().numpy()
        return out

    def predict(self, texts):
        return np.asarray(self.classes_)[self.predict_proba(texts).argmax(1)]

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        (path / 'textlab.json').write_text(json.dumps({
            'family': 'bert', 'model_name': self.model_name, 'max_length': self.max_length,
            'classes': self.classes_, 'loss_history': self.history}, ensure_ascii=False), encoding='utf-8')

    @classmethod
    def load(cls, path: Path):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        meta = json.loads((path / 'textlab.json').read_text(encoding='utf-8'))
        obj = cls(meta['model_name'], meta['max_length'])
        obj.classes_ = meta['classes']
        obj.tokenizer = AutoTokenizer.from_pretrained(path)
        obj.model = AutoModelForSequenceClassification.from_pretrained(path)
        return obj


def load_artifact(path: Path):
    meta = json.loads((path / 'textlab.json').read_text(encoding='utf-8'))
    return (FineTunedClassifier if meta['family'] == 'bert' else EmbeddingLogReg).load(path)


# ------------------------------------------------------------ прогон

def train_and_evaluate(key, model_name, params, Xtr, ytr, Xte, seed, save_dir: Path | None,
                       progress=None, cancel: threading.Event | None = None) -> dict:
    import torch
    from core import Cancelled, PeakMemory
    if key == 'bert':
        m = FineTunedClassifier(model_name, params['max_length'], params['epochs'], params['batch_size'],
                                params['lr'], params['warmup'], params['weight_decay'], seed)
    else:
        _seed_all(seed)
        m = EmbeddingLogReg(model_name, params['max_length'], params['batch_size'], params['C'], seed)
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    try:
        with PeakMemory() as mem:
            t0 = time.perf_counter()
            m.fit(Xtr, ytr, cancel=cancel, progress=(lambda f, msg='': progress(0.9 * f, msg)) if progress else None)
            if cuda:
                torch.cuda.synchronize()
            train_s = time.perf_counter() - t0
        if progress:
            progress(0.92, 'предсказание на тесте')
        t0 = time.perf_counter()
        pred = m.predict(Xte)
        if cuda:
            torch.cuda.synchronize()
        infer = time.perf_counter() - t0
    except _Cancelled:
        raise Cancelled()
    out = {'pred': pred, 'train_seconds': train_s, 'inference_ms': infer * 1000 / len(Xte),
           'peak_ram_mb': mem.delta_mb,
           'gpu_peak_mb': torch.cuda.max_memory_allocated() / 1024 ** 2 if cuda else None,
           'artifact': None, 'model_bytes': None, 'warnings': []}
    if not cuda:
        out['warnings'].append('Обучение шло на CPU — время несопоставимо с GPU-запусками.')
    if save_dir is not None:
        if save_dir.exists():
            shutil.rmtree(save_dir)
        m.save(save_dir)
        out['artifact'] = save_dir.name
        out['model_bytes'] = _dir_bytes(save_dir)
        if key == 'emb_lr':
            out['warnings'].append('Размер модели — только LogReg; веса энкодера берутся из кэша HuggingFace.')
    # Освобождаем видеопамять перед следующей моделью.
    del m
    if cuda:
        torch.cuda.empty_cache()
    return out
