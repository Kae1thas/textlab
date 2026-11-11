"""Скачивает открытые датасеты в ./data (каждый независимо; ошибка одного не мешает другим).

  * UCI SMS Spam Collection — en, 2 класса, дисбаланс 87/13
  * 20 Newsgroups           — en, 20 классов, почти баланс
  * RuReviews               — ru, 3 класса тональности, баланс
Lenta.ru (2 ГБ) не скачивается автоматически: положите lenta-ru-news.csv в data/.
"""
import csv
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
DATA.mkdir(exist_ok=True)
csv.field_size_limit(2**31 - 1)


def fetch(url, timeout=120) -> bytes:
    req = urllib.request.Request(url, headers={'User-Agent': 'TextLab-course-project'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def write_rows(dest: Path, rows):
    tmp = dest.with_suffix('.tmp')
    with tmp.open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['text', 'label'])
        n = 0
        for t, l in rows:
            t = ' '.join(t.split())
            if len(t) >= 3 and l:
                w.writerow([t, l])
                n += 1
    tmp.replace(dest)
    return n


def download_sms():
    dest = DATA / 'sms_spam.csv'
    if dest.exists():
        return 'уже есть'
    blob = fetch('https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip')
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        member = next(n for n in z.namelist() if n.endswith('SMSSpamCollection'))
        lines = z.read(member).decode('utf-8', errors='replace').splitlines()
    rows = []
    for line in lines:
        label, _, text = line.partition('\t')
        rows.append((text, label))
    return f'{write_rows(dest, rows)} SMS'


def download_news():
    dest = DATA / '20newsgroups.csv'
    if dest.exists():
        return 'уже есть'
    from sklearn.datasets import fetch_20newsgroups
    rows = []
    # Заголовки, подписи и цитаты удаляются, чтобы модель не училась по метаданным.
    for split in ('train', 'test'):
        b = fetch_20newsgroups(subset=split, remove=('headers', 'footers', 'quotes'), shuffle=False,
                               download_if_missing=True, data_home=str(DATA / 'sklearn_cache'))
        rows += [(t, b.target_names[int(y)]) for t, y in zip(b.data, b.target) if len(t.strip()) > 20]
    return f'{write_rows(dest, rows)} сообщений'


def download_rureviews():
    dest = DATA / 'rureviews.csv'
    if dest.exists():
        return 'уже есть'
    raw = fetch('https://raw.githubusercontent.com/sismetanin/rureviews/master/'
                'women-clothing-accessories.3-class.balanced.csv').decode('utf-8-sig', errors='replace')
    try:
        dialect = csv.Sniffer().sniff(raw[:20000], delimiters='\t,;')
    except csv.Error:
        dialect = csv.excel_tab
    reader = csv.reader(io.StringIO(raw), dialect)
    header = next(reader)
    low = [h.strip().lower() for h in header]
    if 'review' in low and 'sentiment' in low:
        ti, li = low.index('review'), low.index('sentiment')
    else:  # без заголовка: текст, метка
        ti, li = 0, 1
        reader = csv.reader(io.StringIO(raw), dialect)
    fix = {'neautral': 'neutral'}  # опечатка в исходном файле
    rows = ((r[ti], fix.get(r[li].strip().lower(), r[li].strip().lower())) for r in reader if len(r) > max(ti, li))
    return f'{write_rows(dest, rows)} отзывов'


if __name__ == '__main__':
    failed = 0
    for name, fn in [('SMS Spam', download_sms), ('20 Newsgroups', download_news), ('RuReviews', download_rureviews)]:
        try:
            print(f'{name}: {fn()}')
        except Exception as e:  # сеть, прокси, изменение источника
            failed += 1
            print(f'{name}: НЕ ЗАГРУЖЕН — {type(e).__name__}: {e}', file=sys.stderr)
    if not (DATA / 'lenta-ru-news.csv').exists():
        print('Lenta.ru: файла data/lenta-ru-news.csv нет — набор будет недоступен.')
    sys.exit(0)
