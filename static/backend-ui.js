'use strict';
const $ = id => document.getElementById(id);
const esc = v => String(v ?? '').replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
const nf = (v, d = 3) => v == null || Number.isNaN(v) ? '—' : Number(v).toFixed(d);
const intf = v => v == null ? '—' : Number(v).toLocaleString('ru-RU');
const pct = v => v == null ? '—' : (v * 100).toFixed(1);
const PALETTE = ['#325b89', '#6f8fb3', '#a7bbd2', '#3f7f6a', '#8fb7a5', '#b98b4e', '#d8b98d', '#8a6c9e', '#b9a6c8', '#a75d5d', '#d39c9c', '#5f6b78', '#9aa4ae', '#c2c9d0', '#e1e5e9'];

const LINE_COLORS = ['#325b89', '#d07a2d', '#3f7f6a', '#a75d5d', '#8a6c9e', '#2f9bb5', '#b98b4e', '#5f6b78'];
const state = {datasets: [], models: [], presets: [], current: null, pollTimer: null, journalTimer: null};

async function api(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) {
    let msg = r.statusText;
    try { const j = await r.json(); msg = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail); } catch (_) {}
    throw new Error(msg);
  }
  return r.json();
}
const postJSON = (url, body) => api(url, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});

const METRICS = [
  ['accuracy', 'Accuracy', ''], ['balanced_accuracy', 'Balanced acc.', ''], ['precision_macro', 'Precision', 'macro'],
  ['recall_macro', 'Recall', 'macro'], ['f1_macro', 'F1', 'macro'], ['f1_weighted', 'F1', 'weighted'],
  ['mcc', 'MCC', ''], ['kappa', 'Cohen κ', ''],
];
const COSTS = [['train_seconds', 'Обучение', 'с', 2], ['inference_ms', 'Инференс', 'мс/текст', 3],
  ['peak_ram_mb', 'Пик RAM', 'МБ', 0], ['gpu_peak_mb', 'Пик GPU', 'МБ', 0]];

// ------------------------------------------------------------ датасеты

function currentDataset() { return state.datasets.find(d => d.id === $('datasetSelect').value); }

function renderDatasetMenu(selectId) {
  const sel = $('datasetSelect');
  const groups = [['Встроенные', state.datasets.filter(d => d.builtin)], ['Загруженные', state.datasets.filter(d => !d.builtin)]];
  sel.innerHTML = groups.filter(g => g[1].length).map(([t, items]) =>
    `<optgroup label="${t}">${items.map(d => `<option value="${esc(d.id)}">${esc(d.name)}</option>`).join('')}</optgroup>`).join('');
  if (!state.datasets.length) sel.innerHTML = '<option value="">Нет наборов данных</option>';
  if (selectId && state.datasets.some(d => d.id === selectId)) sel.value = selectId;
  showDataset();
}

function showDataset() {
  const d = currentDataset();
  if (!d) { $('datasetName').textContent = 'Наборы данных не найдены'; $('datasetDesc').textContent = 'Запустите start.ps1 — он скачает открытые датасеты.'; return; }
  const counts = d.class_counts || {};
  const entries = Object.entries(counts);
  const total = entries.reduce((s, [, v]) => s + v, 0) || 1;
  $('datasetName').textContent = d.name;
  $('datasetDelete').hidden = d.builtin;
  $('datasetDesc').textContent = [d.description, `${intf(d.rows)} строк · ${entries.length} классов`, d.language ? `язык: ${d.language}` : '', d.source].filter(Boolean).join(' · ');
  $('rowsBadge').textContent = intf(d.rows) + ' строк';
  $('distribution').innerHTML = entries.map(([l, v], i) =>
    `<span style="flex:${v};background:${PALETTE[i % PALETTE.length]}" title="${esc(l)}: ${intf(v)} (${(v / total * 100).toFixed(1)}%)"></span>`).join('');
  const shown = entries.slice(0, 8);
  $('distLegend').innerHTML = shown.map(([l, v], i) => `<span><i style="background:${PALETTE[i % PALETTE.length]}"></i>${esc(l)} ${(v / total * 100).toFixed(1)}%</span>`).join('')
    + (entries.length > shown.length ? `<span>… ещё ${entries.length - shown.length}</span>` : '');
  const max = Math.max(500, Math.min(500000, d.rows || 500000));
  const s = $('sampleRows');
  s.max = max;
  s.step = max > 20000 ? 1000 : 100;
  if (Number(s.value) > max || !s.dataset.touched) s.value = Math.min(max, d.kind === 'lenta' ? 50000 : max);
  updateSampleLabel();
}

function updateSampleLabel() {
  const d = currentDataset();
  const v = Number($('sampleRows').value);
  $('sampleValue').textContent = intf(v) + (d && v >= d.rows ? ' (весь набор)' : '');
  checkModelLimits();
}

async function loadDatasets(selectId) {
  state.datasets = await api('/api/datasets');
  renderDatasetMenu(selectId);
  const missing = await api('/api/datasets/missing');
  $('missingNotice').hidden = !missing.length;
  $('missingNotice').innerHTML = missing.length ? `Не загружены: ${missing.map(m => esc(m.name)).join(', ')}. ` +
    `Запустите <code>prepare_data.py</code> (start.ps1 делает это сам)${missing.some(m => m.id === 'lenta') ? ', Lenta — положите <code>lenta-ru-news.csv</code> в <code>data/</code>' : ''}.` : '';
}

$('datasetSelect').onchange = () => { $('sampleRows').dataset.touched = ''; showDataset(); fitSampleToSizes(); };
$('datasetDelete').onclick = async () => {
  const d = currentDataset();
  if (!d || !confirm(`Удалить загруженный набор «${d.name}»? Эксперименты с ним останутся в журнале.`)) return;
  try { await api('/api/datasets/' + d.id, {method: 'DELETE'}); await loadDatasets(); } catch (e) { alert(e.message); }
};

// Загрузка CSV: заголовок читается в браузере, столбцы выбирает пользователь.
let pendingFile = null;
function splitHeader(line) {
  const delim = [',', ';', '\t', '|'].map(d => [d, line.split(d).length]).sort((a, b) => b[1] - a[1])[0][0];
  const out = []; let cur = '', q = false;
  for (const ch of line) {
    if (ch === '"') q = !q; else if (ch === delim && !q) { out.push(cur.trim()); cur = ''; } else cur += ch;
  }
  out.push(cur.trim());
  return out.filter(Boolean);
}
function guess(headers, aliases) { return headers.find(h => aliases.includes(h.toLowerCase())) || ''; }

$('csvFile').onchange = async e => {
  const f = e.target.files[0];
  e.target.value = '';
  if (!f) return;
  pendingFile = f;
  const head = await f.slice(0, 65536).text();
  const headers = splitHeader(head.replace(/^﻿/, '').split(/\r?\n/)[0] || '');
  const opts = headers.map(h => `<option>${esc(h)}</option>`).join('');
  $('uploadText').innerHTML = opts; $('uploadLabel').innerHTML = opts;
  $('uploadText').value = guess(headers, ['text', 'review', 'content', 'message', 'body', 'sentence', 'document', 'текст', 'отзыв']) || headers[0] || '';
  $('uploadLabel').value = guess(headers, ['label', 'category', 'class', 'sentiment', 'topic', 'target', 'y', 'метка', 'класс', 'категория']) || headers[1] || '';
  $('uploadName').value = f.name.replace(/\.[^.]+$/, '');
  $('uploadFile').textContent = `${f.name} · ${(f.size / 1024 / 1024).toFixed(1)} МБ · ${headers.length} столбцов`;
  $('uploadStatus').textContent = '';
  $('uploadForm').hidden = false;
};
$('uploadCancel').onclick = () => { $('uploadForm').hidden = true; pendingFile = null; };
$('uploadForm').onsubmit = async e => {
  e.preventDefault();
  if (!pendingFile) return;
  const fd = new FormData();
  fd.append('file', pendingFile); fd.append('text_column', $('uploadText').value);
  fd.append('label_column', $('uploadLabel').value); fd.append('name', $('uploadName').value);
  $('uploadStatus').textContent = 'Загрузка и проверка…';
  try {
    const d = await api('/api/datasets/upload', {method: 'POST', body: fd});
    $('uploadForm').hidden = true; pendingFile = null;
    await loadDatasets(d.id);
  } catch (err) { $('uploadStatus').textContent = err.message; }
};

// ------------------------------------------------------------ модели

const FAMILY = {baseline: 'Базовая линия', classic: 'Классические', neural: 'Нейросеть', transformer: 'Трансформеры'};
const DEFAULT_ON = ['dummy', 'nb', 'lr', 'svm'];

function renderModels() {
  const fams = ['baseline', 'classic', 'neural', 'transformer'];
  $('modelList').innerHTML = fams.map(f => {
    const items = state.models.filter(m => m.family === f);
    return `<div class="model-group"><span>${FAMILY[f]}</span>${items.map(m => `
      <label class="${m.available ? '' : 'disabled'}" title="${esc(m.reason || m.note || '')}">
        <input type="checkbox" value="${m.key}" ${m.available ? '' : 'disabled'} ${DEFAULT_ON.includes(m.key) && m.available ? 'checked' : ''}>
        <span><b>${esc(m.name)}</b><small>${esc(m.features)}${m.available ? '' : ' · ' + esc(m.reason)}</small></span>
      </label>`).join('')}</div>`;
  }).join('');
  $('modelList').querySelectorAll('input').forEach(i => i.onchange = () => { toggleTransformerOpts(); checkModelLimits(); });
  $('trPresets').innerHTML = state.presets.map(p => `<option value="${esc(p.id)}">${esc(p.note)}</option>`).join('');
  toggleTransformerOpts();
}
function selectedModels() { return [...$('modelList').querySelectorAll('input:checked')].map(x => x.value); }
function toggleTransformerOpts() {
  const sel = selectedModels();
  $('transformerOpts').hidden = !sel.some(k => ['bert', 'emb_lr'].includes(k));
  const p = state.presets.find(p => p.id === $('trModel').value);
  $('trPresetNote').textContent = p ? p.note : 'Любая модель с HuggingFace Hub; при первом запуске скачивается в кэш.';
}
$('trModel').oninput = toggleTransformerOpts;
function checkModelLimits() {
  const sizes = $('mode').value === 'curve' ? parseSizes() : [];
  const train = sizes.length ? sizes[sizes.length - 1] : Number($('sampleRows').value) * (1 - Number($('split').value) / 100);
  const msgs = selectedModels().map(k => state.models.find(m => m.key === k)).filter(m => m && m.max_train_rows && train > m.max_train_rows)
    .map(m => `${m.name}: не более ${intf(m.max_train_rows)} обучающих строк`);
  $('runHint').textContent = msgs.length ? msgs.join('; ') : 'Обучение идёт в фоне на сервере; страницу можно закрыть — результаты сохранятся в SQLite.';
  $('runHint').classList.toggle('warn', msgs.length > 0);
}

$('split').oninput = e => { $('splitValue').textContent = e.target.value + '%'; checkModelLimits(); fitSampleToSizes(); };

function parseSizes() {
  return [...new Set($('trainSizes').value.split(/[\s,;]+/).map(x => parseInt(x.replace(/\D/g, ''), 10)).filter(x => x > 0))].sort((a, b) => a - b);
}
// В режиме кривой объём выборки должен вмещать наибольший размер обучения.
function fitSampleToSizes() {
  if ($('mode').value !== 'curve') return;
  const sizes = parseSizes(), d = currentDataset();
  if (!sizes.length || !d) return;
  const need = Math.ceil(sizes[sizes.length - 1] / (1 - Number($('split').value) / 100));
  const s = $('sampleRows');
  if (Number(s.value) < need) { s.value = Math.min(Number(s.max), need); updateSampleLabel(); }
  const avail = Math.floor(Math.min(Number(s.value), d.rows) * (1 - Number($('split').value) / 100));
  $('modeHint').textContent = `Тест общий для всех размеров; обучающие выборки вложены друг в друга (доступно до ${intf(avail)} строк). ` +
    'Сравнение моделей и сохранённые модели — для наибольшего размера.';
  $('modeHint').classList.toggle('warn', sizes[sizes.length - 1] > avail);
}
$('mode').onchange = () => {
  const curve = $('mode').value === 'curve';
  $('sizesLabel').hidden = !curve; $('modeHint').hidden = !curve;
  if (curve) { const r = $('repeats'); if (Number(r.value) > 3) r.value = 3; }
  fitSampleToSizes(); checkModelLimits();
};
$('trainSizes').oninput = () => { fitSampleToSizes(); checkModelLimits(); };
$('sampleRows').oninput = e => { e.target.dataset.touched = '1'; updateSampleLabel(); };

$('run').onclick = async () => {
  const models = selectedModels();
  if (!models.length) return alert('Выберите хотя бы одну модель.');
  const b = $('run'); b.disabled = true;
  try {
    const r = await postJSON('/api/experiments', {
      dataset_id: $('datasetSelect').value, models, test_size: Number($('split').value) / 100,
      seed: Number($('seed').value), repeats: Number($('repeats').value), max_rows: Number($('sampleRows').value),
      transformer_model: $('trModel').value.trim(), transformer_epochs: Number($('trEpochs').value),
      transformer_max_length: Number($('trLen').value),
      train_sizes: $('mode').value === 'curve' ? parseSizes() : null});
    await openExperiment(r.id);
    loadJournal();
  } catch (e) { alert(e.message); } finally { b.disabled = false; }
};

// ------------------------------------------------------------ результаты

const STATUS = {queued: 'В очереди', running: 'Выполняется', completed: 'Завершён', failed: 'Ошибка', cancelled: 'Остановлен', interrupted: 'Прерван'};

async function openExperiment(id, scroll = true) {
  clearTimeout(state.pollTimer);
  const d = await api('/api/experiments/' + id);
  state.current = d;
  renderExperiment(d);
  if (scroll) $('workspace').scrollIntoView({behavior: 'smooth', block: 'start'});
  if (d.status === 'queued' || d.status === 'running') state.pollTimer = setTimeout(() => pollExperiment(id), 1000);
}
async function pollExperiment(id) {
  if (!state.current || state.current.id !== id) return;
  try {
    const d = await api('/api/experiments/' + id);
    const finished = d.status !== state.current.status && !['queued', 'running'].includes(d.status);
    state.current = d;
    renderExperiment(d);
    if (['queued', 'running'].includes(d.status)) state.pollTimer = setTimeout(() => pollExperiment(id), 1000);
    else if (finished) { loadJournal(); loadOverview(); refreshInspect(); }
  } catch (e) { state.pollTimer = setTimeout(() => pollExperiment(id), 3000); }
}

function renderExperiment(d) {
  $('empty').hidden = true;
  const active = ['queued', 'running'].includes(d.status);
  $('resultTitle').textContent = `Результаты · ${d.dataset_name}`;
  $('jobState').hidden = d.status === 'completed';
  $('resultsContent').hidden = !d.summary.length;
  $('exports').hidden = !d.summary.length;
  for (const [el, kind] of [['expRuns', 'runs'], ['expSummary', 'summary'], ['expCmp', 'comparisons']]) $(el).href = `/api/experiments/${d.id}/export.csv?kind=${kind}`;
  if (d.status !== 'completed') {
    $('jobStatus').textContent = STATUS[d.status] + (d.queue_position ? ` · позиция ${d.queue_position}` : '');
    $('jobStatus').className = 'status-' + d.status;
    $('jobBar').style.width = ((d.progress || 0) * 100).toFixed(1) + '%';
    $('jobMessage').textContent = d.status === 'failed' ? d.error : (d.message || '');
    $('jobCancel').hidden = !active;
    $('jobCancel').onclick = async () => { try { await api(`/api/experiments/${d.id}/cancel`, {method: 'POST'}); } catch (e) { alert(e.message); } };
    $('jobPartial').textContent = d.runs.length ? `Готово прогонов: ${d.runs.length} из ${(d.models || []).length * (d.repeats || 1)}` : '';
  }
  if (d.summary.length) renderSummary(d);
}

function cell(stat, digits = 3, scale = 1) {
  if (!stat || stat.mean == null) return '—';
  const m = (stat.mean * scale).toFixed(digits);
  return stat.std == null ? m : `${m}<small> ± ${(stat.std * scale).toFixed(digits)}</small>`;
}

function renderSummary(d) {
  const s = d.summary;
  const reps = d.repeats || 1;
  const cfg = d.config || {};
  $('runMeta').innerHTML = [
    `train ${intf(d.train_rows)} / test ${intf(d.test_rows)}`, `повторов ${reps} (seed ${d.seed}${reps > 1 ? '…' + (d.seed + reps - 1) : ''})`,
    d.sample_note, d.status !== 'completed' ? `<b class="status-${d.status}">частичные результаты</b>` : '',
  ].filter(Boolean).map(x => `<span>${x}</span>`).join('');

  if (!$('chartMetric').options.length) {
    $('chartMetric').innerHTML = METRICS.map(([k, n, sub]) => `<option value="${k}">${n} ${sub}</option>`).join('');
    $('chartMetric').value = 'f1_macro';
    $('chartMetric').onchange = () => { if (state.current) { renderChart(state.current.summary); renderCurve(state.current); } };
  }
  renderChart(s);
  renderCurve(d);

  const best = Math.max(...s.map(x => x.f1_macro.mean ?? -1));
  $('resultsTable').innerHTML = `<thead><tr><th>Модель</th>${METRICS.map(([, n, sub]) => `<th>${n}${sub ? `<br><small>${sub}</small>` : ''}</th>`).join('')}
    ${COSTS.map(([, n, u]) => `<th>${n}<br><small>${u}</small></th>`).join('')}<th>Размер<br><small>МБ</small></th></tr></thead><tbody>` +
    s.map(r => `<tr class="${r.f1_macro.mean === best ? 'toprow' : ''}"><td>${esc(r.model)}${r.warnings.length ? ` <span class="warn-icon" title="${esc(r.warnings.join('\n'))}">⚠</span>` : ''}</td>
      ${METRICS.map(([k]) => `<td>${cell(r[k])}</td>`).join('')}
      ${COSTS.map(([k, , , dg]) => `<td>${cell(r[k], dg)}</td>`).join('')}
      <td>${r.model_bytes ? (r.model_bytes / 1048576).toFixed(1) : '—'}</td></tr>`).join('') + '</tbody>';
  const warned = s.filter(r => r.warnings.length);
  $('tableNote').innerHTML = (reps > 1 ? 'Значение — среднее ± стандартное отклонение по повторам. ' : 'Один повтор: разброс не оценён — для выводов запускайте 5+ повторов. ') +
    'Время обучения включает построение TF-IDF; инференс — полный путь от текста до метки. ' +
    (warned.length ? `<span class="warn">⚠ Предупреждения: ${warned.map(r => esc(r.model) + ' — ' + esc(r.warnings[0])).join('; ')}</span>` : '');

  const c = d.comparisons || [];
  if (!c.length) {
    $('cmpTable').innerHTML = '';
    $('cmpNote').textContent = d.status === 'completed' ? 'Сравнение доступно при двух и более моделях.' : 'Сравнения будут рассчитаны после завершения всех повторов.';
  } else {
    const sig = p => p == null ? '—' : `<span class="${p < 0.05 ? 'sig' : 'nsig'}">${p < 0.001 ? '<0.001' : p.toFixed(3)}</span>`;
    $('cmpTable').innerHTML = `<thead><tr><th>Модель A</th><th>Модель B</th><th>ΔF1 (A−B)</th><th>t</th><th>p<br><small>скорр. t-тест</small></th><th>p<br><small>Холм</small></th><th>Макнемар<br><small>b / c</small></th><th>p<br><small>Макнемар, Холм</small></th></tr></thead><tbody>` +
      c.map(r => `<tr><td>${esc(r.model_a)}</td><td>${esc(r.model_b)}</td><td>${r.mean_diff > 0 ? '+' : ''}${nf(r.mean_diff, 4)}</td><td>${nf(r.t_stat, 2)}</td>
        <td>${sig(r.p_value)}</td><td>${sig(r.p_holm)}</td><td>${r.mcnemar_b ?? '—'} / ${r.mcnemar_c ?? '—'}</td><td>${sig(r.mcnemar_p_holm)}</td></tr>`).join('') + '</tbody>';
    $('cmpNote').innerHTML = 'Скорректированный t-тест повторных разбиений (Nadeau &amp; Bengio, 2003) по Macro F1 учитывает пересечение обучающих выборок; ' +
      'тест Макнемара сравнивает ошибки на тестовом наборе первого повтора (b — A прав, B ошибся; c — наоборот). ' +
      'Поправка Холма — на множественные сравнения. <span class="sig">Выделено</span> — p &lt; 0.05.' + (reps < 2 ? ' <b>Для t-теста нужно не меньше двух повторов.</b>' : '');
  }

  $('matrixArea').innerHTML = s.filter(r => r.confusion).map(r => {
    const max = Math.max(1, ...r.confusion.flat());
    const short = l => l.length > 14 ? l.slice(0, 13) + '…' : l;
    return `<details class="matrix" ${s.length <= 2 ? 'open' : ''}><summary>${esc(r.model)}</summary>
      <div class="table-wrap"><table><tr><th>истина \\ прогноз</th>${r.labels.map(l => `<th title="${esc(l)}">${esc(short(l))}</th>`).join('')}<th>F1</th><th>n</th></tr>
      ${r.confusion.map((line, i) => `<tr><th title="${esc(r.labels[i])}">${esc(short(r.labels[i]))}</th>${line.map((n, j) =>
        `<td style="background:rgba(50,91,137,${(n / max * (i === j ? 0.55 : 0.85)).toFixed(3)});color:${n / max > 0.5 ? '#fff' : 'inherit'}">${n}</td>`).join('')}
        <td><b>${r.per_class ? nf(r.per_class[i].f1, 2) : '—'}</b></td><td>${r.per_class ? intf(r.per_class[i].support) : ''}</td></tr>`).join('')}</table></div></details>`;
  }).join('');

  $('envInfo').textContent = JSON.stringify({config: cfg, models: Object.fromEntries(s.map(r => [r.model, r.params])), environment: d.environment, class_counts: d.class_counts}, null, 2);
}

function renderCurve(d) {
  const curve = d.curve || [];
  $('curveBlock').hidden = !curve.length;
  if (!curve.length) return;
  const key = $('chartMetric').value || 'f1_macro';
  const label = $('chartMetric').selectedOptions[0]?.textContent || key;
  const sizes = d.train_sizes;
  const models = [...new Map(curve.map(c => [c.model_key, c.model])).entries()];
  const val = (m, n) => curve.find(c => c.model_key === m && c.train_size === n)?.[key];
  const vals = curve.map(c => c[key]).filter(v => v && v.mean != null);
  let lo = Math.min(...vals.map(v => v.mean - (v.ci95 || 0))), hi = Math.max(...vals.map(v => v.mean + (v.ci95 || 0)));
  const pad = (hi - lo) * 0.08 || 0.05; lo = Math.max(-1, lo - pad); hi = Math.min(1, hi + pad);
  const W = 860, H = 320, L = 52, R = 190, T = 16, B = 40;
  const lx = Math.log(sizes[0]), rx = Math.log(sizes[sizes.length - 1]);
  const X = n => L + (Math.log(n) - lx) / ((rx - lx) || 1) * (W - L - R);
  const Y = v => T + (hi - v) / ((hi - lo) || 1) * (H - T - B);
  const ticks = [...Array(5)].map((_, i) => lo + (hi - lo) * i / 4);
  let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Кривая обучения: ${esc(label)}">`;
  svg += ticks.map(t => `<line x1="${L}" x2="${W - R}" y1="${Y(t)}" y2="${Y(t)}" class="grid"/><text x="${L - 6}" y="${Y(t) + 4}" text-anchor="end">${t.toFixed(2)}</text>`).join('');
  svg += sizes.map(n => `<text x="${X(n)}" y="${H - B + 18}" text-anchor="middle">${n >= 1000 ? (n / 1000).toLocaleString('ru-RU') + 'k' : n}</text>`).join('');
  svg += `<text x="${(L + W - R) / 2}" y="${H - 4}" text-anchor="middle">Размер обучающей выборки (лог. шкала)</text>`;
  models.forEach(([m, name], i) => {
    const col = LINE_COLORS[i % LINE_COLORS.length];
    const pts = sizes.map(n => [n, val(m, n)]).filter(([, v]) => v && v.mean != null);
    svg += `<polyline fill="none" stroke="${col}" stroke-width="2" points="${pts.map(([n, v]) => `${X(n)},${Y(v.mean)}`).join(' ')}"/>`;
    svg += pts.map(([n, v]) => `${v.ci95 ? `<line x1="${X(n)}" x2="${X(n)}" y1="${Y(v.mean - v.ci95)}" y2="${Y(v.mean + v.ci95)}" stroke="${col}" stroke-width="1.2"/>` : ''}<circle cx="${X(n)}" cy="${Y(v.mean)}" r="3.5" fill="${col}"><title>${esc(name)}, ${intf(n)}: ${nf(v.mean)}${v.ci95 ? ' ± ' + nf(v.ci95) : ''}</title></circle>`).join('');
    const last = pts[pts.length - 1];
    svg += `<rect x="${W - R + 12}" y="${T + i * 20}" width="12" height="3" fill="${col}"/><text x="${W - R + 30}" y="${T + i * 20 + 5}">${esc(name.length > 24 ? name.slice(0, 23) + '…' : name)}</text>`;
  });
  svg += '</svg>';
  $('curveChart').innerHTML = svg;
  $('curveTable').innerHTML = `<thead><tr><th>Модель \\ train</th>${sizes.map(n => `<th>${intf(n)}</th>`).join('')}</tr></thead><tbody>` +
    models.map(([m, name]) => `<tr><td>${esc(name)}</td>${sizes.map(n => `<td>${cell(val(m, n))}</td>`).join('')}</tr>`).join('') + '</tbody>';
  $('summaryTitle').textContent = `Сравнение при наибольшем объёме (train ${intf(sizes[sizes.length - 1])})`;
}

function renderChart(s) {
  const key = $('chartMetric').value || 'f1_macro';
  const lo = Math.min(0, ...s.map(r => (r[key].mean ?? 0) - (r[key].ci95 || 0)));
  const span = 1 - lo;
  const pos = v => ((v - lo) / span * 100).toFixed(2);
  $('chart').innerHTML = s.map(r => {
    const st = r[key];
    if (st.mean == null) return '';
    const ci = st.ci95 || 0;
    return `<div class="hbar"><span class="hbar-name">${esc(r.model)}</span><div class="hbar-track">
      <i class="hbar-fill" style="left:${pos(Math.min(0, st.mean))}%;width:${(Math.abs(st.mean) / span * 100).toFixed(2)}%"></i>
      ${ci ? `<i class="hbar-ci" style="left:${pos(st.mean - ci)}%;width:${(2 * ci / span * 100).toFixed(2)}%"></i>` : ''}
      </div><b>${nf(st.mean)}${ci ? `<small> ±${nf(ci)}</small>` : ''}</b></div>`;
  }).join('');
}

// ------------------------------------------------------------ журнал и сводка

async function loadJournal() {
  const items = await api('/api/experiments');
  $('journalTable').innerHTML = items.length ? `<thead><tr><th>Дата</th><th>Набор</th><th>Статус</th><th>Модели</th><th>Повторы</th><th>Train / test</th><th>Лучшая по Macro F1</th><th></th></tr></thead><tbody>` +
    items.map(e => `<tr class="clickable ${state.current && state.current.id === e.id ? 'selected' : ''}" data-id="${e.id}">
      <td>${new Date(e.created_at).toLocaleString('ru-RU')}</td><td>${esc(e.dataset_name)}</td>
      <td><span class="chip status-${e.status}">${STATUS[e.status] || esc(e.status)}${e.status === 'running' ? ' ' + Math.round((e.progress || 0) * 100) + '%' : ''}</span></td>
      <td>${e.n_models || (e.models || []).length}</td><td>${e.repeats || 1}</td><td>${intf(e.train_rows)} / ${intf(e.test_rows)}</td>
      <td>${e.best_model ? `${esc(e.best_model)} · <b>${nf(e.best_f1)}</b>` : '—'}</td>
      <td><button class="link danger" data-del="${e.id}" title="Удалить запуск и его модели">Удалить</button></td></tr>`).join('') + '</tbody>'
    : '<tbody><tr><td>Проведённые эксперименты появятся здесь.</td></tr></tbody>';
  $('journalTable').querySelectorAll('tr[data-id]').forEach(tr => tr.onclick = ev => {
    if (ev.target.dataset.del) return;
    openExperiment(tr.dataset.id).then(loadJournal);
  });
  $('journalTable').querySelectorAll('[data-del]').forEach(b => b.onclick = async () => {
    if (!confirm('Удалить эксперимент вместе с сохранёнными моделями?')) return;
    try {
      await api('/api/experiments/' + b.dataset.del, {method: 'DELETE'});
      if (state.current && state.current.id === b.dataset.del) { state.current = null; $('resultsContent').hidden = true; $('jobState').hidden = true; $('exports').hidden = true; $('empty').hidden = false; }
      loadJournal(); loadOverview(); refreshInspect();
    } catch (e) { alert(e.message); }
  });
  clearTimeout(state.journalTimer);
  if (items.some(e => ['queued', 'running'].includes(e.status))) state.journalTimer = setTimeout(loadJournal, 3000);
  return items;
}
$('refreshJournal').onclick = () => loadJournal();

async function loadOverview() {
  const rows = await api('/api/analysis');
  $('overviewTable').innerHTML = rows.length ? `<thead><tr><th>Набор</th><th>Train</th><th>Модель</th><th>Запусков</th><th>Accuracy</th><th>F1 macro</th><th>MCC</th><th>Обучение, с</th></tr></thead><tbody>` +
    rows.map(r => `<tr><td>${esc(r.dataset)}</td><td>${intf(r.train_rows)}</td><td>${esc(r.model)}</td><td>${r.runs}</td>
      <td>${cell(r.accuracy)}</td><td>${cell(r.f1_macro)}</td><td>${cell(r.mcc)}</td><td>${cell(r.train_seconds, 1)}</td></tr>`).join('') + '</tbody>'
    : '<tbody><tr><td>Пока нет завершённых экспериментов.</td></tr></tbody>';
}
$('refreshOverview').onclick = () => loadOverview();

// ------------------------------------------------------------ проверка текста

async function refreshInspect() {
  const items = (await api('/api/experiments')).filter(e => e.status === 'completed');
  const prev = $('inspectExperiment').value;
  $('inspectExperiment').innerHTML = items.map(e => `<option value="${esc(e.id)}">${esc(new Date(e.created_at).toLocaleString('ru-RU'))} · ${esc(e.dataset_name)} · ${e.n_models} моделей · ${intf(e.train_rows)} обучающих строк</option>`).join('');
  if (items.some(e => e.id === prev)) $('inspectExperiment').value = prev;
  $('inspectRun').disabled = !items.length;
  $('inspectStatus').textContent = items.length ? '' : 'Сначала обучите хотя бы одну модель.';
}
$('inspectRun').onclick = async () => {
  const button = $('inspectRun'); button.disabled = true;
  $('inspectStatus').textContent = 'Применяем сохранённые модели…'; $('inspectOutput').replaceChildren();
  try {
    const response = await postJSON('/api/inspect', {experiment_id: $('inspectExperiment').value, text: $('inspectText').value});
    $('inspectOutput').innerHTML = response.results.map(r => {
      if (r.error) return `<article class="inspect-card"><h3>${esc(r.model)}</h3><p>${esc(r.error)}</p></article>`;
      const maximum = Math.max(...r.contributions.map(v => Math.abs(v.value)), 1e-6);
      const prob = r.score_kind.startsWith('Вероятность');
      return `<article class="inspect-card"><h3>${esc(r.model)}</h3><div class="prediction">${esc(r.prediction)}</div>
        ${r.known_terms === 0 ? '<p class="inspect-warning">В тексте нет признаков из словаря модели — ответ определяется только смещением модели.</p>' : ''}
        <p class="inspect-note">${esc(r.score_kind)}</p>
        <div class="class-scores">${r.scores.map(s => `<div><span>${esc(s.label)}</span><b>${prob ? (s.value * 100).toFixed(1) + '%' : s.value.toFixed(3)}</b></div>`).join('')}</div>
        <h4>Вклад признаков</h4><p class="inspect-note">${esc(r.explanation)}</p>
        ${r.contributions.map(c => `<div class="contribution"><span>${esc(c.term)}</span><div class="contribution-track"><i style="width:${Math.abs(c.value) / maximum * 100}%;background:${c.value >= 0 ? '#3f7f6a' : '#a75d5d'}"></i></div><b>${c.value > 0 ? '+' : ''}${c.value.toFixed(3)}</b></div>`).join('')}
      </article>`;
    }).join('');
    $('inspectStatus').textContent = 'Готово. Шкалы оценок у моделей разные; вклад слов относится к классу, выбранному каждой моделью.';
  } catch (e) { $('inspectStatus').textContent = e.message; } finally { button.disabled = false; }
};

// ------------------------------------------------------------ старт

(async function init() {
  try {
    const [m, sys] = await Promise.all([api('/api/models'), api('/api/system')]);
    state.models = m.models; state.presets = m.transformer_presets;
    renderModels();
    const hw = [sys.cpu, sys.cpu_threads ? sys.cpu_threads + ' потоков' : '', sys.ram_gb ? sys.ram_gb + ' ГБ RAM' : '', sys.gpu || (sys.torch ? 'GPU не найден' : '')].filter(Boolean).join(' · ');
    $('footerEnv').textContent = `${hw} · Python ${sys.python} · scikit-learn ${sys.sklearn}${sys.torch ? ' · PyTorch ' + sys.torch : ''}`;
    await loadDatasets();
    const items = await loadJournal();
    loadOverview(); refreshInspect();
    const active = items.find(e => ['queued', 'running'].includes(e.status));
    if (active) openExperiment(active.id, false);
  } catch (e) {
    alert('Не удалось подключиться к серверу: ' + e.message + '\nЗапустите start.ps1.');
  }
})();
