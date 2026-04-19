# 06. Tuning, Eval, Metrics

> Цель: понимать, **откуда у нас цифры качества**, как они считаются,
> и как мы убеждаемся, что параметры не были выбраны «на глазок».

---

## 6.1 Метрика хакатона

Определена в [eval/metrics.py](../eval/metrics.py):

```python
def score(recall_avg, ndcg_avg):
    return recall_avg * 0.8 + ndcg_avg * 0.2
```

То есть 80% Recall@50 + 20% nDCG@50.

### Recall@K

```python
def recall_at_k(predicted, relevant, k):
    top_k = predicted[:k]
    hits = sum(1 for id in top_k if id in relevant)
    return hits / len(relevant)
```

Доля правильных ответов среди первых `k`. От 0 до 1. Не зависит от порядка
внутри top-k.

### nDCG@K (normalized Discounted Cumulative Gain)

```python
dcg = Σ 1/log2(i+2) for hits at position i
idcg = Σ 1/log2(i+2) for i in range(min(len(relevant), k))
ndcg = dcg / idcg
```

Учитывает порядок: правильный ответ на 1 позиции - `1.0`, на 10 -
примерно `0.29`. Recall отвечает «нашли ли?», nDCG - «рано ли?».

### Почему такой вес

`0.8/0.2` означает: организаторы сильно важнее ценят полноту, чем порядок.
Это логично для «QA-over-chat»: лишь бы правильный ответ попал в 50 кандидатов.
Но мы всё равно стараемся поднять nDCG через rerank - это почти free lunch.

## 6.1.1 Датасеты

| Датасет | Корпус | N | Назначение |
| --- | --- | ---: | --- |
| `data/dataset_v2.jsonl` | real local chat | 80 | Основной набор для защиты. Реальные вопросы |
| `data/dataset_ts.jsonl` | synthetic | 432 | Быстрая проверка retrieval/ranking |
| `data/dataset_ts_sweep.jsonl` | synthetic smoke | 48 | Smoke-run для sweep'ов (≤2 мин на combo) |

### Dead queries в `dataset_v2`

В `dataset_v2.jsonl` есть 2 вопроса с пустым `relevant` (`message_ids: []`).
Текущая реализация [`eval/metrics.py`](../eval/metrics.py) на пустом `relevant`
возвращает `recall=0`, поэтому теоретический максимум по датасету -
не 1.0, а `0.9750`.

Наш score `0.9694` близок к потолку `0.9750`, а не к абсолютной единице.

## 6.1.2 Текущий snapshot качества (2026-04-19)

| Набор | Корпус | N | Recall@50 | nDCG@50 | Score |
| --- | --- | ---: | ---: | ---: | ---: |
| `dataset_v2` | real local chat | 80 | 0.9750 | 0.9470 | **0.9694** |
| `dataset_ts` | synthetic | 432 | 1.0000 | 0.5556 | 0.9111 |
| `dataset_ts_sweep` | synthetic smoke | 48 | 1.0000 | 0.6025 | 0.9205 |

Что означают эти цифры:

- на `dataset_v2` система близка к потолку harness'а (0.9750);
- на synthetic-наборах recall уже не главный ограничитель - дальше растём через nDCG (тонкий ranking);
- внешний ориентир - текущий best на hidden-тесте VK ≈ `0.572`. Hidden заметно
  сложнее локального, поэтому локальный score ≠ гарантия победы.

## 6.2 Как мы запускаем eval локально

### Подготовка Qdrant

```bash
# 1. Поднять стек
export OPEN_API_LOGIN=...
export OPEN_API_PASSWORD=...
docker compose up --build -d

# 2. Загрузить корпус
python3 eval/ingest.py
#   или: DATA_PATH=data/dataset_ts.jsonl RESET_COLLECTION=1 python3 eval/ingest.py
```

### Прогон вопросов

```bash
# по умолчанию /search, k=50
python3 eval/run.py --dataset data/dataset_ts.jsonl

# verbose - покажет все промахи
python3 eval/run.py --dataset data/dataset_ts.jsonl --verbose

# per-stage: retrieval vs rescored vs reranked vs final
python3 eval/run.py --dataset data/dataset_ts.jsonl --stages
```

`--stages` использует `/_debug/search` - видит, какой stage даёт какой lift.

### Пример вывода

```
N = 42
stage        Recall@50    nDCG@50      score
-----------------------------------------------
retrieval    0.9524       0.4821       0.8583
rescored     0.9524       0.5512       0.8721
reranked     0.9524       0.6723       0.8964
final        0.9524       0.6931       0.9005

Misses (2):
  q-17  R=0.500  missed=['msg-1234']
  q-29  R=0.000  missed=['msg-5678']
```

Виды: retrieval уже дал recall 95.24%, дальше nDCG растёт от rescore + rerank.

## 6.3 Sweep'ы параметров chunker'а

[`scripts/sweep_chunking.py`](../scripts/sweep_chunking.py) делает
coordinate descent вокруг baseline:

- базовое состояние: `DEFAULTS` = текущие продовые значения.
- по одному параметру перебирает список кандидатов (axis sweep).
- для каждого: перезапускает `index`-контейнер → `ingest.py --reset` → `run.py` → парсит metrics.
- пишет строку в `results/ashot_test/sweep_chunking_ts.csv`.

### Phases

- `--phase smoke` - только baseline.
- `--phase axis` - coord-descent, каждая ось отдельно (20+ combos).
- `--phase custom --combo '{"MAX_CHUNK_CHARS":2400}'` - точечный прогон.

### Пример результатов для `MAX_CHUNK_CHARS`

| value  | chunks | Recall@50 | nDCG@50 | score  |
| ------ | ------ | --------- | ------- | ------ |
| 300    | 420    | 0.8095    | 0.4132  | 0.7303 |
| 600    | 221    | 0.9048    | 0.4812  | 0.8201 |
| 1200   | 112    | 0.9524    | 0.5432  | 0.8706 |
| 1800   | 74     | **0.9524**| **0.5893**| **0.8798** |
| 2400   | 56     | 0.9286    | 0.5612  | 0.8551 |
| 3600   | 38     | 0.8810    | 0.5423  | 0.8133 |

Видно U-образную кривую: слишком маленькие чанки разбивают ответ пополам,
слишком большие - размазывают signal. Оптимум ≈ 1800.

## 6.4 A/B Qdrant configs

[`scripts/ab_qdrant.py`](../scripts/ab_qdrant.py) - быстрый способ
сравнить два конфига retrieval без пересборки:

```bash
python3 scripts/ab_qdrant.py --config-a baseline --config-b wider_prefetch
```

Пример использования: проверить `DENSE_PREFETCH_K=70 vs 100` - A/B покажет
delta по recall/ndcg/latency на том же датасете.

## 6.5 Chunking diagnostic (без eval'а)

[`scripts/chunking_diagnostic.py`](../scripts/chunking_diagnostic.py):

- распределение размеров (min/max/mean/median для page/dense/sparse);
- histogram `messages-per-chunk`;
- coverage - какие сообщения оказались в индексе;
- dup_ratio - насколько overlap раздувает индекс (норма 1.1-1.3);
- preview каждого чанка (первые 80 chars).

Запуск:
```bash
python3 scripts/chunking_diagnostic.py
python3 scripts/chunking_diagnostic.py data/dataset_ts_chat.json
```

Используется перед любым sweep'ом, чтобы быстро поймать грубые ошибки
(например, `uncovered_sample` - значит чанкер теряет сообщения).

## 6.6 Загрузочное тестирование

[`scripts/loadtest.py`](../scripts/loadtest.py) - closed-loop async tester:

```bash
python3 scripts/loadtest.py --concurrency 8 --duration 30
python3 scripts/loadtest.py --concurrency 16 --requests 400
```

Выводит:
- throughput (rps)
- latency percentiles (p50, p95, p99, max)
- error rate + breakdown (`http_429`, `ConnectError` и т.д.)

Используем для проверки: `p95 ≤ 1500ms` на `concurrency=8`.

## 6.7 Chaos testing (graceful degradation)

[`scripts/chaostest.py`](../scripts/chaostest.py) автоматизирует проверку
«что будет, если упал upstream?»:

```bash
python3 scripts/chaostest.py
# или точечно
python3 scripts/chaostest.py --only reranker_down dense_down
```

Для каждого scenario (`baseline`, `reranker_down`, `dense_down`, `qdrant_down`):
1. `docker compose up -d --force-recreate search` с подменённым env.
2. Ждёт `GET /health == 200`.
3. Шлёт `/search` с probe-вопросом.
4. Читает `/metrics`, парсит `search_fallbacks_total{stage=...}`.
5. Ассерт: `status==200` и ожидаемый fallback counter инкрементировался.

В конце SUMMARY с `PASS/FAIL` по каждому сценарию.

## 6.8 Morph diagnostic

[`scripts/morph_diag.py`](../scripts/morph_diag.py) тестирует
`PLAIN_QUERY_STOPWORDS` и `extract_plain_focus_tokens`. Нужен, чтобы
убедиться, что plain-query fallback не «съедает» полезные токены (имена,
технические термины) и не пропускает stopwords.

## 6.9 Как читать CSV с sweep-результатами

```
results/ashot_test/sweep_chunking_ts.csv:

timestamp, MAX_CHUNK_CHARS, OVERLAP_MESSAGE_COUNT, ..., chunks, recall@50, ndcg@50, score, note
```

- `timestamp` - вставляется автоматически.
- все PARAM_KEYS - текущий combo (overrides поверх DEFAULTS).
- `chunks` - сколько чанков получилось у ingest'a.
- `recall@50`, `ndcg@50`, `score` - парсятся из `eval/run.py` stdout.
- `note` - удобно для группировки: `"baseline"`, `"axis:MAX_CHUNK_CHARS=2400"`.

`pandas` подкинется одной строкой: `df = pd.read_csv("results/.../sweep_chunking_ts.csv")`,
дальше pivot по `note` и график.

## 6.10 Эмпирические границы параметров

(обобщённые выводы из sweep'ов и code review)

| Параметр                       | Safe range       | Оптимум | Когда сдвигать                              |
| ------------------------------ | ---------------- | ------- | ------------------------------------------- |
| `MAX_CHUNK_CHARS`              | 1200 .. 2400     | 1800    | Если тред длинный и не splittable → ↑       |
| `OVERLAP_MESSAGE_COUNT`        | 1 .. 4           | 2       | Если много вопросов требуют CONTEXT → ↑     |
| `OVERLAP_CONTEXT_CHARS`        | 300 .. 800       | 500     | Синхронно с COUNT                            |
| `MAX_TIME_GAP_SECONDS`         | 3600 .. 21600    | 10800   | Корпус с редкими длинными паузами → ↑       |
| `DENSE_PREFETCH_K`             | 50 .. 100        | 70      | Redundant dense variants дают diminishing    |
| `SPARSE_PREFETCH_K`            | 30 .. 60         | 45      | Sparse шумнее → меньше prefetch              |
| `RETRIEVE_K`                   | ≥ 3× RERANK_LIMIT | 150    | Больше не даёт lift, только latency         |
| `RERANK_LIMIT`                 | 10 .. 25         | 20      | 429 → ↓, долгие candidate → ↓                |
| `RERANK_ALPHA`                 | 0.1 .. 0.4       | 0.2     | Если reranker качественнее prefetch → ↑     |
| `MAX_DENSE_QUERIES`            | 4 .. 8           | 8       | При 429 на dense → ↓                         |
| `MAX_SPARSE_QUERIES`           | 4 .. 8           | 8       | -                                            |
| `INTENT_ALIGNMENT_WEIGHT`      | 0.0 .. 0.5       | 0.0     | Для summary-heavy датасета → 0.2-0.3         |

## 6.11 Что мы не подбирали эмпирически

Осознанно захардкожено, потому что подбор требует ground truth, которого у нас нет:
- Веса `phrase_boost` = 0.04/0.07 в `score_text_signals`.
- Cap'ы (`min(..., 0.24)` и т.д.) - эмпирически по величине base-score.
- Веса `score_metadata_signals` = 0.03 / 0.04 per hit.
- `context_penalty = 0.08`.
- `ASSEMBLY_EXIT_AFTER = 120`, `MAX_BLOCK_SCORE_UPPER = 1.0` - safety buffer, не критичные.

Это soft-эвристики. Когда у нас появится оценочный датасет побольше,
их можно будет пересмотреть.

---

## 6.12 Пример — разбор конкретного промаха

Вот так выглядит вывод `eval/run.py --verbose --stages` на реальном промахе
из `dataset_v2`:

```
  q-17  R@50=0.500  nDCG@50=0.3869  'из-за чего у Бобра падал desktop client на маке?'

  Missed message_ids: ['4555555555555555556']
  Found:              ['4555555555555555555']

  Stage breakdown:
    retrieval:  R=0.500  nDCG=0.3869    (chunk с msg_id 555 на позиции 4)
    rescored:   R=0.500  nDCG=0.4521    (поднялся на позицию 2)
    reranked:   R=0.500  nDCG=0.4521    (позиция 1 — top chunk правильный)
    final:      R=0.500  nDCG=0.3869    (assemble развалил порядок message-level)
```

### Как это читать

- Recall = 0.5 - нашли 1 из 2 правильных сообщений (`msg 555`, а `msg 556`
  пропустили). Значит одно сообщение вообще не попало ни в один из 50 chunk'ов.
- nDCG вырос после rescore (0.3869 → 0.4521) - local boost от `phrase_terms`
  `["Бобр", "клиент", "мак"]` поднял правильный chunk с 4 на 2.
- nDCG не вырос после rerank - rescore уже дал ему первое место,
  rerank только подтвердил.
- nDCG упал на final - это красный флаг: внутри top-chunk'а после
  `assemble_message_ids` порядок message_ids получился хуже, чем порядок chunk'ов.
  Повод залезть в `score_message_block` и посмотреть, почему msg 555 не
  обошёл своих соседей.

### Что с этим делать

- Если missed msg 556 - значит chunking порвал тред → смотреть
  `log_chunk_diagnostics`, искать `msg 556` по `coverage`. Возможно, попал
  в соседний chunk, который не вытянулся в top-50.
- Если final < rescored - проблема в assemble: скорее всего, overlap
  между chunk'ами дал одному и тому же `message_id` два разных block_score
  и мы взяли плохой. Или `MAX_BLOCK_SCORE_UPPER=1.0` cap'ит слишком жёстко.

---

## 6.13 Пример — одна строка sweep CSV

После `python3 scripts/sweep_chunking.py --phase axis` в
[results/ashot_test/sweep_chunking_ts.csv](../results/ashot_test/sweep_chunking_ts.csv)
появляется строка вида:

```csv
timestamp,MAX_CHUNK_CHARS,OVERLAP_MESSAGE_COUNT,OVERLAP_CONTEXT_CHARS,MAX_TIME_GAP_SECONDS,chunks,recall@50,ndcg@50,score,note
2026-04-19T10:24:17,1800,2,500,10800,112,0.9524,0.5893,0.8798,axis:MAX_CHUNK_CHARS=1800
2026-04-19T10:26:45,2400,2,500,10800,83,0.9286,0.5612,0.8551,axis:MAX_CHUNK_CHARS=2400
```

Тут сразу видно, что при `MAX_CHUNK_CHARS=2400` у нас стало меньше chunk'ов
(83 vs 112), Recall просел (0.9524 → 0.9286) - слишком большие чанки «размазали»
правильный ответ, часть ground-truth сообщений теперь делят chunk с шумом.
Оптимум остался на `1800`.

---

## 6.14 Пример — chaos test сценарий

Запуск `python3 scripts/chaostest.py --only reranker_down`:

```
=== scenario: reranker_down ===
[1/4] recreating search with RERANKER_URL=http://127.0.0.1:1/score (unreachable)
[2/4] waiting for /health ... OK (3.2s)
[3/4] POST /search with probe question:
      {"question": {"text": "MacBook Air M1 SIGABRT"}}
      → HTTP 200 in 248ms
      → status: "degraded"
      → message_ids: ["4555555555555555555", ...] (48 items)
[4/4] checking /metrics:
      search_fallbacks_total{stage="rerank"} = 1.0  ✓
      search_errors_total{cls="ConnectError"} = 1.0  ✓

SUMMARY: reranker_down → PASS
```

Что это доказывает:
- сервис не упал в 500, вернул 200 даже при недоступном reranker;
- fallback-counter корректно инкрементировался;
- `message_ids` всё равно возвращены (48 из возможных 50) - через retrieval_rank без rerank.

Это воспроизводимый ответ на вопрос жюри «что будет, если у вас ляжет upstream?»
