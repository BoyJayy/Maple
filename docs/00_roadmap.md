# 00. Roadmap - начать отсюда

> Документ для человека, который впервые открыл репо. За 10 минут чтения
> вы поймёте, что это такое, как это запустить, где лежит код и в каком
> порядке читать остальные доки.

---

## 0.1 Что это за репозиторий одним абзацем

Решение для хакатона «Индексация и поиск по сообщениям». На вход приходит
чат (JSON со списком сообщений) и enriched-вопрос (текст + ключевые слова
+ сущности + hyde). На выход наш `search` возвращает список `message_ids`
сообщений из этого чата, которые отвечают на вопрос. Оценка считается по
формуле `score = 0.8 × Recall@50 + 0.2 × nDCG@50`. Внутри - два FastAPI
сервиса (`index` и `search`) поверх Qdrant, гибридный retrieval (dense +
sparse) с fusion и cross-encoder rerank сверху.

Текущий snapshot качества на реальном датасете `dataset_v2` (N=80):
Recall 0.9750, nDCG 0.9470, score 0.9694 (theoretical max = 0.9750, две
dead queries в датасете).

## 0.2 Карта репозитория

```
Hackathon-Search-Engine-1/
├── index/                # FastAPI сервис индексации (порт 8001)
│   ├── main.py           # /health, /index, /sparse_embedding
│   ├── chunking.py       # normalize → filter → split → chunk → format
│   ├── sparse.py         # локальный BM25 через fastembed + кеш
│   ├── config.py         # env-параметры, thresholds, limits
│   ├── schemas.py        # Pydantic модели запросов/ответов
│   └── Dockerfile / Makefile / requirements.txt
│
├── search/               # FastAPI сервис поиска (порт 8002)
│   ├── main.py           # весь pipeline: context → retrieval → rescore → rerank → assemble
│   └── Dockerfile / Makefile / requirements.txt
│
├── eval/                 # локальная оценка качества
│   ├── ingest.py         # orchestrator: /index → dense API → /sparse_embedding → Qdrant upsert
│   ├── run.py            # гоняет датасет вопросов → Recall@K, nDCG@K, score
│   └── metrics.py        # реализация метрик
│
├── scripts/              # dev/QA инструменты
│   ├── chunking_diagnostic.py  # покрытие сообщений, размеры чанков, dup_ratio
│   ├── sweep_chunking.py       # coord-descent по параметрам чанкера
│   ├── build_ts_chat.py        # сборка synthetic TS-чата под sweep
│   ├── loadtest.py             # async p50/p95/p99 для /search
│   ├── chaostest.py            # kill upstream → assert 200 (graceful)
│   ├── ab_qdrant.py            # A/B сравнение конфигов Qdrant
│   └── morph_diag.py           # диагностика морфологии для stopword filtering
│
├── data/                 # датасеты для eval
│   ├── dataset_ts_chat.json      # локальный пример чата
│   ├── dataset_v2.jsonl          # production eval set (N=80)
│   ├── dataset_ts*.{json,jsonl}  # synthetic TS-датасет для sweep'ов
│   └── index_request_sample.json # пример request body для /index
│
├── docs/                 # вся документация (вы здесь)
├── docker-compose.yml    # qdrant + index + search
├── api.md                # OpenAPI-style контракты трёх endpoint'ов
├── hackathon_roadmap.md  # roadmap по доработке решения (старый)
├── ТЗ_на_хакатон_*.pdf   # исходное ТЗ
└── CLAUDE.md             # guidelines для LLM-ассистентов в репо
```

## 0.3 Первые 15 минут: поднять и пощупать

### Шаг 1 - учётки внешнего API

```bash
export OPEN_API_LOGIN=...
export OPEN_API_PASSWORD=...
```

Хакатон даёт внешний API с dense embeddings (Qwen3-0.6B, 1024-dim) и
cross-encoder reranker (nvidia/nemotron-rerank-1b-v2). Адреса:
`http://83.166.249.64:18001/embeddings` и `/score`.

### Шаг 2 - поднять стек

```bash
docker compose up --build
```

Поднимутся три контейнера:

- `qdrant` на `localhost:6333` (vector DB)
- `index` на `localhost:8001`
- `search` на `localhost:8002`

### Шаг 3 - проиндексировать тестовый чат

```bash
python3 eval/ingest.py
```

Скрипт вызовет `/index` на `dataset_ts_chat.json`, посчитает dense embeddings,
получит sparse через `/sparse_embedding`, запушит всё в Qdrant.

### Шаг 4 - задать вопрос

```bash
curl -X POST "http://localhost:8002/search" \
  -H "Content-Type: application/json" \
  -d '{"question": {"text": "где писали про SIGABRT на M1?"}}'
```

Ответ: `{"message_ids": ["4555...", "4566..."], "status": "ok"}`.

### Шаг 5 - прогнать метрики

```bash
python3 eval/run.py --dataset data/dataset_v2.jsonl --k 50 --verbose
```

Выведет таблицу с Recall@50 / nDCG@50 / score по датасету.

## 0.4 Два сервиса, две ответственности

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ index-service│ ──> │   Qdrant     │ <── │search-service│
│  chunking +  │     │ hybrid vector│     │ retrieval +  │
│ sparse emb.  │     │   database   │     │fusion+rerank │
└──────────────┘     └──────────────┘     └──────────────┘
```

- `index` отвечает за КАЧЕСТВО ЧАНКОВ. Если падает Recall - копать сюда.
- `search` отвечает за КАЧЕСТВО РАНЖИРОВАНИЯ. Если падает nDCG - копать сюда.
- Qdrant отвечает за ХРАНЕНИЕ и PREFETCH (hybrid через `Query API` + DBSF fusion).

## 0.5 Ключевые цифры, которые полезно помнить

| Параметр | Значение | Где настраивается |
| --- | --- | --- |
| Главная метрика | `0.8 × Recall@50 + 0.2 × nDCG@50` | [eval/metrics.py](../eval/metrics.py) |
| Snapshot score | 0.9694 (2026-04-19, dataset_v2) | [docs/06_tuning_and_eval.md](06_tuning_and_eval.md) |
| MAX_CHUNK_CHARS | 1800 | [index/config.py](../index/config.py) |
| CHUNK_OVERLAP | 1 message | [index/config.py](../index/config.py) |
| DENSE_PREFETCH_K | 70 | search env |
| RETRIEVE_K | 150 | search env |
| RERANK_LIMIT | 20 | search env |
| RERANK_ALPHA | 0.2 (blend с retrieval order) | search env |
| Dense model | Qwen3-Embedding-0.6B, 1024-dim | external API |
| Sparse model | `Qdrant/bm25` через fastembed | локально в `index` |
| Reranker | nvidia/llama-nemotron-rerank-1b-v2 | external API |

## 0.6 Контракты API - их менять нельзя

Три endpoint'а, заморожены по ТЗ:

- `POST /index` (index-service) - принимает чат, возвращает chunks.
- `POST /sparse_embedding` (index-service) - возвращает sparse vectors.
- `POST /search` (search-service) - принимает вопрос, возвращает `message_ids`.

Любая внутренняя логика может меняться. Полные схемы - в [api.md](../api.md).

## 0.7 Как читать доки - три маршрута

### Путь A - Понять систему (60 минут, новый участник)

1. [01_architecture.md](01_architecture.md) - общая карта и env-параметры.
2. [02_index_deep_dive.md](02_index_deep_dive.md) - chunking + worked example (§2.15) + API reference всех функций (§2.16).
3. [03_search_deep_dive.md](03_search_deep_dive.md) - search pipeline + API reference всех функций (§3.20).
4. [04_data_flow.md](04_data_flow.md) - сквозной пример: 3 сообщения → чанк → Qdrant point → ответ.
5. [05_qdrant_and_retrieval.md](05_qdrant_and_retrieval.md) - теория hybrid retrieval, DBSF vs RRF.
6. [06_tuning_and_eval.md](06_tuning_and_eval.md) - датасеты, snapshot, sweep'ы.
7. [07_ops_and_deploy.md](07_ops_and_deploy.md) - Docker, compose, registry, observability.

### Путь B - Защита (20 минут)

1. [08_defense_cheatsheet.md](08_defense_cheatsheet.md) - elevator pitch + Q&A.
2. [10_demo_script.md](10_demo_script.md) - сценарий живого демо на 5-7 минут.
3. [12_limitations.md](12_limitations.md) - честные слабые места.
4. [09_glossary.md](09_glossary.md) - словарь терминов и mental models.

### Путь C - Глубокий обзор (40 минут, для ревью)

1. [11_experiments.md](11_experiments.md) - журнал 12 гипотез.
2. [12_limitations.md](12_limitations.md) - риски и next steps.
3. [06_tuning_and_eval.md](06_tuning_and_eval.md) - методика измерения.
4. [05_qdrant_and_retrieval.md](05_qdrant_and_retrieval.md) - почему DBSF + blended rerank.

## 0.8 Где искать, когда хочется X

| Хочу | Смотреть |
| --- | --- |
| Понять chunking (normalize/filter/split/format) | [index/chunking.py](../index/chunking.py) + [docs/02](02_index_deep_dive.md) |
| Понять весь search pipeline | [search/main.py](../search/main.py) + [docs/03](03_search_deep_dive.md) |
| Контракты API | [api.md](../api.md) |
| Hybrid retrieval math (DBSF, prefetch) | [docs/05](05_qdrant_and_retrieval.md) |
| Env-параметры и тюнинг | [docs/01](01_architecture.md) §env + [docs/06](06_tuning_and_eval.md) |
| Как гонять локальный eval | [eval/run.py](../eval/run.py), [eval/ingest.py](../eval/ingest.py) |
| Почему параметр N = X | [docs/11_experiments.md](11_experiments.md) |
| Что ещё не сделано | [docs/12_limitations.md](12_limitations.md) |
| Диагностика чанков | `python3 scripts/chunking_diagnostic.py` |
| Sweep по параметрам | `python3 scripts/sweep_chunking.py --phase smoke` |
| Нагрузочный тест | `python3 scripts/loadtest.py` |
| Chaos test (убить upstream) | `python3 scripts/chaostest.py` |
| Observability (Prometheus / OTEL) | [docs/07](07_ops_and_deploy.md), `/metrics` на 8002 |
| Пуш образов в хакатонный registry | [README.md](../README.md) §"Первая отправка" |

## 0.9 Data flow - offline и online

### Индексация (offline, один раз на чат)

```
chat JSON
  → POST /index  (chunking, 3 варианта текста)
  → POST /embeddings external (dense vectors, 1024-dim)
  → POST /sparse_embedding (BM25 через fastembed)
  → Qdrant upsert
```

### Поиск (online, на каждый вопрос)

```
enriched question (text / search_text / keywords / entities / hyde)
  → build_dense_queries + build_sparse_queries
  → embed_dense_many_safe + embed_sparse_many_safe  (параллельно)
  → Qdrant Query API (DBSF fusion, prefetch K=70/150)
  → local rescoring (phrase/entity/temporal boost)
  → rerank top-20 (nemotron cross-encoder, blend с α=0.2)
  → assemble_message_ids → top-50
```

Полный трассированный пример с реальными значениями - [docs/04_data_flow.md](04_data_flow.md).

## 0.10 Graceful degradation

Любой внешний upstream (dense API, reranker, Qdrant) может упасть -
сервис не падает, а деградирует:

- dense 429/5xx → sparse-only retrieval.
- reranker 429/5xx → возвращаем retrieval order.
- Qdrant недоступен → 503 (единственный случай, когда сервис не отдаёт 200).

Проверяется автоматически через `scripts/chaostest.py`.

## 0.11 Что менять можно, что нельзя

Менять можно:
- chunking и нормализацию;
- формирование `dense_content` / `sparse_content`;
- sparse-модель внутри index (хоть BM42);
- retrieval, fusion, rerank;
- любые эвристики и фильтры.

Менять нельзя:
- `GET /health`
- `POST /index`
- `POST /sparse_embedding`
- `POST /search`

Сломаете request/response - решение перестанет проходить проверку.

## 0.12 Ограничения среды (ТЗ хакатона)

- 4 CPU / 7 GB RAM на каждый контейнер.
- Нет интернета - все модели должны быть забейкены в образ.
- 15 минут SLA на полную индексацию чата.
- `linux/amd64` target architecture.
- Registry: `83.166.249.64:5000/<TEAM_ID>/{index,search}-service:latest`.

## 0.13 Деплой: от локального image до хакатонного registry

Деплой = собрать два `linux/amd64` образа и запушить их в приватный
registry по адресу `83.166.249.64:5000`. Внутренний ingestion/поиск там
уже запустят без нашего участия. Полный разбор и trace - в
[docs/07_ops_and_deploy.md](07_ops_and_deploy.md) (§7.5-§7.11).

### Один раз - настроить Docker

Registry работает без TLS, его надо добавить в `insecure-registries`.

Docker Desktop → Settings → Docker Engine → JSON:

```json
{"insecure-registries": ["83.166.249.64:5000"]}
```

Apply & Restart.

### Каждый релиз - три команды

```bash
export TEAM_ID=31023
export LOGIN='...'      # хакатонные креды
export PASSWORD='...'

docker login 83.166.249.64:5000 -u "$LOGIN" -p "$PASSWORD"

(cd index  && make push)   # build --platform linux/amd64 + push
(cd search && make push)
```

`Makefile` уже содержит `--platform linux/amd64`, даже если вы собираете
с Apple Silicon. Без этого проверочная среда (x86) не поднимет образ.

### Что должно оказаться в registry

```text
83.166.249.64:5000/<TEAM_ID>/index-service:latest
83.166.249.64:5000/<TEAM_ID>/search-service:latest
```

Проверить:

```bash
curl -su "$LOGIN:$PASSWORD" \
  http://83.166.249.64:5000/v2/$TEAM_ID/search-service/tags/list
# → {"name":"31023/search-service","tags":["latest"]}
```

### Чек-лист перед пушем

- [ ] `docker compose up --build` поднимается без ошибок.
- [ ] `GET /health` отвечает 200 на обоих сервисах.
- [ ] `python3 eval/run.py --dataset data/dataset_v2.jsonl` даёт `score ≥ 0.96`.
- [ ] `python3 scripts/chaostest.py` - все PASS (graceful degradation работает).
- [ ] `python3 scripts/loadtest.py --concurrency 8 --duration 30` - `p95 ≤ 1500 ms`, `errors=0`.
- [ ] `docker inspect .../search-service:latest | grep Architecture` = `amd64`.
- [ ] `OPEN_API_LOGIN/PASSWORD` не захардкожены в Dockerfile или в коде.
- [ ] Размер образа ≤ 2 GB (`docker images 83.166.249.64:5000/$TEAM_ID/...`).

### Типичные факапы

| Симптом | Почему | Что делать |
| --- | --- | --- |
| `docker push` → `authorization required` | Registry не в `insecure-registries` | Добавить в Docker Engine JSON, restart Docker Desktop |
| Проверочная среда не поднимает образ | Собрано под arm64, а не amd64 | Пересобрать через `make push` (там уже `--platform linux/amd64`) |
| `search` стартует, но сразу `status: degraded` | `OPEN_API_LOGIN/PASSWORD` не подхватились → upstream 4xx → fallback | Проверить `docker compose exec search env \| grep OPEN_API` |
| Индексация занимает > 15 мин | Sparse-модель не забейкена → качается из интернета на старте | Проверить `Dockerfile` содержит `RUN python -c "from fastembed..."` |
| `qdrant-init` падает в compose | Qdrant ещё не поднялся | `docker compose up -d qdrant-init` вручную |

### Rollback

У registry мы держим только `:latest`. Если релиз сломался - собрать
предыдущий commit и запушить заново:

```bash
git checkout <good-commit>
(cd index && make push) && (cd search && make push)
```

Образы в registry перезапишутся новым `:latest` - никакого staging нет.

## 0.14 Если застряли

- Вопросы по коду → начните с [docs/01](01_architecture.md), потом [docs/02](02_index_deep_dive.md) / [docs/03](03_search_deep_dive.md).
- Вопросы по метрикам → [docs/06](06_tuning_and_eval.md).
- Вопросы «почему именно так» → [docs/11_experiments.md](11_experiments.md).
- Вопросы по деплою → [docs/07](07_ops_and_deploy.md) + [README.md](../README.md) §"Первая отправка".
- Термин непонятен → [docs/09_glossary.md](09_glossary.md).

## 0.15 Пример прогона end-to-end на локальном чате

Чтобы не гадать, что именно увидит человек в терминале - вот реальные
числа от `docker compose up` до `eval/run.py` на нашем тестовом чате.

### Что на входе

```
data/dataset_ts_chat.json     1023 сообщения, 41 участник, ~60 тредов
```

### `docker compose up --build` (первый запуск)

```
[+] Running 3/3
 ✔ Container hackathon-search-engine-1-qdrant-1   Started     1.8s
 ✔ Container hackathon-search-engine-1-index-1    Started     3.2s
 ✔ Container hackathon-search-engine-1-search-1   Started     4.1s

qdrant  | INFO  storage::content_manager::toc  Loaded collections: []
index   | INFO  uvicorn.error                  Application startup complete
search  | INFO  uvicorn.error                  Application startup complete
```

Три контейнера подняты, коллекций в Qdrant пока нет.

### `python3 eval/ingest.py`

```
[ingest] POST /index              → 127 chunks      (took 0.9s)
[ingest] dense /embeddings × 127  → 1024-dim vectors (took 14.2s, 6 batches)
[ingest] POST /sparse_embedding   → 127 sparse vecs  (took 0.4s)
[ingest] qdrant upsert            → 127 points      (took 0.3s)
[ingest] done: chat=local-chat, chunks=127, total=15.8s
```

1023 сообщений превратились в 127 чанков (каждый чанк покрывает 8-12
сообщений). Уложились в 16 секунд - хорошо внутри 15-минутного SLA
(SLA считается для чата с `N ≤ 2000`).

### `curl POST /search` с вопросом про SIGABRT

Запрос:

```json
{"question": {"text": "где писали про SIGABRT на M1?",
              "search_text": "SIGABRT M1 MacBook",
              "keywords": ["SIGABRT", "M1"],
              "entities": {"names": ["MacBook Air", "M1"]}}}
```

Ответ:

```json
{
  "message_ids": [
    "4555555555555555555",
    "4555555555555555557",
    "4566666666666666662",
    "4566666666666666663",
    ...  // всего 50 ids
  ],
  "status": "ok"
}
```

Структурированный лог в stdout контейнера `search`:

```json
{"request_id":"req-7f3a","total_ms":641,
 "stages_ms":{"embed":207,"qdrant":61,"rescore":38,"rerank":298,"assemble":6},
 "counts":{"dense_queries":4,"sparse_queries":6,"prefetch":150,"rerank":20,"returned":50},
 "fallbacks":{"dense":false,"rerank":false},
 "status":"ok"}
```

Половина времени - это rerank (298 мс из 641). Dense embedding уходит
в cache при повторном вопросе и падает до ~20 мс.

### `python3 eval/run.py --dataset data/dataset_v2.jsonl --k 50`

```text
questions=80   dead=2   effective=78
Recall@50  = 0.9750
nDCG@50    = 0.9470
score      = 0.9694   (= 0.8 × 0.9750 + 0.2 × 0.9470)
theoretical max = 0.9750 (2 dead queries не закрываются в принципе)
p50_latency = 512 ms
p95_latency = 893 ms
```

Это наш текущий snapshot (2026-04-19). Если после ваших изменений
`score` не упал ниже 0.9600 и `p95 < 1500 ms` - в целом всё в порядке.

### Что откуда

| Цифра | Откуда | Значение |
| --- | --- | --- |
| 127 chunks | `/index` response | Сколько чанков построил chunker из 1023 msgs |
| 1024 dim | external API `/embeddings` | Размерность dense вектора (Qwen3-0.6B) |
| 150 prefetch | `RETRIEVE_K` env | Сколько кандидатов получает Qdrant fusion |
| 20 rerank | `RERANK_LIMIT` env | Сколько топ-кандидатов уходит в cross-encoder |
| α = 0.2 | `RERANK_ALPHA` env | Вес блендинга rerank_score с retrieval order |
| 50 returned | hard cap в `/search` | Итоговый размер `message_ids` |
| 0.9694 | `eval/metrics.py` | `0.8 × Recall@50 + 0.2 × nDCG@50` |
