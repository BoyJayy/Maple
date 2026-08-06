# Changelog

## 2026-07-15 · Финальная сбалансированная конфигурация поиска

Финальная версия собрана на основе `fable` и проверена тем же внешним runner,
которым сравнивались ветки `main` и `fable`.

- Point-level rescoring выключен по умолчанию
  (`POINT_RESCORE_ENABLED=0`). На обоих eval-наборах порядок без него немного
  лучше, а поиск не тратит CPU на дополнительный проход. Message-level scoring
  при финальной сборке сохранён.
- Временной поиск использует `TIME_FILTER_MODE=hard`: этот режим оказался
  точнее и быстрее дополнительного soft-fusion на датированных диалогах.
- Добавлен ленивый single-flight TTL-кэш глобальных временных границ коллекции.
  Если диапазон вопроса не пересекается с корпусом, гарантированно пустой
  filtered retrieval пропускается. Ошибка чтения границ не ломает поиск.
- Для hard-режима сохранён zero-result retry без фильтра. Экспериментальный
  `soft`-режим также доступен, но не является дефолтом.
- Пустой после нормализации `search_text` теперь откатывается к `text`; валидный
  `date_range` имеет приоритет над более свободными `date_mentions`.
- Опциональный reranker работает fail-open и остаётся выключенным по умолчанию
  из-за дополнительной задержки и памяти.
- `eval/baseline.json` и `eval/baseline_dialogues.json` обновлены под финальный
  профиль, чтобы следующие regression-прогоны сравнивались с новым дефолтом.

Контрольные результаты:

| Набор / метрика | `main` | `fable` | final |
|---|---:|---:|---:|
| Dataset_main Hit@1 | 0.0438 | 0.5375 | **0.5563** |
| Dataset_main nDCG@50 | 0.5356 | 0.7735 | **0.7799** |
| Dataset_main MRR@50 | 0.3867 | 0.7009 | **0.7098** |
| Dataset_dialogues Recall@50 | 0.6275 | 0.8283 | **0.8396** |
| Dataset_dialogues nDCG@50 | 0.3455 | 0.4954 | **0.5016** |
| Dataset_dialogues MRR@50 | 0.2864 | 0.4384 | **0.4446** |

На `Dataset_main` warm latency final-профиля: p50/p95 `27.0/68.2 ms` при
concurrency 1 и `91.2/137.0 ms` при concurrency 8; throughput при concurrency
8 — `83.9 QPS` против `53.6 QPS` у дефолтного `fable`. HTTP-ошибок и retry у
benchmark runner не было.

## 2026-07-08 · Честный eval-бенчмарк на реальных диалогах + эксперименты

Пилотный датасет из HF-диалогов доведён до полноценного бенчмарка, на нём
прогнана матрица конфигураций поиска. Всё локально, без платных API.

### Датасет (`scripts/convert_hf_dialogues.py` v2)

- Корпус: **300 реальных русских диалогов** (HF `Den4ikAI/russian_dialogues_2`,
  MIT) → 1995 сообщений → ~298 чанков; top-50 выдачи ≈ 2.5% корпуса, recall
  стал различающей метрикой (на старой синтетике был тривиальный 1.0).
- Тематические бакеты-дистракторы (авто/кино/деньги/еда/игры/животные...) —
  системе нужно найти *тот самый* разговор, а не единственный по теме.
- Реализм структуры: 2-3 отправителя на диалог, @упоминания (247 сообщений),
  126 interleaved-сессий с тредами, редкие синтетические стектрейсы —
  задействованы chunking по тредам, mentions-бусты и сжатие трейсов.
- Загрузка через бесплатный HF datasets-server API с ретраями на 429.
- **144 вопроса** (`data/Dataset_dialogues_questions.jsonl`), сгенерированы
  субагентами и проверены офлайн-валидатором
  (`scripts/validate_questions.py`): 0 ошибок. Категории: semantic 36,
  exact 30, date 30, participant 24, multihop 12, negative 12.

### Eval-инструменты

- `eval/run.py`: `--ks 10,50` (несколько cutoff'ов), разбивка финальной
  стадии по категориям вопросов, 95% bootstrap-CI, негативные вопросы
  (пустой ground truth) исключаются из ранжирующих метрик.
- `eval/metrics.py`: `bootstrap_ci()`.
- `scripts/validate_questions.py`: офлайн-проверка вопросов против корпуса
  (существование id, покрытие дат, дословные утечки, дубликаты).

### Результаты экспериментов (N=132, embedded Qdrant, те же кодовые пути)

| конфигурация | R@10 | nDCG@10 | R@50 | nDCG@50 |
|---|---|---|---|---|
| dbsf + rescore (дефолт на момент эксперимента) | 0.535 | 0.423 | 0.828 | 0.495 |
| rrf + rescore | 0.547 | 0.424 | 0.821 | 0.492 |
| dbsf без rescore | 0.537 | 0.427 | 0.840 | 0.502 |
| **dbsf + reranker (jina-v2-multilingual)** | **0.576** | **0.442** | **0.891** | **0.522** |
| dbsf + rescore, e5-large | 0.547 | 0.430 | 0.874 | 0.510 |

Выводы:
- **Reranker — главный выигрыш**: +6.3 п.п. R@50, лучший почти во всех
  категориях (semantic 0.25 против 0.19). Включается `RERANK_ENABLED=1`.
- **Rescoring стабильно чуть в минус** (R@50 0.828 против 0.840 без него,
  дельты в пределах CI, но направление совпадает со старым датасетом) —
  кандидат на отключение или пересбор весов через `RESCORE_*`.
- **e5-large** даёт +4.6 п.п. R@50, но парафразы не лечит (semantic 0.21
  против 0.19) — слабость семантики архитектурная (эмбеддинг целого чанка
  разбавлен соседними сообщениями), а не только модельная.
- **Самая большая дыра — semantic-вопросы** (R@10 ≈ 0.2 у всех конфигураций):
  следующий рычаг качества — не fusion и не веса.
- dbsf против rrf — практически ничья.

Baseline зафиксирован в `eval/baseline_dialogues.json` (multi-k формат,
с разбивкой по категориям).

---

## 2026-07-07 — 2026-07-08 · Большой проход по качеству поиска, надёжности и инфраструктуре

Два этапа работ: (1) улучшения ретривала и инженерной обвязки, (2) исправления
по итогам многоагентной ревизии кода (131 агент-ревьюер, 30 подтверждённых
находок, каждая проверена тремя независимыми верификаторами).

Итог на `data/Dataset_main_questions.jsonl` (160 вопросов, k=50):
**Recall@50 = 1.0000, nDCG@50 = 0.7735, MRR@50 = 0.7009, score = 0.9547** —
зафиксировано в `eval/baseline.json`.

---

### 1. Качество поиска (search)

#### Морфологическое сопоставление exact-терминов — `search/querying.py`
- Термины и текст стеммируются Snowball-стеммером (русский для кириллицы,
  английский для латиницы) и сравниваются по границам слов:
  `релиз` находит `релизе`, но `код` больше не совпадает с `кодекс`.
- Токены с цифрами или пунктуацией идентификаторов (email, ссылки, версии)
  сравниваются дословно.
- Пунктуация в конце/начале токена срезается (`«1.18.»` → `1.18`), составные
  идентификаторы дополнительно матчатся по словным частям
  (`plan` находит `release-plan.docx`).
- Результат `text_stems()` кэшируется (`lru_cache`), чтобы не стеммировать
  один и тот же `page_content` в каждой стадии заново.

#### Приоритет и фильтрация exact-терминов — `search/querying.py`
- Бюджет в 12 терминов раньше съедали слова-паразиты из текста вопроса
  («подскажи», «пожалуйста», «что», «итоге»…), а сущности и ключевые слова
  вообще не попадали в термины. Теперь порядок источников:
  **entities → keywords → date_mentions → текст вопроса → asker**,
  плюс стоп-лист русских/английских филлеров (применяется только к обычным
  словам, идентификаторы не трогаются).

#### Ранговый rescoring — `search/pipeline.py`, `search/config.py`
- Rescoring больше не смешивает бонусы с сырым fusion-скором (шкалы DBSF и RRF
  разные) — работает от **позиции** после fusion. Поведение стало одинаковым
  для обоих режимов fusion.
- Все веса вынесены в env: `RESCORE_RANK_BONUS_MAX`, `RESCORE_RANK_BONUS_STEP`,
  `RESCORE_MESSAGE_HIT_WEIGHT`, `RESCORE_CONTEXT_HIT_WEIGHT`,
  `RESCORE_METADATA_HIT_WEIGHT`, `ASSEMBLE_BLOCK_HIT_WEIGHT`,
  `ASSEMBLE_BLOCK_INDEX_PENALTY`.
- Rescoring и сборка `message_ids` (CPU-bound) ушли с event loop в
  `asyncio.to_thread`.

#### Фильтр по времени — `search/querying.py`, `search/pipeline.py`
- Из `question.date_range` и ISO-подобных дат в `date_mentions`
  (`YYYY`, `YYYY-MM`, `YYYY-MM-DD`) строится временнóе окно, расширяется на
  `TIME_FILTER_MARGIN_SECONDS` (по умолчанию сутки) и применяется к Qdrant
  prefetch как range-условие по `metadata.start` / `metadata.end`.
- Date-only верхняя граница (`to: "2023-05-01"`) трактуется как конец дня.
- **Fallback**: если фильтрованный поиск вернул 0 точек (дата в вопросе
  относится к содержимому, а не времени отправки; коллекция со старыми
  строковыми `start/end`), пайплайн логирует предупреждение и повторяет запрос
  без фильтра. Без этого фильтр обнулял recall на всём eval-датасете
  (вопросы несут `date_range` в 2026, корпус — 2024).
- Управляется `TIME_FILTER_ENABLED` (по умолчанию включён).

#### Опциональный reranker — `search/pipeline.py`
- Локальный cross-encoder (`jinaai/jina-reranker-v2-base-multilingual` через
  fastembed) поверх топ-`RERANK_TOP_K` кандидатов после rescoring.
- Выключен по умолчанию (`RERANK_ENABLED=0`): +латентность и ~1.1 GB модель.
- Стадия `reranked` видна в `/_debug/search` и в `eval/run.py --stages`;
  параметр `no_rerank=true` работает.

#### Поддержка смены dense-модели — `search/config.py`, `eval/ingest.py`
- E5-префиксы `query:` / `passage:` подставляются автоматически при
  E5-модели (`DENSE_QUERY_PREFIX` — в search, `DENSE_DOCUMENT_PREFIX` — в
  ingest); переопределяются через env.
- Переключение на `intfloat/multilingual-e5-large` (сильнейшая multilingual
  dense-модель в fastembed 0.8.0) — одна env-переменная + переиндексация
  (см. docs/local_development.md, раздел Tuning).

#### BM25 по-русски — `index/sparse.py`, `search/pipeline.py`
- `Qdrant/bm25` создаётся с `language=russian` (Snowball-стемминг и стоп-слова
  на стороне BM25; латинские токены проходят без изменений).
- Значение `SPARSE_MODEL_LANGUAGE` должно совпадать в index и search;
  смена требует переиндексации.

---

### 2. Надёжность и API

#### message_blocks — `index/schemas.py`, `index/chunking.py`, `search/pipeline.py`
- Index-сервис теперь отдаёт в каждом чанке `message_blocks` — упорядоченные
  пары `{message_id, text}` (по одной на отрендеренное сообщение; фрагменты
  длинного сообщения разделяют один id).
- `eval/ingest.py` кладёт их в `metadata`, search скорит отдельные сообщения
  по ним вместо хрупкого regex-парсинга `page_content` (старый парсинг оставлен
  как fallback для старых точек). Заодно закрыт баг с фрагментированными
  сообщениями, ломавшими инвариант «блоков столько же, сколько id».

#### Сервисные endpoints — `index/main.py`, `search/main.py`
- `/index` стал синхронным обработчиком (FastAPI уводит его в threadpool) —
  CPU-bound chunking больше не блокирует event loop.
- Модели прогреваются в lifespan обоих сервисов — первый запрос не платит
  за загрузку.
- Новые `GET /ready`: у index — прогрев завершён; у search — Qdrant доступен,
  коллекция существует **и размерность dense-вектора совпадает с
  `DENSE_VECTOR_SIZE`** (ловит смену модели без переиндексации).
- `/_debug/search`: параметр `fusion` больше не имеет жёсткого дефолта
  `dbsf` — по умолчанию используется `FUSION_MODE` сервиса.
- `/sparse_embedding` получил `response_model=SparseEmbeddingResponse`.

#### Данные в Qdrant — `eval/ingest.py`, `scripts/qdrant_init.sh`
- `metadata.start` / `metadata.end` теперь **целые числа** (раньше строки) —
  это требование range-фильтра.
- Payload-индексы: `metadata.chat_id` (keyword), `metadata.start`,
  `metadata.end` (integer) — создаются и в `ensure_collection`, и в
  init-скрипте compose.
- Детерминированные id точек (`uuid5`) — идемпотентный ingest (без изменений,
  но теперь покрыт тестами).

> ⚠️ Всё вышеперечисленное требует **переиндексации**:
> `RESET_COLLECTION=1 python3 eval/ingest.py`.

---

### 3. Eval и скрипты

#### `eval/metrics.py`, `eval/run.py`
- Новая метрика **MRR@K** (позиция первого попадания) во всех отчётах.
- Baseline: `--save-baseline file.json` фиксирует результаты,
  `--baseline file.json` (или автоматически `eval/baseline.json`, если
  существует) печатает дельты по каждой стадии — регрессионный контроль.
- `NO_RESCORE=1` / `NO_RERANK=1` теперь работают и **без** `--stages`
  (раньше молча игнорировались — `/search` не принимает query-параметры;
  теперь toggles автоматически направляют запросы в `/_debug/search`).
- Ground-truth `message_ids` приводятся к строкам, как и все предсказания.
- `SYNTHETIC_VIA_INDEX=1` — прогон синтетического JSONL-корпуса через реальный
  `/index` (chunking) вместо ручной сборки один-ответ-один-чанк.

#### `scripts/sweep_chunking.py`
- Починен парсинг вывода `eval/run.py` (сломался после добавления колонки
  MRR); в CSV добавлена колонка `mrr@50`.
- Ingest теперь идёт в ту же коллекцию, что читает search
  (`QDRANT_COLLECTION_NAME`, по умолчанию `messages`; раньше — в `evaluation`,
  и sweep мерил не то, что заливал).
- Удалён мёртвый параметр `OVERLAP_CONTEXT_CHARS` (сервис его никогда не
  читал); дефолты выровнены со значениями сервиса (1600/650).

#### `scripts/ab_qdrant.py`
- Конфиги «3+3»/«5+5» были дубликатами «full» (лимит запросов — 3):
  заменены на 1+1 / 2+2 / full + вариант `no_rescore`.

---

### 4. Docker, compose, CI

#### `index/Dockerfile`, `search/Dockerfile`
- Имена моделей стали **build args** — `docker compose up --build` запекает в
  образ ту модель, что задана в env, а не всегда дефолтную.
- Удалён мёртвый `ENV CHUNK_SIZE=10`.

#### `docker-compose.yml`
- Volume `qdrant_storage` — точки переживают пересоздание контейнера
  (чистый старт: `docker compose down -v`).
- Volumes `index_models` / `search_models` на `/models` — рантайм-загрузки
  моделей (reranker, смена dense-модели без `--build`) скачиваются один раз.
- Init коллекции вынесен в `scripts/qdrant_init.sh` (создаёт коллекцию и
  payload-индексы; схема синхронизирована с `eval/ingest.py`).
- Проброшены **все** документированные тюнинг-переменные: `SPARSE_MODEL_NAME`
  (в т.ч. в index), `SPARSE_MODEL_LANGUAGE`, `FINAL_MESSAGE_LIMIT`,
  `RESCORE_*`, `ASSEMBLE_*`, `TIME_FILTER_*`, `RERANK_*`.

#### CI — `.github/workflows/`
- Новый `tests.yml`: pytest-джоба (3 сьюта) + джоба валидации
  `docker compose config`. Pip-кэш учитывает и `requirements-dev.txt`,
  и `eval/requirements.txt`.
- `telegram.yml`: отсутствие секретов (fork-PR) — теперь skip, а не падение;
  ошибки Telegram API перестали молча глотаться (`curl --fail-with-body`).

#### Прочее
- `.gitignore`: добавлен `.env`; `data/.DS_Store` убран из индекса git;
  зафиксирована политика — `eval/baseline.json` коммитится намеренно.
- `requirements-dev.txt` — общий dev-набор (eval-зависимости + pytest).
- `search/requirements.txt`: + `snowballstemmer==3.1.1`.

---

### 5. Тесты

Раньше тестов не было. Теперь **70 тестов** в `tests/` (три изолированных
сьюта — у `index/` и `search/` совпадают имена модулей, поэтому запуск
раздельными процессами через `scripts/run_tests.sh`):

- `tests/index_service/` — chunking: нормализация, фильтрация шума, сжатие
  трейсов, фрагментация, границы чанков (тред/время/размер), overlap,
  выравнивание `message_blocks`.
- `tests/search_service/` — querying: стемминг, стоп-слова, приоритет
  сущностей, пунктуация, парсинг дат; pipeline: rescoring (независимость от
  fusion-скора), сборка по `message_blocks` и fallback-путям; поток
  `run_search_pipeline` на фейковом Qdrant: **fallback без фильтра**, стадии,
  reranker.
- `tests/eval_service/` — метрики (recall/nDCG/MRR), хелперы run
  (датасеты, baseline), хелперы ingest (стабильность id, синтетика).

Запуск: `sh scripts/run_tests.sh`.

---

### 6. Удалённый мёртвый код

- `normalize_message(..., is_overlap=)` — неиспользуемый параметр.
- `ENV CHUNK_SIZE` в index/Dockerfile.
- `DENSE_DOCUMENT_PREFIX` из search-конфига (документы эмбеддит только ingest).
- Неверная аннотация возврата `embed_sparse_texts`.
- Жёсткий дефолт `fusion="dbsf"` в debug-endpoint.

---

### Миграция

1. `docker compose up --build` — пересборка образов (build args, volumes).
2. `RESET_COLLECTION=1 python3 eval/ingest.py` — переиндексация обязательна:
   русский BM25, целочисленные `start/end`, `message_blocks`.
3. `python3 eval/run.py --dataset data/Dataset_main_questions.jsonl --k 50
   --stages --save-baseline eval/baseline.json` — пересохранить baseline на
   docker-стеке (текущий снят через embedded-Qdrant с теми же кодовыми путями).

### Известные ограничения / отклонённые идеи

- `question.hyde` и `asked_on` по-прежнему не используются (ревизия сочла
  влияние спорным: hyde-тексты конкурируют за слоты dense-запросов).
- Rescoring на синтетическом датасете даёт лёгкий минус nDCG
  (0.7776 → 0.7735) при recall = 1.0 — веса теперь в env, подбираются через
  `NO_RESCORE=1` A/B и sweep.
- Dense-модель по умолчанию оставлена MiniLM: в fastembed 0.8.0 из более
  сильных multilingual только e5-large / jina-v3 (~2.2 GB) — неприемлемый
  дефолт для local-first; переключение задокументировано.
