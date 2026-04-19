# 09. Glossary + Mental Models

> Словарик-справочник всех терминов, которые будут звучать в защите.
> Если слышите что-то непонятное - ищите здесь. Если кто-то спросит
> определение - говорите как тут написано.

---

## A

### API contract
Строго зафиксированная форма request/response. В нашем проекте нельзя менять
контракты `POST /index`, `POST /sparse_embedding`, `POST /search` - иначе
решение не проходит проверку.

### asker
Поле `question.asker` - e-mail того, кто задал вопрос. Используется для
mild boost кандидатов, где `asker ∈ participants/mentions`.

### assemble / `assemble_message_ids`
Финальная стадия search: переход с уровня chunk'ов на уровень отдельных
`message_ids`. Блоки внутри chunk скорятся отдельно через
`score_message_block`, потом мержатся с `point_bonus`, сортируются,
дедуплицируются, cut до `FINAL_MESSAGE_LIMIT=50`.

---

## B

### BM25
Классический sparse retrieval algorithm. Scores term based on frequency
(TF), inverse document frequency (IDF) and document length. Силён на
exact token match (имена, версии, email, URL). У нас реализован через
`fastembed.SparseTextEmbedding("Qdrant/bm25")` + IDF modifier в Qdrant.

### blended score
Линейная комбинация rerank score и retrieval-order score:
`blend = α × rerank + (1-α) × retrieval_rank`. У нас `α = RERANK_ALPHA = 0.2`.

---

## C

### CONTEXT (section)
Верхняя секция `page_content`, содержит overlap-сообщения из предыдущих
чанков. Нужна для recall (даёт retrieval дополнительный якорь), но при
ранжировании мы её вес понижаем (штраф `context_penalty = 0.08` если match
только в CONTEXT).

### coord-descent (sweep)
Метод оптимизации: меняем один параметр, фиксируем остальные; находим
оптимум по оси; переходим к следующему параметру. Применён в
`scripts/sweep_chunking.py --phase axis`.

### chunk
Группа сообщений, которую мы индексируем как единое целое. У нас message-based
(не character split), с `thread_sn`/time-gap/size ограничениями.

### chunking
Процесс нарезки чата на chunks. Сердце index-сервиса
(см. [02_index_deep_dive.md](02_index_deep_dive.md)).

### cross-encoder
Модель, которая принимает пару `(query, document)` и возвращает relevance
score, читая их одновременно. В отличие от embedding'а (bi-encoder), даёт
выше качество, но требует больше compute. У нас используется как reranker
(nemotron).

---

## D

### DBSF (Distribution-Based Score Fusion)
Qdrant'овский алгоритм объединения prefetch-веток: нормализует score каждой
ветки через z-score (mean, std) и объединяет. Учитывает величину score,
не только rank. Наш default.

### dense retrieval
Поиск по cosine-similarity на embedding-векторах. Хорошо ловит семантику
и перефразировки. У нас через Qwen3-Embedding-0.6B (1024-dim).

### `dense_content`
Текст чанка, оптимизированный для embedding-модели: gladkий, без служебки,
без timestamps/flags.

### `date_range`
Поле `question.date_range = {from, to}`. Парсится в unix timestamp;
chunk получает soft boost `+0.06`, если его `[start, end]` пересекается с
query range. Hard filter - не делаем.

---

## E

### enriched question
Обогащённый входной JSON: помимо `text` - `search_text`, `variants`, `hyde`,
`keywords`, `entities`, `date_range`, `asker`. Готовится upstream'ом.

### entity
Поле `question.entities = {people, emails, documents, names, links}`.
Используется в phrase_terms и identity_terms для local rescoring.

### embedding
Векторное представление текста. Dense - плотный вектор фиксированной длины,
sparse - разреженный вектор с token-index → weight.

---

## F

### fallback
Режим graceful degradation. При сбое upstream'а мы не падаем 500, а переходим
на упрощённый путь (sparse-only, retrieval-order, empty result).

### `fastembed`
Python библиотека от Qdrant для локального embedding'a. Мы используем только
для BM25 (sparse), dense - через внешний API.

### final result
То, что идёт в response `SearchAPIResponse`. Одна запись с `message_ids` из
не более 50 элементов.

### Fusion (Query API)
Qdrant-фича: один `query_points` с несколькими `Prefetch` ветками и
`FusionQuery(fusion=...)`. Объединяет кандидатов по алгоритму (RRF/DBSF).

---

## H

### `HYDE` (Hypothetical Document Embeddings)
Query-expansion техника: вместо запроса используется гипотетический ответ
(или несколько). У нас приходит в `question.hyde[]`. Длинные hyde (≥40 chars)
идут отдельным Qdrant pass (`HYDE_PASS_LIMIT=60`) поверх fusion.

---

## I

### IDF (Inverse Document Frequency)
Weight for rare tokens в BM25. У нас включено через
`SparseVectorParams(modifier=Modifier.IDF)` на стороне Qdrant.

### identity_terms
frozenset нормализованных asker + entity people/documents/names.
Используется в `score_metadata_signals` для boost.

### intent
Классификация вопроса: `summary` (куда смотреть? в каком документе?),
`detail` (какой / как / сколько), `neutral` (остальное). Используется в
опциональном intent-aware boost (`INTENT_ALIGNMENT_WEIGHT`).

---

## L

### local rescoring
Дешёвый post-processing кандидатов между fusion и rerank. Чистый Python,
~40ms на 150 кандидатов. Компоненты: phrase/token hits, best message-block,
metadata signals, temporal overlap, context penalty, intent alignment.

---

## M

### MESSAGES (section)
Нижняя секция `page_content` с сообщениями текущего чанка. Бустим match
в MESSAGES и штрафуем match только в CONTEXT.

### metadata (payload)
Qdrant payload field: `chat_id, chat_name, thread_sn, message_ids, start,
end, participants, mentions, contains_forward/quote`. Используется для
scoring и (потенциально) фильтрации.

### `message_ids`
Первичная единица оценки - metric считается по ним. На выходе `/search`
возвращаем список `message_ids` top-50.

---

## N

### nDCG@K (normalized Discounted Cumulative Gain)
Метрика ranking'а, учитывает порядок. `dcg = Σ 1/log2(i+2)` для попаданий;
`ndcg = dcg / idcg`. Ответ на позиции 1 → 1.0, на позиции 10 → ~0.29.

### normalize (text)
`normalize_text(text)` = strip каждой строки + drop пустых строк. Базовый
шаг в chunker'е.

---

## O

### overlap
`overlap_messages` - сообщения из прошлых батчей, которые даются чанкеру для
контекста. У нас thread-aware: тянем только того же `thread_sn`, gap ≤ 3h.
На новых top-level темах сбрасываем полностью.

### OpenTelemetry (OTEL)
Observability стандарт. Включается `OTEL_ENABLED=1`. Авто-инструментация
FastAPI + httpx, плюс ручные span'ы per stage. Экспорт OTLP или console.

---

## P

### `page_content`
Человекочитаемый текст чанка с `CHAT:`, `CONTEXT:`, `MESSAGES:` секциями,
timestamps, flags, Mentions. Идёт в Qdrant payload и в rerank input.

### participants
Уникальные `sender_id`'s сообщений в чанке. Лежит в `metadata.participants`.
Используется в metadata signals.

### Prefetch (Qdrant)
Одна ветка hybrid-retrieval: `Prefetch(query=v, using="dense|sparse", limit=K)`.
Несколько prefetch + FusionQuery = один финальный ranking.

### PipelineTrace
Наш dataclass для per-request observability. Содержит `stages_ms`, `counts`,
`fallbacks`, `errors`. Эмитится в лог как JSON + в Prometheus histograms.

### phrase_terms / token_terms
Два списка signal'ов из вопроса. Phrase - точные имена/даты/ключевые фразы
(boost `+0.04…0.07`). Token - signal-символы (цифры, @./:) (boost `+0.01`,
cap 0.08). Используются в `score_text_signals`.

### prefers_earliest
Boolean из `QueryContext`: `True` если вопрос содержит «первое/исходный/начало/
first/earliest». Даёт маленький boost раннему сообщению внутри chunk'а.

---

## Q

### Qdrant
Vector database. Поддерживает dense + sparse vectors в одной точке,
hybrid retrieval через Query API, payload filters, detrministic point IDs.
У нас `qdrant/qdrant:v1.14.1`.

### Qwen3-Embedding-0.6B
Dense модель, 1024-dim output. Доступна через внешний hackathon API.

---

## R

### Recall@K
Доля правильных `message_ids` среди первых K. `0.8 × Recall@50` - главная
часть метрики.

### rerank
Вторая стадия ranking'а. Cross-encoder читает `(query, candidate)` пары и
ставит score. У нас nemotron-rerank-1b-v2, top-20 кандидатов.

### RRF (Reciprocal Rank Fusion)
Альтернатива DBSF. `score_rrf(d) = Σ 1/(k + rank_i(d))` с `k=60`. У нас
переключаемо, но default - DBSF.

### rescore (local)
См. local rescoring.

---

## S

### `search_text`
Чищенный primary query (поле `question.search_text`). Предпочитаем его
сырому `text` при построении queries.

### sparse retrieval
Поиск по точному token match (BM25-style). Хорошо ловит редкие токены.
Дополняет dense.

### `sparse_content`
Текст чанка для BM25 embedding'а: одной строкой на сообщение, с `sender:`,
mentions, флагами.

### synthetic eval
Режим `eval/ingest.py`, когда `DATA_PATH=*.jsonl` с `question`/`answer`
полями. Не вызывает `POST /index`, вместо этого строит synthetic corpus
из `answer.text` с `answer.message_ids` как ground truth.

### signal tokens
Токены, которые с высокой вероятностью являются важными: содержат цифру
или символы `@./:+-_`, или длина ≥4.

### SIGABRT
Пример технического токена. Если сообщение содержит `sigabrt`, `.ts:`, `.py:`,
`traceback`, `runtimeerror` - оно считается «техническим», сжимается в
`compress_text_for_index`, не дробится на фрагменты.

---

## T

### thread_sn
Идентификатор треда внутри чата. Используется как barrier в chunking'е
(не склеиваем сообщения разных тредов) и в overlap (берём только того же
треда).

### temporal signal
Soft boost `+0.06`, если `chunk.metadata.[start, end]` пересекается с
`query.date_range.[from, to]`.

---

## U

### upstream
Внешние сервисы, от которых мы зависим: dense API (`/embeddings`),
reranker API (`/score`), Qdrant. Все три обёрнуты в `*_safe` функции
для graceful fallback.

---

## 🧠 Mental models

### «Index отвечает за чанки, Search - за ranking»

Если Recall@50 низкий → смотри index (chunking / filtering / coverage).
Если nDCG@50 низкий, а Recall ок → смотри search (rerank / rescore / assemble).

### «Dense видит смысл, sparse видит точность»

Dense не «знает», что такое `SDK 3.2` как точная строка - он видит вектор
«релиз версии продукта». Sparse видит именно токен `sdk` + `3.2` с весами.
Вместе они покрывают оба случая.

### «Fusion - клей между ветками, rerank - корректор первых позиций»

Fusion делает из 11 prefetch-веток один ranking. Rerank берёт первые 20 и
поднимает правильный ответ в первые 5-10 (это и есть nDCG lift). По
отдельности ни то, ни другое полного решения не даёт.

### «Все внешние API могут ложиться»

Первая реакция на `502 Bad Gateway` / `429 Too Many Requests` в production -
паника. У нас эта паника инкапсулирована: любой сбой → graceful mode →
200 + label `status=degraded`. Чтобы ни jury, ни проверочная среда не
получили 500.


---

## 📋 Мини-примеры по ключевым терминам

Для тех, кто читает «по диагонали» — самые частые в защите термины, каждый
с одним конкретным примером.

### `page_content` — что это буквально

```text
CHAT: Atlas Hub
CONTEXT:
[2024-04-18 10:26:40] alice: Кто-нибудь пробовал desktop client 3.4 на M1?

MESSAGES:
[2024-04-18 10:27:40] bob: У меня MacBook Air (M1, 2020) — падает SIGABRT.
[2024-04-18 10:28:40] alice: Stacktrace есть?
  Quoted message: падает SIGABRT
```

Идёт в payload Qdrant и в rerank input. Это тот текст, который видит
cross-encoder.

### `dense_content` — тот же chunk, но для embedding'а

```text
Кто-нибудь пробовал desktop client 3.4 на M1? У меня MacBook Air (M1, 2020) — падает
SIGABRT. Stacktrace есть?
```

Никаких `CHAT:`/`[timestamp]:` — только смысл. Embedding-модель на служебке
теряет precision.

### `sparse_content` — тот же chunk, но для BM25

```text
alice: Кто-нибудь пробовал desktop client 3.4 на M1?
bob: У меня MacBook Air M1 2020 падает SIGABRT
alice: Stacktrace есть? quoted: падает SIGABRT
```

Одной строкой на сообщение, с префиксом `sender:`. BM25 любит такие «бедные»
тексты — каждый токен считается отдельно.

### `phrase_terms` vs `token_terms` — на живом примере

Вопрос: `question.entities = {"names": ["MacBook Air", "M1"], "people": ["Бобр"]}`
+ `question.keywords = ["SIGABRT", "desktop client 3.4"]`.

- `phrase_terms = {"MacBook Air", "M1", "Бобр", "SIGABRT", "desktop client 3.4"}` — точные n-grams.
- `token_terms = {"macbook", "air", "m1", "бобр", "sigabrt", "desktop", "client", "3.4", "34"}` — токенизированные, нормализованные.

Phrase hit даёт `+0.07` (дорогой сигнал — попало целиком), token hit даёт
`+0.01` (дешёвый сигнал — частичное совпадение).

### `blended score` — где именно применяется

Top-20 после retrieval попадают в rerank. Для каждого:

```text
retrieval_rank_score = 1.0 - (rank_in_150 / 150)   # C_при_rank=1 → 0.9933
rerank_score          = cross_encoder(query, page_content)   # ∈ [0, 1]
blended               = 0.2 × rerank_score + 0.8 × retrieval_rank_score
```

Пример: chunk был на позиции 4 в retrieval (score=0.973), reranker дал 0.85.
Blended = `0.2 × 0.85 + 0.8 × 0.973 = 0.948`. Сдвигает на первое место только
если разница в rerank > ~0.1 относительно соседей по rank.

### `date_range` boost — на цифрах

Вопрос: `question.date_range = {"from": "2024-04-15", "to": "2024-04-20"}`
(unix: `1713139200 — 1713571200`).

Chunk с `metadata.start = 1713460000` (18 апреля, внутри range) → `+0.06`.
Chunk с `metadata.start = 1712000000` (неделю раньше) → `0` (не пересекается).

Это soft boost, не hard filter. Даже chunk без overlap по датам
попадает в retrieval — просто без бонуса.

### `context_penalty` — пример, как работает

Ожидаемый ответ — в сообщении #12. Оно попало в chunk_A (в MESSAGES) и
в chunk_B (как overlap в CONTEXT).

- chunk_A: phrase hit `SIGABRT` в MESSAGES → `+0.07`, без penalty.
- chunk_B: phrase hit `SIGABRT` только в CONTEXT → `+0.07 - 0.08 = -0.01`.

Результат: chunk_A выигрывает, mess #12 возвращается из «своего» chunk'а,
а не из overlap'а соседа.
