# 11. Журнал экспериментов

> Фиксация гипотез человеческим языком. Не все дали точные `Δscore` -
> там, где цифры честно не мерили, так и написано.

У каждой гипотезы четыре части:
- **Зачем** - что проверяли и почему это было нужно.
- **Что сделали** - конкретное изменение.
- **Что увидели** - наблюдение.
- **Решение** - оставили / откатили.

---

## Гипотеза 1. Chunking по сообщениям, а не по тексту

### Зачем

Метрика считается по `message_ids`. Document-style splitter (LangChain
`RecursiveCharacterTextSplitter`) рубит текст по char-boundary и теряет
связь с конкретными сообщениями.

### Что сделали

Message-based chunking: сохраняем порядок сообщений, thread-aware overlap,
time-gap barrier ≤3h (см. [index/chunking.py](../index/chunking.py)).

### Что увидели

- Не теряем `message_ids` при разбиении длинных тредов.
- Короткие ответы не тонут рядом с длинными соседями.
- Chunk'и остаются читаемыми глазом - полезно для rerank.

### Решение

Оставили. Базовая идея всей системы.

---

## Гипотеза 2. Три текста на chunk вместо одного

### Зачем

Текст, оптимизированный для rerank (cross-encoder любит «беспорядочное»
сохранение служебки), вредит dense embedding'у (который любит чистый
смысловой текст). Один текст на всё - компромисс, который плох везде.

### Что сделали

Для каждого chunk'а формируются три варианта:
- `page_content` - для payload + rerank (читаемо).
- `dense_content` - для dense retrieval (очищен).
- `sparse_content` - для BM25 (одной строкой на сообщение, с `sender:`).

### Что увидели

Dense-ветка лучше ловит семантику, sparse - точные токены. Конфликт между
задачами снизился.

### Решение

Оставили. Финальная архитектура.

---

## Гипотеза 3. Цитаты и forwards размечать явно

### Зачем

Если quoted-текст смешан с собственной репликой, retrieval переоценивает
цитату (она длиннее и «похожа» на много вопросов).

### Что сделали

- На индексации: quoted-части помечаются как `Quoted message:`.
- На поиске: `QUOTE_DOWNWEIGHT = 0.2` - scores от quoted-block'ов
  уменьшаются в 5 раз относительно собственного ответа автора
  (см. `score_message_block` в [search/main.py](../search/main.py)).

### Что увидели

Меньше случаев, где chunk побеждает только за счёт overlap/quote, а не
реального ответа.

### Решение

Оставили.

---

## Гипотеза 4. `search_text` как primary query

### Зачем

Сырой `question.text` часто разговорный, неполный или содержит мусор
(«ребята, кто-нибудь помнит, где…»).

### Что сделали

Primary query строим из `question.search_text`, если есть, иначе `text`.

### Что увидели

На paraphrase-heavy вопросах стабильнее. Особенно когда в enriched-question
уже собраны технические токены и сущности.

### Решение

Оставили.

---

## Гипотеза 5. Dense retrieval - несколько запросов, а не один

### Зачем

Один embedding плохо покрывает разные формулировки.

### Что сделали

Dense queries строим из объединения (дедуп, `MAX_DENSE_QUERIES=8`):
- `search_text`, `text`
- часть `variants[]`
- часть `hyde[]`

### Что увидели

Recall устойчивее, когда вопрос и ответ говорят об одном и том же разными
словами. Особенно полезен HyDE-pass: «гипотетический ответ» как query
иногда единственный способ найти неочевидный тред.

### Решение

Оставили.

---

## Гипотеза 6. Sparse тоже на enriched question, не только primary

### Зачем

В чатах много exact-match: имена, версии, email, URL, технические токены.
Один sparse query теряет половину сигнала.

### Что сделали

Sparse queries строим из (дедуп, `MAX_SPARSE_QUERIES=8`):
- primary query
- `keywords`, `entities`, `date_mentions`, `asker`
- части `variants`, `hyde`

### Что увидели

Entity-heavy вопросы стали надёжнее.

### Решение

Оставили.

---

## Гипотеза 7. Fusion через DBSF/RRF, не ручная сумма score

### Зачем

Dense cosine-score и sparse BM25-score лежат в разных шкалах.
Ручная сумма (`α × dense + (1-α) × sparse`) хрупкая - требует тонкой
настройки, и легко ломается при смене модели.

### Что сделали

После prefetch - `FusionQuery(fusion=Fusion.DBSF)` (default) или RRF.
Qdrant сам нормализует через z-score или rank-reciprocal.

### Что увидели

Retrieval стабильнее. Ни одна ветка не «перетирает» другую из-за
масштаба score.

### Решение

Оставили. DBSF по дефолту, RRF переключаем env'ом.

---

## Гипотеза 8. Local rescoring перед внешним rerank

### Зачем

Внешний reranker дорогой (~300ms на 20 кандидатов) и не всегда устойчивый.
Часть сигналов дёшево ловится локально.

### Что сделали

`local rescoring` после fusion (~40ms на 150 кандидатов):
- phrase hits / token hits / signal tokens
- `participants` / `mentions` / `asker`
- `date_range` overlap
- best message-block внутри chunk
- context vs messages section penalty
- intent alignment (опционально)

### Что увидели

Особенно помогает:
- exact/entity-heavy вопросам (phrase hits сильнее, чем dense cosine);
- fallback-сценариям, когда reranker упал;
- когда overlap-контекст начинает тянуть score на себя.

### Решение

Оставили.

---

## Гипотеза 9. В reranker - `MESSAGES → CONTEXT`, не наоборот

### Зачем

Reranker читает `(query, document)` целиком. Если сначала показать overlap-
контекст, он может решить, что это основной ответ.

### Что сделали

При подготовке rerank input секции идут `MESSAGES:` (сам chunk) → `CONTEXT:`
(overlap). Если текст обрезается по `RERANKER_MAX_LENGTH`, режется
CONTEXT, а не MESSAGES.

### Что увидели

Меньше случаев, где соседний overlap переезжает настоящий ответ.

### Решение

Оставили.

---

## Гипотеза 10. Pure rerank слишком агрессивен - нужен blend

### Зачем

Reranker качественный, но «самоуверенный»: может полностью перестроить
retrieval order и сломать recall-friendly ветви.

### Что сделали

Blended score: `α × rerank + (1-α) × retrieval_rank`, где
`RERANK_ALPHA = 0.2` (см. [search/main.py:266](../search/main.py#L266)).

### Что увидели

Rerank помогает первым 5-10 позициям (lift nDCG), но не разрушает хороший
prefetch. Сдвиг alpha в сторону 0.4-0.5 на eval стабильно проседал Recall@50.

### Решение

Оставили. Default `RERANK_ALPHA = 0.2`.

---

## Гипотеза 11. In-memory кэш для dense и rerank

### Зачем

В eval и на защите одни и те же вопросы повторяются - лишний раз ходить
в upstream API и провоцировать `429` незачем.

### Что сделали

In-memory `LRUCache` на время жизни контейнера:
- dense embeddings (ключ = `(text, task)`);
- rerank scores (ключ = `(query, candidate_text)`).

### Что увидели

Повторные прогоны eval стабильнее и быстрее.

### Решение

Оставили.

---

## Гипотеза 12. Явный fallback при `429` / upstream down

### Зачем

Красивый score бесполезен, если на демо приходит `500 Internal Server Error`.

### Что сделали

Трёхуровневая graceful degradation:
- `dense_embed_safe` → при фейле dense-ветка пустая, sparse-only retrieval.
- `rerank_safe` → при фейле используем retrieval order (blend = retrieval_rank).
- `qdrant_search` → при фейле возвращаем `empty result + status=degraded`,
  всё равно 200 OK, не 500.

### Что увидели

Chaos-test проходит все сценарии: `reranker_down`, `dense_down`, `qdrant_down` →
`200 + fallback-counter инкрементировался`. Latency p95 в degraded-mode ниже,
чем в normal (sparse-only быстрее dense).

### Решение

Оставили. См. [scripts/chaostest.py](../scripts/chaostest.py).

---

## Что не доведено до конца

### Abstention для no-answer queries

Идея: вернуть пустой `message_ids`, если score top-кандидата ниже порога.
Проблема: текущий harness (`eval/metrics.py`) на пустом `relevant` возвращает `0`,
поэтому калибровать это на основном eval нельзя. Нужен отдельный negative-набор.

### Морфологическая нормализация query

Потенциал есть (русский с падежами + разговорный стиль), но легко переусложнить
query side и уронить precision. В хакатоне не трогали.

### Жёсткие metadata-фильтры (date_range, participants)

В теории - могло бы резать FP. На практике `thread_sn` иногда пустой, даты
не всегда в enriched question, `participants` обогащается неравномерно.
Остановились на soft boost (`+0.06` / `+0.03`) вместо hard filter.

---

## Короткий вывод

Инженерная история была такая:

1. Укрепили архитектурный каркас: message-based chunking, три текста,
   quote-aware разметка.
2. Расширили retrieval: multi-query dense + sparse, HyDE-pass, DBSF fusion.
3. Добавили аккуратный ranking: local rescoring + blended rerank.
4. Закрыли операционные риски: кэш + graceful fallback + metrics.

Главная зона роста сейчас - не «ещё один магический коэффициент», а:

- no-answer queries (abstention logic);
- multi-message ordering внутри top-50 (тоньше nDCG).

---

## Примеры — измеренные deltas

Где у нас есть точные цифры из git-commits и sweep-ов, а не только
качественные наблюдения. Все замеры — на `dataset_ts.jsonl` (N=432),
если не указано иначе.

### Commit-deltas из истории ветки

| Commit | Изменение | Было → Стало | Delta |
| --- | --- | --- | --- |
| `1200af4` | Fusion: RRF → DBSF | nDCG: 0.5265 → 0.5554 | **+0.0289 nDCG** |
| `102e714` | `RERANK_ALPHA`: 0.3 → 0.2 | nDCG: 0.5474 → 0.5556 | **+0.0082 nDCG** |
| `3412a4b` | Plain-query focus-token extraction | nDCG: 0.5334 → 0.5556 | **+0.0222 nDCG** |
| `8d3adbe` | HyDE boost + отдельный retrieval pass | score: 0.690 → 0.956 | **+0.266 score** (early branch) |

Эти цифры из commit-messages — замеры делали в момент принятия решения,
до того, как ветка поехала дальше. Самые «честные» delta: изменение
изолировано.

### RERANK_ALPHA sweep (гипотеза 10)

Полный sweep на `dataset_ts`:

| α   | Recall@50 | nDCG@50 | score  |
| --- | --------- | ------- | ------ |
| 0.1 | 1.0000    | 0.5389  | 0.9078 |
| **0.2** | **1.0000**    | **0.5556**  | **0.9111** ← default |
| 0.3 | 1.0000    | 0.5474  | 0.9095 |
| 0.4 | 0.9977    | 0.5431  | 0.9068 |
| 0.5 | 0.9954    | 0.5421  | 0.9048 |

Выше 0.4 Recall начинает проседать — reranker ломает recall-friendly
prefetch. Оптимум на 0.2.

### MAX_CHUNK_CHARS sweep (гипотеза 1, косвенно)

| value | chunks | Recall@50 | nDCG@50 | score |
| ----- | ------ | --------- | ------- | ----- |
| 600   | 221    | 0.9048    | 0.4812  | 0.8201 |
| 1200  | 112    | 0.9524    | 0.5432  | 0.8706 |
| **1800**  | 74 | **0.9524** | **0.5893** | **0.8798** ← оптимум |
| 2400  | 56     | 0.9286    | 0.5612  | 0.8551 |
| 3600  | 38     | 0.8810    | 0.5423  | 0.8133 |

U-образная кривая: маленькие чанки режут ответ пополам, большие — размазывают
сигнал.

### Chaos scenarios — latency & recall impact

| Scenario | Recall@50 | nDCG@50 | p50 latency | p95 latency |
| --- | --- | --- | --- | --- |
| baseline (full) | 0.9750 | 0.9470 | 410ms | 980ms |
| reranker_down | 0.9750 | **0.8834** (-0.064) | 180ms | 260ms |
| dense_down | 0.9625 (-0.013) | 0.9021 (-0.045) | 120ms | 220ms |
| qdrant_down | 0.0000 | 0.0000 | 55ms | 90ms |

Что видно:

- reranker_down: recall не страдает (retrieval тот же), nDCG падает на ~6% (порядок без cross-encoder).
- dense_down: recall проседает немного (sparse-only теряет парафразы).
- qdrant_down: всё по нулям, но 200 OK + status=degraded, не 500.

### In-memory cache (гипотеза 11) — proxy-измерение

На повторном прогоне `eval/run.py --dataset dataset_v2`:

| Прогон | Total wall-clock | dense_embed avg | rerank avg |
| --- | --- | --- | --- |
| cold (первый) | 98s | 210ms | 320ms |
| warm (cache hit) | 41s | 4ms | 3ms |

Кэш экономит ~60% wall-clock на повторных прогонах — важно для sweep'ов,
которые гоняют один eval много раз.
