# 05. Qdrant, Hybrid Retrieval, Fusion, Rerank

> Это «теоретический» документ: чтобы на защите не плавать в формулировках
> типа «что такое RRF?», «зачем sparse?», «почему DBSF лучше?».

---

## 5.1 Что такое hybrid retrieval

**Проблема**: чистый dense retrieval (семантический поиск через embedding)
отлично ловит перефразировки, но плохо - редкие токены: имена, email'ы,
версии (`SDK 3.2`), коды ошибок (`SIGABRT`), ссылки.

**Решение**: параллельно гнать dense + sparse (BM25) retrieval и
объединять кандидаты. Это и есть hybrid.

| Ветка    | Что ловит                                      | Пример                              |
| -------- | ---------------------------------------------- | ----------------------------------- |
| Dense    | semantic similarity, перефразы                 | «когда релиз» ≈ «дата выхода»       |
| Sparse   | exact token match (BM25-style)                 | `SDK 3.2`, `SIGABRT`, `user@x`      |

## 5.2 Sparse (BM25 через fastembed)

Мы используем `Qdrant/bm25` - Qdrant'овский вариант BM25 с
IDF-модификатором, доступный через `fastembed`:

```python
from fastembed import SparseTextEmbedding
model = SparseTextEmbedding(model_name="Qdrant/bm25")
for item in model.embed(["SDK 3.2 release", ...]):
    item.indices  # np.array - global token indices
    item.values   # np.array - BM25 weights
```

В Qdrant коллекция создаётся с:

```python
sparse_vectors_config={
    "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF),
}
```

`modifier=IDF` означает, что Qdrant дополнительно учитывает inverse document
frequency при scoring - редкие токены дают больший вклад.

Почему локально, а не через внешний API:

1. Нет rate limit - мы не зависим от 429.
2. Детерминированно - тот же текст → тот же вектор.
3. Batch быстрее (µs per doc на CPU).
4. Модель one-time download в Docker build (`RUN python -c "SparseTextEmbedding(...)"`).

## 5.3 Dense (Qwen3-Embedding-0.6B)

Хакатонный внешний API: `POST http://83.166.249.64:18001/embeddings`
с моделью `Qwen/Qwen3-Embedding-0.6B` → 1024-мерный вектор, cosine distance.

Коллекция:

```python
vectors_config={
    "dense": models.VectorParams(size=1024, distance=models.Distance.COSINE),
}
```

Auth - Basic: `OPEN_API_LOGIN:OPEN_API_PASSWORD`.

## 5.4 Fusion - объединяем dense + sparse

Когда у тебя есть несколько prefetch-веток (например, 4 dense + 7 sparse),
Qdrant `Query API` делает `FusionQuery(fusion=...)`, который объединяет
кандидатов из всех веток.

### 5.4.1 RRF (Reciprocal Rank Fusion)

Классический алгоритм. Для каждого документа d в каждой ветке r:

```
score_rrf(d) = Σ (1 / (k + rank_i(d)))    where k=60
```

Плюсы:
- Не требует нормализации score между ветками.
- Устойчив к выбросам.

Минусы:
- Игнорирует величину score, только rank.
- При сильно разных качествах веток - может недоиспользовать лучшую.

### 5.4.2 DBSF (Distribution-Based Score Fusion)

Для каждой ветки Qdrant считает распределение score'ов
(mean, std), нормализует их (z-score clip) и объединяет:

```
score_dbsf(d) = Σ_i normalized_score_i(d)
```

Плюсы:
- Использует фактическое качество score, не только rank.
- Работает лучше на смешанных вопросах (короткие keyword + длинные смысловые).

Минусы:
- Чуть дороже по compute.
- Чувствителен к очень маленьким batch'ам (при 1-2 кандидатах std плохой).

Наш default - DBSF (см. `docker-compose.yml` не задаёт, но код: `fusion: str = "dbsf"`).
RRF оставлен как переключаемый режим.

### 5.4.3 Пример (упрощённый)

3 кандидата, 2 ветки:

| Кандидат | dense rank | dense score | sparse rank | sparse score |
| -------- | ---------- | ----------- | ----------- | ------------ |
| A        | 1          | 0.90        | 3           | 0.40         |
| B        | 3          | 0.55        | 1           | 0.95         |
| C        | 2          | 0.70        | 2           | 0.60         |

RRF (k=60):
- A: 1/61 + 1/63 = 0.0322
- B: 1/63 + 1/61 = 0.0322
- C: 1/62 + 1/62 = 0.0323 ← выигрывает, хотя не лучший ни в одной ветке!

DBSF (упрощённо z-score):
- A: normalized = (0.90-0.72)/0.15 + (0.40-0.65)/0.25 = 1.2 - 1.0 = 0.2
- B: (0.55-0.72)/0.15 + (0.95-0.65)/0.25 = -1.13 + 1.2 = 0.07
- C: (0.70-0.72)/0.15 + (0.60-0.65)/0.25 = -0.13 - 0.2 = -0.33

В DBSF A выигрывает, потому что он сильно лучший в dense.

## 5.5 Что такое «prefetch-ветка»

В Qdrant Query API можно сказать:

```python
client.query_points(
    collection_name=...,
    prefetch=[
        Prefetch(query=v1, using="dense",  limit=70),
        Prefetch(query=v2, using="dense",  limit=70),
        Prefetch(query=sv1, using="sparse", limit=45),
    ],
    query=FusionQuery(fusion=Fusion.DBSF),
    limit=150,
)
```

Qdrant:
1. Исполняет каждый `Prefetch` независимо (каждый возвращает до `limit` кандидатов).
2. Применяет `FusionQuery` - сливает, нормализует, сортирует.
3. Отдаёт финальный top-`limit=150`.

У нас `DENSE_PREFETCH_K=70` и `SPARSE_PREFETCH_K=45` - разные, потому что
sparse обычно даёт больше шума, и брать слишком много - загрязняет fusion.

## 5.6 Local rescoring (между fusion и rerank)

После fusion у нас `~150` кандидатов. Идти со всеми во внешний reranker - дорого
(latency + 429). Зато можно по-дешёвке подправить их порядок эвристиками,
которые reranker всё равно не увидит.

Что бустим:
1. Phrase hits - точные фразы/имена/даты/asker (`+0.04…0.07` each).
2. Token hits - signal-токены (цифры, @./:) - `+0.01` each, cap `+0.08`.
3. Best block score - лучший message-блок внутри chunk - `до +0.28`.
4. Quote-aware - quoted-текст весит × 0.2 от own.
5. Metadata signals - `participants ∩ asker/entities` - `до +0.06`.
6. Temporal - date_range overlap - `+0.06`.
7. Intent alignment (опц.) - summary vs detail - `±0.04…0.12`.
8. Context penalty - попали только в CONTEXT, не в MESSAGES → `-0.08`.

Все эти вклады складываются с исходным Qdrant score:

```
new_score = qdrant_fusion_score + local_boost
```

Boost не абсолютный, а относительный (magnitude `~0.01…0.3`).
Это не заменяет fusion, а только слегка его корректирует.

## 5.7 Rerank (внешний, cross-encoder)

Cross-encoder (nemotron-rerank-1b-v2) принимает пару `(query, candidate)` и
возвращает relevance score. В отличие от embedding'ов, он читает оба
текста одновременно - поэтому качественнее, но в N раз дороже.

Мы реранкаем только `RERANK_LIMIT=20` (первые 20 после rescore).

### Blend с retrieval rank

Если полностью довериться reranker'у, recall может упасть: хороший fusion
«прячет» рядом с топом нужный кандидат, а cross-encoder неудачно его оценит.
Мы смешиваем:

```python
retrieval_rank_score = 1.0 - (index / total_candidates)
rerank_score = nemotron_score + local_boost
blended = RERANK_ALPHA * rerank_score + (1 - RERANK_ALPHA) * retrieval_rank_score
```

`RERANK_ALPHA=0.2` → 80% веса у retrieval-порядка, 20% у rerank. Это
recall-safe: reranker не перетирает хороший prefetch, но всё ещё
поднимает правильные ответы в первые 5-10 позиций.

### Кеш

`RERANK_SCORE_CACHE: dict[(model, query, candidate_text), score]`.
На повторных eval-прогонах одинаковые пары кешируются - меньше 429 и быстрее.

## 5.8 Почему мы не используем hard filters в Qdrant

По метаданным Qdrant даёт:
```python
filter=models.Filter(
    must=[
        models.FieldCondition(
            key="metadata.participants",
            match=models.MatchValue(value="alice@x"),
        ),
    ],
)
```

Казалось бы, если вопрос от Carol про документ X - фильтруй по `participants:[carol@x]`.
Но:

1. Enrichment не всегда точен: `asker` может быть пустым или неверно заполненным.
2. Вопрос может быть про других: «что Bob сказал про X» - тут нужно bob, не asker.
3. Hard filter = 0 recall при промахе. Soft boost `+0.04` лучше.

Поэтому все metadata-сигналы у нас soft (через `score_metadata_signals`),
а date_range - мягкий `+0.06` при overlap.

## 5.9 Точки в коллекции (payload schema)

```json
{
  "id": "uuid5 детерминированный",
  "vector": {
    "dense": [ ...1024 floats... ],
    "sparse": {"indices": [...], "values": [...]}
  },
  "payload": {
    "page_content": "CHAT: ...\n\nMESSAGES:\n\n[... | ...] ...",
    "metadata": {
      "chat_id": "chat-42",
      "chat_name": "Atlas Hub",
      "chat_type": "group",
      "chat_sn": "gn",
      "thread_sn": "T1",
      "message_ids": ["m-1", "m-2"],
      "start": "1714744800",
      "end": "1714744900",
      "participants": ["alice@x", "bob@x"],
      "mentions": [],
      "contains_forward": false,
      "contains_quote": false
    }
  }
}
```

Всё поле `metadata` - уровнем вниз, чтобы фильтр по Qdrant мог адресовать
`metadata.chat_id` через `FieldCondition(key=...)`.

## 5.10 Как выбраны значения hyper-params

| Param                | Value   | Как выбрано                                                    |
| -------------------- | ------- | -------------------------------------------------------------- |
| `DENSE_PREFETCH_K`   | 70      | sweep, 50 - слишком узко, 100 - рост шума, 70 - оптимум         |
| `SPARSE_PREFETCH_K`  | 45      | sparse шумнее, меньше prefetch → чище fusion                   |
| `RETRIEVE_K`         | 150     | ≥3× RERANK_LIMIT, даёт запас на rescore без пересчёта          |
| `RERANK_LIMIT`       | 20      | nemotron справляется за ~400ms на 20, больше → 429 и latency    |
| `RERANK_ALPHA`       | 0.2     | 0.3 был агрессивнее, 0.1 - не хватает lift. 0.2 - recall-safe  |
| `MAX_DENSE_QUERIES`  | 8       | больше 8 → уже diminishing return + rate pressure              |
| `MAX_SPARSE_QUERIES` | 8       | то же                                                          |
| `HYDE_MIN_SIGNATURE` | 40      | короткий hyde ≈ исходный text, дополнительный pass бесполезен  |
| `HYDE_PASS_LIMIT`    | 60      | compromise между recall и latency отдельного pass              |

Все параметры подтверждены локальными sweep'ами через `sweep_chunking.py`
(для chunking) и ручными A/B через `scripts/ab_qdrant.py` (для retrieval).
Результаты в [`results/`](../results/).

## 5.11 Один запрос в одно обращение в Qdrant

Мы собираем все prefetch ветки в один `query_points` call:
- 4-8 dense + 4-8 sparse prefetch одним HTTP-запросом;
- Qdrant делает работу параллельно внутри;
- клиент получает один итоговый ranking.

Это в 3-4 раза быстрее, чем делать по одному запросу на ветку и склеивать
руками, плюс Qdrant делает fusion «правильнее» (видит все score distributions).

## 5.12 Что ещё можно улучшить (сознательно не сделали)

| Идея                          | Почему отложено                                                |
| ----------------------------- | -------------------------------------------------------------- |
| Own dense fine-tuning         | Хакатонный endpoint фиксирует модель                           |
| Matryoshka / MRL embeddings   | Не поддерживается upstream                                     |
| Query rewriting через LLM     | Уже даётся `search_text`+`variants`+`hyde` - не хотим дубль    |
| Hard date filter              | Ломает recall при неточном date_range                          |
| Personalization per asker     | Нет ground truth для обучения                                  |
| Multi-hop retrieval           | Перебор: 2-3 раунда × 800ms = timeout                          |

---

## 5.13 Пример — DBSF и RRF на одном маленьком датасете

Допустим, у нас 5 chunk'ов (C1…C5) и две ветки retrieval — dense и sparse.
Каждая возвращает top-5 со своими score.

### Raw output

| chunk | dense_score (cosine) | dense_rank | sparse_score (BM25) | sparse_rank |
| ----- | -------------------: | ---------: | ------------------: | ----------: |
| C1    | 0.92                 | 1          | 8.4                 | 3           |
| C2    | 0.88                 | 2          | 3.1                 | 5           |
| C3    | 0.71                 | 3          | 14.2                | 1           |
| C4    | 0.68                 | 4          | 11.7                | 2           |
| C5    | 0.40                 | 5          | 6.0                 | 4           |

Правильный ответ: C3 (BM25 его хорошо ловит за счёт exact tokens, а dense
поставил только на 3-е место - семантика близкая, но не «в точку»).

### RRF (Reciprocal Rank Fusion, k=60)

Формула: `score_rrf(d) = Σᵢ 1 / (k + rank_i(d))`.

```
C1: 1/(60+1) + 1/(60+3) = 0.01639 + 0.01587 = 0.03226
C2: 1/(60+2) + 1/(60+5) = 0.01613 + 0.01538 = 0.03151
C3: 1/(60+3) + 1/(60+1) = 0.01587 + 0.01639 = 0.03226   ← tied with C1
C4: 1/(60+4) + 1/(60+2) = 0.01563 + 0.01613 = 0.03176
C5: 1/(60+5) + 1/(60+4) = 0.01538 + 0.01563 = 0.03101
```

Итог RRF: `C1=C3 (0.0323) > C4 (0.0318) > C2 (0.0315) > C5 (0.0310)`.

RRF видит только ранги, поэтому «uncontroversial топ» у двух веток
(C1 - dense-лидер, C3 - sparse-лидер) получают одинаковый score. C3 всего
лишь равен C1, хотя по модулю BM25-score он сильно оторвался от второго места.

### DBSF (Distribution-Based Score Fusion)

Сначала нормализуем score в каждой ветке через z-score (`(x - mean) / std`),
потом суммируем.

```
dense:  mean=0.718, std=0.192
sparse: mean=8.68,  std=3.86

z_dense  = [1.05, 0.84, -0.04, -0.20, -1.66]
z_sparse = [-0.07, -1.44, 1.43, 0.78, -0.69]

DBSF score = z_dense + z_sparse:
C1: 1.05 + (-0.07) =  0.98
C2: 0.84 + (-1.44) = -0.60
C3: -0.04 + 1.43   =  1.39   ← лидер!
C4: -0.20 + 0.78   =  0.58
C5: -1.66 + (-0.69) = -2.35
```

Итог DBSF: `C3 (1.39) > C1 (0.98) > C4 (0.58) > C2 (-0.60) > C5 (-2.35)`.

C3 теперь чётко на первом месте - DBSF увидел, что его sparse-score не просто
ранг 1, а ещё и сильно оторван от распределения (z=1.43, «+1.4 сигмы выше
среднего»).

### Вывод

- RRF - консервативный, игнорирует magnitude. Хорош, когда score-шкалы
  несопоставимы и мы не доверяем их абсолютным значениям.
- DBSF - учитывает «насколько сильно оторвался». Лучше, когда один retriever
  может быть «очень уверен», а другой «просто ок».

Для нашей задачи (exact tokens часто решают - BM25 любит «отстреливаться»
высоким score на правильный chunk, пока dense вяло-похожий) DBSF выигрывает
стабильнее. Отсюда default `QDRANT_FUSION=DBSF`.
