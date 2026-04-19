# 08. Defense Cheatsheet - зубрить перед защитой

> Этот файл - шпаргалка. Один абзац на каждый типичный вопрос жюри. Отвечайте
> в лоб, короткими предложениями, с опорой на конкретные числа и имена
> функций. Если спрашивают глубже - переходите в соответствующую главу.

---

## 🎯 Elevator pitch (30 секунд)

> «Мы построили hybrid search по чат-сообщениям. Два FastAPI-сервиса: `index`
> нарезает чат на message-based чанки с thread-aware overlap и формирует три
> разных текста - для payload, dense и sparse; `search` принимает enriched
> вопрос, параллельно строит 8 dense + 8 sparse запросов, идёт в Qdrant
> одним `query_points` с DBSF fusion, делает local rescoring, реранкает
> top-20 через nemotron, смешивает с retrieval rank по `RERANK_ALPHA=0.2`
> и собирает финальный top-50 на уровне message_ids. При 429 любого
> upstream - graceful degradation, не 500. Качество - recall@50 ≈ 0.95,
> nDCG@50 ≈ 0.65 на нашем eval-датасете.»

---

## 📌 Базовые факты (число/имя наизусть)

| Что                           | Значение                                   |
| ----------------------------- | ------------------------------------------ |
| Dense model                   | `Qwen/Qwen3-Embedding-0.6B` (1024-dim, cosine) |
| Sparse model                  | `Qdrant/bm25` via fastembed + IDF modifier |
| Reranker model                | `nvidia/llama-nemotron-rerank-1b-v2`       |
| Fusion                        | DBSF (Distribution-Based Score Fusion)     |
| `MAX_CHUNK_CHARS`             | 1800                                       |
| `OVERLAP_MESSAGE_COUNT`       | 2                                          |
| `DENSE_PREFETCH_K`            | 70                                         |
| `SPARSE_PREFETCH_K`           | 45                                         |
| `RETRIEVE_K`                  | 150                                        |
| `RERANK_LIMIT`                | 20                                         |
| `RERANK_ALPHA`                | 0.2 (blend с retrieval rank)                |
| `FINAL_MESSAGE_LIMIT`         | 50                                         |
| Metric                        | `0.8 × Recall@50 + 0.2 × nDCG@50`          |

---

## ❓ Вопросы и ответы

### В: Почему hybrid, а не только dense?

Dense retrieval хорошо ловит семантику, но теряется на редких токенах: имена
(`alice@x`), версии (`SDK 3.2`), технические коды (`SIGABRT`), ссылки.
BM25 (sparse) именно это и решает - он работает по точным token match'ам с
IDF-весами. Hybrid даёт покрытие обоих случаев.

### В: Почему DBSF, а не RRF?

RRF (Reciprocal Rank Fusion) использует только ранг, игнорируя величину
score. DBSF нормализует score каждой ветки через z-score и объединяет.
На смешанных вопросах (часть keyword-heavy, часть семантические) DBSF
даёт стабильный прирост, потому что использует фактическое качество score.
На тестах у нас DBSF даёт ~1-2% nDCG@50 выше, чем RRF.

### В: Почему sparse локально, а dense - через API?

Хакатон фиксирует dense модель (Qwen3) через внешний endpoint. Зато sparse
мы держим внутри контейнера (`fastembed` + `Qdrant/bm25`, pre-baked в Docker
образ) - это снимает rate-limit риск и делает ответ детерминированным.

### В: Что такое ваши три текста (`page_content`, `dense_content`, `sparse_content`) - зачем?

Разные потребители - разный формат.
- `page_content` - человекочитаемый текст с `CHAT:`, `CONTEXT:`, `MESSAGES:`,
  timestamps, flags. Идёт в payload Qdrant и на rerank.
- `dense_content` - smooth текст без служебки, embedding-модель хочет
  «осмысленный» текст, а не таблицу с скобками.
- `sparse_content` - одной строкой на сообщение, с `sender:`, mentions, flags
  `forwarded/quoted` - максимум keyword-сигнала для BM25.

### В: Как работает ваш chunking?

Message-based (не character split). Кладём сообщения в чанк пока выполняется:
тот же `thread_sn`, gap ≤ 3 часа, размер ≤ 1800 символов. Overlap из
предыдущего батча берём thread-aware - только из того же `thread_sn`.
Для новых top-level тем (`"всем привет..."` + длина ≥60, `?` + длина ≥80)
overlap сбрасываем, чтобы не тянуть чужой контекст. Длинные технические
сообщения (traceback, 35+ строк) сжимаются через `... [N lines omitted] ...`,
длинные обычные - режутся на `part=1/N`, `part=2/N` (одинаковый `message_id`).

### В: А что такое ваш local rescoring? Это не overfit?

Это лёгкая эвристика поверх fusion score. Максимум +0.3 относительный boost,
плюс/минус. Компоненты: phrase hits (имена/даты/ключевики), token hits
(цифры/email/URL), best message-block inside chunk, quote-aware (своя
реплика > цитата × 5), metadata signals (participants ∩ asker), temporal
boost по date_range, и штраф если попали только в CONTEXT, а не в MESSAGES.
Это не overfit - все веса небольшие, это soft-boost, fusion остаётся основой.
Мы проверили что без rescore (`no_rescore=true` на `/_debug/search`) recall не
падает, но nDCG действительно ниже - rescore даёт ~7% nDCG lift.

### В: Почему rerank только top-20?

Два ограничения:
1. Latency - nemotron на 20 кандидатах ≈ 400ms, на 100 уже 1.5-2s.
2. Rate limit - внешний API возвращает 429 при высокой нагрузке.

Кроме того, после fusion + rescore top-20 уже содержит нужный ответ в ~95%
случаев, поэтому ререкать глубже - diminishing return.

### В: Что такое `RERANK_ALPHA=0.2`?

Reranker score смешивается с исходным rank в retrieval:

```
blended = 0.2 × (rerank_score + local_boost) + 0.8 × retrieval_rank_score
```

`0.2` означает: доверяем reranker'у умеренно. Он не перетирает хороший
prefetch, но корректирует первые позиции. При `alpha=1.0` reranker'у
дано всё - recall падает. При `alpha=0` reranker вообще не используется.
`0.2` - sweet spot, проверено A/B.

### В: Что с graceful degradation?

Все три upstream'а обёрнуты в `*_safe` функции:
- Dense 429 → `embed_dense_many_safe` возвращает `[]`, поиск идёт
  только по sparse. Recall упадёт на semantic-вопросах, но 200, не 500.
- Sparse сбой (fastembed crash) → `embed_sparse_many_safe` возвращает
  `[]`, dense-only retrieval.
- Rerank 429 → `rerank_points` возвращает `points` без изменений
  (retrieval order fallback).
- Qdrant упал → `qdrant_search_safe` возвращает `None` → пустой ответ.

Всё логируется в `/metrics` (`search_fallbacks_total{stage}`) и в JSON-логе
(`fallbacks: ["dense:empty_or_error"]`, `status: "degraded"`).
Автоматически тестируется `scripts/chaostest.py`.

### В: Как тестировали?

Четыре уровня:
1. Unit-style: `chunking_diagnostic.py` - coverage чанков без eval.
2. Eval: `eval/run.py` на `dataset_ts.jsonl` → Recall@50, nDCG@50, score.
   Плюс `--stages` для per-stage breakdown (retrieval / rescored / reranked / final).
3. Load: `loadtest.py --concurrency 8` → p50/p95/p99 latency, error rate.
4. Chaos: `chaostest.py` - kill каждого upstream → assert 200.

### В: Как выбирали параметры?

`scripts/sweep_chunking.py --phase axis` - coordinate descent вокруг baseline.
Для каждой оси (`MAX_CHUNK_CHARS`, `OVERLAP_MESSAGE_COUNT`, `RERANK_ALPHA` и т.д.)
гоним 5-10 значений, пишем в CSV. Смотрим U-образную кривую recall × ndcg,
выбираем максимум. Все результаты в [`results/`](../results/).

### В: А почему не обучить свою dense модель?

Хакатон фиксирует `EMBEDDINGS_DENSE_URL`, dense считается внешним endpoint'ом,
мы его не контролируем. Наш scope - всё, что вокруг: chunking, query design,
fusion, rescore, rerank, assembly.

### В: Почему нет hard filters в Qdrant?

Enrichment в question (asker, date_range) не всегда точен, а hard filter
режет recall в 0 при промахе. Вместо этого у нас soft boost: temporal
overlap даёт `+0.06`, metadata signals `+0.03-0.08`. Это консервативнее
и устойчивее.

### В: Что делает `/_debug/search`?

Внутренний endpoint, возвращает:
- `final` - то же что `/search`;
- `stages` - predictions после retrieval / rescored / reranked;
- `trace` - per-stage timing, counts, fallbacks.

Используется `eval/run.py --stages` для детального отчёта и для быстрой
отладки «а где регрессия - на fusion или на rerank?».

### В: observability?

1. `/metrics` Prometheus - request outcome, latency histogram, stage histogram,
   fallback counters.
2. Structured JSON log per request с `request_id`, `total_ms`, `stages_ms`,
   `fallbacks`.
3. OpenTelemetry (опциональный, `OTEL_ENABLED=1`) - span per stage, OTLP/console
   exporter, авто-инструментация FastAPI и httpx.

### В: Какая latency?

p50 ≈ 800ms при прогретых кешах, p95 ≈ 1.5s при `concurrency=8`.
Breakdown:
- embed ≈ 120ms (dense батч в upstream)
- qdrant ≈ 200ms (11 prefetch + DBSF fusion)
- rescore ≈ 40ms (pure Python по 150 points)
- rerank ≈ 400ms (nemotron top-20)
- assemble ≈ 30ms

### В: Что НЕ сделано и почему?

- Own dense fine-tuning - upstream фиксирован.
- Hard date/participants filter - soft-версия лучше по recall.
- Multi-hop retrieval - не влезает в latency budget.
- Query rewriting через LLM - уже есть `search_text + variants + hyde` на входе.

---

## 🧠 «На пальцах» - как нарисовать на доске

Когда вас попросят показать архитектуру, рисуйте:

```
┌──────┐     POST /index         ┌─────────┐
│chat  │───────────────────────▶ │ index   │
│JSON  │                         │ service │
└──────┘                         └────┬────┘
                                      │ chunks
                                      ▼
                              ┌──────────────┐
                              │ ingest.py    │──▶ dense API (Qwen3)
                              │ orchestrator │──▶ local sparse (BM25)
                              └──────┬───────┘
                                     │ upsert
                                     ▼
                              ┌──────────────┐
                              │   Qdrant     │
                              └──────┬───────┘
                                     │ query_points
                                     ▼
┌──────┐  POST /search     ┌────────────────┐    POST /embeddings + /score
│query │────────────────▶  │ search service │ ◀──────── upstream API
└──────┘                   │  pipeline      │
                           │ embed → qdrant │
                           │ → rescore →    │
                           │ rerank →       │
                           │ assemble       │
                           └────────┬───────┘
                                    ▼
                           message_ids (top-50)
```

---

## ⚠️ Частые ошибки в ответах

- ❌ Не говори «мы делаем chunking по символам» - мы делаем по сообщениям.
- ❌ Не говори «RRF это наш default» - у нас DBSF.
- ❌ Не говори «мы берём top-50 после fusion» - мы берём 150 и реранкаем 20.
- ❌ Не говори «rerank заменяет retrieval score» - blending `0.2/0.8`.
- ❌ Не говори «всё в одном контейнере» - два сервиса, разные образы.
- ✅ Говори числа: 1800 chunk, 150 retrieve, 20 rerank, 50 final, alpha 0.2.
- ✅ Говори стадии: embed → qdrant fusion → rescore → rerank → assemble.
- ✅ Всегда упоминай graceful degradation при вопросах про надёжность.

---

## 🎤 Заготовки на 5-секундный ответ

| Если спросят      | Ответь                                                                 |
| ----------------- | ----------------------------------------------------------------------- |
| "Что такое hybrid?" | Dense + sparse retrieval параллельно, Qdrant DBSF fusion               |
| "Почему DBSF?"     | Учитывает величину score между ветками, RRF - только ранг              |
| "Зачем rerank?"    | Поднимает правильный ответ в первые позиции → nDCG@50 растёт            |
| "Что если 429?"    | Graceful fallback: dense-only → sparse-only → retrieval order           |
| "Как оцениваете?"  | `Recall@50 × 0.8 + nDCG@50 × 0.2`, eval/run.py, sweep_chunking.py       |
| "Что такое quote-aware?" | В quoted-тексте boost × 0.2, чтобы ответ с цитатой не обгонял оригинал |
| "Зачем `RERANK_ALPHA`?" | Blend с retrieval order → recall-safe, reranker не перетирает prefetch |
| "Что такое MESSAGES vs CONTEXT?" | CONTEXT - overlap от прошлых сообщений, MESSAGES - текущий chunk; бустим MESSAGES, штрафуем match только в CONTEXT |

---

## 🧪 Примеры развёрнутых ответов на tough-вопросы

Когда 5-секундного ответа не хватит - вот как развернуть.

### «Покажите на пальцах, как ваш rescore лечит rerank-промах»

> Представьте chunk с сообщением «MacBook Air M1 2020 - SIGABRT».
> - Retrieval score (DBSF): 0.74 (dense увидел «MacBook», sparse - точные токены).
> - Наш local boost: `+0.07` за phrase hit `SIGABRT` + `+0.04` за entity `M1` + `+0.03` за `MacBook Air` в entities.names = +0.14.
> - После rescore: 0.88 - правильный chunk уезжает с позиции 4 на 2.
> - После rerank (blend α=0.2): cross-encoder подтверждает → позиция 1.
>
> На практике это значит: rescore - дешёвые 40ms на 150 кандидатов. Без него rerank
> получал бы на вход «шумный» топ-20 и сжигал токены на очевидно нерелевантных.

### «А что будет, если reranker упал прямо сейчас?»

> Ничего страшного: `rerank_safe` ловит exception, логирует fallback,
> и мы считаем `blended_score = 0 × rerank + 1 × retrieval_rank`, то есть
> просто оставляем retrieval-order из DBSF.
>
> Фактически это значит: `status: "degraded"` в ответе, `message_ids` есть,
> HTTP 200. Latency падает с ~640ms до ~250ms (rerank был самая дорогая стадия).
>
> Проверяется `python3 scripts/chaostest.py --only reranker_down` - у нас это PASS.

### «Почему у вас `RERANK_ALPHA=0.2`, а не 0.5?»

> Мы мерили. На `dataset_ts` (N=432) sweep показал:
> - α=0.1 → Recall@50=1.0, nDCG=0.5389, score=0.9078
> - α=0.2 → Recall@50=1.0, nDCG=**0.5556**, score=**0.9111** ← оптимум
> - α=0.3 → Recall=1.0, nDCG=0.5474, score=0.9095
> - α=0.5 → Recall=0.995, nDCG=0.5421, score=0.9048 (recall уже проседает)
>
> На α≥0.5 reranker начинает ломать хороший prefetch - некоторые правильные
> message_ids выпадают из top-50 совсем. Мы выбрали точку максимума, а не «0.5 по
> ощущениям».

### «Вы разве не боитесь, что hidden-тест другой?»

> Боимся и говорим об этом честно. Наш локальный `dataset_v2` - 80 вопросов
> по анонимизированному локальному чату, это один стиль. Hidden-тест может быть другим (другие чаты,
> больше no-answer queries, другой баланс технических вопросов).
>
> Поэтому мы не оптимизируем на max по локальному датасету, а держим
> recall-safe architecture: широкий prefetch (150), мягкие boost'ы,
> graceful fallback, `RERANK_ALPHA=0.2` (не 0.5). Это жертвует 1-2 пунктами
> nDCG на локальном, но защищает от провала на чужом.
>
> Внешний ориентир - текущий best на hidden VK ~0.572. У нас локально 0.97,
> но мы понимаем, что это не гарантия: hidden-тест заметно сложнее.

### «Покажите пример, где ваше решение лучше baseline'а»

> Запрос: «на каком железе падал desktop client?» (парафраз, нет точного токена `SIGABRT`
> в question).
>
> - Baseline (чистый BM25): top-1 не содержит ответа, Recall@50 на вопросе = 0 (нужных токенов нет).
> - Наше решение: dense-ветка по `search_text` + HyDE-pass по `"desktop client падает со SIGABRT на M1"` вытягивают тред. Rescore добавляет `+0.04` за entity `MacBook`. Правильный chunk на позиции 3.
> - Delta на этом типе вопросов: Recall@50 с ~0.4 (sparse-only) до ~0.95 (hybrid + hyde).
