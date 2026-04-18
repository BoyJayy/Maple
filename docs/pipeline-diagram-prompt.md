# Промпт для генерации схемы pipeline

Скопировать в Claude (claude.ai или Claude Code с артефактами). Просит нарисовать всю схему поиск-движка в двух форматах: **Mermaid** (для README/документации) и **SVG-артефакт** (для слайдов презентации).

---

## ПРОМПТ (копировать целиком)

Нарисуй схему ML-pipeline для гибридного поискового движка по чатам. Нужны **два варианта**:

1. **Mermaid flowchart** — для вставки в markdown-документацию
2. **Самодостаточный SVG-артефакт** (HTML-страница с inline-SVG) — для слайдов презентации, 1920×1080, тёмная тема, крупные подписи, аккуратные стрелки

### Что изображаем

Система состоит из **двух фаз**, которые связаны через **Qdrant** (единственное общее состояние).

---

### ФАЗА 1: INDEXING (выполняется один раз при загрузке данных)

Data flow от сырых данных до векторной БД. Четыре этапа:

**Источник:** `data/Go Nova.json` — JSON с сообщениями чатов.

**[A] INDEX SERVICE** (`index/main.py`, FastAPI, endpoint `POST /index`):
- Входит `ChatData{chat, overlap_messages, new_messages}`
- Функция `build_chunks()` делает:
  - filter (отбрасывает `is_system`, `is_hidden`)
  - render (склеивает `text + parts[].text + file_snippets`)
  - sort по `(time, id)`
  - boundary cut: граница, если смена thread, или gap > 1 час, или размер > 1600 символов
  - overlap tail: последние 2 сообщения переносятся в следующий чанк (только внутри того же thread и без hard-break)
  - эмитим только чанки, содержащие хотя бы одно `new_message`
- Возвращает `IndexAPIItem[]` со ФИР ПОЛЯМИ:
  - `page_content` — пойдёт в payload и как input реранкера
  - `dense_content` — пойдёт в dense embedding модель
  - `sparse_content` — пойдёт в sparse embedding модель
  - `message_ids` — массив id сообщений в чанке

**[B] INGESTION ORCHESTRATOR** (`eval/ingest.py`):
- Считает `stable_chunk_id = uuid5(NAMESPACE, chat_id + sorted(message_ids))` — детерминирован, upsert идемпотентен
- Батчем гонит `dense_content[]` → внешний **DENSE API** (Qwen/Qwen3-Embedding-0.6B, OpenAI-compatible `/v1/embeddings`, Basic Auth) → получает `vec[1024]` float32, метрика cosine
- Батчем гонит `sparse_content[]` → локальный `POST /sparse_embedding` (fastembed Qdrant/bm25) → получает `{indices[], values[]}`
- Строит payload: `page_content` + `metadata{chat_name, chat_type, chat_id, chat_sn, thread_sn, message_ids, start, end, participants, mentions, contains_forward, contains_quote}`
- Делает `upsert` в Qdrant

**[C] QDRANT COLLECTION** `"evaluation"`:
- Named vectors:
  - `"dense"` → `VectorParams(size=1024, distance=Cosine)`
  - `"sparse"` → `SparseVectorParams(modifier=IDF)`
- Payload: `page_content` + `metadata` (см. выше)
- point.id = `stable_chunk_id`

---

### ФАЗА 2: SEARCH (выполняется на каждый вопрос)

**Вход:** `question = {text, variants?, hyde?, keywords?, entities?, date_range?, asker?, ...}` — расширенный объект, сейчас используется только `text`.

**[D] HYBRID QUERY** (`search/main.py`, функция `qdrant_search`):
- `question.text` параллельно:
  - → **DENSE API** → `query_dense[1024]`
  - → локальный fastembed bm25 (lru_cache) → `query_sparse{indices, values}`
- В Qdrant отправляется `query_points` с двумя prefetch:
  - `prefetch_dense = top 10` по cosine через named vector `"dense"`
  - `prefetch_sparse = top 30` по BM25+IDF через named vector `"sparse"`
  - fusion = **RRF** (Reciprocal Rank Fusion), формула `Σ 1/(60 + rank)`
- Возвращает top `RETRIEVE_K = 20` points с payload

**[E] RERANK**:
- `POST` к внешнему **RERANKER API** (`nvidia/llama-nemotron-rerank-1b-v2`)
- input: `(query, [point.payload.page_content, ...])` — 20 пар
- output: `score[]` float per pair
- points пересортируются по score DESC

**[F] FINAL ASSEMBLY** (ещё не реализовано, план в роадмапе):
- Flatten `message_ids` по всем чанкам с сохранением порядка
- Dedup + diversity control (max 5–10 chunks из одного thread, из одного temporal window)
- Обрезать до 50 уникальных `message_ids`

**Выход:** `{"results": [{"message_ids": ["...", "...", ...]}]}`

---

### Обязательные элементы схемы

1. **Два чётких горизонтальных блока** — "INDEXING" сверху, "SEARCH" снизу, связаны через QDRANT посередине
2. **Qdrant нарисовать как цилиндр или DB-иконку** — подчеркнуть, что это shared state
3. **Внешние API** (DENSE, RERANKER) — отдельным стилем (например, пунктирный контур, значок облака) — подчеркнуть, что это вне нашей инфраструктуры
4. **Три поля контента** (`page_content` / `dense_content` / `sparse_content`) показать как разветвление стрелок после [A] с подписями, куда идёт каждое поле
5. **Подписи на стрелках** — не только типы данных, но и объёмы/форматы (например: `vec[1024]`, `sparse{indices,values}`, `top 10/30`, `RRF`, `top 20`, `score[]`, `cap 50`)
6. **Выделить компоненты**, которые ещё не реализованы (блок [F], query expansion из `variants/hyde`) — например, пунктирной рамкой или надписью "TODO"

### Цветовая палитра (для SVG)

- INDEXING блок: холодный синий/циан
- SEARCH блок: тёплый оранжевый/жёлтый
- Qdrant (shared state): нейтральный серый/фиолетовый, крупнее остальных
- Внешние API: зелёный с пунктиром
- TODO-компоненты: полупрозрачные, с подписью
- Фон: тёмный (#0d1117 или похожий)
- Подписи: светлый шрифт, sans-serif, крупно

### Что НЕ делать

- Не рисуй отдельные микросервисы как Kubernetes-поды, docker whale и т.д. — это именно data-flow схема, не deployment-схема
- Не упоминай backend, Docker, deployment — фокус на данных и моделях
- Не вставляй код — только блоки с названиями и стрелки

---

Сначала выдай **Mermaid версию** в code-блоке, потом — **HTML-артефакт с SVG**. Для Mermaid используй `flowchart TB` с subgraph для фаз. Для SVG — один HTML-файл, который можно открыть в браузере и снять скриншот для слайда.

---

## Как использовать

1. Скопировать всё содержимое секции "ПРОМПТ" (от "Нарисуй схему" до конца)
2. Вставить в claude.ai новую беседу
3. Claude выдаст два варианта — Mermaid и HTML-артефакт
4. Mermaid → вставить в README/docs, GitHub отрендерит
5. HTML → открыть в браузере → Cmd+Shift+4 (macOS) → получить картинку для слайда
