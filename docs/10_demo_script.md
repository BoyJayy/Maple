# 10. Demo script (живое демо на 5-7 минут)

> Сценарий показа на защите. Цель не «продемонстрировать всё», а закрыть три
> пункта: контракт не сломан, поиск работает на реальном чате, метрика
> воспроизводима.

---

## 10.1 Подготовка заранее (до выхода на сцену)

```bash
# 1. Credentials
export OPEN_API_LOGIN='...'
export OPEN_API_PASSWORD='...'

# 2. Поднять стек (занимает 1-2 минуты - ДО защиты)
docker compose up --build -d

# 3. Загрузить реальный чат
python3 eval/ingest.py
```

Открыть рядом в IDE / браузере:
- [08_defense_cheatsheet.md](08_defense_cheatsheet.md)
- [01_architecture.md](01_architecture.md) (схема)
- [api.md](api.md) (контракты)

---

## 10.2 План показа

### 0. Вступление (20 секунд)

> Задача - вернуть не абстрактный «документ», а конкретные `message_ids`
> по чату. Метрика `0.8×Recall@50 + 0.2×nDCG@50`. Внешний API мы не меняли,
> усиливали внутренний pipeline: chunking, hybrid retrieval, local rescore,
> rerank, assemble на уровне сообщений.

### 1. Контракты живы (30 секунд)

```bash
curl -sf http://localhost:8001/health   # index
curl -sf http://localhost:8002/health   # search
```

Открыть [api.md](api.md) и сказать:

> Мы не меняли ни `POST /index`, ни `POST /sparse_embedding`, ни `POST /search`.
> Вся оптимизация - внутри пайплайна.

### 2. Индексация (40 секунд)

```bash
curl -s -X POST http://localhost:8001/index \
  -H 'Content-Type: application/json' \
  --data-binary @data/index_request_sample.json \
  | python3 -m json.tool | sed -n '1,80p'
```

Глазами показать в ответе:
- есть поля `page_content`, `dense_content`, `sparse_content`, `message_ids`;
- `page_content` читаемый, с секциями `CHAT:`, `CONTEXT:`, `MESSAGES:`;
- chunk'и - не заглушки, реальный контент.

> На индексации чат не склеивается в одну строку. Нарезаем по сообщениям
> и отдельно готовим текст под payload, dense retrieval и sparse retrieval -
> три разных текста под три разные задачи.

### 3. Три демо-запроса (3-4 минуты)

Три сценария подобраны так, чтобы показать разные сильные стороны системы.

#### Запрос 1 - Технический exact-match

```bash
curl -s -X POST http://localhost:8002/search \
  -H 'Content-Type: application/json' \
  --data-raw '{
    "question": {
      "text": "на каком железе у человека падал desktop client с SIGABRT",
      "search_text": "SIGABRT железо MacBook Air M1 2020 desktop client",
      "keywords": ["SIGABRT", "MacBook Air", "M1"],
      "entities": {"names": ["MacBook Air", "M1"]}
    }
  }' | python3 -m json.tool
```

Ожидание: вверху - `MacBook Air (M1, 2020)`, `message_id` среди первых - `4555555555555555555`.

> Здесь sparse-ветка хорошо ловит точные токены `SIGABRT`, `M1`, `MacBook Air`,
> а rerank поднимает правильное сообщение в первую позицию.

#### Запрос 2 - Парафраз без точного совпадения

```bash
curl -s -X POST http://localhost:8002/search \
  -H 'Content-Type: application/json' \
  --data-raw '{
    "question": {
      "text": "разработчик не может запустить desktop client на новом яблочном ноуте",
      "search_text": "desktop client не запускается Mac ARM чип краш",
      "keywords": ["падает", "не запускается", "мак"],
      "entities": {"names": ["MacBook", "M1"]},
      "variants": [
        "программа валится при старте на арм-маке",
        "запуск desktop client на apple silicon выдаёт ошибку"
      ],
      "hyde": [
        "Пытаюсь запустить сервис на маке на Apple Silicon - не стартует, валится с ошибкой ещё до main."
      ]
    }
  }' | python3 -m json.tool
```

Ожидание: top-5 содержат сообщения из того же треда про SIGABRT.

> Keyword search тут уже плывёт - нет `SIGABRT`, нет «M1». Вытягивают dense-queries
> по `search_text`, `variants[]` и отдельный HyDE-pass: гипотетический ответ
> попадает в эмбеддинг и возвращает правильный тред.

#### Запрос 3 - Анонс / summary-type

```bash
curl -s -X POST http://localhost:8002/search \
  -H 'Content-Type: application/json' \
  --data-raw '{
    "question": {
      "text": "внешний инженерный митап в Панораме с тремя докладами",
      "search_text": "внешний митап Панорама три доклада апрель",
      "keywords": ["внешний митап", "Панорама", "три доклада"],
      "entities": {"names": ["Панорама"]},
      "variants": [
        "три доклада на апрельском инженерном митапе",
        "публичный инженерный митап компании в апреле"
      ]
    }
  }' | python3 -m json.tool
```

Ожидание: первым - анонс April Nova TechTalk, `message_id ≈ 4999999999999999999`.

> Это не технический вопрос, а событие. Работают другие сигналы: phrase_terms
> («Панорама»), entity_terms, и rerank.

### 4. Метрики (30 секунд)

Не гонять длинный eval вживую - открыть [06_tuning_and_eval.md](06_tuning_and_eval.md#616)
и показать готовую таблицу.

> На реальном локальном чате Recall@50 = 0.9750, nDCG@50 = 0.9470,
> итоговый score = 0.9694. Близко к теоретическому потолку 0.9750
> (2 dead queries с пустым ground truth).

Если жюри настаивает на команде:

```bash
python3 eval/run.py --dataset data/dataset_v2.jsonl --k 50
```

### 5. Финальная фраза (15 секунд)

> Главное улучшение - поиск стал message-aware. API не трогали,
> но усилили внутреннюю логику: chunking, hybrid retrieval, blended rerank
> и final assembly на уровне `message_ids`. И всё это переживает `429`
> от внешних сервисов - демо не упадёт из-за чужого upstream.

---

## 10.3 Если останется время (1-2 минуты)

- Открыть [11_experiments.md](11_experiments.md) - «вот 12 гипотез, которые мы проверили».
- Открыть [12_limitations.md](12_limitations.md) - «вот честный список рисков».
- `curl http://localhost:8002/metrics` - показать Prometheus histogram'ы `search_stage_duration_seconds` per stage.

---

## 10.4 Аварийные кнопки

| Если…                                   | Что делать                                                  |
| --------------------------------------- | ----------------------------------------------------------- |
| `/search` отвечает `status: degraded`  | Это нормально - graceful fallback, можно даже показать       |
| Запрос висит >5s                        | `docker compose restart search` и извиниться                 |
| Upstream API лёг                         | Показать chaos-test: `python3 scripts/chaostest.py`          |
| Сеть на сцене отвалилась                | Всё уже наполнено локально, только upstream embedding не доступен - graceful mode |

Никогда не показывать `docker compose down -v` на сцене - потеряете данные Qdrant.
