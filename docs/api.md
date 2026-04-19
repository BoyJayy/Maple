# API

## Services

В проекте есть два сервиса:
- `index-service`
- `search-service`

## Index service

### `GET /health`

Ответ:

```json
{
  "status": "ok"
}
```

### `POST /index`

Назначение:
- принять чат и сообщения;
- вернуть чанки для последующей индексации.

Минимальная структура запроса:

```json
{
  "data": {
    "chat": {
      "id": "string",
      "name": "string",
      "sn": "string",
      "type": "string"
    },
    "overlap_messages": [],
    "new_messages": []
  }
}
```

Структура ответа:

```json
{
  "results": [
    {
      "page_content": "string",
      "dense_content": "string",
      "sparse_content": "string",
      "message_ids": ["string"]
    }
  ]
}
```

### `POST /sparse_embedding`

Запрос:

```json
{
  "texts": ["string"]
}
```

Ответ:

```json
{
  "vectors": [
    {
      "indices": [1, 2, 3],
      "values": [0.1, 0.2, 0.3]
    }
  ]
}
```

## Search service

### `GET /health`

Ответ:

```json
{
  "status": "ok"
}
```

### `POST /search`

Запрос:

```json
{
  "question": {
    "text": "string",
    "search_text": "string",
    "variants": [],
    "hyde": [],
    "keywords": [],
    "entities": null,
    "date_mentions": [],
    "date_range": null,
    "asker": ""
  }
}
```

Ответ:

```json
{
  "results": [
    {
      "message_ids": ["string"]
    }
  ]
}
```

### `POST /_debug/search`

Сервисный endpoint для локальной диагностики.  
Позволяет посмотреть промежуточные стадии retrieval, rescoring и rerank.

## Compatibility

Эти контракты считаются внешними.  
Менять формат запросов и ответов нельзя.  
Изменять можно только внутреннюю реализацию сервисов.
