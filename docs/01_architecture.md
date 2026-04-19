# Architecture

## Overview

Система состоит из трёх частей:
- `index-service`
- `search-service`
- `Qdrant`

`index-service` подготавливает данные для индексации.  
`search-service` строит query, выполняет retrieval и ранжирует кандидатов.  
`Qdrant` хранит dense и sparse представления чанков.

## Components

```text
chat payload
  -> index-service
  -> chunks + dense_content + sparse_content
  -> dense/sparse vectors
  -> Qdrant

question
  -> search-service
  -> dense + sparse retrieval
  -> fusion
  -> rescoring / rerank
  -> message_ids
```

## Indexing model

`index` работает с сообщениями, а не с сырой строкой чата:
- нормализует текст;
- удаляет шум;
- разбивает длинные сообщения;
- собирает чанки;
- возвращает три текстовых представления:
  - `page_content`
  - `dense_content`
  - `sparse_content`

Dense-векторы строятся во внешнем API.  
Sparse-векторы считаются локально через `fastembed`.

## Search model

`search` использует hybrid retrieval:
- dense retrieval для смысловых совпадений;
- sparse retrieval для точных терминов и именованных сущностей.

После retrieval выполняются:
- fusion результатов;
- локальный rescoring;
- optional rerank top-кандидатов;
- сборка финального списка `message_ids`.

## Main design decisions

- Разделение на два сервиса упрощает поддержку API-контрактов.
- Dense и sparse пути разделены, потому что у них разные сильные стороны.
- Для индекса используются разные текстовые поля под retrieval, payload и sparse branch.
- В поиске есть fallback-режимы: при проблемах с внешним dense/rerank API сервис не падает, а продолжает работу в упрощённом режиме.

## Repository layout

```text
index/
  main.py
  config.py
  schemas.py
  chunking.py
  sparse.py

search/
  main.py
  config.py
  schemas.py
  querying.py
  pipeline.py

eval/
scripts/
docs/
```
