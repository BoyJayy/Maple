# 07. Ops, Docker, Deployment

> Практическое руководство: как это поднять локально, как собрать и
> отправить образы в hackathon registry.

---

## 7.1 Dependencies (версии)

**index-service** ([index/requirements.txt](../index/requirements.txt)):
- Python 3.13-slim
- fastapi 0.136.0 + uvicorn[standard] 0.44.0
- pydantic 2.13.2
- fastembed 0.8.0 (для `SparseTextEmbedding("Qdrant/bm25")`)

**search-service** ([search/requirements.txt](../search/requirements.txt)):
- всё выше +
- httpx 0.28.1 (async клиент для upstream)
- qdrant-client 1.17.1
- prometheus-client 0.23.1
- opentelemetry-api/sdk 1.37.0 + instrumentation для fastapi/httpx + OTLP exporter

**Qdrant**: `qdrant/qdrant:v1.14.1` (Docker image).

## 7.2 Dockerfile'ы

Оба сервиса - `python:3.13-slim`, одинаковая структура:

```dockerfile
FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY *.py .
ENV HOST=0.0.0.0 PORT=8000
ENV FASTEMBED_CACHE_PATH=/models/fastembed HF_HOME=/models/huggingface
RUN mkdir -p /models/fastembed /models/huggingface
# Pre-bake sparse model в образ - чтобы runtime не качал из интернета
RUN python -c "from fastembed import SparseTextEmbedding; SparseTextEmbedding(model_name='Qdrant/bm25')"
EXPOSE 8000
CMD ["python", "main.py"]
```

Модель кешируется внутрь образа через `RUN python -c "..."`.
Без этого каждый старт контейнера = 30+ секунд download, и в проверочной
среде без интернета контейнер не поднимется вообще.

## 7.3 docker-compose.yml - локальный стек

[`docker-compose.yml`](../docker-compose.yml) поднимает всё разом:

```yaml
services:
  qdrant:                 # port 6333
    image: qdrant/qdrant:v1.14.1
  qdrant-init:            # один раз: PUT /collections/evaluation
    image: curlimages/curl
  index:                  # port 8001:8000
    build: ./index
    depends_on: [qdrant]
  search:                 # port 8002:8000
    build: ./search
    depends_on:
      qdrant-init: {condition: service_completed_successfully}
```

### Что делает `qdrant-init`

Waits for Qdrant → создаёт коллекцию `evaluation` с:

```json
{
  "vectors": {"dense": {"size": 1024, "distance": "Cosine"}},
  "sparse_vectors": {"sparse": {"modifier": "idf"}}
}
```

Это однократная операция. Если коллекция уже есть - exit 0.

### Env, которые прокидываются в search

Все параметры retrieval/rerank можно переопределить без пересборки:

```
QDRANT_URL=http://qdrant:6333
QDRANT_COLLECTION_NAME=evaluation
EMBEDDINGS_DENSE_URL=http://83.166.249.64:18001/embeddings
RERANKER_URL=http://83.166.249.64:18001/score
OPEN_API_LOGIN / OPEN_API_PASSWORD - required
RERANK_ALPHA / DENSE_PREFETCH_K / RETRIEVE_K / RERANK_LIMIT / ...
OTEL_ENABLED / OTEL_EXPORTER / OTEL_SERVICE_NAME
```

## 7.4 Локальный запуск - шаги

```bash
# 1. Credentials
export OPEN_API_LOGIN='...'
export OPEN_API_PASSWORD='...'

# 2. Построить и поднять всё
docker compose up --build -d

# 3. Убедиться, что контейнеры alive
docker compose ps
curl http://localhost:8001/health  # index
curl http://localhost:8002/health  # search

# 4. Индексация тестового чата
python3 eval/ingest.py                           # → default local chat corpus
# или synthetic eval корпус:
DATA_PATH=data/dataset_ts.jsonl RESET_COLLECTION=1 python3 eval/ingest.py

# 5. Тест поиска
curl -X POST http://localhost:8002/search \
  -H "Content-Type: application/json" \
  -d '{"question":{"text":"Что писали про релиз SDK 3.2?"}}'

# 6. Eval с метриками
python3 eval/run.py --dataset data/dataset_ts.jsonl --k 50 --verbose --stages
```

### Логи / остановка

```bash
docker compose logs -f search      # tail search-а
docker compose logs -f index
docker compose restart search      # перезапуск без пересборки
docker compose down                # убрать всё
docker compose down -v             # + очистить volumes (данные Qdrant)
```

## 7.5 Makefile (build + push)

### `index/Makefile`

```makefile
build:
	docker build --platform linux/amd64 -t 83.166.249.64:5000/$(TEAM_ID)/index-service:latest .

push: build
	docker push 83.166.249.64:5000/$(TEAM_ID)/index-service:latest

run:
	docker run --rm -p 8000:8000 83.166.249.64:5000/$(TEAM_ID)/index-service:latest
```

### `search/Makefile` - аналогично для `search-service`.

Платформа `linux/amd64` принципиальна: проверочная среда x86,
даже если вы на Apple Silicon. Docker сам сделает multi-arch build через buildx.

## 7.6 Отправка в registry

**Registry:** `83.166.249.64:5000` (HTTP, без TLS).

### Настройка Docker Desktop (macOS/Windows)

`Settings → Docker Engine → JSON config`:

```json
{
  "insecure-registries": ["83.166.249.64:5000"]
}
```

→ **Apply & Restart**.

### Логин и push

```bash
docker login 83.166.249.64:5000 -u "$LOGIN" -p "$PASSWORD"

cd index   && TEAM_ID=31023 LOGIN=... PASSWORD=... make push
cd search  && TEAM_ID=31023 LOGIN=... PASSWORD=... make push
```

В итоге в registry:
- `83.166.249.64:5000/31023/index-service:latest`
- `83.166.249.64:5000/31023/search-service:latest`

### Или вручную

```bash
docker build --platform linux/amd64 \
  -t 83.166.249.64:5000/31023/index-service:latest index/
docker push 83.166.249.64:5000/31023/index-service:latest

# аналогично для search
```

## 7.7 Observability на проде

### Prometheus

- Scrape endpoint: `GET http://search:8000/metrics`.
- Дефолтные метрики библиотеки плюс наши:
  - `search_requests_total{status}` - requests by outcome.
  - `search_request_duration_seconds` - p50/p95/p99 latency histogram.
  - `search_stage_duration_seconds{stage}` - breakdown по stage.
  - `search_fallbacks_total{stage}` - degradation events.
  - `search_errors_total{cls}` - exceptions.

### Structured logs

Каждый `/search` → одна JSON-строка в stdout:
```json
{"event":"search","request_id":"...","total_ms":..., "stages_ms":{...},
 "counts":{...},"fallbacks":[...],"errors":[...],"status":"ok|degraded|error"}
```

Легко парсить Loki/ELK.

### OpenTelemetry (опционально)

```bash
OTEL_ENABLED=1 \
OTEL_EXPORTER=otlp \
OTEL_SERVICE_NAME=search-service \
OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4318/v1/traces \
docker compose up -d search
```

Span'ы per stage, авто-инструментация FastAPI + httpx.

## 7.8 Ограничения среды проверки

- Образы отправляются в приватный registry `83.166.249.64:5000`.
- `QDRANT_URL`, `EMBEDDINGS_DENSE_URL`, `RERANKER_URL`, `API_KEY` (или `OPEN_API_LOGIN/PASSWORD`)
  подставляются проверяющей стороной - никогда не хардкодить.
- Qdrant на их стороне может быть уже наполнен их ingestion'ом (другой вариант
  payload), поэтому `search` не должен полагаться на конкретную структуру
  `metadata` сверх того, что описано в [05_qdrant_and_retrieval.md](05_qdrant_and_retrieval.md#59).
- `POST /search` → `POST /index` → `POST /sparse_embedding` контракты
  запрещено менять. Любой сдвиг схемы → отказ.

## 7.9 Ports reference

| Service | Container port | Host port (compose) | Назначение                    |
| ------- | -------------- | ------------------- | ------------------------------ |
| qdrant  | 6333           | 6333                | REST API + gRPC :6334 internal |
| index   | 8000           | 8001                | `/health`, `/index`, `/sparse_embedding` |
| search  | 8000           | 8002                | `/health`, `/search`, `/metrics`, `/_debug/search` |

## 7.10 Чек-лист перед отправкой

- [ ] `docker compose up --build` поднимается без ошибок.
- [ ] `GET /health` отвечает 200 на обоих сервисах.
- [ ] `eval/ingest.py` проходит на локальном чат-корпусе.
- [ ] `eval/run.py --dataset data/dataset_ts.jsonl` даёт `Recall@50 ≥ 0.90`.
- [ ] `scripts/chaostest.py` - все PASS.
- [ ] `scripts/loadtest.py --concurrency 8 --duration 30` - p95 ≤ 1500ms, error rate 0.
- [ ] Размер image'ей ≤ 2GB (проверить `docker images 83.166.249.64:5000/...`).
- [ ] Образы собраны под `linux/amd64`.
- [ ] `OPEN_API_LOGIN/PASSWORD` не захардкожены в Dockerfile / code.

---

## 7.11 Пример - полный trace от нуля до «сдано»

Ниже - всё, что делает инженер, чтобы переехать с чистого чекаута
до запушенных в хакатонский registry образов. Терминальный снимок примерно
такой:

```bash
# 0. Чистый чекаут
$ git clone https://github.com/BoyJayy/Hackathon-Search-Engine.git
$ cd Hackathon-Search-Engine
$ git checkout ashot_test

# 1. Credentials (выдают организаторы)
$ export OPEN_API_LOGIN='hackathon_user'
$ export OPEN_API_PASSWORD='***redacted***'
$ export TEAM_ID='31023'

# 2. Локальная сборка + подъём
$ docker compose up --build -d
[+] Running 4/4
 ✔ Network hackathon-search-engine_default      Created
 ✔ Container qdrant                             Started (port 6333)
 ✔ Container qdrant-init                        Completed (collection "evaluation" created)
 ✔ Container index                              Started (port 8001)
 ✔ Container search                             Started (port 8002)

# 3. Health check
$ curl -sf http://localhost:8001/health && echo OK
{"status":"ok"} OK
$ curl -sf http://localhost:8002/health && echo OK
{"status":"ok"} OK

# 4. Ingest реального корпуса
$ python3 eval/ingest.py
[ingest] loaded 1023 messages from local chat corpus
[ingest] POST /index -> 127 chunks in 2.8s
[ingest] POST /embeddings batch -> 1024-dim vectors in 4.1s
[ingest] qdrant upsert 127 points -> ok
[ingest] done in 7.4s

# 5. Smoke-search
$ curl -s -X POST http://localhost:8002/search \
    -H 'Content-Type: application/json' \
    -d '{"question": {"text": "про desktop client 3.4 и M1?"}}' | python3 -m json.tool | head
{
    "message_ids": [
        "4555555555555555555",
        "4555555555555555556",
        ...

# 6. Полный eval
$ python3 eval/run.py --dataset data/dataset_v2.jsonl --k 50
N = 80
stage        Recall@50    nDCG@50      score
-----------------------------------------------
final        0.9750       0.9470       0.9694

# 7. Chaos + load
$ python3 scripts/chaostest.py
=== baseline      PASS
=== reranker_down PASS
=== dense_down    PASS
=== qdrant_down   PASS

$ python3 scripts/loadtest.py --concurrency 8 --duration 30
rps=12.4  p50=411ms  p95=980ms  p99=1321ms  errors=0

# 8. Multi-arch build под проверочную среду
$ cd index && TEAM_ID=31023 make build
docker build --platform linux/amd64 \
  -t 83.166.249.64:5000/31023/index-service:latest .
$ cd ../search && TEAM_ID=31023 make build
# аналогично

# 9. Push в hackathon registry
$ docker login 83.166.249.64:5000 -u "$OPEN_API_LOGIN" -p "$OPEN_API_PASSWORD"
Login Succeeded
$ cd ../index  && make push
$ cd ../search && make push
The push refers to repository [83.166.249.64:5000/31023/search-service]
latest: digest: sha256:a3f... size: 2621

# 10. Верификация в registry
$ curl -su "$OPEN_API_LOGIN:$OPEN_API_PASSWORD" \
    http://83.166.249.64:5000/v2/31023/search-service/tags/list
{"name":"31023/search-service","tags":["latest"]}
```

Total wall-clock time (опытный запуск): ~9 минут (из них 6 - сборка образов).

### Если что-то сломалось - где смотреть

| Симптом | Где искать |
| --- | --- |
| `qdrant-init` падает | `docker compose logs qdrant-init` - скорее всего Qdrant ещё не поднялся. Перезапустить `docker compose up -d qdrant-init` |
| `index` возвращает 500 на `/index` | `docker compose logs index` - обычно падает на загрузке sparse-модели. Дёрнуть `docker image inspect` и проверить, что `RUN python -c "from fastembed..."` выполнился при build |
| `search` отвечает `status: "degraded"` сразу после старта | `OPEN_API_LOGIN/PASSWORD` не подхватились → smoke test с 4xx от upstream → автоматический fallback. Проверить `docker compose exec search env \| grep OPEN_API` |
| `docker push` висит на `authorization required` | `Settings → Docker Engine → JSON` не имеет `insecure-registries: ["83.166.249.64:5000"]`, restart Docker Desktop |
| Платформа не та (arm64 вместо amd64) | `docker inspect 83.166.249.64:5000/31023/search-service:latest \| grep Architecture` - должно быть `amd64` |
