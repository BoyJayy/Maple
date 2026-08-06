# Сравнение `main`, `fable` и финальной версии поиска

Первичный прогон: 11 июля 2026 года. Финальная валидация: 15 июля 2026 года.

## Вывод

`fable` — явный победитель относительно `main` и правильная основа финальной
версии. На основном `Dataset_main`
он поднимает Hit@1 с `4.38%` до `53.75%`, а MRR@50 — с `0.3867` до
`0.7009`. Парное сравнение: `fable` ставит правильный ответ выше в 116 из 160
запросов, `main` — только в 16; ещё 28 запросов дают ничью.

Результат подтверждается на менее синтетическом `Dataset_dialogues`: Recall@50
растёт с `0.6275` до `0.8283`, MRR@50 — с `0.2864` до `0.4384`.

По latency дефолтный `fable` примерно равен `main`, но `Dataset_main` содержит
ошибочные даты. Из-за них `fable` делает пустой time-filtered retrieval, а затем
повторяет retrieval без фильтра для каждого запроса. После отключения только
этого фильтра `fable` одновременно сохраняет качество и становится быстрее:
при concurrency=1 p50 `28.0 ms` против `30.0 ms`, при concurrency=8 — `69.1`
против `51.0 QPS`.

Финальный профиль сохраняет DBSF и русский BM25 из `fable`, выключает
point-level rescoring, оставляет message-level scoring и использует hard
time-filter с кэшируемой проверкой временных границ коллекции. Он улучшил
Hit@1 до `55.63%`, MRR@50 до `0.7098`, а на `Dataset_dialogues` — Recall@50
до `0.8396`. При concurrency=8 throughput вырос до `83.9 QPS`.

### Финальный профиль

| Настройка | Значение |
|---|---|
| Hybrid retrieval | DBSF, MiniLM-384 + русский BM25 |
| Query branches | до 3 dense + 3 sparse, prefetch 40, retrieve 60 |
| Point rescore | выключен (`POINT_RESCORE_ENABLED=0`) |
| Message assembly scoring | включён |
| Time filter | `hard`, bounds guard + zero-hit fallback |
| Cross-encoder reranker | выключен |

Мягкий time-fusion также был измерен, но отклонён: на диалогах он снизил
Recall@50 с `0.8396` до `0.8056`, nDCG@50 с `0.5016` до `0.4744` и оказался
чуть медленнее. Поэтому `soft` сохранён только как экспериментальный режим.

| Метрика | `main` | `fable` | final |
|---|---:|---:|---:|
| Dataset_main Hit@1 | 0.0438 | 0.5375 | **0.5563** |
| Dataset_main Hit@10 | 0.9438 | 0.9813 | **0.9875** |
| Dataset_main nDCG@50 | 0.5356 | 0.7735 | **0.7799** |
| Dataset_main MRR@50 | 0.3867 | 0.7009 | **0.7098** |
| Dataset_dialogues Recall@50 | 0.6275 | 0.8283 | **0.8396** |
| Dataset_dialogues nDCG@50 | 0.3455 | 0.4954 | **0.5016** |
| Dataset_dialogues MRR@50 | 0.2864 | 0.4384 | **0.4446** |

Warm latency на `Dataset_main`:

| Вариант | Concurrency | p50, ms | p95, ms | QPS |
|---|---:|---:|---:|---:|
| `main` | 1 | 30.0 | 72.7 | 27.8 |
| `fable` | 1 | 33.1 | 75.4 | 24.6 |
| final | 1 | **27.0** | **68.2** | **30.8** |
| `main` | 8 | 150.0 | 219.6 | 51.0 |
| `fable` | 8 | 145.6 | 196.9 | 53.6 |
| final | 8 | **91.2** | **137.0** | **83.9** |

## Что сравнивалось

| Вариант | Commit | Конфигурация |
|---|---|---|
| `main` | `99465f905b5e985be432c0ed7e34025c2dc0a22e` | shipped defaults |
| `fable` | `602421f728f427d21f6921eaef4ac5ea8ccdf57e` | shipped defaults, reranker выключен |
| `fable`, time-filter off | тот же commit | только `TIME_FILTER_ENABLED=0` |
| final | working tree поверх `fable` | no point-rescore, hard time-filter + bounds guard |

Обе ветки используют MiniLM 384, Qdrant/DBSF, prefetch 40+40, retrieve 60 и
максимум 3 dense + 3 sparse query-варианта. Опциональный Jina reranker в этих
прогонах не включался.

Основные отличия `fable`: русский BM25, RU/EN stemming и стоп-слова при
rescoring, другой приоритет структурированных терминов, явные `message_blocks`,
integer time metadata и time-filter. Кроме того, `fable` убирает raw fused
Qdrant score из финального rescoring. Результаты ниже показывают итоговый эффект
всего набора изменений; причинный вклад каждого изменения потребует отдельного
ablation-теста.

## Методика

- Отдельный detached worktree и чистый Qdrant volume для каждой ветки.
- Каждый корпус заново индексировался кодом соответствующей ветки.
- Проверено совпадение SHA-256 корпуса и вопросов между сравниваемыми прогонами.
- Production endpoint `/search`, один и тот же внешний benchmark runner.
- Перед измерением: три model warmup-запроса и один полный проход датасета,
  исключённый из статистики.
- Основной latency-тест: 5 перемешанных фиксированным seed проходов, всего 800
  запросов на каждом уровне concurrency (`1` и `8`).
- Quality считается по одному детерминированному проходу; повторные проходы дали
  те же rankings.
- Для delta quality: paired cluster bootstrap, 10 000 resamples. В
  `Dataset_main` использованы 40 тематических кластеров по 4 вопроса.
- Среда: Apple M5 Pro, 18 CPU, 48 GiB RAM; Docker VM — 18 CPU и 7.75 GiB RAM;
  Docker 29.4.3, Compose 5.1.3.

### Данные

`Dataset_main` идентичен на обеих ветках:

- 800 сообщений, 40 тредов, 162 индексных chunk-а;
- 160 positive-вопросов, один уникальный relevant message на вопрос;
- поэтому Recall@K на этом наборе равен HitRate@K;
- corpus SHA-256:
  `3beab90c01f80dc8b81b6c37180e20403fe017f0eaedf0e2a116a43ebbfe87fa`;
- questions SHA-256:
  `ef680b9dbc9f7f5d426b5a4a424b1aeb542389a157adb07b095d2a61c41ae5dd`.

## Основной результат: `Dataset_main`

### Quality

| Метрика | `main` | `fable` | Delta `fable-main` | 95% CI delta |
|---|---:|---:|---:|---:|
| Hit@1 | 0.0438 | 0.5375 | +0.4938 | [+0.4188, +0.5687] |
| Hit@3 | 0.6688 | 0.8188 | +0.1500 | [+0.0750, +0.2250] |
| Hit@5 | 0.8250 | 0.9250 | +0.1000 | [+0.0500, +0.1500] |
| Hit@10 | 0.9438 | 0.9813 | +0.0375 | [+0.0125, +0.0688] |
| Hit@50 | 1.0000 | 1.0000 | 0.0000 | [0.0000, 0.0000] |
| nDCG@50 | 0.5356 | 0.7735 | +0.2380 | [+0.2012, +0.2743] |
| MRR@50 | 0.3867 | 0.7009 | +0.3143 | [+0.2658, +0.3613] |

Дополнительно:

- wins / ties / losses для `fable`: `116 / 28 / 16`;
- median rank правильного сообщения: `2` у `main`, `1` у `fable`;
- Hit@50 насыщен у обеих веток и не различает качество порядка, поэтому главные
  показатели здесь — Hit@1, MRR и nDCG.

### Warm latency

| Вариант | Concurrency | p50, ms | p95, ms | p99, ms | QPS |
|---|---:|---:|---:|---:|---:|
| `main` | 1 | 30.0 | 72.7 | 80.8 | 27.8 |
| `fable`, default | 1 | 33.1 | 75.4 | 86.2 | 24.6 |
| `fable`, time-filter off | 1 | 28.0 | 68.0 | 74.5 | 29.0 |
| `main` | 8 | 150.0 | 219.6 | 270.3 | 51.0 |
| `fable`, default | 8 | 145.6 | 196.9 | 231.4 | 53.6 |
| `fable`, time-filter off | 8 | 112.7 | 154.8 | 178.5 | 69.1 |

Ошибок и HTTP-retry во всех основных сериях: `0`.

При concurrency=1 дефолтный `fable` имеет p50 на 10.1% выше `main`. Без
невалидного time-filter он на 6.9% быстрее по p50. При concurrency=8 вариант без
time-filter даёт на 35.5% больше QPS и на 29.5% ниже p95, чем `main`.

Индексирование `Dataset_main` заняло `2.97 s` на `main` и `2.85 s` на `fable`;
обе ветки создали 162 chunk-а. Это один прогон, поэтому разницу в 0.12 s не
следует считать значимой.

Одноразовый memory snapshot после прогрева основного теста:

| Контейнер | `main` | `fable` |
|---|---:|---:|
| search RSS | 719 MiB | 728 MiB |
| index RSS | 142 MiB | 175 MiB |
| Qdrant RSS | 380 MiB | 406 MiB |

Это не peak-memory benchmark, а только ориентир по resident memory в момент
замера.

## Дефект дат в `Dataset_main`

Сообщения корпуса имеют timestamps с 1 по 11 мая 2024 года, тогда как
`date_range` всех 160 вопросов лежит в апреле 2026 года. Валидатор считает это
ошибкой для 160 из 160 вопросов.

У `fable` time-filter включён по умолчанию. Для каждого полного вопроса:

1. Qdrant retrieval с фильтром возвращает ноль points;
2. защитный fallback повторяет retrieval без фильтра;
3. итоговый ranking остаётся правильным, но latency включает лишний запрос.

В логах подтверждено 1760 fallback-событий на 1760 full-question запросов
основной серии. Поэтому в отчёте сохранены обе цифры: shipped-default как
реальное текущее поведение и `TIME_FILTER_ENABLED=0` как controlled ablation.
На корректном `Dataset_dialogues` таких fallback-событий не было.

## Проверка без обогащённых полей

Все вопросы `Dataset_main` содержат `search_text`, keywords, entities, variants
и даты. Чтобы проверить зависимость от этой разметки, был сделан отдельный
запуск, где API получал только `question.text`.

| Метрика | `main`, text-only | `fable`, text-only | Delta |
|---|---:|---:|---:|
| Hit@1 | 0.0063 | 0.0000 | -0.0063 |
| Hit@3 | 0.3063 | 0.3063 | 0.0000 |
| Hit@5 | 0.4250 | 0.4563 | +0.0313 |
| Hit@10 | 0.5750 | 0.6500 | +0.0750 |
| Hit@50 | 0.8563 | 0.8875 | +0.0313 |
| nDCG@50 | 0.3450 | 0.3658 | +0.0208 |
| MRR@50 | 0.2021 | 0.2156 | +0.0135 |

Wins / ties / losses: `63 / 47 / 50` в пользу `fable`. Для MRR@50 95% CI
delta равен `[-0.0118, +0.0415]`, то есть text-only улучшение порядка на этом
наборе статистически неубедительно. Большая победа на полном payload связана с
умением `fable` лучше использовать структурированные подсказки. Это часть
текущего API-контракта, но перед production-решением важно проверить долю таких
полей в реальном трафике.

## Дополнительная проверка: `Dataset_dialogues`

Этот набор менее насыщен синтетическими подсказками: 1995 сообщений, 298
chunk-ов, 132 positive и 12 negative-вопросов, 1–3 relevant messages.

| Метрика | `main` | `fable` | Delta | 95% CI delta |
|---|---:|---:|---:|---:|
| Recall@1 | 0.1780 | 0.2664 | +0.0884 | [+0.0379, +0.1402] |
| Recall@10 | 0.3801 | 0.5354 | +0.1553 | [+0.0947, +0.2197] |
| Recall@50 | 0.6275 | 0.8283 | +0.2008 | [+0.1237, +0.2790] |
| nDCG@50 | 0.3455 | 0.4954 | +0.1499 | [+0.1040, +0.1986] |
| MRR@50 | 0.2864 | 0.4384 | +0.1520 | [+0.0967, +0.2124] |

- wins / ties / losses: `70 / 47 / 15`;
- median best rank: `17.5` у `main`, `4` у `fable`;
- warm c=1 latency: p50 `22.1 -> 21.1 ms`, p95 `28.6 -> 27.9 ms`, QPS
  `43.6 -> 46.0`.

Recall по категориям:

| Категория | N | `main` R@10 | `fable` R@10 | `main` R@50 | `fable` R@50 |
|---|---:|---:|---:|---:|---:|
| date | 30 | 0.2500 | 0.5889 | 0.5500 | 1.0000 |
| exact | 30 | 0.9000 | 0.9000 | 0.9833 | 0.9833 |
| multihop | 12 | 0.3472 | 0.3333 | 0.5278 | 0.6944 |
| participant | 24 | 0.2500 | 0.6250 | 0.6250 | 0.8750 |
| semantic | 36 | 0.1528 | 0.1944 | 0.4306 | 0.5694 |

На всех 12 negative-вопросах обе ветки всё равно вернули кандидатов:
no-result rate `0%`, false-positive rate `100%`. Текущий API не умеет
воздерживаться от ответа; это отдельная задача, не преимущество одной ветки.

## Рекомендация и принятое решение

1. Брать `fable` как основу: выигрыш по качеству большой, статистически
   устойчивый и повторяется на втором датасете.
2. Исправить даты `Dataset_main` или отключать time-filter именно для этого
   benchmark fixture. Для production с корректными датами фильтр полезен.
3. Добавить быстрый guard: если диапазон вопроса заведомо не пересекает
   временной диапазон коллекции, не делать сначала гарантированно пустой
   filtered retrieval.
4. Отдельно проверить реальные payload-ы без богатых `keywords/entities`:
   text-only преимущество намного меньше.
5. Добавить abstention/score threshold и метрику качества на negative-запросах.
6. Перед merge провести ablation ключевых изменений `fable` — прежде всего
   нового rescoring без raw fusion score, русского BM25 и time-filter — чтобы
   понять, какие части дают выигрыш, а какие можно упростить.

Итог: пункты 1, 3 и ключевой ablation из пункта 6 выполнены. Финальная версия
основана на `fable`, bounds guard устраняет лишний запрос на заведомо
непересекающихся датах, а point-level rescore отключён по результатам обоих
наборов. Исправление самих дат fixture и abstention для negative-запросов
остаются отдельными задачами, не блокирующими финальный retrieval-профиль.

## Артефакты и воспроизведение

- Runner: [`benchmark_search.py`](benchmark_search.py)
- Paired comparison: [`compare_results.py`](compare_results.py)
- Сырые результаты: [`results/`](results/)

Пример после запуска и чистой индексации нужной ветки:

```bash
python3 -u benchmarks/benchmark_search.py \
  --url http://localhost:8002/search \
  --dataset data/Dataset_main_questions.jsonl \
  --corpus data/Dataset_main.json \
  --index-points 162 \
  --label branch-default \
  --commit COMMIT_SHA \
  --output benchmarks/results/branch-default.json \
  --rounds 5 \
  --concurrencies 1,8
```

Парное сравнение:

```bash
python3 benchmarks/compare_results.py \
  --left benchmarks/results/main-default.json \
  --right benchmarks/results/fable-default.json \
  --dataset data/Dataset_main_questions.jsonl \
  --output benchmarks/results/compare-main-vs-fable.json
```

### Ограничения

- Основной набор синтетический и сильно обогащён полями поиска.
- Latency снят на одной машине и в одном порядке запусков; небольшие различия
  около нескольких процентов следует считать шумом, а не гарантией.
- Docker использует Qdrant server 1.14.1 при `qdrant-client` 1.17.1. Клиент
  выдаёт compatibility warning; обе ветки находятся в одинаковых условиях.
- Memory — одноразовый RSS snapshot, не peak/allocated benchmark.
- Reranker не включался в эту серию `main`/`fable`/final, поскольку в shipped
  defaults он выключен; более ранний отдельный эксперимент приведён в
  `CHANGELOG.md`.
