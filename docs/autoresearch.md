# Autoresearch — ночной цикл улучшения score

Система, которая сама правит код, сама прогоняет eval, сама решает оставлять изменение или откатить. Цель — пока спишь, 20–40 итераций с ограниченным budget'ом (15 мин/итерацию) двигают `score = 0.8*recall + 0.2*nDCG` вверх.

Референсы:
- [karpathy/autoresearch](https://github.com/karpathy/autoresearch) — три файла (`prepare.py` / `train.py` / `program.md`), агент правит **только `train.py`**, 5-мин бюджет на эксперимент, ~12 итераций/час на одной H100, метрика `val_bpb`
- [WecoAI/awesome-autoresearch](https://github.com/WecoAI/awesome-autoresearch) — каталог агент-систем + паттерн "agent-digivolve-harness": persistent state, explicit eval package, bounded mutations per iteration

Оба сходятся на одном принципе: **keep-or-revert loop по одной числовой метрике**. Агент делает одну правку → eval → если метрика ≥ baseline'у — коммит, иначе `git checkout -- <file>`. Повторить.

---

## Предусловие (обязательное)

Без этого autoresearch не запустится — eval пуст.

**Локальный Q&A датасет** `eval/dataset.jsonl` на 20–30 примерах. Формат:

```jsonl
{"id": "q1", "question": {"text": "..."}, "answer": {"message_ids": ["3888...", "3999..."]}}
```

Как собрать: открыть `data/Go Nova.json`, найти тематический кусок переписки, сформулировать вопрос по смыслу, выписать 2–5 `id` сообщений, которые реально отвечают. Держать **фиксированным** во время ночного прогона — иначе метрика несравнима между итерациями.

Без датасета `eval/run.py` молчит, а значит нечего оптимизировать. Это критический блокер.

---

## Scope — что можно менять

Копируем паттерн Карпатого: **один editable файл за раз**, всё остальное заморожено.

| Режим | Editable файл | Что правим |
|-------|---------------|-----------|
| `chunking` | `index/main.py` | `UPPER_CHARS`, `LOWER_CHARS`, `OVERLAP_MESSAGES`, `TIME_GAP_SECONDS`, логика `build_chunks` |
| `retrieval` | `search/main.py` | `DENSE_PREFETCH_K`, `SPARSE_PREFETCH_K`, `RETRIEVE_K`, fusion (RRF → weighted), фильтры |
| `rerank` | `search/main.py` | `RERANK_LIMIT`, score mixing `α*rerank + (1-α)*retrieval`, diversity |
| `content` | `index/main.py` | `render_message`, разделение `dense_content` vs `sparse_content` |

**Один режим на прогон.** Смешивать — шум в ablation.

---

## Три схемы (от простой к сложной)

### A. Grid sweep (без LLM, детерминистично)

Перебор заранее заданной сетки параметров. Подходит для первой ночи — даёт честный baseline и ablation-таблицу.

```python
# autoresearch/grid.py — pseudocode
GRID = {
    "UPPER_CHARS": [800, 1200, 1600, 2000, 2400],
    "LOWER_CHARS": [200, 400, 600],
    "OVERLAP_MESSAGES": [0, 1, 2, 3],
    "TIME_GAP_SECONDS": [1800, 3600, 7200],
}
for config in product(GRID):
    patch_env(config)                 # пишем в index/main.py или .env
    docker compose restart index
    python eval/ingest.py
    result = python eval/run.py --dataset eval/dataset.jsonl
    log_to_jsonl(config, result)
    if result.score <= best: revert()
```

Плюсы: воспроизводимо, без рисков генерации мусорного кода. Минусы: только параметры, не новая логика.

### B. AI-driven (LLM правит код)

Агент читает `program.md` (цели + правила) + `editable.py` + последние 5 строк `iterations.jsonl`, выдаёт patch, прогоняет eval, решает.

```
program.md            ← цели, метрика, ограничения, что уже пробовал
editable.py           ← единственный файл, который агент может менять
eval/run.py           ← фиксирован
iterations.jsonl      ← история (config_diff, score, decision, timestamp)
```

Одна итерация:
1. Агент читает `program.md` + `editable.py` + `iterations.jsonl` (хвост)
2. Предлагает **одну** правку (diff ≤ 30 строк), обновляет `editable.py`
3. Runner: `ingest` → `run.py` → читает `score`
4. Если `score > best - ε`: `git commit -m "autoresearch: <summary> score=X"`
5. Иначе: `git checkout -- editable.py` (откат)
6. Дописать строку в `iterations.jsonl`, перейти к п.1

Плюсы: может придумать новую логику (query expansion, новый фильтр). Минусы: рейт-лимит API, риск закомитить сломанный код, надо guardrails.

### C. Гибрид (рекомендуемый для первой ночи)

Первые 2 часа — grid sweep по chunking (схема A, ~40 конфигов). Следующие N часов — AI-driven (схема B), агент стартует с лучшей конфигурации из A и предлагает нелинейные правки. Grid задаёт baseline, LLM — открытые ходы.

---

## Орchestрат­ор — одна итерация

Все три схемы делят один runner. Псевдокод:

```python
# autoresearch/runner.py
def iterate(mutator, budget_sec=900):
    baseline_score = read_best()
    git_stash_mark = git_current_sha()

    mutator.propose()                    # правит editable file
    try:
        with timeout(budget_sec):
            restart_index()              # docker compose restart index
            run_ingest()                 # eval/ingest.py
            score = run_eval()           # eval/run.py --dataset ...
    except (Timeout, BuildError, EvalError) as e:
        git_checkout(git_stash_mark)
        log("reverted", reason=str(e))
        return

    if score > baseline_score - EPSILON:
        git_add_commit(f"autoresearch: {mutator.summary()} score={score:.4f}")
        update_best(score)
        log("kept", score=score)
    else:
        git_checkout(git_stash_mark)
        log("reverted", score=score, baseline=baseline_score)
```

Запускать в цикле `while True: iterate(mutator)`. Ночью — 30–40 итераций.

---

## Guardrails (без них агент всё сломает)

Эти грабли известны — ставить сразу, не "потом".

**1. Hard timeout 15 мин/итерацию.** Если ingest+eval не уложились — `SIGKILL`, revert. Иначе один зависший прогон съест всю ночь.

**2. Syntax/import check до eval.** `python -c "import editable"` перед запуском. Битый код — мгновенный revert, eval не запускаем.

**3. Здоровая граница метрики.** `EPSILON = 0.002` — изменения внутри шума не коммитим. Иначе агент уйдёт в drift по случайности.

**4. Rate limit внешних API.** Dense + rerank API проверяющей системы. 20 итераций × N вопросов × embedding per question = легко превысить квоту. Кэш dense-вектора для одинакового `dense_content` — обязательно. Cache key: `hash(dense_content)`.

**5. Коммиты в отдельную ветку.** `autoresearch/run-YYYY-MM-DD`, не в `main` и не в `agent2`. Утром — ручной review и cherry-pick того, что выжило.

**6. Лог в JSONL.** `autoresearch/iterations.jsonl` — одна строка на итерацию. Поля: `ts`, `iter`, `mode` (chunking/retrieval/...), `diff_summary`, `config`, `score`, `recall`, `ndcg`, `decision`, `baseline`, `duration_sec`, `error`. Утром строится график score(iter) — видно плато/рост/регресс.

**7. Стоп по плато.** Если 10 подряд итераций без улучшения — перейти на другой режим (`chunking` → `retrieval`) или остановиться. Экономит API-квоту.

**8. Запрет править eval.** Агент может только editable-файл. `eval/*`, `data/*`, `docker-compose.yml` — readonly. Иначе агент "улучшит" метрику, подправив датасет.

---

## Структура репо

```
autoresearch/
  program.md           ← цели, метрика, что пробовали, что нельзя
  runner.py            ← orchestrator loop (keep/revert)
  mutators/
    grid.py            ← схема A
    llm.py             ← схема B (вызывает claude/openai)
  iterations.jsonl     ← история
  best.json            ← {score, config, sha} — текущий лучший
  report.py            ← читает jsonl, печатает таблицу + график
```

`program.md` — это "инструкция для агента". Что в неё класть:

```markdown
# Цель
Максимизировать score = 0.8*recall@50 + 0.2*nDCG@50 на eval/dataset.jsonl.

# Что можно менять
Только index/main.py. Только функции build_chunks, render_message, keep_message
и константы UPPER_CHARS, LOWER_CHARS, TIME_GAP_SECONDS, OVERLAP_MESSAGES.

# Что нельзя
- Менять API (IndexAPIItem, IndexAPIResponse)
- Ломать контракт: message_ids должны покрывать new_messages
- Добавлять внешние зависимости
- Править eval/ или data/

# Текущий baseline
UPPER=1600 LOWER=400 OVERLAP=2 GAP=3600 → score=0.XXXX (обновляется)

# Что уже пробовали (из iterations.jsonl)
<хвост последних 10 итераций>

# Идеи, которые стоит попробовать
- prefix message.sender_id / timestamp в text для sparse
- dense_content = только message.text без file_snippets
- разделить thread на отдельный чанк при переключении темы (heuristic)
```

---

## Скрытые риски

**Рейт-лимит dense/rerank API — самый вероятный провал.** 30 итераций × 30 вопросов × (1 dense + 1 rerank) ≈ 1800 запросов к rerank. Плюс ingest: 30 × ~50 чанков × 1 dense = 1500. Итого ~3300 запросов за ночь. Проверить квоту **до** запуска. Кэш dense по hash(dense_content) даёт ~20x экономию при чанкинге (тот же чанк не переиндексируется).

**Overfit на 20 вопросов.** При маленьком датасете агент может подогнаться под конкретные id. Митигация: датасет ≥ 20 вопросов, финальную метрику мерить на проверяющей системе, относиться к ночному score как к **relative signal**, не как к абсолюту.

**Шум в git history.** 30 коммитов вида "autoresearch: UPPER=1800 score=0.634" замусорят лог. Поэтому — отдельная ветка + утренний squash или cherry-pick 2–3 выживших изменений в `agent2`.

**Docker restart — 20–40 сек.** Это внутри бюджета 15 мин, но если уменьшать до 5 мин — restart съест треть времени. Альтернатива: не рестартить index, а перезагрузить модуль через `importlib.reload` — хрупко, лучше не надо.

**Evaluation drift.** Если меняется датасет, rerank модель, dense API — все прошлые score несравнимы. Зафиксировать snapshot всего (commit sha + model version) в каждую строку `iterations.jsonl`.

---

## План первой ночи

**Прямо сейчас (до запуска):**
1. Собрать `eval/dataset.jsonl`, 20 вопросов минимум — 2–3 часа ручной работы (блокер)
2. Прогнать текущий baseline (UPPER=1600, OVERLAP=2) → зафиксировать как `best.json`
3. Написать `autoresearch/runner.py` + `mutators/grid.py` (схема A только) — ~1 день
4. Добавить кэш dense embeddings по `hash(dense_content)` — без него квоту сожжём

**Ночь 1 (схема A, ~8 часов):**
- Grid sweep chunking: 5×3×4×3 = 180 конфигов → с ингестом по 2–3 мин ≈ не все, ~100 штук пройдут
- Утром: таблица, топ-5 конфигов, ручной review

**Ночь 2 (схема C):**
- Старт с лучшей конфигурации из ночи 1
- Схема B 15–20 итераций по `search/main.py` (retrieval + rerank)

**Что НЕ пытаться в первую ночь:** схема B сразу, оптимизация всего pipeline одновременно, ML-модели через pip install. Сначала grid — это 70% ценности за 10% сложности.

---

## Связь с остальной документацией

- Метрика и её компоненты: `docs/eval-harness.md`
- Что считается "улучшением" каждого блока: `docs/ml-roadmap.md` (Этапы 1–12)
- Почему текущая конфигурация именно такая: `docs/ml_description.md` (секция Chunking + sweep)
- Что править в каком файле: см. таблицу Scope выше

Autoresearch не заменяет ручную работу по Этапам 2–4 (dense/sparse/rerank boundaries, query expansion) — он дожимает параметры внутри уже реализованного pipeline. Сначала имплементация этапа → потом ночь автоподбора по этому этапу.
