# prompt-loop

Исследовательский пайплайн для итеративной оптимизации системного промпта
бинарного классификатора на базе маленькой языковой модели (класс Tiny, до 3B
параметров).

## Назначение

Цель — не ручное написание промпта, а построение цикла, в котором системный
промпт эволюционирует на основе анализа ошибок и reasoning модели. Каждый шаг
цикла логируется, изменения можно откатить.

## Концепция

Системный промпт рассматривается не как текст, а как контракт между
разработчиком и моделью. Он состоит из фиксированных семантических слоёв:

- роль
- задача
- правила
- контракт вывода
- fallback

Слои упорядочены по стабильности — от неизменяемых к динамическим. Это
сохраняет prefix-кэш на inference-сервере.

Модель работает при `temperature=0`. Классификация выполняется через
candidate scoring: модель оценивает каждый явный кандидат (класс) независимо,
а приложение выбирает финальную классификацию через политику.

```
CandidateScorer.score(text, candidates) → Judgment[]
    → ClassificationPolicy.classify(judgments) → Classification
```

Оценка кандидата — это mean candidate-token log-probability: ранжирующий
сигнал, а не вероятность или confidence. Модель не генерирует JSON-ответ
и не выбирает класс сама.

## Архитектура

Проект разделён на четыре слоя по принципу dependency inversion:

```
src/dynamic_prompt_core/
  domain/            — модели без сторонних зависимостей (Candidate, Judgment, Classification, Dataset, Record)
  application/       — use cases и порты
    ports/           — интерфейсы (CandidateScorer, LLMClient, PromptRepository, ...)
    use_cases/       — run_cycle, run_baseline, classify_input, analyze_theses, compose_prompt, ...
    services/        — ClassificationPolicy, metrics, clustering (чистая логика)
  infrastructure/    — конкретные реализации портов (LLMLogitCandidateScorer, AsyncTask, PromptStore)
  interfaces/        — CLI и сервер (composition root)
```

Зависимости направлены внутрь: `domain` не зависит ни от кого, `application`
зависит от портов, `infrastructure` реализует порты, `interfaces` собирает
всё вместе в composition root.

### Candidate scoring

Классификация использует архитектуру candidate scoring, а не генеративный
вывод. Инфраструктура оценивает кандидатов через logit-скоринг:

```
ScoringPromptBuilder → BatchTokenizerAdapter → BatchedCausalLanguageModel
    → LogitScorer → Judgment[]
```

Приложение передаёт набор кандидатов (`Candidate`) в `CandidateScorer.score()`,
получает `Judgment[]` и передаёт их в `ClassificationPolicy.classify()`, которая
возвращает `Classification`. `ArgmaxClassificationPolicy` выбирает кандидата с
наивысшей оценкой; ties разрешаются по порядку входа.

## Цикл оптимизации

Центральный use case — `run_cycle` в `application/use_cases/run_cycle/`. Он
объединяет все компоненты Stage 0–4 в работающий end-to-end цикл:

1. **Прогон активной версии** промпта на dev через `stage1-baseline-runner`.
2. **Подсчёт метрик** через `stage1-metrics`.
3. **ОбCбор тезисов и кластеров** через `stage1-thesis-analyzer`.
4. **Отбор кандидатов** через `stage2-rule-candidate-selector`.
5. **Композиция новой версии** через `stage3-prompt-composer`.
6. **Прогон новой версии** на dev.
7. **Сравнение и решение** (accept/rollback) через `stage2-version-comparator`.
8. **Обновление активной версии** или переход к следующему кандидату.
9. **Запись per-round report** и state dump.

При rollback цикл пробует следующего кандидата из очереди. При accept —
активирует новую версию в prompt store. Цикл останавливается по одной из
причин: `max_rounds_reached`, `max_rollbacks_reached`,
`candidate_queue_exhausted`, `unrecoverable_error`.

### Запуск

```bash
# Запуск цикла (5 кругов по умолчанию)
python -m dynamic_prompt_core.interfaces.cli.main cycle

# Указать количество кругов
python -m dynamic_prompt_core.interfaces.cli.main cycle --rounds 3

# Возобновить с сохранённого состояния
python -m dynamic_prompt_core.interfaces.cli.main cycle --resume data/results/state_*.json

# Teacher refinement (refine theses)
python -m dynamic_prompt_core.interfaces.cli.main refine --results data/results/results_*.jsonl
```

### Конфигурация

Основные параметры в `[cycle_orchestrator]` секции `config.toml`:

| Параметр | По умолчанию | Описание |
|---|---|---|
| `max_rounds` | 5 | Максимум кругов |
| `max_consecutive_rollbacks` | 2 | Лимит подряд идущих откатов |
| `decision_metric` | `macro_f1` | Метрика для решения accept/rollback |
| `tie_breaker_metric` | `minority_f1` | Тай-брейкер при равенстве |
| `duration` | 0 | Количество строк датасета на круг (см. ниже) |
| `use_teacher_refinement` | false | Включить teacher refinement |
| `stop_on_first_error` | false | Остановка при первой ошибке шага |
| `dump_state_after_each_round` | true | State dump после каждого круга |

#### Параметр `duration`

Контролирует сколько строк датасета обрабатывается в каждом круге:

```toml
# Все строки (по умолчанию)
duration = 0

# 100 строк в каждом круге, всего max_rounds кругов
duration = 100

# 3 круга (переопределяет max_rounds): 50, 100, 200 строк
duration = [50, 100, 200]
```

При указании массива `max_rounds` автоматически устанавливается равным
`len(duration)`.

### Артефакты

После каждого круга записываются:

- `report_{run_id}_round{N}_{ts}.json` — per-round отчёт (метрики dev+holdout,
  решение, изменившиеся предсказания)
- `state_{run_id}_{ts}.json` — state dump для возобновления
- `summary_{run_id}_{ts}.json` — финальный отчёт при остановке
- `cycle_log_{run_id}_{ts}.jsonl` — append-only лог событий

## Компоненты

### Stage 1: Baseline

- **run_baseline** — прогон датасета через модель: thesis extraction через
  `extract_theses_detailed` + классификация через candidate scoring
  (`CandidateScorer` → `ClassificationPolicy`), с чекпоинтами и resume.
- **metrics** — accuracy, precision/recall/F1, confusion matrix, minority-class
  F1, распределение предсказаний, bias detection.
- **thesis-analyzer** — сбор тезисов из reasoning, нормализация, эмбеддинги,
  кластеризация, подсчёт частот и precision по кластерам.

### Stage 2: Candidate selection и decision

- **rule-candidate-selector** — отбор кластеров-кандидатов по частоте и
  precision, исключение уже использованных (`in_prompt` flag).
- **version-comparator** — сравнение метрик двух версий, решение accept/rollback,
  подсчёт rollback counter, поиск изменившихся предсказаний.

### Stage 3: Composition и cycle

- **prompt-composer** — формулировка правил из тезисов кластера через LLM,
  проверка дисторции (cosine similarity с центроидом), enforcement лимитов,
  сборка новой версии промпта.
- **prompt-store** — хранение версий промпта, активация/архивация, lineage.
- **cycle-orchestrator** — оркестрация полного цикла (`run_cycle`).

### Stage 4: Teacher refinement

- **teacher-refinement** — refinement тезисов через большую модель (teacher):
  фильтрация шумных, переформулировка нестабильных, добавление пропущенных.
  Опционально включается через `use_teacher_refinement = true`.

## Требования

- Python 3.11+
- aiohttp
- pydantic v2
- numpy
- torch + transformers (candidate scoring via logit-scoring)
- доступ к llama.cpp или vLLM серверу по OpenAI-совместимому API (thesis extraction)

## Статические проверки

```bash
lint-imports   # проверка слоёв и отсутствия циклических импортов
mypy src/      # типизация
ruff check     # линтер
ruff format    # форматирование
pytest         # тесты (unit + integration)
```
