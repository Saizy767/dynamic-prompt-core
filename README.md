# dynamic-prompt-core

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
а классификатор выбирает финальную классификацию.

```
Classifier.classify(text, rules) → Classification
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
    ports/           — интерфейсы (inbound/outbound: Classifier, LLMClient, PromptRepository, ...)
    use_cases/       — run_cycle, run_baseline, analyze_theses, compose_prompt,
                       select_candidates, compare_versions, refine_theses, observe_cycle,
                       evaluate_stop_criteria
    services/        — metrics, clustering (чистая логика)
    schemas/         — DTO и схемы обмена между use cases
  infrastructure/    — конкретные реализации портов (LLMLogitClassifier, AsyncTask, PromptStore)
  interfaces/        — CLI и сервер (composition root)
```

Зависимости направлены внутрь: `domain` не зависит ни от кого, `application`
зависит от портов, `infrastructure` реализует порты, `interfaces` собирает
всё вместе в composition root.

### Candidate scoring

Классификация использует архитектуру candidate scoring, а не генеративный
вывод. Инфраструктура оценивает кандидатов через logit-скоринг:

```
ClassifierPromptBuilder → BatchTokenizerAdapter → BatchedCausalLanguageModel
    → LogitScorer → Judgment[]
```

Приложение передаёт текст и `ClassificationRules` в `Classifier.classify()`,
который возвращает `Classification` (выбранный кандидат + полный набор
`Judgment`). Кандидат с наивысшей оценкой побеждает; ties разрешаются по
порядку входа.

#### Backend selection

Logit-scoring использует единственный backend — **`llama_server`**, выбираемый
в `[model]` секции `config.toml`:

- **`llama_server`** — HTTP-доступ к `llama-server` (Metal GPU acceleration), без
  опциональных зависимостей — используется только core-пакет `aiohttp`. Подключается
  к уже запущенному серверу (например, стартованному секцией `[llm]`).

```toml
[model]
backend = "llama_server"
model_path = "model/qwen2.5-3b-instruct-q4_k_m.gguf"

# [model.llama_server]
# endpoint = "http://127.0.0.1:8080"
# timeout = 30
```

См. [docs/gguf-backend.md](docs/gguf-backend.md) для подробностей о GGUF backend.

## Цикл оптимизации

Центральный use case — `run_cycle` в `application/use_cases/run_cycle/`. Он
объединяет все компоненты Stage 0–4 в работающий end-to-end цикл:

1. **Прогон активной версии** промпта на dev через `stage1-baseline-runner`.
2. **Подсчёт метрик** через `stage1-metrics`.
3. **Обсбор тезисов и кластеров** через `stage1-thesis-analyzer`.
4. **Отбор кандидатов** через `stage2-rule-candidate-selector`.
5. **Композиция новой версии** через `stage3-prompt-composer`.
6. **Прогон новой версии** на dev.
7. **Сравнение и решение** (accept/rollback) через `stage2-version-comparator`.
8. **Обновление активной версии**; при rollback накопление останавливается,
   оставшиеся кандидаты сохраняются в очереди.
9. **Запись per-cycle report** и state dump.

Накопление правил жадное: при accept цикл сразу пробует следующего кандидата
из очереди; первый rollback останавливает накопление (оставшиеся кандидаты
сохраняются в очереди до следующего круга). Финализированная версия
активируется в prompt store. Если за круг не принят ни один кандидат, цикл
останавливается с причиной `no_improvement`. Полный список причин остановки:
`max_cycles_reached`, `no_improvement`, `plateau_detected`,
`unrecoverable_error`.

### Запуск

```bash
# Запуск цикла (max_cycles из config; --cycles переопределяет)
python -m dynamic_prompt_core.interfaces.cli.main cycle

# Указать количество кругов и split
python -m dynamic_prompt_core.interfaces.cli.main cycle --cycles 3 --split dev

# Возобновить с сохранённого состояния
python -m dynamic_prompt_core.interfaces.cli.main cycle --resume data/results/state_cycle/state_*.json

# Teacher refinement (refine theses)
python -m dynamic_prompt_core.interfaces.cli.main refine --results data/results/results_*.jsonl

# Observer: сводный отчёт по завершённому циклу (markdown/json)
python -m dynamic_prompt_core.interfaces.cli.observer --run-id latest
```

### Конфигурация

Основные параметры в `[cycle_orchestrator]` секции `config.toml`:

| Параметр | По умолчанию | Описание |
|---|---|---|
| `max_cycles` | 5 | Максимум кругов |
| `duration` | 0 | Количество строк датасета на круг (см. ниже) |
| `dev_split` | `dev` | Split для активного прогона и решения accept/rollback |
| `holdout_split` | `holdout` | Split для holdout-прогона (только логируется) |
| `prompt_start` | `0` | Стартовая версия: `0` (baseline) или `best` (новейшая из store) |
| `use_teacher_refinement` | false | Включить teacher refinement между baseline и thesis-analysis |
| `dump_state_after_each_cycle` | true | State dump после каждого круга |
| `output_dir` | `data/results` | Каталог артефактов цикла |

Решение accept/rollback настраивается отдельно в секции `[version_comparator]`:
`decision_metric` (по умолчанию `macro_f1`) и `tie_breaker_metric`
(по умолчанию `minority_f1`). Дополнительные критерии ранней остановки
(например, `plateau_detected`) — в опциональной секции `[stop_criteria]`.

#### Параметр `duration`

Контролирует сколько строк датасета обрабатывается в каждом круге:

```toml
# Все строки (по умолчанию)
duration = 0

# 100 строк в каждом круге, всего max_cycles кругов
duration = 100

# 3 круга (переопределяет max_cycles): 50, 100, 200 строк
duration = [50, 100, 200]
```

При указании массива `max_cycles` автоматически устанавливается равным
`len(duration)`.

### Артефакты

После каждого круга записываются:

- `report_cycle/report_{run_id}_cycle{N}_{ts}.json` — per-cycle отчёт (метрики
  dev+holdout, решение, изменившиеся предсказания)
- `state_cycle/state_{run_id}_{ts}.json` — state dump для возобновления
- `report_cycle/summary_{run_id}_{ts}.json` — финальный отчёт при остановке
- `log_cycle/cycle_log_{run_id}_{ts}.jsonl` — append-only лог событий
- `manifest_cycle/manifest_{run_id}_{prompt_version}_{split}.json` — метаданные
  запуска (backend, model checksum, inference settings) для проверки identity при resume

## Компоненты

### Stage 1: Baseline

- **run_baseline** — прогон датасета через модель: thesis extraction через
  `extract_theses_detailed` + классификация через candidate scoring
  (`Classifier`), с чекпоинтами и resume.
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

### Вспомогательные компоненты

- **cycle-observer** — read-only компонент: загружает артефакты завершённого
  цикла и рендерит сводный отчёт (markdown/json). CLI: `interfaces.cli.observer`.
  Настраивается в `[cycle_observer]`.
- **evaluate-stop-criteria** — опциональные критерии ранней остановки цикла
  (например, `plateau_detected`). Настраивается в `[stop_criteria]`.

## Требования

- Python 3.11+
- aiohttp (HTTP-клиент; также используется `llama_server` backend без доп. зависимостей)
- pydantic v2
- numpy, scikit-learn, sentence-transformers (эмбеддинги и кластеризация тезисов)
- pymorphy3 / spaCy (нормализация тезисов: `ru` / `en`)
- pyarrow
- доступ к llama.cpp или vLLM серверу по OpenAI-совместимому API (thesis extraction)

## Статические проверки

```bash
lint-imports   # проверка слоёв и отсутствия циклических импортов
mypy src/      # типизация
ruff check     # линтер
ruff format    # форматирование
pytest         # тесты (unit + integration)
```
