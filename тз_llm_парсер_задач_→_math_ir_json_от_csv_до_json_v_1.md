# ТЗ: LLM‑парсер математических задач → MathIR‑JSON (от CSV до JSON) v1.0

## 0) Цель и результат
**Цель:** из CSV с колонками `task` и `answer` получить для каждой строки валидный файл в формате **MathIR‑JSON**, описывающий задачу строго по спецификации (см. документ «MathIR‑JSON Spec & Parser v0.1» в Canvas — это источник истины по формату).

**Результат:** директория с JSON‑файлами (`<id>.json`), а также сводный `outputs.jsonl` и `run_report.json` с метриками и логами.

---

## 1) Вход/выход
**Вход:** CSV `tasks.csv` с колонками:
- `task` — текст задачи (может содержать LaTeX),
- `answer` — эталон (не используется в промпте, сохраняется в `meta.gold_answer` для последующей валидации).

**Выход:**
- `out_json/<id>.json` — один JSON на строку в формате MathIR‑JSON,
- `out_json/outputs.jsonl` — та же совокупность в JSONL,
- `out_json/run_report.json` — агрегированные метрики/логи.

**Идентификатор `<id>`:** если в CSV нет явного `id`, использовать 1‑based индекс строки с нулями: `task_000123`.

---

## 2) Ограничения и требования
- LLM **не решает** задачу, только парсит → JSON по схеме.
- Вывод **строго JSON** (без текста). Используем **grammar/JSON‑mode** провайдера + Pydantic/JSON Schema валидацию.
- При неполных данных — `unknown/ambiguous` и причина в `meta.notes`.
- Символьные выражения — в формате `expr_format` (по умолчанию LaTeX **без** `$`). Сохранять имена переменных.
- Не передавать в промпт поле `answer`.

---

## 3) Архитектура и компоненты
```
llm_parser/
  ├─ main.py                 # CLI вход
  ├─ config.yaml             # провайдер, модель, параметры, пути
  ├─ schema/mathir.schema.json
  ├─ prompts/
  │    ├─ system.txt         # системный промпт (из Canvas)
  │    └─ fewshot.jsonl      # 5–8 примеров: {input, output}
  ├─ src/
  │    ├─ io.py             # чтение CSV, запись JSON/JSONL, id
  │    ├─ prompt.py         # сборка промпта (system+few‑shot+user)
  │    ├─ client.py         # адаптеры LLM (OpenAI/Anthropic/Local)
  │    ├─ guard.py          # grammar/JSON‑mode, парсинг ответа, валидация
  │    ├─ retry.py          # delta‑feedback ретраи
  │    ├─ normalize.py      # пост‑нормализация MathIR (expr_format, defaults)
  │    ├─ metrics.py        # метрики/агрегация
  │    └─ logging.py        # структурные логи
  └─ tests/
       ├─ test_schema.py
       ├─ test_prompt.py
       └─ test_integration_small.py
```

---

## 4) Конфигурация (`config.yaml`)
```yaml
provider: openai            # openai|anthropic|local
model: gpt-*-*              # имя модели провайдера
json_mode: true             # включить строгий JSON‑вывод / функции
use_grammar: true           # если доступно (schema‑guided)
max_tokens: 1500
temperature: 0.2
top_p: 0.9
concurrency: 8              # число параллельных запросов
rate_limit_rps: 2           # глобальный лимит
retries: 2                  # кол-во авто‑ретраев
retry_hint_template: prompts/retry_hint.txt
system_prompt: prompts/system.txt
fewshot_path: prompts/fewshot.jsonl
schema_path: schema/mathir.schema.json
input_csv: data/tasks.csv
output_dir: out_json
log_level: INFO
seed: 17
```

`retry_hint.txt` содержит шаблон обратной связи об ошибках валидации:
```
Your previous JSON was invalid against the MathIR‑JSON schema.
Issues:
{{ERROR_LIST}}
Return ONLY corrected JSON. Do not change unrelated fields.
```

---

## 5) Поток обработки (dataflow)
1) **Load CSV** → нормализуем пробелы, сохраняем исходный текст как есть (LaTeX не трогаем).
2) **ID** → генерируем `<id>`.
3) **Prompt build** →
   - System: `prompts/system.txt` (из Canvas),
   - Few‑shot: 5–8 пар `{input: «сырой текст задачи», output: «валидный MathIR‑JSON»}` по нашим подтипам,
   - User: сырой текст из `task`.
4) **LLM call** → json‑mode/grammar; таймаут/ретраи/конкурентность.
5) **Parse & Validate** → Pydantic/JSON Schema. Если невалидно →
   - Сформировать список ошибок (`ERROR_LIST`),
   - Отправить **ровно тот же ввод** + `retry_hint` → повторить (до `retries`).
6) **Normalize** → проставить `expr_format` (если не задан), нормализовать поля, добавить `meta.id`, `meta.lang="ru"`, `meta.gold_answer` (из CSV **но не в промпт**), `meta.source="csv"`.
7) **Write** → `<output_dir>/<id>.json`, плюс строчку в `outputs.jsonl`.
8) **Metrics** → накапливаем: valid_rate, retries_used, tokens, латентность.
9) **(Опционально) Quick downstream check** → кидаем JSON в твой MathIR‑парсер (без решения сложных узлов) и считаем `downstream_parsable_rate`.
10) **Report** → `run_report.json` с метриками и сводкой ошибок.

---

## 6) Контракты и интерфейсы
### 6.1. Вход (task)
- Не модифицируем семантику; разрешена только трим/нормализация пробелов.
- Никаких pre‑rule‑based замен LaTeX.

### 6.2. Выход (MathIR‑JSON)
- Строго по схеме из Canvas («MathIR‑JSON Spec & Parser v0.1»).
- Обязательные поля: `expr_format`, `targets[*]`.
- Разрешённые типы `targets[*].type`: `integral_def|limit|sum|solve_for|inequalities|solve_for_matrix|probability|value`.

### 6.3. Ошибки
- Если после ретраев JSON невалиден: сохраняем файл `<id>.error.json` вида:
```json
{"status":"invalid","errors":["<schema path>: <msg>", ...], "raw": "<llm_output_truncated>"}
```
- В `run_report.json` фиксируем счётчики по кодам ошибок.

---

## 7) Поведение LLM (промптинг)
- System‑промпт — как в документе «Системный промпт для LLM‑парсера» (Canvas). Никаких инструкций на решение.
- Few‑shot — 6 примеров: интеграл, предел последовательности, матричное уравнение, вероятность (Bernoulli), геометрия (поворот+касательная+квадрант), unknown‑кейс.
- Generation: `temperature=0.2`, `top_p=0.9`, `presence/frequency_penalty=0`.
- JSON‑mode/grammar‑constraints включены.

---

## 8) Метрики и критерии готовности
- **JSON Validity Rate ≥ 95%** (валидных без ручной правки).
- **Avg retries per sample ≤ 0.5**.
- **Mean latency ≤ 2.5× median** (нет длинного хвоста таймаутов).
- **(Опц.) Downstream parsable ≥ 85%** на `v0.1` узлах.
- Все файлы присутствуют, имена без «дыр» в диапазоне id.

---

## 9) CLI и примеры запуска
```bash
# Базовый прогон
python -m llm_parser.main \
  --config config.yaml \
  --input data/tasks.csv \
  --output out_json

# Переопределение модели/провайдера
python -m llm_parser.main \
  --config config.yaml \
  --model gpt-4o-mini \
  --provider openai

# Локальная модель (пример)
python -m llm_parser.main \
  --provider local --base-url http://localhost:8000/v1 --model qwen2.5-7b-instruct
```

---

## 10) Псевдокод ключевых модулей
### 10.1. `main.py`
```python
args = parse_args()
cfg = load_yaml(args.config)
df = read_csv(args.input)
init_logger(cfg)
client = LLMClient(cfg)
validator = JSONValidator(cfg.schema_path)
writer = Writer(args.output)
metrics = Metrics()

for row_id, row in enumerate(df.itertuples(index=False), start=1):
    id_ = make_id(row_id)
    prompt = build_prompt(system=load_file(cfg.system_prompt),
                          fewshot=load_fewshot(cfg.fewshot_path),
                          user=row.task)
    resp, usage, latency = client.generate_json(prompt)
    ok, data, errs = validator.validate(resp)
    retry_left = cfg.retries
    while not ok and retry_left > 0:
        hint = make_retry_hint(errs)
        resp, usage2, latency2 = client.generate_json(prompt, retry_hint=hint)
        ok, data, errs = validator.validate(resp)
        metrics.add_retry()
        retry_left -= 1
    if ok:
        data = normalize(data, id_, gold=row.answer)
        writer.write_json(id_, data)
        writer.append_jsonl(data)
        metrics.add_success(usage, latency)
    else:
        writer.write_error(id_, resp, errs)
        metrics.add_failure(usage, latency, errs)

writer.write_report(metrics.to_report())
```

### 10.2. `client.generate_json`
- Включает JSON‑mode/grammar/schema у провайдера.
- Возвращает «сырой текст ответа» (который должен быть JSON), usage (токены), latency.

### 10.3. `guard.validate`
- Использует `jsonschema`/`fastjsonschema` или Pydantic‑модели из Canvas.
- Возвращает `(ok: bool, data: dict|None, errs: List[str])`.

---

## 11) Тестирование
- `tests/test_schema.py` — валидация валидных/невалидных примеров (включая unknown‑кейс).
- `tests/test_prompt.py` — проверка сборки промпта (нет утечек `answer`).
- `tests/test_integration_small.py` — прогон 10 задач с заглушкой LLM (возвращает готовые JSON из few‑shot) → проверяем пайплайн E2E.

---

## 12) Безопасность и устойчивость
- Никогда не передавать `answer` в промпт.
- Тримминг и нормализация пробелов, но без изменения математической семантики.
- Таймауты/повторы на сетевые ошибки; экспоненциальный backoff; глобальный rate‑limit.
- Логирование: один JSON‑лог на задачу (id, латентность, токены, ретраи, валидность).
- Идемпотентность: повторный запуск не перезаписывает валидные файлы без флага `--overwrite`.

---

## 13) План по доработкам (после v1.0)
- Расширить few‑shot из реальных «трудных» задач.
- Добавить эвристику маршрутизации (например, короткие «интегралы» не снабжать избыточными подсказками).
- Поддержать локальные модели с grammar‑constraints (Outlines/Guidance/LMQL).
- В отчёт добавить распределения ошибок по типам `targets`.

---

## 14) Критерии приёмки
- Прогон на полном `tasks.csv` отрабатывает без падений.
- ≥95% строк → валидный MathIR‑JSON; все файлы на месте; есть сводный отчёт.
- Нет утечек `answer` в запросы к LLM.
- Повторный запуск детерминирован (при фикс. seed) и идемпотентен (без перегенерации, если файл существует).
